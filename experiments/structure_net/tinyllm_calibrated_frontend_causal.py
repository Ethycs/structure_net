#!/usr/bin/env python3
"""Test whether an observed calibration reference repairs N3 identifiability."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net.tinyllm_conditional_branch_depth_scan import (
    _add_log_loss_gains,
    fit_conditional_probe,
    fit_cosine_only_null,
)
from experiments.structure_net.tinyllm_internal_quotient_probe import FiberDataset
from experiments.structure_net.tinyllm_invariant_frontend_causal import (
    _module_digest,
    _probe_config,
    _task_logits,
    _tensor_digest,
    decode_sensor_tokens,
    endpoint_pass,
)
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    NuisanceScalingConfig,
    PairedCircleDataset,
    _nuisance_values,
    generate_paired_dataset,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleDataset,
    CircleTaskConfig,
    _state_digest,
    _tinyllm_config,
)
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-calibrated-frontend-causal.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-reference-stable-cosine-quotient-v1"
CONDITIONS = (
    "raw_calibrated",
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
CUTS = ("frontend", "full")
REGIMES = ("in_distribution", "composition", "extrapolation")
PRIMARY_REGIMES = ("composition", "extrapolation")
CALIBRATION_WIDTH = 8


@dataclass(frozen=True)
class CalibratedFrontendConfig:
    preset: str = "d8"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    vector_channels: int = 16
    analysis_seed: int = 83
    probe_train_samples: int = 2_048
    probe_validation_samples: int = 512
    probe_test_samples: int = 1_024
    activation_batch_size: int = 256
    probe_batch_size: int = 256
    probe_steps: int = 240
    probe_width: int = 128
    validation_interval: int = 20
    early_stopping_patience: int = 5
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 0
    gpu_memory_per_experiment_gb: Optional[float] = 3.0
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("the preregistered campaign requires five seeds")
        if self.preset != "d8" and not self.allow_underpowered:
            raise ValueError("the preregistered campaign fixes d8")
        if self.training_steps < 1 or self.train_samples < 4:
            raise ValueError("training steps and samples must be positive")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch size must be positive and even")
        if self.vector_channels < 2:
            raise ValueError("vector_channels must be at least two")


@dataclass(frozen=True)
class CalibratedDataset:
    paired: PairedCircleDataset
    calibration: torch.Tensor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def calibration_packet(
    values: Mapping[str, np.ndarray], directions: torch.Tensor
) -> torch.Tensor:
    """Return the phase-independent observed pilot sufficient statistics."""
    orientation = np.asarray(values["orientation"]).reshape(-1)
    amplitude = np.asarray(values["amplitude"]).reshape(-1)
    speed = np.asarray(values["speed"]).reshape(-1)
    offset = np.asarray(values["offset"])[:, 0, :2]
    drift = np.asarray(values["drift"])[:, 0, :2]
    packet = np.column_stack(
        (
            np.cos(orientation),
            np.sin(orientation),
            directions.numpy() * speed,
            amplitude,
            offset,
            drift,
        )
    )
    if packet.shape[1] != CALIBRATION_WIDTH:
        raise AssertionError("calibration packet width changed")
    return torch.tensor(packet, dtype=torch.float32)


def _permute_paired(
    paired: PairedCircleDataset, permutation: torch.Tensor
) -> PairedCircleDataset:
    circle = paired.circle
    fiber = paired.fiber
    return PairedCircleDataset(
        circle=CircleDataset(
            input_ids=circle.input_ids[permutation],
            target_posteriors=circle.target_posteriors[permutation],
            target_bins=circle.target_bins[permutation],
            phases=circle.phases[permutation],
            directions=circle.directions[permutation],
        ),
        fiber=FiberDataset(
            input_ids=fiber.input_ids[permutation],
            cosine=fiber.cosine[permutation],
            branch=fiber.branch[permutation],
            phase=fiber.phase[permutation],
            fiber_id=fiber.fiber_id[permutation],
        ),
    )


def generate_calibrated_dataset(
    task: CircleTaskConfig,
    *,
    sample_count: int,
    seed: int,
    regime: str = "interpolation",
    shuffle: bool = True,
) -> CalibratedDataset:
    paired = generate_paired_dataset(
        task,
        sample_count=sample_count,
        seed=seed,
        support="N3",
        regime=regime,
        shuffle=False,
    )
    nuisance = _nuisance_values(
        "N3", regime, sample_count, np.random.default_rng(seed + 10_003)
    )
    calibration = calibration_packet(nuisance, paired.circle.directions)
    if shuffle:
        permutation = torch.from_numpy(
            np.random.default_rng(seed + 70_001).permutation(sample_count)
        )
        paired = _permute_paired(paired, permutation)
        calibration = calibration[permutation]
    return CalibratedDataset(paired=paired, calibration=calibration)


def identifiability_contract() -> Dict[str, Any]:
    """Exhaust the old gauge orbits and fail if calibration misses a target change."""
    phases = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    orientations = np.linspace(-0.40, 0.40, 9)
    alphas = np.linspace(-0.35, 0.35, 15)
    alphas = alphas[np.abs(alphas) > 1e-12]
    orders = (2, 3, 4)
    history = np.arange(8, dtype=np.float64) - 7.0
    checked = 0
    target_changing = 0
    violations = 0
    maximum_sensor_error = 0.0
    minimum_target_changing_calibration_distance = float("inf")
    for phase in phases:
        for orientation in orientations:
            for order in orders:
                harmonic_phase = 0.17
                angles = phase + 0.35 * history
                xy = np.stack(
                    (
                        np.cos(orientation) * np.cos(angles)
                        - np.sin(orientation) * np.sin(angles),
                        np.sin(orientation) * np.cos(angles)
                        + np.cos(orientation) * np.sin(angles),
                    ),
                    axis=-1,
                )
                harmonic = 0.3 * np.cos(order * angles + harmonic_phase)
                sensor = np.column_stack((xy, harmonic))
                cal = np.array((math.cos(orientation), math.sin(orientation)))
                target = math.cos(phase + 0.55)
                for alpha in alphas:
                    transformed_orientation = orientation - alpha
                    if not -0.8 < transformed_orientation < 0.8:
                        continue
                    transformed_angles = angles + alpha
                    transformed_xy = np.stack(
                        (
                            np.cos(transformed_orientation)
                            * np.cos(transformed_angles)
                            - np.sin(transformed_orientation)
                            * np.sin(transformed_angles),
                            np.sin(transformed_orientation)
                            * np.cos(transformed_angles)
                            + np.cos(transformed_orientation)
                            * np.sin(transformed_angles),
                        ),
                        axis=-1,
                    )
                    transformed_harmonic = 0.3 * np.cos(
                        order * transformed_angles
                        + harmonic_phase
                        - order * alpha
                    )
                    transformed_sensor = np.column_stack(
                        (transformed_xy, transformed_harmonic)
                    )
                    transformed_cal = np.array(
                        (
                            math.cos(transformed_orientation),
                            math.sin(transformed_orientation),
                        )
                    )
                    transformed_target = math.cos(phase + alpha + 0.55)
                    sensor_error = float(np.max(np.abs(sensor - transformed_sensor)))
                    calibration_distance = float(np.linalg.norm(cal - transformed_cal))
                    target_delta = abs(target - transformed_target)
                    maximum_sensor_error = max(maximum_sensor_error, sensor_error)
                    checked += 1
                    if target_delta > 1e-10:
                        target_changing += 1
                        minimum_target_changing_calibration_distance = min(
                            minimum_target_changing_calibration_distance,
                            calibration_distance,
                        )
                        if sensor_error <= 1e-10 and calibration_distance <= 1e-10:
                            violations += 1
    passed = (
        checked > 0
        and target_changing > 0
        and violations == 0
        and maximum_sensor_error <= 1e-10
        and minimum_target_changing_calibration_distance > 1e-6
    )
    record = {
        "passed": passed,
        "proof_kind": "symbolic gauge argument plus exhaustive numerical gauge grid",
        "checked_gauge_pairs": checked,
        "target_changing_pairs": target_changing,
        "violations": violations,
        "maximum_old_observation_sensor_error": maximum_sensor_error,
        "minimum_target_changing_calibration_distance": (
            minimum_target_changing_calibration_distance
        ),
        "logical_statement": "O_cal(z)=O_cal(z') implies T(z)=T(z') on the declared gauge orbits",
        "stochastic_boundary": (
            "Equality is equality of observation distributions; additive noise "
            "realizations are not adversarial latent cancellation variables."
        ),
    }
    if not passed:
        raise RuntimeError(f"identifiability contract failed: {record}")
    return record


class AnalyticCalibratedCanonicalizer:
    def __init__(self, task: CircleTaskConfig):
        history = torch.arange(task.sensor_steps, dtype=torch.float32)
        self.normalized_history = history / max(1.0, float(task.sensor_steps - 1)) - 1.0
        self.future_delta = task.future_delta

    def __call__(self, sensor: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
        history = self.normalized_history.to(sensor.device)
        orientation = calibration[:, :2]
        signed_speed = calibration[:, 2]
        amplitude = calibration[:, 3:4]
        offset = calibration[:, 4:6]
        drift = calibration[:, 6:8]
        corrected = (
            sensor[..., :2]
            - offset[:, None, :]
            - drift[:, None, :] * history[None, :, None]
        ) / amplitude[:, None, :]
        endpoint = corrected[:, -1, :]
        lab_x = (endpoint * orientation).sum(-1)
        lab_y = orientation[:, 0] * endpoint[:, 1] - orientation[:, 1] * endpoint[:, 0]
        norm = torch.sqrt(lab_x.square() + lab_y.square()).clamp_min(1e-6)
        lab_x, lab_y = lab_x / norm, lab_y / norm
        delta = signed_speed.sign() * self.future_delta
        return (lab_x * torch.cos(delta) - lab_y * torch.sin(delta))[:, None]


class CalibratedEquivariantEncoder(nn.Module):
    def __init__(self, sensor_steps: int, vector_channels: int):
        super().__init__()
        history = torch.arange(sensor_steps, dtype=torch.float32)
        self.register_buffer(
            "normalized_history", history / max(1.0, float(sensor_steps - 1)) - 1.0
        )
        self.temporal_weights = nn.Parameter(torch.empty(vector_channels, sensor_steps))
        invariant_width = vector_channels * vector_channels
        hidden = max(32, 2 * vector_channels)
        self.vector_mixer = nn.Sequential(
            nn.Linear(invariant_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, vector_channels),
        )
        self.scalar_map = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        nn.init.normal_(self.temporal_weights, std=1.0 / math.sqrt(sensor_steps))

    def equivariant_vector(
        self, sensor: torch.Tensor, calibration: torch.Tensor
    ) -> torch.Tensor:
        offset = calibration[:, 4:6]
        drift = calibration[:, 6:8]
        amplitude = calibration[:, 3:4]
        corrected = (
            sensor[..., :2]
            - offset[:, None, :]
            - drift[:, None, :]
            * self.normalized_history[None, :, None]
        ) / amplitude[:, None, :]
        channels = torch.einsum("kt,btc->bkc", self.temporal_weights, corrected)
        gram = torch.einsum("bkc,blc->bkl", channels, channels)
        weights = torch.softmax(self.vector_mixer(gram.flatten(1)), dim=-1)
        vector = torch.einsum("bk,bkc->bc", weights, channels)
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, sensor: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
        vector = self.equivariant_vector(sensor, calibration)
        orientation = calibration[:, :2]
        dot = (vector * orientation).sum(-1)
        cross = orientation[:, 0] * vector[:, 1] - orientation[:, 1] * vector[:, 0]
        signed_speed = calibration[:, 2]
        return self.scalar_map(torch.stack((dot, cross, signed_speed), dim=-1))


def _model_config(preset: str, task: CircleTaskConfig, seed: int):
    return replace(_tinyllm_config(preset, task, seed), block_size=task.sequence_length + 1)


class CalibratedTinyLLM(nn.Module):
    def __init__(
        self,
        model: TinyLLMModel,
        condition: str,
        task: CircleTaskConfig,
        config: CalibratedFrontendConfig,
    ):
        super().__init__()
        self.model = model
        self.condition = condition
        self.calibration_embedding: Optional[nn.Linear] = None
        self.scalar_embedding: Optional[nn.Linear] = None
        self.encoder: Optional[CalibratedEquivariantEncoder] = None
        self.analytic = AnalyticCalibratedCanonicalizer(task)
        if condition == "raw_calibrated":
            self.calibration_embedding = nn.Linear(CALIBRATION_WIDTH, model.config.n_embd)
        else:
            self.scalar_embedding = nn.Linear(1, model.config.n_embd)
            if condition == "learned_calibrated_equivariant":
                self.encoder = CalibratedEquivariantEncoder(
                    task.sensor_steps, config.vector_channels
                )

    def feature(self, sensor: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
        if self.condition == "raw_calibrated":
            return torch.cat((sensor.flatten(1), calibration), dim=1)
        if self.encoder is not None:
            return self.encoder(sensor, calibration)
        return self.analytic(sensor, calibration)

    def forward_cuts(
        self,
        input_ids: torch.Tensor,
        sensor: torch.Tensor,
        calibration: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feature = self.feature(sensor, calibration)
        if self.condition == "raw_calibrated":
            if self.calibration_embedding is None:
                raise AssertionError("raw calibration embedding missing")
            prefix = self.model.transformer["wte"](input_ids[:, :-1])
            cal_token = self.calibration_embedding(calibration)[:, None, :]
            query = self.model.transformer["wte"](input_ids[:, -1:])
            value = torch.cat((prefix, cal_token, query), dim=1)
        else:
            if self.scalar_embedding is None:
                raise AssertionError("structured scalar embedding missing")
            bos = self.model.transformer["wte"](input_ids[:, :1])
            scalar = self.scalar_embedding(feature)[:, None, :]
            query = self.model.transformer["wte"](input_ids[:, -1:])
            value = torch.cat((bos, scalar, query), dim=1)
        positions = torch.arange(value.shape[1], device=value.device)
        value = value + self.model.transformer["wpe"](positions)
        for block in self.model.transformer["h"]:
            value = block(value)
        return {"frontend": feature, "full": value[:, -1, :]}


def _protocol_material(
    task: CircleTaskConfig, config: CalibratedFrontendConfig, seed: int
) -> tuple[CalibratedDataset, torch.Tensor, str, str]:
    dataset = generate_calibrated_dataset(
        task, sample_count=config.train_samples, seed=seed + 1_001, shuffle=False
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    pair_batches = torch.randint(
        0,
        config.train_samples // 2,
        (config.training_steps, config.batch_size // 2),
        generator=generator,
    )
    paired = dataset.paired
    data_hash = _tensor_digest(
        paired.circle.input_ids,
        paired.circle.target_posteriors,
        paired.fiber.cosine,
        paired.fiber.branch,
        dataset.calibration,
    )
    return dataset, pair_batches, data_hash, _tensor_digest(pair_batches)


def _condition_directory(root: Path, condition: str, seed: int) -> Path:
    return root / "runs" / condition / f"seed_{seed}"


def train_cell(
    task: CircleTaskConfig,
    config: CalibratedFrontendConfig,
    condition: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[CalibratedTinyLLM, Dict[str, Any]]:
    identifiability_contract()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyLLMModel(
        _model_config(config.preset, task, seed),
        name=f"CalibratedFrontend_{condition}_{seed}",
    )
    system = CalibratedTinyLLM(model, condition, task, config).to(device)
    initial_model = _state_digest(model)
    initial_system = _module_digest(system)
    dataset, pair_batches, data_hash, batch_hash = _protocol_material(task, config, seed)
    paired = dataset.paired
    sensor = decode_sensor_tokens(paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    optimizer = torch.optim.AdamW(
        system.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    system.train()
    for step in range(1, config.training_steps + 1):
        pairs = pair_batches[step - 1]
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        inputs = paired.circle.input_ids[indices].to(device, non_blocking=True)
        observed = sensor[indices].to(device, non_blocking=True)
        calibration = dataset.calibration[indices].to(device, non_blocking=True)
        targets = paired.circle.target_posteriors[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        residual = system.forward_cuts(inputs, observed, calibration)["full"]
        logits = _task_logits(model, residual, answer_ids)
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), config.gradient_clip)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            history.append(
                {
                    "step": float(step),
                    "task_loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    final_model = _state_digest(model)
    final_system = _module_digest(system)
    checkpoint = output_dir / "model.pt"
    model.to("cpu").save_checkpoint(
        checkpoint,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "condition": condition,
            "seed": seed,
            "training_data_sha256": data_hash,
            "minibatch_schedule_sha256": batch_hash,
        },
    )
    frontend_checkpoint = output_dir / "frontend.pt"
    torch.save(
        {
            "condition": condition,
            "calibration_embedding": (
                system.calibration_embedding.to("cpu").state_dict()
                if system.calibration_embedding is not None
                else None
            ),
            "scalar_embedding": (
                system.scalar_embedding.to("cpu").state_dict()
                if system.scalar_embedding is not None
                else None
            ),
            "encoder": system.encoder.to("cpu").state_dict() if system.encoder else None,
        },
        frontend_checkpoint,
    )
    system.to(device)
    return system, {
        "source": "new_matched_calibrated_task_only_training",
        "training_seconds": training_seconds,
        "training_history": history,
        "initial_model_state_sha256": initial_model,
        "initial_system_state_sha256": initial_system,
        "final_model_state_sha256": final_model,
        "final_system_state_sha256": final_system,
        "training_data_sha256": data_hash,
        "minibatch_schedule_sha256": batch_hash,
        "checkpoint": str(checkpoint),
        "frontend_checkpoint": str(frontend_checkpoint),
        "objective": "ordinary cosine-interval task cross-entropy only",
    }


def _analysis_config(config: CalibratedFrontendConfig) -> NuisanceScalingConfig:
    return _probe_config(config)  # type: ignore[arg-type,return-value]


@torch.no_grad()
def extract_features(
    system: CalibratedTinyLLM,
    dataset: CalibratedDataset,
    task: CircleTaskConfig,
    config: CalibratedFrontendConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    system.eval()
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    captured: Dict[str, List[torch.Tensor]] = {cut: [] for cut in CUTS}
    for start in range(0, len(sensor), config.activation_batch_size):
        stop = start + config.activation_batch_size
        cuts = system.forward_cuts(
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        for cut, value in cuts.items():
            captured[cut].append(value.detach().cpu())
    return {cut: torch.cat(parts).float().numpy() for cut, parts in captured.items()}


@torch.no_grad()
def task_metrics(
    system: CalibratedTinyLLM,
    dataset: CalibratedDataset,
    task: CircleTaskConfig,
    config: CalibratedFrontendConfig,
    device: torch.device,
) -> Dict[str, float]:
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    probabilities = []
    for start in range(0, len(sensor), config.activation_batch_size):
        stop = start + config.activation_batch_size
        residual = system.forward_cuts(
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )["full"]
        probabilities.append(
            torch.softmax(_task_logits(system.model, residual, answer_ids), -1).cpu()
        )
    predicted = torch.cat(probabilities)
    target = dataset.paired.circle.target_posteriors
    predicted_bins = predicted.argmax(1)
    target_bins = target.argmax(1)
    delta = (predicted_bins - target_bins).abs()
    circular = torch.minimum(delta, target.shape[1] - delta)
    return {
        "exact_bin_accuracy": float((predicted_bins == target_bins).float().mean()),
        "mean_circular_error_radians": float(
            circular.float().mean() * (2.0 * math.pi / target.shape[1])
        ),
        "mean_target_cross_entropy": float(
            -(target * predicted.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def analyze(
    system: CalibratedTinyLLM,
    task: CircleTaskConfig,
    config: CalibratedFrontendConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    datasets = {
        "train": generate_calibrated_dataset(task, sample_count=config.probe_train_samples, seed=config.analysis_seed + 101),
        "validation": generate_calibrated_dataset(task, sample_count=config.probe_validation_samples, seed=config.analysis_seed + 211),
        "in_distribution": generate_calibrated_dataset(task, sample_count=config.probe_test_samples, seed=config.analysis_seed + 307),
        "composition": generate_calibrated_dataset(task, sample_count=config.probe_test_samples, seed=config.analysis_seed + 1_316, regime="composition"),
        "extrapolation": generate_calibrated_dataset(task, sample_count=config.probe_test_samples, seed=config.analysis_seed + 2_325, regime="extrapolation"),
    }
    features = {
        name: extract_features(system, dataset, task, config, device)
        for name, dataset in datasets.items()
    }
    probe_config = _analysis_config(config)
    evaluation_fibers = {name: datasets[name].paired.fiber for name in REGIMES}
    null = fit_cosine_only_null(
        datasets["train"].paired.fiber,
        datasets["validation"].paired.fiber,
        evaluation_fibers,
        probe_config,
        device,
        kind="nonlinear",
        seed=config.analysis_seed + seed * 101,
    )
    cuts: Dict[str, Any] = {}
    for cut_index, cut in enumerate(CUTS):
        probe = fit_conditional_probe(
            features["train"][cut],
            features["validation"][cut],
            {name: features[name][cut] for name in REGIMES},
            datasets["train"].paired.fiber,
            datasets["validation"].paired.fiber,
            evaluation_fibers,
            probe_config,
            device,
            kind="nonlinear",
            seed=config.analysis_seed + seed * 10_007 + cut_index * 1_009,
        )
        _add_log_loss_gains(probe, null)
        cuts[cut] = {"probe": probe}
    return {
        "cuts": cuts,
        "task_metrics": {
            regime: task_metrics(system, datasets[regime], task, config, device)
            for regime in REGIMES
        },
    }


def _config_from_mapping(values: Mapping[str, Any]) -> CalibratedFrontendConfig:
    converted = dict(values)
    for field in ("seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return CalibratedFrontendConfig(**converted)


def _scientific_fingerprint(experiment: Experiment) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "configuration": experiment.parameters["configuration"],
        "task_config": experiment.parameters["task_config"],
        "condition": experiment.parameters["condition"],
        "seed": experiment.seed,
        "implementation_sha256": experiment.parameters["implementation_sha256"],
        "identifiability_contract": experiment.parameters["identifiability_contract"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def calibrated_frontend_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    if _implementation_digest() != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    contract = identifiability_contract()
    if contract != experiment.parameters["identifiability_contract"]:
        raise RuntimeError("identifiability contract changed after campaign construction")
    output_dir = _condition_directory(Path(experiment.parameters["output_dir"]), condition, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    system, training = train_cell(task, config, condition, seed, device, output_dir)
    analysis = analyze(system, task, config, seed, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "completed_at": _utc_now(),
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in system.model.parameters()),
        "system_parameters": sum(parameter.numel() for parameter in system.parameters()),
        "training": training,
        "analysis": analysis,
        "identifiability_contract": contract,
        "peak_cuda_allocated_gb": peak,
        "wall_seconds": elapsed,
        "scientific_fingerprint": _scientific_fingerprint(experiment),
        "implementation_sha256": _implementation_digest(),
    }
    detail_path = output_dir / "result.json"
    _write_json(detail_path, detail)
    composition = analysis["cuts"]["full"]["probe"]["evaluations"]["composition"]
    extrapolation = analysis["cuts"]["full"]["probe"]["evaluations"]["extrapolation"]
    metrics = {
        "full_composition_cosine": float(composition["cosine_pearson"]),
        "full_composition_branch": float(composition["balanced_accuracy"]),
        "full_extrapolation_cosine": float(extrapolation["cosine_pearson"]),
        "full_extrapolation_branch": float(extrapolation["balanced_accuracy"]),
        "peak_cuda_allocated_gb": peak,
    }
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics=metrics,
        primary_metric=min(metrics["full_composition_cosine"], metrics["full_extrapolation_cosine"]),
        model_architecture=[task.vocab_size, *([system.model.config.n_embd] * system.model.config.n_layer), task.phase_bins],
        model_parameters=detail["system_parameters"],
        training_time=elapsed,
        training_history=training["training_history"],
        model_checkpoint=training["checkpoint"],
        observations=[f"detail={detail_path}", f"condition={condition}"],
    )


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: CalibratedFrontendConfig
) -> Dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    arms: Dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = sorted(
            (item for item in details if item["condition"] == condition),
            key=lambda item: item["seed"],
        )
        cuts: Dict[str, Any] = {}
        for cut in CUTS:
            cuts[cut] = {}
            for regime in REGIMES:
                values = [
                    item["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]
                    for item in selected
                ]
                passes = [endpoint_pass(value) for value in values]
                cuts[cut][regime] = {
                    "cosine_pearson_mean": float(np.mean([value["cosine_pearson"] for value in values])),
                    "branch_balanced_accuracy_mean": float(np.mean([value["balanced_accuracy"] for value in values])),
                    "conditional_log_loss_gain_mean": float(np.mean([value["conditional_log_loss_gain_over_cosine_only"] for value in values])),
                    "pass_count": int(sum(passes)),
                    "pass_by_seed": {str(item["seed"]): passed for item, passed in zip(selected, passes)},
                }
        joint_by_seed = {}
        for item in selected:
            seed = str(item["seed"])
            cells = {
                f"{cut}:{regime}": cuts[cut][regime]["pass_by_seed"][seed]
                for cut in CUTS
                for regime in PRIMARY_REGIMES
            }
            joint_by_seed[seed] = {"cells": cells, "joint_pass": all(cells.values())}
        joint_count = sum(value["joint_pass"] for value in joint_by_seed.values())
        arms[condition] = {
            "cuts": cuts,
            "joint_pass_by_seed": joint_by_seed,
            "joint_pass_count": joint_count,
            "success": joint_count >= required,
        }
    analytic_success = arms["analytic_calibrated"]["success"]
    learned_success = arms["learned_calibrated_equivariant"]["success"]
    if analytic_success and learned_success:
        conclusion = "gauge_repair_makes_quotient_constructible"
    elif analytic_success:
        conclusion = "identifiable_but_learned_family_or_optimization_insufficient"
    elif all(
        arms["analytic_calibrated"]["cuts"][cut][regime]["pass_count"] < required
        for cut in CUTS
        for regime in PRIMARY_REGIMES
    ):
        conclusion = "analytic_control_failed_audit_calibration_or_implementation"
    else:
        conclusion = "gauge_repair_necessary_not_sufficient"
    return {
        "arms": arms,
        "preregistered_gate": {
            "required_seed_passes": required,
            "cosine_threshold": 0.90,
            "branch_accuracy_ceiling": 0.55,
            "cuts": list(CUTS),
            "regimes": list(PRIMARY_REGIMES),
        },
        "conclusion": conclusion,
    }


def _experiments(
    config: CalibratedFrontendConfig,
    task: CircleTaskConfig,
    output_dir: Path,
) -> List[Experiment]:
    contract = identifiability_contract()
    common = {
        "configuration": asdict(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "implementation_sha256": _implementation_digest(),
        "identifiability_contract": contract,
    }
    return [
        Experiment(
            id=f"tinyllm-calibrated-frontend-{condition}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM calibrated front-end {condition} seed {seed}",
            parameters={**common, "condition": condition, "seed": seed},
            seed=seed,
        )
        for condition in CONDITIONS
        for seed in config.seeds
    ]


def _existing_detail(experiment: Experiment, output_dir: Path) -> Optional[Dict[str, Any]]:
    path = _condition_directory(output_dir, str(experiment.parameters["condition"]), int(experiment.seed)) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint") != _scientific_fingerprint(experiment)
    ):
        return None
    checkpoint = Path(detail.get("training", {}).get("checkpoint", ""))
    frontend = Path(detail.get("training", {}).get("frontend_checkpoint", ""))
    return detail if checkpoint.is_file() and frontend.is_file() else None


async def run_campaign(
    config: CalibratedFrontendConfig,
    task: CircleTaskConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    contract = identifiability_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = _experiments(config, task, output_dir)
    existing = {
        experiment.id: detail
        for experiment in experiments
        if (detail := _existing_detail(experiment, output_dir)) is not None
    }
    pending = [experiment for experiment in experiments if experiment.id not in existing]
    lab = LabConfig(
        project_name="tinyllm_calibrated_frontend_causal",
        results_dir=str(output_dir),
        device_ids=list(config.device_ids),
        max_parallel_experiments=config.max_parallel_experiments,
        gpu_slots_per_device=config.gpu_slots_per_device,
        gpu_memory_per_experiment_gb=config.gpu_memory_per_experiment_gb,
        max_gpu_slots_per_device=config.max_gpu_slots_per_device,
        max_experiment_retries=config.max_retries,
        resume_completed_experiments=config.resume,
        auto_balance=False,
        enable_wandb=False,
        verbose=True,
    )
    runner = AsyncExperimentRunner(lab, calibrated_frontend_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful = {result.experiment_id for result in results if result.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id in successful:
            path = _condition_directory(output_dir, str(experiment.parameters["condition"]), int(experiment.seed)) / "result.json"
            details.append(json.loads(path.read_text(encoding="utf-8")))
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if len(details) == len(experiments) else "partial",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "task_config": asdict(task),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "implementation_sha256": _implementation_digest(),
        "identifiability_contract": contract,
        "summary": {
            "requested": len(experiments),
            "reused": len(existing),
            "scheduled": len(pending),
            "completed": len(details),
            "failed": len(experiments) - len(details),
        },
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {str(key): value for key, value in runner.slot_plan.slots_by_device.items()},
            "calibration": runner.slot_plan.calibration,
        },
        "aggregates": aggregate_details(details, config) if len(details) == len(experiments) else {},
        "results": [
            {
                "experiment_id": result.experiment_id,
                "status": result.status.value,
                "metrics": result.metrics,
                "error": result.error,
                "model_checkpoint": result.model_checkpoint,
            }
            for result in results
        ],
        "method_boundaries": [
            "The exact calibration packet is a strong positive-control observation, not a claim about calibration cost in real systems.",
            "Identifiability is proved for observation distributions; arbitrary noise-realization cancellation is outside the latent-state contract.",
            "Fresh nonlinear probes test conditional decodability rather than certified mutual information.",
        ],
    }
    _write_json(output_dir / "campaign_results.json", bundle)
    return bundle


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_devices(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "cpu":
        return (-1,)
    if value.strip().lower() == "auto":
        return tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else (-1,)
    return _comma_ints(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--probe-train-samples", type=int, default=2_048)
    parser.add_argument("--probe-validation-samples", type=int, default=512)
    parser.add_argument("--probe-test-samples", type=int, default=1_024)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--slots-per-gpu", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_calibrated_frontend_causal/20260806_d8_preregistered"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.seeds = "7"
        args.steps = 2
        args.train_samples = 32
        args.batch_size = 8
        args.probe_steps = 20
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 32
        args.allow_underpowered = True
    config = CalibratedFrontendConfig(
        seeds=_comma_ints(args.seeds),
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        probe_steps=args.probe_steps,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
        device_ids=_parse_devices(args.gpus),
        gpu_slots_per_device=args.slots_per_gpu,
        max_parallel_experiments=args.max_parallel,
        max_retries=args.retries,
        resume=args.resume,
        allow_underpowered=args.allow_underpowered,
    )
    task = CircleTaskConfig(train_samples=config.train_samples)
    bundle = asyncio.run(run_campaign(config, task, args.output))
    print(json.dumps({"summary": bundle["summary"], "conclusion": bundle.get("aggregates", {}).get("conclusion")}, indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
