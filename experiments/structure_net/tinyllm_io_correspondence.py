#!/usr/bin/env python3
"""Track gauge-repaired paired I/O relations through TinyLLM layers."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated_source
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant_source
import experiments.structure_net.tinyllm_nuisance_support_scaling as nuisance_source
from experiments.structure_net.tinyllm_calibrated_frontend_causal import (
    CALIBRATION_WIDTH,
    CalibratedDataset,
    CalibratedTinyLLM,
    calibration_packet,
    generate_calibrated_dataset,
)
from experiments.structure_net.tinyllm_conditional_branch_depth_scan import (
    _add_log_loss_gains,
    fit_conditional_probe,
    fit_cosine_only_null,
)
from experiments.structure_net.tinyllm_internal_quotient_probe import FiberDataset
from experiments.structure_net.tinyllm_invariant_frontend_causal import (
    LearnedEquivariantEncoder,
    _module_digest,
    _probe_config,
    _task_logits,
    _tensor_digest,
    decode_sensor_tokens,
    endpoint_pass,
)
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    PairedCircleDataset,
    _nuisance_values,
    _sample_directions,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleDataset,
    CircleTaskConfig,
    _serialize_sensor_values,
    _state_digest,
)
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-io-correspondence.v1"
HYPOTHESIS_ID = "tinyllm-io-correspondence-gauge-descent-v1"
SOURCE_SCHEMA_VERSION = calibrated_source.SCHEMA_VERSION
CONDITIONS = (
    "uncalibrated_absolute_raw",
    "uncalibrated_absolute_equivariant",
    "calibrated_absolute_raw",
    "calibrated_absolute_analytic",
    "calibrated_absolute_equivariant",
    "uncalibrated_relative_raw",
    "uncalibrated_relative_equivariant",
)
CALIBRATED_CONDITIONS = {
    "calibrated_absolute_raw": "raw_calibrated",
    "calibrated_absolute_analytic": "analytic_calibrated",
    "calibrated_absolute_equivariant": "learned_calibrated_equivariant",
}
CUTS = ("frontend", "post_attention", "post_mlp", "full")
REGIMES = ("in_distribution", "composition", "extrapolation")
PRIMARY_REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class IOCorrespondenceConfig:
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
    mapper_components: int = 16
    mapper_cover_intervals: int = 9
    mapper_overlap: float = 0.30
    mapper_neighbors: int = 12
    mapper_min_cluster: int = 4
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 0
    gpu_memory_per_experiment_gb: Optional[float] = 3.0
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    reuse_calibrated_controls: bool = True
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
        if not 0.0 <= self.mapper_overlap < 1.0:
            raise ValueError("Mapper overlap must lie in [0, 1)")
        if self.mapper_cover_intervals < 3 or self.mapper_neighbors < 1:
            raise ValueError("Mapper cover and neighbor counts are too small")


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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__),
        Path(calibrated_source.__file__),
        Path(invariant_source.__file__),
        Path(nuisance_source.__file__),
    )
    for path in sorted(paths, key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def condition_spec(condition: str) -> Dict[str, str]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown I/O correspondence condition {condition}")
    return {
        "observation": "calibrated" if condition.startswith("calibrated") else "uncalibrated",
        "target": "relative" if "relative" in condition else "absolute",
        "frontend": (
            "analytic"
            if condition.endswith("analytic")
            else "equivariant"
            if condition.endswith("equivariant")
            else "raw"
        ),
    }


def io_relation_contract() -> Dict[str, Any]:
    """Audit descent of all three relations on the exact generator gauge."""
    phases = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    orientations = np.linspace(-0.40, 0.40, 9)
    alphas = np.linspace(-0.35, 0.35, 15)
    alphas = alphas[np.abs(alphas) > 1e-12]
    checked = 0
    absolute_target_changes = 0
    relative_violations = 0
    calibrated_violations = 0
    maximum_sensor_error = 0.0
    minimum_calibration_distance = float("inf")
    for phase in phases:
        for orientation in orientations:
            observed_angle = phase + orientation
            sensor = np.array((math.cos(observed_angle), math.sin(observed_angle)))
            calibration = np.array((math.cos(orientation), math.sin(orientation)))
            for alpha in alphas:
                checked += 1
                transformed_phase = phase + alpha
                transformed_orientation = orientation - alpha
                transformed_angle = transformed_phase + transformed_orientation
                transformed_sensor = np.array(
                    (math.cos(transformed_angle), math.sin(transformed_angle))
                )
                transformed_calibration = np.array(
                    (math.cos(transformed_orientation), math.sin(transformed_orientation))
                )
                sensor_error = float(np.linalg.norm(sensor - transformed_sensor))
                calibration_distance = float(
                    np.linalg.norm(calibration - transformed_calibration)
                )
                maximum_sensor_error = max(maximum_sensor_error, sensor_error)
                old_target = math.cos(phase)
                transformed_old_target = math.cos(transformed_phase)
                relative_target = math.cos(observed_angle)
                transformed_relative_target = math.cos(transformed_angle)
                if abs(old_target - transformed_old_target) > 1e-8:
                    absolute_target_changes += 1
                    minimum_calibration_distance = min(
                        minimum_calibration_distance, calibration_distance
                    )
                    if sensor_error <= 1e-10 and calibration_distance <= 1e-10:
                        calibrated_violations += 1
                if sensor_error <= 1e-10 and abs(
                    relative_target - transformed_relative_target
                ) > 1e-10:
                    relative_violations += 1
    contracts = {
        "uncalibrated_absolute": {
            "expected_to_descend": False,
            "passed": absolute_target_changes > 0,
            "counterexample_count": absolute_target_changes,
            "logical_statement": "O(z)=O(z') does not imply cos(phi)=cos(phi')",
        },
        "calibrated_absolute": {
            "expected_to_descend": True,
            "passed": calibrated_violations == 0 and minimum_calibration_distance > 1e-6,
            "violations": calibrated_violations,
            "minimum_target_changing_calibration_distance": minimum_calibration_distance,
            "logical_statement": "O_cal(z)=O_cal(z') implies cos(phi)=cos(phi')",
        },
        "uncalibrated_relative": {
            "expected_to_descend": True,
            "passed": relative_violations == 0,
            "violations": relative_violations,
            "logical_statement": "O(z)=O(z') implies cos(phi+theta)=cos(phi'+theta')",
        },
    }
    passed = all(item["passed"] for item in contracts.values())
    result = {
        "passed": passed,
        "proof_kind": "symbolic gauge identity plus exhaustive numerical orbit grid",
        "checked_gauge_pairs": checked,
        "maximum_gauge_sensor_error": maximum_sensor_error,
        "relations": contracts,
        "stochastic_boundary": (
            "Equality is equality of observation distributions; additive noise realizations "
            "are not adversarial latent cancellation variables."
        ),
    }
    if not passed:
        raise RuntimeError(f"I/O relation contract failed: {result}")
    return result


def _cosine_posteriors(cosine: np.ndarray, task: CircleTaskConfig) -> torch.Tensor:
    centers = np.linspace(-1.0, 1.0, task.phase_bins)
    bin_width = 2.0 / (task.phase_bins - 1)
    probabilities = np.exp(
        -0.5 * ((centers[None, :] - cosine[:, None]) / bin_width) ** 2
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return torch.from_numpy(probabilities.astype(np.float32))


def _permute_dataset(dataset: CalibratedDataset, permutation: torch.Tensor) -> CalibratedDataset:
    paired = dataset.paired
    circle = paired.circle
    fiber = paired.fiber
    return CalibratedDataset(
        paired=PairedCircleDataset(
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
        ),
        calibration=dataset.calibration[permutation],
    )


def _serialize_with_nuisance(
    task: CircleTaskConfig,
    current_phases: np.ndarray,
    directions: np.ndarray,
    values: Mapping[str, np.ndarray],
    generator: np.random.Generator,
) -> torch.Tensor:
    history = np.arange(task.sensor_steps, dtype=np.float64) - (task.sensor_steps - 1)
    angles = (
        current_phases[:, None]
        + directions[:, None] * history[None, :] * values["speed"]
    )
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    cosine = np.cos(values["orientation"])
    sine = np.sin(values["orientation"])
    rotated_x = cosine * base_x - sine * base_y
    rotated_y = sine * base_x + cosine * base_y
    harmonic = values["harmonic_strength"] * np.cos(
        values["harmonic_order"] * angles + values["harmonic_phase"]
    )
    sensor = values["amplitude"] * np.stack((rotated_x, rotated_y, harmonic), axis=-1)
    sensor += values["offset"]
    normalized_history = history / max(1.0, float(task.sensor_steps - 1))
    sensor += values["drift"] * normalized_history[None, :, None]
    sensor += generator.normal(0.0, 1.0, sensor.shape) * values["noise_scale"]
    return torch.from_numpy(_serialize_sensor_values(sensor, task))


def generate_relation_fiber_dataset(
    task: CircleTaskConfig,
    *,
    target_kind: str,
    sample_count: int,
    seed: int,
    regime: str = "interpolation",
    shuffle: bool = True,
) -> CalibratedDataset:
    """Generate exact output-matched pairs for one declared I/O relation."""
    if target_kind not in {"absolute", "relative"}:
        raise ValueError("target_kind must be absolute or relative")
    if sample_count < 4 or sample_count % 2:
        raise ValueError("sample_count must be even and at least four")
    generator = np.random.default_rng(seed)
    pair_count = sample_count // 2
    target_cosine = generator.uniform(-0.95, 0.95, pair_count)
    positive = np.arccos(target_cosine)
    negative = (2.0 * math.pi - positive) % (2.0 * math.pi)
    target_phase = np.column_stack((positive, negative)).reshape(-1)
    branch = np.tile(np.array([0.0, 1.0]), pair_count)
    fiber_id = np.repeat(np.arange(pair_count), 2)
    directions = _sample_directions("N3", regime, sample_count, generator)
    nuisance_generator = np.random.default_rng(seed + 10_003)
    values = _nuisance_values("N3", regime, sample_count, nuisance_generator)
    orientation = np.asarray(values["orientation"]).reshape(-1)
    absolute_future = (
        target_phase
        if target_kind == "absolute"
        else (target_phase - orientation) % (2.0 * math.pi)
    )
    current = (absolute_future - directions * task.future_delta) % (2.0 * math.pi)
    inputs = _serialize_with_nuisance(
        task, current, directions, values, nuisance_generator
    )
    cosine = np.repeat(target_cosine, 2).astype(np.float32)
    posteriors = _cosine_posteriors(cosine, task)
    circle = CircleDataset(
        input_ids=inputs,
        target_posteriors=posteriors,
        target_bins=posteriors.argmax(1),
        phases=torch.from_numpy(current.astype(np.float32)),
        directions=torch.from_numpy(directions.astype(np.float32)),
    )
    paired = PairedCircleDataset(
        circle=circle,
        fiber=FiberDataset(
            input_ids=inputs,
            cosine=torch.from_numpy(cosine),
            branch=torch.from_numpy(branch.astype(np.float32)),
            phase=torch.from_numpy(target_phase.astype(np.float32)),
            fiber_id=torch.from_numpy(fiber_id.astype(np.int64)),
        ),
    )
    result = CalibratedDataset(
        paired=paired,
        calibration=calibration_packet(values, circle.directions),
    )
    if shuffle:
        permutation = torch.from_numpy(
            np.random.default_rng(seed + 70_001).permutation(sample_count)
        )
        result = _permute_dataset(result, permutation)
    return result


def with_relation_target(
    base: CalibratedDataset, task: CircleTaskConfig, target_kind: str
) -> CalibratedDataset:
    """Replace only the target map on a shared latent/nuisance cohort."""
    if target_kind == "absolute":
        return base
    if target_kind != "relative":
        raise ValueError("unknown target kind")
    orientation = torch.atan2(base.calibration[:, 1], base.calibration[:, 0])
    relative_phase = torch.remainder(base.paired.fiber.phase + orientation, 2.0 * math.pi)
    cosine = torch.cos(relative_phase)
    branch = (torch.sin(relative_phase) < 0).float()
    posteriors = _cosine_posteriors(cosine.numpy(), task)
    circle = base.paired.circle
    return CalibratedDataset(
        paired=PairedCircleDataset(
            circle=CircleDataset(
                input_ids=circle.input_ids,
                target_posteriors=posteriors,
                target_bins=posteriors.argmax(1),
                phases=circle.phases,
                directions=circle.directions,
            ),
            fiber=FiberDataset(
                input_ids=base.paired.fiber.input_ids,
                cosine=cosine.float(),
                branch=branch,
                phase=relative_phase.float(),
                fiber_id=base.paired.fiber.fiber_id,
            ),
        ),
        calibration=base.calibration,
    )


class UncalibratedTinyLLM(nn.Module):
    """Raw or SO(2)-equivariant TinyLLM with a common extra context position."""

    def __init__(
        self,
        model: TinyLLMModel,
        frontend: str,
        task: CircleTaskConfig,
        config: IOCorrespondenceConfig,
    ):
        super().__init__()
        if frontend not in {"raw", "equivariant"}:
            raise ValueError("uncalibrated frontend must be raw or equivariant")
        self.model = model
        self.frontend = frontend
        self.encoder = (
            LearnedEquivariantEncoder(task.sensor_steps, config.vector_channels)
            if frontend == "equivariant"
            else None
        )
        self.scalar_embedding = (
            nn.Linear(1, model.config.n_embd) if frontend == "equivariant" else None
        )

    def feature(self, sensor: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return sensor.flatten(1)
        return self.encoder(sensor)

    def forward_cuts(
        self, input_ids: torch.Tensor, sensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        feature = self.feature(sensor)
        if self.encoder is None:
            prefix = self.model.transformer["wte"](input_ids[:, :-1])
            query = self.model.transformer["wte"](input_ids[:, -1:])
            zero = torch.zeros_like(query)
            value = torch.cat((prefix, zero, query), dim=1)
        else:
            if self.scalar_embedding is None:
                raise AssertionError("equivariant scalar embedding missing")
            bos = self.model.transformer["wte"](input_ids[:, :1])
            scalar = self.scalar_embedding(feature)[:, None, :]
            query = self.model.transformer["wte"](input_ids[:, -1:])
            value = torch.cat((bos, scalar, query), dim=1)
        return _transformer_cuts(self.model, value, feature)


def _transformer_cuts(
    model: TinyLLMModel, value: torch.Tensor, feature: torch.Tensor
) -> Dict[str, torch.Tensor]:
    positions = torch.arange(value.shape[1], device=value.device)
    value = value + model.transformer["wpe"](positions)
    block = model.transformer["h"][0]
    attended = value + block.attn(block.ln_1(value))
    post_mlp = attended + block.mlp(block.ln_2(attended))
    final = post_mlp
    for later in model.transformer["h"][1:]:
        final = later(final)
    return {
        "frontend": feature,
        "post_attention": attended[:, -1, :],
        "post_mlp": post_mlp[:, -1, :],
        "full": final[:, -1, :],
    }


def _calibrated_cuts(
    system: CalibratedTinyLLM,
    input_ids: torch.Tensor,
    sensor: torch.Tensor,
    calibration: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    feature = system.feature(sensor, calibration)
    if system.condition == "raw_calibrated":
        if system.calibration_embedding is None:
            raise AssertionError("raw calibration embedding missing")
        prefix = system.model.transformer["wte"](input_ids[:, :-1])
        cal_token = system.calibration_embedding(calibration)[:, None, :]
        query = system.model.transformer["wte"](input_ids[:, -1:])
        value = torch.cat((prefix, cal_token, query), dim=1)
    else:
        if system.scalar_embedding is None:
            raise AssertionError("structured scalar embedding missing")
        bos = system.model.transformer["wte"](input_ids[:, :1])
        scalar = system.scalar_embedding(feature)[:, None, :]
        query = system.model.transformer["wte"](input_ids[:, -1:])
        value = torch.cat((bos, scalar, query), dim=1)
    return _transformer_cuts(system.model, value, feature)


def _forward_cuts(
    system: nn.Module,
    condition: str,
    inputs: torch.Tensor,
    sensor: torch.Tensor,
    calibration: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if condition in CALIBRATED_CONDITIONS:
        return _calibrated_cuts(system, inputs, sensor, calibration)  # type: ignore[arg-type]
    return system.forward_cuts(inputs, sensor)  # type: ignore[attr-defined,no-any-return]


def _model(system: nn.Module) -> TinyLLMModel:
    return system.model  # type: ignore[attr-defined,no-any-return]


def _protocol_material(
    task: CircleTaskConfig, config: IOCorrespondenceConfig, seed: int, target_kind: str
) -> tuple[CalibratedDataset, torch.Tensor, str, str, str]:
    base = generate_calibrated_dataset(
        task, sample_count=config.train_samples, seed=seed + 1_001, shuffle=False
    )
    dataset = with_relation_target(base, task, target_kind)
    generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    pair_batches = torch.randint(
        0,
        config.train_samples // 2,
        (config.training_steps, config.batch_size // 2),
        generator=generator,
    )
    matched_base_hash = _tensor_digest(
        base.paired.circle.input_ids,
        base.paired.circle.target_posteriors,
        base.paired.fiber.cosine,
        base.paired.fiber.branch,
        base.calibration,
    )
    target_hash = _tensor_digest(
        dataset.paired.circle.target_posteriors,
        dataset.paired.fiber.cosine,
        dataset.paired.fiber.branch,
    )
    return dataset, pair_batches, matched_base_hash, target_hash, _tensor_digest(pair_batches)


def _condition_directory(root: Path, condition: str, seed: int) -> Path:
    return root / "runs" / condition / f"seed_{seed}"


def _save_uncalibrated_frontend(system: UncalibratedTinyLLM, path: Path) -> None:
    torch.save(
        {
            "frontend": system.frontend,
            "scalar_embedding": (
                system.scalar_embedding.to("cpu").state_dict()
                if system.scalar_embedding is not None
                else None
            ),
            "encoder": (
                system.encoder.to("cpu").state_dict() if system.encoder is not None else None
            ),
        },
        path,
    )


def train_uncalibrated(
    task: CircleTaskConfig,
    config: IOCorrespondenceConfig,
    condition: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[UncalibratedTinyLLM, Dict[str, Any]]:
    spec = condition_spec(condition)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyLLMModel(
        calibrated_source._model_config(config.preset, task, seed),
        name=f"IOCorrespondence_{condition}_{seed}",
    )
    system = UncalibratedTinyLLM(model, spec["frontend"], task, config).to(device)
    initial_model = _state_digest(model)
    initial_system = _module_digest(system)
    dataset, pair_batches, base_hash, target_hash, batch_hash = _protocol_material(
        task, config, seed, spec["target"]
    )
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
        residual = _forward_cuts(
            system, condition, inputs, observed, calibration
        )["full"]
        logits = _task_logits(model, residual, answer_ids)
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            system.parameters(), config.gradient_clip
        )
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
            "matched_base_sha256": base_hash,
            "target_sha256": target_hash,
            "minibatch_schedule_sha256": batch_hash,
        },
    )
    frontend_checkpoint = output_dir / "frontend.pt"
    _save_uncalibrated_frontend(system, frontend_checkpoint)
    system.to(device)
    return system, {
        "source": "new_matched_io_relation_task_only_training",
        "training_seconds": training_seconds,
        "training_history": history,
        "initial_model_state_sha256": initial_model,
        "initial_system_state_sha256": initial_system,
        "final_model_state_sha256": final_model,
        "final_system_state_sha256": final_system,
        "matched_base_sha256": base_hash,
        "target_sha256": target_hash,
        "minibatch_schedule_sha256": batch_hash,
        "checkpoint": str(checkpoint),
        "frontend_checkpoint": str(frontend_checkpoint),
        "objective": f"ordinary {spec['target']}-cosine interval cross-entropy only",
    }


def _source_paths(source_root: Path, condition: str, seed: int) -> tuple[Path, Path, Path]:
    source_condition = CALIBRATED_CONDITIONS[condition]
    root = source_root / "runs" / source_condition / f"seed_{seed}"
    return root / "result.json", root / "model.pt", root / "frontend.pt"


def verify_calibrated_source(
    task: CircleTaskConfig,
    config: IOCorrespondenceConfig,
    condition: str,
    seed: int,
    source_root: Path,
) -> Dict[str, Any]:
    detail_path, checkpoint, frontend_checkpoint = _source_paths(
        source_root, condition, seed
    )
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    source_condition = CALIBRATED_CONDITIONS[condition]
    if (
        detail.get("schema_version") != SOURCE_SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("condition") != source_condition
        or int(detail.get("seed", -1)) != seed
        or detail.get("implementation_sha256")
        != _file_digest(Path(calibrated_source.__file__))
        or not checkpoint.is_file()
        or not frontend_checkpoint.is_file()
    ):
        raise ValueError(f"invalid calibrated source cell: {detail_path}")
    expected_model = TinyLLMModel(
        calibrated_source._model_config(config.preset, task, seed),
        name="source-verification",
    )
    expected_initial = _state_digest(expected_model)
    del expected_model
    _, _, base_hash, _, batch_hash = _protocol_material(
        task, config, seed, "absolute"
    )
    training = detail["training"]
    if (
        training["initial_model_state_sha256"] != expected_initial
        or training["training_data_sha256"] != base_hash
        or training["minibatch_schedule_sha256"] != batch_hash
    ):
        raise ValueError(f"calibrated source is not matched: {detail_path}")
    return detail


def load_calibrated_source(
    task: CircleTaskConfig,
    config: IOCorrespondenceConfig,
    condition: str,
    seed: int,
    source_root: Path,
    device: torch.device,
) -> tuple[CalibratedTinyLLM, Dict[str, Any]]:
    detail = verify_calibrated_source(task, config, condition, seed, source_root)
    detail_path, checkpoint, frontend_checkpoint = _source_paths(
        source_root, condition, seed
    )
    model = TinyLLMModel.from_checkpoint(checkpoint, map_location="cpu")
    system = CalibratedTinyLLM(
        model, CALIBRATED_CONDITIONS[condition], task, config  # type: ignore[arg-type]
    )
    frontend = torch.load(frontend_checkpoint, map_location="cpu", weights_only=True)
    if system.calibration_embedding is not None:
        system.calibration_embedding.load_state_dict(frontend["calibration_embedding"])
    if system.scalar_embedding is not None:
        system.scalar_embedding.load_state_dict(frontend["scalar_embedding"])
    if system.encoder is not None:
        system.encoder.load_state_dict(frontend["encoder"])
    if (
        _state_digest(model) != detail["training"]["final_model_state_sha256"]
        or _module_digest(system) != detail["training"]["final_system_state_sha256"]
    ):
        raise ValueError(f"calibrated source checkpoint digest mismatch: {detail_path}")
    system.to(device)
    return system, {
        **detail["training"],
        "source": "reused_frozen_calibrated_positive_control",
        "source_result": str(detail_path),
        "source_result_sha256": _file_digest(detail_path),
        "source_schema_version": detail["schema_version"],
    }


@torch.no_grad()
def extract_features(
    system: nn.Module,
    condition: str,
    dataset: CalibratedDataset,
    task: CircleTaskConfig,
    config: IOCorrespondenceConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    system.eval()
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    captured: Dict[str, List[torch.Tensor]] = {cut: [] for cut in CUTS}
    for start in range(0, len(sensor), config.activation_batch_size):
        stop = start + config.activation_batch_size
        cuts = _forward_cuts(
            system,
            condition,
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        for cut, value in cuts.items():
            captured[cut].append(value.detach().cpu())
    return {cut: torch.cat(parts).float().numpy() for cut, parts in captured.items()}


@torch.no_grad()
def task_metrics(
    system: nn.Module,
    condition: str,
    dataset: CalibratedDataset,
    task: CircleTaskConfig,
    config: IOCorrespondenceConfig,
    device: torch.device,
) -> Dict[str, float]:
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    probabilities = []
    for start in range(0, len(sensor), config.activation_batch_size):
        stop = start + config.activation_batch_size
        residual = _forward_cuts(
            system,
            condition,
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )["full"]
        probabilities.append(
            torch.softmax(_task_logits(_model(system), residual, answer_ids), -1).cpu()
        )
    predicted = torch.cat(probabilities)
    target = dataset.paired.circle.target_posteriors
    centers = torch.linspace(-1.0, 1.0, task.phase_bins)
    predicted_value = predicted @ centers
    target_value = dataset.paired.fiber.cosine
    return {
        "exact_bin_accuracy": float(
            (predicted.argmax(1) == target.argmax(1)).float().mean()
        ),
        "mean_absolute_cosine_error": float((predicted_value - target_value).abs().mean()),
        "mean_target_cross_entropy": float(
            -(target * predicted.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def _standardized_pca(
    train: np.ndarray, evaluation: np.ndarray, components: int, seed: int
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    deviation = train.std(axis=0, dtype=np.float64).astype(np.float32)
    deviation = np.maximum(deviation, 1e-5)
    train_normalized = (train - mean) / deviation
    evaluation_normalized = (evaluation - mean) / deviation
    count = min(components, train_normalized.shape[0] - 1, train_normalized.shape[1])
    if count < train_normalized.shape[1]:
        pca = PCA(n_components=count, svd_solver="randomized", random_state=seed)
        train_projected = pca.fit_transform(train_normalized)
        evaluation_projected = pca.transform(evaluation_normalized)
        explained = float(pca.explained_variance_ratio_.sum())
    else:
        train_projected = train_normalized
        evaluation_projected = evaluation_normalized
        explained = 1.0
    return train_projected, evaluation_projected, {
        "input_dimensions": int(train.shape[1]),
        "retained_dimensions": int(count),
        "explained_variance_ratio": explained,
    }


def mapper_summary(
    projected: np.ndarray,
    fiber: FiberDataset,
    config: IOCorrespondenceConfig,
) -> Dict[str, Any]:
    lens = fiber.cosine.numpy()
    branch = fiber.branch.numpy().astype(int)
    left, right = -0.95, 0.95
    cover_count = config.mapper_cover_intervals
    width = (right - left) / (cover_count - (cover_count - 1) * config.mapper_overlap)
    step = width * (1.0 - config.mapper_overlap)
    nodes: List[Dict[str, Any]] = []
    cover_summaries = []
    covered: set[int] = set()
    for cover_index in range(cover_count):
        start = left + cover_index * step
        stop = start + width
        indices = np.flatnonzero((lens >= start) & (lens <= stop))
        components_for_cover: List[np.ndarray] = []
        if len(indices) >= 2:
            neighbors = min(config.mapper_neighbors, len(indices) - 1)
            graph = NearestNeighbors(n_neighbors=neighbors + 1).fit(projected[indices])
            neighbor_indices = graph.kneighbors(
                projected[indices], n_neighbors=neighbors + 1, return_distance=False
            )[:, 1:]
            rows = np.repeat(np.arange(len(indices)), neighbor_indices.shape[1])
            cols = neighbor_indices.reshape(-1)
            adjacency = csr_matrix(
                (np.ones_like(rows), (rows, cols)), shape=(len(indices), len(indices))
            )
            adjacency = adjacency.maximum(adjacency.T)
            count, labels = connected_components(adjacency, directed=False)
            for label in range(count):
                members = indices[labels == label]
                if len(members) >= config.mapper_min_cluster:
                    components_for_cover.append(members)
        purities = []
        majorities = []
        for members in components_for_cover:
            fraction = float(branch[members].mean())
            purities.append(max(fraction, 1.0 - fraction))
            majorities.append(int(fraction >= 0.5))
            node_id = len(nodes)
            nodes.append(
                {
                    "id": node_id,
                    "cover": cover_index,
                    "members": set(int(item) for item in members),
                }
            )
            covered.update(int(item) for item in members)
        two_sheet = (
            len(components_for_cover) == 2
            and all(value >= 0.8 for value in purities)
            and len(set(majorities)) == 2
        )
        cover_summaries.append(
            {
                "cover": cover_index,
                "sample_count": int(len(indices)),
                "node_count": len(components_for_cover),
                "branch_purities": purities,
                "two_branch_sheets": two_sheet,
            }
        )
    edges = set()
    for first in range(len(nodes)):
        for second in range(first + 1, len(nodes)):
            if abs(nodes[first]["cover"] - nodes[second]["cover"]) != 1:
                continue
            if nodes[first]["members"].intersection(nodes[second]["members"]):
                edges.add((first, second))
    if nodes:
        rows = [item for edge in edges for item in edge]
        cols = [item for edge in edges for item in reversed(edge)]
        adjacency = csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes))
        )
        graph_components = int(connected_components(adjacency, directed=False)[0])
    else:
        graph_components = 0
    cycle_rank = max(0, len(edges) - len(nodes) + graph_components)
    interior = cover_summaries[1:-1]
    single_fraction = float(
        np.mean([item["node_count"] == 1 for item in interior])
    )
    two_sheet_fraction = float(
        np.mean([item["two_branch_sheets"] for item in interior])
    )
    coverage = len(covered) / max(1, len(lens))
    interval_like = (
        single_fraction >= 0.75
        and coverage >= 0.90
        and graph_components == 1
        and cycle_rank == 0
    )
    return {
        "scope": "finite-sample target-lens Mapper approximation",
        "cover_intervals": cover_count,
        "cover_overlap": config.mapper_overlap,
        "neighbors": config.mapper_neighbors,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "connected_components": graph_components,
        "cycle_rank": cycle_rank,
        "sample_coverage": coverage,
        "interior_single_sheet_fraction": single_fraction,
        "interior_two_branch_sheet_fraction": two_sheet_fraction,
        "interval_like": interval_like,
        "covers": cover_summaries,
    }


def paired_map_distortion(
    projected: np.ndarray, fiber: FiberDataset
) -> Dict[str, float]:
    ids = fiber.fiber_id.numpy()
    target = fiber.cosine.numpy()
    unique = np.unique(ids)
    pairs = [np.flatnonzero(ids == item) for item in unique]
    if any(len(indices) != 2 for indices in pairs):
        raise ValueError("paired distortion requires exactly two samples per fiber")
    centroids = np.stack([projected[indices].mean(axis=0) for indices in pairs])
    residuals = np.stack(
        [(projected[indices[0]] - projected[indices[1]]) / math.sqrt(2.0) for indices in pairs]
    )
    targets = np.array([target[indices[0]] for indices in pairs])
    scale = residuals.std(axis=0, dtype=np.float64)
    reference = max(1e-8, float(np.median(np.abs(scale[scale > 1e-12]))))
    scale = np.maximum(scale, 1e-3 * reference)
    whitened = centroids / scale
    target_distance = pdist(targets[:, None], metric="euclidean")
    whitened_distance = pdist(whitened, metric="euclidean")
    euclidean_distance = pdist(centroids, metric="euclidean")
    cosine_distance = np.nan_to_num(
        pdist(centroids, metric="cosine"), nan=0.0, posinf=1.0, neginf=0.0
    )

    def normalize(values: np.ndarray) -> np.ndarray:
        denominator = max(1e-8, float(np.quantile(values, 0.95)))
        return values / denominator

    target_normalized = normalize(target_distance)

    def metrics(values: np.ndarray) -> tuple[float, float]:
        normalized = normalize(values)
        correlation = float(spearmanr(target_distance, values).statistic)
        if not np.isfinite(correlation):
            correlation = 0.0
        return float(np.median(np.abs(target_normalized - normalized))), correlation

    task_distortion, task_order = metrics(whitened_distance)
    euclidean_distortion, euclidean_order = metrics(euclidean_distance)
    cosine_distortion, cosine_order = metrics(cosine_distance)
    return {
        "fiber_averaged_whitened_distortion": task_distortion,
        "fiber_averaged_whitened_order_spearman": task_order,
        "euclidean_control_distortion": euclidean_distortion,
        "euclidean_control_order_spearman": euclidean_order,
        "cosine_control_distortion": cosine_distortion,
        "cosine_control_order_spearman": cosine_order,
        "fiber_count": int(len(unique)),
    }


def analyze(
    system: nn.Module,
    condition: str,
    task: CircleTaskConfig,
    config: IOCorrespondenceConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    target_kind = condition_spec(condition)["target"]
    datasets = {
        "train": generate_relation_fiber_dataset(
            task,
            target_kind=target_kind,
            sample_count=config.probe_train_samples,
            seed=config.analysis_seed + 101,
        ),
        "validation": generate_relation_fiber_dataset(
            task,
            target_kind=target_kind,
            sample_count=config.probe_validation_samples,
            seed=config.analysis_seed + 211,
        ),
        "in_distribution": generate_relation_fiber_dataset(
            task,
            target_kind=target_kind,
            sample_count=config.probe_test_samples,
            seed=config.analysis_seed + 307,
        ),
        "composition": generate_relation_fiber_dataset(
            task,
            target_kind=target_kind,
            sample_count=config.probe_test_samples,
            seed=config.analysis_seed + 1_316,
            regime="composition",
        ),
        "extrapolation": generate_relation_fiber_dataset(
            task,
            target_kind=target_kind,
            sample_count=config.probe_test_samples,
            seed=config.analysis_seed + 2_325,
            regime="extrapolation",
        ),
    }
    features = {
        name: extract_features(system, condition, dataset, task, config, device)
        for name, dataset in datasets.items()
    }
    probe_config = _probe_config(config)  # type: ignore[arg-type]
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
        map_geometry = {}
        for regime in PRIMARY_REGIMES:
            _, projected, projection = _standardized_pca(
                features["train"][cut],
                features[regime][cut],
                config.mapper_components,
                config.analysis_seed + seed * 1_003 + cut_index * 101,
            )
            map_geometry[regime] = {
                "projection": projection,
                "mapper": mapper_summary(
                    projected, datasets[regime].paired.fiber, config
                ),
                "paired_distortion": paired_map_distortion(
                    projected, datasets[regime].paired.fiber
                ),
            }
        cuts[cut] = {"probe": probe, "io_map_geometry": map_geometry}
    return {
        "cuts": cuts,
        "task_metrics": {
            regime: task_metrics(
                system, condition, datasets[regime], task, config, device
            )
            for regime in REGIMES
        },
    }


def _config_from_mapping(values: Mapping[str, Any]) -> IOCorrespondenceConfig:
    converted = dict(values)
    for field in ("seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return IOCorrespondenceConfig(**converted)


def _scientific_fingerprint(experiment: Experiment) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "configuration": experiment.parameters["configuration"],
        "task_config": experiment.parameters["task_config"],
        "condition": experiment.parameters["condition"],
        "seed": experiment.seed,
        "implementation_sha256": experiment.parameters["implementation_sha256"],
        "io_relation_contract": experiment.parameters["io_relation_contract"],
        "source_result_sha256": experiment.parameters.get("source_result_sha256"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def io_correspondence_worker(
    experiment: Experiment, device_id: int
) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    if _implementation_digest() != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    contract = io_relation_contract()
    if contract != experiment.parameters["io_relation_contract"]:
        raise RuntimeError("I/O relation contract changed after campaign construction")
    output_dir = _condition_directory(
        Path(experiment.parameters["output_dir"]), condition, seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    if condition in CALIBRATED_CONDITIONS and config.reuse_calibrated_controls:
        system, training = load_calibrated_source(
            task,
            config,
            condition,
            seed,
            Path(experiment.parameters["calibrated_source_root"]),
            device,
        )
    elif condition in CALIBRATED_CONDITIONS:
        source_condition = CALIBRATED_CONDITIONS[condition]
        system, training = calibrated_source.train_cell(
            task, config, source_condition, seed, device, output_dir  # type: ignore[arg-type]
        )
        training["source"] = "shakedown_new_calibrated_training"
    else:
        system, training = train_uncalibrated(
            task, config, condition, seed, device, output_dir
        )
    analysis = analyze(system, condition, task, config, seed, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "completed_at": _utc_now(),
        "condition": condition,
        "condition_spec": condition_spec(condition),
        "seed": seed,
        "device": str(device),
        "model_parameters": sum(
            parameter.numel() for parameter in _model(system).parameters()
        ),
        "system_parameters": sum(parameter.numel() for parameter in system.parameters()),
        "training": training,
        "analysis": analysis,
        "io_relation_contract": contract,
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
        primary_metric=min(
            metrics["full_composition_cosine"],
            metrics["full_extrapolation_cosine"],
        ),
        model_architecture=[
            task.vocab_size,
            *([_model(system).config.n_embd] * _model(system).config.n_layer),
            task.phase_bins,
        ],
        model_parameters=detail["system_parameters"],
        training_time=elapsed,
        training_history=training.get("training_history", []),
        model_checkpoint=training["checkpoint"],
        observations=[f"detail={detail_path}", f"condition={condition}"],
    )


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: IOCorrespondenceConfig
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
                cell: Dict[str, Any] = {
                    "cosine_pearson_mean": float(
                        np.mean([value["cosine_pearson"] for value in values])
                    ),
                    "branch_balanced_accuracy_mean": float(
                        np.mean([value["balanced_accuracy"] for value in values])
                    ),
                    "conditional_log_loss_gain_mean": float(
                        np.mean(
                            [
                                value["conditional_log_loss_gain_over_cosine_only"]
                                for value in values
                            ]
                        )
                    ),
                    "pass_count": int(sum(passes)),
                    "pass_by_seed": {
                        str(item["seed"]): passed
                        for item, passed in zip(selected, passes)
                    },
                }
                if regime in PRIMARY_REGIMES:
                    geometries = [
                        item["analysis"]["cuts"][cut]["io_map_geometry"][regime]
                        for item in selected
                    ]
                    cell["mapper_interval_like_count"] = int(
                        sum(value["mapper"]["interval_like"] for value in geometries)
                    )
                    cell["mapper_single_sheet_fraction_mean"] = float(
                        np.mean(
                            [
                                value["mapper"]["interior_single_sheet_fraction"]
                                for value in geometries
                            ]
                        )
                    )
                    cell["paired_whitened_distortion_mean"] = float(
                        np.mean(
                            [
                                value["paired_distortion"][
                                    "fiber_averaged_whitened_distortion"
                                ]
                                for value in geometries
                            ]
                        )
                    )
                    cell["paired_whitened_order_spearman_mean"] = float(
                        np.mean(
                            [
                                value["paired_distortion"][
                                    "fiber_averaged_whitened_order_spearman"
                                ]
                                for value in geometries
                            ]
                        )
                    )
                cuts[cut][regime] = cell
        joint_by_seed = {}
        for item in selected:
            seed = str(item["seed"])
            cells = {
                f"{cut}:{regime}": cuts[cut][regime]["pass_by_seed"][seed]
                for cut in CUTS
                for regime in PRIMARY_REGIMES
            }
            joint_by_seed[seed] = {"cells": cells, "joint_pass": all(cells.values())}
        task = {
            regime: float(
                np.mean(
                    [
                        item["analysis"]["task_metrics"][regime]["exact_bin_accuracy"]
                        for item in selected
                    ]
                )
            )
            for regime in REGIMES
        }
        joint_count = int(sum(value["joint_pass"] for value in joint_by_seed.values()))
        arms[condition] = {
            "cuts": cuts,
            "task_exact_bin_accuracy_mean": task,
            "joint_pass_by_seed": joint_by_seed,
            "joint_pass_count": joint_count,
            "success": joint_count >= required,
        }
    task_controls = {}
    for learned, raw in (
        ("calibrated_absolute_equivariant", "calibrated_absolute_raw"),
        ("uncalibrated_relative_equivariant", "uncalibrated_relative_raw"),
    ):
        gaps = {
            regime: arms[learned]["task_exact_bin_accuracy_mean"][regime]
            - arms[raw]["task_exact_bin_accuracy_mean"][regime]
            for regime in PRIMARY_REGIMES
        }
        task_controls[learned] = {
            "learned_minus_raw": gaps,
            "passed": all(value >= -0.03 for value in gaps.values()),
        }
    calibrated_success = (
        arms["calibrated_absolute_analytic"]["success"]
        and arms["calibrated_absolute_equivariant"]["success"]
        and task_controls["calibrated_absolute_equivariant"]["passed"]
    )
    relative_success = (
        arms["uncalibrated_relative_equivariant"]["success"]
        and task_controls["uncalibrated_relative_equivariant"]["passed"]
    )
    absolute_failure = not arms["uncalibrated_absolute_equivariant"]["success"]
    if calibrated_success and relative_success and absolute_failure:
        conclusion = "both_descent_repairs_realize_stable_io_quotients"
    elif calibrated_success and not relative_success:
        conclusion = "observation_repair_succeeds_target_side_repair_fails"
    elif relative_success and not calibrated_success:
        conclusion = "target_side_repair_succeeds_observation_repair_fails"
    else:
        conclusion = "descent_repaired_but_stable_realization_not_confirmed"
    return {
        "arms": arms,
        "task_controls": task_controls,
        "preregistered_gate": {
            "required_seed_passes": required,
            "cosine_threshold": 0.90,
            "branch_accuracy_ceiling": 0.55,
            "conditional_log_loss_gain_ceiling": 0.02,
            "cuts": list(CUTS),
            "regimes": list(PRIMARY_REGIMES),
            "task_accuracy_gap_floor": -0.03,
        },
        "prediction_checks": {
            "calibrated_structured_success": calibrated_success,
            "relative_equivariant_success": relative_success,
            "uncalibrated_absolute_global_failure": absolute_failure,
        },
        "conclusion": conclusion,
    }


def _experiments(
    config: IOCorrespondenceConfig,
    task: CircleTaskConfig,
    output_dir: Path,
    calibrated_source_root: Path,
) -> List[Experiment]:
    contract = io_relation_contract()
    common = {
        "configuration": asdict(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "calibrated_source_root": str(calibrated_source_root),
        "implementation_sha256": _implementation_digest(),
        "io_relation_contract": contract,
    }
    experiments = []
    for condition in CONDITIONS:
        for seed in config.seeds:
            parameters = {**common, "condition": condition, "seed": seed}
            if condition in CALIBRATED_CONDITIONS and config.reuse_calibrated_controls:
                detail_path, _, _ = _source_paths(calibrated_source_root, condition, seed)
                verify_calibrated_source(
                    task, config, condition, seed, calibrated_source_root
                )
                parameters["source_result_sha256"] = _file_digest(detail_path)
            experiments.append(
                Experiment(
                    id=f"tinyllm-io-correspondence-{condition}-seed{seed}",
                    hypothesis_id=HYPOTHESIS_ID,
                    name=f"TinyLLM I/O correspondence {condition} seed {seed}",
                    parameters=parameters,
                    seed=seed,
                )
            )
    return experiments


def _existing_detail(
    experiment: Experiment, output_dir: Path
) -> Optional[Dict[str, Any]]:
    path = _condition_directory(
        output_dir, str(experiment.parameters["condition"]), int(experiment.seed)
    ) / "result.json"
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
    return detail if checkpoint.is_file() else None


async def run_campaign(
    config: IOCorrespondenceConfig,
    task: CircleTaskConfig,
    output_dir: Path,
    calibrated_source_root: Path,
) -> Dict[str, Any]:
    contract = io_relation_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = _experiments(config, task, output_dir, calibrated_source_root)
    existing = {
        experiment.id: detail
        for experiment in experiments
        if (detail := _existing_detail(experiment, output_dir)) is not None
    }
    pending = [experiment for experiment in experiments if experiment.id not in existing]
    lab = LabConfig(
        project_name="tinyllm_io_correspondence",
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
    runner = AsyncExperimentRunner(lab, io_correspondence_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful = {result.experiment_id for result in results if result.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id in successful:
            path = _condition_directory(
                output_dir,
                str(experiment.parameters["condition"]),
                int(experiment.seed),
            ) / "result.json"
            details.append(json.loads(path.read_text(encoding="utf-8")))
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if len(details) == len(experiments) else "partial",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "implementation_sha256": _implementation_digest(),
        "io_relation_contract": contract,
        "calibrated_source_root": str(calibrated_source_root),
        "summary": {
            "requested": len(experiments),
            "reused": len(existing),
            "scheduled": len(pending),
            "completed": len(details),
            "failed": len(experiments) - len(details),
            "frozen_source_cells": sum(
                item["training"].get("source")
                == "reused_frozen_calibrated_positive_control"
                for item in details
            ),
            "new_training_cells": sum(
                item["training"].get("source")
                != "reused_frozen_calibrated_positive_control"
                for item in details
            ),
        },
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value for key, value in runner.slot_plan.slots_by_device.items()
            },
            "calibration": runner.slot_plan.calibration,
        },
        "aggregates": (
            aggregate_details(details, config)
            if len(details) == len(experiments)
            else {}
        ),
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
            "Previously inspected calibrated endpoints are retained evidence, not fresh confirmatory outcomes.",
            "Mapper is a finite-sample cover-and-kNN approximation, not a certified Reeb cosheaf.",
            "Fresh nonlinear probes test conditional decodability rather than certified mutual information.",
            "Paired distortion uses a declared fiber-averaged diagonal-whitened representation metric.",
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
        "--calibrated-source",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calibrated_frontend_causal/"
            "20260806_d8_preregistered"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_io_correspondence/"
            "20260806_d8_preregistered"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reuse_calibrated = True
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
        reuse_calibrated = False
    config = IOCorrespondenceConfig(
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
        reuse_calibrated_controls=reuse_calibrated,
        allow_underpowered=args.allow_underpowered,
    )
    task = CircleTaskConfig(train_samples=config.train_samples)
    bundle = asyncio.run(
        run_campaign(config, task, args.output, args.calibrated_source)
    )
    print(
        json.dumps(
            {
                "summary": bundle["summary"],
                "conclusion": bundle.get("aggregates", {}).get("conclusion"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
