#!/usr/bin/env python3
"""Separate relative-vector carrier recovery from a fixed scalar quotient."""

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
import statistics
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_io_correspondence as io_v1
from experiments.structure_net.tinyllm_conditional_branch_depth_scan import (
    _add_log_loss_gains,
    fit_conditional_probe,
    fit_cosine_only_null,
)
from experiments.structure_net.tinyllm_invariant_frontend_causal import (
    _probe_config,
    _task_logits,
    decode_sensor_tokens,
    endpoint_pass,
)
from experiments.structure_net.tinyllm_io_correspondence_v2 import (
    CorrectedUncalibratedTinyLLM,
    EquivariantVectorEncoder,
)
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    _nuisance_values,
    _sample_directions,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
)
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.analyzers import circular_winding_degree
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-relative-carrier-factorization.v1"
HYPOTHESIS_ID = "tinyllm-relative-carrier-fixed-quotient-v1"
CONDITIONS = (
    "analytic_fixed_quotient",
    "learned_vector_carrier",
    "learned_vector_fixed_quotient",
    "scalar_only_baseline",
)
CUTS = ("frontend", "post_attention", "post_mlp", "full")
REGIMES = ("in_distribution", "composition", "extrapolation")
PRIMARY_REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class RelativeCarrierConfig:
    preset: str = "d8"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    training_steps: int = 600
    carrier_steps: int = 600
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
    probe_steps: int = 240
    probe_width: int = 128
    probe_batch_size: int = 256
    activation_batch_size: int = 256
    validation_interval: int = 20
    early_stopping_patience: int = 5
    degree_phase_points: int = 512
    carrier_alignment_threshold: float = 0.95
    carrier_mse_ceiling: float = 0.02
    equivariance_error_ceiling: float = 2e-5
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if self.preset != "d8" and not self.allow_underpowered:
            raise ValueError("confirmatory campaign fixes d8")
        if min(self.training_steps, self.carrier_steps, self.train_samples) < 1:
            raise ValueError("training sizes must be positive")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch size must be positive and even")


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


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_digest(module: nn.Module) -> str:
    return _tensor_digest(*(value for _, value in sorted(module.state_dict().items())))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for module in (Path(__file__), Path(io_v1.__file__), Path(calibrated.__file__)):
        digest.update(str(module).encode())
        digest.update(module.read_bytes())
    return digest.hexdigest()


class AnalyticRelativeVectorCarrier:
    """Observation-only sinusoid fit returning the complete observed-angle vector."""

    def __init__(self, task: CircleTaskConfig):
        self.task = task
        self.speeds = np.linspace(0.15, 0.55, 81, dtype=np.float64)
        history = np.arange(task.sensor_steps, dtype=np.float64) - (task.sensor_steps - 1)
        self.designs = np.asarray(
            [
                np.column_stack(
                    (np.ones_like(history), history, np.cos(speed * history), np.sin(speed * history))
                )
                for speed in self.speeds
            ]
        )
        self.pseudoinverses = np.asarray([np.linalg.pinv(item) for item in self.designs])

    def __call__(self, sensor: torch.Tensor) -> torch.Tensor:
        planar = sensor[..., :2].detach().cpu().double().numpy()
        coefficients = np.einsum("gkt,ntc->ngkc", self.pseudoinverses, planar)
        fitted = np.einsum("gtk,ngkc->ngtc", self.designs, coefficients)
        chosen = np.square(fitted - planar[:, None]).mean(axis=(2, 3)).argmin(axis=1)
        coeff = coefficients[np.arange(len(planar)), chosen]
        cx_cos, cy_cos = coeff[:, 2, 0], coeff[:, 2, 1]
        cx_sin, cy_sin = coeff[:, 3, 0], coeff[:, 3, 1]
        plus_error = np.square(cx_sin + cy_cos) + np.square(cy_sin - cx_cos)
        minus_error = np.square(cx_sin - cy_cos) + np.square(cy_sin + cx_cos)
        positive = plus_error <= minus_error
        endpoint_x = np.where(positive, 0.5 * (cx_cos + cy_sin), 0.5 * (cx_cos - cy_sin))
        endpoint_y = np.where(positive, 0.5 * (cy_cos - cx_sin), 0.5 * (cy_cos + cx_sin))
        endpoint = np.column_stack((endpoint_x, endpoint_y))
        endpoint /= np.linalg.norm(endpoint, axis=1, keepdims=True).clip(1e-8)
        direction = np.where(positive, 1.0, -1.0)
        delta = direction * self.task.future_delta
        cosine, sine = np.cos(delta), np.sin(delta)
        future = np.column_stack(
            (endpoint[:, 0] * cosine - endpoint[:, 1] * sine,
             endpoint[:, 0] * sine + endpoint[:, 1] * cosine)
        )
        return torch.from_numpy(future.astype(np.float32)).to(sensor.device)


class FixedQuotientTinyLLM(nn.Module):
    def __init__(
        self,
        model: TinyLLMModel,
        task: CircleTaskConfig,
        config: RelativeCarrierConfig,
        carrier: Optional[EquivariantVectorEncoder],
    ):
        super().__init__()
        self.model = model
        self.carrier = carrier
        self.analytic = AnalyticRelativeVectorCarrier(task) if carrier is None else None
        self.scalar_embedding = nn.Linear(1, model.config.n_embd)

    def vector(self, sensor: torch.Tensor) -> torch.Tensor:
        if self.carrier is not None:
            return self.carrier(sensor)
        if self.analytic is None:
            raise AssertionError("analytic carrier missing")
        return self.analytic(sensor)

    def forward_cuts(self, input_ids: torch.Tensor, sensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        quotient = self.vector(sensor)[:, :1]
        bos = self.model.transformer["wte"](input_ids[:, :1])
        query = self.model.transformer["wte"](input_ids[:, -1:])
        value = torch.cat((bos, self.scalar_embedding(quotient)[:, None, :], query), dim=1)
        return io_v1._transformer_cuts(self.model, value, quotient)


def _relative_vector(dataset: calibrated.CalibratedDataset) -> torch.Tensor:
    phase = dataset.paired.fiber.phase
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def _training_material(
    task: CircleTaskConfig, config: RelativeCarrierConfig, seed: int
) -> tuple[calibrated.CalibratedDataset, torch.Tensor, torch.Tensor, str, str]:
    proxy = io_v1.IOCorrespondenceConfig(
        seeds=(seed,), training_steps=config.training_steps,
        train_samples=config.train_samples, batch_size=config.batch_size,
        allow_underpowered=True, reuse_calibrated_controls=False,
    )
    dataset, batches, base_hash, target_hash, batch_hash = io_v1._protocol_material(
        task, proxy, seed, "relative"
    )
    return dataset, batches, _relative_vector(dataset), base_hash, _tensor_digest(batches)


def _train_carrier(
    task: CircleTaskConfig,
    config: RelativeCarrierConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[EquivariantVectorEncoder, Dict[str, Any]]:
    torch.manual_seed(seed + 31_337)
    carrier = EquivariantVectorEncoder(task.sensor_steps, config.vector_channels).to(device)
    dataset, batches, targets, base_hash, batch_hash = _training_material(task, config, seed)
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    optimizer = torch.optim.AdamW(
        carrier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    carrier.train()
    for step in range(1, config.carrier_steps + 1):
        pairs = batches[(step - 1) % len(batches)]
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        predicted = carrier(sensor[indices].to(device))
        loss = torch.square(predicted - targets[indices].to(device)).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(carrier.parameters(), config.gradient_clip)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.carrier_steps:
            history.append({"step": step, "vector_mse": float(loss.detach()), "gradient_norm": float(norm)})
    checkpoint = output_dir / "carrier.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "state_dict": carrier.to("cpu").state_dict(),
            "matched_base_sha256": base_hash,
            "minibatch_schedule_sha256": batch_hash,
        },
        checkpoint,
    )
    carrier.to(device)
    return carrier, {
        "carrier_checkpoint": str(checkpoint),
        "carrier_state_sha256": _module_digest(carrier),
        "carrier_history": history,
        "matched_base_sha256": base_hash,
        "minibatch_schedule_sha256": batch_hash,
        "carrier_objective": "direct complete relative-vector coordinate MSE",
    }


def _train_downstream(
    condition: str,
    task: CircleTaskConfig,
    config: RelativeCarrierConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
    carrier: Optional[EquivariantVectorEncoder] = None,
) -> tuple[nn.Module, Dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyLLMModel(calibrated._model_config(config.preset, task, seed)).to(device)
    if condition == "scalar_only_baseline":
        system: nn.Module = CorrectedUncalibratedTinyLLM(model, "equivariant", task, config)  # type: ignore[arg-type]
    else:
        system = FixedQuotientTinyLLM(model, task, config, carrier)
        if carrier is not None:
            for parameter in carrier.parameters():
                parameter.requires_grad_(False)
            carrier.eval()
    system.to(device)
    dataset, batches, _, base_hash, batch_hash = _training_material(task, config, seed)
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    parameters = [item for item in system.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    system.train()
    for step in range(1, config.training_steps + 1):
        pairs = batches[step - 1]
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        inputs = dataset.paired.circle.input_ids[indices].to(device)
        observed = sensor[indices].to(device)
        targets = dataset.paired.circle.target_posteriors[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        residual = system.forward_cuts(inputs, observed)["full"]  # type: ignore[attr-defined]
        logits = _task_logits(model, residual, answer_ids)
        loss = -(targets * torch.log_softmax(logits, -1)).sum(-1).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            history.append({"step": step, "task_loss": float(loss.detach()), "gradient_norm": float(norm)})
    checkpoint = output_dir / "model.pt"
    model.to("cpu").save_checkpoint(
        checkpoint,
        metadata={
            "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
            "condition": condition, "seed": seed, "matched_base_sha256": base_hash,
            "minibatch_schedule_sha256": batch_hash,
        },
    )
    model.to(device)
    frontend = output_dir / "frontend.pt"
    payload: Dict[str, Any] = {"condition": condition}
    if isinstance(system, FixedQuotientTinyLLM):
        payload["scalar_embedding"] = system.scalar_embedding.to("cpu").state_dict()
        payload["carrier"] = system.carrier.to("cpu").state_dict() if system.carrier is not None else None
        system.to(device)
    else:
        payload["scalar_embedding"] = system.scalar_embedding.to("cpu").state_dict()  # type: ignore[attr-defined]
        payload["carrier"] = system.encoder.to("cpu").state_dict()  # type: ignore[attr-defined]
        system.to(device)
    torch.save(payload, frontend)
    return system, {
        "checkpoint": str(checkpoint), "frontend_checkpoint": str(frontend),
        "training_history": history, "final_model_state_sha256": _state_digest(model),
        "matched_base_sha256": base_hash, "minibatch_schedule_sha256": batch_hash,
        "objective": "fixed x-quotient task loss" if condition != "scalar_only_baseline" else "end-to-end scalar task loss on complete equivariant vector",
    }


def _carrier_for(condition: str, system: Optional[nn.Module], carrier: Optional[EquivariantVectorEncoder], task: CircleTaskConfig):
    if condition == "analytic_fixed_quotient":
        return AnalyticRelativeVectorCarrier(task)
    if condition in {"learned_vector_carrier", "learned_vector_fixed_quotient"}:
        if carrier is None:
            raise AssertionError("learned carrier missing")
        return carrier
    if system is None:
        raise AssertionError("scalar system missing")
    return system.encoder  # type: ignore[attr-defined]


def _phase_loop(
    task: CircleTaskConfig, regime: str, seed: int, points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    relative = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
    one = _nuisance_values("N3", regime, 1, np.random.default_rng(seed))
    values = {key: np.repeat(value, points, axis=0) for key, value in one.items()}
    orientation = np.asarray(values["orientation"]).reshape(-1)
    absolute = (relative - orientation) % (2.0 * math.pi)
    directions = np.ones(points)
    current = (absolute - task.future_delta) % (2.0 * math.pi)
    inputs = io_v1._serialize_with_nuisance(
        task, current, directions, values, np.random.default_rng(seed + 91)
    )
    return decode_sensor_tokens(inputs, task), torch.tensor(relative, dtype=torch.float32)


@torch.no_grad()
def _carrier_metrics(carrier: Any, task: CircleTaskConfig, config: RelativeCarrierConfig, seed: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    carrier_device = (
        next(carrier.parameters()).device
        if isinstance(carrier, nn.Module)
        else torch.device("cpu")
    )
    for regime, offset in (("composition", 1_316), ("extrapolation", 2_325)):
        dataset = io_v1.generate_relation_fiber_dataset(
            task, target_kind="relative", sample_count=config.probe_test_samples,
            seed=config.analysis_seed + offset, regime=regime,
        )
        sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
        predicted = carrier(sensor.to(carrier_device)).float().cpu()
        target = _relative_vector(dataset)
        alignment = float((predicted * target).sum(-1).mean())
        mse = float(torch.square(predicted - target).mean())
        loop_sensor, loop_phase = _phase_loop(
            task, regime, config.analysis_seed + seed * 101 + offset, config.degree_phase_points
        )
        loop = carrier(loop_sensor.to(carrier_device)).float().cpu()
        degree = circular_winding_degree(torch.atan2(loop[:, 1], loop[:, 0]).numpy())
        angle = 0.713
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=sensor.dtype,
        )
        rotated_sensor = sensor.clone()
        rotated_sensor[..., :2] = torch.einsum("ij,btj->bti", rotation, sensor[..., :2])
        rotated_output = carrier(rotated_sensor.to(carrier_device)).float().cpu()
        expected = torch.einsum("ij,bj->bi", rotation, predicted)
        equivariance = float((rotated_output - expected).abs().max())
        passed = (
            alignment >= config.carrier_alignment_threshold
            and mse <= config.carrier_mse_ceiling
            and abs(degree - 1.0) <= 0.1
            and equivariance <= config.equivariance_error_ceiling
        )
        result[regime] = {
            "circular_alignment": alignment, "vector_coordinate_mse": mse,
            "winding_degree": float(degree), "maximum_equivariance_error": equivariance,
            "phase_loop_points": config.degree_phase_points, "passed": passed,
        }
    return result


@torch.no_grad()
def _extract_features(system: nn.Module, dataset: calibrated.CalibratedDataset, task: CircleTaskConfig, config: RelativeCarrierConfig, device: torch.device) -> Dict[str, np.ndarray]:
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    captured: Dict[str, List[torch.Tensor]] = {cut: [] for cut in CUTS}
    system.eval()
    for start in range(0, len(sensor), config.activation_batch_size):
        stop = start + config.activation_batch_size
        cuts = system.forward_cuts(  # type: ignore[attr-defined]
            dataset.paired.circle.input_ids[start:stop].to(device), sensor[start:stop].to(device)
        )
        for cut, value in cuts.items():
            captured[cut].append(value.detach().cpu())
    return {cut: torch.cat(parts).float().numpy() for cut, parts in captured.items()}


def _proxy_config(config: RelativeCarrierConfig) -> io_v1.IOCorrespondenceConfig:
    return io_v1.IOCorrespondenceConfig(
        preset=config.preset, seeds=config.seeds, training_steps=config.training_steps,
        train_samples=config.train_samples, batch_size=config.batch_size,
        probe_train_samples=config.probe_train_samples,
        probe_validation_samples=config.probe_validation_samples,
        probe_test_samples=config.probe_test_samples, probe_steps=config.probe_steps,
        probe_width=config.probe_width, probe_batch_size=config.probe_batch_size,
        activation_batch_size=config.activation_batch_size,
        validation_interval=config.validation_interval,
        early_stopping_patience=config.early_stopping_patience,
        allow_underpowered=config.allow_underpowered, reuse_calibrated_controls=False,
    )


def _downstream_analysis(system: nn.Module, task: CircleTaskConfig, config: RelativeCarrierConfig, seed: int, device: torch.device) -> Dict[str, Any]:
    datasets = {
        "train": io_v1.generate_relation_fiber_dataset(task, target_kind="relative", sample_count=config.probe_train_samples, seed=config.analysis_seed + 101),
        "validation": io_v1.generate_relation_fiber_dataset(task, target_kind="relative", sample_count=config.probe_validation_samples, seed=config.analysis_seed + 211),
        "in_distribution": io_v1.generate_relation_fiber_dataset(task, target_kind="relative", sample_count=config.probe_test_samples, seed=config.analysis_seed + 307),
        "composition": io_v1.generate_relation_fiber_dataset(task, target_kind="relative", sample_count=config.probe_test_samples, seed=config.analysis_seed + 1_316, regime="composition"),
        "extrapolation": io_v1.generate_relation_fiber_dataset(task, target_kind="relative", sample_count=config.probe_test_samples, seed=config.analysis_seed + 2_325, regime="extrapolation"),
    }
    features = {name: _extract_features(system, data, task, config, device) for name, data in datasets.items()}
    proxy = _proxy_config(config)
    probe_config = _probe_config(proxy)  # type: ignore[arg-type]
    evaluation_fibers = {name: datasets[name].paired.fiber for name in REGIMES}
    null = fit_cosine_only_null(
        datasets["train"].paired.fiber, datasets["validation"].paired.fiber,
        evaluation_fibers, probe_config, device, kind="nonlinear",
        seed=config.analysis_seed + seed * 101,
    )
    cuts: Dict[str, Any] = {}
    for index, cut in enumerate(CUTS):
        probe = fit_conditional_probe(
            features["train"][cut], features["validation"][cut],
            {name: features[name][cut] for name in REGIMES},
            datasets["train"].paired.fiber, datasets["validation"].paired.fiber,
            evaluation_fibers, probe_config, device, kind="nonlinear",
            seed=config.analysis_seed + seed * 10_007 + index * 1_009,
        )
        _add_log_loss_gains(probe, null)
        cuts[cut] = {"probe": probe}
    task_metrics: Dict[str, Any] = {}
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    centers = torch.linspace(-1.0, 1.0, task.phase_bins)
    for regime in REGIMES:
        dataset = datasets[regime]
        sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
        probabilities = []
        for start in range(0, len(sensor), config.activation_batch_size):
            stop = start + config.activation_batch_size
            residual = system.forward_cuts(  # type: ignore[attr-defined]
                dataset.paired.circle.input_ids[start:stop].to(device), sensor[start:stop].to(device)
            )["full"]
            probabilities.append(torch.softmax(_task_logits(system.model, residual, answer_ids), -1).cpu())  # type: ignore[attr-defined]
        predicted = torch.cat(probabilities)
        target = dataset.paired.circle.target_posteriors
        task_metrics[regime] = {
            "exact_bin_accuracy": float((predicted.argmax(1) == target.argmax(1)).float().mean()),
            "mean_absolute_cosine_error": float(((predicted @ centers) - dataset.paired.fiber.cosine).abs().mean()),
        }
    return {"cuts": cuts, "task_metrics": task_metrics}


def _condition_directory(root: Path, condition: str, seed: int) -> Path:
    return root / "runs" / condition / f"seed_{seed}"


def _config_from_mapping(values: Mapping[str, Any]) -> RelativeCarrierConfig:
    converted = dict(values)
    for field in ("seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return RelativeCarrierConfig(**converted)


def _fingerprint(experiment: Experiment) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "configuration": experiment.parameters["configuration"],
        "task_config": experiment.parameters["task_config"],
        "condition": experiment.parameters["condition"],
        "seed": experiment.seed,
        "implementation_sha256": experiment.parameters["implementation_sha256"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def relative_carrier_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    condition, seed = str(experiment.parameters["condition"]), int(experiment.seed)
    if _implementation_digest() != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    output = _condition_directory(Path(experiment.parameters["output_dir"]), condition, seed)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    carrier: Optional[EquivariantVectorEncoder] = None
    system: Optional[nn.Module] = None
    training: Dict[str, Any] = {}
    if condition in {"learned_vector_carrier", "learned_vector_fixed_quotient"}:
        carrier, carrier_training = _train_carrier(task, config, seed, device, output)
        training.update(carrier_training)
    if condition != "learned_vector_carrier":
        system, task_training = _train_downstream(
            condition, task, config, seed, device, output,
            carrier if condition == "learned_vector_fixed_quotient" else None,
        )
        training.update(task_training)
    active_carrier = _carrier_for(condition, system, carrier, task)
    carrier_analysis = _carrier_metrics(active_carrier, task, config, seed)
    downstream = _downstream_analysis(system, task, config, seed, device) if system is not None else None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    detail = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id, "status": "completed", "completed_at": _utc_now(),
        "condition": condition, "seed": seed, "device": str(device),
        "training": training, "carrier_analysis": carrier_analysis,
        "downstream_analysis": downstream, "wall_seconds": elapsed,
        "peak_cuda_allocated_gb": peak, "scientific_fingerprint": _fingerprint(experiment),
        "implementation_sha256": _implementation_digest(),
    }
    _write_json(output / "result.json", detail)
    metric = min(value["circular_alignment"] for value in carrier_analysis.values())
    if downstream is not None:
        metric = min(metric, *(downstream["cuts"]["full"]["probe"]["evaluations"][regime]["cosine_pearson"] for regime in PRIMARY_REGIMES))
    checkpoint = training.get("checkpoint", training.get("carrier_checkpoint"))
    return ExperimentResult(
        experiment_id=experiment.id, hypothesis_id=HYPOTHESIS_ID,
        metrics={"joint_alignment_floor": float(metric), "peak_cuda_allocated_gb": peak},
        primary_metric=float(metric), model_architecture=[task.vocab_size],
        model_parameters=sum(parameter.numel() for parameter in system.parameters()) if system is not None else sum(parameter.numel() for parameter in carrier.parameters()),  # type: ignore[union-attr]
        training_time=elapsed, training_history=training.get("training_history", training.get("carrier_history", [])),
        model_checkpoint=str(checkpoint), observations=[f"detail={output / 'result.json'}", f"condition={condition}"],
    )


def aggregate_details(details: Sequence[Mapping[str, Any]], config: RelativeCarrierConfig) -> Dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    arms: Dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = sorted((item for item in details if item["condition"] == condition), key=lambda item: item["seed"])
        carrier_by_seed = {
            str(item["seed"]): {
                "cells": {regime: item["carrier_analysis"][regime]["passed"] for regime in PRIMARY_REGIMES},
                "joint_pass": all(item["carrier_analysis"][regime]["passed"] for regime in PRIMARY_REGIMES),
            }
            for item in selected
        }
        arm: Dict[str, Any] = {
            "carrier_joint_pass_by_seed": carrier_by_seed,
            "carrier_joint_pass_count": sum(value["joint_pass"] for value in carrier_by_seed.values()),
            "carrier_success": sum(value["joint_pass"] for value in carrier_by_seed.values()) >= required,
            "carrier_metrics": {
                regime: {
                    key: statistics.fmean(item["carrier_analysis"][regime][key] for item in selected)
                    for key in ("circular_alignment", "vector_coordinate_mse", "winding_degree", "maximum_equivariance_error")
                }
                for regime in PRIMARY_REGIMES
            },
        }
        if condition != "learned_vector_carrier":
            quotient_by_seed = {}
            for item in selected:
                cells = {}
                for cut in ("frontend", "full"):
                    for regime in PRIMARY_REGIMES:
                        endpoint = item["downstream_analysis"]["cuts"][cut]["probe"]["evaluations"][regime]
                        cells[f"{cut}:{regime}"] = endpoint_pass(endpoint)
                quotient_by_seed[str(item["seed"])] = {"cells": cells, "joint_pass": all(cells.values())}
            arm["quotient_joint_pass_by_seed"] = quotient_by_seed
            arm["quotient_joint_pass_count"] = sum(value["joint_pass"] for value in quotient_by_seed.values())
            arm["quotient_success"] = arm["quotient_joint_pass_count"] >= required
            arm["task_accuracy"] = {
                regime: statistics.fmean(item["downstream_analysis"]["task_metrics"][regime]["exact_bin_accuracy"] for item in selected)
                for regime in PRIMARY_REGIMES
            }
        arms[condition] = arm
    learned = arms["learned_vector_fixed_quotient"]
    scalar = arms["scalar_only_baseline"]
    gaps = {regime: learned["task_accuracy"][regime] - scalar["task_accuracy"][regime] for regime in PRIMARY_REGIMES}
    task_control = all(value >= -0.03 for value in gaps.values())
    confirmed = (
        arms["analytic_fixed_quotient"]["carrier_success"]
        and arms["learned_vector_carrier"]["carrier_success"]
        and learned["carrier_success"] and learned["quotient_success"] and task_control
    )
    return {
        "arms": arms,
        "learned_fixed_minus_scalar_task_accuracy": gaps,
        "task_control_passed": task_control,
        "required_joint_seed_passes": required,
        "confirmed": confirmed,
        "conclusion": "carrier_and_fixed_quotient_factorization_supported" if confirmed else "full_factorization_gate_not_confirmed",
    }


def _experiments(config: RelativeCarrierConfig, task: CircleTaskConfig, output: Path) -> List[Experiment]:
    common = {
        "configuration": asdict(config), "task_config": asdict(task),
        "output_dir": str(output), "implementation_sha256": _implementation_digest(),
    }
    return [
        Experiment(
            id=f"tinyllm-relative-carrier-{condition}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM relative carrier {condition} seed {seed}",
            parameters={**common, "condition": condition, "seed": seed}, seed=seed,
        )
        for condition in CONDITIONS for seed in config.seeds
    ]


def _existing(experiment: Experiment, output: Path) -> Optional[Dict[str, Any]]:
    path = _condition_directory(output, str(experiment.parameters["condition"]), int(experiment.seed)) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if detail.get("status") != "completed" or detail.get("scientific_fingerprint") != _fingerprint(experiment):
        return None
    checkpoint = detail.get("training", {}).get("checkpoint", detail.get("training", {}).get("carrier_checkpoint", ""))
    return detail if Path(checkpoint).is_file() else None


async def run_campaign(config: RelativeCarrierConfig, task: CircleTaskConfig, output: Path) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    experiments = _experiments(config, task, output)
    existing = {item.id: detail for item in experiments if (detail := _existing(item, output)) is not None}
    pending = [item for item in experiments if item.id not in existing]
    lab = LabConfig(
        project_name="tinyllm_relative_carrier_factorization", results_dir=str(output),
        device_ids=list(config.device_ids), max_parallel_experiments=config.max_parallel_experiments,
        gpu_slots_per_device=config.gpu_slots_per_device, max_experiment_retries=config.max_retries,
        resume_completed_experiments=config.resume, auto_balance=False, enable_wandb=False, verbose=True,
    )
    runner = AsyncExperimentRunner(lab, relative_carrier_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful = {item.experiment_id for item in results if item.error is None}
    details = list(existing.values())
    for item in pending:
        if item.id in successful:
            path = _condition_directory(output, str(item.parameters["condition"]), int(item.seed)) / "result.json"
            details.append(json.loads(path.read_text()))
    complete = len(details) == len(experiments)
    bundle = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if complete else "partial", "completed_at": _utc_now(),
        "configuration": asdict(config), "task_config": asdict(task),
        "implementation_sha256": _implementation_digest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "summary": {"requested": len(experiments), "reused": len(existing), "scheduled": len(pending), "completed": len(details), "failed": len(experiments) - len(details)},
        "scheduler": {"logical_device_slots": list(runner.slot_plan.slots), "slots_by_device": {str(k): v for k, v in runner.slot_plan.slots_by_device.items()}},
        "aggregates": aggregate_details(details, config) if complete else {},
        "results": [{"experiment_id": item.experiment_id, "status": item.status.value, "error": item.error, "metrics": item.metrics} for item in results],
        "method_boundaries": [
            "Carrier degree is measured on one deterministic fixed-nuisance section per shift.",
            "Fresh nonlinear probes measure tested conditional decodability, not mutual information.",
            "The analytic carrier is a positive control and has no carrier optimization budget.",
        ],
    }
    _write_json(output / "campaign_results.json", bundle)
    return bundle


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--carrier-steps", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--probe-train-samples", type=int, default=2_048)
    parser.add_argument("--probe-validation-samples", type=int, default=512)
    parser.add_argument("--probe-test-samples", type=int, default=1_024)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_relative_carrier_factorization/20260806_d8_preregistered"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.seeds, args.steps, args.carrier_steps = "7", 2, 2
        args.train_samples, args.batch_size = 32, 8
        args.probe_steps, args.probe_train_samples = 20, 64
        args.probe_validation_samples, args.probe_test_samples = 32, 32
        args.allow_underpowered = True
    devices = (-1,) if args.gpus == "cpu" else _ints(args.gpus)
    config = RelativeCarrierConfig(
        seeds=_ints(args.seeds), training_steps=args.steps, carrier_steps=args.carrier_steps,
        train_samples=args.train_samples, batch_size=args.batch_size,
        probe_steps=args.probe_steps, probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples, device_ids=devices,
        gpu_slots_per_device=args.slots_per_gpu, max_parallel_experiments=args.max_parallel,
        max_retries=args.retries, resume=args.resume, allow_underpowered=args.allow_underpowered,
    )
    task = CircleTaskConfig(train_samples=config.train_samples)
    bundle = asyncio.run(run_campaign(config, task, args.output))
    print(json.dumps({"summary": bundle["summary"], "aggregates": bundle.get("aggregates", {})}, indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
