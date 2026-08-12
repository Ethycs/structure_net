#!/usr/bin/env python3
"""Causally test task-relative activation barycenters in frozen TinyLLMs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _resolve_device,
    _serialize_sensor_values,
    _state_digest,
    _wrapped_targets,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-task-relative-activation-barycenter.v1"
HYPOTHESIS_ID = "tinyllm-task-relative-activation-barycenter-v1"
EVIDENCE_ROLE = "preregistered_underpowered_frozen_task_activation_intervention"
SOURCE_TASK_SCHEMA = "nal.tinyllm-task-quotient-contrast.v1"
SOURCE_TASK_HYPOTHESIS = "tinyllm-task-topology-follows-quotient-v1"
SOURCE_ATLAS_SCHEMA = "nal.tinyllm-task-geometry-atlas.v1"
SOURCE_ATLAS_HYPOTHESIS = "tinyllm-layer-task-carrier-and-quotient-atlas-v1"
SOURCE_TASK_RESULT = Path(
    "data/experiments/tinyllm_task_quotient_contrast/"
    "20260805_d6_d8/results.json"
)
SOURCE_ATLAS_RESULT = Path(
    "data/experiments/tinyllm_task_geometry_atlas/"
    "20260805_d6_d8_seed7/results.json"
)
SOURCE_TASK_SHA256 = (
    "129a7a47757d2ecdde7138ca7729c0052f4c4653d5afc38106949f477e87652d"
)
SOURCE_ATLAS_SHA256 = (
    "da6dad8c00738e5fc364de45b970245fa0667731aeedbcb462b85aa94bf1c3d6"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-task-relative-activation-barycenter-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "15eeea68dde87d101f587ea258c9d928b55b9d9a55a9d897ccd027446203926e"
)
PRESETS = ("d6", "d8")
TASKS = ("cosine_interval", "phase_circle")
REGIMES = ("training_support", "outside_range")
DATASET_SEEDS = {"training_support": 850_011, "outside_range": 850_021}
REFERENCE_FRONTS = {
    "d6": "block_5_post_attention",
    "d8": "block_3_post_attention",
}
CHECKPOINTS = {
    ("d6", "cosine_interval"): Path(
        "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/"
        "checkpoints/d6_cosine_interval_trained_seed7.pt"
    ),
    ("d6", "phase_circle"): Path(
        "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/"
        "checkpoints/d6_phase_circle_trained_seed7.pt"
    ),
    ("d8", "cosine_interval"): Path(
        "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/"
        "checkpoints/d8_cosine_interval_trained_seed7.pt"
    ),
    ("d8", "phase_circle"): Path(
        "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/"
        "checkpoints/d8_phase_circle_trained_seed7.pt"
    ),
}
CHECKPOINT_SHA256 = {
    ("d6", "cosine_interval"): (
        "5343fa57d88b356c26138b763d23e8defd0b1860239d6aabca718d09516181e4"
    ),
    ("d6", "phase_circle"): (
        "5c84701e0022f03af90204b43620b227faee65beedf768187c0546c946d0f854"
    ),
    ("d8", "cosine_interval"): (
        "8170856da5e6b1f8b7a7f1c2c2121ffd4c53edaaf2ea5aabe01fa14d2f063f3f"
    ),
    ("d8", "phase_circle"): (
        "3e2053273075d2404a05d9097823814d39b8682875c0e7f4052df7d37d70de4d"
    ),
}
EXPECTED_ARCHITECTURES = {
    "d6": {"n_layer": 6, "n_head": 6, "n_embd": 384},
    "d8": {"n_layer": 8, "n_head": 8, "n_embd": 512},
}


@dataclass(frozen=True)
class TaskBarycenterConfig:
    presets: tuple[str, ...] = PRESETS
    regimes: tuple[str, ...] = REGIMES
    fiber_pairs: int = 512
    batch_size: int = 64
    accuracy_loss_ceiling: float = 0.03
    cross_entropy_increase_ceiling: float = 0.05
    posterior_js_ceiling: float = 0.02
    baseline_accuracy_floor: float = 0.15
    replay_tolerance: float = 2e-6
    front_cut_tolerance: int = 1
    semantic_control_minimum_change: float = 0.50
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.presets or set(self.presets).difference(PRESETS):
            raise ValueError(f"presets must be drawn from {PRESETS}")
        if not self.regimes or set(self.regimes).difference(REGIMES):
            raise ValueError(f"regimes must be drawn from {REGIMES}")
        if self.fiber_pairs < 2 or self.fiber_pairs % 2:
            raise ValueError("fiber_pairs must be positive and even")
        if self.batch_size < 4 or self.batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        if self.front_cut_tolerance < 0:
            raise ValueError("front_cut_tolerance cannot be negative")
        if not self.allow_underpowered and (
            self.presets != PRESETS
            or self.regimes != REGIMES
            or self.fiber_pairs != 512
        ):
            raise ValueError("primary campaign axes are frozen")


@dataclass(frozen=True)
class ExactTaskFiberCohort:
    input_ids: torch.Tensor
    phase_targets: torch.Tensor
    cosine_targets: torch.Tensor
    future_phase: torch.Tensor
    cosine: torch.Tensor
    fiber_id: torch.Tensor
    branch: torch.Tensor
    semantic_partner: torch.Tensor
    contract: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _json_config(config: TaskBarycenterConfig) -> dict[str, Any]:
    value = asdict(config)
    value["presets"] = list(config.presets)
    value["regimes"] = list(config.regimes)
    return value


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _source_digests() -> dict[str, str]:
    files = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "tinyllm_model": Path(
            "src/structure_net/components/models/tinyllm_model.py"
        ),
        "circle_source": Path(
            "experiments/structure_net/tinyllm_predictive_circle.py"
        ),
    }
    return {name: _sha256(path) for name, path in files.items()}


def _implementation_digest(source_digests: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(source_digests), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def cut_names(layer_count: int) -> tuple[str, ...]:
    names = ["query_embedding"]
    for index in range(1, layer_count + 1):
        names.extend(
            (f"block_{index}_post_attention", f"block_{index}_post_mlp")
        )
    return tuple(names)


def _cosine_targets(values: np.ndarray, task: CircleTaskConfig) -> np.ndarray:
    centers = np.linspace(-1.0, 1.0, task.phase_bins)
    width = 2.0 / (task.phase_bins - 1)
    probabilities = np.exp(-0.5 * ((centers[None] - values[:, None]) / width) ** 2)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def generate_exact_task_fibers(
    task: CircleTaskConfig,
    *,
    pair_count: int,
    seed: int,
    regime: str,
    semantic_control_minimum_change: float = 0.50,
) -> ExactTaskFiberCohort:
    """Generate phase-reflected pairs with shared pre-quantization nuisance."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime}")
    if pair_count < 2 or pair_count % 2:
        raise ValueError("pair_count must be positive and even")
    generator = np.random.default_rng(seed)
    cosine = -0.95 + 1.9 * (
        np.arange(pair_count) + generator.uniform(0.0, 1.0, pair_count)
    ) / pair_count
    generator.shuffle(cosine)
    directions = generator.choice(np.array([-1.0, 1.0]), pair_count)
    if regime == "outside_range":
        amplitude = generator.uniform(1.3, 1.6, (pair_count, 1, 1))
        orientation = generator.uniform(0.25, 0.45, (pair_count, 1))
        offsets = generator.uniform(0.18, 0.32, (pair_count, 1, 3))
        harmonic = generator.uniform(0.4, 0.55, (pair_count, 1))
        noise_scale = 0.08
        angular_speed = generator.uniform(0.24, 0.46, (pair_count, 1))
    else:
        amplitude = generator.uniform(0.75, 1.25, (pair_count, 1, 1))
        orientation = generator.uniform(-0.15, 0.15, (pair_count, 1))
        offsets = generator.uniform(-0.1, 0.1, (pair_count, 1, 3))
        harmonic = generator.uniform(0.15, 0.35, (pair_count, 1))
        noise_scale = 0.025
        angular_speed = generator.uniform(0.28, 0.42, (pair_count, 1))
    noise = generator.normal(
        0.0, noise_scale, (pair_count, task.sensor_steps, 3)
    )

    order = np.argsort(cosine)
    half = pair_count // 2
    fiber_order = np.column_stack((order[:half], order[half:])).reshape(-1)
    cosine = cosine[fiber_order]
    directions = directions[fiber_order]
    amplitude = amplitude[fiber_order]
    orientation = orientation[fiber_order]
    offsets = offsets[fiber_order]
    harmonic = harmonic[fiber_order]
    angular_speed = angular_speed[fiber_order]
    noise = noise[fiber_order]
    semantic_partner = np.arange(pair_count, dtype=np.int64) ^ 1
    semantic_change = np.abs(cosine - cosine[semantic_partner])

    positive = np.arccos(cosine)
    negative = np.remainder(2.0 * math.pi - positive, 2.0 * math.pi)
    future = np.column_stack((positive, negative)).reshape(-1)
    repeated_directions = np.repeat(directions, 2)
    current = np.remainder(
        future - repeated_directions * task.future_delta, 2.0 * math.pi
    )
    repeated_amplitude = np.repeat(amplitude, 2, axis=0)
    repeated_orientation = np.repeat(orientation, 2, axis=0)
    repeated_offsets = np.repeat(offsets, 2, axis=0)
    repeated_harmonic = np.repeat(harmonic, 2, axis=0)
    repeated_speed = np.repeat(angular_speed, 2, axis=0)
    repeated_noise = np.repeat(noise, 2, axis=0)

    history = np.arange(task.sensor_steps, dtype=np.float64) - (
        task.sensor_steps - 1
    )
    angles = (
        current[:, None]
        + repeated_directions[:, None] * history[None] * repeated_speed
    )
    x_values = np.cos(angles)
    y_values = np.sin(angles)
    orientation_cosine = np.cos(repeated_orientation)
    orientation_sine = np.sin(repeated_orientation)
    rotated_x = orientation_cosine * x_values - orientation_sine * y_values
    rotated_y = orientation_sine * x_values + orientation_cosine * y_values
    z_values = repeated_harmonic * np.cos(2.0 * angles)
    sensor = repeated_amplitude * np.stack(
        (rotated_x, rotated_y, z_values), axis=-1
    )
    sensor += repeated_offsets + repeated_noise
    input_ids = _serialize_sensor_values(sensor, task)
    phase_targets = _wrapped_targets(future, task)
    repeated_cosine = np.repeat(cosine, 2)
    cosine_targets = _cosine_targets(repeated_cosine, task)
    phase_bins = phase_targets.argmax(1).reshape(pair_count, 2)
    serialized_pairs = input_ids.reshape(pair_count, 2, -1)

    nuisance_arrays = (
        repeated_directions[:, None],
        repeated_amplitude.reshape(len(future), -1),
        repeated_orientation,
        repeated_offsets.reshape(len(future), -1),
        repeated_harmonic,
        repeated_speed,
    )
    shared_nuisance = all(
        np.array_equal(value[0::2], value[1::2]) for value in nuisance_arrays
    )
    shared_noise = np.array_equal(repeated_noise[0::2], repeated_noise[1::2])
    contract = {
        "two_rows_per_fiber": True,
        "cosine_target_maximum_pair_difference": float(
            np.max(np.abs(cosine_targets[0::2] - cosine_targets[1::2]))
        ),
        "phase_target_bins_differ_every_fiber": bool(
            np.all(phase_bins[:, 0] != phase_bins[:, 1])
        ),
        "shared_nuisance_byte_identical": bool(shared_nuisance),
        "shared_prequantization_noise_byte_identical": bool(shared_noise),
        "serialized_sheets_differ_every_fiber": bool(
            np.all(np.any(serialized_pairs[:, 0] != serialized_pairs[:, 1], axis=1))
        ),
        "minimum_semantic_control_cosine_change": float(
            semantic_change.min()
        ),
        "semantic_control_change_pass": bool(
            semantic_change.min() >= semantic_control_minimum_change
        ),
        "finite": bool(
            np.isfinite(sensor).all()
            and np.isfinite(phase_targets).all()
            and np.isfinite(cosine_targets).all()
        ),
    }
    contract["pass"] = bool(
        contract["cosine_target_maximum_pair_difference"] <= 1e-6
        and contract["phase_target_bins_differ_every_fiber"]
        and contract["shared_nuisance_byte_identical"]
        and contract["shared_prequantization_noise_byte_identical"]
        and contract["serialized_sheets_differ_every_fiber"]
        and contract["semantic_control_change_pass"]
        and contract["finite"]
    )
    return ExactTaskFiberCohort(
        input_ids=torch.from_numpy(input_ids),
        phase_targets=torch.from_numpy(phase_targets.astype(np.float32)),
        cosine_targets=torch.from_numpy(cosine_targets.astype(np.float32)),
        future_phase=torch.from_numpy(future.astype(np.float32)),
        cosine=torch.from_numpy(repeated_cosine.astype(np.float32)),
        fiber_id=torch.arange(pair_count).repeat_interleave(2),
        branch=torch.tensor([0, 1], dtype=torch.long).repeat(pair_count),
        semantic_partner=torch.from_numpy(semantic_partner),
        contract=contract,
    )


def _validate_sources(
    config: TaskBarycenterConfig,
) -> tuple[CircleTaskConfig, dict[tuple[str, str], Path], dict[str, str]]:
    task_result = json.loads(SOURCE_TASK_RESULT.read_text(encoding="utf-8"))
    atlas_result = json.loads(SOURCE_ATLAS_RESULT.read_text(encoding="utf-8"))
    task_config = task_result.get("task_config", {})
    if (
        _sha256(SOURCE_TASK_RESULT) != SOURCE_TASK_SHA256
        or task_result.get("schema_version") != SOURCE_TASK_SCHEMA
        or task_result.get("hypothesis_id") != SOURCE_TASK_HYPOTHESIS
        or task_result.get("status") != "completed"
        or _sha256(SOURCE_ATLAS_RESULT) != SOURCE_ATLAS_SHA256
        or atlas_result.get("schema_version") != SOURCE_ATLAS_SCHEMA
        or atlas_result.get("hypothesis_id") != SOURCE_ATLAS_HYPOTHESIS
        or atlas_result.get("status") != "completed"
        or atlas_result.get("task_config") != task_config
        or set(atlas_result.get("checkpoint_paths", []))
        != {str(CHECKPOINTS[key]) for key in CHECKPOINTS}
        or _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256
    ):
        raise ValueError("task-relative barycenter source campaign changed")
    checkpoints: dict[tuple[str, str], Path] = {}
    hashes: dict[str, str] = {
        "source_task_result": SOURCE_TASK_SHA256,
        "source_atlas_result": SOURCE_ATLAS_SHA256,
    }
    for preset in config.presets:
        for task_name in TASKS:
            key = (preset, task_name)
            path = CHECKPOINTS[key]
            expected_hash = CHECKPOINT_SHA256[key]
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(f"checkpoint changed: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            metadata = payload.get("metadata", {})
            model_config = payload.get("config", {})
            architecture = EXPECTED_ARCHITECTURES[preset]
            if (
                metadata.get("schema_version") != SOURCE_TASK_SCHEMA
                or metadata.get("hypothesis_id") != SOURCE_TASK_HYPOTHESIS
                or metadata.get("quotient") != task_name
                or metadata.get("condition") != "trained"
                or int(metadata.get("seed", -1)) != 7
                or any(
                    int(model_config.get(name, -1)) != expected
                    for name, expected in architecture.items()
                )
                or int(model_config.get("block_size", -1))
                != CircleTaskConfig(**task_config).sequence_length
            ):
                raise ValueError(f"invalid checkpoint metadata: {path}")
            checkpoints[key] = path
            hashes[f"checkpoint_{preset}_{task_name}"] = expected_hash
    return CircleTaskConfig(**task_config), checkpoints, hashes


def _model_system_state(model: TinyLLMModel) -> str:
    payload = {
        "model_state": _state_digest(model),
        "training": model.training,
        "requires_grad": [parameter.requires_grad for parameter in model.parameters()],
        "feedback_connections": len(model.feedback_connections),
        "refinement_steps": model.refinement_steps,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_model(
    path: Path, preset: str, task_name: str, device: torch.device
) -> tuple[TinyLLMModel, str]:
    model = TinyLLMModel.from_checkpoint(path, map_location="cpu").to(device)
    if (
        model.config.n_layer != int(preset[1:])
        or any(
            int(getattr(model.config, name)) != expected
            for name, expected in EXPECTED_ARCHITECTURES[preset].items()
        )
        or model.feedback_connections
        or model.refinement_steps
    ):
        raise ValueError(f"unsupported source model {path}")
    metadata = model.checkpoint_metadata or {}
    if metadata.get("quotient") != task_name:
        raise ValueError(f"checkpoint task mismatch {path}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, _model_system_state(model)


@torch.no_grad()
def _capture_batch(
    model: TinyLLMModel, inputs: torch.Tensor, answer_ids: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    positions = torch.arange(inputs.shape[1], device=inputs.device, dtype=torch.long)
    value = model.transformer["wte"](inputs) + model.transformer["wpe"](positions)
    captured = {"query_embedding": value}
    for index, block in enumerate(model.transformer["h"], start=1):
        value = value + block.attn(block.ln_1(value))
        captured[f"block_{index}_post_attention"] = value
        value = value + block.mlp(block.ln_2(value))
        captured[f"block_{index}_post_mlp"] = value
    normalized = model.transformer["ln_f"](value)
    logits = model.lm_head(normalized[:, -1]).index_select(-1, answer_ids)
    return captured, torch.softmax(logits, -1).double()


@torch.no_grad()
def continue_from_cut(
    model: TinyLLMModel,
    cut: str,
    value: torch.Tensor,
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    names = cut_names(model.config.n_layer)
    if cut not in names:
        raise ValueError(f"unknown cut {cut}")
    if cut == "query_embedding":
        start = 0
    else:
        fields = cut.split("_")
        block_index = int(fields[1]) - 1
        if cut.endswith("post_attention"):
            block = model.transformer["h"][block_index]
            value = value + block.mlp(block.ln_2(value))
        start = block_index + 1
    for block in model.transformer["h"][start:]:
        value = block(value)
    normalized = model.transformer["ln_f"](value)
    logits = model.lm_head(normalized[:, -1]).index_select(-1, answer_ids)
    return torch.softmax(logits, -1).double()


def _fiber_barycenters(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(values) % 4:
        raise ValueError("streaming batch must contain complete semantic pairs")
    unique = values.reshape((len(values) // 2, 2) + values.shape[1:]).mean(1)
    return unique.repeat_interleave(2, dim=0), unique


def _semantic_reassignment(unique: torch.Tensor) -> torch.Tensor:
    partner = torch.arange(len(unique), device=unique.device) ^ 1
    return unique[partner].repeat_interleave(2, dim=0)


def _jensen_shannon_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.double().clamp_min(1e-12)
    right = right.double().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(1)
        + (right * (right.log() - middle.log())).sum(1)
    )


def _posterior_metrics(
    posterior: torch.Tensor,
    targets: torch.Tensor,
    task_name: str,
    task: CircleTaskConfig,
    future_phase: torch.Tensor,
    cosine: torch.Tensor,
) -> dict[str, float]:
    posterior = posterior.double().clamp_min(1e-12)
    targets = targets.double()
    predicted_bins = posterior.argmax(1)
    target_bins = targets.argmax(1)
    result = {
        "exact_bin_accuracy": float((predicted_bins == target_bins).double().mean()),
        "mean_target_cross_entropy": float(
            -(targets * posterior.log()).sum(1).mean()
        ),
        "mean_target_kl": float(
            (targets * (targets.clamp_min(1e-12).log() - posterior.log()))
            .sum(1)
            .mean()
        ),
    }
    if task_name == "phase_circle":
        angles = 2.0 * math.pi * torch.arange(task.phase_bins).double() / task.phase_bins
        vector_x = posterior @ torch.cos(angles)
        vector_y = posterior @ torch.sin(angles)
        predicted = torch.atan2(vector_y, vector_x)
        target = future_phase.double()
        delta = predicted - target
        result["task_map_score"] = float(
            torch.sqrt(torch.mean(torch.cos(delta)).square() + torch.mean(torch.sin(delta)).square())
        )
        result["mean_absolute_circular_error"] = float(
            torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta))).mean()
        )
    else:
        centers = torch.linspace(-1.0, 1.0, task.phase_bins, dtype=torch.double)
        predicted = posterior @ centers
        target = cosine.double()
        centered_predicted = predicted - predicted.mean()
        centered_target = target - target.mean()
        denominator = torch.sqrt(
            centered_predicted.square().sum() * centered_target.square().sum()
        ).clamp_min(1e-12)
        result["task_map_score"] = float(
            (centered_predicted * centered_target).sum() / denominator
        )
        result["root_mean_squared_error"] = float(
            torch.sqrt(torch.mean((predicted - target).square()))
        )
    return result


def _candidate_record(
    posterior: torch.Tensor,
    baseline_posterior: torch.Tensor,
    targets: torch.Tensor,
    task_name: str,
    task: CircleTaskConfig,
    cohort: ExactTaskFiberCohort,
    config: TaskBarycenterConfig,
) -> dict[str, Any]:
    baseline = _posterior_metrics(
        baseline_posterior,
        targets,
        task_name,
        task,
        cohort.future_phase,
        cohort.cosine,
    )
    metrics = _posterior_metrics(
        posterior,
        targets,
        task_name,
        task,
        cohort.future_phase,
        cohort.cosine,
    )
    gates = {
        "accuracy_loss": float(
            baseline["exact_bin_accuracy"] - metrics["exact_bin_accuracy"]
        ),
        "cross_entropy_increase": float(
            metrics["mean_target_cross_entropy"]
            - baseline["mean_target_cross_entropy"]
        ),
        "mean_posterior_js": float(
            _jensen_shannon_rows(posterior, baseline_posterior).mean()
        ),
    }
    gates["accuracy_pass"] = bool(
        gates["accuracy_loss"] <= config.accuracy_loss_ceiling
    )
    gates["cross_entropy_pass"] = bool(
        gates["cross_entropy_increase"]
        <= config.cross_entropy_increase_ceiling
    )
    gates["posterior_js_pass"] = bool(
        gates["mean_posterior_js"] <= config.posterior_js_ceiling
    )
    gates["task_sufficient"] = bool(
        gates["accuracy_pass"]
        and gates["cross_entropy_pass"]
        and gates["posterior_js_pass"]
    )
    return {"metrics": metrics, "gates": gates}


def _stream_model_regime(
    model: TinyLLMModel,
    task_name: str,
    cohort: ExactTaskFiberCohort,
    task: CircleTaskConfig,
    config: TaskBarycenterConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    names = cut_names(model.config.n_layer)
    targets = (
        cohort.cosine_targets
        if task_name == "cosine_interval"
        else cohort.phase_targets
    )
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    baseline_parts: list[torch.Tensor] = []
    replay_parts = {cut: [] for cut in names}
    correct_parts = {cut: [] for cut in names}
    semantic_parts = {cut: [] for cut in names}
    pair_rms = {cut: [] for cut in names}
    state_square_sum = {cut: 0.0 for cut in names}
    state_count = {cut: 0 for cut in names}
    maximum_replay_error = {cut: 0.0 for cut in names}
    for start in range(0, len(cohort.input_ids), config.batch_size):
        stop = min(len(cohort.input_ids), start + config.batch_size)
        if (stop - start) % 4:
            raise ValueError("batch split broke a semantic partner group")
        inputs = cohort.input_ids[start:stop].to(device)
        captured, baseline = _capture_batch(model, inputs, answer_ids)
        baseline_cpu = baseline.cpu()
        baseline_parts.append(baseline_cpu)
        for cut in names:
            state = captured[cut]
            correct, unique = _fiber_barycenters(state)
            semantic = _semantic_reassignment(unique)
            replay = continue_from_cut(model, cut, state, answer_ids).cpu()
            candidates = torch.cat((correct, semantic), dim=0)
            continued = continue_from_cut(
                model, cut, candidates, answer_ids
            ).cpu()
            batch = len(state)
            correct_posterior, semantic_posterior = continued.split(batch)
            maximum_replay_error[cut] = max(
                maximum_replay_error[cut],
                float(torch.max(torch.abs(replay - baseline_cpu))),
            )
            replay_parts[cut].append(replay)
            correct_parts[cut].append(correct_posterior)
            semantic_parts[cut].append(semantic_posterior)
            shaped = state.reshape((batch // 2, 2) + state.shape[1:]).double()
            difference = shaped[:, 0] - shaped[:, 1]
            differences = torch.sqrt(
                torch.mean(
                    difference.square(),
                    dim=tuple(range(1, difference.ndim)),
                )
            )
            pair_rms[cut].append(differences.detach().cpu().reshape(-1))
            state_square_sum[cut] += float(state.double().square().sum())
            state_count[cut] += state.numel()
    baseline_posterior = torch.cat(baseline_parts)
    baseline_metrics = _posterior_metrics(
        baseline_posterior,
        targets,
        task_name,
        task,
        cohort.future_phase,
        cohort.cosine,
    )
    cuts: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "baseline_posterior": baseline_posterior.numpy()
    }
    for cut in names:
        replay = torch.cat(replay_parts[cut])
        correct = torch.cat(correct_parts[cut])
        semantic = torch.cat(semantic_parts[cut])
        rms = torch.cat(pair_rms[cut]).double()
        state_rms = math.sqrt(
            state_square_sum[cut] / max(1, state_count[cut])
        )
        cuts[cut] = {
            "maximum_replay_error": maximum_replay_error[cut],
            "correct_barycenter": _candidate_record(
                correct,
                baseline_posterior,
                targets,
                task_name,
                task,
                cohort,
                config,
            ),
            "semantic_reassignment": _candidate_record(
                semantic,
                baseline_posterior,
                targets,
                task_name,
                task,
                cohort,
                config,
            ),
            "activation_pair_geometry": {
                "mean_pair_rms": float(rms.mean()),
                "maximum_pair_rms": float(rms.max()),
                "mean_pair_rms_over_state_rms": float(
                    rms.mean() / max(state_rms, 1e-12)
                ),
            },
        }
        arrays[f"{cut}__replay_posterior"] = replay.numpy()
        arrays[f"{cut}__correct_posterior"] = correct.numpy()
        arrays[f"{cut}__semantic_posterior"] = semantic.numpy()
        arrays[f"{cut}__pair_rms"] = rms.numpy()
    return {
        "baseline": baseline_metrics,
        "baseline_valid": bool(
            baseline_metrics["exact_bin_accuracy"]
            >= config.baseline_accuracy_floor
        ),
        "cuts": cuts,
    }, arrays


def _joint_pass(
    task_result: Mapping[str, Any], cut: str, field: str
) -> bool:
    return all(
        task_result["regimes"][regime]["cuts"][cut][field]["gates"][
            "task_sufficient"
        ]
        for regime in REGIMES
        if regime in task_result["regimes"]
    )


def _all_regimes_fail(
    task_result: Mapping[str, Any], cut: str, field: str
) -> bool:
    return all(
        not regime_result["cuts"][cut][field]["gates"]["task_sufficient"]
        for regime_result in task_result["regimes"].values()
    )


def mature_front(
    task_result: Mapping[str, Any], cuts: Sequence[str]
) -> Optional[str]:
    for index, cut in enumerate(cuts):
        if all(
            _joint_pass(task_result, later, "correct_barycenter")
            for later in cuts[index:]
        ):
            return cut
    return None


def first_isolated_pass(
    task_result: Mapping[str, Any], cuts: Sequence[str]
) -> Optional[str]:
    return next(
        (
            cut
            for cut in cuts
            if _joint_pass(task_result, cut, "correct_barycenter")
        ),
        None,
    )


def _fingerprint(
    preset: str,
    config: TaskBarycenterConfig,
    implementation: str,
    source_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "preset": preset,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_hashes": dict(source_hashes),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def analyze_preset(
    preset: str,
    task: CircleTaskConfig,
    checkpoints: Mapping[tuple[str, str], Path],
    source_hashes: Mapping[str, str],
    config: TaskBarycenterConfig,
    device: torch.device,
    output: Path,
    implementation: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    cohorts = {
        regime: generate_exact_task_fibers(
            task,
            pair_count=config.fiber_pairs,
            seed=DATASET_SEEDS[regime],
            regime=regime,
            semantic_control_minimum_change=config.semantic_control_minimum_change,
        )
        for regime in config.regimes
    }
    if not all(cohort.contract["pass"] for cohort in cohorts.values()):
        raise ValueError("task-fiber input contract failed before model evaluation")
    tasks: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    state_records: dict[str, Any] = {}
    cuts: Optional[tuple[str, ...]] = None
    for task_name in TASKS:
        model, initial_system_state = _load_model(
            checkpoints[(preset, task_name)], preset, task_name, device
        )
        names = cut_names(model.config.n_layer)
        if cuts is None:
            cuts = names
        elif cuts != names:
            raise ValueError("matched task models have different activation cuts")
        regimes: dict[str, Any] = {}
        for regime, cohort in cohorts.items():
            regime_result, regime_arrays = _stream_model_regime(
                model, task_name, cohort, task, config, device
            )
            regimes[regime] = regime_result
            arrays.update(
                {
                    f"{task_name}__{regime}__{name}": value
                    for name, value in regime_arrays.items()
                }
            )
        final_system_state = _model_system_state(model)
        state_records[task_name] = {
            "initial_system_state_sha256": initial_system_state,
            "final_system_state_sha256": final_system_state,
            "unchanged": initial_system_state == final_system_state,
            "model_state_sha256": _state_digest(model),
        }
        tasks[task_name] = {"regimes": regimes}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert cuts is not None
    cosine = tasks["cosine_interval"]
    front = mature_front(cosine, cuts)
    isolated = first_isolated_pass(cosine, cuts)
    reference = REFERENCE_FRONTS[preset]
    front_distance = (
        abs(cuts.index(front) - cuts.index(reference))
        if front is not None
        else None
    )
    selected_cuts = tuple(dict.fromkeys((front, cuts[-1]))) if front else (cuts[-1],)
    correct_selected = bool(
        front is not None
        and all(
            _joint_pass(cosine, cut, "correct_barycenter")
            for cut in selected_cuts
        )
    )
    phase_specific = bool(
        front is not None
        and all(
            _all_regimes_fail(
                tasks["phase_circle"], cut, "correct_barycenter"
            )
            for cut in selected_cuts
        )
    )
    semantic_specific = bool(
        front is not None
        and all(
            _all_regimes_fail(cosine, cut, "semantic_reassignment")
            for cut in selected_cuts
        )
    )
    maximum_replay = max(
        float(cut_result["maximum_replay_error"])
        for task_result in tasks.values()
        for regime_result in task_result["regimes"].values()
        for cut_result in regime_result["cuts"].values()
    )
    baseline_valid = all(
        regime_result["baseline_valid"]
        for task_result in tasks.values()
        for regime_result in task_result["regimes"].values()
    )
    validity = bool(
        baseline_valid
        and maximum_replay <= config.replay_tolerance
        and all(item["unchanged"] for item in state_records.values())
        and all(cohort.contract["pass"] for cohort in cohorts.values())
    )
    front_agreement = bool(
        front_distance is not None
        and front_distance <= config.front_cut_tolerance
    )
    full_gate = bool(
        validity
        and front is not None
        and front_agreement
        and correct_selected
        and phase_specific
        and semantic_specific
    )
    diagnostics_path = output / "diagnostics.npz"
    _write_npz(diagnostics_path, arrays)
    result_path = output / "result.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "experiment_id": f"tinyllm_task_relative_barycenter_{preset}_seed7",
        "preset": preset,
        "seed": 7,
        "scientific_fingerprint": _fingerprint(
            preset, config, implementation, source_hashes
        ),
        "implementation_sha256": implementation,
        "configuration": _json_config(config),
        "cuts": list(cuts),
        "reference_observational_front": reference,
        "cohort_contracts": {
            regime: dict(cohort.contract) for regime, cohort in cohorts.items()
        },
        "tasks": tasks,
        "state_records": state_records,
        "summary": {
            "mature_causal_front": front,
            "first_isolated_passing_cut": isolated,
            "front_cut_distance": front_distance,
            "maximum_replay_error": maximum_replay,
            "block1_mlp_joint_sufficiency": _joint_pass(
                cosine, "block_1_post_mlp", "correct_barycenter"
            ),
        },
        "gates": {
            "validity": validity,
            "baseline_validity": baseline_valid,
            "replay": maximum_replay <= config.replay_tolerance,
            "model_state_unchanged": all(
                item["unchanged"] for item in state_records.values()
            ),
            "mature_causal_front_exists": front is not None,
            "front_agreement_within_one_cut": front_agreement,
            "correct_barycenter_selected_cuts": correct_selected,
            "phase_task_specificity": phase_specific,
            "semantic_reassignment_specificity": semantic_specific,
            "full_preset_gate": full_gate,
        },
        "source": {
            "task_result": str(SOURCE_TASK_RESULT),
            "atlas_result": str(SOURCE_ATLAS_RESULT),
            "source_hashes": dict(source_hashes),
            "checkpoints": {
                task_name: str(checkpoints[(preset, task_name)])
                for task_name in TASKS
            },
        },
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    if not _finite(result):
        raise ValueError(f"non-finite task barycenter result for {preset}")
    _write_json(result_path, result)
    return result


def _reusable_result(
    path: Path, fingerprint: str, implementation: str
) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = Path(value.get("artifacts", {}).get("diagnostics", ""))
    if (
        value.get("status") != "completed"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("scientific_fingerprint") != fingerprint
        or value.get("implementation_sha256") != implementation
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != value.get("artifacts", {}).get("diagnostics_sha256")
    ):
        raise ValueError(f"incompatible completed result {path}")
    return value


def _campaign_reusable(
    campaign: Mapping[str, Any], config: TaskBarycenterConfig, implementation: str
) -> bool:
    entries = campaign.get("results", [])
    expected = set(config.presets)
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and {item.get("preset") for item in entries} == expected
        and all(
            Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["result_sha256"]
            and Path(item["diagnostics_path"]).is_file()
            and _sha256(Path(item["diagnostics_path"]))
            == item["diagnostics_sha256"]
            for item in entries
        )
    )


def run_campaign(
    config: TaskBarycenterConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_reusable(existing, config, implementation):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")
    task, checkpoints, source_hashes = _validate_sources(config)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    results = []
    reused = 0
    for preset in config.presets:
        result_path = output / "runs" / preset / "result.json"
        fingerprint = _fingerprint(
            preset, config, implementation, source_hashes
        )
        reusable = _reusable_result(result_path, fingerprint, implementation)
        if reusable is not None:
            result = reusable
            reused += 1
        else:
            result = analyze_preset(
                preset,
                task,
                checkpoints,
                source_hashes,
                config,
                device,
                result_path.parent,
                implementation,
            )
        results.append(result)
    primary = bool(
        not config.allow_underpowered
        and set(config.presets) == set(PRESETS)
        and all(item["gates"]["full_preset_gate"] for item in results)
    )
    valid = bool(all(item["gates"]["validity"] for item in results))
    if config.allow_underpowered:
        classification = "systems_lifecycle_only_not_quality_evidence"
    elif primary:
        classification = "task_relative_causal_quotient_matches_geometry"
    elif all(item["gates"]["mature_causal_front_exists"] for item in results):
        classification = "causal_quotient_exists_but_atlas_mislocalizes"
    else:
        classification = "observational_task_geometry_not_causally_sufficient"
    entries = [
        {
            "experiment_id": item["experiment_id"],
            "preset": item["preset"],
            "scientific_fingerprint": item["scientific_fingerprint"],
            "path": item["artifacts"]["result"],
            "result_sha256": _sha256(Path(item["artifacts"]["result"])),
            "diagnostics_path": item["artifacts"]["diagnostics"],
            "diagnostics_sha256": item["artifacts"]["diagnostics_sha256"],
        }
        for item in results
    ]
    manifest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "source_hashes": source_hashes,
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "summary": {
            "requested": len(config.presets),
            "scheduled": len(config.presets) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_probes": 0,
            "fitted_alignment_maps": 0,
            "fitted_decoders": 0,
        },
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary,
            "valid": valid,
            "preset_gates": {
                item["preset"]: item["gates"] for item in results
            },
            "mature_causal_fronts": {
                item["preset"]: item["summary"]["mature_causal_front"]
                for item in results
            },
            "reference_observational_fronts": REFERENCE_FRONTS,
        },
        "results": entries,
        "result_manifest_sha256": manifest,
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The four retained checkpoints all use seed 7; d6 and d8 are architecture replications, not independent training seeds.",
            "The intervention patches the complete residual sequence and uses each frozen model's native continuation and answer head.",
            "No probe, alignment, decoder, model parameter, or threshold is fit.",
            "The exact pair changes phase while sharing every declared nuisance and pre-quantization noise draw.",
            "Activation barycenters are off-manifold causal interventions and do not establish literal whole-state information erasure.",
        ],
    }
    _write_json(campaign_path, campaign)
    return campaign


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_task_relative_activation_barycenter/"
            "20260810_seed7_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--presets", type=_strings, default=PRESETS)
    parser.add_argument("--regimes", type=_strings, default=REGIMES)
    parser.add_argument("--fiber-pairs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = TaskBarycenterConfig(
        presets=args.presets,
        regimes=args.regimes,
        fiber_pairs=args.fiber_pairs,
        batch_size=args.batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
