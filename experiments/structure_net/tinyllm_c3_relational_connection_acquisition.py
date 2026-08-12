#!/usr/bin/env python3
"""Run the matched C3 relational connection acquisition campaign."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import experiments.structure_net.tinyllm_c3_relational_connection_function_class as fc
import experiments.structure_net.tinyllm_c3_relational_connection_preflight as rel
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as source
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-c3-relational-connection-acquisition.v1"
HYPOTHESIS_ID = "tinyllm-c3-relational-connection-acquisition-v1"
EVIDENCE_ROLE = "prospective_matched_connection_invariant_acquisition"
ARMS = (
    "learned_true",
    "learned_no_connection",
    "learned_connection_shuffled",
    "learned_target_shuffled",
)
SEEDS = (1453, 1471, 1483, 1531, 1543)
REGIMES = ("composition", "extrapolation")
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-acquisition-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "d1ca34b1d251dc06a69aa293bfbfac2c3ee63fb80de3d4b7ed822e01ad10c015"
)
FUNCTION_CLASS_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_function_class/"
    "20260811_preregistered/result.json"
)
PINNED_SOURCE_SHA256 = {
    "preregistration": PREREGISTRATION_SHA256,
    "function_class_runner": (
        "7010794c3be5fda05a035e5a0b4a178aacd40c934dc1f5f7b36eb2eb03ea96b1"
    ),
    "function_class_result": (
        "2292e971bb655db246565675fece8dfd9e1546692b9b782e940a8bbef49de82c"
    ),
    "relational_runner": (
        "2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214"
    ),
    "relational_result": (
        "ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e"
    ),
}
RELATIONAL_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_preflight/"
    "20260811_preregistered/result.json"
)
PRIMARY_OUTPUT = Path(
    "data/experiments/tinyllm_c3_relational_connection_acquisition/"
    "20260811_preregistered"
)
TRAIN_DATA_SEED_BASE = 1_151_107
EVALUATION_SEED_BASES = {
    "composition": 1_153_107,
    "extrapolation": 1_155_107,
}
MINIBATCH_SEED_BASE = 1_157_107
TRAIN_CONNECTION_SHUFFLE_SEED_BASE = 1_159_107
EVALUATION_CONNECTION_SHUFFLE_SEED_BASES = {
    "composition": 1_161_107,
    "extrapolation": 1_163_107,
}
TRAIN_TARGET_SHUFFLE_SEED_BASE = 1_165_107
ACTION_SEED_BASE = 1_167_107
TOTAL_PARAMETER_COUNT = 187
ACTION_ERROR_MAXIMUM = 2e-5
ANALYTIC_GATES = {
    "scalar_correlation_minimum": 0.999,
    "scalar_rmse_maximum": 0.01,
    "exact_bin_accuracy_minimum": 0.98,
    "target_cross_entropy_maximum": 1.35,
    "predicted_bin_coverage": 16,
}
LEARNED_GATES = {
    "scalar_correlation_minimum": 0.999,
    "scalar_rmse_maximum": 0.01,
    "exact_bin_accuracy_minimum": 0.95,
    "target_cross_entropy_maximum": 1.35,
    "predicted_bin_coverage": 16,
}


@dataclass(frozen=True)
class AcquisitionConfig:
    seeds: tuple[int, ...] = SEEDS
    training_steps: int = 2_400
    train_samples: int = 4_096
    evaluation_samples: int = 1_024
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    required_seed_passes: int = 4
    control_seed_passes_maximum: int = 1
    device_ids: tuple[int, ...] = (0, 1, 2)
    gpu_slots_per_device: int = 2
    gpu_memory_per_experiment_gb: float = 0.25
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 6
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.training_steps < 2 or self.training_steps % 2:
            raise ValueError("training_steps must be positive and even")
        if self.train_samples < self.batch_size or self.batch_size < 2:
            raise ValueError("invalid training sample or batch count")
        if self.evaluation_samples < 16:
            raise ValueError("evaluation cohort is too small")
        if not self.device_ids:
            raise ValueError("at least one logical device is required")
        if not self.allow_underpowered and (
            self.seeds != SEEDS
            or self.training_steps != 2_400
            or self.train_samples != 4_096
            or self.evaluation_samples != 1_024
            or self.batch_size != 64
            or self.learning_rate != 1e-3
            or self.weight_decay != 0.0
            or self.gradient_clip != 1.0
            or self.required_seed_passes != 4
            or self.control_seed_passes_maximum != 1
        ):
            raise ValueError("primary connection acquisition protocol is frozen")

    @property
    def split_step(self) -> int:
        return self.training_steps // 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _manifest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _source_hashes() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "function_class_runner": _sha256(Path(fc.__file__)),
        "function_class_result": _sha256(FUNCTION_CLASS_RESULT_PATH),
        "relational_runner": _sha256(Path(rel.__file__)),
        "relational_result": _sha256(RELATIONAL_RESULT_PATH),
    }


def _validate_sources() -> tuple[dict[str, str], dict[str, Any]]:
    hashes = _source_hashes()
    for name, expected in PINNED_SOURCE_SHA256.items():
        if hashes.get(name) != expected:
            raise RuntimeError(f"connection acquisition source changed: {name}")
    predecessor = json.loads(FUNCTION_CLASS_RESULT_PATH.read_text(encoding="utf-8"))
    aggregates = predecessor.get("aggregates", {})
    if (
        predecessor.get("status") != "completed"
        or aggregates.get("classification") != fc.CLASSIFICATION_PASS
        or aggregates.get("matched_sensor_readout_campaign_licensed") is not True
        or aggregates.get("unrestricted_tinyllm_training_licensed") is not False
    ):
        raise RuntimeError("function-class predecessor no longer licenses acquisition")
    return hashes, predecessor


def _json_config(config: AcquisitionConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(value: Mapping[str, Any]) -> AcquisitionConfig:
    converted = dict(value)
    converted["seeds"] = tuple(converted["seeds"])
    converted["device_ids"] = tuple(converted["device_ids"])
    return AcquisitionConfig(**converted)


def _implementation_digest(sources: Mapping[str, str]) -> str:
    return _json_hash(dict(sources))


def generate_dataset(
    regime: str, *, dataset_seed: int, sample_count: int
) -> rel.RelationalDataset:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    theta_0 = 2.0 * math.pi * torch.rand(
        sample_count, dtype=torch.float64, generator=generator
    )
    delta = math.pi * (
        2.0 * torch.rand(sample_count, dtype=torch.float64, generator=generator)
        - 1.0
    )
    interior = 2.0 * math.pi * torch.rand(
        sample_count,
        rel.TIME_STEPS - 2,
        dtype=torch.float64,
        generator=generator,
    )
    phase = torch.cat(
        (theta_0[:, None], interior, (theta_0 + delta)[:, None]), dim=1
    )
    ranges = source.REGIME_RANGES[regime]

    def uniform(bounds: tuple[float, float]) -> torch.Tensor:
        return torch.empty(sample_count, dtype=torch.float64).uniform_(
            *bounds, generator=generator
        )

    amplitude = uniform(ranges["amplitude"])
    offset = uniform(ranges["offset"])
    drift = uniform(ranges["drift"])
    gauge = torch.randint(
        0,
        rel.CHANNELS,
        (sample_count, rel.TIME_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    continuous = rel._continuous_observation(phase, amplitude, offset, drift)
    canonical = source.quantize(continuous)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    return rel.RelationalDataset(
        canonical_tokens=canonical,
        tokens=rel.apply_local_action(canonical, gauge),
        calibration=calibration,
        target=torch.cos(delta),
        phase=phase,
        delta=delta,
        gauge=gauge,
        connection=rel.edge_connection(gauge),
        saturation_count=int(
            (continuous.abs() >= source.QUANTIZATION_LIMIT).sum()
        ),
        dataset_seed=dataset_seed,
    )


def dataset_digest(dataset: rel.RelationalDataset) -> str:
    return _tensor_digest(
        dataset.canonical_tokens,
        dataset.tokens,
        dataset.calibration,
        dataset.target,
        dataset.phase,
        dataset.delta,
        dataset.gauge,
        dataset.connection,
    )


def _to_device(
    dataset: rel.RelationalDataset, device: torch.device
) -> rel.RelationalDataset:
    values = []
    for name in dataset.__dataclass_fields__:
        value = getattr(dataset, name)
        values.append(value.to(device) if torch.is_tensor(value) else value)
    return rel.RelationalDataset(*values)


def protocol_material(
    config: AcquisitionConfig, seed: int
) -> tuple[rel.RelationalDataset, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
    dataset = generate_dataset(
        "composition",
        dataset_seed=TRAIN_DATA_SEED_BASE + seed,
        sample_count=config.train_samples,
    )
    batches = torch.randint(
        0,
        config.train_samples,
        (config.training_steps, config.batch_size),
        generator=torch.Generator(device="cpu").manual_seed(
            MINIBATCH_SEED_BASE + seed
        ),
        dtype=torch.int64,
    )
    connection_permutation = rel.sattolo_derangement(
        config.train_samples, TRAIN_CONNECTION_SHUFFLE_SEED_BASE + seed
    )
    target_permutation = rel.sattolo_derangement(
        config.train_samples, TRAIN_TARGET_SHUFFLE_SEED_BASE + seed
    )
    hashes = {
        "training_data_sha256": dataset_digest(dataset),
        "minibatches_sha256": _tensor_digest(batches),
        "connection_permutation_sha256": _tensor_digest(connection_permutation),
        "target_permutation_sha256": _tensor_digest(target_permutation),
    }
    return dataset, batches, connection_permutation, target_permutation, hashes


def build_evaluation_material(
    config: AcquisitionConfig, seed: int
) -> tuple[dict[str, rel.RelationalDataset], dict[str, torch.Tensor], dict[str, str]]:
    datasets = {
        regime: generate_dataset(
            regime,
            dataset_seed=EVALUATION_SEED_BASES[regime] + seed,
            sample_count=config.evaluation_samples,
        )
        for regime in REGIMES
    }
    permutations = {
        regime: rel.sattolo_derangement(
            config.evaluation_samples,
            EVALUATION_CONNECTION_SHUFFLE_SEED_BASES[regime] + seed,
        )
        for regime in REGIMES
    }
    hashes: dict[str, str] = {}
    for regime in REGIMES:
        hashes[f"{regime}_data_sha256"] = dataset_digest(datasets[regime])
        hashes[f"{regime}_connection_permutation_sha256"] = _tensor_digest(
            permutations[regime]
        )
    return datasets, permutations, hashes


def protocol_contract(
    dataset: rel.RelationalDataset,
    connection_permutation: torch.Tensor,
    target_permutation: torch.Tensor,
    evaluations: Mapping[str, rel.RelationalDataset],
    evaluation_permutations: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    count = len(dataset.target)
    connection_fixed = int(
        (connection_permutation == torch.arange(count)).sum()
    )
    target_fixed = int((target_permutation == torch.arange(count)).sum())
    connection_change = float(
        (
            dataset.connection
            != dataset.connection[connection_permutation]
        ).any(dim=1).float().mean()
    )
    target_change = float(
        (dataset.target - dataset.target[target_permutation]).abs().mean()
    )
    evaluation_changes = {}
    for regime in REGIMES:
        evaluation = evaluations[regime]
        permutation = evaluation_permutations[regime]
        evaluation_changes[regime] = {
            "fixed_points": int(
                (permutation == torch.arange(len(permutation))).sum()
            ),
            "changed_connection_fraction": float(
                (
                    evaluation.connection
                    != evaluation.connection[permutation]
                ).any(dim=1).float().mean()
            ),
            "target_bin_coverage": int(
                torch.unique(
                    fc.joint.interval_posterior_unclipped(
                        evaluation.target.double(), 16
                    ).argmax(dim=-1)
                ).numel()
            ),
            "saturation_count": evaluation.saturation_count,
        }
    train_coverage = int(
        torch.unique(
            fc.joint.interval_posterior_unclipped(
                dataset.target.double(), 16
            ).argmax(dim=-1)
        ).numel()
    )
    passed = bool(
        dataset.saturation_count == 0
        and train_coverage == 16
        and connection_fixed == 0
        and target_fixed == 0
        and connection_change >= 0.90
        and target_change >= 0.25
        and all(
            item["fixed_points"] == 0
            and item["changed_connection_fraction"] >= 0.90
            and item["target_bin_coverage"] == 16
            and item["saturation_count"] == 0
            for item in evaluation_changes.values()
        )
    )
    return {
        "training_saturation_count": dataset.saturation_count,
        "training_target_bin_coverage": train_coverage,
        "connection_permutation_fixed_points": connection_fixed,
        "target_permutation_fixed_points": target_fixed,
        "training_changed_connection_fraction": connection_change,
        "training_mean_absolute_target_change": target_change,
        "evaluation": evaluation_changes,
        "pass": passed,
    }


def _initial_module(seed: int, device: torch.device) -> fc.ConnectionInvariantRelationalModule:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    module = fc.ConnectionInvariantRelationalModule().to(device)
    if sum(parameter.numel() for parameter in module.parameters()) != TOTAL_PARAMETER_COUNT:
        raise RuntimeError("connection module parameter count changed")
    return module


def _arm_training_material(
    arm: str,
    dataset: rel.RelationalDataset,
    connection_permutation: torch.Tensor,
    target_permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if arm == "learned_true":
        return dataset.connection, dataset.target
    if arm == "learned_no_connection":
        return torch.zeros_like(dataset.connection), dataset.target
    if arm == "learned_connection_shuffled":
        return dataset.connection[connection_permutation], dataset.target
    if arm == "learned_target_shuffled":
        return dataset.connection, dataset.target[target_permutation]
    raise ValueError(f"unknown acquisition arm {arm!r}")


def _arm_evaluation_connection(
    arm: str,
    dataset: rel.RelationalDataset,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if arm in ("learned_true", "learned_target_shuffled"):
        return dataset.connection
    if arm == "learned_no_connection":
        return torch.zeros_like(dataset.connection)
    if arm == "learned_connection_shuffled":
        return dataset.connection[permutation]
    raise ValueError(f"unknown acquisition arm {arm!r}")


def train_range(
    module: fc.ConnectionInvariantRelationalModule,
    optimizer: torch.optim.AdamW,
    dataset: rel.RelationalDataset,
    batches: torch.Tensor,
    connection: torch.Tensor,
    target: torch.Tensor,
    config: AcquisitionConfig,
    start: int,
    stop: int,
) -> list[dict[str, float]]:
    module.train()
    history = []
    device = next(module.parameters()).device
    for step in range(start, stop):
        indices = batches[step].to(device)
        prediction = module(
            dataset.tokens[indices],
            dataset.calibration[indices],
            connection[indices],
        )
        loss = torch.nn.functional.mse_loss(
            prediction.double(), target[indices].double()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            module.parameters(), config.gradient_clip
        )
        optimizer.step()
        history.append(
            {
                "step": float(step + 1),
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return history


@torch.no_grad()
def predict(
    module: fc.ConnectionInvariantRelationalModule,
    dataset: rel.RelationalDataset,
    connection: torch.Tensor,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    module.eval()
    values = []
    for start in range(0, len(dataset.target), batch_size):
        stop = start + batch_size
        values.append(
            module(
                dataset.tokens[start:stop],
                dataset.calibration[start:stop],
                connection[start:stop],
            ).cpu()
        )
    return torch.cat(values)


def _endpoint_pass(
    scalar: Mapping[str, Any], task: Mapping[str, Any], gates: Mapping[str, Any]
) -> bool:
    return bool(
        float(scalar["correlation"]) >= gates["scalar_correlation_minimum"]
        and float(scalar["rmse"]) <= gates["scalar_rmse_maximum"]
        and float(task["exact_bin_accuracy"])
        >= gates["exact_bin_accuracy_minimum"]
        and float(task["target_cross_entropy"])
        <= gates["target_cross_entropy_maximum"]
        and int(task["predicted_bin_coverage"])
        == gates["predicted_bin_coverage"]
    )


@torch.no_grad()
def evaluate_module(
    module: fc.ConnectionInvariantRelationalModule,
    dataset: rel.RelationalDataset,
    connection: torch.Tensor,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = predict(module, dataset, connection)
    target = dataset.target.cpu()
    scalar = rel._scalar_metrics(prediction, target)
    task = rel._task_metrics(prediction, target)
    return {
        "scalar": scalar,
        "task": task,
        "endpoint_pass": _endpoint_pass(scalar, task, gates),
        "prediction_sha256": _tensor_digest(prediction),
    }


@torch.no_grad()
def action_error(
    module: fc.ConnectionInvariantRelationalModule,
    dataset: rel.RelationalDataset,
    connection: torch.Tensor,
    action: torch.Tensor,
) -> float:
    base = predict(module, dataset, connection)
    transformed_dataset = rel.RelationalDataset(
        canonical_tokens=dataset.canonical_tokens,
        tokens=rel.apply_local_action(dataset.tokens, action),
        calibration=dataset.calibration,
        target=dataset.target,
        phase=dataset.phase,
        delta=dataset.delta,
        gauge=(dataset.gauge + action) % rel.CHANNELS,
        connection=rel.transform_connection(connection, action),
        saturation_count=dataset.saturation_count,
        dataset_seed=dataset.dataset_seed,
    )
    transformed = predict(
        module, transformed_dataset, transformed_dataset.connection
    )
    return float((transformed - base).abs().max())


@torch.no_grad()
def winding_diagnostic(
    module: fc.ConnectionInvariantRelationalModule, points: int = 1_025
) -> dict[str, Any]:
    device = next(module.parameters()).device
    theta = torch.linspace(-math.pi, math.pi, points, device=device)
    channel = (
        2.0
        * math.pi
        * torch.arange(rel.CHANNELS, dtype=torch.float32, device=device)
        / rel.CHANNELS
    )
    corrected = torch.cos(theta[:, None] + channel[None, :])[:, None, :]
    encoder = module.encoder
    features = encoder.shared_map(corrected.unsqueeze(-1))
    real = torch.einsum(
        "btck,c->btk", features, encoder.character_real
    )
    imaginary = torch.einsum(
        "btck,c->btk", features, encoder.character_imag
    )
    mixed_real = torch.einsum(
        "btk,k->bt", real, encoder.mixer_real
    ) - torch.einsum("btk,k->bt", imaginary, encoder.mixer_imag)
    mixed_imaginary = torch.einsum(
        "btk,k->bt", real, encoder.mixer_imag
    ) + torch.einsum("btk,k->bt", imaginary, encoder.mixer_real)
    value = torch.complex(mixed_real[:, 0], mixed_imaginary[:, 0])
    increments = torch.angle(value[1:] * value[:-1].conj())
    winding = int(round(float(increments.sum().cpu() / (2.0 * math.pi))))
    magnitude = value.abs()
    return {
        "winding_number": winding,
        "minimum_raw_magnitude": float(magnitude.min().cpu()),
        "median_raw_magnitude": float(magnitude.median().cpu()),
    }


def _checkpoint_payload(
    module: fc.ConnectionInvariantRelationalModule,
    optimizer: torch.optim.AdamW,
    *,
    arm: str,
    seed: int,
    step: int,
    fingerprint: str,
    protocol_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "arm": arm,
        "seed": seed,
        "step": step,
        "scientific_fingerprint": fingerprint,
        "protocol_hashes": dict(protocol_hashes),
        "module_state": module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": _sha256(path)}


def load_checkpoint(
    path: Path,
    module: fc.ConnectionInvariantRelationalModule,
    optimizer: torch.optim.AdamW,
    *,
    arm: str,
    seed: int,
    step: int,
    fingerprint: str,
    protocol_hashes: Mapping[str, str],
    device: torch.device,
) -> None:
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "arm": arm,
        "seed": seed,
        "step": step,
        "scientific_fingerprint": fingerprint,
        "protocol_hashes": dict(protocol_hashes),
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"checkpoint provenance mismatch: {path}")
    module.load_state_dict(payload["module_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in payload["cuda_rng_state"]]
        )


def train_arm(
    *,
    arm: str,
    seed: int,
    config: AcquisitionConfig,
    train_dataset: rel.RelationalDataset,
    batches: torch.Tensor,
    connection_permutation: torch.Tensor,
    target_permutation: torch.Tensor,
    evaluations: Mapping[str, rel.RelationalDataset],
    evaluation_permutations: Mapping[str, torch.Tensor],
    action: torch.Tensor,
    device: torch.device,
    cell_dir: Path,
    fingerprint: str,
    protocol_hashes: Mapping[str, str],
) -> dict[str, Any]:
    module = _initial_module(seed, device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    training_connection, training_target = _arm_training_material(
        arm, train_dataset, connection_permutation, target_permutation
    )
    composition_connection = _arm_evaluation_connection(
        arm, evaluations["composition"], evaluation_permutations["composition"]
    )
    initial_state = fc._state_digest(module)
    initial_action_error = action_error(
        module, evaluations["composition"], composition_connection, action
    )
    winding = {"initial": winding_diagnostic(module)}
    first_history = train_range(
        module,
        optimizer,
        train_dataset,
        batches,
        training_connection,
        training_target,
        config,
        0,
        config.split_step,
    )
    midpoint_action_error = action_error(
        module, evaluations["composition"], composition_connection, action
    )
    winding["midpoint"] = winding_diagnostic(module)
    midpoint_path = cell_dir / f"{arm}_midpoint.pt"
    midpoint = save_checkpoint(
        midpoint_path,
        _checkpoint_payload(
            module,
            optimizer,
            arm=arm,
            seed=seed,
            step=config.split_step,
            fingerprint=fingerprint,
            protocol_hashes=protocol_hashes,
        ),
    )
    second_history = train_range(
        module,
        optimizer,
        train_dataset,
        batches,
        training_connection,
        training_target,
        config,
        config.split_step,
        config.training_steps,
    )
    final_state = fc._state_digest(module)
    final_optimizer = fc.stage0._optimizer_digest(optimizer)
    final_action_error = action_error(
        module, evaluations["composition"], composition_connection, action
    )
    winding["final"] = winding_diagnostic(module)
    final_path = cell_dir / f"{arm}_final.pt"
    final = save_checkpoint(
        final_path,
        _checkpoint_payload(
            module,
            optimizer,
            arm=arm,
            seed=seed,
            step=config.training_steps,
            fingerprint=fingerprint,
            protocol_hashes=protocol_hashes,
        ),
    )
    regimes = {}
    final_predictions = {}
    for regime in REGIMES:
        supplied = _arm_evaluation_connection(
            arm, evaluations[regime], evaluation_permutations[regime]
        )
        regimes[regime] = evaluate_module(
            module, evaluations[regime], supplied, LEARNED_GATES
        )
        final_predictions[regime] = predict(
            module, evaluations[regime], supplied
        )

    reloaded = _initial_module(seed, device)
    reloaded_optimizer = torch.optim.AdamW(
        reloaded.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    load_checkpoint(
        final_path,
        reloaded,
        reloaded_optimizer,
        arm=arm,
        seed=seed,
        step=config.training_steps,
        fingerprint=fingerprint,
        protocol_hashes=protocol_hashes,
        device=device,
    )
    reload_errors = {}
    for regime in REGIMES:
        supplied = _arm_evaluation_connection(
            arm, evaluations[regime], evaluation_permutations[regime]
        )
        reload_errors[regime] = float(
            (
                predict(reloaded, evaluations[regime], supplied)
                - final_predictions[regime]
            ).abs().max()
        )
    reload_state = fc._state_digest(reloaded)
    reload_optimizer = fc.stage0._optimizer_digest(reloaded_optimizer)
    reload_pass = bool(
        reload_state == final_state
        and reload_optimizer == final_optimizer
        and all(value == 0.0 for value in reload_errors.values())
    )

    resumed = _initial_module(seed, device)
    resumed_optimizer = torch.optim.AdamW(
        resumed.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    load_checkpoint(
        midpoint_path,
        resumed,
        resumed_optimizer,
        arm=arm,
        seed=seed,
        step=config.split_step,
        fingerprint=fingerprint,
        protocol_hashes=protocol_hashes,
        device=device,
    )
    resumed_history = train_range(
        resumed,
        resumed_optimizer,
        train_dataset,
        batches,
        training_connection,
        training_target,
        config,
        config.split_step,
        config.training_steps,
    )
    resume_errors = {}
    for regime in REGIMES:
        supplied = _arm_evaluation_connection(
            arm, evaluations[regime], evaluation_permutations[regime]
        )
        resume_errors[regime] = float(
            (
                predict(resumed, evaluations[regime], supplied)
                - final_predictions[regime]
            ).abs().max()
        )
    resume_state = fc._state_digest(resumed)
    resume_optimizer = fc.stage0._optimizer_digest(resumed_optimizer)
    resume_pass = bool(
        resume_state == final_state
        and resume_optimizer == final_optimizer
        and resumed_history == second_history
        and all(value == 0.0 for value in resume_errors.values())
    )
    action_errors = {
        "initial": initial_action_error,
        "midpoint": midpoint_action_error,
        "final": final_action_error,
    }
    action_pass = max(action_errors.values()) <= ACTION_ERROR_MAXIMUM
    finite = _finite(first_history) and _finite(second_history) and _finite(regimes)
    lifecycle_valid = bool(
        finite
        and final_state != initial_state
        and action_pass
        and reload_pass
        and resume_pass
    )
    joint_pass = bool(
        lifecycle_valid
        and all(regimes[regime]["endpoint_pass"] for regime in REGIMES)
    )
    del module, reloaded, resumed, optimizer, reloaded_optimizer, resumed_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "arm": arm,
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "final_optimizer_sha256": final_optimizer,
        "state_changed": final_state != initial_state,
        "training_history": first_history + second_history,
        "midpoint_checkpoint": midpoint,
        "final_checkpoint": final,
        "winding_diagnostic": winding,
        "action_errors": action_errors,
        "action_pass": action_pass,
        "regimes": regimes,
        "reload": {
            "state_sha256": reload_state,
            "optimizer_sha256": reload_optimizer,
            "maximum_prediction_errors": reload_errors,
            "pass": reload_pass,
        },
        "exact_resume": {
            "state_sha256": resume_state,
            "optimizer_sha256": resume_optimizer,
            "history_exact": resumed_history == second_history,
            "maximum_prediction_errors": resume_errors,
            "verification_optimizer_steps": config.training_steps
            - config.split_step,
            "pass": resume_pass,
        },
        "finite": finite,
        "lifecycle_valid": lifecycle_valid,
        "joint_pass": joint_pass,
    }


def evaluate_analytic_seed(
    config: AcquisitionConfig, seed: int
) -> dict[str, Any]:
    datasets, _permutations, hashes = build_evaluation_material(config, seed)
    torch.manual_seed(0)
    module = fc.construct_analytic_witness(
        fc.ConnectionInvariantRelationalModule()
    )
    action = torch.randint(
        0,
        rel.CHANNELS,
        (config.evaluation_samples, rel.TIME_STEPS),
        generator=torch.Generator(device="cpu").manual_seed(
            ACTION_SEED_BASE + seed
        ),
        dtype=torch.int64,
    )
    regimes = {}
    for regime in REGIMES:
        result = evaluate_module(
            module, datasets[regime], datasets[regime].connection, ANALYTIC_GATES
        )
        result["action_error"] = action_error(
            module, datasets[regime], datasets[regime].connection, action
        )
        result["action_pass"] = result["action_error"] <= ACTION_ERROR_MAXIMUM
        result["joint_pass"] = bool(
            result["endpoint_pass"] and result["action_pass"]
        )
        regimes[regime] = result
    return {
        "seed": seed,
        "evaluation_hashes": hashes,
        "regimes": regimes,
        "winding_diagnostic": winding_diagnostic(module),
        "joint_pass": all(item["joint_pass"] for item in regimes.values()),
    }


def _cell_dir(root: Path, seed: int) -> Path:
    return root / "runs" / f"seed_{seed}"


def _scientific_fingerprint(
    *,
    config: AcquisitionConfig,
    seed: int,
    implementation: str,
    protocol_hashes: Mapping[str, str],
    evaluation_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "seed": seed,
            "implementation_sha256": implementation,
            "protocol_hashes": dict(protocol_hashes),
            "evaluation_hashes": dict(evaluation_hashes),
        }
    )


def acquisition_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    seed = int(experiment.seed)
    sources, _license = _validate_sources()
    implementation = _implementation_digest(sources)
    if implementation != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("acquisition implementation changed after scheduling")
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    train_cpu, batches, connection_permutation, target_permutation, protocol_hashes = (
        protocol_material(config, seed)
    )
    evaluations_cpu, evaluation_permutations, evaluation_hashes = (
        build_evaluation_material(config, seed)
    )
    contract = protocol_contract(
        train_cpu,
        connection_permutation,
        target_permutation,
        evaluations_cpu,
        evaluation_permutations,
    )
    if not contract["pass"]:
        raise RuntimeError("connection acquisition protocol contract failed")
    if evaluation_hashes != experiment.parameters["evaluation_hashes"]:
        raise RuntimeError("connection acquisition evaluation cohorts changed")
    fingerprint = _scientific_fingerprint(
        config=config,
        seed=seed,
        implementation=implementation,
        protocol_hashes=protocol_hashes,
        evaluation_hashes=evaluation_hashes,
    )
    if fingerprint != experiment.parameters["scientific_fingerprint"]:
        raise RuntimeError("connection acquisition fingerprint changed")
    train = _to_device(train_cpu, device)
    evaluations = {
        regime: _to_device(evaluations_cpu[regime], device) for regime in REGIMES
    }
    connection_permutation = connection_permutation.to(device)
    target_permutation = target_permutation.to(device)
    evaluation_permutations = {
        regime: evaluation_permutations[regime].to(device) for regime in REGIMES
    }
    action = torch.randint(
        0,
        rel.CHANNELS,
        (config.evaluation_samples, rel.TIME_STEPS),
        generator=torch.Generator(device="cpu").manual_seed(
            ACTION_SEED_BASE + seed
        ),
        dtype=torch.int64,
    ).to(device)
    cell_dir = _cell_dir(Path(experiment.parameters["output_dir"]), seed)
    cell_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    initial = _initial_module(seed, device)
    initial_digest = fc._state_digest(initial)
    del initial
    arms = {
        arm: train_arm(
            arm=arm,
            seed=seed,
            config=config,
            train_dataset=train,
            batches=batches,
            connection_permutation=connection_permutation,
            target_permutation=target_permutation,
            evaluations=evaluations,
            evaluation_permutations=evaluation_permutations,
            action=action,
            device=device,
            cell_dir=cell_dir,
            fingerprint=fingerprint,
            protocol_hashes=protocol_hashes,
        )
        for arm in ARMS
    }
    matched_initialization = all(
        item["initial_state_sha256"] == initial_digest for item in arms.values()
    )
    validity = bool(
        contract["pass"]
        and matched_initialization
        and all(item["lifecycle_valid"] for item in arms.values())
    )
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result_path = cell_dir / "result.json"
    gates = {
        "validity": validity,
        "matched_initialization": matched_initialization,
        **{f"{arm}_joint": arms[arm]["joint_pass"] for arm in ARMS},
    }
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "seed": seed,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "configuration": _json_config(config),
        "source_hashes": sources,
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "protocol_hashes": protocol_hashes,
        "evaluation_hashes": evaluation_hashes,
        "protocol_contract": contract,
        "matched_initialization_sha256": initial_digest,
        "arms": arms,
        "gates": gates,
        "accounting": {
            "tinyllm_models_instantiated": 0,
            "trainable_parameters_per_arm": TOTAL_PARAMETER_COUNT,
            "learned_arms": len(ARMS),
            "primary_optimizer_steps": len(ARMS) * config.training_steps,
            "resume_verification_optimizer_steps": len(ARMS)
            * (config.training_steps - config.split_step),
        },
        "wall_seconds": elapsed,
        "peak_cuda_allocated_gb": peak,
        "artifacts": {
            "result": str(result_path),
            "checkpoints": {
                arm: {
                    "midpoint": item["midpoint_checkpoint"],
                    "final": item["final_checkpoint"],
                }
                for arm, item in arms.items()
            },
        },
    }
    if not _finite(detail):
        raise RuntimeError("non-finite connection acquisition result")
    _write_json(result_path, detail)
    checkpoint = arms["learned_true"]["final_checkpoint"]["path"]
    del arms, train, train_cpu, evaluations, evaluations_cpu
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics={
            "validity": float(validity),
            **{f"{arm}_joint": float(gates[f"{arm}_joint"]) for arm in ARMS},
            "peak_cuda_allocated_gb": peak,
        },
        primary_metric=float(gates["learned_true_joint"]),
        model_architecture=[1, 16, 8, 2, 1],
        model_parameters=TOTAL_PARAMETER_COUNT,
        training_time=elapsed,
        model_checkpoint=checkpoint,
        observations=[
            "Matched connection-invariant relational acquisition campaign.",
            f"Seed={seed}; four learned true/information/control arms.",
        ],
        timestamp=datetime.fromisoformat(detail["completed_at"]),
    )


def _experiment(
    config: AcquisitionConfig,
    seed: int,
    output_dir: Path,
    implementation: str,
) -> Experiment:
    _dataset, _batches, _connection, _target, protocol_hashes = protocol_material(
        config, seed
    )
    _evaluations, _permutations, evaluation_hashes = build_evaluation_material(
        config, seed
    )
    fingerprint = _scientific_fingerprint(
        config=config,
        seed=seed,
        implementation=implementation,
        protocol_hashes=protocol_hashes,
        evaluation_hashes=evaluation_hashes,
    )
    return Experiment(
        id=f"tinyllm-c3-relational-connection-acquisition-seed{seed}",
        hypothesis_id=HYPOTHESIS_ID,
        name=f"TinyLLM C3 relational connection acquisition seed {seed}",
        parameters={
            "configuration": _json_config(config),
            "output_dir": str(output_dir),
            "implementation_sha256": implementation,
            "scientific_fingerprint": fingerprint,
            "evaluation_hashes": evaluation_hashes,
        },
        seed=seed,
    )


def _existing_detail(
    experiment: Experiment, output_dir: Path
) -> Optional[dict[str, Any]]:
    path = _cell_dir(output_dir, int(experiment.seed)) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint")
        != experiment.parameters["scientific_fingerprint"]
        or detail.get("implementation_sha256")
        != experiment.parameters["implementation_sha256"]
    ):
        return None
    for arm in ARMS:
        for point in ("midpoint", "final"):
            artifact = (
                detail.get("artifacts", {})
                .get("checkpoints", {})
                .get(arm, {})
                .get(point, {})
            )
            artifact_path = Path(str(artifact.get("path", "")))
            if (
                not artifact_path.is_file()
                or _sha256(artifact_path) != artifact.get("sha256")
            ):
                return None
    return detail


def aggregate_details(
    details: Sequence[Mapping[str, Any]],
    analytic: Sequence[Mapping[str, Any]],
    config: AcquisitionConfig,
) -> dict[str, Any]:
    analytic_count = sum(bool(item["joint_pass"]) for item in analytic)
    analytic_pass = analytic_count >= config.required_seed_passes
    valid = bool(
        len(details) == len(config.seeds)
        and all(item.get("gates", {}).get("validity") is True for item in details)
    )
    counts = {
        arm: sum(
            bool(item.get("gates", {}).get(f"{arm}_joint")) for item in details
        )
        for arm in ARMS
    }
    controls_specific = all(
        counts[arm] <= config.control_seed_passes_maximum
        for arm in ARMS
        if arm != "learned_true"
    )
    if config.allow_underpowered:
        classification = "lifecycle_completed_not_quality_evidence"
        primary = False
    elif not analytic_pass:
        classification = "analytic_connection_ceiling_failed_on_fresh_streams"
        primary = False
    elif not valid:
        classification = "invalid_connection_acquisition_campaign"
        primary = False
    elif not controls_specific:
        classification = "connection_acquisition_specificity_failed"
        primary = False
    elif counts["learned_true"] >= config.required_seed_passes:
        classification = "connection_invariant_relation_acquired_by_gradient_training"
        primary = True
    else:
        classification = "exact_function_class_but_population_acquisition_unreliable"
        primary = False
    return {
        "valid": valid,
        "analytic_joint_pass_count": analytic_count,
        "analytic_stop_gate_pass": analytic_pass,
        "arm_joint_pass_counts": counts,
        "required_seed_passes": config.required_seed_passes,
        "control_seed_passes_maximum": config.control_seed_passes_maximum,
        "control_specificity_pass": controls_specific,
        "classification": classification,
        "primary_hypothesis_pass": primary,
        "unrestricted_tinyllm_training_licensed": False,
        "seed_gates": {str(item["seed"]): item["gates"] for item in details},
    }


async def run_campaign(
    config: AcquisitionConfig, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, license_record = _validate_sources()
    implementation = _implementation_digest(sources)
    analytic = [evaluate_analytic_seed(config, seed) for seed in config.seeds]
    analytic_count = sum(bool(item["joint_pass"]) for item in analytic)
    analytic_pass = analytic_count >= config.required_seed_passes
    experiments = [
        _experiment(config, seed, output_dir, implementation)
        for seed in config.seeds
    ]
    campaign_fingerprint = _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "analytic_evaluation_hashes": [
                item["evaluation_hashes"] for item in analytic
            ],
            "experiment_fingerprints": [
                item.parameters["scientific_fingerprint"] for item in experiments
            ],
        }
    )
    campaign_path = output_dir / "campaign_results.json"
    if campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing_campaign.get("status") in ("completed", "stopped")
            and existing_campaign.get("campaign_fingerprint")
            == campaign_fingerprint
            and (
                existing_campaign.get("status") == "stopped"
                or all(
                    _existing_detail(item, output_dir) is not None
                    for item in experiments
                )
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing_campaign

    details: list[dict[str, Any]] = []
    runner = None
    reused = 0
    scheduled = 0
    if analytic_pass or config.allow_underpowered:
        existing = {
            item.id: detail
            for item in experiments
            if (detail := _existing_detail(item, output_dir)) is not None
        }
        pending = [item for item in experiments if item.id not in existing]
        reused = len(existing)
        scheduled = len(pending)
        lab = LabConfig(
            project_name="tinyllm_c3_relational_connection_acquisition",
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
        runner = AsyncExperimentRunner(lab, acquisition_worker)
        results = await runner.run_experiments(pending) if pending else []
        successful = {item.experiment_id for item in results if item.error is None}
        details = list(existing.values())
        for experiment in pending:
            if experiment.id in successful:
                detail = _existing_detail(experiment, output_dir)
                if detail is not None:
                    details.append(detail)
        details.sort(key=lambda item: int(item["seed"]))

    aggregates = aggregate_details(details, analytic, config)
    complete = len(details) == len(experiments)
    stopped = not analytic_pass and not config.allow_underpowered
    result_paths = [
        _cell_dir(output_dir, seed) / "result.json" for seed in config.seeds
    ]
    checkpoint_paths = [
        _cell_dir(output_dir, seed) / f"{arm}_{point}.pt"
        for seed in config.seeds
        for arm in ARMS
        for point in ("midpoint", "final")
    ]
    status = (
        "stopped"
        if stopped
        else "completed"
        if complete and aggregates["valid"]
        else "partial"
    )
    scheduler = None
    if runner is not None:
        scheduler = {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value
                for key, value in runner.slot_plan.slots_by_device.items()
            },
            "free_memory_gb": {
                str(key): value
                for key, value in runner.slot_plan.free_memory_gb.items()
            },
            "calibration": runner.slot_plan.calibration,
        }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": status,
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "configuration": _json_config(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "source_hashes": sources,
        "implementation_sha256": implementation,
        "function_class_license": {
            "path": str(FUNCTION_CLASS_RESULT_PATH),
            "sha256": sources["function_class_result"],
            "classification": license_record["aggregates"]["classification"],
        },
        "analytic_positive_control": analytic,
        "campaign_fingerprint": campaign_fingerprint,
        "scheduler": scheduler,
        "summary": {
            "requested": len(experiments),
            "reused": reused,
            "scheduled": scheduled,
            "completed": len(details),
            "failed": 0 if stopped else len(experiments) - len(details),
            "primary_optimizer_steps": sum(
                item["accounting"]["primary_optimizer_steps"] for item in details
            ),
            "resume_verification_optimizer_steps": sum(
                item["accounting"]["resume_verification_optimizer_steps"]
                for item in details
            ),
            "tinyllm_models_instantiated": 0,
        },
        "aggregates": aggregates,
        "result_manifest_sha256": _manifest(result_paths) if complete else None,
        "checkpoint_manifest_sha256": (
            _manifest(checkpoint_paths) if complete else None
        ),
        "maximum_peak_cuda_allocated_gb": max(
            (float(item["peak_cuda_allocated_gb"]) for item in details),
            default=0.0,
        ),
        "aggregate_cell_wall_seconds": sum(
            float(item["wall_seconds"]) for item in details
        ),
    }
    if not _finite(bundle):
        raise RuntimeError("non-finite connection acquisition campaign")
    _write_json(campaign_path, bundle)
    return bundle


def _mode_config(mode: str) -> AcquisitionConfig:
    if mode == "primary":
        return AcquisitionConfig()
    device = (-1,) if mode == "cpu-shakedown" else (0,)
    return AcquisitionConfig(
        seeds=(1409,),
        training_steps=2,
        train_samples=128,
        evaluation_samples=128,
        batch_size=8,
        required_seed_passes=1,
        device_ids=device,
        gpu_slots_per_device=1,
        max_gpu_slots_per_device=1,
        max_parallel_experiments=1,
        allow_underpowered=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cpu-shakedown", "cuda-shakedown", "primary"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _mode_config(args.mode)
    output = args.output
    if output is None:
        output = (
            PRIMARY_OUTPUT
            if args.mode == "primary"
            else Path(f"/tmp/tinyllm-c3-relational-acquisition-{args.mode}")
        )
    bundle = asyncio.run(run_campaign(config, output))
    print(
        json.dumps(
            {
                "output": str(output),
                "status": bundle["status"],
                "classification": bundle["aggregates"]["classification"],
                "analytic_passes": bundle["aggregates"][
                    "analytic_joint_pass_count"
                ],
                "arm_passes": bundle["aggregates"]["arm_joint_pass_counts"],
                "completed": bundle["summary"]["completed"],
            },
            indent=2,
        )
    )
    return 0 if bundle["status"] in ("completed", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
