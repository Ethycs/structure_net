#!/usr/bin/env python3
"""Train only the exact-C3 sensor against a frozen temporal task interface."""

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

import experiments.structure_net.tinyllm_c3_temporal_quotient_analysis as analysis
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as generator
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0
import experiments.structure_net.tinyllm_c3_temporal_sensor_function_class as capacity
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-sensor-only.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-sensor-only-v1"
EVIDENCE_ROLE = "prospective_c3_sensor_only_fixed_operator_training"
ARMS = ("learned_true", "learned_target_shuffled")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-sensor-only-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "e0c16420ea79130b22d6940b6f4943b5dda6bd3e048418eff078c0190e69a2bb"
)
CAPACITY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_function_class/"
    "20260811_preregistered/result.json"
)
PINNED_SOURCE_SHA256 = {
    "preregistration": PREREGISTRATION_SHA256,
    "capacity_result": "6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383",
    "capacity_runner": "95331fa823f193449af1c46f961f22f4aaf902300695f179912b75d8ed4bcade",
    "sensor_family": "dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6",
    "generator": "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118",
    "interval_likelihood": "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6",
}
REFERENCE_SEED = 431_003
EVALUATION_SEEDS = {"composition": 331_003, "extrapolation": 331_021}
TASK_GATES = {
    "posterior_mean_correlation_minimum": 0.90,
    "composition_accuracy_minimum": 0.50,
    "extrapolation_accuracy_minimum": 0.35,
    "composition_cross_entropy_maximum": 1.80,
    "extrapolation_cross_entropy_maximum": 2.20,
    "composition_coverage_minimum": 14,
    "extrapolation_coverage_minimum": 12,
}
ACTION_ERROR_MAXIMUM = 2e-6
CARRIER_DOT_MINIMUM = 0.90
CARRIER_RMSE_MAXIMUM = 0.35
PARAMETER_COUNT = 184


@dataclass(frozen=True)
class SensorOnlyConfig:
    seeds: tuple[int, ...] = SEEDS
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    reference_samples: int = 1_024
    evaluation_samples: int = 1_024
    required_seed_passes: int = 4
    shuffled_seed_passes_maximum: int = 1
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 2
    gpu_memory_per_experiment_gb: float = 0.25
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.training_steps < 2 or self.training_steps % 2:
            raise ValueError("training steps must be positive and even")
        if self.train_samples < 4 or self.train_samples % 2:
            raise ValueError("training examples must form latent pairs")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch size must contain complete pairs")
        if self.reference_samples < 2 or self.evaluation_samples < 2:
            raise ValueError("held-out cohorts are too small")
        if not self.device_ids:
            raise ValueError("at least one logical device is required")
        if not self.allow_underpowered and (
            self.seeds != SEEDS
            or self.training_steps != 600
            or self.train_samples != 4_096
            or self.batch_size != 64
            or self.reference_samples != 1_024
            or self.evaluation_samples != 1_024
            or self.learning_rate != 3e-4
            or self.weight_decay != 0.01
            or self.gradient_clip != 1.0
            or self.required_seed_passes != 4
            or self.shuffled_seed_passes_maximum != 1
        ):
            raise ValueError("primary sensor-only protocol is frozen")

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


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "capacity_result": _sha256(CAPACITY_RESULT_PATH),
        "capacity_runner": _sha256(Path(capacity.__file__)),
        "sensor_family": _sha256(Path(stage0.__file__)),
        "generator": _sha256(Path(generator.__file__)),
        "interval_likelihood": _sha256(Path(joint.__file__)),
        "evaluation_generator": _sha256(Path(analysis.__file__)),
    }


def _validate_sources() -> tuple[dict[str, str], dict[str, Any]]:
    sources = _source_hashes()
    if any(sources[name] != expected for name, expected in PINNED_SOURCE_SHA256.items()):
        raise RuntimeError("sensor-only source contract changed")
    license_record = json.loads(CAPACITY_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        license_record.get("classification") != capacity.CLASSIFICATION_PASS
        or license_record.get("gates", {}).get("campaign_licensed") is not True
        or license_record.get("gates", {}).get("zero_optimizer_steps") is not True
    ):
        raise RuntimeError("sensor-only campaign is not licensed")
    return sources, license_record


def _json_config(config: SensorOnlyConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(value: Mapping[str, Any]) -> SensorOnlyConfig:
    converted = dict(value)
    converted["seeds"] = tuple(converted["seeds"])
    converted["device_ids"] = tuple(converted["device_ids"])
    return SensorOnlyConfig(**converted)


def _implementation_digest(sources: Mapping[str, str]) -> str:
    return _json_hash(dict(sources))


def pair_preserving_target_permutation(sample_count: int, seed: int) -> torch.Tensor:
    if sample_count < 4 or sample_count % 2:
        raise ValueError("target permutation requires complete latent pairs")
    latent_count = sample_count // 2
    shift = 1 + (int(seed) % (latent_count - 1))
    permutation = torch.roll(torch.arange(latent_count), shifts=shift)
    return torch.stack((2 * permutation, 2 * permutation + 1), dim=1).reshape(-1)


def protocol_material(
    task: stage0.C3TaskConfig,
    config: SensorOnlyConfig,
    seed: int,
) -> tuple[stage0.C3TrainingDataset, torch.Tensor, torch.Tensor, dict[str, str]]:
    dataset = stage0.generate_training_dataset(
        task, sample_count=config.train_samples, seed=seed + 1_001
    )
    generator_state = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    pair_batches = torch.randint(
        0,
        config.train_samples // 2,
        (config.training_steps, config.batch_size // 2),
        generator=generator_state,
    )
    target_permutation = pair_preserving_target_permutation(
        config.train_samples, seed + 27_017
    )
    hashes = {
        "training_data_sha256": stage0._digest_tensors(
            dataset.canonical_tokens,
            dataset.tokens,
            dataset.calibration,
            dataset.target,
            dataset.target_posteriors,
            dataset.deck,
            dataset.pair_id,
        ),
        "minibatches_sha256": _tensor_digest(pair_batches),
        "target_permutation_sha256": _tensor_digest(target_permutation),
    }
    return dataset, pair_batches, target_permutation, hashes


def protocol_contract(
    dataset: stage0.C3TrainingDataset,
    target_permutation: torch.Tensor,
) -> dict[str, Any]:
    source = stage0.validate_training_protocol(dataset)
    pair_identity = bool(
        torch.all(target_permutation[0::2] % 2 == 0)
        and torch.all(target_permutation[1::2] == target_permutation[0::2] + 1)
    )
    pair_fixed_points = int(
        (target_permutation[0::2] == torch.arange(0, len(dataset), 2)).sum()
    )
    target_change = float(
        (
            dataset.target_posteriors
            - dataset.target_posteriors[target_permutation]
        ).abs().sum(-1).mean()
    )
    return {
        "source": source,
        "pair_preserving_target_permutation": pair_identity,
        "target_pair_fixed_points": pair_fixed_points,
        "mean_target_posterior_l1_change": target_change,
        "pass": bool(
            source["pass"]
            and pair_identity
            and pair_fixed_points == 0
            and target_change >= 0.25
        ),
    }


def build_held_out_datasets(
    task: stage0.C3TaskConfig,
    config: SensorOnlyConfig,
) -> dict[str, stage0.C3TrainingDataset]:
    values = {
        "reference": analysis.generate_shift_dataset(
            task,
            regime="composition",
            latent_count=config.reference_samples,
            seed=REFERENCE_SEED,
            all_sheets=False,
        )
    }
    for regime in REGIMES:
        values[regime] = analysis.generate_shift_dataset(
            task,
            regime=regime,
            latent_count=config.evaluation_samples,
            seed=EVALUATION_SEEDS[regime],
            all_sheets=False,
        )
    if any(dataset.saturation_count for dataset in values.values()):
        raise RuntimeError("held-out C3 sensor cohort saturated")
    return values


def held_out_hashes(
    datasets: Mapping[str, stage0.C3TrainingDataset],
) -> dict[str, str]:
    return {
        name: stage0._digest_tensors(
            dataset.canonical_tokens,
            dataset.tokens,
            dataset.calibration,
            dataset.target,
            dataset.target_posteriors,
            dataset.deck,
        )
        for name, dataset in datasets.items()
    }


def _initial_encoder(seed: int, device: torch.device) -> stage0.LearnedC3InvariantEncoder:
    torch.manual_seed(seed + 91_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 91_001)
    encoder = stage0.LearnedC3InvariantEncoder().to(device)
    if sum(parameter.numel() for parameter in encoder.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("sensor parameter count changed")
    return encoder


def _training_indices(pair_batch: torch.Tensor) -> torch.Tensor:
    return torch.stack((2 * pair_batch, 2 * pair_batch + 1), dim=-1).reshape(-1)


def _loss(
    encoder: stage0.LearnedC3InvariantEncoder,
    dataset: stage0.C3TrainingDataset,
    indices: torch.Tensor,
    target_posteriors: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    corrected = stage0.corrected_channels(
        dataset.tokens[indices].to(device),
        dataset.calibration[indices].to(device),
    )
    feature = encoder(corrected)
    carrier = torch.complex(feature[..., 0], feature[..., 1])
    temporal = stage0.preflight.temporal_prediction(carrier.to(torch.complex128))
    posterior = joint.interval_posterior_unclipped(temporal, 16)
    target = target_posteriors[indices].to(device=device, dtype=torch.float64)
    return -(target * posterior.clamp_min(1e-12).log()).sum(-1).mean()


def train_range(
    encoder: stage0.LearnedC3InvariantEncoder,
    optimizer: torch.optim.AdamW,
    dataset: stage0.C3TrainingDataset,
    pair_batches: torch.Tensor,
    target_posteriors: torch.Tensor,
    config: SensorOnlyConfig,
    device: torch.device,
    start: int,
    stop: int,
) -> list[dict[str, float]]:
    history = []
    encoder.train()
    for step in range(start, stop):
        indices = _training_indices(pair_batches[step])
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(
            encoder, dataset, indices, target_posteriors, device
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), config.gradient_clip
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


def _checkpoint_payload(
    encoder: stage0.LearnedC3InvariantEncoder,
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
        "encoder_state": encoder.state_dict(),
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
    encoder: stage0.LearnedC3InvariantEncoder,
    optimizer: torch.optim.AdamW,
    *,
    arm: str,
    seed: int,
    step: int,
    fingerprint: str,
    protocol_hashes: Mapping[str, str],
    device: torch.device,
) -> dict[str, Any]:
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
    encoder.load_state_dict(payload["encoder_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in payload["cuda_rng_state"]]
        )
    return payload


@torch.no_grad()
def extract_carrier(
    encoder: stage0.LearnedC3InvariantEncoder,
    dataset: stage0.C3TrainingDataset,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    encoder.eval()
    values = []
    for start in range(0, len(dataset), batch_size):
        stop = start + batch_size
        corrected = stage0.corrected_channels(
            dataset.tokens[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        feature = encoder(corrected)
        values.append(torch.complex(feature[..., 0], feature[..., 1]).cpu())
    return torch.cat(values)


def analytic_carrier(dataset: stage0.C3TrainingDataset) -> torch.Tensor:
    value, _ = generator.analytic_carrier(dataset.tokens, dataset.calibration)
    return value.to(torch.complex64)


def fit_orthogonal_gauge(
    learned: torch.Tensor,
    analytic: torch.Tensor,
) -> dict[str, Any]:
    left = torch.view_as_real(learned).reshape(-1, 2).double()
    right = torch.view_as_real(analytic).reshape(-1, 2).double()
    u, singular, vh = torch.linalg.svd(left.T @ right)
    matrix = u @ vh
    aligned = left @ matrix
    dot = float((aligned * right).sum(-1).mean())
    rmse = float(torch.sqrt((aligned - right).square().mean()))
    return {
        "matrix": matrix,
        "matrix_values": matrix.tolist(),
        "determinant": float(torch.linalg.det(matrix)),
        "singular_values": singular.tolist(),
        "reference_mean_dot": dot,
        "reference_coordinate_rmse": rmse,
    }


def _task_gate(metrics: Mapping[str, Any], regime: str) -> bool:
    return (
        float(metrics["posterior_mean_correlation"])
        >= TASK_GATES["posterior_mean_correlation_minimum"]
        and float(metrics["exact_bin_accuracy"])
        >= TASK_GATES[f"{regime}_accuracy_minimum"]
        and float(metrics["target_cross_entropy"])
        <= TASK_GATES[f"{regime}_cross_entropy_maximum"]
        and int(metrics["predicted_bin_coverage"])
        >= TASK_GATES[f"{regime}_coverage_minimum"]
    )


@torch.no_grad()
def evaluate_carrier(
    encoder: stage0.LearnedC3InvariantEncoder,
    dataset: stage0.C3TrainingDataset,
    gauge: torch.Tensor,
    device: torch.device,
    regime: str,
) -> dict[str, Any]:
    learned = extract_carrier(encoder, dataset, device)
    analytic = analytic_carrier(dataset)
    left = torch.view_as_real(learned).reshape(-1, 2).double()
    right = torch.view_as_real(analytic).reshape(-1, 2).double()
    aligned = left @ gauge.double()
    mean_dot = float((aligned * right).sum(-1).mean())
    coordinate_rmse = float(torch.sqrt((aligned - right).square().mean()))
    direct_error = float((learned.to(torch.complex128) - analytic.to(torch.complex128)).abs().max())
    temporal = generator.temporal_prediction(learned.to(torch.complex128))
    posterior = joint.interval_posterior_unclipped(temporal, 16)
    task_metrics = capacity._task_metrics(posterior, dataset.target)
    action_errors = {}
    for element in (1, 2):
        transformed_dataset = stage0.C3TrainingDataset(
            canonical_tokens=dataset.canonical_tokens,
            tokens=generator.apply_deck_action(dataset.tokens, element),
            calibration=dataset.calibration,
            target=dataset.target,
            target_posteriors=dataset.target_posteriors,
            target_bins=dataset.target_bins,
            phase=dataset.phase,
            speed=dataset.speed,
            deck=(dataset.deck + element) % 3,
            pair_id=dataset.pair_id,
            saturation_count=dataset.saturation_count,
        )
        transformed = extract_carrier(encoder, transformed_dataset, device)
        action_errors[str(element)] = float((transformed - learned).abs().max())
    maximum_action_error = max(action_errors.values())
    carrier_pass = (
        mean_dot >= CARRIER_DOT_MINIMUM
        and coordinate_rmse <= CARRIER_RMSE_MAXIMUM
    )
    action_pass = maximum_action_error <= ACTION_ERROR_MAXIMUM
    task_pass = _task_gate(task_metrics, regime)
    return {
        "mean_aligned_unit_dot": mean_dot,
        "aligned_coordinate_rmse": coordinate_rmse,
        "direct_unaligned_carrier_maximum_error": direct_error,
        "deck_action_maximum_errors": action_errors,
        "maximum_deck_action_error": maximum_action_error,
        "carrier_pass": carrier_pass,
        "action_pass": action_pass,
        "task": task_metrics,
        "task_pass": task_pass,
        "joint_pass": carrier_pass and action_pass and task_pass,
    }


def evaluate_analytic_control(
    datasets: Mapping[str, stage0.C3TrainingDataset],
) -> dict[str, Any]:
    reference = analytic_carrier(datasets["reference"])
    gauge = fit_orthogonal_gauge(reference, reference)
    matrix = gauge.pop("matrix")
    regimes = {}
    for regime in REGIMES:
        analytic = analytic_carrier(datasets[regime])
        left = torch.view_as_real(analytic).reshape(-1, 2).double()
        right = left.clone()
        aligned = left @ matrix
        temporal = generator.temporal_prediction(analytic.to(torch.complex128))
        posterior = joint.interval_posterior_unclipped(temporal, 16)
        task_metrics = capacity._task_metrics(posterior, datasets[regime].target)
        regimes[regime] = {
            "mean_aligned_unit_dot": float((aligned * right).sum(-1).mean()),
            "aligned_coordinate_rmse": float(
                torch.sqrt((aligned - right).square().mean())
            ),
            "carrier_pass": True,
            "action_pass": True,
            "task": task_metrics,
            "task_pass": _task_gate(task_metrics, regime),
        }
        regimes[regime]["joint_pass"] = all(
            (
                regimes[regime]["carrier_pass"],
                regimes[regime]["action_pass"],
                regimes[regime]["task_pass"],
            )
        )
    return {
        "gauge": gauge,
        "regimes": regimes,
        "pass": all(item["joint_pass"] for item in regimes.values()),
    }


def _arm_targets(
    arm: str,
    dataset: stage0.C3TrainingDataset,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if arm == "learned_true":
        return dataset.target_posteriors
    if arm == "learned_target_shuffled":
        return dataset.target_posteriors[permutation]
    raise ValueError(f"unknown sensor-only arm {arm!r}")


def train_arm(
    *,
    arm: str,
    seed: int,
    config: SensorOnlyConfig,
    dataset: stage0.C3TrainingDataset,
    pair_batches: torch.Tensor,
    permutation: torch.Tensor,
    protocol_hashes: Mapping[str, str],
    datasets: Mapping[str, stage0.C3TrainingDataset],
    device: torch.device,
    cell_dir: Path,
    fingerprint: str,
) -> dict[str, Any]:
    encoder = _initial_encoder(seed, device)
    initial_state = stage0._state_digest(encoder)
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    targets = _arm_targets(arm, dataset, permutation)
    first_history = train_range(
        encoder,
        optimizer,
        dataset,
        pair_batches,
        targets,
        config,
        device,
        0,
        config.split_step,
    )
    midpoint_path = cell_dir / f"{arm}_midpoint.pt"
    midpoint = save_checkpoint(
        midpoint_path,
        _checkpoint_payload(
            encoder,
            optimizer,
            arm=arm,
            seed=seed,
            step=config.split_step,
            fingerprint=fingerprint,
            protocol_hashes=protocol_hashes,
        ),
    )
    second_history = train_range(
        encoder,
        optimizer,
        dataset,
        pair_batches,
        targets,
        config,
        device,
        config.split_step,
        config.training_steps,
    )
    final_state = stage0._state_digest(encoder)
    final_optimizer = stage0._optimizer_digest(optimizer)
    final_path = cell_dir / f"{arm}_final.pt"
    final = save_checkpoint(
        final_path,
        _checkpoint_payload(
            encoder,
            optimizer,
            arm=arm,
            seed=seed,
            step=config.training_steps,
            fingerprint=fingerprint,
            protocol_hashes=protocol_hashes,
        ),
    )

    reference_carrier = extract_carrier(encoder, datasets["reference"], device)
    gauge = fit_orthogonal_gauge(
        reference_carrier, analytic_carrier(datasets["reference"])
    )
    gauge_matrix = gauge.pop("matrix")
    regimes = {
        regime: evaluate_carrier(
            encoder, datasets[regime], gauge_matrix, device, regime
        )
        for regime in REGIMES
    }
    final_posteriors = {
        regime: joint.interval_posterior_unclipped(
            generator.temporal_prediction(
                extract_carrier(encoder, datasets[regime], device).to(
                    torch.complex128
                )
            ),
            16,
        ).cpu()
        for regime in REGIMES
    }

    reloaded = _initial_encoder(seed, device)
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
    reload_state = stage0._state_digest(reloaded)
    reload_optimizer = stage0._optimizer_digest(reloaded_optimizer)
    reload_errors = {}
    for regime in REGIMES:
        carrier = extract_carrier(reloaded, datasets[regime], device)
        temporal = generator.temporal_prediction(carrier.to(torch.complex128))
        posterior = joint.interval_posterior_unclipped(temporal, 16).cpu()
        reload_errors[regime] = float(
            (posterior - final_posteriors[regime]).abs().max()
        )

    resumed = _initial_encoder(seed, device)
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
        dataset,
        pair_batches,
        targets,
        config,
        device,
        config.split_step,
        config.training_steps,
    )
    resume_state = stage0._state_digest(resumed)
    resume_optimizer = stage0._optimizer_digest(resumed_optimizer)
    resume_errors = {}
    for regime in REGIMES:
        carrier = extract_carrier(resumed, datasets[regime], device)
        temporal = generator.temporal_prediction(carrier.to(torch.complex128))
        posterior = joint.interval_posterior_unclipped(temporal, 16).cpu()
        resume_errors[regime] = float(
            (posterior - final_posteriors[regime]).abs().max()
        )

    finite = _finite(first_history) and _finite(second_history) and _finite(regimes)
    state_changed = final_state != initial_state
    reload_pass = (
        reload_state == final_state
        and reload_optimizer == final_optimizer
        and all(value == 0.0 for value in reload_errors.values())
    )
    resume_pass = (
        resume_state == final_state
        and resume_optimizer == final_optimizer
        and resumed_history == second_history
        and all(value == 0.0 for value in resume_errors.values())
    )
    joint_pass = bool(
        finite
        and state_changed
        and reload_pass
        and resume_pass
        and all(item["joint_pass"] for item in regimes.values())
    )
    del reloaded, resumed, optimizer, reloaded_optimizer, resumed_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "arm": arm,
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "final_optimizer_sha256": final_optimizer,
        "state_changed": state_changed,
        "training_history": first_history + second_history,
        "midpoint_checkpoint": midpoint,
        "final_checkpoint": final,
        "gauge": gauge,
        "regimes": regimes,
        "reload": {
            "state_sha256": reload_state,
            "optimizer_sha256": reload_optimizer,
            "maximum_posterior_errors": reload_errors,
            "pass": reload_pass,
        },
        "exact_resume": {
            "state_sha256": resume_state,
            "optimizer_sha256": resume_optimizer,
            "history_exact": resumed_history == second_history,
            "maximum_posterior_errors": resume_errors,
            "verification_optimizer_steps": config.training_steps
            - config.split_step,
            "pass": resume_pass,
        },
        "finite": finite,
        "joint_pass": joint_pass,
    }


def _cell_dir(root: Path, seed: int) -> Path:
    return root / "runs" / f"seed_{seed}"


def _scientific_fingerprint(
    *,
    config: SensorOnlyConfig,
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


def sensor_only_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    seed = int(experiment.seed)
    sources, _license = _validate_sources()
    implementation = _implementation_digest(sources)
    if implementation != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("sensor-only implementation changed after scheduling")
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    task = stage0.C3TaskConfig()
    dataset, pair_batches, permutation, protocol_hashes = protocol_material(
        task, config, seed
    )
    contract = protocol_contract(dataset, permutation)
    if not contract["pass"]:
        raise RuntimeError("sensor-only paired protocol contract failed")
    datasets = build_held_out_datasets(task, config)
    evaluation_hashes = held_out_hashes(datasets)
    if evaluation_hashes != experiment.parameters["evaluation_hashes"]:
        raise RuntimeError("sensor-only held-out cohorts changed")
    fingerprint = _scientific_fingerprint(
        config=config,
        seed=seed,
        implementation=implementation,
        protocol_hashes=protocol_hashes,
        evaluation_hashes=evaluation_hashes,
    )
    if fingerprint != experiment.parameters["scientific_fingerprint"]:
        raise RuntimeError("sensor-only scientific fingerprint changed")
    cell_dir = _cell_dir(Path(experiment.parameters["output_dir"]), seed)
    cell_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    initial_reference = _initial_encoder(seed, device)
    initial_digest = stage0._state_digest(initial_reference)
    del initial_reference
    arms = {
        arm: train_arm(
            arm=arm,
            seed=seed,
            config=config,
            dataset=dataset,
            pair_batches=pair_batches,
            permutation=permutation,
            protocol_hashes=protocol_hashes,
            datasets=datasets,
            device=device,
            cell_dir=cell_dir,
            fingerprint=fingerprint,
        )
        for arm in ARMS
    }
    matched_initialization = all(
        item["initial_state_sha256"] == initial_digest for item in arms.values()
    )
    analytic = evaluate_analytic_control(datasets)
    validity = bool(
        contract["pass"]
        and matched_initialization
        and (analytic["pass"] or config.allow_underpowered)
        and all(item["finite"] for item in arms.values())
        and all(item["reload"]["pass"] for item in arms.values())
        and all(item["exact_resume"]["pass"] for item in arms.values())
    )
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result_path = cell_dir / "result.json"
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
        "configuration": _json_config(config),
        "source_hashes": sources,
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "protocol_hashes": protocol_hashes,
        "evaluation_hashes": evaluation_hashes,
        "protocol_contract": contract,
        "matched_initialization_sha256": initial_digest,
        "matched_initialization": matched_initialization,
        "analytic_positive_control": analytic,
        "arms": arms,
        "gates": {
            "validity": validity,
            "analytic_positive_control": analytic["pass"],
            "matched_initialization": matched_initialization,
            "learned_true_joint": arms["learned_true"]["joint_pass"],
            "learned_target_shuffled_joint": arms[
                "learned_target_shuffled"
            ]["joint_pass"],
        },
        "accounting": {
            "tinyllm_models_instantiated": 0,
            "trainable_parameters_per_arm": PARAMETER_COUNT,
            "primary_optimizer_steps": 2 * config.training_steps,
            "resume_verification_optimizer_steps": 2
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
        raise RuntimeError("non-finite sensor-only result")
    _write_json(result_path, detail)
    checkpoint = arms["learned_true"]["final_checkpoint"]["path"]
    del arms, datasets, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics={
            "validity": float(validity),
            "learned_true_joint": float(detail["gates"]["learned_true_joint"]),
            "learned_target_shuffled_joint": float(
                detail["gates"]["learned_target_shuffled_joint"]
            ),
            "peak_cuda_allocated_gb": peak,
        },
        primary_metric=float(detail["gates"]["learned_true_joint"]),
        model_architecture=[3, 16, 8, 2],
        model_parameters=PARAMETER_COUNT,
        training_time=elapsed,
        model_checkpoint=checkpoint,
        observations=[
            "Sensor-only exact-C3 task training with frozen temporal operator and decoder.",
            f"Seed={seed}; matched true and target-shuffled arms.",
        ],
        timestamp=datetime.fromisoformat(detail["completed_at"]),
    )


def _experiment(
    config: SensorOnlyConfig,
    seed: int,
    output_dir: Path,
    implementation: str,
    evaluation_hashes: Mapping[str, str],
) -> Experiment:
    task = stage0.C3TaskConfig()
    _dataset, _batches, _permutation, protocol_hashes = protocol_material(
        task, config, seed
    )
    fingerprint = _scientific_fingerprint(
        config=config,
        seed=seed,
        implementation=implementation,
        protocol_hashes=protocol_hashes,
        evaluation_hashes=evaluation_hashes,
    )
    return Experiment(
        id=f"tinyllm-c3-temporal-sensor-only-seed{seed}",
        hypothesis_id=HYPOTHESIS_ID,
        name=f"TinyLLM C3 temporal sensor-only seed {seed}",
        parameters={
            "configuration": _json_config(config),
            "output_dir": str(output_dir),
            "implementation_sha256": implementation,
            "scientific_fingerprint": fingerprint,
            "evaluation_hashes": dict(evaluation_hashes),
        },
        seed=seed,
    )


def _existing_detail(
    experiment: Experiment,
    output_dir: Path,
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
        for name in ("midpoint", "final"):
            artifact = detail.get("artifacts", {}).get("checkpoints", {}).get(arm, {}).get(name, {})
            artifact_path = Path(str(artifact.get("path", "")))
            if not artifact_path.is_file() or _sha256(artifact_path) != artifact.get("sha256"):
                return None
    return detail


def aggregate_details(
    details: Sequence[Mapping[str, Any]],
    config: SensorOnlyConfig,
) -> dict[str, Any]:
    valid = len(details) == len(config.seeds) and all(
        item.get("gates", {}).get("validity") is True for item in details
    )
    analytic = valid and all(
        item.get("gates", {}).get("analytic_positive_control") is True
        for item in details
    )
    true_count = sum(
        bool(item.get("gates", {}).get("learned_true_joint")) for item in details
    )
    shuffled_count = sum(
        bool(item.get("gates", {}).get("learned_target_shuffled_joint"))
        for item in details
    )
    specificity = shuffled_count <= config.shuffled_seed_passes_maximum
    if config.allow_underpowered:
        classification = "lifecycle_completed_not_quality_evidence"
        primary = False
    elif not valid or not analytic:
        classification = "invalid_campaign"
        primary = False
    elif not specificity:
        classification = "sensor_acquisition_specificity_failed"
        primary = False
    elif true_count >= config.required_seed_passes:
        classification = "task_only_sensor_acquisition_supported"
        primary = True
    else:
        classification = (
            "function_class_present_but_task_only_sensor_optimization_unreliable"
        )
        primary = False
    return {
        "valid": valid,
        "analytic_positive_control_pass": analytic,
        "learned_true_joint_pass_count": true_count,
        "learned_target_shuffled_joint_pass_count": shuffled_count,
        "required_seed_passes": config.required_seed_passes,
        "shuffled_seed_passes_maximum": config.shuffled_seed_passes_maximum,
        "specificity_pass": specificity,
        "classification": classification,
        "primary_hypothesis_pass": primary,
        "seed_gates": {
            str(item["seed"]): item["gates"] for item in details
        },
    }


async def run_campaign(
    config: SensorOnlyConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, license_record = _validate_sources()
    implementation = _implementation_digest(sources)
    task = stage0.C3TaskConfig()
    datasets = build_held_out_datasets(task, config)
    evaluation_hashes = held_out_hashes(datasets)
    del datasets
    experiments = [
        _experiment(
            config, seed, output_dir, implementation, evaluation_hashes
        )
        for seed in config.seeds
    ]
    campaign_fingerprint = _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "evaluation_hashes": evaluation_hashes,
            "experiment_fingerprints": [
                item.parameters["scientific_fingerprint"] for item in experiments
            ],
        }
    )
    campaign_path = output_dir / "campaign_results.json"
    if campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing_campaign.get("status") == "completed"
            and existing_campaign.get("campaign_fingerprint")
            == campaign_fingerprint
            and all(_existing_detail(item, output_dir) is not None for item in experiments)
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing_campaign

    existing = {
        item.id: detail
        for item in experiments
        if (detail := _existing_detail(item, output_dir)) is not None
    }
    pending = [item for item in experiments if item.id not in existing]
    lab = LabConfig(
        project_name="tinyllm_c3_temporal_sensor_only",
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
    runner = AsyncExperimentRunner(lab, sensor_only_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful = {item.experiment_id for item in results if item.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id in successful:
            detail = _existing_detail(experiment, output_dir)
            if detail is not None:
                details.append(detail)
    details.sort(key=lambda item: int(item["seed"]))
    complete = len(details) == len(experiments)
    aggregates = aggregate_details(details, config)
    result_paths = [
        _cell_dir(output_dir, seed) / "result.json" for seed in config.seeds
    ]
    checkpoint_paths = [
        _cell_dir(output_dir, seed) / f"{arm}_{point}.pt"
        for seed in config.seeds
        for arm in ARMS
        for point in ("midpoint", "final")
    ]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if complete and aggregates["valid"] else "partial",
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
        "capacity_license": {
            "path": str(CAPACITY_RESULT_PATH),
            "sha256": sources["capacity_result"],
            "classification": license_record["classification"],
        },
        "evaluation_hashes": evaluation_hashes,
        "campaign_fingerprint": campaign_fingerprint,
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value for key, value in runner.slot_plan.slots_by_device.items()
            },
            "free_memory_gb": {
                str(key): value for key, value in runner.slot_plan.free_memory_gb.items()
            },
            "calibration": runner.slot_plan.calibration,
        },
        "summary": {
            "requested": len(experiments),
            "reused": len(existing),
            "scheduled": len(pending),
            "completed": len(details),
            "failed": len(experiments) - len(details),
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
        "checkpoint_manifest_sha256": _manifest(checkpoint_paths) if complete else None,
        "maximum_peak_cuda_allocated_gb": max(
            (float(item["peak_cuda_allocated_gb"]) for item in details),
            default=0.0,
        ),
        "aggregate_cell_wall_seconds": sum(
            float(item["wall_seconds"]) for item in details
        ),
    }
    if not _finite(bundle):
        raise RuntimeError("non-finite sensor-only campaign")
    _write_json(campaign_path, bundle)
    return bundle


def _mode_config(mode: str) -> SensorOnlyConfig:
    if mode == "primary":
        return SensorOnlyConfig()
    device = (-1,) if mode == "cpu-shakedown" else (0,)
    return SensorOnlyConfig(
        seeds=(7,),
        training_steps=2,
        train_samples=64,
        batch_size=8,
        reference_samples=32,
        evaluation_samples=32,
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
            Path(
                "data/experiments/tinyllm_c3_temporal_sensor_only/"
                "20260811_preregistered"
            )
            if args.mode == "primary"
            else Path(f"/tmp/tinyllm-c3-temporal-sensor-only-{args.mode}")
        )
    bundle = asyncio.run(run_campaign(config, output))
    print(
        json.dumps(
            {
                "output": str(output),
                "status": bundle["status"],
                "classification": bundle["aggregates"]["classification"],
                "completed": bundle["summary"]["completed"],
                "true_passes": bundle["aggregates"][
                    "learned_true_joint_pass_count"
                ],
                "shuffled_passes": bundle["aggregates"][
                    "learned_target_shuffled_joint_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if bundle["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
