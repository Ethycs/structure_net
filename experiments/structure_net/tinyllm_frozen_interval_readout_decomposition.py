#!/usr/bin/env python3
"""Fit closed-form interval readouts on frozen calibrated TinyLLM states."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_frozen_scalar_interface_decomposition as source
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-frozen-interval-readout-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-frozen-interval-readout-decomposition-v1"
EVIDENCE_ROLE = "prospective_frozen_backbone_closed_form_interface_fit"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-interval-readout-decomposition-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "e5e1ee52bdccb420403643cf071adf3a49c80854c70adf5c632bba56e683d62f"
)
PRIMARY_SOURCE_ROOT = source.SOURCE_ROOT
DEVELOPMENT_SOURCE_ROOT = Path(
    "data/experiments/tinyllm_calibrated_frontend_causal/"
    "20260806_d8_preregistered"
)
DEVELOPMENT_SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
PRIMARY_PRESETS = ("d6", "d10")
PRIMARY_CONDITIONS = (
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
ARMS = (
    "input_affine_gauge",
    "untyped_final_readout",
    "typed_interval_readout",
    "frontend_typed_bypass",
)
FIT_ARMS = (
    "input_affine_gauge",
    "untyped_final_readout",
    "typed_interval_readout",
)


@dataclass(frozen=True)
class IntervalReadoutConfig:
    source_root: str = str(PRIMARY_SOURCE_ROOT)
    presets: tuple[str, ...] = PRIMARY_PRESETS
    conditions: tuple[str, ...] = PRIMARY_CONDITIONS
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    regimes: tuple[str, ...] = REGIMES
    train_samples: int = 4_096
    samples_per_regime: int = 1_024
    batch_size: int = 512
    ridge_lambda: float = 1e-4
    standardization_floor: float = 1e-6
    scalar_correlation_minimum: float = 0.90
    required_seed_passes: int = 4
    shuffled_seed_pass_ceiling: int = 1
    replay_tolerance: float = 2e-6
    shuffle_seed: int = 20_260_811
    device: str = "cuda:0"
    allow_underpowered: bool = False
    development_d8: bool = False

    def __post_init__(self) -> None:
        if not self.presets or not self.conditions or not self.seeds:
            raise ValueError("campaign axes cannot be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        if not self.regimes or set(self.regimes).difference(REGIMES):
            raise ValueError("unknown or empty regime selection")
        if self.train_samples < 4 or self.samples_per_regime < 4:
            raise ValueError("sample counts must be at least four")
        if self.batch_size < 1 or self.ridge_lambda <= 0.0:
            raise ValueError("batch size and ridge coefficient must be positive")
        if self.standardization_floor <= 0.0:
            raise ValueError("standardization floor must be positive")
        if not 0.0 < self.scalar_correlation_minimum <= 1.0:
            raise ValueError("scalar correlation threshold must lie in (0, 1]")
        if self.required_seed_passes < 1 or self.shuffled_seed_pass_ceiling < 0:
            raise ValueError("seed thresholds must be non-negative and nonzero")
        if not self.allow_underpowered:
            expected = {
                "source_root": str(PRIMARY_SOURCE_ROOT),
                "presets": PRIMARY_PRESETS,
                "conditions": PRIMARY_CONDITIONS,
                "seeds": PRIMARY_SEEDS,
                "regimes": REGIMES,
                "train_samples": 4_096,
                "samples_per_regime": 1_024,
                "batch_size": 512,
                "ridge_lambda": 1e-4,
                "standardization_floor": 1e-6,
                "scalar_correlation_minimum": 0.90,
                "required_seed_passes": 4,
                "shuffled_seed_pass_ceiling": 1,
                "replay_tolerance": 2e-6,
                "shuffle_seed": 20_260_811,
                "development_d8": False,
            }
            actual = {
                "source_root": self.source_root,
                "presets": self.presets,
                "conditions": self.conditions,
                "seeds": self.seeds,
                "regimes": self.regimes,
                "train_samples": self.train_samples,
                "samples_per_regime": self.samples_per_regime,
                "batch_size": self.batch_size,
                "ridge_lambda": self.ridge_lambda,
                "standardization_floor": self.standardization_floor,
                "scalar_correlation_minimum": self.scalar_correlation_minimum,
                "required_seed_passes": self.required_seed_passes,
                "shuffled_seed_pass_ceiling": self.shuffled_seed_pass_ceiling,
                "replay_tolerance": self.replay_tolerance,
                "shuffle_seed": self.shuffle_seed,
                "development_d8": self.development_d8,
            }
            if actual != expected:
                raise ValueError("primary interval-readout configuration changed")


@dataclass(frozen=True)
class RidgeMap:
    mean: torch.Tensor
    scale: torch.Tensor
    coefficients: torch.Tensor
    ridge_lambda: float
    rank: int
    condition_number: float
    training_rmse: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=512)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns)


def _manifest_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _json_config(config: IntervalReadoutConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "source_interface_runner": _sha256(Path(source.__file__)),
        "calibrated_frontend": _sha256(Path(calibrated.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("interval-readout preregistration changed")
    return values


def _implementation_digest(values: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(values or _implementation_sources()))


def fixed_interval_posterior(
    scalar: torch.Tensor, phase_bins: int = 16
) -> torch.Tensor:
    """Decode a physical cosine scalar with the task's fixed interval chart."""
    scalar = scalar.reshape(-1, 1).to(torch.float64).clamp(-1.0, 1.0)
    centers = torch.linspace(-1.0, 1.0, phase_bins, dtype=torch.float64)
    width = 2.0 / float(phase_bins - 1)
    logits = -0.5 * ((centers[None] - scalar) / width).square()
    return torch.softmax(logits, dim=-1)


def fit_ridge_map(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge_lambda: float,
    standardization_floor: float,
) -> RidgeMap:
    """Fit a deterministic standardized affine ridge map in float64 on CPU."""
    x = features.detach().cpu().to(torch.float64)
    y = targets.detach().cpu().to(torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("ridge inputs must be aligned matrices")
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(standardization_floor)
    normalized = (x - mean) / scale
    design = torch.cat(
        (normalized, torch.ones(len(normalized), 1, dtype=torch.float64)), dim=1
    )
    gram = design.T @ design / float(len(design))
    penalty = torch.eye(gram.shape[0], dtype=torch.float64) * ridge_lambda
    penalty[-1, -1] = 0.0
    system = gram + penalty
    rhs = design.T @ y / float(len(design))
    coefficients = torch.linalg.solve(system, rhs)
    prediction = design @ coefficients
    error = torch.sqrt(torch.mean((prediction - y).square()))
    return RidgeMap(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        ridge_lambda=float(ridge_lambda),
        rank=int(torch.linalg.matrix_rank(design)),
        condition_number=float(torch.linalg.cond(system)),
        training_rmse=float(error),
    )


def apply_ridge_map(fitted: RidgeMap, features: torch.Tensor) -> torch.Tensor:
    x = features.detach().cpu().to(torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    normalized = (x - fitted.mean) / fitted.scale
    design = torch.cat(
        (normalized, torch.ones(len(normalized), 1, dtype=torch.float64)), dim=1
    )
    return design @ fitted.coefficients


def ridge_record(fitted: RidgeMap) -> dict[str, Any]:
    return {
        "input_width": int(len(fitted.mean)),
        "output_width": int(fitted.coefficients.shape[1]),
        "coefficient_count": int(fitted.coefficients.numel()),
        "ridge_lambda": fitted.ridge_lambda,
        "rank": fitted.rank,
        "condition_number": fitted.condition_number,
        "training_rmse": fitted.training_rmse,
    }


def task_metrics(posterior: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    posterior = posterior.detach().cpu().to(torch.float64)
    target = target.detach().cpu().to(torch.float64)
    predicted_bins = posterior.argmax(1)
    target_bins = target.argmax(1)
    return {
        "exact_bin_accuracy": float((predicted_bins == target_bins).double().mean()),
        "mean_absolute_bin_error": float(
            (predicted_bins - target_bins).abs().double().mean()
        ),
        "mean_target_cross_entropy": float(
            -(target * posterior.clamp_min(1e-15).log()).sum(1).mean()
        ),
        "predicted_bin_coverage": int(torch.unique(predicted_bins).numel()),
        "predicted_bins": sorted(int(value) for value in torch.unique(predicted_bins)),
    }


def scalar_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    fiber_id: torch.Tensor,
) -> dict[str, float]:
    predicted = predicted.detach().cpu().reshape(-1).to(torch.float64)
    target = target.detach().cpu().reshape(-1).to(torch.float64)
    centered_x = predicted - predicted.mean()
    centered_y = target - target.mean()
    denominator = torch.sqrt(centered_x.square().sum() * centered_y.square().sum())
    correlation = (
        float((centered_x * centered_y).sum() / denominator)
        if float(denominator) > 0.0
        else 0.0
    )
    order = torch.argsort(fiber_id.detach().cpu().reshape(-1))
    paired = predicted[order].reshape(-1, 2)
    return {
        "cosine_pearson": correlation,
        "cosine_rmse": float(torch.sqrt(torch.mean((predicted - target).square()))),
        "minimum": float(predicted.min()),
        "maximum": float(predicted.max()),
        "mean_opposite_sheet_absolute_difference": float(
            (paired[:, 0] - paired[:, 1]).abs().mean()
        ),
    }


def _centered_log_targets(target: torch.Tensor) -> torch.Tensor:
    values = target.detach().cpu().to(torch.float64).clamp_min(1e-15).log()
    return values - values.mean(1, keepdim=True)


def deterministic_target_permutation(
    count: int, preset: str, condition: str, seed: int, base_seed: int
) -> torch.Tensor:
    material = f"{base_seed}:{preset}:{condition}:{seed}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(derived % (2**63 - 1))
    permutation = torch.randperm(count, generator=generator)
    if bool(torch.equal(permutation, torch.arange(count))):
        permutation = torch.roll(permutation, 1)
    return permutation


def typed_seed_pass(
    regime_records: Mapping[str, Mapping[str, Any]],
    regimes: Sequence[str],
    correlation_minimum: float,
) -> bool:
    return all(
        bool(regime_records[regime]["task_gate"])
        and float(regime_records[regime]["scalar_metrics"]["cosine_pearson"])
        >= correlation_minimum
        for regime in regimes
    )


def _finite(value: Any) -> bool:
    return source._finite(value)


def _primary_sources(
    config: IntervalReadoutConfig,
) -> tuple[
    dict[str, Any],
    CircleTaskConfig,
    dict[tuple[str, str, int], dict[str, Any]],
]:
    source_config = source.ScalarInterfaceConfig(
        source_root=config.source_root,
        device=config.device,
    )
    return source._source_details(source_config)


def _development_sources(
    config: IntervalReadoutConfig,
) -> tuple[
    dict[str, Any],
    CircleTaskConfig,
    dict[tuple[str, str, int], dict[str, Any]],
]:
    root = Path(config.source_root)
    campaign_path = root / "campaign_results.json"
    if _sha256(campaign_path) != DEVELOPMENT_SOURCE_CAMPAIGN_SHA256:
        raise ValueError("development d8 source campaign changed")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    task = CircleTaskConfig(**campaign["task_config"])
    details: dict[tuple[str, str, int], dict[str, Any]] = {}
    for condition in config.conditions:
        for seed in config.seeds:
            directory = root / "runs" / condition / f"seed_{seed}"
            path = directory / "result.json"
            detail = json.loads(path.read_text(encoding="utf-8"))
            model_path = directory / "model.pt"
            frontend_path = directory / "frontend.pt"
            if (
                detail.get("status") != "completed"
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or not model_path.is_file()
                or not frontend_path.is_file()
            ):
                raise ValueError(f"invalid development source {path}")
            detail["preset"] = "d8"
            detail["artifacts"] = {
                "checkpoint": str(model_path),
                "checkpoint_sha256": _sha256(model_path),
                "frontend_checkpoint": str(frontend_path),
                "frontend_checkpoint_sha256": _sha256(frontend_path),
            }
            detail["representation"] = detail["analysis"]
            detail["gates"] = {
                "task_accuracy_floors": {
                    regime: max(
                        0.0,
                        float(detail["analysis"]["task_metrics"][regime]["exact_bin_accuracy"])
                        - 0.03,
                    )
                    for regime in REGIMES
                },
                "task_adequacy_pass": True,
                "joint_seed_pass": True,
            }
            detail["_result_path"] = str(path)
            detail["_result_sha256"] = _sha256(path)
            details[("d8", condition, seed)] = detail
    return campaign, task, details


def _datasets(
    task: CircleTaskConfig, config: IntervalReadoutConfig
) -> dict[str, calibrated.CalibratedDataset]:
    specifications = source.DATASET_SPECS
    return {
        regime: calibrated.generate_calibrated_dataset(
            task,
            sample_count=config.samples_per_regime,
            seed=specifications[regime][0],
            regime=specifications[regime][1],
            shuffle=True,
        )
        for regime in config.regimes
    }


def _dataset_hashes(
    datasets: Mapping[str, calibrated.CalibratedDataset],
) -> dict[str, str]:
    return source._dataset_hashes(datasets)


@torch.inference_mode()
def _extract_state(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: IntervalReadoutConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    scalars: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    posteriors: list[torch.Tensor] = []
    for start in range(0, len(sensor), config.batch_size):
        stop = min(len(sensor), start + config.batch_size)
        inputs = dataset.paired.circle.input_ids[start:stop].to(device)
        observed = sensor[start:stop].to(device)
        packet = dataset.calibration[start:stop].to(device)
        scalar = system.feature(observed, packet)
        final = system.forward_cuts(inputs, observed, packet)["full"]
        normalized = system.model.transformer["ln_f"](final)
        logits = system.model.lm_head(normalized).index_select(-1, answer_ids)
        scalars.append(scalar.reshape(-1).float().cpu())
        residuals.append(normalized.float().cpu())
        posteriors.append(torch.softmax(logits, -1).float().cpu())
    return {
        "natural_scalar": torch.cat(scalars),
        "normalized_final_residual": torch.cat(residuals),
        "source_posterior": torch.cat(posteriors),
    }


@torch.inference_mode()
def _injected_posterior(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    scalar: torch.Tensor,
    task: CircleTaskConfig,
    config: IntervalReadoutConfig,
    device: torch.device,
) -> torch.Tensor:
    if system.scalar_embedding is None:
        raise AssertionError("structured scalar embedding missing")
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    result: list[torch.Tensor] = []
    scalar = scalar.reshape(-1, 1).to(torch.float32)
    for start in range(0, len(input_ids), config.batch_size):
        stop = min(len(input_ids), start + config.batch_size)
        ids = input_ids[start:stop].to(device)
        feature = scalar[start:stop].to(device)
        bos = system.model.transformer["wte"](ids[:, :1])
        scalar_token = system.scalar_embedding(feature)[:, None]
        query = system.model.transformer["wte"](ids[:, -1:])
        value = torch.cat((bos, scalar_token, query), dim=1)
        positions = torch.arange(value.shape[1], device=device)
        value = value + system.model.transformer["wpe"](positions)
        for block in system.model.transformer["h"]:
            value = block(value)
        normalized = system.model.transformer["ln_f"](value[:, -1])
        logits = system.model.lm_head(normalized).index_select(-1, answer_ids)
        result.append(torch.softmax(logits, -1).float().cpu())
    return torch.cat(result)


def _fit_maps(
    training_state: Mapping[str, torch.Tensor],
    training_dataset: calibrated.CalibratedDataset,
    permutation: torch.Tensor,
    config: IntervalReadoutConfig,
) -> dict[str, RidgeMap]:
    cosine = training_dataset.paired.fiber.cosine.reshape(-1).to(torch.float64)
    logits = _centered_log_targets(training_dataset.paired.circle.target_posteriors)
    scalar = training_state["natural_scalar"]
    residual = training_state["normalized_final_residual"]
    kwargs = {
        "ridge_lambda": config.ridge_lambda,
        "standardization_floor": config.standardization_floor,
    }
    return {
        "input_true": fit_ridge_map(scalar, cosine, **kwargs),
        "input_shuffled": fit_ridge_map(scalar, cosine[permutation], **kwargs),
        "typed_true": fit_ridge_map(residual, cosine, **kwargs),
        "typed_shuffled": fit_ridge_map(residual, cosine[permutation], **kwargs),
        "untyped_true": fit_ridge_map(residual, logits, **kwargs),
        "untyped_shuffled": fit_ridge_map(residual, logits[permutation], **kwargs),
    }


def _task_gate(metrics: Mapping[str, Any], floor: float) -> bool:
    return float(metrics["exact_bin_accuracy"]) >= floor


def _fit_arrays(maps: Mapping[str, RidgeMap]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, fitted in maps.items():
        arrays[f"fit__{name}__mean"] = fitted.mean.numpy()
        arrays[f"fit__{name}__scale"] = fitted.scale.numpy()
        arrays[f"fit__{name}__coefficients"] = fitted.coefficients.numpy()
    return arrays


def _cell_directory(root: Path, preset: str, condition: str, seed: int) -> Path:
    return root / "runs" / preset / condition / f"seed_{seed}"


def _fingerprint(
    config: IntervalReadoutConfig,
    preset: str,
    condition: str,
    seed: int,
    implementation: str,
    source_detail: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "preset": preset,
            "condition": condition,
            "seed": seed,
            "implementation_sha256": implementation,
            "source_result_sha256": source_detail["_result_sha256"],
            "dataset_hashes": dict(dataset_hashes),
        }
    )


def _existing_detail(
    output_root: Path,
    config: IntervalReadoutConfig,
    preset: str,
    condition: str,
    seed: int,
    implementation: str,
    source_detail: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> dict[str, Any] | None:
    path = _cell_directory(output_root, preset, condition, seed) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    diagnostics = Path(detail.get("artifacts", {}).get("diagnostics", ""))
    expected = _fingerprint(
        config,
        preset,
        condition,
        seed,
        implementation,
        source_detail,
        dataset_hashes,
    )
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint") != expected
        or detail.get("gates", {}).get("validity") is not True
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != detail.get("artifacts", {}).get("diagnostics_sha256")
    ):
        return None
    return detail


def run_cell(
    config: IntervalReadoutConfig,
    task: CircleTaskConfig,
    preset: str,
    condition: str,
    seed: int,
    source_detail: Mapping[str, Any],
    datasets: Mapping[str, calibrated.CalibratedDataset],
    dataset_hashes: Mapping[str, str],
    implementation_sources: Mapping[str, str],
    implementation: str,
    output_root: Path,
) -> dict[str, Any]:
    output_dir = _cell_directory(output_root, preset, condition, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    system = source._load_system(source_detail, task, preset, condition, device)
    state_before = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    source_config = calibrated.CalibratedFrontendConfig(
        preset=preset,
        seeds=PRIMARY_SEEDS,
        train_samples=config.train_samples,
        allow_underpowered=True,
    )
    training_dataset, _pair_batches, training_hash, _batch_hash = (
        calibrated._protocol_material(task, source_config, seed)
    )
    if training_hash != source_detail["training"]["training_data_sha256"]:
        raise RuntimeError(f"training cohort changed for {preset}/{condition}/{seed}")
    training_state = _extract_state(
        system, training_dataset, task, config, device
    )
    permutation = deterministic_target_permutation(
        len(training_dataset.paired.circle.input_ids),
        preset,
        condition,
        seed,
        config.shuffle_seed,
    )
    maps = _fit_maps(training_state, training_dataset, permutation, config)
    arrays = _fit_arrays(maps)
    records: dict[str, Any] = {}
    maximum_source_replay_error = 0.0
    maximum_decoder_error = 0.0
    for regime in config.regimes:
        dataset = datasets[regime]
        state = _extract_state(system, dataset, task, config, device)
        target = dataset.paired.circle.target_posteriors.to(torch.float64)
        cosine = dataset.paired.fiber.cosine.reshape(-1).to(torch.float64)
        fiber_id = dataset.paired.fiber.fiber_id
        floor = float(source_detail["gates"]["task_accuracy_floors"][regime])

        source_posterior = state["source_posterior"].to(torch.float64)
        natural_injected_posterior = _injected_posterior(
            system,
            dataset.paired.circle.input_ids,
            state["natural_scalar"],
            task,
            config,
            device,
        ).to(torch.float64)
        source_metrics = task_metrics(source_posterior, target)
        stored = source_detail["representation"]["task_metrics"][regime]
        direct_injection_error = float(
            (source_posterior - natural_injected_posterior).abs().max()
        )
        stored_metric_error = (
            max(
                abs(
                    source_metrics["exact_bin_accuracy"]
                    - float(stored["exact_bin_accuracy"])
                ),
                abs(
                    source_metrics["mean_target_cross_entropy"]
                    - float(stored["mean_target_cross_entropy"])
                ),
            )
            if config.samples_per_regime == 1_024
            else None
        )
        replay_error = max(
            direct_injection_error,
            stored_metric_error if stored_metric_error is not None else 0.0,
        )
        maximum_source_replay_error = max(maximum_source_replay_error, replay_error)

        exact_posterior = fixed_interval_posterior(cosine, task.phase_bins)
        decoder_error = float((exact_posterior - target).abs().max())
        maximum_decoder_error = max(maximum_decoder_error, decoder_error)

        input_true = apply_ridge_map(maps["input_true"], state["natural_scalar"]).reshape(-1)
        input_shuffled = apply_ridge_map(
            maps["input_shuffled"], state["natural_scalar"]
        ).reshape(-1)
        injected_true = _injected_posterior(
            system,
            dataset.paired.circle.input_ids,
            input_true.clamp(-1.0, 1.0),
            task,
            config,
            device,
        ).to(torch.float64)
        injected_shuffled = _injected_posterior(
            system,
            dataset.paired.circle.input_ids,
            input_shuffled.clamp(-1.0, 1.0),
            task,
            config,
            device,
        ).to(torch.float64)

        typed_true = apply_ridge_map(
            maps["typed_true"], state["normalized_final_residual"]
        ).reshape(-1)
        typed_shuffled = apply_ridge_map(
            maps["typed_shuffled"], state["normalized_final_residual"]
        ).reshape(-1)
        typed_posterior = fixed_interval_posterior(typed_true, task.phase_bins)
        typed_shuffled_posterior = fixed_interval_posterior(
            typed_shuffled, task.phase_bins
        )

        untyped_logits = apply_ridge_map(
            maps["untyped_true"], state["normalized_final_residual"]
        )
        untyped_shuffled_logits = apply_ridge_map(
            maps["untyped_shuffled"], state["normalized_final_residual"]
        )
        untyped_posterior = torch.softmax(untyped_logits, -1)
        untyped_shuffled_posterior = torch.softmax(untyped_shuffled_logits, -1)

        bypass_posterior = fixed_interval_posterior(input_true, task.phase_bins)
        bypass_shuffled_posterior = fixed_interval_posterior(
            input_shuffled, task.phase_bins
        )

        arms = {
            "input_affine_gauge": {
                "true": task_metrics(injected_true, target),
                "shuffled": task_metrics(injected_shuffled, target),
                "scalar_true": scalar_metrics(input_true, cosine, fiber_id),
                "scalar_shuffled": scalar_metrics(input_shuffled, cosine, fiber_id),
            },
            "untyped_final_readout": {
                "true": task_metrics(untyped_posterior, target),
                "shuffled": task_metrics(untyped_shuffled_posterior, target),
            },
            "typed_interval_readout": {
                "true": task_metrics(typed_posterior, target),
                "shuffled": task_metrics(typed_shuffled_posterior, target),
                "scalar_true": scalar_metrics(typed_true, cosine, fiber_id),
                "scalar_shuffled": scalar_metrics(typed_shuffled, cosine, fiber_id),
            },
            "frontend_typed_bypass": {
                "true": task_metrics(bypass_posterior, target),
                "shuffled": task_metrics(bypass_shuffled_posterior, target),
                "scalar_true": scalar_metrics(input_true, cosine, fiber_id),
                "scalar_shuffled": scalar_metrics(input_shuffled, cosine, fiber_id),
            },
        }
        for arm in ARMS:
            arms[arm]["task_gate_true"] = _task_gate(arms[arm]["true"], floor)
            arms[arm]["task_gate_shuffled"] = _task_gate(
                arms[arm]["shuffled"], floor
            )
        records[regime] = {
            "dataset_sha256": dataset_hashes[regime],
            "task_accuracy_floor": floor,
            "source": {
                "metrics": source_metrics,
                "direct_vs_natural_injection_maximum_absolute_error": (
                    direct_injection_error
                ),
                "stored_metric_replay_error": stored_metric_error,
                "stored_metrics_same_cohort": config.samples_per_regime == 1_024,
                "task_gate": _task_gate(source_metrics, floor),
            },
            "exact_decoder": {
                "maximum_absolute_posterior_error": decoder_error,
                "metrics": task_metrics(exact_posterior, target),
            },
            "arms": arms,
        }
        prefix = f"{regime}__"
        arrays.update(
            {
                f"{prefix}target_posterior": target.numpy(),
                f"{prefix}exact_cosine": cosine.numpy(),
                f"{prefix}natural_scalar": state["natural_scalar"].numpy(),
                f"{prefix}source_posterior": source_posterior.numpy(),
                f"{prefix}natural_injected_posterior": (
                    natural_injected_posterior.numpy()
                ),
                f"{prefix}input_true_scalar": input_true.numpy(),
                f"{prefix}input_shuffled_scalar": input_shuffled.numpy(),
                f"{prefix}input_injected_posterior": injected_true.numpy(),
                f"{prefix}input_shuffled_posterior": injected_shuffled.numpy(),
                f"{prefix}typed_true_scalar": typed_true.numpy(),
                f"{prefix}typed_shuffled_scalar": typed_shuffled.numpy(),
                f"{prefix}typed_posterior": typed_posterior.numpy(),
                f"{prefix}typed_shuffled_posterior": typed_shuffled_posterior.numpy(),
                f"{prefix}untyped_posterior": untyped_posterior.numpy(),
                f"{prefix}untyped_shuffled_posterior": untyped_shuffled_posterior.numpy(),
                f"{prefix}bypass_posterior": bypass_posterior.numpy(),
                f"{prefix}bypass_shuffled_posterior": bypass_shuffled_posterior.numpy(),
            }
        )

    cell_gates: dict[str, Any] = {}
    for arm in ARMS:
        task_true = all(
            records[regime]["arms"][arm]["task_gate_true"]
            for regime in config.regimes
        )
        task_shuffled = all(
            records[regime]["arms"][arm]["task_gate_shuffled"]
            for regime in config.regimes
        )
        if arm == "typed_interval_readout":
            true_pass = task_true and all(
                records[regime]["arms"][arm]["scalar_true"]["cosine_pearson"]
                >= config.scalar_correlation_minimum
                for regime in config.regimes
            )
            shuffled_pass = task_shuffled and all(
                records[regime]["arms"][arm]["scalar_shuffled"]["cosine_pearson"]
                >= config.scalar_correlation_minimum
                for regime in config.regimes
            )
        else:
            true_pass = task_true
            shuffled_pass = task_shuffled
        cell_gates[arm] = {
            "true_both_shifts": bool(true_pass),
            "shuffled_both_shifts": bool(shuffled_pass),
        }

    state_after = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    state_unchanged = state_before == state_after == {
        "model": source_detail["training"]["final_model_state_sha256"],
        "system": source_detail["training"]["final_system_state_sha256"],
    }
    finite = _finite({"records": records, "fits": {k: ridge_record(v) for k, v in maps.items()}})
    finite = finite and all(np.isfinite(value).all() for value in arrays.values())
    validity = bool(
        maximum_source_replay_error <= config.replay_tolerance
        and maximum_decoder_error <= config.replay_tolerance
        and state_unchanged
        and finite
    )
    diagnostics_path = output_dir / "diagnostics.npz"
    _write_npz(diagnostics_path, arrays)
    with np.load(diagnostics_path, allow_pickle=False) as reloaded:
        diagnostics_reload = bool(
            set(reloaded.files) == set(arrays)
            and all(np.array_equal(reloaded[name], arrays[name]) for name in arrays)
        )
    validity = validity and diagnostics_reload
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result_path = output_dir / "result.json"
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": (
            f"tinyllm-interval-readout-{preset}-{condition}-seed{seed}"
        ),
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "implementation_sha256": implementation,
        "implementation_sources": dict(implementation_sources),
        "scientific_fingerprint": _fingerprint(
            config,
            preset,
            condition,
            seed,
            implementation,
            source_detail,
            dataset_hashes,
        ),
        "source": {
            "campaign": str(Path(config.source_root) / "campaign_results.json"),
            "result": source_detail["_result_path"],
            "result_sha256": source_detail["_result_sha256"],
            "source_task_adequacy_pass": bool(
                source_detail["gates"]["task_adequacy_pass"]
            ),
            "model_checkpoint": source_detail["artifacts"]["checkpoint"],
            "model_checkpoint_sha256": source_detail["artifacts"][
                "checkpoint_sha256"
            ],
            "frontend_checkpoint": source_detail["artifacts"]["frontend_checkpoint"],
            "frontend_checkpoint_sha256": source_detail["artifacts"][
                "frontend_checkpoint_sha256"
            ],
            "training_data_sha256": training_hash,
        },
        "fit_protocol": {
            "kind": "closed_form_standardized_ridge",
            "target_permutation_sha256": calibrated._tensor_digest(permutation),
            "trained_model_parameters": 0,
            "fitted_maps": {name: ridge_record(value) for name, value in maps.items()},
            "fitted_parameter_count": int(
                sum(value.coefficients.numel() for value in maps.values())
            ),
        },
        "regimes": records,
        "state_record": {"before": state_before, "after": state_after},
        "gates": {
            "arms": cell_gates,
            "source_replay": maximum_source_replay_error <= config.replay_tolerance,
            "exact_decoder_fidelity": maximum_decoder_error <= config.replay_tolerance,
            "state_unchanged": state_unchanged,
            "diagnostics_reload": diagnostics_reload,
            "finite": finite,
            "validity": validity,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gb": peak,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, detail)
    del system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return detail


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: IntervalReadoutConfig
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for preset in config.presets:
        for condition in config.conditions:
            selected = [
                detail
                for detail in details
                if detail["preset"] == preset and detail["condition"] == condition
            ]
            arm_values = {}
            for arm in ARMS:
                true_count = sum(
                    bool(item["gates"]["arms"][arm]["true_both_shifts"])
                    for item in selected
                )
                shuffled_count = sum(
                    bool(item["gates"]["arms"][arm]["shuffled_both_shifts"])
                    for item in selected
                )
                arm_values[arm] = {
                    "true_pass_count": true_count,
                    "shuffled_pass_count": shuffled_count,
                    "pass_by_seed": {
                        str(item["seed"]): bool(
                            item["gates"]["arms"][arm]["true_both_shifts"]
                        )
                        for item in selected
                    },
                    "family_gate": bool(
                        true_count >= config.required_seed_passes
                        and shuffled_count <= config.shuffled_seed_pass_ceiling
                    ),
                }
            strata[f"{preset}/{condition}"] = {
                "valid_count": sum(bool(item["gates"]["validity"]) for item in selected),
                "arms": arm_values,
            }

    complete_primary = bool(
        not config.allow_underpowered
        and len(details)
        == len(config.presets) * len(config.conditions) * len(config.seeds)
    )
    validity = bool(
        all(detail["gates"]["validity"] for detail in details)
        and (complete_primary or config.allow_underpowered)
    )
    arm_family_gates = {
        arm: bool(
            validity
            and complete_primary
            and all(value["arms"][arm]["family_gate"] for value in strata.values())
        )
        for arm in ARMS
    }
    typed = arm_family_gates["typed_interval_readout"]
    untyped = arm_family_gates["untyped_final_readout"]
    input_gauge = arm_family_gates["input_affine_gauge"]
    bypass = arm_family_gates["frontend_typed_bypass"]
    any_stratum = any(
        value["arms"][arm]["family_gate"]
        for value in strata.values()
        for arm in FIT_ARMS
    )
    if not validity:
        classification = "invalid"
    elif not complete_primary:
        classification = "systems_lifecycle_only_not_scientific_evidence"
    elif typed:
        classification = "frozen_typed_interval_readout_architecture_stable"
    elif untyped:
        classification = "untyped_readout_only_architecture_stable"
    elif input_gauge:
        classification = "input_affine_gauge_architecture_stable"
    elif any_stratum:
        classification = "partial_frozen_interface_repair"
    else:
        classification = "no_frozen_endpoint_interface_repair"

    source_failed = [
        detail for detail in details if not detail["source"]["source_task_adequacy_pass"]
    ]
    repair_counts = {
        arm: sum(
            bool(detail["gates"]["arms"][arm]["true_both_shifts"])
            for detail in source_failed
        )
        for arm in ARMS
    }
    return {
        "valid": validity,
        "complete_primary_population": complete_primary,
        "classification": classification,
        "strata": strata,
        "arm_family_gates": arm_family_gates,
        "source_failed_count": len(source_failed),
        "source_failure_repair_counts": repair_counts,
        "gates": {
            "primary_hypothesis_pass": typed,
            "untyped_readout_family_pass": untyped,
            "input_affine_gauge_family_pass": input_gauge,
            "frontend_bypass_family_pass": bypass,
            "stop_before_joint_retraining": bool(typed or input_gauge),
            "joint_interface_retraining_licensed": bool(
                complete_primary and validity and not typed and not input_gauge
            ),
        },
    }


def _campaign_fingerprint(
    config: IntervalReadoutConfig,
    implementation: str,
    source_manifest: str,
    dataset_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "source_manifest_sha256": source_manifest,
            "dataset_hashes": dict(dataset_hashes),
        }
    )


def run_campaign(config: IntervalReadoutConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    if config.development_d8:
        source_campaign, task, source_details = _development_sources(config)
    else:
        source_campaign, task, source_details = _primary_sources(config)
    datasets = _datasets(task, config)
    dataset_hashes = _dataset_hashes(datasets)
    if (
        not config.allow_underpowered
        and dataset_hashes != source.EXPECTED_DATASET_HASHES
    ):
        raise RuntimeError("primary held-out cohort hashes changed")
    selected = [
        (preset, condition, seed)
        for preset in config.presets
        for condition in config.conditions
        for seed in config.seeds
    ]
    source_manifest = _json_hash(
        {
            f"{preset}/{condition}/seed_{seed}": source_details[
                (preset, condition, seed)
            ]["_result_sha256"]
            for preset, condition, seed in selected
        }
    )
    fingerprint = _campaign_fingerprint(
        config, implementation, source_manifest, dataset_hashes
    )
    campaign_path = output_dir / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("campaign_fingerprint") == fingerprint
            and all(
                _existing_detail(
                    output_dir,
                    config,
                    preset,
                    condition,
                    seed,
                    implementation,
                    source_details[(preset, condition, seed)],
                    dataset_hashes,
                )
                is not None
                for preset, condition, seed in selected
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing

    details: list[dict[str, Any]] = []
    for index, (preset, condition, seed) in enumerate(selected, start=1):
        detail = _existing_detail(
            output_dir,
            config,
            preset,
            condition,
            seed,
            implementation,
            source_details[(preset, condition, seed)],
            dataset_hashes,
        )
        action = "reused"
        if detail is None:
            detail = run_cell(
                config,
                task,
                preset,
                condition,
                seed,
                source_details[(preset, condition, seed)],
                datasets,
                dataset_hashes,
                sources,
                implementation,
                output_dir,
            )
            action = "completed"
        details.append(detail)
        print(
            f"[{index}/{len(selected)}] {action} {preset}/{condition}/seed_{seed} "
            f"typed={detail['gates']['arms']['typed_interval_readout']['true_both_shifts']} "
            f"untyped={detail['gates']['arms']['untyped_final_readout']['true_both_shifts']}",
            flush=True,
        )

    aggregates = aggregate_details(details, config)
    result_paths = [Path(detail["artifacts"]["result"]) for detail in details]
    diagnostic_paths = [Path(detail["artifacts"]["diagnostics"]) for detail in details]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if len(details) == len(selected) else "partial",
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "implementation_sha256": implementation,
        "implementation_sources": sources,
        "campaign_fingerprint": fingerprint,
        "source": {
            "campaign": str(Path(config.source_root) / "campaign_results.json"),
            "campaign_sha256": _sha256(
                Path(config.source_root) / "campaign_results.json"
            ),
            "source_campaign_fingerprint": source_campaign.get(
                "campaign_fingerprint"
            ),
            "selected_source_manifest_sha256": source_manifest,
        },
        "dataset_hashes": dataset_hashes,
        "summary": {
            "requested": len(selected),
            "completed": len(details),
            "failed": len(selected) - len(details),
            "source_checkpoints_loaded": len(details),
            "trained_models": 0,
            "trained_frontends": 0,
            "iterative_optimizers": 0,
            "closed_form_fit_count": 6 * len(details),
            "fitted_parameter_count": sum(
                int(detail["fit_protocol"]["fitted_parameter_count"])
                for detail in details
            ),
        },
        "aggregates": aggregates,
        "result_manifest_sha256": _manifest_sha256(result_paths),
        "diagnostics_manifest_sha256": _manifest_sha256(diagnostic_paths),
        "results": [
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "condition": detail["condition"],
                "seed": detail["seed"],
                "source_task_adequacy_pass": detail["source"][
                    "source_task_adequacy_pass"
                ],
                "arm_gates": detail["gates"]["arms"],
                "result": detail["artifacts"]["result"],
            }
            for detail in details
        ],
        "method_boundaries": [
            "Source checkpoint outcomes were known; fitted interface outcomes were preregistered before inspection.",
            "Closed-form maps use generator-derived cosine as a supervised training target.",
            "The typed readout tests affine accessibility and does not exclude nonlinear alternatives.",
            "No source checkpoint parameter was modified.",
            "The d6 and d10 presets jointly vary depth, width, and head count.",
        ],
    }
    _write_json(campaign_path, bundle)
    return bundle


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(PRIMARY_SOURCE_ROOT))
    parser.add_argument("--presets", default=",".join(PRIMARY_PRESETS))
    parser.add_argument("--conditions", default=",".join(PRIMARY_CONDITIONS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in PRIMARY_SEEDS))
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--samples", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_frozen_interval_readout_decomposition/"
            "20260811_d6_d10_preregistered"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    development_d8 = False
    if args.shakedown:
        args.source_root = str(DEVELOPMENT_SOURCE_ROOT)
        args.presets = "d8"
        args.conditions = "analytic_calibrated"
        args.seeds = "7"
        args.train_samples = 4_096
        args.samples = 64
        args.batch_size = 64
        args.allow_underpowered = True
        development_d8 = True
    config = IntervalReadoutConfig(
        source_root=args.source_root,
        presets=_comma_strings(args.presets),
        conditions=_comma_strings(args.conditions),
        seeds=_comma_ints(args.seeds),
        regimes=_comma_strings(args.regimes),
        train_samples=args.train_samples,
        samples_per_regime=args.samples,
        batch_size=args.batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
        development_d8=development_d8,
    )
    result = run_campaign(config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "valid": result["aggregates"]["valid"],
                "summary": result["summary"],
                "output": str(args.output / "campaign_results.json"),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
