#!/usr/bin/env python3
"""Measure inference-only calibration robustness of retained TinyLLM front ends."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_conditional_branch_depth_scan as probes
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-calibration-degradation-causal.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-reference-robustness-curve-v1"
SOURCE_SCHEMA = "nal.tinyllm-calibrated-frontend-causal.v1"
SOURCE_HYPOTHESIS = "tinyllm-calibrated-reference-stable-cosine-quotient-v1"
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
SOURCE_MANIFEST_SHA256 = (
    "23598aaef5a0d16825ff9a928de57857e7bd58b10feab8853739448acfd983fc"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
CUTS = ("frontend", "full")
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
ABLATIONS = (
    "orientation_default",
    "speed_default",
    "amplitude_default",
    "offset_default",
    "drift_default",
    "all_default",
)


@dataclass(frozen=True)
class CalibrationDegradationConfig:
    source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    noise_levels: tuple[float, ...] = NOISE_LEVELS
    orientation_scale_radians: float = 0.40
    signed_speed_scale: float = 0.50
    log_amplitude_scale: float = 0.50
    offset_scale: float = 0.30
    drift_scale: float = 0.16
    amplitude_floor: float = 0.10
    amplitude_ceiling: float = 4.0
    cosine_floor: float = 0.90
    branch_accuracy_ceiling: float = 0.55
    conditional_log_loss_gain_ceiling: float = 0.02
    task_accuracy_drop_ceiling: float = 0.03
    target_robust_radius: float = 0.20
    required_seed_passes: int = 4
    clean_replay_tolerance: float = 1e-5
    activation_audit_rows: int = 128
    activation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.conditions) != CONDITIONS and not self.allow_underpowered:
            raise ValueError("primary conditions are fixed")
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary seeds are fixed to 7,17,29,41,53")
        if tuple(self.noise_levels) != NOISE_LEVELS:
            raise ValueError("calibration noise grid is fixed")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown calibrated condition")
        if self.required_seed_passes > len(self.seeds):
            raise ValueError("required seed count exceeds requested seeds")
        if self.activation_audit_rows < 2:
            raise ValueError("activation audit requires at least two rows")
        if not 0.0 < self.amplitude_floor < self.amplitude_ceiling:
            raise ValueError("amplitude clamps are invalid")


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


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(calibrated.__file__),
        Path(probes.__file__),
        Path(invariant.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: CalibrationDegradationConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_frozen_checkpoint_underpowered_causal_diagnostic"
    )


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def noise_key(level: float) -> str:
    return f"sigma_{level:.2f}".replace(".", "p")


def evaluation_key(regime: str, intervention: str) -> str:
    return f"{regime}__{intervention}"


def perturb_calibration(
    calibration: torch.Tensor,
    noise: torch.Tensor,
    level: float,
    config: CalibrationDegradationConfig,
) -> torch.Tensor:
    """Apply one nested physically typed calibration perturbation."""
    if calibration.ndim != 2 or calibration.shape[1] != calibrated.CALIBRATION_WIDTH:
        raise ValueError("calibration must have eight fields")
    if noise.shape != (len(calibration), 7):
        raise ValueError("calibration noise must have seven fields")
    if level == 0.0:
        return calibration.clone()
    value = calibration.double().clone()
    noise = noise.double()
    angle = torch.atan2(value[:, 1], value[:, 0])
    angle = angle + level * config.orientation_scale_radians * noise[:, 0]
    value[:, 0], value[:, 1] = torch.cos(angle), torch.sin(angle)
    value[:, 2] += level * config.signed_speed_scale * noise[:, 1]
    log_amplitude = value[:, 3].clamp_min(config.amplitude_floor).log()
    value[:, 3] = torch.exp(
        log_amplitude + level * config.log_amplitude_scale * noise[:, 2]
    ).clamp(config.amplitude_floor, config.amplitude_ceiling)
    value[:, 4:6] += level * config.offset_scale * noise[:, 3:5]
    value[:, 6:8] += level * config.drift_scale * noise[:, 5:7]
    return value.to(dtype=calibration.dtype)


def ablate_calibration(calibration: torch.Tensor, name: str) -> torch.Tensor:
    if name not in ABLATIONS:
        raise ValueError(f"unknown calibration ablation {name}")
    value = calibration.clone()
    if name in {"orientation_default", "all_default"}:
        value[:, 0] = 1.0
        value[:, 1] = 0.0
    if name in {"speed_default", "all_default"}:
        value[:, 2] = 0.35
    if name in {"amplitude_default", "all_default"}:
        value[:, 3] = 1.0
    if name in {"offset_default", "all_default"}:
        value[:, 4:6] = 0.0
    if name in {"drift_default", "all_default"}:
        value[:, 6:8] = 0.0
    return value


def endpoint_pass(
    metrics: Mapping[str, Any], config: CalibrationDegradationConfig
) -> bool:
    return bool(
        metrics["cosine_pearson"] >= config.cosine_floor
        and metrics["balanced_accuracy"] <= config.branch_accuracy_ceiling
        and metrics["conditional_log_loss_gain_over_cosine_only"]
        <= config.conditional_log_loss_gain_ceiling
    )


def activation_drift(
    clean: np.ndarray,
    perturbed: np.ndarray,
    audit_rows: int,
) -> dict[str, float]:
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if clean.shape != perturbed.shape or clean.ndim != 2 or not len(clean):
        raise ValueError("activation arrays must be equal non-empty matrices")
    clean_norm = np.linalg.norm(clean, axis=1)
    perturbed_norm = np.linalg.norm(perturbed, axis=1)
    denominator = np.maximum(clean_norm * perturbed_norm, 1e-24)
    row_cosine = np.sum(clean * perturbed, axis=1) / denominator
    both_zero = (clean_norm <= 1e-12) & (perturbed_norm <= 1e-12)
    row_cosine[both_zero] = 1.0
    normalized_rmse = float(
        np.sqrt(np.mean(np.square(perturbed - clean)))
        / max(float(np.std(clean)), 1e-12)
    )
    count = min(audit_rows, len(clean))
    x = clean[:count] - clean[:count].mean(0, keepdims=True)
    y = perturbed[:count] - perturbed[:count].mean(0, keepdims=True)
    x_gram = x @ x.T
    y_gram = y @ y.T
    numerator = float(np.sum(x_gram * y_gram))
    cka_denominator = math.sqrt(
        max(float(np.sum(x_gram * x_gram)), 0.0)
        * max(float(np.sum(y_gram * y_gram)), 0.0)
    )
    cka = numerator / cka_denominator if cka_denominator > 1e-24 else 1.0
    return {
        "mean_row_cosine_to_clean": float(np.mean(row_cosine)),
        "normalized_rmse_from_clean": normalized_rmse,
        "centered_linear_cka_to_clean": float(cka),
        "cka_rows": count,
    }


def _source_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    records = []
    for condition in CONDITIONS:
        for seed in PRIMARY_SEEDS:
            directory = root / "runs" / condition / f"seed_{seed}"
            for name in ("result.json", "model.pt", "frontend.pt"):
                path = directory / name
                records.append(
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), records


def _load_source_campaign(
    config: CalibrationDegradationConfig,
) -> tuple[dict[str, Any], Path, str, list[dict[str, Any]]]:
    root = Path(config.source_root)
    path = root / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    manifest, records = _source_manifest(root)
    if (
        _sha256(path) != SOURCE_CAMPAIGN_SHA256
        or manifest != SOURCE_MANIFEST_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 15
        or int(campaign.get("summary", {}).get("failed", -1)) != 0
        or any(
            int(
                campaign.get("aggregates", {})
                .get("arms", {})
                .get(condition, {})
                .get("joint_pass_count", -1)
            )
            != 5
            for condition in CONDITIONS
        )
    ):
        raise ValueError(f"invalid calibrated source campaign {path}")
    return campaign, path, manifest, records


def _source_paths(root: Path, condition: str, seed: int) -> tuple[Path, Path, Path]:
    directory = root / "runs" / condition / f"seed_{seed}"
    return directory / "result.json", directory / "model.pt", directory / "frontend.pt"


def _load_system(
    campaign: Mapping[str, Any],
    config: CalibrationDegradationConfig,
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[calibrated.CalibratedTinyLLM, dict[str, Any]]:
    root = Path(config.source_root)
    result_path, model_path, frontend_path = _source_paths(root, condition, seed)
    detail = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        detail.get("schema_version") != SOURCE_SCHEMA
        or detail.get("hypothesis_id") != SOURCE_HYPOTHESIS
        or detail.get("status") != "completed"
        or detail.get("condition") != condition
        or int(detail.get("seed", -1)) != seed
        or detail.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or detail.get("training", {}).get("checkpoint") != str(model_path)
        or detail.get("training", {}).get("frontend_checkpoint") != str(frontend_path)
    ):
        raise ValueError(f"invalid calibrated source result {result_path}")
    task = CircleTaskConfig(**campaign["task_config"])
    source_config = calibrated._config_from_mapping(campaign["configuration"])
    model = TinyLLMModel.from_checkpoint(model_path, map_location="cpu")
    system = calibrated.CalibratedTinyLLM(model, condition, task, source_config)
    frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
    if frontend.get("condition") != condition:
        raise ValueError(f"frontend condition mismatch {frontend_path}")
    if system.calibration_embedding is not None:
        system.calibration_embedding.load_state_dict(frontend["calibration_embedding"])
    if system.scalar_embedding is not None:
        system.scalar_embedding.load_state_dict(frontend["scalar_embedding"])
    if system.encoder is not None:
        system.encoder.load_state_dict(frontend["encoder"])
    model_state = _state_digest(system.model)
    system_state = calibrated._module_digest(system)
    if (
        model_state != detail["training"]["final_model_state_sha256"]
        or system_state != detail["training"]["final_system_state_sha256"]
    ):
        raise ValueError(f"calibrated source state mismatch {result_path}")
    return system.to(device).eval(), {
        "detail": detail,
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "frontend": str(frontend_path),
        "frontend_sha256": _sha256(frontend_path),
        "model_state_sha256": model_state,
        "system_state_sha256": system_state,
    }


def _dataset_hash(dataset: calibrated.CalibratedDataset) -> str:
    paired = dataset.paired
    return invariant._tensor_digest(
        paired.circle.input_ids,
        paired.circle.target_posteriors,
        paired.fiber.cosine,
        paired.fiber.branch,
        dataset.calibration,
    )


def _datasets(
    task: CircleTaskConfig,
    source_config: calibrated.CalibratedFrontendConfig,
) -> dict[str, calibrated.CalibratedDataset]:
    seed = source_config.analysis_seed
    return {
        "train": calibrated.generate_calibrated_dataset(
            task, sample_count=source_config.probe_train_samples, seed=seed + 101
        ),
        "validation": calibrated.generate_calibrated_dataset(
            task, sample_count=source_config.probe_validation_samples, seed=seed + 211
        ),
        "composition": calibrated.generate_calibrated_dataset(
            task,
            sample_count=source_config.probe_test_samples,
            seed=seed + 1_316,
            regime="composition",
        ),
        "extrapolation": calibrated.generate_calibrated_dataset(
            task,
            sample_count=source_config.probe_test_samples,
            seed=seed + 2_325,
            regime="extrapolation",
        ),
    }


def _interventions(
    datasets: Mapping[str, calibrated.CalibratedDataset],
    source_config: calibrated.CalibratedFrontendConfig,
    config: CalibrationDegradationConfig,
) -> tuple[dict[str, calibrated.CalibratedDataset], str]:
    output: dict[str, calibrated.CalibratedDataset] = {}
    noise_payload: list[torch.Tensor] = []
    for regime_index, regime in enumerate(REGIMES):
        base = datasets[regime]
        generator = torch.Generator(device="cpu").manual_seed(
            source_config.analysis_seed + 40_001 + regime_index * 10_007
        )
        noise = torch.randn((len(base.calibration), 7), generator=generator)
        noise_payload.append(noise)
        for level in config.noise_levels:
            key = evaluation_key(regime, noise_key(level))
            output[key] = calibrated.CalibratedDataset(
                paired=base.paired,
                calibration=perturb_calibration(base.calibration, noise, level, config),
            )
        for name in ABLATIONS:
            output[evaluation_key(regime, name)] = calibrated.CalibratedDataset(
                paired=base.paired,
                calibration=ablate_calibration(base.calibration, name),
            )
    return output, invariant._tensor_digest(*noise_payload)


def _clean_replay(
    detail: Mapping[str, Any],
    probe_evaluations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task_evaluations: Mapping[str, Mapping[str, Any]],
    config: CalibrationDegradationConfig,
) -> tuple[bool, float]:
    maximum = 0.0
    for cut in CUTS:
        for regime in REGIMES:
            source_value = detail["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]
            replay = probe_evaluations[cut][evaluation_key(regime, noise_key(0.0))]
            for metric in (
                "cosine_pearson",
                "balanced_accuracy",
                "conditional_log_loss_gain_over_cosine_only",
            ):
                maximum = max(maximum, abs(float(source_value[metric]) - float(replay[metric])))
    for regime in REGIMES:
        source_value = detail["analysis"]["task_metrics"][regime]
        replay = task_evaluations[evaluation_key(regime, noise_key(0.0))]
        for metric in (
            "exact_bin_accuracy",
            "mean_circular_error_radians",
            "mean_target_cross_entropy",
        ):
            maximum = max(maximum, abs(float(source_value[metric]) - float(replay[metric])))
    return maximum <= config.clean_replay_tolerance, maximum


def _fingerprint(
    config: CalibrationDegradationConfig,
    condition: str,
    seed: int,
    source_result_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "condition": condition,
        "seed": seed,
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_result_sha256": source_result_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    result_path: Path,
    arrays_path: Path,
    config: CalibrationDegradationConfig,
    implementation: str,
    condition: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and value.get("condition") == condition
        and int(value.get("seed", -1)) == seed
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("artifacts", {}).get("result") == str(result_path)
        and value.get("artifacts", {}).get("activations") == str(arrays_path)
        and arrays_path.is_file()
        and _sha256(arrays_path) == value.get("artifacts", {}).get("activations_sha256")
    ):
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return value


def _campaign_reusable(
    value: Mapping[str, Any],
    config: CalibrationDegradationConfig,
    implementation: str,
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("summary", {}).get("completed", -1))
        == len(config.conditions) * len(config.seeds)
        and all(
            Path(item.get("path", "")).is_file()
            and Path(item.get("activations", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            and _sha256(Path(item["activations"])) == item.get("activations_sha256")
            for item in value.get("results", [])
        )
    )


def robust_radius(
    pass_counts: Mapping[str, int],
    levels: Sequence[float],
    required: int,
) -> float:
    radius = -1.0
    for level in levels:
        if int(pass_counts[noise_key(float(level))]) < required:
            break
        radius = float(level)
    return radius


def campaign_classification(
    *,
    valid: bool,
    analytic_radius: float,
    learned_radius: float,
    target_radius: float,
    analytic_default_fail_count: int,
    learned_default_fail_count: int,
    required_default_failures: int,
    levels: Sequence[float],
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    indices = {float(level): index for index, level in enumerate(levels)}
    analytic_index = indices.get(analytic_radius, -1)
    learned_index = indices.get(learned_radius, -1)
    primary = bool(
        analytic_radius >= target_radius
        and learned_radius >= target_radius
        and abs(learned_index - analytic_index) <= 1
        and analytic_default_fail_count >= required_default_failures
        and learned_default_fail_count >= required_default_failures
    )
    if primary:
        return "learned_calibration_robustness_tracks_analytic_control", True
    if learned_index > analytic_index + 1:
        return "learned_more_robust_than_analytic_control", False
    if analytic_radius >= target_radius and learned_radius < target_radius:
        return "learned_calibration_brittle", False
    if analytic_radius < target_radius and learned_radius < target_radius:
        return "exact_calibration_required", False
    if analytic_radius < target_radius <= learned_radius:
        return "analytic_canonicalizer_brittle", False
    return "mixed_calibration_robustness", False


def run_campaign(
    config: CalibrationDegradationConfig,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    campaign, source_path, manifest, manifest_records = _load_source_campaign(config)
    task = CircleTaskConfig(**campaign["task_config"])
    source_config = calibrated._config_from_mapping(campaign["configuration"])
    datasets = _datasets(task, source_config)
    interventions, noise_sha256 = _interventions(datasets, source_config, config)
    clean_data_hashes = {name: _dataset_hash(value) for name, value in datasets.items()}
    input_target_hashes = {
        name: invariant._tensor_digest(
            value.paired.circle.input_ids,
            value.paired.circle.target_posteriors,
            value.paired.fiber.cosine,
            value.paired.fiber.branch,
        )
        for name, value in interventions.items()
    }
    for regime in REGIMES:
        expected = input_target_hashes[evaluation_key(regime, noise_key(0.0))]
        if any(
            digest != expected
            for name, digest in input_target_hashes.items()
            if name.startswith(regime + "__")
        ):
            raise RuntimeError(f"input/target identity failed for {regime}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    results: list[dict[str, Any]] = []
    reused = 0
    probe_config = calibrated._analysis_config(source_config)
    for condition in config.conditions:
        for seed in config.seeds:
            cell_started = time.perf_counter()
            result_path = output / "runs" / condition / f"seed_{seed}" / "result.json"
            arrays_path = result_path.with_name("activation_audit.npz")
            source_result_path, _, _ = _source_paths(Path(config.source_root), condition, seed)
            source_result_sha256 = _sha256(source_result_path)
            fingerprint = _fingerprint(
                config, condition, seed, source_result_sha256
            )
            existing = _reusable_result(
                result_path,
                arrays_path,
                config,
                implementation,
                condition,
                seed,
                fingerprint,
            )
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"resuming {condition} seed {seed}", flush=True)
                continue

            system, provenance = _load_system(
                campaign, config, condition, seed, device
            )
            train_features = calibrated.extract_features(
                system, datasets["train"], task, source_config, device
            )
            validation_features = calibrated.extract_features(
                system, datasets["validation"], task, source_config, device
            )
            evaluation_features: dict[str, dict[str, np.ndarray]] = {}
            task_evaluations: dict[str, dict[str, float]] = {}
            activation_arrays: dict[str, np.ndarray] = {}
            for key, dataset in interventions.items():
                features = calibrated.extract_features(
                    system, dataset, task, source_config, device
                )
                evaluation_features[key] = features
                task_evaluations[key] = calibrated.task_metrics(
                    system, dataset, task, source_config, device
                )
                for cut in CUTS:
                    activation_arrays[f"{key}__{cut}"] = features[cut][
                        : config.activation_audit_rows
                    ]

            evaluation_datasets = {
                key: dataset.paired.fiber for key, dataset in interventions.items()
            }
            null = calibrated.fit_cosine_only_null(
                datasets["train"].paired.fiber,
                datasets["validation"].paired.fiber,
                evaluation_datasets,
                probe_config,
                device,
                kind="nonlinear",
                seed=source_config.analysis_seed + seed * 101,
            )
            probe_evaluations: dict[str, dict[str, Any]] = {}
            for cut_index, cut in enumerate(CUTS):
                probe = calibrated.fit_conditional_probe(
                    train_features[cut],
                    validation_features[cut],
                    {key: value[cut] for key, value in evaluation_features.items()},
                    datasets["train"].paired.fiber,
                    datasets["validation"].paired.fiber,
                    evaluation_datasets,
                    probe_config,
                    device,
                    kind="nonlinear",
                    seed=(
                        source_config.analysis_seed
                        + seed * 10_007
                        + cut_index * 1_009
                    ),
                )
                calibrated._add_log_loss_gains(probe, null)
                probe_evaluations[cut] = probe["evaluations"]

            clean_replay_pass, maximum_clean_replay_error = _clean_replay(
                provenance["detail"],
                probe_evaluations,
                task_evaluations,
                config,
            )
            cells = []
            seed_pass_by_intervention: dict[str, bool] = {}
            for intervention in tuple(noise_key(level) for level in config.noise_levels) + ABLATIONS:
                cell_pass = True
                for regime in REGIMES:
                    key = evaluation_key(regime, intervention)
                    clean_key = evaluation_key(regime, noise_key(0.0))
                    representation = {}
                    for cut in CUTS:
                        metrics = probe_evaluations[cut][key]
                        drift = activation_drift(
                            evaluation_features[clean_key][cut],
                            evaluation_features[key][cut],
                            config.activation_audit_rows,
                        )
                        representation[cut] = {
                            "probe": metrics,
                            "activation_drift": drift,
                            "endpoint_pass": endpoint_pass(metrics, config),
                        }
                        cell_pass = bool(
                            cell_pass and representation[cut]["endpoint_pass"]
                        )
                    clean_accuracy = task_evaluations[clean_key]["exact_bin_accuracy"]
                    task_value = task_evaluations[key]
                    accuracy_drop = clean_accuracy - task_value["exact_bin_accuracy"]
                    task_pass = accuracy_drop <= config.task_accuracy_drop_ceiling
                    cell_pass = bool(cell_pass and task_pass)
                    cells.append(
                        {
                            "intervention": intervention,
                            "regime": regime,
                            "representation": representation,
                            "task": {
                                **task_value,
                                "clean_exact_bin_accuracy": clean_accuracy,
                                "exact_bin_accuracy_drop": accuracy_drop,
                                "utility_pass": task_pass,
                            },
                        }
                    )
                seed_pass_by_intervention[intervention] = cell_pass

            pass_counts = {
                noise_key(level): int(seed_pass_by_intervention[noise_key(level)])
                for level in config.noise_levels
            }
            radius = robust_radius(pass_counts, config.noise_levels, 1)
            probe_finite = all(
                np.isfinite(
                    [
                        float(value)
                        for value in probe_evaluations[cut][key].values()
                        if isinstance(value, (int, float))
                    ]
                ).all()
                for cut in CUTS
                for key in probe_evaluations[cut]
            )
            task_finite = all(
                np.isfinite(list(task_evaluations[key].values())).all()
                for key in task_evaluations
            )
            valid = bool(clean_replay_pass and probe_finite and task_finite)
            classification = (
                "invalid"
                if not valid
                else (
                    "checkpoint_robust_through_target"
                    if radius >= config.target_robust_radius
                    else "checkpoint_fails_before_target"
                )
            )
            _write_npz(arrays_path, activation_arrays)
            arrays_sha256 = _sha256(arrays_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-calibration-degradation-{condition}-seed{seed}",
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "source": {key: value for key, value in provenance.items() if key != "detail"},
                "source_manifest_sha256": manifest,
                "data": {
                    "clean_dataset_hashes": clean_data_hashes,
                    "noise_sha256": noise_sha256,
                    "input_target_identity_pass": True,
                },
                "observer": {
                    "kind": "nonlinear_clean_fit_frozen_across_interventions",
                    "fitted_suites": len(CUTS),
                    "fitted_nulls": 1,
                    "source_probe_steps": source_config.probe_steps,
                },
                "clean_replay": {
                    "pass": clean_replay_pass,
                    "maximum_absolute_metric_error": maximum_clean_replay_error,
                    "tolerance": config.clean_replay_tolerance,
                },
                "cells": cells,
                "seed_pass_by_intervention": seed_pass_by_intervention,
                "robust_radius": radius,
                "classification": classification,
                "primary_metric": float(
                    valid and radius >= config.target_robust_radius
                ),
                "analysis_seconds": time.perf_counter() - cell_started,
                "artifacts": {
                    "result": str(result_path),
                    "activations": str(arrays_path),
                    "activations_sha256": arrays_sha256,
                },
            }
            _write_json(result_path, result)
            results.append(result)
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"{condition} seed {seed}: {classification}, radius={radius:.2f}",
                flush=True,
            )
            if _implementation_digest() != implementation:
                raise RuntimeError("calibration-degradation implementation changed")

    valid = all(result["classification"] != "invalid" for result in results)
    arms: dict[str, Any] = {}
    for condition in config.conditions:
        selected = [item for item in results if item["condition"] == condition]
        pass_counts = {
            noise_key(level): sum(
                bool(item["seed_pass_by_intervention"][noise_key(level)])
                for item in selected
            )
            for level in config.noise_levels
        }
        ablation_pass_counts = {
            name: sum(bool(item["seed_pass_by_intervention"][name]) for item in selected)
            for name in ABLATIONS
        }
        arms[condition] = {
            "noise_pass_counts": pass_counts,
            "ablation_pass_counts": ablation_pass_counts,
            "robust_radius": robust_radius(
                pass_counts,
                config.noise_levels,
                config.required_seed_passes,
            ),
            "valid_checkpoint_count": sum(item["classification"] != "invalid" for item in selected),
        }
    analytic = arms["analytic_calibrated"]
    learned = arms["learned_calibrated_equivariant"]
    classification, primary = campaign_classification(
        valid=valid,
        analytic_radius=analytic["robust_radius"],
        learned_radius=learned["robust_radius"],
        target_radius=config.target_robust_radius,
        analytic_default_fail_count=(
            len(config.seeds) - analytic["ablation_pass_counts"]["all_default"]
        ),
        learned_default_fail_count=(
            len(config.seeds) - learned["ablation_pass_counts"]["all_default"]
        ),
        required_default_failures=config.required_seed_passes,
        levels=config.noise_levels,
    )
    campaign_result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": _sha256(source_path),
            "source_implementation_sha256": campaign["implementation_sha256"],
            "source_manifest_sha256": manifest,
            "source_manifest_entries": len(manifest_records),
            "noise_sha256": noise_sha256,
        },
        "summary": {
            "requested": len(config.conditions) * len(config.seeds),
            "scheduled": len(config.conditions) * len(config.seeds) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "fitted_diagnostic_suites": len(results) * len(CUTS),
            "fitted_diagnostic_nulls": len(results),
            "noise_levels": len(config.noise_levels),
            "secondary_ablations": len(ABLATIONS),
        },
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary,
            "arms": arms,
            "required_seed_passes": config.required_seed_passes,
            "target_robust_radius": config.target_robust_radius,
        },
        "results": [
            {
                "experiment_id": item["experiment_id"],
                "condition": item["condition"],
                "seed": item["seed"],
                "scientific_fingerprint": item["scientific_fingerprint"],
                "classification": item["classification"],
                "robust_radius": item["robust_radius"],
                "path": item["artifacts"]["result"],
                "result_sha256": _sha256(Path(item["artifacts"]["result"])),
                "activations": item["artifacts"]["activations"],
                "activations_sha256": item["artifacts"]["activations_sha256"],
            }
            for item in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "Calibration sigma is a synthetic dimensionless stress coordinate, not a real instrument model.",
            "Diagnostic observers are fit once on clean predecessor splits and frozen across interventions.",
            "No TinyLLM model, front end, or task head is trained or updated.",
            "Five retained seeds do not establish architecture-population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "results": str(output / "runs" / "*" / "seed_*" / "result.json"),
            "activations": str(output / "runs" / "*" / "seed_*" / "activation_audit.npz"),
        },
    }
    _write_json(campaign_path, campaign_result)
    return campaign_result


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calibration_degradation_causal/"
            "20260807_d8_existing_checkpoints"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = CalibrationDegradationConfig(
        seeds=args.seeds,
        required_seed_passes=(
            len(args.seeds) if args.allow_underpowered else 4
        ),
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    result = run_campaign(config, args.output)
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
