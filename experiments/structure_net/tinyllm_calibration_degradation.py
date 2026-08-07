#!/usr/bin/env python3
"""Measure calibration-noise breakpoints in frozen calibrated TinyLLMs."""

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

import experiments.structure_net.tinyllm_calibrated_frontend_causal as source
import experiments.structure_net.tinyllm_conditional_branch_depth_scan as branch
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-calibration-degradation.v1"
HYPOTHESIS_ID = "tinyllm-calibration-noise-breakpoint-v1"
SOURCE_SCHEMA_VERSION = "nal.tinyllm-calibrated-frontend-causal.v1"
SOURCE_HYPOTHESIS_ID = "tinyllm-calibrated-reference-stable-cosine-quotient-v1"
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
CUTS = source.CUTS
REGIMES = source.REGIMES
PRIMARY_REGIMES = source.PRIMARY_REGIMES
PRIMARY_LEVELS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
SPLIT_SPECS = {
    "train": (2_048, 184, "interpolation"),
    "validation": (512, 294, "interpolation"),
    "in_distribution": (1_024, 390, "interpolation"),
    "composition": (1_024, 1_399, "composition"),
    "extrapolation": (1_024, 2_408, "extrapolation"),
}


@dataclass(frozen=True)
class CalibrationDegradationConfig:
    source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    conditions: tuple[str, ...] = CONDITIONS
    noise_levels: tuple[float, ...] = PRIMARY_LEVELS
    orientation_std_radians: float = 0.15
    log_speed_std: float = math.log(1.25)
    log_amplitude_std: float = math.log(1.25)
    offset_std: float = 0.10
    drift_std: float = 0.16
    corruption_seed: int = 9_308_003
    replay_tolerance: float = 1e-6
    required_seed_passes: int = 4
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if not self.allow_underpowered and self.seeds != (7, 17, 29, 41, 53):
            raise ValueError("primary seeds are fixed to 7,17,29,41,53")
        if set(self.conditions) != set(CONDITIONS):
            raise ValueError(f"conditions must be exactly {CONDITIONS}")
        if not self.allow_underpowered and self.noise_levels != PRIMARY_LEVELS:
            raise ValueError(f"primary noise levels are fixed to {PRIMARY_LEVELS}")
        if not self.noise_levels or self.noise_levels[0] != 0.0:
            raise ValueError("noise levels must begin at exact zero")
        if any(left >= right for left, right in zip(self.noise_levels, self.noise_levels[1:])):
            raise ValueError("noise levels must be strictly increasing")
        if any(value < 0.0 for value in self.noise_levels):
            raise ValueError("noise levels must be nonnegative")
        if any(
            value <= 0.0
            for value in (
                self.orientation_std_radians,
                self.log_speed_std,
                self.log_amplitude_std,
                self.offset_std,
                self.drift_std,
            )
        ):
            raise ValueError("all corruption scales must be positive")
        if self.required_seed_passes != 4 and not self.allow_underpowered:
            raise ValueError("primary campaign requires four joint seed passes")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(source.__file__),
        Path(branch.__file__),
        Path(invariant.__file__),
    ):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: CalibrationDegradationConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_sequential_frozen_checkpoint_robustness_evidence"
    )


def _load_source_campaign(
    config: CalibrationDegradationConfig,
) -> tuple[dict[str, Any], Path, CircleTaskConfig, source.CalibratedFrontendConfig]:
    path = Path(config.source_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    arms = campaign.get("aggregates", {}).get("arms", {})
    if (
        _sha256(path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_SCHEMA_VERSION
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or source._implementation_digest() != SOURCE_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 15
        or int(campaign.get("summary", {}).get("failed", -1)) != 0
        or any(
            arms.get(condition, {}).get("success") is not True
            or int(arms.get(condition, {}).get("joint_pass_count", -1)) != 5
            for condition in CONDITIONS
        )
    ):
        raise ValueError(f"invalid calibrated source campaign {path}")
    source_seeds = tuple(int(value) for value in campaign["configuration"]["seeds"])
    if not set(config.seeds).issubset(source_seeds):
        raise ValueError("requested seed is absent from source campaign")
    task = CircleTaskConfig(**campaign["task_config"])
    frontend_config = source._config_from_mapping(campaign["configuration"])
    return campaign, path, task, frontend_config


def _source_paths(
    config: CalibrationDegradationConfig, condition: str, seed: int
) -> tuple[Path, Path, Path]:
    root = Path(config.source_root) / "runs" / condition / f"seed_{seed}"
    return root / "result.json", root / "model.pt", root / "frontend.pt"


def _source_provenance(
    config: CalibrationDegradationConfig, condition: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path, model_path, frontend_path = _source_paths(config, condition, seed)
    detail = json.loads(result_path.read_text(encoding="utf-8"))
    training = detail.get("training", {})
    if (
        detail.get("schema_version") != SOURCE_SCHEMA_VERSION
        or detail.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or detail.get("status") != "completed"
        or detail.get("condition") != condition
        or int(detail.get("seed", -1)) != seed
        or detail.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or training.get("checkpoint") != str(model_path)
        or training.get("frontend_checkpoint") != str(frontend_path)
        or not model_path.is_file()
        or not frontend_path.is_file()
    ):
        raise ValueError(f"invalid calibrated source result {result_path}")
    provenance = {
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "scientific_fingerprint": detail["scientific_fingerprint"],
        "model_checkpoint": str(model_path),
        "model_checkpoint_sha256": _sha256(model_path),
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
        "model_state_sha256": training["final_model_state_sha256"],
        "system_state_sha256": training["final_system_state_sha256"],
    }
    return detail, provenance


def _load_system(
    config: CalibrationDegradationConfig,
    frontend_config: source.CalibratedFrontendConfig,
    task: CircleTaskConfig,
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[source.CalibratedTinyLLM, dict[str, Any], dict[str, Any]]:
    detail, provenance = _source_provenance(config, condition, seed)
    model = TinyLLMModel.from_checkpoint(provenance["model_checkpoint"], map_location="cpu")
    system = source.CalibratedTinyLLM(model, condition, task, frontend_config)
    frontend = torch.load(
        provenance["frontend_checkpoint"], map_location="cpu", weights_only=True
    )
    if frontend.get("condition") != condition:
        raise ValueError("source frontend condition mismatch")
    if system.scalar_embedding is None:
        raise AssertionError("structured scalar embedding is missing")
    system.scalar_embedding.load_state_dict(frontend["scalar_embedding"])
    if system.encoder is not None:
        system.encoder.load_state_dict(frontend["encoder"])
    if (
        _state_digest(model) != provenance["model_state_sha256"]
        or source._module_digest(system) != provenance["system_state_sha256"]
    ):
        raise ValueError(f"source checkpoint state mismatch for {condition} seed {seed}")
    return system.to(device).eval(), detail, provenance


def _dataset_digest(dataset: source.CalibratedDataset) -> str:
    circle = dataset.paired.circle
    fiber = dataset.paired.fiber
    return invariant._tensor_digest(
        circle.input_ids,
        circle.target_posteriors,
        circle.target_bins,
        circle.phases,
        circle.directions,
        fiber.cosine,
        fiber.branch,
        fiber.phase,
        fiber.fiber_id,
        dataset.calibration,
    )


def _datasets(task: CircleTaskConfig) -> dict[str, source.CalibratedDataset]:
    return {
        name: source.generate_calibrated_dataset(
            task,
            sample_count=count,
            seed=seed,
            regime=regime,
        )
        for name, (count, seed, regime) in SPLIT_SPECS.items()
    }


def _noise_fields(
    datasets: Mapping[str, source.CalibratedDataset], seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    fields = {}
    permutations = {}
    for index, (name, dataset) in enumerate(datasets.items()):
        generator = torch.Generator(device="cpu").manual_seed(seed + index * 10_007)
        fields[name] = torch.randn(
            len(dataset.calibration), 7, generator=generator, dtype=torch.float32
        )
        permutations[name] = torch.randperm(
            len(dataset.calibration), generator=generator
        )
    return fields, permutations


def corrupt_calibration(
    calibration: torch.Tensor,
    noise: torch.Tensor,
    level: float,
    config: CalibrationDegradationConfig,
) -> torch.Tensor:
    """Apply one nested, physically typed corruption to a calibration packet."""
    if calibration.ndim != 2 or calibration.shape[1] != source.CALIBRATION_WIDTH:
        raise ValueError("calibration must have shape [examples, 8]")
    if noise.shape != (len(calibration), 7):
        raise ValueError("calibration noise must have shape [examples, 7]")
    if level == 0.0:
        return calibration.clone()
    result = calibration.clone()
    angle = torch.atan2(calibration[:, 1], calibration[:, 0])
    angle = angle + level * config.orientation_std_radians * noise[:, 0]
    result[:, 0] = torch.cos(angle)
    result[:, 1] = torch.sin(angle)
    result[:, 2] = (
        calibration[:, 2].sign()
        * calibration[:, 2].abs()
        * torch.exp(level * config.log_speed_std * noise[:, 1])
    )
    result[:, 3] = calibration[:, 3] * torch.exp(
        level * config.log_amplitude_std * noise[:, 2]
    )
    result[:, 4:6] = (
        calibration[:, 4:6] + level * config.offset_std * noise[:, 3:5]
    )
    result[:, 6:8] = (
        calibration[:, 6:8] + level * config.drift_std * noise[:, 5:7]
    )
    return result


def corruption_metrics(
    source_calibration: torch.Tensor, corrupted: torch.Tensor
) -> dict[str, float]:
    source_angle = torch.atan2(source_calibration[:, 1], source_calibration[:, 0])
    corrupted_angle = torch.atan2(corrupted[:, 1], corrupted[:, 0])
    angle_error = torch.atan2(
        torch.sin(corrupted_angle - source_angle),
        torch.cos(corrupted_angle - source_angle),
    )
    speed_log_error = torch.log(
        corrupted[:, 2].abs().clamp_min(1e-12)
        / source_calibration[:, 2].abs().clamp_min(1e-12)
    )
    amplitude_log_error = torch.log(
        corrupted[:, 3].clamp_min(1e-12)
        / source_calibration[:, 3].clamp_min(1e-12)
    )

    def rms(value: torch.Tensor) -> float:
        return float(torch.sqrt(value.double().square().mean()))

    return {
        "orientation_rms_radians": rms(angle_error),
        "log_speed_rms": rms(speed_log_error),
        "log_amplitude_rms": rms(amplitude_log_error),
        "offset_rms_sensor_units": rms(corrupted[:, 4:6] - source_calibration[:, 4:6]),
        "drift_rms_sensor_units": rms(corrupted[:, 6:8] - source_calibration[:, 6:8]),
        "orientation_unit_norm_max_error": float(
            (corrupted[:, :2].norm(dim=1) - 1.0).abs().max()
        ),
        "minimum_amplitude": float(corrupted[:, 3].min()),
        "minimum_absolute_speed": float(corrupted[:, 2].abs().min()),
    }


def _corrupted_datasets(
    datasets: Mapping[str, source.CalibratedDataset],
    fields: Mapping[str, torch.Tensor],
    config: CalibrationDegradationConfig,
    level: float,
) -> dict[str, source.CalibratedDataset]:
    return {
        name: source.CalibratedDataset(
            paired=dataset.paired,
            calibration=corrupt_calibration(
                dataset.calibration, fields[name], level, config
            ),
        )
        for name, dataset in datasets.items()
    }


def _shuffled_datasets(
    datasets: Mapping[str, source.CalibratedDataset],
    permutations: Mapping[str, torch.Tensor],
) -> dict[str, source.CalibratedDataset]:
    return {
        name: source.CalibratedDataset(
            paired=dataset.paired,
            calibration=dataset.calibration[permutations[name]],
        )
        for name, dataset in datasets.items()
    }


def _evaluate_dataset_set(
    system: source.CalibratedTinyLLM,
    datasets: Mapping[str, source.CalibratedDataset],
    task: CircleTaskConfig,
    frontend_config: source.CalibratedFrontendConfig,
    seed: int,
    device: torch.device,
    null: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    features = {
        name: source.extract_features(system, dataset, task, frontend_config, device)
        for name, dataset in datasets.items()
    }
    evaluation_fibers = {name: datasets[name].paired.fiber for name in REGIMES}
    probe_config = source._analysis_config(frontend_config)
    cuts = {}
    for cut_index, cut in enumerate(CUTS):
        probe = branch.fit_conditional_probe(
            features["train"][cut],
            features["validation"][cut],
            {name: features[name][cut] for name in REGIMES},
            datasets["train"].paired.fiber,
            datasets["validation"].paired.fiber,
            evaluation_fibers,
            probe_config,
            device,
            kind="nonlinear",
            seed=frontend_config.analysis_seed + seed * 10_007 + cut_index * 1_009,
        )
        branch._add_log_loss_gains(probe, null)
        cuts[cut] = {"probe": probe}
    return {
        "cuts": cuts,
        "task_metrics": {
            regime: source.task_metrics(
                system, datasets[regime], task, frontend_config, device
            )
            for regime in REGIMES
        },
    }


def _baseline_replay_error(
    evaluated: Mapping[str, Any], source_analysis: Mapping[str, Any]
) -> float:
    maximum = 0.0
    probe_fields = (
        "cosine_pearson",
        "balanced_accuracy",
        "conditional_log_loss_gain_over_cosine_only",
    )
    for cut in CUTS:
        for regime in REGIMES:
            current = evaluated["cuts"][cut]["probe"]["evaluations"][regime]
            expected = source_analysis["cuts"][cut]["probe"]["evaluations"][regime]
            maximum = max(
                maximum,
                *(abs(float(current[name]) - float(expected[name])) for name in probe_fields),
            )
    for regime in REGIMES:
        current = evaluated["task_metrics"][regime]
        expected = source_analysis["task_metrics"][regime]
        maximum = max(
            maximum,
            *(abs(float(current[name]) - float(expected[name])) for name in current),
        )
    return maximum


def joint_seed_pass(evaluated: Mapping[str, Any]) -> bool:
    return all(
        invariant.endpoint_pass(
            evaluated["cuts"][cut]["probe"]["evaluations"][regime]
        )
        for cut in CUTS
        for regime in PRIMARY_REGIMES
    )


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _fingerprint(
    config: CalibrationDegradationConfig,
    condition: str,
    seed: int,
    implementation: str,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    noise_hashes: Mapping[str, str],
    permutation_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "condition": condition,
        "seed": seed,
        "implementation_sha256": implementation,
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source": provenance,
        "dataset_hashes": dataset_hashes,
        "noise_hashes": noise_hashes,
        "permutation_hashes": permutation_hashes,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _result_reusable(
    record: Mapping[str, Any],
    config: CalibrationDegradationConfig,
    condition: str,
    seed: int,
    implementation: str,
    fingerprint: str,
) -> bool:
    return bool(
        record.get("schema_version") == SCHEMA_VERSION
        and record.get("hypothesis_id") == HYPOTHESIS_ID
        and record.get("status") == "completed"
        and record.get("condition") == condition
        and int(record.get("seed", -1)) == seed
        and record.get("configuration")
        == json.loads(json.dumps(asdict(config)))
        and record.get("implementation_sha256") == implementation
        and record.get("scientific_fingerprint") == fingerprint
    )


def _level_key(level: float) -> str:
    return format(level, "g")


def _run_cell(
    config: CalibrationDegradationConfig,
    condition: str,
    seed: int,
    output: Path,
    implementation: str,
    task: CircleTaskConfig,
    frontend_config: source.CalibratedFrontendConfig,
    datasets: Mapping[str, source.CalibratedDataset],
    fields: Mapping[str, torch.Tensor],
    permutations: Mapping[str, torch.Tensor],
    dataset_hashes: Mapping[str, str],
    noise_hashes: Mapping[str, str],
    permutation_hashes: Mapping[str, str],
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    started = time.perf_counter()
    detail, provenance = _source_provenance(config, condition, seed)
    fingerprint = _fingerprint(
        config,
        condition,
        seed,
        implementation,
        provenance,
        dataset_hashes,
        noise_hashes,
        permutation_hashes,
    )
    result_path = output / "runs" / condition / f"seed_{seed}" / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if _result_reusable(
            existing, config, condition, seed, implementation, fingerprint
        ):
            print(f"reuse {condition} seed {seed}", flush=True)
            return existing, True
        raise ValueError(f"incompatible completed result {result_path}")

    system, detail, provenance = _load_system(
        config, frontend_config, task, condition, seed, device
    )
    evaluation_fibers = {name: datasets[name].paired.fiber for name in REGIMES}
    probe_config = source._analysis_config(frontend_config)
    null = branch.fit_cosine_only_null(
        datasets["train"].paired.fiber,
        datasets["validation"].paired.fiber,
        evaluation_fibers,
        probe_config,
        device,
        kind="nonlinear",
        seed=frontend_config.analysis_seed + seed * 101,
    )
    evaluations = {}
    zero_equal = True
    for level in config.noise_levels:
        corrupted = _corrupted_datasets(datasets, fields, config, level)
        if level == 0.0:
            zero_equal = all(
                torch.equal(corrupted[name].calibration, datasets[name].calibration)
                for name in datasets
            )
        analysis = _evaluate_dataset_set(
            system, corrupted, task, frontend_config, seed, device, null
        )
        evaluations[_level_key(level)] = {
            "level": level,
            "calibration_error": {
                name: corruption_metrics(
                    datasets[name].calibration, corrupted[name].calibration
                )
                for name in datasets
            },
            "analysis": analysis,
            "joint_seed_pass": joint_seed_pass(analysis),
        }
        print(f"{condition} seed {seed} q={level:g}", flush=True)
    shuffled = _shuffled_datasets(datasets, permutations)
    shuffled_analysis = _evaluate_dataset_set(
        system, shuffled, task, frontend_config, seed, device, null
    )
    evaluations["shuffled"] = {
        "level": None,
        "calibration_error": {
            name: corruption_metrics(
                datasets[name].calibration, shuffled[name].calibration
            )
            for name in datasets
        },
        "analysis": shuffled_analysis,
        "joint_seed_pass": joint_seed_pass(shuffled_analysis),
    }
    replay_error = _baseline_replay_error(
        evaluations["0"]["analysis"], detail["analysis"]
    )
    numerical = _all_finite(evaluations)
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
        "provenance": {
            "source_campaign": str(Path(config.source_root) / "campaign_results.json"),
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            **provenance,
        },
        "dataset_hashes": dict(dataset_hashes),
        "noise_hashes": dict(noise_hashes),
        "permutation_hashes": dict(permutation_hashes),
        "evaluations": evaluations,
        "gates": {
            "source_provenance_contract": True,
            "source_checkpoint_state_contract": True,
            "zero_calibration_identity_contract": zero_equal,
            "zero_source_metric_replay_contract": bool(
                replay_error <= config.replay_tolerance
            ),
            "finite_numerical_contract": numerical,
        },
        "maximum_zero_source_metric_replay_error": replay_error,
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": {"result": str(result_path)},
    }
    _write_json(result_path, result)
    del system
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, False


def _metric_summary(
    selected: Sequence[Mapping[str, Any]], level_key: str
) -> dict[str, Any]:
    cuts = {}
    for cut in CUTS:
        cuts[cut] = {}
        for regime in REGIMES:
            values = [
                item["evaluations"][level_key]["analysis"]["cuts"][cut]["probe"][
                    "evaluations"
                ][regime]
                for item in selected
            ]
            cuts[cut][regime] = {
                "cosine_pearson_mean": float(
                    np.mean([value["cosine_pearson"] for value in values])
                ),
                "cosine_pearson_minimum": float(
                    np.min([value["cosine_pearson"] for value in values])
                ),
                "cosine_pearson_maximum": float(
                    np.max([value["cosine_pearson"] for value in values])
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
                "endpoint_pass_count": int(
                    sum(invariant.endpoint_pass(value) for value in values)
                ),
            }
    task = {}
    for regime in REGIMES:
        values = [
            item["evaluations"][level_key]["analysis"]["task_metrics"][regime]
            for item in selected
        ]
        task[regime] = {
            name + "_mean": float(np.mean([value[name] for value in values]))
            for name in values[0]
        }
    return {"cuts": cuts, "task_metrics": task}


def breakpoint_summary(
    level_passes: Sequence[bool], levels: Sequence[float]
) -> dict[str, Any]:
    if len(level_passes) != len(levels):
        raise ValueError("level pass vector and grid differ")
    failed_indices = [index for index in range(1, len(levels)) if not level_passes[index]]
    first_failed_index = failed_indices[0] if failed_indices else None
    first_failed = levels[first_failed_index] if first_failed_index is not None else None
    last_passing = None
    if first_failed_index is not None:
        passing_before = [
            levels[index]
            for index in range(first_failed_index)
            if level_passes[index]
        ]
        last_passing = passing_before[-1] if passing_before else None
    elif level_passes:
        last_passing = levels[-1]
    reentry = bool(
        first_failed_index is not None
        and any(level_passes[index] for index in range(first_failed_index + 1, len(levels)))
    )
    return {
        "first_failed_level": first_failed,
        "last_passing_level": last_passing,
        "breakpoint_interval": (
            [last_passing, first_failed] if first_failed is not None else None
        ),
        "right_censored": first_failed is None,
        "gate_reentry": reentry,
        "zero_pass": bool(level_passes[0]),
        "any_nonzero_pass": bool(any(level_passes[1:])),
        "maximum_level_fails": bool(not level_passes[-1]),
    }


def aggregate_results(
    results: Sequence[Mapping[str, Any]], config: CalibrationDegradationConfig
) -> dict[str, Any]:
    required = (
        len(config.seeds) if config.allow_underpowered else config.required_seed_passes
    )
    arms = {}
    for condition in config.conditions:
        selected = sorted(
            (item for item in results if item["condition"] == condition),
            key=lambda item: item["seed"],
        )
        levels = {}
        level_passes = []
        for level in config.noise_levels:
            key = _level_key(level)
            pass_by_seed = {
                str(item["seed"]): bool(item["evaluations"][key]["joint_seed_pass"])
                for item in selected
            }
            pass_count = sum(pass_by_seed.values())
            level_pass = pass_count >= required
            level_passes.append(level_pass)
            levels[key] = {
                "level": level,
                "joint_pass_count": pass_count,
                "joint_pass_required": required,
                "level_pass": level_pass,
                "pass_by_seed": pass_by_seed,
                **_metric_summary(selected, key),
            }
        shuffled_pass_by_seed = {
            str(item["seed"]): bool(
                item["evaluations"]["shuffled"]["joint_seed_pass"]
            )
            for item in selected
        }
        shuffled_pass_count = sum(shuffled_pass_by_seed.values())
        breakpoint = breakpoint_summary(level_passes, config.noise_levels)
        shuffled_control_pass = shuffled_pass_count <= len(config.seeds) - required
        bounded_stable = bool(
            breakpoint["zero_pass"]
            and breakpoint["any_nonzero_pass"]
            and breakpoint["maximum_level_fails"]
            and not breakpoint["gate_reentry"]
            and shuffled_control_pass
        )
        arms[condition] = {
            "levels": levels,
            "shuffled_control": {
                "joint_pass_count": shuffled_pass_count,
                "pass_by_seed": shuffled_pass_by_seed,
                "negative_control_pass": shuffled_control_pass,
                **_metric_summary(selected, "shuffled"),
            },
            "breakpoint": breakpoint,
            "bounded_stable_breakpoint": bounded_stable,
        }
    validity = all(all(item["gates"].values()) for item in results)
    supported = bool(
        not config.allow_underpowered
        and validity
        and all(arms[name]["bounded_stable_breakpoint"] for name in config.conditions)
    )
    if config.allow_underpowered:
        conclusion = "systems_lifecycle_only_not_quality_evidence"
    elif not validity:
        conclusion = "invalid_campaign"
    elif supported:
        conclusion = "bounded_stable_calibration_breakpoints_both_arms"
    elif any(
        arm["breakpoint"]["gate_reentry"] for arm in arms.values()
    ):
        conclusion = "nonmonotone_calibration_curve_unresolved"
    elif any(arm["breakpoint"]["right_censored"] for arm in arms.values()):
        conclusion = "calibration_breakpoint_right_censored"
    elif any(
        not arm["shuffled_control"]["negative_control_pass"]
        for arm in arms.values()
    ):
        conclusion = "shuffled_calibration_control_failed"
    elif any(not arm["breakpoint"]["any_nonzero_pass"] for arm in arms.values()):
        conclusion = "exact_calibration_brittle_on_registered_scale"
    else:
        conclusion = "checkpoint_or_arm_stratified_calibration_breakpoint"
    return {
        "arms": arms,
        "preregistered_gate": {
            "required_seed_passes": required,
            "cosine_threshold": 0.90,
            "branch_accuracy_ceiling": 0.55,
            "cuts": list(CUTS),
            "regimes": list(PRIMARY_REGIMES),
            "noise_levels": list(config.noise_levels),
        },
        "validity_contract": validity,
        "supported": supported,
        "conclusion": conclusion,
        "gate_counts": {
            name: sum(bool(item["gates"][name]) for item in results)
            for name in results[0]["gates"]
        },
    }


def run_campaign(
    config: CalibrationDegradationConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    implementation = _implementation_digest()
    source_campaign, source_path, task, frontend_config = _load_source_campaign(config)
    datasets = _datasets(task)
    fields, permutations = _noise_fields(datasets, config.corruption_seed)
    dataset_hashes = {name: _dataset_digest(value) for name, value in datasets.items()}
    noise_hashes = {
        name: invariant._tensor_digest(value) for name, value in fields.items()
    }
    permutation_hashes = {
        name: invariant._tensor_digest(value)
        for name, value in permutations.items()
    }
    campaign_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "configuration": asdict(config),
                "implementation_sha256": implementation,
                "source_campaign_sha256": _sha256(source_path),
                "dataset_hashes": dataset_hashes,
                "noise_hashes": noise_hashes,
                "permutation_hashes": permutation_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("hypothesis_id") == HYPOTHESIS_ID
            and existing.get("status") == "completed"
            and existing.get("scientific_fingerprint") == campaign_fingerprint
            and existing.get("implementation_sha256") == implementation
            and all(
                Path(item["path"]).is_file()
                and _sha256(Path(item["path"])) == item["result_sha256"]
                for item in existing.get("results", [])
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    results = []
    reused = 0
    for condition in config.conditions:
        for seed in config.seeds:
            result, was_reused = _run_cell(
                config,
                condition,
                seed,
                output,
                implementation,
                task,
                frontend_config,
                datasets,
                fields,
                permutations,
                dataset_hashes,
                noise_hashes,
                permutation_hashes,
                device,
            )
            results.append(result)
            reused += int(was_reused)
            if _implementation_digest() != implementation:
                raise RuntimeError("calibration-degradation implementation changed during run")
    aggregates = aggregate_results(results, config)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "task_config": asdict(task),
        "scientific_fingerprint": campaign_fingerprint,
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": _sha256(source_path),
            "source_implementation_sha256": source_campaign["implementation_sha256"],
            "dataset_hashes": dataset_hashes,
            "noise_hashes": noise_hashes,
            "permutation_hashes": permutation_hashes,
        },
        "summary": {
            "requested": len(config.conditions) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_frontends": 0,
            "fitted_evaluation_probes": len(results)
            * (1 + len(config.noise_levels) * len(CUTS) + len(CUTS)),
        },
        "aggregates": aggregates,
        "results": [
            {
                "experiment_id": item["experiment_id"],
                "condition": item["condition"],
                "seed": item["seed"],
                "scientific_fingerprint": item["scientific_fingerprint"],
                "gates": item["gates"],
                "path": item["artifacts"]["result"],
                "result_sha256": _sha256(Path(item["artifacts"]["result"])),
            }
            for item in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The checkpoints were selected because exact calibrated quotient gates passed.",
            "Calibration is corrupted only at inference after exact-calibration training.",
            "The Gaussian scale grid is synthetic and not an instrument specification.",
            "Conditional branch accuracy is recoverability under the declared nonlinear probe.",
            "The study does not test biased, adversarial, or learned pilot calibration.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "*" / "seed_*" / "result.json"),
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calibration_degradation/"
            "20260807_d8_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", type=_ints, default=(7, 17, 29, 41, 53))
    parser.add_argument("--noise-levels", type=_floats, default=PRIMARY_LEVELS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = CalibrationDegradationConfig(
        seeds=args.seeds,
        noise_levels=args.noise_levels,
        required_seed_passes=(len(args.seeds) if args.allow_underpowered else 4),
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
