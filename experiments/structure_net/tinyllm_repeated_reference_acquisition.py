#!/usr/bin/env python3
"""Test repeated orientation-reference acquisition in frozen TinyLLM systems."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibration_orientation_noise as orientation
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-repeated-reference-acquisition.v1"
HYPOTHESIS_ID = "tinyllm-repeated-reference-acquisition-v1"
SOURCE_ORIENTATION_CAMPAIGN_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_ORIENTATION_IMPLEMENTATION_SHA256 = (
    "990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70"
)
SOURCE_ORIENTATION_NOISE_SHA256 = (
    "b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7"
)
SOURCE_CALIBRATED_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_CALIBRATED_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
MOTIVATING_READOUT_CAMPAIGN_SHA256 = (
    "833392ad7956ddcf715a20211431586f371edee280f167bf5a4fa51437cdc6c6"
)
ARMS = ("analytic_calibrated", "learned_calibrated_equivariant")
RULES = ("analytic_circular_mean", "learned_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
REPLICATE_COUNTS = (1, 4, 16, 64, 256)
RATE_COUNTS = (16, 64, 256)
MASTER_NOISE_SEED = 42_700_019
SHUFFLE_SEED = 50_800_019
SPLIT_SPECS = {
    "composition": (20_399, "composition", 1_024),
    "extrapolation": (21_408, "extrapolation", 1_024),
}


@dataclass(frozen=True)
class RepeatedReferenceConfig:
    source_orientation_root: str = (
        "data/experiments/tinyllm_calibration_orientation_noise/"
        "20260807_d8_preregistered"
    )
    arms: tuple[str, ...] = ARMS
    seeds: tuple[int, ...] = SEEDS
    replicate_counts: tuple[int, ...] = REPLICATE_COUNTS
    measurement_sigma_radians: float = 0.175
    accuracy_loss_ceiling: float = 0.03
    learned_match_accuracy_ceiling: float = 0.02
    required_seed_passes: int = 4
    replay_tolerance: float = 2e-6
    unit_norm_tolerance: float = 1e-6
    equivariance_tolerance: float = 1e-6
    rate_ratio_minimum: float = 0.80
    rate_ratio_maximum: float = 1.20
    rate_slope_minimum: float = -0.60
    rate_slope_maximum: float = -0.40
    master_noise_seed: int = MASTER_NOISE_SEED
    shuffle_seed: int = SHUFFLE_SEED
    activation_batch_size: int = 256
    denoiser_hidden: int = 32
    denoiser_learning_rate: float = 3e-3
    denoiser_batch_size: int = 512
    denoiser_steps: int = 1_200
    denoiser_validation_interval: int = 50
    denoiser_patience: int = 6
    denoiser_seed: int = 271_828
    denoiser_validation_seed: int = 271_829
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if not self.arms or set(self.arms).difference(ARMS):
            raise ValueError("unknown or empty acquisition arm")
        if tuple(sorted(set(self.replicate_counts))) != self.replicate_counts:
            raise ValueError("replicate counts must be unique and ordered")
        if not self.replicate_counts or self.replicate_counts[0] != 1:
            raise ValueError("replicate grid must begin at one")
        if self.required_seed_passes > len(self.seeds):
            raise ValueError("required seed passes exceeds selected seeds")
        if self.measurement_sigma_radians <= 0.0:
            raise ValueError("measurement sigma must be positive")
        if self.denoiser_hidden < 1 or self.denoiser_batch_size < 1:
            raise ValueError("invalid denoiser width or batch size")
        if self.denoiser_steps < 1 or self.denoiser_validation_interval < 1:
            raise ValueError("invalid denoiser training schedule")
        if self.denoiser_patience < 1:
            raise ValueError("denoiser patience must be positive")
        if not self.allow_underpowered:
            if self.arms != ARMS or self.seeds != SEEDS:
                raise ValueError("primary arms and five seeds are fixed")
            if self.replicate_counts != REPLICATE_COUNTS:
                raise ValueError("primary replicate grid is fixed")
            if self.measurement_sigma_radians != 0.175:
                raise ValueError("primary measurement sigma is fixed")
            if self.master_noise_seed != MASTER_NOISE_SEED:
                raise ValueError("primary acquisition seed is fixed")
            if self.shuffle_seed != SHUFFLE_SEED:
                raise ValueError("primary shuffle seed is fixed")
            if self.required_seed_passes != 4:
                raise ValueError("primary population gate is four of five")
            if (
                self.denoiser_hidden != 32
                or self.denoiser_learning_rate != 3e-3
                or self.denoiser_batch_size != 512
                or self.denoiser_steps != 1_200
                or self.denoiser_validation_interval != 50
                or self.denoiser_patience != 6
                or self.denoiser_seed != 271_828
                or self.denoiser_validation_seed != 271_829
            ):
                raise ValueError("primary denoiser protocol is fixed")


class EquivariantReliabilityAggregator(nn.Module):
    """Permutation-invariant SO(2)-equivariant reliability-weighted mean."""

    def __init__(self, hidden: int = 32, max_count: int = 256) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.max_count = int(max_count)
        self.score = nn.Sequential(
            nn.Linear(3, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3 or observations.shape[-1] != 2:
            raise ValueError("observations must have shape (batch, count, 2)")
        count = observations.shape[1]
        if count < 1 or count > self.max_count:
            raise ValueError("observation count is outside the declared range")
        provisional_sum = observations.sum(1)
        provisional = provisional_sum / provisional_sum.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        agreement = (observations * provisional[:, None, :]).sum(-1)
        resultant = observations.mean(1).norm(dim=1, keepdim=True)
        normalized_log_count = math.log(float(count)) / math.log(
            float(self.max_count)
        )
        context = torch.stack(
            (
                agreement,
                resultant.expand(-1, count),
                torch.full_like(agreement, normalized_log_count),
            ),
            dim=-1,
        )
        weights = torch.softmax(self.score(context).squeeze(-1), dim=1)
        result = (weights[..., None] * observations).sum(1)
        return result / result.norm(dim=1, keepdim=True).clamp_min(1e-12)


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


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _runner_protocol_digest() -> str:
    """Hash the producing protocol while excluding non-scientific CLI defaults."""
    protocol_names = (
        "RepeatedReferenceConfig",
        "EquivariantReliabilityAggregator",
        "_evidence_role",
        "replicate_key",
        "_finite",
        "_state_digest",
        "_base_angles",
        "_observed_orientations",
        "_orientation_audit",
        "circular_mean_orientation",
        "learned_mean_orientation",
        "apply_orientation",
        "task_recovery_pass",
        "standard_error_contract",
        "classify_campaign",
        "_load_source",
        "_datasets",
        "_dataset_hash",
        "_ensure_noise_arrays",
        "_synthetic_batch",
        "_denoiser_validation_loss",
        "_denoiser_contracts",
        "_denoiser_fingerprint",
        "_fit_or_load_denoiser",
        "_build_interventions",
        "_source_result",
        "_fingerprint",
        "_reusable_result",
        "_campaign_reusable",
        "run_campaign",
    )
    constants = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "source_orientation_campaign_sha256": (
            SOURCE_ORIENTATION_CAMPAIGN_SHA256
        ),
        "source_orientation_implementation_sha256": (
            SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
        ),
        "source_orientation_noise_sha256": SOURCE_ORIENTATION_NOISE_SHA256,
        "source_calibrated_campaign_sha256": SOURCE_CALIBRATED_CAMPAIGN_SHA256,
        "source_calibrated_implementation_sha256": (
            SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
        ),
        "motivating_readout_campaign_sha256": (
            MOTIVATING_READOUT_CAMPAIGN_SHA256
        ),
        "arms": ARMS,
        "rules": RULES,
        "seeds": SEEDS,
        "regimes": REGIMES,
        "replicate_counts": REPLICATE_COUNTS,
        "rate_counts": RATE_COUNTS,
        "master_noise_seed": MASTER_NOISE_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "split_specs": SPLIT_SPECS,
    }
    material = {
        "protocol_names": protocol_names,
        "sources": {
            name: inspect.getsource(globals()[name]) for name in protocol_names
        },
        "constants": constants,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_digests() -> dict[str, str]:
    paths = {
        "orientation_noise": Path(orientation.__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "tinyllm_model": Path(inspect.getfile(TinyLLMModel)),
    }
    return {
        "runner_protocol": _runner_protocol_digest(),
        **{name: _sha256(path) for name, path in paths.items()},
    }


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    material = dict(digests or _source_digests())
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_role(config: RepeatedReferenceConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "corrective_outcome_informed_frozen_system_acquisition_intervention"
    )


def replicate_key(count: int) -> str:
    return f"m_{count:03d}"


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _base_angles(
    true_orientation: torch.Tensor,
    fiber_inverse: torch.Tensor,
) -> torch.Tensor:
    if true_orientation.ndim != 2 or true_orientation.shape[1] != 2:
        raise ValueError("true orientation must have shape (rows, 2)")
    if len(fiber_inverse) != len(true_orientation):
        raise ValueError("fiber mapping and orientation length differ")
    return torch.atan2(
        true_orientation[:, 1].double(), true_orientation[:, 0].double()
    )


def _observed_orientations(
    true_orientation: torch.Tensor,
    fiber_inverse: torch.Tensor,
    unique_errors: torch.Tensor,
    sigma: float,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unique_errors.ndim != 2 or count < 1 or count > unique_errors.shape[0]:
        raise ValueError("invalid acquisition error matrix or replicate count")
    base_angle = _base_angles(true_orientation, fiber_inverse)
    observed_angles = (
        base_angle[:, None]
        + float(sigma) * unique_errors[:count, fiber_inverse].double().transpose(0, 1)
    )
    observed = torch.stack(
        (torch.cos(observed_angles), torch.sin(observed_angles)), dim=-1
    )
    return observed, base_angle


def _orientation_audit(
    value: torch.Tensor,
    base_angle: torch.Tensor,
    observations: torch.Tensor,
    sigma: float,
    count: int,
) -> dict[str, float]:
    value_angle = torch.atan2(value[:, 1], value[:, 0])
    difference = torch.atan2(
        torch.sin(value_angle - base_angle), torch.cos(value_angle - base_angle)
    )
    return {
        "replicate_count": int(count),
        "angular_mae_radians": float(difference.abs().mean()),
        "angular_rmse_radians": float(torch.sqrt(difference.square().mean())),
        "signed_angular_bias_radians": float(difference.mean()),
        "absolute_error_q95_radians": float(
            torch.quantile(difference.abs(), 0.95)
        ),
        "empirical_resultant_length": float(
            observations.mean(1).norm(dim=1).mean()
        ),
        "maximum_absolute_angular_error_radians": float(
            difference.abs().max()
        ),
        "maximum_unit_norm_error": float(
            (value.norm(dim=1) - 1.0).abs().max()
        ),
        "theoretical_standard_error_radians": float(sigma / math.sqrt(count)),
    }


def circular_mean_orientation(
    true_orientation: torch.Tensor,
    fiber_inverse: torch.Tensor,
    unique_errors: torch.Tensor,
    sigma: float,
    count: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return per-row nested circular means from unique-fiber errors."""
    observations, base_angle = _observed_orientations(
        true_orientation, fiber_inverse, unique_errors, sigma, count
    )
    total = observations.sum(1)
    norm = total.norm(dim=1, keepdim=True)
    if float(norm.min()) <= 1e-12:
        raise ValueError("circular mean is numerically undefined")
    value = total / norm
    audit = _orientation_audit(
        value, base_angle, observations, sigma, count
    )
    return value.float(), audit


@torch.no_grad()
def learned_mean_orientation(
    denoiser: EquivariantReliabilityAggregator,
    true_orientation: torch.Tensor,
    fiber_inverse: torch.Tensor,
    unique_errors: torch.Tensor,
    sigma: float,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    observations, base_angle = _observed_orientations(
        true_orientation, fiber_inverse, unique_errors, sigma, count
    )
    value = denoiser(observations.float().to(device)).double().cpu()
    audit = _orientation_audit(
        value, base_angle, observations, sigma, count
    )
    return value.float(), audit


def apply_orientation(
    dataset: calibrated.CalibratedDataset, orientation_value: torch.Tensor
) -> calibrated.CalibratedDataset:
    if orientation_value.shape != dataset.calibration[:, :2].shape:
        raise ValueError("replacement orientation shape mismatch")
    packet = dataset.calibration.clone()
    packet[:, :2] = orientation_value
    return calibrated.CalibratedDataset(paired=dataset.paired, calibration=packet)


def task_recovery_pass(
    metrics: Mapping[str, Mapping[str, float]],
    clean_metrics: Mapping[str, Mapping[str, float]],
    ceiling: float,
) -> tuple[bool, dict[str, Any]]:
    summary = {}
    passed = True
    for regime in REGIMES:
        loss = float(clean_metrics[regime]["exact_bin_accuracy"]) - float(
            metrics[regime]["exact_bin_accuracy"]
        )
        regime_pass = loss <= ceiling
        summary[regime] = {
            "accuracy_loss_from_clean": loss,
            "pass": regime_pass,
        }
        passed = bool(passed and regime_pass)
    return passed, summary


def standard_error_contract(
    audits: Mapping[str, Mapping[int, Mapping[str, float]]],
    config: RepeatedReferenceConfig,
) -> dict[str, Any]:
    if not set(RATE_COUNTS).issubset(config.replicate_counts):
        return {
            "evaluated": False,
            "pass": bool(config.allow_underpowered),
            "reason": "rate counts absent from lifecycle grid",
        }
    splits = {}
    overall = True
    for regime in REGIMES:
        ratios = {
            replicate_key(count): float(
                audits[regime][count]["angular_rmse_radians"]
                / (config.measurement_sigma_radians / math.sqrt(count))
            )
            for count in RATE_COUNTS
        }
        slope = float(
            np.polyfit(
                np.log(np.asarray(RATE_COUNTS, dtype=np.float64)),
                np.log(
                    np.asarray(
                        [
                            audits[regime][count]["angular_rmse_radians"]
                            for count in RATE_COUNTS
                        ],
                        dtype=np.float64,
                    )
                ),
                1,
            )[0]
        )
        passed = bool(
            all(
                config.rate_ratio_minimum <= value <= config.rate_ratio_maximum
                for value in ratios.values()
            )
            and config.rate_slope_minimum <= slope <= config.rate_slope_maximum
        )
        splits[regime] = {"ratios": ratios, "log_log_slope": slope, "pass": passed}
        overall = bool(overall and passed)
    return {"evaluated": True, "splits": splits, "pass": overall}


def classify_campaign(
    *,
    valid: bool,
    analytic_primary_by_arm: Mapping[str, bool],
    learned_primary_by_arm: Mapping[str, bool],
    learned_match_by_arm: Mapping[str, bool],
    rate_pass: bool,
    controls_pass: bool,
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    analytic_all = all(analytic_primary_by_arm.get(arm, False) for arm in ARMS)
    learned_all = all(learned_primary_by_arm.get(arm, False) for arm in ARMS)
    match_all = all(learned_match_by_arm.get(arm, False) for arm in ARMS)
    if analytic_all and learned_all and match_all and rate_pass and controls_pass:
        return "acquisition_variance_causally_sufficient", True
    if analytic_all and (not learned_all or not match_all):
        return "analytic_averaging_sufficient_learned_gap", False
    if learned_all and not analytic_all:
        return "learned_non_gaussian_advantage", False
    if (
        not any(analytic_primary_by_arm.values())
        and not any(learned_primary_by_arm.values())
        and controls_pass
    ):
        return "repeated_acquisition_insufficient", False
    if (
        len(set(analytic_primary_by_arm.values())) > 1
        or len(set(learned_primary_by_arm.values())) > 1
    ):
        return "arm_stratified_acquisition_limit", False
    return "mixed_acquisition_result", False


def _load_source(
    config: RepeatedReferenceConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[tuple[str, int], Mapping[str, Any]],
    dict[str, Any],
    CircleTaskConfig,
    calibrated.CalibratedFrontendConfig,
]:
    orientation_path = Path(config.source_orientation_root) / "campaign_results.json"
    orientation_campaign = json.loads(orientation_path.read_text(encoding="utf-8"))
    entries = {
        (item["condition"], int(item["seed"])): item
        for item in orientation_campaign.get("results", [])
    }
    expected = {(arm, seed) for arm in config.arms for seed in config.seeds}
    source_noise_path = Path(
        orientation_campaign.get("artifacts", {}).get("noise_arrays", "")
    )
    if (
        _sha256(orientation_path) != SOURCE_ORIENTATION_CAMPAIGN_SHA256
        or orientation_campaign.get("schema_version") != orientation.SCHEMA_VERSION
        or orientation_campaign.get("hypothesis_id") != orientation.HYPOTHESIS_ID
        or orientation_campaign.get("status") != "completed"
        or orientation_campaign.get("implementation_sha256")
        != SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
        or orientation_campaign.get("aggregates", {}).get("classification")
        != "reference_precision_critical"
        or orientation_campaign.get("artifacts", {}).get("noise_arrays_sha256")
        != SOURCE_ORIENTATION_NOISE_SHA256
        or not source_noise_path.is_file()
        or _sha256(source_noise_path) != SOURCE_ORIENTATION_NOISE_SHA256
        or not expected.issubset(entries)
        or any(
            not Path(entries[key]["path"]).is_file()
            or _sha256(Path(entries[key]["path"])) != entries[key]["result_sha256"]
            for key in expected
        )
    ):
        raise ValueError("invalid orientation-noise source campaign")
    orientation_config = orientation.OrientationNoiseConfig(
        seeds=config.seeds,
        required_seed_passes=(
            config.required_seed_passes
            if not config.allow_underpowered
            else len(config.seeds)
        ),
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    calibrated_campaign, calibrated_path, task, source_config = (
        orientation._load_source_campaign(orientation_config)
    )
    if (
        _sha256(calibrated_path) != SOURCE_CALIBRATED_CAMPAIGN_SHA256
        or calibrated_campaign.get("implementation_sha256")
        != SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
    ):
        raise ValueError("calibrated source identity changed")
    return (
        orientation_campaign,
        orientation_path,
        entries,
        calibrated_campaign,
        task,
        source_config,
    )


def _datasets(task: CircleTaskConfig) -> dict[str, calibrated.CalibratedDataset]:
    return {
        name: calibrated.generate_calibrated_dataset(
            task, sample_count=count, seed=seed, regime=regime, shuffle=True
        )
        for name, (seed, regime, count) in SPLIT_SPECS.items()
    }


def _dataset_hash(dataset: calibrated.CalibratedDataset) -> str:
    paired = dataset.paired
    return invariant._tensor_digest(
        paired.circle.input_ids,
        paired.circle.target_posteriors,
        paired.circle.target_bins,
        paired.fiber.cosine,
        paired.fiber.branch,
        paired.fiber.fiber_id,
        dataset.calibration,
    )


def _ensure_noise_arrays(
    path: Path,
    datasets: Mapping[str, calibrated.CalibratedDataset],
    source_noise_path: Path,
    config: RepeatedReferenceConfig,
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    output: dict[str, torch.Tensor] = {}
    contracts: dict[str, Any] = {
        "source_noise_sha256": _sha256(source_noise_path),
        "first_replicate_source_identity": True,
        "prefix_contract": True,
        "all_replicate_columns_distinct": True,
        "maximum_absolute_off_diagonal_correlation": 0.0,
    }
    with np.load(source_noise_path, allow_pickle=False) as source_archive:
        for split_index, (name, dataset) in enumerate(datasets.items()):
            fibers = dataset.paired.fiber.fiber_id.long()
            unique, inverse = torch.unique(fibers, sorted=True, return_inverse=True)
            source_fibers = source_archive[f"{name}__fiber_id"]
            source_rows = source_archive[f"{name}__base_noise"]
            if not np.array_equal(source_fibers, fibers.numpy()):
                raise ValueError(f"source fiber IDs changed for {name}")
            source_unique = np.empty(len(unique), dtype=np.float64)
            for index in range(len(unique)):
                values = source_rows[inverse.numpy() == index]
                if not len(values) or not np.all(values == values[0]):
                    raise ValueError(f"source paired-sheet noise differs for {name}")
                source_unique[index] = values[0]
            generator = np.random.default_rng(
                config.master_noise_seed + split_index * 1_000_003
            )
            errors = np.empty(
                (max(config.replicate_counts), len(unique)), dtype=np.float64
            )
            errors[0] = source_unique
            errors[1:] = generator.standard_normal(errors[1:].shape)
            correlations = np.corrcoef(errors)
            off_diagonal = np.abs(correlations - np.eye(len(errors)))
            maximum_correlation = float(off_diagonal.max())
            contracts["maximum_absolute_off_diagonal_correlation"] = max(
                contracts["maximum_absolute_off_diagonal_correlation"],
                maximum_correlation,
            )
            distinct = len({row.tobytes() for row in errors}) == len(errors)
            contracts["all_replicate_columns_distinct"] = bool(
                contracts["all_replicate_columns_distinct"] and distinct
            )
            contracts["first_replicate_source_identity"] = bool(
                contracts["first_replicate_source_identity"]
                and np.array_equal(errors[0], source_unique)
            )
            arrays[f"{name}__unique_fibers"] = unique.numpy()
            arrays[f"{name}__fiber_inverse"] = inverse.numpy()
            arrays[f"{name}__errors"] = errors
            output[f"{name}__fiber_inverse"] = inverse
            output[f"{name}__errors"] = torch.from_numpy(errors)
    contracts["independence_diagnostic_pass"] = bool(
        contracts["all_replicate_columns_distinct"]
        and contracts["maximum_absolute_off_diagonal_correlation"] <= 0.25
    )
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(archive[key], value)
                for key, value in arrays.items()
            ):
                raise ValueError(f"incompatible acquisition noise arrays {path}")
    else:
        _write_npz(path, arrays)
    return output, _sha256(path), contracts


def _synthetic_batch(
    *,
    batch_size: int,
    count: int,
    sigma: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.rand(batch_size, generator=generator) * (2.0 * math.pi)
    errors = torch.randn(batch_size, count, generator=generator)
    observed_angles = angles[:, None] + float(sigma) * errors
    observations = torch.stack(
        (torch.cos(observed_angles), torch.sin(observed_angles)), dim=-1
    )
    target_error = torch.randn(batch_size, generator=generator)
    target_angle = angles + float(sigma) * target_error
    target = torch.stack((torch.cos(target_angle), torch.sin(target_angle)), dim=-1)
    return observations, target


@torch.no_grad()
def _denoiser_validation_loss(
    denoiser: EquivariantReliabilityAggregator,
    validation: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    losses = []
    denoiser.eval()
    for observations, target in validation.values():
        prediction = denoiser(observations.to(device))
        losses.append(float((1.0 - (prediction * target.to(device)).sum(-1)).mean()))
    return float(np.mean(losses))


@torch.no_grad()
def _denoiser_contracts(
    denoiser: EquivariantReliabilityAggregator,
    device: torch.device,
) -> dict[str, float | bool]:
    generator = torch.Generator(device="cpu").manual_seed(8_675_309)
    count = min(16, denoiser.max_count)
    observations, _ = _synthetic_batch(
        batch_size=64, count=count, sigma=0.175, generator=generator
    )
    observations = observations.to(device)
    base = denoiser(observations)
    permutation = torch.randperm(count, generator=generator).to(device)
    permuted = denoiser(observations[:, permutation])
    angle = torch.tensor(0.731, dtype=observations.dtype, device=device)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    rotated_observations = torch.stack(
        (
            observations[..., 0] * cosine - observations[..., 1] * sine,
            observations[..., 0] * sine + observations[..., 1] * cosine,
        ),
        dim=-1,
    )
    rotated_output = denoiser(rotated_observations)
    expected_rotated = torch.stack(
        (
            base[..., 0] * cosine - base[..., 1] * sine,
            base[..., 0] * sine + base[..., 1] * cosine,
        ),
        dim=-1,
    )
    return {
        "maximum_permutation_error": float((base - permuted).abs().max()),
        "maximum_rotation_equivariance_error": float(
            (rotated_output - expected_rotated).abs().max()
        ),
        "maximum_unit_norm_error": float((base.norm(dim=1) - 1.0).abs().max()),
    }


def _denoiser_fingerprint(
    config: RepeatedReferenceConfig, implementation: str
) -> str:
    payload = {
        "implementation_sha256": implementation,
        "hidden": config.denoiser_hidden,
        "learning_rate": config.denoiser_learning_rate,
        "batch_size": config.denoiser_batch_size,
        "steps": config.denoiser_steps,
        "validation_interval": config.denoiser_validation_interval,
        "patience": config.denoiser_patience,
        "seed": config.denoiser_seed,
        "validation_seed": config.denoiser_validation_seed,
        "sigma": config.measurement_sigma_radians,
        "counts": [count for count in config.replicate_counts if count > 1],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fit_or_load_denoiser(
    path: Path,
    config: RepeatedReferenceConfig,
    implementation: str,
    device: torch.device,
) -> tuple[EquivariantReliabilityAggregator, dict[str, Any], str]:
    fingerprint = _denoiser_fingerprint(config, implementation)
    denoiser = EquivariantReliabilityAggregator(
        hidden=config.denoiser_hidden,
        max_count=max(config.replicate_counts),
    ).to(device)
    if path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError(f"incompatible acquisition denoiser {path}")
        denoiser.load_state_dict(checkpoint["state_dict"])
        denoiser.to(device).eval()
        if _state_digest(denoiser) != checkpoint.get("state_sha256"):
            raise ValueError(f"acquisition denoiser state changed {path}")
        contracts = _denoiser_contracts(denoiser, device)
        if contracts != checkpoint.get("contracts"):
            raise ValueError(f"acquisition denoiser contracts changed {path}")
        return denoiser, checkpoint, _sha256(path)

    torch.manual_seed(config.denoiser_seed)
    denoiser = EquivariantReliabilityAggregator(
        hidden=config.denoiser_hidden,
        max_count=max(config.replicate_counts),
    ).to(device)
    optimizer = torch.optim.Adam(
        denoiser.parameters(), lr=config.denoiser_learning_rate
    )
    train_generator = torch.Generator(device="cpu").manual_seed(
        config.denoiser_seed
    )
    validation_generator = torch.Generator(device="cpu").manual_seed(
        config.denoiser_validation_seed
    )
    train_counts = tuple(count for count in config.replicate_counts if count > 1)
    if not train_counts:
        raise ValueError("denoiser needs at least one repeated-reference count")
    validation = {
        count: _synthetic_batch(
            batch_size=max(1_024, config.denoiser_batch_size),
            count=count,
            sigma=config.measurement_sigma_radians,
            generator=validation_generator,
        )
        for count in train_counts
    }
    best_loss = _denoiser_validation_loss(denoiser, validation, device)
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in denoiser.state_dict().items()
    }
    trace = [{"step": 0, "validation_loss": best_loss}]
    stale = 0
    steps_run = 0
    for step in range(1, config.denoiser_steps + 1):
        count_index = int(
            torch.randint(len(train_counts), (1,), generator=train_generator)
        )
        count = train_counts[count_index]
        observations, target = _synthetic_batch(
            batch_size=config.denoiser_batch_size,
            count=count,
            sigma=config.measurement_sigma_radians,
            generator=train_generator,
        )
        denoiser.train()
        prediction = denoiser(observations.to(device))
        loss = (1.0 - (prediction * target.to(device)).sum(-1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps_run = step
        if step % config.denoiser_validation_interval == 0:
            validation_loss = _denoiser_validation_loss(
                denoiser, validation, device
            )
            trace.append({"step": step, "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in denoiser.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= config.denoiser_patience:
                break
    denoiser.load_state_dict(best_state)
    denoiser.to(device).eval()
    contracts = _denoiser_contracts(denoiser, device)
    contract_pass = bool(
        contracts["maximum_permutation_error"] <= config.equivariance_tolerance
        and contracts["maximum_rotation_equivariance_error"]
        <= config.equivariance_tolerance
        and contracts["maximum_unit_norm_error"] <= config.unit_norm_tolerance
    )
    if not contract_pass:
        raise ValueError("trained acquisition denoiser violates its group contract")
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "fingerprint": fingerprint,
        "implementation_sha256": implementation,
        "state_dict": best_state,
        "state_sha256": _state_digest(denoiser),
        "architecture": {
            "hidden": config.denoiser_hidden,
            "max_count": max(config.replicate_counts),
            "features": ["agreement", "resultant_length", "normalized_log_count"],
            "permutation_invariant": True,
            "so2_equivariant": True,
        },
        "training": {
            "objective": "noise2noise_one_minus_dot",
            "steps_run": steps_run,
            "selected_step": best_step,
            "selected_validation_loss": best_loss,
            "trace": trace,
        },
        "contracts": contracts,
        "contract_pass": contract_pass,
        "created_at": _utc_now(),
    }
    _write_torch(path, checkpoint)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    verifier = EquivariantReliabilityAggregator(
        hidden=config.denoiser_hidden,
        max_count=max(config.replicate_counts),
    )
    verifier.load_state_dict(reloaded["state_dict"])
    if _state_digest(verifier) != checkpoint["state_sha256"]:
        raise ValueError("saved acquisition denoiser failed reload")
    return denoiser, checkpoint, _sha256(path)


def _build_interventions(
    datasets: Mapping[str, calibrated.CalibratedDataset],
    noise: Mapping[str, torch.Tensor],
    denoiser: EquivariantReliabilityAggregator,
    config: RepeatedReferenceConfig,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, dict[int, calibrated.CalibratedDataset]]],
    dict[str, dict[str, dict[int, dict[str, float]]]],
    dict[str, calibrated.CalibratedDataset],
    dict[str, Any],
]:
    interventions = {rule: {} for rule in RULES}
    audits = {rule: {} for rule in RULES}
    shuffled: dict[str, calibrated.CalibratedDataset] = {}
    contracts: dict[str, Any] = {
        "maximum_unit_norm_error": 0.0,
        "maximum_pair_shared_error": 0.0,
        "maximum_m1_rule_difference": 0.0,
    }
    for split_index, (name, dataset) in enumerate(datasets.items()):
        inverse = noise[f"{name}__fiber_inverse"].long()
        errors = noise[f"{name}__errors"].double()
        for rule in RULES:
            interventions[rule][name] = {}
            audits[rule][name] = {}
        for count in config.replicate_counts:
            analytic_value, analytic_audit = circular_mean_orientation(
                dataset.calibration[:, :2],
                inverse,
                errors,
                config.measurement_sigma_radians,
                count,
            )
            learned_value, learned_audit = learned_mean_orientation(
                denoiser,
                dataset.calibration[:, :2],
                inverse,
                errors,
                config.measurement_sigma_radians,
                count,
                device,
            )
            values = {
                "analytic_circular_mean": analytic_value,
                "learned_equivariant": learned_value,
            }
            values_audit = {
                "analytic_circular_mean": analytic_audit,
                "learned_equivariant": learned_audit,
            }
            for rule in RULES:
                value = values[rule]
                interventions[rule][name][count] = apply_orientation(dataset, value)
                audits[rule][name][count] = values_audit[rule]
                contracts["maximum_unit_norm_error"] = max(
                    contracts["maximum_unit_norm_error"],
                    values_audit[rule]["maximum_unit_norm_error"],
                )
                fibers = dataset.paired.fiber.fiber_id
                base_angle = torch.atan2(
                    dataset.calibration[:, 1].double(),
                    dataset.calibration[:, 0].double(),
                )
                value_angle = torch.atan2(
                    value[:, 1].double(), value[:, 0].double()
                )
                angular_error = torch.atan2(
                    torch.sin(value_angle - base_angle),
                    torch.cos(value_angle - base_angle),
                )
                for fiber in torch.unique(fibers, sorted=True):
                    rows = angular_error[fibers == fiber]
                    contracts["maximum_pair_shared_error"] = max(
                        contracts["maximum_pair_shared_error"],
                        float((rows - rows[0]).abs().max()),
                    )
            if count == 1:
                contracts["maximum_m1_rule_difference"] = max(
                    contracts["maximum_m1_rule_difference"],
                    float((analytic_value - learned_value).abs().max()),
                )
        value = interventions["analytic_circular_mean"][name][
            max(config.replicate_counts)
        ].calibration[:, :2]
        unique_count = int(inverse.max()) + 1
        branch = dataset.paired.fiber.branch
        row_groups = []
        value_groups = []
        for index in range(unique_count):
            rows = torch.nonzero(inverse == index, as_tuple=False).flatten()
            if len(rows) != 2:
                raise ValueError("shuffle control requires two rows per fiber")
            rows = rows[torch.argsort(branch[rows])]
            row_groups.append(rows)
            value_groups.append(value[rows])
        grouped_value = torch.stack(value_groups)
        generator = torch.Generator(device="cpu").manual_seed(
            config.shuffle_seed + split_index * 1_000_003
        )
        permutation = torch.randperm(unique_count, generator=generator)
        shuffled_value = torch.empty_like(value)
        for target_index, rows in enumerate(row_groups):
            shuffled_value[rows] = grouped_value[permutation[target_index]]
        shuffled[name] = apply_orientation(dataset, shuffled_value)
        contracts[f"{name}_shuffle_permutation_sha256"] = (
            invariant._tensor_digest(permutation)
        )
    contracts["standard_error_law"] = standard_error_contract(
        audits["analytic_circular_mean"], config
    )
    return interventions, audits, shuffled, contracts


def _source_result(
    entry: Mapping[str, Any], arm: str, seed: int
) -> tuple[dict[str, Any], Path]:
    path = Path(entry["path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    source_level = next(
        (
            level
            for level in value.get("levels", [])
            if float(level.get("sigma_radians", -1)) == 0.175
        ),
        None,
    )
    if (
        _sha256(path) != entry.get("result_sha256")
        or value.get("schema_version") != orientation.SCHEMA_VERSION
        or value.get("hypothesis_id") != orientation.HYPOTHESIS_ID
        or value.get("condition") != arm
        or int(value.get("seed", -1)) != seed
        or value.get("scientific_fingerprint") != entry.get("scientific_fingerprint")
        or source_level is None
        or source_level.get("representation_gate") is not True
        or source_level.get("complete_gate") is not False
    ):
        raise ValueError(f"invalid source orientation result {path}")
    return value, path


def _fingerprint(
    config: RepeatedReferenceConfig,
    implementation: str,
    arm: str,
    seed: int,
    source_result_sha256: str,
    dataset_hashes: Mapping[str, str],
    noise_sha256: str,
    denoiser_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "arm": arm,
        "seed": seed,
        "source_result_sha256": source_result_sha256,
        "dataset_hashes": dict(dataset_hashes),
        "noise_sha256": noise_sha256,
        "denoiser_sha256": denoiser_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    path: Path, fingerprint: str, implementation: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("implementation_sha256") == implementation
    ):
        raise ValueError(f"incompatible completed acquisition result {path}")
    return value


def _campaign_reusable(
    value: Mapping[str, Any], implementation: str
) -> bool:
    artifacts = value.get("artifacts", {})
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("implementation_sha256") == implementation
        and all(
            Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["result_sha256"]
            for item in value.get("results", [])
        )
        and Path(artifacts.get("noise_arrays", "")).is_file()
        and _sha256(Path(artifacts["noise_arrays"]))
        == artifacts.get("noise_arrays_sha256")
        and Path(artifacts.get("denoiser", "")).is_file()
        and _sha256(Path(artifacts["denoiser"]))
        == artifacts.get("denoiser_sha256")
    )


def run_campaign(
    config: RepeatedReferenceConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    implementation_digests = _source_digests()
    implementation = _implementation_digest(implementation_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_reusable(existing, implementation):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    (
        source_campaign,
        source_path,
        source_entries,
        calibrated_campaign,
        task,
        source_config,
    ) = _load_source(config)
    del calibrated_campaign
    datasets = _datasets(task)
    dataset_hashes = {name: _dataset_hash(value) for name, value in datasets.items()}
    expected_hashes = {
        name: source_campaign["dataset_hashes"][name] for name in REGIMES
    }
    if dataset_hashes != expected_hashes:
        raise ValueError("regenerated acquisition cohorts do not replay source")
    source_noise_path = Path(source_campaign["artifacts"]["noise_arrays"])
    noise_path = output / "acquisition_errors.npz"
    noise, noise_sha256, noise_contracts = _ensure_noise_arrays(
        noise_path, datasets, source_noise_path, config
    )
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    denoiser_path = output / "equivariant_reference_denoiser.pt"
    denoiser, denoiser_record, denoiser_sha256 = _fit_or_load_denoiser(
        denoiser_path, config, implementation, device
    )
    denoiser_state_before = _state_digest(denoiser)
    interventions, acquisition_audits, shuffled, acquisition_contracts = (
        _build_interventions(datasets, noise, denoiser, config, device)
    )
    calibrated_root = Path(source_campaign["configuration"]["source_root"])
    analysis_config = replace(
        source_config, activation_batch_size=config.activation_batch_size
    )
    results: list[dict[str, Any]] = []
    reused = 0
    for arm in config.arms:
        for seed in config.seeds:
            cell_started = time.perf_counter()
            source_detail, source_result_path = _source_result(
                source_entries[(arm, seed)], arm, seed
            )
            system, provenance = orientation._load_system(
                calibrated_root, arm, seed, task, analysis_config, device
            )
            for parameter in system.parameters():
                parameter.requires_grad_(False)
            source_result_sha256 = _sha256(source_result_path)
            fingerprint = _fingerprint(
                config,
                implementation,
                arm,
                seed,
                source_result_sha256,
                dataset_hashes,
                noise_sha256,
                denoiser_sha256,
            )
            result_path = output / "runs" / arm / f"seed_{seed}" / "result.json"
            existing = _reusable_result(result_path, fingerprint, implementation)
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"resuming {arm} seed {seed}", flush=True)
                del system
                continue
            clean_metrics = {
                regime: calibrated.task_metrics(
                    system, datasets[regime], task, analysis_config, device
                )
                for regime in REGIMES
            }
            clean_source_level = next(
                level
                for level in source_detail["levels"]
                if float(level["sigma_radians"]) == 0.0
            )
            noisy_source_level = next(
                level
                for level in source_detail["levels"]
                if float(level["sigma_radians"]) == 0.175
            )
            clean_replay_error = max(
                abs(
                    float(clean_metrics[regime][metric])
                    - float(clean_source_level["task_metrics"][regime][metric])
                )
                for regime in REGIMES
                for metric in (
                    "exact_bin_accuracy",
                    "mean_circular_error_radians",
                    "mean_target_cross_entropy",
                )
            )
            rule_results: dict[str, Any] = {}
            for rule in RULES:
                levels = []
                pass_by_m = {}
                for count in config.replicate_counts:
                    metrics = {
                        regime: calibrated.task_metrics(
                            system,
                            interventions[rule][regime][count],
                            task,
                            analysis_config,
                            device,
                        )
                        for regime in REGIMES
                    }
                    passed, recovery = task_recovery_pass(
                        metrics, clean_metrics, config.accuracy_loss_ceiling
                    )
                    pass_by_m[replicate_key(count)] = passed
                    levels.append(
                        {
                            "replicate_count": count,
                            "task_metrics": metrics,
                            "task_recovery": recovery,
                            "acquisition_audit": {
                                regime: acquisition_audits[rule][regime][count]
                                for regime in REGIMES
                            },
                            "seed_gate": passed,
                        }
                    )
                passing = [
                    count
                    for count in config.replicate_counts
                    if pass_by_m[replicate_key(count)]
                ]
                rule_results[rule] = {
                    "levels": levels,
                    "pass_by_m": pass_by_m,
                    "minimum_passing_replicate_count": (
                        min(passing) if passing else None
                    ),
                }
            m1_metrics = rule_results["analytic_circular_mean"]["levels"][0][
                "task_metrics"
            ]
            m1_replay_error = max(
                abs(
                    float(m1_metrics[regime][metric])
                    - float(noisy_source_level["task_metrics"][regime][metric])
                )
                for regime in REGIMES
                for metric in (
                    "exact_bin_accuracy",
                    "mean_circular_error_radians",
                    "mean_target_cross_entropy",
                )
            )
            primary_count = max(config.replicate_counts)
            analytic_primary = next(
                level
                for level in rule_results["analytic_circular_mean"]["levels"]
                if level["replicate_count"] == primary_count
            )
            learned_primary = next(
                level
                for level in rule_results["learned_equivariant"]["levels"]
                if level["replicate_count"] == primary_count
            )
            learned_match = all(
                float(learned_primary["task_metrics"][regime]["exact_bin_accuracy"])
                >= float(
                    analytic_primary["task_metrics"][regime]["exact_bin_accuracy"]
                )
                - config.learned_match_accuracy_ceiling
                for regime in REGIMES
            )
            shuffled_metrics = {
                regime: calibrated.task_metrics(
                    system, shuffled[regime], task, analysis_config, device
                )
                for regime in REGIMES
            }
            shuffled_task_gate, shuffled_recovery = task_recovery_pass(
                shuffled_metrics, clean_metrics, config.accuracy_loss_ceiling
            )
            state_unchanged = bool(
                calibrated._state_digest(system.model)
                == provenance["model_state_sha256"]
                and calibrated._module_digest(system)
                == provenance["system_state_sha256"]
            )
            finite = _finite(clean_metrics) and _finite(rule_results) and _finite(
                shuffled_metrics
            )
            validity = bool(
                clean_replay_error <= config.replay_tolerance
                and m1_replay_error <= config.replay_tolerance
                and state_unchanged
                and finite
                and noise_contracts["first_replicate_source_identity"]
                and noise_contracts["prefix_contract"]
                and noise_contracts["independence_diagnostic_pass"]
                and acquisition_contracts["maximum_unit_norm_error"]
                <= config.unit_norm_tolerance
                and acquisition_contracts["maximum_pair_shared_error"]
                <= config.equivariance_tolerance
                and acquisition_contracts["maximum_m1_rule_difference"]
                <= config.unit_norm_tolerance
                and denoiser_record["contract_pass"]
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-repeated-reference-{arm}-seed{seed}",
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "condition": arm,
                "seed": seed,
                "configuration": asdict(config),
                "implementation_sha256": implementation,
                "scientific_fingerprint": fingerprint,
                "source_orientation_result": str(source_result_path),
                "source_orientation_result_sha256": source_result_sha256,
                "provenance": provenance,
                "dataset_hashes": dataset_hashes,
                "noise_sha256": noise_sha256,
                "denoiser_sha256": denoiser_sha256,
                "clean_task_metrics": clean_metrics,
                "clean_replay": {
                    "pass": clean_replay_error <= config.replay_tolerance,
                    "maximum_absolute_error": clean_replay_error,
                    "tolerance": config.replay_tolerance,
                },
                "single_observation_replay": {
                    "pass": m1_replay_error <= config.replay_tolerance,
                    "maximum_absolute_error": m1_replay_error,
                    "tolerance": config.replay_tolerance,
                },
                "rules": rule_results,
                "learned_matches_analytic_at_primary": learned_match,
                "controls": {
                    "exact_reference_oracle": {
                        "task_metrics": clean_metrics,
                        "seed_gate": True,
                    },
                    "fiber_shuffled_analytic_primary": {
                        "task_metrics": shuffled_metrics,
                        "task_recovery": shuffled_recovery,
                        "task_gate": shuffled_task_gate,
                    },
                },
                "gates": {
                    "clean_replay": clean_replay_error <= config.replay_tolerance,
                    "single_observation_replay": (
                        m1_replay_error <= config.replay_tolerance
                    ),
                    "source_representation_pass_task_fail": True,
                    "state_unchanged": state_unchanged,
                    "finite": finite,
                    "noise_contract": bool(
                        noise_contracts["first_replicate_source_identity"]
                        and noise_contracts["prefix_contract"]
                        and noise_contracts["independence_diagnostic_pass"]
                    ),
                    "acquisition_contract": bool(
                        acquisition_contracts["maximum_unit_norm_error"]
                        <= config.unit_norm_tolerance
                        and acquisition_contracts["maximum_pair_shared_error"]
                        <= config.equivariance_tolerance
                        and acquisition_contracts["maximum_m1_rule_difference"]
                        <= config.unit_norm_tolerance
                    ),
                    "denoiser_contract": denoiser_record["contract_pass"],
                    "validity": validity,
                },
                "analysis_seconds": time.perf_counter() - cell_started,
                "artifacts": {"result": str(result_path)},
            }
            _write_json(result_path, result)
            results.append(result)
            print(
                f"{arm} seed {seed}: "
                f"analytic={rule_results['analytic_circular_mean']['minimum_passing_replicate_count']} "
                f"learned={rule_results['learned_equivariant']['minimum_passing_replicate_count']} "
                f"valid={validity}",
                flush=True,
            )
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("acquisition implementation changed during run")

    denoiser_state_unchanged = _state_digest(denoiser) == denoiser_state_before
    valid = bool(
        all(result["gates"]["validity"] for result in results)
        and denoiser_state_unchanged
    )
    arms: dict[str, Any] = {}
    for arm in config.arms:
        selected = [result for result in results if result["condition"] == arm]
        rule_aggregates = {}
        for rule in RULES:
            counts = {
                replicate_key(count): sum(
                    int(result["rules"][rule]["pass_by_m"][replicate_key(count)])
                    for result in selected
                )
                for count in config.replicate_counts
            }
            passing = [
                count
                for count in config.replicate_counts
                if counts[replicate_key(count)] >= config.required_seed_passes
            ]
            rule_aggregates[rule] = {
                "pass_counts": counts,
                "minimum_population_passing_replicate_count": (
                    min(passing) if passing else None
                ),
                "primary_pass": counts[replicate_key(max(config.replicate_counts))]
                >= config.required_seed_passes,
            }
        arms[arm] = {
            "rules": rule_aggregates,
            "learned_match_pass_count": sum(
                int(result["learned_matches_analytic_at_primary"])
                for result in selected
            ),
            "learned_match_population_pass": sum(
                int(result["learned_matches_analytic_at_primary"])
                for result in selected
            )
            >= config.required_seed_passes,
            "oracle_pass_count": sum(
                int(result["controls"]["exact_reference_oracle"]["seed_gate"])
                for result in selected
            ),
            "shuffled_task_pass_count": sum(
                int(
                    result["controls"]["fiber_shuffled_analytic_primary"][
                        "task_gate"
                    ]
                )
                for result in selected
            ),
            "single_observation_pass_count": sum(
                int(
                    result["rules"]["analytic_circular_mean"]["pass_by_m"][
                        replicate_key(1)
                    ]
                )
                for result in selected
            ),
        }
    rate_pass = bool(acquisition_contracts["standard_error_law"]["pass"])
    controls_pass = all(
        arms[arm]["oracle_pass_count"] == len(config.seeds)
        and arms[arm]["shuffled_task_pass_count"] <= 1
        and arms[arm]["single_observation_pass_count"] < config.required_seed_passes
        for arm in config.arms
    )
    analytic_by_arm = {
        arm: bool(arms[arm]["rules"]["analytic_circular_mean"]["primary_pass"])
        for arm in config.arms
    }
    learned_by_arm = {
        arm: bool(arms[arm]["rules"]["learned_equivariant"]["primary_pass"])
        for arm in config.arms
    }
    learned_match_by_arm = {
        arm: bool(arms[arm]["learned_match_population_pass"])
        for arm in config.arms
    }
    if config.allow_underpowered:
        classification, primary = (
            "systems_lifecycle_only_not_quality_evidence",
            False,
        )
    else:
        classification, primary = classify_campaign(
            valid=valid,
            analytic_primary_by_arm=analytic_by_arm,
            learned_primary_by_arm=learned_by_arm,
            learned_match_by_arm=learned_match_by_arm,
            rate_pass=rate_pass,
            controls_pass=controls_pass,
        )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "source_digests": implementation_digests,
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
            "orientation_campaign": str(source_path),
            "orientation_campaign_sha256": SOURCE_ORIENTATION_CAMPAIGN_SHA256,
            "orientation_implementation_sha256": (
                SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
            ),
            "orientation_noise_sha256": SOURCE_ORIENTATION_NOISE_SHA256,
            "calibrated_campaign_sha256": SOURCE_CALIBRATED_CAMPAIGN_SHA256,
            "calibrated_implementation_sha256": (
                SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
            ),
            "motivating_readout_campaign_sha256": (
                MOTIVATING_READOUT_CAMPAIGN_SHA256
            ),
            "noise_sha256": noise_sha256,
            "denoiser_sha256": denoiser_sha256,
        },
        "dataset_hashes": dataset_hashes,
        "noise_contracts": noise_contracts,
        "acquisition_audits": acquisition_audits,
        "acquisition_contracts": acquisition_contracts,
        "denoiser": {
            key: value
            for key, value in denoiser_record.items()
            if key != "state_dict"
        },
        "summary": {
            "requested": len(config.arms) * len(config.seeds),
            "scheduled": len(config.arms) * len(config.seeds) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "fitted_probes": 0,
            "fitted_observers": 0,
            "fitted_acquisition_denoisers": 1,
        },
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary,
            "valid": valid,
            "arms": arms,
            "analytic_primary_by_arm": analytic_by_arm,
            "learned_primary_by_arm": learned_by_arm,
            "learned_match_by_arm": learned_match_by_arm,
            "rate_pass": rate_pass,
            "controls_pass": controls_pass,
            "denoiser_state_unchanged": denoiser_state_unchanged,
            "required_seed_passes": config.required_seed_passes,
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "condition": result["condition"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The intervention uses synthetic independent zero-mean Gaussian orientation observations.",
            "TinyLLM, front ends, task heads, probes, and observers remain frozen.",
            "One acquisition-only equivariant denoiser is fit without phase, cosine, bins, activations, logits, or checkpoint identity.",
            "The analytic circular mean and exact clean reference are positive controls, not learned task results.",
            "Five retained checkpoints do not establish architecture-population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "noise_arrays": str(noise_path),
            "noise_arrays_sha256": noise_sha256,
            "denoiser": str(denoiser_path),
            "denoiser_sha256": denoiser_sha256,
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_repeated_reference_acquisition/"
            "20260807_d8_corrective_expanded_v8"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--replicate-counts", type=_ints, default=REPLICATE_COUNTS)
    parser.add_argument("--denoiser-steps", type=int, default=1_200)
    parser.add_argument("--denoiser-validation-interval", type=int, default=50)
    parser.add_argument("--denoiser-patience", type=int, default=6)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = RepeatedReferenceConfig(
        seeds=args.seeds,
        replicate_counts=args.replicate_counts,
        required_seed_passes=len(args.seeds) if args.allow_underpowered else 4,
        denoiser_steps=args.denoiser_steps,
        denoiser_validation_interval=args.denoiser_validation_interval,
        denoiser_patience=args.denoiser_patience,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    result = run_campaign(config, args.output)
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
    print(result["artifacts"]["campaign"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
