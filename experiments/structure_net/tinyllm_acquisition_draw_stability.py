#!/usr/bin/env python3
"""Replicate TinyLLM repeated-reference recovery across fresh acquisition draws."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibration_orientation_noise as orientation
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
import experiments.structure_net.tinyllm_repeated_reference_acquisition as acquisition


SCHEMA_VERSION = "nal.tinyllm-acquisition-draw-stability.v1"
HYPOTHESIS_ID = "tinyllm-acquisition-draw-stability-v1"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-acquisition-draw-stability-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "9106b1632e91ecc499e9a36f539556b6dc9977e76119654a8ed0fc9e706590ac"
)
SOURCE_ORIENTATION_CAMPAIGN_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_CALIBRATED_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
PREDECESSOR_ARRAY_SHA256 = (
    "8886c1f6ad0fd307720748e44b1741edb656fdb9e29e913c7ad059b72397eef4",
    "895d398e4ca31611f04f92921e1169704eeba6292813de5b3015415dbd746fec",
)
ARMS = acquisition.ARMS
SEEDS = acquisition.SEEDS
REGIMES = acquisition.REGIMES
REPLICATE_COUNTS = (64, 256)
DRAW_COUNT = 16
DRAW_SEED_ROOT = 81_027_026
SHUFFLE_SEED = 81_027_113
REQUIRED_SEED_PASSES = 4
REQUIRED_PRIMARY_DRAW_PASSES = 15
M64_BROAD_STABILITY_MINIMUM = 12
M64_DRAW_SENSITIVITY_MINIMUM = 4


@dataclass(frozen=True)
class AcquisitionDrawStabilityConfig:
    source_orientation_root: str = (
        "data/experiments/tinyllm_calibration_orientation_noise/"
        "20260807_d8_preregistered"
    )
    arms: tuple[str, ...] = ARMS
    seeds: tuple[int, ...] = SEEDS
    replicate_counts: tuple[int, ...] = REPLICATE_COUNTS
    draw_count: int = DRAW_COUNT
    draw_seed_root: int = DRAW_SEED_ROOT
    shuffle_seed: int = SHUFFLE_SEED
    measurement_sigma_radians: float = 0.175
    accuracy_loss_ceiling: float = 0.03
    replay_tolerance: float = 2e-6
    unit_norm_tolerance: float = 1e-6
    pair_tolerance: float = 1e-6
    maximum_inter_draw_correlation: float = 0.05
    required_seed_passes: int = REQUIRED_SEED_PASSES
    required_primary_draw_passes: int = REQUIRED_PRIMARY_DRAW_PASSES
    m64_broad_stability_minimum: int = M64_BROAD_STABILITY_MINIMUM
    m64_draw_sensitivity_minimum: int = M64_DRAW_SENSITIVITY_MINIMUM
    activation_batch_size: int = 256
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.arms or set(self.arms).difference(ARMS):
            raise ValueError("unknown or empty system arm")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("checkpoint seeds must be non-empty and distinct")
        if self.replicate_counts != tuple(sorted(set(self.replicate_counts))):
            raise ValueError("replicate counts must be ordered and distinct")
        if not self.replicate_counts or min(self.replicate_counts) < 1:
            raise ValueError("replicate counts must be positive")
        if self.draw_count < 1:
            raise ValueError("draw count must be positive")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed passes is outside the selected population")
        if not 1 <= self.required_primary_draw_passes <= self.draw_count:
            raise ValueError("required draw passes is outside the draw population")
        if self.measurement_sigma_radians <= 0.0:
            raise ValueError("measurement sigma must be positive")
        if not 0.0 <= self.maximum_inter_draw_correlation < 1.0:
            raise ValueError("invalid inter-draw correlation ceiling")
        if not self.allow_underpowered:
            if self.arms != ARMS or self.seeds != SEEDS:
                raise ValueError("primary arms and checkpoint population are fixed")
            if self.replicate_counts != REPLICATE_COUNTS:
                raise ValueError("primary acquisition counts are fixed")
            if self.draw_count != DRAW_COUNT:
                raise ValueError("primary draw count is fixed")
            if self.draw_seed_root != DRAW_SEED_ROOT:
                raise ValueError("primary draw seed root is fixed")
            if self.shuffle_seed != SHUFFLE_SEED:
                raise ValueError("primary shuffle seed is fixed")
            if self.measurement_sigma_radians != 0.175:
                raise ValueError("primary measurement sigma is fixed")
            if self.required_seed_passes != REQUIRED_SEED_PASSES:
                raise ValueError("primary checkpoint gate is four of five")
            if self.required_primary_draw_passes != REQUIRED_PRIMARY_DRAW_PASSES:
                raise ValueError("primary draw gate is fifteen of sixteen")
            if self.m64_broad_stability_minimum != M64_BROAD_STABILITY_MINIMUM:
                raise ValueError("primary m64 broad-stability gate is fixed")
            if (
                self.m64_draw_sensitivity_minimum
                != M64_DRAW_SENSITIVITY_MINIMUM
            ):
                raise ValueError("primary m64 draw-sensitivity gate is fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _json_config(config: AcquisitionDrawStabilityConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "repeated_acquisition": Path(acquisition.__file__),
        "orientation_noise": Path(orientation.__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "preregistration": PREREGISTRATION_PATH,
    }
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError("acquisition-draw preregistration changed")
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    values = dict(digests or _source_digests())
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_role(config: AcquisitionDrawStabilityConfig) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    return "preregistered_frozen_system_acquisition_draw_replication"


def draw_key(draw_index: int) -> str:
    return f"draw_{draw_index:02d}"


def replicate_key(count: int) -> str:
    return f"m{count}"


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def classify_campaign(
    *,
    valid: bool,
    m64_complete_draw_passes: int,
    m256_complete_draw_passes: int,
    config: AcquisitionDrawStabilityConfig,
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    primary = (
        m256_complete_draw_passes >= config.required_primary_draw_passes
    )
    if not primary:
        return "m256_not_stable", False
    if m64_complete_draw_passes >= config.required_primary_draw_passes:
        return "m64_stable_population_ceiling", True
    if m64_complete_draw_passes >= config.m64_broad_stability_minimum:
        return "m64_broadly_stable_checkpoint_variable", True
    if m64_complete_draw_passes >= config.m64_draw_sensitivity_minimum:
        return "m64_draw_sensitive_m256_stable", True
    return "m64_unreliable_m256_stable", True


def _load_source(
    config: AcquisitionDrawStabilityConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[tuple[str, int], Mapping[str, Any]],
    Any,
    Any,
]:
    source_config = acquisition.RepeatedReferenceConfig(
        source_orientation_root=config.source_orientation_root,
        arms=config.arms,
        seeds=config.seeds,
        required_seed_passes=config.required_seed_passes,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    (
        source_campaign,
        source_path,
        source_entries,
        calibrated_campaign,
        task,
        calibrated_config,
    ) = acquisition._load_source(source_config)
    del calibrated_campaign
    if _sha256(source_path) != SOURCE_ORIENTATION_CAMPAIGN_SHA256:
        raise ValueError("orientation source campaign identity changed")
    calibrated_path = Path(source_campaign["configuration"]["source_root"]) / (
        "campaign_results.json"
    )
    if _sha256(calibrated_path) != SOURCE_CALIBRATED_CAMPAIGN_SHA256:
        raise ValueError("calibrated source campaign identity changed")
    return source_campaign, source_path, source_entries, task, calibrated_config


def _ensure_draw_arrays(
    path: Path,
    datasets: Mapping[str, calibrated.CalibratedDataset],
    config: AcquisitionDrawStabilityConfig,
) -> tuple[dict[str, torch.Tensor], str, dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    output: dict[str, torch.Tensor] = {}
    draw_sequences = np.random.SeedSequence(config.draw_seed_root).spawn(
        config.draw_count
    )
    split_sequences_by_draw = [
        sequence.spawn(len(REGIMES)) for sequence in draw_sequences
    ]
    maximum_correlation = 0.0
    all_draws_distinct = True
    maximum_pair_shared_error = 0.0
    stream_records: list[dict[str, Any]] = []
    for draw_index, draw_sequence in enumerate(draw_sequences):
        split_sequences = split_sequences_by_draw[draw_index]
        stream_records.append(
            {
                "draw_index": draw_index,
                "spawn_key": list(draw_sequence.spawn_key),
                "split_spawn_keys": [
                    list(sequence.spawn_key) for sequence in split_sequences
                ],
            }
        )
        for split_index, regime in enumerate(REGIMES):
            if draw_index > 0:
                continue
            dataset = datasets[regime]
            fibers = dataset.paired.fiber.fiber_id.long()
            unique, inverse = torch.unique(fibers, sorted=True, return_inverse=True)
            arrays[f"{regime}__unique_fibers"] = unique.numpy()
            arrays[f"{regime}__fiber_inverse"] = inverse.numpy()
            output[f"{regime}__fiber_inverse"] = inverse
            expanded_fibers = inverse.numpy()
            for fiber_index in range(len(unique)):
                rows = np.flatnonzero(expanded_fibers == fiber_index)
                if len(rows) < 1:
                    raise ValueError("empty acquisition fiber")
    for split_index, regime in enumerate(REGIMES):
        fiber_inverse = output[f"{regime}__fiber_inverse"]
        fiber_count = int(fiber_inverse.max()) + 1
        errors = np.empty(
            (config.draw_count, max(config.replicate_counts), fiber_count),
            dtype=np.float64,
        )
        for draw_index in range(config.draw_count):
            split_sequence = split_sequences_by_draw[draw_index][split_index]
            generator = np.random.default_rng(split_sequence)
            errors[draw_index] = generator.standard_normal(
                (max(config.replicate_counts), fiber_count)
            )
        flattened = errors.reshape(config.draw_count, -1)
        if config.draw_count > 1:
            correlations = np.corrcoef(flattened)
            off_diagonal = np.abs(
                correlations - np.eye(config.draw_count, dtype=np.float64)
            )
            maximum_correlation = max(maximum_correlation, float(off_diagonal.max()))
        distinct = len({value.tobytes() for value in errors}) == config.draw_count
        all_draws_distinct = bool(all_draws_distinct and distinct)
        expanded = errors[:, :, fiber_inverse.numpy()]
        for fiber_index in range(fiber_count):
            rows = torch.nonzero(
                fiber_inverse == fiber_index, as_tuple=False
            ).flatten().numpy()
            if len(rows) > 1:
                maximum_pair_shared_error = max(
                    maximum_pair_shared_error,
                    float(
                        np.abs(
                            expanded[:, :, rows]
                            - expanded[:, :, rows[:1]]
                        ).max()
                    ),
                )
        arrays[f"{regime}__errors"] = errors
        output[f"{regime}__errors"] = torch.from_numpy(errors)
    arrays["draw_seed_root"] = np.asarray([config.draw_seed_root], dtype=np.int64)
    arrays["draw_spawn_keys"] = np.asarray(
        [record["spawn_key"] for record in stream_records], dtype=np.int64
    )
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(archive[name], value)
                for name, value in arrays.items()
            ):
                raise ValueError(f"incompatible acquisition draw arrays {path}")
    else:
        _write_npz(path, arrays)
    array_sha256 = _sha256(path)
    contracts = {
        "draw_count": config.draw_count,
        "draw_seed_root": config.draw_seed_root,
        "stream_records": stream_records,
        "all_draws_distinct": all_draws_distinct,
        "maximum_absolute_inter_draw_correlation": maximum_correlation,
        "maximum_pair_shared_error": maximum_pair_shared_error,
        "nested_prefix_contract": True,
        "shared_across_arms_and_checkpoints": True,
        "predecessor_array_alias": array_sha256 in PREDECESSOR_ARRAY_SHA256,
    }
    contracts["pass"] = bool(
        all_draws_distinct
        and maximum_correlation <= config.maximum_inter_draw_correlation
        and maximum_pair_shared_error <= config.pair_tolerance
        and array_sha256 not in PREDECESSOR_ARRAY_SHA256
    )
    return output, array_sha256, contracts


def _pair_shared_angular_error(
    values: torch.Tensor,
    true_orientation: torch.Tensor,
    fiber_inverse: torch.Tensor,
) -> float:
    base_angle = torch.atan2(
        true_orientation[:, 1].double(), true_orientation[:, 0].double()
    )
    value_angle = torch.atan2(values[:, 1].double(), values[:, 0].double())
    angular_error = torch.atan2(
        torch.sin(value_angle - base_angle),
        torch.cos(value_angle - base_angle),
    )
    maximum = 0.0
    for fiber_index in range(int(fiber_inverse.max()) + 1):
        rows = torch.nonzero(
            fiber_inverse == fiber_index, as_tuple=False
        ).flatten()
        if len(rows) > 1:
            maximum = max(
                maximum,
                float((angular_error[rows] - angular_error[rows[:1]]).abs().max()),
            )
    return maximum


def _build_interventions(
    datasets: Mapping[str, calibrated.CalibratedDataset],
    noise: Mapping[str, torch.Tensor],
    config: AcquisitionDrawStabilityConfig,
) -> tuple[
    dict[int, dict[int, dict[str, calibrated.CalibratedDataset]]],
    dict[int, dict[int, dict[str, dict[str, float]]]],
    dict[str, calibrated.CalibratedDataset],
    dict[str, Any],
]:
    interventions: dict[
        int, dict[int, dict[str, calibrated.CalibratedDataset]]
    ] = {}
    audits: dict[int, dict[int, dict[str, dict[str, float]]]] = {}
    maximum_unit_norm_error = 0.0
    maximum_pair_shared_error = 0.0
    for draw_index in range(config.draw_count):
        interventions[draw_index] = {}
        audits[draw_index] = {}
        for count in config.replicate_counts:
            interventions[draw_index][count] = {}
            audits[draw_index][count] = {}
            for regime in REGIMES:
                dataset = datasets[regime]
                inverse = noise[f"{regime}__fiber_inverse"].long()
                errors = noise[f"{regime}__errors"][draw_index].double()
                orientation_value, audit = acquisition.circular_mean_orientation(
                    dataset.calibration[:, :2],
                    inverse,
                    errors,
                    config.measurement_sigma_radians,
                    count,
                )
                pair_error = _pair_shared_angular_error(
                    orientation_value, dataset.calibration[:, :2], inverse
                )
                maximum_pair_shared_error = max(
                    maximum_pair_shared_error, pair_error
                )
                maximum_unit_norm_error = max(
                    maximum_unit_norm_error,
                    float(audit["maximum_unit_norm_error"]),
                )
                audit = {**audit, "maximum_pair_shared_error": pair_error}
                interventions[draw_index][count][regime] = (
                    acquisition.apply_orientation(dataset, orientation_value)
                )
                audits[draw_index][count][regime] = audit

    shuffled: dict[str, calibrated.CalibratedDataset] = {}
    shuffle_records: dict[str, Any] = {}
    for split_index, regime in enumerate(REGIMES):
        dataset = datasets[regime]
        inverse = noise[f"{regime}__fiber_inverse"].long()
        errors = noise[f"{regime}__errors"][0].double()
        value, _ = acquisition.circular_mean_orientation(
            dataset.calibration[:, :2],
            inverse,
            errors,
            config.measurement_sigma_radians,
            max(config.replicate_counts),
        )
        fiber_count = int(inverse.max()) + 1
        branch = dataset.paired.fiber.branch
        row_groups = []
        value_groups = []
        for fiber_index in range(fiber_count):
            rows = torch.nonzero(
                inverse == fiber_index, as_tuple=False
            ).flatten()
            if len(rows) != 2:
                raise ValueError("shuffle control requires two rows per fiber")
            rows = rows[torch.argsort(branch[rows])]
            row_groups.append(rows)
            value_groups.append(value[rows])
        grouped_values = torch.stack(value_groups)
        generator = np.random.default_rng(config.shuffle_seed + split_index)
        shift = int(generator.integers(1, fiber_count))
        shuffled_groups = torch.roll(grouped_values, shifts=shift, dims=0)
        shuffled_value = torch.empty_like(value)
        for target_index, rows in enumerate(row_groups):
            shuffled_value[rows] = shuffled_groups[target_index]
        shuffled[regime] = acquisition.apply_orientation(dataset, shuffled_value)
        shuffle_records[regime] = {
            "fiber_count": fiber_count,
            "cyclic_shift": shift,
            "fixed_points": 0,
            "paired_sheet_groups_preserved": True,
        }
    contracts = {
        "maximum_unit_norm_error": maximum_unit_norm_error,
        "maximum_pair_shared_error": maximum_pair_shared_error,
        "shuffle": shuffle_records,
    }
    contracts["pass"] = bool(
        maximum_unit_norm_error <= config.unit_norm_tolerance
        and maximum_pair_shared_error <= config.pair_tolerance
        and all(
            record["fixed_points"] == 0
            and record["paired_sheet_groups_preserved"]
            for record in shuffle_records.values()
        )
    )
    return interventions, audits, shuffled, contracts


def _fingerprint(
    config: AcquisitionDrawStabilityConfig,
    implementation: str,
    arm: str,
    seed: int,
    source_result_sha256: str,
    dataset_hashes: Mapping[str, str],
    draw_arrays_sha256: str,
) -> str:
    value = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "arm": arm,
        "seed": seed,
        "source_result_sha256": source_result_sha256,
        "dataset_hashes": dict(dataset_hashes),
        "draw_arrays_sha256": draw_arrays_sha256,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    path: Path, fingerprint: str, implementation: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "completed"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("scientific_fingerprint") != fingerprint
        or value.get("implementation_sha256") != implementation
        or value.get("artifacts", {}).get("result") != str(path)
    ):
        raise ValueError(f"incompatible completed acquisition-draw result {path}")
    return value


def _campaign_reusable(
    campaign: Mapping[str, Any],
    config: AcquisitionDrawStabilityConfig,
    implementation: str,
) -> bool:
    artifacts = campaign.get("artifacts", {})
    array_path = Path(artifacts.get("draw_arrays", ""))
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("implementation_sha256") == implementation
        and campaign.get("configuration") == _json_config(config)
        and array_path.is_file()
        and _sha256(array_path) == artifacts.get("draw_arrays_sha256")
        and all(
            Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["result_sha256"]
            for item in campaign.get("results", [])
        )
    )


def _metric_replay_error(
    actual: Mapping[str, Mapping[str, float]],
    expected: Mapping[str, Mapping[str, float]],
) -> float:
    return max(
        abs(float(actual[regime][metric]) - float(expected[regime][metric]))
        for regime in REGIMES
        for metric in (
            "exact_bin_accuracy",
            "mean_circular_error_radians",
            "mean_target_cross_entropy",
        )
    )


def _aggregate_results(
    results: list[dict[str, Any]],
    config: AcquisitionDrawStabilityConfig,
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in config.arms:
        selected = [item for item in results if item["condition"] == arm]
        draw_records: dict[str, Any] = {}
        for draw_index in range(config.draw_count):
            counts: dict[str, Any] = {}
            for count in config.replicate_counts:
                seed_passes = sum(
                    int(
                        item["draws"][draw_key(draw_index)][replicate_key(count)][
                            "seed_gate"
                        ]
                    )
                    for item in selected
                )
                counts[replicate_key(count)] = {
                    "seed_passes": seed_passes,
                    "population_pass": seed_passes >= config.required_seed_passes,
                }
            draw_records[draw_key(draw_index)] = counts
        checkpoint_frequencies = {}
        for item in selected:
            seed = str(item["seed"])
            checkpoint_frequencies[seed] = {}
            for count in config.replicate_counts:
                successes = sum(
                    int(
                        item["draws"][draw_key(draw_index)][replicate_key(count)][
                            "seed_gate"
                        ]
                    )
                    for draw_index in range(config.draw_count)
                )
                low, high = wilson_interval(successes, config.draw_count)
                checkpoint_frequencies[seed][replicate_key(count)] = {
                    "passes": successes,
                    "draws": config.draw_count,
                    "frequency": successes / config.draw_count,
                    "wilson_95": [low, high],
                }
        inherited_negative = sum(
            int(item["inherited_single_observation"]["seed_gate"])
            for item in selected
        )
        shuffled_passes = sum(
            int(item["controls"]["fiber_shuffled_draw0_m256"]["seed_gate"])
            for item in selected
        )
        arms[arm] = {
            "draws": draw_records,
            "checkpoint_frequencies": checkpoint_frequencies,
            "inherited_single_observation_passes": inherited_negative,
            "fiber_shuffled_draw0_m256_passes": shuffled_passes,
        }
    count_aggregates: dict[str, Any] = {}
    for count in config.replicate_counts:
        complete = []
        per_arm = {arm: 0 for arm in config.arms}
        for draw_index in range(config.draw_count):
            arm_passes = {
                arm: bool(
                    arms[arm]["draws"][draw_key(draw_index)][replicate_key(count)][
                        "population_pass"
                    ]
                )
                for arm in config.arms
            }
            for arm, passed in arm_passes.items():
                per_arm[arm] += int(passed)
            complete.append(all(arm_passes.values()))
        complete_passes = sum(int(value) for value in complete)
        low, high = wilson_interval(complete_passes, config.draw_count)
        count_aggregates[replicate_key(count)] = {
            "complete_draw_pass_vector": complete,
            "complete_draw_passes": complete_passes,
            "draws": config.draw_count,
            "frequency": complete_passes / config.draw_count,
            "wilson_95": [low, high],
            "arm_population_draw_passes": per_arm,
        }
    return {"arms": arms, "counts": count_aggregates}


def run_campaign(
    config: AcquisitionDrawStabilityConfig, output: Path
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

    source_campaign, source_path, source_entries, task, source_config = _load_source(
        config
    )
    datasets = acquisition._datasets(task)
    dataset_hashes = {
        regime: acquisition._dataset_hash(dataset)
        for regime, dataset in datasets.items()
    }
    expected_hashes = {
        regime: source_campaign["dataset_hashes"][regime] for regime in REGIMES
    }
    if dataset_hashes != expected_hashes:
        raise ValueError("regenerated acquisition cohorts do not replay source")

    draw_array_path = output / "acquisition_draw_errors.npz"
    noise, draw_arrays_sha256, noise_contracts = _ensure_draw_arrays(
        draw_array_path, datasets, config
    )
    interventions, acquisition_audits, shuffled, acquisition_contracts = (
        _build_interventions(datasets, noise, config)
    )
    if not noise_contracts["pass"] or not acquisition_contracts["pass"]:
        raise ValueError("fresh acquisition arrays failed pre-outcome contracts")

    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    calibrated_root = Path(source_campaign["configuration"]["source_root"])
    analysis_config = replace(
        source_config, activation_batch_size=config.activation_batch_size
    )
    results: list[dict[str, Any]] = []
    reused = 0
    for arm in config.arms:
        for seed in config.seeds:
            cell_started = time.perf_counter()
            source_detail, source_result_path = acquisition._source_result(
                source_entries[(arm, seed)], arm, seed
            )
            source_result_sha256 = _sha256(source_result_path)
            fingerprint = _fingerprint(
                config,
                implementation,
                arm,
                seed,
                source_result_sha256,
                dataset_hashes,
                draw_arrays_sha256,
            )
            result_path = output / "runs" / arm / f"seed_{seed}" / "result.json"
            existing = _reusable_result(result_path, fingerprint, implementation)
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"resuming {arm} seed {seed}", flush=True)
                continue

            system, provenance = orientation._load_system(
                calibrated_root, arm, seed, task, analysis_config, device
            )
            for parameter in system.parameters():
                parameter.requires_grad_(False)
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
            clean_replay_error = _metric_replay_error(
                clean_metrics, clean_source_level["task_metrics"]
            )
            inherited_single_pass, inherited_single_recovery = (
                acquisition.task_recovery_pass(
                    noisy_source_level["task_metrics"],
                    clean_source_level["task_metrics"],
                    config.accuracy_loss_ceiling,
                )
            )

            draw_results: dict[str, Any] = {}
            for draw_index in range(config.draw_count):
                levels: dict[str, Any] = {}
                for count in config.replicate_counts:
                    metrics = {
                        regime: calibrated.task_metrics(
                            system,
                            interventions[draw_index][count][regime],
                            task,
                            analysis_config,
                            device,
                        )
                        for regime in REGIMES
                    }
                    passed, recovery = acquisition.task_recovery_pass(
                        metrics, clean_metrics, config.accuracy_loss_ceiling
                    )
                    levels[replicate_key(count)] = {
                        "replicate_count": count,
                        "task_metrics": metrics,
                        "task_recovery": recovery,
                        "acquisition_audit": acquisition_audits[draw_index][count],
                        "seed_gate": passed,
                    }
                draw_results[draw_key(draw_index)] = levels

            shuffled_metrics = {
                regime: calibrated.task_metrics(
                    system, shuffled[regime], task, analysis_config, device
                )
                for regime in REGIMES
            }
            shuffled_pass, shuffled_recovery = acquisition.task_recovery_pass(
                shuffled_metrics, clean_metrics, config.accuracy_loss_ceiling
            )
            state_unchanged = bool(
                calibrated._state_digest(system.model)
                == provenance["model_state_sha256"]
                and calibrated._module_digest(system)
                == provenance["system_state_sha256"]
            )
            finite = _finite(clean_metrics) and _finite(draw_results) and _finite(
                shuffled_metrics
            )
            validity = bool(
                clean_replay_error <= config.replay_tolerance
                and state_unchanged
                and finite
                and noise_contracts["pass"]
                and acquisition_contracts["pass"]
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-acquisition-draw-{arm}-seed{seed}",
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "condition": arm,
                "seed": seed,
                "configuration": _json_config(config),
                "implementation_sha256": implementation,
                "scientific_fingerprint": fingerprint,
                "source_orientation_result": str(source_result_path),
                "source_orientation_result_sha256": source_result_sha256,
                "provenance": provenance,
                "dataset_hashes": dataset_hashes,
                "draw_arrays_sha256": draw_arrays_sha256,
                "clean_task_metrics": clean_metrics,
                "clean_replay": {
                    "pass": clean_replay_error <= config.replay_tolerance,
                    "maximum_absolute_error": clean_replay_error,
                    "tolerance": config.replay_tolerance,
                },
                "inherited_single_observation": {
                    "task_metrics": noisy_source_level["task_metrics"],
                    "task_recovery": inherited_single_recovery,
                    "seed_gate": inherited_single_pass,
                },
                "draws": draw_results,
                "controls": {
                    "exact_reference_oracle": {
                        "task_metrics": clean_metrics,
                        "seed_gate": True,
                    },
                    "fiber_shuffled_draw0_m256": {
                        "task_metrics": shuffled_metrics,
                        "task_recovery": shuffled_recovery,
                        "seed_gate": shuffled_pass,
                    },
                },
                "gates": {
                    "clean_replay": clean_replay_error
                    <= config.replay_tolerance,
                    "state_unchanged": state_unchanged,
                    "finite": finite,
                    "noise_contract": noise_contracts["pass"],
                    "acquisition_contract": acquisition_contracts["pass"],
                    "validity": validity,
                },
                "analysis_seconds": time.perf_counter() - cell_started,
                "artifacts": {"result": str(result_path)},
            }
            _write_json(result_path, result)
            results.append(result)
            m64_passes = sum(
                int(
                    draw_results[draw_key(index)][replicate_key(64)]["seed_gate"]
                )
                for index in range(config.draw_count)
            )
            m256_passes = sum(
                int(
                    draw_results[draw_key(index)][replicate_key(256)]["seed_gate"]
                )
                for index in range(config.draw_count)
            )
            print(
                f"{arm} seed {seed}: m64={m64_passes}/{config.draw_count} "
                f"m256={m256_passes}/{config.draw_count} valid={validity}",
                flush=True,
            )
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("acquisition-draw implementation changed during run")

    aggregate_detail = _aggregate_results(results, config)
    controls_pass = all(
        aggregate_detail["arms"][arm]["inherited_single_observation_passes"]
        < config.required_seed_passes
        and aggregate_detail["arms"][arm][
            "fiber_shuffled_draw0_m256_passes"
        ]
        <= 1
        for arm in config.arms
    )
    valid = bool(
        all(result["gates"]["validity"] for result in results)
        and controls_pass
    )
    m64_complete = aggregate_detail["counts"]["m64"][
        "complete_draw_passes"
    ]
    m256_complete = aggregate_detail["counts"]["m256"][
        "complete_draw_passes"
    ]
    if config.allow_underpowered:
        classification, primary = (
            "systems_lifecycle_only_not_quality_evidence",
            False,
        )
    else:
        classification, primary = classify_campaign(
            valid=valid,
            m64_complete_draw_passes=m64_complete,
            m256_complete_draw_passes=m256_complete,
            config=config,
        )

    result_entries = [
        {
            "experiment_id": result["experiment_id"],
            "condition": result["condition"],
            "seed": result["seed"],
            "scientific_fingerprint": result["scientific_fingerprint"],
            "path": result["artifacts"]["result"],
            "result_sha256": _sha256(Path(result["artifacts"]["result"])),
        }
        for result in results
    ]
    result_manifest_sha256 = hashlib.sha256(
        json.dumps(result_entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
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
            "calibrated_campaign_sha256": SOURCE_CALIBRATED_CAMPAIGN_SHA256,
            "predecessor_array_sha256": list(PREDECESSOR_ARRAY_SHA256),
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "draw_arrays_sha256": draw_arrays_sha256,
        },
        "dataset_hashes": dataset_hashes,
        "noise_contracts": noise_contracts,
        "acquisition_contracts": acquisition_contracts,
        "acquisition_audits": {
            draw_key(draw): {
                replicate_key(count): acquisition_audits[draw][count]
                for count in config.replicate_counts
            }
            for draw in range(config.draw_count)
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
            "fitted_acquisition_parameters": 0,
        },
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary,
            "valid": valid,
            "controls_pass": controls_pass,
            "required_seed_passes": config.required_seed_passes,
            "required_primary_draw_passes": config.required_primary_draw_passes,
            **aggregate_detail,
        },
        "results": result_entries,
        "result_manifest_sha256": result_manifest_sha256,
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The replication unit is a fresh complete acquisition draw; checkpoints are fixed systems within each draw.",
            "Only the analytic circular mean at m=64 and m=256 is evaluated.",
            "TinyLLM, structured front ends, task heads, probes, observers, and acquisition rules remain frozen.",
            "The intervention assumes independent unbiased homoscedastic Gaussian orientation errors.",
            "Five retained checkpoints do not establish architecture-population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "draw_arrays": str(draw_array_path),
            "draw_arrays_sha256": draw_arrays_sha256,
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_acquisition_draw_stability/"
            "20260810_d16_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--arms", type=_strings, default=ARMS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--replicate-counts", type=_ints, default=REPLICATE_COUNTS)
    parser.add_argument("--draw-count", type=int, default=DRAW_COUNT)
    parser.add_argument("--draw-seed-root", type=int, default=DRAW_SEED_ROOT)
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument(
        "--required-seed-passes", type=int, default=REQUIRED_SEED_PASSES
    )
    parser.add_argument(
        "--required-primary-draw-passes",
        type=int,
        default=REQUIRED_PRIMARY_DRAW_PASSES,
    )
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = AcquisitionDrawStabilityConfig(
        arms=args.arms,
        seeds=args.seeds,
        replicate_counts=args.replicate_counts,
        draw_count=args.draw_count,
        draw_seed_root=args.draw_seed_root,
        shuffle_seed=args.shuffle_seed,
        required_seed_passes=args.required_seed_passes,
        required_primary_draw_passes=args.required_primary_draw_passes,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
