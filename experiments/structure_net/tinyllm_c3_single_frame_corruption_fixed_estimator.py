#!/usr/bin/env python3
"""Test a fixed robust C3 estimator under one unmarked frame substitution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_constant_acceleration_fixed_operator as acceleration
import experiments.structure_net.tinyllm_c3_temporal_fixed_operator_ceiling as fixed
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as source


SCHEMA_VERSION = "nal.tinyllm-c3-single-frame-corruption-fixed-estimator.v1"
HYPOTHESIS_ID = "tinyllm-c3-single-frame-corruption-fixed-estimator-v1"
EVIDENCE_ROLE = "prospective_no_training_observation_scope_preflight"
SEEDS = acceleration.SEEDS
REGIMES = acceleration.REGIMES
ARMS = (
    "clean_all_frame_degree2",
    "corrupted_all_frame_degree2",
    "oracle_drop_one_quadratic",
    "robust_drop_one_quadratic",
)
SAMPLE_COUNT = acceleration.SAMPLE_COUNT
DONOR_SEED_BASES = {"composition": 831_107, "extrapolation": 833_107}
FRAME_SEED_BASES = {"composition": 841_107, "extrapolation": 843_107}
SHUFFLE_SEED_BASES = {"composition": 851_107, "extrapolation": 853_107}
MINIMUM_FRAME_COUNT = 400
ACTION_ERROR_MAXIMUM = 2e-12
CONTINUOUS_ERROR_MAXIMUM = 1e-10
CHART_MARGIN_MINIMUM = 0.20
SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM = 0.10
SHUFFLED_RMSE_MINIMUM = 0.80
FIXED_CEILING_RMSE_MAXIMUM = acceleration.FIXED_CEILING_RMSE_MAXIMUM
FIXED_CEILING_ACCURACY_MINIMUM = acceleration.FIXED_CEILING_ACCURACY_MINIMUM
REPAIR_RMSE_RATIO_MAXIMUM = 0.50
REPAIR_ACCURACY_DELTA_MINIMUM = 0.20
REPAIR_CROSS_ENTROPY_DELTA_MAXIMUM = -0.10
ORACLE_RMSE_SLACK_MAXIMUM = 0.002
ORACLE_ACCURACY_SLACK_MAXIMUM = 0.01
ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM = 0.005
REQUIRED_SEED_PASSES = 4
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-single-frame-corruption-fixed-estimator-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "29b47e522fc46d5993cfba811eeba2d6bd2cc8eca189a1f6205c422f159b572e"
)
ACCELERATION_RUNNER_SHA256 = (
    "6ea952f386b82b12355c3aa2e9552af6bf73e03e7cd47310fec764ce49d0d5e2"
)
ACCELERATION_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_constant_acceleration_fixed_operator/"
    "20260811_preregistered/result.json"
)
ACCELERATION_RESULT_SHA256 = (
    "b04a5574efc658ec1ed73f70fa494041ad16c0ae1342423cdde32925c1c7bc53"
)
GENERATOR_SHA256 = "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
INTERVAL_SHA256 = "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_single_frame_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
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


def validate_sources() -> tuple[dict[str, str], dict[tuple[int, str], str]]:
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "constant_acceleration_runner": _sha256(Path(acceleration.__file__)),
        "constant_acceleration_result": _sha256(ACCELERATION_RESULT_PATH),
        "generator": _sha256(Path(source.__file__)),
        "interval_likelihood": _sha256(Path(acceleration.joint.__file__)),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "constant_acceleration_runner": ACCELERATION_RUNNER_SHA256,
        "constant_acceleration_result": ACCELERATION_RESULT_SHA256,
        "generator": GENERATOR_SHA256,
        "interval_likelihood": INTERVAL_SHA256,
    }
    predecessor = json.loads(
        ACCELERATION_RESULT_PATH.read_text(encoding="utf-8")
    )
    if (
        hashes != expected
        or predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != "fixed_all_frame_degree2_closes_constant_acceleration"
    ):
        raise RuntimeError("single-frame-corruption source contract changed")
    cohort_hashes = {
        (int(cell["seed"]), str(cell["regime"])): str(cell["dataset_sha256"])
        for cell in predecessor["cells"]
    }
    expected_cells = {(seed, regime) for seed in SEEDS for regime in REGIMES}
    if set(cohort_hashes) != expected_cells:
        raise RuntimeError("predecessor cohort population changed")
    return hashes, cohort_hashes


def corruption_plan(
    regime: str,
    replicate_seed: int,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    donor = fixed.sattolo_derangement(
        sample_count, DONOR_SEED_BASES[regime] + replicate_seed
    )
    generator = torch.Generator(device="cpu").manual_seed(
        FRAME_SEED_BASES[regime] + replicate_seed
    )
    frame = torch.randint(
        0,
        source.TIME_STEPS,
        (sample_count,),
        dtype=torch.int64,
        generator=generator,
    )
    return donor, frame


def corrupt_frames(
    values: torch.Tensor,
    donor: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    if values.ndim not in (2, 3) or values.shape[0] != donor.numel():
        raise ValueError("corruption tensor and plan are incompatible")
    corrupted = values.clone()
    row = torch.arange(values.shape[0], dtype=torch.int64)
    corrupted[row, frame] = values[donor, frame]
    return corrupted


def _principal_increment(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def phase_fit_candidates(
    carrier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return time-8 prediction and retained-fit residual for every deletion."""

    if carrier.ndim != 2 or carrier.shape[1] != source.TIME_STEPS:
        raise ValueError("expected one eight-frame carrier per example")
    predictions = []
    residuals = []
    evaluation = torch.tensor([1.0, 8.0, 28.0], dtype=torch.float64)
    all_times = torch.arange(source.TIME_STEPS, dtype=torch.float64)
    for omitted in range(source.TIME_STEPS):
        keep = torch.arange(source.TIME_STEPS) != omitted
        times = all_times[keep]
        phase = torch.angle(carrier[:, keep]).double()
        increments = _principal_increment(phase[:, 1:] - phase[:, :-1])
        unwrapped = torch.cat(
            (phase[:, :1], phase[:, :1] + torch.cumsum(increments, dim=1)),
            dim=1,
        )
        design = torch.stack(
            (torch.ones_like(times), times, 0.5 * times * (times - 1.0)),
            dim=1,
        )
        pseudoinverse = torch.linalg.pinv(design)
        coefficients = unwrapped @ pseudoinverse.T
        fitted = coefficients @ design.T
        residuals.append((unwrapped - fitted).square().mean(dim=1))
        predicted_phase = coefficients @ evaluation
        predictions.append(
            torch.polar(torch.ones_like(predicted_phase), predicted_phase).to(
                torch.complex128
            )
        )
    return torch.stack(predictions, dim=1), torch.stack(residuals, dim=1)


def deletion_predictions(
    carrier: torch.Tensor,
    true_frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates, residuals = phase_fit_candidates(carrier)
    selected = residuals.argmin(dim=1)
    row = torch.arange(carrier.shape[0], dtype=torch.int64)
    return (
        candidates[row, true_frame],
        candidates[row, selected],
        selected,
        residuals,
    )


def operator_complex_predictions(
    clean_carrier: torch.Tensor,
    corrupted_carrier: torch.Tensor,
    true_frame: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    oracle, robust, selected, residuals = deletion_predictions(
        corrupted_carrier, true_frame
    )
    return (
        {
            "clean_all_frame_degree2": acceleration.operator_complex_predictions(
                clean_carrier
            )["all_frame_degree2"],
            "corrupted_all_frame_degree2": acceleration.operator_complex_predictions(
                corrupted_carrier
            )["all_frame_degree2"],
            "oracle_drop_one_quadratic": oracle,
            "robust_drop_one_quadratic": robust,
        },
        selected,
        residuals,
    )


def _chart_margin(
    clean_carrier: torch.Tensor,
    true_frame: torch.Tensor,
) -> float:
    minimum = math.inf
    for omitted in range(source.TIME_STEPS):
        rows = true_frame == omitted
        if not bool(rows.any()):
            continue
        keep = torch.arange(source.TIME_STEPS) != omitted
        phase = torch.angle(clean_carrier[rows][:, keep]).double()
        increments = _principal_increment(phase[:, 1:] - phase[:, :-1])
        minimum = min(minimum, float(math.pi - increments.abs().max()))
    return minimum


def _arm_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shuffled_target: torch.Tensor,
    regime: str,
    action_errors: Mapping[str, float],
) -> dict[str, Any]:
    scalar_prediction = prediction.real
    scalar = acceleration._scalar_metrics(scalar_prediction, target)
    task = acceleration._task_metrics(scalar_prediction, target)
    shuffled_scalar = acceleration._scalar_metrics(
        scalar_prediction, shuffled_target
    )
    shuffled_task = acceleration._task_metrics(
        scalar_prediction, shuffled_target
    )
    maximum_action_error = max(action_errors.values())
    return {
        "scalar": scalar,
        "task": task,
        "task_pass": acceleration._task_gate(task, regime),
        "fixed_ceiling_pass": acceleration._fixed_ceiling_gate(
            scalar, task, regime
        ),
        "deck_action_maximum_errors": dict(action_errors),
        "maximum_deck_action_error": maximum_action_error,
        "action_pass": maximum_action_error <= ACTION_ERROR_MAXIMUM,
        "shuffled_target_scalar": shuffled_scalar,
        "shuffled_target_task": shuffled_task,
        "shuffled_task_pass": acceleration._task_gate(shuffled_task, regime),
        "shuffle_pass": bool(
            abs(shuffled_scalar["correlation"])
            <= SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM
            and shuffled_scalar["rmse"] >= SHUFFLED_RMSE_MINIMUM
            and not acceleration._task_gate(shuffled_task, regime)
        ),
    }


def analyze_cell(
    regime: str,
    replicate_seed: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
    expected_base_hash: str | None = None,
) -> dict[str, Any]:
    registered = replicate_seed in SEEDS and sample_count == SAMPLE_COUNT
    pilot = allow_pilot and replicate_seed == 991 and sample_count == 64
    if regime not in REGIMES or not (registered or pilot):
        raise ValueError(f"unknown corruption cohort {regime}/{replicate_seed}")
    dataset = acceleration.generate_dataset(
        regime,
        replicate_seed,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_hash = acceleration.dataset_hash(dataset)
    base_hash_replayed = expected_base_hash is None or base_hash == expected_base_hash
    donor, frame = corruption_plan(regime, replicate_seed, sample_count)
    corrupted_tokens = corrupt_frames(dataset.tokens, donor, frame)
    corruption_hash = _tensor_hash(corrupted_tokens, donor, frame)
    replay_donor, replay_frame = corruption_plan(regime, replicate_seed, sample_count)
    replay_corrupted = corrupt_frames(dataset.tokens, replay_donor, replay_frame)
    corruption_deterministic = bool(
        torch.equal(donor, replay_donor)
        and torch.equal(frame, replay_frame)
        and torch.equal(corrupted_tokens, replay_corrupted)
    )
    frame_counts = torch.bincount(frame, minlength=source.TIME_STEPS)
    required_frame_count = MINIMUM_FRAME_COUNT if registered else 0

    clean_carrier, clean_magnitude = source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    corrupted_carrier, corrupted_magnitude = source.analytic_carrier(
        corrupted_tokens, dataset.calibration
    )
    complex_predictions, selected, residuals = operator_complex_predictions(
        clean_carrier, corrupted_carrier, frame
    )

    action_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    corruption_equivariance_errors: dict[str, int] = {}
    for element in (1, 2):
        transformed_clean_tokens = source.apply_deck_action(dataset.tokens, element)
        transformed_corrupted_tokens = corrupt_frames(
            transformed_clean_tokens, donor, frame
        )
        expected_transformed_corruption = source.apply_deck_action(
            corrupted_tokens, element
        )
        corruption_equivariance_errors[str(element)] = int(
            (transformed_corrupted_tokens != expected_transformed_corruption).sum()
        )
        transformed_clean_carrier, _ = source.analytic_carrier(
            transformed_clean_tokens, dataset.calibration
        )
        transformed_corrupted_carrier, _ = source.analytic_carrier(
            transformed_corrupted_tokens, dataset.calibration
        )
        transformed_predictions, _, _ = operator_complex_predictions(
            transformed_clean_carrier, transformed_corrupted_carrier, frame
        )
        for arm in ARMS:
            action_errors[arm][str(element)] = float(
                (transformed_predictions[arm] - complex_predictions[arm])
                .abs()
                .max()
            )

    exact_clean = acceleration.continuous_carrier(dataset)
    exact_future = acceleration.continuous_carrier(
        dataset, include_future=True
    )[:, -1]
    exact_corrupted = corrupt_frames(exact_clean, donor, frame)
    continuous_oracle, continuous_robust, continuous_selected, _ = (
        deletion_predictions(exact_corrupted, frame)
    )
    continuous_errors = {
        "oracle_drop_one_quadratic": float(
            (continuous_oracle - exact_future).abs().max()
        ),
        "robust_drop_one_quadratic": float(
            (continuous_robust - exact_future).abs().max()
        ),
    }
    continuous_index_recovery_count = int((continuous_selected == frame).sum())
    chart_margin = _chart_margin(exact_clean, frame)

    shuffle = fixed.sattolo_derangement(
        sample_count, SHUFFLE_SEED_BASES[regime] + replicate_seed
    )
    shuffled_target = dataset.target[shuffle]
    operators = {
        arm: _arm_metrics(
            complex_predictions[arm],
            dataset.target,
            shuffled_target,
            regime,
            action_errors[arm],
        )
        for arm in ARMS
    }
    naive = operators["corrupted_all_frame_degree2"]
    robust = operators["robust_drop_one_quadratic"]
    oracle = operators["oracle_drop_one_quadratic"]
    clean = operators["clean_all_frame_degree2"]
    repair = {
        "rmse_ratio": (
            robust["scalar"]["rmse"] / max(naive["scalar"]["rmse"], 1e-18)
        ),
        "accuracy_delta": (
            robust["task"]["exact_bin_accuracy"]
            - naive["task"]["exact_bin_accuracy"]
        ),
        "cross_entropy_delta": (
            robust["task"]["target_cross_entropy"]
            - naive["task"]["target_cross_entropy"]
        ),
    }
    repair["pass"] = bool(
        robust["fixed_ceiling_pass"]
        and repair["rmse_ratio"] <= REPAIR_RMSE_RATIO_MAXIMUM
        and repair["accuracy_delta"] >= REPAIR_ACCURACY_DELTA_MINIMUM
        and repair["cross_entropy_delta"]
        <= REPAIR_CROSS_ENTROPY_DELTA_MAXIMUM
    )
    oracle_fidelity = {
        "rmse_excess": robust["scalar"]["rmse"] - oracle["scalar"]["rmse"],
        "accuracy_delta": (
            robust["task"]["exact_bin_accuracy"]
            - oracle["task"]["exact_bin_accuracy"]
        ),
        "cross_entropy_excess": (
            robust["task"]["target_cross_entropy"]
            - oracle["task"]["target_cross_entropy"]
        ),
    }
    oracle_fidelity["pass"] = bool(
        oracle_fidelity["rmse_excess"] <= ORACLE_RMSE_SLACK_MAXIMUM
        and oracle_fidelity["accuracy_delta"] >= -ORACLE_ACCURACY_SLACK_MAXIMUM
        and oracle_fidelity["cross_entropy_excess"]
        <= ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
    )
    base_contracts = acceleration._group_contracts(dataset)
    donor_fixed_points = int(
        (donor == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    shuffle_fixed_points = int(
        (shuffle == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    validity = bool(
        dataset.saturation_count == 0
        and base_hash_replayed
        and base_contracts["pass"]
        and corruption_deterministic
        and donor_fixed_points == 0
        and int(frame_counts.min()) >= required_frame_count
        and sum(corruption_equivariance_errors.values()) == 0
        and all(operator["action_pass"] for operator in operators.values())
        and (
            all(operator["shuffle_pass"] for operator in operators.values())
            if registered
            else True
        )
        and continuous_errors["oracle_drop_one_quadratic"]
        <= CONTINUOUS_ERROR_MAXIMUM
        and continuous_errors["robust_drop_one_quadratic"]
        <= CONTINUOUS_ERROR_MAXIMUM
        and continuous_index_recovery_count == sample_count
        and chart_margin >= CHART_MARGIN_MINIMUM
        and shuffle_fixed_points == 0
        and float(clean_magnitude.min()) > 1e-12
        and float(corrupted_magnitude.min()) > 1e-12
        and _finite(operators)
        and _finite(repair)
        and _finite(oracle_fidelity)
        and _finite(residuals)
    )
    return {
        "seed": replicate_seed,
        "regime": regime,
        "dataset_seed": dataset.dataset_seed,
        "base_dataset_sha256": base_hash,
        "base_dataset_hash_replayed": base_hash_replayed,
        "corruption_sha256": corruption_hash,
        "corruption_deterministic": corruption_deterministic,
        "sample_count": sample_count,
        "saturation_count": dataset.saturation_count,
        "donor_fixed_points": donor_fixed_points,
        "frame_counts": [int(value) for value in frame_counts],
        "minimum_frame_count": int(frame_counts.min()),
        "minimum_clean_chart_margin": chart_margin,
        "minimum_clean_character_magnitude": float(clean_magnitude.min()),
        "minimum_corrupted_character_magnitude": float(
            corrupted_magnitude.min()
        ),
        "base_group_contracts": base_contracts,
        "corruption_equivariance_token_errors": corruption_equivariance_errors,
        "continuous_prediction_maximum_errors": continuous_errors,
        "continuous_corruption_index_recovery_count": (
            continuous_index_recovery_count
        ),
        "quantized_corruption_index_recovery_count": int((selected == frame).sum()),
        "quantized_corruption_index_recovery_rate": float(
            (selected == frame).double().mean()
        ),
        "shuffle_fixed_points": shuffle_fixed_points,
        "operators": operators,
        "corruption_material": bool(
            clean["fixed_ceiling_pass"] and not naive["fixed_ceiling_pass"]
        ),
        "robust_repair": repair,
        "oracle_fidelity": oracle_fidelity,
        "valid": validity,
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    def cell(seed: int, regime: str) -> Mapping[str, Any]:
        return next(
            item
            for item in cells
            if item["seed"] == seed and item["regime"] == regime
        )

    def arm_seed_count(arm: str) -> int:
        return sum(
            all(
                bool(cell(seed, regime)["operators"][arm]["fixed_ceiling_pass"])
                for regime in REGIMES
            )
            for seed in SEEDS
        )

    def field_seed_count(field: str) -> int:
        return sum(
            all(bool(cell(seed, regime)[field]) for regime in REGIMES)
            for seed in SEEDS
        )

    def nested_seed_count(field: str) -> int:
        return sum(
            all(bool(cell(seed, regime)[field]["pass"]) for regime in REGIMES)
            for seed in SEEDS
        )

    clean_count = arm_seed_count("clean_all_frame_degree2")
    naive_count = arm_seed_count("corrupted_all_frame_degree2")
    oracle_count = arm_seed_count("oracle_drop_one_quadratic")
    robust_count = arm_seed_count("robust_drop_one_quadratic")
    material_count = field_seed_count("corruption_material")
    repair_count = nested_seed_count("robust_repair")
    fidelity_count = nested_seed_count("oracle_fidelity")
    if not valid:
        classification = "invalid_single_frame_corruption_preflight"
    elif (
        material_count >= REQUIRED_SEED_PASSES
        and robust_count >= REQUIRED_SEED_PASSES
        and repair_count >= REQUIRED_SEED_PASSES
        and fidelity_count >= REQUIRED_SEED_PASSES
    ):
        classification = "fixed_robust_estimator_closes_single_frame_corruption"
    elif naive_count >= REQUIRED_SEED_PASSES:
        classification = "single_frame_corruption_not_material"
    elif (
        material_count >= REQUIRED_SEED_PASSES
        and robust_count < REQUIRED_SEED_PASSES
        and oracle_count >= REQUIRED_SEED_PASSES
    ):
        classification = "recoverable_corruption_exceeds_registered_fixed_estimator"
    elif oracle_count < REQUIRED_SEED_PASSES:
        classification = "declared_corruption_not_recoverable_at_required_ceiling"
    else:
        classification = "inconclusive_single_frame_corruption_preflight"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "clean_fixed_ceiling_seed_pass_count": clean_count,
        "corrupted_naive_fixed_ceiling_seed_pass_count": naive_count,
        "oracle_fixed_ceiling_seed_pass_count": oracle_count,
        "robust_fixed_ceiling_seed_pass_count": robust_count,
        "corruption_material_seed_pass_count": material_count,
        "robust_repair_seed_pass_count": repair_count,
        "oracle_fidelity_seed_pass_count": fidelity_count,
        "tinyllm_training_licensed": False,
        "compact_robust_sensor_comparison_licensed": classification
        == "recoverable_corruption_exceeds_registered_fixed_estimator",
    }


def build_result() -> dict[str, Any]:
    source_hashes, cohort_hashes = validate_sources()
    cells = [
        analyze_cell(
            regime,
            seed,
            expected_base_hash=cohort_hashes[(seed, regime)],
        )
        for seed in SEEDS
        for regime in REGIMES
    ]
    valid = len(cells) == len(SEEDS) * len(REGIMES) and all(
        cell["valid"] for cell in cells
    )
    aggregates = classify(cells, valid)
    source_hashes["runner"] = _sha256(Path(__file__))
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if valid else "invalid",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "source_hashes": source_hashes,
        "configuration": {
            "seeds": list(SEEDS),
            "regimes": list(REGIMES),
            "arms": list(ARMS),
            "sample_count_per_cell": SAMPLE_COUNT,
            "donor_seed_bases": DONOR_SEED_BASES,
            "frame_seed_bases": FRAME_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "minimum_frame_count": MINIMUM_FRAME_COUNT,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "chart_margin_minimum": CHART_MARGIN_MINIMUM,
            "fixed_ceiling_rmse_maximum": FIXED_CEILING_RMSE_MAXIMUM,
            "fixed_ceiling_accuracy_minimum": FIXED_CEILING_ACCURACY_MINIMUM,
            "repair_rmse_ratio_maximum": REPAIR_RMSE_RATIO_MAXIMUM,
            "repair_accuracy_delta_minimum": REPAIR_ACCURACY_DELTA_MINIMUM,
            "repair_cross_entropy_delta_maximum": (
                REPAIR_CROSS_ENTROPY_DELTA_MAXIMUM
            ),
            "oracle_rmse_slack_maximum": ORACLE_RMSE_SLACK_MAXIMUM,
            "oracle_accuracy_slack_maximum": ORACLE_ACCURACY_SLACK_MAXIMUM,
            "oracle_cross_entropy_slack_maximum": (
                ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
            ),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": acceleration.TASK_GATES,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "new_base_evaluation_examples": 0,
            "matched_base_examples_replayed": len(cells) * SAMPLE_COUNT,
            "new_corrupted_evaluations": len(cells) * SAMPLE_COUNT,
            "closed_form_observation_only_fits": (
                len(cells) * SAMPLE_COUNT * (source.TIME_STEPS + 1)
            ),
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "models_instantiated": 0,
            "checkpoints_loaded": 0,
            "target_using_fits": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "The corruption law, estimator family, thresholds, and outcomes were registered before primary corruption generation.",
            "The study tests exactly one unmarked same-time frame substitution under a known constant-acceleration law.",
            "The robust estimator performs target-free closed-form per-example state estimation; it learns no reusable parameter.",
            "A robust failure licenses a compact learned sensor comparison only when the oracle ceiling passes.",
        ],
    }
    if not _finite(result):
        raise RuntimeError("non-finite single-frame-corruption result")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY_RESULT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_result()
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "naive_ceiling_seeds": result["aggregates"][
                    "corrupted_naive_fixed_ceiling_seed_pass_count"
                ],
                "oracle_ceiling_seeds": result["aggregates"][
                    "oracle_fixed_ceiling_seed_pass_count"
                ],
                "robust_ceiling_seeds": result["aggregates"][
                    "robust_fixed_ceiling_seed_pass_count"
                ],
                "robust_repair_seeds": result["aggregates"][
                    "robust_repair_seed_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
