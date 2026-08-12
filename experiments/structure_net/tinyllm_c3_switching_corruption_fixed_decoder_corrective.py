#!/usr/bin/env python3
"""Fresh numerical corrective for the C3 switching-corruption decoder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import sympy
import torch

from experiments.structure_net import (
    tinyllm_c3_switching_corruption_fixed_decoder as base,
)


SCHEMA_VERSION = (
    "nal.tinyllm-c3-switching-corruption-fixed-decoder-corrective.v1"
)
HYPOTHESIS_ID = "tinyllm-c3-switching-corruption-fixed-decoder-corrective-v1"
EVIDENCE_ROLE = "prospective_fresh_cohort_numerical_corrective"
SEEDS = (401, 419, 433, 449, 467)
REGIMES = base.REGIMES
SWITCH_POINTS = base.SWITCH_POINTS
ARMS = base.ARMS
SAMPLE_COUNT = base.SAMPLE_COUNT
DATASET_SEED_BASES = {"composition": 981_107, "extrapolation": 983_107}
PILOT_DATASET_SEED_BASES = {
    "composition": 997_107,
    "extrapolation": 999_107,
}
DONOR_SEED_BASES = {"composition": 985_107, "extrapolation": 987_107}
FRAME_SEED_BASES = {"composition": 989_107, "extrapolation": 991_107}
SHUFFLE_SEED_BASES = {"composition": 993_107, "extrapolation": 995_107}
STABILIZATION_DECIMAL_PLACES = 12
STABILIZATION_SCALE = float(10**STABILIZATION_DECIMAL_PLACES)
STABILIZATION_DISPLACEMENT_MAXIMUM = 1e-12
MINIMUM_SWITCH_COUNT = base.MINIMUM_SWITCH_COUNT
MINIMUM_FRAME_COUNT = base.MINIMUM_FRAME_COUNT
ACTION_ERROR_MAXIMUM = base.ACTION_ERROR_MAXIMUM
CONTINUOUS_ERROR_MAXIMUM = base.CONTINUOUS_ERROR_MAXIMUM
CHART_MARGIN_MINIMUM = base.CHART_MARGIN_MINIMUM
PARETO_RMSE_RATIO_MAXIMUM = base.PARETO_RMSE_RATIO_MAXIMUM
PARETO_ACCURACY_DELTA_MINIMUM = base.PARETO_ACCURACY_DELTA_MINIMUM
PARETO_CROSS_ENTROPY_DELTA_MAXIMUM = base.PARETO_CROSS_ENTROPY_DELTA_MAXIMUM
ORACLE_RMSE_SLACK_MAXIMUM = base.ORACLE_RMSE_SLACK_MAXIMUM
ORACLE_ACCURACY_SLACK_MAXIMUM = base.ORACLE_ACCURACY_SLACK_MAXIMUM
ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM = base.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
REQUIRED_SEED_PASSES = base.REQUIRED_SEED_PASSES
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-"
    "corrective-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "46e40a526c4adf71410beff4d7333da97b4829c2e03f14923744508da602c828"
)
INVALID_RUNNER_SHA256 = (
    "5ebe462c27989e0f09f38bba8ee0e885ea5fb76190bc86c1d7143d79376e2128"
)
INVALID_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_switching_corruption_fixed_decoder/"
    "20260811_preregistered/result.json"
)
INVALID_RESULT_SHA256 = (
    "2dd8f23a2eb7bd5ad7e2c224ee8c08201f648ce905e2fd7d806d7d676badf20c"
)
ORIGINAL_PREREGISTRATION_SHA256 = base.PREREGISTRATION_SHA256
PRIMARY_RESULT_PATH = Path(
    "data/experiments/"
    "tinyllm_c3_switching_corruption_fixed_decoder_corrective/"
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_sources() -> dict[str, str]:
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "invalid_runner": _sha256(Path(base.__file__)),
        "invalid_result": _sha256(INVALID_RESULT_PATH),
        "original_preregistration": _sha256(base.PREREGISTRATION_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "invalid_runner": INVALID_RUNNER_SHA256,
        "invalid_result": INVALID_RESULT_SHA256,
        "original_preregistration": ORIGINAL_PREREGISTRATION_SHA256,
    }
    base.validate_sources()
    invalid = json.loads(INVALID_RESULT_PATH.read_text(encoding="utf-8"))
    invalid_cells = {
        (int(cell["seed"]), str(cell["regime"])): cell
        for cell in invalid.get("cells", [])
        if not cell.get("valid")
    }
    expected_invalid = {(359, "extrapolation"), (389, "composition")}
    if (
        hashes != expected
        or invalid.get("status") != "invalid"
        or invalid.get("aggregates", {}).get("classification")
        != "invalid_switching_corruption_preflight"
        or set(invalid_cells) != expected_invalid
        or any(
            cell["operators"]["corrupted_known_switch_no_drop"][
                "action_pass"
            ]
            is not False
            for cell in invalid_cells.values()
        )
    ):
        raise RuntimeError("switching-corruption corrective sources changed")
    return hashes


def stabilize_carrier(carrier: torch.Tensor) -> tuple[torch.Tensor, float]:
    rounded = torch.complex(
        torch.round(carrier.real * STABILIZATION_SCALE) / STABILIZATION_SCALE,
        torch.round(carrier.imag * STABILIZATION_SCALE) / STABILIZATION_SCALE,
    )
    stabilized = rounded / rounded.abs().clamp_min(1e-12)
    return stabilized, float((stabilized - carrier).abs().max())


def generate_dataset(
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> base.SwitchingDataset:
    primary = seed_value in SEEDS and sample_count == SAMPLE_COUNT
    pilot = allow_pilot and seed_value == 997 and sample_count == 64
    if regime not in REGIMES or not (primary or pilot):
        raise ValueError(f"unknown switching corrective cohort {regime}/{seed_value}")
    bases = DATASET_SEED_BASES if primary else PILOT_DATASET_SEED_BASES
    dataset_seed = bases[regime] + seed_value
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    ranges = base.source.REGIME_RANGES[regime]
    phase = 2.0 * math.pi * torch.rand(
        sample_count, dtype=torch.float64, generator=generator
    )
    velocity = base._signed_uniform(sample_count, ranges["speed"], generator)
    delta_velocity = base._signed_uniform(
        sample_count, base.DELTA_VELOCITY_RANGES[regime], generator
    )
    switch = torch.randint(
        min(SWITCH_POINTS),
        max(SWITCH_POINTS) + 1,
        (sample_count,),
        dtype=torch.int64,
        generator=generator,
    )
    amplitude = base._uniform(sample_count, ranges["amplitude"], generator)
    offset = base._uniform(sample_count, ranges["offset"], generator)
    drift = base._uniform(sample_count, ranges["drift"], generator)
    deck = torch.randint(
        0, base.source.CHANNELS, (sample_count,), generator=generator
    )
    continuous = base._continuous_observation(
        phase,
        velocity,
        delta_velocity,
        switch,
        amplitude,
        offset,
        drift,
    )
    canonical = base.source.quantize(continuous)
    tokens = base.source._roll_by_example(canonical, deck)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    future = base.phase_path(
        phase,
        velocity,
        delta_velocity,
        switch,
        steps=base.source.TIME_STEPS + 1,
    )[:, -1]
    return base.SwitchingDataset(
        canonical_tokens=canonical,
        tokens=tokens,
        calibration=calibration,
        target=torch.cos(3.0 * future),
        phase=phase,
        velocity=velocity,
        delta_velocity=delta_velocity,
        switch=switch,
        deck=deck,
        saturation_count=int(
            (continuous.abs() >= base.source.QUANTIZATION_LIMIT).sum()
        ),
        dataset_seed=dataset_seed,
    )


def corruption_plan(
    regime: str,
    seed_value: int,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    donor = base.corrective.hidden.speed.sattolo_derangement(
        sample_count, DONOR_SEED_BASES[regime] + seed_value
    )
    generator = torch.Generator(device="cpu").manual_seed(
        FRAME_SEED_BASES[regime] + seed_value
    )
    frame = torch.randint(
        0,
        base.source.TIME_STEPS,
        (sample_count,),
        dtype=torch.int64,
        generator=generator,
    )
    return donor, frame


def corrected_predictions(
    clean_carrier: torch.Tensor,
    corrupted_carrier: torch.Tensor,
    true_switch: torch.Tensor,
    true_frame: torch.Tensor,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    stable_clean, clean_displacement = stabilize_carrier(clean_carrier)
    stable_corrupted, corrupted_displacement = stabilize_carrier(
        corrupted_carrier
    )
    predictions, selected, oracle_index, residuals = base.switching_predictions(
        stable_clean,
        stable_corrupted,
        true_switch,
        true_frame,
    )
    return (
        predictions,
        selected,
        oracle_index,
        residuals,
        {
            "clean": clean_displacement,
            "corrupted": corrupted_displacement,
        },
    )


def analyze_cell(
    regime: str,
    seed_value: int,
    audit: Mapping[str, Any],
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    registered = seed_value in SEEDS and sample_count == SAMPLE_COUNT
    dataset = generate_dataset(
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_hash = base.dataset_hash(dataset)
    replay = generate_dataset(
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_deterministic = base_hash == base.dataset_hash(replay)
    donor, frame = corruption_plan(regime, seed_value, sample_count)
    corrupted_tokens = base.corruption.corrupt_frames(
        dataset.tokens, donor, frame
    )
    replay_donor, replay_frame = corruption_plan(
        regime, seed_value, sample_count
    )
    corruption_deterministic = bool(
        torch.equal(donor, replay_donor)
        and torch.equal(frame, replay_frame)
        and torch.equal(
            corrupted_tokens,
            base.corruption.corrupt_frames(
                dataset.tokens, replay_donor, replay_frame
            ),
        )
    )
    clean_carrier, clean_magnitude = base.source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    corrupted_carrier, corrupted_magnitude = base.source.analytic_carrier(
        corrupted_tokens, dataset.calibration
    )
    predictions, selected, oracle_index, residuals, displacements = (
        corrected_predictions(
            clean_carrier,
            corrupted_carrier,
            dataset.switch,
            frame,
        )
    )
    action_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    equivariance_errors: dict[str, int] = {}
    transformed_displacements: dict[str, dict[str, float]] = {}
    for element in (1, 2):
        transformed_clean_tokens = base.source.apply_deck_action(
            dataset.tokens, element
        )
        transformed_corrupted_tokens = base.corruption.corrupt_frames(
            transformed_clean_tokens, donor, frame
        )
        equivariance_errors[str(element)] = int(
            (
                transformed_corrupted_tokens
                != base.source.apply_deck_action(corrupted_tokens, element)
            ).sum()
        )
        transformed_clean_carrier, _ = base.source.analytic_carrier(
            transformed_clean_tokens, dataset.calibration
        )
        transformed_corrupted_carrier, _ = base.source.analytic_carrier(
            transformed_corrupted_tokens, dataset.calibration
        )
        transformed_predictions, _, _, _, element_displacements = (
            corrected_predictions(
                transformed_clean_carrier,
                transformed_corrupted_carrier,
                dataset.switch,
                frame,
            )
        )
        transformed_displacements[str(element)] = element_displacements
        for arm in ARMS:
            action_errors[arm][str(element)] = float(
                (transformed_predictions[arm] - predictions[arm]).abs().max()
            )

    exact_theta = base.phase_path(
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        steps=base.source.TIME_STEPS + 1,
    )
    exact_carrier = torch.polar(
        torch.ones_like(exact_theta), 3.0 * exact_theta
    ).to(torch.complex128)
    exact_corrupted = base.corruption.corrupt_frames(
        exact_carrier[:, :-1], donor, frame
    )
    stable_exact, exact_clean_displacement = stabilize_carrier(
        exact_carrier[:, :-1]
    )
    stable_exact_corrupted, exact_corrupted_displacement = stabilize_carrier(
        exact_corrupted
    )
    del stable_exact
    exact_candidates, exact_residuals, exact_identities = (
        base.switch_deletion_candidates(stable_exact_corrupted)
    )
    exact_selected = exact_residuals.argmin(dim=1)
    exact_index = {
        identity: offset for offset, identity in enumerate(exact_identities)
    }
    exact_oracle_index = torch.tensor(
        [
            exact_index[(int(item_frame), int(item_switch))]
            for item_frame, item_switch in zip(frame, dataset.switch)
        ],
        dtype=torch.int64,
    )
    row = torch.arange(sample_count, dtype=torch.int64)
    exact_oracle = exact_candidates[row, exact_oracle_index]
    exact_fixed = exact_candidates[row, exact_selected]
    exact_future = exact_carrier[:, -1]
    continuous_errors = {
        "oracle_switch_drop": float((exact_oracle - exact_future).abs().max()),
        "fixed_switch_drop": float((exact_fixed - exact_future).abs().max()),
    }
    equivalent_count = int(
        ((exact_fixed - exact_future).abs() <= CONTINUOUS_ERROR_MAXIMUM).sum()
    )

    shuffle = base.corrective.hidden.speed.sattolo_derangement(
        sample_count, SHUFFLE_SEED_BASES[regime] + seed_value
    )
    shuffled_target = dataset.target[shuffle]
    operators = {
        arm: base.corruption._arm_metrics(
            predictions[arm],
            dataset.target,
            shuffled_target,
            regime,
            action_errors[arm],
        )
        for arm in ARMS
    }
    comparisons = base._comparison_metrics(operators)
    switch_counts = torch.bincount(dataset.switch, minlength=6)
    frame_counts = torch.bincount(frame, minlength=base.source.TIME_STEPS)
    selected_frame = torch.tensor(
        [int(index) // len(SWITCH_POINTS) for index in selected],
        dtype=torch.int64,
    )
    selected_switch = torch.tensor(
        [SWITCH_POINTS[int(index) % len(SWITCH_POINTS)] for index in selected],
        dtype=torch.int64,
    )
    group = base.group_contracts(dataset)
    chart_margin = base._chart_margin(exact_carrier[:, :-1], frame)
    donor_fixed_points = int(
        (donor == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    shuffle_fixed_points = int(
        (shuffle == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    all_displacements = [
        *displacements.values(),
        *(
            value
            for element in transformed_displacements.values()
            for value in element.values()
        ),
        exact_clean_displacement,
        exact_corrupted_displacement,
    ]
    maximum_displacement = max(all_displacements)
    required_switch_count = MINIMUM_SWITCH_COUNT if registered else 0
    required_frame_count = MINIMUM_FRAME_COUNT if registered else 0
    validity = bool(
        audit.get("pass") is True
        and dataset.saturation_count == 0
        and base_deterministic
        and group["pass"]
        and corruption_deterministic
        and donor_fixed_points == 0
        and min(int(switch_counts[item]) for item in SWITCH_POINTS)
        >= required_switch_count
        and int(frame_counts.min()) >= required_frame_count
        and sum(equivariance_errors.values()) == 0
        and maximum_displacement <= STABILIZATION_DISPLACEMENT_MAXIMUM
        and all(operator["action_pass"] for operator in operators.values())
        and (
            all(operator["shuffle_pass"] for operator in operators.values())
            if registered
            else True
        )
        and max(continuous_errors.values()) <= CONTINUOUS_ERROR_MAXIMUM
        and equivalent_count == sample_count
        and chart_margin >= CHART_MARGIN_MINIMUM
        and shuffle_fixed_points == 0
        and float(clean_magnitude.min()) > 1e-12
        and float(corrupted_magnitude.min()) > 1e-12
        and base._finite(operators)
        and base._finite(comparisons)
        and base._finite(residuals)
    )
    return {
        "seed": seed_value,
        "regime": regime,
        "dataset_seed": dataset.dataset_seed,
        "base_dataset_sha256": base_hash,
        "base_dataset_deterministic": base_deterministic,
        "corruption_sha256": base._tensor_hash(corrupted_tokens, donor, frame),
        "corruption_deterministic": corruption_deterministic,
        "sample_count": sample_count,
        "saturation_count": dataset.saturation_count,
        "switch_counts": {
            str(item): int(switch_counts[item]) for item in SWITCH_POINTS
        },
        "minimum_switch_count": min(
            int(switch_counts[item]) for item in SWITCH_POINTS
        ),
        "frame_counts": [int(item) for item in frame_counts],
        "minimum_frame_count": int(frame_counts.min()),
        "donor_fixed_points": donor_fixed_points,
        "shuffle_fixed_points": shuffle_fixed_points,
        "minimum_clean_chart_margin": chart_margin,
        "minimum_clean_character_magnitude": float(clean_magnitude.min()),
        "minimum_corrupted_character_magnitude": float(
            corrupted_magnitude.min()
        ),
        "base_group_contracts": group,
        "corruption_equivariance_token_errors": equivariance_errors,
        "stabilization_displacements": {
            "natural": displacements,
            "transformed": transformed_displacements,
            "exact_clean": exact_clean_displacement,
            "exact_corrupted": exact_corrupted_displacement,
            "maximum": maximum_displacement,
        },
        "continuous_prediction_maximum_errors": continuous_errors,
        "continuous_future_equivalent_count": equivalent_count,
        "quantized_corruption_index_recovery_rate": float(
            (selected_frame == frame).double().mean()
        ),
        "quantized_switch_index_recovery_rate": float(
            (selected_switch == dataset.switch).double().mean()
        ),
        "quantized_joint_hidden_index_recovery_rate": float(
            (
                (selected_frame == frame)
                & (selected_switch == dataset.switch)
            ).double().mean()
        ),
        "quantized_oracle_candidate_selected_rate": float(
            (selected == oracle_index).double().mean()
        ),
        "operators": operators,
        "dynamics_material": bool(
            operators["clean_known_switch"]["fixed_ceiling_pass"]
            and not operators["clean_global_quadratic"]["fixed_ceiling_pass"]
        ),
        "corruption_material": bool(
            operators["clean_known_switch"]["fixed_ceiling_pass"]
            and not operators["corrupted_known_switch_no_drop"][
                "fixed_ceiling_pass"
            ]
        ),
        **comparisons,
        "valid": validity,
    }


def classify(
    cells: Sequence[Mapping[str, Any]],
    valid: bool,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    def cell(seed: int, regime: str) -> Mapping[str, Any]:
        return next(
            item
            for item in cells
            if item["seed"] == seed and item["regime"] == regime
        )

    def arm_count(arm: str) -> int:
        return sum(
            all(
                bool(cell(seed, regime)["operators"][arm]["fixed_ceiling_pass"])
                for regime in REGIMES
            )
            for seed in SEEDS
        )

    def field_count(field: str) -> int:
        return sum(
            all(bool(cell(seed, regime)[field]) for regime in REGIMES)
            for seed in SEEDS
        )

    def nested_count(field: str) -> int:
        return sum(
            all(bool(cell(seed, regime)[field]["pass"]) for regime in REGIMES)
            for seed in SEEDS
        )

    clean_count = arm_count("clean_known_switch")
    clean_global_count = arm_count("clean_global_quadratic")
    corrupted_global_count = arm_count("corrupted_global_quadratic")
    corrupted_known_count = arm_count("corrupted_known_switch_no_drop")
    oracle_count = arm_count("oracle_switch_drop")
    fixed_count = arm_count("fixed_switch_drop")
    dynamics_count = field_count("dynamics_material")
    corruption_count = field_count("corruption_material")
    pareto_count = nested_count("pareto_repair")
    fidelity_count = nested_count("oracle_fidelity")
    if not valid:
        classification = "invalid_switching_corruption_corrective"
    elif (
        audit.get("pass") is True
        and dynamics_count >= REQUIRED_SEED_PASSES
        and corruption_count >= REQUIRED_SEED_PASSES
        and oracle_count >= REQUIRED_SEED_PASSES
        and fixed_count >= REQUIRED_SEED_PASSES
        and pareto_count >= REQUIRED_SEED_PASSES
        and fidelity_count >= REQUIRED_SEED_PASSES
    ):
        classification = (
            "fixed_change_point_decoder_closes_identifiable_switching_corruption"
        )
    elif (
        clean_global_count >= REQUIRED_SEED_PASSES
        or corrupted_global_count >= REQUIRED_SEED_PASSES
    ):
        classification = "switching_corruption_scope_not_material"
    elif oracle_count < REQUIRED_SEED_PASSES:
        classification = (
            "identifiable_switching_not_recoverable_at_required_ceiling"
        )
    elif fixed_count < REQUIRED_SEED_PASSES:
        classification = "recoverable_switching_exceeds_fixed_change_point_decoder"
    else:
        classification = "inconclusive_switching_corruption_preflight"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "identifiability_contract_pass": bool(audit.get("pass")),
        "late_support_target_identifiable_under_one_corruption": False,
        "primary_support_one_corruption_correctable": bool(
            audit.get("primary_support_one_corruption_correctable")
        ),
        "clean_known_switch_seed_pass_count": clean_count,
        "clean_global_quadratic_seed_pass_count": clean_global_count,
        "corrupted_global_quadratic_seed_pass_count": corrupted_global_count,
        "corrupted_known_switch_no_drop_seed_pass_count": corrupted_known_count,
        "oracle_switch_drop_seed_pass_count": oracle_count,
        "fixed_switch_drop_seed_pass_count": fixed_count,
        "dynamics_materiality_seed_pass_count": dynamics_count,
        "corruption_materiality_seed_pass_count": corruption_count,
        "pareto_repair_seed_pass_count": pareto_count,
        "oracle_fidelity_seed_pass_count": fidelity_count,
        "tinyllm_training_licensed": False,
        "compact_typed_continuation_comparison_licensed": classification
        == "recoverable_switching_exceeds_fixed_change_point_decoder",
    }


def build_result() -> dict[str, Any]:
    source_hashes = validate_sources()
    audit = base.identifiability_audit()
    cells = [
        analyze_cell(regime, seed, audit)
        for seed in SEEDS
        for regime in REGIMES
    ]
    valid = bool(
        audit["pass"]
        and len(cells) == len(SEEDS) * len(REGIMES)
        and all(cell["valid"] for cell in cells)
    )
    aggregates = classify(cells, valid, audit)
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
            "primary_switch_points": list(SWITCH_POINTS),
            "late_inclusive_switch_points": list(base.LATE_SWITCH_POINTS),
            "arms": list(ARMS),
            "sample_count_per_cell": SAMPLE_COUNT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "donor_seed_bases": DONOR_SEED_BASES,
            "frame_seed_bases": FRAME_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "delta_velocity_ranges": base.DELTA_VELOCITY_RANGES,
            "stabilization_decimal_places": STABILIZATION_DECIMAL_PLACES,
            "stabilization_displacement_maximum": (
                STABILIZATION_DISPLACEMENT_MAXIMUM
            ),
            "minimum_switch_count": MINIMUM_SWITCH_COUNT,
            "minimum_frame_count": MINIMUM_FRAME_COUNT,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "chart_margin_minimum": CHART_MARGIN_MINIMUM,
            "fixed_ceiling_rmse_maximum": (
                base.corruption.FIXED_CEILING_RMSE_MAXIMUM
            ),
            "fixed_ceiling_accuracy_minimum": (
                base.corruption.FIXED_CEILING_ACCURACY_MINIMUM
            ),
            "pareto_rmse_ratio_maximum": PARETO_RMSE_RATIO_MAXIMUM,
            "pareto_accuracy_delta_minimum": PARETO_ACCURACY_DELTA_MINIMUM,
            "pareto_cross_entropy_delta_maximum": (
                PARETO_CROSS_ENTROPY_DELTA_MAXIMUM
            ),
            "oracle_rmse_slack_maximum": ORACLE_RMSE_SLACK_MAXIMUM,
            "oracle_accuracy_slack_maximum": ORACLE_ACCURACY_SLACK_MAXIMUM,
            "oracle_cross_entropy_slack_maximum": (
                ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
            ),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": base.acceleration.TASK_GATES,
        },
        "identifiability_audit": audit,
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "fresh_base_examples": len(cells) * SAMPLE_COUNT,
            "fresh_corrupted_evaluations": len(cells) * SAMPLE_COUNT,
            "observation_only_candidate_fits": len(cells) * SAMPLE_COUNT * 42,
            "continuous_validation_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 24
            ),
            "invalid_primary_examples_pooled": 0,
            "law_label_reads_by_fixed_decoder": 0,
            "reusable_parameter_fits": 0,
            "target_using_fits": 0,
            "models_instantiated": 0,
            "checkpoints_loaded": 0,
            "optimizer_steps": 0,
            "parameters_changed": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sympy": sympy.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "The invalid predecessor remains unchanged and contributes zero examples to the corrective decision.",
            "The only estimator change is fixed twelve-decimal carrier canonicalization before phase unwrapping.",
            "The late-inclusive target remains non-identifiable under one corruption; primary switches require three post-change frames.",
            "Latent switch/corruption recovery is descriptive; continuous future equivalence is the validity endpoint.",
            "No model, checkpoint, optimizer, target fit, or reusable parameter is involved.",
        ],
    }
    if not base._finite(result):
        raise RuntimeError("non-finite switching-corruption corrective result")
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
                "oracle_seed_passes": result["aggregates"][
                    "oracle_switch_drop_seed_pass_count"
                ],
                "fixed_seed_passes": result["aggregates"][
                    "fixed_switch_drop_seed_pass_count"
                ],
                "oracle_fidelity_seed_passes": result["aggregates"][
                    "oracle_fidelity_seed_pass_count"
                ],
                "tinyllm_training_licensed": result["aggregates"][
                    "tinyllm_training_licensed"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
