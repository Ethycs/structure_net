#!/usr/bin/env python3
"""Fresh corrective for the C3 hidden-order corruption fixed estimator."""

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

import experiments.structure_net.tinyllm_c3_hidden_order_corruption_fixed_estimator as hidden


SCHEMA_VERSION = "nal.tinyllm-c3-hidden-order-corruption-corrective.v1"
HYPOTHESIS_ID = "tinyllm-c3-hidden-order-corruption-corrective-v1"
EVIDENCE_ROLE = "prospective_no_training_corrective_replication"
PAIR_SEEDS = ((71, 211), (83, 229), (97, 251), (109, 277), (131, 307))
REGIMES = hidden.REGIMES
FAMILIES = hidden.FAMILIES
POPULATIONS = hidden.POPULATIONS
ARMS = hidden.ARMS
SAMPLE_COUNT = hidden.SAMPLE_COUNT
DATASET_SEED_BASES = {
    "constant_speed": {"composition": 911_107, "extrapolation": 913_107},
    "constant_acceleration": {
        "composition": 915_107,
        "extrapolation": 917_107,
    },
}
PILOT_DATASET_SEED_BASES = {
    "constant_speed": {"composition": 931_107, "extrapolation": 933_107},
    "constant_acceleration": {
        "composition": 935_107,
        "extrapolation": 937_107,
    },
}
SHUFFLE_SEED_BASES = {
    "constant_speed": {"composition": 921_107, "extrapolation": 923_107},
    "constant_acceleration": {
        "composition": 925_107,
        "extrapolation": 927_107,
    },
}
MINIMUM_FRAME_COUNT = hidden.MINIMUM_FRAME_COUNT
ACTION_ERROR_MAXIMUM = hidden.ACTION_ERROR_MAXIMUM
CONTINUOUS_ERROR_MAXIMUM = hidden.CONTINUOUS_ERROR_MAXIMUM
INCLUSION_ERROR_MAXIMUM = hidden.INCLUSION_ERROR_MAXIMUM
CHART_MARGIN_MINIMUM = hidden.CHART_MARGIN_MINIMUM
PARETO_RMSE_RATIO_MAXIMUM = 0.50
PARETO_ACCURACY_DELTA_MINIMUM = -0.005
PARETO_CROSS_ENTROPY_DELTA_MAXIMUM = -0.10
REQUIRED_SEED_PASSES = 4
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-hidden-order-corruption-corrective-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "2778eb0e9ade9c0f084e194865df17b2b9ece1558d396d4e9227f3d599e91834"
)
HIDDEN_RUNNER_SHA256 = (
    "318cf81497960d37f86b1be58af3d819076e7dc9b620fe2ba73ae215ae0adea7"
)
HIDDEN_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)
HIDDEN_RESULT_SHA256 = (
    "88e44e76c44d654331ea647ccedd8dad505030a65ab7c4ac241d8b98d98bd02e"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_corrective/"
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
        "hidden_order_runner": _sha256(Path(hidden.__file__)),
        "hidden_order_result": _sha256(HIDDEN_RESULT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "hidden_order_runner": HIDDEN_RUNNER_SHA256,
        "hidden_order_result": HIDDEN_RESULT_SHA256,
    }
    predecessor = json.loads(HIDDEN_RESULT_PATH.read_text(encoding="utf-8"))
    hidden.validate_sources()
    if (
        hashes != expected
        or predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != "inconclusive_hidden_order_corruption_preflight"
        or predecessor.get("aggregates", {}).get(
            "common_robust_endpoint_seed_pass_count"
        )
        != 5
        or predecessor.get("aggregates", {}).get("repair_seed_pass_count") != 0
    ):
        raise RuntimeError("hidden-order corrective source contract changed")
    return hashes


def _uniform(
    count: int,
    bounds: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.empty(count, dtype=torch.float64).uniform_(
        *bounds, generator=generator
    )


def _signed_uniform(
    count: int,
    bounds: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    magnitude = _uniform(count, bounds, generator)
    sign = (
        2.0
        * torch.randint(
            0, 2, (count,), dtype=torch.int64, generator=generator
        ).double()
        - 1.0
    )
    return magnitude * sign


def generate_dataset(
    family: str,
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> Any:
    primary = sample_count == SAMPLE_COUNT and (
        (family == "constant_speed" and seed_value in {pair[0] for pair in PAIR_SEEDS})
        or (
            family == "constant_acceleration"
            and seed_value in {pair[1] for pair in PAIR_SEEDS}
        )
    )
    pilot = allow_pilot and sample_count == 64 and seed_value == 997
    if family not in FAMILIES or regime not in REGIMES or not (primary or pilot):
        raise ValueError(f"unknown corrective cohort {family}/{regime}/{seed_value}")
    seed_bases = DATASET_SEED_BASES if primary else PILOT_DATASET_SEED_BASES
    dataset_seed = seed_bases[family][regime] + seed_value
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    phase = 2.0 * math.pi * torch.rand(
        sample_count, dtype=torch.float64, generator=generator
    )
    if family == "constant_speed":
        ranges = hidden.source.REGIME_RANGES[regime]
        velocity = _signed_uniform(sample_count, ranges["speed"], generator)
        acceleration_value = None
    else:
        ranges = hidden.acceleration.REGIME_RANGES[regime]
        velocity = _signed_uniform(sample_count, ranges["velocity"], generator)
        acceleration_value = _signed_uniform(
            sample_count, ranges["acceleration"], generator
        )
    amplitude = _uniform(sample_count, ranges["amplitude"], generator)
    offset = _uniform(sample_count, ranges["offset"], generator)
    drift = _uniform(sample_count, ranges["drift"], generator)
    deck = torch.randint(
        0, hidden.source.CHANNELS, (sample_count,), generator=generator
    )
    continuous = (
        hidden.source._continuous_observation(
            phase, velocity, amplitude, offset, drift
        )
        if family == "constant_speed"
        else hidden.acceleration._continuous_observation(
            phase,
            velocity,
            acceleration_value,
            amplitude,
            offset,
            drift,
        )
    )
    canonical = hidden.source.quantize(continuous)
    tokens = torch.stack(
        [
            hidden.source.apply_deck_action(row, int(element))
            for row, element in zip(canonical, deck)
        ]
    )
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    time = float(hidden.source.TIME_STEPS)
    future_phase = (
        phase + velocity * time
        if family == "constant_speed"
        else phase
        + velocity * time
        + 0.5 * acceleration_value * time * (time - 1.0)
    )
    common = {
        "canonical_tokens": canonical,
        "tokens": tokens,
        "calibration": calibration,
        "target": torch.cos(3.0 * future_phase),
        "phase": phase,
        "deck": deck,
        "saturation_count": int(
            (continuous.abs() >= hidden.source.QUANTIZATION_LIMIT).sum()
        ),
        "dataset_seed": dataset_seed,
    }
    if family == "constant_speed":
        return hidden.speed.CeilingDataset(speed=velocity, **common)
    return hidden.acceleration.AccelerationDataset(
        velocity=velocity,
        acceleration=acceleration_value,
        **common,
    )


def _corrective_comparisons(
    operators: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    clean = operators["clean_law_specific"]
    naive = operators["corrupted_law_specific"]
    oracle = operators["oracle_drop_one_quadratic"]
    robust = operators["robust_drop_one_quadratic"]
    pareto = {
        "rmse_ratio": robust["scalar"]["rmse"]
        / max(naive["scalar"]["rmse"], 1e-18),
        "accuracy_delta": robust["task"]["exact_bin_accuracy"]
        - naive["task"]["exact_bin_accuracy"],
        "cross_entropy_delta": robust["task"]["target_cross_entropy"]
        - naive["task"]["target_cross_entropy"],
    }
    pareto["pass"] = bool(
        robust["fixed_ceiling_pass"]
        and pareto["rmse_ratio"] <= PARETO_RMSE_RATIO_MAXIMUM
        and pareto["accuracy_delta"] + 1e-12
        >= PARETO_ACCURACY_DELTA_MINIMUM
        and pareto["cross_entropy_delta"]
        <= PARETO_CROSS_ENTROPY_DELTA_MAXIMUM
    )
    fidelity = {
        "rmse_excess": robust["scalar"]["rmse"] - oracle["scalar"]["rmse"],
        "accuracy_delta": robust["task"]["exact_bin_accuracy"]
        - oracle["task"]["exact_bin_accuracy"],
        "cross_entropy_excess": robust["task"]["target_cross_entropy"]
        - oracle["task"]["target_cross_entropy"],
    }
    fidelity["pass"] = bool(
        fidelity["rmse_excess"]
        <= hidden.corruption.ORACLE_RMSE_SLACK_MAXIMUM
        and fidelity["accuracy_delta"]
        >= -hidden.corruption.ORACLE_ACCURACY_SLACK_MAXIMUM
        and fidelity["cross_entropy_excess"]
        <= hidden.corruption.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
    )
    return {
        "corruption_material": bool(
            clean["fixed_ceiling_pass"] and not naive["fixed_ceiling_pass"]
        ),
        "pareto_repair": pareto,
        "oracle_fidelity": fidelity,
    }


def analyze_family(
    family: str,
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registered = sample_count == SAMPLE_COUNT and seed_value != 997
    dataset = generate_dataset(
        family,
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_hash = hidden._dataset_hash(family, dataset)
    repeated = generate_dataset(
        family,
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_deterministic = base_hash == hidden._dataset_hash(family, repeated)
    donor, frame = hidden.corruption.corruption_plan(
        regime, seed_value, sample_count
    )
    corrupted_tokens = hidden.corruption.corrupt_frames(
        dataset.tokens, donor, frame
    )
    replay_donor, replay_frame = hidden.corruption.corruption_plan(
        regime, seed_value, sample_count
    )
    corruption_deterministic = bool(
        torch.equal(donor, replay_donor)
        and torch.equal(frame, replay_frame)
        and torch.equal(
            corrupted_tokens,
            hidden.corruption.corrupt_frames(
                dataset.tokens, replay_donor, replay_frame
            ),
        )
    )
    clean_carrier, clean_magnitude = hidden.source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    corrupted_carrier, corrupted_magnitude = hidden.source.analytic_carrier(
        corrupted_tokens, dataset.calibration
    )
    predictions, selected = hidden.common_predictions(
        family, clean_carrier, corrupted_carrier, frame
    )
    action_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    equivariance_errors: dict[str, int] = {}
    for element in (1, 2):
        transformed_clean_tokens = hidden.source.apply_deck_action(
            dataset.tokens, element
        )
        transformed_corrupted_tokens = hidden.corruption.corrupt_frames(
            transformed_clean_tokens, donor, frame
        )
        equivariance_errors[str(element)] = int(
            (
                transformed_corrupted_tokens
                != hidden.source.apply_deck_action(corrupted_tokens, element)
            ).sum()
        )
        transformed_clean_carrier, _ = hidden.source.analytic_carrier(
            transformed_clean_tokens, dataset.calibration
        )
        transformed_corrupted_carrier, _ = hidden.source.analytic_carrier(
            transformed_corrupted_tokens, dataset.calibration
        )
        transformed_predictions, _ = hidden.common_predictions(
            family,
            transformed_clean_carrier,
            transformed_corrupted_carrier,
            frame,
        )
        for arm in ARMS:
            action_errors[arm][str(element)] = float(
                (transformed_predictions[arm] - predictions[arm]).abs().max()
            )
    exact_clean, exact_future, expected_beta2 = hidden._continuous_carriers(
        family, dataset
    )
    exact_corrupted = hidden.corruption.corrupt_frames(
        exact_clean, donor, frame
    )
    exact_oracle, exact_robust, exact_selected, _ = (
        hidden.corruption.deletion_predictions(exact_corrupted, frame)
    )
    continuous_errors = {
        "oracle_drop_one_quadratic": float(
            (exact_oracle - exact_future).abs().max()
        ),
        "robust_drop_one_quadratic": float(
            (exact_robust - exact_future).abs().max()
        ),
    }
    coefficients = hidden._true_deletion_coefficients(exact_clean, frame)
    inclusion_error = float((coefficients[:, 2] - expected_beta2).abs().max())
    chart_margin = hidden.corruption._chart_margin(exact_clean, frame)
    shuffle = hidden.speed.sattolo_derangement(
        sample_count, SHUFFLE_SEED_BASES[family][regime] + seed_value
    )
    shuffled_target = dataset.target[shuffle]
    operators = {
        arm: hidden.corruption._arm_metrics(
            predictions[arm],
            dataset.target,
            shuffled_target,
            regime,
            action_errors[arm],
        )
        for arm in ARMS
    }
    comparisons = _corrective_comparisons(operators)
    frame_counts = torch.bincount(frame, minlength=hidden.source.TIME_STEPS)
    donor_fixed_points = int(
        (donor == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    shuffle_fixed_points = int(
        (shuffle == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    base_contracts = (
        hidden._speed_group_contracts(dataset)
        if family == "constant_speed"
        else hidden.acceleration._group_contracts(dataset)
    )
    validity = bool(
        dataset.saturation_count == 0
        and base_deterministic
        and base_contracts["pass"]
        and corruption_deterministic
        and donor_fixed_points == 0
        and int(frame_counts.min())
        >= (MINIMUM_FRAME_COUNT if registered else 0)
        and sum(equivariance_errors.values()) == 0
        and all(operator["action_pass"] for operator in operators.values())
        and (
            all(operator["shuffle_pass"] for operator in operators.values())
            if registered
            else True
        )
        and max(continuous_errors.values()) <= CONTINUOUS_ERROR_MAXIMUM
        and int((exact_selected == frame).sum()) == sample_count
        and inclusion_error <= INCLUSION_ERROR_MAXIMUM
        and chart_margin >= CHART_MARGIN_MINIMUM
        and shuffle_fixed_points == 0
        and float(clean_magnitude.min()) > 1e-12
        and float(corrupted_magnitude.min()) > 1e-12
        and hidden._finite(operators)
        and hidden._finite(comparisons)
    )
    record = {
        "family": family,
        "seed": seed_value,
        "dataset_seed": dataset.dataset_seed,
        "base_dataset_sha256": base_hash,
        "base_dataset_deterministic": base_deterministic,
        "corruption_sha256": hidden._tensor_hash(
            corrupted_tokens, donor, frame
        ),
        "corruption_deterministic": corruption_deterministic,
        "sample_count": sample_count,
        "saturation_count": dataset.saturation_count,
        "donor_fixed_points": donor_fixed_points,
        "frame_counts": [int(value) for value in frame_counts],
        "minimum_frame_count": int(frame_counts.min()),
        "minimum_clean_chart_margin": chart_margin,
        "continuous_law_inclusion_maximum_error": inclusion_error,
        "continuous_prediction_maximum_errors": continuous_errors,
        "continuous_corruption_index_recovery_count": int(
            (exact_selected == frame).sum()
        ),
        "quantized_corruption_index_recovery_count": int((selected == frame).sum()),
        "quantized_corruption_index_recovery_rate": float(
            (selected == frame).double().mean()
        ),
        "minimum_clean_character_magnitude": float(clean_magnitude.min()),
        "minimum_corrupted_character_magnitude": float(
            corrupted_magnitude.min()
        ),
        "base_group_contracts": base_contracts,
        "corruption_equivariance_token_errors": equivariance_errors,
        "shuffle_fixed_points": shuffle_fixed_points,
        "operators": operators,
        **comparisons,
        "valid": validity,
    }
    return record, {
        "target": dataset.target,
        "shuffled_target": shuffled_target,
        "predictions": predictions,
        "action_errors": action_errors,
    }


def analyze_cell(
    replicate: int,
    regime: str,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    if replicate not in range(1, 6) and not allow_pilot:
        raise ValueError(f"unknown corrective replicate {replicate}")
    speed_seed, acceleration_seed = (
        PAIR_SEEDS[replicate - 1] if not allow_pilot else (997, 997)
    )
    speed_record, speed_payload = analyze_family(
        "constant_speed",
        regime,
        speed_seed,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    acceleration_record, acceleration_payload = analyze_family(
        "constant_acceleration",
        regime,
        acceleration_seed,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    mixed_target = torch.cat(
        (speed_payload["target"], acceleration_payload["target"])
    )
    mixed_shuffled = torch.cat(
        (
            speed_payload["shuffled_target"],
            acceleration_payload["shuffled_target"],
        )
    )
    mixture_operators = {}
    for arm in ARMS:
        prediction = torch.cat(
            (
                speed_payload["predictions"][arm],
                acceleration_payload["predictions"][arm],
            )
        )
        action_errors = {
            element: max(
                speed_payload["action_errors"][arm][element],
                acceleration_payload["action_errors"][arm][element],
            )
            for element in ("1", "2")
        }
        mixture_operators[arm] = hidden.corruption._arm_metrics(
            prediction,
            mixed_target,
            mixed_shuffled,
            regime,
            action_errors,
        )
    mixture_comparisons = _corrective_comparisons(mixture_operators)
    mixture = {
        "law_proportions": {
            "constant_speed": 0.5,
            "constant_acceleration": 0.5,
        },
        "sample_count": int(mixed_target.numel()),
        "operators": mixture_operators,
        **mixture_comparisons,
        "valid": bool(
            all(operator["action_pass"] for operator in mixture_operators.values())
            and (
                all(
                    operator["shuffle_pass"]
                    for operator in mixture_operators.values()
                )
                if not allow_pilot
                else True
            )
            and hidden._finite(mixture_operators)
            and hidden._finite(mixture_comparisons)
        ),
    }
    populations = {
        "constant_speed": speed_record,
        "constant_acceleration": acceleration_record,
        "mixture": mixture,
    }
    return {
        "replicate": replicate,
        "regime": regime,
        "speed_seed": speed_seed,
        "acceleration_seed": acceleration_seed,
        "populations": populations,
        "common_robust_endpoint_pass": all(
            populations[name]["operators"]["robust_drop_one_quadratic"][
                "fixed_ceiling_pass"
            ]
            for name in POPULATIONS
        ),
        "oracle_endpoint_pass": all(
            populations[name]["operators"]["oracle_drop_one_quadratic"][
                "fixed_ceiling_pass"
            ]
            for name in POPULATIONS
        ),
        "materiality_pass": all(
            populations[name]["corruption_material"] for name in POPULATIONS
        ),
        "pareto_repair_pass": all(
            populations[name]["pareto_repair"]["pass"] for name in POPULATIONS
        ),
        "oracle_fidelity_pass": all(
            populations[name]["oracle_fidelity"]["pass"] for name in POPULATIONS
        ),
        "valid": all(populations[name]["valid"] for name in POPULATIONS),
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    def seed_count(field: str) -> int:
        return sum(
            all(
                bool(
                    next(
                        cell
                        for cell in cells
                        if cell["replicate"] == replicate
                        and cell["regime"] == regime
                    )[field]
                )
                for regime in REGIMES
            )
            for replicate in range(1, 6)
        )

    common_count = seed_count("common_robust_endpoint_pass")
    oracle_count = seed_count("oracle_endpoint_pass")
    material_count = seed_count("materiality_pass")
    pareto_count = seed_count("pareto_repair_pass")
    fidelity_count = seed_count("oracle_fidelity_pass")
    naive_mixture_count = sum(
        all(
            bool(
                next(
                    cell
                    for cell in cells
                    if cell["replicate"] == replicate
                    and cell["regime"] == regime
                )["populations"]["mixture"]["operators"][
                    "corrupted_law_specific"
                ]["fixed_ceiling_pass"]
            )
            for regime in REGIMES
        )
        for replicate in range(1, 6)
    )
    if not valid:
        classification = "invalid_hidden_order_corruption_corrective"
    elif all(
        count >= REQUIRED_SEED_PASSES
        for count in (common_count, material_count, pareto_count, fidelity_count)
    ):
        classification = "fresh_corrective_confirms_common_robust_nested_law_closure"
    elif naive_mixture_count >= REQUIRED_SEED_PASSES:
        classification = "fresh_hidden_order_corruption_not_material"
    elif oracle_count >= REQUIRED_SEED_PASSES and common_count < REQUIRED_SEED_PASSES:
        classification = "fresh_recoverable_scope_exceeds_common_fixed_estimator"
    elif oracle_count < REQUIRED_SEED_PASSES:
        classification = "fresh_scope_not_recoverable_at_required_ceiling"
    else:
        classification = "inconclusive_hidden_order_corruption_corrective"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "common_robust_endpoint_seed_pass_count": common_count,
        "oracle_endpoint_seed_pass_count": oracle_count,
        "materiality_seed_pass_count": material_count,
        "pareto_repair_seed_pass_count": pareto_count,
        "oracle_fidelity_seed_pass_count": fidelity_count,
        "corrupted_naive_mixture_ceiling_seed_pass_count": naive_mixture_count,
        "tinyllm_training_licensed": False,
        "compact_typed_selector_comparison_licensed": classification
        == "fresh_recoverable_scope_exceeds_common_fixed_estimator",
    }


def build_result() -> dict[str, Any]:
    source_hashes = validate_sources()
    cells = [
        analyze_cell(replicate, regime)
        for replicate in range(1, 6)
        for regime in REGIMES
    ]
    valid = len(cells) == 10 and all(cell["valid"] for cell in cells)
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
            "pair_seeds": [list(pair) for pair in PAIR_SEEDS],
            "regimes": list(REGIMES),
            "families": list(FAMILIES),
            "populations": list(POPULATIONS),
            "arms": list(ARMS),
            "sample_count_per_family_cell": SAMPLE_COUNT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "minimum_frame_count": MINIMUM_FRAME_COUNT,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "inclusion_error_maximum": INCLUSION_ERROR_MAXIMUM,
            "chart_margin_minimum": CHART_MARGIN_MINIMUM,
            "fixed_ceiling_rmse_maximum": (
                hidden.corruption.FIXED_CEILING_RMSE_MAXIMUM
            ),
            "fixed_ceiling_accuracy_minimum": (
                hidden.corruption.FIXED_CEILING_ACCURACY_MINIMUM
            ),
            "pareto_rmse_ratio_maximum": PARETO_RMSE_RATIO_MAXIMUM,
            "pareto_accuracy_delta_minimum": PARETO_ACCURACY_DELTA_MINIMUM,
            "pareto_cross_entropy_delta_maximum": (
                PARETO_CROSS_ENTROPY_DELTA_MAXIMUM
            ),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": hidden.acceleration.TASK_GATES,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "fresh_base_examples": len(cells) * SAMPLE_COUNT * 2,
            "fresh_corrupted_evaluations": len(cells) * SAMPLE_COUNT * 2,
            "closed_form_observation_only_fits": (
                len(cells)
                * SAMPLE_COUNT
                * 2
                * (hidden.source.TIME_STEPS + 1)
            ),
            "old_primary_examples_pooled": 0,
            "law_label_reads_by_primary_estimator": 0,
            "adaptive_selector_decisions": 0,
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
            "All corrective examples and corruption draws are fresh and no predecessor primary outcome is pooled.",
            "The -0.005 accuracy nondegradation guard predates the inconclusive hidden-order outcome.",
            "The common robust estimator is frozen and never reads or selects a law label.",
            "The study covers nested degree-one/degree-two laws, not non-nested or within-sequence switching dynamics.",
        ],
    }
    if not hidden._finite(result):
        raise RuntimeError("non-finite hidden-order corrective result")
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
                "common_robust_seeds": result["aggregates"][
                    "common_robust_endpoint_seed_pass_count"
                ],
                "materiality_seeds": result["aggregates"][
                    "materiality_seed_pass_count"
                ],
                "pareto_repair_seeds": result["aggregates"][
                    "pareto_repair_seed_pass_count"
                ],
                "oracle_fidelity_seeds": result["aggregates"][
                    "oracle_fidelity_seed_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
