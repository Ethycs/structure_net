#!/usr/bin/env python3
"""Audit full framewise C3 gauge closure of the sufficient cubic carrier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch

from experiments.structure_net import (
    tinyllm_c3_gauge_jump_joint_typed_score as source_study,
)


prior = source_study.prior
charged = source_study.charged
base = source_study.base
source = source_study.source
SCHEMA_VERSION = "nal.tinyllm-c3-full-local-gauge-closure.v1"
HYPOTHESIS_ID = "tinyllm-c3-local-gauge-invariant-closure-v1"
EVIDENCE_ROLE = "artifact_lineage_no_training_full_local_gauge_causal_contract"
SEEDS = source_study.SEEDS
REGIMES = source_study.REGIMES
SAMPLE_COUNT = source_study.SAMPLE_COUNT
TIME_STEPS = source.TIME_STEPS
GROUP_ORDER = source.CHANNELS**TIME_STEPS
GENERATOR_ACTIONS = tuple(
    (time, element)
    for time in range(TIME_STEPS)
    for element in (1, 2)
)
ACTION_SEED_BASES = {"composition": 1_103_107, "extrapolation": 1_105_107}
SECOND_ACTION_SEED_BASES = {
    "composition": 1_107_107,
    "extrapolation": 1_109_107,
}
ACTION_ERROR_MAXIMUM = 2e-12
REQUIRED_SEED_PASSES = 4
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-full-local-gauge-closure-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "0f620df51c2f1d4278a8bbd82f8a6071c3159bdeb5e9420f4e08722c41115f90"
)
SOURCE_PREREGISTRATION_SHA256 = source_study.PREREGISTRATION_SHA256
SOURCE_RUNNER_SHA256 = (
    "6a9b1b849c97fc30bef7292ed0bbc097c4829db2920d3abe8f698e7546489944"
)
SOURCE_RESULT_PATH = source_study.PRIMARY_RESULT_PATH
SOURCE_RESULT_SHA256 = (
    "f52ce2103a07086a7118975d69f49b7cbeca01ac0ca7c5a15fd6d2a96fbc51fa"
)
SOURCE_REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score.md"
)
SOURCE_REPORT_SHA256 = (
    "281f375c069fb58b9949b7e2d0c98c895e4bfedf8a9f4e7160204e2dc3bb852b"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_full_local_gauge_closure/"
    "20260811_artifact_audit/result.json"
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


def validate_sources() -> tuple[dict[str, str], dict[str, Any]]:
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "joint_typed_score_preregistration": _sha256(
            source_study.PREREGISTRATION_PATH
        ),
        "joint_typed_score_runner": _sha256(Path(source_study.__file__)),
        "joint_typed_score_result": _sha256(SOURCE_RESULT_PATH),
        "joint_typed_score_report": _sha256(SOURCE_REPORT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "joint_typed_score_preregistration": SOURCE_PREREGISTRATION_SHA256,
        "joint_typed_score_runner": SOURCE_RUNNER_SHA256,
        "joint_typed_score_result": SOURCE_RESULT_SHA256,
        "joint_typed_score_report": SOURCE_REPORT_SHA256,
    }
    source_study.validate_sources()
    result = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        hashes != expected
        or result.get("status") != "completed"
        or result.get("aggregates", {}).get("classification")
        != "both_fixed_connection_scores_close_fresh_scope"
        or result.get("aggregates", {}).get("invariant_fixed_seed_pass_count")
        != 5
        or result.get("aggregates", {}).get("tinyllm_training_licensed")
        is not False
    ):
        raise RuntimeError("full local-gauge source contract changed")
    return hashes, result


def apply_local_action(tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1:] != (
        TIME_STEPS,
        source.CHANNELS,
    ):
        raise ValueError("expected [batch,time,channel] token tensor")
    if action.shape != tokens.shape[:2]:
        raise ValueError("local action must have shape [batch,time]")
    output = tokens.clone()
    for time in range(TIME_STEPS):
        output[:, time] = source._roll_by_example(
            output[:, time], action[:, time] % source.CHANNELS
        )
    return output


def pointwise_cube_pairs(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real, imag = charged.eisenstein_character_pairs(tokens)
    cube_real = real**3 - 9 * real * imag**2
    cube_imag = 3 * real**2 * imag - 3 * imag**3
    return cube_real, cube_imag


def invariant_decoder(
    tokens: torch.Tensor, calibration: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    invariant, magnitude = source.analytic_carrier(tokens, calibration)
    stable, displacement = charged.parent.stabilize_carrier(invariant)
    candidates, residuals, _ = base.switch_deletion_candidates(stable)
    selection = residuals.argmin(dim=1)
    row = torch.arange(tokens.shape[0], dtype=torch.int64)
    return invariant, candidates[row, selection], selection, max(
        displacement, float(magnitude.min()) * 0.0
    )


def action_stream(
    regime: str, seed_value: int, sample_count: int, *, second: bool = False
) -> torch.Tensor:
    bases = SECOND_ACTION_SEED_BASES if second else ACTION_SEED_BASES
    generator = torch.Generator(device="cpu").manual_seed(
        bases[regime] + seed_value
    )
    return torch.randint(
        0,
        source.CHANNELS,
        (sample_count, TIME_STEPS),
        dtype=torch.int64,
        generator=generator,
    )


def token_group_contracts(
    tokens: torch.Tensor, first: torch.Tensor, second: torch.Tensor
) -> dict[str, int | bool]:
    zeros = torch.zeros_like(first)
    identity_errors = int((apply_local_action(tokens, zeros) != tokens).sum())
    acted = apply_local_action(tokens, first)
    inverse_errors = int(
        (
            apply_local_action(acted, (-first) % source.CHANNELS)
            != tokens
        ).sum()
    )
    composed = apply_local_action(acted, second)
    composition_errors = int(
        (
            composed
            != apply_local_action(
                tokens, (first + second) % source.CHANNELS
            )
        ).sum()
    )
    order_three_errors = int(
        (
            apply_local_action(
                apply_local_action(acted, first), first
            )
            != tokens
        ).sum()
    )
    return {
        "identity_token_error_count": identity_errors,
        "inverse_token_error_count": inverse_errors,
        "composition_token_error_count": composition_errors,
        "order_three_token_error_count": order_three_errors,
        "pass": (
            identity_errors
            + inverse_errors
            + composition_errors
            + order_three_errors
            == 0
        ),
    }


def _action_measurement(
    tokens: torch.Tensor,
    calibration: torch.Tensor,
    action: torch.Tensor,
    base_cube: tuple[torch.Tensor, torch.Tensor],
    base_invariant: torch.Tensor,
    base_prediction: torch.Tensor,
    base_selection: torch.Tensor,
) -> dict[str, int | float | bool]:
    transformed = apply_local_action(tokens, action)
    cube = pointwise_cube_pairs(transformed)
    invariant, prediction, selection, displacement = invariant_decoder(
        transformed, calibration
    )
    cube_errors = int(
        (cube[0] != base_cube[0]).sum() + (cube[1] != base_cube[1]).sum()
    )
    carrier_error = float((invariant - base_invariant).abs().max())
    prediction_error = float((prediction - base_prediction).abs().max())
    selection_changes = int((selection != base_selection).sum())
    return {
        "exact_cube_integer_error_count": cube_errors,
        "maximum_cubic_carrier_error": carrier_error,
        "maximum_invariant_forecast_error": prediction_error,
        "selector_change_count": selection_changes,
        "stabilization_displacement": displacement,
        "pass": bool(
            cube_errors == 0
            and carrier_error <= ACTION_ERROR_MAXIMUM
            and prediction_error <= ACTION_ERROR_MAXIMUM
            and displacement <= prior.STABILIZATION_DISPLACEMENT_MAXIMUM
        ),
    }


def analyze_cell(
    regime: str,
    seed_value: int,
    source_cell: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = source_study.generate_dataset(regime, seed_value)
    donor, frame = source_study.corruption_plan(
        regime, seed_value, SAMPLE_COUNT
    )
    tokens = base.corruption.corrupt_frames(dataset.tokens, donor, frame)
    dataset_hash = prior.dataset_hash(dataset)
    corruption_hash = base._tensor_hash(tokens, donor, frame)
    source_hash_match = bool(
        dataset_hash == source_cell["base_dataset_sha256"]
        and corruption_hash == source_cell["corruption_sha256"]
    )
    base_cube = pointwise_cube_pairs(tokens)
    base_invariant, base_prediction, base_selection, base_displacement = (
        invariant_decoder(tokens, dataset.calibration)
    )
    shuffle = base.corrective.hidden.speed.sattolo_derangement(
        SAMPLE_COUNT, source_study.SHUFFLE_SEED_BASES[regime] + seed_value
    )
    identity_metrics = base.corruption._arm_metrics(
        base_prediction,
        dataset.target,
        dataset.target[shuffle],
        regime,
        {"1": 0.0, "2": 0.0},
    )
    source_operator = source_cell["operators"]["fixed_invariant_switch_drop"]
    metric_replay = bool(
        identity_metrics["scalar"] == source_operator["scalar"]
        and identity_metrics["task"] == source_operator["task"]
        and identity_metrics["shuffled_target_scalar"]
        == source_operator["shuffled_target_scalar"]
        and identity_metrics["shuffled_target_task"]
        == source_operator["shuffled_target_task"]
    )
    generator_measurements: dict[str, dict[str, int | float | bool]] = {}
    for time, element in GENERATOR_ACTIONS:
        action = torch.zeros((SAMPLE_COUNT, TIME_STEPS), dtype=torch.int64)
        action[:, time] = element
        generator_measurements[f"time_{time}_element_{element}"] = (
            _action_measurement(
                tokens,
                dataset.calibration,
                action,
                base_cube,
                base_invariant,
                base_prediction,
                base_selection,
            )
        )
    arbitrary = action_stream(regime, seed_value, SAMPLE_COUNT)
    second = action_stream(regime, seed_value, SAMPLE_COUNT, second=True)
    arbitrary_measurement = _action_measurement(
        tokens,
        dataset.calibration,
        arbitrary,
        base_cube,
        base_invariant,
        base_prediction,
        base_selection,
    )
    group_contracts = token_group_contracts(tokens, arbitrary, second)
    all_measurements = [
        *generator_measurements.values(),
        arbitrary_measurement,
    ]
    exact_algebraic_closure = bool(
        group_contracts["pass"]
        and all(item["exact_cube_integer_error_count"] == 0 for item in all_measurements)
    )
    numeric_closure = bool(all(item["pass"] for item in all_measurements))
    inherited_pass = bool(
        source_cell["valid"]
        and source_operator["fixed_ceiling_pass"]
        and identity_metrics["fixed_ceiling_pass"]
    )
    valid = bool(
        source_hash_match
        and metric_replay
        and exact_algebraic_closure
        and inherited_pass
        and base_displacement <= prior.STABILIZATION_DISPLACEMENT_MAXIMUM
        and base._finite(generator_measurements)
        and base._finite(arbitrary_measurement)
    )
    return {
        "seed": seed_value,
        "regime": regime,
        "sample_count": SAMPLE_COUNT,
        "source_dataset_sha256": dataset_hash,
        "source_corruption_sha256": corruption_hash,
        "source_hash_match": source_hash_match,
        "source_cell_valid": source_cell["valid"],
        "source_fixed_invariant_pass": source_operator[
            "fixed_ceiling_pass"
        ],
        "identity_metric_replay": metric_replay,
        "identity_fixed_ceiling_pass": identity_metrics[
            "fixed_ceiling_pass"
        ],
        "identity_scalar": identity_metrics["scalar"],
        "identity_task": identity_metrics["task"],
        "generator_measurements": generator_measurements,
        "arbitrary_local_action": arbitrary_measurement,
        "token_group_contracts": group_contracts,
        "exact_algebraic_closure": exact_algebraic_closure,
        "numeric_local_gauge_closure": numeric_closure,
        "maximum_cubic_carrier_error": max(
            float(item["maximum_cubic_carrier_error"])
            for item in all_measurements
        ),
        "maximum_invariant_forecast_error": max(
            float(item["maximum_invariant_forecast_error"])
            for item in all_measurements
        ),
        "total_selector_changes": sum(
            int(item["selector_change_count"]) for item in all_measurements
        ),
        "maximum_stabilization_displacement": max(
            base_displacement,
            *(
                float(item["stabilization_displacement"])
                for item in all_measurements
            ),
        ),
        "valid": valid,
    }


def exhaustive_pilot_cube_contract(sample_count: int = 4) -> dict[str, Any]:
    dataset = source_study.generate_dataset(
        "composition",
        source_study.PILOT_SEED,
        sample_count=64,
        allow_pilot=True,
    )
    donor, frame = source_study.corruption_plan(
        "composition", source_study.PILOT_SEED, 64
    )
    tokens = base.corruption.corrupt_frames(dataset.tokens, donor, frame)[
        :sample_count
    ]
    reference = pointwise_cube_pairs(tokens)
    errors = 0
    action_count = 0
    for vector in itertools.product(range(source.CHANNELS), repeat=TIME_STEPS):
        action = torch.tensor(vector, dtype=torch.int64).repeat(sample_count, 1)
        transformed = pointwise_cube_pairs(apply_local_action(tokens, action))
        errors += int(
            (transformed[0] != reference[0]).sum()
            + (transformed[1] != reference[1]).sum()
        )
        action_count += 1
    return {
        "sample_count": sample_count,
        "action_count": action_count,
        "expected_group_order": GROUP_ORDER,
        "exact_cube_integer_error_count": errors,
        "pass": action_count == GROUP_ORDER and errors == 0,
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    indexed = {
        (int(cell["seed"]), cell["regime"]): cell for cell in cells
    }
    inherited_count = sum(
        all(
            bool(indexed[(seed, regime)]["source_fixed_invariant_pass"])
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    local_closure_count = sum(
        all(
            bool(indexed[(seed, regime)]["numeric_local_gauge_closure"])
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    exact_count = sum(
        all(
            bool(indexed[(seed, regime)]["exact_algebraic_closure"])
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    if not valid:
        classification = "invalid_full_local_c3_gauge_audit"
    elif (
        inherited_count >= REQUIRED_SEED_PASSES
        and local_closure_count >= REQUIRED_SEED_PASSES
        and exact_count == len(SEEDS)
    ):
        classification = "pointwise_cubic_quotient_closes_full_local_c3_gauge"
    elif exact_count == len(SEEDS) and local_closure_count < REQUIRED_SEED_PASSES:
        classification = "algebraic_local_gauge_closure_numeric_implementation_defect"
    elif inherited_count < REQUIRED_SEED_PASSES:
        classification = "local_gauge_invariant_but_task_decoder_insufficient"
    else:
        classification = "invalid_full_local_c3_gauge_audit"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "inherited_fixed_invariant_seed_pass_count": inherited_count,
        "exact_local_gauge_seed_pass_count": exact_count,
        "numeric_local_gauge_seed_pass_count": local_closure_count,
        "generator_action_count": len(GENERATOR_ACTIONS),
        "certified_local_group_order": GROUP_ORDER,
        "multiple_jump_experiment_licensed": False,
        "compact_connection_model_licensed": False,
        "tinyllm_training_licensed": False,
    }


def build_result() -> dict[str, Any]:
    source_hashes, source_result = validate_sources()
    source_cells = {
        (int(cell["seed"]), cell["regime"]): cell
        for cell in source_result["cells"]
    }
    cells = [
        analyze_cell(regime, seed, source_cells[(seed, regime)])
        for seed in SEEDS
        for regime in REGIMES
    ]
    valid = bool(
        len(cells) == len(SEEDS) * len(REGIMES)
        and all(cell["valid"] for cell in cells)
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
            "source_seeds": list(SEEDS),
            "regimes": list(REGIMES),
            "sample_count_per_cell": SAMPLE_COUNT,
            "time_steps": TIME_STEPS,
            "local_group": f"C3^{TIME_STEPS}",
            "local_group_order": GROUP_ORDER,
            "generator_actions": [list(item) for item in GENERATOR_ACTIONS],
            "action_seed_bases": ACTION_SEED_BASES,
            "second_action_seed_bases": SECOND_ACTION_SEED_BASES,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "required_seed_passes": REQUIRED_SEED_PASSES,
        },
        "algebraic_certificate": {
            "action": "(g.c)_t = exp(-2*pi*i*g_t/3) c_t",
            "quotient": "q_t = c_t^3",
            "identity": "q(g.c)_t = exp(-2*pi*i*g_t) c_t^3 = q(c)_t",
            "generator_count": len(GENERATOR_ACTIONS),
            "finite_group_order": GROUP_ORDER,
            "multiple_suffix_jumps_are_local_group_elements": True,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "source_examples_reused": len(cells) * SAMPLE_COUNT,
            "fresh_examples": 0,
            "generator_counterfactual_evaluations": (
                len(cells) * SAMPLE_COUNT * len(GENERATOR_ACTIONS)
            ),
            "arbitrary_local_counterfactual_evaluations": (
                len(cells) * SAMPLE_COUNT
            ),
            "models_instantiated": 0,
            "checkpoints_loaded": 0,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "reusable_parameter_fits": 0,
            "target_using_fits": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "This is a causal group-action audit on sealed source cohorts, not a new population-performance experiment.",
            "The conclusion applies only while the task computation factors through the sufficient pointwise cubic carrier.",
            "Connections may remain necessary for charged relational targets or representations that do not descend pointwise.",
            "No result licenses a multiple-jump study, compact connection model, or TinyLLM training under the same law.",
        ],
    }
    if not base._finite(result):
        raise RuntimeError("non-finite full local-gauge result")
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
                "inherited_fixed_seed_passes": result["aggregates"][
                    "inherited_fixed_invariant_seed_pass_count"
                ],
                "numeric_local_gauge_seed_passes": result["aggregates"][
                    "numeric_local_gauge_seed_pass_count"
                ],
                "certified_local_group_order": result["aggregates"][
                    "certified_local_group_order"
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
