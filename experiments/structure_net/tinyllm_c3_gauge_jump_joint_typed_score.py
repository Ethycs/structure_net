#!/usr/bin/env python3
"""Test a fixed joint invariant/charged score under one hidden C3 gauge jump."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterator, Mapping, Sequence

import torch

from experiments.structure_net import (
    tinyllm_c3_gauge_jump_corruption_fixed_decoder as prior,
)


charged = prior.charged
base = prior.base
source = prior.source
SCHEMA_VERSION = "nal.tinyllm-c3-gauge-jump-joint-typed-score.v1"
HYPOTHESIS_ID = "tinyllm-c3-gauge-jump-joint-typed-score-v1"
EVIDENCE_ROLE = "fresh_prospective_no_training_joint_typed_score_replication"
SEEDS = (773, 821, 1003, 1031, 1039)
REGIMES = prior.REGIMES
SWITCH_POINTS = prior.SWITCH_POINTS
JUMP_TIMES = prior.JUMP_TIMES
JUMP_ELEMENTS = prior.JUMP_ELEMENTS
ARMS = (*prior.ARMS, "fixed_joint_typed_connection")
SAMPLE_COUNT = prior.SAMPLE_COUNT
JOINT_INVARIANT_WEIGHT = 1.0 / 9.0
DATASET_SEED_BASES = {"composition": 1_075_107, "extrapolation": 1_077_107}
PILOT_DATASET_SEED_BASES = {
    "composition": 1_099_107,
    "extrapolation": 1_101_107,
}
DONOR_SEED_BASES = {"composition": 1_079_107, "extrapolation": 1_081_107}
FRAME_SEED_BASES = {"composition": 1_083_107, "extrapolation": 1_085_107}
JUMP_TIME_SEED_BASES = {"composition": 1_087_107, "extrapolation": 1_089_107}
JUMP_ELEMENT_SEED_BASES = {
    "composition": 1_091_107,
    "extrapolation": 1_093_107,
}
SHUFFLE_SEED_BASES = {"composition": 1_095_107, "extrapolation": 1_097_107}
PILOT_SEED = 1117
REQUIRED_SEED_PASSES = 4
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "e4629a11cac991b1bd64d641f3276b4517296ee31ac3b9e0a3837e5cb5ce4663"
)
PRIOR_PREREGISTRATION_SHA256 = prior.PREREGISTRATION_SHA256
PRIOR_RUNNER_SHA256 = (
    "5b35658103481645aba809f5575d38159dcddc9dc7330ebfa6764ad65ba170a4"
)
PRIOR_RESULT_PATH = prior.PRIMARY_RESULT_PATH
PRIOR_RESULT_SHA256 = (
    "16f98f5c3cbf09fedfc18f12eca24a5fe69da46411d587c48c5d9072c912aca7"
)
PRIOR_REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-gauge-jump-corruption-fixed-decoder.md"
)
PRIOR_REPORT_SHA256 = (
    "2c2fb44afaf9bd35a63cd82d4331943bdbbfa1a2c366080f50c5a5e8fe0bb6b1"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_gauge_jump_joint_typed_score/"
    "20260811_preregistered/result.json"
)


_PRIOR_DECODER_PREDICTIONS = prior.decoder_predictions
_PRIOR_CONTINUOUS_VALIDATION = prior.continuous_validation


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
        "gauge_jump_primary_preregistration": _sha256(
            prior.PREREGISTRATION_PATH
        ),
        "gauge_jump_primary_runner": _sha256(Path(prior.__file__)),
        "gauge_jump_primary_result": _sha256(PRIOR_RESULT_PATH),
        "gauge_jump_primary_report": _sha256(PRIOR_REPORT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "gauge_jump_primary_preregistration": PRIOR_PREREGISTRATION_SHA256,
        "gauge_jump_primary_runner": PRIOR_RUNNER_SHA256,
        "gauge_jump_primary_result": PRIOR_RESULT_SHA256,
        "gauge_jump_primary_report": PRIOR_REPORT_SHA256,
    }
    prior.validate_sources()
    predecessor = json.loads(PRIOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        hashes != expected
        or predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != "recoverable_time_varying_gauge_exceeds_fixed_connection_decoder"
        or predecessor.get("aggregates", {}).get(
            "compact_typed_connection_comparison_licensed"
        )
        is not True
        or predecessor.get("aggregates", {}).get("tinyllm_training_licensed")
        is not False
    ):
        raise RuntimeError("joint typed-score source contract changed")
    return hashes


def _joint_score(
    invariant_residuals: torch.Tensor,
    invariant_identities: Sequence[tuple[int, int]],
    charged_residuals: torch.Tensor,
    connection_identities: Sequence[tuple[int, int, int, int]],
) -> torch.Tensor:
    invariant_index = {
        identity: offset
        for offset, identity in enumerate(invariant_identities)
    }
    physical_index = torch.tensor(
        [invariant_index[identity[:2]] for identity in connection_identities],
        dtype=torch.int64,
    )
    return (
        charged_residuals
        + JOINT_INVARIANT_WEIGHT * invariant_residuals[:, physical_index]
    )


def decoder_predictions(
    invariant: torch.Tensor,
    tokens: torch.Tensor,
    true_switch: torch.Tensor,
    true_frame: torch.Tensor,
    true_jump_time: torch.Tensor,
    true_jump_element: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    stable_invariant, invariant_displacement = charged.parent.stabilize_carrier(
        invariant
    )
    invariant_candidates, invariant_residuals, invariant_identities = (
        base.switch_deletion_candidates(stable_invariant)
    )
    (
        no_connection_candidates,
        no_connection_residuals,
        _,
        no_connection_displacement,
    ) = charged.exact_charged_deletion_candidates(tokens)
    (
        connected_candidates,
        connected_residuals,
        connection_identities,
        connection_displacement,
    ) = prior.connection_candidates(tokens)
    invariant_index = {
        identity: offset
        for offset, identity in enumerate(invariant_identities)
    }
    connection_index = {
        identity: offset
        for offset, identity in enumerate(connection_identities)
    }
    oracle_invariant = torch.tensor(
        [
            invariant_index[(int(frame), int(switch))]
            for frame, switch in zip(true_frame, true_switch)
        ],
        dtype=torch.int64,
    )
    oracle_connection = torch.tensor(
        [
            connection_index[(
                int(frame),
                int(switch),
                int(jump_time),
                int(jump_element),
            )]
            for frame, switch, jump_time, jump_element in zip(
                true_frame,
                true_switch,
                true_jump_time,
                true_jump_element,
            )
        ],
        dtype=torch.int64,
    )
    fixed_invariant = invariant_residuals.argmin(dim=1)
    fixed_no_connection = no_connection_residuals.argmin(dim=1)
    fixed_connection = connected_residuals.argmin(dim=1)
    joint_residuals = _joint_score(
        invariant_residuals,
        invariant_identities,
        connected_residuals,
        connection_identities,
    )
    fixed_joint = joint_residuals.argmin(dim=1)
    row = torch.arange(invariant.shape[0], dtype=torch.int64)
    predictions = {
        "oracle_invariant_switch_drop": invariant_candidates[
            row, oracle_invariant
        ],
        "fixed_invariant_switch_drop": invariant_candidates[
            row, fixed_invariant
        ],
        "fixed_charged_no_connection": no_connection_candidates[
            row, fixed_no_connection
        ],
        "oracle_charged_connection": connected_candidates[
            row, oracle_connection
        ],
        "fixed_charged_connection": connected_candidates[
            row, fixed_connection
        ],
        "fixed_joint_typed_connection": connected_candidates[
            row, fixed_joint
        ],
    }
    selections = {
        "oracle_invariant": oracle_invariant,
        "fixed_invariant": fixed_invariant,
        "fixed_no_connection": fixed_no_connection,
        "oracle_connection": oracle_connection,
        "fixed_connection": fixed_connection,
        "fixed_joint_typed": fixed_joint,
    }
    return predictions, selections, {
        "invariant": invariant_displacement,
        "no_connection": no_connection_displacement,
        "connection": connection_displacement,
    }


def continuous_validation(
    dataset: prior.GaugeJumpDataset,
    donor: torch.Tensor,
    frame: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    exact_theta = base.phase_path(
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        steps=source.TIME_STEPS + 1,
    )
    deck_phase = 2.0 * math.pi * dataset.deck.double() / source.CHANNELS
    global_charged = torch.polar(
        torch.ones_like(exact_theta[:, :-1]),
        exact_theta[:, :-1] - deck_phase[:, None],
    ).to(torch.complex128)
    jumped_charged = prior.apply_complex_suffix_jump(
        global_charged, dataset.jump_time, dataset.jump_element
    )
    corrupted_charged = base.corruption.corrupt_frames(
        jumped_charged, donor, frame
    )
    corrupted_invariant = corrupted_charged.pow(3)
    stable_invariant, invariant_displacement = charged.parent.stabilize_carrier(
        corrupted_invariant
    )
    invariant_candidates, invariant_residuals, invariant_identities = (
        base.switch_deletion_candidates(stable_invariant)
    )
    (
        connected_candidates,
        connected_residuals,
        connection_identities,
        connection_displacement,
    ) = prior.floating_connection_candidates(corrupted_charged)
    invariant_index = {
        identity: offset
        for offset, identity in enumerate(invariant_identities)
    }
    connection_index = {
        identity: offset
        for offset, identity in enumerate(connection_identities)
    }
    oracle_invariant = torch.tensor(
        [
            invariant_index[(int(item_frame), int(item_switch))]
            for item_frame, item_switch in zip(frame, dataset.switch)
        ],
        dtype=torch.int64,
    )
    oracle_connection = torch.tensor(
        [
            connection_index[(
                int(item_frame),
                int(item_switch),
                int(item_jump_time),
                int(item_jump_element),
            )]
            for item_frame, item_switch, item_jump_time, item_jump_element in zip(
                frame,
                dataset.switch,
                dataset.jump_time,
                dataset.jump_element,
            )
        ],
        dtype=torch.int64,
    )
    joint_residuals = _joint_score(
        invariant_residuals,
        invariant_identities,
        connected_residuals,
        connection_identities,
    )
    row = torch.arange(dataset.target.shape[0], dtype=torch.int64)
    predictions = {
        "oracle_invariant_switch_drop": invariant_candidates[
            row, oracle_invariant
        ],
        "fixed_invariant_switch_drop": invariant_candidates[
            row, invariant_residuals.argmin(dim=1)
        ],
        "oracle_charged_connection": connected_candidates[
            row, oracle_connection
        ],
        "fixed_charged_connection": connected_candidates[
            row, connected_residuals.argmin(dim=1)
        ],
        "fixed_joint_typed_connection": connected_candidates[
            row, joint_residuals.argmin(dim=1)
        ],
    }
    exact_future = torch.polar(
        torch.ones_like(exact_theta[:, -1]), 3.0 * exact_theta[:, -1]
    ).to(torch.complex128)
    return (
        {
            arm: float((prediction - exact_future).abs().max())
            for arm, prediction in predictions.items()
        },
        {
            "invariant": invariant_displacement,
            "connection": connection_displacement,
        },
    )


@contextmanager
def _fresh_parent_scope() -> Iterator[None]:
    replacements = {
        "SEEDS": SEEDS,
        "ARMS": ARMS,
        "DATASET_SEED_BASES": DATASET_SEED_BASES,
        "PILOT_DATASET_SEED_BASES": PILOT_DATASET_SEED_BASES,
        "DONOR_SEED_BASES": DONOR_SEED_BASES,
        "FRAME_SEED_BASES": FRAME_SEED_BASES,
        "JUMP_TIME_SEED_BASES": JUMP_TIME_SEED_BASES,
        "JUMP_ELEMENT_SEED_BASES": JUMP_ELEMENT_SEED_BASES,
        "SHUFFLE_SEED_BASES": SHUFFLE_SEED_BASES,
        "PILOT_SEED": PILOT_SEED,
        "decoder_predictions": decoder_predictions,
        "continuous_validation": continuous_validation,
    }
    originals = {name: getattr(prior, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(prior, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(prior, name, value)


def generate_dataset(
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> prior.GaugeJumpDataset:
    with _fresh_parent_scope():
        return prior.generate_dataset(
            regime,
            seed_value,
            sample_count=sample_count,
            allow_pilot=allow_pilot,
        )


def corruption_plan(
    regime: str, seed_value: int, sample_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    with _fresh_parent_scope():
        return prior.corruption_plan(regime, seed_value, sample_count)


def analyze_cell(
    regime: str,
    seed_value: int,
    audit: Mapping[str, Any],
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    with _fresh_parent_scope():
        cell = prior.analyze_cell(
            regime,
            seed_value,
            audit,
            sample_count=sample_count,
            allow_pilot=allow_pilot,
        )
    typed = cell["operators"]["fixed_joint_typed_connection"]
    oracle = cell["operators"]["oracle_charged_connection"]
    typed_fidelity = prior._fidelity(typed, oracle)
    typed_material_repair = bool(
        not cell["operators"]["fixed_charged_connection"][
            "fixed_ceiling_pass"
        ]
        and typed["fixed_ceiling_pass"]
    )
    cell["joint_typed_oracle_fidelity"] = typed_fidelity
    cell["joint_typed_material_repair"] = typed_material_repair
    cell["joint_invariant_weight"] = JOINT_INVARIANT_WEIGHT
    cell["valid"] = bool(
        cell["valid"]
        and prior.base._finite(typed_fidelity)
        and typed["maximum_deck_action_error"] == 0.0
        and cell["continuous_prediction_maximum_errors"][
            "fixed_joint_typed_connection"
        ]
        <= prior.CONTINUOUS_ERROR_MAXIMUM
    )
    return cell


def classify(
    cells: Sequence[Mapping[str, Any]], valid: bool
) -> dict[str, Any]:
    indexed = {
        (int(cell["seed"]), cell["regime"]): cell for cell in cells
    }

    def arm_count(arm: str) -> int:
        return sum(
            all(
                bool(indexed[(seed, regime)]["operators"][arm][
                    "fixed_ceiling_pass"
                ])
                for regime in REGIMES
            )
            for seed in SEEDS
        )

    invariant_oracle_count = arm_count("oracle_invariant_switch_drop")
    invariant_fixed_count = arm_count("fixed_invariant_switch_drop")
    no_connection_count = arm_count("fixed_charged_no_connection")
    connection_oracle_count = arm_count("oracle_charged_connection")
    connection_fixed_count = arm_count("fixed_charged_connection")
    typed_count = arm_count("fixed_joint_typed_connection")
    typed_fidelity_count = sum(
        all(
            bool(indexed[(seed, regime)]["joint_typed_oracle_fidelity"]["pass"])
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    typed_materiality_count = sum(
        all(
            bool(indexed[(seed, regime)]["joint_typed_material_repair"])
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    oracles_pass = bool(
        invariant_oracle_count >= REQUIRED_SEED_PASSES
        and connection_oracle_count >= REQUIRED_SEED_PASSES
    )
    typed_pass = bool(
        typed_count >= REQUIRED_SEED_PASSES
        and typed_fidelity_count >= REQUIRED_SEED_PASSES
        and no_connection_count <= len(SEEDS) - REQUIRED_SEED_PASSES
    )
    if not valid:
        classification = "invalid_joint_typed_connection_score"
    elif (
        oracles_pass
        and typed_pass
        and connection_fixed_count < REQUIRED_SEED_PASSES
        and typed_materiality_count >= REQUIRED_SEED_PASSES
    ):
        classification = (
            "joint_typed_score_closes_time_varying_gauge_connection_tail"
        )
    elif (
        oracles_pass
        and typed_pass
        and connection_fixed_count >= REQUIRED_SEED_PASSES
    ):
        classification = "both_fixed_connection_scores_close_fresh_scope"
    elif (
        connection_oracle_count >= REQUIRED_SEED_PASSES
        and connection_fixed_count < REQUIRED_SEED_PASSES
        and typed_count < REQUIRED_SEED_PASSES
    ):
        classification = "time_varying_gauge_requires_compact_typed_chart_mixture"
    elif not oracles_pass:
        classification = "fresh_time_varying_gauge_not_oracle_recoverable"
    else:
        classification = "inconclusive_joint_typed_connection_score"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "invariant_oracle_seed_pass_count": invariant_oracle_count,
        "invariant_fixed_seed_pass_count": invariant_fixed_count,
        "no_connection_seed_pass_count": no_connection_count,
        "connection_oracle_seed_pass_count": connection_oracle_count,
        "connection_fixed_seed_pass_count": connection_fixed_count,
        "joint_typed_seed_pass_count": typed_count,
        "joint_typed_oracle_fidelity_seed_pass_count": typed_fidelity_count,
        "joint_typed_material_repair_seed_pass_count": typed_materiality_count,
        "compact_typed_chart_mixture_licensed": classification
        == "time_varying_gauge_requires_compact_typed_chart_mixture",
        "tinyllm_training_licensed": False,
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
            "switch_points": list(SWITCH_POINTS),
            "jump_times": list(JUMP_TIMES),
            "jump_elements": list(JUMP_ELEMENTS),
            "arms": list(ARMS),
            "sample_count_per_cell": SAMPLE_COUNT,
            "joint_score": "charged_residual + invariant_residual / 9",
            "joint_invariant_weight": JOINT_INVARIANT_WEIGHT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "donor_seed_bases": DONOR_SEED_BASES,
            "frame_seed_bases": FRAME_SEED_BASES,
            "jump_time_seed_bases": JUMP_TIME_SEED_BASES,
            "jump_element_seed_bases": JUMP_ELEMENT_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "minimum_switch_count": prior.MINIMUM_SWITCH_COUNT,
            "minimum_frame_count": prior.MINIMUM_FRAME_COUNT,
            "minimum_jump_time_count": prior.MINIMUM_JUMP_TIME_COUNT,
            "minimum_jump_element_count": prior.MINIMUM_JUMP_ELEMENT_COUNT,
            "minimum_joint_jump_count": prior.MINIMUM_JOINT_JUMP_COUNT,
            "action_error_maximum": prior.ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": prior.CONTINUOUS_ERROR_MAXIMUM,
            "chart_margin_minimum": prior.CHART_MARGIN_MINIMUM,
            "stabilization_displacement_maximum": (
                prior.STABILIZATION_DISPLACEMENT_MAXIMUM
            ),
            "fixed_ceiling_rmse_maximum": (
                base.corruption.FIXED_CEILING_RMSE_MAXIMUM
            ),
            "fixed_ceiling_accuracy_minimum": (
                base.corruption.FIXED_CEILING_ACCURACY_MINIMUM
            ),
            "oracle_rmse_slack_maximum": (
                charged.invalid.ORACLE_RMSE_SLACK_MAXIMUM
            ),
            "oracle_accuracy_slack_maximum": (
                charged.invalid.ORACLE_ACCURACY_SLACK_MAXIMUM
            ),
            "oracle_cross_entropy_slack_maximum": (
                charged.invalid.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
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
            "primary_observation_only_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 336
            ),
            "continuous_validation_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 312
            ),
            "outcome_known_diagnostic_examples_pooled": 0,
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
            "device": "cpu",
        },
        "method_boundaries": [
            "The score and one-ninth weight were frozen after an explicitly disclosed outcome-known diagnostic.",
            "Fresh seeds and all fresh generator streams are disjoint from the diagnostic and predecessor.",
            "The score has no fitted or target-using parameter and reuses one shared candidate bank.",
            "No result licenses unrestricted TinyLLM training.",
        ],
    }
    if not base._finite(result):
        raise RuntimeError("non-finite joint typed-score result")
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
                "original_fixed_seed_passes": result["aggregates"][
                    "connection_fixed_seed_pass_count"
                ],
                "joint_typed_seed_passes": result["aggregates"][
                    "joint_typed_seed_pass_count"
                ],
                "joint_typed_fidelity_seed_passes": result["aggregates"][
                    "joint_typed_oracle_fidelity_seed_pass_count"
                ],
                "compact_typed_chart_mixture_licensed": result["aggregates"][
                    "compact_typed_chart_mixture_licensed"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
