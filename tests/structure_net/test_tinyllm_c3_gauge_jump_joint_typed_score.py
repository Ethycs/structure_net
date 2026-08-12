from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_gauge_jump_joint_typed_score as study,
)


PRIMARY_RESULT = Path(
    "data/experiments/tinyllm_c3_gauge_jump_joint_typed_score/"
    "20260811_preregistered/result.json"
)
PRIMARY_RESULT_SHA256 = (
    "f52ce2103a07086a7118975d69f49b7cbeca01ac0ca7c5a15fd6d2a96fbc51fa"
)


def test_sources_are_frozen_and_predecessor_licenses_comparison() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "gauge_jump_primary_preregistration": (
            study.PRIOR_PREREGISTRATION_SHA256
        ),
        "gauge_jump_primary_runner": study.PRIOR_RUNNER_SHA256,
        "gauge_jump_primary_result": study.PRIOR_RESULT_SHA256,
        "gauge_jump_primary_report": study.PRIOR_REPORT_SHA256,
    }


def test_fresh_seeds_and_streams_are_disjoint() -> None:
    inherited_seeds = {
        *study.prior.SEEDS,
        *study.prior.charged.SEEDS,
        *study.prior.charged.invalid.SEEDS,
        *study.prior.charged.invalid.base.SEEDS,
        *study.prior.charged.invalid.parent.SEEDS,
        study.prior.PILOT_SEED,
    }
    assert set(study.SEEDS).isdisjoint(inherited_seeds)
    stream_bases = [
        *study.DATASET_SEED_BASES.values(),
        *study.PILOT_DATASET_SEED_BASES.values(),
        *study.DONOR_SEED_BASES.values(),
        *study.FRAME_SEED_BASES.values(),
        *study.JUMP_TIME_SEED_BASES.values(),
        *study.JUMP_ELEMENT_SEED_BASES.values(),
        *study.SHUFFLE_SEED_BASES.values(),
    ]
    inherited_stream_bases = {
        *study.prior.DATASET_SEED_BASES.values(),
        *study.prior.PILOT_DATASET_SEED_BASES.values(),
        *study.prior.DONOR_SEED_BASES.values(),
        *study.prior.FRAME_SEED_BASES.values(),
        *study.prior.JUMP_TIME_SEED_BASES.values(),
        *study.prior.JUMP_ELEMENT_SEED_BASES.values(),
        *study.prior.SHUFFLE_SEED_BASES.values(),
    }
    assert len(stream_bases) == len(set(stream_bases))
    assert set(stream_bases).isdisjoint(inherited_stream_bases)


def test_joint_score_uses_declared_variance_normalization() -> None:
    invariant = torch.tensor([[9.0, 18.0]], dtype=torch.float64)
    charged = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    score = study._joint_score(
        invariant,
        [(0, 2), (1, 3)],
        charged,
        [(0, 2, 1, 1), (1, 3, 1, 1), (0, 2, 2, 2)],
    )
    assert study.JOINT_INVARIANT_WEIGHT == 1.0 / 9.0
    assert torch.equal(score, torch.tensor([[2.0, 4.0, 4.0]]))


def test_lifecycle_cell_exercises_joint_typed_path_and_restores_parent() -> None:
    original_seeds = study.prior.SEEDS
    cell = study.analyze_cell(
        "composition",
        study.PILOT_SEED,
        study.base.identifiability_audit(),
        sample_count=64,
        allow_pilot=True,
    )
    assert study.prior.SEEDS == original_seeds
    assert cell["valid"] is True
    assert cell["joint_invariant_weight"] == 1.0 / 9.0
    assert cell["operators"]["fixed_charged_no_connection"][
        "fixed_ceiling_pass"
    ] is False
    for arm in (
        "oracle_invariant_switch_drop",
        "fixed_invariant_switch_drop",
        "oracle_charged_connection",
        "fixed_charged_connection",
        "fixed_joint_typed_connection",
    ):
        assert cell["operators"][arm]["fixed_ceiling_pass"] is True
    assert cell["joint_typed_oracle_fidelity"]["pass"] is True
    assert cell["operators"]["fixed_joint_typed_connection"][
        "maximum_deck_action_error"
    ] == 0.0
    assert cell["continuous_prediction_maximum_errors"][
        "fixed_joint_typed_connection"
    ] <= study.prior.CONTINUOUS_ERROR_MAXIMUM


def _cells(
    *,
    invariant_oracle: int = 5,
    invariant_fixed: int = 5,
    no_connection: int = 0,
    connection_oracle: int = 5,
    connection_fixed: int = 1,
    typed: int = 5,
    typed_fidelity: int = 5,
    typed_materiality: int = 4,
) -> list[dict]:
    cells = []
    for index, seed in enumerate(study.SEEDS):
        for regime in study.REGIMES:
            cells.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "operators": {
                        "oracle_invariant_switch_drop": {
                            "fixed_ceiling_pass": index < invariant_oracle
                        },
                        "fixed_invariant_switch_drop": {
                            "fixed_ceiling_pass": index < invariant_fixed
                        },
                        "fixed_charged_no_connection": {
                            "fixed_ceiling_pass": index < no_connection
                        },
                        "oracle_charged_connection": {
                            "fixed_ceiling_pass": index < connection_oracle
                        },
                        "fixed_charged_connection": {
                            "fixed_ceiling_pass": index < connection_fixed
                        },
                        "fixed_joint_typed_connection": {
                            "fixed_ceiling_pass": index < typed
                        },
                    },
                    "joint_typed_oracle_fidelity": {
                        "pass": index < typed_fidelity
                    },
                    "joint_typed_material_repair": index < typed_materiality,
                }
            )
    return cells


def test_classifications_and_training_stop_are_locked() -> None:
    closed = study.classify(_cells(), True)
    assert closed["classification"] == (
        "joint_typed_score_closes_time_varying_gauge_connection_tail"
    )
    assert closed["compact_typed_chart_mixture_licensed"] is False
    assert closed["tinyllm_training_licensed"] is False

    both = study.classify(
        _cells(connection_fixed=5, typed_materiality=0), True
    )
    assert both["classification"] == "both_fixed_connection_scores_close_fresh_scope"

    mixture = study.classify(
        _cells(typed=3, typed_fidelity=3, typed_materiality=3), True
    )
    assert mixture["classification"] == (
        "time_varying_gauge_requires_compact_typed_chart_mixture"
    )
    assert mixture["compact_typed_chart_mixture_licensed"] is True
    assert mixture["tinyllm_training_licensed"] is False

    assert study.classify(_cells(connection_oracle=3), True)[
        "classification"
    ] == "fresh_time_varying_gauge_not_oracle_recoverable"
    assert study.classify(_cells(), False)["classification"] == (
        "invalid_joint_typed_connection_score"
    )


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "connection_fixed_seed_pass_count": 1,
            "joint_typed_seed_pass_count": 5,
            "joint_typed_oracle_fidelity_seed_pass_count": 5,
            "compact_typed_chart_mixture_licensed": False,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_authoritative_result_closes_fresh_scope_conservatively() -> None:
    assert hashlib.sha256(PRIMARY_RESULT.read_bytes()).hexdigest() == (
        PRIMARY_RESULT_SHA256
    )
    result = json.loads(PRIMARY_RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["source_hashes"] == {
        "gauge_jump_primary_preregistration": (
            study.PRIOR_PREREGISTRATION_SHA256
        ),
        "gauge_jump_primary_report": study.PRIOR_REPORT_SHA256,
        "gauge_jump_primary_result": study.PRIOR_RESULT_SHA256,
        "gauge_jump_primary_runner": study.PRIOR_RUNNER_SHA256,
        "preregistration": study.PREREGISTRATION_SHA256,
        "runner": (
            "6a9b1b849c97fc30bef7292ed0bbc097c4829db2920d3abe8f698e7546489944"
        ),
    }
    assert result["aggregates"] == {
        "classification": "both_fixed_connection_scores_close_fresh_scope",
        "compact_typed_chart_mixture_licensed": False,
        "connection_fixed_seed_pass_count": 4,
        "connection_oracle_seed_pass_count": 5,
        "invariant_fixed_seed_pass_count": 5,
        "invariant_oracle_seed_pass_count": 5,
        "joint_typed_material_repair_seed_pass_count": 0,
        "joint_typed_oracle_fidelity_seed_pass_count": 5,
        "joint_typed_seed_pass_count": 5,
        "no_connection_seed_pass_count": 0,
        "required_seed_passes": 4,
        "tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["accounting"] == {
        "checkpoints_loaded": 0,
        "continuous_validation_candidate_fits": 12_779_520,
        "fresh_base_examples": 40_960,
        "fresh_corrupted_evaluations": 40_960,
        "models_instantiated": 0,
        "optimizer_steps": 0,
        "outcome_known_diagnostic_examples_pooled": 0,
        "parameters_changed": 0,
        "primary_observation_only_candidate_fits": 13_762_560,
        "reusable_parameter_fits": 0,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    assert all(cell["valid"] is True for cell in result["cells"])
    assert all(
        cell["operators"]["fixed_joint_typed_connection"][
            "fixed_ceiling_pass"
        ]
        is True
        and cell["joint_typed_oracle_fidelity"]["pass"] is True
        and cell["operators"]["fixed_joint_typed_connection"][
            "maximum_deck_action_error"
        ]
        == 0.0
        for cell in result["cells"]
    )
    original_failures = {
        (cell["seed"], cell["regime"])
        for cell in result["cells"]
        if not cell["operators"]["fixed_charged_connection"][
            "fixed_ceiling_pass"
        ]
    }
    assert original_failures == {(821, "composition")}
    assert max(
        cell["continuous_prediction_maximum_errors"][
            "fixed_joint_typed_connection"
        ]
        for cell in result["cells"]
    ) <= study.prior.CONTINUOUS_ERROR_MAXIMUM
