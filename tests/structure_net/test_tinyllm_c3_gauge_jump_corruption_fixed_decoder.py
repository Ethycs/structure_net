from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_gauge_jump_corruption_fixed_decoder as study,
)


PRIMARY_RESULT = Path(
    "data/experiments/"
    "tinyllm_c3_gauge_jump_corruption_fixed_decoder/"
    "20260811_preregistered/result.json"
)
PRIMARY_RESULT_SHA256 = (
    "16f98f5c3cbf09fedfc18f12eca24a5fe69da46411d587c48c5d9072c912aca7"
)


def test_sources_and_preregistration_are_frozen() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "charged_corrective_runner": study.PARENT_RUNNER_SHA256,
        "charged_corrective_result": study.PARENT_RESULT_SHA256,
        "charged_corrective_report": study.PARENT_REPORT_SHA256,
    }


def test_primary_seeds_and_streams_are_fresh() -> None:
    used = {
        *study.charged.SEEDS,
        *study.charged.invalid.SEEDS,
        *study.charged.invalid.base.SEEDS,
        *study.charged.invalid.parent.SEEDS,
        *study.base.acceleration.SEEDS,
        991,
        997,
        1009,
        study.PILOT_SEED,
    }
    assert set(study.SEEDS).isdisjoint(used)
    stream_bases = [
        *study.DATASET_SEED_BASES.values(),
        *study.PILOT_DATASET_SEED_BASES.values(),
        *study.DONOR_SEED_BASES.values(),
        *study.FRAME_SEED_BASES.values(),
        *study.JUMP_TIME_SEED_BASES.values(),
        *study.JUMP_ELEMENT_SEED_BASES.values(),
        *study.SHUFFLE_SEED_BASES.values(),
    ]
    assert len(set(stream_bases)) == len(stream_bases)


def test_suffix_jump_is_exactly_invertible_and_globally_equivariant() -> None:
    dataset = study.generate_dataset(
        "composition",
        study.PILOT_SEED,
        sample_count=64,
        allow_pilot=True,
    )
    assert torch.equal(
        study.apply_suffix_jump(
            dataset.global_tokens,
            dataset.jump_time,
            dataset.jump_element,
        ),
        dataset.tokens,
    )
    assert torch.equal(
        study.apply_suffix_jump(
            dataset.tokens,
            dataset.jump_time,
            dataset.jump_element,
            inverse=True,
        ),
        dataset.global_tokens,
    )
    contracts = study.group_and_jump_contracts(dataset)
    assert contracts["pass"] is True
    assert contracts["jump_construction_token_error_count"] == 0
    assert contracts["jump_inverse_token_error_count"] == 0
    assert sum(contracts["global_jump_commutation_token_errors"].values()) == 0
    assert (
        contracts["maximum_cubic_carrier_jump_invariance_error"]
        <= study.ACTION_ERROR_MAXIMUM
    )


def test_connection_bank_is_complete_and_deterministic() -> None:
    dataset = study.generate_dataset(
        "composition",
        study.PILOT_SEED,
        sample_count=64,
        allow_pilot=True,
    )
    donor, frame = study.corruption_plan(
        "composition", study.PILOT_SEED, 64
    )
    corrupted = study.base.corruption.corrupt_frames(
        dataset.tokens, donor, frame
    )
    predictions, residuals, identities, displacement = (
        study.connection_candidates(corrupted)
    )
    assert predictions.shape == (64, 288)
    assert residuals.shape == (64, 288)
    assert identities == study.connection_identities()
    assert len(set(identities)) == 288
    assert displacement <= study.STABILIZATION_DISPLACEMENT_MAXIMUM


def test_lifecycle_cell_exercises_time_varying_gauge_path() -> None:
    cell = study.analyze_cell(
        "composition",
        study.PILOT_SEED,
        study.base.identifiability_audit(),
        sample_count=64,
        allow_pilot=True,
    )
    assert cell["valid"] is True
    assert cell["base_dataset_deterministic"] is True
    assert cell["corruption_deterministic"] is True
    assert cell["group_and_jump_contracts"]["pass"] is True
    assert all(
        contract["pass"]
        for contract in cell["connected_integer_action_contracts"].values()
    )
    assert cell["operators"]["fixed_charged_no_connection"][
        "fixed_ceiling_pass"
    ] is False
    for arm in (
        "oracle_invariant_switch_drop",
        "fixed_invariant_switch_drop",
        "oracle_charged_connection",
        "fixed_charged_connection",
    ):
        assert cell["operators"][arm]["fixed_ceiling_pass"] is True
    assert cell["connection_oracle_fidelity"]["pass"] is True
    assert cell["gauge_material_repair"] is True
    assert max(cell["continuous_prediction_maximum_errors"].values()) <= 1e-10
    assert max(
        cell["operators"][arm]["maximum_deck_action_error"]
        for arm in (
            "fixed_charged_no_connection",
            "oracle_charged_connection",
            "fixed_charged_connection",
        )
    ) == 0.0


def _classification_cells(
    *,
    invariant_oracle: int,
    invariant_fixed: int,
    no_connection: int,
    connection_oracle: int,
    connection_fixed: int,
    fidelity: int,
    materiality: int,
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
                    },
                    "connection_oracle_fidelity": {
                        "pass": index < fidelity
                    },
                    "gauge_material_repair": index < materiality,
                }
            )
    return cells


def test_classifications_and_training_stop_are_locked() -> None:
    unique = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        no_connection=0,
        connection_oracle=5,
        connection_fixed=5,
        fidelity=5,
        materiality=5,
    )
    decision = study.classify(unique, True)
    assert decision["classification"] == (
        "discrete_c3_connection_closes_time_varying_gauge_and_invariant_gap"
    )
    assert decision["compact_typed_connection_comparison_licensed"] is False
    assert decision["tinyllm_training_licensed"] is False

    both = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=4,
        no_connection=0,
        connection_oracle=5,
        connection_fixed=5,
        fidelity=5,
        materiality=5,
    )
    assert study.classify(both, True)["classification"] == (
        "discrete_c3_connection_closes_gauge_scope_without_"
        "unique_invariant_advantage"
    )

    learned = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        no_connection=0,
        connection_oracle=5,
        connection_fixed=3,
        fidelity=3,
        materiality=3,
    )
    decision = study.classify(learned, True)
    assert decision["classification"] == (
        "recoverable_time_varying_gauge_exceeds_fixed_connection_decoder"
    )
    assert decision["compact_typed_connection_comparison_licensed"] is True
    assert decision["tinyllm_training_licensed"] is False

    oracle_fails = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        no_connection=0,
        connection_oracle=3,
        connection_fixed=3,
        fidelity=3,
        materiality=3,
    )
    assert study.classify(oracle_fails, True)["classification"] == (
        "time_varying_gauge_scope_not_oracle_recoverable"
    )
    assert study.classify(unique, False)["classification"] == (
        "invalid_time_varying_gauge_fixed_decoder"
    )


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "invariant_fixed_seed_pass_count": 3,
            "connection_fixed_seed_pass_count": 5,
            "connection_oracle_fidelity_seed_pass_count": 5,
            "compact_typed_connection_comparison_licensed": False,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_authoritative_primary_result_is_valid_and_preserves_locked_decision() -> None:
    assert hashlib.sha256(PRIMARY_RESULT.read_bytes()).hexdigest() == (
        PRIMARY_RESULT_SHA256
    )
    result = json.loads(PRIMARY_RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["source_hashes"] == {
        "charged_corrective_report": study.PARENT_REPORT_SHA256,
        "charged_corrective_result": study.PARENT_RESULT_SHA256,
        "charged_corrective_runner": study.PARENT_RUNNER_SHA256,
        "preregistration": study.PREREGISTRATION_SHA256,
        "runner": (
            "5b35658103481645aba809f5575d38159dcddc9dc7330ebfa6764ad65ba170a4"
        ),
    }
    aggregates = result["aggregates"]
    assert aggregates == {
        "classification": (
            "recoverable_time_varying_gauge_exceeds_fixed_connection_decoder"
        ),
        "compact_typed_connection_comparison_licensed": True,
        "connection_fixed_seed_pass_count": 1,
        "connection_oracle_fidelity_seed_pass_count": 1,
        "connection_oracle_seed_pass_count": 5,
        "gauge_material_repair_seed_pass_count": 1,
        "invariant_fixed_seed_pass_count": 5,
        "invariant_oracle_seed_pass_count": 5,
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
        "outcome_known_audit_examples_pooled": 0,
        "parameters_changed": 0,
        "primary_observation_only_candidate_fits": 13_762_560,
        "reusable_parameter_fits": 0,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    assert all(cell["valid"] is True for cell in result["cells"])
    assert all(
        cell["operators"]["oracle_charged_connection"]["fixed_ceiling_pass"]
        is True
        for cell in result["cells"]
    )
    assert all(
        cell["operators"]["fixed_invariant_switch_drop"]["fixed_ceiling_pass"]
        is True
        for cell in result["cells"]
    )
    fixed_connection_failures = {
        (cell["seed"], cell["regime"])
        for cell in result["cells"]
        if not cell["operators"]["fixed_charged_connection"][
            "fixed_ceiling_pass"
        ]
    }
    assert fixed_connection_failures == {
        (673, "extrapolation"),
        (691, "composition"),
        (709, "composition"),
        (751, "extrapolation"),
    }
    assert max(
        operator["maximum_deck_action_error"]
        for cell in result["cells"]
        for operator in cell["operators"].values()
    ) <= study.ACTION_ERROR_MAXIMUM
