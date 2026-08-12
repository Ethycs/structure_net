from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.structure_net import (
    tinyllm_c3_switching_corruption_fixed_decoder_corrective as study,
)


PRIMARY_RESULT_SHA256 = (
    "20d40a54ce42a904766cab9eb533f6e2baccff79f75a2703e6bd57ff9638675b"
)
RUNNER_SHA256 = "452b0fa8f8ff54ddb0afa98e32eef9dc38da9f240eccf6a867386d9b65197939"


def test_sources_preserve_invalid_predecessor() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "invalid_runner": study.INVALID_RUNNER_SHA256,
        "invalid_result": study.INVALID_RESULT_SHA256,
        "original_preregistration": study.ORIGINAL_PREREGISTRATION_SHA256,
    }
    invalid = json.loads(study.INVALID_RESULT_PATH.read_text(encoding="utf-8"))
    assert invalid["status"] == "invalid"
    assert invalid["aggregates"]["classification"] == (
        "invalid_switching_corruption_preflight"
    )


def test_fresh_seeds_are_disjoint_from_invalid_and_pilots() -> None:
    assert set(study.SEEDS).isdisjoint(study.base.SEEDS)
    assert 991 not in study.SEEDS
    assert 997 not in study.SEEDS


def test_stabilizer_is_bounded_and_deck_stable_on_failed_cells() -> None:
    for seed, regime in ((359, "extrapolation"), (389, "composition")):
        dataset = study.base.generate_dataset(regime, seed)
        donor, frame = study.base.corruption_plan(
            regime, seed, study.base.SAMPLE_COUNT
        )
        corrupted = study.base.corruption.corrupt_frames(
            dataset.tokens, donor, frame
        )
        carrier, _ = study.base.source.analytic_carrier(
            corrupted, dataset.calibration
        )
        stabilized, displacement = study.stabilize_carrier(carrier)
        assert displacement <= study.STABILIZATION_DISPLACEMENT_MAXIMUM
        natural = study.base.known_switch_prediction(
            stabilized, dataset.switch
        )
        for element in (1, 2):
            transformed_tokens = study.base.source.apply_deck_action(
                dataset.tokens, element
            )
            transformed_corrupted = study.base.corruption.corrupt_frames(
                transformed_tokens, donor, frame
            )
            transformed_carrier, _ = study.base.source.analytic_carrier(
                transformed_corrupted, dataset.calibration
            )
            stable_transformed, transformed_displacement = (
                study.stabilize_carrier(transformed_carrier)
            )
            assert (
                transformed_displacement
                <= study.STABILIZATION_DISPLACEMENT_MAXIMUM
            )
            transformed = study.base.known_switch_prediction(
                stable_transformed, dataset.switch
            )
            assert float((transformed - natural).abs().max()) <= 2e-12


def test_corrective_lifecycle_generator_is_deterministic() -> None:
    for regime in study.REGIMES:
        first = study.generate_dataset(
            regime, 997, sample_count=64, allow_pilot=True
        )
        second = study.generate_dataset(
            regime, 997, sample_count=64, allow_pilot=True
        )
        assert study.base.dataset_hash(first) == study.base.dataset_hash(second)
        assert first.dataset_seed == second.dataset_seed
        assert first.saturation_count == 0
        assert study.base.group_contracts(first)["pass"] is True


def test_corrective_lifecycle_cell_passes_validity() -> None:
    audit = study.base.identifiability_audit()
    cell = study.analyze_cell(
        "composition",
        997,
        audit,
        sample_count=64,
        allow_pilot=True,
    )
    assert cell["valid"] is True
    assert cell["continuous_future_equivalent_count"] == 64
    assert cell["stabilization_displacements"]["maximum"] <= 1e-12
    assert all(
        arm["action_pass"] for arm in cell["operators"].values()
    )


def _classification_cells(
    *,
    clean_global: int,
    corrupted_global: int,
    oracle: int,
    fixed: int,
    dynamics: int,
    corruption: int,
    pareto: int,
    fidelity: int,
) -> list[dict]:
    cells = []
    for index, seed in enumerate(study.SEEDS):
        for regime in study.REGIMES:
            cells.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "operators": {
                        "clean_known_switch": {"fixed_ceiling_pass": True},
                        "clean_global_quadratic": {
                            "fixed_ceiling_pass": index < clean_global
                        },
                        "corrupted_global_quadratic": {
                            "fixed_ceiling_pass": index < corrupted_global
                        },
                        "corrupted_known_switch_no_drop": {
                            "fixed_ceiling_pass": False
                        },
                        "oracle_switch_drop": {
                            "fixed_ceiling_pass": index < oracle
                        },
                        "fixed_switch_drop": {
                            "fixed_ceiling_pass": index < fixed
                        },
                    },
                    "dynamics_material": index < dynamics,
                    "corruption_material": index < corruption,
                    "pareto_repair": {"pass": index < pareto},
                    "oracle_fidelity": {"pass": index < fidelity},
                }
            )
    return cells


def test_corrective_classifications_are_locked() -> None:
    audit = study.base.identifiability_audit()
    closed = _classification_cells(
        clean_global=0,
        corrupted_global=0,
        oracle=5,
        fixed=5,
        dynamics=5,
        corruption=5,
        pareto=5,
        fidelity=5,
    )
    assert study.classify(closed, True, audit)["classification"] == (
        "fixed_change_point_decoder_closes_identifiable_switching_corruption"
    )
    learned = _classification_cells(
        clean_global=0,
        corrupted_global=0,
        oracle=5,
        fixed=3,
        dynamics=5,
        corruption=5,
        pareto=3,
        fidelity=3,
    )
    decision = study.classify(learned, True, audit)
    assert decision["classification"] == (
        "recoverable_switching_exceeds_fixed_change_point_decoder"
    )
    assert decision["compact_typed_continuation_comparison_licensed"] is True
    assert decision["tinyllm_training_licensed"] is False
    assert study.classify(closed, False, audit)["classification"] == (
        "invalid_switching_corruption_corrective"
    )


def test_authoritative_corrective_establishes_oracle_fixed_gap() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == PRIMARY_RESULT_SHA256
    assert result["status"] == "completed"
    assert result["source_hashes"]["runner"] == RUNNER_SHA256
    assert result["aggregates"] == {
        "classification": "recoverable_switching_exceeds_fixed_change_point_decoder",
        "clean_global_quadratic_seed_pass_count": 0,
        "clean_known_switch_seed_pass_count": 5,
        "compact_typed_continuation_comparison_licensed": True,
        "corrupted_global_quadratic_seed_pass_count": 0,
        "corrupted_known_switch_no_drop_seed_pass_count": 0,
        "corruption_materiality_seed_pass_count": 5,
        "dynamics_materiality_seed_pass_count": 5,
        "fixed_switch_drop_seed_pass_count": 3,
        "identifiability_contract_pass": True,
        "late_support_target_identifiable_under_one_corruption": False,
        "oracle_fidelity_seed_pass_count": 1,
        "oracle_switch_drop_seed_pass_count": 5,
        "pareto_repair_seed_pass_count": 3,
        "primary_support_one_corruption_correctable": True,
        "required_seed_passes": 4,
        "tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["accounting"]["fresh_base_examples"] == 40_960
    assert result["accounting"]["invalid_primary_examples_pooled"] == 0
    assert len(result["cells"]) == 10
    assert all(cell["valid"] for cell in result["cells"])
    assert all(
        cell["continuous_future_equivalent_count"] == 4_096
        for cell in result["cells"]
    )
    assert max(
        cell["stabilization_displacements"]["maximum"]
        for cell in result["cells"]
    ) <= 1e-12
    assert max(
        operator["maximum_deck_action_error"]
        for cell in result["cells"]
        for operator in cell["operators"].values()
    ) <= 2e-12


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "oracle_switch_drop_seed_pass_count": 5,
            "fixed_switch_drop_seed_pass_count": 5,
            "oracle_fidelity_seed_pass_count": 5,
            "tinyllm_training_licensed": False,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
