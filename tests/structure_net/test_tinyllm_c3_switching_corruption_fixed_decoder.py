from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.structure_net import (
    tinyllm_c3_switching_corruption_fixed_decoder as study,
)


INVALID_RESULT_SHA256 = (
    "2dd8f23a2eb7bd5ad7e2c224ee8c08201f648ce905e2fd7d806d7d676badf20c"
)


def test_sources_and_preregistration_are_frozen() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "corrective_runner": study.CORRECTIVE_RUNNER_SHA256,
        "corrective_result": study.CORRECTIVE_RESULT_SHA256,
        "single_frame_corruption_runner": study.CORRUPTION_RUNNER_SHA256,
        "generator": study.GENERATOR_SHA256,
        "interval_likelihood": study.INTERVAL_SHA256,
    }


def test_exact_identifiability_boundary_and_collision() -> None:
    audit = study.identifiability_audit()
    assert audit["pass"] is True
    assert audit["late_inclusive_support"][
        "minimum_future_phase_code_distance"
    ] == 2
    assert audit["primary_support"][
        "minimum_future_phase_code_distance"
    ] == 3
    collision = audit["explicit_late_collision"]
    assert collision["equal_clean_token_frames"] == [0, 1, 2, 3, 4, 6]
    assert collision["continuous_collision_maximum_error"] <= 1e-12
    assert collision["token_collision_error_count"] == 0
    assert collision["absolute_target_separation"] >= 0.25
    assert collision["parameters_inside_composition_support"] is True


def test_lifecycle_generator_is_deterministic_and_group_typed() -> None:
    for regime in study.REGIMES:
        first = study.generate_dataset(
            regime, 997, sample_count=64, allow_pilot=True
        )
        second = study.generate_dataset(
            regime, 997, sample_count=64, allow_pilot=True
        )
        assert study.dataset_hash(first) == study.dataset_hash(second)
        assert first.dataset_seed == second.dataset_seed
        assert first.saturation_count == 0
        assert set(first.switch.tolist()) <= set(study.SWITCH_POINTS)
        assert study.group_contracts(first)["pass"] is True


def test_lifecycle_cell_exercises_oracle_and_fixed_decoder() -> None:
    audit = study.identifiability_audit()
    cell = study.analyze_cell(
        "composition",
        997,
        audit,
        sample_count=64,
        allow_pilot=True,
    )
    assert cell["valid"] is True
    assert cell["base_dataset_deterministic"] is True
    assert cell["continuous_future_equivalent_count"] == 64
    assert max(cell["continuous_prediction_maximum_errors"].values()) <= 1e-10
    assert cell["dynamics_material"] is True
    assert cell["corruption_material"] is True
    assert cell["operators"]["oracle_switch_drop"]["fixed_ceiling_pass"] is True
    assert cell["operators"]["fixed_switch_drop"]["fixed_ceiling_pass"] is True


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


def test_population_classifications_are_locked() -> None:
    audit = study.identifiability_audit()
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
    weak = _classification_cells(
        clean_global=5,
        corrupted_global=0,
        oracle=5,
        fixed=5,
        dynamics=0,
        corruption=5,
        pareto=5,
        fidelity=5,
    )
    assert study.classify(weak, True, audit)["classification"] == (
        "switching_corruption_scope_not_material"
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
        "invalid_switching_corruption_preflight"
    )


def test_invalid_primary_is_preserved_as_systems_evidence() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == INVALID_RESULT_SHA256
    assert result["status"] == "invalid"
    assert result["source_hashes"]["runner"] == (
        "5ebe462c27989e0f09f38bba8ee0e885ea5fb76190bc86c1d7143d79376e2128"
    )
    assert result["aggregates"]["classification"] == (
        "invalid_switching_corruption_preflight"
    )
    invalid = [cell for cell in result["cells"] if not cell["valid"]]
    assert {(cell["seed"], cell["regime"]) for cell in invalid} == {
        (359, "extrapolation"),
        (389, "composition"),
    }
    assert all(
        not cell["operators"]["corrupted_known_switch_no_drop"]["action_pass"]
        for cell in invalid
    )


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "identifiability_audit": {
            "late_inclusive_support": {
                "minimum_future_phase_code_distance": 2
            },
            "primary_support": {"minimum_future_phase_code_distance": 3},
        },
        "aggregates": {
            "classification": "test_classification",
            "oracle_switch_drop_seed_pass_count": 5,
            "fixed_switch_drop_seed_pass_count": 5,
            "tinyllm_training_licensed": False,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
