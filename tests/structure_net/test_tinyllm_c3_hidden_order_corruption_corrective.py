from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.structure_net import (
    tinyllm_c3_hidden_order_corruption_corrective as study,
)


PRIMARY_RESULT_SHA256 = (
    "bd5af6550e65c0b9048030a8f6d01cf80e06e9f027edb89066d8338bb8b47c2a"
)
RUNNER_SHA256 = "6d8e408f2347b86440ba6307345b86cad9f2740ee0945c358dea122d1c8d2789"


def test_sources_and_corrective_preregistration_are_frozen() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "hidden_order_runner": study.HIDDEN_RUNNER_SHA256,
        "hidden_order_result": study.HIDDEN_RESULT_SHA256,
    }


def test_corrective_seeds_are_disjoint_from_predecessor() -> None:
    fresh = {seed for pair in study.PAIR_SEEDS for seed in pair}
    predecessor = {seed for pair in study.hidden.PAIR_SEEDS for seed in pair}
    assert fresh.isdisjoint(predecessor)
    assert 991 not in fresh
    assert 997 not in fresh


def test_fresh_generators_are_deterministic() -> None:
    for family in study.FAMILIES:
        first = study.generate_dataset(
            family, "composition", 997, sample_count=64, allow_pilot=True
        )
        second = study.generate_dataset(
            family, "composition", 997, sample_count=64, allow_pilot=True
        )
        assert study.hidden._dataset_hash(
            family, first
        ) == study.hidden._dataset_hash(family, second)
        assert first.dataset_seed == second.dataset_seed
        assert first.saturation_count == 0


def test_disjoint_pilot_exercises_corrective_joint_path() -> None:
    cell = study.analyze_cell(
        1, "composition", sample_count=64, allow_pilot=True
    )
    assert cell["valid"] is True
    assert cell["common_robust_endpoint_pass"] is True
    assert cell["materiality_pass"] is True
    assert cell["pareto_repair_pass"] is True
    assert cell["oracle_fidelity_pass"] is True
    assert cell["populations"]["mixture"]["sample_count"] == 128
    for family in study.FAMILIES:
        population = cell["populations"][family]
        assert population["base_dataset_deterministic"] is True
        assert population["continuous_law_inclusion_maximum_error"] <= 1e-10
        assert population["continuous_corruption_index_recovery_count"] == 64


def test_pareto_repair_gate_is_joint_and_inclusive() -> None:
    operators = {
        "clean_law_specific": {"fixed_ceiling_pass": True},
        "corrupted_law_specific": {
            "fixed_ceiling_pass": False,
            "scalar": {"rmse": 0.2},
            "task": {"exact_bin_accuracy": 0.90, "target_cross_entropy": 2.0},
        },
        "oracle_drop_one_quadratic": {
            "scalar": {"rmse": 0.01},
            "task": {"exact_bin_accuracy": 0.90, "target_cross_entropy": 1.8},
        },
        "robust_drop_one_quadratic": {
            "fixed_ceiling_pass": True,
            "scalar": {"rmse": 0.10},
            "task": {
                "exact_bin_accuracy": 0.895,
                "target_cross_entropy": 1.90,
            },
        },
    }
    comparisons = study._corrective_comparisons(operators)
    assert comparisons["pareto_repair"]["pass"] is True
    operators["robust_drop_one_quadratic"]["task"][
        "exact_bin_accuracy"
    ] = 0.89499
    assert study._corrective_comparisons(operators)["pareto_repair"][
        "pass"
    ] is False


def _classification_cells(
    *,
    common: int,
    oracle: int,
    material: int,
    pareto: int,
    fidelity: int,
    naive: int,
) -> list[dict]:
    cells = []
    for index in range(5):
        for regime in study.REGIMES:
            cells.append(
                {
                    "replicate": index + 1,
                    "regime": regime,
                    "common_robust_endpoint_pass": index < common,
                    "oracle_endpoint_pass": index < oracle,
                    "materiality_pass": index < material,
                    "pareto_repair_pass": index < pareto,
                    "oracle_fidelity_pass": index < fidelity,
                    "populations": {
                        "mixture": {
                            "operators": {
                                "corrupted_law_specific": {
                                    "fixed_ceiling_pass": index < naive
                                }
                            }
                        }
                    },
                }
            )
    return cells


def test_population_classifications_are_locked() -> None:
    close = _classification_cells(
        common=5, oracle=5, material=5, pareto=5, fidelity=5, naive=0
    )
    assert study.classify(close, True)["classification"] == (
        "fresh_corrective_confirms_common_robust_nested_law_closure"
    )
    weak = _classification_cells(
        common=5, oracle=5, material=0, pareto=0, fidelity=5, naive=5
    )
    assert study.classify(weak, True)["classification"] == (
        "fresh_hidden_order_corruption_not_material"
    )
    selector = _classification_cells(
        common=3, oracle=5, material=5, pareto=3, fidelity=3, naive=0
    )
    assert study.classify(selector, True)["classification"] == (
        "fresh_recoverable_scope_exceeds_common_fixed_estimator"
    )
    unidentifiable = _classification_cells(
        common=3, oracle=3, material=5, pareto=3, fidelity=3, naive=0
    )
    assert study.classify(unidentifiable, True)["classification"] == (
        "fresh_scope_not_recoverable_at_required_ceiling"
    )
    assert study.classify(close, False)["classification"] == (
        "invalid_hidden_order_corruption_corrective"
    )


def test_authoritative_fresh_corrective_confirms_fixed_closure() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == PRIMARY_RESULT_SHA256
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["source_hashes"]["runner"] == RUNNER_SHA256
    assert result["aggregates"] == {
        "classification": (
            "fresh_corrective_confirms_common_robust_nested_law_closure"
        ),
        "common_robust_endpoint_seed_pass_count": 5,
        "compact_typed_selector_comparison_licensed": False,
        "corrupted_naive_mixture_ceiling_seed_pass_count": 0,
        "materiality_seed_pass_count": 5,
        "oracle_endpoint_seed_pass_count": 5,
        "oracle_fidelity_seed_pass_count": 5,
        "pareto_repair_seed_pass_count": 5,
        "required_seed_passes": 4,
        "tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["accounting"] == {
        "adaptive_selector_decisions": 0,
        "checkpoints_loaded": 0,
        "closed_form_observation_only_fits": 737_280,
        "fresh_base_examples": 81_920,
        "fresh_corrupted_evaluations": 81_920,
        "law_label_reads_by_primary_estimator": 0,
        "models_instantiated": 0,
        "old_primary_examples_pooled": 0,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    for cell in result["cells"]:
        assert cell["valid"] is True
        assert cell["common_robust_endpoint_pass"] is True
        assert cell["oracle_endpoint_pass"] is True
        assert cell["materiality_pass"] is True
        assert cell["pareto_repair_pass"] is True
        assert cell["oracle_fidelity_pass"] is True
        for population in cell["populations"].values():
            assert population["valid"] is True
            assert population["corruption_material"] is True
            assert population["pareto_repair"]["pass"] is True
            assert population["oracle_fidelity"]["pass"] is True
            assert population["operators"]["clean_law_specific"][
                "fixed_ceiling_pass"
            ] is True
            assert population["operators"]["corrupted_law_specific"][
                "fixed_ceiling_pass"
            ] is False
            for arm in (
                "oracle_drop_one_quadratic",
                "robust_drop_one_quadratic",
            ):
                assert population["operators"][arm]["fixed_ceiling_pass"] is True


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "common_robust_endpoint_seed_pass_count": 5,
            "materiality_seed_pass_count": 5,
            "pareto_repair_seed_pass_count": 5,
            "oracle_fidelity_seed_pass_count": 5,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
