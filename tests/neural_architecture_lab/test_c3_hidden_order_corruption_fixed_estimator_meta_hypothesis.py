import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_hidden_order_corruption_fixed_estimator_meta_hypothesis import (
    build_c3_hidden_order_corruption_fixed_estimator_experiment_results,
    build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis,
    store_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)


def test_meta_preserves_registered_inconclusive_decision() -> None:
    record = build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
        RESULTS
    )
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "inconclusive_registered_speed_repair_accuracy_effect_size_"
        "zero_of_five"
    )
    assert hypothesis["subclaims"]["common_absolute_endpoint"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["full_registered_repair"] == (
        "not_supported_zero_of_five"
    )
    assert hypothesis["subclaims"]["tinyllm_training"] == "not_licensed"


def test_direct_evidence_preserves_positive_components_and_failed_conjunction() -> None:
    record = build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
        RESULTS
    )
    pairs = record["evidence"]["direct_tests"]
    assert len(pairs) == 5
    for pair in pairs:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["common_robust_endpoint_pass"] is True
            assert cell["materiality_pass"] is True
            assert cell["oracle_fidelity_pass"] is True
            assert cell["repair_pass"] is False
            speed = cell["populations"]["constant_speed"]
            assert speed["robust_repair"]["pass"] is False
            assert speed["robust_repair"]["accuracy_delta"] < 0.20
            assert speed["robust_repair"]["rmse_ratio"] < 0.02
            assert cell["populations"]["constant_acceleration"][
                "robust_repair"
            ]["pass"] is True
            assert cell["populations"]["mixture"]["robust_repair"][
                "pass"
            ] is True


def test_experiment_results_preserve_five_failed_joint_units() -> None:
    record = build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
        RESULTS
    )
    experiments = (
        build_c3_hidden_order_corruption_fixed_estimator_experiment_results(
            record, RESULTS
        )
    )
    assert len(experiments) == 5
    assert all(item.primary_metric == 0.0 for item in experiments)
    assert all(item.model_parameters == 0 for item in experiments)
    assert all(item.metrics["common_endpoint_both_shifts"] == 1.0 for item in experiments)
    assert all(item.metrics["registered_repair_both_shifts"] == 0.0 for item in experiments)


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["repair_seed_pass_count"] = 1
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 hidden-order result"):
        build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": (
            "tinyllm-c3-hidden-order-corruption-fixed-estimator-v1"
        ),
        "experiment_count": 5,
    }
