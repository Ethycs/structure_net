import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_hidden_order_corruption_corrective_meta_hypothesis import (
    build_c3_hidden_order_corruption_corrective_experiment_results,
    build_c3_hidden_order_corruption_corrective_meta_hypothesis,
    store_c3_hidden_order_corruption_corrective_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_corrective/"
    "20260811_preregistered/result.json"
)


def test_meta_preserves_fresh_confirmatory_decision() -> None:
    record = build_c3_hidden_order_corruption_corrective_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_fresh_common_robust_nested_law_closure_five_of_five"
    )
    assert hypothesis["evidence_count"] == 5
    assert hypothesis["subclaims"]["fresh_common_absolute_endpoint"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["fresh_oracle_absolute_endpoint"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["fresh_pareto_repair"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["original_inconclusive_classification"] == (
        "unchanged"
    )
    assert hypothesis["subclaims"]["tinyllm_training"] == "not_licensed"


def test_direct_evidence_preserves_all_fresh_joint_gates() -> None:
    record = build_c3_hidden_order_corruption_corrective_meta_hypothesis(RESULTS)
    pairs = record["evidence"]["direct_tests"]
    assert len(pairs) == 5
    for pair in pairs:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["common_robust_endpoint_pass"] is True
            assert cell["oracle_endpoint_pass"] is True
            assert cell["materiality_pass"] is True
            assert cell["pareto_repair_pass"] is True
            assert cell["oracle_fidelity_pass"] is True
            for population in cell["populations"].values():
                assert population["corruption_material"] is True
                assert population["pareto_repair"]["pass"] is True
                assert population["oracle_fidelity"]["pass"] is True


def test_experiment_results_preserve_five_confirmed_joint_units() -> None:
    record = build_c3_hidden_order_corruption_corrective_meta_hypothesis(RESULTS)
    experiments = build_c3_hidden_order_corruption_corrective_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert all(item.model_parameters == 0 for item in experiments)
    assert all(
        item.metrics["common_endpoint_both_shifts"] == 1.0
        for item in experiments
    )
    assert all(
        item.metrics["pareto_repair_both_shifts"] == 1.0
        for item in experiments
    )
    assert all(
        item.metrics["old_primary_examples_pooled"] == 0.0
        for item in experiments
    )


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["pareto_repair_seed_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 hidden-order corrective result"):
        build_c3_hidden_order_corruption_corrective_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_hidden_order_corruption_corrective_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_hidden_order_corruption_corrective_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-hidden-order-corruption-corrective-v1",
        "experiment_count": 5,
    }
