import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_gauge_jump_joint_typed_score_meta_hypothesis import (
    build_c3_gauge_jump_joint_typed_score_experiment_results,
    build_c3_gauge_jump_joint_typed_score_meta_hypothesis,
    store_c3_gauge_jump_joint_typed_score_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_gauge_jump_joint_typed_score/"
    "20260811_preregistered/result.json"
)


def test_meta_confirms_fixed_closure_but_not_unique_typed_repair() -> None:
    record = build_c3_gauge_jump_joint_typed_score_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_fixed_time_varying_gauge_closure_without_unique_typed_advantage"
    )
    assert hypothesis["subclaims"]["fresh_original_fixed_connection"] == (
        "supported_four_of_five"
    )
    assert hypothesis["subclaims"]["fresh_joint_typed_connection"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["unique_joint_typed_tail_repair"] == (
        "not_supported_zero_of_five"
    )
    assert hypothesis["subclaims"]["predecessor_fixed_connection_failure"] == (
        "preserved_one_of_five"
    )
    assert hypothesis["subclaims"]["compact_typed_chart_mixture"] == (
        "not_licensed"
    )
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_direct_and_context_evidence_remain_separate() -> None:
    record = build_c3_gauge_jump_joint_typed_score_meta_hypothesis(RESULTS)
    assert len(record["evidence"]["direct_tests"]) == 5
    assert len(record["evidence"]["predecessor_context_tests"]) == 5
    assert {
        pair["seed"] for pair in record["evidence"]["direct_tests"]
    }.isdisjoint(
        pair["seed"]
        for pair in record["evidence"]["predecessor_context_tests"]
    )
    for pair in record["evidence"]["direct_tests"]:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["operators"]["fixed_joint_typed_connection"][
                "fixed_ceiling_pass"
            ] is True
            assert cell["joint_typed_oracle_fidelity"]["pass"] is True


def test_experiment_results_preserve_five_typed_and_four_original() -> None:
    record = build_c3_gauge_jump_joint_typed_score_meta_hypothesis(RESULTS)
    experiments = build_c3_gauge_jump_joint_typed_score_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 5.0
    assert sum(item.metrics["joint_typed_both_shifts"] for item in experiments) == 5.0
    assert sum(item.metrics["original_fixed_both_shifts"] for item in experiments) == 4.0
    assert sum(
        item.metrics["joint_typed_oracle_fidelity_both_shifts"]
        for item in experiments
    ) == 5.0
    assert sum(
        item.metrics["joint_typed_material_repair_both_shifts"]
        for item in experiments
    ) == 0.0
    assert all(item.metrics["exact_joint_typed_action_error"] == 0.0 for item in experiments)
    assert all(item.model_parameters == 0 for item in experiments)


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["joint_typed_seed_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="joint typed result"):
        build_c3_gauge_jump_joint_typed_score_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_gauge_jump_joint_typed_score_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_gauge_jump_joint_typed_score_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-gauge-jump-joint-typed-score-v1",
        "experiment_count": 5,
    }
