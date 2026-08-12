import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_full_local_gauge_closure_meta_hypothesis import (
    build_c3_full_local_gauge_closure_experiment_results,
    build_c3_full_local_gauge_closure_meta_hypothesis,
    store_c3_full_local_gauge_closure_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_full_local_gauge_closure/"
    "20260811_artifact_audit/result.json"
)


def test_meta_confirms_full_local_gauge_closure_and_training_stop() -> None:
    record = build_c3_full_local_gauge_closure_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_full_c3_eight_local_gauge_task_closure"
    )
    assert hypothesis["subclaims"][
        "exact_pointwise_cubic_local_invariance"
    ] == "supported_five_of_five"
    assert hypothesis["subclaims"]["full_local_group_certificate"] == (
        "supported_order_6561"
    )
    assert hypothesis["subclaims"]["selector_identity_invariance"] == (
        "not_required_42_tie_changes"
    )
    assert hypothesis["subclaims"]["same_law_multiple_jump_experiment"] == (
        "not_licensed"
    )
    assert hypothesis["subclaims"]["compact_connection_model"] == (
        "not_licensed"
    )
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_direct_evidence_preserves_all_generators_and_zero_integer_errors() -> None:
    record = build_c3_full_local_gauge_closure_meta_hypothesis(RESULTS)
    pairs = record["evidence"]["direct_tests"]
    assert len(pairs) == 5
    for pair in pairs:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert len(cell["generator_measurements"]) == 16
            assert cell["exact_algebraic_closure"] is True
            assert cell["numeric_local_gauge_closure"] is True
            assert all(
                item["exact_cube_integer_error_count"] == 0
                for item in (
                    *cell["generator_measurements"].values(),
                    cell["arbitrary_local_action"],
                )
            )


def test_experiment_results_preserve_five_seed_certificate() -> None:
    record = build_c3_full_local_gauge_closure_meta_hypothesis(RESULTS)
    experiments = build_c3_full_local_gauge_closure_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 5.0
    assert sum(item.metrics["exact_both_shifts"] for item in experiments) == 5.0
    assert sum(item.metrics["numeric_both_shifts"] for item in experiments) == 5.0
    assert all(item.metrics["certified_group_order"] == 6561.0 for item in experiments)
    assert all(item.metrics["fresh_examples"] == 0.0 for item in experiments)
    assert all(item.model_parameters == 0 for item in experiments)


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["certified_local_group_order"] = 2187
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="full local-gauge result"):
        build_c3_full_local_gauge_closure_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_full_local_gauge_closure_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_full_local_gauge_closure_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-local-gauge-invariant-closure-v1",
        "experiment_count": 5,
    }
