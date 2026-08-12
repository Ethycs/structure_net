import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_relational_connection_preflight_meta_hypothesis import (
    build_c3_relational_connection_preflight_experiment_results,
    build_c3_relational_connection_preflight_meta_hypothesis,
    store_c3_relational_connection_preflight_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_relational_connection_preflight/"
    "20260811_preregistered/result.json"
)


def test_meta_confirms_connection_identifiability_and_training_boundary() -> None:
    record = build_c3_relational_connection_preflight_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_observed_connection_identifiability_and_specificity"
    )
    assert hypothesis["subclaims"]["observed_connection_fixed_ceiling"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["connection_specificity"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["pointwise_invariant_insufficiency"] == (
        "supported_five_of_five_plus_exact_collision"
    )
    assert hypothesis["subclaims"]["function_class_lifecycle_preflight"] == (
        "licensed"
    )
    assert hypothesis["subclaims"]["matched_training"] == "not_yet_licensed"
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_direct_evidence_preserves_connection_and_control_gates() -> None:
    record = build_c3_relational_connection_preflight_meta_hypothesis(RESULTS)
    pairs = record["evidence"]["direct_tests"]
    assert len(pairs) == 5
    for pair in pairs:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["observed_connection_pass"] is True
            assert cell["connection_specificity_pass"] is True
            assert cell["pointwise_insufficiency_pass"] is True
            assert cell["connection_free_shortcut_pass"] is False
            assert cell["arms"]["observed_connection"]["scalar"]["rmse"] <= 0.01
            assert all(
                cell["arms"][name]["failure_control_pass"] is True
                for name in (
                    "no_connection",
                    "wrong_sign_connection",
                    "shuffled_connection",
                    "principal_pointwise_invariant",
                )
            )


def test_experiment_results_preserve_five_seed_certificate() -> None:
    record = build_c3_relational_connection_preflight_meta_hypothesis(RESULTS)
    experiments = build_c3_relational_connection_preflight_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 5.0
    assert all(
        item.metrics["observed_connection_both_shifts"] == 1.0
        for item in experiments
    )
    assert all(item.metrics["specificity_both_shifts"] == 1.0 for item in experiments)
    assert all(
        item.metrics["pointwise_insufficiency_both_shifts"] == 1.0
        for item in experiments
    )
    assert all(item.model_parameters == 0 for item in experiments)


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["observed_connection_seed_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="relational-connection result"):
        build_c3_relational_connection_preflight_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_relational_connection_preflight_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_relational_connection_preflight_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-relational-connection-preflight-v1",
        "experiment_count": 5,
    }
