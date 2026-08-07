import json
from pathlib import Path

import pytest

from neural_architecture_lab.carrier_jacobian_axis_audit_meta_hypothesis import (
    build_carrier_jacobian_axis_audit_experiment_results,
    build_carrier_jacobian_axis_audit_meta_hypothesis,
    store_carrier_jacobian_axis_audit_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_carrier_jacobian_axis_audit/"
    "20260806_d6_preregistered_diagnostic/campaign_results.json"
)


def test_meta_record_separates_local_success_from_architectural_rejection() -> None:
    record = build_carrier_jacobian_axis_audit_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_mixed_checkpoint_stratified_causal_atlas"
    )
    assert hypothesis["subclaims"]["local_task_linearization"] == (
        "supported_six_of_six"
    )
    assert hypothesis["subclaims"]["universal_weak_third_axis_correction"] == (
        "rejected_zero_of_six_dominant"
    )
    assert record["result"]["confirmed"] is False


def test_experiment_records_preserve_pair_classification_and_local_gates() -> None:
    record = build_carrier_jacobian_axis_audit_meta_hypothesis(RESULTS)
    experiments = build_carrier_jacobian_axis_audit_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 6
    assert all(item.metrics["local_linearization_adequate"] == 1.0 for item in experiments)
    assert all(item.metrics["signed_error_zero_referenced_r2"] >= 0.983 for item in experiments)
    assert sum(item.metrics["axis_3_dominant"] for item in experiments) == 0.0
    assert sum(item.metrics["axes_1_2_dominant"] for item in experiments) == 2.0
    assert sum(item.metrics["mixed"] for item in experiments) == 4.0


def test_campaign_bytes_cannot_promote_universal_axis_story(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["axis_3_dominant_count"] = 6
    campaign["aggregates"]["universal_2d_base_plus_local_scalar_supported"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="axis-audit campaign"):
        build_carrier_jacobian_axis_audit_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_carrier_jacobian_axis_audit_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 6


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_carrier_jacobian_axis_audit_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-carrier-jacobian-axis-audit-v1",
        "experiment_count": 6,
    }
