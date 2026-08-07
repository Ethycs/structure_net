import json
from pathlib import Path

import pytest

from neural_architecture_lab.local_metric_field_specificity_meta_hypothesis import (
    build_local_metric_field_specificity_experiment_results,
    build_local_metric_field_specificity_meta_hypothesis,
    store_local_metric_field_specificity_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_local_metric_field_specificity/"
    "20260807_corrective_v1/campaign_results.json"
)


def test_meta_preserves_corrected_one_of_three_rejection() -> None:
    record = build_local_metric_field_specificity_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "rejected_checkpoint_stratified_one_of_three"
    )
    assert hypothesis["subclaims"]["original_negative_control_validity"] == (
        "rejected_structurally"
    )
    assert hypothesis["subclaims"]["paired_nuisance_geometry"] == (
        "supported_one_of_three"
    )
    assert record["evidence"]["invalidated_antecedent"]["quality_evidence_used"] is False


def test_experiment_records_preserve_tail_failure_and_valid_controls() -> None:
    record = build_local_metric_field_specificity_meta_hypothesis(RESULTS)
    experiments = build_local_metric_field_specificity_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 1.0
    assert all(
        item.metrics["nuisance_equivalence_all_cells"] == 1.0
        for item in experiments
    )
    assert all(
        item.metrics["semantic_phase_specificity_all_cells"] == 1.0
        for item in experiments
    )
    assert min(item.metrics["minimum_paired_median_cosine"] for item in experiments) > 0.99
    assert min(item.metrics["minimum_paired_p10_cosine"] for item in experiments) < 0.60


def test_campaign_cannot_rewrite_one_of_three_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["corrected_checkpoint_pass_count"] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="metric-field specificity campaign"):
        build_local_metric_field_specificity_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_metric_field_specificity_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_metric_field_specificity_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-local-metric-field-specificity-v1",
        "experiment_count": 3,
    }

