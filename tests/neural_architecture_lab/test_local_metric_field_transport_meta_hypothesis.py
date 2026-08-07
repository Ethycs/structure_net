import json
from pathlib import Path

import pytest

from neural_architecture_lab.local_metric_field_transport_meta_hypothesis import (
    build_local_metric_field_transport_experiment_results,
    build_local_metric_field_transport_meta_hypothesis,
    store_local_metric_field_transport_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_local_metric_field_transport/"
    "20260807_d6_fresh_cohort/campaign_results.json"
)


def test_meta_preserves_invalid_control_boundary() -> None:
    record = build_local_metric_field_transport_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == "control_invalid_corrective_required"
    assert hypothesis["subclaims"][
        "identity_transported_tangent_task_equivalence"
    ] == "descriptively_supported_three_of_three_twenty_four_of_twenty_four_cells"
    assert hypothesis["subclaims"]["registered_nuisance_shuffle_control"] == (
        "invalidated_structurally"
    )
    assert hypothesis["subclaims"]["unique_invariant_metric_field"] == (
        "deferred_to_locked_specificity_corrective"
    )
    assert record["evidence"]["validity"][
        "quality_evidence_for_transport_specificity"
    ] is False


def test_experiment_records_keep_zero_primary_and_broad_control_success() -> None:
    record = build_local_metric_field_transport_meta_hypothesis(RESULTS)
    experiments = build_local_metric_field_transport_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["transported_all_cells_pass"] == 1.0 for item in experiments)
    assert all(item.metrics["shuffled_all_cells_pass"] == 1.0 for item in experiments)
    assert all(item.metrics["random_all_cells_pass"] == 1.0 for item in experiments)
    assert all(item.metrics["geometric_transport_pass"] == 0.0 for item in experiments)


def test_campaign_cannot_rewrite_specificity_failure_as_success(
    tmp_path: Path,
) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["gate_counts"]["causal_transport_pass"] = 3
    campaign["aggregates"]["primary_checkpoint_pass_count"] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="local metric-field transport campaign"):
        build_local_metric_field_transport_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_metric_field_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_metric_field_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-local-metric-field-transport-v1",
        "experiment_count": 3,
    }
