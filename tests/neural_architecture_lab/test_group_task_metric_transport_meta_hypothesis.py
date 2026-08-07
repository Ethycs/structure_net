import json
from pathlib import Path

import pytest

from neural_architecture_lab.group_task_metric_transport_meta_hypothesis import (
    build_group_task_metric_transport_experiment_results,
    build_group_task_metric_transport_meta_hypothesis,
    store_group_task_metric_transport_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_group_task_metric_transport/"
    "20260806_d6_preregistered/campaign_results.json"
)


def test_meta_record_preserves_pairwise_support_and_global_rejection() -> None:
    record = build_group_task_metric_transport_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_pairwise_7_53_gauge_class_only"
    )
    assert hypothesis["subclaims"]["global_task_metric_causal_chart"] == (
        "rejected_two_of_six"
    )
    assert hypothesis["subclaims"]["pairwise_seed7_seed53_gauge_equivalence"] == (
        "supported_both_directions"
    )
    assert hypothesis["subclaims"]["seed29_extrapolation_cells"] == (
        "rejected_zero_of_eight"
    )


def test_experiment_records_preserve_exact_pair_split() -> None:
    record = build_group_task_metric_transport_meta_hypothesis(RESULTS)
    experiments = build_group_task_metric_transport_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 6
    assert sum(item.primary_metric for item in experiments) == 2.0
    passed = {
        item.experiment_id
        for item in experiments
        if item.metrics["task_metric_causal_transport"] == 1.0
    }
    assert passed == {
        "tinyllm-c2-task-metric-transport-7-to-53",
        "tinyllm-c2-task-metric-transport-53-to-7",
    }
    assert all(item.metrics["metric_contract"] == 1.0 for item in experiments)


def test_aggregate_cannot_promote_global_confirmation(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["gate_counts"]["task_metric_causal_transport"] = 6
    campaign["aggregates"]["confirmed"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="task-metric transport campaign"):
        build_group_task_metric_transport_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_group_task_metric_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 6


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_group_task_metric_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-group-task-metric-carrier-transport-v1",
        "experiment_count": 6,
    }
