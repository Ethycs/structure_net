import json
from pathlib import Path

import pytest

from neural_architecture_lab.nuisance_fiber_task_metric_transport_meta_hypothesis import (
    build_nuisance_fiber_task_metric_transport_experiment_results,
    build_nuisance_fiber_task_metric_transport_meta_hypothesis,
    store_nuisance_fiber_task_metric_transport_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_nuisance_fiber_task_metric_transport/"
    "20260807_d6_preregistered_diagnostic/campaign_results.json"
)


def test_meta_preserves_nonspecific_causal_transport_result() -> None:
    record = build_nuisance_fiber_task_metric_transport_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_nonspecific_causal_transport_and_extrapolation_tail_geometry"
    )
    assert hypothesis["subclaims"]["transported_tangent_continuous_endpoint"] == (
        "supported_three_of_three_twelve_of_twelve_cells"
    )
    assert hypothesis["subclaims"]["rank_two_specificity"] == (
        "rejected_zero_of_three_random_94_of_96_cells_pass"
    )


def test_experiment_records_preserve_causal_but_zero_primary_passes() -> None:
    record = build_nuisance_fiber_task_metric_transport_meta_hypothesis(RESULTS)
    experiments = build_nuisance_fiber_task_metric_transport_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["causal_transport"] == 1.0 for item in experiments)
    assert all(item.metrics["specificity"] == 0.0 for item in experiments)


def test_campaign_cannot_rewrite_specificity_failure_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["confirmed"] = True
    campaign["aggregates"]["gate_counts"]["specificity"] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="nuisance-fiber metric campaign"):
        build_nuisance_fiber_task_metric_transport_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_nuisance_fiber_task_metric_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_nuisance_fiber_task_metric_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-nuisance-fiber-task-metric-transport-v1",
        "experiment_count": 3,
    }
