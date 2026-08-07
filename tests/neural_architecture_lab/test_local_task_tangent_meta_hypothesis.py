import json
from pathlib import Path

import pytest

from neural_architecture_lab.local_task_tangent_meta_hypothesis import (
    build_local_task_tangent_experiment_results,
    build_local_task_tangent_meta_hypothesis,
    store_local_task_tangent_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_local_task_tangent/"
    "20260807_d6_preregistered_diagnostic/campaign_results.json"
)


def test_meta_preserves_endpoint_and_control_margin_distinction() -> None:
    record = build_local_task_tangent_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_control_margin_two_of_three_despite_tangent_endpoint_three_of_three"
    )
    assert hypothesis["subclaims"]["tangent_continuous_endpoint"] == (
        "supported_three_of_three_twelve_of_twelve_cells"
    )
    assert hypothesis["subclaims"]["all_absolute_control_margins"] == (
        "supported_two_of_three"
    )


def test_experiment_records_preserve_three_endpoints_and_two_primary_passes() -> None:
    record = build_local_task_tangent_meta_hypothesis(RESULTS)
    experiments = build_local_task_tangent_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 2.0
    assert all(item.metrics["tangent_all_cells_pass"] == 1.0 for item in experiments)
    assert sum(item.metrics["all_controls_specific"] for item in experiments) == 2.0


def test_campaign_cannot_rewrite_control_miss_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["local_task_tangent_pass_count"] = 3
    campaign["aggregates"]["supported"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="local task-tangent campaign"):
        build_local_task_tangent_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_task_tangent_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_task_tangent_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-local-task-tangent-v1",
        "experiment_count": 3,
    }
