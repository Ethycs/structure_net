import json
from pathlib import Path

import pytest

from neural_architecture_lab.local_continuation_tangent_meta_hypothesis import (
    build_local_continuation_tangent_experiment_results,
    build_local_continuation_tangent_meta_hypothesis,
    store_local_continuation_tangent_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_local_continuation_tangent/"
    "20260807_d6_preregistered_diagnostic/campaign_results.json"
)


def test_meta_preserves_failed_full_gate_and_narrow_tangent_result() -> None:
    record = build_local_continuation_tangent_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_specificity_two_of_three_despite_tangent_endpoint_three_of_three"
    )
    assert hypothesis["subclaims"]["tangent_continuous_endpoint"] == (
        "supported_three_of_three_twelve_of_twelve_cells"
    )
    assert hypothesis["subclaims"]["random_control_margin"] == (
        "supported_two_of_three"
    )
    assert hypothesis["subclaims"]["full_preregistered_gate"] == (
        "not_confirmed_two_of_three"
    )


def test_experiment_records_preserve_three_tangent_and_two_primary_passes() -> None:
    record = build_local_continuation_tangent_meta_hypothesis(RESULTS)
    experiments = build_local_continuation_tangent_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 2.0
    assert all(item.metrics["tangent_checkpoint_pass"] == 1.0 for item in experiments)
    assert all(item.metrics["kernel_change_fraction"] < 0.01 for item in experiments)


def test_campaign_cannot_rewrite_specificity_miss_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["primary_checkpoint_pass_count"] = 3
    campaign["aggregates"]["gate_counts"]["primary_checkpoint_pass"] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="local continuation tangent campaign"):
        build_local_continuation_tangent_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_continuation_tangent_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_continuation_tangent_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-local-continuation-tangent-v1",
        "experiment_count": 3,
    }
