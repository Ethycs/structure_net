import json
from pathlib import Path

import pytest

from neural_architecture_lab.local_continuation_tangent_kernel_meta_hypothesis import (
    build_local_continuation_experiment_results,
    build_local_continuation_meta_hypothesis,
    store_local_continuation_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_local_continuation_tangent_kernel/"
    "20260807_d6_corrective_v2/campaign_results.json"
)


def test_meta_preserves_failed_stability_and_supported_local_tangent() -> None:
    record = build_local_continuation_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_stable_metric_zero_of_three_local_tangent_supported"
    )
    assert hypothesis["subclaims"]["tangent_causal_sufficiency"] == (
        "supported_three_of_three_all_12_cells"
    )
    assert hypothesis["subclaims"]["cross_cell_metric_stability"] == (
        "rejected_zero_of_three"
    )


def test_experiment_records_preserve_joint_gate_failure() -> None:
    record = build_local_continuation_meta_hypothesis(RESULTS)
    experiments = build_local_continuation_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["tangent_checkpoint_pass"] == 1.0 for item in experiments)
    assert all(item.metrics["kernel_output_inert"] == 1.0 for item in experiments)
    assert all(item.metrics["task_metric_stable"] == 0.0 for item in experiments)
    assert all(
        item.metrics["stable_local_task_metric_gate"] == 0.0
        for item in experiments
    )


def test_campaign_cannot_rewrite_stability_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["stable_gate_count"] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="local continuation campaign"):
        build_local_continuation_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_continuation_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_local_continuation_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-local-continuation-tangent-kernel-v1",
        "experiment_count": 3,
    }
