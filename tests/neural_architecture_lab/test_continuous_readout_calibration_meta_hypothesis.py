import json
from pathlib import Path

import pytest

from neural_architecture_lab.continuous_readout_calibration_meta_hypothesis import (
    build_continuous_readout_calibration_experiment_results,
    build_continuous_readout_calibration_meta_hypothesis,
    store_continuous_readout_calibration_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_continuous_readout_calibration/"
    "20260806_d6_corrective_v3_6a88480c/campaign_results.json"
)


def test_meta_record_preserves_corrective_evidence_boundary() -> None:
    record = build_continuous_readout_calibration_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "supported_post_outcome_corrective_replication_not_fresh_confirmation"
    )
    assert hypothesis["evidence_role"] == (
        "post_outcome_corrective_replication_evidence"
    )
    assert hypothesis["subclaims"]["rank_at_most_three"] == (
        "supported_three_of_three"
    )
    assert record["result"]["confirmed"] is False


def test_corrective_experiment_records_retain_passing_gates_and_ranks() -> None:
    record = build_continuous_readout_calibration_meta_hypothesis(RESULTS)
    experiments = build_continuous_readout_calibration_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert {item.metrics["selected_rank"] for item in experiments} == {2.0, 3.0}
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert sum(item.metrics.get("seed29_margin_repair", 0.0) for item in experiments) == 1.0


def test_fresh_evidence_role_cannot_be_promoted_by_mutating_aggregate(
    tmp_path: Path,
) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["evidence_role"] = "preregistered_underpowered_mechanistic_evidence"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="corrective campaign"):
        build_continuous_readout_calibration_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_continuous_readout_calibration_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_continuous_readout_calibration_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-semantic-carrier-readout-separation-v1",
        "experiment_count": 3,
    }
