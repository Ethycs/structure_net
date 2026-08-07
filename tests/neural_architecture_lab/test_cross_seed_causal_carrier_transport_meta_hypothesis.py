import json
from pathlib import Path

import pytest

from neural_architecture_lab.cross_seed_causal_carrier_transport_meta_hypothesis import (
    build_cross_seed_causal_carrier_transport_experiment_results,
    build_cross_seed_causal_carrier_transport_meta_hypothesis,
    store_cross_seed_causal_carrier_transport_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_cross_seed_causal_carrier_transport/"
    "20260806_d6_preregistered/campaign_results.json"
)


def test_meta_record_preserves_negative_causal_verdict() -> None:
    record = build_cross_seed_causal_carrier_transport_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_geometric_but_not_causal_transport"
    )
    assert hypothesis["subclaims"]["paired_coordinate_transport"] == (
        "supported_six_of_six"
    )
    assert hypothesis["subclaims"]["paired_causal_transport"] == (
        "rejected_zero_of_six"
    )
    assert hypothesis["subclaims"]["common_linear_causal_chart"] == "rejected"
    assert record["result"]["confirmed"] is False


def test_experiment_records_preserve_pair_gates() -> None:
    record = build_cross_seed_causal_carrier_transport_meta_hypothesis(RESULTS)
    experiments = build_cross_seed_causal_carrier_transport_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 6
    assert all(item.primary_metric == 0.0 for item in experiments)
    assert all(item.metrics["paired_coordinate_transport"] == 1.0 for item in experiments)
    assert all(item.metrics["paired_causal_transport"] == 0.0 for item in experiments)
    assert sum(item.metrics["target_control_contract"] for item in experiments) == 4.0


def test_aggregate_cannot_promote_causal_transport(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["gate_counts"]["paired_causal_transport"] = 6
    campaign["aggregates"]["confirmed"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="carrier-transport campaign"):
        build_cross_seed_causal_carrier_transport_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_cross_seed_causal_carrier_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 6


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_cross_seed_causal_carrier_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-cross-seed-causal-carrier-transport-v1",
        "experiment_count": 6,
    }
