import json
from pathlib import Path

import pytest

from neural_architecture_lab.fixed_gauge_writer_capacity_meta_hypothesis import (
    build_fixed_gauge_writer_capacity_experiment_results,
    build_fixed_gauge_writer_capacity_meta_hypothesis,
    store_fixed_gauge_writer_capacity_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
    "20260806_d6_preregistered_diagnostic/campaign_results.json"
)


def test_meta_preserves_unresolved_writer_classification() -> None:
    record = build_fixed_gauge_writer_capacity_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "diagnostic_unresolved_writer_limited_three_of_three"
    )
    assert hypothesis["subclaims"]["low_order_phase_curvature"] == (
        "rejected_zero_of_three"
    )
    assert hypothesis["subclaims"]["declared_calibration_context"] == (
        "rejected_zero_of_three"
    )


def test_experiment_records_preserve_all_candidate_failures() -> None:
    record = build_fixed_gauge_writer_capacity_meta_hypothesis(RESULTS)
    experiments = build_fixed_gauge_writer_capacity_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["predecessor_replay_contract"] == 1.0 for item in experiments)
    for item in experiments:
        for candidate in (
            "quotient_order2",
            "quotient_order4",
            "quotient_order40",
            "quotient_order4_context",
        ):
            assert item.metrics[f"{candidate}_specific_causal_pass"] == 0.0


def test_campaign_cannot_rewrite_a_candidate_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["candidate_specific_pass_counts"][
        "quotient_order4"
    ] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="writer-capacity campaign"):
        build_fixed_gauge_writer_capacity_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_fixed_gauge_writer_capacity_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_fixed_gauge_writer_capacity_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-fixed-gauge-writer-capacity-v1",
        "experiment_count": 3,
    }
