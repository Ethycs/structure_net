import json
from pathlib import Path

import pytest

import neural_architecture_lab.posterior_coordinate_transport_meta_hypothesis as meta


CAMPAIGN = Path(
    "data/experiments/tinyllm_posterior_coordinate_transport/"
    "20260807_d8_preregistered/campaign_results.json"
)


@pytest.mark.skipif(not CAMPAIGN.is_file(), reason="campaign artifacts absent")
def test_meta_record_is_conservative_and_matches_campaign() -> None:
    record = meta.build_posterior_coordinate_transport_meta_hypothesis(CAMPAIGN)
    hypothesis = record["hypothesis"]
    assert hypothesis["id"] == meta.HYPOTHESIS_ID
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "high_rank_answer_chart_compact_prediction_rejected"
    )
    assert record["result"]["confirmed"] is False
    tests = record["evidence"]["direct_tests"]
    assert len(tests) == 10
    assert all(test["k16_rank_gates"]["full"] for test in tests)
    assert not any(test["k16_rank_gates"]["1"] for test in tests)
    assert all(
        test["maximum_source_metric_replay_error"] == 0.0 for test in tests
    )


@pytest.mark.skipif(not CAMPAIGN.is_file(), reason="campaign artifacts absent")
def test_meta_validation_rejects_a_tampered_campaign(tmp_path: Path) -> None:
    tampered = tmp_path / "campaign_results.json"
    value = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    value["aggregates"]["classification"] = "compact_vector_chart"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid posterior-coordinate"):
        meta.build_posterior_coordinate_transport_meta_hypothesis(tampered)


@pytest.mark.skipif(not CAMPAIGN.is_file(), reason="campaign artifacts absent")
def test_experiment_results_expose_the_rank_ladder() -> None:
    record = meta.build_posterior_coordinate_transport_meta_hypothesis(CAMPAIGN)
    experiments = meta.build_posterior_coordinate_transport_experiment_results(
        record, CAMPAIGN
    )
    assert len(experiments) == 10
    for item in experiments:
        assert item.hypothesis_id == meta.HYPOTHESIS_ID
        assert item.primary_metric == 1.0
        assert item.metrics["k16_rank_full_pass"] == 1.0
        assert item.metrics["k16_rank_1_pass"] == 0.0
