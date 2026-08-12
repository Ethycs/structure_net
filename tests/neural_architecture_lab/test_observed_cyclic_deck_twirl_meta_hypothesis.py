import json
from pathlib import Path

import pytest

from neural_architecture_lab.observed_cyclic_deck_twirl_meta_hypothesis import (
    build_observed_cyclic_deck_experiment_results,
    build_observed_cyclic_deck_meta_hypothesis,
    store_observed_cyclic_deck_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_observed_cyclic_deck_twirl/"
    "20260810_d6_preregistered/campaign_results.json"
)


def test_meta_preserves_failed_full_gate_and_positive_mature_result() -> None:
    record = build_observed_cyclic_deck_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "preregistered_full_front_not_confirmed_c2_4of5_c3_2of5_"
        "mature_both_5of5"
    )
    assert hypothesis["direct_experiment_count"] == 10
    assert hypothesis["subclaims"]["mature_observed_C3_twirl"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["C3_front_replication_within_one_cut"] == (
        "not_supported_two_of_five"
    )


def test_meta_retains_every_preregistered_population_count() -> None:
    record = build_observed_cyclic_deck_meta_hypothesis(RESULTS)
    degrees = record["result"]["descriptive_metrics"]["campaign"]["degrees"]
    assert degrees["2"]["mature_observed_twirl_pass_count"] == 5
    assert degrees["2"]["front_replication_pass_count"] == 4
    assert degrees["3"]["mature_observed_twirl_pass_count"] == 5
    assert degrees["3"]["front_replication_pass_count"] == 2
    assert degrees["2"]["semantic_control_preserved_count"] == 0
    assert degrees["3"]["semantic_control_preserved_count"] == 0


def test_checkpoint_summaries_keep_nulls_and_specificity() -> None:
    record = build_observed_cyclic_deck_meta_hypothesis(RESULTS)
    experiments = build_observed_cyclic_deck_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 6.0
    assert all(item.metrics["mature_observed_twirl"] == 1.0 for item in experiments)
    assert all(item.metrics["semantic_control_preserved"] == 0.0 for item in experiments)
    assert all(
        item.metrics["minimum_control_source_target_accuracy_loss"] > 0.50
        for item in experiments
    )


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["degrees"]["3"]["front_replication_pass_count"] = 4
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="observed cyclic-deck campaign"):
        build_observed_cyclic_deck_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_observed_cyclic_deck_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_observed_cyclic_deck_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-observed-cyclic-deck-twirl-front-v1",
        "experiment_count": 10,
    }
