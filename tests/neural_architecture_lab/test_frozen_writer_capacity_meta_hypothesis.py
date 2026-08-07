import json
from pathlib import Path

import pytest

from neural_architecture_lab.frozen_writer_capacity_meta_hypothesis import (
    build_frozen_writer_capacity_experiment_results,
    build_frozen_writer_capacity_meta_hypothesis,
    store_frozen_writer_capacity_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_frozen_writer_capacity/"
    "20260807_d6_preregistered_diagnostic/campaign_results.json"
)


def test_meta_preserves_activation_writer_rejection() -> None:
    record = build_frozen_writer_capacity_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_small_writer_insufficient_three_of_three"
    )
    assert hypothesis["subclaims"]["rank_three_propagated_activation_context"] == (
        "rejected_zero_of_three"
    )
    assert hypothesis["subclaims"]["small_writer_insufficiency"] == (
        "supported_three_of_three"
    )


def test_experiment_records_preserve_all_writer_failures() -> None:
    record = build_frozen_writer_capacity_meta_hypothesis(RESULTS)
    experiments = build_frozen_writer_capacity_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["predecessor_replay_contract"] == 1.0 for item in experiments)
    assert all(
        item.metrics["invariant_context_numerical_contract"] == 1.0
        for item in experiments
    )
    for item in experiments:
        for order in (1, 2, 3, 4):
            assert item.metrics[f"context_m{order:02d}_complete_pass"] == 0.0


def test_campaign_cannot_rewrite_activation_writer_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["classification_by_seed"]["7"] = (
        "invariant_context_required"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="frozen writer-capacity campaign"):
        build_frozen_writer_capacity_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_writer_capacity_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_writer_capacity_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-frozen-writer-capacity-v1",
        "experiment_count": 3,
    }
