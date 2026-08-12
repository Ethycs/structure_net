import json
from pathlib import Path

import pytest

from neural_architecture_lab.frozen_interval_readout_decomposition_meta_hypothesis import (
    build_frozen_interval_readout_decomposition_experiment_results,
    build_frozen_interval_readout_decomposition_meta_hypothesis,
    store_frozen_interval_readout_decomposition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_frozen_interval_readout_decomposition/"
    "20260811_d6_d10_preregistered/campaign_results.json"
)


def test_meta_rejects_architecture_family_typed_readout_claim() -> None:
    record = build_frozen_interval_readout_decomposition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_typed_readout_fails_d10_learned_one_of_five"
    )
    assert hypothesis["subclaims"]["architecture_family_typed_readout"] == (
        "contradicted_d10_learned_one_of_five"
    )


def test_meta_preserves_partial_repair_and_ordered_scalar_result() -> None:
    record = build_frozen_interval_readout_decomposition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    direct = record["result"]["descriptive_metrics"]
    assert hypothesis["subclaims"]["d6_learned_typed_readout"] == (
        "supported_four_of_five"
    )
    assert hypothesis["subclaims"]["source_failure_repair"] == (
        "supported_five_of_ten_partial"
    )
    assert direct["minimum_typed_cosine_correlation"] > 0.95
    assert direct["strata"]["d10/learned_calibrated_equivariant"]["arms"][
        "typed_interval_readout"
    ]["true_pass_count"] == 1


def test_experiment_results_preserve_twenty_seed_units() -> None:
    record = build_frozen_interval_readout_decomposition_meta_hypothesis(RESULTS)
    experiments = build_frozen_interval_readout_decomposition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 20
    assert sum(item.primary_metric for item in experiments) == 15
    assert all(item.metrics["validity"] == 1 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = (
        "frozen_typed_interval_readout_architecture_stable"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen interval-readout campaign"):
        build_frozen_interval_readout_decomposition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_interval_readout_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 20


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_interval_readout_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-frozen-interval-readout-decomposition-v1",
        "experiment_count": 20,
    }
