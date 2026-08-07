import json
from pathlib import Path

import pytest

from neural_architecture_lab.calibration_readout_decomposition_meta_hypothesis import (
    build_calibration_readout_decomposition_experiment_results,
    build_calibration_readout_decomposition_meta_hypothesis,
    store_calibration_readout_decomposition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_calibration_readout_decomposition/"
    "20260807_d8_corrective_v2/campaign_results.json"
)


def test_meta_preserves_reference_precision_classification() -> None:
    record = build_calibration_readout_decomposition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_reference_coordinate_precision_limited"
    )
    assert hypothesis["subclaims"]["frozen_observer_utility_at_0p20"] == (
        "rejected_zero_of_five_both_arms"
    )
    assert hypothesis["subclaims"]["target_oracle_patch_at_0p20"] == (
        "supported_five_of_five_both_arms"
    )
    assert hypothesis["subclaims"]["readout_only_training_licensed"] == "no"


def test_meta_preserves_high_correlation_but_failed_utility() -> None:
    record = build_calibration_readout_decomposition_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]["checkpoint_means"]
    for condition in (
        "analytic_calibrated",
        "learned_calibrated_equivariant",
    ):
        arm = metrics[condition]
        assert arm["observer_cosine_correlation_mean_at_0p20"] > 0.97
        assert arm["observer_cosine_rmse_mean_at_0p20"] > 0.11
        assert arm["observer_accuracy_drop_mean_at_0p20"] > 0.20
        assert arm["observer_patch_post_moment_error_mean_at_0p20"] < 0.002
        assert arm["oracle_patch_accuracy_mean_at_0p20"] > 0.72


def test_experiment_records_are_causal_and_checkpoint_specific() -> None:
    record = build_calibration_readout_decomposition_meta_hypothesis(RESULTS)
    experiments = build_calibration_readout_decomposition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["pass_patches_target_oracle"] == 1.0 for item in experiments)
    assert all(item.metrics["pass_patches_observer"] == 0.0 for item in experiments)
    assert len({item.model_checkpoint for item in experiments}) == 10


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = "decoder_relation_limited"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="readout-decomposition campaign"):
        build_calibration_readout_decomposition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_readout_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_readout_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-calibration-readout-decomposition-v1",
        "experiment_count": 10,
    }
    assert (tmp_path / "experiment_queue").is_dir()
