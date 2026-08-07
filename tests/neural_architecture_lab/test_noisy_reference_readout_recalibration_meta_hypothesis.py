import json
from pathlib import Path

import pytest

from neural_architecture_lab.noisy_reference_readout_recalibration_meta_hypothesis import (
    build_noisy_reference_readout_recalibration_experiment_results,
    build_noisy_reference_readout_recalibration_meta_hypothesis,
    store_noisy_reference_readout_recalibration_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_noisy_reference_readout_recalibration/"
    "20260807_d8_preregistered_cuda1_persistent_mixed_pedigree/"
    "campaign_results.json"
)


def test_meta_preserves_arm_stratified_classification() -> None:
    record = build_noisy_reference_readout_recalibration_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "partially_supported_arm_stratified_readout_repair"
    )
    assert hypothesis["subclaims"]["learned_linear_readout_repair"] == (
        "supported_five_of_five_preregistered_unseen"
    )
    assert hypothesis["subclaims"]["analytic_linear_readout_repair"] == (
        "not_supported_three_of_five_corrective"
    )
    assert hypothesis["subclaims"]["full_model_retraining_licensed"] == "no"


def test_meta_preserves_counts_controls_and_small_scalar_map() -> None:
    record = build_noisy_reference_readout_recalibration_meta_hypothesis(RESULTS)
    campaign = record["result"]["descriptive_metrics"]["campaign"]
    assert campaign["arms"]["analytic_calibrated"][
        "linear_primary_pass_count"
    ] == 3
    assert campaign["arms"]["learned_calibrated_equivariant"][
        "linear_primary_pass_count"
    ] == 5
    assert campaign["arms"]["learned_calibrated_equivariant"][
        "scalar_primary_pass_count"
    ] == 4
    summaries = record["result"]["descriptive_metrics"]["checkpoint_means"]
    for condition in (
        "analytic_calibrated",
        "learned_calibrated_equivariant",
    ):
        assert 0.96 < summaries[condition]["scalar_slope"] < 1.05
        assert abs(summaries[condition]["scalar_intercept"]) < 0.04
        assert summaries[condition]["maximum_target_shuffled_accuracy"] < 0.20


def test_experiment_records_preserve_mixed_evidence_roles() -> None:
    record = build_noisy_reference_readout_recalibration_meta_hypothesis(RESULTS)
    experiments = build_noisy_reference_readout_recalibration_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 8.0
    assert len({item.model_checkpoint for item in experiments}) == 10
    observations = [text for item in experiments for text in item.observations]
    assert any("post_outcome_corrective_replication_evidence" in text for text in observations)
    assert any("preregistered_unseen_arm_evidence" in text for text in observations)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = "linear_readout_recalibration_sufficient"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="noisy-reference readout campaign"):
        build_noisy_reference_readout_recalibration_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_noisy_reference_readout_recalibration_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_noisy_reference_readout_recalibration_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-noisy-reference-readout-recalibration-v1",
        "experiment_count": 10,
    }
    assert (tmp_path / "experiment_queue").is_dir()
