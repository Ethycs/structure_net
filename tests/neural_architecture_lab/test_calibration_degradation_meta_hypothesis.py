import json
from pathlib import Path

import pytest

from neural_architecture_lab.calibration_degradation_meta_hypothesis import (
    build_calibration_degradation_experiment_results,
    build_calibration_degradation_meta_hypothesis,
    store_calibration_degradation_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_calibration_degradation_causal/"
    "20260807_d8_existing_checkpoints/campaign_results.json"
)


def test_meta_rejects_robustness_but_preserves_representation_split() -> None:
    record = build_calibration_degradation_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    split = record["result"]["descriptive_metrics"]["representation_task_split"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_exact_calibration_required"
    )
    assert hypothesis["subclaims"]["full_preregistered_hypothesis"] == "rejected"
    assert hypothesis["subclaims"]["representation_joint_gate_through_0p20"] == (
        "supported_160_of_160_cells"
    )
    assert split["through_target_representation_passes"] == 160
    assert split["primary_branch_passes"] == 240
    assert split["primary_log_loss_passes"] == 240
    assert split["target_task_shift_passes"] == {
        "analytic_calibrated": 0,
        "learned_calibrated_equivariant": 1,
    }


def test_meta_preserves_arm_radii_and_component_ablations() -> None:
    record = build_calibration_degradation_meta_hypothesis(RESULTS)
    campaign = record["result"]["descriptive_metrics"]["campaign"]
    assert campaign["classification"] == "exact_calibration_required"
    assert campaign["arms"]["analytic_calibrated"]["robust_radius"] == 0.0
    assert campaign["arms"]["learned_calibrated_equivariant"][
        "robust_radius"
    ] == 0.05
    assert campaign["arms"]["analytic_calibrated"]["ablation_pass_counts"][
        "amplitude_default"
    ] == 5
    assert campaign["arms"]["learned_calibrated_equivariant"][
        "ablation_pass_counts"
    ]["amplitude_default"] == 0


def test_experiment_records_are_inference_only_and_checkpoint_specific() -> None:
    record = build_calibration_degradation_meta_hypothesis(RESULTS)
    experiments = build_calibration_degradation_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert sorted(item.primary_metric for item in experiments) == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
    ]
    assert all(item.metrics["target_representation_pass_count"] == 4 for item in experiments)
    assert all(item.metrics["clean_replay_pass"] == 1.0 for item in experiments)
    assert len({item.model_checkpoint for item in experiments}) == 10


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["arms"]["analytic_calibrated"]["robust_radius"] = 0.2
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration-degradation campaign"):
        build_calibration_degradation_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_degradation_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_degradation_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-calibrated-reference-robustness-curve-v1",
        "experiment_count": 10,
    }
    assert (tmp_path / "experiment_queue").is_dir()
