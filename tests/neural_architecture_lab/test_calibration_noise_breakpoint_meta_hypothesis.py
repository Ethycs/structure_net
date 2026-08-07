import json
from pathlib import Path

import pytest

from neural_architecture_lab.calibration_noise_breakpoint_meta_hypothesis import (
    build_calibration_noise_breakpoint_experiment_results,
    build_calibration_noise_breakpoint_meta_hypothesis,
    store_calibration_noise_breakpoint_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_calibration_degradation/"
    "20260807_d8_preregistered/campaign_results.json"
)


def test_meta_preserves_bounded_representation_breakpoints() -> None:
    record = build_calibration_noise_breakpoint_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    arms = record["result"]["descriptive_metrics"]["campaign"]["arms"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_bounded_stable_representation_breakpoints"
    )
    assert arms["analytic_calibrated"]["breakpoint"][
        "breakpoint_interval"
    ] == [1, 2]
    assert arms["learned_calibrated_equivariant"]["breakpoint"][
        "breakpoint_interval"
    ] == [0.5, 1]
    assert hypothesis["subclaims"]["frozen_task_utility_robustness"] == (
        "not_an_endpoint"
    )


def test_experiment_records_preserve_seedwise_curves() -> None:
    record = build_calibration_noise_breakpoint_meta_hypothesis(RESULTS)
    experiments = build_calibration_noise_breakpoint_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert sorted(item.primary_metric for item in experiments) == [
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert all(item.metrics["shuffled_joint_pass"] == 0.0 for item in experiments)
    assert all(item.metrics["maximum_branch_accuracy"] <= 0.55 for item in experiments)
    assert len({item.model_checkpoint for item in experiments}) == 10


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["supported"] = False
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration-noise breakpoint campaign"):
        build_calibration_noise_breakpoint_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_noise_breakpoint_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_noise_breakpoint_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-calibration-noise-breakpoint-v1",
        "experiment_count": 10,
    }
    assert (tmp_path / "experiment_queue").is_dir()
