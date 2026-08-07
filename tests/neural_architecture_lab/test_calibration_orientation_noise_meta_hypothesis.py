import json
from pathlib import Path

import pytest

from neural_architecture_lab.calibration_orientation_noise_meta_hypothesis import (
    build_calibration_orientation_noise_experiment_results,
    build_calibration_orientation_noise_meta_hypothesis,
    store_calibration_orientation_noise_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_calibration_orientation_noise/"
    "20260807_d8_preregistered/campaign_results.json"
)


def test_meta_preserves_reference_precision_classification() -> None:
    record = build_calibration_orientation_noise_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    split = record["result"]["descriptive_metrics"][
        "representation_task_split"
    ]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_reference_precision_critical"
    )
    assert hypothesis["subclaims"]["analytic_complete_radius"] == (
        "zero_population_radius"
    )
    assert hypothesis["subclaims"]["learned_complete_radius"] == (
        "zero_population_radius"
    )
    assert split["representation_passes_through_ten_degrees"] == 160
    assert split["first_noise_complete_passes"] == {
        "analytic_calibrated": 0,
        "learned_calibrated_equivariant": 2,
    }


def test_meta_preserves_representation_task_separation() -> None:
    record = build_calibration_orientation_noise_meta_hypothesis(RESULTS)
    split = record["result"]["descriptive_metrics"][
        "representation_task_split"
    ]
    assert split["representation_passes"] == 168
    assert split["representation_cells"] == 280
    assert split["branch_passes"] == 280
    assert split["log_loss_passes"] == 280
    assert split["first_noise_task_passes"] == {
        "analytic_calibrated": 0,
        "learned_calibrated_equivariant": 2,
    }


def test_experiment_records_are_frozen_checkpoint_specific() -> None:
    record = build_calibration_orientation_noise_meta_hypothesis(RESULTS)
    experiments = build_calibration_orientation_noise_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert sorted(item.primary_metric for item in experiments) == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.035,
        0.035,
    ]
    assert all(
        item.metrics["first_noise_representation_pass"] == 1.0
        for item in experiments
    )
    assert all(
        item.metrics["last_representation_pass_sigma_radians"] == 0.175
        for item in experiments
    )
    assert len({item.model_checkpoint for item in experiments}) == 10


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = "learned_matches_analytic_radius"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration-orientation campaign"):
        build_calibration_orientation_noise_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_orientation_noise_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibration_orientation_noise_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-calibrated-orientation-noise-radius-v1",
        "experiment_count": 10,
    }
    assert (tmp_path / "experiment_queue").is_dir()
