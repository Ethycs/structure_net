import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_temporal_sensor_only_meta_hypothesis import (
    build_c3_temporal_sensor_only_experiment_results,
    build_c3_temporal_sensor_only_meta_hypothesis,
    store_c3_temporal_sensor_only_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_only/"
    "20260811_preregistered/campaign_results.json"
)


def test_meta_preserves_five_zero_population_result() -> None:
    record = build_c3_temporal_sensor_only_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_task_only_sensor_acquisition_five_of_five"
    )
    assert hypothesis["subclaims"]["task_only_sensor_acquisition"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["target_shuffle_specificity"] == (
        "supported_zero_of_five"
    )
    assert hypothesis["subclaims"]["tinyllm_utility"] == (
        "not_tested_tinyllm_absent"
    )


def test_direct_evidence_preserves_carrier_and_complete_task_gates() -> None:
    record = build_c3_temporal_sensor_only_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 5
    assert all(cell["gates"]["learned_true_joint"] for cell in cells)
    assert not any(
        cell["gates"]["learned_target_shuffled_joint"] for cell in cells
    )
    for cell in cells:
        true = cell["arms"]["learned_true"]
        assert true["reload"]["pass"] is True
        assert true["exact_resume"]["pass"] is True
        for regime in ("composition", "extrapolation"):
            measured = true["regimes"][regime]
            assert measured["joint_pass"] is True
            assert measured["mean_aligned_unit_dot"] >= 0.99999
            assert measured["aligned_coordinate_rmse"] <= 0.003
            assert measured["maximum_deck_action_error"] <= 2e-6


def test_experiment_results_preserve_five_independent_sensor_units() -> None:
    record = build_c3_temporal_sensor_only_meta_hypothesis(RESULTS)
    experiments = build_c3_temporal_sensor_only_experiment_results(record, RESULTS)
    assert len(experiments) == 5
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert all(item.model_parameters == 184 for item in experiments)
    assert all(item.metrics["tinyllm_models_instantiated"] == 0.0 for item in experiments)
    assert min(item.metrics["true_extrapolation_accuracy"] for item in experiments) >= 0.90
    assert max(item.metrics["shuffled_extrapolation_accuracy"] for item in experiments) < 0.30


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["learned_true_joint_pass_count"] = 4
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 temporal sensor-only campaign"):
        build_c3_temporal_sensor_only_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_sensor_only_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_sensor_only_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-temporal-sensor-only-v1",
        "experiment_count": 5,
    }
