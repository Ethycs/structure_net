import json
from pathlib import Path

import pytest

from neural_architecture_lab.task_activation_scalar_sensor_meta_hypothesis import (
    build_task_activation_scalar_sensor_experiment_results,
    build_task_activation_scalar_sensor_meta_hypothesis,
    store_task_activation_scalar_sensor_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_task_activation_scalar_sensor/"
    "20260807_d6_fresh_e/campaign_results.json"
)


def test_meta_preserves_activation_scalar_component_split() -> None:
    record = build_task_activation_scalar_sensor_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_observable_scalar_not_identified_three_of_three"
    )
    assert hypothesis["subclaims"]["portable_covector_with_exact_scalar"] == (
        "supported_three_of_three_six_of_six_cells"
    )
    assert hypothesis["subclaims"]["primary_activation_scalar_causal_composition"] == (
        "descriptively_supported_three_of_three"
    )
    assert hypothesis["subclaims"]["primary_activation_scalar_causal_extrapolation"] == (
        "not_supported_zero_of_three"
    )


def test_experiment_records_preserve_support_relative_causal_result() -> None:
    record = build_task_activation_scalar_sensor_meta_hypothesis(RESULTS)
    experiments = build_task_activation_scalar_sensor_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(
        item.metrics["composition_primary_causal_pass"] == 1.0
        for item in experiments
    )
    assert all(
        item.metrics["extrapolation_primary_causal_pass"] == 0.0
        for item in experiments
    )
    assert all(
        item.metrics["source_covector_oracle_error_all_fresh_cells_pass"] == 1.0
        for item in experiments
    )


def test_campaign_cannot_rewrite_scalar_failure_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["primary_checkpoint_pass_count"] = 3
    campaign["aggregates"]["conclusion"] = "causal_activation_scalar_supported"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(
        ValueError, match="task-activation scalar-sensor campaign"
    ):
        build_task_activation_scalar_sensor_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_task_activation_scalar_sensor_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_task_activation_scalar_sensor_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-task-activation-scalar-sensor-v2",
        "experiment_count": 3,
    }
