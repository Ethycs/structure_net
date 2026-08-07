import json
from pathlib import Path

import pytest

from neural_architecture_lab.scalar_action_defect_decomposition_meta_hypothesis import (
    build_scalar_action_defect_decomposition_experiment_results,
    build_scalar_action_defect_decomposition_meta_hypothesis,
    store_scalar_action_defect_decomposition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_scalar_action_defect_decomposition/"
    "20260807_d6_stored_arrays_gauge_replay/campaign_results.json"
)


def test_meta_preserves_residual_coordinate_mechanism() -> None:
    record = build_scalar_action_defect_decomposition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_residual_coordinate_defect_three_of_three"
    )
    assert hypothesis["subclaims"]["prospective_output_type"] == (
        "action_conditioned_scalar_amplitude"
    )
    assert all(
        item["classification"] == "residual_coordinate_defect"
        for item in record["evidence"]["direct_tests"]
    )


def test_experiment_records_preserve_component_separation() -> None:
    record = build_scalar_action_defect_decomposition_meta_hypothesis(RESULTS)
    experiments = build_scalar_action_defect_decomposition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 3.0
    assert all(
        item.metrics["minimum_residual_zero_referenced_r2"] > 0.998
        for item in experiments
    )
    assert all(
        item.metrics["maximum_covector_zero_referenced_r2"] < 0.01
        for item in experiments
    )
    assert all(
        item.metrics["residual_component_all_cells_pass"] == 1.0
        for item in experiments
    )


def test_campaign_cannot_rewrite_component_result(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["primary_checkpoint_pass_count"] = 0
    campaign["aggregates"]["conclusion"] = "coupled_action_defect"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="scalar action-defect campaign"):
        build_scalar_action_defect_decomposition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_scalar_action_defect_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_scalar_action_defect_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-scalar-action-defect-decomposition-v1",
        "experiment_count": 3,
    }
