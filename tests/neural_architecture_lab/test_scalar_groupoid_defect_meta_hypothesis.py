import json
from pathlib import Path

import pytest

from neural_architecture_lab.scalar_groupoid_defect_meta_hypothesis import (
    build_scalar_groupoid_defect_experiment_results,
    build_scalar_groupoid_defect_meta_hypothesis,
    store_scalar_groupoid_defect_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_scalar_groupoid_defect/"
    "20260807_d6_existing_group_gauge_replay/campaign_results.json"
)


def test_meta_preserves_two_term_rejection() -> None:
    record = build_scalar_groupoid_defect_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_two_term_groupoid_defect_three_of_three"
    )
    assert hypothesis["subclaims"]["direct_state_term_negligible"] == (
        "rejected_zero_of_twenty_four_cells"
    )
    assert hypothesis["subclaims"]["writer_only_defect_prediction"] == (
        "rejected_zero_of_twenty_four_cells"
    )
    assert hypothesis["subclaims"]["frozen_sidecar_observability"] == (
        "closed_in_tested_scope"
    )
    assert hypothesis["subclaims"]["calibrated_frontend"] == (
        "preferred_constructive_path"
    )


def test_experiment_records_preserve_exact_component_split() -> None:
    record = build_scalar_groupoid_defect_meta_hypothesis(RESULTS)
    experiments = build_scalar_groupoid_defect_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(
        item.metrics["minimum_direct_to_total_rms_ratio"] > 0.80
        for item in experiments
    )
    assert all(
        item.metrics["maximum_writer_only_zero_referenced_r2"] < 0.35
        for item in experiments
    )
    assert all(
        item.metrics["exact_groupoid_identity_all_cells"] == 1.0
        for item in experiments
    )


def test_campaign_cannot_rewrite_two_term_result(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["primary_checkpoint_pass_count"] = 3
    campaign["aggregates"]["conclusion"] = (
        "supported_writer_symmetry_defect_dominant_three_of_three"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="scalar groupoid-defect campaign"):
        build_scalar_groupoid_defect_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_scalar_groupoid_defect_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_scalar_groupoid_defect_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-scalar-groupoid-defect-v1",
        "experiment_count": 3,
    }
    assert (tmp_path / "experiment_queue").is_dir()
