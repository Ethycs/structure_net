import json
from pathlib import Path

import pytest

from neural_architecture_lab.nuisance_scalar_transformation_law_meta_hypothesis import (
    build_nuisance_scalar_transformation_law_experiment_results,
    build_nuisance_scalar_transformation_law_meta_hypothesis,
    store_nuisance_scalar_transformation_law_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_nuisance_scalar_transformation_law/"
    "20260807_d6_existing_group_gauge_replay/campaign_results.json"
)


def test_meta_preserves_action_dependent_type_falsification() -> None:
    record = build_nuisance_scalar_transformation_law_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "rejected_scalar_action_dependent_three_of_three"
    )
    assert hypothesis["subclaims"]["invariant_scalar_output_type"] == (
        "structurally_incompatible"
    )
    assert all(
        item["classification"] == "scalar_action_dependent"
        for item in record["evidence"]["direct_tests"]
    )
    assert all(
        item["gates"]["basis_gauge_replay_contract"] is True
        for item in record["evidence"]["direct_tests"]
    )


def test_experiment_records_preserve_all_cell_failure() -> None:
    record = build_nuisance_scalar_transformation_law_meta_hypothesis(RESULTS)
    experiments = build_nuisance_scalar_transformation_law_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["cell_invariance_pass_count"] == 0.0 for item in experiments)
    assert all(
        item.metrics["maximum_paired_zero_referenced_r2"] < 0.0
        for item in experiments
    )
    assert all(
        item.metrics["minimum_paired_relative_l2"] > 1.0
        for item in experiments
    )


def test_campaign_cannot_rewrite_action_dependence_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["primary_checkpoint_pass_count"] = 3
    campaign["aggregates"]["conclusion"] = (
        "supported_scalar_nuisance_invariance_three_of_three"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="nuisance-scalar transformation campaign"):
        build_nuisance_scalar_transformation_law_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_nuisance_scalar_transformation_law_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_nuisance_scalar_transformation_law_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-nuisance-scalar-transformation-law-v1",
        "experiment_count": 3,
    }
