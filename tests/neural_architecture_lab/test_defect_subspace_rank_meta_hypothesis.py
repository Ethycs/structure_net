from __future__ import annotations

import json
from pathlib import Path

from neural_architecture_lab.defect_subspace_rank_meta_hypothesis import (
    build_defect_subspace_rank_experiment_results,
    build_defect_subspace_rank_meta_hypothesis,
    store_defect_subspace_rank_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_scalar_rejection_and_low_rank_result() -> None:
    record = build_defect_subspace_rank_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["rank_one_sufficiency"] == (
        "not_supported_zero_of_three"
    )
    assert record["hypothesis"]["subclaims"]["low_dimensional_fixed_subspace"] == (
        "supported_ranks_two_eight_four"
    )
    assert record["hypothesis"]["power_profile"] == (
        "underpowered_three_checkpoint_mechanistic_cohort"
    )
    assert [
        item["minimum_fixed_rank"]["causal_pass"]
        for evidence in record["evidence"]["direct_tests"]
        for item in evidence["heldout_cells"]
    ] == [True] * 12


def test_three_cells_convert_with_measured_minimum_ranks() -> None:
    record = build_defect_subspace_rank_meta_hypothesis(RESULTS)
    experiments = build_defect_subspace_rank_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert {item.metrics["minimum_fixed_sufficient_rank"] for item in experiments} == {
        2.0, 4.0, 8.0
    }
    assert sum(item.primary_metric == 1.0 for item in experiments) == 0


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_defect_subspace_rank_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 3
    assert len(read_back["evidence"]["direct_tests"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_defect_subspace_rank_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-defect-subspace-rank-v1",
        "experiment_count": 3,
    }
