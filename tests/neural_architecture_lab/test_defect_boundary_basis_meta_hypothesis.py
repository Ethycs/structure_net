import json
from pathlib import Path

from neural_architecture_lab.defect_boundary_basis_meta_hypothesis import (
    build_defect_boundary_basis_experiment_results,
    build_defect_boundary_basis_meta_hypothesis,
    store_defect_boundary_basis_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_specificity_rejection_and_rank_refinement() -> None:
    record = build_defect_boundary_basis_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["paired_boundary_rank_one_sufficiency"] == (
        "supported_three_of_three"
    )
    assert record["hypothesis"]["subclaims"]["shuffled_and_random_specificity"] == (
        "not_supported_zero_of_three"
    )
    assert record["hypothesis"]["subclaims"]["adjacent_rank_refinement"] == (
        "supported_ranks_two_five_three"
    )
    assert record["hypothesis"]["evidence_role"] == (
        "preregistered_underpowered_mechanistic_evidence"
    )
    assert all(
        item["evidence_role"] == "preregistered_underpowered_mechanistic_evidence"
        for item in record["evidence"]["direct_tests"]
    )


def test_three_checkpoint_records_all_reject_full_hypothesis() -> None:
    record = build_defect_boundary_basis_meta_hypothesis(RESULTS)
    experiments = build_defect_boundary_basis_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert {item.metrics["base_rank"] for item in experiments} == {1.0, 2.0, 4.0}
    assert {item.metrics["minimum_geometric_correction"] for item in experiments} == {1.0}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 0


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_defect_boundary_basis_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_defect_boundary_basis_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-defect-boundary-basis-v1",
        "experiment_count": 3,
    }
