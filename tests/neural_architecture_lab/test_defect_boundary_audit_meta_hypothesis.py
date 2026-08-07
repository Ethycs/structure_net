from __future__ import annotations

import json
from pathlib import Path

from neural_architecture_lab.defect_boundary_audit_meta_hypothesis import (
    build_defect_boundary_audit_experiment_results,
    build_defect_boundary_audit_meta_hypothesis,
    store_defect_boundary_audit_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_defect_boundary_audit/"
    "20260806_d6_preregistered_v3/campaign_results.json"
)


def test_meta_record_preserves_mixed_mechanism_rejection() -> None:
    record = build_defect_boundary_audit_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["boundary_only"] == "supported_one_of_three"
    assert record["hypothesis"]["subclaims"]["continuous_map_distortion"] == (
        "supported_two_of_three"
    )
    assert record["hypothesis"]["subclaims"]["refined_stable_ranks"] == (
        "seed29_rank5_seed53_rank3"
    )
    assert record["hypothesis"]["power_profile"] == (
        "underpowered_post_outcome_corrective_replication"
    )
    assert all(
        item["predecessor_replication"]["passed"]
        for item in record["evidence"]["direct_tests"]
    )


def test_two_checkpoint_records_preserve_refined_ranks() -> None:
    record = build_defect_boundary_audit_meta_hypothesis(RESULTS)
    experiments = build_defect_boundary_audit_experiment_results(record, RESULTS)
    assert len(experiments) == 2
    assert {item.metrics["refined_minimum_sufficient_rank"] for item in experiments} == {
        3.0, 5.0
    }
    assert sum(item.primary_metric == 1.0 for item in experiments) == 1


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_defect_boundary_audit_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 2
    assert len(read_back["source_artifacts"]) == 4


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_defect_boundary_audit_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-defect-boundary-correction-v1",
        "experiment_count": 2,
    }
