from __future__ import annotations

import json
from pathlib import Path

from neural_architecture_lab.c2_attention_head_meta_hypothesis import (
    build_c2_attention_head_experiment_results,
    build_c2_attention_head_meta_hypothesis,
    store_c2_attention_head_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c2_attention_head_decomposition/"
    "20260806_d6_preregistered/campaign_results.json"
)


def test_meta_record_preserves_exact_control_and_sparse_failure() -> None:
    record = build_c2_attention_head_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["evidence_count"] == 5
    assert hypothesis["subclaims"]["exact_projected_head_decomposition"] == (
        "supported_three_of_three_primary"
    )
    assert hypothesis["subclaims"]["one_or_two_head_source_selection"] == (
        "not_supported_zero_of_three_primary"
    )
    primary = [
        item for item in record["evidence"]["direct_tests"] if item["primary_seed"]
    ]
    assert sorted(item["selected_cardinality"] for item in primary) == [4, 4, 5]


def test_all_cells_convert_without_upgrading_the_failed_gate() -> None:
    record = build_c2_attention_head_meta_hypothesis(RESULTS)
    experiments = build_c2_attention_head_experiment_results(record, RESULTS)
    assert len(experiments) == 5
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 0


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c2_attention_head_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 5
    assert len(read_back["evidence"]["direct_tests"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c2_attention_head_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-sparse-attention-head-synthesis-v1",
        "experiment_count": 5,
    }

