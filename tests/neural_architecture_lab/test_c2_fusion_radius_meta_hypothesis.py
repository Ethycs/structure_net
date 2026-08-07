from __future__ import annotations

import json
from pathlib import Path

from neural_architecture_lab.c2_fusion_radius_meta_hypothesis import (
    build_c2_fusion_radius_experiment_results,
    build_c2_fusion_radius_meta_hypothesis,
    store_c2_fusion_radius_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c2_fusion_radius/"
    "20260806_d6_preregistered/campaign_results.json"
)


def test_meta_record_preserves_failed_radius_claim_and_specificity() -> None:
    record = build_c2_fusion_radius_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["evidence_count"] == 5
    assert hypothesis["subclaims"]["early_front_locality"] == (
        "not_supported_zero_of_three"
    )
    assert hypothesis["subclaims"]["character_direction_control_specificity"] == (
        "supported_five_of_five"
    )


def test_all_five_cells_convert_without_upgrading_failure() -> None:
    record = build_c2_fusion_radius_meta_hypothesis(RESULTS)
    experiments = build_c2_fusion_radius_experiment_results(record, RESULTS)
    assert len(experiments) == 5
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 0
    seed17 = next(item for item in experiments if item.experiment_id.endswith("seed17"))
    assert seed17.metrics["composition_onset_radius"] == -1.0


def test_store_json_round_trip_preserves_ids_and_verdict(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c2_fusion_radius_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    read_back = json.loads(output.read_text())
    assert read_back["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert read_back["hypothesis"]["confirmed"] is False
    assert len(read_back["storage"]["experiment_ids"]) == 5
    assert len(read_back["evidence"]["direct_tests"]) == 5


def test_chromadb_round_trip_verifies_hypothesis_and_experiments(
    tmp_path: Path,
) -> None:
    output = tmp_path / "record.json"
    stored = store_c2_fusion_radius_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-character-fusion-radius-v1",
        "experiment_count": 5,
    }
