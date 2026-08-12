import json
from pathlib import Path

import pytest

from neural_architecture_lab.frozen_scalar_interface_decomposition_meta_hypothesis import (
    build_frozen_scalar_interface_decomposition_experiment_results,
    build_frozen_scalar_interface_decomposition_meta_hypothesis,
    store_frozen_scalar_interface_decomposition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_frozen_scalar_interface_decomposition/"
    "20260811_d6_d10_preregistered/campaign_results.json"
)


def test_meta_records_failed_upstream_claim_and_localized_subclaims() -> None:
    record = build_frozen_scalar_interface_decomposition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_exact_cosine_repairs_one_of_ten"
    )
    assert hypothesis["direct_experiment_count"] == 20
    assert hypothesis["subclaims"]["uniformly_upstream_exact_cosine"] == (
        "contradicted_one_of_ten"
    )
    assert hypothesis["subclaims"][
        "target_bin_reachability_clears_source_floor"
    ] == "supported_ten_of_ten_secondary"


def test_meta_preserves_locked_counts_and_exception() -> None:
    record = build_frozen_scalar_interface_decomposition_meta_hypothesis(RESULTS)
    campaign = record["result"]["descriptive_metrics"]["campaign"]
    assert campaign["exact_cosine_repair_count"] == 1
    assert campaign["oracle_repair_count"] == 9
    assert campaign["classification_counts"] == {
        "continuation_or_answer_row_failure": 1,
        "scalar_coordinate_or_boundary_failure": 8,
        "sensor_scalar_estimation_failure": 1,
        "source_passing_positive_control": 10,
    }
    exceptional = next(
        item
        for item in record["evidence"]["direct_tests"]
        if item["preset"] == "d10"
        and item["condition"] == "learned_calibrated_equivariant"
        and item["seed"] == 29
    )
    assert exceptional["classification"] == "continuation_or_answer_row_failure"
    assert exceptional["composition_target_bin_reachability"] == pytest.approx(
        0.845703125
    )


def test_experiment_results_preserve_all_twenty_frozen_units() -> None:
    record = build_frozen_scalar_interface_decomposition_meta_hypothesis(RESULTS)
    experiments = build_frozen_scalar_interface_decomposition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 20
    source_failed = [item for item in experiments if item.metrics["source_failed"]]
    assert len(source_failed) == 10
    assert sum(item.primary_metric for item in source_failed) == 1
    assert sum(item.metrics["oracle_both_shifts"] for item in source_failed) == 9
    assert all(item.metrics["validity"] == 1 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["exact_cosine_repair_count"] = 10
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen scalar-interface campaign"):
        build_frozen_scalar_interface_decomposition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_scalar_interface_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 20


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_scalar_interface_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-frozen-scalar-interface-decomposition-v1",
        "experiment_count": 20,
    }
