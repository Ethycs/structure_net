import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_temporal_continuation_readout_meta_hypothesis import (
    build_c3_temporal_continuation_readout_experiment_results,
    build_c3_temporal_continuation_readout_meta_hypothesis,
    store_c3_temporal_continuation_readout_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_temporal_continuation_readout/"
    "20260811_d6_preregistered/campaign_results.json"
)


def test_meta_preserves_negative_primary_verdict() -> None:
    record = build_c3_temporal_continuation_readout_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_typed_final_readout_one_of_five"
    )
    assert hypothesis["subclaims"]["typed_affine_final_interface"] == (
        "not_supported_one_of_five_complete_seed_passes"
    )


def test_meta_preserves_comparators_and_specificity() -> None:
    record = build_c3_temporal_continuation_readout_meta_hypothesis(RESULTS)
    subclaims = record["hypothesis"]["subclaims"]
    assert subclaims["free_answer_row_replacement"] == (
        "not_supported_zero_of_five"
    )
    assert subclaims["inherited_output_scalar_recalibration"] == (
        "not_supported_zero_of_five"
    )
    assert subclaims["target_shuffle_specificity"] == (
        "supported_zero_of_fifteen"
    )
    assert subclaims["raw_and_learned_c3_arms"] == (
        "untested_predecessor_stop_retained"
    )


def test_direct_evidence_preserves_five_frozen_seed_units() -> None:
    record = build_c3_temporal_continuation_readout_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 5
    assert sum(
        cell["arm_gates"]["typed_final_readout"]["true_both_shifts"]
        for cell in cells
    ) == 1
    assert not any(
        cell["arm_gates"][arm]["shuffled_both_shifts"]
        for cell in cells
        for arm in (
            "output_scalar_recalibration",
            "untyped_final_readout",
            "typed_final_readout",
        )
    )
    assert all(cell["exact_temporal_bypass_pass"] for cell in cells)


def test_typed_state_retains_approximate_extrapolating_coordinate() -> None:
    record = build_c3_temporal_continuation_readout_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    extrapolation = [
        cell["regimes"]["extrapolation"]["arms"]["typed_final_readout"]
        for cell in cells
    ]
    assert min(
        cell["scalar_true"]["cosine_pearson"] for cell in extrapolation
    ) >= 0.90
    assert min(
        cell["true_metrics"]["exact_bin_accuracy"] for cell in extrapolation
    ) >= 0.35
    assert sum(
        cell["true_metrics"]["target_cross_entropy"] <= 2.20
        for cell in extrapolation
    ) == 1


def test_experiment_results_preserve_zero_training_and_one_primary_pass() -> None:
    record = build_c3_temporal_continuation_readout_meta_hypothesis(RESULTS)
    experiments = build_c3_temporal_continuation_readout_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 1
    assert all(item.metrics["optimizer_steps"] == 0 for item in experiments)
    assert all(
        item.metrics["trained_model_parameters"] == 0 for item in experiments
    )
    assert all(item.metrics["exact_temporal_bypass_pass"] == 1 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["primary_hypothesis_pass"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 continuation/readout campaign"):
        build_c3_temporal_continuation_readout_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_continuation_readout_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_continuation_readout_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-temporal-continuation-readout-v1",
        "experiment_count": 5,
    }
