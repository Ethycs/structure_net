import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_temporal_quotient_meta_hypothesis import (
    build_c3_temporal_quotient_experiment_results,
    build_c3_temporal_quotient_meta_hypothesis,
    store_c3_temporal_quotient_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_temporal_quotient/"
    "20260811_d6_preregistered/campaign_results.json"
)


def test_meta_preserves_conservative_stopped_verdict() -> None:
    record = build_c3_temporal_quotient_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_analytic_positive_control_two_of_five_stop"
    )
    assert hypothesis["subclaims"]["learned_exact_c3_usefulness"] == (
        "untested_preregistered_positive_control_stop"
    )
    assert hypothesis["subclaims"]["raw_comparison"] == (
        "untested_preregistered_positive_control_stop"
    )
    assert hypothesis["subclaims"]["d10_extension"] == "unauthorized"


def test_meta_preserves_supported_narrow_subclaims() -> None:
    record = build_c3_temporal_quotient_meta_hypothesis(RESULTS)
    subclaims = record["hypothesis"]["subclaims"]
    assert subclaims["exact_analytic_c3_representation"] == (
        "supported_five_of_five"
    )
    assert subclaims["analytic_causal_closure_all_four_cuts"] == (
        "supported_five_of_five"
    )
    assert subclaims["analytic_natural_extrapolating_utility"] == (
        "failed_population_gate_two_of_five"
    )


def test_direct_evidence_preserves_five_seed_population() -> None:
    record = build_c3_temporal_quotient_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 5
    assert sum(cell["natural_task_pass"] for cell in cells) == 2
    assert all(cell["representation_pass"] for cell in cells)
    assert all(cell["causal_all_cuts_pass"] for cell in cells)
    assert all(cell["exact_action_pass"] for cell in cells)
    assert all(cell["identity_replay_pass"] for cell in cells)
    assert not any(
        any(cell["target_derangement_pass_by_regime"].values())
        for cell in cells
    )
    assert min(
        cell["minimum_registered_target_correlation"] for cell in cells
    ) >= 0.90
    assert max(cell["maximum_conditional_deck_accuracy"] for cell in cells) == (
        pytest.approx(1 / 3)
    )
    assert max(
        cell["maximum_conditional_log_loss_gain"] for cell in cells
    ) <= 0.02


def test_post_outcome_diagnostic_is_separate_and_no_fit() -> None:
    record = build_c3_temporal_quotient_meta_hypothesis(RESULTS)
    diagnostic = record["evidence"]["derived_diagnostic"]
    assert diagnostic["classification"] == (
        "invariant_sensor_valid_trained_continuation_readout_"
        "extrapolation_unreliable"
    )
    assert diagnostic["counts"]["checkpoints_loaded"] == 0
    assert diagnostic["counts"]["optimizer_steps"] == 0
    assert diagnostic["counts"]["trained_parameters"] == 0
    assert diagnostic["gates"]["fixed_no_model_task"] is True
    assert (
        diagnostic["fixed_no_model_positive_control"]["extrapolation"]
        ["temporal_prediction_correlation"]
        > 0.9999
    )


def test_experiment_results_are_only_the_five_executed_analytic_cells() -> None:
    record = build_c3_temporal_quotient_meta_hypothesis(RESULTS)
    experiments = build_c3_temporal_quotient_experiment_results(record, RESULTS)
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 2
    assert all("-analytic-" in item.experiment_id for item in experiments)
    assert all(item.metrics["representation_pass"] == 1 for item in experiments)
    assert all(item.metrics["causal_all_cuts_pass"] == 1 for item in experiments)
    assert all(item.metrics["optimizer_steps"] == 600 for item in experiments)
    assert all(item.metrics["learned_encoder_parameters"] == 0 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = (
        "c3_d6_structured_quotient_supported"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 temporal quotient campaign"):
        build_c3_temporal_quotient_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_quotient_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_quotient_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-temporal-quotient-training-v1",
        "experiment_count": 5,
    }
