import json
from pathlib import Path

import pytest

from neural_architecture_lab.observed_deck_twirl_meta_hypothesis import (
    build_observed_deck_twirl_experiment_results,
    build_observed_deck_twirl_meta_hypothesis,
    store_observed_deck_twirl_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_observed_deck_twirl/"
    "20260810_d10_preregistered/campaign_results.json"
)


def test_meta_confirms_only_the_observed_structured_action_scope() -> None:
    record = build_observed_deck_twirl_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "preregistered_observable_twirl_closed_action_invariant_five_of_five"
    )
    assert hypothesis["direct_experiment_count"] == 10
    assert hypothesis["subclaims"]["oracle_fiber_membership_required"] == (
        "no_for_structured_C2_projection"
    )


def test_meta_preserves_specificity_and_oracle_denoising_boundary() -> None:
    record = build_observed_deck_twirl_meta_hypothesis(RESULTS)
    campaign = record["result"]["descriptive_metrics"]["campaign"]
    for condition in (
        "analytic_calibrated",
        "learned_calibrated_equivariant",
    ):
        assert set(campaign["arms"][condition]["twirl_pass_counts"].values()) == {5}
        assert set(
            campaign["arms"][condition]["control_twirl_pass_counts"].values()
        ) == {0}
    assert record["hypothesis"]["subclaims"][
        "oracle_independent_nuisance_gain_deployable"
    ] == "no_conflates_projection_and_denoising"


def test_checkpoint_summaries_retain_action_and_control_effects() -> None:
    record = build_observed_deck_twirl_meta_hypothesis(RESULTS)
    experiments = build_observed_deck_twirl_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert all(item.metrics["all_control_twirls_fail"] == 1.0 for item in experiments)
    assert all(item.metrics["minimum_control_twirl_accuracy_loss"] > 0.03 for item in experiments)
    assert all(item.metrics["maximum_correct_action_accuracy_loss"] <= 0.001 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["action_contract"]["pass"] = False
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="observed-deck twirl campaign"):
        build_observed_deck_twirl_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_observed_deck_twirl_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_observed_deck_twirl_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-observed-deck-twirl-causal-closure-v1",
        "experiment_count": 10,
    }
