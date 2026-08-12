import json
from pathlib import Path

import pytest

from neural_architecture_lab.observed_cyclic_action_semantics_meta_hypothesis import (
    build_observed_cyclic_action_semantics_experiment_results,
    build_observed_cyclic_action_semantics_meta_hypothesis,
    store_observed_cyclic_action_semantics_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_observed_cyclic_action_semantics/"
    "20260810_d6_preregistered/campaign_results.json"
)


def test_meta_preserves_stage_a_stop_and_oracle_attribution() -> None:
    record = build_observed_cyclic_action_semantics_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "preregistered_stage_a_null_no_observable_candidate_"
        "causal_stage_not_run"
    )
    assert hypothesis["subclaims"]["requantization_alone_explains_gap"] == "no"
    assert hypothesis["subclaims"]["phase_estimation_alone_explains_gap"] == (
        "no_oracle_also_fails"
    )
    assert hypothesis["subclaims"]["causal_stage_authorized"] == "no"


def test_experiment_records_are_input_only_cells() -> None:
    record = build_observed_cyclic_action_semantics_meta_hypothesis(RESULTS)
    experiments = build_observed_cyclic_action_semantics_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 20
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.model_parameters == 0 for item in experiments)
    assert all(item.model_checkpoint is None for item in experiments)
    assert all(
        item.metrics["causal_model_evaluation_run"] == 0.0
        for item in experiments
    )


def test_campaign_cannot_rewrite_stop_result(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["causal_stage_authorized"] = True
    campaign["aggregates"]["selected_variant"] = "residual_fixed_continuous"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="action-semantics campaign"):
        build_observed_cyclic_action_semantics_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_observed_cyclic_action_semantics_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 20


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_observed_cyclic_action_semantics_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-observed-cyclic-action-semantics-front-v1",
        "experiment_count": 20,
    }
