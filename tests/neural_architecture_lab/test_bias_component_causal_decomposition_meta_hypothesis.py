import json
from pathlib import Path

import pytest

from neural_architecture_lab.bias_component_causal_decomposition_meta_hypothesis import (
    build_bias_component_causal_decomposition_experiment_results,
    build_bias_component_causal_decomposition_meta_hypothesis,
    store_bias_component_causal_decomposition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_bias_component_causal_decomposition/"
    "20260810_d10_preregistered/campaign_results.json"
)


def test_meta_preserves_confirmed_causal_component_result() -> None:
    record = build_bias_component_causal_decomposition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_deterministic_positive_mean_sufficient"
    )
    assert hypothesis["evidence_count"] == 10
    assert hypothesis["integrity_valid_experiment_count"] == 10
    assert hypothesis["subclaims"] == {
        "centered_stochastic_utility": "supported_five_of_five_both_arms",
        "positive_mean_sufficiency": (
            "supported_analytic_two_of_five_learned_three_of_five_pass"
        ),
        "source_full_positive_failure": (
            "reproduced_analytic_one_of_five_learned_three_of_five"
        ),
        "sign_reversal_recovery": "supported_four_of_five_both_arms",
        "sign_mechanism": "positive_direction_specific",
        "measurement_robustness_equals_group_closure": "rejected",
        "same_scope_retraining": "not_licensed",
    }


def test_experiment_records_preserve_component_pass_counts() -> None:
    record = build_bias_component_causal_decomposition_meta_hypothesis(RESULTS)
    experiments = build_bias_component_causal_decomposition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert all(item.metrics["validity"] == 1.0 for item in experiments)
    assert sum(item.metrics["centered_seed_pass"] for item in experiments) == 10.0
    assert sum(item.metrics["mean_plus_seed_pass"] for item in experiments) == 5.0
    assert sum(item.metrics["full_plus_seed_pass"] for item in experiments) == 4.0
    assert sum(item.metrics["full_minus_seed_pass"] for item in experiments) == 8.0
    assert sum(item.primary_metric for item in experiments) == 5.0


def test_campaign_cannot_rewrite_confirmation(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["primary_hypothesis_pass"] = False
    campaign["aggregates"]["classification"] = "mean_noise_interaction"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="bias-component campaign"):
        build_bias_component_causal_decomposition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_bias_component_causal_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_bias_component_causal_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-bias-component-causal-decomposition-v1",
        "experiment_count": 10,
    }
