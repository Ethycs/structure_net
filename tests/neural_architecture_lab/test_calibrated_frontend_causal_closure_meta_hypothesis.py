import json
from pathlib import Path

import pytest

from neural_architecture_lab.calibrated_frontend_causal_closure_meta_hypothesis import (
    build_calibrated_frontend_causal_closure_experiment_results,
    build_calibrated_frontend_causal_closure_meta_hypothesis,
    store_calibrated_frontend_causal_closure_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
    "20260810_d15_preregistered/campaign_results.json"
)


def test_meta_confirms_only_the_preregistered_structured_scope() -> None:
    record = build_calibrated_frontend_causal_closure_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "preregistered_frontend_causal_quotient_closed_five_of_five"
    )
    assert hypothesis["direct_experiment_count"] == 10
    assert hypothesis["subclaims"]["raw_calibrated_comparator"] == (
        "descriptive_low_baseline_only"
    )


def test_meta_preserves_causal_gates_controls_and_noninvariance() -> None:
    record = build_calibrated_frontend_causal_closure_meta_hypothesis(RESULTS)
    campaign = record["result"]["descriptive_metrics"]["campaign"]
    for condition in (
        "analytic_calibrated",
        "learned_calibrated_equivariant",
    ):
        assert set(campaign["arms"][condition]["cut_pass_counts"].values()) == {5}
        assert set(campaign["arms"][condition]["shuffle_pass_counts"].values()) == {0}
    assert record["hypothesis"]["subclaims"]["literal_residual_invariance"] == (
        "not_supported_natural_pair_differences_persist"
    )


def test_checkpoint_summaries_retain_effect_sizes() -> None:
    record = build_calibrated_frontend_causal_closure_meta_hypothesis(RESULTS)
    experiments = build_calibrated_frontend_causal_closure_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert all(item.metrics["minimum_exact_accuracy_gain"] > 0.0 for item in experiments)
    assert all(item.metrics["minimum_cross_entropy_reduction"] > 0.0 for item in experiments)
    assert all(item.metrics["all_shuffled_cuts_fail"] == 1.0 for item in experiments)
    assert all(item.metrics["maximum_natural_pair_relative_rms"] > 0.0 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["primary_hypothesis_pass"] = False
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="causal-closure campaign"):
        build_calibrated_frontend_causal_closure_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibrated_frontend_causal_closure_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibrated_frontend_causal_closure_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-calibrated-frontend-causal-closure-v1",
        "experiment_count": 10,
    }
