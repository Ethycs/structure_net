import json
from pathlib import Path

import pytest

from neural_architecture_lab.final_query_semantic_kernel_meta_hypothesis import (
    build_final_query_semantic_kernel_experiment_results,
    build_final_query_semantic_kernel_meta_hypothesis,
    store_final_query_semantic_kernel_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_final_query_semantic_kernel/"
    "20260810_d8_seed7_preregistered/campaign_results.json"
)


def test_meta_preserves_scalar_kernel_null_and_nested_tangent() -> None:
    record = build_final_query_semantic_kernel_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_scalar_kernel_no_contraction_or_shift_stable_preservation"
    )
    assert hypothesis["evidence_count"] == 1
    assert hypothesis["direct_quality_experiment_count"] == 1
    assert hypothesis["fixed_checkpoint_count"] == 1
    assert hypothesis["subclaims"] == {
        "scalar_kernel_preserves_coordinate": "no_both_regimes",
        "scalar_kernel_contains_most_fiber_chord": "no_both_regimes",
        "scalar_posterior_quotient_separation": "no_not_shift_stable",
        "semantic_tangent_nested_in_posterior_tangent": "yes_both_regimes",
        "semantic_tangent_finite_attribution": "support_only",
        "random_rank1_specificity": "yes_both_regimes",
        "raw_final_query_autonomous_scalar_quotient": "no",
        "same_scope_retraining": "no",
    }


def test_experiment_record_preserves_decisive_metrics() -> None:
    record = build_final_query_semantic_kernel_meta_hypothesis(RESULTS)
    experiments = build_final_query_semantic_kernel_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 1
    result = experiments[0]
    assert result.primary_metric == 0.0
    assert result.model_architecture == [8, 512]
    assert result.metrics["scalar_jacobian_rank"] == 1.0
    assert result.metrics["posterior_jacobian_rank"] == 15.0
    assert result.metrics["training_support_mean_scalar_change"] > 0.01
    assert result.metrics["outside_range_mean_scalar_change"] > 0.14
    assert result.metrics["training_support_semantic_remaining_pair_ratio"] > 0.51
    assert result.metrics["outside_range_semantic_remaining_pair_ratio"] > 0.56
    assert result.metrics["training_support_semantic_kernel_posterior_js"] < 0.02
    assert result.metrics["outside_range_semantic_kernel_posterior_js"] > 0.12
    assert result.metrics["training_support_mean_relative_semantic_residual"] < 0.06
    assert result.metrics["outside_range_mean_relative_semantic_residual"] > 0.47
    assert result.metrics["training_support_random_scalar_disadvantage"] > 0.05
    assert result.metrics["outside_range_random_scalar_disadvantage"] > 0.16
    assert result.metrics["training_support_maximum_nesting_leakage"] < 1e-8
    assert result.metrics["outside_range_maximum_nesting_leakage"] < 1e-8


def test_campaign_cannot_rewrite_scalar_kernel_failure_as_success(
    tmp_path: Path,
) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["gates"]["primary_hypothesis_pass"] = True
    campaign["regimes"]["outside_range"]["gates"][
        "semantic_kernel_scalar_preservation"
    ] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="final-query semantic-kernel campaign"):
        build_final_query_semantic_kernel_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_final_query_semantic_kernel_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 1


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_final_query_semantic_kernel_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-final-query-semantic-kernel-v1",
        "experiment_count": 1,
    }
