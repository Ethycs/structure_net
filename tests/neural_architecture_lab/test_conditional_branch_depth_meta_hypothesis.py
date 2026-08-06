from pathlib import Path

from neural_architecture_lab.conditional_branch_depth_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_conditional_branch_depth_experiment_results,
    build_conditional_branch_depth_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_conditional_branch_depth_scan/"
    "20260805_d8_five_seed/results.json"
)


def test_record_preserves_partial_multi_seed_verdict():
    record = build_conditional_branch_depth_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]["cosine_arms"]

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["result"]["independent_seed_count"] == 5
    assert record["hypothesis"]["subclaims"][
        "in_distribution_residual_quotient"
    ] == "supported_five_seeds_all_arms"
    assert record["hypothesis"]["subclaims"][
        "nuisance_robust_internal_quotient"
    ] == "not_supported_cosine_not_preserved"
    assert metrics["standard_final"]["median_residual_quotient_front_id"] == 0.02
    assert metrics["continuous_gate"]["median_residual_quotient_front_id"] == 0.005
    assert metrics["standard_final"][
        "block1_post_attention_branch_accuracy_mean"
    ] > 0.60
    assert metrics["continuous_gate"][
        "block1_post_attention_branch_accuracy_mean"
    ] < 0.53


def test_eighteen_runs_convert_to_content_addressed_results():
    record = build_conditional_branch_depth_meta_hypothesis(RESULTS)
    results = build_conditional_branch_depth_experiment_results(record, RESULTS)

    assert len(results) == 18
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {50_964_992}
    assert all(result.model_checkpoint for result in results)
    assert all(0.8 <= result.primary_metric <= 1.0 for result in results)
