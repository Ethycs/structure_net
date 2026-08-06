from pathlib import Path

from neural_architecture_lab.task_quotient_contrast_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_task_quotient_contrast_experiment_results,
    build_task_quotient_contrast_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/results.json"
)


def test_task_quotient_meta_hypothesis_preserves_partial_result():
    record = build_task_quotient_contrast_meta_hypothesis(RESULTS)

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["tested"] is True
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["topology_follows_task_quotient"] == (
        "supported"
    )
    assert record["hypothesis"]["subclaims"][
        "trained_tasks_exceed_60_percent_exact_bin_accuracy"
    ] == "not_supported"
    assert record["result"]["independent_seed_count"] == 3
    assert record["result"]["model_class_count"] == 2
    assert record["result"]["num_direct_experiments"] == 24
    metrics = record["result"]["descriptive_metrics"]
    assert metrics["trained_circle_alignment_mean"] > 0.98
    assert metrics["trained_interval_pearson_mean"] > 0.99
    assert metrics["trained_circle_normalized_h1_mean"] > 0.75
    assert metrics["trained_interval_normalized_h1_mean"] < 0.02
    assert metrics["trained_interval_accuracy_mean"] < 0.6
    assert metrics["interval_empirical_prior_random_accuracy"] > 0.0625
    assert metrics["interval_empirical_majority_class_accuracy"] > 0.12


def test_all_task_quotient_arms_convert_to_nal_results():
    record = build_task_quotient_contrast_meta_hypothesis(RESULTS)
    results = build_task_quotient_contrast_experiment_results(record, RESULTS)

    assert len(results) == 24
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {29_956_224, 50_964_992}
    assert all(result.metrics["accuracy"] >= 0.0 for result in results)
    assert sum(result.model_checkpoint is not None for result in results) == 4
