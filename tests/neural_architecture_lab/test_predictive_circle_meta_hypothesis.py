from pathlib import Path

from neural_architecture_lab.predictive_circle_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_predictive_circle_experiment_results,
    build_predictive_circle_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_predictive_circle/20260805_d6_d8/results.json"
)


def test_predictive_circle_meta_hypothesis_is_conservative():
    record = build_predictive_circle_meta_hypothesis(RESULTS)

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["tested"] is True
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["confirmation_status"] == (
        "partial_synthetic_success_not_confirmed"
    )
    assert record["result"]["independent_seed_count"] == 3
    assert record["result"]["model_class_count"] == 2
    assert record["result"]["num_direct_experiments"] == 18
    assert record["result"]["descriptive_metrics"]["trained_accuracy_mean"] > 0.7
    assert record["result"]["descriptive_metrics"][
        "trained_nuisance_shift_accuracy_mean"
    ] < 0.31


def test_all_predictive_circle_arms_convert_to_nal_results():
    record = build_predictive_circle_meta_hypothesis(RESULTS)
    results = build_predictive_circle_experiment_results(record, RESULTS)

    assert len(results) == 18
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {29_956_224, 50_964_992}
    assert all(result.metrics["accuracy"] >= 0.0 for result in results)
    assert sum(result.model_checkpoint is not None for result in results) == 2
