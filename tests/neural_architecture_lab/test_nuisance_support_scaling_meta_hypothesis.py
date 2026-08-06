from pathlib import Path

from neural_architecture_lab.nuisance_support_scaling_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_nuisance_support_scaling_experiment_results,
    build_nuisance_support_scaling_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_nuisance_support_scaling/"
    "20260806_d8_full/campaign_results.json"
)


def test_meta_record_preserves_failed_robustness_verdict():
    record = build_nuisance_support_scaling_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["confirmation_status"] == (
        "failure_despite_broad_coverage"
    )
    assert record["result"]["independent_seed_count"] == 5
    assert record["result"]["num_direct_experiments"] == 35
    assert metrics["ordinary_N3_full_composition"]["cosine_pearson_mean"] > 0.97
    assert metrics["ordinary_N3_full_composition"]["branch_accuracy_mean"] > 0.59
    assert metrics["ordinary_N3_full_extrapolation"]["cosine_pearson_mean"] < 0.46


def test_primary_cells_convert_to_content_addressed_results():
    record = build_nuisance_support_scaling_meta_hypothesis(RESULTS)
    results = build_nuisance_support_scaling_experiment_results(record, RESULTS)

    assert len(results) == 35
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {50_964_992}
    assert all(result.model_checkpoint for result in results)
