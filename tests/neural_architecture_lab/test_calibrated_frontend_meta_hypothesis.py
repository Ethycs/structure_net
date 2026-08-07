from pathlib import Path

from neural_architecture_lab.calibrated_frontend_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_calibrated_frontend_experiment_results,
    build_calibrated_frontend_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_calibrated_frontend_causal/"
    "20260806_d8_preregistered/campaign_results.json"
)


def test_meta_record_preserves_confirmed_gauge_repair_verdict():
    record = build_calibrated_frontend_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]
    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is True
    assert record["hypothesis"]["confirmation_status"] == (
        "confirmed_gauge_repair_constructible"
    )
    assert record["result"]["independent_seed_count"] == 5
    assert record["result"]["num_direct_experiments"] == 15
    assert metrics["raw"]["joint_pass_count"] == 0
    assert metrics["analytic"]["joint_pass_count"] == 5
    assert metrics["learned"]["joint_pass_count"] == 5
    assert metrics["learned"]["cuts"]["full"]["extrapolation"]["pass_count"] == 5
    assert metrics["learned"]["cuts"]["full"]["extrapolation"][
        "cosine_pearson_mean"
    ] >= 0.90
    assert metrics["learned"]["cuts"]["full"]["extrapolation"][
        "branch_balanced_accuracy_mean"
    ] <= 0.55


def test_fifteen_cells_convert_to_content_addressed_results():
    record = build_calibrated_frontend_meta_hypothesis(RESULTS)
    results = build_calibrated_frontend_experiment_results(record, RESULTS)
    assert len(results) == 15
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {50_965_504}
    assert all(result.model_checkpoint for result in results)
    assert all("identifiability contract passed" in result.observations[1].lower() for result in results)
