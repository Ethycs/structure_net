from pathlib import Path

from neural_architecture_lab.depth_graded_quotient_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_depth_graded_quotient_experiment_results,
    build_depth_graded_quotient_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_depth_graded_quotient/"
    "20260805_d8_seed7/results.json"
)


def test_record_preserves_depth_family_and_claim_boundaries():
    record = build_depth_graded_quotient_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["result"]["independent_seed_count"] == 1
    assert record["hypothesis"]["subclaims"][
        "continuous_gate_uniquely_outperforms_discrete_multi_exit"
    ] == "not_supported"
    assert record["hypothesis"]["subclaims"]["reeb_cosheaf"] == "not_constructed"
    assert metrics["standard_final:phase_circle"]["refined_front_depth"] == 0.045
    assert metrics["continuous_gate:phase_circle"]["refined_front_depth"] == 0.02
    assert metrics["standard_final:cosine_interval"]["refined_front_depth"] == 1.85
    assert metrics["continuous_gate:cosine_interval"]["refined_front_depth"] == 0.005


def test_six_arms_convert_to_content_addressed_experiment_results():
    record = build_depth_graded_quotient_meta_hypothesis(RESULTS)
    results = build_depth_graded_quotient_experiment_results(record, RESULTS)

    assert len(results) == 6
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {50_964_992}
    assert all(result.model_checkpoint for result in results)
    assert all(result.primary_metric > 0.3 for result in results)
