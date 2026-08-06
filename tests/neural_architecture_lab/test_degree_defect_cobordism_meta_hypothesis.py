from pathlib import Path

from neural_architecture_lab.degree_defect_cobordism_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_degree_defect_cobordism_experiment_results,
    build_degree_defect_cobordism_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_degree_defect_cobordism/"
    "20260805_d6_d8_seed7/results.json"
)


def test_record_preserves_numerical_and_formal_boundaries():
    record = build_degree_defect_cobordism_meta_hypothesis(RESULTS)

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["result"]["independent_seed_count"] == 1
    assert record["hypothesis"]["subclaims"][
        "degree_change_equals_indexed_defect_charge"
    ] == "numerically_supported"
    assert record["hypothesis"]["subclaims"][
        "interval_certified_root_isolation"
    ] == "not_tested"
    assert record["hypothesis"]["subclaims"][
        "hard_tokenizer_training_cobordism"
    ] == "not_defined_due_to_discontinuity"


def test_two_runs_convert_to_nal_results_with_exact_charge_identity():
    record = build_degree_defect_cobordism_meta_hypothesis(RESULTS)
    results = build_degree_defect_cobordism_experiment_results(record, RESULTS)

    assert len(results) == 2
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {29_956_224, 50_964_992}
    assert all(result.metrics["charge_identity_error"] == 0.0 for result in results)
    assert all(result.primary_metric == 1.0 for result in results)
