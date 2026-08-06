from pathlib import Path

from neural_architecture_lab.internal_quotient_probe_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_internal_quotient_probe_experiment_results,
    build_internal_quotient_probe_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_internal_quotient_probe/20260805_d6_d8_seed7/results.json"
)


def test_internal_probe_record_preserves_strong_but_single_seed_result():
    record = build_internal_quotient_probe_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["result"]["independent_seed_count"] == 1
    assert record["result"]["model_class_count"] == 2
    assert metrics["phase_id_branch_accuracy_mean"] > 0.99
    assert 0.48 < metrics["cosine_id_branch_accuracy_mean"] < 0.54
    assert metrics["phase_independent_ph_h1_mean"] > 0.65
    assert metrics["cosine_independent_ph_h1_mean"] < 0.12


def test_internal_probe_arms_convert_to_nal_results():
    record = build_internal_quotient_probe_meta_hypothesis(RESULTS)
    results = build_internal_quotient_probe_experiment_results(record, RESULTS)

    assert len(results) == 4
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {29_956_224, 50_964_992}
    assert all(result.model_checkpoint for result in results)
