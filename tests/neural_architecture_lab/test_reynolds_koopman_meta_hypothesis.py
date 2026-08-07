from pathlib import Path

from neural_architecture_lab.reynolds_koopman_meta_hypothesis import (
    build_reynolds_koopman_experiment_results,
    build_reynolds_koopman_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered/campaign_results.json"
)


def test_meta_record_preserves_primary_failure_and_supported_cover_transition():
    record = build_reynolds_koopman_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["confirmation_status"] == (
        "not_confirmed_predictive_barycenter_front_precedes_causal_quotient_front"
    )
    assert record["hypothesis"]["subclaims"]["cover_gain_transition_k2"] == "supported_five_of_five"
    assert record["result"]["descriptive_metrics"]["degrees"]["3"]["gate_pass_counts"][
        "front_agreement_both_shifts"
    ] == 1


def test_all_ten_frozen_cells_convert_to_experiment_results():
    record = build_reynolds_koopman_meta_hypothesis(RESULTS)
    experiments = build_reynolds_koopman_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert all(item.model_checkpoint for item in experiments)
