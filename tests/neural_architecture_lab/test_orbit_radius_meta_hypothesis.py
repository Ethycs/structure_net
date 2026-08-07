from pathlib import Path

from neural_architecture_lab.orbit_radius_meta_hypothesis import (
    build_orbit_radius_experiment_results,
    build_orbit_radius_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_orbit_radius_titration/20260806_d6_preregistered/campaign_results.json"
)


def test_meta_record_preserves_primary_pass_and_cross_cohort_nonreplication() -> None:
    record = build_orbit_radius_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "primary_gate_passed_but_not_independently_cohort_replicated"
    )
    assert hypothesis["subclaims"]["k2_single_radial_crossing_primary_cohort"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["k2_shift_stable_threshold_primary_cohort"] == (
        "supported_four_of_five"
    )
    assert hypothesis["subclaims"]["k2_cross_cohort_shift_stability"] == (
        "not_supported_three_of_five"
    )
    assert hypothesis["subclaims"]["k3_shift_stable_threshold"] == (
        "not_supported_zero_of_five"
    )
    assert len(record["evidence"]["direct_tests"]) == 10
    assert len(record["evidence"]["related_independent_cohort_tests"]) == 5


def test_all_cells_convert_without_promoting_degree_three() -> None:
    record = build_orbit_radius_meta_hypothesis(RESULTS)
    experiments = build_orbit_radius_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 4
    degree_three = [item for item in experiments if item.model_architecture == [6, 3]]
    assert all(item.primary_metric == 0.0 for item in degree_three)
