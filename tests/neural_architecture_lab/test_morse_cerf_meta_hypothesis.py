from pathlib import Path

from neural_architecture_lab.morse_cerf_meta_hypothesis import (
    build_morse_cerf_experiment_results,
    build_morse_cerf_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_morse_cerf/20260806_d6_preregistered/campaign_results.json"
)


def test_meta_record_preserves_primary_failure_and_local_event_subclaim() -> None:
    record = build_morse_cerf_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["confirmation_status"] == (
        "not_confirmed_near_front_events_without_stable_morse_normal_form"
    )
    assert record["hypothesis"]["subclaims"]["near_front_events_k2"] == (
        "supported_both_shifts_five_of_five"
    )
    assert record["result"]["descriptive_metrics"]["degrees"]["3"]["gate_counts"][
        "extrapolation"
    ]["mature_barycenter_basin"] == 3


def test_all_ten_frozen_cells_convert_to_experiment_results() -> None:
    record = build_morse_cerf_meta_hypothesis(RESULTS)
    experiments = build_morse_cerf_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert all(item.model_checkpoint for item in experiments)
    assert all(item.primary_metric == 0.0 for item in experiments)
