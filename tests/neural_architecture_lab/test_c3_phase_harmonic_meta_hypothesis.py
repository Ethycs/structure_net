from pathlib import Path

from neural_architecture_lab.c3_phase_harmonic_meta_hypothesis import (
    build_c3_phase_harmonic_experiment_results,
    build_c3_phase_harmonic_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_c3_phase_harmonic/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_confirmed_stratified_scope() -> None:
    record = build_c3_phase_harmonic_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is True
    assert record["hypothesis"]["subclaims"]["first_three_theta_harmonic_sufficient"] == (
        "supported_three_of_three"
    )
    assert "frozen_sensitive_stratum" in record["hypothesis"]["confirmation_status"]


def test_five_cells_convert_with_three_primary_successes() -> None:
    record = build_c3_phase_harmonic_meta_hypothesis(RESULTS)
    experiments = build_c3_phase_harmonic_experiment_results(record, RESULTS)
    assert len(experiments) == 5
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 3
