from pathlib import Path

from neural_architecture_lab.reynolds_character_coupling_meta_hypothesis import (
    build_reynolds_character_coupling_experiment_results,
    build_reynolds_character_coupling_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_reynolds_character_coupling/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_exact_k2_localization_and_full_failure() -> None:
    record = build_reynolds_character_coupling_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["k2_exact_synthesis_localization"] == (
        "supported_five_of_five_zero_distance"
    )
    assert record["hypothesis"]["subclaims"]["k2_neutral_quadratic_sufficiency"] == (
        "not_supported_three_of_five"
    )


def test_all_ten_cells_convert_to_experiment_results() -> None:
    record = build_reynolds_character_coupling_meta_hypothesis(RESULTS)
    experiments = build_reynolds_character_coupling_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 3
