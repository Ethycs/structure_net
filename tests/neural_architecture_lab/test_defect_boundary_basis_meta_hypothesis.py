from pathlib import Path

from neural_architecture_lab.defect_boundary_basis_meta_hypothesis import (
    build_defect_boundary_basis_experiment_results,
    build_defect_boundary_basis_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_specificity_rejection_and_rank_refinement() -> None:
    record = build_defect_boundary_basis_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["paired_boundary_rank_one_sufficiency"] == (
        "supported_three_of_three"
    )
    assert record["hypothesis"]["subclaims"]["shuffled_and_random_specificity"] == (
        "not_supported_zero_of_three"
    )
    assert record["hypothesis"]["subclaims"]["adjacent_rank_refinement"] == (
        "supported_ranks_two_five_three"
    )


def test_three_checkpoint_records_all_reject_full_hypothesis() -> None:
    record = build_defect_boundary_basis_meta_hypothesis(RESULTS)
    experiments = build_defect_boundary_basis_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert {item.metrics["base_rank"] for item in experiments} == {1.0, 2.0, 4.0}
    assert {item.metrics["minimum_geometric_correction"] for item in experiments} == {1.0}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 0

