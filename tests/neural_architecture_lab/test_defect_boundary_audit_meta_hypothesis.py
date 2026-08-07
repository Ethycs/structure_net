from pathlib import Path

from neural_architecture_lab.defect_boundary_audit_meta_hypothesis import (
    build_defect_boundary_audit_experiment_results,
    build_defect_boundary_audit_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_mixed_mechanism_rejection() -> None:
    record = build_defect_boundary_audit_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["boundary_only"] == "supported_one_of_three"
    assert record["hypothesis"]["subclaims"]["continuous_map_distortion"] == (
        "supported_two_of_three"
    )
    assert record["hypothesis"]["subclaims"]["refined_stable_ranks"] == (
        "seed29_rank5_seed53_rank3"
    )


def test_two_checkpoint_records_preserve_refined_ranks() -> None:
    record = build_defect_boundary_audit_meta_hypothesis(RESULTS)
    experiments = build_defect_boundary_audit_experiment_results(record, RESULTS)
    assert len(experiments) == 2
    assert {item.metrics["refined_minimum_sufficient_rank"] for item in experiments} == {
        3.0, 5.0
    }
    assert sum(item.primary_metric == 1.0 for item in experiments) == 1

