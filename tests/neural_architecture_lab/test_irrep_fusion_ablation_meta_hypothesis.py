from pathlib import Path

from neural_architecture_lab.irrep_fusion_ablation_meta_hypothesis import (
    build_irrep_fusion_ablation_experiment_results,
    build_irrep_fusion_ablation_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_irrep_fusion_ablation/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_causal_irrep_support_and_c3_gate_failure() -> None:
    record = build_irrep_fusion_ablation_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["charged_mode_necessity"] == (
        "supported_five_of_five_both_degrees"
    )
    assert record["hypothesis"]["subclaims"]["finite_c3_phase_mechanism"] == (
        "not_supported_three_of_five"
    )


def test_all_ten_cells_convert_to_experiment_results() -> None:
    record = build_irrep_fusion_ablation_meta_hypothesis(RESULTS)
    experiments = build_irrep_fusion_ablation_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 8
