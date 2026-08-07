from pathlib import Path

from neural_architecture_lab.attention_head_defect_meta_hypothesis import (
    build_attention_head_defect_experiment_results,
    build_attention_head_defect_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_attention_head_defect/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_sparse_head_rejection() -> None:
    record = build_attention_head_defect_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["singleton_or_pair_sufficiency"] == (
        "not_supported_zero_of_five"
    )
    assert record["hypothesis"]["subclaims"]["exact_additive_contract_and_endpoint"] == (
        "supported_five_of_five"
    )


def test_five_cells_convert_with_no_full_hypothesis_passes() -> None:
    record = build_attention_head_defect_meta_hypothesis(RESULTS)
    experiments = build_attention_head_defect_experiment_results(record, RESULTS)
    assert len(experiments) == 5
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 0
