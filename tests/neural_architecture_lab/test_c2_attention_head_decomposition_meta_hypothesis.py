from pathlib import Path

from neural_architecture_lab.c2_attention_head_decomposition_meta_hypothesis import (
    build_c2_attention_head_decomposition_experiment_results,
    build_c2_attention_head_decomposition_meta_hypothesis,
)


RESULTS = Path("data/experiments/tinyllm_c2_attention_head_decomposition/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_preserves_distributed_head_rejection() -> None:
    record = build_c2_attention_head_decomposition_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["subclaims"]["sparse_source_selection"] == (
        "not_supported_zero_of_three_primary"
    )
    assert record["hypothesis"]["subclaims"]["heldout_endpoint_replication"] == (
        "supported_three_of_three_primary"
    )


def test_five_cells_convert_with_no_full_hypothesis_passes() -> None:
    record = build_c2_attention_head_decomposition_meta_hypothesis(RESULTS)
    experiments = build_c2_attention_head_decomposition_experiment_results(record, RESULTS)
    assert len(experiments) == 5
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert sum(item.primary_metric == 1.0 for item in experiments) == 0
