from pathlib import Path

from neural_architecture_lab.block1_quotient_control_meta_hypothesis import (
    META_HYPOTHESIS_ID,
    build_block1_quotient_control_experiment_results,
    build_block1_quotient_control_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_block1_quotient_control/"
    "20260806_d8_code_frozen/campaign_results.json"
)


def test_meta_record_preserves_failed_causal_verdict():
    record = build_block1_quotient_control_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]

    assert record["hypothesis"]["id"] == META_HYPOTHESIS_ID
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["confirmation_status"] == (
        "not_confirmed_no_joint_improvement"
    )
    assert record["result"]["independent_seed_count"] == 5
    assert record["result"]["num_direct_experiments"] == 10
    assert metrics["joint_gate"]["joint_pass_count"] == 0
    assert metrics["controlled"]["full"]["extrapolation"][
        "cosine_pearson_mean"
    ] < 0.56
    assert metrics["controlled"]["post_mlp"]["composition"][
        "branch_accuracy_mean"
    ] > metrics["ordinary"]["post_mlp"]["composition"][
        "branch_accuracy_mean"
    ]


def test_ten_cells_convert_to_content_addressed_results():
    record = build_block1_quotient_control_meta_hypothesis(RESULTS)
    results = build_block1_quotient_control_experiment_results(record, RESULTS)

    assert len(results) == 10
    assert {result.hypothesis_id for result in results} == {META_HYPOTHESIS_ID}
    assert {result.model_parameters for result in results} == {50_964_992}
    assert all(result.model_checkpoint for result in results)
