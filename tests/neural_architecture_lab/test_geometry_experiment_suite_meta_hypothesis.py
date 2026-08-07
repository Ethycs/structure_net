from pathlib import Path

from neural_architecture_lab.geometry_experiment_suite_meta_hypothesis import build_suite_records


def _records():
    return build_suite_records(
        Path("data/experiments/tinyllm_relative_carrier_factorization/20260806_d8_preregistered/campaign_results.json"),
        Path("data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered/campaign_results.json"),
        Path("data/experiments/tinyllm_defect_certification/20260806_d6_step15/results.json"),
        Path("data/experiments/tinyllm_normal_kernel_radius/20260806_d6_step15_refined/results.json"),
    )


def test_suite_preserves_four_conservative_verdicts():
    records = {item["hypothesis"]["id"]: item for item in _records()}
    assert len(records) == 4
    assert records["tinyllm-relative-carrier-fixed-quotient-v1"]["hypothesis"]["confirmed"] is False
    assert records["tinyllm-degree-k-finite-quotient-ladder-v1"]["hypothesis"]["confirmed"] is False
    assert records["tinyllm-d6-step15-defect-certificate-v1"]["hypothesis"]["confirmed"] is False
    assert records["tinyllm-normal-jet-kernel-structural-radius-v1"]["hypothesis"]["confirmed"] is True


def test_suite_retains_all_direct_cells_and_narrow_kernel_scope():
    records = {item["hypothesis"]["id"]: item for item in _records()}
    assert len(records["tinyllm-relative-carrier-fixed-quotient-v1"]["evidence"]["direct_tests"]) == 20
    assert len(records["tinyllm-degree-k-finite-quotient-ladder-v1"]["evidence"]["direct_tests"]) == 15
    kernel = records["tinyllm-normal-jet-kernel-structural-radius-v1"]
    assert kernel["result"]["descriptive_metrics"]["primary_gates"]["resolved_degree_transition"] is True
    assert "one d6 seed-7 transition" in kernel["hypothesis"]["tested_scope"]
