import pytest

from neural_architecture_lab.qwen_moba_feasibility_hypothesis import QwenMoBAEvidence, build_hypothesis
from experiments.neural_architecture_lab.finalize_qwen_moba_feasibility import attention_pair_fraction


def evidence(classification, passes=0, final=0):
    return QwenMoBAEvidence(classification, 120 + final, passes, final, "campaign.json", "c" * 64, "report.md", "d" * 64)


def test_shared_gpu_support_does_not_confirm_idle_gpu_or_accuracy():
    h = build_hypothesis(evidence("shared_gpu_prefix_feasibility_supported", 10, 10))
    assert h.tested
    assert h.control_parameters["confirmed"] is False
    assert h.control_parameters["uncontended_speed_confirmed"] is False
    assert h.control_parameters["application_accuracy_tested"] is False
    assert h.control_parameters["trained_parameters"] == 0


def test_registration_rejects_incomplete_joint_gate():
    with pytest.raises(ValueError, match="all ten"):
        evidence("shared_gpu_prefix_feasibility_supported", 9, 10)
    with pytest.raises(ValueError, match="sealed"):
        evidence("no_shared_gpu_candidate_passed", 0, 10)
    with pytest.raises(ValueError, match="completed exploratory"):
        evidence("95_percent_accuracy")


def test_causal_pair_count_matches_enumerated_token_budget():
    length, block, k = 16, 4, 2
    retained = sum((token % block + 1) + block * min(token // block, k - 1) for token in range(length))
    assert attention_pair_fraction(length, block, k) == retained / sum(range(1, length + 1))
    assert attention_pair_fraction(length, block, 4) == 1
