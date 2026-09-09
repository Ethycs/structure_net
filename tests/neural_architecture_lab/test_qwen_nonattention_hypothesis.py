import pytest
from neural_architecture_lab.qwen_nonattention_hypothesis import NonattentionEvidence


def test_proxy_support_does_not_confirm_application_accuracy():
    value = NonattentionEvidence("campaign", "a" * 64, "report", "b" * 64,
                                 "execution_regression_supported", 16, 10, 10, "fused_fp16").record()
    assert value["tested"] and not value["confirmed"]
    assert value["trained_parameters"] == 0 and not value["application_accuracy_tested"]
    with pytest.raises(ValueError, match="all ten"):
        NonattentionEvidence("campaign", "a" * 64, "report", "b" * 64,
                             "execution_regression_supported", 16, 10, 9, "fused_fp16")


def test_failed_screen_blocks_conditional_regression():
    with pytest.raises(ValueError, match="failed screen"):
        NonattentionEvidence("campaign", "a" * 64, "report", "b" * 64,
                             "no_execution_candidate_passed", 16, 10, 0, "fused_fp16")
