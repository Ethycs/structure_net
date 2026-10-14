from dataclasses import replace

import pytest

from neural_architecture_lab.llama_summary_throughput_hypothesis import SummaryThroughputEvidence


def evidence():
    return SummaryThroughputEvidence("c", "a" * 64, "r", "b" * 64, 2.0, 1.9, 1.4, 19, 20, 19, True, 400)


def test_twofold_and_ninety_five_percent_thresholds_are_inclusive():
    record = evidence().record()
    assert record["confirmation_status"] == "twofold_fp16_speedup_with_proxy_quality_supported"
    assert record["confirmed"] is False
    assert record["application_accuracy_tested"] is False


def test_quality_and_latency_cannot_be_replaced_by_token_throughput():
    assert replace(evidence(), quality_passed=18).record()["quality_gate"] is False
    assert replace(evidence(), baseline_quality_passed=20).record()["quality_gate"] is False
    assert replace(evidence(), source_review_completed=False).record()["quality_gate"] is False
    assert replace(evidence(), latency_speedup=0.9).record()["confirmation_status"] == "twofold_speed_target_not_met"
    assert replace(evidence(), speedup=1.999).record()["confirmation_status"] == "twofold_speed_target_not_met"


def test_invalid_evidence_is_rejected():
    with pytest.raises(ValueError):
        replace(evidence(), speedup=float("nan"))
    with pytest.raises(ValueError):
        replace(evidence(), quality_passed=21)
