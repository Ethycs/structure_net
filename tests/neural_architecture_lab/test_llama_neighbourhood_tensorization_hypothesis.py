from dataclasses import replace
import pytest
from neural_architecture_lab.llama_neighbourhood_tensorization_hypothesis import NeighbourhoodEvidence


def evidence():
    return NeighbourhoodEvidence("campaign", "a"*64, "report", "b"*64, 14, 0, 8, .65)


def test_local_compression_cannot_rescue_failed_task_gate():
    record = evidence().record()
    assert record["confirmation_status"] == "tensorized_student_development_quality_failed"
    assert not record["confirmed"] and not record["application_accuracy_tested"]
    with pytest.raises(ValueError):
        replace(evidence(), heldout_tested=True)
    with pytest.raises(ValueError):
        replace(evidence(), heldout_quality_gate=True)


def test_promotion_requires_all_stages():
    e = replace(evidence(), development_passed=8)
    assert "unmeasured" in e.record()["confirmation_status"]
    e = replace(e, heldout_tested=True, heldout_quality_gate=True)
    assert e.record()["confirmation_status"] == "proxy_quality_retained_twofold_speed_target_failed"
    assert replace(e, speed_target_met=True).record()["confirmation_status"] == "proxy_quality_and_twofold_speed_supported"
