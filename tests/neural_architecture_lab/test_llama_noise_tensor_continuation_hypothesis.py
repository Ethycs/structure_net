import pytest
from neural_architecture_lab.llama_noise_tensor_continuation_hypothesis import ContinuationEvidence, heldout_gate
from experiments.structure_net.campaign_llama_noise_tensor_heldout import select_arm


def test_secondary_noise_gain_cannot_rescue_quality():
    e = ContinuationEvidence("c", "a"*64, "r", "b"*64, 2, 3, 96)
    record = e.record()
    assert record["confirmation_status"] == "continued_tensor_student_development_quality_failed"
    assert record["noise_benefit_observed"] and not record["confirmed"]


def test_final_stage_order_and_tie():
    e = ContinuationEvidence("c", "a"*64, "r", "b"*64, 8, 8, 96)
    assert not e.record()["noise_benefit_observed"]
    assert e.record()["confirmation_status"] == "development_recovered_final_quality_unmeasured"
    with pytest.raises(ValueError):
        ContinuationEvidence("c", "a"*64, "r", "b"*64, 7, 7, 96, heldout_tested=True)


def test_final_candidate_selection_requires_pass_then_kl_then_clean():
    assert select_arm({"clean": 7, "noise": 8}, {"clean": .01, "noise": .1}) == "noise"
    assert select_arm({"clean": 8, "noise": 8}, {"clean": .03, "noise": .02}) == "noise"
    assert select_arm({"clean": 8, "noise": 8}, {"clean": .02, "noise": .02}) == "clean"
    with pytest.raises(ValueError):
        select_arm({"clean": 7, "noise": 7}, {"clean": .01, "noise": .01})


def test_final_quality_95_percent_inclusive_and_no_baseline_loss():
    assert heldout_gate(19, 19, 18)
    assert not heldout_gate(19, 20, 19)
    assert not heldout_gate(18, 18, 18)
    assert heldout_gate(20, 20, 20)
