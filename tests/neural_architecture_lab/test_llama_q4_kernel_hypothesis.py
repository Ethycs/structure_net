import pytest
from neural_architecture_lab.llama_q4_kernel_hypothesis import PAYLOAD_BYTES,KernelEvidence,classification


def test_development_stop_and_complete_decisions():
    assert classification({'packed':7,'staged':7,'transient':7})=='development_quality_failed'
    with pytest.raises(ValueError):classification({'packed':8})


def test_no_quality_or_payload_tradeoff_can_pass_speed_gate():
    def c(**kw):
        p=dict(development={'packed':8,'staged':8,'transient':8},quality=20,old_quality=20,
               payload=PAYLOAD_BYTES,latency_ratio=1.05,tps_ratio=1.05)
        p.update(kw);return classification(**p)
    assert c()=='kernel_speed_improved'
    assert c(quality=19)=='final_quality_failed'
    assert c(quality=18,old_quality=18)=='final_quality_failed'
    assert c(tps_ratio=1.049999)=='speed_target_unmet'
    assert c(latency_ratio=1.049999)=='speed_target_unmet'
    with pytest.raises(ValueError):c(payload=PAYLOAD_BYTES+2)
    with pytest.raises(ValueError):c(tps_ratio=float('nan'))


def test_one_checkpoint_result_stays_exploratory():
    r=KernelEvidence('campaign','a'*64,'report','b'*64,'kernel_speed_improved',372).record()
    assert r['tested'] and not r['confirmed'] and not r['application_accuracy_tested']
    with pytest.raises(ValueError):KernelEvidence('campaign','a'*64,'report','b'*64,'kernel_speed_improved',71).record()
