import pytest
from neural_architecture_lab.llama_fp16_core_q4_tail_hypothesis import BASELINE_BYTES,CoreQ4Evidence,classification


def test_development_stop_precedes_final_metrics():
    assert classification({64:7,128:7})=='development_quality_failed'
    with pytest.raises(ValueError):classification({64:8,128:7})


def test_joint_quality_size_speed_thresholds():
    def c(**kw):
        args=dict(development={64:8,128:8},quality=19,reference_quality=19,payload=BASELINE_BYTES*.9,latency_ratio=1.05,tps_ratio=1.05)
        args.update(kw);return classification(**args)
    assert c()=='smaller_and_faster_proxy_quality_retained'
    assert c(quality=19,reference_quality=20)=='final_quality_failed'
    assert c(payload=BASELINE_BYTES*.9+1)=='no_useful_compression'
    assert c(tps_ratio=1.049999)=='smaller_speed_unmet'
    assert c(latency_ratio=1.049999)=='smaller_speed_unmet'
    with pytest.raises(ValueError):c(tps_ratio=float('nan'))


def test_exploratory_evidence_never_claims_confirmed():
    r=CoreQ4Evidence('campaign','a'*64,'report','b'*64,'development_quality_failed',48).record()
    assert r['tested'] and not r['confirmed'] and not r['application_accuracy_tested']
    with pytest.raises(ValueError):CoreQ4Evidence('campaign','a'*64,'report','b'*64,'development_quality_failed',47).record()
