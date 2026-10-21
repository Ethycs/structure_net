import pytest
from neural_architecture_lab.ncm_quality_hypothesis import classification,QualityEvidence
from experiments.neural_architecture_lab.finalize_ncm_quality import agreement,paired_ci


def test_scientific_gate_and_inclusive_target():
    assert classification(False,200,200)=='independent_quality_evidence_invalid'
    assert classification(True,189,150)=='independent_summary_quality_below_95'
    assert classification(True,190,191)=='independent_summary_quality_not_retained'
    assert classification(True,190,190)=='independent_summary_quality_point_target_met'
    with pytest.raises(ValueError):classification(True,201,190)
    with pytest.raises(ValueError):classification(True,100,100,100)


def test_paired_estimator_and_agreement_edge_cases():
    x={'a':False,'b':True};y={'a':True,'b':True}
    assert paired_ci(x,y)['teacher_minus_student']==.5
    assert paired_ci(x,y)['teacher_only_pass']==1
    assert agreement(x,y)['agreement']==.5
    assert agreement(y,y)['cohen_kappa'] is None
    with pytest.raises(ValueError):paired_ci(x,{'c':True})


def test_registry_keeps_proxy_and_one_seed_boundaries():
    e=QualityEvidence('independent_summary_quality_below_95','c','a'*64,'r','b'*64,480)
    r=e.record();assert r['tested'] and not r['confirmed'] and not r['application_accuracy_tested']
    assert r['underpowered'] and not r['speed_tested']
