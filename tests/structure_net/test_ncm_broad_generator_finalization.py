import pytest

from experiments.neural_architecture_lab.finalize_ncm_broad_generator import classify


def joint(count=61, valid=True):
    return {'cases':64,'joint_passed':count,'all_valid':valid,'qualified':valid and count >= 61}


def test_invalid_transport_cannot_be_counted_as_model_inaccuracy():
    assert classify(127,127,False,None) == 'DEV_transport_invalid'
    assert classify(0,0,False,None) == 'DEV_transport_invalid'
    assert classify(128,128,True,{'short':joint(), 'long':joint(valid=False)}) == 'DEV_transport_invalid'


def test_complete_secondary_result_separates_failure_from_partial_progress():
    assert classify(128,0,False,None) == 'DEV_semantic_quality_failed'
    assert classify(128,1,False,None) == 'DEV_semantic_partial_progress'
    assert classify(128,121,False,None) == 'DEV_semantic_partial_progress'


def test_qualification_needs_both_same_case_joint_cohorts():
    assert classify(128,122,True,{'short':joint(), 'long':joint()}) == 'DEV_qualified'
    assert classify(128,128,True,{'short':joint(64), 'long':joint(60)}) == 'DEV_semantic_partial_progress'


@pytest.mark.parametrize('args',[
    (127,128,False,None), (129,128,False,None), (128.,0,False,None),
    (128,121,True,{'short':joint(),'long':joint()}),
    (128,122,True,{'short':joint()}), (128,122,False,{'short':joint(),'long':joint()}),
    (128,122,True,{'short':{**joint(60),'qualified':True},'long':joint()}),
])
def test_inconsistent_denominators_or_admission_fail_closed(args):
    with pytest.raises(ValueError): classify(*args)
