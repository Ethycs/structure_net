from copy import deepcopy

import pytest

from experiments.neural_architecture_lab import finalize_ncm_onpolicy_residual as final


def arm(passes, joint=None, *, valid=128):
    admitted = joint is not None
    joints = {str(i):{'cases':64,'joint_passed':n,'all_valid':True,'qualified':n >= 61}
              for i,n in enumerate(joint)} if admitted else None
    classification = final.classify(valid,passes,admitted,joints)
    return {'valid':valid,'passed':passes,'primary_admitted':admitted,'joint':joints,
            'classification':classification,'DEV_qualified':classification == 'DEV_qualified',
            'joint_task_accuracy':sum(joint)/128 if admitted else None}


def decisions(values):
    return [{'id':str(i),'valid':v is not None,'passed':v} for i,v in enumerate(values)]


def test_pairing_preserves_invalid_cases_and_withholds_accuracy_difference():
    value = final.paired_passes(decisions([True,True,False,False,None]),
                               decisions([True,False,True,False,True]))
    assert value == {'cases':5,'both_pass':1,'fixed_only':1,'residual_only':1,'both_fail':1,
                     'unresolved':1,'valid_pairs':4,'accuracy_difference_residual_minus_fixed':None}
    valid = final.paired_passes(decisions([False,True,False]),decisions([True,True,False]))
    assert valid['accuracy_difference_residual_minus_fixed'] == 1/3


@pytest.mark.parametrize('change',['reorder','duplicate','missing','invalid_as_failure','non_boolean'])
def test_pairing_rejects_denominator_identity_or_invalidity_changes(change):
    left,right = decisions([True,False,None]),decisions([False,True,None])
    if change == 'reorder': right.reverse()
    elif change == 'duplicate': left[1]['id'] = right[1]['id'] = '0'
    elif change == 'missing': right.pop()
    elif change == 'invalid_as_failure': right[-1]['passed'] = False
    else: right[0]['passed'] = 1
    with pytest.raises(ValueError): final.paired_passes(left,right)


@pytest.mark.parametrize('fixed,residual,expected',[
    (arm(0),arm(1),None),
    (arm(122,[62,60]),arm(0),None),
    (arm(124,[62,62]),arm(124,[63,61]),'residual_targets'),
    (arm(126,[63,63]),arm(124,[62,62]),'fixed_targets'),
    (arm(127,[64,63]),arm(128,[64,64]),'residual_targets'),
    (arm(122,[61,61]),arm(0,valid=127),'fixed_targets'),
])
def test_selection_uses_both_cohort_joint_thresholds_and_prospective_tiebreak(fixed,residual,expected):
    assert final.select_candidate({'fixed_targets':fixed,'residual_targets':residual}) == expected


@pytest.mark.parametrize('change',['qualification','joint_accuracy','missing_arm'])
def test_selection_rejects_unearned_qualification_or_misstated_joint_accuracy(change):
    values = {'fixed_targets':arm(122,[61,61]),'residual_targets':arm(0)}
    values = deepcopy(values)
    if change == 'qualification': values['residual_targets']['DEV_qualified'] = True
    elif change == 'joint_accuracy': values['fixed_targets']['joint_task_accuracy'] = 1.
    else: del values['residual_targets']
    with pytest.raises(ValueError): final.select_candidate(values)
