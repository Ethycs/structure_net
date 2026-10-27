from copy import deepcopy
from dataclasses import asdict

import pytest

from experiments.structure_net import ncm_task_supervision_development as dev
from experiments.neural_architecture_lab.audit_ncm_task_supervision_development import preserve


def scores(short=0,long=0,primary=False):
    result = {}
    for cohort,count in zip(dev.COHORTS,(short,long)):
        rows = [{'id':cohort+str(i),'valid':True,'passed':i<count} for i in range(64)]
        result[cohort] = {'secondary':rows}
        if primary:result[cohort]['primary'] = deepcopy(rows)
    return result


def test_zero_means_no_whole_summary_pass_not_unmeasured():
    value = dev.summarize(scores())
    assert value['valid']==128 and value['passed']==0 and value['task_accuracy']==0
    assert not value['DEV_qualified'] and value['joint_passed'] is None


def test_invalid_semantic_decision_cannot_become_an_accuracy_or_comparison_gain():
    a,b = scores(),scores(8,0)
    b[dev.COHORTS[0]]['secondary'][10].update(valid=False,passed=None)
    result = dev.compare(a,b)
    assert result['arms']['expanded']['task_accuracy'] is None
    assert result['additional_passes'] is None and result['descriptive_supervision_benefit'] is None
    assert result['paired_Terra']['unresolved']==1 and result['selected_DEV_candidate'] is None


@pytest.mark.parametrize('change',[{'valid':True,'passed':None},{'valid':True,'passed':1},
    {'valid':False,'passed':True},{'valid':False,'passed':False},{'valid':1,'passed':True}])
def test_malformed_or_contradictory_decisions_are_rejected(change):
    value=scores();value[dev.COHORTS[0]]['secondary'][0].update(change)
    with pytest.raises(ValueError):dev.summarize(value)


def test_seven_additional_passes_is_descriptive_not_goal_success():
    assert not dev.compare(scores(),scores(6,0))['descriptive_supervision_benefit']
    value = dev.compare(scores(),scores(7,0))
    assert value['additional_passes']==7 and value['descriptive_supervision_benefit']
    assert value['selected_DEV_candidate'] is None and not value['final_task_goal_achieved']


def test_one_cohort_cannot_compensate_for_other_missing_61_gate():
    result = dev.summarize(scores(64,60))
    assert result['passed']==124 and not result['primary_admitted'] and not result['DEV_qualified']


def test_marginal_judge_passes_do_not_replace_same_case_joint_passes():
    value = scores(61,61,primary=True)
    for c in dev.COHORTS:
        for i,row in enumerate(value[c]['primary']):row['passed'] = i>=3
    result = dev.summarize(value)
    assert result['primary_admitted'] and not result['DEV_qualified']
    assert all(r['joint_passed']==58 for r in result['joint'].values())


def test_selection_uses_joint_passes_and_existing_wins_tie():
    assert dev.compare(scores(61,61,True),scores(61,61,True))['selected_DEV_candidate']=='existing'
    assert dev.compare(scores(61,61,True),scores(62,61,True))['selected_DEV_candidate']=='expanded'


def test_qualification_requires_complete_primary_and_valid_transport():
    with pytest.raises(ValueError):dev.summarize(scores(61,61))
    value = scores(61,61,True)
    value[dev.COHORTS[0]]['primary'][63].update(valid=False,passed=None)
    assert not dev.summarize(value)['DEV_qualified']
    with pytest.raises(ValueError):dev.summarize(scores(60,60,True))


def test_paired_comparison_rejects_missing_duplicate_or_misaligned_sources():
    value = scores();value[dev.COHORTS[0]]['secondary'].pop()
    with pytest.raises(ValueError):dev.summarize(value)
    value = scores();value[dev.COHORTS[1]]['secondary'][0]['id']=value[dev.COHORTS[0]]['secondary'][0]['id']
    with pytest.raises(ValueError):dev.summarize(value)
    value = scores();value[dev.COHORTS[0]]['secondary'].reverse()
    with pytest.raises(ValueError):dev.compare(scores(),value)


def test_first_response_replay_requires_complete_exact_extension():
    before={'historical':{'raw':'first'}}
    preserve(before,{**before,'new':{'raw':'next'}},{'new'})
    for after,allowed in (({'historical':{'raw':'rerolled'}},{'historical'}),
                          (before,{'new'}),({**before,'extra':{}},set())):
        with pytest.raises(ValueError):preserve(before,after,allowed)


def training_record():
    value={'status':'task_supervision_training_completed','arm':'existing','stage':'train',
        'config':asdict(dev.study.CONFIG),'updates':2048,'presentations':65536,'cases':1376,
        'strict_final_reload_passed':True,'protected_tensors_unchanged':True,
        'systems_lifecycle_only_not_quality_evidence':False,'fresh_optimizer_masters_match_initial_weights':True,
        'checkpoint_sha256':'actual-final'}
    audited={'status':'task_supervision_training_replayed','arm':'existing','stage':'train',
        'cases':1376,'updates':2048,'source_checkpoint_sha256':'actual-final'}
    return value,audited


def test_new_checkpoint_schema_is_validated_without_historical_aliases():
    value,audited=training_record()
    assert 'strict_reload_passed' not in value
    dev.check_training_record(value,audited,'existing',1376)


@pytest.mark.parametrize('change',[{'updates':2},{'presentations':64},{'strict_final_reload_passed':False},
    {'protected_tensors_unchanged':False},{'systems_lifecycle_only_not_quality_evidence':True},
    {'fresh_optimizer_masters_match_initial_weights':False},{'arm':'expanded'}])
def test_pilot_or_incomplete_training_never_enters_semantic_comparison(change):
    value,audited=training_record();value.update(change)
    with pytest.raises(ValueError):dev.check_training_record(value,audited,'existing',1376)


def test_audit_of_different_checkpoint_does_not_qualify_final_model():
    value,audited=training_record();audited['source_checkpoint_sha256']='different'
    with pytest.raises(ValueError):dev.check_training_record(value,audited,'existing',1376)
