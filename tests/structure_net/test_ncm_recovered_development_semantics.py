from copy import deepcopy

import pytest

from experiments.structure_net import ncm_recovered_development_semantics as dev
from tests.structure_net.test_ncm_task_supervision_development import scores


def outputs():
    row = dict(id='source',source_sha256='source-hash',split='short_development',feature_sha256='feature',
        source_positions=193,codes=[4]*641,raw_word_ids=[1]*128,text='parent')
    new = {**deepcopy(row),'raw_word_ids':[2]*128,'text':'new'}
    return {'parent_with_source':[row],'new_codec_only':[new]}


def test_export_cannot_change_source_code_pairing_or_population():
    original = outputs()
    result = dev.selected_rows(original,'new_codec_only',['source'])
    assert result[0]['text']=='new' and result[0]['codes']==original['parent_with_source'][0]['codes']
    result[0]['codes'][0] = 3
    assert original['new_codec_only'][0]['codes'][0]==4
    for key,value in (('codes',[5]*641),('source_sha256','other'),('source_positions',192),('id','other')):
        bad = outputs();bad['new_codec_only'][0][key] = value
        with pytest.raises(ValueError):dev.selected_rows(bad,'new_codec_only',['source'])
    with pytest.raises(ValueError):dev.selected_rows(original,'new_codec_only',['other'])
    with pytest.raises(ValueError):dev.selected_rows({'new_codec_only':original['new_codec_only']},'new_codec_only',['source'])


def test_fixed_readout_preference_is_not_score_maximization_or_final_success():
    value = dev.compare(scores(61,61,True),scores(64,64,True))
    assert value['selected_DEV_candidate']=='new_codec_only'
    assert value['arms']['new_codec_only']['joint_passed']==122
    assert not value['final_task_goal_achieved'] and not value['final_test_opened']
    value = dev.compare(scores(60,60),scores(61,61,True))
    assert value['selected_DEV_candidate']=='parent_with_source'


def test_unknown_decisions_have_no_accuracy_or_improvement_claim():
    new = scores(8,0);new[dev.COHORTS[0]]['secondary'][10].update(valid=False,passed=None)
    value = dev.compare(new,scores())
    assert value['arms']['new_codec_only']['task_accuracy'] is None
    assert value['additional_passes_new_decoder'] is None and value['paired_Terra']['unresolved']==1
    assert value['selected_DEV_candidate'] is None
    wrong = scores();wrong[dev.COHORTS[0]]['secondary'].reverse()
    with pytest.raises(ValueError):dev.compare(scores(),wrong)


def test_each_cohort_requires_same_case_dual_judge_passes():
    value = dev.compare(scores(64,60),scores())
    assert value['arms']['new_codec_only']['passed']==124
    assert value['selected_DEV_candidate'] is None
    new = scores(61,61,True)
    for cohort in dev.COHORTS:
        for i,row in enumerate(new[cohort]['primary']):row['passed']=i>=3
    value = dev.compare(new,scores())
    assert all(v['joint_passed']==58 for v in value['arms']['new_codec_only']['joint'].values())
    assert value['selected_DEV_candidate'] is None
