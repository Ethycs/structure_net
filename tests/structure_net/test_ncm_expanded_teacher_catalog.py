from copy import deepcopy

import pytest

from experiments.structure_net import ncm_expanded_teacher_catalog as study


@pytest.fixture
def inputs(monkeypatch):
    source={'id':'new','source_id':'new_family','source_component':'new_component','group_id':'new_group',
        'source':'A complete original source.','source_cl100k_tokens':256,'split':'additional_train',
        'source_truncated':False,'spans':[{'source_id':'new_family','turn_text_sha256':'turn'}]}
    source['source_sha256']=study.old.text_hash(source['source'])
    candidate={**source,'target_text':'Original reviewed summary.','teacher_valid':True,'capacity_fit':True,
               'stored_tokens':3,'native_lexical_tokens':3,'native_word_ids':[101,7,8,9,101]+[0]*123}
    prior=[{'id':'old','source_sha256':'old_hash','source_component':'old_component','target_text':'Unchanged old target'}]
    protected={key:set() for key in ('families','groups','hashes','turns')}
    class Tokenizer:
        all_special_ids=[0,100,101,102,103]
        def encode(self,text,**kwargs):return [7,8,9] if text else []
    calls=[]
    monkeypatch.setattr(study.collection.original,'replay_candidate',lambda *args:calls.append(args[0]['id']))
    return prior,source,candidate,protected,Tokenizer(),calls


def materialize(inputs):
    prior,source,candidate,protected,tokenizer,_=inputs
    return study.materialize(prior,[source],[candidate],protected,[],tokenizer,tokenizer,'frozen/labels.json')


def test_preserves_old_rows_and_new_first_text_without_mutating_inputs(inputs):
    prior,source,candidate,_,_,calls=inputs
    old=deepcopy(prior);first=deepcopy(candidate);rows,excluded,observed=materialize(inputs)
    assert rows[:1]==old==prior and candidate==first
    assert rows[1]['target_text']==candidate['target_text']
    assert rows[1]['split']=='TRAIN' and source['split']=='additional_train'
    assert rows[1]['target_sha256']==study.old.text_hash(candidate['target_text'])
    assert not rows[1]['semantic_qualified'] and not excluded
    assert observed[0]['admitted'] and calls==['new']


def test_invalid_first_is_retained_as_an_exclusion_not_replaced(inputs):
    candidate=inputs[2];candidate.update(target_text='',teacher_valid=False,capacity_fit=False)
    rows,excluded,observed=materialize(inputs)
    assert rows==inputs[0] and len(excluded)==len(observed)==1
    assert 'invalid_first_preserved' in excluded[0]['reasons']
    assert observed[0]['teacher_valid'] is False and observed[0]['admitted'] is False


def test_native_unknown_is_excluded_even_when_teacher_capacity_says_fit(inputs):
    inputs[4].encode=lambda text,**kwargs:[100]
    rows,excluded,_=materialize(inputs)
    assert rows==inputs[0] and 'native_special_or_unknown' in excluded[0]['reasons']


def test_rejects_source_edit_instead_of_accepting_summary(inputs):
    inputs[2]['source']='Edited source'
    with pytest.raises(ValueError,match='unaltered'):materialize(inputs)


def test_protected_informative_turn_is_excluded(inputs):
    inputs[3]['turns'].add('turn')
    rows,excluded,_=materialize(inputs)
    assert rows==inputs[0] and 'protected_informative_turn' in excluded[0]['reasons']


def test_duplicate_component_is_excluded_even_with_another_source_hash(inputs):
    inputs[0][0]['source_component']='new_component'
    rows,excluded,_=materialize(inputs)
    assert rows==inputs[0] and 'source_component_duplicate' in excluded[0]['reasons']


def test_feature_union_preserves_old_entries_and_only_admitted_new():
    old=[{'id':'a'}];new=[{'id':'b'},{'id':'c'}]
    index,manifest=study.merge_bindings(old,{'a':'old'},new,{'b':'B','c':'C'},{'a','c'})
    assert index==[{'id':'a'},{'id':'c'}] and manifest=={'a':'old','c':'C'}
    for bad in ({'c'},{'a','d'}):
        with pytest.raises(ValueError):study.merge_bindings(old,{'a':'old'},new,{'b':'B','c':'C'},bad)
    with pytest.raises(ValueError):study.merge_bindings(old,{'a':'old'},old,{'a':'new'},{'a'})


def test_training_gate_requires_full_population_and_never_admits_pilot():
    assert study.training_admission(1024,900,pilot=False)
    assert not study.training_admission(1024,899,pilot=False)
    assert not study.training_admission(1000,1000,pilot=False)
    assert not study.training_admission(8,8,pilot=True)
    assert not study.training_admission(1024,1024,pilot=True)
    with pytest.raises(ValueError):study.training_admission(8,9,pilot=False)
