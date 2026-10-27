from copy import deepcopy

import pytest

from experiments.structure_net import ncm_teacher1024 as study
from experiments.neural_architecture_lab import ncm_teacher1024_service as service


@pytest.fixture
def population():
    rows = [{'id':f'{band}-{i}','source_component':f'component-{band}-{i}',
             'bin':band,'source_cl100k_tokens':192+band*200+i} for band in range(5) for i in range(6)]
    observed = [rows[0],rows[7]]
    unused = [r['id'] for r in rows if r not in observed]
    return rows, unused, observed, {0:4,1:2,2:2,3:2,4:2}


def test_deterministic_selection_keeps_all_prior_firsts_out(population):
    rows, unused, observed, quotas = population
    selected = study.select(rows, unused, observed, quotas)
    assert selected == study.select(list(reversed(rows)), list(reversed(unused)), observed, quotas)
    assert len(selected) == 12
    assert not {r['id'] for r in selected} & {r['id'] for r in observed}
    assert {band:sum(r['bin']==band for r in selected) for band in quotas} == quotas


@pytest.mark.parametrize('kind',['duplicate_row','duplicate_unused','duplicate_observed','missing','overlap','component','quota'])
def test_partition_and_component_errors_stop_before_submission(population,kind):
    rows, unused, observed, quotas = deepcopy(population)
    if kind == 'duplicate_row':rows.append(rows[0])
    elif kind == 'duplicate_unused':unused.append(unused[0])
    elif kind == 'duplicate_observed':observed.append(observed[0])
    elif kind == 'missing':unused.pop()
    elif kind == 'overlap':unused.append(observed[0]['id'])
    elif kind == 'component':
        for r in rows:
            if r['id'] in unused:r['source_component']='same-component'
    elif kind == 'quota':quotas[4]=20
    with pytest.raises(ValueError):study.select(rows,unused,observed,quotas)


def test_pilot_covers_all_bins_first_and_longest(population):
    rows, unused, observed, quotas = population
    selected = study.select(rows,unused,observed,quotas);ids=study.pilot_ids(selected)
    assert len(ids)==len(set(ids))==8
    assert selected[0]['id'] in ids
    assert max(selected,key=lambda r:r['source_cl100k_tokens'])['id'] in ids
    assert {r['bin'] for r in selected if r['id'] in ids} == set(range(5))


def test_audit_rejects_edited_or_truncated_first_target(monkeypatch):
    source={'id':'source','source':'complete source'}
    class Tokenizer:
        def encode(self,text,**kwargs):return [42] if text else []
    monkeypatch.setattr(study,'replay',lambda source,stage,value,encoding,draft=None:{'summary':'first summary'})
    value={**source,'draft':{},'review':{},'target_text':'first summary',
           'teacher_valid':True,'capacity_fit':True,'stored_tokens':1,'native_lexical_tokens':1,
           'native_word_ids':[101,42,101]+[0]*125,'semantic_qualified':False}
    study.replay_candidate(source,value,Tokenizer(),Tokenizer())
    value['target_text']='changed summary'
    with pytest.raises(ValueError,match='first teacher'):study.replay_candidate(source,value,Tokenizer(),Tokenizer())


@pytest.mark.parametrize('stage,count,status',[('pilot',8,'teacher_pilot_completed'),('generate',1024,'all1024_first_candidates_completed')])
def test_service_requires_correct_complete_stage(stage,count,status):
    value={'status':status,'cases':count,'student_training_admitted':False,
           'canonical_semantic_cache_changed':False,'final_test_opened':False}
    assert service.valid_result(stage,value)
    assert not service.valid_result(stage,{**value,'cases':count-1})
    assert not service.valid_result(stage,{**value,'student_training_admitted':True})
    assert service.command(stage)[0]=='/home/rabbit/.pixi/bin/pixi'
    assert service.command(stage)[-2:]==['experiments.structure_net.ncm_teacher1024',stage]
