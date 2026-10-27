from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

from experiments.structure_net import ncm_oracle_prefix as p


def sample():
    spec = importlib.util.spec_from_file_location('oracle_prefix_test_model',
        Path(__file__).with_name('test_ncm_sequential_levels.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,f,m,gold = module.sample();model.eval().requires_grad_(False)
    return model,f,m,gold


def test_empty_prefix_matches_existing_source_only_generation():
    model,f,m,_ = sample()
    empty = torch.empty(1,0,dtype=torch.long)
    assert torch.equal(p.generate_with_prefix(model,f,m,empty),p.sequence.generate(model,f,m))


def test_selected_prefix_only_and_target_immutability():
    model,f,m,gold = sample();saved = gold.clone()
    before = p.generate_with_prefix(model,f,m,gold[:,:1])
    changed = gold.clone();changed[:,1:] = 0
    assert torch.equal(before,p.generate_with_prefix(model,f,m,changed[:,:1]))
    assert torch.equal(gold,saved) and torch.equal(before[:,:1],gold[:,:1])
    with pytest.raises(TypeError):p.generate_with_prefix(model,f,m,gold=gold)


def test_all_code_control_does_not_call_generator(monkeypatch):
    model,f,m,gold = sample()
    def forbidden(*args,**kwargs):raise AssertionError('all codes supplied')
    monkeypatch.setattr(p.sequence,'level_logits',forbidden)
    assert torch.equal(p.generate_with_prefix(model,f,m,gold),gold)


@pytest.mark.parametrize('prefix',[torch.zeros(1,2,dtype=torch.long),torch.zeros(2,1,dtype=torch.long),
                                 torch.zeros(1,1),torch.tensor([[-1]]),torch.tensor([[8192]])])
def test_invalid_prefix_rejected(prefix):
    model,f,m,_ = sample()
    with pytest.raises(ValueError):p.generate_with_prefix(model,f,m,prefix)


def test_training_mode_is_rejected():
    model,f,m,gold=sample();model.train()
    with pytest.raises(ValueError):p.generate_with_prefix(model,f,m,gold[:,:1])


def case(i):
    return {'id':str(i),'split':'reserved_component','source_sha256':'source'+str(i),
        'target_sha256':'target'+str(i),'oracle_codes':[1]*641,
        'native_word_ids':[101,5,101]+[0]*125,'native_reference':'gold'}


def output(row,arm):
    count = p.ARMS[arm]
    out = {'id':row['id'],'arm':arm,'source_id_used':row['id'],'source_sha256':row['source_sha256'],
        'target_sha256':row['target_sha256'],'injected_codes':count,'codes':[1]*count+[2]*(641-count),
        'raw_word_ids':[101,6,101]+[0]*125,'text':'wrong','task_accuracy':None}
    out['metrics']=p.case_metrics(out,row,arm);return out


def test_injected_matches_are_excluded_from_suffix_metric():
    row=case(0);out=output(row,'oracle25');score=p.aggregate([out],[row],'oracle25')
    assert score['remaining_positions']==616 and score['correct_remaining_codes']==0
    assert score['remaining_code_agreement']==0
    assert sum(v['remaining_positions'] for v in score['levels'])==616
    full=p.aggregate([output(row,'oracle641')],[row],'oracle641')
    assert full['remaining_positions']==0 and full['remaining_code_agreement'] is None


def test_complete_replay_cannot_qualify_all_oracle_control():
    rows=[case(0),case(1)]
    outputs={arm:[output(r,arm) for r in rows] for arm in p.ARMS}
    v=p.replay(outputs,rows)
    assert v['classification']=='early_prefix_rescue_not_established'
    assert v['passing_interventions']==[]
    assert set(v['effects'])=={'oracle1','oracle9','oracle25'}
    assert all(x['remaining_code_gain']==0 for x in v['effects'].values())
    with pytest.raises(ValueError):p.replay({k:v for k,v in outputs.items() if k!='oracle641'},rows)


@pytest.mark.parametrize('key,value',[('source_id_used','other'),('target_sha256','other'),('task_accuracy',.95),('injected_codes',9)])
def test_bad_binding_rejected(key,value):
    row=case(0);out=output(row,'oracle1');out[key]=value
    with pytest.raises(ValueError):p.case_metrics(out,row,'oracle1')


def test_corrupt_prefix_counts_text_and_control_are_rejected():
    row=case(0);out=output(row,'oracle1');out['codes'][0]=2
    with pytest.raises(ValueError):p.case_metrics(out,row,'oracle1')
    out=output(row,'oracle1');out['metrics']['correct_remaining_codes']=1
    with pytest.raises(ValueError):p.aggregate([out],[row],'oracle1')
    out=output(row,'source_only')
    with pytest.raises(ValueError):p.validate_text({'source_only':[out]},lambda ids:'different')
    outputs={a:[output(row,a)] for a in p.ARMS}
    baseline={row['id']:deepcopy(outputs['source_only'][0])};oracle={row['id']:deepcopy(outputs['oracle641'][0])}
    p.compare_controls(outputs,[row],baseline,oracle)
    oracle[row['id']]['raw_word_ids'][1]=7
    with pytest.raises(ValueError):p.compare_controls(outputs,[row],baseline,oracle)


def test_joint_gate_inclusive_and_both_endpoints_required():
    value={'remaining_positions':100,'correct_remaining_codes':10,'content_tokens':100,'content_matches':20}
    assert p.effect(value,0,0)['joint_gate_passed']
    assert not p.effect({**value,'correct_remaining_codes':9},0,0)['joint_gate_passed']
    assert not p.effect({**value,'content_matches':19},0,0)['joint_gate_passed']
    with pytest.raises(ValueError):p.effect({**value,'remaining_positions':0},0,0)


def test_every_real_fingerprint_input_exists():
    assert all(path.is_file() for path in p.input_paths())
