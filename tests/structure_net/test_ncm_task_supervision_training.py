from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from experiments.structure_net import ncm_task_supervision_training as study
from experiments.neural_architecture_lab import audit_ncm_task_supervision as audit
from experiments.neural_architecture_lab import ncm_task_supervision_service as service


def population():
    return [{'id':str(i),'oracle_codes':[(j+i)%8192 for j in range(641)],
        'native_lexical_tokens':10+i} for i in range(7)]


def synthetic_trace():
    config = replace(study.CONFIG,updates=4,effective_examples=8,microbatch=4,warmup=1)
    rows = population();sampler = study.schedule.Schedule(len(rows),config);records = []
    for step in range(1,5):
        refs = sampler.update(step-1);micro = []
        for offset in (0,4):
            selected = refs[offset:offset+4]
            gold = torch.tensor([rows[e.index]['oracle_codes'] for e in selected])
            _,changes = study.engine.histories(gold,selected,audit.LEVELS)
            micro.append({'ids':[rows[e.index]['id'] for e in selected],'positions':[e.position for e in selected],
                'populations':[e.population for e in selected],'noisy':[e.noisy for e in selected],
                'history_changes':changes,'weighted_CE':.25})
        records.append({'step':step,'epoch':1,'examples':8,'lr':study.schedule.shared.rate(step,config),
            'weighted_CE':.5,'gradient_norm':1.,'scale':128.,'microbatches':micro})
    return config,rows,records


def test_controls_include_true_longest_source_and_longest_target():
    rows = population();index = [{'id':r['id'],'source_positions':100+i} for i,r in enumerate(rows)]
    index[2]['source_positions'] = 2048
    selected = study.control_ids(rows,index)
    assert len(selected) == len(set(selected)) == 4
    assert {'0','2','6'} <= set(selected)
    with pytest.raises(ValueError):study.control_ids(rows,index[:-1])


def test_fresh_cpu_optimizer_uses_loaded_parent_weights(monkeypatch):
    model = torch.nn.Linear(3,2)
    with torch.no_grad():
        model.weight.fill_(7);model.bias.fill_(-3)
    monkeypatch.setattr(torch.amp,'GradScaler',lambda *a,**k: object())
    opt,_ = study.optimizer_for(model)
    assert not opt.state_dict()['state']
    for p,m in zip(model.parameters(),opt.masters):assert torch.equal(p,m) and p.data_ptr() != m.data_ptr()
    model(torch.ones(1,3)).sum().backward();opt.step()
    assert model.weight.max() < 7 and model.bias.min() < -3


def test_trace_replays_complete_order_noise_and_resume_suffix(tmp_path):
    config,rows,records = synthetic_trace()
    full = audit.replay(records,rows,0,4,config)
    assert full['presentations'] == 32 and full['noisy_presentations'] == 16
    assert audit.replay(records[2:],rows,2,4,config)['presentations'] == 16
    path = tmp_path/'trace.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    assert study.verify_trace(path,rows,config)['distinct_sources_visited'] == 7


@pytest.mark.parametrize('change',['id','history','noise','denominator','loss','scale','missing','extra'])
def test_independent_replay_rejects_corrupted_training_evidence(change):
    config,rows,records = synthetic_trace();records = deepcopy(records)
    if change == 'id':records[0]['microbatches'][0]['ids'][0] = 'wrong'
    elif change == 'history':records[0]['microbatches'][0]['history_changes'][0] += 1
    elif change == 'noise':records[0]['microbatches'][0]['noisy'][0] = not records[0]['microbatches'][0]['noisy'][0]
    elif change == 'denominator':records[0]['examples'] = 4
    elif change == 'loss':records[0]['weighted_CE'] = .7
    elif change == 'scale':records[1]['scale'] = 64
    elif change == 'missing':records.pop()
    else:records.append(records[-1])
    with pytest.raises(ValueError):audit.replay(records,rows,0,4,config)


def test_full_training_cannot_start_without_complete_expanded_admission(monkeypatch):
    monkeypatch.setattr(study,'dataset',lambda arm:([],[],{},[]))
    monkeypatch.setattr(study,'require_parent',lambda:[])
    def refused():raise ValueError('full data not admitted')
    monkeypatch.setattr(study,'require_expanded',refused)
    assert study.prerequisites('existing','pilot') == ([],[],{},[])
    with pytest.raises(ValueError,match='full data not admitted'):study.prerequisites('existing','train')


def test_service_separates_lifecycle_success_from_training_success():
    result = {'status':'task_supervision_lifecycle_passed','arm':'existing','stage':'pilot','updates':2,
        'strict_final_reload_passed':True,'protected_tensors_unchanged':True,
        'fresh_optimizer_masters_match_initial_weights':True,'gradient_parity_passed':True,
        'strict_resumed_update_passed':True,'source_only_outputs_reproduced':True,
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False,'new_judge_requests':0}
    assert service.valid_result('existing','pilot',result)
    assert not service.valid_result('existing','train',result)
    assert not service.valid_result('expanded','pilot',result)
    result['strict_resumed_update_passed'] = False
    assert not service.valid_result('existing','pilot',result)
    assert service.command('existing','pilot')[-3:] == ['pilot','--arm','existing']
    assert service.ENV['CUDA_VISIBLE_DEVICES'] == '0' and service.ENV['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


def test_noisy_cpu_replay_never_loads_features(monkeypatch):
    config,rows,records = synthetic_trace()
    monkeypatch.setattr(study.catalog.FeatureCache,'get',lambda *a: pytest.fail('feature loading in label-only replay'))
    assert audit.replay(records,rows,0,4,config)['recorded_history_changes_replayed']
