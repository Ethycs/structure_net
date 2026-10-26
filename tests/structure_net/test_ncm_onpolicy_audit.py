from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import audit_ncm_onpolicy_training as audit


def fixture(arm='fixed_targets'):
    config = replace(audit.study.CONFIG,updates=2,warmup=1)
    rows = [{'id':str(i),'split':'TRAIN','oracle_codes':[i]+[0]*640} for i in range(5)]
    rng = torch.Generator().manual_seed(config.sampler_seed)
    remaining,trace = [],[]
    for step in (1,2):
        metrics = []
        for micro in range(4):
            if not remaining: remaining = torch.randperm(len(rows),generator=rng).tolist()
            row = rows[remaining.pop()]
            history,targets = list(row['oracle_codes']),list(row['oracle_codes'])
            if micro%2:
                history[0] += 1
                if arm == 'residual_targets': targets[1] += 1
            metrics.append({'id':row['id'],'onpolicy':bool(micro%2),'level_CE':[1.]*12,'mean_CE':1.,
                'history_errors':int(bool(micro%2)),
                'target_changes':int(bool(micro%2) and arm == 'residual_targets'),
                'history_codes':history if micro%2 else None,'target_codes':targets if micro%2 else None})
        trace.append({'step':step,'updates':2,'arm':arm,'lr':audit.study.broad.rate(step,config),
                      'metrics':metrics,'gradient_norm':1.,'mean_CE':1.,'peak_bytes':1})
    return rows,trace,config


@pytest.mark.parametrize('arm',['fixed_targets','residual_targets'])
def test_trace_replays_sampling_modes_and_raw_code_counts(arm):
    rows,trace,config = fixture(arm)
    result = audit.replay_trace(rows,trace,arm,config)
    assert sum(result['exposures'].values()) == 8 and len(result['policy_rows']) == 4
    assert sum(result['history_errors_by_level']) == 4
    assert sum(result['target_changes_by_level']) == (4 if arm == 'residual_targets' else 0)


@pytest.mark.parametrize('corruption',['source','mode','fixed_label','count','loss','budget','cap'])
def test_trace_rejects_wrong_source_label_mode_loss_or_denominator(corruption):
    rows,trace,config = fixture()
    trace = deepcopy(trace)
    metric = trace[0]['metrics'][1]
    if corruption == 'source': metric['id'] = 'wrong'
    elif corruption == 'mode': metric['onpolicy'] = False
    elif corruption == 'fixed_label':
        metric['target_codes'][1] = 1; metric['target_changes'] = 1
    elif corruption == 'count': metric['history_errors'] += 1
    elif corruption == 'loss': metric['level_CE'][0] = float('nan')
    elif corruption == 'budget': trace.pop()
    elif corruption == 'cap': trace[0]['peak_bytes'] = config.peak_limit_bytes+1
    with pytest.raises(ValueError): audit.replay_trace(rows,trace,'fixed_targets',config)


def test_nearest_label_audit_subtracts_history_not_earlier_targets():
    weights = torch.tensor([[-2.],[0.],[2.]])
    vq = SimpleNamespace(levels=[1,1,1],embed_dim=1,
        embeddings=[torch.nn.Embedding.from_pretrained(weights) for _ in range(3)],
        _down_for_quant=lambda x,i,n:x, _up_to_full=lambda x,i,n:x,
        _apply_quant_resi=lambda x,i:x,
        _distances=lambda x,w:(x[:,None,:]-w[None,:,:]).square().sum(-1))
    latent = torch.tensor([[[2.]]])
    history = torch.tensor([[1,1,1]])
    audit.check_nearest_labels(vq,latent,history,torch.tensor([[2,2,2]]))
    with pytest.raises(ValueError,match='nearest residual'):
        audit.check_nearest_labels(vq,latent,history,torch.tensor([[2,1,1]]))
    assert latent.tolist() == [[[2.]]] and history.tolist() == [[1,1,1]]
