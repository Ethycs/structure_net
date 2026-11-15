"""Protected-selection lifecycle and real tiny Qwen recovery coverage."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_3b_staged_train as study


def metrics(correct=45, ce=1., kl=.606, count=64):
    return dict(correct=correct,count=count,ce=ce,original_kl=kl)


def test_lower_lr_preserves_original_objective_budget_and_native_gate():
    a,b=study.CONFIG,study.inherited.CONFIG
    assert replace(a,learning_rate=b.learning_rate)==b
    assert a.learning_rate==2e-5 and (a.epochs,a.microbatch,a.accumulation)==(12,4,4)
    assert (a.temperature,a.kl_weight,a.ce_weight,a.dtype)==(2.,.7,.3,'bfloat16')
    check=study.inherited.gates(45,native_correct=48,parent_correct=52,count=64)
    assert check['passed'] and not check['parent']['passed']


@pytest.mark.parametrize('candidate,eligible',[
    (metrics(46,1.075,1.075),False), (metrics(64,.9,.607),False),
    (metrics(64,1.001,.5),False), (metrics(46,.9,.5),True),
    (metrics(46,1.+.5e-8,.606+.5e-8),True),
    (metrics(46,float('nan'),.5),False), (metrics(46,.9,float('inf')),False),
])
def test_joint_candidates_cannot_trade_original_drift_for_correct_count(candidate,eligible):
    assert study.candidate_eligibility(candidate,metrics())['eligible'] is eligible


def test_best_candidate_requires_honest_eligibility_and_protected_wins_tie():
    protected=metrics();history=[]
    for epoch,value in [(-1,protected),(0,metrics()),(1,metrics(64,.9,.9))]:
        history.append(dict(epoch=epoch,metrics=value,eligibility=study.candidate_eligibility(value,protected)))
    assert study.best_eligible(history,protected)['epoch']==-1
    history[-1]['eligibility']['eligible']=True
    with pytest.raises(ValueError,match='eligibility'):study.best_eligible(history,protected)


class FakeStack(nn.Module):
    def __init__(self):
        super().__init__();self.mark=nn.Parameter(torch.tensor(0.));self.register_buffer('fixed',torch.tensor(7.))
        self.layer_indices=(34,35);self.blocks=[None,None]


def lifecycle(tmp_path, monkeypatch, improve):
    trainer=object.__new__(study.CautiousTrainer)
    trainer.stack=FakeStack();trainer.norm=nn.Linear(1,1,bias=False)
    trainer.readout=torch.tensor([2.]);trainer.rotary=nn.Identity()
    trainer.config=study.CONFIG;trainer.updates=0;trainer.finished=False;trainer.guard=None
    trainer.native_correct=48;trainer.parent_correct=52;trainer.development_count=64
    trainer.dataset=SimpleNamespace(training=list(range(512)))
    # This represents an optimizer already initialized from warm weights.
    master=torch.tensor(0.);trainer.optimizer=SimpleNamespace(memory_stats=lambda:{'cpu_state_bytes':4},master=master)
    initial=metrics();table={-1:initial,0:metrics(46,1.1,1.075)}
    table.update({i:metrics(47,.9,.5) if improve and i>=2 else metrics(46,1.1,.7) for i in range(1,13)})
    def score():
        key=int(trainer.stack.mark);value=deepcopy(table[key]);return value,[{'state':key,'correct':value['correct']}]
    trainer.packed_development=score
    rates=[];first=[]
    def update(indices,learning_rate):
        if not first:first.append((float(trainer.stack.mark),float(master)))
        assert all(i in trainer.dataset.training for i in indices)
        trainer.updates+=1;rates.append(learning_rate)
        with torch.no_grad():trainer.stack.mark.fill_((trainer.updates+31)//32)
        return dict(update=trainer.updates,gradients_present_by_block=[True,True],cases=len(indices))
    trainer.train_window=update
    protected=trainer.snapshot();protected['stack']['mark'].fill_(-1.)
    raw_records=[{'state':-1,'correct':45}]
    result=trainer.fit(tmp_path/'run',protected,initial,raw_records)
    return trainer,result,protected,rates,first


@pytest.mark.parametrize('improve',[False,True])
def test_all384_updates_restore_warm_start_and_select_guarded_checkpoint(tmp_path,monkeypatch,improve):
    value,result,protected,rates,first=lifecycle(tmp_path,monkeypatch,improve)
    assert first==[(0.,0.)]  # Temporary raw scoring did not replace warm optimizer/model.
    assert result['updates']==384 and len(rates)==384 and rates[0]==2e-5
    assert [r['epoch'] for r in result['history']]==list(range(-1,13))
    assert result['selected_epoch']==(2 if improve else -1)
    assert result['strict_reload_exact'] and result['fixed_state_unchanged']
    assert result['protected_replay']['warm_restore_exact']
    assert result['protected_replay']['warm_state_fingerprint_before']==result['protected_replay']['warm_state_fingerprint_after']
    assert not result['history'][1]['eligibility']['eligible']
    state=study._load(tmp_path/'run/selected.pt')
    assert float(state['stack']['mark'])==result['selected_epoch']
    assert float(protected['stack']['mark'])==-1 and float(value.optimizer.master)==0
    assert study.best_eligible(result['history'],result['initial'])['epoch']==result['selected_epoch']
    assert result['quality']['passed']


def test_real_tiny_qwen_fit_and_raw_replay_rejection(tmp_path):
    # Reuse the existing actual BF16 affine/poly Qwen fixture; its producer
    # remains unchanged. Only the new fit lifecycle/selection is exercised.
    from test_jspace_qwen3b_progressive_train import components,data
    stack,norm,E,rotary,*_=components(cut=34);dataset=data(tmp_path)
    config=replace(study.CONFIG,epochs=1,microbatch=2,accumulation=2,optimizer_chunk_bytes=128)
    trainer=study.CautiousTrainer(stack,norm,E,rotary,dataset,[2,3],native_correct=3,parent_correct=4,count=4,config=config)
    protected=trainer.snapshot();raw,rows=trainer.packed_development()
    result=trainer.fit(tmp_path/'real',protected,raw,rows)
    assert result['updates']==1 and result['completed_epochs']==1
    assert result['strict_reload_exact'] and result['fixed_state_unchanged']
    assert result['history'][0]['epoch']==-1
    assert result['history'][2]['training'][0]['gradients_present_by_block']==[True,True]
    dataset.close()


def test_wrong_protected_report_fails_before_updates_and_restores_warm(tmp_path):
    from test_jspace_qwen3b_progressive_train import components,data
    stack,norm,E,rotary,*_=components(cut=34);dataset=data(tmp_path)
    trainer=study.CautiousTrainer(stack,norm,E,rotary,dataset,[2,3],native_correct=3,parent_correct=4,count=4,
        config=replace(study.CONFIG,epochs=1,microbatch=2,accumulation=2,optimizer_chunk_bytes=128))
    state=trainer.snapshot();raw,rows=trainer.packed_development();raw['ce']+=.1
    before=study.state_fingerprint(state)
    with pytest.raises(RuntimeError,match='Protected initializer'):trainer.fit(tmp_path/'bad',state,raw,rows)
    assert trainer.updates==0 and study.state_fingerprint(trainer.snapshot())==before
    assert all(s['step']==0 for s in trainer.optimizer.state.values())
    dataset.close()
