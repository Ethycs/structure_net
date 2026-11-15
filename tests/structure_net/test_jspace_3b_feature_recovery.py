"""CPU controls for live feature gradients, train-only calibration and lifecycle."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_feature_recovery as study
from test_jspace_3b_hidden_feature_run import tiny, inputs, cases


@pytest.mark.parametrize('checkpointing',[False,True])
def test_actual_feature_forward_matches_original_outputs_and_all_gradients(checkpointing):
    stack=tiny()
    for p in stack.parameters():p.data=p.data.bfloat16()
    stack.requires_grad_(True).train();stack.checkpoint_blocks=checkpointing
    reference=deepcopy(stack);x,batch,rope=inputs();x=x.requires_grad_();y=x.detach().clone().requires_grad_()
    kwargs=dict(position_ids=batch['position_ids'],position_embeddings=rope,attention_mask=batch['attention_mask'])
    expected=reference(x,cache_position=batch['position_ids'][0],use_cache=False,**kwargs)[0]
    actual,z=study.forward_features(stack,y,**kwargs)
    assert torch.equal(actual,expected) and z.requires_grad and z.shape==(2,5,4)
    expected.float().square().mean().backward();actual.float().square().mean().backward()
    assert torch.equal(x.grad,y.grad)
    for a,b in zip(stack.parameters(),reference.parameters()):assert torch.equal(a.grad,b.grad)


def test_feature_only_backward_reaches_entrance_through_all_checkpointed_layers():
    stack=tiny();stack.checkpoint_blocks=True;stack.requires_grad_(True).train()
    for p in stack.parameters():p.data=p.data.bfloat16()
    x,batch,rope=inputs();before=deepcopy(stack.state_dict())
    _,z=study.forward_features(stack,x,position_ids=batch['position_ids'],position_embeddings=rope,attention_mask=batch['attention_mask'])
    targets=study.AnswerTargets(torch.ones(2,4,dtype=torch.bfloat16),batch['indices'])
    loss=study.feature_loss(z,batch,targets);loss.backward()
    assert all(all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters()) for b in stack.blocks)
    assert sum(float(p.grad.float().square().sum()) for p in stack.blocks[0].parameters())>0
    assert study.old.state_equal(stack.state_dict(),before)
    assert all(not b.requires_grad and b.grad is None for b in stack.buffers())


def test_train_only_answer_positions_and_frozen_normalization():
    target=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    frozen=float(target.float().square().mean())
    targets=study.AnswerTargets(target,[5,9],normalization_energy=frozen)
    z=torch.zeros(2,4,2,requires_grad=True);batch=dict(indices=[9,5],lengths=[2,4])
    loss=study.feature_loss(z,batch,targets)
    assert float(loss)==1.;loss.backward()
    assert torch.count_nonzero(z.grad[0,:1])==0 and torch.count_nonzero(z.grad[0,2:])==0
    assert torch.count_nonzero(z.grad[1,:3])==0
    assert target.grad is None and targets.energy==frozen
    with pytest.raises(ValueError,match='training-only'):targets.batch([0],'cpu')
    with pytest.raises(ValueError,match='energy'):study.AnswerTargets(target,[5,9],normalization_energy=frozen*2)


def test_fixed_calibration_is_first_eight_per_family_in_original_source_order():
    items=cases();items[0]['split']='development'
    indices=study.calibration_indices(items)
    assert len(indices)==32 and indices==sorted(indices) and 0 not in indices
    for family in {items[i]['family'] for i in indices}:
        assert [i for i in indices if items[i]['family']==family]==[i for i,c in enumerate(items) if c['split']=='train' and c['family']==family][:8]


class TinyData:
    width=8;vocabulary=16;original=True
    def __init__(self):
        self.items=[dict(identity=f'c{i}',family=('a','b','c','d')[i%4],split='train' if i<32 else 'development',
            choices=['a','b'],answer=i%2,targets=[1,2,3,i%2]) for i in range(34)]
        self.training=list(range(32));self.development=[32,33];self.lengths=[3 if i%2==0 else 5 for i in range(34)]
        torch.manual_seed(789);self.x=torch.randn(34,5,8).bfloat16();self.teacher=torch.randn(34,4,16).bfloat16()
    def batch(self,indices,*,device='cpu',dtype=torch.bfloat16,include_original=False):
        lengths=[self.lengths[i] for i in indices];length=max(lengths);ids=torch.arange(length,device=device)[None].expand(len(indices),-1)
        valid=ids<torch.tensor(lengths,device=device)[:,None];allowed=(ids[:,None,:]<=ids[:,:,None])&valid[:,None,:]
        mask=torch.zeros(len(indices),1,length,length,device=device,dtype=dtype).masked_fill(~allowed[:,None],torch.finfo(dtype).min)
        result=dict(indices=list(indices),hidden=self.x[indices,:length].to(device=device,dtype=dtype),position_ids=ids,
            attention_mask=mask,lengths=lengths,loss_positions=torch.tensor([[0,0,1,n-1] for n in lengths],device=device),
            targets=torch.tensor([self.items[i]['targets'] for i in indices],device=device),teacher=self.teacher[indices].to(device))
        if include_original:result['original']=result['teacher']
        return result


def trainer():
    stack=tiny();norm=Qwen2RMSNorm(8).bfloat16();E=torch.randn(16,8).bfloat16().requires_grad_(False)
    config=Qwen2Config(hidden_size=8,num_attention_heads=2,max_position_embeddings=16)
    rotary=Qwen2RotaryEmbedding(config);data=TinyData()
    targets=study.AnswerTargets(torch.randn(32,4).bfloat16(),data.training)
    config=replace(study.CONFIG,epochs=1,microbatch=4,accumulation=1,optimizer_chunk_bytes=64)
    return study.FeatureTrainer(stack,norm,E,rotary,data,[0,1],targets=targets,config=config,native_correct=2,parent_correct=2,count=2)


def test_real_train_only_calibration_keeps_weights_optimizer_and_finalnorm_unchanged():
    instance=trainer();before=instance.snapshot();states={id(p):deepcopy(s) for p,s in instance.optimizer.state.items()}
    calibration=instance.calibrate(study.calibration_indices(instance.dataset.items))
    assert calibration['count']==32 and calibration['optimizer_updates']==0 and calibration['development_used'] is False
    assert calibration['alpha']==pytest.approx(.1*(calibration['endpoint_gradient_squared']/calibration['feature_gradient_squared'])**.5)
    assert study.training.state_fingerprint(before)==study.training.state_fingerprint(instance.snapshot())
    for p,state in instance.optimizer.state.items():
        assert p.grad is None
        for key,value in state.items():
            assert torch.equal(value,states[id(p)][key]) if torch.is_tensor(value) else value==states[id(p)][key]
    assert all(g['indices']==calibration['indices'][4*i:4*i+4] for i,g in enumerate(calibration['groups']))
    update=instance.train_window([0,1,2,3])
    assert update['gradients_present_by_block']==[True]*3 and update['update']==1
    assert update['loss']==pytest.approx(update['endpoint_loss']+instance.alpha*update['feature_loss'],rel=1e-6)
    assert update['feature_loss']>0 and all(p.grad is None for p in instance.parameters)
    with pytest.raises(ValueError,match='training'):instance.train_window([32])


def test_original_fit_packed_evaluation_and_eligible_selection_are_inherited():
    assert study.FeatureTrainer.fit is study.training.CautiousTrainer.fit
    assert study.FeatureTrainer.packed_development is study.training.CautiousTrainer.packed_development
    assert study.FeatureTrainer.quality is study.training.CautiousTrainer.quality
    assert study.CONFIG is study.training.CONFIG


def test_calibration_receipt_rejects_wrong_rows_coefficient_and_development():
    indices=list(range(32));groups=[dict(indices=indices[i:i+4],endpoint_gradient_squared=4.,feature_gradient_squared=1.) for i in range(0,32,4)]
    plan=dict(calibration_indices=indices,target_receipt={'normalization_energy':2.})
    receipt=dict(groups=groups,indices=indices,endpoint_gradient_squared=32.,feature_gradient_squared=8.,ratio=.1,alpha=.2,
        normalization_energy=2.,unchanged_warm_state=True,state_before='same',state_after='same',development_used=False,optimizer_updates=0)
    study.validate_calibration(receipt,plan)
    for key,value in (('alpha',.3),('development_used',True),('state_after','changed')):
        with pytest.raises(ValueError,match='calibration'):study.validate_calibration(dict(receipt,**{key:value}),plan)
    altered=deepcopy(receipt);altered['groups'][0]['indices'][0]=513
    with pytest.raises(ValueError,match='calibration'):study.validate_calibration(altered,plan)


def diagnostic_fixture(tmp_path,monkeypatch):
    root=tmp_path/'diagnostic';path=root/'run';path.mkdir(parents=True)
    rows=study.capture.case_rows(cases(),list(range(576)))
    plan=dict(fingerprint='diagnostic',rows=rows)
    monkeypatch.setattr(study.capture,'verify',lambda *_:plan)
    values=torch.arange(512*1024,dtype=torch.float32).reshape(512,1024).remainder(13).bfloat16()+1
    metadata=dict(rows=rows[:512],shape=[512,1024],teacher_checkpoint_sha256=study.capture.PARENT_SHA,coordinate_layer=35,
        position='last valid token',normalization_fitted_on_development=False,energy_mean=float(values.float().square().mean()))
    study.base.write(path/'parent_training_final_answers.json',metadata)
    with (path/'parent_training_final_answers.npy').open('xb') as stream:np.save(stream,values.view(torch.int16).numpy())
    study.base.write(path/'result.json',dict(status='completed',fingerprint='diagnostic',modes={'parent_packed':{'source_replay':{'passed':True}}}))
    study.base.write(path/'manifest.json',dict(status='completed',fingerprint='diagnostic',artifacts={p.name:study.base.sha(p) for p in path.iterdir()}))
    return root,metadata


def test_authenticated_targets_exclude_development_and_reject_tampered_bytes(tmp_path,monkeypatch):
    root,metadata=diagnostic_fixture(tmp_path,monkeypatch)
    targets,receipt=study.load_targets(root)
    assert targets.energy==metadata['energy_mean'] and receipt['training_count']==512 and 512 not in targets.lookup
    (root/'run/parent_training_final_answers.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):study.load_targets(root)


def test_train_requires_completed_pilot_before_targets_or_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'verify_plan',lambda *_:({'fingerprint':'plan'},None,None))
    def denied(*_):raise ValueError('pilot not complete')
    monkeypatch.setattr(study,'checked',denied)
    monkeypatch.setattr(study,'load_targets',lambda *_:pytest.fail('targets read before pilot admission'))
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('CUDA touched'))
    with pytest.raises(ValueError,match='pilot'):study.run(tmp_path,'train')


def test_diagnostic_best_task_checkpoint_remains_separate_from_eligible_winner(tmp_path):
    class Model:
        def snapshot(self):return {'stack':{'weight':torch.tensor(2.)},'norm':{}}
    raw={'stack':{'weight':torch.tensor(1.)},'norm':{}}
    protected=dict(count=64,correct=46,ce=1.,original_kl=.01)
    (tmp_path/'selected.pt').write_bytes(b'protected selected winner')
    recorder=study.DiagnosticCheckpoint(tmp_path,Model(),raw,protected)
    recorder.observe(-1,protected)
    better=dict(count=64,correct=49,ce=2.,original_kl=.1)
    recorder.observe(1,better)
    assert (tmp_path/'selected.pt').read_bytes()==b'protected selected winner'
    assert recorder.best['eligibility']['eligible'] is False and recorder.best['automatically_adopted'] is False
    assert torch.load(tmp_path/'diagnostic_best_task.pt',weights_only=True)['stack']['weight']==2
    assert (tmp_path/'diagnostic_improvement_-1.json').exists() and (tmp_path/'diagnostic_improvement_01.json').exists()
