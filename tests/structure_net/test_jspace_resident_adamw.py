"""CPU exact-reference tests; these do not establish CUDA speed or rounding."""
import json

import pytest
import torch

from experiments.structure_net.jspace_resident_adamw import ResidentSpillAdamW, STATE_KEYS, placement_plan
from experiments.structure_net.jspace_streaming_adamw import StreamingCPUAdamW


def parameters():
    return [torch.nn.Parameter(torch.linspace(-.7,1.3,n,dtype=dtype))
            for n,dtype in ((7,torch.bfloat16),(11,torch.float32),(3,torch.bfloat16))]


def cpu_optimizer(params):
    return StreamingCPUAdamW(params,lr=.002,betas=(.8,.95),eps=2e-7,
        weight_decay=.03,chunk_bytes=16,pin_memory=False)


def gradients(params,step):
    for i,p in enumerate(params):
        p.grad = (None if (step+i)%4==1 else
                  (torch.sin(torch.arange(p.numel()).float()+step+i)+.3).to(p.dtype)*8)


def same_state(left,right):
    for p,q in zip(left._parameters,right._parameters):
        assert torch.equal(p,q)
        assert left.state[p]['step']==right.state[q]['step']
        for key in STATE_KEYS:
            assert torch.equal(left.state[p][key],right.state[q][key])
            assert right.state[q][key].dtype==torch.float32


@pytest.mark.parametrize('budget,expected',[(0,[False,False,False]),(55,[False,False,False]),
    (92,[False,False,True]),(140,[True,False,False]),(308,[True,True,True])])
def test_real_work_budget_and_deterministic_whole_parameter_first_fit(budget,expected):
    params=parameters();plan=placement_plan(params,4,budget)
    assert plan['resident']==expected
    assert plan['resident_planned_peak_bytes']<=budget
    assert plan['resident_state_bytes']+plan['cpu_state_bytes']==21*12
    assert plan['resident_work_peak_bytes']==(56 if any(expected) else 0)
    optimizer=ResidentSpillAdamW.from_cpu_optimizer(cpu_optimizer(params),resident_device='cpu',
        resident_budget_bytes=budget,pin_memory=False)
    assert optimizer.placement==plan
    assert optimizer.memory_stats()['resident_backend']=='cpu'
    if any(expected):
        live=optimizer._resident_gradient.numel()*4+sum(t.numel()*t.element_size() for t in optimizer._resident_staging.values())
        assert live+optimizer.chunk_bytes==plan['resident_work_peak_bytes']
    else:
        assert optimizer._resident_gradient is None and not optimizer._resident_staging


@pytest.mark.parametrize('budget',[0,140,308])
def test_adopted_state_exactly_matches_frozen_cpu_reference_with_missing_gradients(budget):
    a,b=parameters(),parameters();reference,source=cpu_optimizer(a),cpu_optimizer(b)
    for step in range(3):
        gradients(a,step);gradients(b,step)
        reference.step(max_grad_norm=.9,grad_scale=8);source.step(max_grad_norm=.9,grad_scale=8)
    before=[{k:(v.clone() if torch.is_tensor(v) else v) for k,v in source.state[p].items()} for p in b]
    actual=ResidentSpillAdamW.from_cpu_optimizer(source,resident_device='cpu',
        resident_budget_bytes=budget,pin_memory=False)
    same_state(reference,actual)
    assert source._parameters==() and source.state=={} and source.param_groups==[]
    with pytest.raises(ValueError):source.step()
    for p,row in zip(b,before):
        for key in STATE_KEYS:assert torch.equal(actual.state[p][key],row[key])
    visited_resident=visited_spill=False
    for step,lr in enumerate([.007,.0001,0.,.0025,.00003,.001,.006],3):
        gradients(a,step);gradients(b,step)
        previous=[actual.state[p]['step'] for p in b]
        reference.param_groups[0]['lr']=actual.param_groups[0]['lr']=lr
        expected=reference.step(max_grad_norm=.9,grad_scale=8)
        measured=actual.step(max_grad_norm=.9,grad_scale=8)
        same_state(reference,actual)
        for key in ('unclipped_gradient_norm','clip_coefficient','updated_scalars','chunks'):
            assert measured[key]==expected[key]
        for p,old in zip(b,previous):assert actual.state[p]['step']==old+int(p.grad is not None)
        visited_resident |= measured['resident_chunks']>0
        visited_spill |= measured['spill_chunks']>0
    assert visited_resident==(budget>0) and visited_spill==(budget<308)


def test_checkpoint_preserves_sub_bf16_master_updates_and_resumes_other_placement(tmp_path):
    p=torch.nn.Parameter(torch.ones(7,dtype=torch.bfloat16))
    cpu=StreamingCPUAdamW([p],lr=1e-4,chunk_bytes=12,pin_memory=False)
    optimizer=ResidentSpillAdamW.from_cpu_optimizer(cpu,resident_device='cpu',resident_budget_bytes=1000,pin_memory=False)
    for _ in range(10):p.grad=torch.ones_like(p);optimizer.step()
    assert torch.equal(p,torch.ones_like(p)) and torch.all(optimizer.state[p]['master']<.9991)
    receipt=optimizer.save_checkpoint(tmp_path/'state');saved=optimizer.fingerprint()
    manifest=json.loads((tmp_path/'state/manifest.json').read_text())
    assert receipt['manifest_sha256']==optimizer._sha(tmp_path/'state/manifest.json')
    for row in manifest['parameters']:
        for chunk in row['chunks']:
            payload=torch.load(tmp_path/'state'/chunk['file'],map_location='cpu',weights_only=True)
            assert set(payload)==set(STATE_KEYS)
            assert sum(t.untyped_storage().nbytes() for t in payload.values())<=36
    q=torch.nn.Parameter(torch.full_like(p,8))
    other=ResidentSpillAdamW.from_cpu_optimizer(StreamingCPUAdamW([q],lr=.1,chunk_bytes=12,pin_memory=False),
        resident_device='cpu',resident_budget_bytes=0,pin_memory=False)
    assert other.load_checkpoint(tmp_path/'state')['restored']
    assert other.fingerprint()==saved
    for step in range(30):
        p.grad=q.grad=None if step==5 else torch.ones_like(p)
        optimizer.param_groups[0]['lr']=other.param_groups[0]['lr']=1e-4/(1+step/20)
        optimizer.step();other.step();same_state(optimizer,other)
    assert torch.all(p<1)


def test_checkpoint_chunk_tamper_and_incomplete_manifest_rejected(tmp_path):
    optimizer=ResidentSpillAdamW.from_cpu_optimizer(cpu_optimizer(parameters()),resident_device='cpu',
        resident_budget_bytes=140,pin_memory=False)
    optimizer.save_checkpoint(tmp_path/'tamper')
    manifest=json.loads((tmp_path/'tamper/manifest.json').read_text())
    path=tmp_path/'tamper'/manifest['parameters'][0]['chunks'][0]['file']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Changed optimizer checkpoint'):optimizer.load_checkpoint(tmp_path/'tamper')
    optimizer.save_checkpoint(tmp_path/'incomplete')
    path=tmp_path/'incomplete/manifest.json';manifest=json.loads(path.read_text());manifest['status']='running'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Invalid optimizer checkpoint'):optimizer.load_checkpoint(tmp_path/'incomplete')


def test_nonfinite_global_gradient_rejects_before_resident_or_spill_mutation():
    optimizer=ResidentSpillAdamW.from_cpu_optimizer(cpu_optimizer(parameters()),resident_device='cpu',
        resident_budget_bytes=140,pin_memory=False)
    before=optimizer.fingerprint()
    for p in optimizer._parameters:p.grad=torch.ones_like(p)
    optimizer._parameters[-1].grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):optimizer.step(max_grad_norm=1)
    assert optimizer.fingerprint()==before
    assert all(state['step']==0 for state in optimizer.state.values())


def test_cuda_guard_runs_before_allocation_and_failed_adoption_keeps_source(monkeypatch):
    source=cpu_optimizer(parameters());original=list(source._parameters)
    monkeypatch.setattr(torch,'empty',lambda *_a,**_k:pytest.fail('allocation before authorization'))
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('CUDA initialization before authorization'))
    with pytest.raises(ValueError,match='reservation guard'):
        ResidentSpillAdamW.from_cpu_optimizer(source,resident_device='cuda:0',resident_budget_bytes=140,pin_memory=True)
    def denied():raise PermissionError('lease denied')
    with pytest.raises(PermissionError,match='lease denied'):
        ResidentSpillAdamW.from_cpu_optimizer(source,resident_device='cuda:0',resident_budget_bytes=140,guard=denied,pin_memory=True)
    assert list(source._parameters)==original and len(source.state)==3


def test_step_guard_rejection_prevents_any_update():
    optimizer=ResidentSpillAdamW.from_cpu_optimizer(cpu_optimizer(parameters()),resident_device='cpu',
        resident_budget_bytes=140,pin_memory=False)
    before=optimizer.fingerprint()
    for p in optimizer._parameters:p.grad=torch.ones_like(p)
    def denied():raise PermissionError('expired')
    optimizer.guard=denied
    with pytest.raises(PermissionError):optimizer.step()
    optimizer.guard=None
    assert optimizer.fingerprint()==before
