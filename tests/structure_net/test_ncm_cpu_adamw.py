import copy

import pytest
import torch

from experiments.structure_net.ncm_cpu_adamw import CPUAdamW, assert_cpu_state


def test_offload_matches_standard_adamw_and_roundtrips_moments():
    torch.manual_seed(11)
    actual=[torch.nn.Parameter(torch.randn(7,3)),torch.nn.Parameter(torch.randn(3))]
    reference=[torch.nn.Parameter(p.detach().clone()) for p in actual]
    offload=CPUAdamW(actual,lr=.001)
    standard=torch.optim.AdamW(reference,lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    for step in range(4):
        for a,b in zip(actual,reference):a.grad=torch.randn_like(a);b.grad=a.grad.clone()
        offload.param_groups[0]['lr']=standard.param_groups[0]['lr']=.001/(step+1)
        offload.step();standard.step();assert_cpu_state(offload)
        assert all(torch.equal(a,b) for a,b in zip(actual,reference))
    state=copy.deepcopy(offload.state_dict())
    restored=CPUAdamW(actual,lr=.7);restored.load_state_dict(state)
    assert restored.param_groups[0]['lr']==offload.param_groups[0]['lr']
    for a,b in zip(actual,reference):a.grad=torch.randn_like(a);b.grad=a.grad.clone()
    restored.step();standard.step()
    assert all(torch.equal(a,b) for a,b in zip(actual,reference))
    assert_cpu_state(restored)


def test_parameters_without_gradients_do_not_receive_weight_decay():
    a=torch.nn.Parameter(torch.ones(4));b=torch.nn.Parameter(torch.ones(4))
    optimizer=CPUAdamW([a,b],lr=.1,weight_decay=.2)
    a.grad=torch.ones_like(a);optimizer.step()
    assert torch.equal(b,torch.ones_like(b))
    with pytest.raises(ValueError,match='FP32'):
        CPUAdamW([torch.nn.Parameter(torch.ones(4,dtype=torch.float16))],lr=.1)
