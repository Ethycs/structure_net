import copy

import pytest
import torch

from experiments.structure_net import ncm_source_hierarchy_mechanism as study
from experiments.structure_net.ncm_lm1b_source_attention import SourceAttention
from tests.structure_net.test_ncm_source_coverage import toy,forward


def test_dense_install_retains_parameters_rng_and_initial_function():
    model=toy(); x,source=torch.randn(2,4,8),torch.randn(2,17,8)
    expected=forward(model,x,source); old=dict(model.named_parameters()); rng=torch.get_rng_state()
    study.install(model,'dense')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(dict(model.named_parameters())[k] is p for k,p in old.items())
    assert torch.equal(expected,forward(model,x,source))
    assert model.concepts.layers==tuple(range(12))
    assert all(isinstance(a,study.CheckpointedSourceAttention) for a in model.concepts.adapters.values())


@pytest.mark.parametrize('dropout',[0.,.2])
def test_dense_recomputation_matches_gradients_and_rng(dropout):
    torch.manual_seed(817); original=SourceAttention(8,2,dropout)
    with torch.no_grad(): original.out.weight.normal_(std=.1)
    tested=study.CheckpointedSourceAttention(copy.deepcopy(original))
    values=[torch.randn(2,7,8),torch.randn(2,2,17,4),torch.randn(2,2,17,4)]
    mask=torch.ones(2,1,1,17,dtype=torch.bool); mask[0,...,3:]=False
    def run(module):
        x,k,v=[a.clone().requires_grad_(True) for a in values];torch.manual_seed(819)
        out=module(x,(k,v,mask));rng=torch.get_rng_state().clone();out.square().sum().backward()
        assert torch.equal(rng,torch.get_rng_state())
        return [out.detach(),x.grad,k.grad,v.grad,*[p.grad for p in module.parameters()]]
    assert all(torch.equal(a,b) for a,b in zip(run(original),run(tested)))
