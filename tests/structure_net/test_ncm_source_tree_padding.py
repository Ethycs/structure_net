import copy

import pytest
import torch

from experiments.structure_net import ncm_source_tree as tree
from experiments.structure_net import ncm_source_tree_padding as study
from tests.structure_net.test_ncm_source_tree_checkpointing import Leaf


def memory(k,v,mask):
    return study.per_source_levels(tree.build_tree((k,v,mask)),8)


def test_per_source_depth_includes_no_ancestor_case():
    lengths = [1,3,17,65]
    mask = torch.arange(65)[None,:] < torch.tensor(lengths)[:,None]
    k = torch.randn(4,2,65,4)
    value = memory(k,k,mask[:,None,None,:])
    assert value.active_levels.tolist() == [[False,False,False],[True,False,False],[True,True,False],[True,True,True]]


def test_padded_batch_matches_unpadded_outputs_and_gradients():
    torch.manual_seed(721)
    serial = study.PerSourceTreeAttention(Leaf(0.),tree.Config())
    with torch.no_grad():
        serial.tree_out.normal_(std=.1); serial.level_weights.normal_(std=.1)
    batched = copy.deepcopy(serial)
    lengths = [1,3,17,65]; x = torch.randn(4,7,8); k,v = [torch.randn(4,2,65,4) for _ in range(2)]
    xs,ks,vs = [a.clone().requires_grad_(True) for a in (x,k,v)]
    outputs = []
    for i,n in enumerate(lengths):
        mask = torch.ones(1,1,1,n,dtype=torch.bool)
        out = serial(xs[i:i+1],memory(ks[i:i+1,:,:n],vs[i:i+1,:,:n],mask))
        outputs.append(out); (out.square().sum()/4).backward()
    xb,kb,vb = [a.clone().requires_grad_(True) for a in (x,k,v)]
    mask = (torch.arange(65)[None,:]<torch.tensor(lengths)[:,None])[:,None,None,:]
    out = batched(xb,memory(kb,vb,mask)); (out.square().sum()/4).backward()
    torch.testing.assert_close(torch.cat(outputs),out,rtol=2e-5,atol=2e-6)
    for a,b in zip([xs.grad,ks.grad,vs.grad,*[p.grad for p in serial.parameters()]],
                   [xb.grad,kb.grad,vb.grad,*[p.grad for p in batched.parameters()]]):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    # Regression sensitivity: the previous unmasked mixture changes short examples.
    old = tree.TreeAttention(copy.deepcopy(serial.leaf),tree.Config()); old.load_state_dict(serial.state_dict())
    wrong = old(x,tree.build_tree((k,v,mask)))
    assert not torch.allclose(wrong,out.detach(),rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('dropout',[0.,.2])
def test_masked_checkpoint_preserves_gradient_and_random_state(dropout):
    torch.manual_seed(723)
    serial = study.PerSourceTreeAttention(Leaf(dropout),tree.Config())
    with torch.no_grad(): serial.tree_out.normal_(std=.1); serial.level_weights.normal_(std=.1)
    recomputed = study.CheckpointedPerSourceTreeAttention(copy.deepcopy(serial.leaf),tree.Config())
    recomputed.load_state_dict(serial.state_dict())
    values = [torch.randn(2,7,8),torch.randn(2,2,17,4),torch.randn(2,2,17,4)]
    mask = torch.ones(2,1,1,17,dtype=torch.bool); mask[0,...,3:] = False
    def run(module):
        x,k,v = [a.clone().requires_grad_(True) for a in values]
        mem = memory(k,v,mask); torch.manual_seed(725)
        out = module(x,mem); rng = torch.get_rng_state().clone(); out.square().sum().backward()
        assert torch.equal(rng,torch.get_rng_state())
        return [out.detach(),x.grad,k.grad,v.grad,*[p.grad for p in module.parameters()]]
    assert all(torch.equal(a,b) for a,b in zip(run(serial),run(recomputed)))
