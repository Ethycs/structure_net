import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_source_tree as tree
from experiments.structure_net import ncm_source_tree_checkpointing as study


class Leaf(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.width, self.heads, self.dropout = 8, 2, dropout
        self.norm = nn.LayerNorm(8)
        self.q, self.out = nn.Linear(8,8,bias=False), nn.Linear(8,8,bias=False)

    def forward(self, targets, memory):
        k,v,mask = memory
        q = self.q(self.norm(targets)).reshape(len(targets),-1,2,4).transpose(1,2)
        attended = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.dropout if self.training else 0.)
        return self.out(attended.transpose(1,2).reshape_as(targets))


@pytest.mark.parametrize('dropout',[0.,.2])
def test_checkpoint_preserves_outputs_gradients_and_rng(dropout):
    torch.manual_seed(917)
    reference = tree.TreeAttention(Leaf(dropout), tree.Config())
    with torch.no_grad():
        reference.tree_out.normal_(std=.1); reference.level_weights.normal_(std=.1)
    tested = study.CheckpointedTreeAttention(copy.deepcopy(reference.leaf),reference.config)
    tested.load_state_dict(reference.state_dict(),strict=True)
    source = [torch.randn(2,7,8),torch.randn(2,2,17,4),torch.randn(2,2,17,4)]
    mask = torch.ones(2,1,1,17,dtype=torch.bool); mask[1,...,13:] = False
    def run(module):
        x,k,v = [item.clone().requires_grad_(True) for item in source]
        memory = tree.build_tree((k,v,mask))
        torch.manual_seed(919)
        output = module(x,memory); after = torch.get_rng_state().clone()
        output.square().sum().backward()
        assert torch.equal(torch.get_rng_state(),after)
        return [output.detach(), x.grad, k.grad, v.grad, *[p.grad for p in module.parameters()]]
    expected,actual = run(reference),run(tested)
    assert len(expected)==len(actual)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
