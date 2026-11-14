from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net.jspace_compact_model import RotatedAttention


@pytest.mark.parametrize("head_dim",[4,128])
@pytest.mark.parametrize("group",[0,1])
def test_split_attention_without_compression_preserves_outputs_and_input_gradient(head_dim,group):
    torch.manual_seed(190)
    width=4*head_dim
    config=SimpleNamespace(hidden_size=width,num_attention_heads=4,num_key_value_heads=2)
    original=SimpleNamespace(layer_idx=0,q_proj=nn.Linear(width,width).double(),
        k_proj=nn.Linear(width,2*head_dim).double(),v_proj=nn.Linear(width,2*head_dim).double(),
        o_proj=nn.Linear(width,width,bias=False).double())
    basis=torch.linalg.qr(torch.randn(width,width,dtype=torch.double)).Q
    native=RotatedAttention(original,torch.rand(width,dtype=torch.double)+.5,basis,1e-6,config)
    split=deepcopy(native)
    s=torch.linalg.qr(torch.randn(head_dim,head_dim,dtype=torch.double)).Q
    split.compile(width,torch.randn(width,dtype=torch.double),group,list(range(head_dim//2)),s)
    x=torch.randn(1,5,width,dtype=torch.double,requires_grad=True)
    angle=torch.rand(1,5,head_dim//2,dtype=torch.double);angle=torch.cat((angle,angle),-1)
    positions=(angle.cos(),angle.sin())
    expected=native(x,positions);actual=split(x,positions)
    assert torch.allclose(expected,actual,atol=1e-11,rtol=1e-10)
    grad_a=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    grad_e=torch.autograd.grad(expected.square().sum(),x)[0]
    assert torch.allclose(grad_a,grad_e,atol=1e-10,rtol=1e-10)
    split.pack()
    assert torch.allclose(expected,split(x,positions),atol=1e-11,rtol=1e-10)
