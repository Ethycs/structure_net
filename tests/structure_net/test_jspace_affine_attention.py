from types import SimpleNamespace
import pytest
import torch
from torch import nn

from experiments.structure_net.jspace_compact_model import RotatedAttention
from experiments.structure_net.jspace_affine_attention import AffinePackedAttention,value_mean_offset


def fixture(constant=False):
    torch.manual_seed(614)
    config=SimpleNamespace(hidden_size=16,num_attention_heads=4,num_key_value_heads=2)
    original=SimpleNamespace(layer_idx=0,q_proj=nn.Linear(16,16).double(),k_proj=nn.Linear(16,8).double(),
        v_proj=nn.Linear(16,8).double(),o_proj=nn.Linear(16,16,bias=False).double())
    if constant:
        with torch.no_grad():
            for layer in (original.q_proj,original.k_proj,original.v_proj):layer.weight.zero_()
            original.q_proj.bias.zero_();original.k_proj.bias.zero_()
    basis=torch.linalg.qr(torch.randn(16,16,dtype=torch.double)).Q;gain=torch.rand(16,dtype=torch.double)+.5
    args=(original,gain,basis,1e-6,config)
    s=torch.linalg.qr(torch.randn(4,2,dtype=torch.double)).Q
    x=torch.randn(1,7,16,dtype=torch.double,requires_grad=True)
    angles=torch.randn(1,7,2,dtype=torch.double);angles=torch.cat((angles,angles),-1)
    return original,args,s,x,(angles.cos(),angles.sin())


@pytest.mark.parametrize("group",[0,1])
def test_affine_value_offset_preserves_constant_mean_that_centered_pca_discards(group):
    original,args,s,x,pos=fixture(constant=True)
    full=RotatedAttention(*args);corrected=AffinePackedAttention(*args);uncorrected=RotatedAttention(*args)
    mean=original.v_proj.bias[group*4:(group+1)*4].detach()
    offset=value_mean_offset(corrected,group,s,mean)
    for a in (corrected,uncorrected):a.compile(8,torch.zeros(16,dtype=torch.double),group,[0],s)
    corrected.output_bias=corrected.output_bias+offset
    expected=full(x,pos)
    assert not torch.allclose(uncorrected(x,pos),expected,atol=1e-5)
    assert torch.allclose(corrected(x,pos),expected,atol=1e-11)
    corrected.pack()
    assert torch.allclose(corrected(x,pos),expected,atol=1e-11)


def test_padded_attention_and_single_output_gemm_match_separate_branches_and_cache():
    original,args,s,x,pos=fixture()
    separate=RotatedAttention(*args);combined=AffinePackedAttention(*args)
    for a in (separate,combined):a.compile(8,torch.randn(16,dtype=torch.double)*0,0,[1],s)
    expected=separate(x,pos);actual=combined(x,pos)
    assert torch.allclose(actual,expected,atol=1e-11)
    da=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    de=torch.autograd.grad(expected.square().sum(),x)[0]
    assert torch.allclose(da,de,atol=1e-10)
    combined.pack()
    assert torch.allclose(combined(x,pos),expected,atol=1e-11)
    class Cache:
        def __init__(self):self.k=self.v=None
        def update(self,k,v,layer,kwargs):
            self.k=k if self.k is None else torch.cat((self.k,k),2)
            self.v=v if self.v is None else torch.cat((self.v,v),2)
            return self.k,self.v
    cache=Cache();cos,sin=pos
    sequential=torch.cat([combined(x[:,i:i+1],(cos[:,i:i+1],sin[:,i:i+1]),past_key_value=cache) for i in range(7)],1)
    assert torch.allclose(sequential,expected,atol=1e-11)
