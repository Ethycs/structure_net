from copy import deepcopy
from types import SimpleNamespace
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net.jspace_compact_model import (
    RotatedAttention,RotatedBlock,RotatedMLP,RotatedEmbedding,RotatedFinalNorm,attend,denominator,fold_input,rope,
)
from experiments.structure_net.jspace_groebner_tensor import features


def toy():
    torch.manual_seed(916)
    d,m=16,24
    config=SimpleNamespace(hidden_size=d,num_attention_heads=4,num_key_value_heads=2)
    attention=SimpleNamespace(layer_idx=0,q_proj=nn.Linear(d,d).double(),k_proj=nn.Linear(d,8).double(),
                              v_proj=nn.Linear(d,8).double(),o_proj=nn.Linear(d,d,bias=False).double())
    norm=lambda:SimpleNamespace(weight=torch.rand(d,dtype=torch.double)+.7,variance_epsilon=1e-6)
    mlp=SimpleNamespace(gate_proj=nn.Linear(d,m,bias=False).double(),up_proj=nn.Linear(d,m,bias=False).double(),
                        down_proj=nn.Linear(m,d,bias=False).double())
    block=SimpleNamespace(self_attn=attention,mlp=mlp,input_layernorm=norm(),post_attention_layernorm=norm(),
                          attention_type="full_attention")
    q=torch.linalg.qr(torch.randn(d,d,dtype=torch.double)).Q
    return block,q,config


def positions(length,head_dim=4):
    angle=torch.arange(length,dtype=torch.double)[:,None]*torch.linspace(.1,.6,head_dim//2,dtype=torch.double)[None]
    angle=torch.cat((angle,angle),-1)[None]
    return angle.cos(),angle.sin()


def original_forward(block,x,pos):
    norm=lambda h,n:h/denominator(h,n.variance_epsilon)*n.weight
    h=norm(x,block.input_layernorm);b,t,d=h.shape
    q=block.self_attn.q_proj(h).view(b,t,4,4).transpose(1,2)
    k=block.self_attn.k_proj(h).view(b,t,2,4).transpose(1,2)
    v=block.self_attn.v_proj(h).view(b,t,2,4).transpose(1,2)
    q,k=rope(q,k,*pos)
    mixed=attend(q,k,v,None,.5).transpose(1,2).reshape(b,t,d)
    h=x+block.self_attn.o_proj(mixed)
    normalized=norm(h,block.post_attention_layernorm)
    return h+block.mlp.down_proj(F.silu(block.mlp.gate_proj(normalized))*block.mlp.up_proj(normalized))


def test_full_coordinate_rms_block_folding_preserves_function_and_gradient():
    original,q,config=toy();compiled=RotatedBlock(original,q,config)
    x=torch.randn(2,5,16,dtype=torch.double,requires_grad=True)
    target=original_forward(original,x,positions(5))
    actual=compiled(x@q,position_embeddings=positions(5))[0]@q.T
    assert torch.allclose(actual,target,atol=1e-11,rtol=1e-11)
    da=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    dt=torch.autograd.grad(target.square().sum(),x)[0]
    assert torch.allclose(da,dt,atol=1e-10,rtol=1e-10)


def test_affine_rms_fold_uses_original_width_and_retained_residual_energy():
    original,q,config=toy();u=q[:,:8];v=q[:,8:]
    z=torch.randn(5,8,dtype=torch.double);r=torch.randn(5,8,dtype=torch.double)
    h=z@u.T+r@v.T
    gain=original.input_layernorm.weight;w=original.self_attn.q_proj.weight
    denom=((z.square().sum(-1,keepdim=True)+r.square().sum(-1,keepdim=True))/16+1e-6).sqrt()
    target=F.linear(h/denominator(h,1e-6)*gain,w)
    assert torch.allclose(F.linear(torch.cat((z,r),-1),fold_input(w,gain,q))/denom,target,atol=1e-11)
    anchor=torch.randn(16,dtype=torch.double);anchor=anchor-u@(u.T@anchor)
    represented=anchor+z@u.T
    folded=(F.linear(z,(w*gain)@u)+(w*gain)@anchor)/((z.square().sum(-1,keepdim=True)+anchor.square().sum())/16+1e-6).sqrt()
    assert torch.allclose(folded,F.linear(represented/denominator(represented,1e-6)*gain,w),atol=1e-11)


def test_selected_rope_frequency_pairs_commute_and_keep_original_attention_scale():
    torch.manual_seed(3)
    q=torch.randn(1,2,5,8,dtype=torch.double);k=torch.randn(1,1,5,8,dtype=torch.double)
    cos,sin=positions(5,8);cols=[0,3,4,7]
    fullq,fullk=rope(q,k,cos,sin)
    smallq,smallk=rope(q[...,cols],k[...,cols],cos[...,cols],sin[...,cols])
    assert torch.equal(fullq[...,cols],smallq) and torch.equal(fullk[...,cols],smallk)
    expected=(fullq[...,cols]@fullk[...,cols].transpose(-1,-2))/math.sqrt(8)
    assert torch.equal(expected,(smallq@smallk.transpose(-1,-2))/math.sqrt(8))


def test_normal_form_output_is_compiled_from_deleted_neurons():
    original,q,config=toy()
    full=RotatedMLP(original.mlp,original.post_attention_layernorm.weight,q,1e-6)
    compact=deepcopy(full)
    fit={"anchors":[0,2,5,7],"mean":torch.randn(24,dtype=torch.double),
         "scale":torch.rand(24,dtype=torch.double)+1,
         "coefficients":{"2":torch.randn(24,15,dtype=torch.double)/10}}
    removed=[1,4,8,9,10,12,13,17]
    compact.compile(fit,removed)
    x=torch.randn(2,5,16,dtype=torch.double,requires_grad=True)
    gate,up=(F.linear(x,full.gate_up)/denominator(x,1e-6)).chunk(2,-1)
    hidden=F.silu(gate)*up
    z=(hidden[...,fit["anchors"]]-fit["mean"][fit["anchors"]])/fit["scale"][fit["anchors"]]
    predicted=features(z,2)@fit["coefficients"]["2"][removed].T
    substituted=hidden.clone();substituted[...,removed]=predicted*fit["scale"][removed]+fit["mean"][removed]
    target=F.linear(substituted,full.output)
    assert torch.allclose(compact(x),target,atol=1e-10)
    compact.pack()
    assert torch.allclose(compact(x),target,atol=1e-10)


def test_compact_attention_gradient_routing_and_cache_equivalence():
    original,q,config=toy()
    attention=RotatedAttention(original.self_attn,original.input_layernorm.weight,q,1e-6,config)
    s=torch.linalg.qr(torch.randn(4,2,dtype=torch.double)).Q
    attention.compile(8,torch.randn(16,dtype=torch.double),0,[1],s)
    x=torch.randn(1,5,16,dtype=torch.double,requires_grad=True)
    output=attention(x,positions(5));output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in attention.compact_parameters())
    assert attention.native_qkv.grad is None and attention.native_output.grad is None
    class Cache:
        def __init__(self):self.k=self.v=None
        def update(self,k,v,layer,kwargs):
            self.k=k if self.k is None else torch.cat((self.k,k),2)
            self.v=v if self.v is None else torch.cat((self.v,v),2)
            return self.k,self.v
    cache=Cache();cos,sin=positions(5)
    sequential=torch.cat([attention(x[:,i:i+1],(cos[:,i:i+1],sin[:,i:i+1]),past_key_value=cache,
                                      cache_position=torch.tensor([i])) for i in range(5)],1)
    assert torch.allclose(output,sequential,atol=1e-10,rtol=1e-10)


def test_hf_complete_coordinate_model_preserves_tied_readout_and_logits():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(29)
    config=Qwen2Config(hidden_size=16,intermediate_size=24,num_attention_heads=4,num_key_value_heads=2,
                      num_hidden_layers=2,vocab_size=32,tie_word_embeddings=True,max_position_embeddings=64)
    config._attn_implementation="eager"
    native=Qwen2ForCausalLM(config).eval().requires_grad_(False)
    rotated=deepcopy(native);q=torch.linalg.qr(torch.randn(16,16)).Q
    for i,block in enumerate(list(rotated.model.layers)):rotated.model.layers[i]=RotatedBlock(block,q,config)
    embedding=RotatedEmbedding(rotated.model.embed_tokens,q)
    rotated.model.norm=RotatedFinalNorm(rotated.model.norm,embedding);rotated.model.embed_tokens=embedding
    ids=torch.tensor([[2,4,6,8,10]])
    with torch.no_grad():
        expected=native(ids,use_cache=False).logits;actual=rotated(ids,use_cache=False).logits
    assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-5)
    assert rotated.lm_head.weight.data_ptr()==rotated.model.embed_tokens.weight.data_ptr()


def test_disk_teacher_cache_preserves_bfloat_bits(tmp_path):
    import numpy as np
    from experiments.structure_net.jspace_full_capture import write_bits,read_bits
    original=torch.tensor([0.,-1.3,9.999,1e-19,3e17],dtype=torch.bfloat16)
    path=tmp_path/"bits.npy"
    mapped=np.lib.format.open_memmap(path,mode="w+",dtype=np.int16,shape=(1,5))
    write_bits(mapped,0,original);mapped.flush()
    actual=read_bits(np.load(path,mmap_mode="r"),0,device="cpu")
    assert torch.equal(actual.view(torch.int16),original.view(torch.int16))


def test_reservation_wait_requires_a_fresh_success(monkeypatch):
    from experiments.structure_net import jspace_full_lease as lease
    calls=[];slept=[]
    def guard(devices):
        calls.append(devices)
        if len(calls)==1:raise lease.ReservationError("GPU lease is active but its memory is not ready")
        return {"ready":True}
    monkeypatch.setattr(lease,"require_gpu_lease",guard);monkeypatch.setattr(lease.time,"sleep",slept.append)
    assert lease.wait_ready([0])=={"ready":True}
    assert calls==[[0],[0]] and slept==[3]


def test_reservation_wait_does_not_retry_expired_or_invalid_lease(monkeypatch):
    import pytest
    from experiments.structure_net import jspace_full_lease as lease
    slept=[]
    def guard(devices):raise lease.ReservationError("GPU lease is not active or has expired")
    monkeypatch.setattr(lease,"require_gpu_lease",guard);monkeypatch.setattr(lease.time,"sleep",slept.append)
    with pytest.raises(lease.ReservationError,match="expired"):lease.wait_ready([0])
    assert not slept


def test_reservation_wait_times_out_while_still_unready(monkeypatch):
    import pytest
    from experiments.structure_net import jspace_full_lease as lease
    clock=iter([0,46]);slept=[]
    def guard(devices):raise lease.ReservationError("GPU lease is active but its memory is not ready")
    monkeypatch.setattr(lease,"require_gpu_lease",guard);monkeypatch.setattr(lease.time,"monotonic",lambda:next(clock))
    monkeypatch.setattr(lease.time,"sleep",slept.append)
    with pytest.raises(lease.ReservationError,match="not ready"):lease.wait_ready([0])
    assert not slept
