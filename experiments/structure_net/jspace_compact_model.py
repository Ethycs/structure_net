"""Orthogonal residual coordinates, affine RMS folding and polynomial cores."""
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def arithmetic_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def denominator(x, eps):
    return (x.to(arithmetic_dtype(x)).square().mean(-1, keepdim=True)+eps).sqrt()


def fold_input(weight, gain, basis):
    dtype=arithmetic_dtype(weight)
    return ((weight.to(dtype)*gain.to(dtype)[None,:])@basis.to(dtype)).to(weight.dtype)


def fold_output(weight, basis):
    dtype=arithmetic_dtype(weight)
    return (basis.to(dtype).T@weight.to(dtype)).to(weight.dtype)


def parameter(value, trainable=False):
    return nn.Parameter(value.detach().contiguous(), requires_grad=trainable)


def rotate_half(x):
    left,right=x.chunk(2,-1)
    return torch.cat((-right,left),-1)


def rope(q,k,cos,sin):
    return q*cos[:,None]+rotate_half(q)*sin[:,None], k*cos[:,None]+rotate_half(k)*sin[:,None]


def attend(q,k,v,mask,scale):
    groups=q.shape[1]//k.shape[1]
    if groups!=1:
        k=k.repeat_interleave(groups,1);v=v.repeat_interleave(groups,1)
    scores=(q@k.transpose(-1,-2))*scale
    if mask is not None:
        scores=scores+mask[...,:q.shape[-2],:k.shape[-2]]
    elif q.shape[-2]>1:
        causal=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).triu(1+k.shape[-2]-q.shape[-2])
        scores=scores.masked_fill(causal,torch.finfo(scores.dtype).min)
    probability=scores.to(arithmetic_dtype(q)).softmax(-1).to(q.dtype)
    return probability@v


class RotatedAttention(nn.Module):
    def __init__(self,original,gain,basis,eps,config):
        super().__init__()
        self.config=config;self.layer_idx=original.layer_idx
        self.head_dim=config.hidden_size//config.num_attention_heads
        self.nq=config.num_attention_heads;self.nkv=config.num_key_value_heads
        self.group=self.nq//self.nkv;self.width=config.hidden_size
        self.eps=eps;self.scale=self.head_dim**-.5;self.compact=False
        weights=torch.cat([layer.weight for layer in (original.q_proj,original.k_proj,original.v_proj)])
        self.qkv=parameter(fold_input(weights,gain,basis))
        biases=[layer.bias if layer.bias is not None else layer.weight.new_zeros(layer.out_features)
                for layer in (original.q_proj,original.k_proj,original.v_proj)]
        self.bias=parameter(torch.cat(biases))
        self.output=parameter(fold_output(original.o_proj.weight,basis))
        if original.o_proj.bias is not None:
            self.output_bias=parameter((basis.T@original.o_proj.bias).to(weights.dtype))
        else:self.register_buffer("output_bias",weights.new_zeros(self.width))

    def compile(self,rank,mean_coordinates,group,pairs,value_basis):
        if self.compact or self.nkv!=2:raise ValueError("Compilation requires two original GQA groups")
        if not 0<rank<=self.width:raise ValueError("Invalid functional width")
        d=self.head_dim;hd=2*len(pairs)
        if value_basis.shape!=(d,hd):raise ValueError("Value basis and selected RoPE width differ")
        device=self.qkv.device;dtype=arithmetic_dtype(self.qkv)
        cols=torch.tensor(list(pairs)+[p+d//2 for p in pairs],device=device)
        native=1-group
        qr=torch.arange(native*self.group*d,(native+1)*self.group*d,device=device)
        kr=self.nq*d+torch.arange(native*d,(native+1)*d,device=device)
        vr=(self.nq+self.nkv)*d+torch.arange(native*d,(native+1)*d,device=device)
        rows=torch.cat((qr,kr,vr))
        self.native_qkv=parameter(self.qkv[rows]);self.native_bias=parameter(self.bias[rows])
        qr_comp=torch.cat([cols+h*d for h in range(group*self.group,(group+1)*self.group)])
        kr_comp=self.nq*d+group*d+cols
        vr_full=(self.nq+self.nkv)*d+torch.arange(group*d,(group+1)*d,device=device)
        s=value_basis.to(device=device,dtype=dtype)
        w=torch.cat((self.qkv[qr_comp].to(dtype),self.qkv[kr_comp].to(dtype),s.T@self.qkv[vr_full].to(dtype)))
        b=torch.cat((self.bias[qr_comp].to(dtype),self.bias[kr_comp].to(dtype),s.T@self.bias[vr_full].to(dtype)))
        self.core_qkv=parameter(w[:,:rank],True)
        self.core_shift=parameter(w[:,rank:]@mean_coordinates[rank:].to(device=device,dtype=dtype),True)
        self.core_bias=parameter(b,True)
        self.native_output=parameter(self.output[:,qr])
        blocks=[self.output[:,h*d:(h+1)*d].to(dtype)@s for h in range(group*self.group,(group+1)*self.group)]
        self.core_output=parameter(torch.cat(blocks,1),True)
        self.register_buffer("rope_columns",cols)
        self.rank=rank;self.compact_dim=hd;self.compact_group=group;self.compact=True
        del self.qkv,self.bias,self.output

    def compact_parameters(self):
        if not self.compact:return []
        return [self.core_qkv,self.core_shift,self.core_bias,self.core_output]

    def forward(self,x,position_embeddings,attention_mask=None,past_key_value=None,cache_position=None):
        batch,length,_=x.shape
        scale=denominator(x,self.eps)
        cos,sin=position_embeddings
        if not self.compact:
            projected=(F.linear(x,self.qkv).to(arithmetic_dtype(x))/scale+self.bias).to(x.dtype)
            q,k,v=projected.split((self.nq*self.head_dim,self.nkv*self.head_dim,self.nkv*self.head_dim),-1)
            q=q.view(batch,length,self.nq,self.head_dim).transpose(1,2)
            k=k.view(batch,length,self.nkv,self.head_dim).transpose(1,2)
            v=v.view(batch,length,self.nkv,self.head_dim).transpose(1,2)
            q,k=rope(q,k,cos,sin)
            if past_key_value is not None:k,v=past_key_value.update(k,v,self.layer_idx,{"cache_position":cache_position})
            mixed=attend(q,k,v,attention_mask,self.scale).transpose(1,2).reshape(batch,length,-1)
            return (F.linear(mixed,self.output)+self.output_bias).to(x.dtype)
        native=(F.linear(x,self.native_qkv).to(arithmetic_dtype(x))/scale+self.native_bias).to(x.dtype)
        compact=((F.linear(x[...,:self.rank].to(self.core_qkv.dtype),self.core_qkv)+self.core_shift)/scale+self.core_bias).to(x.dtype)
        qn,kn,vn=native.split((self.group*self.head_dim,self.head_dim,self.head_dim),-1)
        qc,kc,vc=compact.split((self.group*self.compact_dim,self.compact_dim,self.compact_dim),-1)
        qn=qn.view(batch,length,self.group,self.head_dim).transpose(1,2)
        kn=kn.view(batch,length,1,self.head_dim).transpose(1,2);vn=vn.view(batch,length,1,self.head_dim).transpose(1,2)
        qc=qc.view(batch,length,self.group,self.compact_dim).transpose(1,2)
        kc=kc.view(batch,length,1,self.compact_dim).transpose(1,2);vc=vc.view(batch,length,1,self.compact_dim).transpose(1,2)
        qn,kn=rope(qn,kn,cos,sin)
        qc,kc=rope(qc,kc,cos[...,self.rope_columns],sin[...,self.rope_columns])
        if past_key_value is not None:
            padding=self.head_dim-self.compact_dim
            packed_k=torch.cat((kn,F.pad(kc,(0,padding))),1)
            packed_v=torch.cat((vn,F.pad(vc,(0,padding))),1)
            cached_k,cached_v=past_key_value.update(packed_k,packed_v,self.layer_idx,{"cache_position":cache_position})
            kn,vn=cached_k[:,:1],cached_v[:,:1]
            kc,vc=cached_k[:,1:2,:,:self.compact_dim],cached_v[:,1:2,:,:self.compact_dim]
        on=attend(qn,kn,vn,attention_mask,self.scale).transpose(1,2).reshape(batch,length,-1)
        oc=attend(qc,kc,vc,attention_mask,self.scale).transpose(1,2).reshape(batch,length,-1)
        value=F.linear(on,self.native_output).to(arithmetic_dtype(x))
        value=value+F.linear(oc.to(self.core_output.dtype),self.core_output)+self.output_bias
        return value.to(x.dtype)

    def pack(self):
        if self.compact:
            dtype=self.native_qkv.dtype
            for name in ("core_qkv","core_shift","core_bias","core_output"):
                setattr(self,name,parameter(getattr(self,name).to(dtype)))


class RotatedMLP(nn.Module):
    def __init__(self,original,gain,basis,eps):
        super().__init__()
        if any(layer.bias is not None for layer in (original.gate_proj,original.up_proj,original.down_proj)):
            raise ValueError("Qwen MLP projections must be bias free")
        self.eps=eps;self.compact=False;self.packed=False
        self.intermediate=original.gate_proj.out_features
        self.gate_up=parameter(fold_input(torch.cat((original.gate_proj.weight,original.up_proj.weight)),gain,basis))
        self.output=parameter(fold_output(original.down_proj.weight,basis))

    def compile(self,fit,removed):
        if self.compact:raise ValueError("Already compiled")
        n=self.intermediate;anchors=list(fit["anchors"]);removed=list(removed)
        if set(anchors)&set(removed):raise ValueError("Anchor elimination")
        keep=[i for i in range(n) if i not in set(removed)]
        self.gate_up=parameter(self.gate_up[keep+[i+n for i in keep]])
        dtype=arithmetic_dtype(self.output);device=self.output.device
        mean=fit["mean"].to(device=device,dtype=dtype);std=fit["scale"].to(device=device,dtype=dtype)
        c=fit["coefficients"]["2"][removed].to(device=device,dtype=dtype)
        weight=self.output[:,removed].to(dtype)
        self.poly_output=parameter((weight*std[removed])@c,True)
        self.poly_bias=parameter(weight@mean[removed],True)
        self.output=parameter(self.output[:,keep])
        self.register_buffer("anchor_positions",torch.tensor([keep.index(i) for i in anchors],device=device))
        self.register_buffer("anchor_mean",mean[anchors]);self.register_buffer("anchor_scale",std[anchors])
        pairs=torch.triu_indices(len(anchors),len(anchors),device=device)
        self.register_buffer("pair_i",pairs[0]);self.register_buffer("pair_j",pairs[1])
        self.intermediate=len(keep);self.removed=removed;self.compact=True

    def compact_parameters(self):
        return [self.poly_output,self.poly_bias] if self.compact and not self.packed else []

    def forward(self,x):
        pre=(F.linear(x,self.gate_up).to(arithmetic_dtype(x))/denominator(x,self.eps)).to(x.dtype)
        gate,up=pre.chunk(2,-1);hidden=F.silu(gate)*up
        if not self.compact:return F.linear(hidden,self.output)
        z=(hidden[...,self.anchor_positions].to(self.anchor_mean.dtype)-self.anchor_mean)/self.anchor_scale
        phi=torch.cat((torch.ones_like(z[...,:1]),z,z[...,self.pair_i]*z[...,self.pair_j]),-1)
        if self.packed:
            return (F.linear(torch.cat((hidden,phi.to(hidden.dtype)),-1),self.output).to(arithmetic_dtype(x))+self.poly_bias).to(x.dtype)
        value=F.linear(hidden,self.output).to(arithmetic_dtype(x))+F.linear(phi,self.poly_output)+self.poly_bias
        return value.to(x.dtype)

    def pack(self):
        if self.compact and not self.packed:
            self.output=parameter(torch.cat((self.output,self.poly_output.to(self.output.dtype)),1))
            self.poly_bias.requires_grad_(False)
            del self.poly_output
            self.packed=True


class RotatedBlock(nn.Module):
    def __init__(self,original,basis,config):
        super().__init__()
        self.attention_type=original.attention_type
        self.self_attn=RotatedAttention(original.self_attn,original.input_layernorm.weight,basis,
                                       original.input_layernorm.variance_epsilon,config)
        self.mlp=RotatedMLP(original.mlp,original.post_attention_layernorm.weight,basis,
                            original.post_attention_layernorm.variance_epsilon)

    def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None,**kwargs):
        hidden_states=hidden_states+self.self_attn(hidden_states,position_embeddings,attention_mask,past_key_value,cache_position)
        hidden_states=hidden_states+self.mlp(hidden_states)
        return (hidden_states,None) if output_attentions else (hidden_states,)

    def compact_parameters(self):
        return self.self_attn.compact_parameters()+self.mlp.compact_parameters()

    def pack(self):
        self.self_attn.pack();self.mlp.pack()


class RotatedEmbedding(nn.Module):
    def __init__(self,original,basis):
        super().__init__();self.original=original
        self.register_buffer("basis",basis.to(original.weight.dtype))

    @property
    def weight(self):return self.original.weight

    def forward(self,ids):return F.linear(self.original(ids),self.basis.T)


class RotatedFinalNorm(nn.Module):
    def __init__(self,original,embedding):
        super().__init__();self.original=original
        object.__setattr__(self,"embedding_reference",embedding)

    @property
    def weight(self):return self.original.weight

    def forward(self,coordinates):
        return self.original(F.linear(coordinates,self.embedding_reference.basis))
