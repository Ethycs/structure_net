"""Affine value reconstruction and a single packed attention output projection."""
import torch
from torch.nn import functional as F

from .jspace_compact_model import RotatedAttention,arithmetic_dtype,denominator,rope,attend,parameter


def value_mean_offset(attention,group,value_basis,value_mean):
    dtype=arithmetic_dtype(attention.output);device=attention.output.device
    s=value_basis.to(device=device,dtype=dtype);mean=value_mean.to(device=device,dtype=dtype)
    residual=mean-s@(s.T@mean);d=attention.head_dim
    return sum(attention.output[:,h*d:(h+1)*d].to(dtype)@residual
               for h in range(group*attention.group,(group+1)*attention.group))


class AffinePackedAttention(RotatedAttention):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.packed=False

    def compact_parameters(self):
        return [] if self.packed else super().compact_parameters()

    def forward(self,x,position_embeddings,attention_mask=None,past_key_value=None,cache_position=None):
        if not self.compact:return super().forward(x,position_embeddings,attention_mask,past_key_value,cache_position)
        batch,length,_=x.shape;scale=denominator(x,self.eps);cos,sin=position_embeddings
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
        padding=self.head_dim-self.compact_dim
        q=torch.cat((qn,F.pad(qc,(0,padding))),1)
        k=torch.cat((kn,F.pad(kc,(0,padding))),1);v=torch.cat((vn,F.pad(vc,(0,padding))),1)
        if past_key_value is not None:k,v=past_key_value.update(k,v,self.layer_idx,{"cache_position":cache_position})
        mixed=attend(q,k,v,attention_mask,self.scale).transpose(1,2)
        features=torch.cat((mixed[:,:,:self.group].reshape(batch,length,-1),
                            mixed[:,:,self.group:,:self.compact_dim].reshape(batch,length,-1)),-1)
        if self.packed:
            constants=self.output_constants.expand(batch,length,-1)
            return F.linear(torch.cat((features,constants),-1),self.packed_output)
        dtype=self.core_output.dtype
        weight=torch.cat((self.native_output.to(dtype),self.core_output),1)
        return F.linear(features.to(dtype),weight,self.output_bias.to(dtype)).to(x.dtype)

    def pack(self):
        if not self.compact or self.packed:return
        dtype=self.native_qkv.dtype
        for name in ("core_qkv","core_shift","core_bias"):
            setattr(self,name,parameter(getattr(self,name).to(dtype)))
        weight=torch.cat((self.native_output,self.core_output.to(dtype)),1)
        # Two BF16 terms retain the fixed FP32 affine offset inside the GEMM's
        # accumulation, avoiding an extra rounded group output before addition.
        extra=(-weight.shape[1])%16
        if extra<2:extra+=16
        bias=self.output_bias.to(arithmetic_dtype(weight));high=bias.to(dtype)
        low=(bias-high.to(bias.dtype)).to(dtype)
        columns=weight.new_zeros(weight.shape[0],extra);columns[:,0]=high;columns[:,1]=low
        self.packed_output=parameter(torch.cat((weight,columns),1))
        constants=weight.new_zeros(extra);constants[:2]=1
        self.register_buffer("output_constants",constants)
        del self.native_output,self.core_output,self.output_bias
        self.packed=True
