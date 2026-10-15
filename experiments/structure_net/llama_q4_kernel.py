"""Fused execution of a frozen FP16 core plus packed Q4 residual."""
from dataclasses import dataclass

import torch
from torch import nn
import triton
import triton.language as tl

from experiments.structure_net.llama_fp16_core_q4_tail import CoreResidualLinear


@dataclass(frozen=True)
class Tile:
    bm:int=16
    bn:int=64
    bk:int=64
    split:int=8
    warps:int=4
    stages:int=2
    group:int=8
    staged:bool=False

    def __post_init__(self):
        if self.bm not in (16,32,64,128) or self.bn not in (32,64,128,256) or self.bk not in (64,128):
            raise ValueError("supported power-of-two matrix tiles required")
        if self.split not in (1,4,8) or self.warps not in (4,8) or self.stages not in (2,3) or self.group not in (1,4,8):
            raise ValueError("supported launch/reduction layout required")


def signed_half(nibbles):
    """CPU numerical control for exact int4-to-FP16 bit conversion."""
    return (((nibbles.to(torch.int16)^8)|0x6400).view(torch.float16)-1032).half()


@triton.jit
def decode_half(v):
    bits=((v^8)|0x6400).to(tl.uint16)
    return bits.to(tl.float16,bitcast=True).to(tl.float32)-1032.


@triton.jit
def core_dot(U,L,rows,cols,M,BM:tl.constexpr,BN:tl.constexpr):
    r=tl.arange(0,64)
    u=tl.load(U+rows[:,None]*64+r[None,:],rows[:,None]<M,other=0)
    left=tl.load(L+cols[None,:]*64+r[:,None])
    return tl.dot(u,left,tl.zeros((BM,BN),tl.float32)).to(tl.float16)


@triton.jit
def packed_core_mm(A,Q,S,U,L,C,M,N:tl.constexpr,K:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr,
                   GROUP:tl.constexpr,STAGED:tl.constexpr):
    pid=tl.program_id(0);part=tl.program_id(1)
    nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN)
    first=(pid//(GROUP*nn))*GROUP
    gm=tl.minimum(nm-first,GROUP)
    mi=first+pid%gm;ni=(pid%(GROUP*nn))//gm
    rows=mi*BM+tl.arange(0,BM);cols=ni*BN+tl.arange(0,BN)
    kr=tl.arange(0,BK)
    pairs=tl.arange(0,BK//2)
    groups=tl.arange(0,BK//64)
    acc=tl.zeros((BM,BN),tl.float32)
    for step in range(tl.cdiv(K,BK*SPLIT)):
        start=part+step*SPLIT;k=start*BK+kr
        a=tl.load(A+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        pk=start*(BK//2)+pairs
        p=tl.load(Q+pk[:,None]*N+cols[None,:],(pk[:,None]<K//2)&(cols[None,:]<N),other=0).to(tl.int32)
        v=tl.join(p&15,p>>4).permute(0,2,1).reshape(BK,BN)
        signed=decode_half(v)
        sg=start*(BK//64)+groups
        scales=tl.load(S+sg[:,None]*N+cols[None,:],(sg[:,None]<K//64)&(cols[None,:]<N),other=0).to(tl.float32)
        b=(signed.reshape(BK//64,64,BN)*scales[:,None,:]).reshape(BK,BN).to(tl.float16)
        if STAGED:
            partial=tl.dot(a,b,tl.zeros((BM,BN),tl.float16),out_dtype=tl.float16)
            acc+=partial.to(tl.float32)
        else:
            acc=tl.dot(a,b,acc)
    if SPLIT==1:
        core=core_dot(U,L,rows,cols,M,BM,BN)
        value=acc.to(tl.float16).to(tl.float32)+core.to(tl.float32)
    else:
        value=acc
    tl.store(C+part*M*N+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def reduce_core(P,U,L,C,M,N:tl.constexpr,SPLIT:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    parts=tl.arange(0,SPLIT)
    p=tl.load(P+parts[:,None,None]*M*N+rows[None,:,None]*N+cols[None,None,:],
              (rows[None,:,None]<M)&(cols[None,None,:]<N),other=0)
    tail=tl.sum(p,0).to(tl.float16)
    core=core_dot(U,L,rows,cols,M,BM,BN)
    value=tail.to(tl.float32)+core.to(tl.float32)
    tl.store(C+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def dequant_weight(Q,S,W,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    n=i//K;k=i%K
    p=tl.load(Q+(k//2)*N+n,i<N*K,other=0).to(tl.int32)
    v=(p>>((k%2)*4))&15
    s=tl.load(S+(k//64)*N+n,i<N*K,other=0).to(tl.float32)
    tl.store(W+i,(decode_half(v)*s).to(tl.float16),i<N*K)


def fused_projection(x,core,tail,tile):
    a=x.reshape(-1,tail.in_features).contiguous();m,k=a.shape;n=tail.out_features
    u=torch.nn.functional.linear(a,core.right)
    out=torch.empty((m,n),device=x.device,dtype=torch.float16)
    partial=out if tile.split==1 else torch.empty((tile.split,m,n),device=x.device,dtype=torch.float32)
    packed_core_mm[(triton.cdiv(m,tile.bm)*triton.cdiv(n,tile.bn),tile.split)](
        a,tail.packed,tail.scales,u,core.left,partial,m,n,k,tile.bm,tile.bn,tile.bk,tile.split,tile.group,tile.staged,
        num_warps=tile.warps,num_stages=tile.stages)
    if tile.split>1:
        reduce_core[(triton.cdiv(m,16),triton.cdiv(n,64))](partial,u,core.left,out,m,n,tile.split,16,64,num_warps=4)
    return out.reshape(*x.shape[:-1],n)


def transient_projection(x,core,tail):
    n,k=tail.out_features,tail.in_features
    w=torch.empty((n,k),device=x.device,dtype=torch.float16)
    dequant_weight[(triton.cdiv(n*k,256),)](tail.packed,tail.scales,w,n,k,256)
    return core(x)+torch.nn.functional.linear(x,w)


class OptimizedCore(nn.Module):
    def __init__(self,original,family,decode,prefill):
        super().__init__()
        if not isinstance(original,CoreResidualLinear) or original.core.left.shape[1]!=64 or family not in ("packed","staged","transient"):
            raise ValueError("fixed rank64 trained core and supported execution family required")
        self.core,self.tail=original.core,original.tail
        self.in_features,self.out_features=original.in_features,original.out_features
        self.family,self.decode,self.prefill=family,decode,prefill

    def forward(self,x):
        if x.dtype!=torch.float16 or not x.is_cuda or x.shape[-1]!=self.in_features:
            raise ValueError("CUDA FP16 activation required")
        m=x.numel()//self.in_features
        if m>16 and self.family=="transient":return transient_projection(x,self.core,self.tail)
        return fused_projection(x,self.core,self.tail,self.decode if m<=16 else self.prefill)


def install(model,family,decode,prefill):
    before={n:(x.untyped_storage().data_ptr(),x.dtype,tuple(x.shape)) for n,x in model.state_dict().items()}
    count=0
    for layer in model.model.layers:
        for name in ("gate_proj","up_proj","down_proj"):
            old=getattr(layer.mlp,name)
            setattr(layer.mlp,name,OptimizedCore(old,family,decode,prefill));count+=1
    after={n:(x.untyped_storage().data_ptr(),x.dtype,tuple(x.shape)) for n,x in model.state_dict().items()}
    if before!=after or count!=84:raise RuntimeError("kernel installation changed physical model state")
    return count
