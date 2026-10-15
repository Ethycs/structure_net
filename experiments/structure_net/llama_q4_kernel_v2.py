"""Direct Q4 loads with fused epilogues and coalesced transient expansion."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from experiments.structure_net.llama_q4_kernel import (
    Tile as FirstTile,OptimizedCore as FirstCore,decode_half,core_dot,reduce_core)
from experiments.structure_net.llama_fp16_core_q4_tail import q4_mm


@dataclass(frozen=True)
class Tile(FirstTile):
    def __post_init__(self):
        if self.bk==32:
            FirstTile(self.bm,self.bn,64,self.split,self.warps,self.stages,self.group,self.staged)
        else:super().__post_init__()


@triton.jit
def direct_core_mm(A,Q,S,U,L,C,M,N:tl.constexpr,K:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                   GROUP:tl.constexpr,STAGED:tl.constexpr):
    pid=tl.program_id(0);nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN)
    if GROUP==1:
        mi=pid//nn;ni=pid%nn
    else:
        first=pid//(GROUP*nn)*GROUP;gm=tl.minimum(nm-first,GROUP)
        mi=first+pid%gm;ni=pid%(GROUP*nn)//gm
    rows=mi*BM+tl.arange(0,BM);cols=ni*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
    for step in range(tl.cdiv(K,BK)):
        k=step*BK+kk
        a=tl.load(A+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        p=tl.load(Q+(k[:,None]//2)*N+cols[None,:],(k[:,None]<K)&(cols[None,:]<N),other=0).to(tl.int32)
        v=(p>>((k[:,None]%2)*4))&15
        if BK<=64:
            s=tl.load(S+(step*BK//64)*N+cols,cols<N,other=0).to(tl.float32)
            b=(decode_half(v)*s[None,:]).to(tl.float16)
        else:
            s=tl.load(S+(k[:,None]//64)*N+cols[None,:],(k[:,None]<K)&(cols[None,:]<N),other=0).to(tl.float32)
            b=(decode_half(v)*s).to(tl.float16)
        if STAGED:
            partial=tl.dot(a,b,tl.zeros((BM,BN),tl.float16),out_dtype=tl.float16)
            acc+=partial.to(tl.float32)
        else:acc=tl.dot(a,b,acc)
    core=core_dot(U,L,rows,cols,M,BM,BN)
    value=acc.to(tl.float16).to(tl.float32)+core.to(tl.float32)
    tl.store(C+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def coalesced_dequant(Q,S,W,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    k=i//N;n=i%N
    p=tl.load(Q+(k//2)*N+n,i<N*K,other=0).to(tl.int32)
    v=(p>>((k%2)*4))&15
    s=tl.load(S+(k//64)*N+n,i<N*K,other=0).to(tl.float32)
    tl.store(W+i,(decode_half(v)*s).to(tl.float16),i<N*K)


def fused_projection(x,core,tail,tile):
    a=x.reshape(-1,tail.in_features).contiguous();m,k=a.shape;n=tail.out_features
    u=torch.nn.functional.linear(a,core.right)
    out=torch.empty((m,n),device=x.device,dtype=torch.float16)
    if m<=16:
        partial=torch.empty((tile.split,m,n),device=x.device,dtype=torch.float32)
        # Preserve the old fast decode load layout and FP32 accumulation.
        q4_mm[(triton.cdiv(m,tile.bm),triton.cdiv(n,tile.bn),tile.split)](
            a,tail.packed,tail.scales,partial,m,n,k,tile.bm,tile.bn,tile.bk,tile.split,
            num_warps=tile.warps,num_stages=tile.stages)
        reduce_core[(triton.cdiv(m,16),triton.cdiv(n,64))](partial,u,core.left,out,m,n,tile.split,16,64,num_warps=4)
    else:
        direct_core_mm[(triton.cdiv(m,tile.bm)*triton.cdiv(n,tile.bn),)](
            a,tail.packed,tail.scales,u,core.left,out,m,n,k,tile.bm,tile.bn,tile.bk,tile.group,tile.staged,
            num_warps=tile.warps,num_stages=tile.stages)
    return out.reshape(*x.shape[:-1],n)


def transient_projection(x,core,tail):
    n,k=tail.out_features,tail.in_features
    w=torch.empty((k,n),device=x.device,dtype=torch.float16)
    coalesced_dequant[(triton.cdiv(n*k,256),)](tail.packed,tail.scales,w,n,k,256)
    return core(x)+torch.nn.functional.linear(x,w.T)


class OptimizedCore(FirstCore):
    def forward(self,x):
        if x.dtype!=torch.float16 or not x.is_cuda or x.shape[-1]!=self.in_features:
            raise ValueError("CUDA FP16 activation required")
        m=x.numel()//self.in_features
        if m<=16 and self.decode is None:return self.core(x)+self.tail(x)
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
