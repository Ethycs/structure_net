"""Q4 tile kernel with an FP16 core-output epilogue and small live state."""
import torch
import triton
import triton.language as tl

from experiments.structure_net.llama_q4_kernel_v2 import (
    Tile,OptimizedCore as PreviousCore,decode_half,transient_projection,fused_projection as decode_projection)


@triton.jit
def core_output_mm(A,Q,S,CORE,C,M,N:tl.constexpr,K:tl.constexpr,
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
    core=tl.load(CORE+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&(cols[None,:]<N),other=0)
    value=acc.to(tl.float16).to(tl.float32)+core.to(tl.float32)
    tl.store(C+rows[:,None]*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))


def fused_projection(x,core,tail,tile):
    a=x.reshape(-1,tail.in_features).contiguous();m,k=a.shape;n=tail.out_features
    if m<=16:return decode_projection(x,core,tail,tile)
    core_output=core(a)
    out=torch.empty((m,n),device=x.device,dtype=torch.float16)
    core_output_mm[(triton.cdiv(m,tile.bm)*triton.cdiv(n,tile.bn),)](
        a,tail.packed,tail.scales,core_output,out,m,n,k,tile.bm,tile.bn,tile.bk,tile.group,tile.staged,
        num_warps=tile.warps,num_stages=tile.stages)
    return out.reshape(*x.shape[:-1],n)


class OptimizedCore(PreviousCore):
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
