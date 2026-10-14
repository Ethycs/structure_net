"""Trained FP16 factor core with a packed uniform-Q4 residual, not random noise."""
import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import nn
import triton
import triton.language as tl

from experiments.structure_net.llama_neighbourhood_tensorization import FactorLinear, digest
from experiments.structure_net.llama_noise_tensor_continuation import assembled, FIXTURES
from experiments.structure_net.llama_tensor_runtime import CHECKPOINT, TensorRuntime
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json

HYPOTHESIS_ID = "llama32-3b-fp16-core-q4-tail-v1"
SCHEMA_VERSION = "nal.llama-fp16-core-q4-tail.v1"
PLAN = Path("docs/07 - Status Reports/2026-10-14_llama32-fp16-core-q4-tail-plan.md")


@dataclass(frozen=True)
class Config:
    rank: int = 64
    group_size: int = 64
    bits: int = 4

    def __post_init__(self):
        if self.rank not in (64, 128) or self.group_size != 64 or self.bits != 4:
            raise ValueError("fixed core ranks and Q4/64 quantizer required")


def pack_signed(q):
    if q.dtype != torch.int8 or q.ndim != 2 or q.shape[1] % 2:
        raise ValueError("signed matrix with even input dimension required")
    if (q < -8).any() or (q > 7).any():
        raise ValueError("signed nibble range required")
    return ((q[:, 0::2].to(torch.int16) & 15) | ((q[:, 1::2].to(torch.int16) & 15) << 4)).byte().T.contiguous()


def unpack_signed(packed):
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed byte matrix required")
    p = packed.T.to(torch.int16)
    result = torch.empty((p.shape[0], 2*p.shape[1]), dtype=torch.int8, device=p.device)
    result[:, 0::2] = ((p & 15) ^ 8)-8
    result[:, 1::2] = (((p >> 4) & 15) ^ 8)-8
    return result


@torch.no_grad()
def quantize(weight, group=64):
    if weight.ndim != 2 or weight.shape[1] % group or group != 64 or not torch.isfinite(weight).all():
        raise ValueError("finite matrix and aligned Q4 group required")
    w = weight.float().reshape(weight.shape[0], -1, group)
    # Store and use the same rounded scale; nonzero floor is representable in FP16.
    scale = (w.abs().amax(-1)/7).clamp_min(2**-24).half()
    q = torch.round(w/scale.float()[..., None]).clamp(-7, 7).to(torch.int8).reshape(weight.shape)
    return pack_signed(q), scale.T.contiguous()


def dequantize(packed, scales, group=64):
    q = unpack_signed(packed).float()
    if scales.shape != (q.shape[1]//group, q.shape[0]) or scales.dtype != torch.float16:
        raise ValueError("scale shape and FP16 storage mismatch")
    return (q.reshape(q.shape[0], -1, group)*scales.T.float()[..., None]).reshape(q.shape).half()


@triton.jit
def q4_mm(A, Q, S, C, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
          BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr, SPLIT:tl.constexpr=1):
    rows = tl.program_id(0)*BM+tl.arange(0,BM)
    cols = tl.program_id(1)*BN+tl.arange(0,BN)
    kk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN),tl.float32)
    part = tl.program_id(2)
    for step in range(tl.cdiv(K,BK*SPLIT)):
        start = part+step*SPLIT
        k = start*BK+kk
        a = tl.load(A+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        p = tl.load(Q+(k[:,None]//2)*N+cols[None,:],(k[:,None]<K)&(cols[None,:]<N),other=0).to(tl.int32)
        nibble = (p >> ((k[:,None]%2)*4)) & 15
        signed = ((nibble ^ 8)-8).to(tl.float32)
        # BK divides the quantizer group: one vector load per tile, not BK copies.
        scale = tl.load(S+(start*BK//64)*N+cols,(start*BK<K)&(cols<N),other=0).to(tl.float32)
        b = (signed*scale[None,:]).to(tl.float16)
        acc = tl.dot(a,b,acc)
    tl.store(C+part*M*N+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


def packed_linear(x, packed, scales, tuning):
    bm,bn,bk,split,warps,stages=tuning
    if bk not in (32,64):raise ValueError("K tile must divide the 64-channel group")
    m,k=x.shape;n=packed.shape[1]
    out=torch.empty((split,m,n),device=x.device,dtype=torch.float32 if split>1 else torch.float16)
    q4_mm[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
        x,packed,scales,out,m,n,k,bm,bn,bk,split,num_warps=warps,num_stages=stages)
    return out.sum(0).half() if split>1 else out[0]


class Q4Linear(nn.Module):
    def __init__(self, packed, scales):
        super().__init__()
        if packed.ndim != 2 or packed.dtype != torch.uint8 or packed.shape[0]*2%64 or scales.dtype != torch.float16 or scales.shape != (packed.shape[0]*2//64,packed.shape[1]):
            raise ValueError("valid packed Q4 and scale buffers required")
        self.register_buffer("packed",packed)
        self.register_buffer("scales",scales)
        self.in_features, self.out_features = packed.shape[0]*2, packed.shape[1]

    def forward(self,x):
        if not x.is_cuda or x.dtype != torch.float16 or x.shape[-1] != self.in_features:
            raise ValueError("CUDA FP16 activation with matching input dimension required")
        a = x.reshape(-1,self.in_features).contiguous()
        # Fixed by the pre-quality operator tuning in operator_tuning_v2.
        tuning=(16,64,64,8,4,2) if a.shape[0]<=16 else (128,128,32,1,8,3)
        out=packed_linear(a,self.packed,self.scales,tuning)
        return out.reshape(*x.shape[:-1],self.out_features)


class CoreResidualLinear(nn.Module):
    def __init__(self,left,right,packed,scales):
        super().__init__()
        self.core = FactorLinear(left,right).requires_grad_(False)
        self.tail = Q4Linear(packed,scales)
        self.in_features, self.out_features = right.shape[1], left.shape[0]
        if (self.tail.in_features,self.tail.out_features)!=(self.in_features,self.out_features):
            raise ValueError("core and residual dimensions differ")

    def forward(self,x):
        return self.core(x)+self.tail(x)


@torch.no_grad()
def replace(model,config):
    records=[]
    torch.backends.cuda.matmul.allow_tf32=False
    for layer,block in enumerate(model.model.layers):
        for name in ("gate_proj","up_proj","down_proj"):
            old=getattr(block.mlp,name)
            if not isinstance(old,FactorLinear) or old.left.shape[1]!=1024:
                raise ValueError("frozen trained rank1024 factors required")
            tail=old.left[:,config.rank:].float()@old.right[config.rank:,:].float()
            packed,scales=quantize(tail,config.group_size)
            relative=float((dequantize(packed,scales).float()-tail).square().mean()/tail.square().mean().clamp_min(1e-12))
            new=CoreResidualLinear(old.left[:,:config.rank].contiguous(),old.right[:config.rank,:].contiguous(),packed,scales)
            setattr(block.mlp,name,new)
            records.append({"layer":layer,"projection":name,"tail_relative_weight_mse":relative,
                "core_rank":config.rank,"packed_bytes":packed.numel(),"scale_bytes":scales.numel()*2})
            del tail,old
    gc.collect()
    torch.cuda.empty_cache()
    return records


def unique_bytes(model):
    seen=set();total=0
    for tensor in list(model.parameters())+list(model.buffers()):
        key=(tensor.device,tensor.untyped_storage().data_ptr())
        if key not in seen:
            seen.add(key);total+=tensor.untyped_storage().nbytes()
    return total


def seal(output,stage,config,extra=()):
    if output.exists():raise ValueError("new evidence root required")
    output.mkdir(parents=True)
    paths=[Path(__file__),PLAN,Path("experiments/structure_net/llama_tensor_runtime.py"),
        Path("experiments/structure_net/llama_neighbourhood_tensorization.py"),
        Path("experiments/structure_net/llama_noise_tensor_continuation.py"),
        Path("experiments/structure_net/prepare_llama_cotenancy.py"),CHECKPOINT/"training.json",FIXTURES]+list(extra)
    hashes={str(p):sha256(p) for p in paths}
    snapshots=output/"producer_source";snapshots.mkdir()
    for i,p in enumerate(paths):shutil.copyfile(p,snapshots/f"{i:03d}_{p.name}")
    record={"schema":SCHEMA_VERSION,"hypothesis_id":HYPOTHESIS_ID,"stage":stage,
        "config":config,"source_hashes":hashes,"command":sys.argv,"torch":torch.__version__,
        "triton":triton.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
        "dense_tail_cached":False,"cross_request_kv_reuse":False}
    record["fingerprint"]=digest(record);write_json(output/"producer.json",record)
    return record


@torch.inference_mode()
def operator_pilot(output):
    producer=seal(output,"systems_operator_only_not_quality_evidence",{"group":64,"bits":4,"seed":51014})
    torch.manual_seed(51014)
    rows=[]
    for n,k in ((8192,3072),(3072,8192),(1024,3072)):
        weight=torch.randn((n,k),device="cuda",dtype=torch.float16)*.03
        packed,scales=quantize(weight)
        module=Q4Linear(packed,scales)
        restored=dequantize(packed,scales)
        for m in (1,9,512):
            x=torch.randn((1,m,k),device="cuda",dtype=torch.float16)
            expected=torch.nn.functional.linear(x,restored)
            actual=module(x)
            torch.cuda.synchronize()
            relative=float((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt())
            cosine=float(torch.nn.functional.cosine_similarity(actual.float().flatten(),expected.float().flatten(),dim=0))
            if not torch.isfinite(actual).all() or relative>.002 or cosine<.9999:raise RuntimeError("packed operator numerical gate failed")
            times={}
            for label,fn in (("q4",lambda:module(x)),("fp16",lambda:torch.nn.functional.linear(x,restored))):
                for _ in range(3):fn()
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(10):fn()
                torch.cuda.synchronize();times[label]=(time.perf_counter()-start)/10
            rows.append({"m":m,"n":n,"k":k,"relative_rms":relative,"cosine":cosine,"seconds":times})
            print(json.dumps(rows[-1]),flush=True)
        del module,restored,weight,packed,scales,x,expected,actual
    write_json(output/"result.json",{"status":"passed","label":"systems_operator_only_not_quality_evidence",
        "rows":rows,"producer_fingerprint":producer["fingerprint"],"peak_cuda_bytes":torch.cuda.max_memory_allocated()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("operator-pilot",))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one selected CUDA GPU required")
    operator_pilot(args.output)


if __name__=="__main__":main()
