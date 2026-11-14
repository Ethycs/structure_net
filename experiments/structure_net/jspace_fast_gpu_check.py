"""Reserved GPU parity/graph shakedown for the new compact Triton kernels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from .jspace_bench_data import sha, write
from .jspace_full_lease import wait_ready
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_affine_fast import FastAffineQwenBlock, affine_qkv_post, gate_swiglu_features


def fixture(polynomial):
    torch.manual_seed(201)
    config=Qwen2Config(hidden_size=64,intermediate_size=256,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=1,vocab_size=64,attention_dropout=0.,max_position_embeddings=128)
    config._attn_implementation="eager"
    native=Qwen2DecoderLayer(config,layer_idx=0).float().cuda().eval().requires_grad_(False)
    basis=torch.linalg.qr(torch.randn(64,64,device="cuda:0"))[0][:,:32]
    block=AffineQwenBlock(native,basis,torch.randn(64,64,device="cuda:0")*.1)
    if polynomial:
        anchors=[0,2,4,6]
        fit={"anchors":anchors,"mean":torch.zeros(256),"scale":torch.ones(256),
             "coefficients":{"2":torch.randn(256,15)*.01}}
        block.compile_polynomial(fit,list(range(16,48)))
    return block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16).eval()


def metric(actual,expected):
    difference=actual.float()-expected.float()
    if not torch.isfinite(actual).all():raise FloatingPointError("Nonfinite Triton output")
    return {"exact":bool(torch.equal(actual,expected)),"max_absolute":float(difference.abs().max()),
            "relative_rmse":float(difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-20))}


@torch.no_grad()
def run(output):
    lease=wait_ready([0]);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(201)
    torch.cuda.reset_peak_memory_stats();results=[]
    for polynomial in (False,True):
        block=fixture(polynomial);fast=FastAffineQwenBlock.from_packed(deepcopy(block),pad_to_multiple=128).eval()
        for length in (1,8,32):
            ids=torch.arange(length,device="cuda:0")[None]
            denominator=torch.rand(1,length,1,device="cuda:0")+.3
            raw=torch.randn(1,length,block.qkv_weight.shape[0],device="cuda:0",dtype=torch.bfloat16)
            expected=affine_qkv_post(raw,denominator,block,ids,use_triton=False)
            actual=affine_qkv_post(raw,denominator,block,ids,use_triton=True)
            qkv=metric(actual,expected)
            raw=torch.randn(1,length,2*block.intermediate_size,device="cuda:0",dtype=torch.bfloat16)*3
            kwargs={"feature_i":fast.feature_i_0,"feature_j":fast.feature_j_0,"padded_width":fast.original.mlp_output.shape[1]}
            expected=gate_swiglu_features(raw,denominator,block,ids,use_triton=False,**kwargs)
            actual=gate_swiglu_features(raw,denominator,block,ids,use_triton=True,**kwargs)
            features=metric(actual,expected)
            if not qkv["exact"] or features["relative_rmse"]>1e-3:raise ValueError("Fused arithmetic parity failed")
            for geometry in ("fp32","bf16"):
                reference=FastAffineQwenBlock.from_packed(deepcopy(block),geometry_mode=geometry,pad_to_multiple=128,enabled=False).eval()
                candidate=FastAffineQwenBlock.from_packed(deepcopy(block),geometry_mode=geometry,pad_to_multiple=128,enabled=True).eval()
                x=torch.randn(1,length,64,device="cuda:0",dtype=torch.bfloat16)
                angle=ids[:,:,None].float()*torch.arange(1,9,device="cuda:0")[None,None,:]/100
                angle=torch.cat((angle,angle),-1);position_embeddings=(angle.cos().bfloat16(),angle.sin().bfloat16())
                if not candidate._can_fuse(x):raise ValueError("Triton path was silently bypassed")
                def forward():return candidate(x,position_embeddings=position_embeddings,position_ids=ids)[0]
                expected=reference(x,position_embeddings=position_embeddings,position_ids=ids)[0]
                actual=forward();whole=metric(actual,expected)
                if whole["relative_rmse"]>2e-3:raise ValueError("Whole-block reference parity failed")
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=forward()
                graph.replay();torch.cuda.synchronize()
                if not torch.equal(captured,actual):raise ValueError("CUDA graph changed fused output")
                results.append({"polynomial":polynomial,"length":length,"geometry":geometry,"qkv":qkv,"features":features,
                                "whole_block":whole,"graph_exact":True})
                del graph,candidate,reference
    sources=[Path(__file__),Path(__file__).with_name("jspace_qwen_affine_fast.py"),
             Path(__file__).with_name("jspace_qwen_affine_block.py"),Path(__file__).with_name("jspace_qwen_affine_stack.py")]
    peak=torch.cuda.max_memory_reserved()
    if peak>sum(lease["allocation"].values())*1024**2:raise RuntimeError("Reservation exceeded")
    value={"status":"completed","systems_only":True,"lease_id":lease["id"],"gpu":torch.cuda.get_device_name(0),
           "peak_reserved_bytes":peak,"results":results,"source_sha256":{str(p):sha(p) for p in sources}}
    for p in sources:(output/p.name).write_bytes(p.read_bytes())
    write(output/"result.json",value);return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    print(json.dumps(run(args.output),allow_nan=False),flush=True)
