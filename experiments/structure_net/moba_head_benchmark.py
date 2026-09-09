#!/usr/bin/env python3
"""Standalone-head MoBA timing; synthetic systems evidence, never task accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from experiments.structure_net import moba_sparse_kernel as custom

UPSTREAM = Path("data/experiments/moba_paper/upstream")
UPSTREAM_COMMIT = "b5d58363311d3ca946f1ec444182727c15e338b5"
PROTOCOL = Path("docs/07 - Status Reports/2026-09-09_moba-attention-application-protocol.md")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def reference(q, k, v, block, topk):
    """FP32 selected-token softmax; input-dtype mean pooling follows upstream."""
    heads, kvheads, length = q.shape[1], k.shape[1], q.shape[2]
    k = k.repeat_interleave(heads // kvheads, 1)
    v = v.repeat_interleave(heads // kvheads, 1)
    means = torch.stack([part.mean(-2) for part in k.split(block, dim=-2)], dim=-2).float()
    q,k,v = q.float(),k.float(),v.float()
    blocks = torch.arange(means.shape[-2], device=q.device)
    keypos = torch.arange(length, device=q.device)
    output = []
    for start in range(0, length, 128):
        querypos = torch.arange(start,min(start+128,length),device=q.device)
        local = querypos // block
        scores = q[...,start:start+128,:] @ means.transpose(-1,-2)
        eligible = blocks[None,:] < local[:,None]
        remote = scores.masked_fill(~eligible,-torch.inf).argsort(dim=-1,descending=True,stable=True)[...,:topk-1]
        selected = torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,remote,True) & eligible
        selected.scatter_(-1,local[None,None,:,None].expand(*scores.shape[:-1],1),True)
        allowed = selected[...,keypos//block] & (keypos[None,:] <= querypos[:,None])
        logits = q[...,start:start+128,:] @ k.transpose(-1,-2) * (q.shape[-1] ** -.5)
        output.append(logits.masked_fill(~allowed,-torch.inf).softmax(-1) @ v)
    return torch.cat(output,-2)


def official_call(q,k,v,block,topk,cu=None):
    from moba.moba_efficient import moba_attn_varlen
    b,h,n,d = q.shape
    kh = k.shape[1]
    # Upstream needs equal head counts. Expansion is part of the timed call.
    k = k.repeat_interleave(h//kh,1) if kh != h else k
    v = v.repeat_interleave(h//kh,1) if kh != h else v
    if cu is None:
        cu = torch.arange(b+1,device=q.device,dtype=torch.int32) * n
    out = moba_attn_varlen(q.transpose(1,2).reshape(b*n,h,d),k.transpose(1,2).reshape(b*n,h,d),
                          v.transpose(1,2).reshape(b*n,h,d),cu,n,block,topk)
    return out.reshape(b,n,h,d).transpose(1,2)


def flash_call(q,k,v):
    from flash_attn import flash_attn_func
    return flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),causal=True).transpose(1,2)


def sdpa_call(q,k,v):
    # Compatible with the paper's PyTorch 2.4 environment. Charge expansion.
    groups=q.shape[1]//k.shape[1]
    if groups != 1:
        k,v=k.repeat_interleave(groups,1),v.repeat_interleave(groups,1)
    return F.scaled_dot_product_attention(q,k,v,is_causal=True)


def timed(fn, repeats):
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        out = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def paired_times(functions, samples=9, repeats=20):
    for fn in functions.values():
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    collected = {name: [] for name in functions}
    names = list(functions)
    for sample in range(samples):
        order = names if sample % 2 == 0 else names[::-1]
        for name in order:
            collected[name].append(timed(functions[name],repeats))
    return {name:{"samples_ms":values,"median_ms":statistics.median(values),"min_ms":min(values),"max_ms":max(values)}
            for name,values in collected.items()}


def run(output, official=False, shakedown=False):
    if output.exists():
        raise RuntimeError("preserve existing benchmark directory; choose a new root")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA benchmark required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    files = [Path(__file__),Path(custom.__file__),PROTOCOL]
    if official:
        actual = subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"],text=True).strip()
        if actual != UPSTREAM_COMMIT:
            raise RuntimeError("upstream commit changed")
        sys.path.insert(0,str(UPSTREAM.resolve()))
        files += sorted((UPSTREAM/"moba").glob("*.py"))
    producer = {"source":{str(p):sha256(p) for p in files},"torch":str(torch.__version__),"cuda":torch.version.cuda,
                "gpu":torch.cuda.get_device_name(),"device_capability":torch.cuda.get_device_capability(),
                "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"tf32":False,
                "upstream_commit":UPSTREAM_COMMIT if official else None,
                "configuration":{"samples":9,"repeats":20,"warmups":5,"official":official,"shakedown":shakedown,
                                 "layout":"BTHD allocation, BHTD view; native head input, no model weights"}}
    producer["fingerprint"] = hashlib.sha256(json.dumps(producer,sort_keys=True).encode()).hexdigest()
    for p in files:
        target = output/"producer_source"/p.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    write_json(output/"producer.json",producer)
    shapes = [("tiny256",1,8,8,256,64,32,2), ("heads2048",1,8,8,2048,64,128,2),
              ("heads8192",1,8,8,8192,64,128,2), ("qwen3_8b_shape8192",1,32,8,8192,128,128,2)]
    if shakedown:
        shapes = shapes[:1]
    cells = []
    with torch.inference_mode():
        for label,b,h,kh,n,d,block,topk in shapes:
            for dtype in (torch.float32,torch.float16):
                torch.manual_seed(951)
                q = torch.randn(b,n,h,d,device="cuda",dtype=dtype).transpose(1,2)
                k,v = [torch.randn(b,n,kh,d,device="cuda",dtype=dtype).transpose(1,2) for _ in range(2)]
                cell = {"label":label,"shape":[b,h,kh,n,d],"dtype":str(dtype),"block_size":block,"top_k":topk,
                        "application_accuracy":None,"application_accuracy_status":"application_contract_missing",
                        "functions":{},"failures":[]}
                cu = torch.arange(b+1,device=q.device,dtype=torch.int32) * n
                functions = {"full_sdpa":lambda:sdpa_call(q,k,v),
                             "experimental_sparse":lambda:custom.sparse_attention(q,k,v,block)}
                if official and dtype == torch.float16:
                    functions.update(full_flash=lambda:flash_call(q,k,v),paper_moba=lambda:official_call(q,k,v,block,topk,cu))
                ref = reference(q,k,v,block,topk)
                valid = {}
                for name,fn in functions.items():
                    try:
                        actual = fn()
                        if not torch.isfinite(actual).all():
                            raise RuntimeError("nonfinite output")
                        if name in ("experimental_sparse","paper_moba"):
                            difference = actual.float()-ref
                            error = {"max_abs_error":float(difference.abs().max()),"mean_abs_error":float(difference.abs().mean()),
                                     "relative_l2_error":float(difference.norm()/ref.norm())}
                            cell["functions"][name] = error
                            tolerance = 1e-5 if dtype == torch.float32 else .02
                            if not torch.allclose(actual.float(),ref,atol=tolerance,rtol=tolerance):
                                raise RuntimeError("selected-softmax parity failed")
                        valid[name] = fn
                    except Exception as exc:
                        cell["failures"].append({"backend":name,"error":repr(exc)})
                if not any(key in valid for key in ("full_sdpa","full_flash")):
                    cell.update(timing={},speedup_vs_fastest_full={},status="baseline_unavailable_no_speed_claim")
                else:
                    cell["timing"] = paired_times(valid)
                    full = min(cell["timing"][key]["median_ms"] for key in ("full_sdpa","full_flash") if key in valid)
                    cell["speedup_vs_fastest_full"] = {key:full/value["median_ms"] for key,value in cell["timing"].items() if key not in ("full_sdpa","full_flash")}
                    cell["status"] = "completed_with_backend_failures" if cell["failures"] else "completed"
                cells.append(cell)
                write_json(output/f"{label}_{str(dtype).split('.')[-1]}.json",cell)
                print(json.dumps({"label":label,"dtype":str(dtype),"speedup":cell["speedup_vs_fastest_full"],"failures":cell["failures"]}),flush=True)
                del q,k,v,ref
                torch.cuda.empty_cache()
    result = {"schema":"nal.moba-standalone-head-benchmark.v1","status":"completed",
              "classification":"synthetic_kernel_evidence_application_unverified","producer_fingerprint":producer["fingerprint"],
              "profile":"systems_lifecycle_only_not_quality_evidence","cells":cells,
              "application_accuracy_tested":False,"timestamp":time.time()}
    write_json(output/"campaign_results.json",result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--official",action="store_true")
    parser.add_argument("--shakedown",action="store_true")
    args = parser.parse_args()
    run(args.output,args.official,args.shakedown)
