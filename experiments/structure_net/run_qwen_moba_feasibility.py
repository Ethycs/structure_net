#!/usr/bin/env python3
"""Measured public-Qwen prefix feasibility using the paper's optimized MoBA."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net import moba_head_benchmark as bench

PROTOCOL=Path("docs/07 - Status Reports/2026-09-09_qwen8b-moba-feasibility.md")
INPUT_ROOT=Path("data/experiments/qwen_moba_prefix/inputs_v2")
SOURCES={name:Path("data/corpora/babylm_10M")/(name+".txt") for name in ("wikipedia","bnc_spoken")}
SEEDS=(7,17,29,41,53)
LENGTHS=(512,2048,8192,16384,32768)


def prepare_inputs():
    tokenizer=Tokenizer.from_file(str(q.MODEL_ROOT/"tokenizer.json"))
    INPUT_ROOT.mkdir(parents=True,exist_ok=True)
    metadata={"tokenizer_sha256":q.sha256(q.MODEL_ROOT/"tokenizer.json"),"source_sha256":{k:q.sha256(v) for k,v in SOURCES.items()}}
    path=INPUT_ROOT/"metadata.json"
    if path.exists():
        existing=json.loads(path.read_text())
        if any(existing[k]!=v for k,v in metadata.items()):
            raise RuntimeError("token inputs changed; choose a new input root")
        for source,artifact in existing["tokens"].items():
            if q.sha256(artifact["path"])!=artifact["sha256"]:
                raise RuntimeError("cached tokens changed")
        return existing
    metadata["tokens"]={}
    for name,source in SOURCES.items():
        # Encode independent text chunks to avoid retaining a huge Encoding object.
        text=source.read_text()
        encoded=[]
        for start in range(0,len(text),100000):
            encoded.extend(tokenizer.encode(text[start:start+100000],add_special_tokens=False).ids)
        tokens=np.asarray(encoded,dtype=np.int32)
        token_path=INPUT_ROOT/(name+".npy")
        np.save(token_path,tokens)
        metadata["tokens"][name]={"path":str(token_path),"sha256":q.sha256(token_path),"count":len(tokens)}
    bench.write_json(path,metadata)
    return metadata


def window_schedules(inputs):
    schedules={}
    for name,artifact in inputs["tokens"].items():
        count=artifact["count"]
        train_end=int(count*.6)
        validation_end=int(count*.8)
        validation=int(np.random.default_rng((7,1961)).integers(train_end,validation_end-max(LENGTHS)+1))
        bins=np.arange(validation_end,count-max(LENGTHS)+1,max(LENGTHS))
        if len(bins)<len(SEEDS):
            raise RuntimeError(f"not enough disjoint final windows for {name}")
        final=np.random.default_rng((1962,7)).choice(bins,len(SEEDS),replace=False)
        schedules[name]={"train_end":train_end,"validation_end":validation_end,"validation_start":validation,
                         "final_starts":{str(seed):int(start) for seed,start in zip(SEEDS,final)}}
    return schedules


def producer(output,inputs,schedules,stage,head_chunk_size,selection_artifact=None,memory_fraction=1.0):
    import flash_attn,transformers
    upstream=subprocess.check_output(["git","-C",str(bench.UPSTREAM),"rev-parse","HEAD"],text=True).strip()
    if upstream!=bench.UPSTREAM_COMMIT:
        raise RuntimeError("paper source changed")
    paths=[Path(__file__),Path(q.__file__),Path(bench.__file__),Path(bench.custom.__file__),PROTOCOL]
    paths+=sorted((bench.UPSTREAM/"moba").glob("*.py"))
    package=Path(transformers.__file__).parent
    external=[package/"models/qwen3/modeling_qwen3.py",package/"models/qwen3/configuration_qwen3.py",package/"modeling_rope_utils.py"]
    sources={str(p.resolve()):q.sha256(p) for p in paths+external}
    value={"schema":"nal.qwen3-8b-prefix6-moba-feasibility.v1","hypothesis_id":"qwen3-8b-prefix6-moba-feasibility-v1",
           "stage":stage,"sources":sources,"inputs":inputs,"schedules":schedules,
           "model":{"revision":q.MODEL_REVISION,"hashes":json.loads((q.MODEL_ROOT/"download_hashes.json").read_text())},
           "settings":[asdict(s) for s in q.SETTINGS],"lengths":list(LENGTHS),
           "execution":{"mlp_chunk_tokens":2048,"moba_head_chunk_size":head_chunk_size,"memory_fraction":memory_fraction},"selection_artifact":selection_artifact,
           "environment":{"torch":str(torch.__version__),"flash_attn":flash_attn.__version__,"transformers":transformers.__version__,
                          "numpy":np.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                          "visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"dtype":"float16","pooled_dtype":"float32","tf32":False},
           "upstream_commit":upstream,"command":sys.argv}
    import hashlib
    value["fingerprint"]=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    for index,p in enumerate(paths+external):
        target=output/"producer_source"/f"{index:02d}_{p.name}"
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
    bench.write_json(output/"producer.json",value)
    return value


def hf_parity(model,ids):
    old=model.config._attn_implementation
    model.config._attn_implementation="sdpa"
    with torch.inference_mode():
        actual=model(ids,q.FULL,capture=True)
        x=model.embed_tokens(ids)
        positions=torch.arange(ids.shape[1],device=ids.device)[None,:]
        rope=model.rotary_emb(x,positions)
        for layer in model.layers[:5]:
            x=layer(x,attention_mask=None,position_ids=positions,position_embeddings=rope)[0]
        projection=model.layers[5].self_attn(model.layers[5].input_layernorm(x),position_embeddings=rope,attention_mask=None)[0]
        target=projection.float().reshape(1,ids.shape[1]//128,128,-1).mean(2)
        torch.testing.assert_close(actual["hidden_before_final"],x,atol=.03,rtol=.03)
        torch.testing.assert_close(actual["ov"],target,atol=.03,rtol=.03)
        result={"passed":True,"hidden_max_error":float((actual["hidden_before_final"]-x).abs().max()),
                "ov_max_error":float((actual["ov"]-target).abs().max()),"atol":.03,"rtol":.03}
    model.config._attn_implementation=old
    return result


def evaluate_cell(model,ids,setting,baseline,samples=5,measure_timing=True):
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        actual=model(ids,setting)
        metrics=q.readout_metrics(actual,baseline)
        timing=bench.paired_times({"full":lambda:model(ids,q.FULL),"moba":lambda:model(ids,setting)},samples=samples,repeats=1) if measure_timing else None
        speedup=timing["full"]["median_ms"]/timing["moba"]["median_ms"] if timing else None
        active=any(setting.enabled(i,ids.shape[1]) for i in range(6))
    return {"status":"completed","setting":asdict(setting),"name":setting.name,"length":ids.shape[1],
            "metrics":metrics,"timing":timing,"prefix_speedup":speedup,"sparse_active":active,
            "gate":q.feasibility_gate(metrics,speedup,active) if measure_timing else None,
            "timing_status":"measured" if measure_timing else "deferred_gpu_contention",
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
            "application_accuracy":None,"application_accuracy_status":"not_measured_by_this_feasibility_proxy"}


def select_candidate(cells):
    for length in LENGTHS:
        passing=[]
        for setting in q.SETTINGS:
            rows=[r for r in cells if r["name"]==setting.name and r["length"]==length and r["status"]=="completed"]
            if len(rows)==len(SOURCES) and all(r["gate"] for r in rows):
                passing.append((statistics.mean(r["prefix_speedup"] for r in rows),
                                statistics.mean(r["metrics"]["change_pearson"] for r in rows),setting))
        # Prefer the shortest demonstrated useful context, then speed/fidelity.
        passing.sort(key=lambda r:(r[0],r[1]),reverse=True)
        if passing:
            return {"setting":asdict(passing[0][2]),"length":length}
    return None


def run(output,stage="screen",selection_path=None,head_chunk_size=32,memory_fraction=1.0):
    if output.exists():
        raise RuntimeError("preserve existing root; choose a new artifact directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not 0 < memory_fraction <= 1:
        raise ValueError("invalid per-process memory fraction")
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    sys.path.insert(0,str(bench.UPSTREAM.resolve()))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    inputs=prepare_inputs()
    schedules=window_schedules(inputs)
    selection_artifact={"path":str(selection_path),"sha256":q.sha256(selection_path),"contents":json.loads(selection_path.read_text())} if selection_path else None
    producing=producer(output,inputs,schedules,stage,head_chunk_size,selection_artifact,memory_fraction)
    tokens={name:np.load(artifact["path"],mmap_mode="r") for name,artifact in inputs["tokens"].items()}
    model=q.load_prefix()
    model.moba_head_chunk_size=head_chunk_size
    original=q.frozen_hash(model)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("zero-shot study must not update any weights")
    first_source=next(iter(SOURCES))
    start=schedules[first_source]["validation_start"]
    ids=torch.tensor(tokens[first_source][start:start+128].copy(),device="cuda",dtype=torch.long)[None,:]
    validity=hf_parity(model,ids)
    bench.write_json(output/"hf_prefix_parity.json",validity)
    cells=[]
    settings=q.SETTINGS
    lengths=LENGTHS
    sources=list(SOURCES)
    seeds=(7,)
    if stage=="pilot":
        settings=(q.MoBASetting(),)
        lengths=(512,max(LENGTHS))
        sources=sources[:1]
    elif stage=="final":
        if selection_path is None:
            raise ValueError("final evaluation requires a frozen screen selection")
        screen=selection_artifact["contents"]
        if screen["classification"]!="zero_shot_candidate_selected" or not screen["selection"]:
            raise RuntimeError("no eligible sealed selection")
        settings=(q.MoBASetting(**screen["selection"]["setting"]),)
        lengths=(screen["selection"]["length"],)
        seeds=SEEDS
    for source in sources:
        for seed in seeds:
            start=schedules[source]["final_starts"][str(seed)] if stage=="final" else schedules[source]["validation_start"]
            for length in lengths:
                ids=torch.tensor(tokens[source][start:start+length].copy(),device="cuda",dtype=torch.long)[None,:]
                try:
                    with torch.inference_mode():
                        baseline=model(ids,q.FULL)
                except torch.cuda.OutOfMemoryError as exc:
                    for setting in settings:
                        result={"status":"baseline_out_of_memory","name":setting.name,"setting":asdict(setting),"length":length,
                                "error":str(exc),"gate":False,"source":source,"seed":seed,"start":int(start)}
                        cells.append(result)
                        bench.write_json(output/"cells"/f"{source}_seed{seed}_n{length}_{setting.name}.json",result)
                    del ids
                    torch.cuda.empty_cache()
                    print(json.dumps({"source":source,"length":length,"status":"baseline_out_of_memory"}),flush=True)
                    continue
                for setting in settings:
                    try:
                        result=evaluate_cell(model,ids,setting,baseline,samples=5,measure_timing=stage!="readouts")
                    except torch.cuda.OutOfMemoryError as exc:
                        result={"status":"out_of_memory","name":setting.name,"setting":asdict(setting),"length":length,"error":str(exc),"gate":False}
                        torch.cuda.empty_cache()
                    result.update(source=source,seed=seed,start=int(start))
                    cells.append(result)
                    bench.write_json(output/"cells"/f"{source}_seed{seed}_n{length}_{setting.name}.json",result)
                    print(json.dumps({k:result.get(k) for k in ("source","seed","length","name","status","prefix_speedup","gate")}),flush=True)
                del baseline,ids
                torch.cuda.empty_cache()
    after=q.frozen_hash(model)
    if original!=after:
        raise RuntimeError("frozen Qwen weights changed")
    selection=select_candidate(cells) if stage=="screen" else None
    classification="readout_screen_completed_timing_pending" if stage=="readouts" else ("systems_lifecycle_only_not_quality_evidence" if stage=="pilot" else (
        ("zero_shot_candidate_selected" if selection else "no_zero_shot_candidate_passed") if stage=="screen" else
        ("prefix_feasibility_supported" if all(r["gate"] for r in cells) else "prefix_feasibility_not_confirmed")))
    result={"schema":"nal.qwen3-8b-prefix6-moba-feasibility.v1","status":"completed","classification":classification,
            "hypothesis_id":"qwen3-8b-prefix6-moba-feasibility-v1","profile":"EXPLORATORY_FEASIBILITY" if stage!="pilot" else "systems_lifecycle_only_not_quality_evidence",
            "stage":stage,"selection":selection,"producer_fingerprint":producing["fingerprint"],"validity":validity,
            "frozen_weights_before":original,"frozen_weights_after":after,"model_weights_unchanged":True,
            "summary":{"requested":len(settings)*len(lengths)*len(sources)*len(seeds),"completed":sum(c["status"]=="completed" for c in cells),
                       "failed":sum(c["status"]!="completed" for c in cells),"joint_passes":sum(c["gate"] is True for c in cells)},
            "cells":cells,"test_evaluated":stage=="final","application_accuracy_tested":False,"completed_at":time.time()}
    bench.write_json(output/"campaign_results.json",result)
    print(json.dumps({"classification":classification,"summary":result["summary"],"selection":selection}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("pilot","screen","readouts","final"),default="screen")
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--head-chunk-size",type=int,default=32)
    parser.add_argument("--memory-fraction",type=float,default=1.0)
    args=parser.parse_args()
    run(args.output,args.stage,args.selection,args.head_chunk_size,args.memory_fraction)
