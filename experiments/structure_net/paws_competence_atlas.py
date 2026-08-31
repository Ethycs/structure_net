#!/usr/bin/env python3
"""Experiment 06: freeze local heads and build the held-out A/B/C competence atlas."""
from __future__ import annotations

import argparse, asyncio, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner

try:
    from experiments.structure_net.paws_dataset_contract import render_prompt
    from experiments.structure_net.paws_teacher_annotation import annotate, eligible_rows, key, sha256
except ModuleNotFoundError:
    from paws_dataset_contract import render_prompt
    from paws_teacher_annotation import annotate, eligible_rows, key, sha256

SCHEMA_VERSION="nal.paws-competence-atlas.v1"; HYPOTHESIS_ID="paws-held-out-competence-atlas-v1"


def best_checkpoint(campaign: Path) -> Path:
    payload=json.loads(campaign.read_text(encoding="utf-8")); candidates=[]
    for experiment_id,result in payload["results"].items():
        if result["error"] is None: candidates.append((result["metrics"]["balanced_accuracy"],result["metrics"]["accuracy"],experiment_id,result["checkpoint"]))
    if not candidates: raise RuntimeError(f"no successful checkpoints in {campaign}")
    return Path(sorted(candidates,key=lambda item:(-item[0],-item[1],item[2]))[0][3])


def local_probabilities(cache_path: Path, checkpoint_path: Path) -> tuple[np.ndarray,np.ndarray]:
    cache=np.load(cache_path); checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    x=torch.from_numpy(cache["dev_x"].astype(np.float32)); mean=checkpoint["mean"].float(); std=checkpoint["std"].float()
    head=torch.nn.Linear(x.shape[1],2); head.load_state_dict(checkpoint["head_state"]); head.eval()
    with torch.no_grad(): probabilities=torch.softmax(head((x-mean)/std),1).numpy()
    return cache["dev_groups"],probabilities


def qwen_dev(rows:list[dict[str,Any]],path:Path,concurrency:int)->dict[str,dict[str,Any]]:
    existing={}
    if path.exists():
        existing={record["group_id"]:record for record in map(json.loads,path.read_text(encoding="utf-8").splitlines())}
    pending=[row for row in rows if existing.get(row["group_id"],{}).get("status")!="success"]
    api_key=key()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
      for start in range(0,len(pending),64):
        batch=pending[start:start+64]
        if batch:
            records=list(pool.map(lambda row:annotate(row,base_url="https://central-dev.zt:4000/v1",model="qwen3-8b",ca_path="/home/rabbit/.config/lakefs/caddy-root.crt",api_key=api_key),batch))
            existing.update((record["group_id"],record) for record in records)
            temporary=path.with_suffix(".jsonl.tmp")
            with temporary.open("w",encoding="utf-8") as handle:
                for group_id in sorted(existing): handle.write(json.dumps(existing[group_id],sort_keys=True)+"\n")
            temporary.replace(path)
    return existing


def worker(experiment:Experiment,_device_id:int)->ExperimentResult:
    started=time.perf_counter(); p=experiment.parameters; output=Path(p["output"]); output.mkdir(parents=True,exist_ok=True)
    rows=eligible_rows(Path("data/datasets/paws-wiki/labeled/dev.csv")); groups=np.asarray([row["group_id"] for row in rows])
    tiny_checkpoint=best_checkpoint(Path(p["tiny_campaign"])); smol_checkpoint=best_checkpoint(Path(p["smol_campaign"]))
    tiny_groups,tiny_prob=local_probabilities(Path(p["tiny_cache"]),tiny_checkpoint); smol_groups,smol_prob=local_probabilities(Path(p["smol_cache"]),smol_checkpoint)
    if not np.array_equal(groups,tiny_groups) or not np.array_equal(groups,smol_groups): raise RuntimeError("development feature caches are misaligned")
    qwen=qwen_dev(rows,output/"qwen_dev.jsonl",p["concurrency"])
    if set(qwen)!={row["group_id"] for row in rows} or any(record["status"]!="success" for record in qwen.values()): raise RuntimeError("Qwen development annotation gate failed")
    labels=np.asarray([row["label"] for row in rows],dtype=np.int8); a=tiny_prob.argmax(1).astype(np.int8); b=smol_prob.argmax(1).astype(np.int8); c=np.asarray([qwen[group]["teacher_label"] for group in groups],dtype=np.int8)
    partitions=np.asarray([hashlib.sha256(("paws-dev-partition-v1:"+group).encode()).digest()[0]%4 for group in groups],dtype=np.int8)
    smol_cache=np.load(Path(p["smol_cache"])); carrier=smol_cache["dev_x"].astype(np.float16)
    np.savez_compressed(output/"competence_atlas.npz",groups=groups,labels=labels,partition=partitions,a_prediction=a,b_prediction=b,c_prediction=c,a_probability=tiny_prob.astype(np.float16),b_probability=smol_prob.astype(np.float16),embedding=carrier)
    mask=partitions==1; signatures=[f"{int(a[i]==labels[i])}{int(b[i]==labels[i])}{int(c[i]==labels[i])}" for i in np.flatnonzero(mask)]
    counts={signature:signatures.count(signature) for signature in ("000","001","010","011","100","101","110","111")}
    non_nested=float(np.mean([signature in {"010","100","101","110"} for signature in signatures]))
    summary={"schema_version":SCHEMA_VERSION,"complete":True,"atlas_rows":int(mask.sum()),"signature_counts":counts,"non_nested_rate":non_nested,"tiny_checkpoint":str(tiny_checkpoint),"smol_checkpoint":str(smol_checkpoint),"tiny_checkpoint_sha256":sha256(tiny_checkpoint),"smol_checkpoint_sha256":sha256(smol_checkpoint),"atlas_sha256":sha256(output/"competence_atlas.npz"),"qwen_sha256":sha256(output/"qwen_dev.jsonl")}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return ExperimentResult(experiment_id=experiment.id,hypothesis_id=HYPOTHESIS_ID,metrics={"non_nested_rate":non_nested,"atlas_rows":float(mask.sum())},primary_metric=1-non_nested,model_architecture=[],model_parameters=0,training_time=time.perf_counter()-started,observations=[f"summary={output/'summary.json'}"])


async def run(output:Path,concurrency:int)->dict[str,Any]:
    p={"output":str(output),"concurrency":concurrency,"tiny_campaign":"data/experiments/paws_abc_routing/2026-08-16_experiment_05/campaign_results.json","smol_campaign":"data/experiments/paws_abc_routing/2026-08-16_experiment_04/campaign_results.json","tiny_cache":"data/experiments/paws_abc_routing/2026-08-16_experiment_05/tinyllm_features.npz","smol_cache":"data/experiments/paws_abc_routing/2026-08-16_experiment_04/smollm_features.npz"}
    experiment=Experiment(id="paws-held-out-competence-atlas",hypothesis_id=HYPOTHESIS_ID,name="PAWS held-out competence atlas",parameters=p)
    runner=AsyncExperimentRunner(LabConfig(project_name="paws_competence_atlas",results_dir=str(output/"nal"),device_ids=[-1],max_parallel_experiments=1,min_experiments_per_hypothesis=1,require_statistical_significance=False,enable_wandb=False),worker)
    result=(await runner.run_experiments([experiment]))[0]; return {"complete":result.error is None,"metrics":result.metrics,"error":result.error}


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("data/experiments/paws_abc_routing/2026-08-16_experiment_06")); parser.add_argument("--concurrency",type=int,default=16); args=parser.parse_args()
    result=asyncio.run(run(args.output,args.concurrency)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["complete"] else 1)
if __name__=="__main__": main()
