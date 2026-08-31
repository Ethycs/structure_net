#!/usr/bin/env python3
"""Experiment 02: undistilled PAWS baselines for TinyLLM, SmolLM, and Qwen."""
from __future__ import annotations

import argparse, asyncio, csv, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.models import TinyLLMConfig, TinyLLMModel
try:
    from experiments.structure_net.paws_dataset_contract import pair_group_id, render_prompt
except ModuleNotFoundError:  # Support direct script execution from the repository root.
    from paws_dataset_contract import pair_group_id, render_prompt

SCHEMA_VERSION = "nal.paws-undistilled-baseline.v1"
HYPOTHESIS_ID = "paws-undistilled-abc-baseline-v1"
MODELS = ("tinyllm", "smollm", "qwen")

@dataclass(frozen=True)
class Config:
    samples: int = 256
    selection_seed: int = 17
    dev_path: str = "data/datasets/paws-wiki/labeled/dev.csv"
    tiny_checkpoint: str = "data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/checkpoint_step12000.pt"
    tiny_tokenizer: str = "data/corpora/babylm_10M_bpe16k.tokenizer.json"
    smol_path: str = "/data/models/SmolLM2-360M-Instruct"
    qwen_base_url: str = "https://central-dev.zt:4000/v1"
    qwen_model: str = "qwen3-8b"
    qwen_ca_path: str = "/home/rabbit/.config/lakefs/caddy-root.crt"

def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"); tmp.replace(path)

def _sha256(path: str) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def selected(config: Config) -> list[dict[str, Any]]:
    if config.samples < 2 or config.samples % 2: raise ValueError("samples must be an even integer >= 2")
    rows=[]
    with Path(config.dev_path).open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f,delimiter="\t"):
            r["label"]=int(r["label"]); r["group_id"]=pair_group_id(r["sentence1"],r["sentence2"]); rows.append(r)
    seen=set(); unique=[]
    for r in rows:
        if r["group_id"] not in seen: seen.add(r["group_id"]); unique.append(r)
    chosen=[]
    for label in (0,1):
        pool=[r for r in unique if r["label"]==label]
        pool.sort(key=lambda r: hashlib.sha256(f"{config.selection_seed}:{r['group_id']}".encode()).hexdigest())
        chosen.extend(pool[:config.samples//2])
    chosen.sort(key=lambda r:r["group_id"])
    if len(chosen) != config.samples: raise ValueError(f"requested {config.samples}, selected {len(chosen)}")
    return chosen

def _tiny_model(config: Config, device: torch.device) -> tuple[TinyLLMModel,Tokenizer]:
    m=TinyLLMModel(TinyLLMConfig.from_preset("d8",block_size=256,vocab_size=50257,initialization_seed=7))
    payload=torch.load(config.tiny_checkpoint,map_location="cpu",weights_only=True); m.load_state_dict(payload["model_state"],strict=True)
    return m.to(device).eval(),Tokenizer.from_file(config.tiny_tokenizer)

@torch.no_grad()
def local_predictions(name: str, rows: list[dict[str,Any]], config: Config, device: torch.device) -> list[dict[str,Any]]:
    if name=="tinyllm": model,tok=_tiny_model(config,device); limit=256
    else: tok=AutoTokenizer.from_pretrained(config.smol_path,local_files_only=True); model=AutoModelForCausalLM.from_pretrained(config.smol_path,local_files_only=True,torch_dtype=torch.float16 if device.type=="cuda" else torch.float32).to(device).eval(); limit=8192
    out=[]
    for row in rows:
        prompt=render_prompt(row["sentence1"],row["sentence2"])
        pids=tok.encode(prompt).ids if name=="tinyllm" else tok.encode(prompt,add_special_tokens=True)
        max_new=min(12,limit-len(pids))
        if max_new <= 0: raise RuntimeError(f"{name} prompt does not fit its context")
        ids=torch.tensor([pids],device=device)
        generated=[]
        for _ in range(max_new):
            if name=="tinyllm": logits,_=model(ids,return_full_logits=True)
            else: logits=model(ids).logits
            next_id=int(torch.argmax(logits[0,-1]).item()); generated.append(next_id)
            ids=torch.cat((ids,torch.tensor([[next_id]],device=device)),dim=1)
            eos=None if name=="tinyllm" else tok.eos_token_id
            if eos is not None and next_id==eos: break
            partial=(tok.decode(generated) if name=="tinyllm" else tok.decode(generated,skip_special_tokens=True)).strip()
            if partial and not any(label.startswith(partial) for label in ("DIFFERENT","PARAPHRASE")): break
            if partial in ("DIFFERENT","PARAPHRASE"): break
        text=(tok.decode(generated) if name=="tinyllm" else tok.decode(generated,skip_special_tokens=True)).strip()
        pred={"DIFFERENT":0,"PARAPHRASE":1}.get(text)
        out.append({"group_id":row["group_id"],"label":row["label"],"prediction":pred,"raw":text,"method":"greedy_exact_generation"})
    return out

def _key() -> str:
    for line in Path(".env").read_text().splitlines():
        if line.startswith("LITELLM_KEY="): return line.split("=",1)[1].strip().strip('"').strip("'")
    raise RuntimeError("LITELLM_KEY missing")

def qwen_predictions(rows:list[dict[str,Any]],config:Config)->list[dict[str,Any]]:
    import requests
    key=_key()
    def predict(row:dict[str,Any])->dict[str,Any]:
        error=None
        for attempt in range(4):
            try:
                response=requests.post(config.qwen_base_url.rstrip("/")+"/chat/completions",headers={"Authorization":f"Bearer {key}"},json={"model":config.qwen_model,"messages":[{"role":"user","content":render_prompt(row["sentence1"],row["sentence2"])}],"temperature":0,"max_tokens":16},verify=config.qwen_ca_path,timeout=60); response.raise_for_status()
                text=response.json()["choices"][0]["message"]["content"].strip(); pred={"DIFFERENT":0,"PARAPHRASE":1}.get(text)
                return {"group_id":row["group_id"],"label":row["label"],"prediction":pred,"raw":text,"attempts":attempt+1}
            except requests.RequestException as exc:
                error=exc
                if attempt<3: time.sleep(2**attempt)
        raise RuntimeError(f"Qwen request failed after retries: {error}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(predict,rows))

def worker(experiment:Experiment,device_id:int)->ExperimentResult:
    started=time.perf_counter(); config=Config(**experiment.parameters["configuration"]); name=experiment.parameters["model"]; rows=selected(config)
    predictions=qwen_predictions(rows,config) if name=="qwen" else local_predictions(name,rows,config,torch.device(f"cuda:{device_id}" if device_id>=0 and torch.cuda.is_available() else "cpu"))
    valid=[p for p in predictions if p["prediction"] is not None]; accuracy=sum(p["prediction"]==p["label"] for p in valid)/len(rows); malformed=1-len(valid)/len(rows)
    recalls=[]
    for label in (0,1):
        label_rows=[p for p in predictions if p["label"]==label]
        recalls.append(sum(p["prediction"]==label for p in label_rows)/len(label_rows))
    detail={"schema_version":SCHEMA_VERSION,"hypothesis_id":HYPOTHESIS_ID,"model":name,"configuration":asdict(config),"dataset_sha256":_sha256(config.dev_path),"selection_ids_sha256":hashlib.sha256("\n".join(r["group_id"] for r in rows).encode()).hexdigest(),"accuracy":accuracy,"balanced_accuracy":sum(recalls)/2,"malformed_rate":malformed,"predictions":predictions}
    path=Path(experiment.parameters["output"]); _write(path,detail)
    return ExperimentResult(experiment_id=experiment.id,hypothesis_id=HYPOTHESIS_ID,metrics={"accuracy":accuracy,"balanced_accuracy":sum(recalls)/2,"malformed_rate":malformed},primary_metric=accuracy,model_architecture=[],model_parameters=0,training_time=time.perf_counter()-started,observations=[f"detail={path}"])

async def run(output:Path,config:Config)->dict[str,Any]:
    experiments=[Experiment(id=f"paws-undistilled-{m}",hypothesis_id=HYPOTHESIS_ID,name=f"PAWS undistilled {m}",parameters={"configuration":asdict(config),"model":m,"output":str(output/"runs"/m/"result.json")}) for m in MODELS]
    runner=AsyncExperimentRunner(LabConfig(project_name="paws_undistilled_baseline",results_dir=str(output/"nal"),device_ids=[0],max_parallel_experiments=1,min_experiments_per_hypothesis=1,require_statistical_significance=False,enable_wandb=False),worker)
    results=await runner.run_experiments(experiments); summary={r.experiment_id:r.metrics for r in results}; payload={"schema_version":SCHEMA_VERSION,"hypothesis_id":HYPOTHESIS_ID,"configuration":asdict(config),"results":summary,"complete":len(results)==3 and all(r.error is None for r in results)}; _write(output/"campaign_results.json",payload); return payload

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--samples",type=int,default=256); p.add_argument("--output",type=Path,default=Path("data/experiments/paws_abc_routing/2026-08-16_experiment_02")); a=p.parse_args(); result=asyncio.run(run(a.output,Config(samples=a.samples))); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["complete"] else 1)
if __name__=="__main__": main()
