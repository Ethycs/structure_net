"""Exact-width attention control and full-model polynomial-MLP follow-up."""
import argparse
import gc
import json
from pathlib import Path

import torch

from . import jspace_full_compact as experiment
from . import jspace_full_capture as base
from .jspace_full_attribution import DEFAULT_ROOT as ATTRIBUTION,ARMS
from .jspace_full_fit import derive_root
from .jspace_bench_data import sha,write
from .jspace_bench_tensor import run_cases,accuracy,comparison,model_bytes

DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_refinement_v1"


def prepare(root,parent):
    for arm in ARMS:base.checked(ATTRIBUTION,arm)
    return derive_root(root,parent,[Path(__file__),base.ROOT/"tests/structure_net/test_jspace_compact_attention_identity.py"],
        [parent/"controls_manifest.json",parent/"controls.json",parent/"result.json",
         *[ATTRIBUTION/arm/"result.json" for arm in ARMS]],reference_root=str(parent.resolve()),
        followup={"classification":"posthoc_component_and_execution_measurement","additional_training_steps":0,
            "known_before_freeze":"full adapted 103/256, 36.35 vs original 40.90 graph tok/s; development MLP-only 46/64 vs native 48/64; attention variants 18-26/64",
            "arms":["attention_identity","mlp_runtime"],"purpose":"Check the split implementation at full rank; benchmark the less damaging polynomial MLP branch on every layer"})


def run(root,arm):
    if arm not in ("attention_identity","mlp_runtime"):raise ValueError("Unknown follow-up")
    stage,fit=experiment.start(root,arm);experiment.rotate(stage,fit)
    for i,block in enumerate(stage.model.model.layers):
        entry=fit["layers"][str(i)]
        if arm=="attention_identity":
            block.self_attn.compile(2048,entry["mean_coordinates"],entry["attention_group"],list(range(64)),torch.eye(128))
        else:block.mlp.compile(entry,entry["removed"])
        block.pack()
    reference=Path(stage.plan["reference_root"])/"controls"
    eval_rows=run_cases(stage,stage.tasks["evaluation"],arm+"_evaluation")
    dev=run_cases(stage,stage.tasks["development"],arm+"_development")
    result={"arm":arm,"status":"completed","confirmed":False,"fingerprint":stage.plan["scientific_fingerprint"],
        "evaluation":accuracy(eval_rows),"development":accuracy(dev),"tensor_bytes":model_bytes(stage.model),
        "versus_original":comparison(json.loads((reference/"original_evaluation.json").read_text()),eval_rows),
        "versus_rotation":comparison(json.loads((reference/"rotated_evaluation.json").read_text()),eval_rows),
        "scope":"posthoc follow-up on reused evaluation; no additional training or checkpoint selection"}
    if arm=="mlp_runtime":
        result["runtime"]=experiment.timed_generation(stage,arm)
        original=json.loads((Path(stage.plan["reference_root"])/"controls.json").read_text())["original"]
        result["matched_graph_speedup"]=result["runtime"]["timings"]["cuda_graph"]["tokens_per_second"]/original["runtime"]["timings"]["cuda_graph"]["tokens_per_second"]
        result["tensor_reduction_fraction"]=1-result["tensor_bytes"]/original["tensor_bytes"]
        if stage.model.lm_head.weight.data_ptr()!=stage.model.model.embed_tokens.weight.data_ptr():raise ValueError("Readout tie lost")
        state={}
        for name,value in stage.model.state_dict().items():
            state[name]=state["model.embed_tokens.original.weight"] if name=="lm_head.weight" else value.detach().cpu()
        base.save(stage.path/"model.pt",state);del state
        result["checkpoint_bytes"]=(stage.path/"model.pt").stat().st_size
        result["checkpoint_sha256"]=sha(stage.path/"model.pt")
    write(stage.path/"result.json",result)
    manifest=stage.finish();del stage,fit;gc.collect();torch.cuda.empty_cache()
    return {"result":result,"manifest":manifest}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","all","attention_identity","mlp_runtime"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--parent",type=Path,default=experiment.DEFAULT_ROOT)
    a=p.parse_args()
    value=prepare(a.root,a.parent) if a.stage=="prepare" else ({arm:run(a.root,arm) for arm in ("attention_identity","mlp_runtime")} if a.stage=="all" else run(a.root,a.stage))
    print(json.dumps(value,allow_nan=False),flush=True)
