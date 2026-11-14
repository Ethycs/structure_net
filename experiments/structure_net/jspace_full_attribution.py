"""Posthoc development-only attribution of the initialized compression loss."""
import argparse
import gc
import json
from pathlib import Path

import torch

from . import jspace_full_compact as experiment
from . import jspace_full_capture as base
from .jspace_full_fit import derive_root
from .jspace_bench_data import write
from .jspace_bench_tensor import run_cases,accuracy,comparison,model_bytes

DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_attribution_v1"
ARMS=("mlp_only","attention_only","attention_full_input","attention_full_head")


def prepare(root,parent):
    base.checked(parent,"controls")
    return derive_root(root,parent,[Path(__file__)],[parent/"controls_manifest.json",parent/"controls.json"],
        reference_root=str(parent.resolve()),diagnostic={"arms":list(ARMS),"split":"development_only",
        "known_before_freeze":"evaluation original 210/256, rotated 209/256, initialized compact 89/256; adaptation is running",
        "purpose":"Attribute initialized loss; no checkpoint/rank selection and no changes to the running fixed adaptation"})


def run(root,arm):
    if arm not in ARMS:raise ValueError("Unknown diagnostic")
    stage,fit=experiment.start(root,arm);experiment.rotate(stage,fit)
    for i,block in enumerate(stage.model.model.layers):
        entry=fit["layers"][str(i)]
        if arm=="mlp_only":block.mlp.compile(entry,entry["removed"])
        else:
            rank=2048 if arm=="attention_full_input" else 1024
            pairs=list(range(64)) if arm=="attention_full_head" else entry["rope_pairs"]
            values=torch.eye(128) if arm=="attention_full_head" else entry["value_basis"]
            block.self_attn.compile(rank,entry["mean_coordinates"],entry["attention_group"],pairs,values)
        block.pack()
    rows=run_cases(stage,stage.tasks["development"],arm+"_development")
    reference=Path(stage.plan["reference_root"])/"controls"
    value={"arm":arm,"scope":"posthoc development-only initialized-component attribution; not a trained candidate",
        "accuracy":accuracy(rows),"tensor_bytes":model_bytes(stage.model),
        "versus_original":comparison(json.loads((reference/"original_development.json").read_text()),rows),
        "versus_rotation":comparison(json.loads((reference/"rotated_development.json").read_text()),rows)}
    write(stage.path/"result.json",value)
    manifest=stage.finish();del stage,fit;gc.collect();torch.cuda.empty_cache()
    return {"result":value,"manifest":manifest}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","all",*ARMS))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--parent",type=Path,default=experiment.DEFAULT_ROOT)
    a=p.parse_args()
    if a.stage=="prepare":value=prepare(a.root,a.parent)
    elif a.stage=="all":value={arm:run(a.root,arm) for arm in ARMS}
    else:value=run(a.root,a.stage)
    print(json.dumps(value,allow_nan=False),flush=True)
