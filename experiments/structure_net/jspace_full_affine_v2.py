"""Correct the missing value mean and fuse output projection; preserve v1."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import jspace_full_compact as original
from . import jspace_full_capture as base
from . import jspace_compact_model as model_module
from .jspace_full_fit import derive_root
from .jspace_affine_attention import AffinePackedAttention,value_mean_offset

DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_affine_v2"
SHAKEDOWN_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_affine_shakedown_v2"
original_start=original.start
original_compile=original.compile_layer


def prepare(root,parent,systems_only=False):
    plan=base.verify(parent)
    source=Path(plan["capture_root"])/"capture/statistics.pt"
    paths=[Path(__file__),Path(__file__).with_name("jspace_affine_attention.py"),
           base.ROOT/"tests/structure_net/test_jspace_affine_attention.py",
           base.ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-affine-correction.md"]
    return derive_root(root,parent,paths,[parent/"result.json",source],systems_only=systems_only,
        affine_value_mean_source=str(source),amendment={"known_before_freeze":"v1 trained 103/256, 36.35 vs 40.90 graph tok/s; dropped 70.87%/81.89% of selected value-mean energy in layers 0/1",
            "corrections":["Fold O(I-SS^T) mean(V) into a fixed output offset","One padded attention call and one output projection; affine offset accumulated as two BF16 constant terms"],
            "unchanged":"datasets, ranks, channel budget, anchor basis, seed, optimizer, 96 steps/layer, loss and 36-layer adaptation"})


def start(root,name):
    stage,fit=original_start(root,name)
    stats=torch.load(Path(stage.plan["affine_value_mean_source"]),map_location="cpu",weights_only=True)
    for i in range(36):
        entry=fit["layers"][str(i)];entry["value_mean"]=stats["value_sum"][i,entry["attention_group"]]/stats["count"]
    return stage,fit


def compile_layer(stage,fit,i):
    entry=fit["layers"][str(i)];attention=stage.model.model.layers[i].self_attn
    with torch.no_grad():offset=value_mean_offset(attention,entry["attention_group"],entry["value_basis"],entry["value_mean"])
    block=original_compile(stage,fit,i)
    block.self_attn.output_bias=block.self_attn.output_bias.float()+offset
    return block


def run(root,stage):
    with patch.object(model_module,"RotatedAttention",AffinePackedAttention),patch.object(original,"start",start),patch.object(original,"compile_layer",compile_layer):
        if stage=="export_reload":
            from .jspace_full_export import reload
            return reload(root)
        return {"shakedown":original.shakedown,"controls":original.controls,"train":original.train,"reload":original.reload}[stage](root)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","shakedown","controls","train","reload","export_reload"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--parent",type=Path,default=original.DEFAULT_ROOT)
    p.add_argument("--systems-only",action="store_true");a=p.parse_args()
    print(json.dumps(prepare(a.root,a.parent,a.systems_only) if a.stage=="prepare" else run(a.root,a.stage),allow_nan=False),flush=True)
