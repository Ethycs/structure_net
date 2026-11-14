"""Deduplicate tied checkpoint storage without changing any model tensor."""
import argparse
import gc
import json
from pathlib import Path

import torch

from . import jspace_full_capture as base
from . import jspace_full_compact as experiment
from .jspace_full_fit import derive_root
from .jspace_bench_data import sha,write
from .jspace_bench_tensor import run_cases,accuracy,model_bytes

DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_export_v1"


def prepare(root,parent):
    base.cpu_only();plan=base.verify(parent)
    manifest=json.loads((parent/"train_manifest.json").read_text())
    if manifest["status"]!="completed" or manifest["fingerprint"]!=plan["scientific_fingerprint"]:raise ValueError("Incomplete training")
    return derive_root(root,parent,[Path(__file__)],[parent/"result.json",parent/"train_manifest.json"],
        trained_root=str(parent.resolve()),export_scope="Identical tensor values; restore the existing input/output embedding storage alias")


def export(root):
    base.cpu_only();plan=base.verify(root);parent=Path(plan["trained_root"])
    result=json.loads((parent/"result.json").read_text());source=parent/"compact_model.pt"
    if sha(source)!=result["checkpoint_sha256"]:raise ValueError("Training checkpoint changed")
    state=torch.load(source,map_location="cpu",weights_only=True)
    embedding="model.embed_tokens.original.weight";readout="lm_head.weight"
    if not torch.equal(state[embedding],state[readout]):raise ValueError("Embedding/readout values are not tied")
    raw_alias=state[embedding].data_ptr()==state[readout].data_ptr()
    state[readout]=state[embedding]
    target=root/"compact_model_tied.pt";base.save(target,state)
    reloaded=torch.load(target,map_location="cpu",weights_only=True)
    if state.keys()!=reloaded.keys() or any(not torch.equal(value,reloaded[name]) for name,value in state.items()):
        raise ValueError("Export changed tensor values")
    if reloaded[embedding].data_ptr()!=reloaded[readout].data_ptr():raise ValueError("Export alias was not preserved")
    value={"status":"completed","fingerprint":plan["scientific_fingerprint"],"all_state_tensors_exact":True,
        "tensor_count":len(state),"source_had_embedding_alias":raw_alias,"export_has_embedding_alias":True,
        "source_checkpoint":str(source),"source_sha256":sha(source),"export_sha256":sha(target),
        "source_file_bytes":source.stat().st_size,"export_file_bytes":target.stat().st_size,
        "bytes_saved_by_restoring_alias":source.stat().st_size-target.stat().st_size,
        "scope":"Serialization-only correction. Shapes, precision, learned values and model computation are unchanged."}
    write(root/"export.json",value)
    del state,reloaded;gc.collect()
    return value


def reload(root):
    plan=base.verify(root);exported=json.loads((root/"export.json").read_text())
    if sha(root/"compact_model_tied.pt")!=exported["export_sha256"]:raise ValueError("Export changed")
    stage,fit=experiment.start(root,"reload");experiment.rotate(stage,fit)
    for i in range(36):experiment.compile_layer(stage,fit,i).pack()
    checkpoint=torch.load(root/"compact_model_tied.pt",map_location="cpu",weights_only=True)
    stage.model.load_state_dict(checkpoint,strict=True);del checkpoint
    actual=run_cases(stage,stage.tasks["evaluation"],"reloaded_evaluation")
    expected=json.loads((Path(plan["trained_root"])/"train/adapted_36_evaluation.json").read_text())
    if actual!=expected:raise ValueError("Export changed evaluation logits")
    write(stage.path/"verification.json",{"strict_state_load":True,"all_256_records_exact":True,"accuracy":accuracy(actual),
        "model_tensor_bytes":model_bytes(stage.model),"checkpoint_sha256":exported["export_sha256"],
        "source_checkpoint_sha256":exported["source_sha256"],"checkpoint":str(root/"compact_model_tied.pt")})
    return stage.finish()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","export","reload"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--parent",type=Path,default=experiment.DEFAULT_ROOT)
    a=p.parse_args();value=prepare(a.root,a.parent) if a.stage=="prepare" else {"export":export,"reload":reload}[a.stage](a.root)
    print(json.dumps(value,allow_nan=False),flush=True)
