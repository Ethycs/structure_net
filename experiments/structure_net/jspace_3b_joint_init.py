"""CPU materialization of a fresh, fully compact 3B joint-training initializer.

All learned parameters are freshly projected from the original native model.
Historical task-Jacobian bases and polynomial relations remain fixed. No
trained suffix weights, development selections, or audit labels initialize it.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_3b_joint_relation as relations
from . import jspace_3b_broad_recovery_cache as cache
from . import jspace_qwen3b_progressive_model as builder
from .jspace_incremental_suffix import file_identity,pin,fingerprint,fixed_state

base=relations.base
ROOT=relations.ROOT
SCHEMA="jspace.3b-joint-initializer.v1"
REVISION="aa8e72537993ba99e69dfaafa59ed015b17504d1"
read=relations.read


def validated_items(items):
    if (len(items)!=4864 or Counter(c["split"] for c in items)!={"train":4608,"development":256}
            or len({c["identity"] for c in items})!=len(items)):
        raise ValueError("Canonical 4608 TRAIN/256 DEV identities required; no audit rows")
    offset=0
    for case in items:
        if (case["offset"]!=offset or case["stop"]-offset!=len(case["input_ids"])
                or not 3<=len(case["input_ids"])<=384 or len(case["content_loss_positions"])!=4
                or len(case["content_targets"])!=4 or case["content_loss_positions"][-1]!=len(case["input_ids"])-1):
            raise ValueError("Canonical input offsets/content positions changed")
        offset=case["stop"]
    return offset


def completed_relations(root):
    root=Path(root);p=relations.verify(root);r=read(root/"result.json");m=read(root/"manifest.json")
    if (m.get("status")!="completed" or r.get("status")!="completed"
            or m.get("fingerprint")!=p["fingerprint"] or r.get("fingerprint")!=p["fingerprint"]
            or r.get("fit_map")!={k:v["directory"] for k,v in p["layers"].items()}):
        raise ValueError("Completed matching all36 relation map required")
    for name,digest in m["artifacts"].items():
        if base.sha(root/name)!=digest:raise ValueError("Relation completion changed")
    for layer in relations.WIDTHS:
        directory=Path(r["fit_map"][str(layer)]);manifest=read(directory/"manifest.json")
        if manifest.get("status")!="completed" or manifest.get("fingerprint")!=p["fingerprint"]:
            raise ValueError("Both replacement relations must be complete")
        for name,digest in manifest["artifacts"].items():
            if base.sha(directory/name)!=digest:raise ValueError("Replacement relation changed")
    return p,r


def prepare(root,relation_root,cache_root):
    relations.cpu_only();root,relation_root,cache_root=map(lambda p:Path(p).resolve(),(root,relation_root,cache_root))
    if root.exists():raise FileExistsError(root)
    rp,rr=completed_relations(relation_root);cp=cache.verify(cache_root)
    native=cache.checked(cache_root,"native",cp);cache.checked(cache_root,"parent",cp)
    items=read(cache_root/"items.json")["items"];tokens=validated_items(items)
    if cp["model"]["revision"]!=REVISION or cp["model"]!=rp["model"]:
        raise ValueError("Original pinned native Qwen2.5-3B revision required")
    inputs=dict(rp["input_sha256"])
    for name,digest in cp["input_sha256"].items():
        if name in inputs and inputs[name]!=digest:raise ValueError("Conflicting original input provenance")
        inputs[name]=digest
    for path in (relation_root/"plan.json",relation_root/"result.json",relation_root/"manifest.json",
                 cache_root/"plan.json",cache_root/"items.json"):
        pin(inputs,path)
    for layer in relations.WIDTHS:
        directory=Path(rr["fit_map"][str(layer)])
        pin(inputs,directory/"manifest.json")
        for name,digest in read(directory/"manifest.json")["artifacts"].items():pin(inputs,directory/name,digest)
    for phase in ("native","parent"):
        directory=cache_root/phase;pin(inputs,directory/"manifest.json")
        for name,digest in read(directory/"manifest.json")["artifacts"].items():
            inputs[str((directory/name).resolve())]=digest  # checked above, once.
    for name,digest in cp["model"]["files"].items():pin(inputs,Path(cp["model"]["path"])/name,digest)
    pin(inputs,cp["parent_checkpoint"],cp["parent_checkpoint_sha256"])
    for name in ("fit.pt","relation.pt"):
        pin(inputs,Path(cp["fits"]["35"])/name)
    files=relations.local_import_closure([Path(__file__),ROOT/"tests/structure_net/test_jspace_3b_joint_init.py"])
    feature=read(cache_root/"parent/parent_feature_receipt.json")["features"]
    plan=dict(schema=SCHEMA,study_id="JSPACE-3B-ALL36-JOINT-INITIALIZER-01",relation_root=str(relation_root),
        relation_fingerprint=rp["fingerprint"],cache_root=str(cache_root),cache_fingerprint=cp["fingerprint"],
        model=cp["model"],fit_map=rr["fit_map"],anchor_widths=rr["anchor_widths"],cut=0,rank=1024,head_dim=64,
        native_correct=native["correct"],native_count=native["count"],label_ids=cp["label_ids"],
        item_digest=relations.protocol_digest(items),shape=[tokens,2048],training_count=4608,development_count=256,
        teacher_logits_path=str(cache_root/"native/teacher_logits.npy"),teacher_slots=[4,5,6,7],
        teacher_semantics="Original native BF16 full-vocabulary logits at the frozen three content positions plus answer",
        parent_checkpoint=cp["parent_checkpoint"],parent_checkpoint_sha256=cp["parent_checkpoint_sha256"],
        feature_teacher_fit35=cp["fits"]["35"],diagnostic_parent_feature_receipt=feature,
        feature_alpha=0.0,objective="Original-native content4 KD+CE only; no parent feature auxiliary",
        initializer="Fresh original-native projection for all36 blocks; no recovered model parameters loaded",
        precision="All stack/norm learned parameters and tied E BF16; fixed geometry untouched from BF16-mode builder",
        rotary="CPU constructor fingerprint recorded; runtime CUDA constructor is separately defined and must not be equated bitwise",
        source_fit_teacher_provenance=rr["teacher_provenance"],confirmed=False,
        input_sha256=inputs,input_identity={p:file_identity(p) for p in inputs},
        source_sha256={str(p):base.sha(p) for p in sorted(files)})
    plan["fingerprint"]=relations.protocol_digest(plan);root.mkdir(parents=True)
    for name,digest in plan["source_sha256"].items():
        target=root/"producer"/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        if base.sha(target)!=digest:raise ValueError("Initializer source archive changed")
    shutil.copy2(cache_root/"items.json",root/"items.json");base.write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],shape=plan["shape"],training=4608,development=256)


def verify(root):
    root=Path(root);p=read(root/"plan.json")
    if (p.get("schema")!=SCHEMA or p["fingerprint"]!=relations.protocol_digest({k:v for k,v in p.items() if k!="fingerprint"})
            or p.get("cut")!=0 or p.get("model",{}).get("revision")!=REVISION
            or p.get("teacher_slots")!=[4,5,6,7] or p.get("feature_alpha")!=0.0
            or p.get("confirmed") is not False):
        raise ValueError("Initializer protocol changed")
    for name,identity in p["input_identity"].items():
        if file_identity(name)!=identity:raise ValueError("Initializer input changed: "+name)
    for name,digest in p["source_sha256"].items():
        if base.sha(name)!=digest or base.sha(root/"producer"/Path(name).relative_to(ROOT))!=digest:
            raise ValueError("Initializer producer/archive changed")
    items=read(root/"items.json")["items"]
    if relations.protocol_digest(items)!=p["item_digest"] or validated_items(items)!=p["shape"][0]:
        raise ValueError("Canonical training/development mapping changed")
    return p


@torch.no_grad()
def canonicalize(stack,norm):
    for parameter in (*stack.parameters(),*norm.parameters()):parameter.data=parameter.data.to(dtype=torch.bfloat16)
    if any(p.dtype!=torch.bfloat16 for p in (*stack.parameters(),*norm.parameters())):
        raise ValueError("Every learned parameter must use explicit BF16 storage")


def inventory(stack,norm,E,rotary):
    parameters=[*stack.parameters(),*norm.parameters()]
    buffers=[*stack.buffers(),*norm.buffers()]
    learned=sum(p.numel() for p in parameters);learned_bytes=sum(p.numel()*p.element_size() for p in parameters)
    geometry=sum(t.numel()*t.element_size() for t in buffers)
    embedding=E.numel()*E.element_size();rope=sum(t.numel()*t.element_size() for t in rotary.buffers())
    return dict(trainable_scalar_count=learned,trainable_tensor_bytes=learned_bytes,
        fixed_geometry_tensor_bytes=geometry,tied_embedding_readout_bytes=embedding,cpu_rotary_bytes=rope,
        total_unpacked_tensor_bytes=learned_bytes+geometry+embedding+rope,
        cpu_adam_master_moments_bytes=12*learned,cpu_parameter_dtype_gradient_bank_bytes=learned_bytes,
        cpu_optimizer_and_gradient_bank_bytes=12*learned+learned_bytes,
        parameter_dtypes=dict(Counter(str(p.dtype) for p in parameters)),
        geometry_dtypes=dict(Counter(str(t.dtype) for t in buffers)),
        note="Actual materialized unpacked inventory; activations/workspaces/runtime rounding and peak reserved memory are not measured")


def final_geometry(stack,parent_state,*,fit_hashes_match):
    layers=list(parent_state["stack"]["_extra_state"]["layer_indices"])
    if not layers or layers[-1]!=35 or stack.layer_indices[-1]!=35:raise ValueError("Original final layer35 geometry required")
    prefix=f"blocks.{len(layers)-1}.";block=stack.blocks[-1]
    comparisons={}
    for name in ("basis","anchor","anchor_energy"):
        actual=getattr(block,name).detach().cpu();expected=parent_state["stack"][prefix+name]
        comparisons[name]=dict(equal=actual.dtype==expected.dtype and torch.equal(actual,expected),
            shape=list(actual.shape),dtype=str(actual.dtype),reference_dtype=str(expected.dtype),
            max_abs=float((actual.float()-expected.float()).abs().max()))
    return dict(passed=fit_hashes_match and all(v["equal"] for v in comparisons.values()),
        fit_files_exact=fit_hashes_match,comparisons=comparisons,
        policy="Diagnostic coordinate comparison only; this initializer uses native KD+CE with no parent feature auxiliary")


@torch.no_grad()
def write_boundary(path,items,E,*,chunk_tokens=4096):
    if E.device.type!="cpu" or E.dtype!=torch.bfloat16:raise ValueError("Original CPU BF16 tied E required")
    if not items or chunk_tokens<1:raise ValueError("Nonempty canonical items and positive chunk size required")
    if Path(path).exists():raise FileExistsError(path)
    bank=np.lib.format.open_memmap(path,mode="w+",dtype=np.int16,shape=(items[-1]["stop"],E.shape[1]))
    cursor=0
    try:
        for case in items:
            ids=case["input_ids"]
            if case["offset"]!=cursor or case["stop"]-cursor!=len(ids):raise ValueError("Noncanonical token offsets")
            if any(type(i) is not int or i<0 or i>=E.shape[0] for i in ids):raise ValueError("Token outside original embedding vocabulary")
            for start in range(0,len(ids),chunk_tokens):
                tokens=torch.tensor(ids[start:start+chunk_tokens],dtype=torch.long)
                value=E[tokens].contiguous()
                bank[cursor+start:cursor+start+len(tokens)]=value.view(torch.int16).numpy()
            cursor=case["stop"]
        bank.flush()
    finally:bank._mmap.close()
    return dict(shape=[cursor,E.shape[1]],dtype="native-endian int16 lossless BF16 bits",source="Raw canonical input_ids indexed into original tied native E; no learned prefix")


def run(root):
    relations.cpu_only();root=Path(root).resolve();p=verify(root)
    if any((root/name).exists() for name in ("started.json","failure.json","result.json","initializer.pt")):
        raise FileExistsError("Initializer attempts are append-only, without automatic retry")
    base.write(root/"started.json",dict(fingerprint=p["fingerprint"],pid=os.getpid()));torch.set_num_threads(2)
    started=time.monotonic()
    try:
        stack,norm,E,rotary,proof=builder.build_stack(p["model"]["path"],{int(k):v for k,v in p["fit_map"].items()},
            0,None,device="cpu",trainable=True,checkpoint_blocks=True)
        if tuple(stack.layer_indices)!=tuple(range(36)) or not proof.get("native_initialization"):
            raise ValueError("Every block must be freshly compiled from native weights")
        canonicalize(stack,norm)
        if E.dtype!=torch.bfloat16 or E.requires_grad:raise ValueError("Original frozen BF16 tied E required")
        fits_exact=all(base.sha(Path(p["fit_map"]["35"])/name)==base.sha(Path(p["feature_teacher_fit35"])/name)
            for name in ("fit.pt","relation.pt"))
        parent=torch.load(p["parent_checkpoint"],map_location="cpu",weights_only=True,mmap=True)
        match=final_geometry(stack,parent,fit_hashes_match=fits_exact);del parent
        sizes=inventory(stack,norm,E,rotary)
        geometry=fingerprint([("stack."+n,v) for n,v in stack.named_buffers()]+[("readout",E)])
        cpu_state=fixed_state(stack,E,rotary)
        with (root/"initializer.pt").open("xb") as stream:torch.save({"stack":stack.state_dict(),"norm":norm.state_dict()},stream)
        boundary=write_boundary(root/"boundary_00.npy",read(root/"items.json")["items"],E)
        if fixed_state(stack,E,rotary)!=cpu_state:raise ValueError("Materialization changed fixed geometry/readout")
        result=dict(status="completed",fingerprint=p["fingerprint"],model=p["model"],fit_map=p["fit_map"],
            initializer=str(root/"initializer.pt"),initializer_sha256=base.sha(root/"initializer.pt"),
            boundary_path=str(root/"boundary_00.npy"),boundary_sha256=base.sha(root/"boundary_00.npy"),
            cache_root=p["cache_root"],items_path=str(root/"items.json"),item_digest=p["item_digest"],shape=p["shape"],
            teacher_logits_path=p["teacher_logits_path"],teacher_slots=p["teacher_slots"],
            final_geometry_matches_feature_teacher=match["passed"],final_geometry_control=match,
            parent_feature_receipt=None,feature_alpha=0.0,objective=p["objective"],
            diagnostic_parent_feature_receipt=p["diagnostic_parent_feature_receipt"],
            inventory=sizes,geometry_readout_fingerprint=geometry,cpu_fixed_state=cpu_state,
            cpu_rotary_fingerprint=fingerprint(list(rotary.named_buffers())),native_initialization=proof,
            boundary=boundary,source_fit_teacher_provenance=p["source_fit_teacher_provenance"],
            elapsed_seconds=time.monotonic()-started,confirmed=False,
            scope="CPU initializer and exact embedding bank; no training/accuracy/GPU memory/speed claim")
        base.write(root/"result.json",result)
        base.write(root/"manifest.json",dict(status="completed",fingerprint=p["fingerprint"],
            artifacts={name:base.sha(root/name) for name in ("initializer.pt","boundary_00.npy","items.json","result.json")}))
        return {k:result[k] for k in ("status","fingerprint","inventory","final_geometry_matches_feature_teacher","elapsed_seconds")}
    except Exception as error:
        base.write(root/"failure.json",dict(status="failed",fingerprint=p["fingerprint"],error=repr(error)));raise
    finally:gc.collect()


def checked(root):
    root=Path(root);p=verify(root);r=read(root/"result.json");m=read(root/"manifest.json")
    if (m.get("status")!="completed" or r.get("status")!="completed" or m.get("fingerprint")!=p["fingerprint"]
            or r.get("fingerprint")!=p["fingerprint"] or r.get("fit_map")!=p["fit_map"]
            or r.get("model")!=p["model"] or r.get("item_digest")!=p["item_digest"]
            or r.get("cache_root")!=p["cache_root"] or r.get("shape")!=p["shape"]
            or set(m.get("artifacts",{}))!={"initializer.pt","boundary_00.npy","items.json","result.json"}):
        raise ValueError("Completed matching initializer required")
    for name,digest in m["artifacts"].items():
        if base.sha(root/name)!=digest:raise ValueError("Initializer artifact changed")
    if (r["initializer_sha256"]!=m["artifacts"]["initializer.pt"] or r["boundary_sha256"]!=m["artifacts"]["boundary_00.npy"]
            or Path(r["initializer"]).resolve()!=(root/"initializer.pt").resolve()
            or Path(r["boundary_path"]).resolve()!=(root/"boundary_00.npy").resolve()
            or r["final_geometry_matches_feature_teacher"]!=r["final_geometry_control"]["passed"]
            or r["feature_alpha"]!=0.0 or r["parent_feature_receipt"] is not None
            or r.get("teacher_slots")!=[4,5,6,7] or r.get("confirmed") is not False):
        raise ValueError("Initializer bytes or feature-target qualification changed")
    return r


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","run","check"))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--relations-root",type=Path)
    parser.add_argument("--cache-root",type=Path,default=ROOT/"data/experiments/jspace_3b_broad_recovery_cache/20261115_v1")
    args=parser.parse_args()
    if args.phase=="prepare" and args.relations_root is None:parser.error("prepare requires --relations-root")
    value=prepare(args.root,args.relations_root,args.cache_root) if args.phase=="prepare" else run(args.root) if args.phase=="run" else checked(args.root)
    print(json.dumps(value,allow_nan=False),flush=True)
