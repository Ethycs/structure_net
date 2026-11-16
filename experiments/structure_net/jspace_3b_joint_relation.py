"""CPU-only completion of two missing TRAIN-derived polynomial interfaces.

This is a new exploratory configuration, not a repair of the old failed gate:
layer 2 uses 15 anchors admitted by its unchanged historical numerical cutoff;
layer 5 requests 24 and must independently pass that same cutoff. Existing
fits are immutable. No model, development labels or sealed audit is evaluated.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_allblocks_rankfit as fitter
from . import jspace_bench_data as base
from .jspace_incremental_suffix import file_identity, pin
from .jspace_qwen_parallel import digest as protocol_digest
from .jspace_3b_staged_campaign import local_import_closure
from .jspace_qwen_parallel_model import group_fit_path

ROOT=Path(__file__).resolve().parents[2]
SCHEMA="jspace.3b-joint-relations.v1"
STUDY="JSPACE-3B-ALL36-JOINT-RELATIONS-01"
WIDTHS={2:15,5:24}
SETTINGS=dict(remove_count=1024,pad_to_multiple=128,svd_rank=64,oversample=16,
    power_iterations=1,ridge=.01,eligibility_nrmse=.5,chunk_rows=256)
SCOPE="TRAIN-only graph coefficients/removal; historical DEV activations reported only; one fixed width per layer; no task-label or audit evaluation"
read=lambda p:json.loads(Path(p).read_text())


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU-only preparation/fitting requires CUDA hidden")


def manifest_pin(inputs,library,path,selected):
    manifest=read(path)
    if manifest.get("status")!="completed" or manifest.get("fingerprint")!=read(library/"plan.json")["fingerprint"]:
        raise ValueError("Completed matching historical producing manifest required")
    pin(inputs,path)
    for p in selected:
        relative=str(p.relative_to(library))
        expected=manifest["artifacts"].get(relative)
        if expected is None:raise ValueError("Consumed historical file absent from its producing manifest")
        pin(inputs,p,expected)
    return manifest


def check_calibration(metadata,items):
    indices=metadata["calibration_indices"]
    if (metadata.get("status")!="completed" or len(indices)!=128 or len(set(indices))!=128
            or len({items[i]["identity"] for i in indices})!=128
            or any(items[i]["split"]!="train" for i in indices)
            or Counter(items[i]["family"] for i in indices)!=dict.fromkeys(("arc","boolq","hellaswag","winogrande"),32)):
        raise ValueError("Historical calibration must be 128 distinct TRAIN cases, 32 per family")
    return dict(count=128,families=dict(Counter(items[i]["family"] for i in indices)),
        identities=[items[i]["identity"] for i in indices],split="train")


def validate_spectrum_provenance(diagnostic,interface,interface_path):
    def normalized(values):
        result={}
        for name,value in values.items():
            key=str(Path(name).resolve())
            if key in result and result[key]!=value:raise ValueError("Conflicting spectrum input aliases")
            result[key]=value
        return result
    expected=normalized(interface["input_sha256"])
    expected[str(Path(interface_path).resolve())]=base.sha(interface_path)
    if normalized(diagnostic.get("input_sha256",{}))!=expected:
        raise ValueError("Historical full-row spectrum is not bound to this exact Jacobian/interface input map")


def prepare(root,library):
    cpu_only();root,library=Path(root).resolve(),Path(library).resolve()
    if root.exists():raise FileExistsError(root)
    parent=read(library/"plan.json")
    if (parent.get("schema")!="jspace.qwen-parallel.v2" or parent.get("systems_only") is not False
            or parent["fingerprint"]!=protocol_digest({k:v for k,v in parent.items() if k!="fingerprint"})
            or any(parent["config"].get(k)!=v for k,v in dict(rank=1024,head_dim=64,anchors=24,removed_channels=1024).items())):
        raise ValueError("Original full task-Jacobian/Groebner library required")
    inputs={};pin(inputs,library/"plan.json");layers={};groups={}
    for group in range(6):
        directory=library/"groups"/f"group_{group:02d}";capture=directory/"capture"
        metadata=read(capture/"metadata.json");items=read(capture/"items.json")["items"]
        calibration=check_calibration(metadata,items)
        manifest_pin(inputs,library,capture/"manifest.json",[capture/"metadata.json",capture/"items.json"])
        basis=read(directory/"basis/manifest.json")
        groups[str(group)]=dict(teacher_fingerprint=metadata["teacher_fingerprint"],calibration=calibration,
            basis_manifest_sha256=base.sha(directory/"basis/manifest.json"),
            historical_reuse=basis.get("reused_from"))
        for layer in parent["groups"][group]:
            block=group_fit_path(library,layer)
            manifest_pin(inputs,library,directory/"basis/manifest.json",[block/"fit.pt",block/"fit.json",block/"interface.json"])
            fit,interface=read(block/"fit.json"),read(block/"interface.json")
            if (fit.get("rank")!=1024 or fit.get("head_dim")!=64 or interface.get("layer")!=layer
                    or fit["teacher_fingerprint"]!=metadata["teacher_fingerprint"]
                    or interface["teacher_fingerprint"]!=metadata["teacher_fingerprint"]):
                raise ValueError("Fitted geometry/teacher provenance differs")
            if layer in WIDTHS:
                if (block/"relation.pt").exists():raise ValueError("Expected missing historical relation now exists")
                for name,digest in interface["input_sha256"].items():pin(inputs,name,digest)
                if layer==2:
                    diag=read(block/"rank_diagnostic.json")
                    validate_spectrum_provenance(diag,interface,block/"interface.json")
                    if (diag["layer"]!=2 or diag["seed"]!=20863 or diag["teacher_fingerprint"]!=metadata["teacher_fingerprint"]
                            or diag["diagnostics"]["requested_anchors"]!=24 or diag["diagnostics"]["admitted"] is not False
                            or diag["diagnostics"]["small_svd_gate"]["rank"]!=15):
                        raise ValueError("Original layer2 rank15/24 failure must remain explicit")
                    pin(inputs,block/"rank_diagnostic.json");pin(inputs,block/"rank_spectrum.pt",diag["spectral_state_sha256"])
                destination=root/f"block_{layer:02d}"
            else:
                manifest_pin(inputs,library,block/"fit_result/manifest.json",[block/"relation.pt",block/"relations.json"])
                relation=read(block/"relations.json")
                if (relation.get("fixed_anchor_width")!=24 or relation.get("removed_channels")!=1024
                        or relation["teacher_fingerprint"]!=metadata["teacher_fingerprint"]):
                    raise ValueError("Existing relation configuration/teacher differs")
                destination=block
            layers[str(layer)]=dict(source=str(block),directory=str(destination),group=group,
                anchors=WIDTHS.get(layer,24),teacher_fingerprint=metadata["teacher_fingerprint"],
                new_relation=layer in WIDTHS)
    files=local_import_closure([Path(__file__),ROOT/"tests/structure_net/test_jspace_3b_joint_relation.py"])
    plan=dict(schema=SCHEMA,study_id=STUDY,library=str(library),library_fingerprint=parent["fingerprint"],
        model=parent["model"],settings=SETTINGS,anchor_widths={str(k):v for k,v in WIDTHS.items()},
        layers=layers,groups=groups,compute="CPU FP32; one thread; no model execution",scope=SCOPE,
        layer2_spectrum_reuse="Original full-row FP32 SVD and greedy pivot prefix; exact pooled TRAIN scale equality required",
        initializer="All parameters will be freshly projected from original native weights; no trained tail reused",
        teacher_provenance="Each six-layer group retains its historical accepted compact-suffix teacher; this is not a fresh all-native Jacobian calibration",
        confirmed=False,input_sha256=inputs,input_identity={p:file_identity(p) for p in inputs},
        source_sha256={str(p):base.sha(p) for p in sorted(files)})
    plan["fingerprint"]=protocol_digest(plan);root.mkdir(parents=True)
    for name,digest in plan["source_sha256"].items():
        target=root/"producer"/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        if base.sha(target)!=digest:raise ValueError("Producer archive changed")
    for layer in WIDTHS:
        source=Path(layers[str(layer)]["source"]);destination=Path(layers[str(layer)]["directory"]);destination.mkdir()
        for name in ("fit.pt","fit.json","interface.json"):
            shutil.copy2(source/name,destination/name)
            if base.sha(source/name)!=base.sha(destination/name):raise ValueError("Copied original geometry changed")
    base.write(root/"plan.json",plan)
    return dict(status="prepared",study_id=STUDY,fingerprint=plan["fingerprint"],new_layers=WIDTHS,existing_relations=34)


def verify(root):
    cpu_only();root=Path(root);p=read(root/"plan.json")
    if (p.get("schema")!=SCHEMA or p["fingerprint"]!=protocol_digest({k:v for k,v in p.items() if k!="fingerprint"})
            or p.get("settings")!=SETTINGS or p.get("anchor_widths")!={str(k):v for k,v in WIDTHS.items()}
            or set(p.get("layers",{}))!={str(i) for i in range(36)} or p.get("confirmed") is not False):
        raise ValueError("New relation configuration changed")
    for name,identity in p["input_identity"].items():
        if file_identity(name)!=identity:raise ValueError("Immutable historical input changed: "+name)
    for name,digest in p["source_sha256"].items():
        if base.sha(name)!=digest or base.sha(root/"producer"/Path(name).relative_to(ROOT))!=digest:
            raise ValueError("Fitting producer changed")
    for layer in WIDTHS:
        row=p["layers"][str(layer)]
        for name in ("fit.pt","fit.json","interface.json"):
            if base.sha(Path(row["directory"])/name)!=p["input_sha256"][str(Path(row["source"])/name)]:
                raise ValueError("Copied fitting interface changed")
    return p


def saved_selector(report,state,*,on_spectrum):
    """Use a previously computed TRAIN sketch only when its input scale is exact."""
    def select(rows,scale,max_anchors,*,seed,svd_rank,oversample,power_iterations,chunk_rows):
        old=report["diagnostics"]
        if (seed!=report["seed"] or svd_rank!=64 or oversample!=16 or power_iterations!=1 or chunk_rows!=256
                or max_anchors!=15 or tuple(rows.shape)!=tuple(old["original_shape"])
                or not torch.equal(scale.cpu(),state["scale"])):
            raise ValueError("Saved full-row spectrum has different TRAIN normalization or sketch settings")
        singular=state["singular"];cutoff=singular[0]*max(state["small"].shape)*torch.finfo(singular.dtype).eps
        rank=int((singular>cutoff).sum())
        if (rank!=old["small_svd_gate"]["rank"] or rank<max_anchors
                or float(cutoff)!=old["small_svd_gate"]["absolute_cutoff"]):
            raise ValueError("Unchanged numerical-rank cutoff does not admit requested anchors")
        diagnostic=deepcopy(old);diagnostic.update(requested_anchors=max_anchors,admitted=True,
            diagnostic_anchor_indices=old["diagnostic_anchor_indices"][:max_anchors],
            diagnostic_pivot_norms=old["diagnostic_pivot_norms"][:max_anchors],
            scope="New fixed15-anchor fit; unchanged original TRAIN spectrum, cutoff and greedy pivot prefix")
        on_spectrum(diagnostic,state)
        anchors=diagnostic["diagnostic_anchor_indices"]
        if len(anchors)!=15 or len(set(anchors))!=15:raise ValueError("Independent historical pivot prefix required")
        return anchors,dict(anchors=anchors,seed=seed,all_rows_used=True,
            reused_spectrum=True,original_spectrum_sha256=report["spectral_state_sha256"],
            source_requested_anchors=24,source_admitted=False,rank_admission=diagnostic,
            normalization_scale_bit_exact=True,sketch_width=80,retained_svd_components=64,
            coordinate_normalization="J multiplied by pooled training activation scale",row_normalization="none")
    return select


@torch.no_grad()
def fit_saved_layer2(directory,source,*,progress):
    interface=read(directory/"interface.json")
    training=fitter._read_bits(interface["native_calibration"],slice(None),"cpu").float()
    native_dev=fitter._read_bits(interface["native_development"],slice(None),"cpu").float()
    compiled=fitter._read_bits(interface["compiled_development"],slice(None),"cpu").float()
    weight=torch.load(interface["output_weight"],map_location="cpu",weights_only=True)
    if weight.dtype!=torch.float32 or not torch.equal(weight,weight.bfloat16().float()):
        raise ValueError("Original BF16-rounded compiled output map required")
    original=read(source/"rank_diagnostic.json")
    state=torch.load(source/"rank_spectrum.pt",map_location="cpu",weights_only=True,mmap=True)
    def save_spectrum(diagnostic,spectrum):
        shutil.copy2(source/"rank_spectrum.pt",directory/"rank_spectrum.pt")
        base.write(directory/"rank_diagnostic.json",dict(layer=2,seed=20863,
            teacher_fingerprint=interface["teacher_fingerprint"],diagnostics=diagnostic,
            spectral_state_sha256=base.sha(directory/"rank_spectrum.pt"),
            reused_from=str(source/"rank_spectrum.pt"),source_diagnostic_sha256=base.sha(source/"rank_diagnostic.json")))
    with fitter._ProgressRows(interface["jacobian"],device="cpu",max_rows=256,progress=progress,layer=2,guard=None) as rows:
        result=fitter.fit_benchmark_groebner(training,interface["calibration_families"],rows,weight,
            native_dev,interface["development_families"],compressed_dev_activations=compiled,
            anchor_widths=(15,),seed=20863,selector=saved_selector(original,state,on_spectrum=save_spectrum),
            **{k:v for k,v in SETTINGS.items() if k!="pad_to_multiple"})
        streamed=rows.metadata()
    fitter._save(directory/"relation.pt",result["fits"]["15"])
    report={k:v for k,v in result.items() if k!="fits"}
    logical=training.shape[1]-1024+136
    report.update(layer=2,teacher_fingerprint=interface["teacher_fingerprint"],fixed_anchor_width=15,
        removed_channels=1024,runtime_padding=dict(multiple=128,logical_width=logical,padded_width=math.ceil(logical/128)*128),
        training_families=dict(Counter(interface["calibration_families"])),development_families=dict(Counter(interface["development_families"])),
        relation_sha256=base.sha(directory/"relation.pt"),rank_diagnostic_sha256=base.sha(directory/"rank_diagnostic.json"),
        rank_spectrum_sha256=base.sha(directory/"rank_spectrum.pt"),streamed_jacobian=streamed,
        input_sha256={**interface["input_sha256"],str(directory/"interface.json"):base.sha(directory/"interface.json")},
        source="Historical accepted-teacher full-row J spectrum, original native TRAIN activations, fresh fixed15-anchor graph fit",scope=SCOPE)
    base.write(directory/"relations.json",report)
    return report


def run(root,layer):
    root=Path(root).resolve();p=verify(root)
    if layer not in WIDTHS:raise ValueError("Only missing layers2 and5 may be fitted")
    row=p["layers"][str(layer)];directory=Path(row["directory"])
    if any((directory/name).exists() for name in ("started.json","manifest.json","failure.json","relation.pt")):
        raise FileExistsError("An attempted relation fit is never overwritten or retried")
    base.write(directory/"started.json",dict(fingerprint=p["fingerprint"],layer=layer,pid=os.getpid()))
    torch.set_num_threads(1);started=time.monotonic()
    def progress(value):
        with (directory/"progress.jsonl").open("a") as stream:stream.write(json.dumps(value,allow_nan=False)+"\n")
        print(json.dumps(value,allow_nan=False),flush=True)
    try:
        progress(dict(phase="cpu_fit_start",layer=layer,anchors=WIDTHS[layer]))
        result=fit_saved_layer2(directory,Path(row["source"]),progress=progress) if layer==2 else fitter.fit_relation(
            directory,device="cpu",seed=20861+layer,anchors=WIDTHS[layer],progress=progress,**SETTINGS)
        diagnostic=read(directory/"rank_diagnostic.json")["diagnostics"]
        if not diagnostic["admitted"] or result["fixed_anchor_width"]!=WIDTHS[layer] or result["removed_channels"]!=1024:
            raise ValueError("Configured rank/removal budget did not complete")
        receipt=dict(status="completed",fingerprint=p["fingerprint"],layer=layer,anchors=WIDTHS[layer],
            removed_channels=1024,elapsed_seconds=time.monotonic()-started,compute="CPU FP32",confirmed=False,scope=SCOPE)
        base.write(directory/"result.json",receipt)
        base.write(directory/"manifest.json",dict(status="completed",fingerprint=p["fingerprint"],layer=layer,
            artifacts={f.name:base.sha(f) for f in directory.iterdir() if f.is_file()}))
        return receipt
    except Exception as error:
        base.write(directory/"failure.json",dict(status="failed",fingerprint=p["fingerprint"],layer=layer,error=repr(error)));raise
    finally:gc.collect()


def finalize(root):
    root=Path(root).resolve();p=verify(root)
    for layer in WIDTHS:
        directory=Path(p["layers"][str(layer)]["directory"]);m=read(directory/"manifest.json")
        if m.get("status")!="completed" or m.get("fingerprint")!=p["fingerprint"] or m.get("layer")!=layer:
            raise ValueError("Both new relations must complete")
        for name,digest in m["artifacts"].items():
            if base.sha(directory/name)!=digest:raise ValueError("New relation artifact changed")
    fit_map={k:v["directory"] for k,v in p["layers"].items()}
    result=dict(status="completed",fingerprint=p["fingerprint"],study_id=STUDY,fit_map=fit_map,
        anchor_widths={k:v["anchors"] for k,v in p["layers"].items()},rank=1024,head_dim=64,removed_channels=1024,
        teacher_provenance=p["teacher_provenance"],initializer=p["initializer"],confirmed=False,scope=SCOPE)
    base.write(root/"result.json",result);base.write(root/"manifest.json",dict(status="completed",fingerprint=p["fingerprint"],
        artifacts={"result.json":base.sha(root/"result.json"),**{f"block_{i:02}/manifest.json":base.sha(root/f"block_{i:02}/manifest.json") for i in WIDTHS}}))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","fit","finalize"))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--layer",type=int,choices=tuple(WIDTHS))
    parser.add_argument("--library",type=Path,default=ROOT/"data/experiments/jspace_qwen_parallel/20261114_v2")
    args=parser.parse_args()
    if args.phase=="fit" and args.layer is None:parser.error("fit requires --layer")
    value=prepare(args.root,args.library) if args.phase=="prepare" else run(args.root,args.layer) if args.phase=="fit" else finalize(args.root)
    print(json.dumps(value,allow_nan=False),flush=True)
