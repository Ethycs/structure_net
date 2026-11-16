"""Frozen prepare/run/check boundaries for the fresh progressive response fit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_3b_rebuild_data as dataset
from . import jspace_3b_rebuild_fit as fitting

SCHEMA="jspace.3b-response-rebuild-fit-campaign.v1"
ROOT=Path(__file__).resolve().parents[2]
read,write,sha,identity=dataset.read,dataset.write,dataset.sha,dataset.identity


def _pin(inputs,path,expected=None):
    path=Path(path).resolve();actual=sha(path)
    if expected is not None and actual!=expected:raise ValueError("Authenticated fit input changed: "+str(path))
    inputs[str(path)]=actual
    return path


def accepted_parent(parent_root,cut,*,checkpoint=None,data_fingerprint=None):
    """Only a completed, eligible adjacent recovery checkpoint may propagate."""
    parent_root=Path(parent_root).resolve();p=read(parent_root/"plan.json");r=read(parent_root/"run/result.json")
    body=dict(p);fp=body.pop("fingerprint")
    if dataset.data.digest(body)!=fp or r.get("fingerprint")!=fp:
        raise ValueError("Parent recovery plan/result fingerprint differs")
    if (p.get("schema")!="jspace.3b-progressive-response-recovery.v1" or p.get("cut")!=cut+1 or r.get("status")!="completed"
            or data_fingerprint is not None and p.get("data_fingerprint")!=data_fingerprint
            or r.get("relative_retention_pass") is not True):
        raise ValueError("An accepted adjacent packed-evaluated parent is required")
    path=Path(r["selected_checkpoint"])
    path=(parent_root/path).resolve() if not path.is_absolute() else path.resolve()
    if checkpoint is not None and path!=Path(checkpoint).resolve():
        raise ValueError("Requested checkpoint is not the accepted parent selection")
    manifest=read(parent_root/"run/manifest.json")
    if (manifest.get("status")!="completed" or manifest.get("fingerprint")!=fp
            or manifest.get("artifacts",{}).get("result.json")!=sha(parent_root/"run/result.json")):
        raise ValueError("Accepted parent completion manifest differs")
    inputs={};_pin(inputs,parent_root/"plan.json");_pin(inputs,parent_root/"run/result.json");_pin(inputs,parent_root/"run/manifest.json")
    _pin(inputs,path,r["selected_checkpoint_sha256"])
    fits={str(int(k)):str(Path(v).resolve()) for k,v in p["fits"].items()}
    if set(fits)!={str(i) for i in range(cut+1,36)}:
        raise ValueError("Accepted parent lacks an exact contiguous fit map")
    for directory in fits.values():
        for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):
            fit_path=Path(directory)/name
            expected=p.get("input_sha256",{}).get(str(fit_path.resolve()))
            if expected is None:raise ValueError("Parent fit lacks its recovery-plan content pin")
            _pin(inputs,fit_path,expected)
    return dict(root=str(parent_root),fingerprint=fp,checkpoint=str(path),
                checkpoint_sha256=inputs[str(path)],fits=fits),inputs


def _parent_root_from_checkpoint(checkpoint):
    checkpoint=Path(checkpoint).resolve()
    for parent in list(checkpoint.parents)[:3]:
        if (parent/"plan.json").is_file() and (parent/"run/result.json").is_file():return parent
    raise ValueError("Provide --parent-root containing the accepted recovery plan/result")


def prepare(root,data_root,*,cut,parent_root=None,parent_checkpoint=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU preparation requires CUDA hidden")
    root,data_root=Path(root).resolve(),Path(data_root).resolve()
    if root.exists():raise FileExistsError(root)
    if cut not in (35,34):raise ValueError("Only native-first35 and accepted-adjacent34 are admitted")
    p,records=dataset.verify(data_root,live=False)
    if cut not in p["cuts"]:raise ValueError("Requested native boundary was not prepared")
    fitting.calibration_cases(records,p["calibration_indices"],p["diagnostic_indices"])
    inputs=dict(p["input_sha256"])
    _pin(inputs,data_root/"plan.json")
    # Native capture already hashed its large banks. Bind that completed
    # manifest plus each unchanged file identity, avoiding repeated bulk reads.
    inputs.update(dataset._manifest_files(data_root/"native",p["fingerprint"],hash_payloads=False))
    native=read(data_root/"native/result.json")
    if not all(native.get(k) for k in ("full_teacher_bit_exact","existing_boundary_bit_exact","teacher_unchanged")):
        raise ValueError("Native boundary/teacher controls failed")
    parent=None
    if cut==35:
        if parent_root is not None or parent_checkpoint is not None:
            raise ValueError("Stage35 must bootstrap from the original native model")
    else:
        if parent_root is None and parent_checkpoint is not None:parent_root=_parent_root_from_checkpoint(parent_checkpoint)
        if parent_root is None:raise ValueError("Stage34 needs a completed accepted stage35 parent")
        parent,parent_inputs=accepted_parent(parent_root,cut,checkpoint=parent_checkpoint,data_fingerprint=p["fingerprint"]);inputs.update(parent_inputs)
    configuration=asdict(fitting.FitConfig())
    teacher=dataset.data.digest(dict(native_model=p["model"],records=p["records_sha256"],cut=cut,
        calibration=p["calibration_indices"],diagnostics=p["diagnostic_indices"],parent=parent,configuration=configuration))
    sources=dataset.cache.prior.source.inherited.local_import_closure([Path(__file__),dataset.PROTOCOL,
        ROOT/"tests/structure_net/test_jspace_3b_rebuild_fit.py",
        ROOT/"tests/structure_net/test_jspace_3b_rebuild_fit_campaign.py"])
    plan=dict(schema=SCHEMA,data_root=str(data_root),data_fingerprint=p["fingerprint"],cut=cut,model=p["model"],
        calibration_indices=p["calibration_indices"],diagnostic_indices=p["diagnostic_indices"],
        configuration=configuration,teacher_fingerprint=teacher,parent=parent,allocation_mib=11264,limit_mib=11008,
        input_sha256=inputs,input_identity={name:identity(name) for name in inputs},
        source_sha256={str(path):sha(path) for path in sorted(sources)},confirmed=False,
        scope="Fresh full-response functional/head/MLP fit; no recovery or quality claim")
    plan["fingerprint"]=dataset.data.digest(plan)
    root.mkdir(parents=True);write(root/"plan.json",plan)
    for path,digest in plan["source_sha256"].items():
        target=root/"producer"/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(target)!=digest:raise ValueError("Fit producer archive mismatch")
    return dict(status="prepared",fingerprint=plan["fingerprint"],cut=cut,teacher_fingerprint=teacher)


def verify(root,*,live=True):
    root=Path(root).resolve();p=read(root/"plan.json");body=dict(p);fp=body.pop("fingerprint")
    if p.get("schema")!=SCHEMA or dataset.data.digest(body)!=fp:raise ValueError("Fit plan changed")
    if (p["configuration"]!=asdict(fitting.FitConfig()) or p["cut"] not in (35,34)
            or p["allocation_mib"]!=11264 or p["limit_mib"]!=11008):
        raise ValueError("Unadmitted scientific fit configuration")
    for path,old in p["input_identity"].items():
        if identity(path)!=old:raise ValueError("Frozen fit input changed: "+path)
    for path,digest in p["source_sha256"].items():
        if sha(root/"producer"/Path(path).relative_to(ROOT))!=digest or live and sha(path)!=digest:
            raise ValueError("Frozen fit producer changed: "+path)
    return p


def run(root,*,limit_mib=11008):
    root=Path(root).resolve();p=verify(root)
    if limit_mib!=p["limit_mib"]:raise ValueError("Allocator differs from the frozen fit plan")
    if any((root/name).exists() for name in ("result.json","failure.json","work")):
        raise FileExistsError("Fit attempt has already started")
    started=time.monotonic();lease=None;data=None
    try:
        lease,guard=dataset.cache.reservation(limit_mib)
        if next(iter(lease["allocation"].values()))!=p["allocation_mib"]:
            raise ValueError("Fit worker lacks its declared11264MiB allocation")
        torch.set_num_threads(2)
        data=dataset.RebuildDataset(p["data_root"],cut=p["cut"])
        parent=p["parent"]
        with dataset.cache.execution.arithmetic_policy():
            receipt=fitting.fit_stage(root/"work",data,model_path=p["model"]["path"],cut=p["cut"],
                calibration_indices=p["calibration_indices"],diagnostic_indices=p["diagnostic_indices"],
                teacher_fingerprint=p["teacher_fingerprint"],
                parent_fits={int(k):v for k,v in parent["fits"].items()} if parent else None,
                parent_checkpoint=parent["checkpoint"] if parent else None,device="cuda:0",guard=guard,
                config=fitting.FitConfig(**p["configuration"]))
        guard()
        files={str(path.relative_to(root)):identity(path) for path in (root/"work").rglob("*") if path.is_file()}
        result=dict(schema=SCHEMA,status="completed",fingerprint=p["fingerprint"],cut=p["cut"],
            lease_id=lease["id"],fit_directory=receipt["fit_directory"],
            fit_receipt=str(root/"work/fit_receipt.json"),fit_receipt_sha256=sha(root/"work/fit_receipt.json"),
            output_identity=files,elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            confirmed=False)
        write(root/"result.json",result);return result
    except Exception as error:
        write(root/"failure.json",dict(schema=SCHEMA,status="failed",fingerprint=p["fingerprint"],
            lease_id=lease["id"] if lease else None,error_type=type(error).__name__,error=str(error),
            elapsed_seconds=time.monotonic()-started,confirmed=False))
        raise
    finally:
        if data is not None:data.close()


def check(root):
    root=Path(root).resolve();p=verify(root,live=False)
    if (root/"failure.json").exists():return read(root/"failure.json")
    if not (root/"result.json").exists():return dict(status="prepared",fingerprint=p["fingerprint"])
    r=read(root/"result.json")
    if r.get("fingerprint")!=p["fingerprint"] or r.get("status")!="completed":raise ValueError("Fit result differs from plan")
    if sha(r["fit_receipt"])!=r["fit_receipt_sha256"]:raise ValueError("Completed fit receipt changed")
    for name,old in r["output_identity"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root/"work") or identity(path)!=old:raise ValueError("Completed fit artifact changed")
    receipt=read(r["fit_receipt"])
    if (receipt.get("status")!="completed" or receipt.get("teacher_fingerprint")!=p["teacher_fingerprint"]
            or receipt.get("configuration")!=p["configuration"] or receipt.get("cut")!=p["cut"]
            or not receipt.get("native_seam_bit_exact") or not receipt.get("saved_fit_compilation_passed")):
        raise ValueError("Completed fit controls differ from the prospective plan")
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    prepare_parser=sub.add_parser("prepare");prepare_parser.add_argument("--root",required=True)
    prepare_parser.add_argument("--data-root",required=True);prepare_parser.add_argument("--cut",type=int,choices=(35,34),required=True)
    prepare_parser.add_argument("--parent-root");prepare_parser.add_argument("--parent-checkpoint")
    run_parser=sub.add_parser("run");run_parser.add_argument("--root",required=True);run_parser.add_argument("--limit-mib",type=int,default=11008)
    check_parser=sub.add_parser("check");check_parser.add_argument("--root",required=True)
    args=vars(parser.parse_args());command=args.pop("command")
    result={"prepare":prepare,"run":run,"check":check}[command](**args)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
