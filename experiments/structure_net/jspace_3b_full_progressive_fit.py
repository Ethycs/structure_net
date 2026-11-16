"""Continue fresh response-calibrated compression from the one-block seed to36.

Quality and exploratory continuation are separate recorded decisions. The old
fit arithmetic is reused unchanged; old two-block admission is not patched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_3b_full_progressive_data as dataset
from . import jspace_3b_rebuild_fit as fitting
from .jspace_3b_rebuild_fit_campaign import _pin

ROOT=Path(__file__).resolve().parents[2]
SCHEMA="jspace.3b-full-progressive-fit.v1"
SEED_SCHEMA="jspace.3b-full-progressive-seed.v1"
RECOVERY_SCHEMA="jspace.3b-full-progressive-recovery.v1"
read,write,sha,identity=dataset.read,dataset.write,dataset.sha,dataset.identity


def _cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Preparation requires CUDA hidden")


def _plan(path):
    p=read(path);body=dict(p);fp=body.pop("fingerprint")
    if dataset.data.digest(body)!=fp:raise ValueError("Plan fingerprint changed")
    return p


def _record_identity(plan,data_plan):
    if (plan.get("model")!=data_plan["model"]
            or plan.get("corpus_fingerprint",plan.get("records_sha256"))!=data_plan["records_sha256"]):
        raise ValueError("Parent differs from the exact native model or response records")


def _sources():
    files=dataset.cache.prior.source.inherited.local_import_closure([Path(__file__),dataset.PROTOCOL,
        ROOT/"tests/structure_net/test_jspace_3b_full_progressive_fit.py"])
    return {str(path):sha(path) for path in sorted(files)}


def _archive(root,sources):
    for path,digest in sources.items():
        target=root/"producer"/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if sha(target)!=digest:raise ValueError("Full progressive fit source archive changed")


def _fit_inputs(fits,source_plan,inputs):
    for directory in fits.values():
        for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):
            path=Path(directory)/name;expected=source_plan["input_sha256"].get(str(path.resolve()))
            # Older plans authenticated every new fit artifact but only the
            # inherited compiler tensors. All newly frozen wrapper plans pin
            # the full descriptive inventory too.
            if expected is None and name in ("fit.pt","relation.pt"):
                raise ValueError("Inherited compiler input lacks an authenticated content pin")
            _pin(inputs,path,expected)


def _completion(root,p,*,gpu=False):
    result_name="gpu_result.json" if gpu else "result.json"
    manifest_name="gpu_manifest.json" if gpu else "manifest.json"
    r=read(root/"run"/result_name);m=read(root/"run"/manifest_name)
    if (r.get("status")!="completed" or r.get("fingerprint")!=p["fingerprint"]
            or m.get("status")!="completed" or m.get("fingerprint")!=p["fingerprint"]
            or m.get("artifacts",{}).get(result_name)!=sha(root/"run"/result_name)):
        raise ValueError("Completed parent run/result manifest required")
    if gpu and r.get("systems_passed") is not True:
        raise ValueError("Parent GPU systems checks must pass before exploratory continuation")
    return r,m


def seed(root,data_root,source_root,checkpoint):
    """Authenticate the existing v2 layer35/update46; do not refit or relabel it."""
    _cpu();root,data_root,source_root,checkpoint=map(lambda p:Path(p).resolve(),(root,data_root,source_root,checkpoint))
    if root.exists():raise FileExistsError(root)
    dp,_=dataset.verify(data_root,live=False);p=_plan(source_root/"plan.json");_record_identity(p,dp)
    if p.get("schema")!="jspace.3b-progressive-response-recovery.v1" or p.get("cut")!=35:
        raise ValueError("The original one-block response-rebuild recovery is the only seed")
    old_data=_plan(Path(p["data_root"])/"plan.json")
    if (old_data["model"]!=dp["model"] or old_data["records_sha256"]!=dp["records_sha256"]
            or old_data["rows"]!=dp["rows"] or old_data["calibration_indices"]!=dp["calibration_indices"]
            or old_data["diagnostic_indices"]!=dp["diagnostic_indices"]):
        raise ValueError("New native bank does not preserve the seed's exact records and fitting cases")
    result,manifest=_completion(source_root,p)
    matches=[c for c in result.get("candidates",[]) if c.get("checkpoint",{}).get("path")==str(checkpoint)]
    if len(matches)!=1 or matches[0].get("update")!=46 or checkpoint.name!="update_0046.pt":
        raise ValueError("Explicit existing update46 must be present in the completed seed results")
    candidate=matches[0]
    if (not candidate.get("packed_reload_passed") or not candidate.get("packed_reload",{}).get("packed_state_exact")):
        raise ValueError("Seed packed-state/reload systems evidence is missing")
    inputs={};_pin(inputs,source_root/"plan.json");_pin(inputs,source_root/"run/result.json");_pin(inputs,source_root/"run/manifest.json")
    _pin(inputs,Path(p["data_root"])/"plan.json");_pin(inputs,data_root/"plan.json")
    _pin(inputs,checkpoint,candidate["checkpoint"]["sha256"])
    if manifest.get("checkpoints",{}).get(str(checkpoint))!=inputs[str(checkpoint)]:
        raise ValueError("Seed checkpoint differs from completed checkpoint inventory")
    fits={str(k):str(Path(v).resolve()) for k,v in p["fits"].items()}
    if set(fits)!={"35"}:raise ValueError("Seed must contain only compressed layer35")
    _fit_inputs(fits,p,inputs)
    value=dict(schema=SEED_SCHEMA,status="completed",cut=35,data_root=str(data_root),data_fingerprint=dp["fingerprint"],
        records_sha256=dp["records_sha256"],model=dp["model"],source_root=str(source_root),source_fingerprint=p["fingerprint"],
        source_data_fingerprint=p["data_fingerprint"],checkpoint=str(checkpoint),checkpoint_sha256=inputs[str(checkpoint)],fits=fits,
        relative_retention_pass=bool(result.get("relative_retention_pass")),
        source_quality=dict(nll=candidate.get("nll"),mc_accuracy=candidate.get("mc_accuracy"),
            cheap_gate=candidate.get("cheap_gate"),eligibility=candidate.get("eligibility"),grading_status=candidate.get("grading_status")),
        input_sha256=inputs,input_identity={path:identity(path) for path in inputs},
        source_sha256=_sources(),
        continuation_reason="User-authorized full36 exploratory progression; existing seed NLL failure retained, not reclassified",
        confirmed=False)
    value["fingerprint"]=dataset.data.digest(value);root.mkdir(parents=True);write(root/"seed.json",value);_archive(root,value["source_sha256"])
    return value


def continuation_parent(root,cut,data_plan,*,checkpoint=None):
    root=Path(root).resolve();inputs={}
    if (root/"seed.json").exists():
        p=_plan(root/"seed.json")
        if p.get("schema")!=SEED_SCHEMA or p.get("status")!="completed" or cut!=34:
            raise ValueError("The one-block seed can only feed cut34")
        _record_identity(p,data_plan)
        if p["data_fingerprint"]!=data_plan["fingerprint"]:raise ValueError("Seed belongs to another full data plan")
        for path,old in p["input_identity"].items():
            if identity(path)!=old:raise ValueError("Seed input changed: "+path)
        for path,digest in p["source_sha256"].items():
            if sha(root/"producer"/Path(path).relative_to(ROOT))!=digest:raise ValueError("Archived seed producer changed")
        inputs.update(p["input_sha256"]);_pin(inputs,root/"seed.json")
        chosen=p["checkpoint"];chosen_sha=p["checkpoint_sha256"];fits=p["fits"]
        quality=p["relative_retention_pass"];quality_detail=p["source_quality"]
    else:
        p=_plan(root/"plan.json");_record_identity(p,data_plan)
        if (p.get("schema")!=RECOVERY_SCHEMA or p.get("cut")!=cut+1
                or p.get("data_fingerprint")!=data_plan["fingerprint"]):
            raise ValueError("An adjacent completed full-progressive stage on identical data is required")
        result,manifest=_completion(root,p,gpu=True)
        chosen=result.get("continuation_checkpoint");chosen_sha=result.get("continuation_checkpoint_sha256")
        if not chosen or not chosen_sha:raise ValueError("Parent has no recorded exploratory continuation checkpoint")
        if manifest.get("checkpoints",{}).get(chosen)!=chosen_sha:
            raise ValueError("Continuation checkpoint differs from completed checkpoint inventory")
        fits=p["fits"];quality=result.get("relative_retention_pass")
        quality_detail=dict(outcome=result.get("outcome"),relative_retention_pass=quality,
                            quality_status=result.get("quality_status","pending"),
                            continuation_selection=result.get("continuation_selection"))
        for name in ("plan.json","run/gpu_result.json","run/gpu_manifest.json"):_pin(inputs,root/name)
        _fit_inputs(fits,p,inputs)
    chosen=str(Path(chosen).resolve())
    if checkpoint is not None and chosen!=str(Path(checkpoint).resolve()):
        raise ValueError("Requested parent differs from the frozen continuation selection")
    _pin(inputs,chosen,chosen_sha)
    if set(fits)!={str(i) for i in range(cut+1,36)}:raise ValueError("Parent fit map is not the exact adjacent suffix")
    return dict(root=str(root),fingerprint=p["fingerprint"],checkpoint=chosen,checkpoint_sha256=chosen_sha,
        fits=fits,relative_retention_pass=quality,quality=quality_detail,
        admission="Completed authenticated exploratory parent; quality failure recorded and does not stop progression"),inputs


def prepare(root,data_root,*,cut,parent_root,parent_checkpoint=None):
    _cpu();root,data_root=Path(root).resolve(),Path(data_root).resolve()
    if root.exists():raise FileExistsError(root)
    if type(cut)is not int or not 0<=cut<=34:raise ValueError("Continue existing seed through cuts34..0; never refit35")
    dp,records=dataset.verify(data_root,live=False)
    if cut not in dp["cuts"]:raise ValueError("Native cut was not included in the full progressive data plan")
    fitting.calibration_cases(records,dp["calibration_indices"],dp["diagnostic_indices"],production=True)
    parent,inputs=continuation_parent(parent_root,cut,dp,checkpoint=parent_checkpoint)
    _pin(inputs,data_root/"plan.json");inputs.update(dp["input_sha256"])
    inputs.update(dataset._manifest_files(data_root/"native",dp["fingerprint"],hash_payloads=False))
    native=read(data_root/"native/result.json")
    if not all(native.get(k) for k in ("full_teacher_bit_exact","existing_boundary_bit_exact","teacher_unchanged")):
        raise ValueError("Original native bank systems controls failed")
    if (native.get("seed_boundary35_bit_exact") is not True
            or native.get("seed_data_fingerprint")!=dp["seed_data"]["fingerprint"]):
        raise ValueError("Full native bank does not reproduce the authenticated seed boundary35")
    config=asdict(fitting.FitConfig())
    teacher=dataset.data.digest(dict(native_model=dp["model"],records=dp["records_sha256"],cut=cut,
        calibration=dp["calibration_indices"],diagnostics=dp["diagnostic_indices"],parent=parent,configuration=config))
    p=dict(schema=SCHEMA,data_root=str(data_root),data_fingerprint=dp["fingerprint"],records_sha256=dp["records_sha256"],
        model=dp["model"],cut=cut,parent=parent,configuration=config,teacher_fingerprint=teacher,
        calibration_indices=dp["calibration_indices"],diagnostic_indices=dp["diagnostic_indices"],
        allocation_mib=11264,limit_mib=11008,input_sha256=inputs,input_identity={path:identity(path) for path in inputs},
        source_sha256=_sources(),confirmed=False,
        admission="Exact existing arithmetic and full-response geometry; all depths34..0; no quality-gate stop")
    p["fingerprint"]=dataset.data.digest(p);root.mkdir(parents=True);write(root/"plan.json",p)
    _archive(root,p["source_sha256"])
    return dict(status="prepared",fingerprint=p["fingerprint"],cut=cut,parent_quality_pass=parent["relative_retention_pass"])


def verify(root,*,live=True):
    root=Path(root).resolve();p=_plan(root/"plan.json")
    if (p.get("schema")!=SCHEMA or type(p["cut"])is not int or not 0<=p["cut"]<=34
            or p["configuration"]!=asdict(fitting.FitConfig()) or p["allocation_mib"]!=11264 or p["limit_mib"]!=11008):
        raise ValueError("Frozen full progressive fitting admission changed")
    for path,old in p["input_identity"].items():
        if identity(path)!=old:raise ValueError("Frozen full progressive fit input changed: "+path)
    for path,digest in p["source_sha256"].items():
        if sha(root/"producer"/Path(path).relative_to(ROOT))!=digest or live and sha(path)!=digest:
            raise ValueError("Full progressive fit producer changed: "+path)
    return p


def fit_stage(root,data,*,plan,guard,device="cuda:0"):
    """Strict full-depth admission around the frozen generic mathematical path."""
    if plan["configuration"]!=asdict(fitting.FitConfig()) or not 0<=plan["cut"]<=34:
        raise ValueError("Only the frozen full-progressive fit recipe is admitted")
    fitting.calibration_cases(data.records,plan["calibration_indices"],plan["diagnostic_indices"],production=True)
    parent=plan["parent"]
    return fitting.fit_stage(root,data,model_path=plan["model"]["path"],cut=plan["cut"],
        calibration_indices=plan["calibration_indices"],diagnostic_indices=plan["diagnostic_indices"],
        teacher_fingerprint=plan["teacher_fingerprint"],parent_fits={int(k):v for k,v in parent["fits"].items()},
        parent_checkpoint=parent["checkpoint"],device=device,guard=guard,config=fitting.FitConfig(),production=False)


def run(root,*,limit_mib=11008):
    root=Path(root).resolve();p=verify(root)
    if limit_mib!=p["limit_mib"]:raise ValueError("Allocator differs from frozen plan")
    if any((root/name).exists() for name in ("work","result.json","failure.json")):raise FileExistsError("Attempt already started")
    started=time.monotonic();lease=None;data=None
    try:
        lease,guard=dataset.cache.reservation(limit_mib)
        if next(iter(lease["allocation"].values()))!=p["allocation_mib"]:raise ValueError("Declared reservation required")
        torch.set_num_threads(2);data=dataset.RebuildDataset(p["data_root"],cut=p["cut"])
        with dataset.cache.execution.arithmetic_policy():receipt=fit_stage(root/"work",data,plan=p,guard=guard)
        guard()
        result=dict(schema=SCHEMA,status="completed",fingerprint=p["fingerprint"],cut=p["cut"],lease_id=lease["id"],
            fit_directory=receipt["fit_directory"],fit_receipt=str(root/"work/fit_receipt.json"),
            fit_receipt_sha256=sha(root/"work/fit_receipt.json"),
            output_identity={str(path.relative_to(root)):identity(path) for path in (root/"work").rglob("*") if path.is_file()},
            elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),parent_quality_pass=p["parent"]["relative_retention_pass"],confirmed=False)
        write(root/"result.json",result);return result
    except Exception as error:
        write(root/"failure.json",dict(status="failed",fingerprint=p["fingerprint"],error_type=type(error).__name__,error=str(error),
            lease_id=lease["id"] if lease else None,elapsed_seconds=time.monotonic()-started,confirmed=False));raise
    finally:
        if data is not None:data.close()


def check(root):
    root=Path(root).resolve();p=verify(root,live=False)
    if (root/"failure.json").exists():return read(root/"failure.json")
    if not (root/"result.json").exists():return dict(status="prepared",fingerprint=p["fingerprint"])
    r=read(root/"result.json")
    if r.get("status")!="completed" or r.get("fingerprint")!=p["fingerprint"]:raise ValueError("Result/plan mismatch")
    if sha(r["fit_receipt"])!=r["fit_receipt_sha256"]:raise ValueError("Fit receipt changed")
    for name,old in r["output_identity"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root/"work") or identity(path)!=old:raise ValueError("Completed fit artifact changed")
    receipt=read(r["fit_receipt"])
    if (receipt.get("cut")!=p["cut"] or receipt.get("configuration")!=p["configuration"]
            or receipt.get("teacher_fingerprint")!=p["teacher_fingerprint"]
            or receipt.get("parent_checkpoint_sha256")!=p["parent"]["checkpoint_sha256"]
            or not receipt.get("native_seam_bit_exact") or not receipt.get("saved_fit_compilation_passed")):
        raise ValueError("Fresh fit does not match admitted full-progressive stage")
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("seed")
    for name in ("root","data-root","source-root","checkpoint"):p.add_argument("--"+name,required=True)
    p=sub.add_parser("prepare")
    for name in ("root","data-root","parent-root"):p.add_argument("--"+name,required=True)
    p.add_argument("--cut",type=int,required=True);p.add_argument("--parent-checkpoint")
    p=sub.add_parser("run");p.add_argument("--root",required=True);p.add_argument("--limit-mib",type=int,default=11008)
    p=sub.add_parser("check");p.add_argument("--root",required=True)
    args=vars(parser.parse_args());command=args.pop("command")
    print(json.dumps({"seed":seed,"prepare":prepare,"run":run,"check":check}[command](**args),sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
