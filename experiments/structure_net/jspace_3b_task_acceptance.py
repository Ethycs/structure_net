"""Explicit task-based checkpoint acceptance, separate from cautious selection."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_3b_feature_recovery as feature
from . import jspace_3b_resident_campaign as resident

source=feature.source
base,old,read,training=feature.base,feature.old,feature.read,feature.training
SCHEMA="jspace.3b-feature-seed-verification.v1"
POLICY="highest_correct_then_lowest_ce; baseline_then_scientific_then_best_task_on_exact_ties"
PROTOCOL=base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-parent-feature-resident-continuation.md"


def native_gate(metrics):
    if (type(metrics.get("correct")) is not int or metrics.get("count")!=64 or
            not 0<=metrics["correct"]<=64 or not all(math.isfinite(metrics[k]) for k in ("ce","original_kl")) or
            any(isinstance(value,(int,float)) and not math.isfinite(value) for value in metrics.values())):
        raise ValueError("Finite matched64-case task metrics required")
    return dict(passed=metrics["correct"]>=45,native_correct=48,minimum_correct=45,count=64,maximum_drop=.05,
        ce_kl_increase_is_veto=False)


def choose(candidates):
    """Order is prospective; Python max preserves the first exact score/CE tie."""
    if not candidates or len({c["kind"] for c in candidates})!=len(candidates):
        raise ValueError("Distinct ordered candidates required")
    for candidate in candidates:native_gate(candidate["metrics"])
    winner=max(candidates,key=lambda c:training.selection_key(c["metrics"]))
    quality=native_gate(winner["metrics"])
    return dict(accepted=quality["passed"],candidate=winner,quality=quality,policy=POLICY)


def descriptor(kind,directory,trained,*,diagnostic=False):
    directory=Path(directory)
    if diagnostic:
        saved=read(directory/"diagnostic_best_task.json");epoch=saved["epoch"]
        checkpoint=directory/"diagnostic_best_task.pt";digest=saved["checkpoint_sha256"];metrics=saved["metrics"]
        task_winner=max(trained["history"],key=lambda row:training.selection_key(row["metrics"]))
        if (saved.get("diagnostic_only") is not True or saved.get("automatically_adopted") is not False or
                epoch!=task_winner["epoch"] or metrics!=task_winner["metrics"]):
            raise ValueError("Saved task candidate does not match actual epoch history")
    else:
        epoch=trained["selected_epoch"];checkpoint=directory/"selected.pt"
        digest=trained["selected_checkpoint_sha256"];metrics=trained["selected"]
    report=directory/f"epoch_{epoch:02d}_development.json"
    if base.sha(checkpoint)!=digest or read(report)["metrics"]!=metrics:
        raise ValueError("Candidate checkpoint/epoch metrics changed")
    row=next(r for r in trained["history"] if r["epoch"]==epoch)
    if row["development_sha256"]!=base.sha(report):raise ValueError("Candidate epoch records changed")
    source.validate_quality(read(report));native_gate(metrics)
    return dict(kind=kind,epoch=epoch,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest,
        report=str(report.resolve()),report_sha256=base.sha(report),metrics=metrics)


def seed_candidates(seed_root,feature_root):
    seed_root,feature_root=Path(seed_root).resolve(),Path(feature_root).resolve()
    parent,stage=source.verify_stage(seed_root,23);directory=source.stage_dir(seed_root,23)
    baseline=source.checked(directory,"train",stage);baseline_verified=source.checked(directory,"verify",stage)
    plan,feature_parent,feature_stage=feature.verify_plan(feature_root)
    pilot=feature.checked(feature_root,"pilot",plan)
    trained=feature.checked(feature_root,"train",plan);verified=feature.checked(feature_root,"verify",plan)
    if (baseline["selected_checkpoint_sha256"]!=resident.SEED_SHA or not baseline_verified["passed"] or
            parent["fingerprint"]!=feature_parent["fingerprint"] or stage["fingerprint"]!=feature_stage["fingerprint"] or
            Path(plan["source_root"]).resolve()!=seed_root or not verified["passed"] or not pilot["passed"]):
        raise ValueError("Feature candidate must use the same authenticated thirteen-block baseline")
    candidates=[descriptor("baseline",directory/"train",baseline),
        descriptor("feature_selected",feature_root/"train",trained),
        descriptor("feature_best_task",feature_root/"train",trained,diagnostic=True)]
    return dict(seed_root=str(seed_root),feature_root=str(feature_root),parent=parent,stage=stage,
        feature_plan=plan,feature_result=trained,candidates=candidates,selection=choose(candidates))


def prepare(root,seed_root,feature_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Seed verification preparation must hide CUDA")
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    evidence=seed_candidates(seed_root,feature_root);winner=evidence["selection"]["candidate"]
    if winner["kind"]!="feature_best_task":
        return dict(status="existing_fresh_verification_sufficient",kind=winner["kind"],candidate=winner)
    inputs={}
    for path in (Path(feature_root)/"plan.json",Path(feature_root)/"train/manifest.json",Path(feature_root)/"verify/manifest.json",
                 winner["checkpoint"],winner["report"]):old.pin(inputs,path)
    sources={}
    for path in source.local_import_closure([Path(__file__),PROTOCOL,
            base.ROOT/"tests/structure_net/test_jspace_3b_feature_resident_campaign.py"]):sources[str(path)]=base.sha(path)
    plan=dict(schema=SCHEMA,seed_root=evidence["seed_root"],feature_root=evidence["feature_root"],
        source_fingerprint=evidence["parent"]["fingerprint"],stage_fingerprint=evidence["stage"]["fingerprint"],
        feature_fingerprint=evidence["feature_plan"]["fingerprint"],candidate=winner,selection_policy=POLICY,
        allocation_mib=11264,allocator_mib=11008,input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        source_sha256=sources,confirmed=False)
    plan["fingerprint"]=base.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        target=root/"producer"/Path(path).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if base.sha(target)!=digest:raise ValueError("Seed verifier archive changed")
    shutil.copy2(winner["report"],root/"reference.json")
    if base.sha(root/"reference.json")!=winner["report_sha256"]:raise ValueError("Candidate record archive changed")
    base.write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],candidate=winner)


def verify_plan(root):
    root=Path(root);plan=read(root/"plan.json");source.check_plan(plan)
    if (plan.get("schema")!=SCHEMA or plan.get("selection_policy")!=POLICY or
            plan.get("allocation_mib")!=11264 or plan.get("allocator_mib")!=11008):
        raise ValueError("Seed verifier protocol changed")
    for path,digest in plan["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError("Seed verifier source/archive changed")
    for path,identity in plan["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Candidate input changed")
    if base.sha(root/"reference.json")!=plan["candidate"]["report_sha256"]:raise ValueError("Archived epoch records changed")
    evidence=seed_candidates(plan["seed_root"],plan["feature_root"])
    if evidence["selection"]["candidate"]!=plan["candidate"] or evidence["feature_plan"]["fingerprint"]!=plan["feature_fingerprint"]:
        raise ValueError("Winning seed candidate changed")
    return plan,evidence


def checked(root,candidate):
    root=Path(root);plan,_=verify_plan(root);manifest=read(root/"run/manifest.json")
    if plan["candidate"]!=candidate or manifest.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"]:
        raise ValueError("Matching completed seed verification required")
    for name,digest in manifest.get("artifacts",{}).items():
        path=root/"run"/name
        if not path.resolve().is_relative_to((root/"run").resolve()) or base.sha(path)!=digest:
            raise ValueError("Seed verification artifact changed")
    if "result.json" not in manifest.get("artifacts",{}):raise ValueError("Authenticated seed result required")
    result=read(root/"run/result.json");reference=read(root/"reference.json")
    if (result.get("status")!="completed" or not result.get("passed") or not result.get("strict_state_load") or not result.get("fixed_state_unchanged") or
            result.get("checkpoint_sha256")!=candidate["checkpoint_sha256"] or result.get("metrics")!=reference["metrics"] or
            result.get("records")!=reference["records"] or not native_gate(result["metrics"])["passed"]):
        raise ValueError("Exact fresh best-task seed verification failed")
    return result


def run(root):
    from .jspace_full_lease import wait_ready
    root=Path(root).resolve();plan,evidence=verify_plan(root);output=root/"run"
    if output.exists():raise FileExistsError(output)
    lease=None;data=None;started=time.monotonic()
    try:
        lease=wait_ready([0]);allocator=source.cap_allocator(lease,plan)
        torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4);torch.manual_seed(feature.CONFIG.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Seed reservation changed")
            if torch.cuda.max_memory_reserved()>allocator["allocator_limit_bytes"]:raise MemoryError("Seed reservation cap exceeded")
        parent,stage=evidence["parent"],evidence["stage"]
        stack,norm,E,rotary,_=source.make_stack(parent,stage,"cuda:0");before=old.fixed_state(stack,E,rotary)
        state=source.native._load(plan["candidate"]["checkpoint"])
        stack.load_state_dict(state["stack"],strict=True);norm.load_state_dict(state["norm"],strict=True);del state
        data=source.data_for(parent,23,source.stage_dir(plan["seed_root"],23)/"capture/packed_teacher_logits.npy")
        scorer=source._trainer(stack,norm,E,rotary,data,parent,stage,guard)
        metrics,rows=scorer.packed_development();reference=read(root/"reference.json")
        fixed=old.fixed_state(stack,E,rotary)==before
        if not fixed or metrics!=reference["metrics"] or rows!=reference["records"]:
            raise RuntimeError("Fresh best-task seed changed actual metrics/logits or geometry")
        output.mkdir();result=dict(status="completed",passed=True,strict_state_load=True,fixed_state_unchanged=fixed,
            checkpoint_sha256=plan["candidate"]["checkpoint_sha256"],metrics=metrics,records=rows,
            native_gate=native_gate(metrics),selection_policy=POLICY,confirmed=False)
        base.write(output/"result.json",result);guard()
        base.write(output/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],lease_id=lease["id"],
            allocation=lease["allocation"],allocator=allocator,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-started,artifacts={"result.json":base.sha(output/"result.json")}))
        return result
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        base.write(output/"failure.json",dict(status="failed",error=repr(error),lease_id=None if lease is None else lease["id"]))
        raise
    finally:
        if data is not None:data.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--seed-root",type=Path);parser.add_argument("--feature-root",type=Path)
    args=parser.parse_args()
    if args.phase=="prepare":
        if args.seed_root is None or args.feature_root is None:parser.error("prepare requires both source roots")
        result=prepare(args.root,args.seed_root,args.feature_root)
    else:result=run(args.root)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
