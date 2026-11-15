"""Prospective continuation under the original-native accuracy budget.

The stopped v3 dual-gate result remains unchanged. Only admission/continuation
policy changes; frozen architecture, calibration, selection and training are
reused explicitly, without patching module globals.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch
from . import jspace_qwen3b_progressive as prior
from . import jspace_qwen3b_native_budget_train as training

old,native,core,model,calibration,targets = prior.old,prior.native,prior.core,prior.model,prior.calibration,prior.targets
base,read = prior.base,prior.read
CONFIG=training.CONFIG
SCHEMA="jspace.qwen3b-native-budget.v1"
STUDY="JSPACE-QWEN3B-NATIVE-BUDGET-01"
PROTOCOL=base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-native-budget-continuation.md"
SEED_SHA="d6954494f3fd74a5765c3f623b39891cae383339d1888bbef067dd6879bb4cb5"
PHASES=prior.PHASES
records,pilot_result,check_plan=prior.records,prior.pilot_result,prior.check_plan
data_for,make_stack,cap_allocator=prior.data_for,prior.make_stack,prior.cap_allocator


def stage_dir(root,cut):
    if type(cut) is not int or not 0<=cut<=25:raise ValueError("New cut must be an integer in [0,25]")
    return Path(root)/"stages"/f"block_{cut:02d}"


def seed_gate(result,verified,outcome):
    """Admit only the named completed old miss; never rewrite its old gate."""
    if (result.get("selected_checkpoint_sha256")!=SEED_SHA or result.get("selected",{}).get("correct")!=47
        or result.get("quality_pass") is not False or not verified.get("passed")
        or verified.get("selected_checkpoint_sha256")!=SEED_SHA
        or outcome.get("status")!="quality_stop" or outcome.get("failed_cut")!=26
        or outcome.get("last_accepted_cut")!=27 or outcome.get("selected")!=result["selected"]):
        raise ValueError("Expected the completed, freshly verified v3 cut26 original dual-gate miss")
    expected=prior.training.gates(47,native_correct=48,parent_correct=52,count=64)
    if result.get("quality")!=expected:raise ValueError("Historical dual-gate evidence changed")
    revised=training.gates(47,native_correct=48,parent_correct=52,count=64)
    if not revised["passed"] or revised["parent"]["passed"]:raise ValueError("Invalid prospective seed policy")
    return dict(status="admitted",seed_cut=26,seed_correct=47,seed_checkpoint_sha256=SEED_SHA,
        historical_quality=expected,historical_quality_pass=False,prospective_quality=revised,
        old_result_reclassified=False,development_reused=True,confirmed=False)


def prepare(root,seed_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Preparation must hide CUDA")
    root,seed_root=Path(root).resolve(),Path(seed_root).resolve()
    if root.exists():raise FileExistsError(root)
    seed=prior.verify_root(seed_root)
    if seed.get("seed_cut")!=29 or seed.get("native_correct")!=48:raise ValueError("Expected original v3 campaign")
    inputs=dict(seed["input_sha256"]); chain=[]
    for source,digest in seed["source_sha256"].items():
        inputs[str(seed_root/"producer"/Path(source).relative_to(base.ROOT))]=digest
    old.pin(inputs,seed_root/"plan.json");old.pin(inputs,seed_root/"outcome.json")
    for cut in (28,27,26):
        _,stage=prior.verify_stage(seed_root,cut);directory=prior.stage_dir(seed_root,cut)
        inputs.update(stage["input_sha256"])
        evidence={phase:prior.checked(directory,phase,stage) for phase in PHASES}
        if cut>26 and not evidence["train"]["quality_pass"]:raise ValueError("Earlier v3 stages were not accepted")
        old.pin(inputs,directory/"plan.json")
        for phase in PHASES:
            manifest=directory/phase/"manifest.json";old.pin(inputs,manifest)
            for name,digest in read(manifest)["artifacts"].items():inputs[str((directory/phase/name).resolve())]=digest
        chain.append(dict(cut=cut,fingerprint=stage["fingerprint"],selected_checkpoint_sha256=evidence["train"]["selected_checkpoint_sha256"],
            selected_correct=evidence["train"]["selected"]["correct"],historical_quality_pass=evidence["train"]["quality_pass"]))
    result,verified=evidence["train"],evidence["verify"]
    receipt=seed_gate(result,verified,read(seed_root/"outcome.json"))
    receipt.update(source_root=str(seed_root),source_fingerprint=seed["fingerprint"],authenticated_stages=chain)
    fits=stage["fits"]
    if sorted(map(int,fits))!=list(range(26,36)):raise ValueError("Seed must contain exactly ten compact blocks")
    for fit in fits.values():
        for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):
            path=Path(fit)/name
            if str(path.resolve()) not in inputs:old.pin(inputs,path)
    sources=dict(seed["source_sha256"])
    for path in (Path(__file__),Path(training.__file__),PROTOCOL,
        base.ROOT/"tests/structure_net/test_jspace_qwen3b_native_budget.py",
        base.ROOT/"tests/structure_net/test_jspace_qwen3b_native_budget_train.py"):
        sources[str(path.resolve())]=base.sha(path)
    volume=root.parent
    while not volume.exists():volume=volume.parent
    if shutil.disk_usage(volume).free<350*1024**3:raise OSError("At least350GiB free required for preserved continuation artifacts")
    root.mkdir(parents=True);receipt_path=root/"seed_gate_receipt.json";base.write(receipt_path,receipt)
    old.pin(inputs,receipt_path)
    plan=dict(schema=SCHEMA,study_id=STUDY,quality_policy="original_native_only",seed_root=str(seed_root),
        seed_fingerprint=seed["fingerprint"],seed_cut=26,seed_correct=47,target_cut=0,native_correct=48,
        seed_checkpoint=str(directory/"train/selected.pt"),seed_checkpoint_sha256=SEED_SHA,
        seed_records=str(directory/"train"/f"epoch_{result['selected_epoch']:02d}_development.json"),
        seed_gate_receipt=str(receipt_path),seed_gate_receipt_sha256=base.sha(receipt_path),
        model=seed["model"],fits=fits,native_cache=seed["native_cache"],items_path=seed["items_path"],
        original_indices=seed["original_indices"],label_ids=seed["label_ids"],calibration_indices=seed["calibration_indices"],
        config=asdict(CONFIG),allocation_mib=11264,allocator_mib=11008,input_sha256=inputs,
        input_identity={p:old.file_identity(p) for p in inputs},source_sha256=sources,confirmed=False)
    plan["fingerprint"]=base.digest(plan)
    for path,digest in sources.items():
        if base.sha(path)!=digest:raise ValueError("Source changed during preparation")
        target=root/"producer"/Path(path).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    base.write(root/"plan.json",plan)
    base.event(root/"progress.jsonl","prepared",fingerprint=plan["fingerprint"],compressed_blocks=10,quality_policy=plan["quality_policy"])
    return dict(status="prepared",fingerprint=plan["fingerprint"])


def verify_root(root):
    root=Path(root);plan=read(root/"plan.json");check_plan(plan)
    if plan.get("schema")!=SCHEMA or plan["config"]!=asdict(CONFIG) or plan.get("quality_policy")!="original_native_only":raise ValueError("Campaign protocol changed")
    for path,digest in plan["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError("Frozen producer changed: "+path)
    for path,identity in plan["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Pinned original input changed: "+path)
    if base.sha(plan["seed_gate_receipt"])!=plan["seed_gate_receipt_sha256"]:raise ValueError("Seed admission receipt changed")
    return plan


def checked(directory,phase,stage):
    """Authenticate complete phase evidence and its prospective validity gates."""
    if phase not in PHASES:raise ValueError("Unknown phase")
    path=Path(directory)/phase;manifest=read(path/"manifest.json")
    if manifest.get("status")!="completed" or manifest.get("fingerprint")!=stage["fingerprint"]:
        raise ValueError("Completed matching phase required: "+phase)
    artifacts=manifest.get("artifacts",{})
    if "result.json" not in artifacts:raise ValueError("Phase result must be authenticated")
    for name,digest in artifacts.items():
        target=path/name
        if not target.resolve().is_relative_to(path.resolve()) or base.sha(target)!=digest:
            raise ValueError("Phase artifact changed: "+name)
    result=read(path/"result.json")
    if result.get("status")!="completed":raise ValueError("Phase execution is incomplete")
    layers=list(range(stage["cut"],36))
    if phase=="capture":
        if (not result.get("controls",{}).get("passed") or
                not result.get("teacher_cache_diagnostic",{}).get("predictions_exact") or
                result.get("teacher_fingerprint")!=stage["teacher_fingerprint"]):
            raise ValueError("Capture parent/seam/teacher validity failed")
    elif phase=="fit":
        if result.get("cut")!=stage["cut"] or result.get("teacher_fingerprint")!=stage["teacher_fingerprint"]:
            raise ValueError("Fit teacher/cut provenance changed")
    elif phase=="pilot":
        if (not result.get("passed") or not result.get("systems_only") or not result.get("discarded") or
                result.get("layer_indices")!=layers or result.get("update",{}).get("update")!=1 or
                result.get("fp32_master_changed_by_block")!=[True]*len(layers) or
                not result.get("norm_fp32_master_changed") or not result.get("fixed_state_unchanged") or
                not result.get("checkpoint_strict_load") or not result.get("inheritance",{}).get("passed")):
            raise ValueError("Discarded pilot failed update, fixed-state or reload validity")
    elif phase=="train":
        history=result.get("history",[])
        if (result.get("config")!=asdict(CONFIG) or result.get("completed_epochs")!=12 or result.get("updates")!=384 or
                result.get("layer_indices")!=layers or [r.get("epoch") for r in history]!=list(range(13)) or
                not result.get("strict_reload_exact") or not result.get("fixed_state_unchanged")):
            raise ValueError("Training budget, layer layout or reload validity failed")
        updates=[u for row in history for u in row.get("training",[])]
        if ([u.get("update") for u in updates]!=list(range(1,385)) or
                any(u.get("gradients_present_by_block")!=[True]*len(layers) for u in updates) or
                any(len(row.get("training",[]))!=(0 if row["epoch"]==0 else 32) for row in history)):
            raise ValueError("Training updates or block gradient participation incomplete")
        for row in history:
            filename=f"epoch_{row['epoch']:02d}_development.json"
            if artifacts.get(filename)!=row.get("development_sha256"):
                raise ValueError("Epoch development receipt missing from manifest")
            if read(path/filename).get("metrics")!=row.get("metrics") or row["metrics"].get("count")!=64:
                raise ValueError("Epoch development metrics/count changed")
        winner=max(history,key=lambda row:prior.training.selection_key(row["metrics"]))
        if (winner["epoch"]!=result.get("selected_epoch") or winner["metrics"]!=result.get("selected") or
                winner.get("checkpoint_sha256")!=result.get("selected_checkpoint_sha256") or
                artifacts.get("selected.pt")!=result.get("selected_checkpoint_sha256") or
                result.get("initial")!=history[0]["metrics"]):
            raise ValueError("Accuracy-first selected checkpoint provenance changed")
        quality=training.gates(result["selected"]["correct"],native_correct=48,
            parent_correct=stage["parent_correct"],count=64)
        if result.get("quality")!=quality or result.get("quality_pass")!=quality["passed"]:
            raise ValueError("Original-native budget or parent diagnostic changed")
    elif phase=="verify":
        trained=checked(directory,"train",stage)
        expected_quality=training.gates(trained["selected"]["correct"],native_correct=48,parent_correct=stage["parent_correct"],count=64)
        if (not result.get("passed") or not result.get("comparison",{}).get("passed") or
                not result.get("fixed_state_unchanged") or result.get("metrics")!=trained["selected"] or
                result.get("selected_checkpoint_sha256")!=trained["selected_checkpoint_sha256"] or
                result.get("quality")!=expected_quality or result.get("compressed_blocks")!=len(layers)):
            raise ValueError("Fresh verification, selected state or quality provenance failed")
    return result


def prepare_stage(root,cut):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Stage preparation must hide CUDA")
    root=Path(root).resolve();plan=verify_root(root);directory=stage_dir(root,cut)
    if directory.exists():raise FileExistsError(directory)
    fits=dict(plan["fits"]);inputs={}
    if cut==25:
        checkpoint=plan["seed_checkpoint"];reference=plan["seed_records"];parent_correct=plan["seed_correct"]
        parent_fingerprint=plan["seed_fingerprint"]
    else:
        parent_dir=stage_dir(root,cut+1);parent=read(parent_dir/"plan.json");check_plan(parent)
        if parent.get("root_fingerprint")!=plan["fingerprint"] or parent.get("cut")!=cut+1:
            raise ValueError("Parent must be the immediately preceding stage of this campaign")
        result=checked(parent_dir,"train",parent);verification=checked(parent_dir,"verify",parent)
        if not result["quality_pass"] or not verification["passed"]:raise ValueError("Cannot expand an unaccepted parent")
        fits=parent["fits"];checkpoint=str(parent_dir/"train/selected.pt")
        reference=str(parent_dir/"train"/f"epoch_{result['selected_epoch']:02d}_development.json")
        parent_correct=result["selected"]["correct"];parent_fingerprint=parent["fingerprint"]
        for path in (parent_dir/"plan.json",parent_dir/"train/manifest.json",parent_dir/"verify/manifest.json"):
            old.pin(inputs,path)
    old.pin(inputs,checkpoint);old.pin(inputs,reference)
    for fit in fits.values():
        for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):
            old.pin(inputs,Path(fit)/name)
    fits=dict(fits);fits[str(cut)]=str(directory/"fit"/f"block_{cut:02d}")
    stage=dict(schema=SCHEMA,root_fingerprint=plan["fingerprint"],cut=cut,parent_cut=cut+1,layers=list(range(cut,36)),
        parent_checkpoint=checkpoint,parent_checkpoint_sha256=base.sha(checkpoint),parent_records=reference,
        parent_correct=parent_correct,parent_fingerprint=parent_fingerprint,native_correct=48,quality_policy="original_native_only",
        fits=fits,input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        teacher_fingerprint=base.digest(base.sha(checkpoint)),calibration_seed=20861+100000*(29-cut))
    stage["fingerprint"]=base.digest(stage);directory.mkdir(parents=True);base.write(directory/"plan.json",stage)
    base.event(root/"progress.jsonl","stage_prepared",cut=cut,parent_correct=parent_correct)
    return stage


def verify_stage(root,cut):
    plan=verify_root(root);stage=read(stage_dir(root,cut)/"plan.json");check_plan(stage)
    if stage["root_fingerprint"]!=plan["fingerprint"] or stage["cut"]!=cut:raise ValueError("Stage/root mismatch")
    for path,identity in stage["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Stage parent input changed: "+path)
    if base.sha(stage["parent_checkpoint"])!=stage["parent_checkpoint_sha256"]:raise ValueError("Parent checkpoint changed")
    return plan,stage


def controls(plan,stage,stack,norm,readout,rotary,guard):
    cut=stage["cut"];data=data_for(plan,cut);parent_data=data_for(plan,cut+1)
    cases=[data.items[i] for i in data.development];reference=records(stage["parent_records"])
    try:
        parent_rows=old.score(stack,norm,readout,rotary,parent_data,plan["label_ids"],guard)
        parent_check=old.compare_rows(parent_rows,reference,cases,expected_correct=stage["parent_correct"])
        block=native.load_native_block(plan["model"]["path"],cut,device=readout.device)
        seam=old.NativeTail([block,stack]).eval().requires_grad_(False)
        seam_rows=old.score(seam,norm,readout,rotary,data,plan["label_ids"],guard)
        seam_check=old.compare_rows(seam_rows,reference,cases,expected_correct=stage["parent_correct"])
        del seam,block
        result=dict(parent=dict(**parent_check,records=parent_rows),seam=dict(**seam_check,records=seam_rows),
            passed=parent_check["passed"] and seam_check["passed"])
        if cut==25:
            _,native_norm,_=native.load_readout_norm(plan["model"]["path"],device=readout.device)
            tail=old.NativeTail([native.load_native_block(plan["model"]["path"],i,device=readout.device) for i in range(cut,36)])
            rows=old.score(tail,native_norm,readout,rotary,data,plan["label_ids"],guard,batch_size=1)
            result["native"]={**old.compare_rows(rows,old.rows_from_cache(data,plan["label_ids"]),cases,expected_correct=48),"records":rows}
            result["passed"] &=result["native"]["passed"]
        return result
    finally:data.close();parent_data.close()


def run_phase(root,cut,phase):
    from .jspace_full_lease import wait_ready
    root=Path(root).resolve();plan,stage=verify_stage(root,cut);directory=stage_dir(root,cut);output=directory/phase
    if phase not in PHASES:raise ValueError("Unknown GPU phase")
    if output.exists():raise FileExistsError(output)
    for previous in PHASES[:PHASES.index(phase)]:checked(directory,previous,stage)
    lease=None;data=None;teacher=None;started=time.monotonic();cuda_started=False
    try:
        lease=wait_ready([0]);allocator=cap_allocator(lease,plan);torch.cuda.init();cuda_started=True
        torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4);torch.manual_seed(CONFIG.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Reservation changed")
            if torch.cuda.max_memory_reserved()>allocator["allocator_limit_bytes"]:raise MemoryError("Allocator reservation exceeded")
        def event(kind,**values):base.event(root/"progress.jsonl",kind,cut=cut,phase=phase,**values)
        def progress(value):
            payload=dict(value);kind=payload.pop("kind", "calibration_progress")
            payload.pop("cut",None);payload.pop("phase",None)
            event(kind,**payload)
        if phase=="capture":
            stack,norm,readout,rotary,inheritance=make_stack(plan,stage,"cpu",parent=True)
            stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).to("cuda:0")
            norm.to("cuda:0");readout=readout.to("cuda:0");rotary.to("cuda:0")
            control=controls(plan,stage,stack,norm,readout,rotary,guard)
            if not control["passed"]:raise RuntimeError("Parent/seam/native replay failed: "+json.dumps(control))
            block=native.load_native_block(plan["model"]["path"],cut,device="cuda:0")
            teacher=calibration.CachedPrefixTeacher(read(plan["items_path"])["items"],
                Path(plan["native_cache"])/f"boundary_{cut:02d}.npy",cut,block,stack,norm,readout,rotary,guard=guard)
            calibration.capture(output,teacher,plan["calibration_indices"],teacher_fingerprint=stage["teacher_fingerprint"],progress=progress)
            data=data_for(plan,cut,output/"accepted_teacher_logits.npy")
            differentiable=old.compare_rows(old.rows_from_cache(data,plan["label_ids"],"accepted"),records(stage["parent_records"]),
                [data.items[i] for i in data.development])
            data.close();data=data_for(plan,cut+1)
            target_receipt=targets.capture_packed_teacher(output/"packed_teacher_logits.npy",stack,norm,readout,rotary,data,
                guard=guard,progress=progress)
            data.close();data=data_for(plan,cut,output/"packed_teacher_logits.npy")
            diagnostic=old.compare_rows(old.rows_from_cache(data,plan["label_ids"],"accepted"),records(stage["parent_records"]),
                [data.items[i] for i in data.development],expected_correct=stage["parent_correct"])
            if not diagnostic["passed"]:raise RuntimeError("Packed teacher targets changed accepted parent predictions/logits")
            result=dict(status="completed",controls=control,inheritance=inheritance,teacher_cache_diagnostic=diagnostic,
                differentiable_teacher_cache_diagnostic=differentiable,packed_target_receipt=target_receipt,
                teacher_cache_logit_gate_applied=True,teacher_fingerprint=stage["teacher_fingerprint"],calibration_reused=False)
        elif phase=="fit":
            output.mkdir();calibration.finalize(directory/"capture",output,device="cuda:0",guard=guard,progress=progress)
            block=native.load_native_block(plan["model"]["path"],cut,device="cuda:0")
            _,_,rotary=native.load_readout_norm(plan["model"]["path"],device="cuda:0")
            calibration.capture_interface(output/f"block_{cut:02d}",directory/"capture",cut,block,rotary,
                device="cuda:0",guard=guard,progress=progress)
            calibration.fit_relation(output/f"block_{cut:02d}",device="cuda:0",guard=guard,progress=progress)
            result=dict(status="completed",cut=cut,teacher_fingerprint=stage["teacher_fingerprint"])
        else:
            stack,norm,readout,rotary,inheritance=make_stack(plan,stage,"cuda:0")
            data=data_for(plan,cut,directory/"capture/packed_teacher_logits.npy")
            trainer=training.NativeBudgetTrainer(stack,norm,readout,rotary,data,plan["label_ids"],config=CONFIG,guard=guard,
                native_correct=48,parent_correct=stage["parent_correct"])
            if phase=="pilot":
                output.mkdir();initial,initial_rows=trainer.packed_development()
                order=sorted(data.training,key=lambda i:data.lengths[i],reverse=True)
                probe=trainer.master_update_probe(order[:16]);old.save(output/"discarded_state.pt",trainer.snapshot())
                if not probe["passed"]:raise RuntimeError("Discarded pilot gradient/fixed-state proof failed")
                snapshot=native._load(output/"discarded_state.pt")
                stack.load_state_dict(snapshot["stack"],strict=True);norm.load_state_dict(snapshot["norm"],strict=True)
                result=pilot_result(probe,initial,initial_rows,inheritance)
            elif phase=="train":
                result=trainer.fit(output,event=event,initial_reference=checked(directory,"pilot",stage))
            else:
                output.mkdir();trained=checked(directory,"train",stage);snapshot=native._load(directory/"train/selected.pt")
                before=old.fixed_state(stack,readout,rotary)
                stack.load_state_dict(snapshot["stack"],strict=True);norm.load_state_dict(snapshot["norm"],strict=True)
                metrics,rows=trainer.packed_development()
                reference=read(directory/"train"/f"epoch_{trained['selected_epoch']:02d}_development.json")
                comparison=old.compare_rows(rows,reference["records"],[data.items[i] for i in data.development],expected_correct=trained["selected"]["correct"])
                fixed=old.fixed_state(stack,readout,rotary)==before
                result=dict(status="completed",passed=comparison["passed"] and metrics==reference["metrics"] and fixed,
                    comparison=comparison,metrics=metrics,records=rows,fixed_state_unchanged=fixed,
                    storage=model.inventory(plan["model"]["path"],stack,norm,rotary),
                    selected_checkpoint_sha256=base.sha(directory/"train/selected.pt"),quality=training.gates(metrics["correct"],native_correct=48,parent_correct=stage["parent_correct"],count=64),
                    compressed_blocks=36-cut,delta_from_continuation_seed=metrics["correct"]-47,confirmed=False)
                if not result["passed"]:raise RuntimeError("Fresh packed verification failed")
        if not (output/"result.json").exists():base.write(output/"result.json",result)
        guard();base.write(output/"manifest.json",dict(status="completed",fingerprint=stage["fingerprint"],
            artifacts={str(p.relative_to(output)):base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
            lease_id=lease["id"],allocation=lease["allocation"],allocator=allocator,
            elapsed_seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        event("phase_completed",elapsed_seconds=time.monotonic()-started,
            correct=result.get("selected",result.get("metrics",{})).get("correct"))
        return dict(status="completed",cut=cut,phase=phase)
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        if not (output/"failure.json").exists():base.write(output/"failure.json",dict(status="failed",error=repr(error),
            error_type=type(error).__name__,failure_kind="resource_failure" if isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)) else "execution_failure",
            lease_id=None if lease is None else lease["id"],peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise
    finally:
        if data is not None:data.close()
        if teacher is not None:teacher.close()


def campaign(root,*,stop_cut=0):
    """CPU-only supervisor; every GPU subprocess obtains and releases its own lease."""
    root=Path(root).resolve();plan=verify_root(root)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Supervisor must hide CUDA")
    if type(stop_cut) is not int or not 0<=stop_cut<=25:raise ValueError("Invalid stopping cut")
    for cut in range(25,stop_cut-1,-1):
        directory=stage_dir(root,cut)
        if not directory.exists():prepare_stage(root,cut)
        _,stage=verify_stage(root,cut)
        for phase in PHASES:
            if (directory/phase/"manifest.json").exists():checked(directory,phase,stage);continue
            if (directory/phase).exists():raise RuntimeError("Incomplete phase requires a new versioned attempt: "+str(directory/phase))
            command=["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation","GPU-0370fdb0=11264",
                "--minutes","120","--reason",f"{STUDY} cut{cut} {phase}","--",
                "pixi","run","--locked","--offline","-e","ncm","python","-m",__name__,phase,"--root",str(root),"--cut",str(cut)]
            # __name__ is __main__ under python -m; use the importable module explicitly.
            command[command.index("-m")+1]="experiments.structure_net.jspace_qwen3b_native_budget"
            with (directory/f"{phase}.log").open("xb") as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            if completed.returncode:raise RuntimeError(f"cut{cut} {phase} failed: see {directory/phase}.log")
            checked(directory,phase,stage)
        result=checked(directory,"train",stage);verification=checked(directory,"verify",stage)
        base.event(root/"progress.jsonl","stage_completed",cut=cut,compressed_blocks=36-cut,
            correct=result["selected"]["correct"],quality_pass=result["quality_pass"],quality=result["quality"],storage=verification["storage"])
        if not result["quality_pass"]:
            outcome=dict(status="quality_stop",last_accepted_cut=cut+1,failed_cut=cut,compressed_blocks=35-cut,
                selected=result["selected"],quality=result["quality"],confirmed=False)
            base.write(root/"outcome.json",outcome);return outcome
    outcome=dict(status="all_blocks_recovered" if stop_cut==0 else "requested_prefix_completed",last_accepted_cut=stop_cut,
        compressed_blocks=36-stop_cut,heldout_and_speed_pending=True,confirmed=False)
    base.write(root/("outcome.json" if stop_cut==0 else f"through_cut_{stop_cut:02d}.json"),outcome);return outcome


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","prepare-stage","campaign",*PHASES));parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--seed-root",type=Path);parser.add_argument("--cut",type=int);parser.add_argument("--stop-cut",type=int,default=0)
    args=parser.parse_args()
    if args.phase=="prepare":result=prepare(args.root,args.seed_root)
    elif args.phase=="prepare-stage":result=prepare_stage(args.root,args.cut)
    elif args.phase=="campaign":result=campaign(args.root,stop_cut=args.stop_cut)
    else:result=run_phase(args.root,args.cut,args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)
