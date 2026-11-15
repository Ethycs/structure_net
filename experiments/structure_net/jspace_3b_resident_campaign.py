"""Qualified resident-Adam continuation of the accepted thirteen-block suffix.

The six numerical phases, model, objective and protected selection are inherited.
Only fresh joint pilot/train optimizers use spare-GPU state with CPU spill.
No existing producer, supervisor or accepted checkpoint is modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from . import jspace_3b_staged_campaign as inherited
from . import jspace_resident_adam_replay as replay

base, old, read = inherited.base, inherited.old, inherited.read
native, model, calibration, targets = inherited.native, inherited.model, inherited.calibration, inherited.targets
training, entrance = inherited.training, inherited.entrance
CONFIG, ENTRANCE_CONFIG, PHASES = inherited.CONFIG, inherited.ENTRANCE_CONFIG, inherited.PHASES
check_plan, records, data_for, make_stack = inherited.check_plan, inherited.records, inherited.data_for, inherited.make_stack
SCHEMA = "jspace.3b-resident-campaign.v1"
STUDY = "JSPACE-3B-RESIDENT-CONTINUATION-01"
MODULE = "experiments.structure_net.jspace_3b_resident_campaign"
PROTOCOL = base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-resident-optimizer-continuation.md"
SEED_CUT = 23
SEED_SHA = "e12dad834ef48acf76fce441fbb919ab2749cd5c707396df75018646805e0e8e"
OPTIMIZER = dict(kind="resident_fp32_adam_with_cpu_spill", resident_budget_mib=6144,
    placement="deterministic_parameter_order_first_fit", resident_phases=["pilot", "train"],
    entrance_optimizer="unchanged_cpu", arithmetic="same_ATen_sequence; CPU_CUDA_rounding_may_differ",
    gradient_norm="unchanged_model_device_global_chunk_reduction", transport="pinned_host_bounce_no_P2P")


def stage_dir(root, cut):
    if type(cut) is not int or not 0 <= cut < SEED_CUT:
        raise ValueError("New stages are cuts22 through0")
    return Path(root)/"stages"/f"block_{cut:02d}"


def qualification(replay_root):
    """A real completed resident replay qualifies the implementation, not a seed."""
    replay_root = Path(replay_root).resolve()
    plan, _, _ = replay.verify_plan(replay_root)
    pilots = {arm: replay.checked(replay_root, arm, "pilot", plan) for arm in replay.ARMS}
    trained = replay.checked(replay_root, "resident", "train", plan)
    verified = replay.checked(replay_root, "resident", "verify", plan)
    quality = training.inherited.gates(trained["selected"]["correct"], native_correct=48,
        parent_correct=trained["controls"]["parent_correct"], count=64)
    cpu_probe,resident_probe=pilots["cpu"]["probe"],pilots["resident"]["probe"]
    if (plan["cut"] != 24 or plan["resident_budget_mib"] != OPTIMIZER["resident_budget_mib"] or
            trained.get("quality") != quality or trained.get("quality_pass") is not True or not quality["passed"] or
            verified["metrics"] != trained["selected"] or not all(p["warm_replay_exact"] for p in pilots.values()) or
            pilots["cpu"]["warm"] != pilots["resident"]["warm"] or pilots["cpu"]["records"] != pilots["resident"]["records"] or
            len(cpu_probe.get("fp32_master_before",[]))!=36-plan["cut"] or
            cpu_probe["fp32_master_before"]!=resident_probe.get("fp32_master_before") or
            not cpu_probe.get("norm_fp32_master_before") or
            cpu_probe["norm_fp32_master_before"]!=resident_probe.get("norm_fp32_master_before")):
        raise ValueError("Matched warm controls, original-native quality and exact own-selected verification required")
    return dict(root=str(replay_root), fingerprint=plan["fingerprint"], cut=plan["cut"],
        selected=trained["selected"], selected_checkpoint_sha256=trained["selected_checkpoint_sha256"],
        quality=quality, historical_selected=plan["historical_selected"],
        cpu_full_replay_required=False, cross_backend_weight_equality_required=False,
        qualification_only_not_campaign_seed=True,
        receipts={f"{arm}/{phase}/manifest.json":base.sha(replay_root/arm/phase/"manifest.json")
            for arm, phase in (("cpu","pilot"),("resident","pilot"),("resident","train"),("resident","verify"))})


def _pin_replay(inputs, root, receipt):
    root = Path(root)
    old.pin(inputs, root/"plan.json")
    for relative, digest in receipt["receipts"].items():
        manifest_path = root/relative
        old.pin(inputs, manifest_path, digest)
        for name, expected in read(manifest_path)["artifacts"].items():
            target = manifest_path.parent/name
            if not target.resolve().is_relative_to(manifest_path.parent.resolve()):
                raise ValueError("Replay artifact escapes its phase")
            old.pin(inputs, target, expected)


def prepare(root, seed_root, replay_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation must hide CUDA")
    root, seed_root = Path(root).resolve(), Path(seed_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    qualified = qualification(replay_root)  # No work is admitted on an unfinished replay.
    parent, stage = inherited.verify_stage(seed_root, SEED_CUT)
    directory = inherited.stage_dir(seed_root, SEED_CUT)
    evidence = {phase: inherited.checked(directory, phase, stage) for phase in PHASES}
    trained, verified = evidence["train"], evidence["verify"]
    if (not trained["quality_pass"] or not verified["passed"] or trained["selected"]["correct"] != 46 or
            trained["selected_checkpoint_sha256"] != SEED_SHA or verified["selected_checkpoint_sha256"] != SEED_SHA or
            sorted(map(int,stage["fits"])) != list(range(SEED_CUT,36))):
        raise ValueError("The authenticated accepted thirteen-block seed is required")
    inputs = dict(parent["input_sha256"])
    inputs.update(stage["input_sha256"])
    for path in (seed_root/"plan.json", directory/"plan.json"):
        old.pin(inputs,path)
    # Retain the completed source chain from the original seed to cut23.
    chain = [];previous_sha=parent["seed_checkpoint_sha256"];previous_correct=parent["seed_correct"]
    for cut in range(parent["seed_cut"]-1, SEED_CUT-1, -1):
        _, previous = inherited.verify_stage(seed_root,cut)
        previous_dir = inherited.stage_dir(seed_root,cut)
        if previous["parent_checkpoint_sha256"]!=previous_sha or previous["parent_correct"]!=previous_correct:
            raise ValueError("Inherited accepted seed-parent chain changed")
        results = {phase:inherited.checked(previous_dir,phase,previous) for phase in PHASES}
        if not results["train"]["quality_pass"] or not results["verify"]["passed"]:
            raise ValueError("All inherited seed transitions must be accepted and freshly verified")
        old.pin(inputs,previous_dir/"plan.json")
        for phase in PHASES:
            inherited._pin_phase(inputs,previous_dir,phase)
        chain.append(dict(cut=cut,fingerprint=previous["fingerprint"],
            checkpoint_sha256=results["train"]["selected_checkpoint_sha256"],correct=results["train"]["selected"]["correct"]))
        previous_sha=results["train"]["selected_checkpoint_sha256"];previous_correct=results["train"]["selected"]["correct"]
    _pin_replay(inputs,qualified["root"],qualified)
    sources = dict(parent["source_sha256"])
    extras = [Path(__file__),Path(replay.__file__),PROTOCOL,
        base.ROOT/"tests/structure_net/test_jspace_3b_resident_campaign.py"]
    for path in inherited.local_import_closure(extras):
        digest = base.sha(path)
        if str(path) in sources and sources[str(path)] != digest:
            raise ValueError("Inherited frozen producer changed")
        sources[str(path)] = digest
    plan = dict(schema=SCHEMA,study_id=STUDY,seed_root=str(seed_root),seed_fingerprint=parent["fingerprint"],
        seed_cut=SEED_CUT,seed_correct=46,target_cut=0,native_correct=48,seed_checkpoint_sha256=SEED_SHA,
        seed_checkpoint=str(directory/"train/selected.pt"),
        seed_records=str(directory/"train"/f"epoch_{trained['selected_epoch']:02d}_development.json"),
        seed_chain=chain,qualification=qualified,optimizer=OPTIMIZER,
        model=parent["model"],fits=stage["fits"],native_cache=parent["native_cache"],items_path=parent["items_path"],
        original_indices=parent["original_indices"],label_ids=parent["label_ids"],calibration_indices=parent["calibration_indices"],
        config=asdict(CONFIG),entrance_config=asdict(ENTRANCE_CONFIG),selection_policy=training.SELECTION,
        eligibility_tolerance=training.TOLERANCE,quality_policy="original_native_only",seam_policy=inherited.SEAM_POLICY,
        allocation_mib=11264,allocator_mib=11008,resident_budget_mib=6144,
        input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},source_sha256=sources,confirmed=False)
    plan["fingerprint"] = base.digest(plan)
    volume=root.parent
    while not volume.exists():volume=volume.parent
    if shutil.disk_usage(volume).free<425*1024**3:
        raise OSError("At least 425 GiB free required for preserved continuation")
    root.mkdir(parents=True)
    for path,digest in sources.items():
        destination=root/"producer"/Path(path).relative_to(base.ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        if base.sha(destination)!=digest:raise ValueError("Producer archive changed")
    base.write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],seed_cut=SEED_CUT,compressed_blocks=13)


def verify_root(root):
    root=Path(root);plan=read(root/"plan.json");check_plan(plan)
    if (plan.get("schema")!=SCHEMA or plan.get("seed_cut")!=SEED_CUT or plan.get("seed_correct")!=46 or
            plan.get("seed_checkpoint_sha256")!=SEED_SHA or plan.get("config")!=asdict(CONFIG) or
            plan.get("entrance_config")!=asdict(ENTRANCE_CONFIG) or plan.get("optimizer")!=OPTIMIZER or
            plan.get("quality_policy")!="original_native_only" or plan.get("selection_policy")!=training.SELECTION or
            plan.get("eligibility_tolerance")!=training.TOLERANCE or plan.get("seam_policy")!=inherited.SEAM_POLICY):
        raise ValueError("Resident continuation protocol changed")
    for path,digest in plan["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError("Resident campaign source/archive changed")
    for path,identity in plan["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Resident campaign input changed")
    return plan


def prepare_stage(root,cut):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Stage preparation must hide CUDA")
    root=Path(root).resolve();plan=verify_root(root);directory=stage_dir(root,cut)
    if directory.exists():raise FileExistsError(directory)
    inputs={}
    if cut==SEED_CUT-1:
        fits=plan["fits"];checkpoint=plan["seed_checkpoint"];reference=plan["seed_records"]
        parent_correct=plan["seed_correct"];parent_fingerprint=plan["seed_fingerprint"]
    else:
        parent_dir=stage_dir(root,cut+1);parent=read(parent_dir/"plan.json");check_plan(parent)
        if parent.get("root_fingerprint")!=plan["fingerprint"] or parent.get("cut")!=cut+1:
            raise ValueError("Parent must be the immediately preceding stage")
        trained=checked(parent_dir,"train",parent);verified=checked(parent_dir,"verify",parent)
        if not trained["quality_pass"] or not verified["passed"]:raise ValueError("Cannot grow a rejected parent")
        fits=parent["fits"];checkpoint=str(parent_dir/"train/selected.pt")
        reference=str(parent_dir/"train"/f"epoch_{trained['selected_epoch']:02d}_development.json")
        parent_correct=trained["selected"]["correct"];parent_fingerprint=parent["fingerprint"]
        for path in (parent_dir/"plan.json",parent_dir/"train/manifest.json",parent_dir/"verify/manifest.json"):
            old.pin(inputs,path)
    old.pin(inputs,checkpoint);old.pin(inputs,reference)
    for fit in fits.values():
        for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):old.pin(inputs,Path(fit)/name)
    fits=dict(fits);fits[str(cut)]=str(directory/"fit"/f"block_{cut:02d}")
    stage=dict(schema=SCHEMA,root_fingerprint=plan["fingerprint"],cut=cut,parent_cut=cut+1,layers=list(range(cut,36)),
        parent_checkpoint=checkpoint,parent_checkpoint_sha256=base.sha(checkpoint),parent_records=reference,
        parent_correct=parent_correct,parent_fingerprint=parent_fingerprint,native_correct=48,fits=fits,
        input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        teacher_fingerprint=base.digest(base.sha(checkpoint)),calibration_seed=20861+100000*(29-cut),
        calibration_reuse=None,quality_policy="original_native_only",optimizer=OPTIMIZER)
    stage["fingerprint"]=base.digest(stage);directory.mkdir(parents=True);base.write(directory/"plan.json",stage)
    return stage


def verify_stage(root,cut):
    plan=verify_root(root);stage=read(stage_dir(root,cut)/"plan.json");check_plan(stage)
    if (stage.get("schema")!=SCHEMA or stage.get("root_fingerprint")!=plan["fingerprint"] or
            stage.get("cut")!=cut or stage.get("optimizer")!=OPTIMIZER):
        raise ValueError("Resident stage/root mismatch")
    for path,identity in stage["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Stage parent input changed")
    if base.sha(stage["parent_checkpoint"])!=stage["parent_checkpoint_sha256"]:raise ValueError("Parent checkpoint changed")
    return plan,stage


def checked(directory,phase,stage):
    result=inherited.checked(directory,phase,stage)
    if phase in OPTIMIZER["resident_phases"]:
        path=Path(directory)/phase
        manifest=read(path/"manifest.json")
        if manifest["artifacts"].get("resident_optimizer.json")!=base.sha(path/"resident_optimizer.json"):
            raise ValueError("Authenticated resident placement receipt required")
        receipt=read(path/"resident_optimizer.json")
        memory=receipt.get("memory",{})
        if (receipt.get("optimizer")!=OPTIMIZER or receipt.get("transfer_accounting")!=replay.TRANSFER_ACCOUNTING or
                not str(memory.get("resident_backend","")).startswith("cuda:") or
                memory.get("resident_budget_bytes")!=6144*1024**2 or memory.get("resident_state_bytes",0)<=0 or
                memory.get("resident_planned_peak_bytes",float("inf"))>6144*1024**2):
            raise ValueError("Resident placement/accounting receipt failed")
        allocation=manifest.get("allocation",{});settings=manifest.get("allocator",{})
        if len(allocation)!=2 or set(settings)!={"front","tail"}:
            raise ValueError("Two-card resident reservation receipt required")
        for role,prefix,mib in (("front","GPU-0370fdb0",11264),("tail","GPU-7b1db3cb",7168)):
            values=settings[role];uuid=values.get("uuid","")
            if (not uuid.startswith(prefix) or allocation.get(uuid)!=mib or
                    values.get("allocator_limit_bytes")!=(mib-256)*1024**2 or
                    manifest.get("peak_reserved_bytes",{}).get(role,float("inf"))>values["allocator_limit_bytes"]):
                raise ValueError("Resident reservation mapping or measured peak invalid")
        if memory["resident_backend"]!=f"cuda:{settings['tail']['device']}":
            raise ValueError("Resident state was not assigned to the leased SUPER")
    if phase in ("prealign","train","verify"):
        checked(directory,"pilot",stage)
    if phase=="verify":
        checked(directory,"train",stage)
    return result


def _resident_trainer(stack,norm,E,rotary,data,plan,stage,settings,guard):
    return replay.make_trainer(stack,norm,E,rotary,data,plan,stage,plan,"resident",settings,guard)


def run_phase(root,cut,phase):
    from .jspace_full_lease import wait_ready
    root=Path(root).resolve();plan,stage=verify_stage(root,cut);directory=stage_dir(root,cut);output=directory/phase
    if phase not in PHASES:raise ValueError("Unknown phase")
    if output.exists():raise FileExistsError(output)
    for previous in PHASES[:PHASES.index(phase)]:checked(directory,previous,stage)
    dual=phase in OPTIMIZER["resident_phases"]
    lease=None;settings=None;data=pairs=teacher=trainer=None;started=time.monotonic();cuda_started=False
    try:
        indices=[0,1] if dual else [0]
        lease=wait_ready(indices)
        if dual:
            settings=replay.allocator_settings(lease);front=settings["front"]["device"]
            for value in settings.values():torch.cuda.reset_peak_memory_stats(value["device"])
        else:
            settings={"front":{**inherited.cap_allocator(lease,plan),"device":0}}
            front=0;torch.cuda.reset_peak_memory_stats(front)
        cuda_started=True;torch.cuda.set_device(front);torch.set_num_threads(4);torch.manual_seed(CONFIG.seed)
        torch.backends.cuda.matmul.allow_tf32=False;device=f"cuda:{front}"
        def guard():
            current=wait_ready(indices)
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Reservation changed")
            for value in settings.values():
                if torch.cuda.max_memory_reserved(value["device"])>value["allocator_limit_bytes"]:
                    raise MemoryError("Reservation allocator cap exceeded")
        def event(kind,**values):base.event(root/"progress.jsonl",kind,cut=cut,phase=phase,**values)
        def progress(value):
            values=dict(value);kind=values.pop("kind","calibration_progress");values.pop("cut",None);values.pop("phase",None)
            event(kind,**values)
        if phase=="capture":
            stack,norm,E,rotary,inheritance=make_stack(plan,stage,"cpu",parent=True)
            stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).to(device)
            norm.to(device);E=E.to(device);rotary.to(device)
            control=inherited.controls(plan,stage,stack,norm,E,rotary,guard)
            if not control["passed"]:raise RuntimeError("Parent/native singleton seam replay failed")
            block=native.load_native_block(plan["model"]["path"],cut,device=device)
            teacher=calibration.CachedPrefixTeacher(read(plan["items_path"])["items"],
                Path(plan["native_cache"])/f"boundary_{cut:02d}.npy",cut,block,stack,norm,E,rotary,guard=guard)
            calibration.capture(output,teacher,plan["calibration_indices"],teacher_fingerprint=stage["teacher_fingerprint"],progress=progress)
            data=data_for(plan,cut+1)
            target_receipt=targets.capture_packed_teacher(output/"packed_teacher_logits.npy",stack,norm,E,rotary,data,guard=guard,progress=progress)
            data.close();data=data_for(plan,cut,output/"accepted_teacher_logits.npy")
            differentiable=old.compare_rows(old.rows_from_cache(data,plan["label_ids"],"accepted"),records(stage["parent_records"]),
                [data.items[i] for i in data.development]);data.close()
            data=data_for(plan,cut,output/"packed_teacher_logits.npy")
            diagnostic=old.compare_rows(old.rows_from_cache(data,plan["label_ids"],"accepted"),records(stage["parent_records"]),
                [data.items[i] for i in data.development],expected_correct=stage["parent_correct"])
            if not diagnostic["passed"]:raise RuntimeError("Packed teacher changed accepted parent")
            result=dict(status="completed",controls=control,inheritance=inheritance,teacher_cache_diagnostic=diagnostic,
                differentiable_teacher_cache_diagnostic=differentiable,teacher_cache_logit_gate_applied=True,
                teacher_fingerprint=stage["teacher_fingerprint"],calibration_reused=False,packed_target_receipt=target_receipt,
                teacher_target_path=str(output/"packed_teacher_logits.npy"),teacher_target_sha256=base.sha(output/"packed_teacher_logits.npy"))
        elif phase=="fit":
            output.mkdir();calibration.finalize(directory/"capture",output,device=device,guard=guard,progress=progress)
            block=native.load_native_block(plan["model"]["path"],cut,device=device)
            _,_,rotary=native.load_readout_norm(plan["model"]["path"],device=device)
            calibration.capture_interface(output/f"block_{cut:02d}",directory/"capture",cut,block,rotary,device=device,guard=guard,progress=progress)
            calibration.fit_relation(output/f"block_{cut:02d}",device=device,guard=guard,progress=progress)
            result=dict(status="completed",cut=cut,teacher_fingerprint=stage["teacher_fingerprint"],calibration_reused=False)
        else:
            stack,norm,E,rotary,inheritance=make_stack(plan,stage,device)
            data=data_for(plan,cut,directory/"capture/packed_teacher_logits.npy")
            if phase=="pilot":
                output.mkdir();scorer=inherited._trainer(stack,norm,E,rotary,data,plan,stage,guard)
                initial,initial_rows=scorer.packed_development();raw=scorer.snapshot();del scorer;gc.collect()
                old.save(output/"initializer.pt",raw);base.write(output/"initial_development.json",dict(metrics=initial,records=initial_rows))
                names=[n for n,_ in stack.blocks[0].named_parameters()];buffers=[n for n,_ in stack.blocks[0].named_buffers()]
                pairs=inherited.entrance_pairs(plan,cut);order=sorted(data.training,key=lambda i:data.lengths[i],reverse=True)[:16]
                local=entrance.EntranceTrainer(stack,norm,E,rotary,pairs,config=ENTRANCE_CONFIG,guard=guard)
                local_probe=local.pilot(order);del local;gc.collect()
                stack.load_state_dict(raw["stack"],strict=True);norm.load_state_dict(raw["norm"],strict=True)
                restored=training.state_fingerprint(dict(stack=stack.state_dict(),norm=norm.state_dict()))==training.state_fingerprint(raw)
                if not local_probe["passed"] or not restored:raise RuntimeError("Entrance pilot/restore failed")
                trainer=_resident_trainer(stack,norm,E,rotary,data,plan,stage,settings,guard)
                probe=trainer.master_update_probe(order);old.save(output/"discarded_joint_state.pt",trainer.snapshot())
                if not probe["passed"]:raise RuntimeError("Resident joint pilot failed")
                snapshot=native._load(output/"discarded_joint_state.pt")
                stack.load_state_dict(snapshot["stack"],strict=True);norm.load_state_dict(snapshot["norm"],strict=True)
                result=inherited.prior.pilot_result(probe,initial,initial_rows,inheritance)
                result.update(entrance_probe=local_probe,entrance_restore_exact=restored,
                    initializer_sha256=base.sha(output/"initializer.pt"),initial_development_sha256=base.sha(output/"initial_development.json"),
                    entrance_parameter_names=names,entrance_buffer_names=buffers)
            elif phase=="prealign":
                pilot=checked(directory,"pilot",stage);raw=native._load(directory/"pilot/initializer.pt")
                stack.load_state_dict(raw["stack"],strict=True);norm.load_state_dict(raw["norm"],strict=True)
                pairs=inherited.entrance_pairs(plan,cut)
                local=entrance.EntranceTrainer(stack,norm,E,rotary,pairs,config=ENTRANCE_CONFIG,guard=guard)
                result=local.fit(output,event=event);delta=native._load(output/"selected_entrance.pt")
                base.write(output/"delta_validation.json",inherited.validate_delta(raw,delta,pilot,cut))
            elif phase=="train":
                pilot=checked(directory,"pilot",stage);raw=native._load(directory/"pilot/initializer.pt")
                delta=native._load(directory/"prealign/selected_entrance.pt");inherited.validate_delta(raw,delta,pilot,cut)
                stack.load_state_dict(raw["stack"],strict=True);norm.load_state_dict(raw["norm"],strict=True)
                stack.blocks[0].load_state_dict(delta["block"],strict=True)
                trainer=_resident_trainer(stack,norm,E,rotary,data,plan,stage,settings,guard)
                diagnostic=replay.DiagnosticCheckpoint(output,trainer,raw,pilot["initial"])
                def joint_event(kind,**values):
                    if kind=="development":diagnostic.observe(values["epoch"],{k:values[k] for k in pilot["initial"]})
                    event(kind,**values)
                result=trainer.fit(output,protected_state=raw,protected_metrics=pilot["initial"],protected_records=pilot["initial_records"],event=joint_event)
            else:
                output.mkdir();trained=checked(directory,"train",stage);before=old.fixed_state(stack,E,rotary)
                snapshot=native._load(directory/"train/selected.pt")
                stack.load_state_dict(snapshot["stack"],strict=True);norm.load_state_dict(snapshot["norm"],strict=True)
                scorer=inherited._trainer(stack,norm,E,rotary,data,plan,stage,guard);metrics,rows=scorer.packed_development()
                reference=read(directory/"train"/f"epoch_{trained['selected_epoch']:02d}_development.json")
                comparison=old.compare_rows(rows,reference["records"],[data.items[i] for i in data.development],expected_correct=trained["selected"]["correct"])
                fixed=old.fixed_state(stack,E,rotary)==before
                result=dict(status="completed",passed=comparison["passed"] and metrics==reference["metrics"] and fixed,
                    comparison=comparison,metrics=metrics,records=rows,fixed_state_unchanged=fixed,
                    storage=model.inventory(plan["model"]["path"],stack,norm,rotary),selected_checkpoint_sha256=base.sha(directory/"train/selected.pt"),
                    quality=trained["quality"],compressed_blocks=36-cut,confirmed=False)
                if not result["passed"]:raise RuntimeError("Fresh packed verification failed")
        if dual:
            base.write(output/"resident_optimizer.json",dict(optimizer=OPTIMIZER,memory=trainer.optimizer.memory_stats(),
                transfers=trainer.optimizer.transferred_bytes,transfer_accounting=replay.TRANSFER_ACCOUNTING))
        if not (output/"result.json").exists():base.write(output/"result.json",result)
        guard();base.write(output/"manifest.json",dict(status="completed",fingerprint=stage["fingerprint"],
            lease_id=lease["id"],allocation=lease["allocation"],allocator=settings,elapsed_seconds=time.monotonic()-started,
            peak_reserved_bytes={role:torch.cuda.max_memory_reserved(v["device"]) for role,v in settings.items()},
            artifacts={str(p.relative_to(output)):base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()}))
        return dict(status="completed",cut=cut,phase=phase)
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        if not (output/"failure.json").exists():base.write(output/"failure.json",dict(status="failed",error=repr(error),
            error_type=type(error).__name__,lease_id=None if lease is None else lease["id"],
            cuda_started=cuda_started))
        raise
    finally:
        for value in (data,pairs,teacher):
            if value is not None:value.close()


def campaign(root,*,stop_cut=0):
    root=Path(root).resolve();plan=verify_root(root)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Supervisor must hide CUDA")
    if type(stop_cut) is not int or not 0<=stop_cut<SEED_CUT:raise ValueError("Invalid stopping cut")
    for cut in range(SEED_CUT-1,stop_cut-1,-1):
        directory=stage_dir(root,cut)
        if not directory.exists():prepare_stage(root,cut)
        _,stage=verify_stage(root,cut)
        for phase in PHASES:
            if (directory/phase/"manifest.json").exists():checked(directory,phase,stage);continue
            if (directory/phase).exists():raise RuntimeError("Incomplete phase requires a new versioned attempt")
            allocation=["GPU-0370fdb0=11264"]+(["GPU-7b1db3cb=7168"] if phase in OPTIMIZER["resident_phases"] else [])
            command=["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation",",".join(allocation)]
            command += ["--minutes","120","--reason",f"{STUDY} cut{cut} {phase}","--","pixi","run","--locked","--offline","-e","ncm","python", "-m",MODULE,phase,"--root",str(root),"--cut",str(cut)]
            with (directory/f"{phase}.log").open("xb") as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            if process.returncode:raise RuntimeError(f"cut{cut} {phase} failed")
            checked(directory,phase,stage)
        trained=checked(directory,"train",stage);verified=checked(directory,"verify",stage)
        base.event(root/"progress.jsonl","stage_completed",cut=cut,compressed_blocks=36-cut,
            correct=trained["selected"]["correct"],quality=trained["quality"],storage=verified["storage"])
        if not trained["quality_pass"]:
            outcome=dict(status="quality_stop",last_accepted_cut=cut+1,failed_cut=cut,compressed_blocks=35-cut,
                selected=trained["selected"],quality=trained["quality"],confirmed=False)
            base.write(root/"outcome.json",outcome);return outcome
    outcome=dict(status="all_blocks_recovered" if stop_cut==0 else "requested_prefix_completed",
        last_accepted_cut=stop_cut,compressed_blocks=36-stop_cut,heldout_and_speed_pending=True,confirmed=False)
    base.write(root/("outcome.json" if stop_cut==0 else f"through_cut_{stop_cut:02d}.json"),outcome)
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","prepare-stage","campaign",*PHASES))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--seed-root",type=Path)
    parser.add_argument("--replay-root",type=Path);parser.add_argument("--cut",type=int)
    parser.add_argument("--stop-cut",type=int,default=0);args=parser.parse_args()
    if args.phase=="prepare":
        if args.seed_root is None or args.replay_root is None:parser.error("prepare requires both seed and replay roots")
        result=prepare(args.root,args.seed_root,args.replay_root)
    elif args.phase=="prepare-stage":result=prepare_stage(args.root,args.cut)
    elif args.phase=="campaign":result=campaign(args.root,stop_cut=args.stop_cut)
    else:result=run_phase(args.root,args.cut,args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
