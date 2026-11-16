"""Postselection recovery audit and raw generation evidence; never select weights."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_3b_broad_recovery as trial
from . import jspace_3b_broad_recovery_data as data_source
from . import jspace_3b_partial_generation as generation
from .jspace_full_lease import wait_ready

cache=trial.cache
base,old,read,sanity=cache.base,cache.old,cache.read,cache.sanity
SCHEMA="jspace.3b-broad-assessment.v1"
GENERATION_CONFIG=base.ROOT/"experiments/structure_net/jspace_3b_broad_generation_v1.json"
PHASES=("audit_native","audit_compact","generate_native","generate_seed","generate_selected")


def canary_inputs(config,tokenizer):
    if (config.get("schema")!="jspace.3b-broad-generation.v1" or len(config.get("canaries",[]))!=6
            or config.get("generation",{}).get("use_cache") is not False
            or config["generation"].get("max_new_tokens")!=64 or config["generation"].get("position_capacity")!=416):
        raise ValueError("Frozen six-task cache-free generation configuration required")
    result=[]
    for case in config["canaries"]:
        ids=list(tokenizer.apply_chat_template([dict(role="user",content=case["prompt"])],
            tokenize=True,add_generation_prompt=True))
        if not ids or len(ids)+64>416:raise ValueError("Untruncated canary plus budget exceeds capacity")
        result.append(dict(case,input_ids=ids))
    if len({c["id"] for c in result})!=6:raise ValueError("Distinct canary IDs required")
    return result


def audit_summary(native,compact,cases,*,development_passed):
    sanity.validate_records(native,cases);sanity.validate_records(compact,cases)
    quality=sanity.paired_quality(native,compact)
    overall=quality["overall"]["accuracy_drop"]<=.05
    families={f:v["accuracy_drop"]<=.10 for f,v in quality["by_family"].items()}
    return dict(quality=quality,descriptive_audit_passed=overall and all(families.values()),
        overall_passed=overall,family_passed=families,maximum_overall_drop=.05,maximum_family_drop=.10,
        development_passed=development_passed,audit_can_rescue_development=False,confirmed=False,
        scope="One sealed recovery audit after global selection; not a population noninferiority proof or final512 confirmation")


def reused_generation(seed,*,seed_sha,selected_sha,seed_manifest_sha):
    if seed_sha!=selected_sha or seed.get("checkpoint_sha256")!=seed_sha or seed.get("phase")!="generate_seed":
        raise ValueError("Generation reuse requires identical authenticated checkpoint bytes")
    return dict(deepcopy(seed),phase="generate_selected",reused_from="generate_seed",
        reused_manifest_sha256=seed_manifest_sha,reuse_reason="Identical checkpoint, tokenizer, prompts and generation settings")


def development_controls(recovery_root,cache_root,candidate):
    """Bind one existing DEV row to each checkpoint before opening the sealed audit."""
    recovery_root,cache_root=Path(recovery_root),Path(cache_root)
    arm=candidate["arm"]
    if arm not in cache.ARMS:raise ValueError("Selected recovery arm required")
    pilot=read(recovery_root/arm/"pilot/result.json")
    trained=read(recovery_root/arm/"train/result.json")
    items=read(cache_root/"items.json")["items"]
    dev=[c for c in items if c["split"]=="development"]
    selected=read(candidate["report"])["records"]
    sanity.validate_records(selected,dev);sanity.validate_records(pilot["initial_records"],dev)
    if len(dev)!=256 or pilot["probe"]["fixed_state"]!=trained["fixed_state"]:
        raise ValueError("Original matched DEV256 and complete fixed-state receipt required")
    return dict(case=dev[0],boundary_path=str((cache_root/"native/boundary_22.npy").resolve()),
        boundary_shape=[items[-1]["stop"],2048],
        seed=dict(record=pilot["initial_records"][0],fixed_state=pilot["probe"]["fixed_state"],
                  checkpoint_sha256=pilot["checkpoint_sha256"]),
        selected=dict(record=selected[0],fixed_state=trained["fixed_state"],
                      checkpoint_sha256=candidate["checkpoint_sha256"]),
        scope="First existing DEV case, singleton: exact cached native prefix and saved choice logits; no selection")


def control_boundary(controls):
    case=controls["case"];bank=np.load(controls["boundary_path"],mmap_mode="r")
    try:
        if bank.dtype!=np.int16 or list(bank.shape)!=controls["boundary_shape"]:
            raise ValueError("Pinned DEV prefix bank layout changed")
        return torch.from_numpy(np.array(bank[case["offset"]:case["stop"]],copy=True)).view(torch.bfloat16)[None]
    finally:bank._mmap.close()


def require_development_record(actual,expected):
    if actual!=expected:raise ValueError("Compact development choice record differs from saved checkpoint")
    return dict(passed=True,records_exact=True,actual=actual,expected=expected)


def partial_development_control(model,controls,which,label_ids,guard):
    actual,boundary=sanity.native_case(model,controls["case"],label_ids,22,guard)
    prefix=sanity.boundary_control(boundary,control_boundary(controls))
    return dict(**require_development_record(actual,controls[which]["record"]),boundary=prefix)


def prepare(root,recovery_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Assessment preparation requires CUDA hidden")
    root,recovery_root=Path(root).resolve(),Path(recovery_root).resolve()
    if root.exists():raise FileExistsError(root)
    # This MUST precede reading either the sealed audit or any of its labels.
    selection=trial.checked_selection(recovery_root)
    cp=cache.verify(selection["cache_root"]);data_root=Path(cp["data_root"])
    data_source.verify(data_root)  # Hash/identity verification never decodes audit content.
    candidate=selection["candidate"]
    if base.sha(candidate["checkpoint"])!=candidate["checkpoint_sha256"]:
        raise ValueError("Globally selected checkpoint changed")
    controls=development_controls(recovery_root,selection["cache_root"],candidate)
    if controls["seed"]["checkpoint_sha256"]!=cp["checkpoint_sha256"]:
        raise ValueError("Seed development control belongs to another checkpoint")
    audit=read(data_root/"audit_tasks.json")["splits"]["audit"]
    identities=read(data_root/"audit_identities.json")
    if (len(audit)!=128 or len({c["identity"] for c in audit})!=128
            or {c["identity"] for c in audit}!=set(identities["identities"])
            or Counter(c["family"] for c in audit)!=dict.fromkeys(("arc","boolq","hellaswag","winogrande"),32)
            or any(c["split"]!="audit" or not 3<=len(c["input_ids"])<=384 for c in audit)):
        raise ValueError("Original sealed128 balanced audit identity required")
    offset=0;items=[]
    for case in audit:
        items.append(dict(case,offset=offset,stop=offset+len(case["input_ids"])));offset=items[-1]["stop"]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cp["model"]["path"],local_files_only=True,trust_remote_code=False)
    config=read(GENERATION_CONFIG);canaries=canary_inputs(config,tokenizer)
    inputs=dict(cp["input_sha256"])
    paths=[recovery_root/"plan.json",recovery_root/"selection.json",recovery_root/"manifest.json",
        Path(selection["cache_root"])/"plan.json",data_root/"plan.json",data_root/"manifest.json",
        data_root/"audit_tasks.json",data_root/"audit_identities.json",GENERATION_CONFIG,cache.PROTOCOL,
        Path(candidate["checkpoint"]),Path(candidate["report"]),
        Path(selection["cache_root"])/"items.json",Path(selection["cache_root"])/"native/manifest.json",
        Path(controls["boundary_path"])]
    for arm in cache.ARMS:
        for phase in ("pilot","train","verify"):
            directory=recovery_root/arm/phase
            paths.append(directory/"manifest.json")
            # checked_selection already authenticated these; pin identities without rehashing large weights.
            for name,digest in read(directory/"manifest.json")["artifacts"].items():
                inputs[str((directory/name).resolve())]=digest
    for path in paths:old.pin(inputs,path)
    files=cache.source.inherited.local_import_closure([Path(__file__),GENERATION_CONFIG,cache.PROTOCOL,
        base.ROOT/"tests/structure_net/test_jspace_3b_broad_assess.py",
        base.ROOT/"tests/structure_net/test_jspace_3b_partial_generation.py"])
    plan=dict(schema=SCHEMA,recovery_root=str(recovery_root),recovery_fingerprint=selection["fingerprint"],
        cache_root=selection["cache_root"],data_root=str(data_root),selection=selection,
        model=cp["model"],fits=cp["fits"],label_ids=cp["label_ids"],cut=22,hidden_size=2048,
        checkpoint=candidate["checkpoint"],checkpoint_sha256=candidate["checkpoint_sha256"],
        seed_checkpoint=cp["checkpoint"],seed_checkpoint_sha256=cp["checkpoint_sha256"],
        compact_development_controls=controls,
        development_control=cp["development_control"],native_boundary=str(Path(cp["native_cache"])/"boundary_22.npy"),
        items_digest=base.digest(items),canaries_digest=base.digest(canaries),total_tokens=offset,
        generation_config_sha256=base.sha(GENERATION_CONFIG),generation=config["generation"],
        phases=list(PHASES),allocation_mib=11264,allocator_mib=11008,confirmed=False,
        input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        source_sha256={str(p):base.sha(p) for p in sorted(files)})
    plan["fingerprint"]=base.digest(plan);root.mkdir(parents=True)
    for name,digest in plan["source_sha256"].items():
        target=root/"producer"/Path(name).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        if base.sha(target)!=digest:raise ValueError("Assessment producer archive changed")
    base.write(root/"items.json",dict(items=items));base.write(root/"canaries.json",dict(canaries=canaries))
    shutil.copy2(GENERATION_CONFIG,root/"generation_config.json");base.write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],audit_count=128,canary_count=6)


def verify(root):
    root=Path(root);p=read(root/"plan.json");cache.source.check_plan(p)
    if (p.get("schema")!=SCHEMA or p.get("cut")!=22 or p.get("phases")!=list(PHASES)
            or p.get("allocation_mib")!=11264 or p.get("allocator_mib")!=11008 or p.get("confirmed") is not False
            or p["checkpoint_sha256"]!=p["selection"]["candidate"]["checkpoint_sha256"]
            or p["seed_checkpoint_sha256"]!=cache.SEED_SHA
            or p["compact_development_controls"]["selected"]["checkpoint_sha256"]!=p["checkpoint_sha256"]
            or p["compact_development_controls"]["seed"]["checkpoint_sha256"]!=p["seed_checkpoint_sha256"]):
        raise ValueError("Assessment admission policy changed")
    for path,identity in p["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Assessment input changed: "+path)
    for path,digest in p["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError("Assessment source changed")
    if (base.digest(read(root/"items.json")["items"])!=p["items_digest"]
            or base.digest(read(root/"canaries.json")["canaries"])!=p["canaries_digest"]
            or base.sha(root/"generation_config.json")!=p["generation_config_sha256"]):
        raise ValueError("Frozen assessment prompts/configuration changed")
    return p


def validate_generation(rows,canaries):
    if len(rows)!=len(canaries):raise ValueError("All six generation records required")
    for row,case in zip(rows,canaries):
        if (row.get("id")!=case["id"] or row.get("input_ids")!=case["input_ids"]
                or not isinstance(row.get("response"),str) or len(row.get("output_ids",[]))>64
                or any(type(t)is not int or t<0 for t in row.get("output_ids",[]))
                or row.get("use_cache") is not False or row.get("stop_reason") not in ("eos","token_limit")
                or row.get("generation_config",{}).get("use_cache") is not False
                or row["generation_config"].get("max_new_tokens")!=64):
            raise ValueError("Generation prompt, budget, cache or output receipt changed")


def check_compact_controls(result,plan,which,*,full_partial):
    expected=plan["compact_development_controls"][which]
    if (result.get("fixed_state_control")!={"passed":True,"expected":expected["fixed_state"],"actual":expected["fixed_state"]}
            or not result.get("development_control",{}).get("records_exact")
            or result["development_control"].get("actual")!=expected["record"]
            or result["development_control"].get("expected")!=expected["record"]
            or (full_partial and not result["development_control"].get("boundary",{}).get("bit_exact"))):
        raise ValueError("Original fixed-state and compact development replay controls required")


def checked(root,phase,plan=None):
    root=Path(root);p=verify(root) if plan is None else plan
    if phase not in PHASES:raise ValueError("Unknown assessment phase")
    directory=root/phase;m=read(directory/"manifest.json");r=read(directory/"result.json")
    if (m.get("status")!="completed" or m.get("phase")!=phase or m.get("fingerprint")!=p["fingerprint"]
            or r.get("status")!="completed" or r.get("phase")!=phase or r.get("fingerprint")!=p["fingerprint"]
            or m.get("artifacts",{}).get("result.json")!=base.sha(directory/"result.json")):
        raise ValueError("Completed matching assessment phase required")
    for name,digest in m["artifacts"].items():
        if not (directory/name).resolve().is_relative_to(directory.resolve()) or base.sha(directory/name)!=digest:
            raise ValueError("Assessment artifact changed")
    if r.get("reused_from") is not None:
        seed=checked(root,"generate_seed",p)
        expected=reused_generation(seed,seed_sha=p["seed_checkpoint_sha256"],selected_sha=p["checkpoint_sha256"],
            seed_manifest_sha=base.sha(root/"generate_seed/manifest.json"))
        if phase!="generate_selected" or r!=expected or m.get("reused_from")!="generate_seed":
            raise ValueError("Unauthenticated generation reuse")
    else:
        a=m.get("allocator",{});uuid=a.get("physical_uuid","")
        if (not uuid.startswith("GPU-0370fdb0") or m.get("allocation")!={uuid:11264}
                or a.get("allocator_limit_bytes")!=11008*1024**2 or not 0<m.get("peak_reserved_bytes",0)<=11008*1024**2):
            raise ValueError("Bounded RTX3060 reservation required")
    if phase.startswith("audit_"):
        cases=read(root/"items.json")["items"];sanity.validate_records(r["records"],cases)
        if r.get("count")!=128 or r.get("correct")!=sum(x["prediction"]==x["answer"] for x in r["records"]):
            raise ValueError("Audit counts do not match actual records")
        if phase=="audit_native":
            if not r.get("boundary_control",{}).get("bit_exact") or "boundary_22.npy" not in m["artifacts"]:
                raise ValueError("Original native prefix control/bank required")
        elif (not r.get("strict_state_load") or r.get("checkpoint_sha256")!=p["checkpoint_sha256"]
                or r.get("boundary_sha256")!=base.sha(root/"audit_native/boundary_22.npy")
                or not r.get("inheritance",{}).get("passed")
                or r.get("inheritance",{}).get("unchanged_layers")!=list(range(22,36))
                or not r.get("inheritance",{}).get("norm_exact")):
            raise ValueError("Selected compact state or native prefix changed")
        if phase=="audit_compact":check_compact_controls(r,p,"selected",full_partial=False)
    else:
        validate_generation(r["records"],read(root/"canaries.json")["canaries"])
        expected_sha=None if phase=="generate_native" else p["seed_checkpoint_sha256"] if phase=="generate_seed" else p["checkpoint_sha256"]
        if (r.get("checkpoint_sha256")!=expected_sha or r.get("generation_config_sha256")!=p["generation_config_sha256"]):
            raise ValueError("Wrong generation checkpoint or pinned configuration")
        if phase!="generate_native":
            check_compact_controls(r,p,"seed" if phase=="generate_seed" else "selected",full_partial=True)
    return r


def run(root,phase):
    root=Path(root).resolve();p=verify(root)
    if phase not in PHASES:raise ValueError("Unknown assessment phase")
    for previous in PHASES[:PHASES.index(phase)]:checked(root,previous,p)
    out=root/phase
    if out.exists():raise FileExistsError(out)
    if phase=="generate_selected" and p["checkpoint_sha256"]==p["seed_checkpoint_sha256"]:
        result=reused_generation(checked(root,"generate_seed",p),seed_sha=p["seed_checkpoint_sha256"],
            selected_sha=p["checkpoint_sha256"],seed_manifest_sha=base.sha(root/"generate_seed/manifest.json"))
        out.mkdir();base.write(out/"result.json",result)
        base.write(out/"manifest.json",dict(status="completed",phase=phase,fingerprint=p["fingerprint"],
            reused_from="generate_seed",artifacts={"result.json":base.sha(out/"result.json")}))
        return dict(status="completed",phase=phase,reused_from="generate_seed")
    lease=settings=None;cuda=False;started=time.monotonic()
    try:
        lease=wait_ready([0]);settings=cache.source.inherited.cap_allocator(lease,p)
        torch.cuda.init();cuda=True;torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4)
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Lease changed")
            if torch.cuda.max_memory_reserved()>settings["allocator_limit_bytes"]:raise MemoryError("Allocator cap exceeded")
        out.mkdir();records=[];detail={}
        with sanity.execution.arithmetic_policy():
            if phase=="audit_native":
                model=sanity.execution.load_native_model(p["model"]["path"],device="cuda:0",attention="eager",guard=guard)
                c=p["development_control"];_,actual=sanity.native_case(model,c,p["label_ids"],22,guard)
                prior=np.load(p["native_boundary"],mmap_mode="r")
                expected=torch.from_numpy(np.array(prior[c["offset"]:c["stop"]],copy=True)).view(torch.bfloat16)[None]
                detail["boundary_control"]=sanity.boundary_control(actual,expected);del prior,actual,expected
                bank=np.lib.format.open_memmap(out/"boundary_22.npy",mode="w+",dtype=np.int16,shape=(p["total_tokens"],2048))
                for case in read(root/"items.json")["items"]:
                    row,hidden=sanity.native_case(model,case,p["label_ids"],22,guard)
                    bank[case["offset"]:case["stop"]]=hidden[0].contiguous().view(torch.int16).numpy();records.append(row)
                bank.flush();bank._mmap.close();del bank,model
            elif phase=="audit_compact":
                stack,norm,E,cpu_rotary,proof=cache.builder.build_stack(p["model"]["path"],{int(k):v for k,v in p["fits"].items()},
                    22,p["checkpoint"],device="cpu",trainable=False,checkpoint_blocks=False)
                controls=p["compact_development_controls"]
                rotary=generation.training_rotary(cache.builder.native._config(p["model"]["path"],None),"cuda:0",guard=guard)
                fixed=generation.assert_fixed_state(stack,E,rotary,controls["selected"]["fixed_state"])
                packed=cache.builder.packed_clone(stack,device="cpu");del stack,cpu_rotary
                packed,norm,E=packed.to("cuda:0"),norm.to("cuda:0"),E.to("cuda:0")
                control=sanity.compact_case(packed,norm,E,rotary,control_boundary(controls),controls["case"],p["label_ids"],guard)
                replay=require_development_record(control,controls["selected"]["record"])
                bank=np.load(root/"audit_native/boundary_22.npy",mmap_mode="r")
                if bank.dtype!=np.int16 or bank.shape!=(p["total_tokens"],2048):raise ValueError("Native audit bank changed")
                for case in read(root/"items.json")["items"]:
                    hidden=torch.from_numpy(np.array(bank[case["offset"]:case["stop"]],copy=True)).view(torch.bfloat16)[None]
                    records.append(sanity.compact_case(packed,norm,E,rotary,hidden,case,p["label_ids"],guard))
                detail=dict(strict_state_load=True,inheritance=proof,checkpoint_sha256=p["checkpoint_sha256"],
                    boundary_sha256=base.sha(root/"audit_native/boundary_22.npy"),
                    fixed_state_control=fixed,development_control=replay);bank._mmap.close()
            else:
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(p["model"]["path"],local_files_only=True,trust_remote_code=False)
                checkpoint=None if phase=="generate_native" else p["seed_checkpoint"] if phase=="generate_seed" else p["checkpoint"]
                which="seed" if phase=="generate_seed" else "selected"
                model=sanity.execution.load_native_model(p["model"]["path"],device="cuda:0",attention="eager",guard=guard) if checkpoint is None else generation.build_partial(
                    p["model"]["path"],{int(k):v for k,v in p["fits"].items()},checkpoint,cut=22,device="cuda:0",guard=guard,
                    expected_fixed_state=p["compact_development_controls"][which]["fixed_state"])
                if checkpoint is not None:
                    detail.update(fixed_state_control=model.partial_fixed_state_receipt,
                        development_control=partial_development_control(model,p["compact_development_controls"],which,p["label_ids"],guard),
                        rotary_scope=model.partial_generation_scope)
                for case in read(root/"canaries.json")["canaries"]:
                    value=generation.generate(model,tokenizer,case["input_ids"],max_new_tokens=64,position_capacity=416,guard=guard)
                    records.append(dict(id=case["id"],category=case["category"],**value))
                detail.update(checkpoint_sha256=None if checkpoint is None else base.sha(checkpoint),strict_state_load=True,
                    semantic_review_pending=True,use_cache=False,generation_config_sha256=p["generation_config_sha256"])
        guard();result=dict(status="completed",phase=phase,fingerprint=p["fingerprint"],records=records,**detail)
        if phase.startswith("audit_"):
            sanity.validate_records(records,read(root/"items.json")["items"])
            result.update(count=len(records),correct=sum(r["prediction"]==r["answer"] for r in records))
        else:validate_generation(records,read(root/"canaries.json")["canaries"])
        base.write(out/"result.json",result)
        base.write(out/"manifest.json",dict(status="completed",phase=phase,fingerprint=p["fingerprint"],lease_id=lease["id"],
            allocation=lease["allocation"],allocator=settings,peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started,
            artifacts={f.name:base.sha(f) for f in out.iterdir() if f.is_file()}))
        return dict(status="completed",phase=phase)
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True);base.write(out/"failure.json",dict(status="failed",phase=phase,error=repr(error),
            lease_id=None if lease is None else lease["id"],cuda_started=cuda));raise
    finally:gc.collect()


def finalize(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Assessment finalization requires CUDA hidden")
    root=Path(root);p=verify(root);results={phase:checked(root,phase,p) for phase in PHASES}
    summary=audit_summary(results["audit_native"]["records"],results["audit_compact"]["records"],read(root/"items.json")["items"],
        development_passed=p["selection"]["development_passed"])
    generation_results={phase:results[phase]["records"] for phase in PHASES if phase.startswith("generate_")}
    settings=[row["generation_config"] for rows in generation_results.values() for row in rows]
    if any(value!=settings[0] for value in settings[1:]):raise ValueError("Generation settings differ between arms/prompts")
    result=dict(status="completed",fingerprint=p["fingerprint"],selection=p["selection"],audit=summary,
        generation=generation_results,semantic_requirements=read(root/"canaries.json")["canaries"],semantic_review_pending=True,
        phase_manifest_sha256={phase:base.sha(root/phase/"manifest.json") for phase in PHASES},
        confirmed=False,trained=False,speed_measured=False,compressed_layers=14,full36=False,
        scope="Frozen postselection128 audit plus six raw generation canaries; cannot select another candidate or rescue failed development")
    base.write(root/"result.json",result);base.write(root/"manifest.json",dict(status="completed",fingerprint=p["fingerprint"],
        artifacts={"result.json":base.sha(root/"result.json")},confirmed=False))
    return dict(status="completed",audit=summary,semantic_review_pending=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare",*PHASES,"finalize"))
    parser.add_argument("--root",type=Path,default=base.ROOT/"data/experiments/jspace_3b_broad_assess/20261115_v1")
    parser.add_argument("--recovery-root",type=Path,default=base.ROOT/"data/experiments/jspace_3b_broad_recovery/20261115_v1")
    args=parser.parse_args();value=prepare(args.root,args.recovery_root) if args.phase=="prepare" else finalize(args.root) if args.phase=="finalize" else run(args.root,args.phase)
    print(json.dumps(value,allow_nan=False),flush=True)
