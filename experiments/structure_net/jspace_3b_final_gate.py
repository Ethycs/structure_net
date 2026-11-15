"""Reserved final validation of accepted full36 Qwen2.5-3B compression.

A separate immutable lifecycle: no training, candidate selection, native-prefix
bypass or reuse of the old already-inspected512-task evaluation population.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from . import jspace_qwen3b_progressive as campaign
from . import jspace_recovery_data as confirmation
from . import jspace_3b_final_evaluate as evaluator

base=campaign.base; old=campaign.old; read=campaign.read
SCHEMA="jspace.qwen3b-final-gate.v1"
STUDY="JSPACE-QWEN3B-FULLMODEL-GATE-01"
PROTOCOL=base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-full-model-gate.md"
PHASES=("compact_export","compact_validate","native_eager_before","native_sdpa_before","compact","native_sdpa_after")
CONFIG=dict(seed=20861,confirmation_count=512,per_family=128,max_input_tokens=384,
    benchmark_prompts=8,repeats=5,generated_tokens=32,allocation_mib=11264,allocator_mib=11008,
    cache_capacity=416,logit_atol=.03125,logit_rtol=1/128,
    maximum_accuracy_drop=.05,minimum_size_reduction=.38,minimum_speedup=1.4)


def _cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU lifecycle must hide CUDA")


def _pin(inputs,path,expected=None):
    path=Path(path).resolve();digest=base.sha(path)
    if expected is not None and digest!=expected:raise ValueError("Changed input: "+str(path))
    inputs[str(path)]=digest


def admit_campaign(root):
    """Authenticate every accepted parent transition before opening confirmation."""
    root=Path(root).resolve();schema=read(root/"plan.json").get("schema")
    if schema==campaign.SCHEMA:
        producer=campaign
    elif schema=="jspace.qwen3b-native-budget.v1":
        producer=importlib.import_module(".jspace_qwen3b_native_budget",package=__package__)
    else:
        raise ValueError("Unsupported compression campaign schema")
    plan=producer.verify_root(root);outcome=read(root/"outcome.json")
    if outcome.get("status")!="all_blocks_recovered" or outcome.get("last_accepted_cut")!=0 or outcome.get("compressed_blocks")!=36:
        raise ValueError("Accepted, completed full36 campaign required")
    chain=[];inputs={};previous_sha=plan["seed_checkpoint_sha256"];previous_correct=plan["seed_correct"]
    seed_gate=None
    if "seed_gate_receipt" in plan:
        seed_path=Path(plan["seed_gate_receipt"]);_pin(inputs,seed_path)
        seed_gate=read(seed_path)
    for cut in range(plan["seed_cut"]-1,-1,-1):
        directory=producer.stage_dir(root,cut);stage=read(directory/"plan.json");producer.check_plan(stage)
        if (stage.get("root_fingerprint")!=plan["fingerprint"] or stage.get("cut")!=cut or
                stage.get("parent_checkpoint_sha256")!=previous_sha or stage.get("parent_correct")!=previous_correct):
            raise ValueError("Broken accepted-parent chain")
        trained=producer.checked(directory,"train",stage);verified=producer.checked(directory,"verify",stage)
        if not trained["quality_pass"] or not verified["passed"]:raise ValueError("Failed stage cannot enter final confirmation")
        for path in (directory/"plan.json",directory/"train/manifest.json",directory/"train/result.json",
                     directory/"verify/manifest.json",directory/"verify/result.json"):_pin(inputs,path)
        # checked() already hashes this actual selected checkpoint, not only metadata.
        inputs[str((directory/"train/selected.pt").resolve())]=trained["selected_checkpoint_sha256"]
        chain.append(dict(cut=cut,fingerprint=stage["fingerprint"],checkpoint_sha256=trained["selected_checkpoint_sha256"],
            correct=trained["selected"]["correct"],quality=trained["quality"]))
        previous_sha,previous_correct=trained["selected_checkpoint_sha256"],trained["selected"]["correct"]
    selected=root/"stages/block_00/train/selected.pt"
    reference=root/"stages/block_00/train"/f"epoch_{trained['selected_epoch']:02d}_development.json"
    for path in (root/"plan.json",root/"outcome.json",reference):_pin(inputs,path)
    return dict(root=str(root),plan=plan,stage=stage,training=trained,verification=verified,chain=chain,
        seed_gate_receipt=seed_gate,checkpoint=str(selected),selected_reference=str(reference),input_sha256=inputs)


def _recorded_predictions(value):
    found=set()
    if isinstance(value,dict):
        if isinstance(value.get("identity"),str) and any(k in value for k in ("prediction","teacher_prediction")):
            found.add(value["identity"])
        for child in value.values():found.update(_recorded_predictions(child))
    elif isinstance(value,list):
        for child in value:found.update(_recorded_predictions(child))
    return found


def validate_confirmation(cases,used):
    ids=[c["identity"] for c in cases]
    if (len(ids)!=512 or len(set(ids))!=512 or set(ids)&used or
            Counter(c["family"] for c in cases)!={f:128 for f in confirmation.DATASETS}):
        raise ValueError("Confirmation count/families/identity separation failed")
    if any(not 0<len(c["input_ids"])<=384 or not 0<=c["answer"]<len(c["choices"]) for c in cases):
        raise ValueError("Invalid original confirmation token/label layout")
    return dict(count=512,families=dict(Counter(c["family"] for c in cases)),used_identity_count=len(used),overlap=0)


def historical_usage(root,confirmation_root):
    """Inspect task roles and actual prediction records; exclusions are not use."""
    used=set();paths=[]
    for path in sorted(Path(root).glob("jspace*/**/*.json")):
        if "producer" in path.parts or Path(confirmation_root).resolve() in path.resolve().parents:continue
        name=path.name
        # Score consumers have used several filenames (including quality.json
        # and native_records.json). Inspect every JSON payload, not just names.
        value=read(path);local=_recorded_predictions(value)
        if name=="tasks.json" and isinstance(value,dict):
            for split,rows in value.get("splits",{}).items():
                if split!="confirmation":local.update(c["identity"] for c in rows if isinstance(c,dict) and "identity" in c)
        if name=="items.json":
            rows=value.get("items",[]) if isinstance(value,dict) else value
            if isinstance(rows,list):local.update(c["identity"] for c in rows if isinstance(c,dict) and
                c.get("split") in ("train","development","calibration","evaluation","test") and "identity" in c)
        if local:used.update(local);paths.append(path)
    return used,paths


def benchmark_cases(items):
    selected=[]
    for i,family in enumerate(confirmation.DATASETS):
        rows=[c for c in items if c.get("split")=="development" and c["family"]==family]
        if len(rows)!=16:raise ValueError("Expected original16 development cases per family")
        random.Random(CONFIG["seed"]+i).shuffle(rows);selected.extend(rows[:2])
    if any(len(c["input_ids"])+32>CONFIG["cache_capacity"] for c in selected):raise ValueError("Benchmark exceeds deployed position capacity")
    return selected


def prepare(root,campaign_root,confirmation_root=confirmation.DEFAULT_ROOT):
    _cpu();root=Path(root).resolve();confirmation_root=Path(confirmation_root).resolve()
    if root.exists():raise FileExistsError(root)
    accepted=admit_campaign(campaign_root)  # No confirmation is opened before this gate.
    confirmation_receipt=confirmation.verify(confirmation_root)
    conf_plan=read(confirmation_root/"plan.json");tasks=read(confirmation_root/"tasks.json")
    cases=tasks["splits"]["confirmation"]
    used,usage_paths=historical_usage(base.ROOT/"data/experiments",confirmation_root)
    items=read(accepted["plan"]["items_path"])["items"]
    used.update(c["identity"] for c in items)
    separation=validate_confirmation(cases,used)
    if conf_plan["tokenizer_model"]["revision"]!=accepted["plan"]["model"]["revision"]:
        raise ValueError("Confirmation tokenization differs from pinned native model")
    inputs=dict(accepted["input_sha256"])
    for path in [*[confirmation_root/n for n in ("plan.json","tasks.json","exclusions.json","manifest.json")],
                 accepted["plan"]["items_path"],*usage_paths]:_pin(inputs,path)
    model=accepted["plan"]["model"]
    for name,digest in model["files"].items():_pin(inputs,Path(model["path"])/name,digest)
    for directory in accepted["stage"]["fits"].values():
        for name in ("fit.pt","relation.pt","fit.json","relations.json","interface.json"):_pin(inputs,Path(directory)/name)
    boundary=Path(accepted["plan"]["native_cache"])/"boundary_00.npy";_pin(inputs,boundary)
    source=dict(accepted["plan"]["source_sha256"])
    extras=[Path(__file__),Path(__file__).with_name("jspace_3b_final_evaluate.py"),
        Path(__file__).with_name("jspace_allblocks_artifact.py"),Path(__file__).with_name("jspace_qwen_parallel_evaluate.py"),
        Path(confirmation.__file__),PROTOCOL,base.ROOT/"pyproject.toml",base.ROOT/"pixi.lock",
        base.ROOT/"tests/structure_net/test_jspace_3b_final_gate.py",
        base.ROOT/"tests/structure_net/test_jspace_3b_final_evaluate.py"]
    for path in extras:source[str(path.resolve())]=base.sha(path)
    plan=dict(schema=SCHEMA,study_id=STUDY,config=CONFIG,campaign_root=accepted["root"],
        campaign_fingerprint=accepted["plan"]["fingerprint"],stage_fingerprint=accepted["stage"]["fingerprint"],
        campaign_schema=accepted["plan"]["schema"],seed_gate_receipt=accepted["seed_gate_receipt"],
        accepted_chain=accepted["chain"],selected_checkpoint=accepted["checkpoint"],selected_checkpoint_sha256=accepted["training"]["selected_checkpoint_sha256"],
        selected_reference=accepted["selected_reference"],fits=accepted["stage"]["fits"],model=model,
        confirmation_root=str(confirmation_root),confirmation_receipt=confirmation_receipt,separation=separation,
        label_ids=accepted["plan"]["label_ids"],native_boundary_zero=str(boundary),
        input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},source_sha256=source,confirmed=False)
    frozen=dict(confirmation=cases,development=[c for c in items if c["split"]=="development"],
        all_original_items=items,benchmark=benchmark_cases(items))
    plan["tasks_digest"]=base.digest(frozen);plan["fingerprint"]=base.digest(plan)
    root.mkdir(parents=True)
    for path,digest in source.items():
        if base.sha(path)!=digest:raise ValueError("Source changed during final preparation")
        target=root/"producer"/Path(path).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    base.write(root/"tasks.json",frozen);base.write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],confirmation_count=512,compressed_blocks=36)


def verify(root):
    root=Path(root);plan=read(root/"plan.json");campaign.check_plan(plan)
    if plan.get("schema")!=SCHEMA or plan.get("config")!=CONFIG:raise ValueError("Final protocol changed")
    if base.digest(read(root/"tasks.json"))!=plan["tasks_digest"]:raise ValueError("Frozen final tasks changed")
    for path,identity in plan["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Final selected input changed: "+path)
    for path,digest in plan["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError("Frozen final producer changed: "+path)
    return plan


def checked(root,phase,plan):
    path=Path(root)/phase;manifest=read(path/"manifest.json")
    if manifest.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"] or "result.json" not in manifest.get("artifacts",{}):
        raise ValueError("Complete authenticated final phase required")
    for name,digest in manifest["artifacts"].items():
        file=path/name
        if not file.resolve().is_relative_to(path.resolve()) or base.sha(file)!=digest:raise ValueError("Final phase artifact changed")
    result=read(path/"result.json")
    if result.get("status")!="completed" or result.get("phase")!=phase or result.get("fingerprint")!=plan["fingerprint"]:
        raise ValueError("Incomplete or mismatched final phase")
    if phase in ("compact_export","compact_validate") and "development.json" not in manifest["artifacts"]:
        raise ValueError("Selected-development records must be authenticated")
    if phase=="compact_export":
        if (not result.get("embedding_control",{}).get("exact") or not result.get("selected_development",{}).get("passed") or
                any(not result.get("native_cache_controls",{}).get(a,{}).get("passed") for a in ("eager","sdpa"))):
            raise ValueError("Native/embedding/selected-development export controls failed")
    elif phase=="compact_validate":
        if (not result.get("fresh_export_reload_exact") or not result.get("selected_development",{}).get("passed") or
                not result.get("cache",{}).get("passed")):
            raise ValueError("Fresh packed validation must pass before confirmation")
    else:
        if "quality.json" not in manifest["artifacts"]:raise ValueError("Confirmation records are not authenticated")
        rows=read(path/"quality.json")["records"]
        expected=read(Path(root)/"tasks.json")["confirmation"]
        if len(rows)!=512 or len(expected)!=512 or any(
                any(a[k]!=b[k] for k in ("identity","family","answer")) for a,b in zip(rows,expected)):
            raise ValueError("Confirmation record population/order changed")
        old.compare_rows(rows,rows,expected)  # finite logits, argmax and choice-count integrity
        runtime=result["runtime"];timing=runtime["timings"]["cuda_graph"]
        samples=timing["samples"]
        if (not runtime["graph_static_exact"] or len(samples)!=40 or timing["total_tokens"]!=1280 or
                any(row["tokens"]!=32 or not math.isfinite(row["seconds"]) or row["seconds"]<=0 for row in samples) or
                not math.isclose(sum(row["seconds"] for row in samples),timing["total_seconds"],rel_tol=1e-12) or
                not math.isclose(1280/timing["total_seconds"],timing["tokens_per_second"],rel_tol=1e-12)):
            raise ValueError("Runtime samples/settings/parity are invalid")
        benchmark=read(Path(root)/"tasks.json")["benchmark"]
        if Counter((row["repeat"],row["identity"]) for row in samples)!=Counter(
                (repeat,case["identity"]) for repeat in range(5) for case in benchmark):
            raise ValueError("Runtime prompt/repeat pairing changed")
        if phase=="compact" and (not result.get("fresh_export_reload_exact") or not result.get("confirmation_reload_exact") or
                "reload_quality.json" not in manifest["artifacts"] or read(path/"reload_quality.json")["records"]!=rows):
            raise ValueError("Packed confirmation reload must be exact")
        if phase=="native_sdpa_after" and (not result.get("baseline_quality_exact") or
                rows!=read(Path(root)/"native_sdpa_before/quality.json")["records"]):
            raise ValueError("Native SDPA baseline changed across the bracket")
    return result


def _model_bytes(model):return evaluator.builder.tensor_bytes(model)


@torch.inference_mode()
def embedding_control(model,items,path,guard):
    bank=np.load(path,mmap_mode="r",allow_pickle=False)
    try:
        if bank.dtype!=np.dtype("int16") or bank.ndim!=2:raise ValueError("Original native BF16-bit boundary0 required")
        count=0
        for case in items:
            guard();ids=torch.tensor(case["input_ids"],device=model.lm_head.weight.device)
            actual=model.model.embed_tokens(ids).detach().cpu()
            expected=torch.from_numpy(np.array(bank[case["offset"]:case["stop"]],copy=True)).view(torch.bfloat16)
            if actual.shape!=expected.shape or not torch.equal(actual,expected):raise ValueError("Raw embedding differs from frozen native boundary0")
            count+=len(ids)
        return dict(passed=True,exact=True,cases=len(items),tokens=count,learned_prefix=False)
    finally:bank._mmap.close()


def _cache_controls(model,cases,guard):
    results=[]
    for case in cases:
        ids=torch.tensor([case["input_ids"]],device=model.lm_head.weight.device)
        result=evaluator.validate_cache(model,ids,guard=guard,capacity=CONFIG["cache_capacity"],
            atol=CONFIG["logit_atol"],rtol=CONFIG["logit_rtol"])
        results.append(dict(identity=case["identity"],**result))
    return dict(passed=all(r["passed"] for r in results),records=results)


def _selected_check(rows,reference,cases):
    return old.compare_rows(rows,reference,cases,expected_correct=sum(r["prediction"]==r["answer"] for r in reference))


def run(root,phase):
    from .jspace_full_lease import wait_ready
    root=Path(root).resolve();plan=verify(root)
    if phase not in PHASES:raise ValueError("Unknown final GPU phase")
    output=root/phase
    if output.exists():raise FileExistsError(output)
    for before in PHASES[:PHASES.index(phase)]:checked(root,before,plan)
    lease=None;cuda_started=False;started=time.monotonic();model=None
    output.mkdir()
    try:
        lease=wait_ready([0]);allocator=campaign.cap_allocator(lease,CONFIG);torch.cuda.init();cuda_started=True
        torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4);torch.manual_seed(CONFIG["seed"])
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Final reservation changed")
            if torch.cuda.max_memory_reserved()>allocator["allocator_limit_bytes"]:raise MemoryError("Final allocator cap exceeded")
        def event(**values):base.event(output/"progress.jsonl","scoring",**values)
        tasks=read(root/"tasks.json");dev=tasks["development"];bench=tasks["benchmark"]
        reference=read(plan["selected_reference"])["records"]
        tokenizer=AutoTokenizer.from_pretrained(plan["model"]["path"],local_files_only=True,trust_remote_code=False)
        result=dict(status="completed",phase=phase,fingerprint=plan["fingerprint"],confirmed=False)
        with evaluator.arithmetic_policy():
            if phase=="compact_export":
                native_controls={}
                for attention in ("eager","sdpa"):
                    model=evaluator.load_native_model(plan["model"]["path"],device="cuda:0",attention=attention,guard=guard)
                    native_controls[attention]=_cache_controls(model,bench,guard)
                    base.write(output/f"native_{attention}_cache.json",native_controls[attention])
                    if not native_controls[attention]["passed"]:raise RuntimeError("Native "+attention+" cache control failed; cannot attribute this to compression")
                    del model;model=None;gc.collect();torch.cuda.empty_cache()
                model=evaluator.build_compact_model(plan["model"]["path"],{int(k):v for k,v in plan["fits"].items()},
                    plan["selected_checkpoint"],device="cuda:0",guard=guard)
                result["embedding_control"]=embedding_control(model,tasks["all_original_items"],plan["native_boundary_zero"],guard)
                rows=evaluator.score_development(model,dev,plan["label_ids"],guard=guard)
                check=_selected_check(rows,reference,dev);base.write(output/"development.json",dict(records=rows,comparison=check))
                if not check["passed"]:raise RuntimeError("Raw-token compact model changed accepted selected-development records")
                result.update(native_cache_controls=native_controls,selected_development=check,
                    export=evaluator.export_model_bundle(output/"model",model,plan["model"]["path"],plan["fingerprint"],guard=guard),
                    tensor_bytes=_model_bytes(model))
            elif phase in ("compact_validate","compact"):
                model,tokenizer=evaluator.load_model_bundle(root/"compact_export/model",device="cuda:0",guard=guard,
                    expected_fingerprint=plan["fingerprint"])
                rows=evaluator.score_development(model,dev,plan["label_ids"],guard=guard)
                selected=_selected_check(rows,reference,dev)
                exported=read(root/"compact_export/development.json")["records"]
                exact=rows==exported
                if not selected["passed"] or not exact:raise RuntimeError("Fresh bundle changed selected-development records")
                result.update(selected_development=selected,fresh_export_reload_exact=exact,tensor_bytes=_model_bytes(model))
                if phase=="compact_validate":
                    result["cache"]=_cache_controls(model,bench,guard)
                    if not result["cache"]["passed"]:raise RuntimeError("Compact cache control failed after native controls passed")
                    base.write(output/"development.json",dict(records=rows,comparison=selected))
                else:
                    candidates=evaluator.score_model(model,tasks["confirmation"],plan["label_ids"],guard=guard,event=event)
                    base.write(output/"quality.json",dict(records=candidates))
                    result["runtime"]=evaluator.runtime_adapter(model,tokenizer,bench,seed=CONFIG["seed"],guard=guard,repeats=5)
                    del model;model=None;gc.collect();torch.cuda.empty_cache()
                    model,tokenizer=evaluator.load_model_bundle(root/"compact_export/model",device="cuda:0",guard=guard,expected_fingerprint=plan["fingerprint"])
                    reloaded=evaluator.score_model(model,tasks["confirmation"],plan["label_ids"],guard=guard,event=event)
                    base.write(output/"reload_quality.json",dict(records=reloaded))
                    result["confirmation_reload_exact"]=reloaded==candidates
                    if not result["confirmation_reload_exact"]:raise RuntimeError("Reloaded packed model changed confirmation predictions/logits")
            else:
                attention="eager" if phase=="native_eager_before" else "sdpa"
                model=evaluator.load_native_model(plan["model"]["path"],device="cuda:0",attention=attention,guard=guard)
                result["tensor_bytes"]=_model_bytes(model)
                native_rows=evaluator.score_model(model,tasks["confirmation"],plan["label_ids"],guard=guard,event=event)
                base.write(output/"quality.json",dict(records=native_rows))
                if phase=="native_sdpa_after":
                    before=read(root/"native_sdpa_before/quality.json")["records"]
                    result["baseline_quality_exact"]=native_rows==before
                    if not result["baseline_quality_exact"]:raise RuntimeError("Native SDPA before/after baseline changed")
                result["runtime"]=evaluator.runtime_adapter(model,tokenizer,bench,seed=CONFIG["seed"],guard=guard,repeats=5)
        guard();result["elapsed_seconds"]=time.monotonic()-started;base.write(output/"result.json",result)
        base.write(output/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],lease_id=lease["id"],
            allocation=lease["allocation"],allocator=allocator,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            artifacts={str(p.relative_to(output)):base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()}))
        return dict(status="completed",phase=phase)
    except BaseException as error:
        base.write(output/"failure.json",dict(status="failed",error_type=type(error).__name__,error=str(error),
            lease_id=None if lease is None else lease["id"],peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise
    finally:
        if model is not None:del model
        gc.collect()


def finalize(root):
    _cpu();root=Path(root);plan=verify(root)
    phases={p:checked(root,p,plan) for p in PHASES}
    rows={p:read(root/p/"quality.json")["records"] for p in ("native_eager_before","native_sdpa_before","compact","native_sdpa_after")}
    quality=evaluator.scoring.paired_quality(rows["native_eager_before"],rows["compact"])
    sdpa_quality=evaluator.scoring.paired_quality(rows["native_eager_before"],rows["native_sdpa_before"])
    before=phases["native_sdpa_before"]["runtime"]["timings"]["cuda_graph"]
    after=phases["native_sdpa_after"]["runtime"]["timings"]["cuda_graph"]
    candidate=phases["compact"]["runtime"]["timings"]["cuda_graph"]
    pooled=sum(row["tokens"] for part in (before,after) for row in part["samples"])/(before["total_seconds"]+after["total_seconds"])
    gates=evaluator.final_gates(quality,native_bytes=phases["native_eager_before"]["tensor_bytes"],
        compact_bytes=phases["compact"]["tensor_bytes"],sdpa_before=before,
        sdpa_after=after,compact_rate=candidate["tokens_per_second"],
        reload_exact=phases["compact"]["confirmation_reload_exact"] and phases["compact_validate"]["fresh_export_reload_exact"],
        cache_valid=phases["compact_validate"]["cache"]["passed"],
        graph_static_exact=all(phases[p]["runtime"]["graph_static_exact"] for p in ("native_eager_before","native_sdpa_before","compact","native_sdpa_after")))
    result=dict(status="completed",study_id=STUDY,fingerprint=plan["fingerprint"],confirmed=False,
        quality=quality,sdpa_quality_diagnostic=sdpa_quality,decision=gates,
        speedup_vs_eager=candidate["tokens_per_second"]/phases["native_eager_before"]["runtime"]["timings"]["cuda_graph"]["tokens_per_second"],
        phase_artifacts={p:base.sha(root/p/"manifest.json") for p in PHASES},pooled_sdpa_tokens_per_second=pooled,
        classification="all_three_full_model_gates_passed" if gates["passed"] else "full_model_targets_not_jointly_met",
        scope="One selected full36 model on512 untouched confirmation cases; no tuning or family-specific gates")
    base.write(root/"result.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","finalize",*PHASES))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--campaign",type=Path)
    parser.add_argument("--confirmation-root",type=Path,default=confirmation.DEFAULT_ROOT);args=parser.parse_args()
    result=prepare(args.root,args.campaign,args.confirmation_root) if args.phase=="prepare" else finalize(args.root) if args.phase=="finalize" else run(args.root,args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)
