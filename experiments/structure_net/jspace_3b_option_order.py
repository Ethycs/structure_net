"""Paired option-order diagnostic on the already exposed 32 WinoGrande cases."""
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
from . import jspace_3b_sanity_evaluate as sanity

base,old,read=sanity.base,sanity.old,sanity.read
SCHEMA="jspace.3b-option-order.v1"
PROTOCOL=base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-option-order-diagnostic.md"
POLICY=dict(checkpoint_sha256=sanity.CHECKPOINT_SHA,cut=22,original_cases=32,variants=["original","swapped"],
    original_replay="exact_records_and_logits_before_each_arm_scores_swapped",dtype="bfloat16",
    attention="eager",batch_size=1,padding=False,confirmed=False)


def make_pairs(cases,tokenizer):
    originals=[];swapped=[]
    for case in cases:
        if case["family"]!="winogrande":continue
        if len(case["choices"])!=2 or case["choices"][0]==case["choices"][1]:raise ValueError("Two distinct options required")
        suffix="\n\n"+"\n".join(f"{chr(65+i)}. {s}" for i,s in enumerate(case["choices"]))
        if not case["prompt"].endswith(suffix):raise ValueError("Original prompt does not match its options")
        question=case["prompt"][:-len(suffix)]
        for variant,destination,choices,answer in (("original",originals,case["choices"],case["answer"]),
                ("swapped",swapped,list(reversed(case["choices"])),1-case["answer"])):
            prompt=question+"\n\n"+"\n".join(f"{chr(65+i)}. {s}" for i,s in enumerate(choices))
            ids=list(tokenizer.apply_chat_template([dict(role="system",content=sanity.data_source.SYSTEM),
                dict(role="user",content=prompt)],tokenize=True,add_generation_prompt=True))
            identity=sanity.data_source.identity(question,choices)
            if variant=="original" and (ids!=case["input_ids"] or identity!=case["identity"]):
                raise ValueError("Original tokenization or normalized identity changed")
            if not 3<=len(ids)<=384:raise ValueError("Variant exceeds the frozen prompt range")
            destination.append(dict(case,identity=identity,prompt=prompt,input_ids=ids,choices=choices,answer=answer,
                variant=variant,original_identity=case["identity"]))
    items=originals+swapped
    if not originals or len({c["identity"] for c in items})!=len(items):raise ValueError("Distinct original/swapped identities required")
    offset=0
    for case in items:case.update(offset=offset,stop=offset+len(case["input_ids"]));offset=case["stop"]
    return items


def original_replay(records,reference,cases):
    sanity.validate_records(records,cases)
    if records!=reference:raise ValueError("Original32 must reproduce the saved sanity records/logits exactly")
    return dict(passed=True,exact_records=True,exact_logits=True,count=len(cases))


def score_variants(items,reference,score,progress=lambda *_:None):
    n=len(reference)
    if not n or len(items)!=2*n:raise ValueError("Two complete variants required")
    records=[]
    for i,case in enumerate(items):
        if i==n:control=original_replay(records,reference,items[:n])
        records.append(score(case));progress(i,case)
    sanity.validate_records(records,items)
    return records,control


def order_metrics(records,items):
    sanity.validate_records(records,items);n=len(items)//2
    if len(items)!=2*n or not n:raise ValueError("Complete paired variants required")
    original,swapped=records[:n],records[n:]
    for a,b in zip(items[:n],items[n:]):
        if (a["variant"]!="original" or b["variant"]!="swapped" or b["original_identity"]!=a["identity"]
                or b["choices"]!=list(reversed(a["choices"])) or b["answer"]!=1-a["answer"]):
            raise ValueError("Semantic option correspondence changed")
    both=sum(a["prediction"]==a["answer"] and b["prediction"]==b["answer"] for a,b in zip(original,swapped))
    either=sum(a["prediction"]==a["answer"] or b["prediction"]==b["answer"] for a,b in zip(original,swapped))
    a=sum(r["prediction"]==r["answer"] for r in original);b=sum(r["prediction"]==r["answer"] for r in swapped)
    consistent=sum(x["prediction"]==1-y["prediction"] for x,y in zip(original,swapped))
    return dict(count=n,original_correct=a,swapped_correct=b,original_accuracy=a/n,swapped_accuracy=b/n,
        mean_variant_accuracy=(a+b)/(2*n),both_correct=both,either_correct=either,
        remapped_choice_consistent=consistent,remapped_choice_consistency=consistent/n,
        original_A_predictions=sum(r["prediction"]==0 for r in original),swapped_A_predictions=sum(r["prediction"]==0 for r in swapped),
        original_text_option0_chosen_before=sum(r["prediction"]==0 for r in original),
        original_text_option0_chosen_after=sum(r["prediction"]==1 for r in swapped))


def prepare(root,source_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Preparation requires CUDA hidden")
    root,source_root=Path(root).resolve(),Path(source_root).resolve()
    if root.exists():raise FileExistsError(root)
    parent=sanity.verify(source_root)
    previous={phase:sanity.checked(source_root,phase,parent) for phase in ("native","compact")}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(parent["model"]["path"],local_files_only=True,trust_remote_code=False)
    source_cases=read(source_root/"items.json")["items"]
    items=make_pairs(source_cases,tokenizer)
    if len(items)!=64:raise ValueError("Exactly32 previously scored WinoGrande cases required")
    references={phase:[r for r in previous[phase]["records"] if r["family"]=="winogrande"] for phase in previous}
    for rows in references.values():sanity.validate_records(rows,items[:32])
    inputs=dict(parent["input_sha256"])
    for path in (source_root/"plan.json",source_root/"items.json",*(source_root/p/n for p in previous for n in ("manifest.json","result.json"))):old.pin(inputs,path)
    sources=dict(parent["source_sha256"])
    for path in sanity.source.inherited.local_import_closure([Path(__file__),PROTOCOL,
            base.ROOT/"tests/structure_net/test_jspace_3b_option_order.py"]):
        digest=base.sha(path)
        if str(path) in sources and sources[str(path)]!=digest:raise ValueError("Frozen numerical source changed")
        sources[str(path)]=digest
    plan=dict(schema=SCHEMA,policy=POLICY,source_root=str(source_root),source_fingerprint=parent["fingerprint"],
        model=parent["model"],fits=parent["fits"],checkpoint=parent["checkpoint"],checkpoint_sha256=parent["checkpoint_sha256"],
        label_ids=parent["label_ids"],native_boundary=parent["native_boundary"],development_control=parent["development_control"],
        items_digest=base.digest(items),references_digest=base.digest(references),total_tokens=items[-1]["stop"],hidden_size=2048,
        allocation_mib=11264,allocator_mib=11008,input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        source_sha256=sources,confirmed=False)
    plan["fingerprint"]=base.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        dst=root/"producer"/Path(path).relative_to(base.ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
        if base.sha(dst)!=digest:raise ValueError("Producer archive changed")
    base.write(root/"items.json",dict(items=items));base.write(root/"references.json",references);base.write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],original_cases=32,variant_cases=64)


def verify(root):
    root=Path(root);plan=read(root/"plan.json");sanity.source.check_plan(plan)
    if (plan.get("schema")!=SCHEMA or plan.get("policy")!=POLICY or plan.get("checkpoint_sha256")!=sanity.CHECKPOINT_SHA
            or plan.get("allocation_mib")!=11264 or plan.get("allocator_mib")!=11008):raise ValueError("Diagnostic protocol changed")
    for path,identity in plan["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Pinned source input changed")
    for path,digest in plan["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:raise ValueError("Frozen producer changed")
    if (base.digest(read(root/"items.json")["items"])!=plan["items_digest"] or
            base.digest(read(root/"references.json"))!=plan["references_digest"]):raise ValueError("Diagnostic cases or references changed")
    return plan


def checked(root,phase,plan=None):
    root=Path(root);plan=verify(root) if plan is None else plan
    result=sanity.checked(root,phase,plan)
    cases=read(root/"items.json")["items"]
    control=original_replay(result["records"][:32],read(root/"references.json")[phase],cases[:32])
    if result.get("original_replay")!=control:raise ValueError("Original replay proof changed")
    return result


def run(root,phase):
    root=Path(root).resolve();plan=verify(root)
    if phase not in ("native","compact"):raise ValueError("Unknown diagnostic phase")
    if phase=="compact":checked(root,"native",plan)
    output=root/phase
    if output.exists():raise FileExistsError(output)
    lease=None;started=time.monotonic();cuda=False
    try:
        lease=sanity.wait_ready([0]);settings=sanity.source.inherited.cap_allocator(lease,plan)
        torch.cuda.init();cuda=True;torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4)
        def guard():
            current=sanity.wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Reservation changed")
            if torch.cuda.max_memory_reserved()>settings["allocator_limit_bytes"]:raise MemoryError("Allocator cap exceeded")
        output.mkdir();items=read(root/"items.json")["items"];reference=read(root/"references.json")[phase];records=[]
        with sanity.execution.arithmetic_policy():
            if phase=="native":
                model=sanity.execution.load_native_model(plan["model"]["path"],device="cuda:0",attention="eager",guard=guard)
                c=plan["development_control"];_,actual=sanity.native_case(model,c,plan["label_ids"],22,guard)
                prior=np.load(plan["native_boundary"],mmap_mode="r")
                if prior.dtype!=np.int16:raise ValueError("Original boundary must be BF16 bits")
                expected=torch.from_numpy(np.array(prior[c["offset"]:c["stop"]],copy=True)).view(torch.bfloat16)[None]
                detail=dict(boundary_control=sanity.boundary_control(actual,expected));del prior,expected,actual
                bank=np.lib.format.open_memmap(output/"boundary_22.npy",mode="w+",dtype=np.int16,shape=(plan["total_tokens"],2048))
            else:
                stack,norm,E,rotary,inheritance=sanity.builder.build_stack(plan["model"]["path"],{int(k):v for k,v in plan["fits"].items()},
                    22,plan["checkpoint"],device="cpu",trainable=False,checkpoint_blocks=False)
                packed=sanity.builder.packed_clone(stack,device="cpu");del stack
                packed,norm,E,rotary=packed.to("cuda:0"),norm.to("cuda:0"),E.to("cuda:0"),rotary.to("cuda:0")
                bank=np.load(root/"native/boundary_22.npy",mmap_mode="r")
                if bank.dtype!=np.int16 or list(bank.shape)!=[plan["total_tokens"],2048]:raise ValueError("Diagnostic boundary bank changed")
                detail=dict(strict_state_load=True,inheritance=inheritance,checkpoint_sha256=plan["checkpoint_sha256"],boundary_sha256=base.sha(root/"native/boundary_22.npy"))
            def score(case):
                if phase=="native":
                    record,value=sanity.native_case(model,case,plan["label_ids"],22,guard)
                    bank[case["offset"]:case["stop"]]=value[0].contiguous().view(torch.int16).numpy()
                else:
                    value=torch.from_numpy(np.array(bank[case["offset"]:case["stop"]],copy=True)).view(torch.bfloat16)[None]
                    record=sanity.compact_case(packed,norm,E,rotary,value,case,plan["label_ids"],guard)
                return record
            records,control=score_variants(items,reference,score,lambda i,c:base.event(root/"progress.jsonl","score",
                phase=phase,completed=i+1,total=64,variant=c["variant"]))
            if phase=="native":bank.flush()
            del bank
        sanity.validate_records(records,items);guard()
        result=dict(status="completed",phase=phase,fingerprint=plan["fingerprint"],records=records,count=64,
            correct=sum(r["prediction"]==r["answer"] for r in records),original_replay=control,**detail)
        base.write(output/"result.json",result)
        base.write(output/"manifest.json",dict(status="completed",phase=phase,fingerprint=plan["fingerprint"],lease_id=lease["id"],
            allocation=lease["allocation"],allocator=settings,peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started,
            artifacts={p.name:base.sha(p) for p in output.iterdir() if p.is_file()}))
        return dict(status="completed",phase=phase)
    except Exception as error:
        output.mkdir(parents=True,exist_ok=True);base.write(output/"failure.json",dict(status="failed",phase=phase,error=repr(error),
            lease_id=None if lease is None else lease["id"],cuda_started=cuda));raise
    finally:gc.collect()


def finalize(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Finalization requires CUDA hidden")
    root=Path(root);plan=verify(root);items=read(root/"items.json")["items"]
    results={phase:checked(root,phase,plan) for phase in ("native","compact")}
    value=dict(status="completed",fingerprint=plan["fingerprint"],checkpoint_sha256=plan["checkpoint_sha256"],
        models={p:order_metrics(r["records"],items) for p,r in results.items()},
        paired_by_variant={v:sanity.paired_summary(results["native"]["records"][start:start+32],
            results["compact"]["records"][start:start+32],items[start:start+32])["quality"] for v,start in (("original",0),("swapped",32))},
        confirmed=False,full36=False,compressed_layers=14,speed_measured=False,trained=False,
        scope="Option-order sensitivity on32 already exposed sanity cases; not proof of cause, restored ability, or independent generalization.",
        phase_manifest_sha256={p:base.sha(root/p/"manifest.json") for p in results})
    base.write(root/"result.json",value);base.write(root/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
        artifacts={"result.json":base.sha(root/"result.json")},confirmed=False));return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","native","compact","finalize"))
    parser.add_argument("--root",type=Path,default=base.ROOT/"data/experiments/jspace_3b_option_order/20261115_v1")
    parser.add_argument("--source-root",type=Path,default=base.ROOT/"data/experiments/jspace_3b_sanity_evaluate/20261115_v1")
    a=parser.parse_args();result=prepare(a.root,a.source_root) if a.phase=="prepare" else finalize(a.root) if a.phase=="finalize" else run(a.root,a.phase)
    print(json.dumps(result,allow_nan=False),flush=True)
