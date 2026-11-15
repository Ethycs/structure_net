"""Read-only development diagnosis of BF16 native seams and suffix packing.

No training, selection, tolerance changes, confirmation data or speed claims.
The fixed target is native layer24 followed by the selected compact25..35.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from . import jspace_qwen3b_native_budget as source

base, old = source.base, source.old
SCHEMA = "jspace.3b-seam-diagnostic.v1"
MODES = ("packed_inference", "packed_torch", "unpacked_eval", "unpacked_grad")


def _forward(module, rotary, batch, hidden):
    output = module(hidden, attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"], cache_position=batch["position_ids"][0],
        position_embeddings=rotary(hidden, batch["position_ids"]), use_cache=False)
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def native_variants(block, rotary, batch):
    """All outputs share the original padded shape; only valid rows are scored."""
    hidden = batch["hidden"]
    outputs = {"batch4": _forward(block, rotary, batch, hidden),
               "singleton": torch.zeros_like(hidden), "singleton_padded": torch.zeros_like(hidden)}
    for row, length in enumerate(batch["lengths"]):
        for mode, width in (("singleton", length), ("singleton_padded", hidden.shape[1])):
            single = dict(batch, position_ids=batch["position_ids"][row:row+1, :width],
                attention_mask=batch["attention_mask"][row:row+1, :, :width, :width])
            value = _forward(block, rotary, single, hidden[row:row+1, :width])
            outputs[mode][row, :length] = value[0, :length]
    for value in outputs.values():
        for row, length in enumerate(batch["lengths"]):
            value[row, length:] = 0
    return outputs


def boundary_metrics(actual, expected, lengths):
    if actual.shape != expected.shape or actual.ndim != 3 or len(lengths) != len(actual):
        raise ValueError("Aligned boundary tensors and lengths required")
    rows = []
    for a, b, length in zip(actual, expected, lengths):
        if type(length) is not int or not 0 < length <= len(a):
            raise ValueError("Invalid valid-token length")
        a, b = a[:length].float(), b[:length].float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite boundary")
        error = a-b
        rows.append(dict(exact=bool(torch.equal(a,b)), max_abs=float(error.abs().max()),
            mismatch_fraction=float((a!=b).float().mean()),
            relative_rms=float((error.square().mean()/b.square().mean().clamp_min(1e-30)).sqrt()),
            answer_relative_rms=float((error[-1].square().mean()/b[-1].square().mean().clamp_min(1e-30)).sqrt())))
    return dict(cases=rows, exact=all(r["exact"] for r in rows), max_abs=max(r["max_abs"] for r in rows),
        case_mean_relative_rms=sum(r["relative_rms"] for r in rows)/len(rows))


def suffix_logits(stack, norm, E, rotary, batch, hidden, mode="packed_inference"):
    if mode not in MODES:
        raise ValueError("Unknown diagnostic execution mode")
    if bool(stack.packed) != mode.startswith("packed"):
        raise ValueError("Mode must match actual packed/unpacked weights")
    if any(p.requires_grad for m in (stack,norm,rotary) for p in m.parameters()) or E.requires_grad:
        raise ValueError("Diagnostic parameters must be frozen")
    enabled = stack.enabled
    try:
        if mode == "packed_torch":
            stack.enabled = False
        context = torch.enable_grad() if mode == "unpacked_grad" else torch.inference_mode()
        with context:
            value = hidden.detach().clone().requires_grad_(True) if mode == "unpacked_grad" else hidden
            output = _forward(stack, rotary, batch, value)
            selected = output[torch.arange(len(value), device=value.device)[:,None],batch["loss_positions"]]
            logits = F.linear(norm(selected), E).float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite diagnostic logits")
            return logits.detach()
    finally:
        stack.enabled = enabled


def logits_comparison(actual, expected, items, label_ids):
    if actual.shape != expected.shape or actual.ndim != 3 or actual.shape[1] != 4 or len(items)!=len(actual):
        raise ValueError("Aligned four-position full-vocabulary logits required")
    if any(c.get("split") != "development" for c in items):
        raise ValueError("Only existing development cases are allowed")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Nonfinite logits")
    rows=[];reference=[]
    for index,case in enumerate(items):
        for logits,destination in ((actual,rows),(expected,reference)):
            values=logits[index,-1,label_ids[:len(case["choices"])]]
            destination.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
                prediction=int(values.argmax()),logits=values.cpu().tolist()))
    tlog=(expected.float()/2).log_softmax(-1);slog=(actual.float()/2).log_softmax(-1)
    per_case=(tlog.exp()*(tlog-slog)).sum(-1).mean(-1)
    return dict(**old.compare_rows(rows,reference,items),records=rows,
        full_vocab_kl_t2_by_case=per_case.cpu().tolist(),full_vocab_kl_t2=float(per_case.mean()),
        full_vocab_max_abs=float((actual-expected).abs().max()))


def prepare(root, campaign):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("Preparation must hide CUDA")
    root,campaign=Path(root).resolve(),Path(campaign).resolve()
    if root.exists():raise FileExistsError(root)
    parent_plan,stage=source.verify_stage(campaign,24)
    parent_dir=source.stage_dir(campaign,25);parent_stage=source.read(parent_dir/"plan.json")
    trained=source.checked(parent_dir,"train",parent_stage);verified=source.checked(parent_dir,"verify",parent_stage)
    if not trained["quality_pass"] or not verified["passed"] or stage["parent_checkpoint_sha256"]!=trained["selected_checkpoint_sha256"]:
        raise ValueError("Authenticated accepted eleven-block parent required")
    inputs={}
    for path in (campaign/"plan.json",source.stage_dir(campaign,24)/"plan.json",source.stage_dir(campaign,24)/"capture/failure.json",
        parent_dir/"train/manifest.json",parent_dir/"train/result.json",parent_dir/"verify/manifest.json",parent_dir/"verify/result.json",
        Path(stage["parent_checkpoint"]),Path(stage["parent_records"]),Path(parent_plan["items_path"]),
        Path(parent_plan["native_cache"])/"boundary_24.npy",Path(parent_plan["native_cache"])/"boundary_25.npy",
        parent_dir/"capture/packed_teacher_logits.npy"):
        old.pin(inputs,path)
    cases=[c for c in source.read(parent_plan["items_path"])["items"] if c["split"]=="development"]
    if len(cases)!=64:raise ValueError("Original complete64 development cases required")
    sources=dict(parent_plan["source_sha256"])
    for path in (Path(__file__).resolve(),base.ROOT/"tests/structure_net/test_jspace_3b_seam_diagnostic.py"):
        sources[str(path)]=base.sha(path)
    plan=dict(schema=SCHEMA,campaign=str(campaign),campaign_fingerprint=parent_plan["fingerprint"],cut=24,parent_cut=25,
        parent_checkpoint_sha256=stage["parent_checkpoint_sha256"],parent_correct=trained["selected"]["correct"],
        teacher_logits=str(parent_dir/"capture/packed_teacher_logits.npy"),identities=[c["identity"] for c in cases],
        input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},source_sha256=sources,
        allocation_mib=11264,allocator_mib=11008,modes=list(MODES),native_variants=["singleton","singleton_padded","batch4"],
        scope="Read-only existing development diagnostic; no training, selection, new gates or heldout",confirmed=False)
    plan["fingerprint"]=base.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        destination=root/"producer"/Path(path).relative_to(base.ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if base.sha(destination)!=digest:raise ValueError("Producer changed during preparation")
    base.write(root/"plan.json",plan);return plan


def verify(root):
    root=Path(root);plan=source.read(root/"plan.json");source.check_plan(plan)
    if plan["schema"]!=SCHEMA:raise ValueError("Unsupported diagnostic schema")
    for path,digest in plan["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError("Diagnostic producer changed")
    for path,identity in plan["input_identity"].items():
        if old.file_identity(path)!=identity:raise ValueError("Diagnostic input changed")
    return plan


def run(root):
    from .jspace_full_lease import wait_ready
    root=Path(root);plan=verify(root)
    if (root/"run").exists():raise FileExistsError(root/"run")
    source_plan,stage=source.verify_stage(Path(plan["campaign"]),24)
    lease=wait_ready([0]);allocator=source.cap_allocator(lease,plan)
    torch.cuda.init();torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    output=root/"run";output.mkdir();started=time.monotonic();datasets=[]
    def guard():
        current=wait_ready([0])
        if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Reservation changed")
        if torch.cuda.max_memory_reserved()>allocator["allocator_limit_bytes"]:raise MemoryError("Reservation cap exceeded")
    try:
        unpacked,norm,E,rotary,inheritance=source.make_stack(source_plan,stage,"cpu",parent=True)
        unpacked.checkpoint_blocks=False
        packed=source.model.packed_clone(unpacked,device="cuda:0")
        unpacked.to("cuda:0");norm.to("cuda:0");E=E.to("cuda:0");rotary.to("cuda:0")
        native=source.native.load_native_block(source_plan["model"]["path"],24,device="cuda:0").eval().requires_grad_(False)
        datasets=[source.data_for(source_plan,cut,plan["teacher_logits"]) for cut in (24,25)]
        def fingerprint():
            values=[("E",E)]
            for name,module in (("unpacked",unpacked),("packed",packed),("norm",norm),("rotary",rotary),("native",native)):
                values.extend((name+"."+n,v) for n,v in [*module.named_parameters(),*module.named_buffers()])
            return old.fingerprint(values)
        before=fingerprint();summaries={};boundary_rows={};reference_rows=[]
        def add(name,logits,reference,batch,items):
            value=logits_comparison(logits,reference,items,source_plan["label_ids"])
            loss,kl,ce=source.core.output_objective(logits,batch["teacher"],batch["targets"],source.CONFIG)
            summaries.setdefault(name,[]).append(dict(**value,objective=dict(loss=float(loss),kl=float(kl),ce=float(ce))))
        for start in range(0,len(datasets[0].development),4):
            guard();indices=datasets[0].development[start:start+4]
            incoming,cached=[data.batch(indices,device=E.device) for data in datasets]
            items=[datasets[0].items[i] for i in indices]
            reference=suffix_logits(packed,norm,E,rotary,cached,cached["hidden"])
            add("packed_inference",reference,reference,cached,items)
            reference_rows.extend(summaries["packed_inference"][-1]["records"])
            for mode in MODES[1:]:
                stack=packed if mode.startswith("packed") else unpacked
                logits=suffix_logits(stack,norm,E,rotary,cached,cached["hidden"],mode)
                add(mode,logits,reference,cached,items)
            for name,hidden in native_variants(native,rotary,incoming).items():
                differences=boundary_metrics(hidden,cached["hidden"],incoming["lengths"])
                boundary_rows.setdefault(name,[]).extend(differences["cases"])
                logits=suffix_logits(packed,norm,E,rotary,cached,hidden)
                add("native_"+name,logits,reference,cached,items)
            base.event(root/"progress.jsonl","development_batch",completed=start+len(indices),total=64)
            del incoming,cached,reference,logits,hidden
        cases=[datasets[0].items[i] for i in datasets[0].development]
        if [c["identity"] for c in cases]!=plan["identities"]:raise ValueError("Development identity/order changed")
        control=old.compare_rows(reference_rows,source.records(stage["parent_records"]),cases,expected_correct=plan["parent_correct"])
        results={}
        for name,batches in summaries.items():
            rows=[row for b in batches for row in b["records"]]
            kls=[value for b in batches for value in b["full_vocab_kl_t2_by_case"]]
            results[name]=dict(**old.compare_rows(rows,reference_rows,cases),records=rows,
                full_vocab_kl_t2=sum(kls)/len(kls),full_vocab_kl_t2_by_case=kls,
                full_vocab_max_abs=max(b["full_vocab_max_abs"] for b in batches),
                objective={key:sum(b["objective"][key]*b["count"] for b in batches)/64 for key in ("loss","kl","ce")})
        after=fingerprint();guard()
        result=dict(status="completed",fingerprint=plan["fingerprint"],control=control,modes=results,
            boundary={name:dict(cases=rows,exact=all(r["exact"] for r in rows),max_abs=max(r["max_abs"] for r in rows),
                case_mean_relative_rms=sum(r["relative_rms"] for r in rows)/64) for name,rows in boundary_rows.items()},
            unchanged_state=before==after,state_before=before,state_after=after,inheritance=inheritance,
            passed=control["passed"] and before==after,elapsed_seconds=time.monotonic()-started,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope=plan["scope"],confirmed=False)
        base.write(output/"result.json",result)
        if not result["passed"]:raise RuntimeError("Cached parent or immutable-state control failed")
        base.write(output/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],lease_id=lease["id"],
            allocation=lease["allocation"],allocator=allocator,artifacts={"result.json":base.sha(output/"result.json")}))
        return result
    except BaseException as error:
        base.write(output/"failure.json",dict(error=repr(error),error_type=type(error).__name__,lease_id=lease["id"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        raise
    finally:
        for data in datasets:data.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--campaign",type=Path)
    args=parser.parse_args();value=prepare(args.root,args.campaign) if args.phase=="prepare" else run(args.root)
    print(json.dumps({"status":value.get("status","prepared"),"fingerprint":value["fingerprint"]}),flush=True)
