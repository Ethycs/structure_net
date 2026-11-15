"""One authenticated seven-block expansion of the recovered six-block suffix.

Only native prefix0..28 is cached. All seven trainable blocks execute on every
minibatch; no hidden targets or later historical recovery weights are imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import jspace_allblocks_v2 as v2
from . import jspace_qwen_parallel_model as builder
from . import jspace_qwen_parallel_train as core

base=v2.base
SCHEMA="jspace.incremental-suffix.v1"
STUDY="JSPACE-QWEN-INCREMENTAL-SUFFIX-01"
PROTOCOL=base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen-incremental-suffix-recovery.md"
PARENT_SHA="81026565492400de386d56f5765e6a4e8335e609af96dc96197e67ed0a4ca675"
ATOL,RTOL=.03125,1/128
CONFIG=core.RecoveryConfig(seed=20861,max_cpu_state_bytes=32*1024**3)
SCOPE="Seven-block cached-frozen-prefix development experiment; no further expansion, confirmation or speed claim"
read=lambda p:json.loads(Path(p).read_text())


def save(path,value):
    with Path(path).open("xb") as stream:torch.save(value,stream)


def file_identity(path):
    stat=Path(path).stat()
    return [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]


def state_equal(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k].detach().cpu(),b[k].detach().cpu())
        if isinstance(a[k],torch.Tensor) and isinstance(b[k],torch.Tensor) else a[k]==b[k] for k in a)


def fingerprint(named):
    digest=hashlib.sha256();size=0
    for name,value in named:
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)]).encode());flat=value.detach().reshape(-1)
        step=max(1,32*1024**2//value.element_size())
        for start in range(0,flat.numel(),step):digest.update(flat[start:start+step].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        size+=value.numel()*value.element_size()
    return dict(sha256=digest.hexdigest(),tensor_bytes=size)


def fixed_state(stack,readout,rotary):
    return fingerprint([("stack."+n,v) for n,v in stack.named_buffers()]+[("readout",readout)]+
        [("rotary."+n,v) for n,v in rotary.named_buffers()])


def master_fingerprint(module,optimizer):
    return fingerprint([(name,optimizer.state[p]["master"]) for name,p in module.named_parameters()])


def preserve_inheritance(stack,norm,previous):
    """Restore saved interior bridges; only the new entrance bridge is new."""
    old=list(previous["stack"]["_extra_state"]["layer_indices"])
    if list(stack.layer_indices)!=[old[0]-1,*old]:raise ValueError("Exactly one preceding block may be added")
    norm.load_state_dict(previous["norm"],strict=True)
    for index,bridge in enumerate(stack.bridges[1:]):
        prefix=f"bridges.{index}.";bridge.load_state_dict({k[len(prefix):]:v for k,v in previous["stack"].items() if k.startswith(prefix)},strict=True)
    for index,block in enumerate(stack.blocks[1:]):
        prefix=f"blocks.{index}.";wanted={k[len(prefix):]:v for k,v in previous["stack"].items() if k.startswith(prefix)}
        if not state_equal(block.state_dict(),wanted):raise ValueError("Inherited block weights or geometry changed")
        for kind in ("feature_i","feature_j"):
            if not torch.equal(getattr(stack,f"{kind}_{index+1}").cpu(),previous["stack"][f"{kind}_{index}"]):
                raise ValueError("Inherited polynomial feature definitions changed")
    if not state_equal(norm.state_dict(),previous["norm"]):raise ValueError("Inherited norm changed")
    return dict(passed=True,unchanged_layers=old,interior_bridges_preserved=len(stack.bridges)-1,
        new_layer=old[0]-1,new_bridge=[old[0]-1,old[0]],norm_exact=True)


def rows_from_cache(dataset,label_ids,which="original"):
    bank=dataset.original if which=="original" else dataset.accepted;rows=[]
    for index in dataset.development:
        case=dataset.items[index];source=dataset.original_indices[index] if which=="original" else index
        logits=dataset._bits(bank[source,-1],"cpu").float()[label_ids[:len(case["choices"])]]
        rows.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
            prediction=int(logits.argmax()),logits=logits.tolist()))
    return rows


def compare_rows(rows,reference,cases,*,expected_correct=None):
    if len(rows)!=len(cases) or len(reference)!=len(cases):raise ValueError("Complete aligned development records required")
    exact=True;logits_pass=True;maximum=0.
    for actual,wanted,case in zip(rows,reference,cases):
        for value in (actual,wanted):
            if any(value[k]!=case[k] for k in ("identity","family","answer")) or len(value["logits"])!=len(case["choices"]):
                raise ValueError("Changed development identity/choice alignment")
            if not all(math.isfinite(z) for z in value["logits"]) or value["prediction"]!=max(range(len(value["logits"])),key=value["logits"].__getitem__):
                raise ValueError("Nonfinite logits or inconsistent prediction")
        exact &= actual["prediction"]==wanted["prediction"]
        for x,y in zip(actual["logits"],wanted["logits"]):
            maximum=max(maximum,abs(x-y));logits_pass &= abs(x-y)<=ATOL+RTOL*abs(y)
    correct=sum(r["prediction"]==r["answer"] for r in rows)
    return dict(passed=exact and logits_pass and (expected_correct is None or correct==expected_correct),
        predictions_exact=exact,logits_pass=logits_pass,max_abs_logit_delta=maximum,correct=correct,count=len(rows),
        accuracy=correct/len(rows),atol=ATOL,rtol=RTOL)


def gates(correct,native=48,parent=51,count=64):
    native_min=math.ceil(native-.05*count);parent_min=math.ceil(parent-.05*count)
    return dict(passed=correct>=max(native_min,parent_min),minimum_correct=max(native_min,parent_min),
        native=dict(correct=native,minimum_correct=native_min,passed=correct>=native_min),
        parent=dict(correct=parent,minimum_correct=parent_min,passed=correct>=parent_min))


def selection_key(metrics):return metrics["correct"],-metrics["ce"]


def dataset(plan,cut=29):
    items=read(plan["items_path"])["items"]
    return core.BoundaryDataset(items,plan[f"boundary_{cut}"],plan["accepted_logits"],
        original_logits_path=plan["original_logits"],original_indices=plan["original_indices"])


def build(plan,device):
    previous=builder.load_recovered_checkpoint(plan["parent_root"],0)
    stack=builder.build_stack(plan["parent_root"],29,True,0,device=device,checkpoint_blocks=True)
    readout,norm,rotary=builder.load_readout_norm(plan["model"]["path"],device=device)
    inheritance=preserve_inheritance(stack,norm,previous)
    return stack,norm,readout,rotary,inheritance


class NativeTail(nn.Module):
    def __init__(self,blocks):super().__init__();self.blocks=nn.ModuleList(blocks)
    def forward(self,hidden,**kwargs):
        for block in self.blocks:
            value=block(hidden,**kwargs);hidden=value[0] if isinstance(value,tuple) else value
        return (hidden,)


@torch.inference_mode()
def score(stack,norm,readout,rotary,data,label_ids,guard,*,batch_size=4):
    if type(batch_size) is not int or batch_size<1:raise ValueError("Positive scoring batch size required")
    rows=[];device=readout.device
    for start in range(0,len(data.development),batch_size):
        guard();batch=data.batch(data.development[start:start+batch_size],device=device)
        h=batch["hidden"];rope=rotary(h,batch["position_ids"])
        h=stack(h,attention_mask=batch["attention_mask"],position_ids=batch["position_ids"],
            cache_position=batch["position_ids"][0],position_embeddings=rope,use_cache=False)[0]
        h=h[torch.arange(len(batch["indices"]),device=device)[:,None],batch["loss_positions"]]
        logits=F.linear(norm(h),readout).float()
        for j,index in enumerate(batch["indices"]):
            c=data.items[index];values=logits[j,-1,label_ids[:len(c["choices"])]]
            rows.append(dict(identity=c["identity"],family=c["family"],answer=c["answer"],prediction=int(values.argmax()),logits=values.cpu().tolist()))
    return rows


def controls(plan,device,guard):
    data=dataset(plan);data30=dataset(plan,30);cases=[data.items[i] for i in data.development]
    readout,norm,rotary=builder.load_readout_norm(plan["model"]["path"],device=device)
    try:
        native=NativeTail([builder.load_native_block(plan["model"]["path"],i,device=device) for i in range(29,36)]).eval().requires_grad_(False)
        # The original native logit bank was captured one unpadded case at a time.
        native_rows=score(native,norm,readout,rotary,data,plan["label_ids"],guard,batch_size=1)
        native_check=compare_rows(native_rows,rows_from_cache(data,plan["label_ids"]),cases,expected_correct=48)
        del native;gc.collect();guard()
        previous=builder.load_recovered_checkpoint(plan["parent_root"],0)
        parent=builder.build_stack(plan["parent_root"],30,True,0,device="cpu")
        parent.load_state_dict(previous["stack"],strict=True)
        parent.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).to(device).eval().requires_grad_(False)
        norm.load_state_dict(previous["norm"],strict=True)
        parent_rows=score(parent,norm,readout,rotary,data30,plan["label_ids"],guard)
        parent_reference=read(plan["parent_records"])
        parent_check=compare_rows(parent_rows,parent_reference,cases,expected_correct=51)
        seam=NativeTail([builder.load_native_block(plan["model"]["path"],29,device=device),parent]).eval().requires_grad_(False)
        seam_rows=score(seam,norm,readout,rotary,data,plan["label_ids"],guard)
        seam_check=compare_rows(seam_rows,parent_reference,cases,expected_correct=51)
        cache=compare_rows(rows_from_cache(data,plan["label_ids"],"accepted"),parent_reference,cases,expected_correct=51)
        return dict(passed=all(v["passed"] for v in (native_check,parent_check,seam_check)) and cache["predictions_exact"],
            native=dict(**native_check,records=native_rows),parent=dict(**parent_check,records=parent_rows),
            seam=dict(**seam_check,records=seam_rows),teacher_cache_diagnostic=dict(**cache,logit_gate_applied=False))
    finally:data.close();data30.close()


class Trainer(core.SuffixTrainer):
    @torch.no_grad()
    def packed_development(self):
        metrics,rows=super().packed_development();metrics["selection_metric"]="highest_correct_then_lowest_full_vocab_ce_earlier_tie"
        metrics["by_family"]={f:dict(correct=sum(r["prediction"]==r["answer"] for r in rows if r["family"]==f),
            count=sum(r["family"]==f for r in rows)) for f in sorted({r["family"] for r in rows})}
        return metrics,rows

    def fit(self,path,*,event=lambda *_a,**_kw:None,initial_reference=None):
        path=Path(path);path.mkdir(exist_ok=False)
        if self.updates or self.finished:raise ValueError("Scientific fit must start with untouched initializer")
        before=fixed_state(self.stack,self.readout,self.rotary);history=[];best=None;best_epoch=None
        total=self.config.epochs*math.ceil(len(self.dataset.training)/(self.config.microbatch*self.config.accumulation))
        effective=self.config.microbatch*self.config.accumulation
        for epoch in range(self.config.epochs+1):
            training=[]
            if epoch:
                order=list(self.dataset.training);random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0,len(order),effective):
                    lr=self.config.learning_rate*.5*(1+math.cos(math.pi*self.updates/total))
                    update=self.train_window(order[start:start+effective],learning_rate=lr);training.append(update)
                    event("update",epoch=epoch,**update)
            metrics,records=self.packed_development();report=path/f"epoch_{epoch:02d}_development.json"
            if epoch==0 and initial_reference is not None:
                check=compare_rows(records,initial_reference["initial_records"],
                    [self.dataset.items[i] for i in self.dataset.development],expected_correct=initial_reference["initial"]["correct"])
                base.write(path/"initializer_replay.json",check)
                if not check["passed"]:raise RuntimeError("Scientific initializer differs from discarded pilot before training")
            base.write(report,dict(metrics=metrics,records=records))
            changed=best is None or selection_key(metrics)>selection_key(best)
            row=dict(epoch=epoch,metrics=metrics,training=training,development_sha256=base.sha(report),checkpoint_updated=changed)
            if changed:
                pending=path/"selected.pt.pending";save(pending,self.snapshot());os.replace(pending,path/"selected.pt")
                best,best_epoch=metrics,epoch;row["checkpoint_sha256"]=base.sha(path/"selected.pt")
            history.append(row);event("development",epoch=epoch,selected_epoch=best_epoch,**metrics)
        if self.updates!=total:raise RuntimeError("Wrong update budget")
        state=builder._load(path/"selected.pt");self.stack.load_state_dict(state["stack"],strict=True);self.norm.load_state_dict(state["norm"],strict=True)
        final,rows=self.packed_development();reference=read(path/f"epoch_{best_epoch:02d}_development.json")
        if final!=best or rows!=reference["records"]:raise RuntimeError("Selected packed reload changed results")
        if fixed_state(self.stack,self.readout,self.rotary)!=before:raise RuntimeError("Recovery changed fixed geometry or readout")
        self.finished=True
        result=dict(status="completed",systems_only=False,confirmed=False,completed_epochs=self.config.epochs,updates=self.updates,
            config=asdict(self.config),initial=history[0]["metrics"],selected=best,selected_epoch=best_epoch,
            initializer_selected=best_epoch==0,history=history,selected_checkpoint_sha256=base.sha(path/"selected.pt"),
            quality=gates(best["correct"]),quality_pass=gates(best["correct"])["passed"],strict_reload_exact=True,
            fixed_state_unchanged=True,fixed_state=before,optimizer_memory=self.optimizer.memory_stats(),scope=SCOPE)
        base.write(path/"result.json",result);return result


def pin(inputs,path,expected=None):
    path=Path(path).resolve();value=base.sha(path)
    if expected is not None and value!=expected:raise ValueError("Changed consumed artifact: "+str(path))
    if str(path) in inputs and inputs[str(path)]!=value:raise ValueError("Conflicting input provenance")
    inputs[str(path)]=value


def manifest_inputs(parent,stage,plan,names,inputs):
    path=Path(parent)/stage;manifest=read(path/"manifest.json")
    if manifest.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"]:raise ValueError("Matching completed parent stage required")
    pin(inputs,path/"manifest.json")
    if "reused_from" in manifest:
        reuse=manifest["reused_from"];pin(inputs,Path(reuse["root"])/reuse["manifest"],reuse["manifest_sha256"])
    for name in names:pin(inputs,Path(parent)/name,manifest["artifacts"][name])
    return manifest


def validate_items(items,original):
    wanted={(s,f):n for s,n in (("train",128),("development",16)) for f in ("arc","boolq","hellaswag","winogrande")}
    if Counter((c["split"],c["family"]) for c in items)!=wanted or len({c["identity"] for c in items})!=576:
        raise ValueError("Original512 train/64 balanced development only")
    by_id={c["identity"]:(i,c) for i,c in enumerate(original)};indices=[];offset=0
    for c in items:
        index,old=by_id[c["identity"]]
        if any(c[k]!=old[k] for k in ("input_ids","loss_positions","targets","split","family","answer","choices")):
            raise ValueError("Original task identity/tokens/targets changed")
        if c["offset"]!=offset or c["stop"]!=offset+len(c["input_ids"]):raise ValueError("Boundary offsets must be contiguous")
        offset=c["stop"];indices.append(index)
    return indices


def prepare(root,parent_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU preparation must hide CUDA")
    root,parent=Path(root).resolve(),Path(parent_root).resolve()
    if root.exists():raise FileExistsError(root)
    with v2.protocol_context():original=base.verify(parent,model=True)
    inputs={};pin(inputs,parent/"plan.json");pin(inputs,parent/"tasks.json")
    # Already hashed by verify(model=True); retain their identities for every later stage.
    inputs.update({str(Path(original["model"]["path"])/name):digest for name,digest in original["model"]["files"].items()})
    g0,g1="groups/group_00","groups/group_01"
    recovered=manifest_inputs(parent,g0+"/recovery",original,[g0+"/recovery/"+n for n in
        ("training.json","recovered_unpacked.pt","selected_packed_development.json")],inputs)
    if recovered["selected"]["correct"]!=51 or recovered["selected_epoch"]!=2:raise ValueError("Exact accepted six-block result required")
    pin(inputs,parent/g0/"recovered_unpacked.pt",PARENT_SHA)
    if inputs[str(parent/g0/"recovery/recovered_unpacked.pt")]!=PARENT_SHA:raise ValueError("Wrong parent checkpoint")
    for group,cut in ((g0,30),(g1,29)):
        names=[group+"/capture/"+n for n in ("items.json","metadata.json","accepted_teacher_logits.npy",f"block_{cut:02d}/boundary.npy")]
        manifest_inputs(parent,group+"/capture",original,names,inputs)
    metadata=read(parent/g1/"capture/metadata.json")
    if metadata["teacher_fingerprint"]!=base.digest(PARENT_SHA):raise ValueError("New block was not calibrated against exact six-block teacher")
    for layer in range(29,36):
        group=g1 if layer==29 else g0;block=group+f"/block_{layer:02d}"
        manifest_inputs(parent,group+"/basis",original,[block+"/"+n for n in ("fit.pt","fit.json","interface.json")],inputs)
        manifest_inputs(parent,block+"/fit_result",original,[block+"/relation.pt",block+"/relations.json"],inputs)
        if layer==29 and read(parent/block/"interface.json")["teacher_fingerprint"]!=metadata["teacher_fingerprint"]:
            raise ValueError("Layer29 fit interface teacher differs")
    oldcache=Path(original["original_cache"])
    for name in ("items.json","teacher_logits.npy"):
        pin(inputs,oldcache/name,original["input_sha256"][str(oldcache/name)])
    items=read(parent/g1/"capture/items.json")["items"]
    if items!=read(parent/g0/"capture/items.json")["items"]:raise ValueError("Six/seven cache layouts differ")
    indices=validate_items(items,read(oldcache/"items.json")["items"])
    sources=dict(original["source_sha256"])
    for path in (Path(__file__).resolve(),PROTOCOL,base.ROOT/"tests/structure_net/test_jspace_incremental_suffix.py"):
        if str(path) in sources and sources[str(path)]!=base.sha(path):raise ValueError("Conflicting source closure")
        sources[str(path)]=base.sha(path)
    plan=dict(schema=SCHEMA,study_id=STUDY,parent_root=str(parent),parent_fingerprint=original["fingerprint"],model=original["model"],
        parent_checkpoint_sha256=PARENT_SHA,parent_records=str(parent/g0/"recovery/selected_packed_development.json"),
        items_path=str(parent/g1/"capture/items.json"),original_indices=indices,original_logits=str(oldcache/"teacher_logits.npy"),
        accepted_logits=str(parent/g1/"capture/accepted_teacher_logits.npy"),boundary_29=str(parent/g1/"capture/block_29/boundary.npy"),
        boundary_30=str(parent/g0/"capture/block_30/boundary.npy"),cut=29,parent_cut=30,layers=list(range(29,36)),
        label_ids=original["label_ids"],config=asdict(CONFIG),native_correct=48,parent_correct=51,minimum_correct=48,
        input_sha256=inputs,input_identity={p:file_identity(p) for p in inputs},source_sha256=sources,
        gpu_prefix="GPU-0370fdb0",allocation_mib=8192,allocator_mib=7936,logit_atol=ATOL,logit_rtol=RTOL,
        native_control_microbatch=1,other_scoring_microbatch=4,
        metric_atol=1e-7,metric_rtol=1e-6,confirmed=False,scope=SCOPE)
    plan["fingerprint"]=base.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        if base.sha(path)!=digest:raise ValueError("Source changed during prepare")
        dest=root/"producer"/Path(path).relative_to(base.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    base.write(root/"plan.json",plan);base.write(root/"items.json",dict(items=items))
    return dict(status="prepared",fingerprint=plan["fingerprint"])


def verify(root):
    root=Path(root);plan=read(root/"plan.json")
    if plan.get("schema")!=SCHEMA or plan["fingerprint"]!=base.digest({k:v for k,v in plan.items() if k!="fingerprint"}) or plan["config"]!=asdict(CONFIG):
        raise ValueError("Incremental protocol changed")
    with v2.protocol_context():parent=base.verify(plan["parent_root"])
    if parent["fingerprint"]!=plan["parent_fingerprint"] or read(root/"items.json")!=read(plan["items_path"]):raise ValueError("Changed parent or cases")
    for path,digest in plan["source_sha256"].items():
        if base.sha(path)!=digest or base.sha(root/"producer"/Path(path).relative_to(base.ROOT))!=digest:raise ValueError("Frozen incremental producer changed")
    for path,identity in plan["input_identity"].items():
        if file_identity(path)!=identity:raise ValueError("Frozen incremental input changed")
    if base.sha(Path(plan["parent_root"])/"groups/group_00/recovered_unpacked.pt")!=PARENT_SHA:raise ValueError("Parent checkpoint changed")
    return plan


def checked(root,stage,plan):
    path=Path(root)/stage;manifest=read(path/"manifest.json");result=read(path/"result.json")
    if manifest.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"] or result.get("status")!="completed":
        raise ValueError("Completed matching stage required")
    for name,digest in manifest["artifacts"].items():
        if base.sha(path/name)!=digest:raise ValueError("Stage artifact changed: "+name)
    if "result.json" not in manifest["artifacts"]:raise ValueError("Result must be authenticated")
    if stage=="pilot" and (result.get("systems_only") is not True or result.get("updates")!=1 or result.get("fixed_state_unchanged") is not True
        or result["changed_layers"]!=list(range(29,36)) or result.get("norm_changed") is not True
        or any(not result["controls"][k]["passed"] for k in ("native","parent","seam"))):raise ValueError("Passing seven-block pilot required")
    if stage=="train":
        if result.get("updates")!=384 or result.get("completed_epochs")!=12 or result.get("strict_reload_exact") is not True or result.get("fixed_state_unchanged") is not True:
            raise ValueError("Complete fixed recovery required")
        if [r["epoch"] for r in result["history"]]!=list(range(13)):raise ValueError("All initialization/epoch records required")
        winner=max(result["history"],key=lambda r:selection_key(r["metrics"]))
        if winner["epoch"]!=result["selected_epoch"] or winner["metrics"]!=result["selected"] or winner["checkpoint_sha256"]!=result["selected_checkpoint_sha256"]:
            raise ValueError("Selected epoch changed")
        if base.sha(path/"selected.pt")!=result["selected_checkpoint_sha256"]:raise ValueError("Selected checkpoint changed")
    return result


def cap_allocator(lease,*,cuda=None,visible=None):
    cuda=torch.cuda if cuda is None else cuda;visible=os.environ.get("CUDA_VISIBLE_DEVICES","") if visible is None else visible
    if (len(lease["allocation"])!=1 or visible not in lease["allocation"] or not visible.startswith("GPU-0370fdb0")
            or lease["allocation"][visible]!=8192 or cuda.is_initialized()):raise ValueError("Fresh reserved3060 at8192MiB required")
    total=int(cuda.get_device_properties(0).total_memory);limit=7936*1024**2
    if limit>total or cuda.memory.get_allocator_backend()!="native":raise ValueError("Unsupported GPU capacity/allocator")
    cuda.set_per_process_memory_fraction(limit/total,device=0)
    return dict(physical_uuid=visible,lease_bytes=8192*1024**2,allocator_limit_bytes=limit)


def tensor_bytes(module):
    storages={}
    for value in [*module.parameters(),*module.buffers()]:
        storage=value.untyped_storage();storages[(str(value.device),storage.data_ptr())]=storage.nbytes()
    return sum(storages.values())


def inventory(plan,stack,norm,rotary):
    packed=core.cpu_module_clone(stack).pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True)
    suffix=tensor_bytes(packed)+tensor_bytes(norm);rotation=tensor_bytes(rotary)
    prefix=embedding=native=0;seen=set()
    for path in sorted(Path(plan["model"]["path"]).glob("*.safetensors")):
        with path.open("rb") as stream:header=json.loads(stream.read(int.from_bytes(stream.read(8),"little")))
        for name,value in header.items():
            if name=="__metadata__" or name in seen or name=="lm_head.weight":continue
            seen.add(name)
            if value["dtype"]!="BF16":raise ValueError("Pinned native storage must be BF16")
            size=math.prod(value["shape"])*2;native+=size
            if name=="model.embed_tokens.weight":embedding+=size
            elif name.startswith("model.layers.") and int(name.split('.')[2])<29:prefix+=size
    if not prefix or not embedding:raise ValueError("Native prefix and tiedE inventory missing")
    return dict(packed_suffix_and_norm_bytes=suffix,frozen_native_prefix_bytes=prefix,tied_embedding_readout_bytes=embedding,
        rotary_bytes=rotation,full_model_tensor_bytes=prefix+embedding+suffix+rotation,native_full_tensor_bytes=native+rotation,
        accounting="BF16 nativeprefix0..28 and tiedE once; actual packedBF16 suffix/norm buffers including geometry; no KV/activations")


def run(root,stage):
    from .jspace_full_lease import wait_ready
    root=Path(root).resolve();plan=verify(root);directory=root/("verification" if stage=="verify" else stage)
    if directory.exists():raise FileExistsError(directory)
    if stage not in ("pilot","train","verify"):raise ValueError("Explicit stage required")
    pilot=checked(root,"pilot",plan) if stage!="pilot" else None
    result_train=checked(root,"train",plan) if stage=="verify" else None
    lease=allocator=data=None;started=time.monotonic();cuda_started=False
    try:
        lease=wait_ready([0]);allocator=cap_allocator(lease);torch.cuda.init();cuda_started=True;torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4);torch.manual_seed(CONFIG.seed);torch.backends.cuda.matmul.allow_tf32=False
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Reservation changed")
            if torch.cuda.max_memory_reserved()>allocator["allocator_limit_bytes"]:raise MemoryError("Incremental allocator cap exceeded")
        if stage=="pilot":
            directory.mkdir();control=controls(plan,"cuda:0",guard);base.write(directory/"controls.json",control)
            if not control["passed"]:raise RuntimeError("Native48/parent51/seam replay failed")
        stack,norm,readout,rotary,inheritance=build(plan,"cuda:0");data=dataset(plan)
        trainer=Trainer(stack,norm,readout,rotary,data,plan["label_ids"],config=CONFIG,guard=guard)
        before=fixed_state(stack,readout,rotary)
        if stage=="pilot":
            initial,initial_rows=trainer.packed_development()
            hashes=[fingerprint(b.named_parameters()) for b in stack.blocks];norm_before=fingerprint(norm.named_parameters())
            masters=[master_fingerprint(b,trainer.optimizer) for b in stack.blocks];norm_master=master_fingerprint(norm,trainer.optimizer)
            order=sorted(data.training,key=lambda i:data.lengths[i],reverse=True)
            update=trainer.train_window(order[:16])
            after_masters=[master_fingerprint(b,trainer.optimizer) for b in stack.blocks]
            changed=[layer for layer,after,digest in zip(stack.layer_indices,after_masters,masters) if after!=digest]
            bf16_changed=[layer for layer,b,digest in zip(stack.layer_indices,stack.blocks,hashes) if fingerprint(b.named_parameters())!=digest]
            norm_after=master_fingerprint(norm,trainer.optimizer);norm_changed=norm_after!=norm_master
            bf16_norm_changed=fingerprint(norm.named_parameters())!=norm_before;unchanged=fixed_state(stack,readout,rotary)==before
            snapshot=trainer.snapshot();save(directory/"pilot_state.pt",snapshot)
            stack.load_state_dict(builder._load(directory/"pilot_state.pt")["stack"],strict=True)
            norm.load_state_dict(snapshot["norm"],strict=True)
            if changed!=list(range(29,36)) or not norm_changed or not unchanged:raise RuntimeError("Pilot update/frozen-state proof failed")
            result=dict(status="completed",systems_only=True,updates=1,controls=control,inheritance=inheritance,
                initial=initial,initial_records=initial_rows,update=update,changed_layers=changed,norm_changed=norm_changed,
                parameter_change_gate="FP32 optimizer masters; BF16 stored changes are separate rounding diagnostics",
                master_changes=dict(blocks=[dict(layer=layer,before=a,after=b,changed=a!=b) for layer,a,b in
                    zip(stack.layer_indices,masters,after_masters)],norm=dict(before=norm_master,after=norm_after,changed=norm_changed)),
                gradient_proof="Core optimizer validates all gradients finite; each first-step zero-weight-decay FP32 master change proves a nonzero gradient in that block/norm",
                bf16_changed_layers=bf16_changed,bf16_norm_changed=bf16_norm_changed,
                fixed_state_unchanged=True,fixed_state=before,checkpoint_strict_load=True,confirmed=False,scope=SCOPE)
            base.write(directory/"result.json",result)
        elif stage=="train":
            def event(kind,**values):base.event(root/"progress.jsonl",kind,**values)
            result=trainer.fit(directory,event=event,initial_reference=pilot)
        else:
            directory.mkdir();state=builder._load(root/"train/selected.pt")
            stack.load_state_dict(state["stack"],strict=True);norm.load_state_dict(state["norm"],strict=True)
            metrics,rows=trainer.packed_development();reference=read(root/"train"/f"epoch_{result_train['selected_epoch']:02d}_development.json")
            comparison=compare_rows(rows,reference["records"],[data.items[i] for i in data.development],expected_correct=result_train["selected"]["correct"])
            metrics_pass=all(a==reference["metrics"][k] if not isinstance(a,float) else math.isclose(a,reference["metrics"][k],abs_tol=plan["metric_atol"],rel_tol=plan["metric_rtol"]) for k,a in metrics.items())
            unchanged=fixed_state(stack,readout,rotary)==before
            result=dict(status="completed",passed=comparison["passed"] and metrics_pass and unchanged,selected_epoch=result_train["selected_epoch"],
                selected_checkpoint_sha256=result_train["selected_checkpoint_sha256"],strict_state_load=True,fixed_state_unchanged=unchanged,
                comparison=comparison,metrics_pass=metrics_pass,metrics=metrics,records=rows,quality=gates(metrics["correct"]),
                storage=inventory(plan,stack,norm,rotary),packing_validated=comparison["passed"] and metrics_pass and unchanged,confirmed=False,scope=SCOPE)
            base.write(directory/"result.json",result)
            if not result["passed"]:raise RuntimeError("Fresh selected packed verification failed")
        guard();base.write(directory/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
            artifacts={p.name:base.sha(p) for p in directory.iterdir() if p.is_file()},lease_id=lease["id"],allocation=lease["allocation"],
            allocator=allocator,peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started,
            passed=result.get("passed"),quality_pass=result.get("quality_pass")))
        return dict(status="completed",stage=stage,fingerprint=plan["fingerprint"])
    except BaseException as error:
        directory.mkdir(parents=True,exist_ok=True);base.write(directory/"failure.json",dict(status="failed",fingerprint=plan["fingerprint"],error=repr(error),
            lease_id=None if lease is None else lease["id"],peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None));raise
    finally:
        if data is not None:data.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","pilot","train","verify"))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--parent-root",type=Path);args=parser.parse_args()
    if args.stage=="prepare" and args.parent_root is None:parser.error("prepare requires --parent-root")
    print(json.dumps(prepare(args.root,args.parent_root) if args.stage=="prepare" else run(args.root,args.stage),allow_nan=False),flush=True)
