"""Local grouped calibration, parallel polynomial fits, and joint Qwen recovery.

GPU phases are always children of inference-ctl leases. The coordinator itself
is CPU-only and starts a fresh process for every phase to release model memory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "JSPACE-QWEN-ALLBLOCKS-01"
SCHEMA = "jspace.qwen-parallel.v1"
MODULE = "experiments.structure_net.jspace_qwen_parallel"
DEFAULT_ROOT = ROOT / "data/experiments/jspace_qwen_parallel/20261114_v1"
SYSTEMS_ROOT = ROOT / "data/experiments/jspace_qwen_parallel/20261114_shakedown_v1"
PROTOCOL = ROOT / "docs/07 - Status Reports/jspace/2026-11-14_jspace-allblocks-parallel-plan.md"
CAPTURE = ROOT / "data/experiments/jspace_full_compact/20261113_capture_v1"
PRIOR = [ROOT / "data/experiments" / p for p in (
    "jspace_qwen_affine/20261113_v2", "jspace_qwen_layerkd/20261113_v1",
    "jspace_fast_recovery/20261113_v2")]
CONFIG = dict(seed=20861, layers=36, group_size=6, rank=1024, head_dim=64,
    anchors=24, removed_channels=1024, position_capacity=416, pad_to_multiple=128,
    probe_groups=2, probes_per_group=8, activation_positions=16,
    epochs=12, microbatch=4, accumulation=4, learning_rate=1.2e-4,
    learning_rate_schedule="cosine_over_fixed_update_budget",
    weight_decay=0., temperature=2., kl_weight=.7, ce_weight=.3,
    gradient_clip=1., checkpoint_blocks=True, train_final_norm=True,
    train_readout=False, minimum_size_reduction=.38, minimum_speedup=1.4,
    maximum_accuracy_drop=1/100, maximum_family_drop=4/128,
    quality_early_stop=False, native_bytes=6171877632,
    scope="exploratory_single_checkpoint_all_blocks_retained_residual")
CARDS = [("GPU-0370fdb0",11264), ("GPU-8cf440df",5632), ("GPU-7b1db3cb",7168)]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8*1024**2), b""):
            result.update(data)
    return result.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")


def event(path,kind,**values):
    row={"kind":kind,"time":datetime.now(timezone.utc).isoformat(),**values}
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row,allow_nan=False)+"\n")
    print(json.dumps(row,allow_nan=False),flush=True)


def group_layers(index):
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<6:
        raise ValueError("Group index must be 0..5")
    stop=36-index*6
    return list(range(stop-6,stop))


def group_path(root,index):
    group_layers(index)
    return Path(root)/"groups"/f"group_{index:02d}"


def sources():
    from . import jspace_qwen_affine as base
    required=["jspace_qwen_parallel_model.py","jspace_qwen_parallel_fit.py",
              "jspace_qwen_parallel_train.py","jspace_qwen_parallel_evaluate.py"]
    extra=[Path(__file__),PROTOCOL]+[Path(__file__).with_name(name) for name in required]
    for path in extra:
        if not path.is_file():raise FileNotFoundError("Required producer is not ready: "+str(path))
    for pattern in ("jspace_qwen_parallel*.py","jspace_qwen_progressive_calibration.py",
                    "jspace_qwen_compact_stack.py","jspace_streaming_adamw.py",
                    "jspace_qwen_affine_fast.py","jspace_qwen_affine_stack.py","jspace_streamed_jacobian.py"):
        extra+=list(Path(__file__).parent.glob(pattern))
    for pattern in ("test_jspace_qwen_parallel*.py","test_jspace_qwen_progressive_calibration.py",
                    "test_jspace_qwen_compact_stack.py","test_jspace_streaming_adamw.py"):
        extra+=list((ROOT/"tests/structure_net").glob(pattern))
    return sorted(set(base.source_paths()+extra))


def prepare(root,*,systems_only=False,systems_root=SYSTEMS_ROOT):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU preparation requires hidden CUDA")
    from . import jspace_qwen_affine as base
    from .jspace_qwen_compact_stack import estimate_qwen3b_stack_bytes
    from transformers import AutoTokenizer
    root=Path(root)
    if root.exists(): raise FileExistsError(root)
    parent=json.loads((CAPTURE/"plan.json").read_text())
    cached=json.loads((CAPTURE/"capture/items.json").read_text())
    capture_manifest=json.loads((CAPTURE/"capture_manifest.json").read_text())
    if capture_manifest["status"]!="completed":raise ValueError("Original teacher capture is incomplete")
    for name in ("items.json","teacher_logits.npy","boundary_35.npy"):
        if sha(CAPTURE/"capture"/name)!=capture_manifest["artifact_sha256"]["capture/"+name]:
            raise ValueError("Original teacher capture changed: "+name)
    inputs={str(CAPTURE/p):sha(CAPTURE/p) for p in (
        "plan.json","tasks.json","capture_manifest.json","capture/items.json",
        "capture/teacher_logits.npy","capture/boundary_35.npy")}
    dataset_manifest=json.loads((base.DATA_ROOT/"manifest.json").read_text())
    inputs[str(base.DATA_ROOT/"manifest.json")]=sha(base.DATA_ROOT/"manifest.json")
    for entry in dataset_manifest.values():
        for info in entry["files"].values():
            path=base.DATA_ROOT/info["local"];value=sha(path)
            if value!=info["sha256"]:raise ValueError("Benchmark source changed")
            inputs[str(path)]=value
    if systems_only:
        for name in ("fit.pt","relation.pt"):inputs[str(PRIOR[0]/name)]=sha(PRIOR[0]/name)
    else:
        systems_root=Path(systems_root);system_plan=verify(systems_root)
        if not system_plan["systems_only"]:raise ValueError("Systems prerequisite must be separately labeled")
        prerequisites=[systems_root/"pilot",group_path(systems_root,0)/"capture",group_path(systems_root,0)/"basis"]
        prerequisites += [group_path(systems_root,0)/f"block_{layer:02d}"/"fit_result" for layer in group_layers(0)]
        for path in prerequisites:
            checked(systems_root,path)
            inputs[str(path/"manifest.json")]=sha(path/"manifest.json")
        inputs[str(systems_root/"plan.json")]=sha(systems_root/"plan.json")
    excluded={}
    for index,path in enumerate([CAPTURE,*PRIOR]):
        data=json.loads((path/"tasks.json").read_text())
        inputs[str(path/"tasks.json")]=sha(path/"tasks.json")
        for key,cases in data["splits"].items(): excluded[f"{index}_{key}"]=cases
    tokenizer=AutoTokenizer.from_pretrained(parent["model"]["path"],local_files_only=True,trust_remote_code=False)
    fresh,skipped=base.fresh_cases(tokenizer,{"splits":excluded})
    original=json.loads((CAPTURE/"tasks.json").read_text())["splits"]
    tasks={"splits":{"train":original["train"],"development":original["development"],"evaluation":fresh},
           "previous_identities":sorted({c["identity"] for rows in excluded.values() for c in rows}),
           "fresh_exclusions":skipped}
    if set(tasks["previous_identities"]) & {c["identity"] for c in fresh}:
        raise ValueError("Fresh evaluation overlaps previous evidence")
    for path,value in parent["model"]["files"].items():
        if sha(Path(parent["model"]["path"])/path)!=value:raise ValueError("Native model changed")
    total_tokens=sum(len(c["input_ids"]) for c in cached["items"])
    calibration_tokens=sum(len(cached["items"][i]["input_ids"]) for i in cached["calibration_indices"])
    budget={"jacobian_bytes":36*16*calibration_tokens*11008*2,
            "boundary_bytes":36*total_tokens*2048*2,
            "additional_artifact_allowance_bytes":70*1024**3,
            "free_disk_bytes":shutil.disk_usage(root.parent if root.parent.exists() else ROOT).free,
            "packed_estimate":estimate_qwen3b_stack_bytes()}
    if sum(budget[k] for k in ("jacobian_bytes","boundary_bytes","additional_artifact_allowance_bytes"))>budget["free_disk_bytes"]:
        raise OSError("Insufficient disk for complete retained campaign evidence")
    plan={"schema":SCHEMA,"study_id":STUDY_ID,"config":CONFIG,"systems_only":systems_only,
          "created_at":datetime.now(timezone.utc).isoformat(),"model":parent["model"],
          "label_ids":parent["label_ids"],"source_sha256":{str(p):sha(p) for p in sources()},
          "input_sha256":inputs,"task_digest":digest(tasks),"budget":budget,
          "original_cache":str(CAPTURE/"capture"),"groups":[group_layers(i) for i in range(6)]}
    plan["fingerprint"]=digest(plan)
    root.mkdir(parents=True)
    for path in sources():
        target=root/"producer"/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(root/"plan.json",plan);write(root/"tasks.json",tasks)
    return {"status":"prepared","fingerprint":plan["fingerprint"],"budget":budget}


def verify(root,*,model=False):
    root=Path(root);plan=json.loads((root/"plan.json").read_text())
    if plan["schema"]!=SCHEMA or plan["config"]!=CONFIG or plan["fingerprint"]!=digest({k:v for k,v in plan.items() if k!="fingerprint"}):
        raise ValueError("Campaign protocol changed")
    if set(plan["source_sha256"])!={str(p) for p in sources()}:
        raise ValueError("Producer dependency closure changed")
    for path,value in plan["source_sha256"].items():
        if sha(path)!=value or sha(root/"producer"/Path(path).relative_to(ROOT))!=value:
            raise ValueError("Frozen producer changed: "+path)
    if digest(json.loads((root/"tasks.json").read_text()))!=plan["task_digest"]:
        raise ValueError("Campaign tasks changed")
    if model:
        for path,value in plan["model"]["files"].items():
            if sha(Path(plan["model"]["path"])/path)!=value:raise ValueError("Model changed")
    return plan


def complete(root,path,plan,**values):
    path=Path(path)
    files=[p for p in path.rglob("*") if p.is_file() and p.name!="manifest.json"]
    result={"schema":SCHEMA,"status":"completed","fingerprint":plan["fingerprint"],
            "artifacts":{str(p.relative_to(Path(root))):sha(p) for p in files},**values}
    write(path/"manifest.json",result)
    return {k:v for k,v in result.items() if k!="artifacts"}


def checked(root,path,*,hashes=True):
    plan=verify(root);value=json.loads((Path(path)/"manifest.json").read_text())
    if value["status"]!="completed" or value["fingerprint"]!=plan["fingerprint"]:
        raise ValueError("Incomplete/mismatched stage")
    if hashes:
        for name,expected in value["artifacts"].items():
            if sha(Path(root)/name)!=expected:raise ValueError("Artifact changed: "+name)
    return value


def gpu_setup(root):
    from .jspace_full_lease import wait_ready
    lease=wait_ready([0]);plan=verify(root,model=True)
    for name,expected in plan["input_sha256"].items():
        if sha(name)!=expected:raise ValueError("Campaign input changed: "+name)
    import torch
    torch.set_num_threads(4);torch.manual_seed(CONFIG["seed"])
    torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    return plan,lease,lambda:wait_ready([0])


def capture(root,index,*,limit=None):
    from .jspace_qwen_parallel_model import build_teacher
    from .jspace_qwen_progressive_calibration import GroupedTeacherCapture
    import torch
    plan,lease,guard=gpu_setup(root);group=group_path(root,index)
    group.mkdir(parents=True,exist_ok=True)
    if limit is not None and not plan["systems_only"]:raise ValueError("Capture limit is systems-only")
    if index:checked(root,group_path(root,index-1)/"recovery")
    meta=json.loads((Path(plan["original_cache"])/"items.json").read_text())
    cases=meta["items"];indices=meta["calibration_indices"]
    if limit is not None:
        selected=indices[:limit]+meta["development_indices"][:1]
        cases=[cases[i] for i in selected];indices=list(range(limit))
    model,tokenizer=build_teacher(root,index-1 if index else None,device="cuda:0")
    teacher_id=plan["model"] if index==0 else sha(group_path(root,index-1)/"recovered_unpacked.pt")
    start=time.perf_counter()
    with GroupedTeacherCapture(group/"capture",model,group_layers(index),cases,indices,
            teacher_fingerprint=digest(teacher_id),probe_groups=2,probes_per_group=8,
            activation_positions=16,position_capacity=416,seed=CONFIG["seed"]+100000*index,guard=guard) as recorder:
        for i,case in enumerate(cases):
            ids=torch.tensor([case["input_ids"]],device="cuda:0")
            recorder.capture_case(i,lambda:model.model(input_ids=ids,use_cache=False).last_hidden_state)
            if i%8==0 or i+1==len(cases):
                event(group/"capture_events.jsonl","capture",group=index,cases=i+1,total=len(cases),
                      seconds=time.perf_counter()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved())
        recorder.finish()
    guard()
    if torch.cuda.max_memory_reserved()>sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("Capture exceeded reservation")
    return complete(root,group/"capture",plan,seconds=time.perf_counter()-start,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),lease_id=lease["id"],systems_only=plan["systems_only"])


def basis(root,index):
    from .jspace_qwen_parallel_fit import finalize_basis,capture_compiled_development
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    from .jspace_qwen_affine import positions
    import torch
    plan,lease,guard=gpu_setup(root);group=group_path(root,index)
    checked(root,group/"capture")
    finalize_basis(group/"capture",group,rank=CONFIG["rank"],head_dim=64,device="cuda:0",guard=guard)
    readout,norm,rotary=load_readout_norm(plan["model"]["path"],device="cuda:0")
    del readout,norm
    def forward(block,x,case):
        ids=torch.arange(x.shape[1],device=x.device)[None]
        mask=torch.full((x.shape[1],x.shape[1]),torch.finfo(x.dtype).min,device=x.device,dtype=x.dtype).triu(1)[None,None]
        return block(x,attention_mask=mask,position_ids=ids,cache_position=ids[0],position_embeddings=rotary(x,ids))[0]
    with torch.no_grad():
        for layer in group_layers(index):
            native=load_native_block(plan["model"]["path"],layer,device="cuda:0")
            capture_compiled_development(group/f"block_{layer:02d}",group/"capture",layer,native,forward,device="cuda:0",guard=guard)
            del native
            event(group/"basis_events.jsonl","basis_interface_complete",layer=layer)
    path=group/"basis";path.mkdir()
    # Include the generated external block files in this phase's provenance.
    files={str(p.relative_to(Path(root))):sha(p) for layer in group_layers(index)
           for p in (group/f"block_{layer:02d}").rglob("*") if p.is_file()}
    write(path/"manifest.json",{"status":"completed","fingerprint":plan["fingerprint"],"artifacts":files,
          "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"lease_id":lease["id"]})
    return {"status":"completed","group":index,"layers":group_layers(index)}


def relation(root,index,layer):
    from .jspace_qwen_parallel_fit import fit_relation
    import torch
    plan,lease,guard=gpu_setup(root)
    if layer not in group_layers(index):raise ValueError("Layer outside group")
    group=group_path(root,index);checked(root,group/"basis")
    path=group/f"block_{layer:02d}"
    result=fit_relation(path,device="cuda:0",guard=guard,seed=CONFIG["seed"]+layer)
    record=path/"fit_result";record.mkdir()
    write(record/"manifest.json",{"status":"completed","fingerprint":plan["fingerprint"],
          "artifacts":{str(p.relative_to(Path(root))):sha(p) for p in (path/"relation.pt",path/"relations.json")},
          "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"lease_id":lease["id"],"gpu":torch.cuda.get_device_name(0)})
    return {"status":"completed","layer":layer}


def recovery_config(index=0):
    from dataclasses import fields
    from .jspace_qwen_parallel_train import RecoveryConfig
    keys={f.name for f in fields(RecoveryConfig)}
    return RecoveryConfig(**{k:v for k,v in CONFIG.items() if k in keys},
                          max_cpu_state_bytes=32*1024**3)


def train(root,index):
    from .jspace_qwen_parallel_model import build_stack,load_readout_norm,load_recovered_checkpoint
    from .jspace_qwen_parallel_train import BoundaryDataset,SuffixTrainer
    import torch
    plan,lease,guard=gpu_setup(root);group=group_path(root,index)
    checked(root,group/"capture")
    for earlier in range(index+1):
        for layer in group_layers(earlier):checked(root,group_path(root,earlier)/f"block_{layer:02d}"/"fit_result")
    items=json.loads((group/"capture/items.json").read_text())["items"]
    original=json.loads((Path(plan["original_cache"])/"items.json").read_text())["items"]
    original_indices={c["identity"]:i for i,c in enumerate(original)}
    cut=group_layers(index)[0]
    dataset=BoundaryDataset(items,group/"capture"/f"block_{cut:02d}"/"boundary.npy",
        group/"capture/accepted_teacher_logits.npy",
        original_logits_path=Path(plan["original_cache"])/"teacher_logits.npy",
        original_indices=[original_indices[c["identity"]] for c in items])
    stack=build_stack(root,cut,True,index-1 if index else None,device="cuda:0",checkpoint_blocks=True)
    readout,norm,rotary=load_readout_norm(plan["model"]["path"],device="cuda:0")
    if index:norm.load_state_dict(load_recovered_checkpoint(root,index-1)["norm"],strict=True)
    trainer=SuffixTrainer(stack,norm,readout,rotary,dataset,plan["label_ids"],config=recovery_config(index),guard=guard)
    started=time.perf_counter()
    def progress(kind,**values):
        # Every update is durable; emit roughly one line per four updates.
        row={"kind":kind,"seconds":time.perf_counter()-started,"group":index,**values}
        with (group/"recovery_events.jsonl").open("a") as stream:stream.write(json.dumps(row,allow_nan=False)+"\n")
        if kind!="update" or values["update"]%4==1:print(json.dumps(row,allow_nan=False),flush=True)
    try:
        report=trainer.fit(group/"recovery",event=progress)
        os.link(group/"recovery/recovered_unpacked.pt",group/"recovered_unpacked.pt")
        guard()
        peak=torch.cuda.max_memory_reserved()
        if peak>sum(lease["allocation"].values())*1024**2:raise RuntimeError("Recovery exceeded reservation")
        return complete(root,group/"recovery",plan,group=index,selected_epoch=report["selected_epoch"],
                        selected=report["selected"],peak_reserved_bytes=peak,lease_id=lease["id"])
    finally:dataset.close()


def pilot(root):
    """Representative shape-equivalent 36-block backward; never quality evidence."""
    from .jspace_qwen_parallel_model import systems_stack,load_readout_norm,state_dict_cpu
    from .jspace_qwen_parallel_train import BoundaryDataset,SuffixTrainer,cpu_module_clone
    import gc
    import torch
    plan,lease,guard=gpu_setup(root)
    if not plan["systems_only"]:raise ValueError("Pilot requires separate systems-only root")
    path=Path(root)/"pilot";path.mkdir()
    cache=Path(plan["original_cache"]);items=json.loads((cache/"items.json").read_text())["items"]
    dataset=BoundaryDataset(items,cache/"boundary_35.npy",cache/"teacher_logits.npy")
    stack=systems_stack(plan["model"]["path"],PRIOR[0]/"fit.pt",PRIOR[0]/"relation.pt",
                        device="cuda:0",stabilized=True)
    metadata=stack.systems_metadata
    readout,norm,rotary=load_readout_norm(plan["model"]["path"],device="cuda:0")
    started=time.perf_counter()
    trainer=SuffixTrainer(stack,norm,readout,rotary,dataset,plan["label_ids"],config=recovery_config(),guard=guard)
    setup=time.perf_counter()-started
    selected=sorted(dataset.training,key=lambda i:dataset.lengths[i],reverse=True)[:16]
    event(path/"events.jsonl","full_stack_backward_start",blocks=36,case_lengths=[dataset.lengths[i] for i in selected],
          optimizer_setup_seconds=setup,optimizer_memory=trainer.optimizer.memory_stats())
    torch.cuda.synchronize();started=time.perf_counter()
    update=trainer.train_window(selected)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    train_peak=torch.cuda.max_memory_reserved()
    event(path/"events.jsonl","full_stack_backward_complete",seconds=elapsed,update=update,peak_reserved_bytes=train_peak)
    # Match the real selection lifecycle: a packed inference copy coexists
    # with live training weights and the CPU optimizer's master/moment state.
    torch.cuda.synchronize();development_started=time.perf_counter()
    development,_=trainer.packed_development()
    torch.cuda.synchronize();development_seconds=time.perf_counter()-development_started
    event(path/"events.jsonl","packed_development_lifecycle_complete",seconds=development_seconds,
          peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope="systems-only values are not task-quality evidence")
    snapshot=trainer.snapshot();torch.save(snapshot,path/"unpacked.pt")
    restored=torch.load(path/"unpacked.pt",map_location="cpu",weights_only=True)
    stack.load_state_dict(restored["stack"],strict=True);norm.load_state_dict(restored["norm"],strict=True)
    # Exercise the same CPU fold used at each packed-development selection.
    packed=cpu_module_clone(stack).pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).eval().requires_grad_(False)
    packed.checkpoint_blocks=False
    packed_bytes=packed.storage_breakdown()
    del snapshot,restored,trainer,stack;gc.collect();torch.cuda.empty_cache()
    packed.to("cuda:0");norm.eval()
    batch=dataset.batch(selected[:1],device="cuda:0")
    x=batch["hidden"];pos=batch["position_ids"];rope=rotary(x,pos)
    def forward():return packed(x,position_ids=pos,cache_position=pos[0],attention_mask=batch["attention_mask"],position_embeddings=rope)[0]
    with torch.inference_mode():
        packed.enabled=False;reference=forward()
        packed.enabled=True;candidate=forward()
        difference=candidate.float()-reference.float()
        errors={"max_absolute_error":float(difference.abs().max()),
                "relative_rmse":float(difference.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-20)),
                "exact":bool(torch.equal(reference,candidate))}
        write(path/"fused_comparison.json",errors)
        if not errors["exact"]:raise ValueError("Full-stack fused/reference values differ; error metrics retained")
        packed_state=state_dict_cpu(packed)
        torch.save({"stack":packed_state,"norm":state_dict_cpu(norm)},path/"packed.pt")
        packed.load_state_dict(torch.load(path/"packed.pt",map_location="cpu",weights_only=True)["stack"],strict=True)
        if not torch.equal(candidate,forward()):raise ValueError("Full-stack packed reload differs")
    # Assemble the HF container without allocating another native 3B model.
    # All native cache indices remain 0..35 inside the single compact module.
    from transformers import AutoConfig,AutoModelForCausalLM,GenerationConfig
    from .jspace_qwen_affine import graph_decoder
    config=AutoConfig.from_pretrained(plan["model"]["path"],local_files_only=True)
    config._attn_implementation="eager"
    with torch.device("meta"):
        model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16)
    model.model.layers=torch.nn.ModuleList([packed])
    tied=torch.nn.Parameter(readout,requires_grad=False)
    model.model.embed_tokens.weight=tied;model.lm_head.weight=tied
    model.model.norm=norm;model.model.rotary_emb=rotary
    model.generation_config=GenerationConfig.from_pretrained(plan["model"]["path"],local_files_only=True)
    model.eval().requires_grad_(False)
    if any(t.is_meta for t in [*model.parameters(),*model.buffers()]):raise ValueError("Unmaterialized systems model")
    decoder=graph_decoder(model)
    event(path/"events.jsonl","full_stack_cache_graph_start")
    with torch.inference_mode():
        graph_setup_seconds=decoder.capture()
        prompt=torch.tensor([items[dataset.development[0]]["input_ids"]],device="cuda:0")
        uncaptured=decoder.generate(prompt,False);captured=decoder.generate(prompt,True)
        if not torch.equal(uncaptured,captured):raise ValueError("Full-stack cache graph changes generated tokens")
    event(path/"events.jsonl","full_stack_cache_graph_complete",capture_seconds=graph_setup_seconds,
          generated_tokens=len(captured),cache_layers=len(decoder.cache.key_cache))
    peak=torch.cuda.max_memory_reserved()
    if peak>sum(lease["allocation"].values())*1024**2:raise RuntimeError("Systems pilot exceeded reservation")
    result={"scope":"systems_lifecycle_only_not_quality_evidence","initialization":metadata,
            "optimizer_setup_seconds":setup,"window_seconds":elapsed,"window":update,
            "packed_development_seconds":development_seconds,"packed_development_cases":development["count"],
            "train_peak_reserved_bytes":train_peak,"peak_reserved_bytes":peak,
            "case_lengths":[dataset.lengths[i] for i in selected],"packed_stack_bytes":packed_bytes,
            "whole_tensor_bytes":packed_bytes["total"]+readout.numel()*readout.element_size()+sum(p.numel()*p.element_size() for p in norm.parameters()),
            "fused_same_geometry_exact":True,"packed_reload_exact":True,"unpacked_reload_strict":True}
    result.update(cache_graph_tokens_exact=True,cache_layers=len(decoder.cache.key_cache),graph_setup_seconds=graph_setup_seconds)
    write(path/"result.json",result);dataset.close()
    return complete(root,path,plan,**result,lease_id=lease["id"])


def leased_command(root,stage,*,index=None,layer=None,card=CARDS[0],minutes=240,extra=()):
    args=["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run",
          "--allocation",f"{card[0]}={card[1]}","--minutes",str(minutes),
          "--reason",f"{STUDY_ID} local {stage} group={index} layer={layer}","--",
          "pixi","run","--locked","--offline","-e","ncm","python","-m",MODULE,stage,"--root",str(root)]
    if index is not None:args += ["--group",str(index)]
    if layer is not None:args += ["--layer",str(layer)]
    return args+list(extra)


def run_phase(root,stage,*,index=None,layer=None,card=CARDS[0],minutes=240,extra=()):
    logs=Path(root)/"logs";logs.mkdir(exist_ok=True)
    name=f"{stage}_g{index}_l{layer}"
    path=logs/(name+".log")
    if path.exists():raise FileExistsError("Inspect existing attempt before retry: "+str(path))
    command=leased_command(root,stage,index=index,layer=layer,card=card,minutes=minutes,extra=extra)
    event(Path(root)/"coordinator.jsonl","phase_start",stage=stage,group=index,layer=layer,command=command)
    env={**os.environ,"PYTHONUNBUFFERED":"1","TRITON_CACHE_DIR":"/tmp/structure-net-triton-cache",
         "PIXI_CACHE_DIR":"/tmp/structure-net-pixi-cache","UV_CACHE_DIR":"/tmp/structure-net-uv-cache",
         "OPENBLAS_NUM_THREADS":"4","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
    with path.open("x") as stream:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    event(Path(root)/"coordinator.jsonl","phase_end",stage=stage,group=index,layer=layer,returncode=result.returncode)
    if result.returncode:raise RuntimeError(f"{name} failed; see {path}")


def run(root):
    plan=verify(root)
    if plan["systems_only"]:raise ValueError("Use explicit pilot phases for a systems root")
    if os.environ.get("IC_LEASE"):raise ValueError("Coordinator must not hold a parent GPU lease")
    for index in range(6):
        group=group_path(root,index)
        for phase in ("capture","basis"):
            if (group/phase/"manifest.json").exists():checked(root,group/phase)
            else:run_phase(root,phase,index=index)
        # One worker per physical card, two consecutive fits per worker.
        def worker(slot):
            for layer in group_layers(index)[slot::3]:
                record=group/f"block_{layer:02d}"/"fit_result"
                if (record/"manifest.json").exists():checked(root,record)
                else:run_phase(root,"relation",index=index,layer=layer,card=CARDS[slot])
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(worker,slot) for slot in range(3)]):future.result()
        if (group/"recovery/manifest.json").exists():checked(root,group/"recovery")
        else:run_phase(root,"train",index=index,minutes=720)
    if not (Path(root)/"evaluation/manifest.json").exists():run_phase(root,"evaluate",minutes=240)
    return json.loads((Path(root)/"campaign_results.json").read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","capture","basis","relation","train","pilot","evaluate","run"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--group",type=int,default=0)
    parser.add_argument("--layer",type=int)
    parser.add_argument("--systems-only",action="store_true")
    parser.add_argument("--systems-root",type=Path,default=SYSTEMS_ROOT)
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.stage=="prepare":result=prepare(args.root,systems_only=args.systems_only,systems_root=args.systems_root)
    elif args.stage=="capture":result=capture(args.root,args.group,limit=args.limit)
    elif args.stage=="basis":result=basis(args.root,args.group)
    elif args.stage=="relation":result=relation(args.root,args.group,args.layer)
    elif args.stage=="train":result=train(args.root,args.group)
    elif args.stage=="pilot":result=pilot(args.root)
    elif args.stage=="evaluate":
        from .jspace_qwen_parallel_evaluate import evaluate
        result=evaluate(args.root)
    elif args.stage=="run":result=run(args.root)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
