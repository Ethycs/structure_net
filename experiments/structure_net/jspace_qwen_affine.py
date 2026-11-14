"""A local, full-output-Jacobian affine compression experiment on Qwen."""
import argparse
from collections import Counter
import copy
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_bench_data import DATASETS,normalize,sha,write
from .jspace_bench_tensor import accuracy,comparison,model_bytes,run_cases
from .jspace_full_capture import read_bits
from .jspace_full_lease import wait_ready
from .jspace_lens import load_small_qwen
from .jspace_logit_sketch import project_sign_probe_groups,normalization_metadata,subspace_energy_fraction
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_benchmark_groebner import fit_benchmark_groebner

ROOT=Path(__file__).resolve().parents[2]
STUDY_ID="JSPACE-QWEN-AFFINE-01"
DEFAULT_ROOT=ROOT/"data/experiments/jspace_qwen_affine/20261113_v2"
SHAKEDOWN_ROOT=ROOT/"data/experiments/jspace_qwen_affine/20261113_shakedown_v2"
CAPTURE=ROOT/"data/experiments/jspace_full_compact/20261113_capture_v1"
DATA_ROOT=ROOT/"data/benchmarks/jspace_four_tasks_20261113"
PROTOCOL=ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-qwen-affine-plan.md"
CONFIG={"seed":861,"layer":35,"ranks":[1536,1024],"calibration_per_task":32,
        "probe_groups":2,"probes_per_group":8,"position_capacity":416,"max_tokens":384,
        "fresh_evaluation_per_task":128,"epochs":12,"accumulation":4,"learning_rate":3e-5,
        "weight_decay":0.,"gradient_clip":1.,"temperature":2.,"kl_weight":.7,"ce_weight":.3,
        "patience":4,"minimum_epochs":4,"minimum_improvement":1e-4,"graph_repeats":3,
        "new_tokens":32,"full_rank_minimum_dev_agreement":63/64,"full_rank_maximum_kl":.01,
        "groebner_anchor_widths":[16,24,32],"groebner_removed_channels":1024,
        "groebner_ridge":.01,"groebner_eligibility_nrmse":.5,
        "groebner_svd_rank":64,"groebner_oversample":16,"groebner_power_iterations":1,
        "groebner_chunk_rows":256,
        "scope":"exploratory_qwen_task_benchmark_groebner_affine_full_vocabulary_sketch"}
ARMS={"r1536_h128":{"rank":1536,"head_dim":128},
      "r1536_h64":{"rank":1536,"head_dim":64},"r1024_h64":{"rank":1024,"head_dim":64},
      "r1536_h64_gb":{"rank":1536,"head_dim":64,"groebner":True}}
CONFIG["arms"]=ARMS


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("Hide CUDA for CPU preparation")


def save(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    torch.save(value,path)


def source_paths():
    names=["jspace_qwen_affine.py","jspace_qwen_affine_block.py","jspace_logit_sketch.py","jspace_benchmark_groebner.py",
           "jspace_bench_data.py","jspace_bench_tensor.py","jspace_groebner_tensor.py",
           "jspace_lens.py","jspace_full_capture.py","jspace_full_lease.py",
           "jspace_compact_model.py","jspace_stacked_graph_decode.py","jspace_stacked_graph_decode_v2.py",
           "jspace_stacked_graph_decode_v3.py","jspace_bench_stacked.py","jspace_bench_stacked_timing.py",
           "jspace_packed_core.py"]
    paths=[Path(__file__).with_name(n) for n in names]
    paths += [PROTOCOL,ROOT/"pixi.lock",ROOT/"pyproject.toml",
              ROOT/"tests/structure_net/test_jspace_qwen_affine_block.py",
              ROOT/"tests/structure_net/test_jspace_logit_sketch.py",
              ROOT/"tests/structure_net/test_jspace_benchmark_groebner.py",
              ROOT/"tests/structure_net/test_jspace_qwen_affine_protocol.py"]
    return paths+sorted((ROOT/"src/structure_net_resources").glob("*.py"))


def fresh_cases(tokenizer,old):
    import pyarrow.parquet as pq
    manifest=json.loads((DATA_ROOT/"manifest.json").read_text())
    seen={c["identity"] for rows in old["splits"].values() for c in rows}
    rows=[];excluded={}
    for fi,family in enumerate(DATASETS):
        entry=manifest[family];split=entry["evaluation_source_split"]
        source=pq.read_table(DATA_ROOT/entry["files"][split]["local"]).to_pylist()
        order=list(range(len(source)));random.Random(CONFIG["seed"]+9000+fi).shuffle(order)
        selected=[];skipped=Counter()
        for index in order:
            try:question,choices,answer=normalize(family,source[index])
            except ValueError as error:
                if str(error)!="Unsupported or unlabeled multiple-choice item":raise
                skipped["unsupported_choice_format"]+=1;continue
            identity=hashlib.sha256(json.dumps([question,choices]).encode()).hexdigest()
            if identity in seen:skipped["overlap"]+=1;continue
            user=question+"\n\n"+"\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
            messages=[{"role":"system","content":"Answer the multiple-choice question. Reply with only the letter of the correct option."},
                      {"role":"user","content":user}]
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            if len(ids)>CONFIG["max_tokens"]:skipped["long"]+=1;continue
            seen.add(identity);selected.append({"family":family,"source_split":split,"source_row":index,
                "identity":identity,"prompt":user,"input_ids":ids,"choices":choices,"answer":answer})
            if len(selected)==CONFIG["fresh_evaluation_per_task"]:break
        if len(selected)!=CONFIG["fresh_evaluation_per_task"]:raise ValueError("Insufficient fresh evaluation")
        rows+=selected;excluded[family]=dict(skipped)
    return rows,excluded


def prepare(root,systems_only=False):
    cpu_only()
    from transformers import AutoTokenizer
    if root.exists():raise FileExistsError(root)
    parent=json.loads((CAPTURE/"plan.json").read_text())
    manifest=json.loads((CAPTURE/"capture_manifest.json").read_text())
    if manifest["status"]!="completed":raise ValueError("Incomplete teacher cache")
    cache_names=["items.json","boundary_35.npy","boundary_36.npy","teacher_logits.npy",
                 "mlp_calibration_35.npy","mlp_development_35.npy"]
    inputs={}
    for name in cache_names:
        p=CAPTURE/"capture"/name;value=sha(p)
        if value!=manifest["artifact_sha256"]["capture/"+name]:raise ValueError("Teacher cache changed")
        inputs[str(p)]=value
    inputs.update({str(CAPTURE/n):sha(CAPTURE/n) for n in ("plan.json","tasks.json","capture_manifest.json")})
    dataset=json.loads((DATA_ROOT/"manifest.json").read_text())
    for entry in dataset.values():
        for f in entry["files"].values():
            p=DATA_ROOT/f["local"];value=sha(p)
            if value!=f["sha256"]:raise ValueError("Dataset changed")
            inputs[str(p)]=value
    old=json.loads((CAPTURE/"tasks.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained(parent["model"]["path"],local_files_only=True,trust_remote_code=False)
    fresh,excluded=fresh_cases(tokenizer,old)
    tasks={"splits":{**old["splits"],"previous_evaluation":old["splits"]["evaluation"],"evaluation":fresh},
           "datasets":dataset,"fresh_exclusions":excluded}
    for a in ("train","development","previous_evaluation"):
        if {c["identity"] for c in tasks["splits"][a]}&{c["identity"] for c in fresh}:raise ValueError("Evaluation leakage")
    plan={"study_id":STUDY_ID,"config":CONFIG,"systems_only":systems_only,"model":parent["model"],
          "label_ids":parent["label_ids"],"cache_root":str(CAPTURE/"capture"),"created_at":datetime.now(timezone.utc).isoformat(),
          "task_digest":digest(tasks),"input_sha256":inputs,"source_sha256":{str(p):sha(p) for p in source_paths()}}
    plan["fingerprint"]=digest(plan);root.mkdir(parents=True)
    for p in source_paths():
        target=root/"producer"/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(root/"tasks.json",tasks);write(root/"plan.json",plan)
    return {"root":str(root),"fingerprint":plan["fingerprint"],"counts":{k:len(v) for k,v in tasks["splits"].items()}}


def verify(root,inputs=False,model=False):
    plan=json.loads((root/"plan.json").read_text())
    if plan["config"]!=CONFIG or plan["study_id"]!=STUDY_ID or plan["fingerprint"]!=digest({k:v for k,v in plan.items() if k!="fingerprint"}):raise ValueError("Changed protocol")
    for name,expected in plan["source_sha256"].items():
        if sha(name)!=expected or sha(root/"producer"/Path(name).relative_to(ROOT))!=expected:raise ValueError("Changed producer: "+name)
    if digest(json.loads((root/"tasks.json").read_text()))!=plan["task_digest"]:raise ValueError("Changed tasks")
    if inputs:
        for name,expected in plan["input_sha256"].items():
            if sha(name)!=expected:raise ValueError("Changed input: "+name)
    if model:
        for name,expected in plan["model"]["files"].items():
            if sha(Path(plan["model"]["path"])/name)!=expected:raise ValueError("Changed teacher weights")
    return plan


def checked(root,name):
    plan=verify(root);manifest=json.loads((root/f"{name}_manifest.json").read_text())
    if manifest["status"]!="completed" or manifest["fingerprint"]!=plan["fingerprint"]:raise ValueError("Incomplete stage")
    for p,value in manifest["artifact_sha256"].items():
        if sha(root/p)!=value:raise ValueError("Changed stage artifact: "+p)
    return manifest


class Stage:
    def __init__(self,root,name):
        self.lease=wait_ready([0]);self.guard=wait_ready;self.plan=verify(root,inputs=True,model=True)
        self.root=root;self.path=root/name;self.path.mkdir(exist_ok=False);self.name=name;self.started=time.perf_counter();self.counts=Counter()
        torch.set_num_threads(4);torch.manual_seed(CONFIG["seed"]);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        self.model,self.tokenizer=load_small_qwen(self.plan["model"]["path"],device="cuda:0",dtype=torch.bfloat16)
        self.native=copy.deepcopy(self.model.model.layers[-1]).cpu()
        self.native_norm={k:v.detach().cpu().clone() for k,v in self.model.model.norm.state_dict().items()}
        self.tasks=json.loads((root/"tasks.json").read_text())["splits"];self.label_ids=torch.tensor(self.plan["label_ids"],device="cuda:0")
        cache=Path(self.plan["cache_root"]);self.items=json.loads((cache/"items.json").read_text())["items"]
        self.input=np.load(cache/"boundary_35.npy",mmap_mode="r");self.targets=np.load(cache/"teacher_logits.npy",mmap_mode="r")
        self.indices={c["identity"]:i for i,c in enumerate(self.items)}
        write(self.path/"lease.json",{k:self.lease[k] for k in ("id","holder","allocation","state","ready","end")})

    def event(self,kind,**values):
        self.counts[kind]+=1;row={"kind":kind,"elapsed_seconds":time.perf_counter()-self.started,**values}
        with (self.path/"events.jsonl").open("a") as stream:stream.write(json.dumps(row,allow_nan=False)+"\n")
        if self.counts[kind]==1 or self.counts[kind]%16==0 or kind in ("epoch","complete"):
            self.guard([0]);print(json.dumps(row,allow_nan=False),flush=True)

    def finish(self,names=(),**extra):
        self.guard([0]);verify(self.root);peak=torch.cuda.max_memory_reserved()
        if peak>sum(self.lease["allocation"].values())*1024**2:raise RuntimeError("Reservation exceeded")
        paths=[p for p in self.path.rglob("*") if p.is_file()]+[self.root/n for n in names]
        value={"study_id":STUDY_ID,"status":"completed","stage":self.name,"fingerprint":self.plan["fingerprint"],
               "elapsed_seconds":time.perf_counter()-self.started,"peak_reserved_bytes":peak,"lease_id":self.lease["id"],
               "gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),"counts":dict(self.counts),
               "artifact_sha256":{str(p.relative_to(self.root)):sha(p) for p in paths},**extra}
        write(self.root/f"{self.name}_manifest.json",value)
        return {k:v for k,v in value.items() if k!="artifact_sha256"}

    def x(self,case):return read_bits(self.input,slice(case["offset"],case["stop"]))[None]

    def reset(self):
        self.model.model.layers[-1]=copy.deepcopy(self.native).cuda()
        self.model.model.norm=self.model.model.norm.to(dtype=torch.bfloat16)
        self.model.model.norm.load_state_dict(self.native_norm);self.model.requires_grad_(False)
        gc.collect();torch.cuda.empty_cache()

    def install(self,fit,rank,trainable=False,head_dim=128,groebner=False):
        self.reset();source=self.model.model.layers[-1]
        projection=None if head_dim==128 else {"rope_pairs":fit["rope_pairs"].cuda(),"value_basis":fit["value_basis"].cuda()}
        block=AffineQwenBlock(source,fit["basis"][:,:rank].cuda(),fit["means"].cuda(),trainable=trainable,
                              parameter_dtype=torch.float32 if trainable else torch.bfloat16,head_projection=projection)
        if groebner:
            checked(self.root,"relations")
            relation=torch.load(self.root/"relation.pt",map_location="cpu",weights_only=True)
            block.compile_polynomial(relation,relation["removed"])
        self.model.model.layers[-1]=block
        if trainable:self.model.model.norm.float().requires_grad_(True)
        return block


def positions(stage,x):
    ids=torch.arange(x.shape[1],device=x.device)[None]
    return ids,stage.model.model.rotary_emb(x,ids)


def forward_block(stage,x):
    ids,pos=positions(stage,x);length=x.shape[1]
    mask=torch.full((length,length),torch.finfo(x.dtype).min,device=x.device,dtype=x.dtype).triu(1)[None,None]
    return stage.model.model.layers[-1](x,attention_mask=mask,position_ids=ids,
                cache_position=ids[0],position_embeddings=pos,use_cache=False)[0]


def cached_logits(stage,case):
    x=stage.x(case);hidden=forward_block(stage,x)
    return F.linear(stage.model.model.norm(hidden[:,case["loss_positions"]]),stage.model.lm_head.weight)[0].float()


def objective(logits,teacher,case):
    t=CONFIG["temperature"];target=(teacher.float()/t).log_softmax(-1)
    kd=(target.exp()*(target-(logits.float()/t).log_softmax(-1))).sum(-1).mean()
    ce=F.cross_entropy(logits.float(),torch.tensor(case["targets"],device=logits.device))
    return CONFIG["kl_weight"]*t*t*kd+CONFIG["ce_weight"]*ce,kd,ce


@torch.no_grad()
def development(stage,tag):
    rows=[];losses=[];kls=[];ces=[]
    for i,case in enumerate(stage.items):
        if case["split"]!="development":continue
        with torch.autocast("cuda",dtype=torch.bfloat16):logits=cached_logits(stage,case)
        teacher=read_bits(stage.targets,i);loss,kd,ce=objective(logits,teacher,case)
        values=logits[-1,stage.label_ids[:len(case["choices"])]]
        rows.append({"identity":case["identity"],"family":case["family"],"answer":case["answer"],
                     "prediction":int(values.argmax()),"logits":values.cpu().tolist()})
        losses.append(float(loss));kls.append(float(kd));ces.append(float(ce))
    value={"accuracy":accuracy(rows),"loss":sum(losses)/len(losses),"kl":sum(kls)/len(kls),"ce":sum(ces)/len(ces)}
    write(stage.path/(tag+"_development.json"),rows);write(stage.path/(tag+"_metrics.json"),value)
    return value,rows


def packed_development(stage,tag):
    """Select checkpoints in the actual deployment precision, preserving Adam state."""
    live_block=stage.model.model.layers[-1];live_norm=stage.model.model.norm
    try:
        stage.model.model.layers[-1]=copy.deepcopy(live_block).pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
        stage.model.model.norm=copy.deepcopy(live_norm).to(dtype=torch.bfloat16).requires_grad_(False)
        return development(stage,tag)
    finally:
        stage.model.model.layers[-1]=live_block;stage.model.model.norm=live_norm


def checkpoint(stage,rank,head_dim=128):
    return {"rank":rank,"head_dim":head_dim,"block":{k:v.detach().cpu().clone() for k,v in stage.model.model.layers[-1].state_dict().items()},
            "norm":{k:v.detach().cpu().clone() for k,v in stage.model.model.norm.state_dict().items()}}


def calibration(root):
    if not verify(root)["systems_only"]:checked(SHAKEDOWN_ROOT,"shakedown")
    stage=Stage(root,"calibration");width=stage.model.config.hidden_size
    sums=torch.zeros(CONFIG["position_capacity"],width,device="cuda:0");counts=torch.zeros(len(sums),device="cuda:0")
    for case in stage.items:
        if case["split"]!="train":continue
        x=stage.x(case)[0].float();sums[:len(x)]+=x;counts[:len(x)]+=1
    global_mean=sums.sum(0)/counts.sum();means=sums/counts[:,None].clamp_min(1)
    means[counts==0]=global_mean
    selected=[]
    for f in DATASETS:selected += [c for c in stage.items if c["split"]=="train" and c["family"]==f][:CONFIG["calibration_per_task"]]
    if stage.plan["systems_only"]:selected=selected[:2]
    grams=torch.zeros(CONFIG["probe_groups"],width,width,device="cuda:0")
    native_dim=stage.model.config.hidden_size//stage.model.config.num_attention_heads
    kv=stage.model.config.num_key_value_heads;group_size=stage.model.config.num_attention_heads//kv
    pair_scores=torch.zeros(CONFIG["probe_groups"],kv,native_dim//2,device="cuda:0")
    mask_grams=torch.zeros(CONFIG["probe_groups"],kv*native_dim//2,kv*native_dim//2,device="cuda:0")
    value_moments=torch.zeros(kv,native_dim,native_dim,device="cuda:0")
    value_means=torch.zeros(kv,native_dim,device="cuda:0")
    generator=torch.Generator(device="cuda:0").manual_seed(CONFIG["seed"]+20000)
    mlp_jacobian=np.lib.format.open_memmap(stage.path/"mlp_full_output_jacobian.npy",mode="w+",dtype=np.float32,
        shape=(len(selected)*CONFIG["probe_groups"]*CONFIG["probes_per_group"]*4,stage.model.config.intermediate_size))
    jacobian_row=0
    for number,case in enumerate(selected):
        projected={};handles=[];attention=stage.model.model.layers[-1].self_attn
        for kind in ("q","k","v"):
            handles.append(getattr(attention,kind+"_proj").register_forward_hook(lambda m,a,o,key=kind:projected.__setitem__(key,o)))
        handles.append(stage.model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(
            lambda m,a:projected.__setitem__("mlp",a[0])))
        x=stage.x(case).requires_grad_(True)
        try:hidden=forward_block(stage,x)
        finally:
            for handle in handles:handle.remove()
        values=projected["v"][0].detach().float().reshape(-1,kv,native_dim).transpose(0,1)
        value_moments.add_(values.transpose(1,2)@values/values.shape[1])
        value_means.add_(values.mean(1))
        output=stage.model.model.norm(hidden[:,case["loss_positions"]])[0].float()
        vectors=project_sign_probe_groups(stage.model.lm_head.weight,len(case["loss_positions"]),
                    CONFIG["probes_per_group"],generator,groups=CONFIG["probe_groups"])
        for group in range(CONFIG["probe_groups"]):
            for probe in range(CONFIG["probes_per_group"]):
                last=group==CONFIG["probe_groups"]-1 and probe==CONFIG["probes_per_group"]-1
                scalar=(output*vectors[group,probe]).sum()
                gx,gq,gk,gm=torch.autograd.grad(scalar,(x,projected["q"],projected["k"],projected["mlp"]),retain_graph=not last)
                rows=gm[0,case["loss_positions"]].float().detach()
                if not torch.isfinite(rows).all():raise FloatingPointError("Nonfinite MLP full-output derivative")
                mlp_jacobian[jacobian_row:jacobian_row+len(rows)]=rows.cpu().numpy();jacobian_row+=len(rows)
                gradient=gx[0].float()
                if not torch.isfinite(gradient).all():raise FloatingPointError("Nonfinite full-output derivative")
                grams[group].addmm_(gradient.T,gradient,alpha=1/CONFIG["probes_per_group"])
                q=(projected["q"].detach().float()*gq.float())[0].reshape(-1,kv,group_size,native_dim).sum((0,2))
                k=(projected["k"].detach().float()*gk.float())[0].reshape(-1,kv,native_dim).sum(0)
                mask_gradient=q[:,:native_dim//2]+q[:,native_dim//2:]+k[:,:native_dim//2]+k[:,native_dim//2:]
                pair_scores[group].add_(mask_gradient.square()/CONFIG["probes_per_group"])
                flat=mask_gradient.flatten();mask_grams[group].add_(flat[:,None]*flat[None,:]/CONFIG["probes_per_group"])
        stage.event("calibration",completed=number+1,total=len(selected));del x,hidden,output,vectors,gradient,scalar
    if jacobian_row!=len(mlp_jacobian):raise ValueError("Incomplete MLP derivative bank")
    mlp_jacobian.flush()
    write(stage.path/"mlp_jacobian_metadata.json",{"rows":jacobian_row,"precision":"FP32 storage of BF16 backward values",
        "cases":[{"identity":c["identity"],"family":c["family"],"positions":c["loss_positions"]} for c in selected],
        "row_order":"case, probe group, probe, output position","systems_only":stage.plan["systems_only"]})
    grams/=len(selected);combined=grams.mean(0)
    if not torch.isfinite(grams).all():raise FloatingPointError("Nonfinite Gramian")
    eigenvalues,basis=torch.linalg.eigh(combined);basis=basis.flip(1).contiguous();eigenvalues=eigenvalues.flip(0)
    orthogonality=float((basis.T@basis-torch.eye(width,device=basis.device)).abs().max())
    if not torch.isfinite(eigenvalues).all() or not torch.isfinite(basis).all() or orthogonality>1e-4:raise ValueError("Invalid eigenspace")
    independent=[torch.linalg.eigh(g)[1].flip(1).contiguous() for g in grams]
    stability={}
    for rank in CONFIG["ranks"]:
        u,v=[q[:,:rank] for q in independent]
        stability[str(rank)]={"projector_overlap":float((u.T@v).square().sum()/rank),
            "random_subspace_overlap_reference":rank/width,
            "group0_basis_on_group1_energy":float(subspace_energy_fraction(grams[1],u)),
            "group1_basis_on_group0_energy":float(subspace_energy_fraction(grams[0],v)),
            "combined_energy":float(eigenvalues[:rank].sum()/eigenvalues.sum())}
    pair_scores/=len(selected);mask_grams/=len(selected);scores=pair_scores.mean(0);pairs=scores.topk(32,dim=-1).indices.sort(-1).values
    value_moments/=len(selected);value_means/=len(selected);ve,s=torch.linalg.eigh(value_moments);s=s[:,:,-64:].flip(-1).contiguous()
    removed_mask=torch.ones_like(scores).scatter_(1,pairs,0.).flatten()
    mask_metric=mask_grams.mean(0);removed_quadratic=float(removed_mask@mask_metric@removed_mask)
    reconstructed_mean=(s@(s.transpose(1,2)@value_means[:,:,None]))[:,:,0]
    mean_loss=(reconstructed_mean-value_means).square().sum(1)/value_means.square().sum(1).clamp_min(1e-30)
    fit={"basis":basis.cpu(),"means":means.cpu(),"position_counts":counts.cpu(),"grams":grams.cpu(),"eigenvalues":eigenvalues.cpu(),
         "rope_pairs":pairs.cpu(),"value_basis":s.cpu(),"pair_scores":pair_scores.cpu(),"value_moments":value_moments.cpu(),
         "pair_mask_grams":mask_grams.cpu(),"value_means":value_means.cpu()}
    save(root/"fit.pt",fit)
    info={"calibration_cases":len(selected),"position_mean_training_cases":512,"unobserved_positions":(counts==0).nonzero().flatten().cpu().tolist(),
          "orthogonality_max":orthogonality,"stability":stability,
          "estimator":"independent full-vocabulary Rademacher VJPs; sum every input token; average over output positions and logits",
          "probe_groups":CONFIG["probe_groups"],"probes_per_group":CONFIG["probes_per_group"],"forward_precision":"BF16 forward/backward rounding; FP32 Gramian accumulation",
          "normalization":normalization_metadata(stage.model.config.vocab_size,4,CONFIG["probes_per_group"]),
          "head_projection":{"head_dim":64,"native_head_dim":128,"rope_pairs":pairs.cpu().tolist(),
             "pair_sensitivity_fraction":(scores.gather(1,pairs).sum(1)/scores.sum(1)).cpu().tolist(),
             "joint_removed_mask_linearized_output_energy":removed_quadratic,
             "joint_removed_mask_energy_over_mask_gramian_trace":removed_quadratic/float(mask_metric.trace()),
             "value_second_moment_fraction":(ve[:,-64:].sum(1)/ve.sum(1)).cpu().tolist(),
             "value_mean_projection_loss_fraction":mean_loss.cpu().tolist(),
             "scope":"paired RoPE selector from full-output derivative of shared Q/K pair masks; V basis from uncentered second moment"}}
    write(root/"fit.json",info);return stage.finish(["fit.pt","fit.json"])


@torch.no_grad()
def relations(root):
    """Learn graph normal forms from the shared original-model task benchmark."""
    checked(root,"calibration");stage=Stage(root,"relations")
    fit=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    cache=Path(stage.plan["cache_root"]);meta=json.loads((cache/"items.json").read_text())
    training=read_bits(np.load(cache/"mlp_calibration_35.npy",mmap_mode="r"),slice(None)).float()
    native_dev=read_bits(np.load(cache/"mlp_development_35.npy",mmap_mode="r"),slice(None)).float()
    jac=torch.from_numpy(np.array(np.load(root/"calibration/mlp_full_output_jacobian.npy",mmap_mode="r"))).cuda()
    block=stage.install(fit,1536,head_dim=64);block.set_capture_mlp(True)
    compiled=[]
    for index in meta["development_indices"]:
        case=stage.items[index];x=stage.x(case)
        with torch.autocast("cuda",dtype=torch.bfloat16):forward_block(stage,x)
        sampled=torch.linspace(0,x.shape[1]-1,16,device=x.device).long()
        compiled.append(block.last_mlp_hidden[0,sampled].detach().float())
    block.set_capture_mlp(False);compiled_dev=torch.cat(compiled)
    if training.shape[0]!=len(meta["calibration_families"]) or compiled_dev.shape!=native_dev.shape:
        raise ValueError("Task benchmark activation alignment failed")
    stage.event("relation_fit",training_tokens=len(training),development_tokens=len(native_dev),jacobian_rows=len(jac))
    result=fit_benchmark_groebner(training,meta["calibration_families"],jac,block.mlp_output.detach().float(),
        native_dev,meta["development_families"],compressed_dev_activations=compiled_dev,
        anchor_widths=tuple(CONFIG["groebner_anchor_widths"]),remove_count=CONFIG["groebner_removed_channels"],
        seed=CONFIG["seed"],ridge=CONFIG["groebner_ridge"],eligibility_nrmse=CONFIG["groebner_eligibility_nrmse"],
        svd_rank=CONFIG["groebner_svd_rank"],oversample=CONFIG["groebner_oversample"],
        power_iterations=CONFIG["groebner_power_iterations"],chunk_rows=CONFIG["groebner_chunk_rows"])
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    selected=result["fits"][str(result["selected_anchor_width"])]
    save(root/"relation.pt",cpu(selected));save(stage.path/"all_relation_fits.pt",cpu(result["fits"]))
    report={k:v for k,v in result.items() if k!="fits"}
    report.update(training_families=dict(Counter(meta["calibration_families"])),
        development_families=dict(Counter(meta["development_families"])),
        source="original-model task-benchmark MLP activations and full-output task Jacobian",
        selection="minimum compiled-interface development projected-output relative reconstruction error",
        systems_only=stage.plan["systems_only"])
    write(root/"relations.json",report);stage.event("complete",selected_anchor_width=result["selected_anchor_width"])
    return stage.finish(["relation.pt","relations.json"])


def shakedown(root):
    checked(root,"calibration");checked(root,"relations")
    stage=Stage(root,"shakedown");fit=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    case=max((c for c in stage.items if c["split"]=="train"),key=lambda c:len(c["input_ids"]))
    index=stage.indices[case["identity"]];teacher=read_bits(stage.targets,index)
    with torch.no_grad():native=cached_logits(stage,case)
    if not torch.equal(native,teacher.float()):raise ValueError("Cached prefix does not reproduce native teacher logits")
    checks=[]
    for arm,spec in ARMS.items():
        rank=spec["rank"];head_dim=spec["head_dim"]
        block=stage.install(fit,rank,True,head_dim,spec.get("groebner",False));parameters=[p for p in stage.model.parameters() if p.requires_grad]
        if len(parameters)!=len([p for p in block.parameters() if p.requires_grad])+1:raise ValueError("Training escaped block and final norm")
        optimizer=torch.optim.AdamW(parameters,lr=CONFIG["learning_rate"],weight_decay=0.)
        before={k:v.detach().clone() for k,v in block.named_parameters() if v.requires_grad}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):logits=cached_logits(stage,case)
            loss,kd,ce=objective(logits,teacher,case);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if any(p.grad is None for p in parameters):raise ValueError("Missing compact gradient")
            optimizer.step();stage.event("backward",arm=arm,step=step,loss=float(loss),gradient_norm=float(norm))
        if not any(not torch.equal(before[k],p) for k,p in block.named_parameters() if k in before):raise ValueError("No optimizer update")
        optimizer.zero_grad(set_to_none=True);del optimizer,parameters,before,loss,logits
        block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16);stage.model.model.norm.to(dtype=torch.bfloat16).requires_grad_(False)
        packed=checkpoint(stage,rank,head_dim);save(stage.path/f"{arm}.pt",packed)
        with torch.no_grad():expected=cached_logits(stage,case)
        del packed,block
        block=stage.install(fit,rank,False,head_dim,spec.get("groebner",False));block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
        loaded=torch.load(stage.path/f"{arm}.pt",map_location="cpu",weights_only=True)
        block.load_state_dict(loaded["block"],strict=True);stage.model.model.norm.load_state_dict(loaded["norm"])
        with torch.no_grad():actual=cached_logits(stage,case)
        if not torch.equal(expected,actual):raise ValueError("Checkpoint reload changed logits")
        checks.append({"arm":arm,"checkpoint_logits_exact":True,"trained_steps":2});del loaded
    decoder=graph_decoder(stage.model);capture=decoder.capture()
    ids=torch.tensor([stage.tasks["development"][0]["input_ids"]],device="cuda:0")
    with torch.no_grad():a=decoder.generate(ids,False);b=decoder.generate(ids,True)
    if not torch.equal(a,b):raise ValueError("Static/graph decode mismatch")
    write(stage.path/"checks.json",{"teacher_cache_exact":True,"ranks":checks,"graph_tokens_exact":True,"graph_capture_seconds":capture})
    return stage.finish(systems_only=True)


def controls(root):
    checked(root,"calibration");checked(root,"relations")
    stage=Stage(root,"controls");fit=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    native,rows=development(stage,"native");values={"native":native,"native_tensor_bytes":model_bytes(stage.model)}
    block=stage.install(fit,2048);block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
    folded,other=development(stage,"full_rank");cmp=comparison(rows,other)
    values.update(full_rank=folded,full_rank_comparison=cmp)
    if cmp["agreement"]<CONFIG["full_rank_minimum_dev_agreement"] or folded["kl"]>CONFIG["full_rank_maximum_kl"]:
        write(stage.path/"invalid_control.json",values);raise ValueError("Full-rank control failed")
    for arm,spec in ARMS.items():
        block=stage.install(fit,**spec);block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
        result,_=development(stage,f"initialized_{arm}");values[arm]={**result,"tensor_bytes":model_bytes(stage.model)}
    write(root/"controls.json",values);return stage.finish(["controls.json"])


def train(root,arm):
    spec=ARMS[arm];rank=spec["rank"];head_dim=spec["head_dim"]
    checked(root,"controls");stage=Stage(root,f"train_{arm}");fit=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    block=stage.install(fit,trainable=True,**spec);parameters=[p for p in stage.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
    train=[i for i,c in enumerate(stage.items) if c["split"]=="train"]
    initial,_=packed_development(stage,"epoch_00");best=initial["loss"];patience_reference=best;best_epoch=0;stale=0;history=[]
    save(stage.path/"epoch_00.pt",checkpoint(stage,rank,head_dim))
    for epoch in range(1,CONFIG["epochs"]+1):
        order=train.copy();random.Random(CONFIG["seed"]+epoch).shuffle(order);optimizer.zero_grad(set_to_none=True)
        sums=np.zeros(3);started=time.perf_counter()
        for step,index in enumerate(order):
            fraction=((epoch-1)*len(order)+step)/(CONFIG["epochs"]*len(order))
            for group in optimizer.param_groups:group["lr"]=CONFIG["learning_rate"]*.5*(1+math.cos(math.pi*fraction))
            case=stage.items[index];teacher=read_bits(stage.targets,index)
            with torch.autocast("cuda",dtype=torch.bfloat16):logits=cached_logits(stage,case)
            loss,kd,ce=objective(logits,teacher,case)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite distillation loss")
            (loss/CONFIG["accumulation"]).backward();sums+=[float(loss.detach()),float(kd.detach()),float(ce.detach())]
            if (step+1)%CONFIG["accumulation"]==0 or step+1==len(order):
                torch.nn.utils.clip_grad_norm_(parameters,CONFIG["gradient_clip"],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            if (step+1)%64==0:stage.event("train",epoch=epoch,cases=step+1,loss=float(loss.detach()))
            del logits,teacher,loss,kd,ce
        dev,_=packed_development(stage,f"epoch_{epoch:02d}")
        row={"epoch":epoch,"train_loss":float(sums[0]/len(order)),"train_kl":float(sums[1]/len(order)),
             "train_ce":float(sums[2]/len(order)),"development":dev,"seconds":time.perf_counter()-started}
        history.append(row);stage.event("epoch",arm=arm,**row)
        if dev["loss"]<best:
            best=dev["loss"];best_epoch=epoch;save(stage.path/f"epoch_{epoch:02d}.pt",checkpoint(stage,rank,head_dim))
        if dev["loss"]<patience_reference-CONFIG["minimum_improvement"]:
            patience_reference=dev["loss"];stale=0
        else:stale+=1
        if epoch>=CONFIG["minimum_epochs"] and stale>=CONFIG["patience"]:break
    optimizer.zero_grad(set_to_none=True);del optimizer,parameters
    selected=torch.load(stage.path/f"epoch_{best_epoch:02d}.pt",map_location="cpu",weights_only=True)
    block.load_state_dict(selected["block"],strict=True);stage.model.model.norm.load_state_dict(selected["norm"]);del selected
    block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16);stage.model.model.norm.to(dtype=torch.bfloat16).requires_grad_(False)
    packed,records=development(stage,"selected_packed")
    target=root/f"{arm}.pt";save(target,checkpoint(stage,rank,head_dim))
    expected=json.loads((stage.path/f"epoch_{best_epoch:02d}_development.json").read_text())
    if expected!=records:raise ValueError("Selected packed checkpoint differs from selection-time model")
    if stage.model.lm_head.weight.data_ptr()!=stage.model.model.embed_tokens.weight.data_ptr():raise ValueError("Vocabulary tie lost")
    value={"arm":arm,**spec,"selected_epoch":best_epoch,"completed_epochs":len(history),"training_examples_seen":len(history)*len(train),
           "initial":initial,"history":history,"selected_packed":packed,"checkpoint_sha256":sha(target),
           "checkpoint_bytes":target.stat().st_size,"model_tensor_bytes":model_bytes(stage.model),"scope":"block-plus-final-norm delta checkpoint; native pinned prefix and tied vocabulary required"}
    write(root/f"training_{arm}.json",value);return stage.finish([target.name,f"training_{arm}.json"])


def load_selected(stage,fit,arm):
    spec=ARMS[arm];rank=spec["rank"]
    result=json.loads((stage.root/f"training_{arm}.json").read_text());target=stage.root/f"{arm}.pt"
    if sha(target)!=result["checkpoint_sha256"]:raise ValueError("Checkpoint changed")
    block=stage.install(fit,**spec);block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
    state=torch.load(target,map_location="cpu",weights_only=True)
    block.load_state_dict(state["block"],strict=True);stage.model.model.norm.load_state_dict(state["norm"])
    stage.model.requires_grad_(False)


def graph_decoder(model):
    from .jspace_stacked_graph_decode_v3 import DynamicPrefillGraphDecoder
    from transformers import DynamicCache
    class CompactPrefillDecoder(DynamicPrefillGraphDecoder):
        @torch.inference_mode()
        def prefill(self,ids):
            self.seen.zero_();self.seen.index_fill_(0,ids.reshape(-1),True)
            dynamic=DynamicCache()
            hidden=self.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),
                past_key_values=dynamic,use_cache=True).last_hidden_state[:,-1]
            self.cache.reset()
            for i in range(len(self.cache.key_cache)):
                k,v=dynamic.key_cache[i],dynamic.value_cache[i]
                self.cache.key_cache[i][:,:,:ids.shape[1],:k.shape[-1]].copy_(k)
                self.cache.value_cache[i][:,:,:ids.shape[1],:v.shape[-1]].copy_(v)
            return hidden
    return CompactPrefillDecoder(model)


def runtime(stage,tag):
    cases=[c for f in DATASETS for c in [r for r in stage.tasks["development"] if r["family"]==f][:2]]
    decoder=graph_decoder(stage.model);setup=decoder.capture();rows=[];outputs={}
    warm=torch.tensor([cases[0]["input_ids"]],device="cuda:0")
    with torch.no_grad():
        if not torch.equal(decoder.generate(warm,False),decoder.generate(warm,True)):raise ValueError("Graph mismatch")
        for arm in ("hf","cuda_graph"):
            def generate(ids):
                if arm=="cuda_graph":return decoder.generate(ids,True)
                return stage.model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,use_cache=True,
                    min_new_tokens=32,max_new_tokens=32,pad_token_id=stage.tokenizer.eos_token_id)[0,-32:]
            generate(warm);torch.cuda.synchronize()
            for repeat in range(CONFIG["graph_repeats"] if arm=="cuda_graph" else 1):
                order=list(range(len(cases)));random.Random(CONFIG["seed"]+repeat).shuffle(order)
                for ci in order:
                    stage.guard([0]);case=cases[ci];ids=torch.tensor([case["input_ids"]],device="cuda:0")
                    torch.cuda.synchronize();start=time.perf_counter();tokens=generate(ids);torch.cuda.synchronize()
                    rows.append({"arm":arm,"repeat":repeat,"identity":case["identity"],"tokens":len(tokens),"seconds":time.perf_counter()-start})
                    if repeat==0:outputs[arm+"/"+str(ci)]=tokens.cpu().tolist()
    timings={}
    for arm in ("hf","cuda_graph"):
        subset=[r for r in rows if r["arm"]==arm]
        timings[arm]={"tokens_per_second":sum(r["tokens"] for r in subset)/sum(r["seconds"] for r in subset),"samples":subset}
    del decoder;gc.collect();torch.cuda.empty_cache()
    block_rows=[]
    with torch.no_grad():
        for case in cases:
            cached=stage.items[stage.indices[case["identity"]]];x=stage.x(cached)
            for _ in range(3):forward_block(stage,x)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(20):forward_block(stage,x)
            torch.cuda.synchronize()
            block_rows.append({"identity":case["identity"],"tokens":x.shape[1],"seconds_per_call":(time.perf_counter()-start)/20})
    value={"timings":timings,"graph_capture_seconds":setup,"generated_tokens":outputs,"isolated_block_prefill":block_rows,
           "scope":"same eight dev prompts; batch1; 32 greedy tokens including prefill; graph setup amortized; native repeated at end"}
    write(stage.path/(tag+"_timing.json"),value);return value


def evaluate(root):
    for arm in ARMS:checked(root,f"train_{arm}")
    stage=Stage(root,"evaluation");fit=torch.load(root/"fit.pt",map_location="cpu",weights_only=True);results={}
    for tag in ("native",*ARMS):
        if tag=="native":stage.reset()
        else:load_selected(stage,fit,tag)
        fresh=run_cases(stage,stage.tasks["evaluation"],tag+"_fresh")
        previous=run_cases(stage,stage.tasks["previous_evaluation"],tag+"_previous")
        dev,_=development(stage,tag+"_reload")
        if tag!="native":
            expected=json.loads((root/f"train_{tag}/selected_packed_development.json").read_text())
            actual=json.loads((stage.path/(tag+"_reload_development.json")).read_text())
            if actual!=expected:raise ValueError("Fresh-process reload changed development logits")
        results[tag]={"fresh_accuracy":accuracy(fresh),"previous_accuracy":accuracy(previous),"development":dev,
                      "tensor_bytes":model_bytes(stage.model),"runtime":runtime(stage,tag),"reload_dev_logits_exact":tag!="native"}
        if tag=="native":reference=fresh
        else:results[tag]["comparison_to_native"]=comparison(reference,fresh)
        stage.event("complete",arm=tag,fresh_correct=results[tag]["fresh_accuracy"]["correct"])
    stage.reset();results["native_repeat_runtime"]=runtime(stage,"native_repeat")
    result={"study_id":STUDY_ID,"status":"completed","confirmed":False,"fingerprint":stage.plan["fingerprint"],
            "results":results,"scope":CONFIG["scope"],"fresh_evaluation_count":512,"previous_evaluation_count":256}
    write(root/"result.json",result);return stage.finish(["result.json"])


def reload(root):
    checked(root,"evaluation");stage=Stage(root,"reload");fit=torch.load(root/"fit.pt",map_location="cpu",weights_only=True);checks={}
    for arm in ARMS:
        load_selected(stage,fit,arm);actual=run_cases(stage,stage.tasks["evaluation"],arm+"_fresh")
        expected=json.loads((root/"evaluation"/(arm+"_fresh.json")).read_text())
        if actual!=expected:raise ValueError("Reload changed fresh-test logits")
        checks[arm]={"all_512_records_exact":True,"accuracy":accuracy(actual)}
    write(stage.path/"verification.json",checks);return stage.finish()


def campaign(root):
    """Run dependent stages in fresh processes under the caller's GPU lease."""
    plan=verify(root);wait_ready([0])
    schedule=[("calibration",None),("relations",None)]
    if plan["systems_only"]:schedule += [("shakedown",None)]
    else:schedule += [("controls",None)]+[("train",arm) for arm in ARMS]+[("evaluate",None),("reload",None)]
    # No automatic retries or reuse of partial stages. Completed fingerprints
    # are checked on each invocation; unfinished stages remain explicit failures.
    for name,arm in schedule:
        stage_name=name if arm is None else f"{name}_{arm}"
        if (root/f"{stage_name}_manifest.json").exists():checked(root,stage_name);continue
        command=[sys.executable,"-m","experiments.structure_net.jspace_qwen_affine",name,"--root",str(root)]
        if arm is not None:command += ["--arm",arm]
        print(json.dumps({"kind":"stage_launch","stage":stage_name,"command":command}),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    return {"status":"completed","root":str(root),"systems_only":plan["systems_only"]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","calibration","relations","shakedown","controls","train","evaluate","reload","campaign"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--arm",choices=ARMS);p.add_argument("--systems-only",action="store_true")
    a=p.parse_args()
    if a.stage=="prepare":value=prepare(a.root,a.systems_only)
    elif a.stage=="train":
        if a.arm is None:p.error("train requires --arm")
        value=train(a.root,a.arm)
    else:value={"calibration":calibration,"relations":relations,"shakedown":shakedown,"controls":controls,"evaluate":evaluate,"reload":reload,"campaign":campaign}[a.stage](a.root)
    print(json.dumps(value,allow_nan=False),flush=True)
