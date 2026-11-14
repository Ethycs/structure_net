"""Matched boundary-distillation comparison after task-benchmark compression."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_qwen_affine as base
from .jspace_bench_data import DATASETS, sha, write
from .jspace_bench_tensor import accuracy, comparison, model_bytes, run_cases
from .jspace_full_capture import read_bits
from .jspace_full_lease import wait_ready
from .jspace_layer_distill import projected_boundary_mse, rms_readout_jacobian_loss, gradient_statistics, coefficient_from_stats
from .jspace_qwen_affine_block import AffineQwenBlock

ROOT=base.ROOT
STUDY_ID="JSPACE-QWEN-LAYERKD-01"
PARENT=ROOT/"data/experiments/jspace_qwen_affine/20261113_v2"
DEFAULT_ROOT=ROOT/"data/experiments/jspace_qwen_layerkd/20261113_v1"
SHAKEDOWN_ROOT=ROOT/"data/experiments/jspace_qwen_layerkd/20261113_shakedown_v1"
PROTOCOL=ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-layerwise-distillation-plan.md"
ARMS=("endpoint","euclidean","jacobian")
CONFIG={"seed":861,"layer":35,"rank":1536,"head_dim":64,
        "groebner":True,"calibration_per_task":8,"target_gradient_ratio":.1,
        "jacobian_metric":"analytic original RMSNorm differential and full readout Gramian",
        "readout_chunk_rows":4096,
        "epochs":12,"accumulation":4,"learning_rate":3e-5,"weight_decay":0.,"gradient_clip":1.,
        "temperature":2.,"kl_weight":.7,"ce_weight":.3,"early_stopping":False,
        "local_positions":"same four cached output positions in both auxiliary arms",
        "normalization":"fixed mean initial auxiliary loss over 32 training calibration cases",
        "coefficient":"0.1 times RMS endpoint gradient / RMS raw auxiliary gradient; block parameters only",
        "selection":"minimum packed development endpoint KL+CE, including epoch zero",
        "arms":list(ARMS),"scope":"exploratory single-final-block layer-distillation; no multi-block stability claim"}


def source_paths():
    return sorted(set(base.source_paths()+[Path(__file__),PROTOCOL,
        Path(__file__).with_name("jspace_layer_distill.py"),
        ROOT/"tests/structure_net/test_jspace_layer_distill.py",
        ROOT/"tests/structure_net/test_jspace_qwen_layerkd_protocol.py"]))


def prepare(root,systems_only=False):
    base.cpu_only()
    if root.exists():raise FileExistsError(root)
    parent=base.verify(PARENT,inputs=True)
    for name in ("calibration","relations","controls"):base.checked(PARENT,name)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(parent["model"]["path"],local_files_only=True,trust_remote_code=False)
    prior=json.loads((PARENT/"tasks.json").read_text())
    fresh,exclusions=base.fresh_cases(tokenizer,prior)
    tasks={"splits":{"train":prior["splits"]["train"],"development":prior["splits"]["development"],
                     "evaluation":fresh},"fresh_exclusions":exclusions}
    inputs=dict(parent["input_sha256"])
    for name in ("plan.json","tasks.json","fit.pt","relation.pt","calibration_manifest.json",
                 "relations_manifest.json","controls_manifest.json"):
        inputs[str(PARENT/name)]=sha(PARENT/name)
    plan={"study_id":STUDY_ID,"config":CONFIG,"systems_only":systems_only,"model":parent["model"],
        "parent_root":str(PARENT),"parent_fingerprint":parent["fingerprint"],"cache_root":parent["cache_root"],
        "label_ids":parent["label_ids"],"input_sha256":inputs,"task_digest":base.digest(tasks),
        "created_at":datetime.now(timezone.utc).isoformat(),"source_sha256":{str(p):sha(p) for p in source_paths()}}
    plan["fingerprint"]=base.digest(plan);root.mkdir(parents=True)
    for p in source_paths():
        target=root/"producer"/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ("fit.pt","relation.pt"):shutil.copy2(PARENT/name,root/name)
    write(root/"tasks.json",tasks);write(root/"plan.json",plan)
    return {"root":str(root),"fingerprint":plan["fingerprint"],"systems_only":systems_only,
            "counts":{k:len(v) for k,v in tasks["splits"].items()}}


def verify(root,inputs=False,model=False):
    plan=json.loads((root/"plan.json").read_text())
    if plan["config"]!=CONFIG or plan["study_id"]!=STUDY_ID or plan["fingerprint"]!=base.digest({k:v for k,v in plan.items() if k!="fingerprint"}):
        raise ValueError("Changed layer-distillation protocol")
    for name,expected in plan["source_sha256"].items():
        if sha(name)!=expected or sha(root/"producer"/Path(name).relative_to(ROOT))!=expected:raise ValueError("Changed producer: "+name)
    if base.digest(json.loads((root/"tasks.json").read_text()))!=plan["task_digest"]:raise ValueError("Changed tasks")
    for name in ("fit.pt","relation.pt"):
        if sha(root/name)!=plan["input_sha256"][str(PARENT/name)]:raise ValueError("Changed inherited compression")
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
    for path,expected in manifest["artifact_sha256"].items():
        if sha(root/path)!=expected:raise ValueError("Changed stage artifact: "+path)
    return manifest


def calibration_indices(items):
    return [i for f in DATASETS for i in [j for j,c in enumerate(items) if c["split"]=="train" and c["family"]==f][:CONFIG["calibration_per_task"]]]


def training_order(indices,epoch):
    order=list(indices);random.Random(CONFIG["seed"]+epoch).shuffle(order);return order


class Stage(base.Stage):
    def __init__(self,root,name):
        self.lease=wait_ready([0]);self.guard=wait_ready;self.plan=verify(root,inputs=True,model=True)
        self.root=root;self.path=root/name;self.path.mkdir(exist_ok=False);self.name=name;self.started=time.perf_counter();self.counts=Counter()
        torch.set_num_threads(4);torch.manual_seed(CONFIG["seed"]);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        self.model,self.tokenizer=base.load_small_qwen(self.plan["model"]["path"],device="cuda:0",dtype=torch.bfloat16)
        self.native=copy.deepcopy(self.model.model.layers[-1]).cpu()
        self.native_norm={k:v.detach().cpu().clone() for k,v in self.model.model.norm.state_dict().items()}
        self.tasks=json.loads((root/"tasks.json").read_text())["splits"];self.label_ids=torch.tensor(self.plan["label_ids"],device="cuda:0")
        cache=Path(self.plan["cache_root"]);self.items=json.loads((cache/"items.json").read_text())["items"]
        self.input=np.load(cache/"boundary_35.npy",mmap_mode="r");self.output=np.load(cache/"boundary_36.npy",mmap_mode="r")
        self.targets=np.load(cache/"teacher_logits.npy",mmap_mode="r");self.indices={c["identity"]:i for i,c in enumerate(self.items)}
        self.training=[i for i,c in enumerate(self.items) if c["split"]=="train"]
        self.training_rows={index:row for row,index in enumerate(self.training)}
        self.fit=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
        self.relation=torch.load(root/"relation.pt",map_location="cpu",weights_only=True)
        self.output_metric=None
        write(self.path/"lease.json",{k:self.lease[k] for k in ("id","holder","allocation","state","ready","end")})

    def teacher_boundary(self,case):
        return read_bits(self.output,slice(case["offset"],case["stop"]))[case["loss_positions"]]

    def install(self,trainable=False,initialized=False,packed=False):
        self.reset();source=self.model.model.layers[-1]
        block=AffineQwenBlock(source,self.fit["basis"][:,:CONFIG["rank"]].cuda(),self.fit["means"].cuda(),
            trainable=trainable,parameter_dtype=torch.float32 if trainable else torch.bfloat16,
            head_projection={"rope_pairs":self.fit["rope_pairs"].cuda(),"value_basis":self.fit["value_basis"].cuda()})
        block.compile_polynomial(self.relation,self.relation["removed"]);self.model.model.layers[-1]=block
        if trainable:self.model.model.norm.float().requires_grad_(True)
        if initialized:
            calibration=json.loads((self.root/"calibration.json").read_text())
            if sha(self.root/"initialized.pt")!=calibration["initialized_checkpoint_sha256"]:raise ValueError("Changed initializer")
            state=torch.load(self.root/"initialized.pt",map_location="cpu",weights_only=True)
            block.load_state_dict(state["block"],strict=True);self.model.model.norm.load_state_dict(state["norm"])
        if packed:
            block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
            self.model.model.norm.to(dtype=torch.bfloat16).requires_grad_(False)
        return block

    def metric(self):
        if self.output_metric is None:
            value=torch.load(self.root/"setup/output_metric.pt",map_location="cpu",weights_only=True)
            self.output_metric={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in value.items()}
        return self.output_metric

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


def losses(stage,index,auxiliary=True):
    case=stage.items[index];x=stage.x(case)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        hidden=base.forward_block(stage,x)[0,case["loss_positions"]]
        logits=F.linear(stage.model.model.norm(hidden),stage.model.lm_head.weight).float()
    endpoint,kd,ce=base.objective(logits,read_bits(stage.targets,index),case)
    if not auxiliary:return endpoint,kd,ce,{}
    if case["split"]!="train":raise ValueError("Auxiliary fitting may only consume training cases")
    teacher=stage.teacher_boundary(case);block=stage.model.model.layers[-1];aux={}
    if auxiliary is True or auxiliary=="euclidean":
        anchor=block.anchor[case["loss_positions"]]
        aux["euclidean"],aux["discarded_floor"]=projected_boundary_mse(hidden,teacher,block.basis,anchor)
    if auxiliary is True or auxiliary=="jacobian":
        metric=stage.metric()
        aux["jacobian"]=rms_readout_jacobian_loss(hidden,teacher,metric["gain"],metric["epsilon"],
            metric["readout_gram"],metric["vocabulary_size"])
    return endpoint,kd,ce,aux


def setup(root):
    if not verify(root)["systems_only"]:checked(SHAKEDOWN_ROOT,"shakedown")
    stage=Stage(root,"setup");cal=calibration_indices(stage.items)
    if len(cal)!=CONFIG["calibration_per_task"]*len(DATASETS):raise ValueError("Insufficient balanced calibration")
    longest=max(stage.training,key=lambda i:len(stage.items[i]["input_ids"]))
    width=stage.model.config.hidden_size;gram=torch.zeros(width,width,device="cuda:0")
    readout=stage.model.lm_head.weight
    with torch.no_grad():
        for start in range(0,len(readout),CONFIG["readout_chunk_rows"]):
            chunk=readout[start:start+CONFIG["readout_chunk_rows"]].float();gram.addmm_(chunk.T,chunk)
        gram=(gram+gram.T)*.5
    if not torch.isfinite(gram).all():raise FloatingPointError("Nonfinite readout Gramian")
    metric={"readout_gram":gram.cpu(),"gain":stage.model.model.norm.weight.detach().float().cpu(),
            "epsilon":float(stage.model.model.norm.variance_epsilon),"vocabulary_size":len(readout)}
    base.save(stage.path/"output_metric.pt",metric)
    write(stage.path/"metric_metadata.json",{"shape":list(gram.shape),"dtype":"FP32 Gramian and analytic RMS differential from BF16 teacher weights/states",
        "metric":"sum_position dRMS(e)^T (E^T E) dRMS(e) / (positions * vocabulary_size)",
        "boundary":"original final block output, before original frozen final RMSNorm",
        "scope":"exact real-arithmetic first-order full-logit metric for this positionwise suffix; floating-point approximation in execution",
        "random_probe_approximation":False,"teacher_metric_tensor_bytes":sum(v.numel()*v.element_size() for v in metric.values() if isinstance(v,torch.Tensor))})
    del gram,metric,chunk
    stage.event("metric",vocabulary_size=len(readout),width=width)
    with torch.no_grad():native=base.cached_logits(stage,stage.items[longest])
    if not torch.equal(native,read_bits(stage.targets,longest).float()):raise ValueError("Native cached teacher changed")
    block=stage.install(trainable=True);initial=base.checkpoint(stage,CONFIG["rank"],CONFIG["head_dim"])
    base.save(root/"initialized.pt",initial)
    params=[p for p in block.parameters() if p.requires_grad];stats={arm:[] for arm in ARMS[1:]};raw={arm:[] for arm in ARMS[1:]};floors=[]
    for index in cal:
        endpoint,kd,ce,aux=losses(stage,index)
        for arm in ARMS[1:]:
            stat=gradient_statistics(endpoint,aux[arm],params)
            stats[arm].append({"identity":stage.items[index]["identity"],"family":stage.items[index]["family"],**stat})
            raw[arm].append(float(aux[arm].detach()))
        floors.append(float(aux["discarded_floor"]))
        del endpoint,kd,ce,aux
    coefficients={}
    for arm in ARMS[1:]:
        scale=sum(raw[arm])/len(raw[arm])
        if not math.isfinite(scale) or scale<=1e-20:raise ValueError("Degenerate auxiliary normalizer")
        numeric=[{k:row[k] for k in ("task_grad_norm","local_grad_norm","grad_dot","grad_cosine")} for row in stats[arm]]
        coefficient=coefficient_from_stats(numeric,target_ratio=CONFIG["target_gradient_ratio"])
        if not math.isfinite(coefficient) or coefficient<=0:raise ValueError("Degenerate calibration gradient ratio")
        coefficients[arm]={"raw_coefficient":coefficient,"normalizer":scale,"normalized_coefficient":coefficient*scale,
                           "calibration_gradient_statistics":stats[arm],"initial_raw_losses":raw[arm]}
    for key,value in block.state_dict().items():
        if not torch.equal(value.detach().cpu(),initial["block"][key]):raise ValueError("Coefficient fitting changed initialization")
    if any(p.grad is not None for p in params):raise ValueError("Coefficient fitting wrote optimizer gradients")
    report={"coefficients":coefficients,"calibration_indices":cal,"calibration_families":dict(Counter(stage.items[i]["family"] for i in cal)),
            "mean_discarded_floor":sum(floors)/len(floors),"initialized_checkpoint_sha256":sha(root/"initialized.pt"),
            "teacher_cache_exact":True,"systems_only":stage.plan["systems_only"]}
    write(root/"calibration.json",report)
    return stage.finish(["initialized.pt","calibration.json"])


def total_loss(endpoint,aux,arm,calibration):
    if arm=="endpoint":return endpoint
    if arm not in ARMS:raise ValueError("Unknown arm")
    specification=calibration["coefficients"][arm]
    return endpoint+specification["normalized_coefficient"]*(aux[arm]/specification["normalizer"])


def shakedown(root):
    checked(root,"setup");stage=Stage(root,"shakedown")
    calibration=json.loads((root/"calibration.json").read_text());index=max(stage.training,key=lambda i:len(stage.items[i]["input_ids"]))
    checks=[]
    for arm in ARMS:
        block=stage.install(trainable=True,initialized=True);params=[p for p in stage.model.parameters() if p.requires_grad]
        if len(params)!=len([p for p in block.parameters() if p.requires_grad])+1:raise ValueError("Training escaped block and norm")
        optimizer=torch.optim.AdamW(params,lr=CONFIG["learning_rate"],weight_decay=0.)
        before=block.mlp_output.detach().clone()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);endpoint,kd,ce,aux=losses(stage,index,arm if arm!="endpoint" else False)
            loss=total_loss(endpoint,aux,arm,calibration);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,CONFIG["gradient_clip"],error_if_nonfinite=True)
            if any(p.grad is None for p in params):raise ValueError("Missing trainable gradient")
            optimizer.step();stage.event("backward",arm=arm,step=step,loss=float(loss.detach()),gradient_norm=float(norm))
        if torch.equal(before,block.mlp_output):raise ValueError("No optimizer update")
        optimizer.zero_grad(set_to_none=True);del optimizer,params,before,endpoint,kd,ce,aux,loss
        block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16);stage.model.model.norm.to(dtype=torch.bfloat16).requires_grad_(False)
        state=base.checkpoint(stage,CONFIG["rank"],CONFIG["head_dim"]);base.save(stage.path/f"{arm}.pt",state)
        with torch.no_grad():expected=base.cached_logits(stage,stage.items[index])
        block=stage.install(packed=True);loaded=torch.load(stage.path/f"{arm}.pt",map_location="cpu",weights_only=True)
        block.load_state_dict(loaded["block"],strict=True);stage.model.model.norm.load_state_dict(loaded["norm"])
        with torch.no_grad():actual=base.cached_logits(stage,stage.items[index])
        if not torch.equal(expected,actual):raise ValueError("Packed checkpoint reload changed logits")
        checks.append({"arm":arm,"steps":2,"packed_reload_exact":True});del loaded,state
    decoder=base.graph_decoder(stage.model);seconds=decoder.capture()
    ids=torch.tensor([stage.tasks["development"][0]["input_ids"]],device="cuda:0")
    with torch.no_grad():a=decoder.generate(ids,False);b=decoder.generate(ids,True)
    if not torch.equal(a,b):raise ValueError("CUDA graph decode mismatch")
    write(stage.path/"checks.json",{"arms":checks,"graph_exact":True,"capture_seconds":seconds})
    return stage.finish(systems_only=True)


def train(root,arm):
    if arm not in ARMS:raise ValueError("Unknown arm")
    if verify(root)["systems_only"]:raise ValueError("Systems roots cannot produce training evidence")
    checked(root,"setup");stage=Stage(root,f"train_{arm}");block=stage.install(trainable=True,initialized=True)
    calibration=json.loads((root/"calibration.json").read_text())
    params=[p for p in stage.model.parameters() if p.requires_grad];block_params=[p for p in block.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
    initial,_=base.packed_development(stage,"epoch_00");best=initial["loss"];best_epoch=0;history=[]
    base.save(stage.path/"epoch_00.pt",base.checkpoint(stage,CONFIG["rank"],CONFIG["head_dim"]))
    diagnostic=calibration_indices(stage.items)[0]
    for epoch in range(1,CONFIG["epochs"]+1):
        order=training_order(stage.training,epoch)
        optimizer.zero_grad(set_to_none=True);sums=np.zeros(5);started=time.perf_counter();diagnostics=None
        for step,index in enumerate(order):
            fraction=((epoch-1)*len(order)+step)/(CONFIG["epochs"]*len(order))
            for group in optimizer.param_groups:group["lr"]=CONFIG["learning_rate"]*.5*(1+math.cos(math.pi*fraction))
            endpoint,kd,ce,aux=losses(stage,index,arm if arm!="endpoint" else False)
            loss=total_loss(endpoint,aux,arm,calibration)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
            if index==diagnostic and arm!="endpoint":
                diagnostics=gradient_statistics(endpoint,aux[arm],block_params)
                raw_coefficient=calibration["coefficients"][arm]["raw_coefficient"]
                diagnostics["weighted_local_to_endpoint_norm"]=raw_coefficient*diagnostics["local_grad_norm"]/max(diagnostics["task_grad_norm"],1e-30)
            (loss/CONFIG["accumulation"]).backward()
            local=0. if arm=="endpoint" else float(aux[arm].detach())/calibration["coefficients"][arm]["normalizer"]
            sums += [float(loss.detach()),float(endpoint.detach()),float(kd.detach()),float(ce.detach()),local]
            if (step+1)%CONFIG["accumulation"]==0 or step+1==len(order):
                torch.nn.utils.clip_grad_norm_(params,CONFIG["gradient_clip"],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            if (step+1)%128==0:stage.event("train",arm=arm,epoch=epoch,cases=step+1,loss=float(loss.detach()))
            del endpoint,kd,ce,aux,loss
        dev,_=base.packed_development(stage,f"epoch_{epoch:02d}")
        row={"epoch":epoch,"total_loss":float(sums[0]/len(order)),"endpoint_loss":float(sums[1]/len(order)),
            "kl":float(sums[2]/len(order)),"ce":float(sums[3]/len(order)),"normalized_local":float(sums[4]/len(order)),
            "gradient_diagnostic":diagnostics,"development":dev,"seconds":time.perf_counter()-started}
        history.append(row);stage.event("epoch",arm=arm,**row)
        if dev["loss"]<best:
            best=dev["loss"];best_epoch=epoch;base.save(stage.path/f"epoch_{epoch:02d}.pt",base.checkpoint(stage,CONFIG["rank"],CONFIG["head_dim"]))
    optimizer.zero_grad(set_to_none=True);del optimizer,params,block_params
    state=torch.load(stage.path/f"epoch_{best_epoch:02d}.pt",map_location="cpu",weights_only=True)
    block.load_state_dict(state["block"],strict=True);stage.model.model.norm.load_state_dict(state["norm"]);del state
    block.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16);stage.model.model.norm.to(dtype=torch.bfloat16).requires_grad_(False)
    selected,records=base.development(stage,"selected_packed")
    expected=json.loads((stage.path/f"epoch_{best_epoch:02d}_development.json").read_text())
    if expected!=records:raise ValueError("Selected packed checkpoint changed from validation")
    target=root/f"{arm}.pt";base.save(target,base.checkpoint(stage,CONFIG["rank"],CONFIG["head_dim"]))
    result={"arm":arm,"selected_epoch":best_epoch,"completed_epochs":len(history),"updates":len(history)*len(stage.training)//CONFIG["accumulation"],
        "initialized_checkpoint_sha256":calibration["initialized_checkpoint_sha256"],"initial":initial,"selected_packed":selected,
        "history":history,"checkpoint_sha256":sha(target),"checkpoint_bytes":target.stat().st_size,"tensor_bytes":model_bytes(stage.model)}
    write(root/f"training_{arm}.json",result);return stage.finish([target.name,f"training_{arm}.json"])


def load_selected(stage,arm):
    if arm not in ARMS:raise ValueError("Unknown arm")
    result=json.loads((stage.root/f"training_{arm}.json").read_text())
    if sha(stage.root/f"{arm}.pt")!=result["checkpoint_sha256"]:raise ValueError("Changed selected checkpoint")
    block=stage.install(packed=True);state=torch.load(stage.root/f"{arm}.pt",map_location="cpu",weights_only=True)
    block.load_state_dict(state["block"],strict=True);stage.model.model.norm.load_state_dict(state["norm"]);stage.model.requires_grad_(False)


def evaluate(root):
    for arm in ARMS:checked(root,f"train_{arm}")
    stage=Stage(root,"evaluation");results={};records={}
    for arm in ("native",*ARMS):
        if arm=="native":stage.reset()
        else:load_selected(stage,arm)
        rows=run_cases(stage,stage.tasks["evaluation"],arm+"_fresh");records[arm]=rows
        dev,dev_rows=base.development(stage,arm+"_reload")
        if arm!="native":
            expected=json.loads((root/f"train_{arm}/selected_packed_development.json").read_text())
            if expected!=dev_rows:raise ValueError("Fresh-process reload changed development logits")
        results[arm]={"accuracy":accuracy(rows),"development":dev,"tensor_bytes":model_bytes(stage.model),"runtime":base.runtime(stage,arm)}
        if arm!="native":results[arm]["comparison_to_native"]=comparison(records["native"],rows)
        if arm in ARMS[1:]:results[arm]["comparison_to_endpoint"]=comparison(records["endpoint"],rows)
        stage.event("complete",arm=arm,correct=results[arm]["accuracy"]["correct"])
    byte_counts={results[arm]["tensor_bytes"] for arm in ARMS}
    if len(byte_counts)!=1:raise ValueError("Distillation arms differ in deployed tensor footprint")
    stage.reset();results["native_repeat_runtime"]=base.runtime(stage,"native_repeat")
    result={"study_id":STUDY_ID,"status":"completed","confirmed":False,"fingerprint":stage.plan["fingerprint"],
        "results":results,"scope":CONFIG["scope"],"fresh_evaluation_count":len(stage.tasks["evaluation"])}
    write(root/"result.json",result);return stage.finish(["result.json"])


def reload(root):
    checked(root,"evaluation");stage=Stage(root,"reload");checks={}
    for arm in ARMS:
        load_selected(stage,arm);actual=run_cases(stage,stage.tasks["evaluation"],arm+"_fresh")
        expected=json.loads((root/"evaluation"/f"{arm}_fresh.json").read_text())
        if actual!=expected:raise ValueError("Reload changed held-out records")
        checks[arm]={"records_exact":True,"accuracy":accuracy(actual)}
    write(stage.path/"verification.json",checks);return stage.finish()


def campaign(root):
    plan=verify(root);wait_ready([0]);schedule=[("setup",None)]
    if plan["systems_only"]:schedule += [("shakedown",None)]
    else:schedule += [("train",arm) for arm in ARMS]+[("evaluate",None),("reload",None)]
    for name,arm in schedule:
        stage_name=name if arm is None else f"train_{arm}"
        if (root/f"{stage_name}_manifest.json").exists():checked(root,stage_name);continue
        command=[sys.executable,"-m","experiments.structure_net.jspace_qwen_layerkd",name,"--root",str(root)]
        if arm is not None:command += ["--arm",arm]
        print(json.dumps({"kind":"stage_launch","stage":stage_name,"command":command}),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    return {"status":"completed","root":str(root),"systems_only":plan["systems_only"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","setup","shakedown","train","evaluate","reload","campaign"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT);parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--systems-only",action="store_true");args=parser.parse_args()
    if args.stage=="prepare":value=prepare(args.root,args.systems_only)
    elif args.stage=="train":
        if args.arm is None:parser.error("train requires --arm")
        value=train(args.root,args.arm)
    else:value={"setup":setup,"shakedown":shakedown,"evaluate":evaluate,"reload":reload,"campaign":campaign}[args.stage](args.root)
    print(json.dumps(value,allow_nan=False),flush=True)
