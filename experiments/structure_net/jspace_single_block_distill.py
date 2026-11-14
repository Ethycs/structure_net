"""Distill only enriched block0; qualify it through the35 original blocks.

Teacher inputs/outputs are immutable native BF16 boundary caches. The training
objective is equal-case/equal-valid-token squared output error divided by
teacher UPDATE energy, floored using training data only. Native embedding,
readout, RMS norm and layers1..35 never train. No confirmation data is opened.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict,dataclass
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from . import jspace_joint_recovery as joint
from . import jspace_quadratic_recipe as recipe
from .jspace_semantic_recovery import build_enriched_block,ranking_summary
from .jspace_quadratic_evaluate import stream_score

SCHEMA="jspace.single-block-distillation.v1"
PROTOCOL=recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-single-layer-recovery.md"


@dataclass(frozen=True)
class Config:
    epochs:int=12
    microbatch:int=4
    accumulation:int=4
    learning_rate:float=1.2e-4
    gradient_clip:float=1.
    seed:int=27864
    floor_fraction:float=1e-4

    def __post_init__(self):
        joint.JointConfig(epochs=self.epochs,microbatch=self.microbatch,accumulation=self.accumulation,
            learning_rate=self.learning_rate,gradient_clip=self.gradient_clip,seed=self.seed)
        if not 0<self.floor_fraction<1:raise ValueError("Positive fractional training-energy floor required")


class NativePairs:
    def __init__(self,items,input_path,target_path):
        self.items=list(items);self.inputs=self.targets=None
        if (not self.items or any(c["split"] not in ("train","development") for c in self.items)
                or len({c["identity"] for c in self.items})!=len(self.items)):
            raise ValueError("Distinct training/development native pairs required")
        self.training=[i for i,c in enumerate(items) if c["split"]=="train"]
        self.development=[i for i,c in enumerate(items) if c["split"]=="development"]
        if not self.training or not self.development:raise ValueError("Both training and development required")
        cursor=0
        for c in self.items:
            if c["offset"]!=cursor or c["stop"]!=cursor+len(c["input_ids"]) or len(c["input_ids"])<2:
                raise ValueError("Native cache offsets must partition nonempty prompts")
            cursor=c["stop"]
        try:
            self.inputs=np.load(input_path,mmap_mode="r",allow_pickle=False)
            self.targets=np.load(target_path,mmap_mode="r",allow_pickle=False)
            if (self.inputs.dtype!=np.int16 or self.targets.dtype!=np.int16
                    or self.inputs.shape!=self.targets.shape or self.inputs.ndim!=2 or self.inputs.shape[0]!=cursor):
                raise ValueError("Aligned original BF16-bit native banks required")
        except Exception:self.close();raise
        self.normalization=None

    def close(self):
        for value in (self.inputs,self.targets):
            if value is not None and not value._mmap.closed:value._mmap.close()

    def batch(self,indices,device):
        cases=[self.items[i] for i in indices];lengths=[len(c["input_ids"]) for c in cases]
        tensors=[]
        for bank in (self.inputs,self.targets):
            values=np.zeros((len(indices),max(lengths),bank.shape[1]),dtype=np.int16)
            for row,c in enumerate(cases):values[row,:lengths[row]]=bank[c["offset"]:c["stop"]]
            tensor=torch.from_numpy(values).view(torch.bfloat16).to(device,torch.float32)
            if not torch.isfinite(tensor).all():raise FloatingPointError("Nonfinite native cache")
            tensors.append(tensor)
        positions=torch.arange(max(lengths),device=device).expand(len(indices),-1)
        valid=positions<torch.tensor(lengths,device=device)[:,None]
        mask=(positions[:,:,None]>=positions[:,None,:])&valid[:,:,None]&valid[:,None,:]
        return dict(indices=indices,hidden=tensors[0],target=tensors[1],positions=positions,valid=valid,mask=mask[:,None])

    @torch.no_grad()
    def fit_floor(self,device,config,guard):
        energies=[]
        for start in range(0,len(self.training),config.microbatch):
            guard();batch=self.batch(self.training[start:start+config.microbatch],device)
            energy=(batch["target"]-batch["hidden"]).square().mean(-1)
            energies.append(energy[batch["valid"]].cpu())
        values=torch.cat(energies);median=float(values.median())
        if median<=0 or not torch.isfinite(values).all():raise ValueError("Degenerate native training update energy")
        self.normalization=dict(train_token_count=len(values),train_case_count=len(self.training),
            median_teacher_update_energy=median,floor=config.floor_fraction*median,floor_fraction=config.floor_fraction,
            denominator="detached mean((native_h1-native_h0)^2) over native channels")
        return self.normalization


def feature_losses(student,hidden,target,valid,*,floor,basis=None):
    if student.shape!=hidden.shape or student.shape!=target.shape or valid.shape!=student.shape[:2] or floor<=0:
        raise ValueError("Aligned native features, mask and positive floor required")
    target,hidden=target.detach().float(),hidden.detach().float()
    squared=(student.float()-target).square().mean(-1)
    energy=(target-hidden).square().mean(-1).clamp_min(floor)
    relative=squared/energy
    def average(value,mask):
        count=mask.sum(-1);cases=count>0
        return ((value*mask).sum(-1)/count.clamp_min(1))[cases].mean() if cases.any() else value.sum()*0
    ordinary=average(squared,valid);loss=average(relative,valid)
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite single-block feature loss")
    positions=torch.arange(valid.shape[1],device=valid.device)[None]
    lengths=valid.sum(-1)
    groups=dict(first=valid&(positions==0),content=valid&(positions>0)&(positions<lengths[:,None]-1),
        answer=valid&(positions==lengths[:,None]-1))
    result=dict(loss=loss,relative_mse=loss,mse=ordinary,pooled_mse=(squared*valid).sum()/valid.sum())
    for name,mask in groups.items():
        result[name+"_relative_mse"]=average(relative,mask)
        result[name+"_mse"]=average(squared,mask)
    if basis is not None:
        with torch.no_grad():
            update=target-hidden
            outside=update-(update@basis)@basis.T
            floor_per_token=outside.square().mean(-1)/energy
            result["irreducible_outside_relative_mse"]=average(floor_per_token,valid)
            result["retained_relative_mse"]=loss.detach()-result["irreducible_outside_relative_mse"]
    return result


class Trainer:
    def __init__(self,block,rotary,data,quality,*,config=Config(),guard=lambda:None):
        self.block,self.rotary,self.data,self.quality,self.config,self.guard=block,rotary,data,quality,config,guard
        self.device=next(block.parameters()).device
        self.parameters=list(block.parameters())
        if any(p.dtype!=torch.float32 or not p.requires_grad for p in self.parameters):
            raise ValueError("Only the FP32 compressed block parameters may train")
        self.optimizer=torch.optim.AdamW(self.parameters,lr=config.learning_rate,weight_decay=0.,foreach=False)
        self.updates=0
        if data.normalization is None:data.fit_floor(self.device,config,guard)

    def forward(self,batch):
        return self.block(batch["hidden"],attention_mask=batch["mask"],position_ids=batch["positions"],
            position_embeddings=self.rotary(batch["hidden"],batch["positions"]))

    def losses(self,batch,*,diagnostics=False):
        return feature_losses(self.forward(batch),batch["hidden"],batch["target"],batch["valid"],floor=self.data.normalization["floor"],
            basis=self.block.basis if diagnostics else None)

    def train_window(self,indices):
        if not indices or any(i not in self.data.training for i in indices):raise ValueError("Training cases only")
        self.guard();self.block.train();self.optimizer.zero_grad(set_to_none=True);totals={}
        for start in range(0,len(indices),self.config.microbatch):
            self.guard();chosen=indices[start:start+self.config.microbatch]
            values=self.losses(self.data.batch(chosen,self.device));weight=len(chosen)/len(indices)
            (values["loss"]*weight).backward()
            for k,v in values.items():totals[k]=totals.get(k,0.)+float(v.detach())*weight
        if any(p.grad is None for p in self.parameters):raise RuntimeError("A compressed block parameter has no gradient")
        norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,self.config.gradient_clip,error_if_nonfinite=True,foreach=False))
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True);self.updates+=1
        return dict(**totals,update=self.updates,cases=len(indices),gradient_norm_before_clip=norm,
            gradient_parameter_count=len(self.parameters),gradient_scalar_count=sum(p.numel() for p in self.parameters))

    @torch.no_grad()
    def development(self):
        self.guard();self.block.eval();self.optimizer.zero_grad(set_to_none=True);totals={}
        for start in range(0,len(self.data.development),self.config.microbatch):
            self.guard();indices=self.data.development[start:start+self.config.microbatch]
            values=self.losses(self.data.batch(indices,self.device),diagnostics=True)
            for key,value in values.items():totals[key]=totals.get(key,0.)+float(value)*len(indices)
        metrics={k:v/len(self.data.development) for k,v in totals.items()}
        # Production quality uses a detached snapshot. Keep this defensive
        # restoration for injectable scorers without changing parameter IDs.
        try:quality=self.quality(self.block)
        finally:
            for p in self.parameters:p.requires_grad_(True)
        records=quality["records"];metrics.update(ranking_summary(records))
        metrics["selection"]="highest_dev_task_accuracy_then_lowest_teacher_update_relative_mse"
        return metrics,records

    def fit(self,directory,*,event=lambda *_a,**_k:None,native_correct=50,minimum_correct=47,expected_initial_correct=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        history=[];best=None;best_epoch=None;best_records=None
        effective=self.config.microbatch*self.config.accumulation
        for epoch in range(self.config.epochs+1):
            if epoch:
                order=list(self.data.training);random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0,len(order),effective):event("update",epoch=epoch,**self.train_window(order[start:start+effective]))
            metrics,records=self.development()
            if epoch==0 and expected_initial_correct is not None and metrics["correct"]!=expected_initial_correct:
                raise ValueError("Untrained enriched block0 baseline did not reproduce the declared46/64")
            path=directory/f"epoch_{epoch:02d}.pt"
            recipe.save(path,dict(block=joint._cpu(self.block.state_dict()),epoch=epoch,normalization=self.data.normalization))
            recipe.write(directory/f"epoch_{epoch:02d}_development.json",dict(metrics=metrics,records=records))
            history.append(dict(epoch=epoch,metrics=metrics,checkpoint=path.name,checkpoint_sha256=recipe.sha(path),
                development_sha256=recipe.sha(directory/f"epoch_{epoch:02d}_development.json")))
            if best is None or (metrics["correct"],-metrics["relative_mse"])>(best["correct"],-best["relative_mse"]):
                best,best_epoch,best_records=metrics,epoch,records
            event("development",epoch=epoch,**metrics)
        selected=directory/f"epoch_{best_epoch:02d}.pt"
        self.block.load_state_dict(recipe.load(selected)["block"],strict=True)
        metrics,records=self.development()
        if metrics!=best or records!=best_records:raise RuntimeError("Selected single-block reload changed feature/task results")
        result=dict(status="completed",systems_only=False,selected_epoch=best_epoch,selected_checkpoint=str(selected),
            selected=best,history=history,updates=self.updates,strict_reload_exact=True,
            native_correct=native_correct,minimum_correct=minimum_correct,quality_pass=best["correct"]>=minimum_correct,
            normalization=self.data.normalization,scope="Development-only single block0; remaining35 native, no confirmation or speed claim")
        recipe.write(directory/"result.json",result);return result


def make_quality(source,items,directory,guard,*,device="cuda:0"):
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    parent=recipe.read(Path(source)/"plan.json");cases=[c for c in items if c["split"]=="development"]
    embedding,norm,_=load_readout_norm(parent["model"]["path"],device="cpu",dtype=torch.bfloat16)
    embedding,norm=embedding.float(),norm.float();rotary=recipe.rotary_module(parent,device);calls=0
    def evaluate(block):
        nonlocal calls
        # Release the copied snapshot after layer0; the live trainer/AdamW
        # remains separate and untouched while the native suffix streams.
        snapshots=[deepcopy(block).pack(dtype=torch.float32,buffer_dtype=torch.float32)]
        def loader(layer):
            guard()
            return snapshots.pop() if layer==0 else load_native_block(parent["model"]["path"],layer,device=device,dtype=torch.bfloat16).float()
        calls+=1
        result=stream_score(cases,input_embedding=embedding,block_loader=loader,final_readout=(norm,embedding),
            label_ids=parent["label_ids"],scratch_dir=Path(directory)/f"scratch_{calls:02d}",rotary=rotary,
            layer_count=36,device=device,guard=guard,microbatch=4)
        result["metrics"]["max_live_decoder_blocks"]=2
        result["metrics"]["resident_training_optimizer_included_in_peak"]=True
        return result
    return evaluate


def prepare(root,source,basis_root,native_baseline,*,config=Config()):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Hide CUDA for CPU preparation")
    root,source,basis_root,native_baseline=map(lambda p:Path(p).resolve(),(root,source,basis_root,native_baseline))
    if root.exists():raise FileExistsError(root)
    parent=recipe.verify(source);items=recipe.read(source/"items.json")
    if (sum(c["split"]=="train" for c in items["items"])!=512 or sum(c["split"]=="development" for c in items["items"])!=64
            or any(len(c["input_ids"])>384 for c in items["items"])):
        raise ValueError("Original512train/64development population within384 tokens required")
    fitplan=recipe.read(basis_root/"plan.json");manifest=recipe.read(basis_root/"basis/manifest.json")
    if (fitplan.get("schema") not in ("jspace.all-layer-enriched-basis.v1","jspace.first-block-enriched-basis.v1")
            or fitplan["fingerprint"]!=recipe.digest({k:v for k,v in fitplan.items() if k!="fingerprint"})
            or fitplan["source_fingerprint"]!=parent["fingerprint"] or manifest["status"]!="completed"
            or manifest["fingerprint"]!=fitplan["fingerprint"]):raise ValueError("Completed matching enriched geometry required")
    train_ids={c["identity"] for c in items["items"] if c["split"]=="train"}
    fit_ids,check_ids=set(fitplan["fit_training_identities"]),set(fitplan["internal_training_check_identities"])
    if not fit_ids or not check_ids or fit_ids&check_ids or not ((fit_ids|check_ids)<=train_ids):
        raise ValueError("Enriched geometry fitting/check identities must be disjoint and train-only")
    for path,digest in fitplan["source_sha256"].items():
        if recipe.sha(path)!=digest or recipe.sha(basis_root/"producer"/Path(path).relative_to(recipe.ROOT))!=digest:
            raise ValueError("Enrichment producer changed")
    baseline=recipe.read(native_baseline);cases=[c for c in items["items"] if c["split"]=="development"]
    baseline_plan=recipe.read(native_baseline.parent/"plan.json")
    baseline_manifest=recipe.read(native_baseline.parent/"manifest.json")
    if (baseline_plan["fingerprint"]!=recipe.digest({k:v for k,v in baseline_plan.items() if k!="fingerprint"})
            or baseline_plan.get("variant")!="full_rank_full_head" or baseline_plan.get("dtype")!="float32"
            or baseline_plan.get("source_fingerprint")!=parent["fingerprint"]
            or baseline_manifest.get("status")!="completed" or baseline_manifest["fingerprint"]!=baseline_plan["fingerprint"]
            or baseline.get("fingerprint")!=baseline_plan["fingerprint"]
            or baseline_manifest["artifacts"]["result.json"]!=recipe.sha(native_baseline)):
        raise ValueError("Pinned matched full-rank/full-head native-equivalent control required")
    records=baseline["records"]
    if len(cases)!=64 or len(records)!=64 or {r["identity"] for r in records}!={c["identity"] for c in cases}:
        raise ValueError("Matched original64-case native baseline required")
    truth={c["identity"]:(c["answer"],c["family"]) for c in cases}
    if any((r["answer"],r["family"])!=truth[r["identity"]] for r in records):raise ValueError("Native baseline labels differ")
    correct=sum(r["prediction"]==r["answer"] for r in records)
    if correct!=50:raise ValueError("Declared original native baseline is50/64")
    fitpath=basis_root/"basis/block_00/fit.pt";cache=Path(parent["teacher_root"])/"capture"
    inputs={str(fitpath):manifest["artifacts"]["basis/block_00/fit.pt"]}
    for path in (cache/"boundary_00.npy",cache/"boundary_01.npy"):
        inputs[str(path)]=parent["input_sha256"][str(path)]
    for path,expected in inputs.items():
        if recipe.sha(path)!=expected:raise ValueError("Changed consumed native pair/geometry")
    for path in (native_baseline,native_baseline.parent/"plan.json",native_baseline.parent/"manifest.json",
            source/"plan.json",source/"items.json",basis_root/"plan.json",basis_root/"basis/manifest.json"):
        inputs[str(path)]=recipe.sha(path)
    producers=[Path(__file__),Path(__file__).with_name("jspace_semantic_recovery.py"),Path(__file__).with_name("jspace_semantic_loss.py"),Path(__file__).with_name("jspace_joint_recovery.py"),
        Path(__file__).with_name("jspace_residual_carry.py"),Path(__file__).with_name("jspace_quadratic_evaluate.py"),
        recipe.ROOT/"tests/structure_net/test_jspace_single_block_distill.py",PROTOCOL]
    plan=dict(schema=SCHEMA,source=str(source),source_fingerprint=parent["fingerprint"],basis_root=str(basis_root),
        basis_fingerprint=fitplan["fingerprint"],fit_path=str(fitpath),input_path=str(cache/"boundary_00.npy"),target_path=str(cache/"boundary_01.npy"),
        config=asdict(config),items_digest=recipe.digest(items),native_baseline=str(native_baseline),native_correct=50,minimum_correct=47,
        native_baseline_scope="Measured native-equivalent full-rank/full-head block0 +35 original blocks; not a separately rerun raw-native model",
        expected_initial_correct=46,protocol=str(PROTOCOL),
        trained_layers=[0],native_suffix=list(range(1,36)),rank=1024,head_dim=128,intermediate_size=11008,
        initializer="native BF16-rounded weights promoted to FP32, enriched basis; no previous recovery checkpoint",
        objective="equal-case mean of equal-valid-token output MSE / detached teacher UPDATE energy; floor1e-4 training median updateenergy",
        dtype="FP32 weights, forward, gradients and GPU AdamW; immutable native BF16 cached pairs",
        qualification="Detached FP32-packed block0 snapshot, original35-layer FP32 suffix; training parameters never packed",
        selection="actual native-suffix development accuracy first, then relative feature MSE",confirmation_used=False,
        input_sha256=inputs,input_identity={p:recipe.file_identity(p) for p in inputs},source_sha256={str(p):recipe.sha(p) for p in producers})
    plan["fingerprint"]=recipe.digest(plan);root.mkdir(parents=True)
    for path in producers:
        target=root/"producer"/path.relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    recipe.write(root/"plan.json",plan);recipe.write(root/"items.json",items)
    return dict(status="prepared",fingerprint=plan["fingerprint"])


def verify(root):
    root=Path(root);plan=recipe.read(root/"plan.json")
    if plan["schema"]!=SCHEMA or plan["fingerprint"]!=recipe.digest({k:v for k,v in plan.items() if k!="fingerprint"}):raise ValueError("Changed single-block plan")
    parent=recipe.verify(plan["source"])
    if parent["fingerprint"]!=plan["source_fingerprint"] or recipe.digest(recipe.read(root/"items.json"))!=plan["items_digest"]:
        raise ValueError("Changed teacher or paired cases")
    for path,digest in plan["source_sha256"].items():
        if recipe.sha(path)!=digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT))!=digest:raise ValueError("Changed single-block producer")
    for path,identity in plan["input_identity"].items():
        if recipe.file_identity(path)!=identity:raise ValueError("Changed frozen native inputs")
    return plan


def run(root,stage):
    if stage not in ("pilot","train"):raise ValueError("Only pilot/train execute GPU work")
    root=Path(root).resolve();plan=verify(root);directory=root/stage
    if directory.exists():raise FileExistsError(directory)
    if stage=="train":joint.verify_pilot(root,plan["fingerprint"])
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block
    started=time.monotonic();lease=None;data=None;cuda_started=False
    try:
        lease=wait_ready([0]);allocation=dict(lease["allocation"])
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=allocation:raise RuntimeError("Single-block reservation changed")
            joint.check_reserved_peak(allocation,torch.cuda.max_memory_reserved())
        torch.set_num_threads(4);torch.manual_seed(plan["config"]["seed"]);torch.cuda.init();cuda_started=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats();config=Config(**plan["config"])
        parent=recipe.read(Path(plan["source"])/"plan.json")
        native=load_native_block(parent["model"]["path"],0,device="cuda:0",dtype=torch.bfloat16)
        block=build_enriched_block(native,recipe.load(plan["fit_path"]),parameter_dtype=torch.float32);del native
        if (block.width,block.rank,block.head_dim,block.intermediate_size)!=(2048,1024,128,11008):raise ValueError("Runtime architecture differs from frozen plan")
        data=NativePairs(recipe.read(root/"items.json")["items"],plan["input_path"],plan["target_path"])
        # Pilot includes native-suffix evaluation with optimizer states still
        # resident, so its measured peak covers BOTH training and qualification.
        quality=make_quality(plan["source"],data.items,directory,guard)
        trainer=Trainer(block,recipe.rotary_module(parent,"cuda:0"),data,quality,config=config,guard=guard)
        if stage=="pilot":
            directory.mkdir(exist_ok=False)
            initial,initial_records=trainer.development()
            recipe.write(directory/"initial_development.json",dict(metrics=initial,records=initial_records))
            if initial["correct"]!=plan["expected_initial_correct"]:
                raise ValueError("Untrained enriched block0 baseline failed46/64 reproduction before pilot update")
            order=sorted(data.training,key=lambda i:len(data.items[i]["input_ids"]),reverse=True)
            update=trainer.train_window(order[:config.microbatch*config.accumulation])
            feature,records=trainer.development()
            result=dict(status="completed",systems_only=True,updates=1,update=update,development=feature,initial=initial,
                longest_context=max(len(data.items[i]["input_ids"]) for i in order),normalization=data.normalization,
                scope="Systems-only discarded one-update model; pilot weights are not training initialization")
            recipe.write(directory/"result.json",result)
        else:result=trainer.fit(directory,event=lambda kind,**values:recipe.event(root,kind,stage=stage,**values),
            native_correct=plan["native_correct"],minimum_correct=plan["minimum_correct"],expected_initial_correct=plan["expected_initial_correct"])
        guard()
        metrics=dict(status="completed",systems_only=stage=="pilot",fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"),lease_id=lease["id"],allocation=allocation,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic()-started,reserved_budget_bytes=joint.check_reserved_peak(allocation,torch.cuda.max_memory_reserved()))
        recipe.write(directory/"manifest.json",metrics);return dict(**result,execution=metrics)
    except Exception as error:
        directory.mkdir(parents=True,exist_ok=True)
        recipe.write(directory/"failure.json",dict(status="failed",fingerprint=plan["fingerprint"],stage=stage,
            error_type=type(error).__name__,error=str(error),lease_id=None if lease is None else lease["id"],
            allocation=None if lease is None else lease["allocation"],peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise
    finally:
        if data is not None:data.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","pilot","train"))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--source",type=Path)
    parser.add_argument("--basis-root",type=Path);parser.add_argument("--native-baseline",type=Path)
    args=parser.parse_args()
    if args.stage=="prepare":
        if any(v is None for v in (args.source,args.basis_root,args.native_baseline)):parser.error("prepare requires source, basis-root and native-baseline")
        value=prepare(args.root,args.source,args.basis_root,args.native_baseline)
    else:value=run(args.root,args.stage)
    print(json.dumps(value,allow_nan=False),flush=True)


if __name__=="__main__":main()
