"""Jointly repair a frozen search-selected triple in the actual compact path.

Only those three blocks train. Feature targets come from a new original FP32
teacher pass; endpoint supervision uses the existing matched FP32 logit bank.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import jspace_recovered_joint as base
from . import jspace_recovered_teacher as teacher

recipe,semantic,joint=base.recipe,base.semantic,base.joint
SCHEMA="jspace.layer-group-repair.v1"
PROTOCOL=recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-layer-combination-repair.md"
SCOPE="Development-only joint repair of exactly three search-selected blocks; no confirmation or speed claim"
FEATURE_KEYS=("endpoint_loss","feature_loss","weighted_feature_loss")


@dataclass(frozen=True)
class Config(base.Config):
    epochs:int=6
    seed:int=27866
    feature_weight:float=1.

    def __post_init__(self):
        super().__post_init__()
        if self.feature_weight!=1. or self.epochs!=6:raise ValueError("Frozen6-epoch/unit-feature objective required")


def selected_layers(value):
    if (not isinstance(value,list) or len(value)!=3 or value!=sorted(set(value))
            or any(type(i) is not int or not 0<=i<36 for i in value)):
        raise ValueError("Exactly three sorted distinct absolute layers required")
    return value


def configure_group(model,layers):
    layers=selected_layers(layers);model.requires_grad_(False)
    for index in layers:model.blocks[index].requires_grad_(True)
    model.selected_layers=tuple(layers)
    if model.norm_gain.requires_grad or model.embedding.requires_grad:raise ValueError("Native norm and tiedE must remain fixed")
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if len(names)!=7*len(layers):raise ValueError("Only seven compact parameter tensors per selected block may train")
    return model


class GroupModel(base.PartitionedCarryModel):
    def forward_with_features(self,input_ids,position_ids,attention_mask,loss_positions):
        hidden=F.embedding(input_ids,self.embedding);base_rope=self.rotary(hidden,position_ids)
        contexts={};features={};first=min(self.selected_layers)
        for index,block in enumerate(self.blocks):
            device=next(block.parameters()).device
            if device not in contexts:
                contexts[device]=(position_ids.to(device),attention_mask.to(device),tuple(t.to(device) for t in base_rope))
            ids,mask,rope=contexts[device];hidden=hidden.to(device)
            def apply(value,current=block,p=ids,a=mask,r=rope):
                return current(value,attention_mask=a,position_ids=p,position_embeddings=r)
            if index<first:
                with torch.no_grad():hidden=apply(hidden)
            elif self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                hidden=checkpoint(apply,hidden,use_reentrant=False)
            else:hidden=apply(hidden)
            if index in self.selected_layers:features[index]=hidden
        rows=torch.arange(hidden.shape[0],device=hidden.device)[:,None]
        chosen=hidden[rows,loss_positions.to(hidden.device)].to(self.embedding.device)
        normalized=chosen*torch.rsqrt(chosen.square().mean(-1,keepdim=True)+self.epsilon)
        return F.linear(normalized*self.norm_gain,self.embedding),features

    def forward(self,input_ids,position_ids,attention_mask,loss_positions):
        return self.forward_with_features(input_ids,position_ids,attention_mask,loss_positions)[0]


def build_model(parent,devices,layers,guard):
    original=base.build_model(parent,devices,guard=guard)
    model=GroupModel(original.embedding,list(original.blocks),original.norm_gain,original.epsilon,
        original.rotary,split=24,checkpoint_blocks=True)
    return configure_group(model,layers)


def feature_objective(features,targets,bases,valid,device):
    """Full derivative through output y; detached fixed teacher-only denominator."""
    if set(features)!=set(targets) or set(features)!=set(bases) or not features:
        raise ValueError("Matching selected next-boundary features, targets and bases required")
    losses=[];per_layer={}
    for layer,y in features.items():
        target=targets[layer].detach().to(device=y.device,dtype=y.dtype);u=bases[layer].detach().to(y.device)
        mask=valid.to(y.device)
        if target.shape!=y.shape or mask.shape!=y.shape[:2] or u.shape[0]!=y.shape[-1]:
            raise ValueError("Teacher, valid-token mask and basis shapes differ")
        count=mask.sum(-1)
        if torch.any(count<=0):raise ValueError("Each case needs valid tokens")
        error=(y-target)@u
        teacher_coordinates=target@u
        numerator=(error.square().sum(-1)*mask).sum(-1)/count
        denominator=((teacher_coordinates.square().sum(-1)*mask).sum(-1)/count).clamp_min(1e-8).detach()
        values=(numerator/denominator).to(device)
        if not torch.isfinite(values).all():raise FloatingPointError("Nonfinite normalized feature error")
        losses.append(values);per_layer[str(layer)]=values
    per_case=torch.stack(losses).mean(0)
    return dict(feature_loss=per_case.mean(),per_case_feature_loss=per_case,per_layer_feature_loss=per_layer)


class GroupDataset(teacher.FP32JointDataset):
    def __init__(self,root,plan):
        super().__init__(plan["teacher_root"])
        if recipe.digest(recipe.read(Path(root)/"items.json"))!=plan["items_digest"]:
            self.close();raise ValueError("Repair item identities changed")
        self.feature_banks={}
        try:
            offset=0
            for c in self.items:
                if c["offset"]!=offset or c["stop"]!=offset+len(c["input_ids"]):raise ValueError("Contiguous original-token layout required")
                offset=c["stop"]
            for layer in plan["selected_layers"]:
                bank=np.load(Path(root)/"capture"/f"boundary_{layer+1:02d}.npy",mmap_mode="r",allow_pickle=False)
                self.feature_banks[layer]=bank
                if bank.dtype!=np.float32 or bank.shape!=(offset,2048):raise ValueError("Original FP32 next-boundary bank required")
        except Exception:self.close();raise

    def close(self):
        super().close()
        for bank in getattr(self,"feature_banks",{}).values():
            if not bank._mmap.closed:bank._mmap.close()

    def batch(self,indices,device):
        value=super().batch(indices,device);length=value["input_ids"].shape[1]
        targets={}
        for layer,bank in self.feature_banks.items():
            rows=np.zeros((len(indices),length,2048),dtype=np.float32)
            for row,index in enumerate(indices):
                c=self.items[index];rows[row,:len(c["input_ids"])]=bank[c["offset"]:c["stop"]]
            if not np.isfinite(rows).all():raise FloatingPointError("Nonfinite teacher boundary")
            targets[layer]=torch.from_numpy(rows)
        lengths=torch.tensor([len(self.items[i]["input_ids"]) for i in indices],device=device)
        return dict(**value,feature_targets=targets,valid_tokens=value["position_ids"]<lengths[:,None])


def frozen_fingerprint(model):
    import hashlib
    digest=hashlib.sha256();count=0;size=0
    values=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]+list(model.named_buffers())
    for name,value in values:
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)]).encode())
        flat=value.detach().reshape(-1);step=max(1,32*1024**2//value.element_size())
        for start in range(0,flat.numel(),step):digest.update(flat[start:start+step].cpu().numpy().tobytes())
        count+=1;size+=value.numel()*value.element_size()
    return dict(sha256=digest.hexdigest(),tensor_count=count,bytes=size)


def parameter_fingerprints(model,layers):
    import hashlib
    result={}
    for layer in layers:
        digest=hashlib.sha256()
        for name,p in model.blocks[layer].named_parameters():
            digest.update(name.encode());flat=p.detach().reshape(-1)
            for start in range(0,flat.numel(),8*1024**2):digest.update(flat[start:start+8*1024**2].cpu().numpy().tobytes())
        result[str(layer)]=digest.hexdigest()
    return result


def save_delta(model,directory,plan,metadata):
    directory=Path(directory);directory.mkdir(exist_ok=False)
    files={}
    for layer in plan["selected_layers"]:
        path=directory/f"block_{layer:02d}.pt";recipe.save(path,joint._cpu(model.blocks[layer].state_dict()));files[path.name]=recipe.sha(path)
    value=dict(schema=SCHEMA,selected_layers=plan["selected_layers"],parent_assembly_fingerprint=plan["assembly_fingerprint"],
        search_fingerprint=plan["selection"]["search_fingerprint"],metadata=metadata,artifacts=files,
        fixed_state="Other33 blocks, native norm, tiedE and all geometry reconstructed from authenticated parent; no norm delta")
    recipe.write(directory/"manifest.json",value);return value


def load_delta(model,directory,plan):
    directory=Path(directory);value=recipe.read(directory/"manifest.json")
    names={f"block_{i:02d}.pt" for i in plan["selected_layers"]}
    if (value["schema"]!=SCHEMA or value["selected_layers"]!=plan["selected_layers"]
            or value["parent_assembly_fingerprint"]!=plan["assembly_fingerprint"]
            or value["search_fingerprint"]!=plan["selection"]["search_fingerprint"] or set(value["artifacts"])!=names):
        raise ValueError("Delta selection/provenance changed")
    for name,digest in value["artifacts"].items():
        if recipe.sha(directory/name)!=digest:raise ValueError("Selected delta bytes changed")
    for layer in plan["selected_layers"]:model.blocks[layer].load_state_dict(recipe.load(directory/f"block_{layer:02d}.pt"),strict=True)
    return value


def replace_delta(model,path,plan,metadata):
    path=Path(path);pending=path.with_name(path.name+".pending");retired=path.with_name(path.name+".retired")
    if pending.exists() or retired.exists():raise FileExistsError("Incomplete delta replacement preserved")
    save_delta(model,pending,plan,metadata)
    if path.exists():os.replace(path,retired)
    os.replace(pending,path)
    if retired.exists():shutil.rmtree(retired)
    return recipe.sha(path/"manifest.json")


class Trainer(base.Trainer):
    def __init__(self,model,dataset,plan,*,guard=None):
        super().__init__(model,dataset,config=Config(),guard=guard);self.plan=plan
        expected={f"blocks.{i}." for i in plan["selected_layers"]}
        if any(not any(name.startswith(prefix) for prefix in expected) for name,_ in self.named_parameters):
            raise ValueError("Optimizer includes a frozen block or norm")

    def losses(self,batch):
        positions=batch["loss_positions"][:,[1,3]]
        logits,features=self.model.forward_with_features(batch["input_ids"],batch["position_ids"],batch["attention_mask"],positions)
        cases=[self.dataset.items[i] for i in batch["indices"]]
        values=semantic.semantic_recovery_loss(logits,batch["teacher"],label_ids=self.dataset.label_ids,
            choice_counts=torch.tensor([len(c["choices"]) for c in cases],device=self.device),
            answers=torch.tensor([c["answer"] for c in cases],device=self.device),config=self.config.objective)
        aux=feature_objective(features,batch["feature_targets"],
            {i:self.model.blocks[i].basis for i in self.plan["selected_layers"]},batch["valid_tokens"],self.device)
        values.update(aux,endpoint_loss=values["loss"],weighted_feature_loss=aux["feature_loss"])
        values["loss"]=values["endpoint_loss"]+values["weighted_feature_loss"]
        return logits,values

    def _records(self,indices,logits,values):
        rows=super()._records(indices,logits,values)
        for index,row in enumerate(rows):
            row["feature_loss"]=float(values["per_case_feature_loss"][index].detach())
            row["feature_loss_by_layer"]={k:float(v[index].detach()) for k,v in values["per_layer_feature_loss"].items()}
        return rows

    def train_window(self,indices):
        if not indices or any(i not in self.dataset.training for i in indices):raise ValueError("Training identities only")
        self._guard();self.model.train();self.optimizer.zero_grad(set_to_none=True)
        totals={k:0. for k in (*semantic.METRICS,*FEATURE_KEYS)};records=[]
        for start in range(0,len(indices),self.config.microbatch):
            self._guard();selected=indices[start:start+self.config.microbatch];batch=self.dataset.batch(selected,self.device)
            logits,values=self.losses(batch);weight=len(selected)/len(indices);(values["loss"]*weight).backward()
            for key in totals:totals[key]+=float(values[key].detach())*weight
            records+=self._records(selected,logits,values)
            del logits,values,batch
        if any(p.grad is None for _,p in self.named_parameters):raise RuntimeError("Selected parameter missing gradient")
        step=self.optimizer.step(max_grad_norm=self.config.gradient_clip);self.optimizer.zero_grad(set_to_none=True);self.updates+=1
        return dict(**totals,cases=len(indices),update=self.updates,optimizer=step,ranking=semantic.ranking_summary(records),
            gradient_parameter_count=len(self.named_parameters),gradient_scalar_count=sum(p.numel() for _,p in self.named_parameters))

    @torch.no_grad()
    def development(self):
        self._guard();self.model.eval();self.optimizer.zero_grad(set_to_none=True)
        totals={k:0. for k in (*semantic.METRICS,*FEATURE_KEYS)};records=[]
        for start in range(0,len(self.dataset.development),self.config.microbatch):
            self._guard();indices=self.dataset.development[start:start+self.config.microbatch]
            logits,values=self.losses(self.dataset.batch(indices,self.device))
            for key in totals:totals[key]+=float(values[key])*len(indices)
            records+=self._records(indices,logits,values)
        metrics={k:v/len(records) for k,v in totals.items()};metrics.update(semantic.ranking_summary(records))
        metrics.update(kl=metrics["full_kl"],ce=metrics["answer_full_vocab_ce"],selection="highest_dev_accuracy_then_lowest_conditional_choice_CE")
        return metrics,records

    def fit(self,directory,reference,event=lambda *_a,**_k:None,*,frozen_before=None):
        if frozen_before is None:frozen_before=frozen_fingerprint(self.model)
        directory=Path(directory);directory.mkdir(exist_ok=False);path=directory/"selected"
        best=None;history=[];best_records=None;best_epoch=None
        for epoch in range(self.config.epochs+1):
            if epoch:
                order=list(self.dataset.training);random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0,len(order),16):event("update",epoch=epoch,**self.train_window(order[start:start+16]))
            metrics,records=self.development()
            if epoch==0:
                control=base.baseline_control(records,reference,[self.dataset.items[i] for i in self.dataset.development])
                recipe.write(directory/"initial_control.json",control)
                if not control["passed"]:raise RuntimeError("Selected-group initialization did not reproduce27/64")
            report=directory/f"epoch_{epoch:02d}_development.json";recipe.write(report,dict(metrics=metrics,records=records))
            changed=best is None or base.selection_key(metrics)>base.selection_key(best)
            row=dict(epoch=epoch,metrics=metrics,development_sha256=recipe.sha(report),checkpoint_updated=changed)
            if changed:
                row["checkpoint_manifest_sha256"]=replace_delta(self.model,path,self.plan,dict(epoch=epoch,development=metrics))
                best,best_records,best_epoch=metrics,records,epoch
            history.append(row);event("development",epoch=epoch,**metrics)
        load_delta(self.model,path,self.plan);metrics,records=self.development()
        if metrics!=best or records!=best_records:raise RuntimeError("Selected delta reload changed results")
        if frozen_fingerprint(self.model)!=frozen_before:raise RuntimeError("Scientific training changed frozen state")
        result=dict(status="completed",systems_only=False,confirmed=False,selected_epoch=best_epoch,selected=best,
            initial=history[0]["metrics"],history=history,updates=self.updates,strict_reload_exact=True,
            selected_directory=str(path),selected_manifest_sha256=recipe.sha(path/"manifest.json"),
            selected_layers=self.plan["selected_layers"],initializer_selected=best_epoch==0,quality_pass=best["correct"]>=47,
            minimum_correct=47,native_correct=50,optimizer=self.optimizer.memory_stats(),scope=SCOPE,
            frozen_state_unchanged=True,frozen_state=frozen_before)
        recipe.write(directory/"result.json",result);return result


class FeatureTeacher(teacher.PartitionedNativeTeacher):
    @torch.inference_mode()
    def forward_features(self,batch,layers):
        hidden=F.embedding(batch["input_ids"],self.embedding);ids=batch["position_ids"]
        mask=torch.zeros(batch["attention_mask"].shape,device=self.front,dtype=torch.float32).masked_fill_(
            ~batch["attention_mask"],torch.finfo(torch.float32).min)
        rope=self.rotary_front(hidden,ids);features={}
        for index,block in enumerate(self.blocks):
            if index==self.cut:
                hidden=hidden.to(self.tail);ids=ids.to(self.tail);mask=mask.to(self.tail);rope=self.rotary_tail(hidden,ids)
            value=block(hidden,attention_mask=mask,position_ids=ids,position_embeddings=rope,cache_position=ids[0],use_cache=False)
            hidden=value[0] if isinstance(value,tuple) else value
            if index in layers:features[index]=hidden
        rows=torch.arange(hidden.shape[0],device=self.tail)[:,None]
        chosen=hidden[rows,batch["loss_positions"][:,[1,3]].to(self.tail)].to(self.front)
        return F.linear(self.norm(chosen),self.embedding),features


@torch.inference_mode()
def capture_boundaries(model,items,layers,labels,directory,*,native_cases,native_banks,reference,guard,progress=lambda **_k:None):
    directory=Path(directory);directory.mkdir(exist_ok=False);banks={};records=[]
    native_by_id={c["identity"]:c for c in native_cases};parity={str(i):dict(passed=True,max_abs_delta=0.) for i in layers}
    try:
        for layer in layers:
            banks[layer]=np.lib.format.open_memmap(directory/f"boundary_{layer+1:02d}.npy",mode="w+",dtype=np.float32,
                shape=(items[-1]["stop"],model.embedding.shape[1]))
        for start in range(0,len(items),4):
            guard();indices=list(range(start,min(start+4,len(items))));batch=teacher.token_batch(items,indices,model.front)
            logits,features=model.forward_features(batch,layers)
            if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite native logits")
            for layer,hidden in features.items():
                if hidden.dtype!=torch.float32 or not torch.isfinite(hidden).all():raise FloatingPointError("Invalid native FP32 hidden state")
                host=hidden.cpu().numpy()
                for row,index in enumerate(indices):
                    c=items[index];valid=host[row,:len(c["input_ids"])];banks[layer][c["offset"]:c["stop"]]=valid
                    if c["split"]=="development":
                        ref=native_by_id[c["identity"]]
                        if any(ref[k]!=c[k] for k in ("input_ids","loss_positions","answer","family","choices")):
                            raise ValueError("Native development identity/token alignment changed")
                        wanted=native_banks[layer][ref["offset"]:ref["stop"]]
                        parity[str(layer)]["max_abs_delta"]=max(parity[str(layer)]["max_abs_delta"],float(np.max(np.abs(valid-wanted))))
                        parity[str(layer)]["passed"] &= bool(np.allclose(valid,wanted,atol=1e-5,rtol=2e-4))
            choices=logits[:,-1,labels].cpu()
            for row,index in enumerate(indices):
                c=items[index]
                if c["split"]=="development":
                    z=choices[row,:len(c["choices"])];records.append(dict(identity=c["identity"],family=c["family"],
                        answer=c["answer"],prediction=int(z.argmax()),logits=z.tolist()))
            del logits,features,batch
            progress(cases_written=min(start+4,len(items)),total_cases=len(items))
        for bank in banks.values():bank.flush()
    finally:
        for bank in banks.values():bank._mmap.close()
    control=teacher.check_control(records,reference,[c for c in items if c["split"]=="development"])
    return dict(passed=control["passed"] and all(v["passed"] for v in parity.values()),records=records,native_control=control,
        development_boundary_parity=parity,parity_atol=1e-5,parity_rtol=2e-4,selected_layers=layers,
        dtype="float32",shape=[items[-1]["stop"],model.embedding.shape[1]],teacher_hidden_states="Original native trajectory; never substituted into student forward")


def native_target_inputs(native_root,layers,source_fingerprint):
    """Authenticate only the three consumed native banks, not all37 banks."""
    from . import jspace_native_dev_cache as native_cache
    native_root=Path(native_root);plan=native_cache.verify(native_root,completed=False)
    manifest=recipe.read(native_root/"cache/manifest.json");result=recipe.read(native_root/"cache/result.json")
    if (plan["source_fingerprint"]!=source_fingerprint or manifest.get("status")!="completed"
            or manifest.get("fingerprint")!=plan["fingerprint"] or result.get("status")!="completed"
            or result.get("fingerprint")!=plan["fingerprint"] or result.get("correct")!=50
            or result.get("count")!=64 or result.get("native_baseline_predictions_exact") is not True):
        raise ValueError("Authenticated original FP32 native50 development cache required")
    inputs={}
    for name in ["items.json","cache/result.json",*[f"cache/boundary_{i+1:02d}.npy" for i in layers]]:
        path=native_root/name;expected=manifest["artifact_sha256"][name]
        if recipe.sha(path)!=expected:raise ValueError("Changed consumed native feature/control artifact: "+name)
        inputs[str(path)]=expected
    for name in ("plan.json","cache/manifest.json"):inputs[str(native_root/name)]=recipe.sha(native_root/name)
    return inputs


def prepare(root,assembly_root,teacher_root,search_root):
    from .jspace_layer_combination_search import verified_selection
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Hide CUDA for CPU preparation")
    root,assembly_root,teacher_root,search_root=map(lambda p:Path(p).resolve(),(root,assembly_root,teacher_root,search_root))
    if root.exists():raise FileExistsError(root)
    selection=verified_selection(search_root);layers=selected_layers(selection["selected_layers"])
    parent=base.assembly.verify(assembly_root);baseline=base.assembly.completed_arm(assembly_root,parent,"all_compact")
    tp=teacher.verify(teacher_root,completed=True);items=recipe.read(Path(parent["source"])/"items.json")
    if (selection["assembly_fingerprint"]!=parent["fingerprint"] or Path(selection["assembly_root"]).resolve()!=assembly_root
            or baseline["accuracy"]["correct"]!=27 or tp["source_fingerprint"]!=parent["source_fingerprint"]
            or tp["items_digest"]!=recipe.digest(items)):
        raise ValueError("Matching completed triple search, assembly27 and originalFP32teacher required")
    teacher.validate_items(items["items"],151936,tp["label_ids"],full=True)
    offset=0
    for c in items["items"]:
        if c["offset"]!=offset or c["stop"]!=offset+len(c["input_ids"]):raise ValueError("Original contiguous token layout changed")
        offset=c["stop"]
    inputs=dict(parent["input_sha256"])
    inputs.update(native_target_inputs(tp["native_cache_root"],layers,parent["source_fingerprint"]))
    for path in [assembly_root/"plan.json",assembly_root/"all_compact/manifest.json",assembly_root/"all_compact/result.json",
        teacher_root/"plan.json",teacher_root/"cache/manifest.json",search_root/"plan.json",
        *[search_root/"search"/name for name in ("manifest.json","selection.json","result.json")]]:
        inputs[str(path)]=recipe.sha(path)
    for name,digest in recipe.read(teacher_root/"cache/manifest.json")["artifact_sha256"].items():inputs[str(teacher_root/name)]=digest
    sources=dict(parent["source_sha256"]);base.assembly._merge_hashes(sources,tp["source_sha256"])
    paths=[Path(__file__),Path(base.__file__),Path(__file__).with_name("jspace_layer_combination_search.py"),
        recipe.ROOT/"tests/structure_net/test_jspace_layer_group_repair.py",PROTOCOL,
        Path(__file__).with_name("jspace_streaming_adamw.py"),Path(__file__).with_name("jspace_semantic_loss.py")]
    for path in paths:base.assembly._merge_hashes(sources,{str(path):recipe.sha(path)})
    plan=dict(schema=SCHEMA,assembly_root=str(assembly_root),assembly_fingerprint=parent["fingerprint"],
        source=parent["source"],source_fingerprint=parent["source_fingerprint"],model=parent["model"],
        teacher_root=str(teacher_root),teacher_fingerprint=tp["fingerprint"],search_root=str(search_root),selection=selection,
        selected_layers=layers,config=asdict(Config()),items_digest=recipe.digest(items),tokens=offset,label_ids=tp["label_ids"],
        native_cache_root=tp["native_cache_root"],initial_correct=27,native_correct=50,minimum_correct=47,confirmed=False,scope=SCOPE,
        feature_objective="mean_selected_layers mean_cases [mean_valid ||(y-native_next)@U||^2 / max(mean_valid ||native_next@U||^2,1e-8)]; full y derivative; detached teacher",
        freezing="Only the selected3 blocks train; other33 blocks, native norm, tiedE, bases/anchors and all buffers fixed",
        partition={"3060":list(range(24)),"2060_SUPER":list(range(24,36))},
        input_sha256=inputs,input_identity={p:recipe.file_identity(p) for p in inputs},source_sha256=sources)
    plan["fingerprint"]=recipe.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        if recipe.sha(path)!=digest:raise ValueError("Producer changed during preparation")
        destination=root/"producer"/Path(path).relative_to(recipe.ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    recipe.write(root/"plan.json",plan);recipe.write(root/"items.json",items)
    return dict(status="prepared",fingerprint=plan["fingerprint"],selected_layers=layers)


def verify(root):
    from .jspace_layer_combination_search import verified_selection
    root=Path(root);plan=recipe.read(root/"plan.json")
    if (plan.get("schema")!=SCHEMA or plan["fingerprint"]!=recipe.digest({k:v for k,v in plan.items() if k!="fingerprint"})
            or plan["config"]!=asdict(Config()) or verified_selection(plan["search_root"])!=plan["selection"]):
        raise ValueError("Changed frozen subgroup repair/search selection")
    selected_layers(plan["selected_layers"])
    if plan["selected_layers"]!=plan["selection"]["selected_layers"]:raise ValueError("Changed selected triple")
    if base.assembly.verify(plan["assembly_root"])["fingerprint"]!=plan["assembly_fingerprint"]:raise ValueError("Changed parent assembly")
    if recipe.digest(recipe.read(root/"items.json"))!=plan["items_digest"]:raise ValueError("Changed repair cases")
    for path,digest in plan["source_sha256"].items():
        if recipe.sha(path)!=digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT))!=digest:raise ValueError("Changed repair producer")
    for path,value in plan["input_identity"].items():
        if recipe.file_identity(path)!=value:raise ValueError("Changed frozen input")
    return plan


def checked(root,stage,plan):
    path=Path(root)/stage;manifest=recipe.read(path/"manifest.json");result=recipe.read(path/"result.json")
    if (manifest.get("status")!="completed" or result.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"]
            or manifest.get("result_sha256")!=recipe.sha(path/"result.json")):
        raise ValueError("Completed matching stage required: "+stage)
    if stage=="capture":
        names={f"boundary_{i+1:02d}.npy" for i in plan["selected_layers"]}
        if result.get("passed") is not True or set(manifest["artifacts"])!=names:raise ValueError("Passing originalFP32feature capture required")
        for name,digest in manifest["artifacts"].items():
            if recipe.sha(path/name)!=digest:raise ValueError("Teacher boundary bytes changed")
    elif stage=="pilot":
        if (result.get("systems_only") is not True or result.get("updates")!=1 or result["initial_control"]["passed"] is not True
                or result["changed_selected_layers"]!=plan["selected_layers"] or result["frozen_state_unchanged"] is not True):
            raise ValueError("Group-only update/frozen-state pilot required")
    elif stage=="train":
        if (result.get("updates")!=192 or result.get("strict_reload_exact") is not True or result.get("frozen_state_unchanged") is not True
                or result.get("systems_only") is not False or result.get("selected_layers")!=plan["selected_layers"]
                or [r["epoch"] for r in result["history"]]!=list(range(7))):raise ValueError("Completed6-epoch group recovery required")
        if set(manifest["artifacts"])!={"frozen_state.json"} or recipe.sha(path/"frozen_state.json")!=manifest["artifacts"]["frozen_state.json"]:
            raise ValueError("Authenticated frozen-state receipt required")
        frozen=recipe.read(path/"frozen_state.json")
        if frozen.get("passed") is not True or frozen["before"]!=frozen["after"] or frozen["before"]!=result["frozen_state"]:
            raise ValueError("Frozen-state receipt does not match completed training")
        winner=max(result["history"],key=lambda r:base.selection_key(r["metrics"]))
        if (winner["epoch"]!=result["selected_epoch"] or winner["metrics"]!=result["selected"]
                or winner["checkpoint_manifest_sha256"]!=result["selected_manifest_sha256"]
                or recipe.sha(path/"selected/manifest.json")!=result["selected_manifest_sha256"]):raise ValueError("Selected delta selection changed")
        for row in result["history"]:
            if recipe.sha(path/f"epoch_{row['epoch']:02d}_development.json")!=row["development_sha256"]:raise ValueError("Changed epoch record")
    return result


def run(root,stage):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    from .jspace_single_block_verify import Tolerances,validate_records
    if stage not in ("capture","pilot","train","verify"):raise ValueError("Explicit repair stage required")
    root=Path(root).resolve();plan=verify(root);directory=root/("verification" if stage=="verify" else stage)
    if directory.exists():raise FileExistsError(directory)
    if stage!="capture":checked(root,"capture",plan)
    if stage=="train":checked(root,"pilot",plan)
    selected=checked(root,"train",plan) if stage=="verify" else None
    lease=settings=dataset=None;started=time.monotonic()
    try:
        lease=wait_ready([0,1]);settings=base.allocator_settings(lease)
        for row in settings["settings"]:torch.cuda.reset_peak_memory_stats(row["logical_device"])
        torch.set_num_threads(4);torch.manual_seed(Config().seed);torch.backends.cuda.matmul.allow_tf32=False
        def guard():
            current=wait_ready([0,1])
            if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Repair reservation changed")
            base.device_peaks(settings)
        parent=recipe.read(Path(plan["assembly_root"])/"plan.json");items=recipe.read(root/"items.json")["items"]
        if stage=="capture":
            first,second=settings["devices"]["3060"],settings["devices"]["2060_SUPER"]
            guard();embedding,norm,rotary=load_readout_norm(plan["model"]["path"],device=first,dtype=torch.float32)
            blocks=[]
            for layer in range(36):
                guard();blocks.append(load_native_block(plan["model"]["path"],layer,device=first if layer<24 else second,dtype=torch.float32))
            original=recipe.read(Path(plan["source"])/"plan.json")
            model=FeatureTeacher(embedding,blocks,norm,rotary,recipe.rotary_module(original,second),cut=24)
            native_root=Path(plan["native_cache_root"]);native_cases=recipe.read(native_root/"items.json")["items"]
            native_banks={i:np.load(native_root/"cache"/f"boundary_{i+1:02d}.npy",mmap_mode="r",allow_pickle=False) for i in plan["selected_layers"]}
            try:
                result=capture_boundaries(model,items,plan["selected_layers"],plan["label_ids"],directory,
                    native_cases=native_cases,native_banks=native_banks,reference=recipe.read(native_root/"cache/result.json")["records"],guard=guard,
                    progress=lambda **kw:recipe.event(root,"capture",**kw))
            finally:
                for bank in native_banks.values():bank._mmap.close()
            result.update(status="completed" if result["passed"] else "failed",fingerprint=plan["fingerprint"])
            recipe.write(directory/"result.json",result)
            if not result["passed"]:raise RuntimeError("OriginalFP32teacher feature/control parity failed")
            artifacts={f"boundary_{i+1:02d}.npy":recipe.sha(directory/f"boundary_{i+1:02d}.npy") for i in plan["selected_layers"]}
        else:
            model=build_model(parent,settings["devices"],plan["selected_layers"],guard);dataset=GroupDataset(root,plan)
            trainer=Trainer(model,dataset,plan,guard=guard);before=frozen_fingerprint(model);guard()
            reference=recipe.read(Path(plan["assembly_root"])/"all_compact/result.json")
            cases=[dataset.items[i] for i in dataset.development]
            if stage=="pilot":
                directory.mkdir();initial,records=trainer.development();control=base.baseline_control(records,reference,cases)
                if not control["passed"]:raise RuntimeError("Pilot did not reproduce27/64 initializer")
                hashes=parameter_fingerprints(model,plan["selected_layers"])
                order=sorted(dataset.training,key=lambda i:len(dataset.items[i]["input_ids"]),reverse=True)
                update=trainer.train_window(order[:16]);after=parameter_fingerprints(model,plan["selected_layers"])
                changed=[i for i in plan["selected_layers"] if hashes[str(i)]!=after[str(i)]]
                unchanged=before==frozen_fingerprint(model)
                if not unchanged or changed!=plan["selected_layers"]:raise RuntimeError("Pilot changed frozen state or failed to update selected triple")
                development,_=trainer.development()
                result=dict(status="completed",systems_only=True,updates=1,initial=initial,initial_control=control,update=update,
                    development=development,changed_selected_layers=changed,frozen_state_unchanged=unchanged,frozen_state=before,confirmed=False)
                recipe.write(directory/"result.json",result)
            elif stage=="train":
                result=trainer.fit(directory,reference,frozen_before=before,event=lambda kind,**kw:recipe.event(root,kind,stage=stage,**kw))
                recipe.write(directory/"frozen_state.json",dict(passed=True,before=before,after=result["frozen_state"]))
            else:
                directory.mkdir();manifest=load_delta(model,root/"train/selected",plan)
                metrics,records=trainer.development();ref=recipe.read(root/"train"/f"epoch_{selected['selected_epoch']:02d}_development.json")
                validate_records(records,cases);validate_records(ref["records"],cases);tol=Tolerances()
                predictions=all(a["prediction"]==b["prediction"] for a,b in zip(records,ref["records"]))
                logits=all(abs(x-y)<=tol.logit_atol+tol.logit_rtol*abs(y) for a,b in zip(records,ref["records"]) for x,y in zip(a["logits"],b["logits"]))
                metrics_pass=all(math.isclose(metrics[k],ref["metrics"][k],abs_tol=tol.feature_atol,rel_tol=tol.feature_rtol)
                    for k in (*semantic.METRICS,*FEATURE_KEYS,"kl","ce"))
                unchanged=before==frozen_fingerprint(model)
                result=dict(status="completed",passed=predictions and logits and metrics_pass and unchanged,fingerprint=plan["fingerprint"],
                    selected_epoch=selected["selected_epoch"],selected_manifest_sha256=recipe.sha(root/"train/selected/manifest.json"),
                    selected_layers=plan["selected_layers"],strict_state_load=True,predictions_exact=predictions,logits_pass=logits,
                    metrics_pass=metrics_pass,frozen_state_unchanged=unchanged,metrics=metrics,records=records,checkpoint_metadata=manifest["metadata"],
                    training_quality_pass=selected["quality_pass"],tolerances=asdict(tol),confirmed=False,scope=SCOPE)
                recipe.write(directory/"result.json",result)
                if not result["passed"]:raise RuntimeError("Fresh selected-group verification failed")
            artifacts={}
            if stage=="train":artifacts["frozen_state.json"]=recipe.sha(directory/"frozen_state.json")
        guard();recipe.write(directory/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"),artifacts=artifacts,lease_id=lease["id"],allocation=lease["allocation"],
            allocator=settings,device_peaks=base.device_peaks(settings),elapsed_seconds=time.monotonic()-started,
            passed=result.get("passed"),quality_pass=result.get("quality_pass",result.get("training_quality_pass"))))
        return dict(status="completed",stage=stage,fingerprint=plan["fingerprint"],selected_layers=plan["selected_layers"])
    except BaseException as error:
        directory.mkdir(parents=True,exist_ok=True);recipe.write(directory/"failure.json",dict(status="failed",fingerprint=plan["fingerprint"],
            lease_id=None if lease is None else lease["id"],error=repr(error),elapsed_seconds=time.monotonic()-started));raise
    finally:
        if dataset is not None:dataset.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","capture","pilot","train","verify"))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--assembly-root",type=Path)
    parser.add_argument("--teacher-root",type=Path);parser.add_argument("--search-root",type=Path);args=parser.parse_args()
    if args.stage=="prepare" and any(x is None for x in (args.assembly_root,args.teacher_root,args.search_root)):
        parser.error("prepare requires --assembly-root, --teacher-root and --search-root")
    output=prepare(args.root,args.assembly_root,args.teacher_root,args.search_root) if args.stage=="prepare" else run(args.root,args.stage)
    print(json.dumps(output,allow_nan=False),flush=True)
