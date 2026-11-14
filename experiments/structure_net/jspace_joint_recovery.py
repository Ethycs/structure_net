"""Joint task recovery of all compact branches with a native residual stream.

The original tied embedding/readout is fixed. Every compact branch and the
final RMS gain train together from token IDs, without cached student prefixes.
Teacher logits are immutable full-vocabulary BF16 cache entries. A separate
one-update systems pilot never supplies weights to the scientific training run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
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
from torch.utils.checkpoint import checkpoint

from . import jspace_quadratic_recipe as recipe
from .jspace_streaming_adamw import StreamingCPUAdamW

SCHEMA = "jspace.joint-residual-recovery.v1"


@dataclass(frozen=True)
class JointConfig:
    epochs: int = 3
    microbatch: int = 1
    accumulation: int = 16
    learning_rate: float = 1e-5
    temperature: float = 2.
    kl_weight: float = .7
    ce_weight: float = .3
    gradient_clip: float = 1.
    seed: int = 27862
    optimizer_chunk_bytes: int = 32*1024**2
    max_cpu_state_bytes: int = 32*1024**3

    def __post_init__(self):
        for name in ("epochs", "microbatch", "accumulation", "optimizer_chunk_bytes", "max_cpu_state_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("Positive integer required: "+name)
        if self.optimizer_chunk_bytes < 4:
            raise ValueError("Optimizer chunk must contain an FP32 scalar")
        if any(not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.temperature, self.gradient_clip)):
            raise ValueError("Positive finite optimization settings required")
        if (any(not math.isfinite(v) or v < 0 for v in (self.kl_weight,self.ce_weight))
                or not math.isclose(self.kl_weight+self.ce_weight, 1.)):
            raise ValueError("Nonnegative objective weights must sum to one")


class JointCarryModel(nn.Module):
    """One fixed E used for input lookup and full-vocabulary output projection."""
    def __init__(self, embedding, blocks, norm_gain, epsilon, rotary, *, checkpoint_blocks=True):
        super().__init__()
        if (embedding.ndim != 2 or norm_gain.shape != (embedding.shape[1],)
                or not blocks or not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError("Valid embedding, blocks and final RMS normalization required")
        self.register_buffer("embedding", embedding.detach())
        self.blocks = nn.ModuleList(blocks)
        self.norm_gain = nn.Parameter(norm_gain.detach().float().clone())
        self.rotary = rotary.eval().requires_grad_(False)
        self.epsilon = float(epsilon)
        self.checkpoint_blocks = bool(checkpoint_blocks)

    def forward(self, input_ids, position_ids, attention_mask, loss_positions):
        # The fixed lookup has no gradient; nonreentrant checkpointing still
        # differentiates every first-block parameter, unlike the legacy API.
        hidden = F.embedding(input_ids, self.embedding).float()
        rope = self.rotary(hidden, position_ids)
        for block in self.blocks:
            def apply(value, current=block):
                return current(value, attention_mask=attention_mask, position_ids=position_ids,
                    position_embeddings=rope)
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                hidden = checkpoint(apply, hidden, use_reentrant=False)
            else:
                hidden = apply(hidden)
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected = hidden[rows, loss_positions]
        normalized = selected*torch.rsqrt(selected.square().mean(-1, keepdim=True)+self.epsilon)
        normalized = normalized*self.norm_gain
        # Only four positions enter the full V-way head. E remains BF16 on the
        # real model; full-vocabulary objective arithmetic below is FP32.
        return F.linear(normalized.to(self.embedding.dtype), self.embedding).float()


class JointDataset:
    def __init__(self, items, teacher_logits_path, label_ids):
        self.items = list(items)
        if not self.items or any(c.get("split") not in ("train", "development") for c in self.items):
            raise ValueError("Only training/development cases may enter joint recovery")
        if len({c["identity"] for c in self.items}) != len(self.items):
            raise ValueError("Duplicate joint-recovery identity")
        self.training = [i for i,c in enumerate(self.items) if c["split"] == "train"]
        self.development = [i for i,c in enumerate(self.items) if c["split"] == "development"]
        if not self.training or not self.development:
            raise ValueError("Both train and development cases required")
        self.label_ids = list(label_ids)
        self.teacher_logits = np.load(teacher_logits_path, mmap_mode="r", allow_pickle=False)
        if (self.teacher_logits.dtype != np.int16 or self.teacher_logits.ndim != 3
                or self.teacher_logits.shape[:2] != (len(self.items),4)):
            self.close(); raise ValueError("Teacher cache must be BF16 bits [cases,4,full vocabulary]")
        self.vocabulary = self.teacher_logits.shape[-1]
        if not self.label_ids or any(type(i) is not int or not 0 <= i < self.vocabulary for i in self.label_ids):
            self.close(); raise ValueError("Invalid native choice-label token IDs")
        for c in self.items:
            positions = c["loss_positions"]
            if (not c["input_ids"] or any(not 0 <= i < self.vocabulary for i in c["input_ids"])
                    or len(positions) != 4 or any(type(p) is not int or not 0 <= p < len(c["input_ids"]) for p in positions)
                    or positions[-1] != len(c["input_ids"])-1
                    or not 0 <= c["answer"] < len(c["choices"]) <= len(self.label_ids)):
                self.close(); raise ValueError("Invalid tokens, final-answer position or choices")

    def close(self):
        value = getattr(self, "teacher_logits", None)
        if value is not None and not value._mmap.closed: value._mmap.close()

    def batch(self, indices, device):
        if not indices or any(type(i) is not int or not 0 <= i < len(self.items) for i in indices):
            raise ValueError("Valid nonempty batch indices required")
        cases = [self.items[i] for i in indices]
        lengths = [len(c["input_ids"]) for c in cases]
        ids = torch.zeros((len(cases),max(lengths)), dtype=torch.long, device=device)
        for row,c in enumerate(cases): ids[row,:lengths[row]] = torch.tensor(c["input_ids"],device=device)
        positions = torch.arange(max(lengths), device=device).expand(len(cases),-1)
        valid = positions < torch.tensor(lengths,device=device)[:,None]
        mask = (positions[:,:,None] >= positions[:,None,:]) & valid[:,:,None] & valid[:,None,:]
        teacher = torch.from_numpy(np.array(self.teacher_logits[indices],copy=True)).view(torch.bfloat16).to(device,torch.float32)
        if not torch.isfinite(teacher).all(): raise FloatingPointError("Nonfinite original teacher logits")
        return dict(indices=indices, input_ids=ids, position_ids=positions, attention_mask=mask[:,None],
            loss_positions=torch.tensor([c["loss_positions"] for c in cases],device=device), teacher=teacher,
            target_tokens=torch.tensor([self.label_ids[c["answer"]] for c in cases],device=device))


def joint_losses(student, teacher, target_tokens, config=JointConfig()):
    if student.shape != teacher.shape or student.ndim != 3 or student.shape[1] != 4:
        raise ValueError("Student and teacher must share [batch,4,full vocabulary]")
    log_p = (teacher.detach().float()/config.temperature).log_softmax(-1)
    log_q = (student.float()/config.temperature).log_softmax(-1)
    per_case_kl = (log_p.exp()*(log_p-log_q)).sum(-1).mean(-1)
    per_case_ce = F.cross_entropy(student[:,-1].float(),target_tokens,reduction="none")
    kl, ce = per_case_kl.mean(), per_case_ce.mean()
    loss = config.kl_weight*config.temperature**2*kl+config.ce_weight*ce
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite joint recovery objective")
    return dict(loss=loss, kl=kl, ce=ce, per_case_kl=per_case_kl, per_case_ce=per_case_ce)


def selection_key(metrics):
    """Higher actual development accuracy first; lower KL breaks ties."""
    return (metrics["correct"], -metrics["kl"])


def _cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:_cpu(v) for k,v in value.items()}
    return value


def save_selected(model, directory, *, metadata):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
    files = []
    for i,block in enumerate(model.blocks):
        path = directory/f"block_{i:02d}.pt"
        recipe.save(path,_cpu(block.state_dict())); files.append(path)
    path=directory/"norm_gain.pt";recipe.save(path,_cpu(model.norm_gain));files.append(path)
    manifest=dict(schema=SCHEMA,blocks=len(model.blocks),metadata=metadata,
        fixed_embedding="Original pinned native E, used once and tied for lookup/readout",
        optimizer_resume="New run must explicitly reset optimizer; selected weights are a warm start only",
        artifacts={p.name:recipe.sha(p) for p in files})
    recipe.write(directory/"manifest.json",manifest)
    return manifest


def load_selected(model, directory):
    directory=Path(directory); manifest=recipe.read(directory/"manifest.json")
    expected={f"block_{i:02d}.pt" for i in range(len(model.blocks))}|{"norm_gain.pt"}
    if manifest["schema"] != SCHEMA or manifest["blocks"] != len(model.blocks) or set(manifest["artifacts"]) != expected:
        raise ValueError("Selected checkpoint architecture/artifacts differ")
    for name,digest in manifest["artifacts"].items():
        if recipe.sha(directory/name)!=digest:raise ValueError("Changed selected checkpoint: "+name)
    for i,block in enumerate(model.blocks):
        block.load_state_dict(recipe.load(directory/f"block_{i:02d}.pt"),strict=True)
    gain=recipe.load(directory/"norm_gain.pt")
    if gain.shape != model.norm_gain.shape or gain.dtype != model.norm_gain.dtype:
        raise ValueError("Final gain checkpoint shape/precision differs")
    with torch.no_grad():model.norm_gain.copy_(gain)
    return manifest


class JointTrainer:
    def __init__(self,model,dataset,*,config=JointConfig(),guard=None):
        self.model,self.dataset,self.config,self.guard=model,dataset,config,guard
        self.device=model.embedding.device
        if self.device.type=="cuda" and guard is None:raise ValueError("Reservation guard required for GPU training")
        if model.embedding.requires_grad or model.embedding.shape[0]!=dataset.vocabulary:
            raise ValueError("Original frozen full-vocabulary tied E required")
        self.named_parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if any(p.device!=self.device for _,p in self.named_parameters):raise ValueError("Parameters must share one device")
        self._guard()
        self.optimizer=StreamingCPUAdamW([p for _,p in self.named_parameters],lr=config.learning_rate,
            weight_decay=0.,chunk_bytes=config.optimizer_chunk_bytes,max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.updates=0

    def _guard(self):
        if self.guard is not None:self.guard()

    def losses(self,batch):
        logits=self.model(batch["input_ids"],batch["position_ids"],batch["attention_mask"],batch["loss_positions"])
        values=joint_losses(logits,batch["teacher"],batch["target_tokens"],self.config)
        return logits,values

    def train_window(self,indices):
        if not indices or any(i not in self.dataset.training for i in indices):
            raise ValueError("Optimizer updates consume training identities only")
        self._guard();self.model.train();self.optimizer.zero_grad(set_to_none=True)
        metrics=dict(loss=0.,kl=0.,ce=0.)
        for start in range(0,len(indices),self.config.microbatch):
            self._guard()
            selected=indices[start:start+self.config.microbatch]
            batch=self.dataset.batch(selected,self.device)
            logits,values=self.losses(batch)
            weight=len(selected)/len(indices)
            (values["loss"]*weight).backward()
            for key in metrics:metrics[key]+=float(values[key].detach())*weight
            del logits,values,batch
        missing=[n for n,p in self.named_parameters if p.grad is None]
        if missing:raise RuntimeError("Joint parameters missing gradients: "+", ".join(missing))
        # Streaming optimizer checks the unscaled norm of EVERY gradient before
        # mutating any weights/moments, rejecting both NaN and infinity.
        step=self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        self.optimizer.zero_grad(set_to_none=True);self.updates+=1
        return dict(**metrics,cases=len(indices),update=self.updates,optimizer=step,
            gradient_parameter_count=len(self.named_parameters),gradient_scalar_count=sum(p.numel() for _,p in self.named_parameters))

    @torch.no_grad()
    def development(self):
        self._guard();self.model.eval();self.optimizer.zero_grad(set_to_none=True)
        totals=dict(loss=0.,kl=0.,ce=0.);records=[]
        for start in range(0,len(self.dataset.development),self.config.microbatch):
            self._guard()
            indices=self.dataset.development[start:start+self.config.microbatch]
            batch=self.dataset.batch(indices,self.device)
            logits,values=self.losses(batch)
            for key in totals:totals[key]+=float(values[key])*len(indices)
            for row,index in enumerate(indices):
                case=self.dataset.items[index]
                label_logits=logits[row,-1,self.dataset.label_ids[:len(case["choices"])]]
                records.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
                    prediction=int(label_logits.argmax()),logits=label_logits.cpu().tolist(),
                    kl=float(values["per_case_kl"][row]),ce=float(values["per_case_ce"][row])))
        count=len(records);correct=sum(r["answer"]==r["prediction"] for r in records)
        metrics={k:v/count for k,v in totals.items()}
        metrics.update(count=count,correct=correct,accuracy=correct/count,selection="highest_dev_accuracy_then_lowest_full_vocab_kl")
        metrics["by_family"]={f:dict(correct=sum(r["prediction"]==r["answer"] for r in records if r["family"]==f),
            count=sum(r["family"]==f for r in records)) for f in sorted({r["family"] for r in records})}
        return metrics,records

    def fit(self,directory,*,event=lambda *_a,**_k:None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        initial,records=self.development();best,best_epoch=initial,0
        recipe.write(directory/"epoch_00_development.json",dict(metrics=initial,records=records))
        save_selected(self.model,directory/"epoch_00_selected",metadata=dict(epoch=0,development=initial))
        best_records=records;history=[dict(epoch=0,development=initial)]
        event("development",epoch=0,**initial)
        effective=self.config.microbatch*self.config.accumulation
        for epoch in range(1,self.config.epochs+1):
            order=list(self.dataset.training);random.Random(self.config.seed+epoch).shuffle(order)
            for start in range(0,len(order),effective):
                values=self.train_window(order[start:start+effective]);event("update",epoch=epoch,**values)
            metrics,records=self.development()
            recipe.write(directory/f"epoch_{epoch:02d}_development.json",dict(metrics=metrics,records=records))
            if selection_key(metrics)>selection_key(best):
                best,best_epoch,best_records=metrics,epoch,records
                save_selected(self.model,directory/f"epoch_{epoch:02d}_selected",metadata=dict(epoch=epoch,development=metrics))
            history.append(dict(epoch=epoch,development=metrics));event("development",epoch=epoch,**metrics)
        selected=directory/f"epoch_{best_epoch:02d}_selected"
        load_selected(self.model,selected)
        reloaded,reloaded_records=self.development()
        if reloaded_records!=best_records or reloaded!=best:
            raise RuntimeError("Strict selected-state reload changed development logits/metrics")
        result=dict(status="completed",systems_only=False,selected_epoch=best_epoch,selected_directory=str(selected),
            selected=best,initial=initial,history=history,updates=self.updates,strict_reload_exact=True,
            optimizer=self.optimizer.memory_stats(),optimizer_resume="Not saved; explicit optimizer reset required for a new warm-start run")
        recipe.write(directory/"result.json",result)
        return result


def build_model(source,*,arm,initialization,device):
    from .jspace_qwen_parallel_model import load_readout_norm
    from .jspace_residual_carry import ResidualCarryBlock
    plan=recipe.read(Path(source)/"plan.json")
    if initialization not in ("native_projected","selected"):raise ValueError("Unknown joint initialization")
    config=recipe.native_config(plan)
    if not config.tie_word_embeddings:raise ValueError("Joint recipe requires native tied E")
    embedding,norm,rotary=load_readout_norm(plan["model"]["path"],device=device,dtype=torch.bfloat16)
    blocks=[]
    for layer in range(plan["config"]["layers"]):
        wrapped=recipe.build_block(source,arm,layer,device=device,selected=initialization=="selected")
        # Deliberately do not call module.to(dtype): geometry must stay FP32.
        for p in wrapped.parameters():
            p.data=p.data.to(torch.bfloat16);p.requires_grad_(True)
        block=ResidualCarryBlock(wrapped)
        if block.basis.dtype!=torch.float32:raise ValueError("Residual basis must remain FP32")
        blocks.append(block)
    return JointCarryModel(embedding,blocks,norm.weight,norm.variance_epsilon,rotary)


def prepare(root,source,*,arm="softmax",initialization="selected",config=JointConfig()):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA with CUDA_VISIBLE_DEVICES='' for CPU preparation")
    root,source=Path(root).resolve(),Path(source).resolve()
    if root.exists():raise FileExistsError(root)
    if arm not in recipe.ARMS or initialization not in ("native_projected","selected"):
        raise ValueError("Invalid attention arm or initializer")
    plan=recipe.verify(source)
    items=recipe.read(source/"items.json")
    teacher=Path(plan["teacher_root"])/"capture/teacher_logits.npy"
    inputs={str(teacher):plan["input_sha256"][str(teacher)]}
    basis=recipe.read(source/"basis/manifest.json")
    if basis["status"]!="completed" or basis["fingerprint"]!=plan["fingerprint"]:raise ValueError("Invalid fitted source")
    for layer in range(plan["config"]["layers"]):
        path=recipe.fit_path(source,layer);inputs[str(path)]=basis["artifacts"][str(path.relative_to(source))]
        if initialization=="selected":
            path=recipe.train_path(source,arm,layer)/"selected.pt";manifest=recipe.read(path.parent/"manifest.json")
            if manifest["status"]!="completed" or manifest["fingerprint"]!=plan["fingerprint"]:raise ValueError("Invalid selected source")
            inputs[str(path)]=manifest["artifacts"][str(path.relative_to(source))]
    for name,expected in inputs.items():
        if recipe.sha(name)!=expected:raise ValueError("Changed initialization/teacher file: "+name)
    files=[Path(__file__),Path(__file__).with_name("jspace_residual_carry.py"),Path(__file__).with_name("jspace_streaming_adamw.py"),
        recipe.ROOT/"tests/structure_net/test_jspace_joint_recovery.py"]
    value=dict(schema=SCHEMA,source=str(source),source_fingerprint=plan["fingerprint"],source_config=plan["config"],
        config=asdict(config),arm=arm,initialization=initialization,items_digest=recipe.digest(items),label_ids=plan["label_ids"],
        input_sha256=inputs,input_identity={p:recipe.file_identity(p) for p in inputs},
        source_sha256={str(p):recipe.sha(p) for p in files},teacher_path=str(teacher),
        objective="0.7*T^2 full-vocabulary KL at four positions + 0.3 final-answer full-vocabulary CE",
        parameter_dtype="bfloat16 compact branches; float32 final RMS gain",geometry_residual_dtype="float32",
        checkpoint="nonreentrant per-block activation checkpointing",embedding_readout="one frozen native tied BF16 E",
        pilot="separate systems-only one accumulated update; no pilot weights enter train",
        optimizer_resume="selected weights only; continuing requires explicit new-run optimizer reset")
    value["fingerprint"]=recipe.digest(value)
    root.mkdir(parents=True)
    for path in files:
        target=root/"producer"/path.relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    recipe.write(root/"plan.json",value);recipe.write(root/"items.json",items)
    return dict(status="prepared",fingerprint=value["fingerprint"],initialization=initialization,arm=arm)


def verify(root):
    root=Path(root);plan=recipe.read(root/"plan.json")
    if plan["schema"]!=SCHEMA or plan["fingerprint"]!=recipe.digest({k:v for k,v in plan.items() if k!="fingerprint"}):
        raise ValueError("Changed joint recovery plan")
    parent=recipe.verify(plan["source"])
    if parent["fingerprint"]!=plan["source_fingerprint"] or recipe.digest(recipe.read(root/"items.json"))!=plan["items_digest"]:
        raise ValueError("Changed original source or case identities")
    for path,digest in plan["source_sha256"].items():
        copy=root/"producer"/Path(path).relative_to(recipe.ROOT)
        if recipe.sha(path)!=digest or recipe.sha(copy)!=digest:raise ValueError("Changed joint recovery producer")
    for path,identity in plan["input_identity"].items():
        if recipe.file_identity(path)!=identity:raise ValueError("Changed frozen training input metadata")
    return plan


def verify_pilot(root, fingerprint):
    directory=Path(root)/"pilot"
    manifest=recipe.read(directory/"manifest.json")
    result=recipe.read(directory/"result.json")
    if (manifest.get("status")!="completed" or manifest.get("systems_only") is not True
            or manifest.get("fingerprint")!=fingerprint
            or manifest.get("result_sha256")!=recipe.sha(directory/"result.json")
            or result.get("status")!="completed" or result.get("systems_only") is not True
            or result.get("updates")!=1):
        raise ValueError("A successful matching one-update systems pilot is required before training")
    check_reserved_peak(manifest["allocation"],manifest["peak_reserved_bytes"])
    return manifest


def check_reserved_peak(allocation, peak):
    if (not isinstance(allocation,dict) or not allocation
            or any(type(v) is not int or v<=0 for v in allocation.values())):
        raise ValueError("A positive MiB reservation allocation is required")
    if type(peak) is not int or peak < 0:
        raise ValueError("Reserved memory peak must be a nonnegative integer byte count")
    budget=sum(allocation.values())*1024**2
    if peak>budget:
        raise MemoryError(f"Reserved CUDA peak {peak} exceeds lease allocation {budget} bytes")
    return budget


def run(root,stage):
    if stage not in ("pilot","train"):raise ValueError("Only pilot/train execute GPU work")
    root=Path(root).resolve();plan=verify(root);directory=root/stage
    if directory.exists():raise FileExistsError(directory)
    if stage=="train":verify_pilot(root,plan["fingerprint"])
    from .jspace_full_lease import wait_ready
    started=time.monotonic();lease=None;dataset=None;cuda_started=False
    try:
        lease=wait_ready([0])
        allocation=dict(lease["allocation"])
        check_reserved_peak(allocation,0)
        def guard():
            current=wait_ready([0])
            if current["id"]!=lease["id"] or current["allocation"]!=allocation:
                raise RuntimeError("Joint recovery reservation identity/allocation changed")
            check_reserved_peak(allocation,torch.cuda.max_memory_reserved())
        torch.set_num_threads(4);torch.manual_seed(plan["config"]["seed"])
        torch.cuda.init();cuda_started=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
        config=JointConfig(**plan["config"])
        model=build_model(plan["source"],arm=plan["arm"],initialization=plan["initialization"],device="cuda:0")
        guard()
        dataset=JointDataset(recipe.read(root/"items.json")["items"],plan["teacher_path"],plan["label_ids"])
        trainer=JointTrainer(model,dataset,config=config,guard=guard)
        if stage=="pilot":
            directory.mkdir(exist_ok=False)
            # Longest training cases exercise the actual peak prefill/backward
            # shape, with the full scientific accumulation window.
            order=sorted(dataset.training,key=lambda i:len(dataset.items[i]["input_ids"]),reverse=True)
            update=trainer.train_window(order[:config.microbatch*config.accumulation])
            result=dict(status="completed",systems_only=True,updates=1,update=update,
                longest_context=max(len(dataset.items[i]["input_ids"]) for i in order),
                initialization=plan["initialization"],trainable_scalars=sum(p.numel() for _,p in trainer.named_parameters))
            recipe.write(directory/"result.json",result)
        else:
            result=trainer.fit(directory,event=lambda kind,**values:recipe.event(root,kind,stage=stage,**values))
        guard()
        metrics=dict(elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),lease_id=lease["id"],fingerprint=plan["fingerprint"],
            allocation=allocation,reserved_budget_bytes=check_reserved_peak(allocation,torch.cuda.max_memory_reserved()),
            status="completed",systems_only=stage=="pilot",result_sha256=recipe.sha(directory/"result.json"))
        recipe.write(directory/"manifest.json",metrics)
        return dict(**result,execution=metrics)
    except Exception as error:
        directory.mkdir(parents=True,exist_ok=True)
        failure=dict(status="failed",stage=stage,systems_only=stage=="pilot",fingerprint=plan["fingerprint"],
            error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,
            lease_id=None if lease is None else lease["id"],allocation=None if lease is None else lease["allocation"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda_started else None)
        recipe.write(directory/"failure.json",failure)
        raise
    finally:
        if dataset is not None:dataset.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","pilot","train"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--arm",choices=recipe.ARMS,default="softmax")
    parser.add_argument("--initialization",choices=("selected","native_projected"),default="selected")
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    args=parser.parse_args()
    if args.stage=="prepare":
        if args.source is None:parser.error("prepare requires --source")
        result=prepare(args.root,args.source,arm=args.arm,initialization=args.initialization,
            config=JointConfig(epochs=args.epochs,learning_rate=args.learning_rate))
    else:result=run(args.root,args.stage)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
