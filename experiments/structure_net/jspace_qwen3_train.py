"""Output-only Qwen3 suffix recovery with declared CPU gradient accumulation.

The caller owns model/data provenance, lease acquisition, pilot exclusion and
fresh-process verification. No prefix is trainable or retained here. BF16 leaf
gradients are accumulated on CPU in BF16, then chunked FP32 CPU Adam updates
are copied back. This is a declared arithmetic mode, not a bitwise claim about
CUDA reduction/accumulation. No optimizer step runs inside a backward hook.
"""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .jspace_qwen3_block import Qwen3CompactStack
from .jspace_qwen_parallel_train import RecoveryConfig, cpu_module_clone, recursive_cpu, output_objective
from .jspace_streaming_adamw import StreamingCPUAdamW

DEFAULT_CONFIG = RecoveryConfig(seed=20861, max_cpu_state_bytes=48 * 1024**3)
GRADIENT_MODE = "cpu_parameter_dtype"
SELECTION = "highest_correct_then_lowest_full_vocab_ce_earlier_tie"


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(named):
    value = hashlib.sha256(); size = 0
    for name, tensor in named:
        value.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        flat = tensor.detach().reshape(-1); step = max(1, 32*1024**2 // tensor.element_size())
        for start in range(0, flat.numel(), step):
            value.update(flat[start:start+step].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        size += tensor.numel() * tensor.element_size()
    return dict(sha256=value.hexdigest(), tensor_bytes=size)


def fixed_state(stack, readout, rotary):
    return fingerprint([("stack."+n,v) for n,v in stack.named_buffers()] + [("readout", readout)] +
                       [("rotary."+n,v) for n,v in rotary.named_buffers()])


def available_host_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"): return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot establish available host memory")


def require_host_memory(additional_bytes, *, reserve_bytes=1024**3):
    available = available_host_bytes()
    if additional_bytes + reserve_bytes > available:
        raise MemoryError(f"Host stage requires {additional_bytes} additional bytes plus {reserve_bytes} reserve; available={available}")
    return dict(additional_bytes=additional_bytes, reserve_bytes=reserve_bytes, available_bytes=available)


def memory_plan(stack, norm, config):
    parameters = [*stack.parameters(), *norm.parameters()]
    state = 12 * sum(p.numel() for p in parameters)
    gradients = sum(p.numel()*p.element_size() for p in parameters)
    model = sum(t.numel()*t.element_size() for m in (stack,norm) for t in list(m.parameters())+list(m.buffers()))
    workspace = 6 * config.optimizer_chunk_bytes
    return dict(cpu_optimizer_state_bytes=state, cpu_gradient_bank_bytes=gradients,
        combined_cpu_optimizer_gradient_bytes=state+gradients,
        live_suffix_tensor_bytes=model, conservative_packed_evaluation_extra_bytes=2*model+workspace,
        conservative_host_peak_bytes=state+gradients+2*model+workspace,
        gradient_mode=GRADIENT_MODE, gpu_persistent_parameter_gradients=False,
        note="Host peak estimate excludes caller-owned data/readout and allocator overhead; live MemAvailable checked separately")


class CPUGradientAdamW(StreamingCPUAdamW):
    """Same chunked master Adam arithmetic, consuming a parameter-dtype CPU bank."""
    def __init__(self, params, *, max_cpu_state_bytes=None, **kwargs):
        params = list(params)
        total = sum((12+p.element_size())*p.numel() for p in params)
        if max_cpu_state_bytes is not None and total > max_cpu_state_bytes:
            raise MemoryError(f"Combined CPU masters/moments/gradient bank requires {total} bytes")
        self.cpu_gradients, self.received, self.handles = {}, {}, []
        self.offload_bytes = 0
        super().__init__(params, max_cpu_state_bytes=None, **kwargs)
        for parameter in self._parameters:
            self.handles.append(parameter.register_post_accumulate_grad_hook(self._capture))

    @torch.no_grad()
    def _capture(self, parameter):
        gradient = parameter.grad
        if gradient is None or gradient.is_sparse or not gradient.is_contiguous():
            raise ValueError("Dense contiguous completed leaf gradients required")
        if gradient.dtype != parameter.dtype:
            raise ValueError("Gradient accumulation dtype must match the parameter")
        if parameter not in self.cpu_gradients:
            self.cpu_gradients[parameter] = torch.zeros_like(parameter, device="cpu")
            self.received[parameter] = 0
        target = self.cpu_gradients[parameter].view(-1)
        for start, end in self._chunks(parameter):
            stage = self._buffer(parameter.dtype)[:end-start]
            stage.copy_(gradient.detach().view(-1)[start:end], non_blocking=False)
            target[start:end].add_(stage)
            if gradient.device.type == "cuda": self.offload_bytes += (end-start)*gradient.element_size()
        self.received[parameter] += 1
        parameter.grad = None

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.cpu_gradients.clear(); self.received.clear()

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear(); self.zero_grad()

    @torch.no_grad()
    def gradient_norm(self, grad_scale=1.):
        if not math.isfinite(grad_scale) or grad_scale <= 0: raise ValueError("Invalid grad_scale")
        squared = torch.zeros((), dtype=torch.float32)
        for parameter, gradient in self.cpu_gradients.items():
            flat = gradient.view(-1)
            for start,end in self._chunks(parameter):
                squared.add_((torch.linalg.vector_norm(flat[start:end].float()) / grad_scale).square())
        result = math.sqrt(float(squared))
        if not math.isfinite(result):
            raise FloatingPointError("Nonfinite accumulated gradient; no optimizer state was updated")
        return result

    @torch.no_grad()
    def step(self, closure=None, *, max_grad_norm=None, grad_scale=1.):
        if closure is not None: raise ValueError("Closures unsupported")
        group = self._group()
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("Positive finite max_grad_norm required")
        norm = self.gradient_norm(grad_scale)
        clip = min(1., max_grad_norm/(norm+1e-6)) if max_grad_norm is not None else 1.
        beta1,beta2 = group["betas"]; changes=[]; updated=chunks=h2d=0
        for parameter in self._parameters:
            if parameter not in self.cpu_gradients:
                changes.append(False); continue
            state = self.state[parameter]; state["step"] += 1; step = state["step"]
            master, mean, square = (state[k].view(-1) for k in ("master","exp_avg","exp_avg_sq"))
            changed = False; gradient_bank = self.cpu_gradients[parameter].view(-1)
            for start,end in self._chunks(parameter):
                n=end-start; gradient=self._gradient[:n]
                gradient.copy_(gradient_bank[start:end]).mul_(clip/grad_scale)
                weight,m,v=master[start:end],mean[start:end],square[start:end]
                before=weight.clone()
                weight.mul_(1-group["lr"]*group["weight_decay"])
                m.lerp_(gradient,1-beta1); v.mul_(beta2).addcmul_(gradient,gradient,value=1-beta2)
                denominator=v.sqrt().div_(math.sqrt(1-beta2**step)).add_(group["eps"])
                weight.addcdiv_(m,denominator,value=-group["lr"]/(1-beta1**step))
                changed |= not torch.equal(before,weight)
                del before,denominator
                stage=self._buffer(parameter.dtype)[:n]; stage.copy_(weight)
                parameter.view(-1)[start:end].copy_(stage,non_blocking=False)
                if parameter.device.type=="cuda": h2d += n*parameter.element_size()
                chunks += 1
            changes.append(changed); updated += parameter.numel()
        self.transferred_bytes["weight_host_to_device"] += h2d
        self.transferred_bytes["gradient_device_to_host"] = self.offload_bytes
        self.last_step=dict(unclipped_gradient_norm=norm,clip_coefficient=clip,grad_scale=grad_scale,
            updated_scalars=updated,chunks=chunks,master_changed_by_parameter=changes,
            host_to_device_bytes=h2d,gradient_mode=GRADIENT_MODE,**self.memory_stats())
        return self.last_step

    def memory_stats(self):
        base=super().memory_stats()
        size=sum(v.numel()*v.element_size() for v in self.cpu_gradients.values())
        return {**base,"cpu_gradient_bank_bytes":size,"combined_cpu_optimizer_gradient_bytes":base["cpu_state_bytes"]+size,
                "maximum_master_change_work_bytes":self.chunk_bytes}


class Qwen3SuffixTrainer:
    def __init__(self, stack, norm, readout_weight, rotary, dataset, label_ids, *,
                 config=DEFAULT_CONFIG, guard=None, gradient_mode=GRADIENT_MODE, host_reserve_bytes=1024**3):
        if not isinstance(stack,Qwen3CompactStack) or stack.packed or stack.stripped:
            raise ValueError("Unpacked unstripped Qwen3CompactStack required")
        if gradient_mode != GRADIENT_MODE or config.dtype not in ("float32","bfloat16"):
            raise ValueError("Explicit CPU-parameter-dtype gradient accumulation and FP32/BF16 execution required")
        self.stack,self.norm,self.readout,self.rotary=stack,norm,readout_weight,rotary
        self.dataset,self.config,self.guard=dataset,config,guard
        self.device=next(stack.parameters()).device; self.dtype=getattr(torch,config.dtype)
        self.host_reserve_bytes=host_reserve_bytes
        if self.device.type=="cuda" and guard is None: raise ValueError("CUDA requires the experiment reservation guard")
        self._guard()
        if dataset.width!=stack.width or readout_weight.shape!=(dataset.vocabulary,stack.width) or readout_weight.device!=self.device or readout_weight.dtype!=self.dtype:
            raise ValueError("Boundary/readout dimensions, device or precision mismatch")
        if any(b.position_count>1 and max(dataset.lengths)>b.position_count for b in stack.blocks):
            raise ValueError("Dataset exceeds anchor table capacity")
        if not label_ids or any(i<0 or i>=dataset.vocabulary for i in label_ids): raise ValueError("Invalid label ids")
        if any(not 0<=c["answer"]<len(c["choices"])<=len(label_ids) or c["targets"][-1]!=label_ids[c["answer"]] for c in dataset.items):
            raise ValueError("Choice labels disagree with supervised target")
        self.label_ids=torch.tensor(label_ids,device=self.device)
        self.readout.requires_grad_(False); self.rotary.requires_grad_(False)
        self.stack.checkpoint_blocks=config.checkpoint_blocks
        self.stack.requires_grad_(True); self.norm.requires_grad_(True)
        # Parameters only: geometric buffers must remain FP32. Polynomial bias
        # stays FP32, matching the compiler's explicit arithmetic convention.
        for name,p in self.stack.named_parameters():
            p.data=p.data.to(device=self.device,dtype=torch.float32 if name.endswith("polynomial_bias") else self.dtype)
        for p in self.norm.parameters(): p.data=p.data.to(device=self.device,dtype=self.dtype)
        self.parameters=[*self.stack.parameters(),*self.norm.parameters()]
        self.parameter_ids=tuple(map(id,self.parameters))
        self.memory=memory_plan(stack,norm,config)
        self.host_admission=require_host_memory(self.memory["conservative_host_peak_bytes"],reserve_bytes=host_reserve_bytes)
        self.optimizer=CPUGradientAdamW(self.parameters,lr=config.learning_rate,weight_decay=0.,
            chunk_bytes=config.optimizer_chunk_bytes,max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.updates=0; self.finished=False

    def _guard(self):
        if self.guard is not None: self.guard()

    def _identity(self):
        if tuple(map(id,[*self.stack.parameters(),*self.norm.parameters()]))!=self.parameter_ids:
            raise RuntimeError("Module transfer changed parameter identity or gradient hooks")

    def logits(self,batch,*,stack=None,norm=None):
        stack,norm=stack or self.stack,norm or self.norm
        hidden=batch["hidden"]; positions=self.rotary(hidden,batch["position_ids"])
        output=stack(hidden,attention_mask=batch["attention_mask"],position_ids=batch["position_ids"],
            cache_position=batch["position_ids"][0],position_embeddings=positions,use_cache=False)[0]
        selected=output[torch.arange(len(batch["indices"]),device=self.device)[:,None],batch["loss_positions"]]
        return F.linear(norm(selected),self.readout).float()

    def snapshot(self):
        require_host_memory(self.memory["live_suffix_tensor_bytes"],reserve_bytes=self.host_reserve_bytes)
        return {"stack":recursive_cpu(self.stack.state_dict()),"norm":recursive_cpu(self.norm.state_dict())}

    def train_window(self,indices,*,learning_rate=None):
        indices=list(indices)
        if self.finished: raise RuntimeError("Recovery finished")
        if not indices or any(i not in self.dataset.training for i in indices): raise ValueError("Training cases only")
        self._guard(); self._identity(); self.stack.train(); self.norm.train(); self.optimizer.zero_grad()
        if learning_rate is not None: self.optimizer.param_groups[0]["lr"]=learning_rate
        totals=dict(loss=0.,kl=0.,ce=0.); expected=math.ceil(len(indices)/self.config.microbatch)
        for start in range(0,len(indices),self.config.microbatch):
            self._guard(); chosen=indices[start:start+self.config.microbatch]
            batch=self.dataset.batch(chosen,device=self.device,dtype=self.dtype)
            logits=self.logits(batch); loss,kl,ce=output_objective(logits,batch["teacher"],batch["targets"],self.config)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite distillation loss")
            weight=len(chosen)/len(indices); (loss*weight).backward()
            for key,value in zip(totals,(loss,kl,ce)): totals[key]+=float(value.detach())*weight
            del batch,logits,loss,kl,ce
        present=[all(self.optimizer.received.get(p)==expected for p in block.parameters()) for block in self.stack.blocks]
        norm_present=all(self.optimizer.received.get(p)==expected for p in self.norm.parameters())
        if not all(present) or not norm_present or any(p.grad is not None for p in self.parameters):
            raise RuntimeError("Missing gradient hooks or retained device gradients")
        update=self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        changes=dict(zip(self.parameters,update["master_changed_by_parameter"]))
        changed=[any(changes[p] for p in block.parameters()) for block in self.stack.blocks]
        norm_changed=any(changes[p] for p in self.norm.parameters())
        self.optimizer.zero_grad(); self.updates+=1; self._guard()
        return {**totals,"cases":len(indices),"update":self.updates,"learning_rate":self.optimizer.param_groups[0]["lr"],
            "gradients_present_by_block":present,"norm_gradients_present":norm_present,
            "fp32_master_changed_by_block":changed,"norm_fp32_master_changed":norm_changed,"optimizer":update}

    @torch.no_grad()
    def packed_development(self):
        self._guard(); self.optimizer.zero_grad(); self._identity()
        require_host_memory(self.memory["conservative_packed_evaluation_extra_bytes"],reserve_bytes=self.host_reserve_bytes)
        clone=packed_norm=None
        # GPU holds one suffix at a time. CPU masters refer to these same leaf
        # Parameter objects; .to() must preserve their registered hook identity.
        self.stack.to("cpu"); self.norm.to("cpu"); self._identity()
        if self.device.type=="cuda": torch.cuda.empty_cache()
        totals=dict(loss=0.,kl=0.,ce=0.,original_kl=0.,accepted_original_kl=0.)
        rows=[]; count=correct=accepted_agreement=original_agreement=0
        try:
            clone=cpu_module_clone(self.stack).pack(dtype=self.dtype,buffer_dtype=torch.float32,strip_unused=True).eval()
            clone.checkpoint_blocks=False; packed_norm=cpu_module_clone(self.norm).eval()
            self._guard(); clone.to(self.device); packed_norm.to(self.device)
            with torch.inference_mode():
                for start in range(0,len(self.dataset.development),self.config.microbatch):
                    self._guard(); indices=self.dataset.development[start:start+self.config.microbatch]
                    batch=self.dataset.batch(indices,device=self.device,dtype=self.dtype,include_original=True)
                    logits=self.logits(batch,stack=clone,norm=packed_norm)
                    loss,kl,ce=output_objective(logits,batch["teacher"],batch["targets"],self.config)
                    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite packed development loss")
                    for key,value in zip(("loss","kl","ce"),(loss,kl,ce)): totals[key]+=float(value)*len(indices)
                    if "original" in batch:
                        _,a,_=output_objective(logits,batch["original"],batch["targets"],self.config)
                        _,b,_=output_objective(batch["teacher"],batch["original"],batch["targets"],self.config)
                        totals["original_kl"]+=float(a)*len(indices); totals["accepted_original_kl"]+=float(b)*len(indices)
                    for row,index in enumerate(indices):
                        case=self.dataset.items[index]; labels=self.label_ids[:len(case["choices"])]; values=logits[row,-1,labels]
                        prediction=int(values.argmax()); correct+=prediction==case["answer"]
                        accepted_agreement+=prediction==int(batch["teacher"][row,-1,labels].argmax())
                        if "original" in batch: original_agreement+=prediction==int(batch["original"][row,-1,labels].argmax())
                        rows.append(dict(identity=case["identity"],family=case["family"],answer=case["answer"],
                            prediction=prediction,logits=values.cpu().tolist()))
                    count+=len(indices)
                    del batch,logits,loss,kl,ce
        finally:
            del clone,packed_norm; gc.collect()
            if self.device.type=="cuda": torch.cuda.empty_cache()
            self._guard(); self.stack.to(self.device); self.norm.to(self.device); self._identity()
        metrics={k:v/count for k,v in totals.items()}
        if self.dataset.original is None: metrics["original_kl"]=metrics["accepted_original_kl"]=None
        metrics.update(correct=correct,count=count,accuracy=correct/count,accepted_choice_agreement=accepted_agreement/count,
            original_choice_agreement=original_agreement/count if self.dataset.original is not None else None,
            selection_metric=SELECTION,precision=self.config.dtype,
            by_family={f:dict(correct=sum(r["prediction"]==r["answer"] for r in rows if r["family"]==f),
                count=sum(r["family"]==f for r in rows)) for f in sorted({r["family"] for r in rows})})
        return metrics,rows

    def fit(self,path,*,event=None,initial_reference=None):
        if self.updates or self.finished: raise ValueError("Fit requires untouched initializer; discard pilot instances")
        path=Path(path); path.mkdir(parents=True,exist_ok=False); event=event or (lambda *a,**k:None)
        before=fixed_state(self.stack,self.readout,self.rotary); history=[]; best=None; best_epoch=None
        started=time.perf_counter(); torch.manual_seed(self.config.seed)
        effective=self.config.microbatch*self.config.accumulation
        total=self.config.epochs*math.ceil(len(self.dataset.training)/effective)
        for epoch in range(self.config.epochs+1):
            training=[]
            if epoch:
                order=list(self.dataset.training); random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0,len(order),effective):
                    lr=self.config.learning_rate*.5*(1+math.cos(math.pi*self.updates/total))
                    update=self.train_window(order[start:start+effective],learning_rate=lr); training.append(update)
                    event("update",epoch=epoch,**update)
            metrics,records=self.packed_development()
            if epoch==0 and initial_reference is not None and (metrics!=initial_reference["metrics"] or records!=initial_reference["records"]):
                raise RuntimeError("Initializer differs from discarded pilot before any update")
            development=path/f"epoch_{epoch:02d}_development.json"; _write(development,dict(metrics=metrics,records=records))
            changed=best is None or (metrics["correct"],-metrics["ce"])>(best["correct"],-best["ce"])
            row=dict(epoch=epoch,metrics=metrics,training=training,development_sha256=sha(development),checkpoint_updated=changed)
            if changed:
                pending=path/"selected.pt.pending"
                with pending.open("xb") as stream: torch.save(self.snapshot(),stream)
                os.replace(pending,path/"selected.pt")
                best,best_epoch=metrics,epoch; row["checkpoint_sha256"]=sha(path/"selected.pt")
            history.append(row); _write(path/f"epoch_{epoch:02d}.json",row)
            event("development",epoch=epoch,selected_epoch=best_epoch,**metrics)
        if self.updates!=total: raise RuntimeError("Wrong update budget")
        state=torch.load(path/"selected.pt",map_location="cpu",mmap=True,weights_only=True)
        self.stack.load_state_dict(state["stack"],strict=True); self.norm.load_state_dict(state["norm"],strict=True)
        del state
        final,records=self.packed_development(); reference=json.loads((path/f"epoch_{best_epoch:02d}_development.json").read_text())
        if final!=best or records!=reference["records"]: raise RuntimeError("Selected packed reload changed results")
        if fixed_state(self.stack,self.readout,self.rotary)!=before: raise RuntimeError("Fixed geometry/readout changed")
        self.finished=True
        result=dict(status="completed",confirmed=False,completed_epochs=self.config.epochs,updates=self.updates,
            config=asdict(self.config),gradient_mode=GRADIENT_MODE,initial=history[0]["metrics"],selected=best,
            selected_epoch=best_epoch,initializer_selected=best_epoch==0,history=history,
            selected_checkpoint_sha256=sha(path/"selected.pt"),strict_reload_exact=True,fixed_state_unchanged=True,
            fixed_state=before,memory_plan=self.memory,optimizer_memory=self.optimizer.memory_stats(),
            host_admission=self.host_admission,elapsed_seconds=time.perf_counter()-started,
            scope="Cached-frozen-prefix joint suffix output distillation; development only; fresh verification is caller-owned")
        _write(path/"result.json",result); return result
