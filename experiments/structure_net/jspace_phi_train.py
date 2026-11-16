"""Single final-block Phi recovery with complete two-token MC responses.

The caller owns the frozen study plan, fit/source admission and GPU lease.
Only the compiled block learns: native prefix, geometry, final norm and untied
head remain fixed. Cached teacher states are BEFORE that original final norm.
This bounded pilot makes no whole-model, generation or packed-runtime claim.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from experiments.neural_architecture_lab import distillation_recovery as nal
from .jspace_incremental_suffix import fingerprint, state_equal
from .jspace_phi_affine_block import AffinePhiBlock
from .jspace_qwen_affine_stack import forward_reduced
from .jspace_recovery_data import digest, read, sha, write
from .jspace_streaming_adamw import StreamingCPUAdamW

response_data = nal.module("distillation_data")
response_training = nal.module("distillation_training")
SCHEMA = "jspace.phi-single-block-training.v1"


@dataclass(frozen=True)
class PhiTrainConfig:
    epochs: int = 3
    accumulation: int = 4
    learning_rate: float = 2e-5
    temperature: float = 2.
    kd_weight: float = .7
    ce_weight: float = .3
    feature_weight: float = 1.
    gradient_clip: float = 1.
    seed: int = 20861
    token_chunk_size: int = 1
    optimizer_chunk_bytes: int = 32 * 1024**2
    max_cpu_state_bytes: int = 4 * 1024**3

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.epochs, self.accumulation,
                self.token_chunk_size, self.optimizer_chunk_bytes, self.max_cpu_state_bytes)):
            raise ValueError("Positive epoch, accumulation, token-chunk and memory settings required")
        if (any(not math.isfinite(v) or v <= 0 for v in (self.learning_rate, self.temperature,
                self.gradient_clip)) or any(not math.isfinite(v) or v < 0 for v in
                (self.kd_weight, self.ce_weight, self.feature_weight)) or self.kd_weight+self.ce_weight <= 0):
            raise ValueError("Finite explicit loss coefficients and learning rate required")


def selection_key(row):
    """Higher DEV accuracy first; earliest checkpoint wins every accuracy tie."""
    return (-row["correct"], row["epoch"])


class PhiTrainingData:
    def __init__(self, pilot_root, response_path, *, expected_fingerprint=None,
                 require_population=True, layer=39, label_ids=None):
        self.root, response_path = Path(pilot_root).resolve(), Path(response_path).resolve()
        native = self.root/"native"
        plan, manifest = read(self.root/"plan.json"), read(native/"manifest.json")
        if (manifest.get("schema") != "jspace.phi-native.v1" or manifest.get("status") != "completed"
                or manifest.get("fingerprint") != plan["fingerprint"]
                or expected_fingerprint is not None and plan["fingerprint"] != expected_fingerprint):
            raise ValueError("Completed matching native Phi capture required")
        self.input_sha256 = {}
        for name, expected in manifest["artifacts"].items():
            relative = Path(name)
            path = (self.root/relative if relative.parts[:1] == ("native",) else native/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(native) or sha(path) != expected:
                raise ValueError("Native artifact path/hash changed")
            self.input_sha256[str(path)] = expected
        for name in ("items.json", "prefix.npy", "teacher_hidden.npy", "prompt_prefix.npy", "prompt_teacher_hidden.npy"):
            if str((native/name).resolve()) not in self.input_sha256:
                raise ValueError("Native manifest must bind items and both hidden-state banks")
        self.items = read(native/"items.json")
        if not isinstance(self.items, list):
            raise ValueError("Native Phi items must be the original ordered array")
        with response_path.open() as stream:
            self.records = [response_data.ResponseRecord(**json.loads(line)) for line in stream]
        self.inputs = np.load(native/"prefix.npy", mmap_mode="r")
        self.teacher = np.load(native/"teacher_hidden.npy", mmap_mode="r")
        self.prompt_inputs = np.load(native/"prompt_prefix.npy", mmap_mode="r")
        self.prompt_teacher = np.load(native/"prompt_teacher_hidden.npy", mmap_mode="r")
        if (self.inputs.dtype != np.int16 or self.teacher.dtype != np.int16
                or self.inputs.ndim != 2 or self.inputs.shape != self.teacher.shape
                or len(self.items) != len(self.records)):
            raise ValueError("Aligned lossless BF16 full-sequence native banks required")
        self.layer = layer; self.width = self.inputs.shape[1]
        if (self.prompt_inputs.dtype != np.int16 or self.prompt_teacher.dtype != np.int16
                or self.prompt_inputs.ndim != 2 or self.prompt_inputs.shape != self.prompt_teacher.shape
                or self.prompt_inputs.shape[1] != self.width):
            raise ValueError("Separate prompt-only native input/output banks required")
        if label_ids is None:
            label_plan = response_path.parent/"plan.json"
            label_ids = read(label_plan)["label_ids"]
            self.input_sha256[str(label_plan)] = sha(label_plan)
        if len(label_ids) != 4 or len(set(label_ids)) != 4 or any(type(v) is not int or v < 0 for v in label_ids):
            raise ValueError("Four explicit distinct Phi label IDs required")
        self.label_ids = list(label_ids)
        cursor = prompt_cursor = 0
        groups = response_data.SplitGuard()
        for item, record in zip(self.items, self.records):
            groups.add(record)
            if (item["input_ids"] != list(record.prompt_ids) or item["identity"] != record.example_id
                    or item["split"] != record.split or item["family"] != record.task
                    or len(record.response_ids) != 2 or record.finish_reason != "eos"
                    or record.response_ids[-1] != record.eos_token_id
                    or record.response_ids[0] != self.label_ids[item["answer"]]
                    or item["offset"] != cursor or item["stop"] != cursor+len(record.prompt_ids)
                    or item["full_stop"] != item["stop"]+2 or item["prompt_offset"] != prompt_cursor
                    or item["prompt_stop"] != prompt_cursor+len(record.prompt_ids)):
                raise ValueError("Native full sequence differs from actual letter-plus-EOS response")
            cursor = item["full_stop"]
            prompt_cursor = item["prompt_stop"]
        if cursor != self.inputs.shape[0] or prompt_cursor != self.prompt_inputs.shape[0]:
            raise ValueError("Native bank offsets leave missing or trailing rows")
        self.training = [i for i,c in enumerate(self.items) if c["split"] == "train"]
        self.development = [i for i,c in enumerate(self.items) if c["split"] == "development"]
        if not self.training or not self.development or len(self.training)+len(self.development) != len(self.items):
            raise ValueError("Only nonempty TRAIN/DEV populations are allowed")
        if require_population and (len(self.training) != 128 or len(self.development) != 64
                or sorted(Counter(c["family"] for c in self.items if c["split"] == "train").values()) != [32]*4
                or sorted(Counter(c["family"] for c in self.items if c["split"] == "development").values()) != [16]*4):
            raise ValueError("Pilot requires 128 TRAIN and 64 DEV balanced over four families")
        for path in (self.root/"plan.json", native/"manifest.json", response_path):
            self.input_sha256[str(path)] = sha(path)
        self.fingerprint = digest(dict(native=plan["fingerprint"], inputs=self.input_sha256,
                                       layer=layer,label_ids=self.label_ids))

    def case(self, index, device):
        item, record = self.items[index], self.records[index]
        full = len(record.prompt_ids)+len(record.response_ids)
        result = response_data.response_batch([record], pad_token_id=record.eos_token_id,
            max_length=full, device=device)
        for key, bank in (("hidden", self.inputs), ("teacher_hidden", self.teacher)):
            result[key] = torch.from_numpy(np.array(bank[item["offset"]:item["full_stop"]], copy=True))
            result[key] = result[key].view(torch.bfloat16).unsqueeze(0).to(device)
        result["index"] = index
        return result

    def prompt_case(self,index,device):
        item=self.items[index]; result={}
        for key,bank in (("hidden",self.prompt_inputs),("teacher_hidden",self.prompt_teacher)):
            result[key]=torch.from_numpy(np.array(bank[item["prompt_offset"]:item["prompt_stop"]],copy=True))
            result[key]=result[key].view(torch.bfloat16).unsqueeze(0).to(device)
        result["position_ids"]=torch.arange(len(item["input_ids"]),device=device).unsqueeze(0)
        return result

    def close(self):
        for bank in (self.inputs, self.teacher,self.prompt_inputs,self.prompt_teacher):
            if not bank._mmap.closed:
                bank._mmap.close()


def forward_coordinates(block, hidden, positions, rotary):
    location = block._positions(hidden, positions[0], positions, None)
    z = block.encode(hidden)
    return forward_reduced(block, z, location, cache_position=positions[0],
                           position_embeddings=rotary(hidden, positions))[0], location


def decoded_rows(block, z, location, predicting_positions):
    """Same native-width exit as AffinePhiBlock, only at supervised rows."""
    anchor = block._rows(block.anchor, location, *z.shape[:2])
    rows = predicting_positions.to(z.device)
    value = F.linear(z[rows[:,0],rows[:,1]].to(block.basis.dtype), block.basis)
    return (value+anchor[rows[:,0],rows[:,1]]).to(z.dtype)


def _state(module):
    return {k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else deepcopy(v)
            for k,v in module.state_dict().items()}


class PhiTrainer:
    def __init__(self, block, norm, head, rotary, data, *, config=PhiTrainConfig(),
                 guard=None, identities=None, native_block_bytes=None):
        if not isinstance(block, AffinePhiBlock) or block.packed or block.layer_idx != data.layer:
            raise ValueError("An unpacked compiled final Phi block is required")
        self.block, self.norm, self.head, self.rotary, self.data = block, norm, head, rotary, data
        self.config, self.guard = config, guard
        self.device = next(block.parameters()).device
        if self.device.type == "cuda" and guard is None:
            raise ValueError("Live reservation guard required for Phi training")
        self._guard()
        if (block.width != data.width or head.ndim != 2 or head.shape[1] != data.width
                or head.device != self.device or head.dtype != torch.bfloat16
                or norm.weight.dtype != torch.bfloat16 or norm.weight.device != self.device):
            raise ValueError("Matching BF16 block, frozen native norm and untied readout required")
        if max(len(r.prompt_ids)+len(r.response_ids) for r in data.records) > block.position_count and block.position_count != 1:
            raise ValueError("Full assistant sequence exceeds fitted position capacity")
        block.requires_grad_(True); block.set_capture_mlp(False)
        for p in block.parameters(): p.data = p.data.to(torch.bfloat16); p.grad = None
        norm.eval().requires_grad_(False); norm.zero_grad(set_to_none=True)
        head.requires_grad_(False); head.grad = None
        rotary.eval().requires_grad_(False); rotary.zero_grad(set_to_none=True)
        self.parameters = list(block.parameters())
        self.parameter_ids = tuple(map(id,self.parameters))
        self.identities = dict(identities or {}, data=data.fingerprint,
            initializer=fingerprint(block.named_parameters())["sha256"])
        self.fixed = self.fixed_state()
        self.optimizer = StreamingCPUAdamW(self.parameters,lr=config.learning_rate,weight_decay=0.,
            chunk_bytes=config.optimizer_chunk_bytes,max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.loss_config = response_training.DistillationConfig(temperature=config.temperature,
            kd_weight=config.kd_weight,ce_weight=config.ce_weight,token_chunk_size=config.token_chunk_size)
        self.updates, self.finished = 0, False
        self.normalization = self.training_energy()
        self.native_block_bytes = native_block_bytes

    def _guard(self):
        if self.guard is not None: self.guard()

    def fixed_state(self):
        return fingerprint([("block."+n,v) for n,v in self.block.named_buffers()]
            +[("norm."+n,v) for n,v in [*self.norm.named_parameters(),*self.norm.named_buffers()]]
            +[("head",self.head)]+[("rotary."+n,v) for n,v in self.rotary.named_buffers()])

    def unchanged(self):
        return (self.fixed_state() == self.fixed and tuple(map(id,self.block.parameters())) == self.parameter_ids
                and self.head.grad is None and all(p.grad is None for p in self.norm.parameters()))

    @torch.no_grad()
    def training_energy(self):
        energies = []
        for i in self.data.training:
            self._guard(); case = self.data.case(i,self.device)
            energies.append(float(self.block.encode(case["teacher_hidden"]).float().square().mean()))
        mean = sum(energies)/len(energies)
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError("Degenerate TRAIN-only projected teacher energy")
        return dict(energy=mean,denominator=max(mean,1e-8),training_cases=len(energies),
            scope="Equal-case mean over all valid full-sequence tokens; TRAIN only; fixed once")

    def _readout(self, hidden):
        return F.linear(self.norm(hidden),self.head)

    def losses(self, case, *, backward=False, scale=1.):
        response_training.validate_batch(case,training=backward)
        z, location = forward_coordinates(self.block,case["hidden"],case["position_ids"],self.rotary)
        target = self.block.encode(case["teacher_hidden"]).detach()
        feature = (z.float()-target.float()).square().mean()/self.normalization["denominator"]
        positions = case["loss_mask"].nonzero()
        if len(positions) != 2:
            raise ValueError("This pilot supervises both actual response tokens exactly once")
        selected = decoded_rows(self.block,z,location,positions)
        bridge = selected.detach().requires_grad_(backward)
        teacher = case["teacher_hidden"][positions[:,0],positions[:,1]].detach()
        targets = case["targets"][positions[:,0],positions[:,1]]
        totals = dict(endpoint=0.,kl_t2=0.,ce=0.,feature=float(feature.detach()))
        logits_rows, teacher_rows = [], []
        for start in range(0,len(positions),self.config.token_chunk_size):
            stop = min(start+self.config.token_chunk_size,len(positions))
            with torch.no_grad(): teacher_logits = self._readout(teacher[start:stop])
            logits = self._readout(bridge[start:stop])
            values = response_training.loss_vectors(logits,teacher_logits,targets[start:stop],self.loss_config)
            if backward: (values["loss"].sum()*scale/len(positions)).backward()
            totals["endpoint"] += float(values["loss"].detach().sum())/len(positions)
            totals["kl_t2"] += float(values["forward_kl"].detach().sum())/len(positions)
            totals["ce"] += float(values["ce"].detach().sum())/len(positions)
            logits_rows.append(logits.detach()); teacher_rows.append(teacher_logits.detach())
        if backward:
            if bridge.grad is None or not torch.isfinite(bridge.grad).all():
                raise FloatingPointError("Response readout produced no finite hidden gradient")
            torch.autograd.backward((feature,selected),
                (feature.new_tensor(self.config.feature_weight*scale),bridge.grad))
        totals["loss"] = totals["endpoint"]+self.config.feature_weight*totals["feature"]
        if not all(math.isfinite(v) for v in totals.values()): raise FloatingPointError("Nonfinite Phi objective")
        return totals,torch.cat(logits_rows),torch.cat(teacher_rows)

    def train_window(self, indices):
        if self.finished or not indices or any(i not in self.data.training for i in indices):
            raise ValueError("Fresh trainer and TRAIN-only update cases required")
        self._guard(); self.block.train(); self.optimizer.zero_grad(set_to_none=True)
        values = []
        for i in indices:
            self._guard(); values.append(self.losses(self.data.case(i,self.device),backward=True,scale=1/len(indices))[0])
        if any(p.grad is None for p in self.parameters): raise RuntimeError("Some compiled block parameters lack gradients")
        update = self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        self.optimizer.zero_grad(set_to_none=True); self.updates += 1
        if update["updated_scalars"] != sum(p.numel() for p in self.parameters):
            raise RuntimeError("Optimizer did not update the complete compact block")
        return dict(update=self.updates,cases=len(indices),supervised_tokens=2*len(indices),
            **{k:sum(v[k] for v in values)/len(values) for k in values[0]},optimizer=update)

    @torch.no_grad()
    def development(self):
        self.block.eval(); records=[]; totals={}; native_correct=0
        for i in self.data.development:
            self._guard(); case=self.data.case(i,self.device)
            values,logits,teacher=self.losses(case)
            item=self.data.items[i]
            prompt=self.data.prompt_case(i,self.device)
            z,location=forward_coordinates(self.block,prompt["hidden"],prompt["position_ids"],self.rotary)
            last=torch.tensor([[0,z.shape[1]-1]],device=self.device)
            prompt_logits=self._readout(decoded_rows(self.block,z,location,last))[0]
            native_logits=self._readout(prompt["teacher_hidden"][:,-1])[0]
            label_ids=self.data.label_ids[:len(item["choices"])]
            choice=prompt_logits[label_ids].float(); native=native_logits[label_ids].float()
            predicted=int(choice.argmax()); native_predicted=int(native.argmax())
            correct=predicted == item["answer"]; native_correct += native_predicted == item["answer"]
            records.append(dict(index=i,identity=item["identity"],family=item["family"],answer=item["answer"],
                prediction=predicted,native_prediction=native_predicted,correct=correct,
                task_scoring="separately executed prompt-only native prefix; no gold response context",
                choice_logits=choice.cpu().tolist(),native_choice_logits=native.cpu().tolist(),
                response_ce=F.cross_entropy(logits.float(),case["targets"][case["loss_mask"]],reduction="none").cpu().tolist(),
                response_tokens=case["targets"][case["loss_mask"]].cpu().tolist()))
            for k,v in values.items():totals[k]=totals.get(k,0.)+v
        count=len(records);correct=sum(r["correct"] for r in records)
        families={f:dict(count=len(rows:=[r for r in records if r["family"]==f]),correct=sum(r["correct"] for r in rows),
            native_correct=sum(r["native_prediction"]==r["answer"] for r in rows)) for f in sorted({r["family"] for r in records})}
        return dict(count=count,correct=correct,accuracy=correct/count,native_correct=native_correct,
            native_accuracy=native_correct/count,native_agreement=sum(r["prediction"]==r["native_prediction"] for r in records)/count,
            **{k:v/count for k,v in totals.items()},families=families,records=records)

    def pilot(self, indices):
        if self.updates or self.finished:raise ValueError("Pilot must use a disposable fresh trainer")
        before=fingerprint([(n,self.optimizer.state[p]["master"]) for n,p in self.block.named_parameters()])
        update=self.train_window(indices)
        after=fingerprint([(n,self.optimizer.state[p]["master"]) for n,p in self.block.named_parameters()])
        self.finished=True;fixed=self.unchanged()
        return dict(passed=before!=after and fixed and update["optimizer"]["unclipped_gradient_norm"]>0,
            systems_only=True,discarded=True,update=update,master_before=before,master_after=after,
            fixed_state_unchanged=fixed)

    def _save(self,path,epoch):
        payload=dict(schema=SCHEMA,block=_state(self.block),epoch=epoch,updates=self.updates,
            config=asdict(self.config),identities=self.identities,fixed=self.fixed,normalization=self.normalization)
        with Path(path).open("xb") as stream:torch.save(payload,stream)
        return sha(path)

    def fit(self,directory,*,event=None):
        if self.updates or self.finished:raise ValueError("Fit requires the unchanged scientific initializer")
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        event=event or (lambda value:None);history=[];update_rows=[]
        for epoch in range(self.config.epochs+1):
            if epoch:
                order=list(self.data.training);random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0,len(order),self.config.accumulation):
                    value=self.train_window(order[start:start+self.config.accumulation]);value["epoch"]=epoch
                    update_rows.append(value);event(dict(phase="training",**value))
            report=self.development()
            if not self.unchanged():raise RuntimeError("Frozen Phi geometry/norm/readout changed")
            cp=directory/f"epoch_{epoch:02d}.pt";cp_sha=self._save(cp,epoch)
            rp=directory/f"development_{epoch:02d}.json";write(rp,report)
            row=dict(epoch=epoch,updates=self.updates,correct=report["correct"],ce=report["ce"],
                checkpoint=cp.name,checkpoint_sha256=cp_sha,report=rp.name,report_sha256=sha(rp))
            history.append(row);event(dict(phase="development",**row))
        selected=min(history,key=selection_key);self.finished=True
        load_selected(self,directory/selected["checkpoint"],selected["checkpoint_sha256"])
        actual=self.development();expected=read(directory/selected["report"])
        if actual!=expected:raise RuntimeError("Strict selected checkpoint reload changed DEV output")
        block_bytes=fingerprint([*self.block.named_parameters(),*self.block.named_buffers()])["tensor_bytes"]
        result=dict(schema=SCHEMA,status="completed",confirmed=False,config=asdict(self.config),
            identities=self.identities,input_sha256=self.data.input_sha256,normalization=self.normalization,
            epochs=self.config.epochs,updates=self.updates,supervised_tokens=sum(v["supervised_tokens"] for v in update_rows),
            selection="highest DEV correct; earliest epoch on all accuracy ties; initializer eligible; CE diagnostic only",
            history=history,selected=selected,metrics=actual,fixed=self.fixed,fixed_state_unchanged=self.unchanged(),
            internal_reload=dict(passed=True,strict_state_load=True,records_and_metrics_exact=True),
            storage=dict(compact_unpacked_block_bytes=block_bytes,native_decoded_block_bytes=self.native_block_bytes,
                frozen_norm_head_bytes=fingerprint([("head",self.head),*self.norm.named_parameters()])["tensor_bytes"],
                scope="One block only; no packed or whole-model compression measurement"),
            scope="128 TRAIN/64 DEV final-block pilot on decoded GPTQ source; no heldout/generation/speed claim")
        write(directory/"updates.json",dict(rows=update_rows));write(directory/"result.json",result)
        write(directory/"manifest.json",dict(status="completed",schema=SCHEMA,
            artifacts={p.name:sha(p) for p in directory.iterdir() if p.is_file()}))
        return result


def load_selected(trainer,path,expected_sha256):
    if sha(path)!=expected_sha256:raise ValueError("Selected Phi checkpoint changed")
    state=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    current=trainer.block.state_dict()
    if (state.get("schema")!=SCHEMA or state.get("identities")!=trainer.identities
            or state.get("config")!=asdict(trainer.config) or state.get("fixed")!=trainer.fixed
            or state.get("normalization")!=trainer.normalization or current.keys()!=state["block"].keys()
            or any(isinstance(v,torch.Tensor) and (v.shape!=current[k].shape or v.dtype!=current[k].dtype)
                   for k,v in state["block"].items())):
        raise ValueError("Selected Phi checkpoint geometry, precision or provenance differs")
    names=dict(trainer.block.named_buffers())
    if any(not torch.equal(v.cpu(),state["block"][k]) for k,v in names.items() if k in state["block"]):
        raise ValueError("Selected checkpoint changed frozen geometry")
    trainer.block.load_state_dict(state["block"],strict=True)
    if not state_equal(trainer.block.state_dict(),state["block"]) or not trainer.unchanged():
        raise ValueError("Strict Phi checkpoint reload changed state")
    return state


def verify_checkpoint(trainer,directory):
    """Call in a fresh process/trainer with identical admitted inputs and fits."""
    directory=Path(directory);manifest=read(directory/"manifest.json")
    if manifest.get("status")!="completed" or manifest.get("schema")!=SCHEMA:
        raise ValueError("Completed Phi training required before fresh verification")
    required={"result.json","updates.json"}|{f"epoch_{i:02d}.pt" for i in range(trainer.config.epochs+1)}|{
        f"development_{i:02d}.json" for i in range(trainer.config.epochs+1)}
    if set(manifest.get("artifacts",{}))!=required:
        raise ValueError("Complete initializer/epoch/checkpoint manifest required")
    for name,expected in manifest["artifacts"].items():
        if Path(name).name!=name or sha(directory/name)!=expected:raise ValueError("Training artifact changed")
    result=read(directory/"result.json");history=result["history"]
    expected_updates=trainer.config.epochs*math.ceil(len(trainer.data.training)/trainer.config.accumulation)
    if (result.get("schema")!=SCHEMA or result.get("status")!="completed" or result.get("config")!=asdict(trainer.config)
            or result["input_sha256"]!=trainer.data.input_sha256
            or len(history)!=trainer.config.epochs+1 or [v["epoch"] for v in history]!=list(range(trainer.config.epochs+1))
            or result["updates"]!=expected_updates or result["selected"]!=min(history,key=selection_key)
            or result["fixed_state_unchanged"] is not True or result["identities"]!=trainer.identities):
        raise ValueError("Training completion or earliest-accuracy selection differs")
    for row in history:
        report=read(directory/row["report"])
        if (row["checkpoint"]!=f"epoch_{row['epoch']:02d}.pt" or row["report"]!=f"development_{row['epoch']:02d}.json"
                or row["checkpoint_sha256"]!=manifest["artifacts"][row["checkpoint"]]
                or row["report_sha256"]!=manifest["artifacts"][row["report"]]
                or row["correct"]!=report["correct"] or row["ce"]!=report["ce"]
                or row["updates"]!=row["epoch"]*math.ceil(len(trainer.data.training)/trainer.config.accumulation)):
            raise ValueError("Epoch evidence disagrees with checkpoint/report history")
    updates=read(directory/"updates.json")["rows"]
    if (len(updates)!=expected_updates or [v["update"] for v in updates]!=list(range(1,expected_updates+1))
            or result["supervised_tokens"]!=2*trainer.config.epochs*len(trainer.data.training)
            or sum(v["supervised_tokens"] for v in updates)!=result["supervised_tokens"]
            or any(v["optimizer"]["updated_scalars"]!=sum(p.numel() for p in trainer.parameters) for v in updates)):
        raise ValueError("Actual optimizer/token budget is incomplete")
    for path,expected in result["input_sha256"].items():
        if sha(path)!=expected:raise ValueError("Native capture or response identities changed")
    winner=result["selected"]
    load_selected(trainer,directory/winner["checkpoint"],winner["checkpoint_sha256"])
    actual=trainer.development();expected=read(directory/winner["report"])
    if actual!=expected or actual!=result["metrics"]:raise ValueError("Fresh Phi DEV replay differs")
    return dict(status="completed",passed=True,strict_state_load=True,checkpoint_sha256=winner["checkpoint_sha256"],
        selected_epoch=winner["epoch"],metrics=actual,fixed_state_unchanged=trainer.unchanged(),
        records_and_metrics_exact=True,training_result_sha256=sha(directory/"result.json"),confirmed=False)
