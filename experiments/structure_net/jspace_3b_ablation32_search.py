"""Joint discarded-subspace search with native frozen Qwen and complete responses.

Raw coordinates, QR and Adam remain FP64. Detached FP32 application bases
accumulate gradients over a complete optimizer window, followed by one QR VJP
per interface. Original BF16 weights never train. The caller owns reservations,
native-model/cache authentication, data selections and append-only run roots.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict,dataclass
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import jspace_3b_ablation32_geometry as geometry
from .jspace_3b_ablation32_compile import projected_native_block, PROJECTION_SITES
from .jspace_3b_response_train import CachedHiddenTeacher, module
from .jspace_3b_response_cache import sha

nal=module("distillation_training")
SCHEMA="jspace.ablation32-search.v1"
GEOMETRY_SCHEMA="ablation32.geometry-bank.v1"


@dataclass(frozen=True)
class SearchConfig:
    retained: int = 1024
    epochs: int = 8
    learning_rate: float = .01
    cosine_floor: float = .15
    accumulation_examples: int = 32
    temperature: float = 2.
    gradient_clip: float = 1.
    token_chunk_size: int = 8
    max_length: int = 416
    basis_seed: int = 220101
    shuffle_seed: int = 220301
    cpu_threads: int = 8

    def __post_init__(self):
        if (any(type(getattr(self,k))is not int or getattr(self,k)<1 for k in
                ("retained","epochs","accumulation_examples","token_chunk_size","max_length","cpu_threads"))
                or not 0<self.cosine_floor<=1 or self.learning_rate<=0 or self.temperature<=0 or self.gradient_clip<=0):
            raise ValueError("Positive search settings and a cosine floor in(0,1] required")

    def loss(self):
        return nal.DistillationConfig(temperature=self.temperature,objective="forward_kl",ce_weight=0.,
            token_chunk_size=self.token_chunk_size,grad_clip_norm=self.gradient_clip)


def _write(path,value):
    with Path(path).open("x") as stream:json.dump(value,stream,sort_keys=True,allow_nan=False,indent=2)


def _save(path,value):
    with Path(path).open("xb") as stream:torch.save(value,stream)
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def epoch_windows(indices,epoch,config):
    order=list(indices);random.Random(config.shuffle_seed+epoch).shuffle(order)
    return [order[i:i+config.accumulation_examples] for i in range(0,len(order),config.accumulation_examples)]


def learning_rate_at(update,total_updates,config):
    if not 0<=update<total_updates:raise ValueError("Learning-rate update outside the fixed schedule")
    phase=update/max(1,total_updates-1)
    return config.learning_rate*(config.cosine_floor+(1-config.cosine_floor)*.5*(1+math.cos(math.pi*phase)))


def backend_discarded_basis(raw):
    """Same FP64 reduced-QR/DC embedding on a caller-selected backend.

    Keep this small functional adapter local while the probed geometry source
    is frozen. The raw CPU optimizer Parameter is never moved or replaced.
    """
    if raw.dtype!=torch.float64:raise ValueError("Search QR remains FP64 on every backend")
    q,r=torch.linalg.qr(raw,mode="reduced");diagonal=r.diagonal()
    tolerance=torch.finfo(r.dtype).eps*max(raw.shape)*r.detach().abs().amax()
    if not bool(torch.isfinite(r).all()) or bool((diagonal.detach().abs()<=tolerance).any()):
        raise ValueError("Discarded-space QR is nonfinite or rank deficient")
    return geometry._embed_mean_free(q*diagonal.sign().unsqueeze(0))


def validate_selections(dataset,*,production=False):
    train=list(dataset.training_indices);selection=list(dataset.selection_indices);all_indices=train+selection
    if (not train or not selection or len(set(all_indices))!=len(all_indices)
            or any(type(i)is not int or not 0<=i<len(dataset.records) for i in all_indices)):
        raise ValueError("Unique disjoint nonempty search and selection records required")
    if any(dataset.records[i].split!="train" or dataset.records[i].finish_reason!="eos" for i in all_indices):
        raise ValueError("Search and selection use complete observed-EOS TRAIN records only")
    if {dataset.records[i].group_id for i in train}&{dataset.records[i].group_id for i in selection}:
        raise ValueError("Search and selection source groups overlap")
    if production and (len(train)!=2500 or len(selection)!=500):
        raise ValueError("Scientific search requires2500 TRAIN calibration and500 disjoint TRAIN selection examples")
    return train,selection


class ProjectedModel(nn.Module):
    """Frozen native blocks with exactly three matched projection sites per block."""
    def __init__(self,native,anchors,*,max_length=416,guard=None):
        super().__init__();self.native=native;self.guard=guard;self.max_length=max_length
        self.layers=len(native.model.layers);self.width=native.config.hidden_size
        if (native.training or any(p.requires_grad for p in native.parameters())
                or any(p.dtype!=torch.bfloat16 or p.is_inference() for p in native.parameters())):
            raise ValueError("Frozen normal-tensor BF16 original model required")
        if native.lm_head.weight is not native.model.embed_tokens.weight:
            raise ValueError("The original tied embedding/readout must remain shared")
        self.device=native.lm_head.weight.device
        if self.device.type=="cuda" and guard is None:raise ValueError("Native CUDA model requires its caller-owned reservation guard")
        if anchors.shape!=(self.layers,max_length,self.width) or anchors.dtype!=torch.float32 or not torch.isfinite(anchors).all():
            raise ValueError("Finite TRAIN position anchors[all layers,capacity,native width] FP32 required")
        if guard is not None:guard()
        self.register_buffer("anchors",anchors.detach().to(self.device))
        self.discarded=[];self.checkpoint_blocks=True
        self.fixed=[(id(t),t._version) for t in (*native.parameters(),*native.buffers())]

    def check_fixed(self):
        if (self.native.training or any(p.requires_grad for p in self.native.parameters())
                or self.fixed!=[(id(t),t._version) for t in (*self.native.parameters(),*self.native.buffers())]):
            raise ValueError("Original native teacher weights or buffers changed")

    def hidden_forward(self,batch):
        if self.guard is not None:self.guard()
        ids=batch["input_ids"].to(self.device)
        if (ids.ndim!=2 or ids.shape[0]!=1 or not 1<ids.shape[1]<=self.max_length
                or "attention_mask" in batch and not bool(batch["attention_mask"].all())
                or len(self.discarded)!=self.layers):
            raise ValueError("One complete unpadded sequence and every application basis required")
        hidden=self.native.model.embed_tokens(ids)
        positions=torch.arange(ids.shape[1],device=self.device)[None]
        mask=torch.full((1,1,ids.shape[1],ids.shape[1]),torch.finfo(hidden.dtype).min,
                        device=self.device,dtype=hidden.dtype).triu(1)
        rotary=self.native.model.rotary_emb(hidden,positions)
        for index,block in enumerate(self.native.model.layers):
            def forward(value,D,block=block,index=index):
                return projected_native_block(block,value,D,self.anchors[index],position_ids=positions,
                    position_embeddings=rotary,attention_mask=mask,cache_position=positions[0])
            if self.checkpoint_blocks and torch.is_grad_enabled():
                # D is an explicit requires-grad input, including block0 where
                # the native embedding itself is frozen.
                hidden=checkpoint(forward,hidden,self.discarded[index],use_reentrant=True,preserve_rng_state=False)
            else:hidden=forward(hidden,self.discarded[index])
        return self.native.model.norm(hidden)

    def readout(self,rows):
        return F.linear(rows.to(self.native.lm_head.weight.dtype),self.native.lm_head.weight)


class SearchTrainer:
    def __init__(self,native,dataset,*,teacher_id,identities,guard=None,config=SearchConfig(),
                 systems_only=False,qr_device="cpu"):
        self.config,self.dataset,self.guard=config,dataset,guard
        self.identities=dict(identities);self.teacher_id=teacher_id;self.systems_only=systems_only
        if not teacher_id or not self.identities:raise ValueError("Pinned original teacher and dataset identities required")
        self.training,self.selection=validate_selections(dataset,production=not systems_only)
        self.qr_device=torch.device(qr_device)
        if self.qr_device.type=="cuda" and guard is None:raise ValueError("QR CUDA execution requires a reservation covering that device")
        torch.set_num_threads(config.cpu_threads)
        anchors=getattr(dataset,"anchors",None);self.zero_anchor_pilot=anchors is None
        if anchors is None:
            if not systems_only:raise ValueError("Scientific search requires authenticated TRAIN anchors")
            anchors=torch.zeros(len(native.model.layers),config.max_length,native.config.hidden_size,dtype=torch.float32)
        if not systems_only and (len(native.model.layers)!=36 or native.config.hidden_size!=2048 or config!=SearchConfig()):
            raise ValueError("Original36-block/2048-width model and prospective search recipe required")
        self.model=ProjectedModel(native,anchors,max_length=config.max_length,guard=guard)
        self.coordinates=nn.ModuleList([geometry.DiscardedSubspace(self.model.width,config.retained,
            seed=config.basis_seed+i,preserve_mean=True,dtype=torch.float64,device="cpu") for i in range(self.model.layers)])
        self.parameters=list(self.coordinates.parameters())
        self.optimizer=torch.optim.Adam(self.parameters,lr=config.learning_rate,weight_decay=0.,foreach=False)
        self.teacher=CachedHiddenTeacher(native.lm_head.weight,teacher_id=teacher_id)
        self.teacher.temperature=config.temperature
        self.loss_config=config.loss();self.graphs=[];self.handles=[];self.gradient_bank=[];self.gradient_calls=[]
        self.updates=self.supervised_tokens=self.examples=0
        self.total_updates=math.ceil(len(self.training)/config.accumulation_examples)*config.epochs
        self.finished=False

    def _guard(self):
        if self.guard is not None:self.guard()
        self.model.check_fixed()

    def _basis(self,coordinates,*,differentiable):
        context=torch.enable_grad() if differentiable else torch.no_grad()
        with context:
            if self.qr_device.type=="cpu":return coordinates.discarded_basis()
            # Save QR factors on host; only one interface and its workspace is
            # resident on the QR card during forward/backward.
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                raw=coordinates.raw.to(self.qr_device)
                return backend_discarded_basis(raw).to("cpu")

    def _clear_window(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.graphs=[];self.gradient_bank=[];self.gradient_calls=[]
        self.model.discarded=[]

    def prepare_window(self,*,differentiable):
        self._clear_window();self._guard();started=time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        for index,coordinates in enumerate(self.coordinates):
            D=self._basis(coordinates,differentiable=differentiable)
            self.graphs.append(D)
            leaf=D.detach().float().to(self.model.device).requires_grad_(differentiable)
            self.model.discarded.append(leaf)
            if differentiable:
                self.gradient_bank.append(torch.zeros_like(D,dtype=torch.float32,device="cpu"));self.gradient_calls.append(0)
                def collect(parameter,index=index):
                    grad=parameter.grad
                    if grad is None or not torch.isfinite(grad).all():raise FloatingPointError("Missing/nonfinite application-basis gradient")
                    self.gradient_bank[index].add_(grad.detach().to(device="cpu",dtype=torch.float32))
                    self.gradient_calls[index]+=1;parameter.grad=None
                self.handles.append(leaf.register_post_accumulate_grad_hook(collect))
        return time.monotonic()-started

    def train_window(self,indices):
        indices=list(indices)
        if (not indices or len(indices)>self.config.accumulation_examples or len(set(indices))!=len(indices)
                or any(i not in self.training for i in indices) or self.updates>=self.total_updates):
            raise ValueError("A bounded unique admitted TRAIN window is required")
        started=time.monotonic();qr_forward=self.prepare_window(differentiable=True)
        total_tokens=0;example_kl=0.;token_kl=0.;ce_sum=0.;model_seconds=0.
        try:
            for index in indices:
                batch=self.dataset.batch([index]);count=nal.validate_batch(batch)
                self.teacher.prepare(batch);self._guard();begin=time.monotonic()
                hidden=self.model.hidden_forward(batch)
                values=nal.chunked_backward(hidden,self.model.readout,self.teacher,batch,self.loss_config,
                    normalizer=count*len(indices))
                del hidden;self.teacher.clear_cache()
                model_seconds+=time.monotonic()-begin
                example_kl+=values["forward_kl"]/count;token_kl+=values["forward_kl"];ce_sum+=values["ce"];total_tokens+=count
            if any(count!=len(indices) for count in self.gradient_calls):
                raise RuntimeError("Every interface must receive each complete-response gradient exactly once")
            application_norms=[float(g.double().norm()) for g in self.gradient_bank]
            if any(not math.isfinite(n) or n<=0 for n in application_norms):
                raise FloatingPointError("An intended discarded space received no finite functional gradient")
            begin=time.monotonic()
            for D,grad in zip(self.graphs,self.gradient_bank):
                self._guard();D.backward(grad.to(dtype=torch.float64))
            qr_backward=time.monotonic()-begin
            if any(p.grad is None or p.grad.dtype!=torch.float64 or not torch.isfinite(p.grad).all() for p in self.parameters):
                raise FloatingPointError("QR chain did not produce finite FP64 raw-coordinate gradients")
            raw_norms=[float(p.grad.norm()) for p in self.parameters]
            norm=torch.nn.utils.clip_grad_norm_(self.parameters,self.config.gradient_clip,error_if_nonfinite=True)
            rate=learning_rate_at(self.updates,self.total_updates,self.config)
            for group in self.optimizer.param_groups:group["lr"]=rate
            begin=time.monotonic();self.optimizer.step();optimizer_seconds=time.monotonic()-begin
            if any(not torch.isfinite(p).all() for p in self.parameters):raise FloatingPointError("Nonfinite updated raw coordinates")
            self.updates+=1;self.examples+=len(indices);self.supervised_tokens+=total_tokens;self._guard()
            return dict(update=self.updates,examples=len(indices),supervised_tokens=total_tokens,
                cumulative_examples=self.examples,cumulative_supervised_tokens=self.supervised_tokens,
                per_example_kl=example_kl/len(indices),token_weighted_kl=token_kl/total_tokens,
                token_nll=ce_sum/total_tokens,learning_rate=rate,gradient_norm=float(norm),
                application_gradient_norms=application_norms,raw_gradient_norms=raw_norms,
                qr_forward_seconds=qr_forward,qr_backward_seconds=qr_backward,model_forward_backward_seconds=model_seconds,
                optimizer_seconds=optimizer_seconds,window_seconds=time.monotonic()-started,
                full_response_and_eos=True,gradient_calls=list(self.gradient_calls))
        finally:self.teacher.clear_cache();self._clear_window()

    @torch.no_grad()
    def evaluate(self,indices=None):
        indices=self.selection if indices is None else list(indices)
        if not indices or any(i not in self.selection for i in indices):raise ValueError("Only the held-out TRAIN selection set may rank search checkpoints")
        started=time.monotonic();qr_seconds=self.prepare_window(differentiable=False)
        example_kl=token_kl=ce=0.;tokens=0
        try:
            for index in indices:
                batch=self.dataset.batch([index]);count=nal.validate_batch(batch)
                self.teacher.prepare(batch);hidden=self.model.hidden_forward(batch)
                positions=batch["loss_mask"].nonzero();case_kl=0.
                for start in range(0,count,self.config.token_chunk_size):
                    pos=positions[start:start+self.config.token_chunk_size]
                    rows=hidden[pos[:,0].to(hidden.device),pos[:,1].to(hidden.device)]
                    target=batch["targets"][pos[:,0],pos[:,1]]
                    values=nal.loss_vectors(self.model.readout(rows),self.teacher(batch,pos),target,self.loss_config)
                    case_kl+=float(values["forward_kl"].sum());ce+=float(values["ce"].sum())
                example_kl+=case_kl/count;token_kl+=case_kl;tokens+=count
                del hidden;self.teacher.clear_cache()
            self._guard()
            return dict(examples=len(indices),supervised_tokens=tokens,per_example_kl=example_kl/len(indices),
                token_weighted_kl=token_kl/tokens,token_nll=ce/tokens,temperature=self.config.temperature,
                qr_seconds=qr_seconds,elapsed_seconds=time.monotonic()-started,
                selection_objective="mean across examples of complete-response T-squared full-vocabulary KL")
        finally:self.teacher.clear_cache();self._clear_window()

    def save_checkpoint(self,path,*,epoch,selection,save_optimizer=True):
        self._guard()
        return _save(path,dict(schema=SCHEMA,identities=self.identities,teacher_id=self.teacher_id,
            geometry=self.coordinates.state_dict(),configuration=asdict(self.config),epoch=epoch,
            updates=self.updates,supervised_tokens=self.supervised_tokens,selection=selection,
            examples=self.examples,optimizer=self.optimizer.state_dict() if save_optimizer else None,
            optimizer_saved=save_optimizer,resume_supported=False,
            projection_sites=PROJECTION_SITES,qr_device=str(self.qr_device),raw_precision="float64",application_precision="float32"))

    def load_checkpoint(self,path,*,expected_sha256):
        if sha(path)!=expected_sha256:raise ValueError("Search checkpoint changed")
        saved=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        if (saved.get("schema")!=SCHEMA or saved.get("identities")!=self.identities
                or saved.get("teacher_id")!=self.teacher_id or saved.get("configuration")!=asdict(self.config)
                or saved.get("projection_sites")!=PROJECTION_SITES):
            raise ValueError("Search checkpoint provenance or objective differs")
        self._clear_window();self.coordinates.load_state_dict(saved["geometry"],strict=True)
        return saved

    def export(self,directory,*,epoch,checkpoint_receipt,arm):
        if sha(checkpoint_receipt["path"])!=checkpoint_receipt["sha256"]:
            raise ValueError("Export geometry checkpoint changed")
        saved=torch.load(checkpoint_receipt["path"],map_location="cpu",weights_only=True,mmap=True)
        if (saved.get("identities")!=self.identities or saved.get("teacher_id")!=self.teacher_id
                or saved.get("epoch")!=epoch or any(not torch.equal(value,saved["geometry"].get(name))
                    for name,value in self.coordinates.state_dict().items())):
            raise ValueError("Export must describe the exact current saved geometry")
        del saved
        directory=Path(directory)
        if directory.exists():raise FileExistsError(directory)
        directory.mkdir(parents=True);layers={}
        for index,coordinates in enumerate(self.coordinates):
            artifact=coordinates.export(anchor=self.model.anchors[index].cpu())
            path=directory/f"layer_{index:02d}.pt";receipt=_save(path,artifact)
            layers[str(index)]=dict(receipt,width=self.model.width,retained=self.config.retained)
            del artifact
        reference=self.export_projection_reference(directory/"projection_reference.pt",checkpoint_receipt)
        manifest=dict(schema=GEOMETRY_SCHEMA,arm=arm,epoch=epoch,identities=self.identities,teacher_id=self.teacher_id,
            model_configuration=self.model.native.config.to_dict(),layers=layers,projection_sites=PROJECTION_SITES,
            checkpoint=checkpoint_receipt,raw_precision="float64",qr_precision="float64",application_precision="float32",
            native_precision="bfloat16",qr_device=str(self.qr_device),rms_denominator_width=self.model.width,
            native_heads_and_mlp_preserved=True,projection_reference=reference,confirmed=False)
        _write(directory/"manifest.json",manifest)
        return dict(directory=str(directory.resolve()),manifest_sha256=sha(directory/"manifest.json"),epoch=epoch,arm=arm)

    @torch.no_grad()
    def export_projection_reference(self,path,checkpoint_receipt):
        """Actual searched full-model outputs for later compiled-model replay.

        Two held-out TRAIN records per family, selected without inspecting
        outputs, bind the learned/random geometry to its actual BF16 execution.
        Numerical deviations from folded execution are diagnostic; this does
        not substitute for generated task-quality evaluation.
        """
        families={}
        for index in self.selection:
            record=self.dataset.records[index]
            family=record.provenance.get("source_family",record.task)
            families.setdefault(family,[]).append(index)
        if not self.systems_only and (len(families)!=8 or any(len(v)<2 for v in families.values())):
            raise ValueError("Projection reference requires two selection records from each of eight families")
        indices=[i for family in sorted(families) for i in families[family][:2]]
        started=time.monotonic();qr_seconds=self.prepare_window(differentiable=False);records=[]
        try:
            for index in indices:
                record=self.dataset.records[index];batch=self.dataset.batch([index]);nal.validate_batch(batch)
                hidden=self.model.hidden_forward(batch);positions=batch["loss_mask"].nonzero()
                last=positions[-1];last_logits=self.model.readout(hidden[last[0],last[1]].unsqueeze(0))
                records.append(dict(index=index,example_id=record.example_id,group_id=record.group_id,
                    family=record.provenance.get("source_family",record.task),
                    input_ids=batch["input_ids"].detach().cpu().clone(),
                    response_prediction_positions=positions.detach().cpu().clone(),
                    normalized_hidden=hidden.detach().cpu().clone(),
                    last_prediction_position=int(last[1]),last_prediction_logits=last_logits.detach().cpu().clone()))
                del hidden,last_logits
            self._guard()
            result=_save(path,dict(schema="jspace.ablation32-projection-reference.v1",records=records,
                identities=self.identities,teacher_id=self.teacher_id,checkpoint=checkpoint_receipt,
                projection_sites=PROJECTION_SITES,qr_device=str(self.qr_device),
                application_precision="float32",native_precision="bfloat16"))
            return dict(result,examples=len(records),indices=indices,qr_seconds=qr_seconds,
                elapsed_seconds=time.monotonic()-started,
                scope="Actual projected native full model; compiled replay remains a separate measurement")
        finally:self._clear_window()

    @torch.no_grad()
    def unprojected_control(self,indices):
        """Measure the original model/cache path before introducing any ablation."""
        rows=[]
        for index in indices:
            batch=self.dataset.batch([index]);count=nal.validate_batch(batch);self.teacher.prepare(batch);self._guard()
            actual=self.model.native.model(input_ids=batch["input_ids"].to(self.model.device),use_cache=False).last_hidden_state
            expected=batch["teacher_hidden"].to(actual)
            difference=actual.float()-expected.float();positions=batch["loss_mask"].nonzero();total=0.
            for start in range(0,count,self.config.token_chunk_size):
                pos=positions[start:start+self.config.token_chunk_size]
                h=actual[pos[:,0].to(actual.device),pos[:,1].to(actual.device)]
                target=batch["targets"][pos[:,0],pos[:,1]]
                total+=float(nal.loss_vectors(self.model.readout(h),self.teacher(batch,pos),target,self.loss_config)["forward_kl"].sum())
            rows.append(dict(index=index,hidden_bit_exact=bool(torch.equal(actual,expected)),
                hidden_max_abs=float(difference.abs().max()),
                hidden_relative_rms=float(difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-30)),
                response_kl_T2=total/count,response_tokens=count))
            self.teacher.clear_cache()
        self._guard()
        return dict(records=rows,all_hidden_bit_exact=all(r["hidden_bit_exact"] for r in rows),
            all_finite=all(math.isfinite(r["response_kl_T2"]) and math.isfinite(r["hidden_relative_rms"]) for r in rows),
            scope="Unprojected original BF16 device/cache numerical control; not compressed quality")

    @torch.no_grad()
    def _projector_actions(self,coordinates,probes):
        result=[]
        for coordinate in coordinates:
            D=self._basis(coordinate,differentiable=False);result.append(D@(D.T@probes))
        return result

    def pilot(self,indices,*,directory):
        if not self.systems_only or self.updates:raise ValueError("Pilot must be a fresh disposable systems-only instance")
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        native_control=self.unprojected_control(list(indices))
        if not native_control["all_finite"]:raise FloatingPointError("Native cache/device control is nonfinite")
        before=[p.detach().clone() for p in self.parameters]
        # Eight fixed probes demonstrate changed projector action rather than
        # merely a rotated coordinate parameterization of the same subspace.
        probes=torch.randn(self.model.width,8,dtype=torch.float64,
            generator=torch.Generator().manual_seed(self.config.basis_seed+999))
        initial=self._projector_actions(self.coordinates,probes)
        result=self.train_window(indices)
        raw_changes=[float((p.detach()-old).abs().max()) for p,old in zip(self.parameters,before)]
        actions=self._projector_actions(self.coordinates,probes)
        projection_changes=[float((value-old).abs().max()) for value,old in zip(actions,initial)]
        if not all(v>0 and math.isfinite(v) for v in raw_changes+projection_changes):
            raise RuntimeError("Pilot did not update every actual discarded subspace")
        saved=self.save_checkpoint(directory/"pilot.pt",epoch=0,selection=None,save_optimizer=False)
        reference=_save(directory/"projector_reference.pt",dict(probes=probes,actions=actions))
        loaded=torch.load(saved["path"],map_location="cpu",weights_only=True,mmap=True)
        fresh=nn.ModuleList([geometry.DiscardedSubspace(self.model.width,self.config.retained,
            seed=self.config.basis_seed+i,dtype=torch.float64) for i in range(self.model.layers)])
        fresh.load_state_dict(loaded["geometry"],strict=True)
        raw_exact=all(torch.equal(a,b) for a,b in zip(fresh.parameters(),self.parameters))
        replay=self._projector_actions(fresh,probes)
        replay_exact=all(torch.equal(a,b) for a,b in zip(actions,replay))
        if not raw_exact or not replay_exact:raise ValueError("Fresh saved geometry/projector replay changed")
        del fresh,loaded;self._guard()
        receipt=dict(status="systems_passed",discarded=True,all_interfaces_updated=True,
            interfaces=self.model.layers,raw_max_changes=raw_changes,projector_probe_max_changes=projection_changes,
            optimizer_state_dtypes=sorted({str(value.dtype) for state in self.optimizer.state.values()
                for key,value in state.items() if key in {"exp_avg","exp_avg_sq"}}),
            zero_anchor_systems_only=self.zero_anchor_pilot,teacher_unchanged=True,update=result,
            unprojected_native_control=native_control,checkpoint=saved,projector_reference=reference,
            fresh_raw_reload_exact=raw_exact,fresh_projector_replay_exact=replay_exact,
            qr_device=str(self.qr_device),cpu_threads=self.config.cpu_threads,confirmed=False)
        _write(directory/"result.json",receipt);return receipt

    def fit(self,directory,*,progress=lambda *_:None):
        if self.systems_only or self.finished or self.updates:raise ValueError("Fresh scientific state required; pilot updates cannot enter search")
        self.finished=True;directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
        initial_score=self.evaluate();initial=self.save_checkpoint(directory/"epoch_00.pt",epoch=0,selection=initial_score)
        random_export=self.export(directory/"random",epoch=0,checkpoint_receipt=initial,arm="matched_initial_random")
        best=dict(epoch=0,checkpoint=initial,selection=initial_score);history=[dict(best)]
        progress(dict(phase="selection",epoch=0,**initial_score))
        for epoch in range(1,self.config.epochs+1):
            seen=[]
            for indices in epoch_windows(self.training,epoch,self.config):
                row=self.train_window(indices);seen.extend(indices);progress(dict(phase="basis_search",epoch=epoch,**row))
            if sorted(seen)!=sorted(self.training):raise RuntimeError("An epoch omitted or repeated a calibration example")
            score=self.evaluate();checkpoint_receipt=self.save_checkpoint(directory/f"epoch_{epoch:02d}.pt",epoch=epoch,selection=score)
            item=dict(epoch=epoch,checkpoint=checkpoint_receipt,selection=score);history.append(item)
            if score["per_example_kl"]<best["selection"]["per_example_kl"]:best=item
            progress(dict(phase="selection",epoch=epoch,**score))
        selected=self.load_checkpoint(best["checkpoint"]["path"],expected_sha256=best["checkpoint"]["sha256"])
        replay=self.evaluate()
        if replay["per_example_kl"]!=best["selection"]["per_example_kl"]:
            raise ValueError("Selected search checkpoint reload did not reproduce the held-out objective")
        learned_export=self.export(directory/"learned",epoch=best["epoch"],checkpoint_receipt=best["checkpoint"],arm="selected_learned")
        result=dict(schema=SCHEMA,status="completed",configuration=asdict(self.config),identities=self.identities,
            history=history,selected_epoch=best["epoch"],selected_checkpoint=best["checkpoint"],selected_objective=best["selection"],
            selected_reload_exact=True,initial_random_export=random_export,learned_export=learned_export,
            total_updates=self.updates,total_response_tokens=self.supervised_tokens,original_teacher_unchanged=True,
            elapsed_seconds=time.monotonic()-started,projection_sites=PROJECTION_SITES,qr_device=str(self.qr_device),
            scope="Selection-set functional KL search, not final task accuracy",confirmed=False)
        _write(directory/"result.json",result);return result

    def close(self):
        self.teacher.clear_cache();self._clear_window();self.optimizer.zero_grad(set_to_none=True);gc.collect()
