"""Fresh, full-response functional calibration for native-first suffix rebuilds.

The caller owns authentication, model/corpus pins and reservations. This module
never launches CUDA workers or reads sealed evaluations. Native softmax and its
original scale are retained; quadratic relations replace selected MLP channels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from . import jspace_qwen3b_progressive_model as builder
from . import jspace_qwen_parallel_fit as fitting
from . import jspace_allblocks_rankfit as relations
from . import jspace_3b_response_model as response_model
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_progressive_calibration import GroupedTeacherCapture, write_bfloat16_bits
from .jspace_qwen3b_progressive_calibrate import block_arguments, _hidden
from .jspace_logit_sketch import normalization_metadata

SCHEMA = "jspace.3b-response-rebuild-fit.v1"


@dataclass(frozen=True)
class FitConfig:
    rank: int = 1024
    head_dim: int = 64
    anchors: int = 24
    remove_count: int = 1024
    svd_rank: int = 64
    oversample: int = 16
    power_iterations: int = 1
    probe_groups: int = 2
    probes_per_group: int = 8
    position_capacity: int = 416
    seed: int = 20861
    ridge: float = .01
    eligibility_nrmse: float = .5
    chunk_rows: int = 256

    def __post_init__(self):
        if (self.probe_groups != 2 or any(type(getattr(self, k)) is not int or getattr(self, k) < 1
                for k in ("rank", "head_dim", "anchors", "remove_count", "svd_rank", "probes_per_group",
                          "position_capacity", "chunk_rows")) or self.head_dim % 2
                or self.svd_rank < self.anchors or self.oversample < 0 or self.power_iterations < 0):
            raise ValueError("Two independent groups and valid fixed compression dimensions required")


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2); stream.write("\n")


def _save(path, value):
    with Path(path).open("xb") as stream: torch.save(value, stream)


def calibration_cases(records, calibration_indices, diagnostic_indices, *, production=True):
    """All response prediction positions, including observed EOS, never DEV."""
    calibration, diagnostic = list(calibration_indices), list(diagnostic_indices)
    all_indices = calibration + diagnostic
    if (not calibration or not diagnostic or len(set(all_indices)) != len(all_indices)
            or any(type(i) is not int or not 0 <= i < len(records) for i in all_indices)):
        raise ValueError("Unique disjoint nonempty TRAIN calibration and diagnostic records required")
    cases = []
    for index in all_indices:
        record = records[index]
        if record.split != "train" or record.finish_reason != "eos" or not record.response_ids:
            raise ValueError("Only complete observed-EOS TRAIN responses may fit the model")
        positions = list(range(len(record.prompt_ids)-1, len(record.input_ids)-1))
        targets = list(record.response_ids)
        if len(positions) != len(targets) or positions[0] < 0 or positions[-1] >= len(record.input_ids):
            raise ValueError("Complete response supervision is misaligned")
        cases.append(dict(identity=record.example_id, group_id=record.group_id, source_index=index, split="train",
            role="calibration" if index in calibration else "diagnostic",
            family=record.provenance.get("source_family", record.task),
            input_ids=list(record.input_ids), loss_positions=positions, targets=targets))
    a, b = (Counter(c["family"] for c in cases if c["role"] == role)
            for role in ("calibration", "diagnostic"))
    if set(a) != set(b): raise ValueError("Calibration and diagnostic task families must agree")
    if len({c["identity"] for c in cases}) != len(cases):
        raise ValueError("Calibration and diagnostics must use distinct example identities")
    if {c["group_id"] for c in cases if c["role"]=="calibration"} & {c["group_id"] for c in cases if c["role"]=="diagnostic"}:
        raise ValueError("Calibration and diagnostic source groups overlap")
    if production and (len(a) != 8 or set(a.values()) != {16} or set(b.values()) != {4}):
        raise ValueError("Production requires128 calibration and32 diagnostic TRAIN records across8 families")
    return cases


class BoundaryTeacher(nn.Module):
    """One native entrance, optional accepted suffix, and actual final RMSNorm."""
    def __init__(self, dataset, cases, cut, native_block, suffix, norm, readout,
                 native_rotary, compact_rotary, *, guard=None):
        super().__init__()
        self.dataset, self.cases, self.cut, self.guard = dataset, cases, cut, guard
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(cut)] + [native_block])
        self.model.norm, self.suffix = norm, suffix
        self.native_rotary, self.compact_rotary = native_rotary, compact_rotary
        self.lm_head = nn.Module(); self.lm_head.register_buffer("weight", readout.detach())
        self.eval().requires_grad_(False)
        self.native_seam_checks = 0

    def forward_case(self, index):
        if self.guard is not None: self.guard()
        batch = self.dataset.batch([self.cases[index]["source_index"]],
                                   device=self.lm_head.weight.device, dtype=torch.bfloat16)
        x = batch["boundary_hidden"]
        x = _hidden(self.model.layers[self.cut](x, **block_arguments(x, self.native_rotary)))
        if not torch.equal(x.detach(), batch["next_boundary_hidden"]):
            raise ValueError("Native entrance did not reproduce the authenticated next-boundary bits")
        self.native_seam_checks += 1
        if self.suffix is not None:
            x = _hidden(self.suffix(x, **block_arguments(x, self.compact_rotary)))
        return self.model.norm(x)


@torch.no_grad()
def training_position_means(dataset, *, width, capacity, progress=None):
    """All-TRAIN native means from disk; no teacher forward or DEV statistics."""
    sums=torch.zeros(capacity,width,dtype=torch.float32);counts=torch.zeros(capacity,dtype=torch.float32)
    indices=[i for i,r in enumerate(dataset.records) if r.split=="train"]
    if not indices:raise ValueError("Native position means require TRAIN records")
    for step,index in enumerate(indices):
        length=len(dataset.records[index].input_ids)
        if not 0<length<=capacity:raise ValueError("TRAIN mean sequence exceeds positional capacity")
        if hasattr(dataset,"boundary") and hasattr(dataset,"rows"):
            row=dataset.rows[index]
            bits=np.array(dataset.boundary[row["offset"]:row["stop"]],copy=True)
            if bits.dtype!=np.int16:raise ValueError("Mean bank must preserve original BF16 bits")
            value=torch.from_numpy(bits).view(torch.bfloat16)
        else:
            value=dataset.batch([index],device="cpu",dtype=torch.bfloat16)["boundary_hidden"][0]
        if value.shape!=(length,width) or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError("Native TRAIN boundary shape/precision changed")
        sums[:length].add_(value.float());counts[:length].add_(1)
        if (step+1)%256==0 or step+1==len(indices):
            fitting._event(progress,"train_position_means",completed=step+1,total=len(indices))
    means=sums/counts[:,None].clamp_min(1)
    means[counts==0]=sums.sum(0)/counts.sum()
    return dict(means=means,counts=counts,training_records=len(indices),training_tokens=int(counts.sum()),
                unobserved_positions=(counts==0).nonzero().flatten().tolist(),
                fill_rule="global token-weighted native TRAIN mean for unobserved positions")


class ResponseCapture(GroupedTeacherCapture):
    """Original VJP math with variable full-response outputs and streamed rows.

The frozen recorder's hooks and VJP implementation are reused unchanged. Each
record sets its true number of output positions; no padding or sampled response
positions enter the functional metric. No huge full-vocabulary logit bank is
written: the recovery teacher already has an authenticated hidden-state cache.
"""
    def __init__(self, root, teacher, *, teacher_fingerprint, config=FitConfig(), guard=None, mean_override=None):
        self.root, self.model, self.cases = Path(root), teacher, teacher.cases
        if self.root.exists(): raise FileExistsError(self.root)
        self.layers, self.modules = (teacher.cut,), {teacher.cut: teacher.model.layers[teacher.cut]}
        self.calibration = tuple(i for i,c in enumerate(self.cases) if c["role"] == "calibration")
        self.development = tuple(i for i,c in enumerate(self.cases) if c["role"] == "diagnostic")
        self.cal_rows, self.dev_rows = dict.fromkeys(self.calibration), dict.fromkeys(self.development)
        self.fixed_tensors = list(teacher.parameters()) + list(teacher.buffers())
        if (teacher.training or any(p.requires_grad for p in teacher.parameters())
                or any(t.is_inference() for t in self.fixed_tensors)):
            raise ValueError("A frozen differentiable teacher is required")
        self.versions = [(id(t),t._version) for t in self.fixed_tensors]
        self.readout, self.guard = teacher.lm_head.weight, guard
        self.device = fitting._guard(self.readout.device, guard)
        self.groups, self.probes = config.probe_groups, config.probes_per_group
        self.lengths = [len(c["input_ids"]) for c in self.cases]
        if not self.lengths or max(self.lengths) > config.position_capacity:
            raise ValueError("Calibration exceeds the prospective positional envelope")
        self.capacity = config.position_capacity
        self.mean_override=mean_override
        self.offsets = np.cumsum([0,*self.lengths]).tolist()
        self.response_offsets = {}; cursor = {"calibration":0,"diagnostic":0}
        self.jac_offsets = {}; jac_rows = 0
        for i,case in enumerate(self.cases):
            role=case["role"]; self.response_offsets[i]=cursor[role]
            cursor[role]+=len(case["loss_positions"])
            if role == "calibration":
                self.jac_offsets[i]=jac_rows; jac_rows+=self.lengths[i]*self.groups*self.probes
        block=self.modules[teacher.cut]; a=block.self_attn; cfg=a.config
        d,m=cfg.hidden_size,block.mlp.down_proj.in_features
        kv,h=cfg.num_key_value_heads,a.head_dim
        self.shapes={teacher.cut:dict(width=d,intermediate=m,heads=cfg.num_attention_heads,kv_heads=kv,head_dim=h)}
        self.root.mkdir(parents=True); path=self.root/f"block_{teacher.cut:02d}"; path.mkdir()
        shapes={"boundary":(self.offsets[-1],d),"mlp_calibration":(cursor["calibration"],m),
                "mlp_development":(cursor["diagnostic"],m),"mlp_jacobian_bf16":(jac_rows,m)}
        self.arrays={teacher.cut:{name:np.lib.format.open_memmap(path/(name+".npy"), mode="w+", dtype=np.int16, shape=shape)
                                for name,shape in shapes.items()}}
        zero=lambda *shape:torch.zeros(*shape,device=self.device,dtype=torch.float32)
        self.statistics={teacher.cut:dict(input_grams=zero(self.groups,d,d),mean_sums=zero(self.capacity,d),
            position_counts=zero(self.capacity),pair_scores=zero(self.groups,kv,h//2),
            mask_grams=zero(self.groups,kv*h//2,kv*h//2),value_moments=zero(kv,h,h),value_means=zero(kv,h))}
        self.generator=torch.Generator(device=self.device).manual_seed(config.seed+teacher.cut)
        self.seen,self.order,self.failed,self.finished=set(),[],False,False
        self.metadata=dict(schema=SCHEMA,teacher_fingerprint=teacher_fingerprint,layers=list(self.layers),
            seed=config.seed+teacher.cut,probe_groups=self.groups,probes_per_group=self.probes,
            calibration_indices=list(self.calibration),development_indices=list(self.development),
            diagnostic_split="independent TRAIN records; DEV is not opened",
            activation_positions="every response prediction position including EOS",
            row_order="calibration record, independent probe group, probe, ALL context positions",
            normalization={"per_record":[normalization_metadata(len(self.readout),len(c["loss_positions"]),self.probes)
                                         for c in self.cases if c["role"]=="calibration"],
                           "record_reduction":"mean; each complete response contributes equal metric weight"},
            storage="lossless actual BF16 forward/backward bits as int16",fixed_configuration=asdict(config),
            estimated_payload_bytes=sum(np.prod(shape).item()*2 for shape in shapes.values()))

    def _record_forward(self,index,captured,output):
        case=self.cases[index]; length=self.lengths[index]
        with torch.no_grad():
            for layer,values in captured.items():
                if set(values)!={"input","q","k","v","mlp"}: raise ValueError("Native entrance hooks incomplete")
                arrays,stats,spec=self.arrays[layer],self.statistics[layer],self.shapes[layer]
                write_bfloat16_bits(arrays["boundary"],slice(self.offsets[index],self.offsets[index+1]),values["input"][0])
                name="mlp_calibration" if case["role"]=="calibration" else "mlp_development"
                start=self.response_offsets[index]; selected=case["loss_positions"]
                write_bfloat16_bits(arrays[name],slice(start,start+len(selected)),values["mlp"][0,selected])
                if index in self.cal_rows:
                    stats["mean_sums"][:length].add_(values["input"][0].float());stats["position_counts"][:length].add_(1)
                    v=values["v"][0].float().reshape(length,spec["kv_heads"],spec["head_dim"]).transpose(0,1)
                    stats["value_moments"].add_(v.transpose(1,2)@v/length);stats["value_means"].add_(v.mean(1))

    def _record_vjps(self,index,captured,output):
        self.output_positions=len(self.cases[index]["loss_positions"])
        super()._record_vjps(index,captured,output)

    def finish(self):
        if self.failed or self.finished or self.seen!=set(range(len(self.cases))):
            raise ValueError("Capture is incomplete or already finished")
        self._check_teacher(); blocks={}
        for layer,stats in self.statistics.items():
            counts=stats["position_counts"]; means=stats["mean_sums"]/counts[:,None].clamp_min(1)
            means[counts==0]=stats["mean_sums"].sum(0)/counts.sum()
            if self.mean_override is not None:
                means=self.mean_override["means"].to(self.device);counts=self.mean_override["counts"].to(self.device)
                if means.shape!=stats["mean_sums"].shape or counts.shape!=stats["position_counts"].shape:
                    raise ValueError("All-TRAIN mean override differs from native geometry")
            result={name:value/len(self.calibration) for name,value in stats.items()
                    if name not in {"mean_sums","position_counts"}}
            result.update(means=means,counts=counts,position_counts=counts,grams=result["input_grams"])
            if any(not torch.isfinite(v).all() for v in result.values()): raise FloatingPointError("Nonfinite statistics")
            path=self.root/f"block_{layer:02d}";_save(path/"statistics.pt",{k:v.cpu() for k,v in result.items()})
            for array in self.arrays[layer].values():array.flush()
            blocks[str(layer)]={"statistics":str(path/"statistics.pt"),"shape":self.shapes[layer],
                               **{name:str(path/(name+".npy")) for name in self.arrays[layer]}}
        items=[dict(case,offset=self.offsets[i],stop=self.offsets[i+1],activation_positions=case["loss_positions"],
                    response_row_offset=self.response_offsets[i],jacobian_row_offset=self.jac_offsets.get(i))
               for i,case in enumerate(self.cases)]
        metadata=dict(self.metadata,status="completed",capture_order=self.order,blocks=blocks,
            calibration_families=[c["family"] for c in self.cases if c["role"]=="calibration" for _ in c["loss_positions"]],
            development_families=[c["family"] for c in self.cases if c["role"]=="diagnostic" for _ in c["loss_positions"]])
        metadata["mean_provenance"]={k:v for k,v in self.mean_override.items() if k not in {"means","counts"}} if self.mean_override else {"scope":"calibration-only fallback"}
        _write(self.root/"items.json",{"items":items});_write(self.root/"metadata.json",metadata)
        self.finished=True;return metadata

    def close(self):
        for arrays in self.arrays.values():
            for array in arrays.values():
                if not array._mmap.closed:array.flush();array._mmap.close()


@torch.no_grad()
def capture_compiled_interface(block_dir,capture_root,native_block,rotary,*,device="cpu",guard=None,progress=None):
    """Capture actual post-projection SwiGLU before fitting polynomial relations."""
    device=fitting._guard(device,guard);block_dir,capture_root=Path(block_dir),Path(capture_root)
    metadata=json.loads((capture_root/"metadata.json").read_text());layer=metadata["layers"][0]
    paths=metadata["blocks"][str(layer)];items=json.loads((capture_root/"items.json").read_text())["items"]
    fit=torch.load(block_dir/"fit.pt",map_location="cpu",weights_only=True)
    block=AffineQwenBlock(native_block,fit["basis"][:,:fit["rank"]].to(device),fit["means"].to(device),
        trainable=False,parameter_dtype=torch.bfloat16,capture_mlp=True,
        head_projection={"rope_pairs":fit["rope_pairs"].to(device),"value_basis":fit["value_basis"].to(device)}).eval()
    arrays={};before=fitting._module_digest(native_block)
    try:
        for role,families in (("calibration",metadata["calibration_families"]),("diagnostic",metadata["development_families"])):
            path=block_dir/("compiled_train.npy" if role=="calibration" else "compiled_dev.npy")
            if path.exists():raise FileExistsError(path)
            arrays[role]=np.lib.format.open_memmap(path,mode="w+",dtype=np.int16,shape=(len(families),paths["shape"]["intermediate"]))
        for case in items:
            fitting._guard(device,guard)
            x=fitting._read_bits(paths["boundary"],slice(case["offset"],case["stop"]),device)[None]
            block.last_mlp_hidden=None;block(x,**block_arguments(x,rotary))
            if block.last_mlp_hidden is None:raise ValueError("Compiled MLP interface was not executed")
            positions=case["activation_positions"];start=case["response_row_offset"]
            write_bfloat16_bits(arrays[case["role"]],slice(start,start+len(positions)),block.last_mlp_hidden[0,positions])
        for array in arrays.values():array.flush()
        _save(block_dir/"output_weight.pt",block.mlp_output.detach().float().cpu())
    finally:
        block.set_capture_mlp(False)
        for array in arrays.values():array._mmap.close()
    if before!=fitting._module_digest(native_block):raise ValueError("Native entrance changed during interface capture")
    inputs=[capture_root/"metadata.json",capture_root/"items.json",block_dir/"fit.pt",block_dir/"fit.json",
            Path(paths["boundary"]),Path(paths["mlp_development"]),Path(paths["mlp_jacobian_bf16"]),
            block_dir/"compiled_train.npy",block_dir/"compiled_dev.npy",block_dir/"output_weight.pt"]
    interface=dict(layer=layer,teacher_fingerprint=metadata["teacher_fingerprint"],rank=fit["rank"],head_dim=fit["head_dim"],
        calibration_families=metadata["calibration_families"],development_families=metadata["development_families"],
        native_calibration=str(block_dir/"compiled_train.npy"),native_development=paths["mlp_development"],
        compiled_development=str(block_dir/"compiled_dev.npy"),jacobian=paths["mlp_jacobian_bf16"],
        output_weight=str(block_dir/"output_weight.pt"),native_block_state_sha256=before,
        training_activation_interface="actual compiled residual/head interface; legacy fitter key native_calibration names this file",
        diagnostic_scope="independent TRAIN-only full-response native and compiled rows; not DEV",
        order=["native functional VJP", "residual/head projection", "compiled SwiGLU capture", "quadratic relation fit"],
        input_sha256={str(p):fitting.sha(p) for p in inputs})
    _write(block_dir/"interface.json",interface)
    fitting._event(progress,"compiled_response_interface_complete",layer=layer)
    return interface


def fit_stage(root,dataset,*,model_path,cut,calibration_indices,diagnostic_indices,
              teacher_fingerprint,parent_fits=None,parent_checkpoint=None,device="cpu",guard=None,
              config=FitConfig(),production=True,progress=None):
    """Return ordinary fit directory for build_stack(..., previous=parent).

    The dataset exposes records and batch([source_index]) with boundary_hidden
    and next_boundary_hidden from the ORIGINAL native model. Stage35 needs no
    parent. Every later stage needs precisely the accepted adjacent suffix.
    """
    device=fitting._guard(device,guard);root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    if not isinstance(teacher_fingerprint,str) or not teacher_fingerprint:
        raise ValueError("An authenticated fixed-teacher identity is required")
    cfg=builder.native._config(model_path,None)
    if type(cut)is not int or not 0<=cut<cfg.num_hidden_layers:raise ValueError("Invalid native entrance")
    if getattr(dataset,"cut",cut)!=cut:raise ValueError("Dataset boundary cut differs from fitted entrance")
    if (cut==cfg.num_hidden_layers-1)!=(parent_checkpoint is None):
        raise ValueError("Only the final native block can bootstrap without an accepted parent")
    if production and (cut not in (35,34) or config!=FitConfig()):
        raise ValueError("Initial scientific rebuild is fixed-recipe stages35 then34 only")
    cases=calibration_cases(dataset.records,calibration_indices,diagnostic_indices,production=production)
    root.mkdir(parents=True);started=time.monotonic();suffix=None
    means=training_position_means(dataset,width=cfg.hidden_size,capacity=config.position_capacity,progress=progress)
    if parent_checkpoint is not None:
        if set(parent_fits or {})!=set(range(cut+1,cfg.num_hidden_layers)):
            raise ValueError("Exact contiguous accepted-parent fit map required")
        suffix,norm,readout,_,inheritance=builder.build_stack(model_path,parent_fits,cut+1,parent_checkpoint,
            device=device,trainable=False,rank=config.rank,checkpoint_blocks=False)
    else:
        readout,norm,_=builder.native.load_readout_norm(model_path,device=device,dtype=torch.bfloat16,config=cfg)
        norm.eval().requires_grad_(False);readout.requires_grad_(False)
        inheritance={"native_initialization":True,"unchanged_layers":[]}
    native=builder.native.load_native_block(model_path,cut,device=device,dtype=torch.bfloat16,config=cfg)
    native.eval().requires_grad_(False)
    native_rotary=response_model.partial.training_rotary(cfg,"cpu").to(device)
    compact_rotary=response_model.partial.training_rotary(cfg,device,guard=guard)
    teacher=BoundaryTeacher(dataset,cases,cut,native,suffix,norm,readout,native_rotary,compact_rotary,guard=guard)
    with ResponseCapture(root/"capture",teacher,teacher_fingerprint=teacher_fingerprint,config=config,guard=guard,
                         mean_override=means) as capture:
        for index in range(len(cases)):
            capture.capture_case(index,lambda i=index:teacher.forward_case(i))
            fitting._event(progress,"response_vjp_capture",layer=cut,completed=index+1,total=len(cases))
        metadata=capture.finish()
    seam_checks=teacher.native_seam_checks
    del capture,teacher,suffix,norm,readout;gc.collect()
    fitting.finalize_basis(root/"capture",root/"fits",layer_ids=[cut],rank=config.rank,head_dim=config.head_dim,
                           device=device,guard=guard,progress=progress)
    directory=root/"fits"/f"block_{cut:02d}"
    capture_compiled_interface(directory,root/"capture",native,compact_rotary,device=device,guard=guard,progress=progress)
    relations.fit_relation(directory,device=device,guard=guard,seed=861,anchors=config.anchors,remove_count=config.remove_count,
        pad_to_multiple=128,svd_rank=config.svd_rank,oversample=config.oversample,power_iterations=config.power_iterations,
        ridge=config.ridge,eligibility_nrmse=config.eligibility_nrmse,chunk_rows=config.chunk_rows,progress=progress)
    fit=torch.load(directory/"fit.pt",map_location="cpu",weights_only=True)
    relation=torch.load(directory/"relation.pt",map_location="cpu",weights_only=True)
    compiled=AffineQwenBlock(native,fit["basis"][:,:config.rank].to(device),fit["means"].to(device),
        trainable=False,parameter_dtype=torch.bfloat16,
        head_projection={"rope_pairs":fit["rope_pairs"].to(device),"value_basis":fit["value_basis"].to(device)})
    compiled.compile_polynomial(relation,relation["removed"])
    if not compiled.has_polynomial or compiled.polynomial_removed.numel()!=config.remove_count:
        raise ValueError("Saved relation did not compile the prescribed retained/residual structure")
    del compiled,fit,relation
    del native,native_rotary,compact_rotary;gc.collect();fitting._guard(device,guard)
    receipt=dict(schema=SCHEMA,status="completed",cut=cut,fit_directory=str(directory),
        teacher_fingerprint=teacher_fingerprint,parent_checkpoint_sha256=fitting.sha(parent_checkpoint) if parent_checkpoint else None,
        inheritance=inheritance,configuration=asdict(config),native_seam_bit_exact=True,native_seam_checks=seam_checks,
        calibration_records=len(calibration_indices),diagnostic_train_records=len(diagnostic_indices),
        mean_provenance={k:v for k,v in means.items() if k not in {"means","counts"}},
        response_positions=sum(len(c["loss_positions"]) for c in cases if c["role"]=="calibration"),
        calibration_families=dict(Counter(c["family"] for c in cases if c["role"]=="calibration")),
        functional_basis="full eigendecomposition of two-group full-response task-Jacobian Gramian; no truncated Krylov approximation",
        attention="native softmax and original scale; fitted RoPE-pair/value head projections",
        mlp="compiled-interface quadratic relations with native residual channels retained",
        order=["fresh functional basis", "compile residual/head projections", "capture compiled SwiGLU", "fit quadratic relations", "compile polynomial residual structure"],
        saved_fit_compilation_passed=True,
        artifacts={str(p.relative_to(root)):fitting.sha(p) for p in sorted(root.rglob("*")) if p.is_file()},
        elapsed_seconds=time.monotonic()-started,confirmed=False)
    _write(root/"fit_receipt.json",receipt)
    return receipt
