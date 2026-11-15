"""One-layer 3B calibration against an accepted compact suffix.

This is an adapter around the original grouped full-vocabulary VJP estimator,
not a new estimator. Prefix states are immutable native BF16 cache rows. Only
the new native block, accepted packed suffix, final norm and tied readout are
resident. The caller authenticates the parent, source and case identities and
owns the reservation and allocator limit.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .jspace_bench_data import sha, write
from .jspace_qwen_progressive_calibration import GroupedTeacherCapture
from . import jspace_qwen_parallel_fit as original_fit
from . import jspace_allblocks_rankfit as corrected_fit


FAMILIES = ("arc", "boolq", "hellaswag", "winogrande")


def calibration_seed(cut):
    if type(cut) is not int or not 0 <= cut <= 29:
        raise ValueError("Progressive cut must lie in [0,29]")
    return 20861 + 100000 * (29-cut)


def _guard(device,guard):
    device=torch.device(device)
    if device.type=="cuda" and guard is None:
        raise ValueError("CUDA calibration requires a reservation guard")
    if guard is not None:guard()
    return device


def _hidden(value):
    return value[0] if isinstance(value,tuple) else value


def block_arguments(x,rotary):
    positions=torch.arange(x.shape[1],device=x.device)[None].expand(x.shape[0],-1)
    length=x.shape[1]
    mask=torch.full((x.shape[0],1,length,length),torch.finfo(x.dtype).min,
                    device=x.device,dtype=x.dtype).triu(1)
    return dict(attention_mask=mask,position_ids=positions,cache_position=positions[0],
                position_embeddings=rotary(x,positions),use_cache=False)


class CachedPrefixTeacher(nn.Module):
    """Expose the original hook layout without constructing a complete LM."""
    def __init__(self,cases,boundary_path,cut,native_block,parent_suffix,norm,readout,rotary,*,guard=None):
        super().__init__();calibration_seed(cut)
        self.cut,self.cases,self.guard=cut,list(cases),guard
        self.boundary_path=Path(boundary_path).resolve()
        _guard(readout.device,guard)
        modules=(native_block,parent_suffix,norm,rotary)
        if any(m.training or any(p.requires_grad for p in m.parameters()) for m in modules):
            raise ValueError("Native block, accepted suffix, norm and rotary must already be frozen and eval")
        if readout.requires_grad or readout.dtype!=torch.bfloat16 or readout.ndim!=2:
            raise ValueError("Frozen full-vocabulary BF16 readout required")
        if native_block.self_attn.layer_idx!=cut or native_block.self_attn.config.model_type!="qwen2":
            raise ValueError("Native Qwen2 layer must match the requested absolute cut")
        if native_block.self_attn.q_proj.weight.dtype!=torch.bfloat16:
            raise ValueError("Native teacher execution must remain BF16")
        tensors=[readout,*(t for m in modules for t in [*m.parameters(),*m.buffers()])]
        if any(t.device!=readout.device or t.is_inference() for t in tensors):
            raise ValueError("Teacher tensors must share a device and permit input autograd")
        self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Identity() for _ in range(cut)]+[native_block])
        self.model.norm=norm;self.accepted_suffix=parent_suffix;self.rotary=rotary
        self.lm_head=nn.Module();self.lm_head.register_buffer("weight",readout.detach())
        self.eval()
        if not self.cases or len({c["identity"] for c in self.cases})!=len(self.cases):
            raise ValueError("Nonempty unique task identities required")
        lengths=[len(c["input_ids"]) for c in self.cases]
        if any(c["split"] not in ("train","development") or not 0<n<=416 for c,n in zip(self.cases,lengths)):
            raise ValueError("Only bounded train/development cases may enter calibration")
        self.offsets=np.cumsum([0,*lengths]).tolist()
        if any(c.get("offset",self.offsets[i])!=self.offsets[i] or
               c.get("stop",self.offsets[i+1])!=self.offsets[i+1] for i,c in enumerate(self.cases)):
            raise ValueError("Native prefix offsets do not match task order")
        self._bank=np.load(self.boundary_path,mmap_mode="r",allow_pickle=False)
        if (not isinstance(self._bank,np.memmap) or self._bank.dtype!=np.dtype("int16") or
                not self._bank.dtype.isnative or self._bank.shape!=(self.offsets[-1],readout.shape[1])):
            self.close();raise ValueError("Expected aligned lossless native BF16-bit [tokens,width] boundary cache")
        self.read_bytes=0

    @property
    def device(self):return self.lm_head.weight.device

    def boundary(self,index):
        _guard(self.device,self.guard)
        if self._bank is None:raise ValueError("Native prefix cache is closed")
        if type(index) is not int or not 0<=index<len(self.cases):raise ValueError("Invalid case index")
        bits=np.array(self._bank[self.offsets[index]:self.offsets[index+1]],copy=True)
        self.read_bytes+=bits.nbytes
        x=torch.from_numpy(bits).view(torch.bfloat16).to(self.device)[None]
        if not torch.isfinite(x).all():raise FloatingPointError("Nonfinite native boundary")
        return x

    def forward_case(self,index):
        x=self.boundary(index);arguments=block_arguments(x,self.rotary)
        x=_hidden(self.model.layers[self.cut](x,**arguments))
        x=_hidden(self.accepted_suffix(x,**arguments))
        return self.model.norm(x)

    def close(self):
        if getattr(self,"_bank",None) is not None:
            if isinstance(self._bank,np.memmap):self._bank._mmap.close()
            self._bank=None

    def __enter__(self):return self
    def __exit__(self,*_args):self.close()


def capture(root,teacher,calibration_indices,*,teacher_fingerprint,systems_only=False,progress=None):
    """Original two independent groups × eight full-vocabulary sign probes."""
    if not isinstance(teacher,CachedPrefixTeacher):raise TypeError("CachedPrefixTeacher required")
    indices=list(calibration_indices);cases=teacher.cases
    if any(type(i) is not int or not 0<=i<len(cases) for i in indices):raise ValueError("Invalid calibration indices")
    if not systems_only:
        splits=Counter(c["split"] for c in cases)
        families=Counter(cases[i]["family"] for i in indices)
        if splits!={"train":512,"development":64} or families!={f:32 for f in FAMILIES}:
            raise ValueError("Scientific capture requires original 512/64 cases and 128 balanced training indices")
    _guard(teacher.device,teacher.guard)
    with GroupedTeacherCapture(root,teacher,[teacher.cut],cases,indices,
            teacher_fingerprint=teacher_fingerprint,probe_groups=2,probes_per_group=8,
            activation_positions=16,position_capacity=416,seed=calibration_seed(teacher.cut),guard=teacher.guard) as recorder:
        for index in range(len(cases)):
            recorder.capture_case(index,lambda i=index:teacher.forward_case(i))
            if progress is not None:progress(dict(phase="capture",cut=teacher.cut,completed=index+1,total=len(cases)))
        metadata=recorder.finish()
    _guard(teacher.device,teacher.guard)
    write(Path(root)/"progressive_wrapper.json",dict(schema="jspace.qwen3b-progressive-calibration.v1",
        status="completed",systems_only=bool(systems_only),cut=teacher.cut,seed=calibration_seed(teacher.cut),
        teacher_fingerprint=teacher_fingerprint,metadata_sha256=sha(Path(root)/"metadata.json"),
        native_boundary=str(teacher.boundary_path),native_boundary_sha256=sha(teacher.boundary_path),
        boundary_read_bytes=teacher.read_bytes,
        source="Unchanged GroupedTeacherCapture full-vocabulary VJP estimator; cached native prefix then new native block and accepted packed suffix"))
    return metadata


def _completed(capture_root):
    root=Path(capture_root)
    value=json.loads((root/"progressive_wrapper.json").read_text())
    metadata=json.loads((root/"metadata.json").read_text())
    if (value.get("status")!="completed" or metadata.get("status")!="completed" or
            value.get("schema")!="jspace.qwen3b-progressive-calibration.v1" or
            metadata.get("layers")!=[value["cut"]] or metadata.get("seed")!=calibration_seed(value["cut"]) or
            value["metadata_sha256"]!=sha(root/"metadata.json") or
            metadata["teacher_fingerprint"]!=value["teacher_fingerprint"]):
        raise ValueError("Completed matching progressive capture required")
    return metadata,value


def finalize(capture_root,fit_group_root,*,device="cpu",guard=None,progress=None,rank=1024,head_dim=64):
    """Separate basis lease; unchanged two-Gram eigenspace/head construction."""
    _guard(device,guard);metadata,_=_completed(capture_root)
    return original_fit.finalize_basis(capture_root,fit_group_root,layer_ids=metadata["layers"],
        rank=rank,head_dim=head_dim,device=device,guard=guard,progress=progress)


def capture_interface(block_dir,capture_root,cut,native_block,rotary,*,device="cpu",guard=None,progress=None):
    _guard(device,guard);metadata,_=_completed(capture_root)
    if metadata["layers"]!=[cut] or native_block.self_attn.layer_idx!=cut:
        raise ValueError("Compiled interface layer does not match captured layer")
    return original_fit.capture_compiled_development(block_dir,capture_root,cut,native_block,
        lambda block,x,_case:block(x,**block_arguments(x,rotary)),device=device,guard=guard,progress=progress)


def fit_relation(block_dir,*,device="cpu",guard=None,progress=None):
    """Same fixed24-anchor/1024-removal graph fit, corrected numerical rank gate."""
    _guard(device,guard)
    return corrected_fit.fit_relation(block_dir,device=device,guard=guard,progress=progress,
        seed=861,anchors=24,remove_count=1024,pad_to_multiple=128,svd_rank=64,
        oversample=16,power_iterations=1,ridge=.01,eligibility_nrmse=.5,chunk_rows=256)
