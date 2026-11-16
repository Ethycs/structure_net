"""Arbitrary-cut calibration/fit adapter for the expanded original-native recovery data.

The caller authenticates the prospective plan and completed parent selection,
reserves the RTX3060, sets its allocator cap, and supplies a live guard. These
functions never launch a job, change a frozen producer, or open audit examples.
The original native logit bank remains the recovery teacher; differentiable
accepted-parent logits written by the VJP recorder are calibration diagnostics.
"""
from __future__ import annotations
from collections import Counter
import gc
from pathlib import Path
import time

import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from . import jspace_3b_broad_recovery_cache as cache
from . import jspace_3b_broad_recovery as trial
from . import jspace_3b_broad_verify_repair as repair
from . import jspace_qwen3b_progressive_calibrate as calibration
from . import jspace_qwen_parallel_fit as fitting
from . import jspace_allblocks_rankfit as rankfit
from .jspace_qwen_progressive_calibration import GroupedTeacherCapture

base,read,builder=cache.base,cache.read,cache.builder
CachedPrefixTeacher=calibration.CachedPrefixTeacher
SCHEMA='jspace.3b-progressive-capture.v1'
PARENT_SCHEMAS={'jspace.3b-broad-step-capture.v1',SCHEMA}
DEVICE='cuda:0'
FAMILIES=('arc','boolq','hellaswag','winogrande')


def native_rotary(plan,device,guard):
    """Original native loader semantics: construct FP32 RoPE on CPU, then move."""
    guard()
    config=builder.native._config(plan['model']['path'],None)
    return Qwen2RotaryEmbedding(config=config,device='cpu').to(device).eval().requires_grad_(False)


class SplitRotaryTeacher(CachedPrefixTeacher):
    """Exact native entrance and accepted compact suffix keep their own RoPE."""
    def __init__(self,*args,native_rotary,**kwargs):
        super().__init__(*args,**kwargs)
        if (native_rotary.training or any(p.requires_grad for p in native_rotary.parameters()) or
                any(t.device!=self.device or t.is_inference() for t in native_rotary.buffers())):
            self.close();raise ValueError('Frozen original native rotary on the teacher device required')
        self.native_rotary=native_rotary
    def native_forward(self,index):
        x=self.boundary(index)
        return calibration._hidden(self.model.layers[self.cut](x,**calibration.block_arguments(x,self.native_rotary)))
    def forward_case(self,index):
        x=self.native_forward(index)
        return self.model.norm(calibration._hidden(self.accepted_suffix(x,**calibration.block_arguments(x,self.rotary))))


@torch.no_grad()
def check_native_seam(teacher,boundary_path):
    """Singleton original native entrance must reproduce every cached next state."""
    bank=np.load(boundary_path,mmap_mode='r',allow_pickle=False)
    try:
        if bank.dtype!=np.int16 or bank.shape!=(teacher.offsets[-1],teacher.lm_head.weight.shape[1]):
            raise ValueError('Native seam bank layout changed')
        for index in range(len(teacher.cases)):
            expected=torch.from_numpy(np.array(bank[teacher.offsets[index]:teacher.offsets[index+1]],copy=True)).view(torch.bfloat16)[None]
            cache.sanity.boundary_control(teacher.native_forward(index).cpu(),expected)
        return dict(passed=True,bit_exact=True,count=len(teacher.cases),max_abs=0.)
    finally:bank._mmap.close()


def content_items(rows):
    """Keep exact source offsets/order; alias the four meaningful positions."""
    return [dict(c,loss_positions=list(c['content_loss_positions']),
        targets=list(c['content_targets'])) for c in rows]


def validate_plan_items(plan):
    """Production admission, with no small-set/systems-only bypass."""
    if (type(plan.get('cut'))is not int or not 0<=plan['cut']<=20 or
            plan.get('parent_cut')!=plan['cut']+1 or plan.get('allocation_mib')!=11264 or
            plan.get('allocator_mib')!=11008):raise ValueError('Only bounded cuts20 through0 with adjacent accepted parent are supported')
    cp=cache.verify(plan['cache_root']);rows=read(Path(plan['cache_root'])/'items.json')['items']
    if (plan['model']!=cp['model'] or plan['label_ids']!=cp['label_ids'] or
            Path(plan['data_root']).resolve()!=Path(cp['data_root']).resolve() or
            plan['training_order']!=cp['training_order'] or
            any(k not in plan['parent_fits'] or Path(plan['parent_fits'][k]).resolve()!=Path(v).resolve()
                for k,v in cp['fits'].items())):
        raise ValueError('Changed model, expanded data, order, labels or inherited fits')
    cache.validate_items(rows,plan['label_ids'])
    train=[i for i,c in enumerate(rows) if c['split']=='train']
    dev=[i for i,c in enumerate(rows) if c['split']=='development']
    indices=plan['calibration_indices']
    if (len(rows)!=4864 or len(train)!=4608 or len(dev)!=256 or sorted(plan['training_order'])!=train or
            Counter(rows[i]['family'] for i in dev)!=dict.fromkeys(FAMILIES,64) or
            len(indices)!=128 or len(set(indices))!=128 or any(type(i)is not int or i not in train for i in indices) or
            Counter(rows[i]['family'] for i in indices)!=dict.fromkeys(FAMILIES,32)):
        raise ValueError('Complete4608TRAIN/256DEV and128 balanced TRAIN-only calibration required')
    if set(plan['parent_fits'])!={str(i) for i in range(plan['parent_cut'],36)}:
        raise ValueError('Exact contiguous accepted-parent fit map required')
    fixed=plan['expected_parent_fixed_state']
    if (set(fixed)!={'sha256','tensor_bytes'} or len(fixed['sha256'])!=64 or
            type(fixed['tensor_bytes'])is not int or fixed['tensor_bytes']<=0 or
            base.sha(plan['parent_checkpoint'])!=plan['parent_checkpoint_sha256']):
        raise ValueError('Authenticated parent checkpoint and complete fixed inventory required')
    roots=[Path(plan[k]).resolve() for k in ('boundary_root','capture_root','fit_root')]
    if len(set(roots))!=3 or any(a.is_relative_to(b) for a in roots for b in roots if a!=b):
        raise ValueError('Distinct nonnested append-only phase roots required')
    if any(root.is_relative_to(Path(plan[k]).resolve()) or Path(plan[k]).resolve().is_relative_to(root)
           for root in roots for k in ('cache_root','data_root')):
        raise ValueError('New outputs must remain separate from immutable input roots')
    parent_roots=[Path(plan['parent_boundary_receipt']).resolve().parent,
        Path(plan['parent_checkpoint']).resolve().parent,*[Path(v).resolve() for v in plan['parent_fits'].values()]]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for root in roots for p in parent_roots):
        raise ValueError('New outputs must remain separate from accepted-parent artifacts')
    items=content_items(rows)
    parent_boundary(plan,items)
    return items


def parent_boundary(plan,items,*,hash_payload=False):
    """Bind the previous stage's ORIGINAL-native bank, never its student output."""
    source=Path(plan['parent_boundary_path']).resolve()
    receipt=Path(plan['parent_boundary_receipt']).resolve()
    if (source.name!=f"boundary_{plan['parent_cut']:02d}.npy" or source.parent!=receipt.parent or
            base.sha(receipt)!=plan['parent_boundary_receipt_sha256']):
        raise ValueError('Authenticated previous native-boundary receipt required')
    r=read(receipt)
    if (r.get('schema') not in PARENT_SCHEMAS or r.get('status')!='completed' or r.get('phase')!='boundary'
            or r.get('cut')!=plan['parent_cut'] or r.get('items_sha256')!=base.digest(items)
            or r.get('shape')!=[items[-1]['stop'],2048] or not r.get('fixed_state_unchanged')
            or r.get('artifacts',{}).get(source.name)!=plan['parent_boundary_sha256']):
        raise ValueError('Parent native boundary belongs to a different cut, data or source')
    if r['schema']=='jspace.3b-broad-step-capture.v1' and plan['parent_cut']!=21:
        raise ValueError('Original single-step boundary schema only produced cut21')
    control=r.get('native_parent_boundary_control',r.get('native_pre22_control'))
    if (not isinstance(control,dict) or not control.get('passed') or not control.get('bit_exact')
            or control.get('count')!=len(items) or control.get('max_abs')!=0.):
        raise ValueError('Previous native boundary seam was not exact')
    if hash_payload and base.sha(source)!=plan['parent_boundary_sha256']:
        raise ValueError('Authenticated parent native bank changed')
    return source


def teacher_fingerprint(plan,items):
    return base.digest(dict(parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],cut=plan['cut'],
        content_items_sha256=base.digest(items),probe_groups=2,probes_per_group=8,
        seed=calibration.calibration_seed(plan['cut'])))


def recovery_dataset(plan,*,parent_input=False):
    """Original-native content slots4:8; explicit new-cut or accepted-parent input."""
    if type(parent_input)is not bool:raise ValueError('Explicit parent-input boolean required')
    items=validate_plan_items(plan);root=Path(plan['cache_root'])
    boundary=Path(plan['parent_boundary_path']) if parent_input else Path(plan['boundary_root'])/f"boundary_{plan['cut']:02d}.npy"
    return cache.RecoveryDataset(items,boundary,root/'native/teacher_logits.npy',
        arm='native_content',label_ids=plan['label_ids'])


def _phase(out,plan,phase,guard):
    if not callable(guard):raise ValueError('Caller-owned reservation guard required')
    guard()  # Before any CUDA context or model allocation.
    items=validate_plan_items(plan);out=Path(out).resolve()
    if out!=Path(plan[phase+'_root']).resolve():raise ValueError('Phase destination differs from prospective plan')
    if out.exists():raise FileExistsError(out)
    return out,items


def _progress(out,value):
    values=dict(value);values['producer_phase']=values.pop('phase',None)
    base.event(out/'progress.jsonl','progress',**values)


def _save_receipt(out,plan,phase,items,**values):
    result=dict(schema=SCHEMA,status='completed',phase=phase,cut=plan['cut'],
        parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],items_sha256=base.digest(items),
        teacher_fingerprint=teacher_fingerprint(plan,items),confirmed=False,**values)
    base.write(out/'step_receipt.json',result);return result


def _checked_phase(plan,phase,items):
    root=Path(plan[phase+'_root']);r=read(root/'step_receipt.json')
    if (r.get('schema')!=SCHEMA or r.get('status')!='completed' or r.get('phase')!=phase or
            r.get('cut')!=plan['cut'] or r.get('parent_checkpoint_sha256')!=plan['parent_checkpoint_sha256'] or
            r.get('items_sha256')!=base.digest(items) or r.get('teacher_fingerprint')!=teacher_fingerprint(plan,items)):
        raise ValueError('Prior phase belongs to a different parent/data/cut')
    for name,digest in r['artifacts'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or base.sha(path)!=digest:raise ValueError('Prior phase artifact changed')
    return r


def _inventory(out):
    return {str(p.relative_to(out)):base.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}


@torch.inference_mode()
def native_boundaries(model,case,cut,guard):
    """Original unpadded HF prefix; stop before unused blocks after both captures."""
    guard()
    if type(cut)is not int or not 0<=cut<len(model.model.layers)-1:raise ValueError('Adjacent native cuts required')
    if model.lm_head.weight is not model.model.embed_tokens.weight:raise ValueError('Original tied readout required')
    captured={};handles=[]
    class ReachedBoundary(Exception):pass
    for index in (cut,cut+1):
        def hook(_module,args,layer=index):
            captured[layer]=args[0].detach().cpu().clone()
            if layer==cut+1:raise ReachedBoundary()
        handles.append(model.model.layers[index].register_forward_pre_hook(hook))
    try:
        ids=torch.tensor([case['input_ids']],device=model.lm_head.weight.device)
        try:model.model(input_ids=ids,use_cache=False)
        except ReachedBoundary:pass
    finally:
        for handle in handles:handle.remove()
    if set(captured)!={cut,cut+1} or any(x.dtype!=torch.bfloat16 or x.shape[:2]!=(1,len(case['input_ids'])) or
            not torch.isfinite(x).all() for x in captured.values()):raise ValueError('Incomplete finite BF16 native boundaries')
    return captured


def run_boundary(out,plan,guard):
    out,items=_phase(out,plan,'boundary',guard);started=time.monotonic();reference=bank=None;model=None
    source=parent_boundary(plan,items,hash_payload=True);cut=plan['cut'];parent_cut=plan['parent_cut']
    try:
        reference=np.load(source,mmap_mode='r',allow_pickle=False)
        if reference.dtype!=np.int16 or reference.shape!=(items[-1]['stop'],2048):raise ValueError('Original native bank layout changed')
        with cache.sanity.execution.arithmetic_policy(),trial._model_device(DEVICE):
            model=cache.sanity.execution.load_native_model(plan['model']['path'],device=DEVICE,attention='eager',guard=guard)
            before=[(id(t),t._version) for t in (*model.parameters(),*model.buffers())]
            out.mkdir(parents=True);bank=np.lib.format.open_memmap(out/f'boundary_{cut:02d}.npy',mode='w+',dtype=np.int16,shape=reference.shape)
            for index,case in enumerate(items):
                values=native_boundaries(model,case,cut,guard)
                expected=torch.from_numpy(np.array(reference[case['offset']:case['stop']],copy=True)).view(torch.bfloat16)[None]
                cache.sanity.boundary_control(values[parent_cut],expected)
                bank[case['offset']:case['stop']]=values[cut][0].contiguous().view(torch.int16).numpy()
                _progress(out,dict(phase='boundary',completed=index+1,total=len(items)))
            bank.flush()
            if before!=[(id(t),t._version) for t in (*model.parameters(),*model.buffers())]:raise ValueError('Native model mutated')
        guard()
        return _save_receipt(out,plan,'boundary',items,artifacts=_inventory(out),
            shape=list(reference.shape),storage='lossless BF16 bits as int16',
            native_parent_boundary_control=dict(passed=True,bit_exact=True,count=len(items),max_abs=0.),
            native_parent_boundary_sha256=plan['parent_boundary_sha256'],parent_cut=parent_cut,fixed_state_unchanged=True,
            reused_native_logits=str(Path(plan['cache_root'])/'native/teacher_logits.npy'),
            native_executed_layers=list(range(cut+1)),elapsed_seconds=time.monotonic()-started)
    finally:
        for value in (reference,bank):
            if value is not None:value._mmap.close()
        del model;gc.collect()


def run_capture(out,plan,guard):
    out,items=_phase(out,plan,'capture',guard);started=time.monotonic()
    boundary=_checked_phase(plan,'boundary',items);teacher=None
    with cache.sanity.execution.arithmetic_policy(),trial._model_device(DEVICE):
        guard();stack,norm,E,_,inheritance=builder.build_stack(plan['model']['path'],
            {int(k):v for k,v in plan['parent_fits'].items()},plan['parent_cut'],plan['parent_checkpoint'],
            device='cpu',trainable=False,checkpoint_blocks=False)
        rotary=repair.restore_rotary(plan,DEVICE,guard=guard)
        fixed=repair.assert_fixed(stack,E,rotary,plan['expected_parent_fixed_state'])
        packed=builder.packed_clone(stack,device='cpu');repair.assert_fixed(stack,E,rotary,fixed);del stack
        packed,norm,E=packed.to(DEVICE),norm.to(DEVICE),E.to(DEVICE)
        native=builder.native.load_native_block(plan['model']['path'],plan['cut'],device=DEVICE,dtype=torch.bfloat16)
        original_rope=native_rotary(plan,DEVICE,guard)
        try:
            teacher=SplitRotaryTeacher(items,Path(plan['boundary_root'])/f"boundary_{plan['cut']:02d}.npy",plan['cut'],
                native,packed,norm,E,rotary,guard=guard,native_rotary=original_rope)
            seam_path=parent_boundary(plan,items,hash_payload=True)
            if plan['parent_boundary_sha256']!=boundary['native_parent_boundary_sha256']:raise ValueError('Original native seam target changed')
            seam=check_native_seam(teacher,seam_path)
            ropes=dict(native=dict(constructor='CPU FP32 then moved to model GPU',
                    **cache.old.fingerprint(list(original_rope.named_buffers()))),
                parent=dict(constructor='FP32 constructed directly on model GPU',
                    **cache.old.fingerprint(list(rotary.named_buffers()))))
            with GroupedTeacherCapture(out,teacher,[plan['cut']],items,plan['calibration_indices'],
                    teacher_fingerprint=teacher_fingerprint(plan,items),probe_groups=2,probes_per_group=8,
                    activation_positions=16,position_capacity=416,seed=calibration.calibration_seed(plan['cut']),guard=guard) as recorder:
                for index in range(len(items)):
                    recorder.capture_case(index,lambda i=index:teacher.forward_case(i))
                    _progress(out,dict(phase='jacobian_capture',completed=index+1,total=len(items)))
                metadata=recorder.finish()
            guard()
            return _save_receipt(out,plan,'capture',items,artifacts=_inventory(out),
                calibration_indices=list(plan['calibration_indices']),metadata_sha256=base.sha(out/'metadata.json'),
                boundary_sha256=boundary['artifacts'][f"boundary_{plan['cut']:02d}.npy"],parent_fixed_state=fixed,inheritance=inheritance,
                native_seam=seam,rotary_provenance=ropes,
                parent_parameters_frozen=True,positions='content4',probe_groups=2,probes_per_group=8,
                calibration_count=128,training_means_count=4608,development_count=256,
                recovery_teacher='unchanged original-native content logits; accepted_teacher_logits.npy is diagnostic only',
                estimated_payload_bytes=metadata['estimated_payload_bytes'],elapsed_seconds=time.monotonic()-started)
        finally:
            if teacher is not None:teacher.close()
            del native,packed,norm,E,rotary,original_rope;gc.collect()


def run_fit(out,plan,guard):
    out,items=_phase(out,plan,'fit',guard);started=time.monotonic();capture=_checked_phase(plan,'capture',items)
    root=Path(plan['capture_root']);metadata=read(root/'metadata.json')
    if (metadata.get('layers')!=[plan['cut']] or metadata.get('calibration_indices')!=sorted(plan['calibration_indices']) or
            metadata.get('capture_order')!=list(range(len(items))) or metadata.get('probe_groups')!=2 or
            metadata.get('probes_per_group')!=8 or metadata.get('activation_positions')!=16 or
            metadata.get('seed')!=calibration.calibration_seed(plan['cut']) or
            metadata.get('teacher_fingerprint')!=teacher_fingerprint(plan,items)):
        raise ValueError('Changed full-context grouped calibration metadata')
    out.mkdir(parents=True)  # The fitter emits progress before creating its block directory.
    with cache.sanity.execution.arithmetic_policy(),trial._model_device(DEVICE):
        guard();fitting.finalize_basis(root,out,layer_ids=[plan['cut']],rank=1024,head_dim=64,device=DEVICE,guard=guard,
            progress=lambda v:_progress(out,v))
        native=builder.native.load_native_block(plan['model']['path'],plan['cut'],device=DEVICE,dtype=torch.bfloat16)
        rotary=repair.restore_rotary(plan,DEVICE,guard=guard);directory=out/f"block_{plan['cut']:02d}"
        try:
            fitting.capture_compiled_development(directory,root,plan['cut'],native,
                lambda block,x,_case:block(x,**calibration.block_arguments(x,rotary)),device=DEVICE,guard=guard,
                progress=lambda v:_progress(out,v))
            rankfit.fit_relation(directory,device=DEVICE,guard=guard,progress=lambda v:_progress(out,v),
                seed=861,anchors=24,remove_count=1024,pad_to_multiple=128,svd_rank=64,
                oversample=16,power_iterations=1,ridge=.01,eligibility_nrmse=.5,chunk_rows=256)
        finally:del native,rotary;gc.collect()
    guard()
    return _save_receipt(out,plan,'fit',items,artifacts=_inventory(out),
        capture_receipt_sha256=base.sha(root/'step_receipt.json'),rank=1024,head_dim=64,anchors=24,removed=1024,
        compiled_interface_rotary='FP32 constructed directly on model GPU, matching candidate suffix execution',
        fit_directory=str(directory.resolve()),order=['basis','compiled_nonGB_interface','TRAIN_relation'],
        elapsed_seconds=time.monotonic()-started)
