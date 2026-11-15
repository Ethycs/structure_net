"""Generic packed-parent targets and qualified resident feature-KD adapter.

The caller authenticates the actual accepted checkpoint and native input
boundary. This helper does not select parents, admit a campaign or reserve GPUs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_3b_feature_recovery as feature
from . import jspace_3b_teacher_targets as original
from . import jspace_qwen_compact_stack as compact
from .jspace_qwen_progressive_calibration import write_bfloat16_bits
from .jspace_resident_adamw import ResidentSpillAdamW
from .jspace_resident_adam_replay import TimedOptimizer

SCHEMA='jspace.3b-parent-feature-targets.v1'
NAMES=('packed_teacher_logits.npy','parent_training_final_answers.npy',
    'parent_training_final_answers.json','parent_feature_receipt.json')
base,read=feature.base,feature.read


@torch.inference_mode()
def packed_coordinates(stack,hidden,*,position_ids,position_embeddings,attention_mask=None,guard=None):
    """Unchanged uncached packed executor, returning its live final coordinates.

    No hooks, detached reconstructed surrogate or global binding mutation.
    Fusion eligibility is selected by the existing stack predicate.
    """
    original._guard(hidden.device,guard)
    if (not isinstance(stack,compact.CompactQwenStack) or not stack.packed or stack.training
            or any(p.requires_grad for p in stack.parameters()) or hidden.ndim!=3 or hidden.shape[-1]!=stack.width
            or position_ids.shape!=hidden.shape[:2] or position_embeddings is None
            or list(stack.layer_indices)!=list(range(stack.layer_indices[0],36))):
        raise ValueError('Frozen packed contiguous Qwen suffix and explicit native input/positions required')
    if stack.geometry_compute_dtype is not None and hidden.dtype!=stack.geometry_compute_dtype:
        raise ValueError('Matching worker geometry dtype required')
    positions=[b._positions(hidden,position_ids[0],position_ids,None) for b in stack.blocks]
    fused=stack._can_fuse(hidden)
    if hidden.is_cuda and hidden.dtype==torch.bfloat16 and not fused:
        raise RuntimeError('Packed BF16 parent CUDA capture must exercise its fused inference path')
    z=stack._geometry(hidden,stack.blocks[0].basis.T)
    for index,block in enumerate(stack.blocks):
        if index:
            bridge=stack.bridges[index-1];p=positions[index-1+bridge.position_source]
            z=stack._geometry(z,bridge.weight,block._rows(bridge.shift,p,*z.shape[:2]))
        z,_=compact._run_reduced(block,z,positions[index],attention_mask=attention_mask,past_key_value=None,
            cache_position=position_ids[0],position_embeddings=position_embeddings,fuse_qkv=fused,fuse_mlp=fused,
            padded_width=block.mlp_output.shape[1],feature_i=getattr(stack,f'feature_i_{index}'),feature_j=getattr(stack,f'feature_j_{index}'))
    last=stack.blocks[-1];anchor=last._rows(last.anchor,positions[-1],*hidden.shape[:2])
    return stack._geometry(z,last.basis,anchor),z,dict(fuse_qkv=fused,fuse_mlp=fused)


def target_rows(items):
    rows=[]
    for index,case in enumerate(items):
        if case.get('split') not in ('train','development'):raise ValueError('Only original training/development cases allowed')
        length=case['stop']-case['offset']
        if (not 0<length<=416 or case['offset']<0 or length!=len(case['input_ids'])
                or len(case['loss_positions'])!=4 or case['loss_positions'][-1]!=length-1):
            raise ValueError('Original token offsets and last-valid answer position required')
        if case['split']=='train':
            rows.append(dict(source_index=index,identity=case['identity'],family=case['family'],split='train',
                native_offset=case['offset'],native_stop=case['stop'],length=length,answer_position=length-1))
    if not rows or len({c['identity'] for c in items})!=len(items):raise ValueError('Distinct original identities required')
    return rows


def _parent_sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Explicit actual accepted checkpoint SHA256 required')


def _feature_receipt(metadata,path):
    return dict(schema=SCHEMA,status='completed',path=str((path/NAMES[1]).resolve()),
        sha256=original.sha(path/NAMES[1]),metadata_path=str((path/NAMES[2]).resolve()),metadata_sha256=original.sha(path/NAMES[2]),
        shape=metadata['shape'],training_count=len(metadata['rows']),normalization_energy=metadata['energy_mean'],
        normalization_scope='all TRAIN answer targets only',parent_checkpoint_sha256=metadata['parent_checkpoint_sha256'],
        parent_cut=metadata['parent_cut'],coordinate_layer=35,position='last valid token',
        source_indices=[r['source_index'] for r in metadata['rows']],rows_sha256=base.digest(metadata['rows']),
        parent_layers=metadata['parent_layers'],development_fitted=False)


@torch.inference_mode()
def capture_packed_parent(directory,stack,norm,E,rotary,dataset,*,parent_checkpoint_sha256,
                          guard=None,progress=None,label_ids=None):
    """One split-specific batch4 pass writes parent logits and TRAIN z35 answers.

    ``dataset`` must read the native boundary at the accepted parent's cut,
    which is one above the newly added block. The caller owns that provenance.
    Original source-row ordering is retained even with interleaved splits.
    """
    original._guard(E.device,guard);_parent_sha(parent_checkpoint_sha256)
    if E.dtype!=torch.bfloat16 or E.shape!=(dataset.vocabulary,dataset.width) or not stack.packed:
        raise ValueError('Packed parent and original full-vocabulary BF16 readout required')
    before=original._fixed(stack,norm,E,rotary);items=dataset.items;rows=target_rows(items)
    train=list(dataset.training);dev=list(dataset.development)
    if (train!=[r['source_index'] for r in rows] or dev!=[i for i,c in enumerate(items) if c['split']=='development']
            or not dev or sorted(train+dev)!=list(range(len(items)))):
        raise ValueError('Complete split-specific original case ordering required')
    if label_ids is not None and (len(label_ids)!=4 or len(set(label_ids))!=4
            or any(type(i) is not int or not 0<=i<len(E) for i in label_ids)):
        raise ValueError('Four distinct original choice-label tokens required')
    path=Path(directory)
    if any((path/name).exists() for name in NAMES):raise FileExistsError('Partial or completed parent-target outputs already exist')
    path.mkdir(parents=True,exist_ok=True);rank=stack.blocks[-1].rank
    for name in NAMES[:2]:
        with (path/name).open('xb'):pass
    logits_array=np.lib.format.open_memmap(path/NAMES[0],mode='w+',dtype=np.int16,shape=(len(items),4,len(E)))
    answer_array=np.lib.format.open_memmap(path/NAMES[1],mode='w+',dtype=np.int16,shape=(len(train),rank))
    row_lookup={index:row for row,index in enumerate(train)};seen=set();seen_targets=set();batches=[];records=[];executions=[]
    reads=dict(getattr(dataset,'read_bytes',{}))
    try:
        for split,indices in (('train',train),('development',dev)):
            for start in range(0,len(indices),4):
                original._guard(E.device,guard)
                if original._fixed(stack,norm,E,rotary)!=before:raise ValueError('Accepted teacher changed')
                chosen=indices[start:start+4];batch=dataset.batch(chosen,device=E.device)
                if batch['indices']!=chosen or batch['hidden'].dtype!=torch.bfloat16 or batch['loss_positions'].shape!=(len(chosen),4):
                    raise ValueError('Exact BF16 input rows and four output positions required')
                rope=rotary(batch['hidden'],batch['position_ids'])
                output,z,execution=packed_coordinates(stack,batch['hidden'],position_ids=batch['position_ids'],
                    position_embeddings=rope,attention_mask=batch['attention_mask'],guard=guard)
                selected=output[torch.arange(len(chosen),device=E.device)[:,None],batch['loss_positions']]
                logits=F.linear(norm(selected),E)
                if logits.dtype!=torch.bfloat16 or logits.shape!=(len(chosen),4,len(E)) or z.dtype!=torch.bfloat16:
                    raise ValueError('Capture stores actual BF16 logits/coordinates without new quantization')
                write_bfloat16_bits(logits_array,chosen,logits)
                if split=='train':
                    answers=z[torch.arange(len(chosen),device=E.device),torch.tensor(batch['lengths'],device=E.device)-1]
                    write_bfloat16_bits(answer_array,[row_lookup[i] for i in chosen],answers);seen_targets.update(chosen)
                    del answers
                elif label_ids is not None:
                    for row,index in enumerate(chosen):
                        case=items[index];values=logits[row,-1,label_ids[:len(case['choices'])]].float()
                        records.append(dict(identity=case['identity'],family=case['family'],answer=case['answer'],
                            prediction=int(values.argmax()),logits=values.cpu().tolist()))
                if original._fixed(stack,norm,E,rotary)!=before:raise ValueError('Accepted teacher changed')
                seen.update(chosen);batches.append(dict(split=split,indices=chosen));executions.append(execution)
                if progress is not None:progress(dict(phase='packed_parent_feature_targets',split=split,completed=len(seen),total=len(items),indices=chosen))
                del batch,rope,output,z,selected,logits
        if seen!=set(range(len(items))) or seen_targets!=set(train):raise ValueError('Incomplete parent output/feature rows')
        logits_array.flush();answer_array.flush()
        # Freeze the same equal-case TRAIN energy convention as the isolated arm.
        values=torch.from_numpy(np.array(answer_array,copy=True)).view(torch.bfloat16)
        energy=float(values.float().square().mean(-1).mean())
        if not math.isfinite(energy) or energy<=0:raise ValueError('Positive finite TRAIN target energy required')
    finally:
        logits_array._mmap.close();answer_array._mmap.close()
    original._guard(E.device,guard)
    if original._fixed(stack,norm,E,rotary)!=before:raise ValueError('Accepted teacher changed')
    metadata=dict(schema=SCHEMA,rows=rows,shape=[len(train),rank],storage='Lossless BF16 bits as native-endian int16',
        parent_checkpoint_sha256=parent_checkpoint_sha256,parent_cut=stack.layer_indices[0],parent_layers=list(stack.layer_indices),
        coordinate_layer=35,position='last valid token',energy_mean=energy,development_fitted=False,
        coordinate_location='actual post-block35 output before reconstruction; no reprojection',execution=executions)
    base.write(path/NAMES[2],metadata)
    logits_receipt=dict(status='completed',path=str((path/NAMES[0]).resolve()),sha256=original.sha(path/NAMES[0]),
        shape=[len(items),4,len(E)],storage='Lossless existing BF16 logit bits in native-endian int16; no dtype conversion before storage',
        teacher_execution='Packed accepted parent in inference_mode; original native prefix boundary at parent cut',
        row_order='Original dataset indices; training batches then independently grouped development batches',
        microbatch=4,batches=batches,training_count=len(train),development_count=len(dev),fixed_state_unchanged=True,
        records=records if label_ids is not None else None,label_ids=label_ids,
        read_bytes={k:v-reads.get(k,0) for k,v in getattr(dataset,'read_bytes',{}).items()},calibration_jacobians_changed=False,
        scope='Packed parent output recovery targets plus actual TRAIN final-answer coordinates; differentiable VJP teacher remains separate')
    receipt=dict(schema=SCHEMA,status='completed',logits=logits_receipt,features=_feature_receipt(metadata,path))
    base.write(path/NAMES[3],receipt);return receipt


def load_targets(capture_dir,items,parent_sha):
    """Validate only the consumed small feature bank; caller checks stage manifest."""
    _parent_sha(parent_sha);path=Path(capture_dir);receipt=read(path/NAMES[3]);metadata=read(path/NAMES[2])
    rows=target_rows(items);layers=metadata.get('parent_layers',[])
    if (receipt.get('status')!='completed' or receipt.get('schema')!=SCHEMA or metadata.get('schema')!=SCHEMA
            or metadata.get('parent_checkpoint_sha256')!=parent_sha or metadata.get('rows')!=rows
            or not layers or layers!=list(range(layers[0],36)) or metadata.get('parent_cut')!=layers[0]
            or metadata.get('coordinate_layer')!=35 or metadata.get('position')!='last valid token'
            or metadata.get('development_fitted') is not False or metadata.get('shape',[None])[0]!=len(rows)):
        raise ValueError('Actual accepted-parent TRAIN feature provenance changed')
    expected=_feature_receipt(metadata,path)
    if receipt.get('features')!=expected:raise ValueError('Target metadata or tensor hash changed')
    bits=np.load(path/NAMES[1],allow_pickle=False)
    if bits.dtype!=np.dtype('int16') or not bits.dtype.isnative or list(bits.shape)!=metadata['shape']:
        raise ValueError('Native-endian BF16 bit bank required')
    targets=feature.AnswerTargets(torch.from_numpy(bits).view(torch.bfloat16),[r['source_index'] for r in rows],
        normalization_energy=metadata['energy_mean'])
    return targets,expected


def validate_calibration(calibration,items,target_receipt):
    return feature.validate_calibration(calibration,dict(calibration_indices=feature.calibration_indices(items),target_receipt=target_receipt))


class ResidentFeatureTrainer(feature.FeatureTrainer):
    """Original feature trainer, then exact CPU-state adoption into qualified Adam."""
    def __init__(self,*args,resident_device,resident_budget_bytes,**kwargs):
        super().__init__(*args,**kwargs)
        self.optimizer=TimedOptimizer(ResidentSpillAdamW.from_cpu_optimizer(self.optimizer,
            resident_device=resident_device,resident_budget_bytes=resident_budget_bytes,guard=self.guard),
            devices=(self.device,torch.device(resident_device)),guard=self.guard)
    def train_window(self,*args,**kwargs):
        self.optimizer.synchronize();started=time.perf_counter()
        result=super().train_window(*args,**kwargs);self.optimizer.synchronize()
        return dict(result,synchronized_train_window_seconds=time.perf_counter()-started)
