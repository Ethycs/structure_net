"""Matched native/parent teachers for the prospective fourteen-block trial.

Native logits have eight slots: legacy four followed by content four, including
intentional duplicate positions. Boundaries and logits preserve BF16 bits.
Only TRAIN and development rows enter this module; sealed audits stay outside.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_3b_feature_resident_campaign as source
from . import jspace_3b_parent_feature as feature
from . import jspace_3b_sanity_evaluate as sanity
from . import jspace_qwen_parallel_train as core
from .jspace_full_lease import wait_ready

base,old,read,builder=source.base,source.old,source.read,source.model
SCHEMA="jspace.3b-broad-recovery-cache.v1"
PROTOCOL=base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-broad-recovery.md"
SEED_SHA=sanity.CHECKPOINT_SHA
PARENT_SHA='abc72e97e8f800517bc1b106a86bd740013647d90537fd68bcb7576bb0de9eec'
ALPHA=.4192096950932534
ARMS=("parent_legacy","native_content")


def canonical_items(items):
    """Give generic parent-feature helpers their unchanged legacy aliases."""
    offset=0;rows=[]
    for case in items:
        rows.append(dict(case,offset=offset,stop=offset+len(case['input_ids']),
            loss_positions=list(case['legacy_loss_positions']),targets=list(case['legacy_targets'])))
        offset=rows[-1]['stop']
    return rows


def validate_items(items, label_ids):
    if not items or len({c['identity'] for c in items})!=len(items):raise ValueError('Distinct recovery rows required')
    offset=0
    for c in items:
        length=len(c['input_ids'])
        if c['split'] not in ('train','development') or not 3<=length<=384:
            raise ValueError('Only bounded TRAIN/development prompts are permitted')
        if c['offset']!=offset or c['stop']!=offset+length:raise ValueError('Contiguous exact token offsets required')
        offset=c['stop']
        for kind in ('legacy','content'):
            positions,targets=c[kind+'_loss_positions'],c[kind+'_targets']
            if (len(positions)!=4 or len(targets)!=4 or positions[-1]!=length-1 or
                    any(type(p)is not int or not 0<=p<length for p in positions) or
                    any(type(t)is not int or t<0 for t in targets) or targets[-1]!=label_ids[c['answer']]):
                raise ValueError('Four explicit positions/next-token targets and gold answer required')
    return offset


class _LogitView:
    def __init__(self, array, start):
        self.array,self.start=array,start
        self.shape=(len(array),4,array.shape[-1])
    def __getitem__(self, indices):return self.array[indices,self.start:self.start+4]


class RecoveryDataset:
    """Reuse the proven padded batch copier with an explicit teacher-slot view."""
    _bits=staticmethod(core.BoundaryDataset._bits)
    batch=core.BoundaryDataset.batch
    def __init__(self, items, boundary_path, logits_path, *, arm, label_ids):
        if arm not in ARMS and arm!='native_legacy':raise ValueError('Unknown teacher arm')
        kind='content' if arm=='native_content' else 'legacy'
        self.items=[dict(c,loss_positions=c[kind+'_loss_positions'],targets=c[kind+'_targets']) for c in items]
        total=validate_items(items,label_ids)
        self.boundary=core.BoundaryDataset._open(boundary_path,2)
        self._logits=core.BoundaryDataset._open(logits_path,3)
        expected=4 if arm=='parent_legacy' else 8
        if self._logits.shape[:2]!=(len(items),expected) or len(self.boundary)!=total:
            raise ValueError('Teacher/boundary shape does not match the complete row inventory')
        self.accepted=_LogitView(self._logits,4 if arm=='native_content' else 0)
        self.original=self.original_indices=None
        self.width,self.vocabulary=self.boundary.shape[1],self._logits.shape[-1]
        self.lengths=[len(c['input_ids']) for c in items]
        self.training=[i for i,c in enumerate(items) if c['split']=='train']
        self.development=[i for i,c in enumerate(items) if c['split']=='development']
        if not self.training or not self.development:raise ValueError('Both TRAIN and development are required')
        self.read_bytes=dict(boundary=0,accepted=0,original=0)
        if any(t>=self.vocabulary for c in self.items for t in c['targets']):raise ValueError('Target outside vocabulary')
    def close(self):
        for value in (self.boundary,self._logits):
            if not value._mmap.closed:value._mmap.close()


@torch.inference_mode()
def native_outputs(model, case, label_ids, *, cuts=(22,23), guard=lambda:None):
    guard();captured={};handles=[]
    if model.lm_head.weight is not model.model.embed_tokens.weight:raise ValueError('Native tied E required')
    for cut in cuts:
        def hook(_module,args,index=cut):captured[index]=args[0].detach().cpu().clone()
        handles.append(model.model.layers[cut].register_forward_pre_hook(hook))
    try:
        ids=torch.tensor([case['input_ids']],device=model.lm_head.weight.device)
        normalized=model.model(input_ids=ids,use_cache=False).last_hidden_state
        positions=case['legacy_loss_positions']+case['content_loss_positions']
        logits=F.linear(normalized[:,positions],model.lm_head.weight)[0]
        record=sanity._record(case,normalized[:,-1],model.lm_head.weight,label_ids)
    finally:
        for handle in handles:handle.remove()
    if set(captured)!=set(cuts) or logits.dtype!=torch.bfloat16 or not torch.isfinite(logits).all():
        raise ValueError('Native capture incomplete or nonfinite')
    return captured,logits.cpu(),record


def prepare(root,source_root,data_root):
    from . import jspace_3b_broad_recovery_data as data_source
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires CUDA hidden')
    root,source_root,data_root=map(lambda p:Path(p).resolve(),(root,source_root,data_root))
    if root.exists():raise FileExistsError(root)
    parent,stage=source.verify_stage(source_root,22);directory=source.stage_dir(source_root,22)
    proof=source.checked(directory,'verify',stage)
    if (not proof['acceptance']['accepted'] or proof['acceptance']['candidate']['checkpoint_sha256']!=SEED_SHA or
            stage['parent_checkpoint_sha256']!=PARENT_SHA):
        raise ValueError('Exact fresh accepted fourteen-block seed required')
    data_source.verify(data_root)
    dp=read(data_root/'plan.json');items=read(data_root/'items.json')['items']
    if dp['model']!=parent['model'] or dp['label_ids']!=parent['label_ids']:
        raise ValueError('Data/model/tokenizer mismatch')
    rows=canonical_items(items);offset=rows[-1]['stop']
    validate_items(rows,dp['label_ids'])
    training=[i for i,c in enumerate(rows) if c['split']=='train'];development=[i for i,c in enumerate(rows) if c['split']=='development']
    if len(training)!=4608 or len(development)!=256 or sorted(dp['training_order'])!=training:
        raise ValueError('Frozen 4608 TRAIN/256 DEV and complete common order required')
    inputs={}
    for p in (source_root/'plan.json',directory/'plan.json',directory/'verify/manifest.json',
              directory/'verify/result.json',directory/'verify/accepted_development.json',
              data_root/'plan.json',data_root/'manifest.json',data_root/'items.json'):
        old.pin(inputs,p)
    checkpoint=directory/'verify/accepted.pt';old.pin(inputs,checkpoint,SEED_SHA)
    old.pin(inputs,stage['parent_checkpoint'],stage['parent_checkpoint_sha256'])
    for name,digest in parent['model']['files'].items():old.pin(inputs,Path(parent['model']['path'])/name,digest)
    for fit in stage['fits'].values():
        for name in ('fit.pt','relation.pt'):old.pin(inputs,Path(fit)/name)
    old.pin(inputs,parent['items_path'])
    controls=read(parent['items_path'])['items'];control=next(c for c in controls if c['split']=='development')
    for cut in (22,23):
        p=Path(parent['native_cache'])/f'boundary_{cut:02d}.npy'
        old.pin(inputs,p,parent['input_sha256'][str(p)])
    files=source.inherited.local_import_closure([Path(__file__),PROTOCOL,
        base.ROOT/'tests/structure_net/test_jspace_3b_broad_recovery_cache.py'])
    plan=dict(schema=SCHEMA,source_root=str(source_root),source_fingerprint=parent['fingerprint'],
        stage_fingerprint=stage['fingerprint'],data_root=str(data_root),data_fingerprint=dp['fingerprint'],
        model=parent['model'],fits=stage['fits'],checkpoint=str(checkpoint),checkpoint_sha256=SEED_SHA,
        parent_checkpoint=stage['parent_checkpoint'],parent_checkpoint_sha256=stage['parent_checkpoint_sha256'],
        alpha=ALPHA,label_ids=dp['label_ids'],training_order=dp['training_order'],cases_digest=base.digest(rows),
        count=len(rows),training_count=len(training),development_count=len(development),total_tokens=offset,
        hidden_size=2048,vocabulary=151936,native_cache=parent['native_cache'],development_control=control,
        native_slots='legacy4 concatenated content4; repeated positions intentionally retained',
        native_execution='singleton unpadded BF16 eager; raw tokens; full-vocabulary teacher slots',
        development_scoring='singleton unpadded identical choice-row linear, separate from teacher full-vocabulary GEMM',
        allocation_mib=11264,allocator_mib=11008,confirmed=False,
        input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        source_sha256={str(p):base.sha(p) for p in sorted(files)})
    plan['fingerprint']=base.digest(plan);root.mkdir(parents=True)
    for p,digest in plan['source_sha256'].items():
        target=root/'producer'/Path(p).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        if base.sha(target)!=digest:raise ValueError('Source archive mismatch')
    base.write(root/'items.json',dict(items=rows));base.write(root/'plan.json',plan)
    return dict(status='prepared',fingerprint=plan['fingerprint'])


def verify(root):
    root=Path(root);p=read(root/'plan.json');source.check_plan(p)
    if (p.get('schema')!=SCHEMA or p.get('checkpoint_sha256')!=SEED_SHA or p.get('alpha')!=ALPHA or
            p.get('parent_checkpoint_sha256')!=PARENT_SHA or p.get('allocation_mib')!=11264 or
            p.get('allocator_mib')!=11008):
        raise ValueError('Broad cache policy changed')
    for path,identity in p['input_identity'].items():
        if old.file_identity(path)!=identity:raise ValueError('Pinned cache input changed: '+path)
    for path,digest in p['source_sha256'].items():
        if base.sha(path)!=digest or base.sha(root/'producer'/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError('Cache producer changed')
    rows=read(root/'items.json')['items']
    if base.digest(rows)!=p['cases_digest']:raise ValueError('Recovery rows changed')
    validate_items(rows,p['label_ids'])
    return p


def checked(root,phase,plan=None):
    root=Path(root);p=verify(root) if plan is None else plan
    if phase not in ('native','parent'):raise ValueError('Unknown cache phase')
    directory=root/phase;m=read(directory/'manifest.json');result=read(directory/'result.json')
    if m.get('status')!='completed' or m.get('phase')!=phase or m.get('fingerprint')!=p['fingerprint']:
        raise ValueError('Incomplete cache phase')
    for name,digest in m['artifacts'].items():
        if not (directory/name).resolve().is_relative_to(directory.resolve()) or base.sha(directory/name)!=digest:
            raise ValueError('Cache artifact changed')
    uuid=m.get('allocator',{}).get('physical_uuid','')
    if (not uuid.startswith('GPU-0370fdb0') or m.get('allocation')!={uuid:11264} or
            m['allocator'].get('allocator_limit_bytes')!=11008*1024**2 or
            not 0<m.get('peak_reserved_bytes',0)<=11008*1024**2 or
            m['artifacts'].get('result.json')!=base.sha(directory/'result.json') or result.get('status')!='completed'):
        raise ValueError('Invalid cache resource/result receipt')
    if phase=='native':
        if (set(result['boundary_controls'])!={'22','23'} or
                not all(v.get('bit_exact') and v.get('passed') for v in result['boundary_controls'].values()) or
                not result.get('fixed_state_unchanged')):raise ValueError('Native controls failed')
        dev=[c for c in read(root/'items.json')['items'] if c['split']=='development']
        sanity.validate_records(result['records'],dev)
        if result.get('count')!=len(dev) or result.get('correct')!=sum(r['prediction']==r['answer'] for r in result['records']):
            raise ValueError('Native development count changed')
    else:
        if result.get('parent_checkpoint_sha256')!=p['parent_checkpoint_sha256']:raise ValueError('Parent changed')
        _,receipt=feature.load_targets(directory,read(root/'items.json')['items'],p['parent_checkpoint_sha256'])
        if (result.get('receipt')!=read(directory/'parent_feature_receipt.json') or
                result.get('normalization_energy')!=receipt['normalization_energy'] or result.get('alpha')!=ALPHA):
            raise ValueError('Parent target/normalization receipt changed')
    return result


def run(root,phase):
    root=Path(root).resolve();p=verify(root);out=root/phase
    if phase not in ('native','parent'):raise ValueError('Unknown cache phase')
    if phase=='parent':checked(root,'native',p)
    if out.exists():raise FileExistsError(out)
    lease=settings=None;cuda=False;dataset=None;started=time.monotonic()
    try:
        lease=wait_ready([0]);settings=source.inherited.cap_allocator(lease,p);torch.cuda.init();cuda=True
        torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4)
        def guard():
            current=wait_ready([0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:raise RuntimeError('Lease changed')
            if torch.cuda.max_memory_reserved()>settings['allocator_limit_bytes']:raise MemoryError('Lease cap exceeded')
        def progress(value):
            payload=dict(value)
            if 'phase' in payload:payload['capture_phase']=payload.pop('phase')
            base.event(root/'progress.jsonl','capture',phase=phase,**payload)
        out.mkdir();items=read(root/'items.json')['items']
        with sanity.execution.arithmetic_policy():
            if phase=='native':
                native=sanity.execution.load_native_model(p['model']['path'],device='cuda:0',attention='eager',guard=guard)
                frozen=[(id(v),v._version) for v in (*native.parameters(),*native.buffers())]
                control=dict(p['development_control']);control['legacy_loss_positions']=control['loss_positions']
                control['content_loss_positions']=control['loss_positions']
                boundaries,_,_=native_outputs(native,control,p['label_ids'],guard=guard);controls={}
                for cut in (22,23):
                    bank=np.load(Path(p['native_cache'])/f'boundary_{cut:02d}.npy',mmap_mode='r')
                    expected=torch.from_numpy(np.array(bank[control['offset']:control['stop']],copy=True)).view(torch.bfloat16)[None]
                    controls[str(cut)]=sanity.boundary_control(boundaries[cut],expected);del bank,expected
                banks={cut:np.lib.format.open_memmap(out/f'boundary_{cut:02d}.npy',mode='w+',dtype=np.int16,
                    shape=(p['total_tokens'],p['hidden_size'])) for cut in (22,23)}
                logits=np.lib.format.open_memmap(out/'teacher_logits.npy',mode='w+',dtype=np.int16,
                    shape=(len(items),8,p['vocabulary']));records=[]
                for index,case in enumerate(items):
                    values,teacher,row=native_outputs(native,case,p['label_ids'],guard=guard)
                    for cut in (22,23):banks[cut][case['offset']:case['stop']]=values[cut][0].contiguous().view(torch.int16).numpy()
                    logits[index]=teacher.contiguous().view(torch.int16).numpy()
                    if case['split']=='development':records.append(row)
                    progress(dict(completed=index+1,total=len(items)))
                for bank in (*banks.values(),logits):bank.flush();bank._mmap.close()
                unchanged=frozen==[(id(v),v._version) for v in (*native.parameters(),*native.buffers())]
                if not unchanged:raise ValueError('Original native teacher state changed')
                result=dict(status='completed',records=records,count=len(records),
                    correct=sum(r['prediction']==r['answer'] for r in records),boundary_controls=controls,
                    native_slots=p['native_slots'],storage='Lossless BF16 bits, native-endian int16',fixed_state_unchanged=unchanged)
                del native,banks,logits
            else:
                stack,norm,E,rotary,proof=builder.build_stack(p['model']['path'],{int(k):v for k,v in p['fits'].items()},
                    23,p['parent_checkpoint'],device='cpu',trainable=False,checkpoint_blocks=False)
                packed=builder.packed_clone(stack,device='cpu');del stack
                packed,norm,E,rotary=packed.to('cuda:0'),norm.to('cuda:0'),E.to('cuda:0'),rotary.to('cuda:0')
                dataset=RecoveryDataset(items,root/'native/boundary_23.npy',root/'native/teacher_logits.npy',
                    arm='native_legacy',label_ids=p['label_ids'])
                receipt=feature.capture_packed_parent(out,packed,norm,E,rotary,dataset,
                    parent_checkpoint_sha256=p['parent_checkpoint_sha256'],guard=guard,
                    progress=lambda value:progress(dict(parent_capture=value)),label_ids=p['label_ids'])
                result=dict(status='completed',parent_checkpoint_sha256=p['parent_checkpoint_sha256'],
                    receipt=receipt,inheritance=proof,alpha=ALPHA,normalization_energy=receipt['features']['normalization_energy'])
        guard();base.write(out/'result.json',result)
        base.write(out/'manifest.json',dict(status='completed',phase=phase,fingerprint=p['fingerprint'],
            artifacts={f.name:base.sha(f) for f in out.iterdir() if f.is_file()},lease_id=lease['id'],
            allocation=lease['allocation'],allocator=settings,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic()-started))
        return result
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True);base.write(out/'failure.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            fingerprint=p['fingerprint'],lease=lease,allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None));raise
    finally:
        if dataset is not None:dataset.close()
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','native','parent'))
    parser.add_argument('--root',required=True);parser.add_argument('--source');parser.add_argument('--data-root');args=parser.parse_args()
    if args.phase=='prepare':
        if not args.source or not args.data_root:parser.error('prepare requires source and data-root')
        result=prepare(args.root,args.source,args.data_root)
    else:result=run(args.root,args.phase)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
