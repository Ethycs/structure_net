"""Authenticated, read-only hidden trajectories after completed cut23 recovery.

Prepare on CPU; run only under a caller-owned RTX3060 reservation. Models are
reconstructed sequentially. No training, checkpoint selection or held-out data.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_3b_staged_campaign as source
from . import jspace_3b_hidden_feature_diagnostic as diagnostic

base, old, read = source.base, source.old, source.read
SCHEMA = 'jspace.3b-hidden-feature-run.v1'
PARENT_SHA = 'db53db32597ebb4947e38f7324e31e6eb9c7686f491a9e8b9745b0c43bca8292'
PROTOCOL = base.ROOT / 'docs/07 - Status Reports/jspace/2026-11-15_qwen3b-hidden-feature-recovery.md'
LAYERS = list(range(24,36))
SCOPE = 'Existing development trajectories only; training targets kept separate; no recovery, selection, held-out or speed claim'


def case_rows(items, original_indices):
    """Preserve case order and both cached-token and original-logit row mappings."""
    if (len(items)!=576 or len(original_indices)!=len(items) or len(set(original_indices))!=len(items)
            or any(type(i) is not int or i<0 for i in original_indices)
            or len({c['identity'] for c in items})!=len(items)):
        raise ValueError('Distinct original512 training and64 development identities required')
    rows=[]
    for index,(case,original) in enumerate(zip(items,original_indices)):
        length=case['stop']-case['offset']
        if (case['split'] not in ('train','development') or length!=len(case['input_ids'])
                or not 0<length<=416 or case['offset']<0 or len(case['loss_positions'])!=4
                or case['loss_positions'][-1]!=length-1):
            raise ValueError('Invalid original case split, offsets or answer position')
        rows.append(dict(source_index=index,original_logit_index=original,identity=case['identity'],family=case['family'],
            split=case['split'],native_offset=case['offset'],native_stop=case['stop'],length=length,answer_position=length-1))
    if sum(r['split']=='train' for r in rows)!=512 or sum(r['split']=='development' for r in rows)!=64:
        raise ValueError('Original512/64 split required; confirmation cases are excluded')
    return rows


def admit(campaign):
    """Use existing completion checks, including independent selected reload."""
    campaign=Path(campaign).resolve();parent,stage=source.verify_stage(campaign,23)
    if stage['parent_checkpoint_sha256']!=PARENT_SHA or stage['parent_cut']!=24:
        raise ValueError('The pinned accepted twelve-block parent is required')
    directory=source.stage_dir(campaign,23)
    evidence={phase:source.checked(directory,phase,stage) for phase in ('pilot','prealign','train','verify')}
    if not evidence['verify']['passed']:
        raise ValueError('Cut23 must complete fresh selected-checkpoint verification first')
    _,parent_stage=source.verify_stage(campaign,24)
    parent_train=source.checked(source.stage_dir(campaign,24),'train',parent_stage)
    parent_verify=source.checked(source.stage_dir(campaign,24),'verify',parent_stage)
    if (not parent_train['quality_pass'] or not parent_verify['passed']
            or parent_train['selected_checkpoint_sha256']!=PARENT_SHA):
        raise ValueError('The twelve-block reference must be accepted and freshly verified')
    return parent,stage,evidence


def prepare(root,campaign):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation must hide CUDA')
    root,campaign=Path(root).resolve(),Path(campaign).resolve()
    if root.exists():raise FileExistsError(root)
    parent,stage,evidence=admit(campaign);directory=source.stage_dir(campaign,23)
    items=read(parent['items_path'])['items'];rows=case_rows(items,parent['original_indices']);inputs={}
    paths=[campaign/'plan.json',directory/'plan.json',Path(parent['items_path']),
        Path(stage['parent_checkpoint']),Path(stage['parent_records']),directory/'pilot/initializer.pt',
        directory/'prealign/selected_entrance.pt',directory/'train/selected.pt',directory/'capture/packed_teacher_logits.npy']
    for cut in (23,24,36):paths.append(Path(parent['native_cache'])/f'boundary_{cut:02d}.npy')
    paths.append(Path(parent['native_cache'])/'teacher_logits.npy')
    for cut in (23,24):
        d=source.stage_dir(campaign,cut);paths.append(d/'plan.json')
        for phase in source.PHASES:
            paths.extend((d/phase/'manifest.json',d/phase/'result.json'))
    # Authenticate only consumed fits/arrays, not unused large Jacobian banks.
    for fit in stage['fits'].values():
        paths.extend(Path(fit)/name for name in ('fit.pt','fit.json','relation.pt','relations.json','interface.json'))
    for row in evidence['train']['history']:
        if row['epoch'] in (-1,0,evidence['train']['selected_epoch']):
            paths.append(directory/'train'/f"epoch_{row['epoch']:02d}_development.json")
    for path in set(paths):old.pin(inputs,path)
    sources=dict(parent['source_sha256'])
    extras=[Path(__file__),Path(diagnostic.__file__),PROTOCOL,
        base.ROOT/'tests/structure_net/test_jspace_3b_hidden_feature_run.py',
        base.ROOT/'tests/structure_net/test_jspace_3b_hidden_feature_diagnostic.py']
    for path in source.local_import_closure(extras):
        digest=base.sha(path)
        if str(path) in sources and sources[str(path)]!=digest:raise ValueError('Inherited source changed')
        sources[str(path)]=digest
    plan=dict(schema=SCHEMA,campaign=str(campaign),campaign_fingerprint=parent['fingerprint'],stage_fingerprint=stage['fingerprint'],
        cut=23,parent_cut=24,parent_checkpoint_sha256=PARENT_SHA,selected_checkpoint_sha256=evidence['train']['selected_checkpoint_sha256'],
        selected_epoch=evidence['train']['selected_epoch'],source_sha256=sources,input_sha256=inputs,
        input_identity={p:old.file_identity(p) for p in inputs},inherited_input_identity=parent['input_identity'],
        rows=rows,layers=LAYERS,batch_size=4,training_target='accepted_parent_actual_final_answer_coordinates',
        allocation_mib=11264,allocator_mib=11008,scope=SCOPE,confirmed=False)
    plan['fingerprint']=base.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        destination=root/'producer'/Path(path).relative_to(base.ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if base.sha(path)!=digest or base.sha(destination)!=digest:raise ValueError('Source changed during preparation')
    base.write(root/'plan.json',plan)
    return dict(status='prepared',fingerprint=plan['fingerprint'],training_cases=512,development_cases=64)


def verify(root):
    root=Path(root);plan=read(root/'plan.json');source.check_plan(plan)
    if (plan['schema']!=SCHEMA or plan['cut']!=23 or plan['parent_checkpoint_sha256']!=PARENT_SHA
            or plan['layers']!=LAYERS or plan['batch_size']!=4):raise ValueError('Diagnostic configuration changed')
    for path,digest in plan['source_sha256'].items():
        if base.sha(path)!=digest or base.sha(root/'producer'/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError('Diagnostic source/archive changed: '+path)
    for identities in (plan['input_identity'],plan['inherited_input_identity']):
        for path,identity in identities.items():
            if old.file_identity(path)!=identity:raise ValueError('Authenticated input changed: '+path)
    return plan


class Metrics:
    """Aggregate per-case errors before taking RMS, never average batch RMS."""
    def __init__(self):self.rows={}
    def add(self,name,actual,expected,batch,items,guard=None):
        value=diagnostic.feature_metrics(actual,expected,batch['lengths'],guard=guard)
        rows=self.rows.setdefault(name,[])
        for index,case,length,metric in zip(batch['indices'],items,batch['lengths'],value['per_case']):
            rows.append(dict(source_index=index,identity=case['identity'],family=case['family'],length=length,**metric))
    def finish(self,identities):
        result={}
        for name,rows in self.rows.items():
            if [r['identity'] for r in rows]!=identities:raise ValueError('Feature comparison lost development row alignment')
            mean=lambda key:sum(r[key] for r in rows)/len(rows)
            result[name]=dict(count=len(rows),valid_tokens=sum(r['length'] for r in rows),mse=mean('mse'),
                relative_rmse=math.sqrt(mean('relative_mse')),answer_mse=mean('answer_mse'),
                answer_relative_rmse=math.sqrt(mean('answer_relative_mse')),max_abs=max(r['max_abs'] for r in rows),per_case=rows)
        return result


class TraceBank:
    """Lossless BF16 bits of valid development rows; no padded or training rows."""
    def __init__(self,directory,rows,layers,rank,width,*,create=False):
        self.rows={};offset=0;self.layers=list(layers);self.directory=Path(directory)
        for row in rows:
            if row['split']!='development':raise ValueError('Development bank cannot include training cases')
            self.rows[row['source_index']]=(offset,offset+row['length']);offset+=row['length']
        if create:
            self.directory.mkdir()
            self.z=np.lib.format.open_memmap(self.directory/'coordinates.npy',mode='w+',dtype=np.int16,shape=(offset,len(layers),rank))
            self.h=np.lib.format.open_memmap(self.directory/'output.npy',mode='w+',dtype=np.int16,shape=(offset,width))
            base.write(self.directory/'rows.json',dict(rows=rows,layers=self.layers,coordinate_location='post-block before outgoing bridge'))
        else:
            self.z=np.load(self.directory/'coordinates.npy',mmap_mode='r');self.h=np.load(self.directory/'output.npy',mmap_mode='r')
    def write(self,trace,batch):
        if trace['output'].dtype!=torch.bfloat16 or any(z.dtype!=torch.bfloat16 for z in trace['coordinates'].values()):
            raise ValueError('Trace bank stores native BF16 bits without new quantization')
        for row,index in enumerate(batch['indices']):
            start,stop=self.rows[index];length=stop-start
            self.z[start:stop]=torch.stack([trace['coordinates'][layer][row,:length] for layer in self.layers],1).cpu().view(torch.int16).numpy()
            self.h[start:stop]=trace['output'][row,:length].cpu().view(torch.int16).numpy()
    def batch(self,indices,length,device):
        z=np.zeros((len(indices),length,*self.z.shape[1:]),dtype=np.int16);h=np.zeros((len(indices),length,self.h.shape[1]),dtype=np.int16)
        for row,index in enumerate(indices):
            start,stop=self.rows[index];z[row,:stop-start]=self.z[start:stop];h[row,:stop-start]=self.h[start:stop]
        z=torch.from_numpy(z).view(torch.bfloat16).to(device);h=torch.from_numpy(h).view(torch.bfloat16).to(device)
        return dict(coordinates={layer:z[:,:,i] for i,layer in enumerate(self.layers)},output=h)
    def close(self):
        for value in (self.z,self.h):
            if value.mode!='r':value.flush()
            value._mmap.close()


def trace_rank(stack,layers):
    ranks={b.layer_idx:b.rank for b in stack.blocks if b.layer_idx in layers}
    if set(ranks)!=set(layers) or len(set(ranks.values()))!=1:
        raise ValueError('Observed layers require an explicit common coordinate width')
    return next(iter(ranks.values()))


def final_answers(trace,batch,items):
    if any(c['split']!='train' for c in items):raise ValueError('Only training targets may enter future normalization')
    z=trace['coordinates'][35]
    return z[torch.arange(len(items),device=z.device),torch.tensor(batch['lengths'],device=z.device)-1].detach()


def load_mode(parent,stage,mode,device):
    is_parent=mode.startswith('parent')
    stack,norm,E,rotary,inheritance=source.make_stack(parent,stage,'cpu',parent=is_parent)
    # Match SuffixTrainer's parameter canonicalization, without an optimizer.
    for p in (*stack.parameters(),*norm.parameters()):p.data=p.data.bfloat16()
    directory=source.stage_dir(Path(parent['_campaign']),23)
    path=Path(stage['parent_checkpoint']) if is_parent else directory/('train/selected.pt' if mode=='selected' else 'pilot/initializer.pt')
    state=source.native._load(path);stack.load_state_dict(state['stack'],strict=True);norm.load_state_dict(state['norm'],strict=True)
    if not old.state_equal(stack.state_dict(),state['stack']) or not old.state_equal(norm.state_dict(),state['norm']):
        raise ValueError('Strict reconstruction changed stored checkpoint tensors')
    del state
    if mode=='warm':
        delta=source.native._load(directory/'prealign/selected_entrance.pt')
        stack.blocks[0].load_state_dict(delta['block'],strict=True);del delta
    stack.checkpoint_blocks=False
    if mode!='parent_unpacked':stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True)
    for module in (stack,norm,rotary):module.eval().requires_grad_(False).to(device)
    E=E.detach().to(device)
    return stack,norm,E,rotary,dict(checkpoint=str(path),checkpoint_sha256=base.sha(path),inheritance=inheritance,
        packed=bool(stack.packed),strict_state_load=True,parameters_bf16=True)


@torch.inference_mode()
def task_rows(hidden,norm,E,batch,items,label_ids):
    h=hidden[torch.arange(len(items),device=hidden.device)[:,None],batch['loss_positions']]
    logits=F.linear(norm(h),E).float()
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite diagnostic logits')
    rows=[]
    for row,case in enumerate(items):
        values=logits[row,-1,label_ids[:len(case['choices'])]]
        rows.append(dict(identity=case['identity'],family=case['family'],answer=case['answer'],prediction=int(values.argmax()),logits=values.cpu().tolist()))
    return rows


def compare_trace(metrics,prefix,actual,expected,batch,items,guard):
    for layer,z in actual['coordinates'].items():metrics.add(f'{prefix}/layer_{layer:02d}',z,expected['coordinates'][layer],batch,items,guard)
    metrics.add(prefix+'/output',actual['output'],expected['output'],batch,items,guard)


def run(root):
    from .jspace_full_lease import wait_ready
    root=Path(root);plan=verify(root);output=root/'run'
    if output.exists():raise FileExistsError(output)
    parent,stage=source.verify_stage(Path(plan['campaign']),23)
    if parent['fingerprint']!=plan['campaign_fingerprint'] or stage['fingerprint']!=plan['stage_fingerprint']:
        raise ValueError('Source campaign/stage changed')
    parent=dict(parent,_campaign=plan['campaign']);directory=source.stage_dir(Path(plan['campaign']),23)
    items=read(parent['items_path'])['items']
    if case_rows(items,parent['original_indices'])!=plan['rows']:raise ValueError('Case identity/row map changed')
    lease=wait_ready([0]);allocator=source.cap_allocator(lease,plan)
    torch.cuda.init();torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    output.mkdir();started=time.monotonic();data=[];bank=None
    def guard():
        current=wait_ready([0])
        if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:raise RuntimeError('Reservation changed')
        if torch.cuda.max_memory_reserved()>allocator['allocator_limit_bytes']:raise MemoryError('Reservation cap exceeded')
    try:
        teacher=directory/'capture/packed_teacher_logits.npy'
        data=[source.data_for(parent,cut,teacher) for cut in (23,24,36)]
        if any(d.items!=items for d in data):raise ValueError('Native boundary row identities changed')
        dev=data[0].development;dev_rows=[plan['rows'][i] for i in dev];identities=[r['identity'] for r in dev_rows]
        cases=[items[i] for i in dev];all_metrics=Metrics();modes={};parent_records=[];native_records=[]
        for mode in ('parent_packed','parent_unpacked','raw','warm','selected'):
            guard();stack,norm,E,rotary,provenance=load_mode(parent,stage,mode,'cuda:0')
            if mode=='parent_packed':bank=TraceBank(output/'parent_development',dev_rows,LAYERS,trace_rank(stack,LAYERS),stack.width,create=True)
            rows=[];execution=[];diff_rows=[]
            incoming=data[1] if mode.startswith('parent') else data[0]
            for start in range(0,len(dev),4):
                guard();indices=dev[start:start+4];batch=incoming.batch(indices,device=E.device,include_original=mode=='parent_packed')
                current=[items[i] for i in indices]
                with torch.no_grad():rope=rotary(batch['hidden'],batch['position_ids'])
                kwargs=dict(position_ids=batch['position_ids'],position_embeddings=rope,attention_mask=batch['attention_mask'],layers=LAYERS,guard=guard)
                trace=diagnostic.trace_coordinates(stack,batch['hidden'],mode='inference',**kwargs)
                rows.extend(task_rows(trace['output'],norm,E,batch,current,parent['label_ids']));execution.append(trace['execution'])
                if mode=='parent_packed':
                    if not all(v['fuse_qkv'] and v['fuse_mlp'] for v in trace['execution'].values()):
                        raise ValueError('Packed inference did not exercise the frozen fused path')
                    bank.write(trace,batch)
                    differentiated=diagnostic.trace_coordinates(stack,batch['hidden'],mode='differentiable',**kwargs)
                    if any(v['fuse_qkv'] or v['fuse_mlp'] for v in differentiated['execution'].values()):raise ValueError('Input-autograd path unexpectedly fused')
                    compare_trace(all_metrics,'parent_packed_input_autograd',differentiated,trace,batch,current,guard)
                    diff_rows.extend(task_rows(differentiated['output'],norm,E,batch,current,parent['label_ids']));del differentiated
                    for row,case in enumerate(current):
                        values=batch['original'][row,-1,parent['label_ids'][:len(case['choices'])]].float()
                        native_records.append(dict(identity=case['identity'],family=case['family'],answer=case['answer'],prediction=int(values.argmax()),logits=values.cpu().tolist()))
                else:
                    reference=bank.batch(indices,batch['hidden'].shape[1],E.device)
                    compare_trace(all_metrics,mode+'/versus_parent',trace,reference,batch,current,guard);del reference
                native_batch=data[2].batch(indices,device=E.device)
                projected=diagnostic.native_final_coordinates(stack,native_batch['hidden'],guard=guard)
                all_metrics.add(mode+'/versus_native_projected_final',trace['coordinates'][35],projected,batch,current,guard)
                reconstructed=diagnostic.reconstruct_final(stack,trace['coordinates'][35],batch['position_ids'],guard=guard)
                roundtrip=diagnostic.native_final_coordinates(stack,reconstructed,guard=guard)
                all_metrics.add(mode+'/actual_exit_roundtrip',roundtrip,trace['coordinates'][35],batch,current,guard)
                if mode=='parent_packed':
                    native_reconstructed=diagnostic.reconstruct_final(stack,projected,batch['position_ids'],guard=guard)
                    native_roundtrip=diagnostic.native_final_coordinates(stack,native_reconstructed,guard=guard)
                    all_metrics.add('native_projected_roundtrip',native_roundtrip,projected,batch,current,guard)
                    all_metrics.add('native_projected_reconstruction',native_reconstructed,native_batch['hidden'],batch,current,guard)
                    del native_reconstructed,native_roundtrip
                del trace,batch,native_batch,projected,reconstructed,roundtrip,rope
                base.event(root/'progress.jsonl','development_batch',mode=mode,completed=start+len(indices),total=64)
            reference=source.records(stage['parent_records']) if mode.startswith('parent') else read(directory/'train'/f"epoch_{({'raw':-1,'warm':0,'selected':plan['selected_epoch']}[mode]):02d}_development.json")['records']
            parity=old.compare_rows(rows,reference,cases)
            if mode in ('parent_packed','raw','warm','selected') and not parity['passed']:
                raise RuntimeError('Authenticated packed development replay failed: '+mode)
            if mode=='parent_packed':
                parent_records=rows
                modes['parent_packed_input_autograd']=dict(records=diff_rows,comparison=old.compare_rows(diff_rows,rows,cases),diagnostic_only=True)
                targets=[];target_rows=[]
                for start in range(0,len(incoming.training),4):
                    indices=incoming.training[start:start+4];batch=incoming.batch(indices,device=E.device);current=[items[i] for i in indices]
                    with torch.no_grad():rope=rotary(batch['hidden'],batch['position_ids'])
                    trace=diagnostic.trace_coordinates(stack,batch['hidden'],position_ids=batch['position_ids'],position_embeddings=rope,
                        attention_mask=batch['attention_mask'],layers=[35],guard=guard)
                    targets.append(final_answers(trace,batch,current).cpu());target_rows.extend(plan['rows'][i] for i in indices)
                    del batch,trace,rope
                    if start%64==0:base.event(root/'progress.jsonl','training_target_batch',completed=start+len(indices),total=512)
                target=torch.cat(targets);energy=target.float().square().mean(-1)
                if target.dtype!=torch.bfloat16 or not torch.isfinite(energy).all():raise ValueError('Invalid BF16 training targets')
                with (output/'parent_training_final_answers.npy').open('xb') as stream:np.save(stream,target.view(torch.int16).numpy(),allow_pickle=False)
                base.write(output/'parent_training_final_answers.json',dict(rows=target_rows,shape=list(target.shape),dtype='BF16 bits as int16',
                    teacher_checkpoint_sha256=PARENT_SHA,coordinate_layer=35,position='last valid token',
                    energy_mean=float(energy.mean()),energy_median=float(energy.median()),energy_min=float(energy.min()),
                    coefficient_selected=False,normalization_fitted_on_development=False))
                del target,targets,energy
            modes[mode]=dict(**provenance,records=rows,source_replay=parity,versus_parent=old.compare_rows(rows,parent_records,cases),execution=execution,
                no_parameter_gradients=all(p.grad is None and not p.requires_grad for p in stack.parameters()))
            if not modes[mode]['no_parameter_gradients']:raise RuntimeError('Diagnostic changed gradient state')
            base.write(output/f'{mode}.json',modes[mode]);del stack,norm,E,rotary;gc.collect();torch.cuda.empty_cache()
        bank.close();bank=None;guard()
        result=dict(status='completed',fingerprint=plan['fingerprint'],modes=modes,native_cached_records=native_records,
            feature_metrics=all_metrics.finish(identities),training_target_count=512,development_count=64,
            scope=SCOPE,confirmed=False,elapsed_seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved())
        base.write(output/'result.json',result)
        base.write(output/'manifest.json',dict(status='completed',fingerprint=plan['fingerprint'],lease_id=lease['id'],allocation=lease['allocation'],allocator=allocator,
            artifacts={str(p.relative_to(output)):base.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
        return dict(status='completed',fingerprint=plan['fingerprint'],result=str(output/'result.json'))
    except BaseException as error:
        base.write(output/'failure.json',dict(status='failed',error=repr(error),error_type=type(error).__name__,lease_id=lease['id'],peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        raise
    finally:
        if bank is not None:bank.close()
        for dataset in data:dataset.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','run'))
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--source-root',type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.source_root) if args.phase=='prepare' else run(args.root),allow_nan=False),flush=True)
