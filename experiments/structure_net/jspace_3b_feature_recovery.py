"""Arm A: accepted-parent final-answer features plus unchanged cautious output KD.

This separate branch reuses the authenticated cut23 raw/warm initializers and
CPU-master optimizer. It neither advances the campaign nor opens confirmation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from torch.utils.checkpoint import checkpoint

from . import jspace_3b_hidden_feature_run as capture
from . import jspace_qwen_compact_stack as compact
from .jspace_resident_adam_replay import DiagnosticCheckpoint

source=capture.source
base,old,read,training,core=source.base,source.old,source.read,source.training,source.core
CONFIG=training.CONFIG
SCHEMA='jspace.3b-final-feature-recovery.v1'
PROTOCOL=base.ROOT/'docs/07 - Status Reports/jspace/2026-11-15_qwen3b-final-feature-recovery.md'
RATIO=.1
PHASES=('pilot','train','verify')


def calibration_indices(items):
    families=sorted({c['family'] for c in items if c['split']=='train'})
    if len(families)!=4:raise ValueError('Four training families required')
    selected=[]
    for family in families:
        rows=[i for i,c in enumerate(items) if c['split']=='train' and c['family']==family]
        if len(rows)<8:raise ValueError('Eight training cases per family required')
        selected.extend(rows[:8])
    return sorted(selected)


class AnswerTargets:
    """Only authenticated training targets, indexed by original source row."""
    def __init__(self,values,indices,*,normalization_energy=None):
        if (values.ndim!=2 or values.dtype!=torch.bfloat16 or values.device.type!='cpu'
                or len(values)!=len(indices) or len(set(indices))!=len(indices) or not len(indices)
                or any(type(i) is not int or i<0 for i in indices) or not torch.isfinite(values).all()):
            raise ValueError('Finite distinct BF16 CPU training-answer targets required')
        self.values=values.detach();self.lookup={index:row for row,index in enumerate(indices)}
        measured=float(values.float().square().mean())
        self.energy=measured if normalization_energy is None else float(normalization_energy)
        if not math.isfinite(self.energy) or self.energy<=0:raise ValueError('Positive finite training-only energy required')
        if not math.isclose(self.energy,measured,rel_tol=1e-6,abs_tol=0):raise ValueError('Authenticated training energy does not match targets')
    def batch(self,indices,device):
        if not indices or any(i not in self.lookup for i in indices):raise ValueError('Feature targets are training-only')
        return self.values[[self.lookup[i] for i in indices]].to(device=device,dtype=torch.float32)


def load_targets(root):
    root=Path(root);plan=capture.verify(root);path=root/'run';manifest=read(path/'manifest.json')
    if manifest.get('status')!='completed' or manifest.get('fingerprint')!=plan['fingerprint']:
        raise ValueError('Completed matching hidden-feature diagnostic required')
    names=('result.json','parent_training_final_answers.json','parent_training_final_answers.npy')
    for name in names:
        if manifest.get('artifacts',{}).get(name)!=base.sha(path/name):raise ValueError('Diagnostic target/result changed')
    result=read(path/'result.json');metadata=read(path/names[1])
    expected=[r for r in plan['rows'] if r['split']=='train']
    if (result.get('status')!='completed' or result.get('fingerprint')!=plan['fingerprint']
            or not result['modes']['parent_packed']['source_replay']['passed']
            or metadata.get('rows')!=expected or len(expected)!=512 or metadata.get('shape')!=[512,1024]
            or metadata.get('teacher_checkpoint_sha256')!=capture.PARENT_SHA or metadata.get('coordinate_layer')!=35
            or metadata.get('position')!='last valid token' or metadata.get('normalization_fitted_on_development') is not False):
        raise ValueError('Actual accepted-parent training feature identity changed')
    bits=np.load(path/names[2],allow_pickle=False)
    if bits.dtype!=np.dtype('int16') or not bits.dtype.isnative or list(bits.shape)!=metadata['shape']:
        raise ValueError('Expected lossless BF16 target bits')
    targets=AnswerTargets(torch.from_numpy(bits).view(torch.bfloat16),[r['source_index'] for r in expected],normalization_energy=metadata['energy_mean'])
    return targets,dict(diagnostic_root=str(root.resolve()),diagnostic_fingerprint=plan['fingerprint'],
        teacher_checkpoint_sha256=capture.PARENT_SHA,target_sha256=base.sha(path/names[2]),
        metadata_sha256=base.sha(path/names[1]),manifest_sha256=base.sha(path/'manifest.json'),
        training_count=512,normalization_energy=targets.energy,normalization_scope='all512TRAIN targets only')


def forward_features(stack,hidden,*,position_ids,position_embeddings,attention_mask=None):
    """Original uncached Torch forward, also returning actual pre-exit coordinates.

    No global observer or reconstruction/projection surrogate survives backward.
    Per-block checkpointing, fixed bridges, masks and arithmetic match the
    frozen CompactQwenStack.forward; packed inference still uses that original.
    """
    if not isinstance(stack,compact.CompactQwenStack) or stack.packed or stack.stripped:
        raise ValueError('Training requires the unpacked unstripped compact stack')
    if hidden.ndim!=3 or hidden.shape[-1]!=stack.width or position_embeddings is None:
        raise ValueError('Explicit native-width hidden state and rotary embeddings required')
    if stack.geometry_compute_dtype is not None and hidden.dtype!=stack.geometry_compute_dtype:
        raise ValueError('Worker geometry dtype mismatch')
    if any(b.capture_mlp for b in stack.blocks):raise ValueError('Feature training cannot capture intermediate MLP activations')
    positions=[b._positions(hidden,position_ids[0],position_ids,None) for b in stack.blocks]
    z=stack._geometry(hidden,stack.blocks[0].basis.T)
    for index,block in enumerate(stack.blocks):
        if index:
            bridge=stack.bridges[index-1];p=positions[index-1+bridge.position_source]
            z=stack._geometry(z,bridge.weight,block._rows(bridge.shift,p,*z.shape[:2]))
        def execute(value,i=index,current=block):
            return compact._run_reduced(current,value,positions[i],attention_mask=attention_mask,
                past_key_value=None,cache_position=position_ids[0],position_embeddings=position_embeddings,
                fuse_qkv=False,fuse_mlp=False,padded_width=current.intermediate_size+current.polynomial_feature_count,
                feature_i=getattr(stack,f'feature_i_{i}'),feature_j=getattr(stack,f'feature_j_{i}'))[0]
        z=checkpoint(execute,z,use_reentrant=False) if stack.checkpoint_blocks and torch.is_grad_enabled() else execute(z)
    last=stack.blocks[-1];anchor=last._rows(last.anchor,positions[-1],*hidden.shape[:2])
    return stack._geometry(z,last.basis,anchor),z


def feature_loss(coordinates,batch,targets):
    answer=coordinates[torch.arange(len(batch['indices']),device=coordinates.device),
        torch.tensor(batch['lengths'],device=coordinates.device)-1].float()
    teacher=targets.batch(batch['indices'],coordinates.device).detach()
    if answer.shape!=teacher.shape:raise ValueError('Actual final compact coordinate width changed')
    return (answer-teacher).square().mean()/targets.energy


def gradient_squared(loss,parameters,*,retain_graph=False):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=False)
    value=sum(float(g.detach().float().square().sum()) for g in gradients)
    del gradients
    if not math.isfinite(value):raise FloatingPointError('Nonfinite calibration gradient')
    return value


class FeatureTrainer(training.CautiousTrainer):
    """Only train_window changes; fit, packed scoring and selection are inherited."""
    def __init__(self,*args,targets,alpha=None,**kwargs):
        super().__init__(*args,**kwargs)
        if type(self.optimizer) is not core.StreamingCPUAdamW:raise ValueError('This arm freezes the CPU-master optimizer')
        if set(targets.lookup)!=set(self.dataset.training):raise ValueError('Feature targets must match only all training rows')
        self.feature_targets,self.alpha=targets,alpha
        if alpha is not None and (not math.isfinite(alpha) or alpha<=0):raise ValueError('Positive fixed feature coefficient required')
    def objectives(self,batch):
        rope=self.rotary(batch['hidden'],batch['position_ids'])
        hidden,z=forward_features(self.stack,batch['hidden'],position_ids=batch['position_ids'],position_embeddings=rope,
            attention_mask=batch['attention_mask'])
        h=hidden[torch.arange(len(batch['indices']),device=self.device)[:,None],batch['loss_positions']]
        logits=F.linear(self.norm(h),self.readout).float()
        endpoint,kl,ce=core.output_objective(logits,batch['teacher'],batch['targets'],self.config)
        return endpoint,kl,ce,feature_loss(z,batch,self.feature_targets)
    def calibrate(self,indices):
        if self.updates or self.finished or self.alpha is not None:raise ValueError('Calibrate untouched warm weights once')
        if list(indices)!=calibration_indices(self.dataset.items):raise ValueError('Fixed first8TRAIN/family calibration required')
        self._guard();self.stack.train();self.norm.train();self.optimizer.zero_grad(set_to_none=True)
        before=training.state_fingerprint(dict(stack=self.stack.state_dict(),norm=self.norm.state_dict()))
        parameters=list(self.stack.parameters());groups=[]
        for start in range(0,len(indices),self.config.microbatch):
            self._guard();chosen=indices[start:start+self.config.microbatch]
            batch=self.dataset.batch(chosen,device=self.device,dtype=self.dtype)
            endpoint,kl,ce,feature=self.objectives(batch)
            endpoint_norm=gradient_squared(endpoint,parameters,retain_graph=True)
            feature_norm=gradient_squared(feature,parameters)
            groups.append(dict(indices=chosen,identities=[self.dataset.items[i]['identity'] for i in chosen],
                endpoint_gradient_squared=endpoint_norm,feature_gradient_squared=feature_norm,
                endpoint_loss=float(endpoint.detach()),feature_loss=float(feature.detach())))
            del batch,endpoint,kl,ce,feature
        a=sum(g['endpoint_gradient_squared'] for g in groups);b=sum(g['feature_gradient_squared'] for g in groups)
        if not all(math.isfinite(v) and v>0 for v in (a,b)):raise ValueError('Degenerate gradient calibration; no implicit fallback')
        self.alpha=RATIO*math.sqrt(a/b)
        after=training.state_fingerprint(dict(stack=self.stack.state_dict(),norm=self.norm.state_dict()))
        unchanged=before==after and self.updates==0 and all(p.grad is None for p in self.parameters)
        if not unchanged:raise RuntimeError('Calibration mutated warm weights or left gradients')
        return dict(alpha=self.alpha,ratio=RATIO,groups=groups,indices=list(indices),count=32,
            endpoint_gradient_squared=a,feature_gradient_squared=b,normalization_energy=self.feature_targets.energy,
            normalization_scope='all512TRAIN targets only',gradient_parameters='all compressed block parameters; final norm excluded',
            grouping='source order; identical microbatches for both objectives',state_before=before,state_after=after,
            unchanged_warm_state=True,optimizer_updates=0,development_used=False)
    def train_window(self,indices,*,learning_rate=None):
        indices=list(indices)
        if self.finished or self.alpha is None:raise ValueError('Unfinished trainer with frozen calibration required')
        if not indices or any(i not in self.dataset.training for i in indices):raise ValueError('Only training updates are allowed')
        self._guard();self.stack.train();self.norm.train();self.optimizer.zero_grad(set_to_none=True)
        if learning_rate is not None:self.optimizer.param_groups[0]['lr']=learning_rate
        totals=dict(loss=0.,endpoint_loss=0.,kl=0.,ce=0.,feature_loss=0.)
        for start in range(0,len(indices),self.config.microbatch):
            chosen=indices[start:start+self.config.microbatch];batch=self.dataset.batch(chosen,device=self.device,dtype=self.dtype)
            endpoint,kl,ce,feature=self.objectives(batch);loss=endpoint+self.alpha*feature
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint feature-distillation loss')
            weight=len(chosen)/len(indices);(loss*weight).backward()
            for key,value in zip(totals,(loss,endpoint,kl,ce,feature)):totals[key]+=float(value.detach())*weight
            del batch,endpoint,kl,ce,feature,loss
        present=[all(p.grad is not None for p in b.parameters()) for b in self.stack.blocks]
        if not all(present) or any(p.grad is None for p in self.norm.parameters()):raise RuntimeError('All blocks and norm must receive gradients')
        update=self.optimizer.step(max_grad_norm=self.config.gradient_clip);self.optimizer.zero_grad(set_to_none=True);self.updates+=1
        return dict(**totals,alpha=self.alpha,normalization_energy=self.feature_targets.energy,cases=len(indices),update=self.updates,
            learning_rate=self.optimizer.param_groups[0]['lr'],gradients_present_by_block=present,optimizer=update)


def prepare(root,diagnostic_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation must hide CUDA')
    root,diagnostic_root=Path(root).resolve(),Path(diagnostic_root).resolve()
    if root.exists():raise FileExistsError(root)
    diagnostic_plan=capture.verify(diagnostic_root);targets,target_receipt=load_targets(diagnostic_root)
    parent,stage,evidence=capture.admit(diagnostic_plan['campaign']);directory=source.stage_dir(diagnostic_plan['campaign'],23)
    items=read(parent['items_path'])['items'];indices=calibration_indices(items)
    if set(targets.lookup)!={i for i,c in enumerate(items) if c['split']=='train'}:raise ValueError('Training target identity mismatch')
    inputs=dict(diagnostic_plan['input_sha256'])
    for name in ('plan.json','run/manifest.json','run/result.json','run/parent_training_final_answers.json','run/parent_training_final_answers.npy'):
        old.pin(inputs,diagnostic_root/name)
    sources=dict(diagnostic_plan['source_sha256'])
    for path in source.local_import_closure([Path(__file__),PROTOCOL,base.ROOT/'tests/structure_net/test_jspace_3b_feature_recovery.py']):
        digest=base.sha(path)
        if str(path) in sources and sources[str(path)]!=digest:raise ValueError('Frozen inherited source changed')
        sources[str(path)]=digest
    plan=dict(schema=SCHEMA,arm='accepted_parent_final_answer_features',cut=23,source_root=diagnostic_plan['campaign'],
        source_fingerprint=parent['fingerprint'],stage_fingerprint=stage['fingerprint'],diagnostic_root=str(diagnostic_root),
        target_receipt=target_receipt,config=asdict(CONFIG),calibration_indices=indices,gradient_ratio=RATIO,
        teacher_checkpoint_sha256=capture.PARENT_SHA,raw_checkpoint_sha256=evidence['pilot']['initializer_sha256'],
        prealigned_checkpoint_sha256=evidence['prealign']['selected_checkpoint_sha256'],
        historical_selected=evidence['train']['selected'],historical_selected_epoch=evidence['train']['selected_epoch'],
        protected_initial=evidence['pilot']['initial'],historical_warm=evidence['train']['warm_initial'],
        optimizer='StreamingCPUAdamW',selection_policy=training.SELECTION,eligibility_tolerance=training.TOLERANCE,
        allocation_mib=11264,allocator_mib=11008,input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        inherited_input_identity=diagnostic_plan['inherited_input_identity'],source_sha256=sources,
        diagnostic_checkpoint_policy='highest_correct_then_lowest_ce_earliest; never_auto_adopt',confirmed=False)
    plan['fingerprint']=base.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        destination=root/'producer'/Path(path).relative_to(base.ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        if base.sha(path)!=digest or base.sha(destination)!=digest:raise ValueError('Source archival changed')
    base.write(root/'plan.json',plan);return dict(status='prepared',fingerprint=plan['fingerprint'])


def verify_plan(root):
    root=Path(root);plan=read(root/'plan.json');source.check_plan(plan)
    if (plan.get('schema')!=SCHEMA or plan.get('cut')!=23 or plan.get('config')!=asdict(CONFIG)
            or plan.get('gradient_ratio')!=RATIO or plan.get('optimizer')!='StreamingCPUAdamW'
            or plan.get('selection_policy')!=training.SELECTION or plan.get('teacher_checkpoint_sha256')!=capture.PARENT_SHA
            or plan.get('eligibility_tolerance')!=training.TOLERANCE or plan.get('allocation_mib')!=11264
            or plan.get('allocator_mib')!=11008 or plan.get('arm')!='accepted_parent_final_answer_features'):
        raise ValueError('Feature-recovery prospective recipe changed')
    for path,digest in plan['source_sha256'].items():
        if base.sha(path)!=digest or base.sha(root/'producer'/Path(path).relative_to(base.ROOT))!=digest:raise ValueError('Producer/source archive changed')
    for identities in (plan['input_identity'],plan['inherited_input_identity']):
        for path,identity in identities.items():
            if old.file_identity(path)!=identity:raise ValueError('Authenticated input changed: '+path)
    parent,stage=source.verify_stage(plan['source_root'],23)
    if parent['fingerprint']!=plan['source_fingerprint'] or stage['fingerprint']!=plan['stage_fingerprint']:raise ValueError('Source stage changed')
    return plan,parent,stage


def validate_calibration(value,plan):
    groups=value.get('groups',[]);indices=[i for group in groups for i in group.get('indices',[])]
    a=value.get('endpoint_gradient_squared',0.);b=value.get('feature_gradient_squared',0.)
    if (indices!=plan['calibration_indices'] or value.get('indices')!=indices or len(indices)!=32
            or len(groups)!=8 or any(len(g['indices'])!=4 for g in groups)
            or not all(math.isfinite(x) and x>0 for x in (a,b))
            or a!=sum(g['endpoint_gradient_squared'] for g in groups)
            or b!=sum(g['feature_gradient_squared'] for g in groups)
            or value.get('ratio')!=RATIO or value.get('alpha')!=RATIO*math.sqrt(a/b)
            or value.get('normalization_energy')!=plan['target_receipt']['normalization_energy']
            or not value.get('unchanged_warm_state') or value.get('state_before')!=value.get('state_after')
            or value.get('development_used') is not False or value.get('optimizer_updates')!=0):
        raise ValueError('Authenticated train-only gradient calibration changed')


def checked(root,phase,plan):
    path=Path(root)/phase;manifest=read(path/'manifest.json')
    if manifest.get('status')!='completed' or manifest.get('fingerprint')!=plan['fingerprint'] or manifest.get('phase')!=phase:
        raise ValueError('Matching completed feature phase required')
    artifacts=manifest['artifacts']
    if 'result.json' not in artifacts:raise ValueError('Missing authenticated result')
    for name,digest in artifacts.items():
        target=path/name
        if not target.resolve().is_relative_to(path.resolve()) or base.sha(target)!=digest:raise ValueError('Phase artifact changed')
    result=read(path/'result.json')
    if result.get('status')!='completed':raise ValueError('Incomplete feature phase')
    if phase=='pilot':
        calibration=read(path/'calibration.json');probe=result.get('probe',{})
        validate_calibration(calibration,plan)
        if (not result.get('passed') or not result.get('discarded') or not result.get('systems_only')
                or not result.get('raw_replay_exact') or not result.get('warm_replay_exact')
                or calibration!=result.get('calibration')
                or not probe.get('passed') or probe.get('fp32_master_changed_by_block')!=[True]*13
                or probe.get('update',{}).get('update')!=1 or not probe.get('norm_fp32_master_changed')
                or not probe.get('fixed_state_unchanged') or result.get('raw')!=plan['protected_initial']
                or result.get('warm')!=plan['historical_warm']):
            raise ValueError('Discarded feature pilot or train-only calibration invalid')
    elif phase=='train':
        pilot=checked(root,'pilot',plan);history=result.get('history',[])
        if [r.get('epoch') for r in history]!=list(range(-1,13)):raise ValueError('Full protected/warm/twelve-epoch history required')
        source._epoch_checks(path,history,artifacts,quality=True)
        updates=[u for row in history for u in row['training']]
        if ([u.get('update') for u in updates]!=list(range(1,385))
                or any(u.get('gradients_present_by_block')!=[True]*13 or u.get('alpha')!=pilot['calibration']['alpha']
                    or u.get('normalization_energy')!=plan['target_receipt']['normalization_energy']
                    or not all(math.isfinite(u[k]) for k in ('loss','endpoint_loss','kl','ce','feature_loss')) for u in updates)
                or result.get('config')!=asdict(CONFIG) or result.get('completed_epochs')!=12 or result.get('updates')!=384
                or result.get('initial')!=plan['protected_initial'] or result.get('warm_initial')!=plan['historical_warm']
                or not result.get('fixed_state_unchanged') or not result.get('strict_reload_exact')):
            raise ValueError('Feature-training budget, fixed state or objective changed')
        winner=training.best_eligible(history,result['initial'])
        diagnostic=read(path/'diagnostic_best_task.json');task_winner=max(history,key=lambda r:training.selection_key(r['metrics']))
        if (winner['metrics']!=result['selected'] or winner['epoch']!=result['selected_epoch']
                or winner.get('checkpoint_sha256')!=result.get('selected_checkpoint_sha256')
                or artifacts.get('selected.pt')!=result.get('selected_checkpoint_sha256')
                or diagnostic.get('epoch')!=task_winner['epoch'] or diagnostic.get('metrics')!=task_winner['metrics']
                or diagnostic.get('automatically_adopted') is not False
                or artifacts.get('diagnostic_best_task.pt')!=diagnostic.get('checkpoint_sha256')):
            raise ValueError('Eligible or separate diagnostic checkpoint selection changed')
        objective=read(path/'feature_objective.json')
        if objective!=dict(target_receipt=plan['target_receipt'],calibration=pilot['calibration'],optimizer='StreamingCPUAdamW',selection_unchanged=True):
            raise ValueError('Training feature objective differs from pilot')
    elif phase=='verify':
        trained=checked(root,'train',plan);reference=read(Path(root)/'train'/f"epoch_{trained['selected_epoch']:02d}_development.json")
        if (not result.get('passed') or not result.get('exact_packed_replay') or not result.get('fixed_state_unchanged')
                or result.get('selected_checkpoint_sha256')!=trained['selected_checkpoint_sha256']
                or result.get('metrics')!=reference['metrics'] or result.get('records')!=reference['records']):
            raise ValueError('Fresh strict packed verification failed')
    else:raise ValueError('Unknown feature phase')
    return result


def run(root,phase):
    from .jspace_full_lease import wait_ready
    root=Path(root);plan,parent,stage=verify_plan(root);output=root/phase
    if phase not in PHASES:raise ValueError('Unknown phase')
    if output.exists():raise FileExistsError(output)
    previous=checked(root,'pilot',plan) if phase in ('train','verify') else None
    trained=checked(root,'train',plan) if phase=='verify' else None
    targets,receipt=load_targets(plan['diagnostic_root'])
    if receipt!=plan['target_receipt']:raise ValueError('Target provenance changed')
    directory=source.stage_dir(plan['source_root'],23);pilot=source.checked(directory,'pilot',stage)
    source.checked(directory,'prealign',stage);lease=None;data=None;started=time.monotonic()
    try:
        lease=wait_ready([0]);allocator=source.cap_allocator(lease,plan);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(CONFIG.seed)
        def guard():
            current=wait_ready([0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:raise RuntimeError('Reservation changed')
            if torch.cuda.max_memory_reserved()>allocator['allocator_limit_bytes']:raise MemoryError('Reservation cap exceeded')
        stack,norm,E,rotary,inheritance=source.make_stack(parent,stage,'cuda:0')
        for p in (*stack.parameters(),*norm.parameters()):p.data=p.data.bfloat16()
        raw=source.native._load(directory/'pilot/initializer.pt');delta=source.native._load(directory/'prealign/selected_entrance.pt')
        source.validate_delta(raw,delta,pilot,23);stack.load_state_dict(raw['stack'],strict=True);norm.load_state_dict(raw['norm'],strict=True)
        stack.blocks[0].load_state_dict(delta['block'],strict=True);del delta
        data=source.data_for(parent,23,directory/'capture/packed_teacher_logits.npy')
        trainer=FeatureTrainer(stack,norm,E,rotary,data,parent['label_ids'],targets=targets,
            alpha=None if previous is None else previous['calibration']['alpha'],config=CONFIG,guard=guard,
            native_correct=48,parent_correct=stage['parent_correct'])
        if phase=='pilot':
            output.mkdir();warm=trainer.snapshot();warm_digest=training.state_fingerprint(warm)
            try:
                stack.load_state_dict(raw['stack'],strict=True);norm.load_state_dict(raw['norm'],strict=True)
                raw_metrics,raw_rows=trainer.packed_development()
                if raw_metrics!=pilot['initial'] or raw_rows!=pilot['initial_records']:raise RuntimeError('Raw packed initializer replay changed')
            finally:
                stack.load_state_dict(warm['stack'],strict=True);norm.load_state_dict(warm['norm'],strict=True)
            if training.state_fingerprint(dict(stack=stack.state_dict(),norm=norm.state_dict()))!=warm_digest:raise RuntimeError('Raw replay changed warm state')
            del warm;gc.collect();metrics,rows=trainer.packed_development();reference=read(directory/'train/epoch_00_development.json')
            if metrics!=reference['metrics'] or rows!=reference['records']:raise RuntimeError('Warm packed initializer replay changed')
            calibration=trainer.calibrate(plan['calibration_indices']);base.write(output/'calibration.json',calibration)
            longest=sorted(data.training,key=lambda i:data.lengths[i],reverse=True)[:16];probe=trainer.master_update_probe(longest)
            if not probe['passed']:raise RuntimeError('Feature pilot update/fixed-state proof failed')
            result=dict(status='completed',passed=True,discarded=True,systems_only=True,raw_replay_exact=True,warm_replay_exact=True,
                raw=raw_metrics,warm=metrics,calibration=calibration,probe=probe,inheritance=inheritance)
        elif phase=='train':
            diagnostic=DiagnosticCheckpoint(output,trainer,raw,pilot['initial'])
            def event(kind,**values):
                if kind=='development':diagnostic.observe(values['epoch'],{k:values[k] for k in pilot['initial']})
                base.event(root/'progress.jsonl',kind,**values)
            result=trainer.fit(output,protected_state=raw,protected_metrics=pilot['initial'],protected_records=pilot['initial_records'],event=event)
            base.write(output/'feature_objective.json',dict(target_receipt=plan['target_receipt'],calibration=previous['calibration'],
                optimizer='StreamingCPUAdamW',selection_unchanged=True))
        else:
            output.mkdir();before=training.prior.fixed_state(stack,E,rotary);state=source.native._load(root/'train/selected.pt')
            stack.load_state_dict(state['stack'],strict=True);norm.load_state_dict(state['norm'],strict=True);del state
            metrics,rows=trainer.packed_development();reference=read(root/'train'/f"epoch_{trained['selected_epoch']:02d}_development.json")
            fixed=before==training.prior.fixed_state(stack,E,rotary);passed=fixed and metrics==reference['metrics'] and rows==reference['records']
            result=dict(status='completed',passed=passed,exact_packed_replay=passed,metrics=metrics,records=rows,fixed_state_unchanged=fixed,
                selected_checkpoint_sha256=base.sha(root/'train/selected.pt'),quality=trained['quality'],confirmed=False)
            if not passed:raise RuntimeError('Fresh packed selected replay changed')
        if not (output/'result.json').exists():base.write(output/'result.json',result)
        guard();base.write(output/'manifest.json',dict(status='completed',fingerprint=plan['fingerprint'],phase=phase,
            lease_id=lease['id'],allocation=lease['allocation'],allocator=allocator,elapsed_seconds=time.monotonic()-started,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),artifacts={str(p.relative_to(output)):base.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
        return dict(status='completed',phase=phase,selected=result.get('selected',result.get('metrics')))
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        base.write(output/'failure.json',dict(status='failed',error=repr(error),error_type=type(error).__name__,lease_id=None if lease is None else lease['id']))
        raise
    finally:
        if data is not None:data.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare',*PHASES))
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--diagnostic-root',type=Path);args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.diagnostic_root) if args.phase=='prepare' else run(args.root,args.phase),allow_nan=False),flush=True)
