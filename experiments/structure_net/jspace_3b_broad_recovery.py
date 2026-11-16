"""One matched, constant-LR recovery pass in two teacher/position arms.

All quarter-pass weights are retained. Development eligibility is explicit;
the best unqualified task checkpoint remains diagnostic, never a promotion.
No audit or final-confirmation contents are opened here.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict,replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from . import jspace_3b_broad_recovery_cache as cache
from . import jspace_3b_parent_feature as feature
from . import jspace_qwen_parallel_train as core
from .jspace_full_lease import wait_ready

source,base,old,read=cache.source,cache.base,cache.old,cache.read
SCHEMA='jspace.3b-broad-recovery.v1'
MODULE='experiments.structure_net.jspace_3b_broad_recovery'
CONFIG=replace(source.CONFIG,epochs=1,learning_rate=5e-6)
CHECKPOINT_UPDATES=(0,72,144,216,288)
POLICY='eligible_highest_correct_then_fewest_updates; cross_arm_exact_tie_parent_legacy; no_CE_KL_veto'
RESET='Weights-only BF16 warm start; fresh FP32 masters, zero moments/counters and identical RNG/order in both arms'


def _model_device(device):
    """Triton selects its target from the current device, not tensor placement."""
    device=torch.device(device)
    return torch.cuda.device(device) if device.type=='cuda' else nullcontext()


def eligibility(records,native_records,cases):
    cache.sanity.validate_records(records,cases);cache.sanity.validate_records(native_records,cases)
    quality=cache.sanity.paired_quality(native_records,records)
    overall=quality['overall']['accuracy_drop']<=.05
    families={f:v['accuracy_drop']<=.10 for f,v in quality['by_family'].items()}
    return dict(passed=overall and all(families.values()),overall_passed=overall,family_passed=families,
        maximum_overall_drop=.05,maximum_family_drop=.10,quality=quality)


def choose(candidates):
    """Task accuracy, then earliest update; an exact arm tie favors the control."""
    if not candidates:raise ValueError('Checkpoint candidates required')
    for c in candidates:
        if (c.get('arm') not in cache.ARMS or type(c.get('updates'))is not int or c['updates']<0 or
                type(c.get('metrics',{}).get('correct'))is not int or
                not 0<=c['metrics']['correct']<=c['metrics'].get('count',-1) or
                type(c.get('eligibility',{}).get('passed'))is not bool):
            raise ValueError('Invalid candidate counts/update/arm/gate')
    key=lambda c:(-c['metrics']['correct'],c['updates'],cache.ARMS.index(c['arm']))
    diagnostic=min(candidates,key=key)
    valid=[c for c in candidates if c['eligibility']['passed']]
    winner=min(valid,key=key) if valid else diagnostic
    return dict(candidate=winner,diagnostic_best=diagnostic,development_passed=bool(valid),policy=POLICY)


def global_candidates(candidates):
    """Identical retained weight files count once; prefer their earlier origin."""
    result={}
    for c in sorted(candidates,key=lambda c:(c['updates'],cache.ARMS.index(c['arm']))):
        result.setdefault(c['checkpoint_sha256'],c)
    return list(result.values())


def matched_initializers(pilots):
    a,b=(pilots[arm] for arm in cache.ARMS)
    if (a['initial_records']!=b['initial_records'] or a['checkpoint_sha256']!=b['checkpoint_sha256'] or
            a['checkpoint_sha256']!=cache.SEED_SHA or a['alpha']!=b['alpha'] or
            a['normalization_energy']!=b['normalization_energy'] or
            any(a['probe'][key]!=b['probe'][key] for key in
                ('fp32_master_before','norm_fp32_master_before','fixed_state'))):
        raise ValueError('Matched arms did not start from identical weights, masters, fixed state and targets')
    return True


@torch.inference_mode()
def score_packed(stack,norm,E,rotary,dataset,label_ids,guard=lambda:None):
    """Singleton choice-row accuracy, plus separately computed arm loss metrics."""
    if not stack.packed:raise ValueError('Packed evaluation required')
    rows=[];totals=dict(loss=0.,kl=0.,ce=0.)
    for index in dataset.development:
        guard()
        # Allocator setup and the resident optimizer also touch the SUPER.
        # Restore the model GPU after the guard, before any fused invocation.
        with _model_device(E.device):
            batch=dataset.batch([index],device=E.device);h=batch['hidden']
            output=stack(h,attention_mask=batch['attention_mask'],position_ids=batch['position_ids'],
                cache_position=batch['position_ids'][0],position_embeddings=rotary(h,batch['position_ids']),use_cache=False)[0]
            selected=output[torch.arange(1,device=E.device)[:,None],batch['loss_positions']]
            logits=F.linear(norm(selected),E).float()
            loss,kl,ce=core.output_objective(logits,batch['teacher'],batch['targets'],CONFIG)
            if not all(torch.isfinite(v) for v in (loss,kl,ce)):raise ValueError('Nonfinite development objective')
            for k,v in zip(totals,(loss,kl,ce)):totals[k]+=float(v)
            rows.append(cache.sanity._record(dataset.items[index],norm(output[:,-1]),E,label_ids))
    count=len(rows)
    if not count:raise ValueError('No development records')
    correct=sum(r['answer']==r['prediction'] for r in rows)
    return dict(**{k:v/count for k,v in totals.items()},count=count,correct=correct,accuracy=correct/count,
        by_family={f:dict(count=sum(r['family']==f for r in rows),
            correct=sum(r['family']==f and r['prediction']==r['answer'] for r in rows))
            for f in sorted({r['family'] for r in rows})},
        scoring='singleton BF16 choice-row; loss metrics use separate full-vocabulary four-position GEMM'),rows


class BroadTrainer(feature.ResidentFeatureTrainer):
    """Only scoring/fit orchestration changes; objective/update math is inherited."""
    @torch.no_grad()
    def packed_development(self):
        self._guard();self.optimizer.zero_grad(set_to_none=True)
        stack=cache.builder.packed_clone(self.stack,device='cpu')
        norm=core.cpu_module_clone(self.norm).eval().requires_grad_(False)
        try:
            return score_packed(stack.to(self.device),norm.to(self.device),self.readout,self.rotary,
                self.dataset,self.label_ids.tolist(),self._guard)
        finally:
            del stack,norm;gc.collect()


def fit_trial(trainer,path,*,arm,order,native_records,seed_checkpoint,pilot_reference,event=lambda *_a,**_kw:None,
              checkpoints=CHECKPOINT_UPDATES):
    """Exactly one pass; checkpoints and validity failures never alter its order."""
    if trainer.updates or trainer.finished or list(order)!=list(dict.fromkeys(order)) or sorted(order)!=sorted(trainer.dataset.training):
        raise ValueError('Fresh optimizer and complete unique common TRAIN order required')
    effective=trainer.config.microbatch*trainer.config.accumulation
    total=math.ceil(len(order)/effective)
    if list(checkpoints)!=sorted(set(checkpoints)) or checkpoints[0]!=0 or checkpoints[-1]!=total:
        raise ValueError('Checkpoint schedule must include initializer and the complete pass')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    before=source.old.fixed_state(trainer.stack,trainer.readout,trainer.rotary)
    rows_cases=[trainer.dataset.items[i] for i in trainer.dataset.development]
    history=[];updates=[];started=time.monotonic()
    def record(step):
        metrics,rows=trainer.packed_development()
        if step==0 and (metrics!=pilot_reference['initial'] or rows!=pilot_reference['initial_records']):
            raise ValueError('Fresh scientific initializer differs from discarded pilot')
        report=path/f'update_{step:03d}_development.json'
        base.write(report,dict(metrics=metrics,records=rows))
        checkpoint=Path(seed_checkpoint) if step==0 else path/f'update_{step:03d}.pt'
        if step:old.save(checkpoint,trainer.snapshot())
        candidate=dict(arm=arm,updates=step,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=base.sha(checkpoint),
            report=str(report.resolve()),report_sha256=base.sha(report),metrics=metrics,
            eligibility=eligibility(rows,native_records,rows_cases),protected_source=step==0)
        history.append(candidate);event('development',candidate=candidate)
    record(0)
    for start in range(0,len(order),effective):
        indices=list(order[start:start+effective]);value=trainer.train_window(indices,learning_rate=trainer.config.learning_rate)
        value=dict(value,indices=indices);updates.append(value);event('update',**value)
        if trainer.updates in checkpoints:record(trainer.updates)
    if trainer.updates!=total or source.old.fixed_state(trainer.stack,trainer.readout,trainer.rotary)!=before:
        raise ValueError('Wrong update count or fixed state changed')
    trainer.finished=True
    result=dict(status='completed',arm=arm,config=asdict(trainer.config),updates=total,training=updates,history=history,
        **choose(history),fixed_state_unchanged=True,fixed_state=before,alpha=trainer.alpha,
        normalization_energy=trainer.feature_targets.energy,optimizer_restart=RESET,order_digest=base.digest(list(order)),
        elapsed_seconds=time.monotonic()-started,confirmed=False)
    base.write(path/'candidates.json',dict(history=history,**choose(history)))
    return result


def prepare(root,cache_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation requires CUDA hidden')
    root,cache_root=Path(root).resolve(),Path(cache_root).resolve()
    if root.exists():raise FileExistsError(root)
    cp=cache.verify(cache_root);native=cache.checked(cache_root,'native',cp);parent=cache.checked(cache_root,'parent',cp)
    items=read(cache_root/'items.json')['items'];dev=[c for c in items if c['split']=='development']
    if Counter(c['family'] for c in dev)!=dict.fromkeys(('arc','boolq','hellaswag','winogrande'),64):
        raise ValueError('Fresh DEV must have64 cases in each family')
    targets,receipt=feature.load_targets(cache_root/'parent',items,cp['parent_checkpoint_sha256'])
    if len(targets.lookup)!=4608 or receipt['shape']!=[4608,1024]:raise ValueError('Shared full TRAIN feature bank required')
    inputs=dict(cp['input_sha256'])
    for path in (cache_root/'plan.json',cache_root/'items.json'):
        old.pin(inputs,path)
    for phase in ('native','parent'):
        m=read(cache_root/phase/'manifest.json');old.pin(inputs,cache_root/phase/'manifest.json')
        for name,digest in m['artifacts'].items():old.pin(inputs,cache_root/phase/name,digest)
    files=source.inherited.local_import_closure([Path(__file__),cache.PROTOCOL,
        base.ROOT/'tests/structure_net/test_jspace_3b_broad_recovery.py'])
    plan=dict(schema=SCHEMA,cache_root=str(cache_root),cache_fingerprint=cp['fingerprint'],source_root=cp['source_root'],
        config=asdict(CONFIG),alpha=cache.ALPHA,feature_receipt=receipt,arms=list(cache.ARMS),cut=22,
        checkpoint=cp['checkpoint'],checkpoint_sha256=cache.SEED_SHA,model=cp['model'],fits=cp['fits'],
        label_ids=cp['label_ids'],order=cp['training_order'],checkpoints=list(CHECKPOINT_UPDATES),policy=POLICY,
        optimizer_restart=RESET,learning_rate_schedule='constant',confirmed=False,
        native_records=native['records'],native_correct=native['correct'],
        allocation_mib=11264,allocator_mib=11008,resident_budget_mib=6144,
        input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        source_sha256={str(p):base.sha(p) for p in sorted(files)})
    plan['fingerprint']=base.digest(plan);root.mkdir(parents=True)
    for name,digest in plan['source_sha256'].items():
        target=root/'producer'/Path(name).relative_to(base.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        if base.sha(target)!=digest:raise ValueError('Archive changed')
    base.write(root/'plan.json',plan);return dict(status='prepared',fingerprint=plan['fingerprint'])


def verify(root):
    root=Path(root);p=read(root/'plan.json');source.check_plan(p)
    if (p.get('schema')!=SCHEMA or p.get('config')!=asdict(CONFIG) or p.get('alpha')!=cache.ALPHA or
            p.get('checkpoint_sha256')!=cache.SEED_SHA or p.get('policy')!=POLICY or
            p.get('checkpoints')!=list(CHECKPOINT_UPDATES) or p.get('arms')!=list(cache.ARMS) or
            p.get('optimizer_restart')!=RESET or p.get('learning_rate_schedule')!='constant'):
        raise ValueError('Prospective broad-recovery recipe changed')
    for name,identity in p['input_identity'].items():
        if old.file_identity(name)!=identity:raise ValueError('Pinned recovery input changed: '+name)
    for name,digest in p['source_sha256'].items():
        if base.sha(name)!=digest or base.sha(root/'producer'/Path(name).relative_to(base.ROOT))!=digest:
            raise ValueError('Recovery producer changed')
    return p


def _items(plan):return read(Path(plan['cache_root'])/'items.json')['items']


def dataset_for(plan,arm):
    c=Path(plan['cache_root'])
    return cache.RecoveryDataset(_items(plan),c/'native/boundary_22.npy',
        c/('parent/packed_teacher_logits.npy' if arm=='parent_legacy' else 'native/teacher_logits.npy'),
        arm=arm,label_ids=plan['label_ids'])


def validate_history(result,plan):
    history=result.get('history',[]);updates=result.get('training',[])
    if (result.get('config')!=plan['config'] or result.get('alpha')!=cache.ALPHA or
            result.get('normalization_energy')!=plan['feature_receipt']['normalization_energy'] or
            result.get('optimizer_restart')!=RESET or result.get('order_digest')!=base.digest(plan['order']) or
            [c.get('updates') for c in history]!=plan['checkpoints'] or len(updates)!=288 or
            result.get('updates')!=288 or [u.get('update') for u in updates]!=list(range(1,289)) or
            [i for u in updates for i in u.get('indices',[])]!=plan['order'] or
            any(u.get('cases')!=16 or u.get('learning_rate')!=5e-6 or
                u.get('alpha')!=cache.ALPHA or u.get('normalization_energy')!=plan['feature_receipt']['normalization_energy'] or
                u.get('gradients_present_by_block')!=[True]*14 or
                not all(math.isfinite(u[k]) for k in ('loss','endpoint_loss','kl','ce','feature_loss')) for u in updates)):
        raise ValueError('Complete matched288-update budget/order/objective required')
    dev=[c for c in _items(plan) if c['split']=='development']
    for c in history:
        if (c.get('arm')!=result.get('arm') or c.get('protected_source')!=(c['updates']==0) or
                c['updates']==0 and (c.get('checkpoint_sha256')!=cache.SEED_SHA or
                    Path(c['checkpoint']).resolve()!=Path(plan['checkpoint']).resolve())):
            raise ValueError('Candidate arm or protected source identity changed')
        if base.sha(c['checkpoint'])!=c['checkpoint_sha256'] or base.sha(c['report'])!=c['report_sha256']:
            raise ValueError('Retained checkpoint/report changed')
        report=read(c['report']);cache.sanity.validate_records(report['records'],dev)
        if (report['metrics']!=c['metrics'] or c['metrics']['count']!=len(dev) or
                c['metrics']['correct']!=sum(r['prediction']==r['answer'] for r in report['records']) or
                c['metrics']['accuracy']!=c['metrics']['correct']/len(dev) or
                not all(math.isfinite(c['metrics'][k]) for k in ('loss','kl','ce')) or
                c['eligibility']!=eligibility(report['records'],plan['native_records'],dev)):
            raise ValueError('Candidate metrics or eligibility disagree with actual records')
    expected=choose(history)
    if any(result.get(k)!=v for k,v in expected.items()) or not result.get('fixed_state_unchanged'):
        raise ValueError('Prospective selection/fixed-state proof changed')
    return expected


def checked(root,arm,phase,plan=None):
    root=Path(root);p=verify(root) if plan is None else plan
    if arm not in cache.ARMS or phase not in ('pilot','train','verify'):raise ValueError('Unknown arm/phase')
    out=root/arm/phase;m=read(out/'manifest.json');r=read(out/'result.json')
    if (m.get('status')!='completed' or m.get('arm')!=arm or m.get('phase')!=phase or
            m.get('fingerprint')!=p['fingerprint'] or r.get('status')!='completed' or r.get('arm')!=arm or
            m.get('artifacts',{}).get('result.json')!=base.sha(out/'result.json')):
        raise ValueError('Incomplete/mismatched recovery receipt')
    for name,digest in m['artifacts'].items():
        if not (out/name).resolve().is_relative_to(out.resolve()) or base.sha(out/name)!=digest:raise ValueError('Recovery artifact changed')
    if phase in ('pilot','train'):
        for role,prefix,mib in (('front','GPU-0370fdb0',11264),('tail','GPU-7b1db3cb',7168)):
            a=m.get('allocator',{}).get(role,{});uuid=a.get('uuid','')
            if (not uuid.startswith(prefix) or m.get('allocation',{}).get(uuid)!=mib or
                    a.get('allocator_limit_bytes')!=(mib-256)*1024**2 or
                    not 0<=m.get('peak_reserved_bytes',{}).get(role,-1)<=a['allocator_limit_bytes']):
                raise ValueError('Missing bounded two-card reservation')
    else:
        a=m.get('allocator',{});uuid=a.get('physical_uuid','')
        if (not uuid.startswith('GPU-0370fdb0') or m.get('allocation')!={uuid:11264} or
                a.get('allocator_limit_bytes')!=11008*1024**2 or not 0<m.get('peak_reserved_bytes',0)<=11008*1024**2):
            raise ValueError('Missing bounded verification reservation')
    if phase=='pilot':
        if (not r.get('passed') or not r.get('discarded') or not r.get('probe',{}).get('passed') or
                r['probe'].get('layer_indices')!=list(range(22,36)) or not r['probe'].get('fixed_state_unchanged') or
                not r.get('geometry_readout_unchanged') or r.get('alpha')!=cache.ALPHA or
                r.get('normalization_energy')!=p['feature_receipt']['normalization_energy'] or
                r.get('checkpoint_sha256')!=cache.SEED_SHA):raise ValueError('Discarded all-block systems probe required')
    elif phase=='train':
        pilot=checked(root,arm,'pilot',p);validate_history(r,p)
        if (r['history'][0]['metrics']!=pilot['initial'] or read(r['history'][0]['report'])['records']!=pilot['initial_records'] or
                r.get('fixed_state')!=pilot['probe'].get('fixed_state')):
            raise ValueError('Training initializer differs from pilot')
    else:
        trained=checked(root,arm,'train',p)
        candidates={c['checkpoint_sha256']:c for c in (trained['candidate'],trained['diagnostic_best'])}
        if not r.get('passed') or set(r.get('checks',{}))!=set(candidates):raise ValueError('Fresh selection proofs incomplete')
        for digest,c in candidates.items():
            check=r['checks'][digest];reference=read(c['report'])
            if (not check.get('strict_state_load') or not check.get('fixed_state_unchanged') or
                    check.get('fixed_state')!=trained.get('fixed_state') or
                    check.get('records')!=reference['records'] or check.get('metrics')!=reference['metrics'] or
                    check.get('checkpoint_sha256')!=digest):raise ValueError('Fresh selected replay changed')
    return r


def _build(plan,checkpoint,device,trainable):
    return cache.builder.build_stack(plan['model']['path'],{int(k):v for k,v in plan['fits'].items()},22,checkpoint,
        device=device,trainable=trainable,checkpoint_blocks=trainable)


def run(root,arm,phase):
    root=Path(root).resolve();p=verify(root)
    if arm not in cache.ARMS or phase not in ('pilot','train','verify'):raise ValueError('Unknown arm/phase')
    prior=checked(root,arm,'pilot',p) if phase=='train' else checked(root,arm,'train',p) if phase=='verify' else None
    out=root/arm/phase
    if out.exists():raise FileExistsError(out)
    dual=phase!='verify';lease=settings=None;cuda=False;data=None;started=time.monotonic()
    try:
        lease=wait_ready([0,1] if dual else [0])
        settings=source.replay.allocator_settings(lease) if dual else source.inherited.cap_allocator(lease,p)
        front=settings['front']['device'] if dual else 0;device=f'cuda:{front}';cuda=True
        for d in ([v['device'] for v in settings.values()] if dual else [0]):torch.cuda.reset_peak_memory_stats(d)
        torch.cuda.set_device(front)
        torch.set_num_threads(4);torch.manual_seed(CONFIG.seed)
        def guard():
            current=wait_ready([0,1] if dual else [0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:raise RuntimeError('Reservation changed')
            limits=list(settings.values()) if dual else [dict(device=0,**settings)]
            if any(torch.cuda.max_memory_reserved(v['device'])>v['allocator_limit_bytes'] for v in limits):raise MemoryError('Reservation cap exceeded')
        out.parent.mkdir(parents=True,exist_ok=True)
        def event(kind,**values):base.event(root/'progress.jsonl',kind,arm=arm,phase=phase,**values)
        data=dataset_for(p,arm)
        with cache.sanity.execution.arithmetic_policy():
            if dual:
                stack,norm,E,rotary,_=_build(p,p['checkpoint'],device,True)
                bank,receipt=feature.load_targets(Path(p['cache_root'])/'parent',_items(p),
                    read(Path(p['cache_root'])/'plan.json')['parent_checkpoint_sha256'])
                if receipt!=p['feature_receipt']:raise ValueError('Shared feature target changed')
                trainer=BroadTrainer(stack,norm,E,rotary,data,p['label_ids'],config=CONFIG,count=256,
                    native_correct=p['native_correct'],parent_correct=p['native_correct'],guard=guard,
                    targets=bank,alpha=p['alpha'],resident_device=f"cuda:{settings['tail']['device']}",
                    resident_budget_bytes=p['resident_budget_mib']*1024**2)
                if phase=='pilot':
                    out.mkdir();before=source.old.fixed_state(stack,E,rotary)
                    initial,rows=trainer.packed_development()
                    longest=sorted(data.training,key=lambda i:(-data.lengths[i],i))[:16]
                    probe=trainer.master_update_probe(longest)
                    result=dict(status='completed',arm=arm,passed=probe['passed'],discarded=True,systems_only=True,
                        initial=initial,initial_records=rows,probe=probe,checkpoint_sha256=p['checkpoint_sha256'],
                        geometry_readout_unchanged=source.old.fixed_state(stack,E,rotary)==before,
                        optimizer_restart=RESET,alpha=p['alpha'],normalization_energy=bank.energy)
                    if not result['passed'] or not result['geometry_readout_unchanged']:raise ValueError('Systems probe failed')
                else:
                    result=fit_trial(trainer,out,arm=arm,order=p['order'],native_records=p['native_records'],
                        seed_checkpoint=p['checkpoint'],pilot_reference=prior,event=event)
                base.write(out/'resident_optimizer.json',dict(memory=trainer.optimizer.memory_stats(),
                    transfer_accounting=source.replay.TRANSFER_ACCOUNTING))
            else:
                out.mkdir();checks={}
                for c in (prior['candidate'],prior['diagnostic_best']):
                    digest=c['checkpoint_sha256']
                    if digest in checks:continue
                    if base.sha(c['checkpoint'])!=digest:raise ValueError('Selected weights changed')
                    stack,norm,E,rotary,inheritance=_build(p,c['checkpoint'],'cpu',False)
                    before=source.old.fixed_state(stack,E,rotary)
                    if before!=prior['fixed_state']:raise ValueError('Selected geometry/readout differs from scientific initializer')
                    packed=cache.builder.packed_clone(stack,device='cpu')
                    fixed=source.old.fixed_state(stack,E,rotary)==before
                    del stack
                    packed,norm,E,rotary=packed.to(device),norm.to(device),E.to(device),rotary.to(device)
                    metrics,rows=score_packed(packed,norm,E,rotary,data,p['label_ids'],guard)
                    reference=read(c['report'])
                    if metrics!=reference['metrics'] or rows!=reference['records'] or not fixed:
                        raise ValueError('Fresh packed development replay differs')
                    checks[digest]=dict(checkpoint_sha256=digest,strict_state_load=True,fixed_state_unchanged=fixed,
                        fixed_state=before,inheritance=inheritance,records=rows,metrics=metrics)
                    del packed,norm,E,rotary;gc.collect();torch.cuda.empty_cache()
                result=dict(status='completed',arm=arm,passed=True,checks=checks,
                    development_passed=prior['development_passed'],confirmed=False)
        guard();base.write(out/'result.json',result)
        peaks={role:torch.cuda.max_memory_reserved(v['device']) for role,v in settings.items()} if dual else torch.cuda.max_memory_reserved()
        base.write(out/'manifest.json',dict(status='completed',arm=arm,phase=phase,fingerprint=p['fingerprint'],
            artifacts={str(f.relative_to(out)):base.sha(f) for f in out.rglob('*') if f.is_file()},
            lease_id=lease['id'],allocation=lease['allocation'],allocator=settings,peak_reserved_bytes=peaks,
            elapsed_seconds=time.monotonic()-started))
        return result
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True);base.write(out/'failure.json',dict(status='failed',arm=arm,phase=phase,
            error=f'{type(error).__name__}: {error}',fingerprint=p['fingerprint'],lease=lease,allocator=settings));raise
    finally:
        if data is not None:data.close()
        gc.collect()


def select(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Selection requires CUDA hidden')
    root=Path(root);p=verify(root);candidates=[];receipts={};pilots={}
    for arm in cache.ARMS:
        checked(root,arm,'verify',p);trained=checked(root,arm,'train',p)
        pilots[arm]=checked(root,arm,'pilot',p)
        candidates.extend((trained['candidate'],trained['diagnostic_best']))
        receipts[arm]=base.sha(root/arm/'verify/manifest.json')
    matched_initializers(pilots);candidates=global_candidates(candidates);decision=choose(candidates)
    result=dict(status='completed',fingerprint=p['fingerprint'],cache_root=p['cache_root'],**decision,verification_sha256=receipts,
        source_update_zero_deduplicated=True,distinct_candidate_count=len(candidates),matched_initializers=True,
        confirmed=False,selection_only='No audit outcomes consumed')
    base.write(root/'selection.json',result)
    base.write(root/'manifest.json',dict(status='completed',fingerprint=p['fingerprint'],confirmed=False,
        artifacts={'selection.json':base.sha(root/'selection.json')}))
    return result


def checked_selection(root):
    """Authenticate both completed arms before a separate consumer opens audit data."""
    root=Path(root);p=verify(root);value=read(root/'selection.json');manifest=read(root/'manifest.json')
    if (manifest.get('status')!='completed' or manifest.get('fingerprint')!=p['fingerprint'] or
            manifest.get('artifacts',{}).get('selection.json')!=base.sha(root/'selection.json') or
            value.get('status')!='completed' or value.get('fingerprint')!=p['fingerprint'] or
            value.get('cache_root')!=p['cache_root'] or value.get('confirmed')is not False):
        raise ValueError('Completed authenticated global selection required')
    candidates=[];pilots={}
    for arm in cache.ARMS:
        checked(root,arm,'verify',p);trained=checked(root,arm,'train',p)
        pilots[arm]=checked(root,arm,'pilot',p)
        if value.get('verification_sha256',{}).get(arm)!=base.sha(root/arm/'verify/manifest.json'):
            raise ValueError('Selected arm fresh verification changed')
        candidates.extend((trained['candidate'],trained['diagnostic_best']))
    matched_initializers(pilots);candidates=global_candidates(candidates)
    if (value.get('distinct_candidate_count')!=len(candidates) or value.get('matched_initializers')is not True or
            any(value.get(k)!=v for k,v in choose(candidates).items())):raise ValueError('Global candidate ranking changed')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','pilot','train','verify','select'))
    p.add_argument('--root',required=True);p.add_argument('--cache-root');p.add_argument('--arm',choices=cache.ARMS);a=p.parse_args()
    if a.phase=='prepare':
        if not a.cache_root:p.error('prepare requires cache-root')
        result=prepare(a.root,a.cache_root)
    elif a.phase=='select':result=select(a.root)
    else:
        if not a.arm:p.error('GPU phases require arm')
        result=run(a.root,a.arm,a.phase)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
