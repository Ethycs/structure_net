"""Reservation-backed independent all36 distillation on the spare RTX2060."""
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

import torch

from . import jspace_3b_joint_init as initializer
from . import jspace_3b_joint_train as training
from . import jspace_3b_partial_generation as generation
from .jspace_full_lease import wait_ready

broad=training.broad
base,old,read,cache=broad.base,broad.old,broad.read,broad.cache
SCHEMA='jspace.3b-all36-joint.v1'
PROTOCOL=base.ROOT/'docs/07 - Status Reports/jspace/2026-11-15_qwen3b-all36-joint.md'
CHECKPOINTS=(0,72,144,216,288)
PHASES=('pilot','train','verify')
DEPENDENCIES={'pilot':(), 'train':('pilot',), 'verify':('train',)}
ALLOCATION=5632
LIMIT=5376


def choose(history):
    if not history:raise ValueError('Retained checkpoint history required')
    if any(type(c.get('updates'))is not int or c['updates']<0 or
           type(c.get('metrics',{}).get('correct'))is not int or
           not 0<=c['metrics']['correct']<=c['metrics'].get('count',-1) or
           type(c.get('eligibility',{}).get('passed'))is not bool for c in history):
        raise ValueError('Finite valid checkpoint counts, update indices and eligibility required')
    key=lambda c:(-c['metrics']['correct'],c['updates'])
    best=min(history,key=key);eligible=[c for c in history if c['eligibility']['passed']]
    return dict(candidate=min(eligible,key=key) if eligible else best,diagnostic_best=best,
                development_passed=bool(eligible),confirmed=False)


def prepare(root,initializer_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only preparation required')
    root,initializer_root=Path(root).resolve(),Path(initializer_root).resolve()
    if root.exists():raise FileExistsError(root)
    init=initializer.checked(initializer_root);ip=initializer.verify(initializer_root)
    cp=cache.verify(init['cache_root']);native=cache.checked(init['cache_root'],'native',cp)
    if not init['native_initialization'].get('native_initialization'):
        raise ValueError('Fresh native projection, not trained suffix inheritance, is required')
    inputs=dict(ip['input_sha256'])
    for path in (initializer_root/'plan.json',initializer_root/'result.json',initializer_root/'manifest.json',
                 Path(init['cache_root'])/'plan.json',Path(init['cache_root'])/'items.json'):
        old.pin(inputs,path)
    for name,digest in read(initializer_root/'manifest.json')['artifacts'].items():
        inputs[str((initializer_root/name).resolve())]=digest
    native_dir=Path(init['cache_root'])/'native'
    for name,digest in read(native_dir/'manifest.json')['artifacts'].items():
        inputs[str((native_dir/name).resolve())]=digest
    files=broad.source.inherited.local_import_closure([Path(__file__),PROTOCOL,
        base.ROOT/'tests/structure_net/test_jspace_3b_joint.py',
        base.ROOT/'tests/structure_net/test_jspace_3b_joint_train.py'])
    p=dict(schema=SCHEMA,initializer_root=str(initializer_root),initializer=init['initializer'],
        initializer_sha256=init['initializer_sha256'],model=init['model'],fit_map=init['fit_map'],
        boundary_path=init['boundary_path'],cache_root=init['cache_root'],items_path=init['items_path'],
        teacher_logits_path=init['teacher_logits_path'],label_ids=cp['label_ids'],order=cp['training_order'],
        native_records=native['records'],native_correct=native['correct'],config=asdict(training.CONFIG),
        geometry_readout_fingerprint=init['geometry_readout_fingerprint'],inventory=init['inventory'],
        phases=list(PHASES),checkpoints=list(CHECKPOINTS),cut=0,alpha=0.,
        objective='original-native content4 KL+CE only; no compressed-parent feature auxiliary',
        executor='unpacked BF16 Torch; CPU BF16 gradient bank and FP32 Adam',
        allocation_mib=ALLOCATION,allocator_mib=LIMIT,confirmed=False,
        input_sha256=inputs,input_identity={name:old.file_identity(name) for name in inputs},
        source_sha256={str(f):base.sha(f) for f in sorted(files)})
    items=read(p['items_path'])['items']
    cases=[c for c in items if c['split']=='development']
    cache.sanity.validate_records(p['native_records'],cases)
    if (p['native_correct']!=195 or len(cases)!=256 or len(p['order'])!=4608 or
            sorted(p['order'])!=[i for i,c in enumerate(items) if c['split']=='train']):
        raise ValueError('Original 195/256 reference and all4608 unique training cases required')
    p['fingerprint']=base.digest(p);root.mkdir(parents=True)
    for name,digest in p['source_sha256'].items():
        out=root/'producer'/Path(name).relative_to(base.ROOT);out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,out)
        if base.sha(out)!=digest:raise ValueError('Archived producer changed')
    base.write(root/'plan.json',p)
    return dict(status='prepared',fingerprint=p['fingerprint'])


def verify(root):
    root=Path(root);p=read(root/'plan.json');broad.source.check_plan(p)
    if (p.get('schema')!=SCHEMA or p.get('config')!=asdict(training.CONFIG) or p.get('cut')!=0
            or p.get('phases')!=list(PHASES) or set(p.get('fit_map',{}))!={str(i) for i in range(36)}
            or p.get('alpha')!=0. or p.get('checkpoints')!=list(CHECKPOINTS)
            or p.get('allocation_mib')!=ALLOCATION or p.get('allocator_mib')!=LIMIT
            or p.get('native_correct')!=195 or p.get('confirmed')is not False):
        raise ValueError('All36 prospective experiment changed')
    for path,identity in p['input_identity'].items():
        if old.file_identity(path)!=identity:raise ValueError('Prepared input changed: '+path)
    for path,digest in p['source_sha256'].items():
        if base.sha(path)!=digest or base.sha(root/'producer'/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError('Frozen producer changed')
    return p


def validate_metrics(metrics,records,p):
    cases=[c for c in read(p['items_path'])['items'] if c['split']=='development']
    cache.sanity.validate_records(records,cases)
    correct=sum(r['prediction']==r['answer'] for r in records)
    families={f:dict(count=sum(r['family']==f for r in records),
                    correct=sum(r['family']==f and r['prediction']==r['answer'] for r in records))
              for f in sorted({r['family'] for r in records})}
    if (len(cases)!=256 or metrics.get('count')!=256 or metrics.get('correct')!=correct
            or metrics.get('accuracy')!=correct/256 or metrics.get('by_family')!=families
            or not all(math.isfinite(metrics[k]) for k in ('loss','kl','ce'))):
        raise ValueError('Retained development metrics differ from paired records')
    return broad.eligibility(records,p['native_records'],cases)


def validate_candidate(root,c,p,manifest):
    out=Path(root)/'train';step=c['updates'];report=out/f'update_{step:03d}_development.json'
    if (Path(c['report']).resolve()!=report.resolve() or c['report_sha256']!=base.sha(report)
            or manifest['artifacts'].get(report.name)!=c['report_sha256']):
        raise ValueError('Retained candidate report linkage changed')
    value=read(report)
    if (c['metrics']!=value['metrics'] or
            c['eligibility']!=validate_metrics(value['metrics'],value['records'],p)):
        raise ValueError('Candidate quality or eligibility changed')
    checkpoint=Path(p['initializer']) if step==0 else out/f'update_{step:03d}.pt'
    expected=p['initializer_sha256'] if step==0 else manifest['artifacts'].get(checkpoint.name)
    if Path(c['checkpoint']).resolve()!=checkpoint.resolve() or c['checkpoint_sha256']!=expected:
        raise ValueError('Actual checkpoint linkage changed')
    return value


def checked(root,phase,p=None):
    root=Path(root);p=verify(root) if p is None else p
    if phase not in PHASES:raise ValueError('Unknown phase')
    out=root/phase;m=read(out/'manifest.json');r=read(out/'result.json')
    if (phase not in p['phases'] or m.get('status')!='completed' or m.get('phase')!=phase
            or m.get('fingerprint')!=p['fingerprint'] or r.get('status')!='completed'
            or m.get('artifacts',{}).get('result.json')!=base.sha(out/'result.json')
            or set(m.get('artifact_identity',{}))!=set(m.get('artifacts',{}))):
        raise ValueError('Completed matching all36 phase required')
    for name,digest in m['artifacts'].items():
        f=out/name
        if not f.resolve().is_relative_to(out.resolve()) or old.file_identity(f)!=m['artifact_identity'][name]:
            raise ValueError('Immutable phase artifact changed')
        if f.suffix!='.pt' and base.sha(f)!=digest:raise ValueError('Phase metadata changed')
    prior={dep:checked(root,dep,p) for dep in DEPENDENCIES[phase]}
    if m.get('dependencies')!={dep:base.sha(root/dep/'manifest.json') for dep in prior}:
        raise ValueError('Pilot/train/verification evidence lineage changed')
    a=m.get('allocator',{});uuid=a.get('uuid','')
    if (not uuid.startswith('GPU-8cf440df') or m.get('allocation')!={uuid:ALLOCATION}
            or a.get('allocator_limit_bytes')!=LIMIT*1024**2
            or not 0<m.get('peak_reserved_bytes',0)<=LIMIT*1024**2):
        raise ValueError('Dedicated bounded RTX2060 reservation required')
    if phase=='pilot' and (not r.get('passed') or not r.get('discarded') or
            r.get('gradients_present_by_block')!=[True]*36 or r.get('masters_changed_by_block')!=[True]*36
            or not r.get('norm_master_changed') or not r.get('fixed_state_unchanged')):
        raise ValueError('Discarded all36 systems pilot required')
    if phase=='pilot':
        validate_metrics(r['initial'],r['initial_records'],p)
        update=r.get('update',{})
        if (update.get('cases')!=16 or update.get('update')!=1 or
                update.get('gradients_present_by_block')!=[True]*36 or
                update.get('alpha')!=0. or update.get('feature_loss')!=0. or
                update.get('normalization_energy') is not None or
                not all(math.isfinite(update[k]) for k in ('loss','endpoint_loss','kl','ce'))):
            raise ValueError('Pilot must exercise the declared endpoint-only objective')
    if phase=='train':
        history=r.get('history',[]);updates=r.get('training',[])
        if (len(updates)!=288 or [u.get('update') for u in updates]!=list(range(1,289))
                or [i for u in updates for i in u['indices']]!=p['order']
                or any(u.get('cases')!=16 or u.get('gradients_present_by_block')!=[True]*36
                       or u.get('learning_rate')!=5e-6 or u.get('alpha')!=0.
                       or u.get('feature_loss')!=0. or u.get('normalization_energy') is not None
                       or not all(math.isfinite(u[k]) for k in ('loss','endpoint_loss','kl','ce')) for u in updates)
                or [c['updates'] for c in history]!=list(CHECKPOINTS)
                or any(r.get(k)!=v for k,v in choose(history).items()) or not r.get('fixed_state_unchanged')
                or r.get('config')!=asdict(training.CONFIG) or r.get('fixed_state')!=prior['pilot']['fixed_state']
                or r.get('initial_masters')!=prior['pilot']['initial_masters']):
            raise ValueError('Fixed full-pass budget or selection changed')
        for c in history:validate_candidate(root,c,p,m)
        initial=read(history[0]['report'])
        if initial!={'metrics':prior['pilot']['initial'],'records':prior['pilot']['initial_records']}:
            raise ValueError('Scientific initializer differs from the discarded pilot')
    if phase=='verify':
        train=prior['train'];selected={c['checkpoint_sha256']:c for c in (train['candidate'],train['diagnostic_best'])}
        if not r.get('passed') or r.get('confirmed') is not False or set(r.get('checks',{}))!=set(selected):
            raise ValueError('Fresh selected and diagnostic replay proofs required')
        tm=read(root/'train/manifest.json')
        for digest,c in selected.items():
            reference=validate_candidate(root,c,p,tm);proof=r['checks'][digest]
            if (not proof.get('passed') or proof.get('checkpoint_sha256')!=digest or
                    not proof.get('strict_state_load') or not proof.get('fixed_state_unchanged') or
                    proof.get('fixed_state')!=train['fixed_state'] or
                    proof.get('metrics')!=reference['metrics'] or proof.get('records')!=reference['records']):
                raise ValueError('Fresh checkpoint, fixed-state or exact replay evidence changed')
    return r


def dataset(p):
    return cache.RecoveryDataset(read(p['items_path'])['items'],p['boundary_path'],p['teacher_logits_path'],
                                 arm='native_content',label_ids=p['label_ids'])


def geometry(stack,E):
    return old.fingerprint([('stack.'+n,v) for n,v in stack.named_buffers()]+[('readout',E)])


def build(p,checkpoint,device,guard):
    guard()
    state=cache.builder.load_previous(checkpoint)
    stack,norm,E,_,proof=cache.builder.build_stack(p['model']['path'],
        {int(k):v for k,v in p['fit_map'].items()},0,state,device='cpu',trainable=True,checkpoint_blocks=True)
    initializer.canonicalize(stack,norm)
    for name,module in (('stack',stack),('norm',norm)):
        actual=module.state_dict()
        if not old.state_equal(actual,state[name]) or any(isinstance(v,torch.Tensor)
                and v.dtype!=actual[k].dtype for k,v in state[name].items()):
            raise ValueError('Strict BF16 checkpoint reconstruction changed a tensor or dtype')
    del state;gc.collect()
    if geometry(stack,E)!=p['geometry_readout_fingerprint']:
        raise ValueError('Frozen native geometry/readout changed')
    guard();rotary=generation.training_rotary(cache.builder.native._config(p['model']['path'],None),device,guard=guard)
    stack,norm,E=stack.to(device),norm.to(device),E.to(device)
    return stack,norm,E,rotary,proof


def master_records(trainer):
    return dict(blocks=[old.master_fingerprint(b,trainer.optimizer) for b in trainer.stack.blocks],
                norm=old.master_fingerprint(trainer.norm,trainer.optimizer))


def run(root,phase):
    root=Path(root).resolve();p=verify(root)
    if phase not in p['phases']:raise ValueError('Unknown phase')
    out=root/phase
    if out.exists():raise FileExistsError(out)
    prior=checked(root,'pilot',p) if phase=='train' else checked(root,'train',p) if phase=='verify' else None
    dependencies={dep:base.sha(root/dep/'manifest.json') for dep in DEPENDENCIES[phase]}
    lease=settings=None;data=trainer=None;cuda=False;started=time.monotonic()
    try:
        lease=wait_ready([0]);uuid=next(iter(lease['allocation']))
        if not uuid.startswith('GPU-8cf440df') or lease['allocation']!={uuid:ALLOCATION}:
            raise ValueError('Only the separately reserved 6GB2060 is admitted')
        torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(LIMIT*1024**2/props.total_memory,0)
        torch.cuda.reset_peak_memory_stats();cuda=True;torch.set_num_threads(2)
        torch.manual_seed(training.CONFIG.seed)
        settings=dict(uuid=uuid,allocator_limit_bytes=LIMIT*1024**2,total_memory=props.total_memory)
        def guard():
            current=wait_ready([0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:
                raise RuntimeError('Separate all36 reservation changed')
            if torch.cuda.max_memory_reserved()>LIMIT*1024**2:raise MemoryError('2060 allocator cap exceeded')
        out.mkdir();data=dataset(p)
        with cache.sanity.execution.arithmetic_policy():
            if phase=='verify':
                checks={}
                for c in (prior['candidate'],prior['diagnostic_best']):
                    digest=c['checkpoint_sha256']
                    if digest in checks:continue
                    if base.sha(c['checkpoint'])!=digest:raise ValueError('Selected checkpoint changed')
                    stack,norm,E,rotary,_=build(p,c['checkpoint'],'cuda:0',guard)
                    fixed=old.fixed_state(stack,E,rotary)
                    if fixed!=prior['fixed_state']:raise ValueError('Runtime fixed state changed')
                    metrics,records=training.score_unpacked(stack,norm,E,rotary,data,p['label_ids'],guard)
                    reference=read(c['report'])
                    if metrics!=reference['metrics'] or records!=reference['records']:
                        raise ValueError('Fresh all36 Torch replay differs')
                    if old.fixed_state(stack,E,rotary)!=fixed:raise ValueError('Fresh evaluation mutated fixed state')
                    checks[digest]=dict(passed=True,checkpoint_sha256=digest,strict_state_load=True,
                        fixed_state_unchanged=True,metrics=metrics,records=records,fixed_state=fixed)
                    del stack,norm,E,rotary;gc.collect();torch.cuda.empty_cache()
                result=dict(status='completed',passed=True,checks=checks,confirmed=False)
            else:
                if base.sha(p['initializer'])!=p['initializer_sha256']:raise ValueError('Initializer changed')
                stack,norm,E,rotary,_=build(p,p['initializer'],'cuda:0',guard)
                trainer=training.JointTrainer(stack,norm,E,rotary,data,p['label_ids'],guard=guard)
                fixed=old.fixed_state(stack,E,rotary);masters=master_records(trainer)
                stamp=time.monotonic();initial,initial_rows=trainer.development();dev_seconds=time.monotonic()-stamp
                if phase=='pilot':
                    indices=sorted(data.training,key=lambda i:(-data.lengths[i],i))[:16]
                    update=trainer.train_window(indices);after=master_records(trainer)
                    changed=[a!=b for a,b in zip(masters['blocks'],after['blocks'])]
                    unchanged=old.fixed_state(stack,E,rotary)==fixed
                    passed=all(changed) and masters['norm']!=after['norm'] and unchanged
                    if not passed:raise ValueError('All36 master-update or fixed-state pilot failed')
                    result=dict(status='completed',passed=True,discarded=True,systems_only=True,
                        initial=initial,initial_records=initial_rows,initial_masters=masters,fixed_state=fixed,
                        update=update,gradients_present_by_block=update['gradients_present_by_block'],
                        masters_changed_by_block=changed,norm_master_changed=True,fixed_state_unchanged=unchanged,
                        development_seconds=dev_seconds,memory=trainer.memory,host_admission=trainer.host_admission,
                        suggested_train_minutes=math.ceil((1.5*update['train_window_seconds']*288+6*dev_seconds+600)/60))
                else:
                    if (initial!=prior['initial'] or initial_rows!=prior['initial_records']
                            or masters!=prior['initial_masters'] or fixed!=prior['fixed_state']):
                        raise ValueError('Scientific initializer differs from discarded pilot')
                    history=[];updates=[]
                    def retain(step,metrics=None,records=None):
                        if metrics is None:metrics,records=trainer.development()
                        report=out/f'update_{step:03d}_development.json'
                        base.write(report,dict(metrics=metrics,records=records))
                        checkpoint=Path(p['initializer']) if step==0 else out/f'update_{step:03d}.pt'
                        if step:old.save(checkpoint,trainer.snapshot())
                        c=dict(updates=step,checkpoint=str(checkpoint),checkpoint_sha256=base.sha(checkpoint),
                            report=str(report),report_sha256=base.sha(report),metrics=metrics,
                            eligibility=broad.eligibility(records,p['native_records'],[data.items[i] for i in data.development]))
                        history.append(c);base.event(root/'progress.jsonl','development',candidate=c)
                    retain(0,initial,initial_rows)
                    for start in range(0,len(p['order']),16):
                        u=trainer.train_window(p['order'][start:start+16],learning_rate=5e-6)
                        updates.append(u);base.event(root/'progress.jsonl','update',**u)
                        if trainer.updates in CHECKPOINTS:retain(trainer.updates)
                    unchanged=old.fixed_state(stack,E,rotary)==fixed
                    if trainer.updates!=288 or not unchanged:raise ValueError('Full budget or fixed state changed')
                    result=dict(status='completed',training=updates,history=history,**choose(history),
                        initial_masters=masters,fixed_state=fixed,fixed_state_unchanged=True,
                        config=asdict(training.CONFIG),elapsed_seconds=time.monotonic()-started)
        guard();base.write(out/'result.json',result)
        artifacts={str(f.relative_to(out)):base.sha(f) for f in out.rglob('*') if f.is_file()}
        base.write(out/'manifest.json',dict(status='completed',phase=phase,fingerprint=p['fingerprint'],
            dependencies=dependencies,
            artifacts=artifacts,artifact_identity={n:old.file_identity(out/n) for n in artifacts},
            lease_id=lease['id'],allocation=lease['allocation'],allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started))
        checked(root,phase,p)
        return {k:result[k] for k in ('status','passed','development_passed','suggested_train_minutes') if k in result}
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        base.write(out/'failure.json',dict(status='failed',error=repr(error),lease=lease,allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None))
        raise
    finally:
        if trainer is not None:trainer.close()
        if data is not None:data.close()
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','pilot','train','verify'))
    parser.add_argument('--root',required=True);parser.add_argument('--initializer-root')
    args=parser.parse_args()
    if args.phase=='prepare':
        if not args.initializer_root:parser.error('prepare requires initializer-root')
        result=prepare(args.root,args.initializer_root)
    else:result=run(args.root,args.phase)
    print(json.dumps(result,indent=2),flush=True)
