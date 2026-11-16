"""Append-only verification repair for device-constructed nonpersistent RoPE.

Training built RoPE on the model GPU. The frozen verifier rebuilt it on CPU,
whose FP32 power results can differ by a few ULPs. Reconstructing RoPE on its
original device must restore the ENTIRE fixed-state hash; no field or tolerance
is removed. Training, caches, selection and scoring remain unchanged.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from . import jspace_3b_broad_recovery as trial
from .jspace_full_lease import wait_ready

base,old,read=trial.base,trial.old,trial.read
SCHEMA='jspace.3b-broad-verification-repair.v1'
METHOD='CPU strict checkpoint load; reconstruct original nonpersistent RoPE on model CUDA device; require unchanged whole fixed-state hash before packing'
FAILURE='ValueError: Selected geometry/readout differs from scientific initializer'


def destination_state(directory,arm,fingerprint):
    """Allow a first attempt or exactly the recorded constructor failure."""
    directory=Path(directory)
    if not directory.exists():return dict(mode='first_verification',failure_sha256=None)
    if not directory.is_dir() or {p.name for p in directory.iterdir()}!={'failure.json'}:
        raise ValueError('Completed or unexpected existing verification files; no overwrite permitted')
    failure=read(directory/'failure.json')
    if (failure.get('status')!='failed' or failure.get('arm')!=arm or failure.get('phase')!='verify' or
            failure.get('fingerprint')!=fingerprint or failure.get('error')!=FAILURE):
        raise ValueError('Only the authenticated nonpersistent-RoPE constructor failure may be repaired')
    return dict(mode='append_after_constructor_failure',failure_sha256=base.sha(directory/'failure.json'))


def assert_fixed(stack,E,rotary,expected):
    actual=old.fixed_state(stack,E,rotary)
    if actual!=expected:raise ValueError('Complete fixed geometry/readout/rotary fingerprint still differs')
    return actual


def restore_rotary(plan,device,*,guard,rotary_factory=None):
    guard()
    config=trial.cache.builder.native._config(plan['model']['path'],None)
    with trial._model_device(device):
        return (rotary_factory or Qwen2RotaryEmbedding)(config=config,device=device).eval().requires_grad_(False)


def prepare(root,recovery_root,arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Repair preparation requires CUDA hidden')
    root,recovery_root=Path(root).resolve(),Path(recovery_root).resolve()
    if root.exists():raise FileExistsError(root)
    if arm not in trial.cache.ARMS:raise ValueError('Unknown recovery arm')
    p=trial.verify(recovery_root);trained=trial.checked(recovery_root,arm,'train',p)
    target=recovery_root/arm/'verify';prior=destination_state(target,arm,p['fingerprint'])
    inputs={}
    for name in ('plan.json',f'{arm}/train/manifest.json',f'{arm}/train/result.json'):
        old.pin(inputs,recovery_root/name)
    if prior['failure_sha256']:old.pin(inputs,target/'failure.json',prior['failure_sha256'])
    candidates={c['checkpoint_sha256']:c for c in (trained['candidate'],trained['diagnostic_best'])}
    for c in candidates.values():
        old.pin(inputs,c['checkpoint'],c['checkpoint_sha256']);old.pin(inputs,c['report'],c['report_sha256'])
    files=trial.source.inherited.local_import_closure([Path(__file__),
        base.ROOT/'tests/structure_net/test_jspace_3b_broad_verify_repair.py'])
    plan=dict(schema=SCHEMA,method=METHOD,recovery_root=str(recovery_root),arm=arm,
        original_fingerprint=p['fingerprint'],target=str(target),prior_destination=prior,
        train_manifest_sha256=base.sha(recovery_root/arm/'train/manifest.json'),
        expected_fixed_state=trained['fixed_state'],candidates=candidates,
        allocation_mib=11264,allocator_mib=11008,confirmed=False,
        input_sha256=inputs,input_identity={name:old.file_identity(name) for name in inputs},
        source_sha256={str(f):base.sha(f) for f in sorted(files)})
    plan['fingerprint']=base.digest(plan);root.mkdir(parents=True)
    for name,digest in plan['source_sha256'].items():
        target_file=root/'producer'/Path(name).relative_to(base.ROOT)
        target_file.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target_file)
        if base.sha(target_file)!=digest:raise ValueError('Repair producer archive changed')
    base.write(root/'plan.json',plan)
    return dict(status='prepared',fingerprint=plan['fingerprint'],original_fingerprint=p['fingerprint'])


def verify(root):
    root=Path(root);plan=read(root/'plan.json');trial.source.check_plan(plan)
    if (plan.get('schema')!=SCHEMA or plan.get('method')!=METHOD or plan.get('arm')not in trial.cache.ARMS or
            plan.get('allocation_mib')!=11264 or plan.get('allocator_mib')!=11008 or plan.get('confirmed')is not False):
        raise ValueError('Repair protocol changed')
    for name,identity in plan['input_identity'].items():
        if old.file_identity(name)!=identity or base.sha(name)!=plan['input_sha256'][name]:
            raise ValueError('Authenticated repair input changed')
    for name,digest in plan['source_sha256'].items():
        if base.sha(name)!=digest or base.sha(root/'producer'/Path(name).relative_to(base.ROOT))!=digest:
            raise ValueError('Repair source or archive changed')
    p=trial.verify(plan['recovery_root']);trained=trial.checked(plan['recovery_root'],plan['arm'],'train',p)
    candidates={c['checkpoint_sha256']:c for c in (trained['candidate'],trained['diagnostic_best'])}
    if (p['fingerprint']!=plan['original_fingerprint'] or trained['fixed_state']!=plan['expected_fixed_state'] or
            candidates!=plan['candidates'] or base.sha(Path(plan['recovery_root'])/plan['arm']/'train/manifest.json')!=plan['train_manifest_sha256']):
        raise ValueError('Original completed training or selected checkpoints changed')
    return plan,p,trained


def publish(root,plan,result,resources):
    """Publish only successful exact checks, retaining the old failure bytes."""
    root=Path(root);target=Path(plan['target'])
    allocator=resources.get('allocator',{});uuid=allocator.get('physical_uuid','')
    peak=resources.get('peak_reserved_bytes',0)
    if (not uuid.startswith('GPU-0370fdb0') or resources.get('allocation')!={uuid:11264} or
            allocator.get('lease_bytes')!=11264*1024**2 or
            allocator.get('allocator_limit_bytes')!=11008*1024**2 or
            not isinstance(peak,int) or isinstance(peak,bool) or not 0<peak<=11008*1024**2):
        raise ValueError('Verification resource receipt violates the original reservation limits')
    if destination_state(target,plan['arm'],plan['original_fingerprint'])!=plan['prior_destination']:
        raise ValueError('Verification destination changed since repair preparation')
    if (not result.get('passed') or result.get('status')!='completed' or result.get('arm')!=plan['arm'] or
            set(result.get('checks',{}))!=set(plan['candidates'])):
        raise ValueError('All selected and diagnostic checkpoints must pass before publication')
    for digest,c in plan['candidates'].items():
        proof=result['checks'][digest];reference=read(c['report'])
        if (not proof.get('strict_state_load') or not proof.get('fixed_state_unchanged') or
                proof.get('fixed_state')!=plan['expected_fixed_state'] or proof.get('checkpoint_sha256')!=digest or
                proof.get('metrics')!=reference['metrics'] or proof.get('records')!=reference['records']):
            raise ValueError('Unverified checkpoint or changed metrics/records cannot be published')
    metadata=dict(schema=SCHEMA,method=METHOD,fingerprint=plan['fingerprint'],repair_root=str(root.resolve()),
        original_fingerprint=plan['original_fingerprint'],prior_destination=plan['prior_destination'],
        source_sha256=plan['source_sha256'])
    target.mkdir(parents=True,exist_ok=True)
    archive=target/'repair';archive.mkdir()
    shutil.copy2(root/'plan.json',archive/'plan.json');shutil.copytree(root/'producer',archive/'producer')
    for name,digest in plan['source_sha256'].items():
        if base.sha(archive/'producer'/Path(name).relative_to(base.ROOT))!=digest:raise ValueError('Published repair source differs')
    base.write(archive/'receipt.json',metadata)
    base.write(target/'result.json',dict(result,verification_producer=metadata))
    if plan['prior_destination']['failure_sha256'] and base.sha(target/'failure.json')!=plan['prior_destination']['failure_sha256']:
        raise ValueError('Original failure receipt changed')
    base.write(target/'manifest.json',dict(status='completed',arm=plan['arm'],phase='verify',
        fingerprint=plan['original_fingerprint'],repair_fingerprint=plan['fingerprint'],**resources,
        artifacts={str(f.relative_to(target)):base.sha(f) for f in target.rglob('*') if f.is_file()}))
    receipt=dict(status='completed',fingerprint=plan['fingerprint'],original_fingerprint=plan['original_fingerprint'],
        target=str(target),result_sha256=base.sha(target/'result.json'),manifest_sha256=base.sha(target/'manifest.json'),
        prior_failure_sha256=plan['prior_destination']['failure_sha256'],passed=True)
    base.write(root/'result.json',receipt)
    base.write(root/'manifest.json',dict(status='completed',fingerprint=plan['fingerprint'],
        artifacts={'plan.json':base.sha(root/'plan.json'),'result.json':base.sha(root/'result.json')}))
    return receipt


def run(root):
    root=Path(root).resolve();plan,p,trained=verify(root)
    if destination_state(plan['target'],plan['arm'],p['fingerprint'])!=plan['prior_destination']:
        raise ValueError('Repair destination already completed or changed')
    if (root/'result.json').exists() or (root/'run_failure.json').exists():raise FileExistsError('Repair attempt already recorded')
    lease=settings=None;data=None;started=time.monotonic();cuda=False
    try:
        lease=wait_ready([0]);settings=trial.source.inherited.cap_allocator(lease,p)
        torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();cuda=True;torch.set_num_threads(4)
        def guard():
            current=wait_ready([0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:raise RuntimeError('Repair reservation changed')
            if torch.cuda.max_memory_reserved()>settings['allocator_limit_bytes']:raise MemoryError('Repair allocator cap exceeded')
        data=trial.dataset_for(p,plan['arm']);checks={}
        with trial.cache.sanity.execution.arithmetic_policy():
            for digest,c in plan['candidates'].items():
                guard();stack,norm,E,cpu_rotary,inheritance=trial._build(p,c['checkpoint'],'cpu',False)
                cpu_fixed=old.fixed_state(stack,E,cpu_rotary)
                rotary=restore_rotary(p,'cuda:0',guard=guard)
                differences=[dict(name=name,shape=list(a.shape),dtype=str(a.dtype),
                    equal=torch.equal(a,b.cpu()),max_abs=float((a.float()-b.cpu().float()).abs().max()))
                    for (name,a),(other,b) in zip(cpu_rotary.named_buffers(),rotary.named_buffers()) if name==other]
                before=assert_fixed(stack,E,rotary,trained['fixed_state'])
                packed=trial.cache.builder.packed_clone(stack,device='cpu')
                assert_fixed(stack,E,rotary,trained['fixed_state'])
                del stack,cpu_rotary
                packed,norm,E=packed.to('cuda:0'),norm.to('cuda:0'),E.to('cuda:0')
                metrics,rows=trial.score_packed(packed,norm,E,rotary,data,p['label_ids'],guard)
                reference=read(c['report'])
                if metrics!=reference['metrics'] or rows!=reference['records']:
                    raise ValueError('Corrected fresh packed replay differs from selected development')
                checks[digest]=dict(checkpoint_sha256=digest,strict_state_load=True,fixed_state_unchanged=True,
                    fixed_state=before,inheritance=inheritance,records=rows,metrics=metrics,
                    cpu_constructed_fixed_state=cpu_fixed,rotary_reconstruction=differences)
                del packed,norm,E,rotary;gc.collect();torch.cuda.empty_cache()
                base.event(root/'progress.jsonl','checkpoint_verified',checkpoint_sha256=digest,updates=c['updates'])
        guard()
        result=dict(status='completed',arm=plan['arm'],passed=True,checks=checks,
            development_passed=trained['development_passed'],confirmed=False)
        receipt=publish(root,plan,result,dict(lease_id=lease['id'],allocation=lease['allocation'],allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started))
        trial.checked(plan['recovery_root'],plan['arm'],'verify',p)
        return receipt
    except Exception as error:
        base.write(root/'run_failure.json',dict(status='failed',fingerprint=plan['fingerprint'],
            error=f'{type(error).__name__}: {error}',lease=lease,allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None));raise
    finally:
        if data is not None:data.close()
        gc.collect()


def checked(root):
    root=Path(root);plan,p,_=verify(root);receipt=read(root/'result.json');manifest=read(root/'manifest.json')
    if (manifest.get('status')!='completed' or manifest.get('fingerprint')!=plan['fingerprint'] or
            manifest.get('artifacts')!={'plan.json':base.sha(root/'plan.json'),'result.json':base.sha(root/'result.json')} or
            receipt.get('fingerprint')!=plan['fingerprint'] or not receipt.get('passed')):
        raise ValueError('Repair completion not authenticated')
    target=Path(plan['target']);result=trial.checked(plan['recovery_root'],plan['arm'],'verify',p)
    if (base.sha(target/'result.json')!=receipt['result_sha256'] or base.sha(target/'manifest.json')!=receipt['manifest_sha256'] or
            result.get('verification_producer',{}).get('fingerprint')!=plan['fingerprint'] or
            read(target/'manifest.json').get('repair_fingerprint')!=plan['fingerprint'] or
            base.sha(target/'repair/plan.json')!=base.sha(root/'plan.json')):
        raise ValueError('Corrected verification linkage changed')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','run','check'))
    parser.add_argument('--root',required=True);parser.add_argument('--recovery-root');parser.add_argument('--arm',choices=trial.cache.ARMS)
    args=parser.parse_args()
    if args.phase=='prepare':
        if not args.recovery_root or not args.arm:parser.error('prepare requires recovery-root and arm')
        value=prepare(args.root,args.recovery_root,args.arm)
    else:value=run(args.root) if args.phase=='run' else checked(args.root)
    print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
