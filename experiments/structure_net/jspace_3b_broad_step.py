"""One prospective cut21 expansion after accepted broad-recovery assessment."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_3b_broad_step_capture as capture
from . import jspace_3b_broad_recovery as trial
from . import jspace_3b_broad_assess as assessment
from . import jspace_3b_parent_feature as feature
from . import jspace_3b_entrance_distill as entrance
from .jspace_full_lease import wait_ready

cache,base,old,read=trial.cache,trial.base,trial.old,trial.read
builder,core,source=cache.builder,trial.core,trial.source
SCHEMA='jspace.3b-broad-step.v1'
MODULE='experiments.structure_net.jspace_3b_broad_step'
PROTOCOL=base.ROOT/'docs/07 - Status Reports/jspace/2026-11-15_qwen3b-broad-fifteen-block-step.md'
PHASES=('boundary','capture','fit','parentfeatures','pilot','prealign','train','verify')
DEPENDENCIES={'boundary':(),'capture':('boundary',),'fit':('capture',),'parentfeatures':('fit',),
    'pilot':('fit','parentfeatures'),'prealign':('pilot',),'train':('pilot','prealign','parentfeatures'),
    'verify':('train',)}
CONFIG=trial.CONFIG
ENTRANCE_CONFIG=replace(entrance.EntranceConfig(),epochs=1)
ALPHA=cache.ALPHA
ARM='native_content'
POLICY='DEV_eligible_highest_correct_then_earliest_joint_update; raw15_precedes_equal_warm0; parent14_fallback; review_only_no_quality_promotion'


def _pin(inputs,path,expected=None):
    path=str(Path(path).resolve())
    if path not in inputs:inputs[path]=base.sha(path)
    if expected is not None and inputs[path]!=expected:raise ValueError('Pinned input SHA mismatch: '+path)


def admit_parent_assessment(selection,result,*,exploratory_parent=False):
    """A failed independent audit permits only explicitly authorized exploration."""
    if type(exploratory_parent)is not bool:raise ValueError('Explicit exploratory-parent boolean required')
    audit=result.get('audit',{})
    if (result.get('status')!='completed' or result.get('selection')!=selection or
            type(audit.get('descriptive_audit_passed'))is not bool or not selection.get('development_passed')):
        raise ValueError('Completed authenticated parent assessment and DEV selection required')
    if not audit['descriptive_audit_passed'] and not exploratory_parent:
        raise ValueError('Failed parent audit requires explicit --exploratory-parent admission')
    return dict(exploratory_parent=exploratory_parent,parent_audit=audit,
        parent_audit_passed=audit['descriptive_audit_passed'],confirmed=False,
        scope='Exploratory compression; parent audit is admission evidence only, never training or checkpoint-selection data')


def prepare(root,recovery_root,assessment_root,*,exploratory_parent=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires CUDA hidden')
    root,recovery_root,assessment_root=map(lambda p:Path(p).resolve(),(root,recovery_root,assessment_root))
    if root.exists():raise FileExistsError(root)
    selection=trial.checked_selection(recovery_root);rp=trial.verify(recovery_root)
    ap=assessment.verify(assessment_root);am=read(assessment_root/'manifest.json');ar=read(assessment_root/'result.json')
    if (not selection['development_passed'] or am.get('status')!='completed' or ar.get('status')!='completed' or
            am.get('fingerprint')!=ap['fingerprint'] or ar.get('fingerprint')!=ap['fingerprint'] or
            am.get('artifacts',{}).get('result.json')!=base.sha(assessment_root/'result.json') or
            ar.get('selection')!=selection or ap['selection']!=selection):
        raise ValueError('Completed global selection and authenticated postselection assessment required')
    admission=admit_parent_assessment(selection,ar,exploratory_parent=exploratory_parent)
    assessed={}
    for phase in assessment.PHASES:
        assessed[phase]=assessment.checked(assessment_root,phase,ap)
        if ar['phase_manifest_sha256'][phase]!=base.sha(assessment_root/phase/'manifest.json'):
            raise ValueError('Completed assessment phase changed')
    expected_audit=assessment.audit_summary(assessed['audit_native']['records'],assessed['audit_compact']['records'],
        read(assessment_root/'items.json')['items'],development_passed=selection['development_passed'])
    if ar['audit']!=expected_audit:raise ValueError('Parent audit summary differs from its authenticated paired records')
    cp=cache.verify(selection['cache_root']);candidate=selection['candidate']
    trained=trial.checked(recovery_root,candidate['arm'],'train',rp)
    items=capture.content_items(read(Path(selection['cache_root'])/'items.json')['items'])
    # Fixed first32 TRAIN cases per family, without looking at outcomes.
    indices=sorted(i for family in capture.FAMILIES for i in
        [j for j,c in enumerate(items) if c['split']=='train' and c['family']==family][:32])
    inputs=dict(rp['input_sha256'])
    for path in (recovery_root/'plan.json',recovery_root/'selection.json',recovery_root/'manifest.json',
            recovery_root/candidate['arm']/'train/manifest.json',recovery_root/candidate['arm']/'train/result.json',
            recovery_root/candidate['arm']/'verify/manifest.json',recovery_root/candidate['arm']/'verify/result.json',
            assessment_root/'plan.json',assessment_root/'manifest.json',assessment_root/'result.json'):_pin(inputs,path)
    _pin(inputs,candidate['checkpoint'],candidate['checkpoint_sha256']);_pin(inputs,candidate['report'],candidate['report_sha256'])
    for phase in assessment.PHASES:_pin(inputs,assessment_root/phase/'manifest.json')
    files=source.inherited.local_import_closure([Path(__file__),PROTOCOL,
        base.ROOT/'tests/structure_net/test_jspace_3b_broad_step.py'])
    plan=dict(schema=SCHEMA,cut=21,parent_cut=22,layers=list(range(21,36)),phases=list(PHASES),
        recovery_root=str(recovery_root),recovery_fingerprint=rp['fingerprint'],assessment_root=str(assessment_root),
        assessment_fingerprint=ap['fingerprint'],parent_checkpoint=candidate['checkpoint'],
        parent_admission=admission,exploratory_parent=exploratory_parent,
        parent_checkpoint_sha256=candidate['checkpoint_sha256'],parent_candidate=candidate,parent_fits=cp['fits'],
        expected_parent_fixed_state=trained['fixed_state'],model=cp['model'],data_root=cp['data_root'],
        cache_root=selection['cache_root'],training_order=cp['training_order'],label_ids=cp['label_ids'],
        boundary_root=str(root/'boundary'),capture_root=str(root/'capture'),fit_root=str(root/'fit'),
        calibration_indices=indices,config=asdict(CONFIG),entrance_config=asdict(ENTRANCE_CONFIG),alpha=ALPHA,
        checkpoints=list(trial.CHECKPOINT_UPDATES),policy=POLICY,native_records=rp['native_records'],native_correct=rp['native_correct'],
        allocation_mib=11264,allocator_mib=11008,resident_budget_mib=6144,
        input_sha256=inputs,input_identity={name:old.file_identity(name) for name in inputs},
        source_sha256={str(f):base.sha(f) for f in sorted(files)},
        environment=dict(torch=torch.__version__,transformers=__import__('transformers').__version__),
        optimizer_restart=trial.RESET,confirmed=False,full36=False,
        audit_scope='Completed parent assessment retained honestly; no audit examples/labels enter fitting, training or selection')
    if plan['native_correct']!=195:raise ValueError('Original matched native195/256 control required')
    capture.validate_plan_items(plan);plan['fingerprint']=base.digest(plan);root.mkdir(parents=True)
    for name,digest in plan['source_sha256'].items():
        dest=root/'producer'/Path(name).relative_to(base.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
        if base.sha(dest)!=digest:raise ValueError('Producer archive changed')
    base.write(root/'plan.json',plan)
    return dict(status='prepared',fingerprint=plan['fingerprint'],parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],cut=21)


def verify(root):
    root=Path(root);p=read(root/'plan.json');source.check_plan(p)
    if (p.get('schema')!=SCHEMA or p.get('cut')!=21 or p.get('parent_cut')!=22 or p.get('layers')!=list(range(21,36)) or
            p.get('phases')!=list(PHASES) or p.get('config')!=asdict(CONFIG) or p.get('entrance_config')!=asdict(ENTRANCE_CONFIG) or
            p.get('alpha')!=ALPHA or p.get('checkpoints')!=list(trial.CHECKPOINT_UPDATES) or p.get('policy')!=POLICY or
            p.get('native_correct')!=195 or p.get('allocation_mib')!=11264 or p.get('allocator_mib')!=11008 or
            p.get('resident_budget_mib')!=6144 or p.get('optimizer_restart')!=trial.RESET or
            p.get('confirmed')is not False or p.get('full36')is not False):raise ValueError('Prospective step policy changed')
    assessment_result=read(Path(p['assessment_root'])/'result.json')
    if p.get('parent_admission')!=admit_parent_assessment(assessment_result['selection'],assessment_result,
            exploratory_parent=p.get('exploratory_parent')):
        raise ValueError('Parent audit evidence or explicit exploratory admission changed')
    for name,identity in p['input_identity'].items():
        if old.file_identity(name)!=identity:raise ValueError('Immutable prepared input changed: '+name)
    for name,digest in p['source_sha256'].items():
        if base.sha(name)!=digest or base.sha(root/'producer'/Path(name).relative_to(base.ROOT))!=digest:
            raise ValueError('Pinned producer or archive changed')
    if p['environment']!={'torch':torch.__version__,'transformers':__import__('transformers').__version__}:
        raise ValueError('Pinned numerical environment changed')
    return p


def _items(p):return capture.content_items(read(Path(p['cache_root'])/'items.json')['items'])

def _dataset(p,parent=False):
    return cache.RecoveryDataset(_items(p),Path(p['cache_root'])/'native/boundary_22.npy' if parent else Path(p['boundary_root'])/'boundary_21.npy',
        Path(p['cache_root'])/'native/teacher_logits.npy',arm=ARM,label_ids=p['label_ids'])

def _consume(path,digest):
    if base.sha(path)!=digest:raise ValueError('Consumed artifact changed: '+str(path))
    return builder.native._load(path)

def _build(p,checkpoint,device,trainable,*,parent=False):
    fits={int(k):v for k,v in p['parent_fits'].items()}
    if not parent:fits[21]=str(Path(p['fit_root'])/'block_21')
    result=builder.build_stack(p['model']['path'],fits,22 if parent else 21,checkpoint,
        device=device,trainable=trainable,checkpoint_blocks=trainable)
    # Newly compiled polynomial_bias can begin FP32. Canonicalize learned
    # parameters before any raw score/snapshot, as the trainer does later.
    # Fixed bases, anchors and bridge buffers must retain their original dtype.
    for module in result[:2]:
        for parameter in module.parameters():parameter.data=parameter.data.to(torch.bfloat16)
    return result


def score_stack(stack,norm,E,rotary,data,label_ids,guard):
    packed=builder.packed_clone(stack,device='cpu');n=core.cpu_module_clone(norm).eval().requires_grad_(False)
    try:return trial.score_packed(packed.to(E.device),n.to(E.device),E,rotary,data,label_ids,guard)
    finally:del packed,n;gc.collect()


@torch.inference_mode()
def capture_parent_features(out,stack,norm,E,rotary,data,*,parent_sha,guard,progress=lambda *_:None):
    """Only actual TRAIN final coordinates; never a full-vocabulary logit bank."""
    guard()
    if list(stack.layer_indices)!=list(range(22,36)):raise ValueError('Actual fourteen-block parent required')
    rows=feature.target_rows(data.items)
    if data.training!=[r['source_index'] for r in rows]:raise ValueError('Complete original TRAIN feature order required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    before=[(id(t),t._version) for m in (stack,norm,rotary) for t in (*m.parameters(),*m.buffers())]+[(id(E),E._version)]
    path=out/feature.NAMES[1];bank=np.lib.format.open_memmap(path,mode='w+',dtype=np.int16,shape=(len(rows),stack.blocks[-1].rank))
    lookup={r['source_index']:i for i,r in enumerate(rows)}
    try:
        for start in range(0,len(data.training),4):
            guard();indices=data.training[start:start+4]
            with trial._model_device(E.device):
                batch=data.batch(indices,device=E.device)
                _,z,_=feature.packed_coordinates(stack,batch['hidden'],position_ids=batch['position_ids'],
                    position_embeddings=rotary(batch['hidden'],batch['position_ids']),attention_mask=batch['attention_mask'],guard=guard)
                values=z[torch.arange(len(indices),device=E.device),torch.tensor(batch['lengths'],device=E.device)-1]
                if values.dtype!=torch.bfloat16 or not torch.isfinite(values).all():raise ValueError('Finite actual BF16 feature targets required')
                bank[[lookup[i] for i in indices]]=values.cpu().contiguous().view(torch.int16).numpy()
            progress(dict(completed=start+len(indices),total=len(data.training)))
        bank.flush();values=torch.from_numpy(np.array(bank,copy=True)).view(torch.bfloat16)
        energy=float(values.float().square().mean(-1).mean())
    finally:bank._mmap.close()
    after=[(id(t),t._version) for m in (stack,norm,rotary) for t in (*m.parameters(),*m.buffers())]+[(id(E),E._version)]
    if before!=after or not math.isfinite(energy) or energy<=0:raise ValueError('Parent changed or invalid TRAIN feature energy')
    metadata=dict(schema=feature.SCHEMA,rows=rows,shape=list(values.shape),storage='lossless BF16 bits as int16',
        parent_checkpoint_sha256=parent_sha,parent_cut=22,parent_layers=list(range(22,36)),coordinate_layer=35,
        position='last valid token',energy_mean=energy,development_fitted=False,
        coordinate_location='actual post-block35 output before reconstruction; no reprojection')
    base.write(out/feature.NAMES[2],metadata)
    result=dict(schema=feature.SCHEMA,status='completed',features=feature._feature_receipt(metadata,out),
        fixed_state_unchanged=True,logits_written=False,training_count=len(rows))
    base.write(out/feature.NAMES[3],result);return result


def candidate(path,report,metrics,rows,*,updates,p,native_records,origin):
    cases=[c for c in _items(p) if c['split']=='development']
    return dict(arm=ARM,updates=updates,origin=origin,checkpoint=str(Path(path).resolve()),checkpoint_sha256=base.sha(path),
        report=str(Path(report).resolve()),report_sha256=base.sha(report),metrics=metrics,
        eligibility=trial.eligibility(rows,native_records,cases),protected_source=updates==0)


def fit_step(trainer,path,p,raw_candidate,*,event=lambda *_a,**_kw:None,checkpoints=trial.CHECKPOINT_UPDATES):
    if trainer.updates or trainer.finished:raise ValueError('Fresh warm optimizer required')
    order=p['training_order'];effective=trainer.config.microbatch*trainer.config.accumulation
    if len(order)%effective or sorted(order)!=sorted(trainer.dataset.training):raise ValueError('Complete fixed TRAIN pass required')
    total=len(order)//effective
    if tuple(checkpoints)!=tuple(sorted(set(checkpoints))) or checkpoints[0]!=0 or checkpoints[-1]!=total:
        raise ValueError('Initializer and complete pass checkpoint schedule required')
    out=Path(path);out.mkdir(parents=True,exist_ok=False);before=old.fixed_state(trainer.stack,trainer.readout,trainer.rotary)
    history=[];updates=[];started=time.monotonic()
    def record(step):
        metrics,rows=trainer.packed_development();report=out/f'update_{step:03d}_development.json'
        base.write(report,dict(metrics=metrics,records=rows));checkpoint=out/f'update_{step:03d}.pt'
        old.save(checkpoint,trainer.snapshot())
        c=candidate(checkpoint,report,metrics,rows,updates=step,p=p,native_records=p['native_records'],origin='warm15' if step==0 else 'joint15')
        history.append(c);event('development',candidate=c)
    record(0)
    for start in range(0,len(order),effective):
        indices=list(order[start:start+effective]);value=trainer.train_window(indices,learning_rate=trainer.config.learning_rate)
        value=dict(value,indices=indices);updates.append(value);event('update',**value)
        if trainer.updates in checkpoints:record(trainer.updates)
    if trainer.updates!=total or old.fixed_state(trainer.stack,trainer.readout,trainer.rotary)!=before:raise ValueError('Budget or geometry changed')
    trainer.finished=True;pool=[raw_candidate,*history];decision=trial.choose(pool)
    return dict(status='completed',config=asdict(trainer.config),updates=total,training=updates,history=history,raw_candidate=raw_candidate,
        **decision,step_policy=POLICY,fixed_state=before,fixed_state_unchanged=True,alpha=trainer.alpha,
        normalization_energy=trainer.feature_targets.energy,optimizer_restart=trial.RESET,order_digest=base.digest(order),
        fallback_parent_checkpoint=p['parent_checkpoint'],fallback_parent_checkpoint_sha256=p['parent_checkpoint_sha256'],
        promoted=False,elapsed_seconds=time.monotonic()-started,confirmed=False)


def _validate_candidate(c,p):
    report=read(c['report']);cases=[x for x in _items(p) if x['split']=='development']
    cache.sanity.validate_records(report['records'],cases)
    metrics=report['metrics'];correct=sum(x['prediction']==x['answer'] for x in report['records'])
    if (base.sha(c['report'])!=c['report_sha256'] or c.get('arm')!=ARM or c['metrics']!=metrics or
            metrics['count']!=256 or metrics['correct']!=correct or metrics['accuracy']!=correct/256 or
            not all(math.isfinite(metrics[k]) for k in ('loss','kl','ce')) or
            c['eligibility']!=trial.eligibility(report['records'],p['native_records'],cases)):
        raise ValueError('Candidate records/metrics or original-native eligibility changed')
    return report


def validate_train(r,p,raw,energy):
    updates=r.get('training',[]);history=r.get('history',[])
    if (r.get('config')!=asdict(CONFIG) or r.get('updates')!=288 or len(updates)!=288 or
            [u.get('update') for u in updates]!=list(range(1,289)) or
            [i for u in updates for i in u.get('indices',[])]!=p['training_order'] or
            [c.get('updates') for c in history]!=list(trial.CHECKPOINT_UPDATES) or
            r.get('raw_candidate')!=raw or r.get('step_policy')!=POLICY or r.get('alpha')!=ALPHA or
            r.get('normalization_energy')!=energy or r.get('order_digest')!=base.digest(p['training_order']) or
            r.get('optimizer_restart')!=trial.RESET or not r.get('fixed_state_unchanged') or
            any(u.get('cases')!=16 or u.get('learning_rate')!=5e-6 or u.get('gradients_present_by_block')!=[True]*15 or
                u.get('alpha')!=ALPHA or u.get('normalization_energy')!=energy or
                not all(math.isfinite(u[k]) for k in ('loss','endpoint_loss','kl','ce','feature_loss')) for u in updates)):
        raise ValueError('Complete unchanged fifteen-block288-update recovery required')
    for c in [raw,*history]:_validate_candidate(c,p)
    choice=trial.choose([raw,*history])
    if any(r.get(k)!=v for k,v in choice.items()):raise ValueError('Wrong absolute-native winner or earliest tie')
    if (r.get('fallback_parent_checkpoint')!=p['parent_checkpoint'] or
            r.get('fallback_parent_checkpoint_sha256')!=p['parent_checkpoint_sha256'] or r.get('promoted')is not False):
        raise ValueError('Training cannot promote before fresh verification')
    return choice


def _resource_check(m,dual):
    roles=(('front','GPU-0370fdb0',11264),('tail','GPU-7b1db3cb',7168)) if dual else (('front','GPU-0370fdb0',11264),)
    expected={}
    for role,prefix,mib in roles:
        a=m.get('allocator',{}).get(role,{}) if dual else m.get('allocator',{})
        uuid=a.get('uuid' if dual else 'physical_uuid','');peak=m.get('peak_reserved_bytes',{}).get(role,-1) if dual else m.get('peak_reserved_bytes',-1)
        if not uuid.startswith(prefix) or a.get('allocator_limit_bytes')!=(mib-256)*1024**2 or not 0<peak<=a['allocator_limit_bytes']:
            raise ValueError('Wrong device, cap or peak in reservation receipt')
        expected[uuid]=mib
    if m.get('allocation')!=expected:raise ValueError('Unexpected experiment reservation allocation')


def checked(root,phase,plan=None):
    root=Path(root);p=verify(root) if plan is None else plan
    if phase not in PHASES:raise ValueError('Unknown step phase')
    out=root/phase;m=read(out/'manifest.json');r=read(out/'result.json')
    if (m.get('status')!='completed' or m.get('phase')!=phase or m.get('fingerprint')!=p['fingerprint'] or
            r.get('status')!='completed' or m.get('artifacts',{}).get('result.json')!=base.sha(out/'result.json') or
            set(m.get('artifact_identity',{}))!=set(m.get('artifacts',{}))):raise ValueError('Incomplete matching phase required')
    for name,identity in m['artifact_identity'].items():
        path=(out/name).resolve()
        if not path.is_relative_to(out.resolve()) or old.file_identity(path)!=identity:raise ValueError('Phase artifact identity changed')
        if path.suffix in ('.json','.jsonl') and base.sha(path)!=m['artifacts'][name]:raise ValueError('Phase evidence changed')
    expected={d:base.sha(root/d/'manifest.json') for d in DEPENDENCIES[phase]}
    if m.get('dependencies')!=expected:raise ValueError('Phase dependency chain changed')
    _resource_check(m,phase in ('pilot','train'))
    if phase in ('boundary','capture','fit'):
        if r!=read(out/'step_receipt.json') or r.get('schema')!=capture.SCHEMA or r.get('parent_checkpoint_sha256')!=p['parent_checkpoint_sha256']:
            raise ValueError('Wrong capture/fit receipt')
        if phase=='boundary' and (r.get('native_pre22_control')!={'passed':True,'bit_exact':True,'count':4864,'max_abs':0.} or
                not r.get('fixed_state_unchanged')):raise ValueError('Original native prefix control failed')
        if phase=='capture' and (r.get('parent_fixed_state')!=p['expected_parent_fixed_state'] or
                r.get('native_seam')!={'passed':True,'bit_exact':True,'count':4864,'max_abs':0.} or
                r.get('calibration_indices')!=p['calibration_indices'] or r.get('positions')!='content4'):
            raise ValueError('Calibration parent, seam or TRAIN indices changed')
        if phase=='fit' and (r.get('rank'),r.get('head_dim'),r.get('anchors'),r.get('removed'))!=(1024,64,24,1024):
            raise ValueError('Fixed structure changed')
    elif phase=='parentfeatures':
        _,receipt=feature.load_targets(out,_items(p),p['parent_checkpoint_sha256'])
        if (r.get('features')!=receipt or receipt['parent_cut']!=22 or receipt['training_count']!=4608 or
                r.get('logits_written')is not False or not r.get('fixed_state_unchanged')):
            raise ValueError('Actual accepted-parent TRAIN targets required')
    elif phase=='pilot':
        probe=r.get('probe',{})
        if (not r.get('passed') or not r.get('discarded') or not r.get('entrance_probe',{}).get('passed') or
                not r.get('restored_raw_exact') or not probe.get('passed') or probe.get('layer_indices')!=list(range(21,36)) or
                not probe.get('fixed_state_unchanged') or probe.get('fixed_state')!=r.get('fixed_state') or
                not r.get('inheritance',{}).get('passed') or r['inheritance'].get('unchanged_layers')!=list(range(22,36)) or
                r.get('geometry_readout_unchanged')is not True):raise ValueError('Discarded entrance/joint pilot or inheritance failed')
        _validate_candidate(r['raw_candidate'],p)
        if (Path(r['raw_candidate']['checkpoint']).resolve()!=(out/'initializer.pt').resolve() or
                r['raw_candidate']['checkpoint_sha256']!=m['artifacts'].get('initializer.pt')):
            raise ValueError('Raw initializer checkpoint linkage changed')
    elif phase=='prealign':
        training=[u for row in r.get('history',[]) for u in row.get('training',[])]
        if (r.get('config')!=asdict(ENTRANCE_CONFIG) or r.get('completed_epochs')!=1 or r.get('updates')!=288 or
                [x.get('epoch') for x in r.get('history',[])]!=[0,1] or len(training)!=288 or
                [u.get('update') for u in training]!=list(range(1,289)) or r.get('layer_idx')!=21 or
                r.get('inherited_layers')!=list(range(22,36)) or not r.get('strict_reload_exact') or
                not r.get('fixed_state_unchanged') or r.get('selection')!='lowest_development_feature_loss_earliest_tie'):
            raise ValueError('One bounded entrance-only pass required')
        best=min(r['history'],key=lambda x:(x['metrics']['loss'],x['epoch']))
        if r['selected_epoch']!=best['epoch'] or r['selected']!=best['metrics']:raise ValueError('Wrong interface-MSE winner')
        if r['selected_checkpoint_sha256']!=m['artifacts'].get('selected_entrance.pt'):
            raise ValueError('Selected entrance checkpoint linkage changed')
        for row in r['history']:
            report=out/f"epoch_{row['epoch']:02d}_development.json"
            if base.sha(report)!=row['development_sha256'] or read(report)['metrics']!=row['metrics']:
                raise ValueError('Entrance metrics no longer match retained reports')
    elif phase=='train':
        pilot=checked(root,'pilot',p);targets=checked(root,'parentfeatures',p)
        validate_train(r,p,pilot['raw_candidate'],targets['features']['normalization_energy'])
        if r.get('fixed_state')!=pilot['fixed_state']:raise ValueError('Joint geometry differs from raw initializer')
        for c in r['history']:
            name=f"update_{c['updates']:03d}.pt"
            if Path(c['checkpoint']).resolve()!=(out/name).resolve() or c['checkpoint_sha256']!=m['artifacts'].get(name):
                raise ValueError('Retained joint checkpoint linkage changed')
    else:
        prior=checked(root,'train',p);candidates={c['checkpoint_sha256']:c for c in (prior['candidate'],prior['diagnostic_best'])}
        if not r.get('passed') or set(r.get('checks',{}))!=set(candidates):raise ValueError('Fresh candidate proofs incomplete')
        for digest,c in candidates.items():
            proof=r['checks'][digest];report=_validate_candidate(c,p)
            if (proof.get('checkpoint_sha256')!=digest or not proof.get('strict_state_load') or
                    proof.get('fixed_state')!=prior['fixed_state'] or not proof.get('fixed_state_unchanged') or
                    proof.get('metrics')!=report['metrics'] or proof.get('records')!=report['records']):
                raise ValueError('Fresh selected metrics/state changed')
        if (r.get('promoted')is not False or
                r.get('development_passed')!=prior['development_passed'] or
                r.get('review_checkpoint')!=(prior['candidate']['checkpoint'] if prior['development_passed'] else p['parent_checkpoint']) or
                r.get('review_checkpoint_sha256')!=(prior['candidate']['checkpoint_sha256'] if prior['development_passed'] else p['parent_checkpoint_sha256']) or
                r.get('fallback_parent_checkpoint_sha256')!=p['parent_checkpoint_sha256'] or
                r.get('compressed_blocks')!=(15 if prior['development_passed'] else 14) or
                r.get('parent_admission')!=p['parent_admission'] or 'accepted_checkpoint' in r or
                r.get('full36')is not False or r.get('confirmed')is not False or r.get('speed_measured')is not False):
            raise ValueError('Exploratory review checkpoint/fallback or quality scope changed')
    return r


def _finish(out,p,phase,result,resources,dependencies):
    if (out/'result.json').exists():
        if read(out/'result.json')!=result:raise ValueError('Existing phase result changed')
    else:base.write(out/'result.json',result)
    artifacts={str(f.relative_to(out)):base.sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    _resource_check(resources,phase in ('pilot','train'))
    base.write(out/'manifest.json',dict(status='completed',phase=phase,fingerprint=p['fingerprint'],dependencies=dependencies,
        artifacts=artifacts,artifact_identity={name:old.file_identity(out/name) for name in artifacts},**resources))


def _resident(stack,norm,E,rotary,data,p,settings,guard,targets):
    stack.requires_grad_(True);norm.requires_grad_(True)
    return trial.BroadTrainer(stack,norm,E,rotary,data,p['label_ids'],config=CONFIG,count=256,
        native_correct=p['native_correct'],parent_correct=p['native_correct'],guard=guard,
        targets=targets,alpha=ALPHA,resident_device=f"cuda:{settings['tail']['device']}",
        resident_budget_bytes=p['resident_budget_mib']*1024**2)


def run(root,phase):
    root=Path(root).resolve();p=verify(root)
    if phase not in PHASES:raise ValueError('Unknown step phase')
    previous={dep:checked(root,dep,p) for dep in DEPENDENCIES[phase]}
    out=root/phase
    if out.exists():raise FileExistsError(out)
    dependencies={dep:base.sha(root/dep/'manifest.json') for dep in DEPENDENCIES[phase]}
    dual=phase in ('pilot','train');lease=settings=None;cuda=False;data=None;pairs=None;limits=[];started=time.monotonic()
    try:
        lease=wait_ready([0,1] if dual else [0])
        settings=source.replay.allocator_settings(lease) if dual else source.inherited.cap_allocator(lease,p)
        front=settings['front']['device'] if dual else 0;device=f'cuda:{front}'
        limits=list(settings.values()) if dual else [dict(device=0,**settings)]
        cuda=True
        for value in limits:torch.cuda.reset_peak_memory_stats(value['device'])
        torch.cuda.set_device(front);torch.set_num_threads(4);torch.manual_seed(CONFIG.seed)
        def guard():
            current=wait_ready([0,1] if dual else [0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:raise RuntimeError('Reservation changed')
            if any(torch.cuda.max_memory_reserved(v['device'])>v['allocator_limit_bytes'] for v in limits):raise MemoryError('Allocator cap exceeded')
        def event(kind,**values):base.event(root/'progress.jsonl',kind,phase=phase,**values)
        with cache.sanity.execution.arithmetic_policy():
            if phase in ('boundary','capture','fit'):
                result=getattr(capture,'run_'+phase)(out,p,guard)
            elif phase=='parentfeatures':
                state=_consume(p['parent_checkpoint'],p['parent_checkpoint_sha256'])
                stack,norm,E,_,_= _build(p,state,'cpu',False,parent=True)
                rotary=capture.repair.restore_rotary(p,device,guard=guard)
                capture.repair.assert_fixed(stack,E,rotary,p['expected_parent_fixed_state'])
                packed=builder.packed_clone(stack,device='cpu');del stack,state
                data=_dataset(p,True)
                result=capture_parent_features(out,packed.to(device),norm.to(device),E.to(device),rotary,data,
                    parent_sha=p['parent_checkpoint_sha256'],guard=guard,progress=lambda v:event('parent_features',**v))
            elif phase=='pilot':
                parent=_consume(p['parent_checkpoint'],p['parent_checkpoint_sha256'])
                stack,norm,E,rotary,inheritance=_build(p,parent,device,True);del parent
                if not inheritance['passed'] or inheritance['unchanged_layers']!=list(range(22,36)):raise ValueError('Parent state/bridges not preserved')
                data=_dataset(p);targets,_=feature.load_targets(root/'parentfeatures',_items(p),p['parent_checkpoint_sha256'])
                metrics,rows=score_stack(stack,norm,E,rotary,data,p['label_ids'],guard)
                out.mkdir();raw=dict(stack=core.recursive_cpu(stack.state_dict()),norm=core.recursive_cpu(norm.state_dict()))
                old.save(out/'initializer.pt',raw);base.write(out/'initial_development.json',dict(metrics=metrics,records=rows))
                raw_candidate=candidate(out/'initializer.pt',out/'initial_development.json',metrics,rows,updates=0,p=p,
                    native_records=p['native_records'],origin='raw15')
                before=old.fixed_state(stack,E,rotary);longest=sorted(data.training,key=lambda i:(-data.lengths[i],i))[:16]
                pairs=entrance.EntrancePairs(_items(p),Path(p['boundary_root'])/'boundary_21.npy',Path(p['cache_root'])/'native/boundary_22.npy')
                local=entrance.EntranceTrainer(stack,norm,E,rotary,pairs,config=ENTRANCE_CONFIG,guard=guard)
                local_probe=local.pilot(longest);del local;gc.collect()
                stack.load_state_dict(raw['stack'],strict=True);norm.load_state_dict(raw['norm'],strict=True)
                restored=builder.state_equal(stack.state_dict(),raw['stack']) and builder.state_equal(norm.state_dict(),raw['norm'])
                if not restored or not local_probe['passed']:raise ValueError('Discarded entrance update/reset failed')
                del raw
                trainer=_resident(stack,norm,E,rotary,data,p,settings,guard,targets)
                probe=trainer.master_update_probe(longest)
                result=dict(status='completed',passed=probe['passed'],discarded=True,entrance_probe=local_probe,restored_raw_exact=restored,
                    raw_candidate=raw_candidate,probe=probe,inheritance=inheritance,fixed_state=before,
                    geometry_readout_unchanged=old.fixed_state(stack,E,rotary)==before,alpha=ALPHA,normalization_energy=targets.energy)
                if not result['passed'] or not result['geometry_readout_unchanged']:raise ValueError('Resident systems pilot failed')
                base.write(out/'resident_optimizer.json',dict(memory=trainer.optimizer.memory_stats(),transfer_accounting=source.replay.TRANSFER_ACCOUNTING))
            elif phase in ('prealign','train'):
                pilot=previous.get('pilot') or checked(root,'pilot',p)
                raw=_consume(pilot['raw_candidate']['checkpoint'],pilot['raw_candidate']['checkpoint_sha256'])
                stack,norm,E,rotary,_=_build(p,raw,device,True)
                capture.repair.assert_fixed(stack,E,rotary,pilot['fixed_state'])
                if phase=='prealign':
                    pairs=entrance.EntrancePairs(_items(p),Path(p['boundary_root'])/'boundary_21.npy',Path(p['cache_root'])/'native/boundary_22.npy')
                    local=entrance.EntranceTrainer(stack,norm,E,rotary,pairs,config=ENTRANCE_CONFIG,guard=guard)
                    result=local.fit(out,event=event)
                else:
                    prealign=previous['prealign'];delta=_consume(root/'prealign/selected_entrance.pt',prealign['selected_checkpoint_sha256'])
                    if delta.get('layer_idx')!=21:raise ValueError('Wrong entrance delta layer')
                    if set(delta['block'])!=set(stack.blocks[0].state_dict()):raise ValueError('Entrance delta structure changed')
                    for name,value in stack.blocks[0].named_buffers():
                        if not torch.equal(delta['block'][name].cpu(),value.cpu()):raise ValueError('Entrance delta changed fixed geometry')
                    stack.blocks[0].load_state_dict(delta['block'],strict=True);capture.repair.assert_fixed(stack,E,rotary,pilot['fixed_state'])
                    del raw,delta;data=_dataset(p);targets,_=feature.load_targets(root/'parentfeatures',_items(p),p['parent_checkpoint_sha256'])
                    trainer=_resident(stack,norm,E,rotary,data,p,settings,guard,targets)
                    result=fit_step(trainer,out,p,pilot['raw_candidate'],event=event)
                    base.write(out/'resident_optimizer.json',dict(memory=trainer.optimizer.memory_stats(),transfer_accounting=source.replay.TRANSFER_ACCOUNTING))
            else:
                prior=previous['train'];out.mkdir();data=_dataset(p);checks={}
                for c in (prior['candidate'],prior['diagnostic_best']):
                    digest=c['checkpoint_sha256']
                    if digest in checks:continue
                    state=_consume(c['checkpoint'],digest);stack,norm,E,_,inheritance=_build(p,state,'cpu',False);del state
                    rotary=capture.repair.restore_rotary(p,device,guard=guard)
                    fixed=capture.repair.assert_fixed(stack,E,rotary,prior['fixed_state'])
                    packed=builder.packed_clone(stack,device='cpu');capture.repair.assert_fixed(stack,E,rotary,fixed);del stack
                    metrics,rows=trial.score_packed(packed.to(device),norm.to(device),E.to(device),rotary,data,p['label_ids'],guard)
                    reference=_validate_candidate(c,p)
                    if metrics!=reference['metrics'] or rows!=reference['records']:raise ValueError('Fresh packed metrics/records differ')
                    checks[digest]=dict(checkpoint_sha256=digest,strict_state_load=True,fixed_state_unchanged=True,
                        fixed_state=fixed,metrics=metrics,records=rows,inheritance=inheritance)
                    del packed,norm,E,rotary;gc.collect();torch.cuda.empty_cache()
                eligible=prior['development_passed'];winner=prior['candidate']
                result=dict(status='completed',passed=True,checks=checks,promoted=False,confirmed=False,
                    review_checkpoint=winner['checkpoint'] if eligible else p['parent_checkpoint'],
                    review_checkpoint_sha256=winner['checkpoint_sha256'] if eligible else p['parent_checkpoint_sha256'],
                    compressed_blocks=15 if eligible else 14,development_passed=eligible,parent_admission=p['parent_admission'],
                    fallback_parent_checkpoint_sha256=p['parent_checkpoint_sha256'],speed_measured=False,full36=False)
        guard();peaks={role:torch.cuda.max_memory_reserved(v['device']) for role,v in settings.items()} if dual else torch.cuda.max_memory_reserved()
        _finish(out,p,phase,result,dict(lease_id=lease['id'],allocation=lease['allocation'],allocator=settings,
            peak_reserved_bytes=peaks,elapsed_seconds=time.monotonic()-started),dependencies)
        checked(root,phase,p);return result
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        base.write(out/'failure.json',dict(status='failed',phase=phase,fingerprint=p['fingerprint'],error=f'{type(error).__name__}: {error}',
            lease=lease,allocator=settings,peak_reserved_bytes={str(v['device']):torch.cuda.max_memory_reserved(v['device']) for v in limits} if cuda else None));raise
    finally:
        if data is not None:data.close()
        if pairs is not None:pairs.close()
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare',*PHASES,'check'))
    parser.add_argument('--root',required=True);parser.add_argument('--recovery-root');parser.add_argument('--assessment-root');parser.add_argument('--check-phase',choices=PHASES)
    parser.add_argument('--exploratory-parent',action='store_true',help='Explicitly admit a completed failed parent audit for an unconfirmed experiment')
    args=parser.parse_args()
    if args.phase=='prepare':
        if not args.recovery_root or not args.assessment_root:parser.error('prepare requires recovery-root and assessment-root')
        value=prepare(args.root,args.recovery_root,args.assessment_root,exploratory_parent=args.exploratory_parent)
    elif args.phase=='check':
        if not args.check_phase:parser.error('check requires check-phase')
        value=checked(args.root,args.check_phase)
    else:value=run(args.root,args.phase)
    print(json.dumps(value,indent=2),flush=True)

if __name__=='__main__':main()
