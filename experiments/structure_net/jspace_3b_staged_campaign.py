"""Staged entrance alignment and cautious joint recovery from accepted nine blocks.

Old campaigns remain immutable. Each new entrance is aligned before joint KD;
the raw initializer remains an eligible protected checkpoint. All GPU phases
run in separate reservation-owned processes. No held-out or speed claim.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch
from torch.nn import functional as F

from . import jspace_qwen3b_progressive as prior
from . import jspace_3b_entrance_distill as entrance
from . import jspace_3b_staged_train as training

old,native,core,model,calibration,targets=prior.old,prior.native,prior.core,prior.model,prior.calibration,prior.targets
base,read=prior.base,prior.read
records,check_plan,data_for,make_stack,cap_allocator=prior.records,prior.check_plan,prior.data_for,prior.make_stack,prior.cap_allocator
CONFIG=training.CONFIG
ENTRANCE_CONFIG=entrance.EntranceConfig()
SCHEMA='jspace.3b-staged-campaign.v1'
STUDY='JSPACE-3B-STAGED-CAMPAIGN-01'
MODULE='experiments.structure_net.jspace_3b_staged_campaign'
PROTOCOL=base.ROOT/'docs/07 - Status Reports/jspace/2026-11-15_qwen3b-staged-recovery.md'
SEED_SHA='5c341443b3a288e55709b0771e13d517d1d2e286fbf6702b719e57b5e6e2882a'
PHASES=('capture','fit','pilot','prealign','train','verify')
SEAM_POLICY='native_singleton_unpadded_then_parent_batch4_exact_boundary'


def stage_dir(root,cut):
    if type(cut) is not int or not 0<=cut<=26:raise ValueError('New cut must be an integer in [0,26]')
    return Path(root)/'stages'/f'block_{cut:02d}'


def local_import_closure(paths):
    """Pin local relative imports, including helpers imported inside functions.

    This traverses explicit imports, never a changing source glob or sys.modules.
    Prior authenticated sources retain their original digests independently.
    """
    pending=[Path(p).resolve() for p in paths];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        if not path.is_file() or not path.is_relative_to(base.ROOT):raise ValueError('Local producer missing: '+str(path))
        found.add(path)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node,ast.ImportFrom) or not node.level:continue
            parent=path.parent
            for _ in range(node.level-1):parent=parent.parent
            names=[node.module] if node.module else [a.name for a in node.names]
            for name in names:
                candidate=parent.joinpath(*name.split('.')).with_suffix('.py')
                if candidate.is_file():pending.append(candidate.resolve())
    return found


def _pin_phase(inputs,directory,phase):
    path=directory/phase;manifest=read(path/'manifest.json')
    old.pin(inputs,path/'manifest.json')
    for name,digest in manifest['artifacts'].items():
        target=(path/name).resolve()
        if not target.is_relative_to(path.resolve()):raise ValueError('Prior artifact escapes its phase')
        inputs[str(target)]=digest
    return dict(directory=str(path),source_fingerprint=manifest['fingerprint'],
        manifest_sha256=base.sha(path/'manifest.json'),artifacts=manifest['artifacts'])


def prepare(root,seed_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation must hide CUDA')
    root,seed_root=Path(root).resolve(),Path(seed_root).resolve()
    if root.exists():raise FileExistsError(root)
    seed=prior.verify_root(seed_root)
    if seed.get('seed_cut')!=29 or seed.get('native_correct')!=48:raise ValueError('Expected frozen v3 seed campaign')
    inputs=dict(seed['input_sha256']);chain=[]
    old.pin(inputs,seed_root/'plan.json')
    for source,digest in seed['source_sha256'].items():
        inputs[str(seed_root/'producer'/Path(source).relative_to(base.ROOT))]=digest
    for cut in (28,27):
        _,stage=prior.verify_stage(seed_root,cut);directory=prior.stage_dir(seed_root,cut)
        evidence={phase:prior.checked(directory,phase,stage) for phase in prior.PHASES}
        if not evidence['train']['quality_pass'] or not evidence['verify']['passed']:
            raise ValueError('Seed chain must be freshly verified and accepted')
        inputs.update(stage['input_sha256']);old.pin(inputs,directory/'plan.json')
        for phase in prior.PHASES:_pin_phase(inputs,directory,phase)
        chain.append(dict(cut=cut,fingerprint=stage['fingerprint'],selected_checkpoint_sha256=evidence['train']['selected_checkpoint_sha256']))
    trained,verified=evidence['train'],evidence['verify']
    if (trained['selected_checkpoint_sha256']!=SEED_SHA or trained['selected']['correct']!=52
            or verified['selected_checkpoint_sha256']!=SEED_SHA or sorted(map(int,stage['fits']))!=list(range(27,36))):
        raise ValueError('The named accepted nine-block checkpoint is required')
    seed_stage=stage;seed_directory=directory
    _,reuse_stage=prior.verify_stage(seed_root,26);reuse_directory=prior.stage_dir(seed_root,26)
    if (reuse_stage['parent_checkpoint_sha256']!=SEED_SHA or reuse_stage['parent_correct']!=52
            or reuse_stage['teacher_fingerprint']!=base.digest(SEED_SHA) or reuse_stage['calibration_seed']!=320861):
        raise ValueError('Reused calibration must have the exact same nine-block parent and seed')
    old.pin(inputs,reuse_directory/'plan.json');reuse={}
    for phase in ('capture','fit'):
        prior.checked(reuse_directory,phase,reuse_stage)
        reuse[phase]=_pin_phase(inputs,reuse_directory,phase)
        reuse[phase]['source_root']=str(seed_root)
        reuse[phase]['campaign_fingerprint']=seed['fingerprint']
    for fit in seed_stage['fits'].values():
        for name in ('fit.pt','fit.json','relation.pt','relations.json','interface.json'):old.pin(inputs,Path(fit)/name)
    sources=dict(seed['source_sha256'])
    extras=[Path(__file__),Path(entrance.__file__),Path(training.__file__),PROTOCOL,
        base.ROOT/'tests/structure_net/test_jspace_3b_staged_campaign.py',
        base.ROOT/'tests/structure_net/test_jspace_3b_entrance_distill.py',
        base.ROOT/'tests/structure_net/test_jspace_3b_staged_train.py']
    for path in local_import_closure(extras):
        digest=base.sha(path)
        if str(path) in sources and sources[str(path)]!=digest:raise ValueError('Frozen inherited producer changed')
        sources[str(path)]=digest
    volume=root.parent
    while not volume.exists():volume=volume.parent
    if shutil.disk_usage(volume).free<425*1024**3:raise OSError('At least425GiB free required for preserved full continuation')
    plan=dict(schema=SCHEMA,study_id=STUDY,seed_root=str(seed_root),seed_fingerprint=seed['fingerprint'],
        seed_cut=27,seed_correct=52,target_cut=0,native_correct=48,
        seed_checkpoint=str(seed_directory/'train/selected.pt'),seed_checkpoint_sha256=SEED_SHA,
        seed_records=str(seed_directory/'train'/f"epoch_{trained['selected_epoch']:02d}_development.json"),
        seed_chain=chain,calibration_reuse=reuse,model=seed['model'],fits=seed_stage['fits'],
        native_cache=seed['native_cache'],items_path=seed['items_path'],original_indices=seed['original_indices'],
        label_ids=seed['label_ids'],calibration_indices=seed['calibration_indices'],
        config=asdict(CONFIG),entrance_config=asdict(ENTRANCE_CONFIG),selection_policy=training.SELECTION,
        eligibility_tolerance=training.TOLERANCE,quality_policy='original_native_only',seam_policy=SEAM_POLICY,
        allocation_mib=11264,allocator_mib=11008,input_sha256=inputs,
        input_identity={p:old.file_identity(p) for p in inputs},source_sha256=sources,confirmed=False)
    plan['fingerprint']=base.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        if base.sha(path)!=digest:raise ValueError('Source changed during preparation')
        destination=root/'producer'/Path(path).relative_to(base.ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        if base.sha(destination)!=digest:raise ValueError('Archived producer mismatch')
    base.write(root/'plan.json',plan);base.event(root/'progress.jsonl','prepared',fingerprint=plan['fingerprint'],compressed_blocks=9)
    return dict(status='prepared',fingerprint=plan['fingerprint'])


def verify_root(root):
    root=Path(root);plan=read(root/'plan.json');check_plan(plan)
    if (plan.get('schema')!=SCHEMA or plan.get('seed_checkpoint_sha256')!=SEED_SHA or plan.get('seed_cut')!=27
            or plan.get('seed_correct')!=52 or plan.get('native_correct')!=48 or plan['config']!=asdict(CONFIG)
            or plan.get('entrance_config')!=asdict(ENTRANCE_CONFIG) or plan.get('seam_policy')!=SEAM_POLICY
            or plan.get('selection_policy')!=training.SELECTION or plan.get('eligibility_tolerance')!=training.TOLERANCE
            or plan.get('quality_policy')!='original_native_only'):
        raise ValueError('Staged campaign protocol changed')
    for path,digest in plan['source_sha256'].items():
        if base.sha(path)!=digest or base.sha(root/'producer'/Path(path).relative_to(base.ROOT))!=digest:
            raise ValueError('Frozen producer changed: '+path)
    for path,identity in plan['input_identity'].items():
        if old.file_identity(path)!=identity:raise ValueError('Pinned original input changed: '+path)
    return plan


def prepare_stage(root,cut):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Stage preparation must hide CUDA')
    root=Path(root).resolve();plan=verify_root(root);directory=stage_dir(root,cut)
    if directory.exists():raise FileExistsError(directory)
    fits=dict(plan['fits']);inputs={};reuse=None
    if cut==26:
        checkpoint=plan['seed_checkpoint'];reference=plan['seed_records'];parent_correct=52
        parent_fingerprint=plan['seed_fingerprint'];reuse=plan['calibration_reuse']
    else:
        parent_dir=stage_dir(root,cut+1);parent=read(parent_dir/'plan.json');check_plan(parent)
        if parent.get('root_fingerprint')!=plan['fingerprint'] or parent.get('cut')!=cut+1:
            raise ValueError('Parent must be immediately preceding stage')
        result=checked(parent_dir,'train',parent);verification=checked(parent_dir,'verify',parent)
        if not result['quality_pass'] or not verification['passed']:raise ValueError('Cannot expand an unaccepted parent')
        fits=parent['fits'];checkpoint=str(parent_dir/'train/selected.pt')
        reference=str(parent_dir/'train'/f"epoch_{result['selected_epoch']:02d}_development.json")
        parent_correct=result['selected']['correct'];parent_fingerprint=parent['fingerprint']
        for path in (parent_dir/'plan.json',parent_dir/'train/manifest.json',parent_dir/'verify/manifest.json'):old.pin(inputs,path)
    old.pin(inputs,checkpoint);old.pin(inputs,reference)
    for fit in fits.values():
        for name in ('fit.pt','fit.json','relation.pt','relations.json','interface.json'):old.pin(inputs,Path(fit)/name)
    fits=dict(fits);fits[str(cut)]=str(directory/'fit'/f'block_{cut:02d}')
    stage=dict(schema=SCHEMA,root_fingerprint=plan['fingerprint'],cut=cut,parent_cut=cut+1,layers=list(range(cut,36)),
        parent_checkpoint=checkpoint,parent_checkpoint_sha256=base.sha(checkpoint),parent_records=reference,
        parent_correct=parent_correct,parent_fingerprint=parent_fingerprint,native_correct=48,
        fits=fits,input_sha256=inputs,input_identity={p:old.file_identity(p) for p in inputs},
        teacher_fingerprint=base.digest(base.sha(checkpoint)),calibration_seed=20861+100000*(29-cut),
        calibration_reuse=reuse,quality_policy='original_native_only')
    stage['fingerprint']=base.digest(stage);directory.mkdir(parents=True);base.write(directory/'plan.json',stage)
    base.event(root/'progress.jsonl','stage_prepared',cut=cut,parent_correct=parent_correct)
    return stage


def verify_stage(root,cut):
    plan=verify_root(root);stage=read(stage_dir(root,cut)/'plan.json');check_plan(stage)
    if stage.get('schema')!=SCHEMA or stage['root_fingerprint']!=plan['fingerprint'] or stage['cut']!=cut:
        raise ValueError('Stage/root mismatch')
    for path,identity in stage['input_identity'].items():
        if old.file_identity(path)!=identity:raise ValueError('Stage parent input changed: '+path)
    if base.sha(stage['parent_checkpoint'])!=stage['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
    return plan,stage


def copy_reused_phase(reuse,output):
    """Copy only authenticated bytes; retain original receipts under distinct names."""
    source=Path(reuse['directory']);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if base.sha(source/'manifest.json')!=reuse['manifest_sha256']:raise ValueError('Reuse manifest changed')
    manifest=read(source/'manifest.json')
    if (manifest.get('status')!='completed' or manifest.get('fingerprint')!=reuse['source_fingerprint']
            or manifest.get('artifacts')!=reuse['artifacts']):raise ValueError('Invalid completed reuse source')
    output.mkdir(parents=True)
    for name,digest in reuse['artifacts'].items():
        original=source/name
        if not original.resolve().is_relative_to(source.resolve()) or base.sha(original)!=digest:
            raise ValueError('Reused artifact changed or escaped source: '+name)
        target=output/('original_result.json' if name=='result.json' else name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
        if base.sha(target)!=digest or os.stat(original).st_ino==os.stat(target).st_ino:
            raise ValueError('Reused artifact is not an independent byte-identical copy')
    shutil.copy2(source/'manifest.json',output/'original_manifest.json')
    base.write(output/'reuse.json',reuse)
    return read(output/'original_result.json')


def validate_quality(report):
    rows,metrics=report.get('records',[]),report.get('metrics',{})
    if len(rows)!=64 or len({r['identity'] for r in rows})!=64 or metrics.get('count')!=64:
        raise ValueError('Complete distinct64 development records required')
    for row in rows:
        values=row['logits']
        if (len(values) not in (2,4) or not all(math.isfinite(v) for v in values)
                or type(row['answer']) is not int or not 0<=row['answer']<len(values)
                or row['prediction']!=max(range(len(values)),key=values.__getitem__)):
            raise ValueError('Invalid choice logits, answer or prediction')
    correct=sum(r['prediction']==r['answer'] for r in rows)
    if metrics.get('correct')!=correct or metrics.get('accuracy')!=correct/64:
        raise ValueError('Development count disagrees with actual answers')


def _epoch_checks(path,history,artifacts,*,quality=False):
    for row in history:
        filename=f"epoch_{row['epoch']:02d}_development.json"
        if artifacts.get(filename)!=row.get('development_sha256'):
            raise ValueError('Epoch development receipt missing from manifest')
        if read(path/filename).get('metrics')!=row.get('metrics'):raise ValueError('Epoch development metrics changed')
        if quality:validate_quality(read(path/filename))


def validate_delta(raw,delta,pilot,cut):
    """Only original entrance parameters can change; all geometry stays exact."""
    if set(delta)!={'block','layer_idx','normalization','config'} or delta['layer_idx']!=cut:
        raise ValueError('Prealignment must export only the selected entrance delta')
    original={k[len('blocks.0.'):]:v for k,v in raw['stack'].items() if k.startswith('blocks.0.')}
    if set(delta['block'])!=set(original):raise ValueError('Delta entrance state keys differ from raw initializer')
    parameters=set(pilot['entrance_parameter_names']);buffers=set(pilot['entrance_buffer_names'])
    if parameters & buffers or not parameters:raise ValueError('Invalid entrance parameter/buffer inventory')
    for key,value in original.items():
        proposed=delta['block'][key]
        if isinstance(value,torch.Tensor):
            if not isinstance(proposed,torch.Tensor) or value.shape!=proposed.shape or value.dtype!=proposed.dtype or not torch.isfinite(proposed).all():
                raise ValueError('Delta tensor shape, dtype or finiteness changed: '+key)
            if key not in parameters and not torch.equal(value,proposed):raise ValueError('Prealignment changed frozen entrance geometry: '+key)
        elif value!=proposed:raise ValueError('Prealignment changed entrance extra state')
    if not (parameters|buffers) <=set(original):raise ValueError('Inventory contains unknown entrance keys')
    return dict(passed=True,layer_idx=cut,only_entrance_parameters=True,fixed_geometry_exact=True,
        parameter_count=len(parameters),buffer_count=len(buffers))


def checked(directory,phase,stage):
    if phase not in PHASES:raise ValueError('Unknown phase')
    path=Path(directory)/phase;manifest=read(path/'manifest.json')
    if manifest.get('status')!='completed' or manifest.get('fingerprint')!=stage['fingerprint']:
        raise ValueError('Completed matching phase required: '+phase)
    artifacts=manifest.get('artifacts',{})
    if 'result.json' not in artifacts:raise ValueError('Authenticated phase result required')
    for name,digest in artifacts.items():
        target=path/name
        if not target.resolve().is_relative_to(path.resolve()) or base.sha(target)!=digest:raise ValueError('Phase artifact changed: '+name)
    result=read(path/'result.json');layers=list(range(stage['cut'],36))
    if result.get('status')!='completed':raise ValueError('Phase execution incomplete')
    if phase in ('capture','fit'):
        reuse=(stage.get('calibration_reuse') or {}).get(phase)
        if reuse:
            if manifest.get('reused_from')!=reuse or read(path/'reuse.json')!=reuse:
                raise ValueError('Copied calibration provenance changed')
            if artifacts.get('original_manifest.json')!=reuse['manifest_sha256']:
                raise ValueError('Original calibration manifest not retained')
            for name,digest in reuse['artifacts'].items():
                copied='original_result.json' if name=='result.json' else name
                if artifacts.get(copied)!=digest:raise ValueError('Reused calibration bytes changed')
    if phase=='capture':
        controls=result.get('controls',{})
        if (not controls.get('passed') or not controls.get('boundary',{}).get('exact')
                or controls.get('seam_policy')!=SEAM_POLICY
                or not result.get('teacher_cache_diagnostic',{}).get('passed')
                or result.get('teacher_fingerprint')!=stage['teacher_fingerprint']):
            raise ValueError('Capture boundary/parent/teacher validity failed')
    elif phase=='fit':
        if result.get('cut')!=stage['cut'] or result.get('teacher_fingerprint')!=stage['teacher_fingerprint']:
            raise ValueError('Fit teacher/cut provenance changed')
    elif phase=='pilot':
        local=result.get('entrance_probe',{});changes=local.get('fp32_master_changed_by_parameter',{})
        if (not result.get('passed') or not result.get('systems_only') or not result.get('discarded')
                or result.get('layer_indices')!=layers or result.get('update',{}).get('update')!=1
                or result.get('fp32_master_changed_by_block')!=[True]*len(layers)
                or not result.get('norm_fp32_master_changed') or not result.get('fixed_state_unchanged')
                or not result.get('checkpoint_strict_load') or not result.get('inheritance',{}).get('passed')
                or not local.get('passed') or not local.get('discarded') or not local.get('systems_only')
                or not local.get('fixed_state_unchanged') or local.get('update',{}).get('update')!=1
                or set(changes)!=set(result.get('entrance_parameter_names',[])) or not changes or not all(changes.values())
                or not result.get('entrance_restore_exact')
                or artifacts.get('initializer.pt')!=result.get('initializer_sha256')
                or artifacts.get('initial_development.json')!=result.get('initial_development_sha256')):
            raise ValueError('Discarded entrance/joint pilot validity failed')
        initial=read(path/'initial_development.json');validate_quality(initial)
        if initial!={'metrics':result.get('initial'),'records':result.get('initial_records')}:
            raise ValueError('Protected initializer development reference changed')
    elif phase=='prealign':
        history=result.get('history',[])
        if (result.get('config')!=asdict(ENTRANCE_CONFIG) or result.get('completed_epochs')!=12 or result.get('updates')!=384
                or result.get('layer_idx')!=stage['cut'] or result.get('inherited_layers')!=layers[1:]
                or [r.get('epoch') for r in history]!=list(range(13)) or not result.get('strict_reload_exact')
                or not result.get('fixed_state_unchanged') or result.get('task_quality_guaranteed') is not False):
            raise ValueError('Entrance-only prealignment budget or frozen-parent proof failed')
        _epoch_checks(path,history,artifacts)
        updates=[u for row in history for u in row.get('training',[])]
        if ([u.get('update') for u in updates]!=list(range(1,385)) or any(not u.get('only_entrance_gradients') for u in updates)
                or any(len(r.get('training',[]))!=(0 if r['epoch']==0 else 32) for r in history)
                or any(not math.isfinite(r['metrics']['loss']) for r in history)):
            raise ValueError('Entrance-only gradient participation or finite feature history failed')
        winner=min(history,key=lambda row:row['metrics']['loss'])
        if (result.get('selected_epoch')!=winner['epoch'] or result.get('selected')!=winner['metrics']
                or result.get('initial')!=history[0]['metrics']
                or winner.get('checkpoint_sha256')!=result.get('selected_checkpoint_sha256')
                or artifacts.get('selected_entrance.pt')!=result.get('selected_checkpoint_sha256')):
            raise ValueError('Feature-selected entrance provenance changed')
        pilot=checked(directory,'pilot',stage)
        delta=native._load(path/'selected_entrance.pt');raw=native._load(Path(directory)/'pilot/initializer.pt')
        proof=validate_delta(raw,delta,pilot,stage['cut'])
        if delta['normalization']!=result['normalization'] or delta['config']!=result['config'] or read(path/'delta_validation.json')!=proof:
            raise ValueError('Entrance export normalization/configuration changed')
    elif phase=='train':
        history=result.get('history',[]);pilot=checked(directory,'pilot',stage)
        if (result.get('config')!=asdict(CONFIG) or result.get('completed_epochs')!=12 or result.get('updates')!=384
                or result.get('layer_indices')!=layers or [r.get('epoch') for r in history]!=list(range(-1,13))
                or not result.get('strict_reload_exact') or not result.get('fixed_state_unchanged')
                or result.get('protected_epoch')!=-1 or result.get('initial')!=pilot['initial']
                or result.get('warm_initial')!=history[1]['metrics'] or result.get('selection_policy')!=training.SELECTION
                or result.get('eligibility_tolerance')!=training.TOLERANCE):
            raise ValueError('Cautious training budget, protected reference or fixed state changed')
        _epoch_checks(path,history,artifacts,quality=True)
        updates=[u for row in history for u in row.get('training',[])]
        if ([u.get('update') for u in updates]!=list(range(1,385))
                or any(u.get('gradients_present_by_block')!=[True]*len(layers) for u in updates)
                or any(len(r.get('training',[]))!=(0 if r['epoch']<=0 else 32) for r in history)
                or any(r['metrics'].get('count')!=64 for r in history)):
            raise ValueError('Joint updates, counts or all-block gradient participation incomplete')
        protected=read(path/'epoch_-1_development.json')
        if protected!={'metrics':pilot['initial'],'records':pilot['initial_records']}:
            raise ValueError('Protected raw candidate changed')
        replay=result.get('protected_replay',{})
        if (read(path/'protected_replay.json')!=replay or not replay.get('protected_replay_exact')
                or not replay.get('warm_restore_exact') or not replay.get('optimizer_untouched')
                or replay.get('warm_state_fingerprint_before')!=replay.get('warm_state_fingerprint_after')):
            raise ValueError('Protected raw replay/warm restoration invalid')
        protected_reference=result.get('protected_reference',{})
        if protected_reference!=dict(metrics=pilot['initial'],development_sha256=history[0]['development_sha256'],checkpoint_sha256=history[0].get('checkpoint_sha256')):
            raise ValueError('Protected candidate provenance changed')
        winner=training.best_eligible(history,pilot['initial'])
        if (winner['epoch']!=result.get('selected_epoch') or winner['metrics']!=result.get('selected')
                or winner.get('checkpoint_sha256')!=result.get('selected_checkpoint_sha256')
                or artifacts.get('selected.pt')!=result.get('selected_checkpoint_sha256')):
            raise ValueError('Eligible accuracy-first checkpoint selection changed')
        quality=training.inherited.gates(result['selected']['correct'],native_correct=48,parent_correct=stage['parent_correct'],count=64)
        if result.get('quality')!=quality or result.get('quality_pass')!=quality['passed']:
            raise ValueError('Original-native quality budget changed')
    elif phase=='verify':
        trained=checked(directory,'train',stage)
        if (not result.get('passed') or not result.get('comparison',{}).get('passed')
                or not result.get('fixed_state_unchanged') or result.get('metrics')!=trained['selected']
                or result.get('selected_checkpoint_sha256')!=trained['selected_checkpoint_sha256']
                or result.get('quality')!=trained['quality'] or result.get('compressed_blocks')!=len(layers)):
            raise ValueError('Fresh verification, selected state or quality provenance failed')
    return result


@torch.inference_mode()
def seam_rows(block,parent,norm,E,rotary,data,next_data,label_ids,guard):
    """Reproduce singleton native capture, then preserve parent batch4 arithmetic."""
    if data.items!=next_data.items:raise ValueError('Seam boundary identities/order differ')
    rows=[];boundaries=[]
    for start in range(0,len(data.development),4):
        guard();indices=data.development[start:start+4]
        batch=data.batch(indices,device=E.device);cached=next_data.batch(indices,device=E.device)
        restored=torch.zeros_like(batch['hidden'])
        for row,length in enumerate(batch['lengths']):
            guard();hidden=batch['hidden'][row:row+1,:length];positions=batch['position_ids'][row:row+1,:length]
            value=block(hidden,attention_mask=batch['attention_mask'][row:row+1,:,:length,:length],
                position_ids=positions,cache_position=positions[0],position_embeddings=rotary(hidden,positions),use_cache=False)
            value=value[0] if isinstance(value,tuple) else value
            restored[row,:length]=value[0]
            expected=cached['hidden'][row,:length]
            boundaries.append(dict(identity=data.items[indices[row]]['identity'],exact=torch.equal(value[0],expected),
                max_abs=float((value[0].float()-expected.float()).abs().max())))
        h=parent(restored,attention_mask=batch['attention_mask'],position_ids=batch['position_ids'],
            cache_position=batch['position_ids'][0],position_embeddings=rotary(restored,batch['position_ids']),use_cache=False)[0]
        h=h[torch.arange(len(indices),device=E.device)[:,None],batch['loss_positions']]
        logits=F.linear(norm(h),E).float()
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite seam logits')
        for row,index in enumerate(indices):
            case=data.items[index];values=logits[row,-1,label_ids[:len(case['choices'])]]
            rows.append(dict(identity=case['identity'],family=case['family'],answer=case['answer'],
                prediction=int(values.argmax()),logits=values.cpu().tolist()))
    return rows,dict(exact=all(r['exact'] for r in boundaries),max_abs=max(r['max_abs'] for r in boundaries),cases=boundaries)


def controls(plan,stage,stack,norm,E,rotary,guard):
    cut=stage['cut'];data=data_for(plan,cut);next_data=data_for(plan,cut+1)
    try:
        cases=[data.items[i] for i in data.development];reference=records(stage['parent_records'])
        parent=old.score(stack,norm,E,rotary,next_data,plan['label_ids'],guard)
        parent_check=old.compare_rows(parent,reference,cases,expected_correct=stage['parent_correct'])
        block=native.load_native_block(plan['model']['path'],cut,device=E.device).eval().requires_grad_(False)
        seam,boundary=seam_rows(block,stack,norm,E,rotary,data,next_data,plan['label_ids'],guard)
        del block
        seam_check=old.compare_rows(seam,reference,cases,expected_correct=stage['parent_correct'])
        result=dict(parent=dict(**parent_check,records=parent),seam=dict(**seam_check,records=seam),boundary=boundary,
            seam_policy=SEAM_POLICY,passed=parent_check['passed'] and seam_check['passed'] and boundary['exact'])
        if cut==26:
            _,norm_native,_=native.load_readout_norm(plan['model']['path'],device=E.device)
            tail=old.NativeTail([native.load_native_block(plan['model']['path'],i,device=E.device) for i in range(cut,36)]).eval().requires_grad_(False)
            rows=old.score(tail,norm_native,E,rotary,data,plan['label_ids'],guard,batch_size=1)
            result['native']=dict(**old.compare_rows(rows,old.rows_from_cache(data,plan['label_ids']),cases,expected_correct=48),records=rows)
            result['passed'] &= result['native']['passed']
        return result
    finally:data.close();next_data.close()


def entrance_pairs(plan,cut):
    cache=Path(plan['native_cache'])
    return entrance.EntrancePairs(read(plan['items_path'])['items'],cache/f'boundary_{cut:02d}.npy',cache/f'boundary_{cut+1:02d}.npy')


def _trainer(stack,norm,E,rotary,data,plan,stage,guard):
    return training.CautiousTrainer(stack,norm,E,rotary,data,plan['label_ids'],config=CONFIG,guard=guard,
        native_correct=48,parent_correct=stage['parent_correct'])


def run_phase(root,cut,phase):
    from .jspace_full_lease import wait_ready
    root=Path(root).resolve();plan,stage=verify_stage(root,cut);directory=stage_dir(root,cut);output=directory/phase
    if phase not in PHASES:raise ValueError('Unknown GPU phase')
    if output.exists():raise FileExistsError(output)
    for previous in PHASES[:PHASES.index(phase)]:checked(directory,previous,stage)
    lease=None;data=pairs=teacher=None;started=time.monotonic();cuda_started=False
    try:
        lease=wait_ready([0]);allocator=cap_allocator(lease,plan);torch.cuda.init();cuda_started=True
        torch.cuda.reset_peak_memory_stats();torch.set_num_threads(4);torch.manual_seed(CONFIG.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        def guard():
            current=wait_ready([0])
            if current['id']!=lease['id'] or current['allocation']!=lease['allocation']:raise RuntimeError('Reservation changed')
            if torch.cuda.max_memory_reserved()>allocator['allocator_limit_bytes']:raise MemoryError('Reservation cap exceeded')
        def event(kind,**values):base.event(root/'progress.jsonl',kind,cut=cut,phase=phase,**values)
        def progress(value):
            values=dict(value);kind=values.pop('kind','calibration_progress');values.pop('cut',None);values.pop('phase',None)
            event(kind,**values)
        reuse=(stage.get('calibration_reuse') or {}).get(phase)
        if phase=='capture':
            stack,norm,E,rotary,inheritance=make_stack(plan,stage,'cpu',parent=True)
            stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).to('cuda:0')
            norm.to('cuda:0');E=E.to('cuda:0');rotary.to('cuda:0')
            control=controls(plan,stage,stack,norm,E,rotary,guard)
            if not control['passed']:raise RuntimeError('Parent/native singleton seam replay failed: '+json.dumps(control))
            if reuse:
                original_capture=copy_reused_phase(reuse,output)
                target_receipt=original_capture.get('packed_target_receipt')
            else:
                block=native.load_native_block(plan['model']['path'],cut,device='cuda:0')
                teacher=calibration.CachedPrefixTeacher(read(plan['items_path'])['items'],
                    Path(plan['native_cache'])/f'boundary_{cut:02d}.npy',cut,block,stack,norm,E,rotary,guard=guard)
                calibration.capture(output,teacher,plan['calibration_indices'],teacher_fingerprint=stage['teacher_fingerprint'],progress=progress)
                data=data_for(plan,cut+1)
                target_receipt=targets.capture_packed_teacher(output/'packed_teacher_logits.npy',stack,norm,E,rotary,data,guard=guard,progress=progress)
                data.close();data=None
            data=data_for(plan,cut,output/'accepted_teacher_logits.npy')
            differentiable=old.compare_rows(old.rows_from_cache(data,plan['label_ids'],'accepted'),records(stage['parent_records']),
                [data.items[i] for i in data.development]);data.close()
            data=data_for(plan,cut,output/'packed_teacher_logits.npy')
            diagnostic=old.compare_rows(old.rows_from_cache(data,plan['label_ids'],'accepted'),records(stage['parent_records']),
                [data.items[i] for i in data.development],expected_correct=stage['parent_correct'])
            if not diagnostic['passed']:raise RuntimeError('Packed teacher targets changed accepted parent')
            result=dict(status='completed',controls=control,inheritance=inheritance,teacher_cache_diagnostic=diagnostic,
                differentiable_teacher_cache_diagnostic=differentiable,teacher_cache_logit_gate_applied=True,
                teacher_fingerprint=stage['teacher_fingerprint'],calibration_reused=bool(reuse),packed_target_receipt=target_receipt,
                teacher_target_path=str(output/'packed_teacher_logits.npy'),teacher_target_sha256=base.sha(output/'packed_teacher_logits.npy'))
        elif phase=='fit':
            if reuse:copy_reused_phase(reuse,output)
            else:
                output.mkdir();calibration.finalize(directory/'capture',output,device='cuda:0',guard=guard,progress=progress)
                block=native.load_native_block(plan['model']['path'],cut,device='cuda:0')
                _,_,rotary=native.load_readout_norm(plan['model']['path'],device='cuda:0')
                calibration.capture_interface(output/f'block_{cut:02d}',directory/'capture',cut,block,rotary,device='cuda:0',guard=guard,progress=progress)
                calibration.fit_relation(output/f'block_{cut:02d}',device='cuda:0',guard=guard,progress=progress)
            result=dict(status='completed',cut=cut,teacher_fingerprint=stage['teacher_fingerprint'],calibration_reused=bool(reuse))
        else:
            stack,norm,E,rotary,inheritance=make_stack(plan,stage,'cuda:0')
            data=data_for(plan,cut,directory/'capture/packed_teacher_logits.npy')
            if phase=='pilot':
                output.mkdir();trainer=_trainer(stack,norm,E,rotary,data,plan,stage,guard)
                initial,initial_rows=trainer.packed_development();raw=trainer.snapshot()
                old.save(output/'initializer.pt',raw);base.write(output/'initial_development.json',dict(metrics=initial,records=initial_rows))
                names=[n for n,_ in stack.blocks[0].named_parameters()];buffers=[n for n,_ in stack.blocks[0].named_buffers()]
                del trainer;gc.collect();pairs=entrance_pairs(plan,cut)
                order=sorted(data.training,key=lambda i:data.lengths[i],reverse=True)[:16]
                local=entrance.EntranceTrainer(stack,norm,E,rotary,pairs,config=ENTRANCE_CONFIG,guard=guard)
                local_probe=local.pilot(order);del local;gc.collect()
                stack.load_state_dict(raw['stack'],strict=True);norm.load_state_dict(raw['norm'],strict=True)
                restored=training.state_fingerprint(dict(stack=stack.state_dict(),norm=norm.state_dict()))==training.state_fingerprint(raw)
                if not local_probe['passed'] or not restored:raise RuntimeError('Entrance pilot failed update or raw-state restoration')
                trainer=_trainer(stack,norm,E,rotary,data,plan,stage,guard)
                probe=trainer.master_update_probe(order);old.save(output/'discarded_joint_state.pt',trainer.snapshot())
                if not probe['passed']:raise RuntimeError('Joint pilot failed all-block update/fixed-state proof')
                snapshot=native._load(output/'discarded_joint_state.pt')
                stack.load_state_dict(snapshot['stack'],strict=True);norm.load_state_dict(snapshot['norm'],strict=True)
                result=prior.pilot_result(probe,initial,initial_rows,inheritance)
                result.update(entrance_probe=local_probe,entrance_restore_exact=restored,
                    initializer_sha256=base.sha(output/'initializer.pt'),initial_development_sha256=base.sha(output/'initial_development.json'),
                    entrance_parameter_names=names,entrance_buffer_names=buffers)
            elif phase=='prealign':
                pilot=checked(directory,'pilot',stage);raw=native._load(directory/'pilot/initializer.pt')
                stack.load_state_dict(raw['stack'],strict=True);norm.load_state_dict(raw['norm'],strict=True)
                pairs=entrance_pairs(plan,cut)
                local=entrance.EntranceTrainer(stack,norm,E,rotary,pairs,config=ENTRANCE_CONFIG,guard=guard)
                result=local.fit(output,event=event)
                delta=native._load(output/'selected_entrance.pt');proof=validate_delta(raw,delta,pilot,cut)
                base.write(output/'delta_validation.json',proof)
            elif phase=='train':
                pilot=checked(directory,'pilot',stage);checked(directory,'prealign',stage)
                raw=native._load(directory/'pilot/initializer.pt');delta=native._load(directory/'prealign/selected_entrance.pt')
                validate_delta(raw,delta,pilot,cut)
                stack.load_state_dict(raw['stack'],strict=True);norm.load_state_dict(raw['norm'],strict=True)
                stack.blocks[0].load_state_dict(delta['block'],strict=True)
                trainer=_trainer(stack,norm,E,rotary,data,plan,stage,guard)
                result=trainer.fit(output,protected_state=raw,protected_metrics=pilot['initial'],protected_records=pilot['initial_records'],event=event)
            else:
                output.mkdir();trained=checked(directory,'train',stage)
                before=old.fixed_state(stack,E,rotary);snapshot=native._load(directory/'train/selected.pt')
                stack.load_state_dict(snapshot['stack'],strict=True);norm.load_state_dict(snapshot['norm'],strict=True)
                trainer=_trainer(stack,norm,E,rotary,data,plan,stage,guard);metrics,rows=trainer.packed_development()
                reference=read(directory/'train'/f"epoch_{trained['selected_epoch']:02d}_development.json")
                comparison=old.compare_rows(rows,reference['records'],[data.items[i] for i in data.development],expected_correct=trained['selected']['correct'])
                fixed=old.fixed_state(stack,E,rotary)==before
                result=dict(status='completed',passed=comparison['passed'] and metrics==reference['metrics'] and fixed,
                    comparison=comparison,metrics=metrics,records=rows,fixed_state_unchanged=fixed,
                    storage=model.inventory(plan['model']['path'],stack,norm,rotary),selected_checkpoint_sha256=base.sha(directory/'train/selected.pt'),
                    quality=trained['quality'],compressed_blocks=36-cut,delta_from_continuation_seed=metrics['correct']-52,confirmed=False)
                if not result['passed']:raise RuntimeError('Fresh packed verification failed')
        if not (output/'result.json').exists():base.write(output/'result.json',result)
        guard();manifest=dict(status='completed',fingerprint=stage['fingerprint'],
            artifacts={str(p.relative_to(output)):base.sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
            lease_id=lease['id'],allocation=lease['allocation'],allocator=allocator,
            elapsed_seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if reuse:manifest['reused_from']=reuse
        base.write(output/'manifest.json',manifest)
        event('phase_completed',correct=result.get('selected',result.get('metrics',{})).get('correct'),elapsed_seconds=time.monotonic()-started)
        return dict(status='completed',cut=cut,phase=phase)
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        if not (output/'failure.json').exists():base.write(output/'failure.json',dict(status='failed',error=repr(error),error_type=type(error).__name__,
            failure_kind='resource_failure' if isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)) else 'execution_failure',
            lease_id=None if lease is None else lease['id'],peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise
    finally:
        for value in (data,pairs,teacher):
            if value is not None:value.close()


def campaign(root,*,stop_cut=0):
    root=Path(root).resolve();plan=verify_root(root)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Supervisor must hide CUDA')
    if type(stop_cut) is not int or not 0<=stop_cut<=26:raise ValueError('Invalid stopping cut')
    for cut in range(26,stop_cut-1,-1):
        directory=stage_dir(root,cut)
        if not directory.exists():prepare_stage(root,cut)
        _,stage=verify_stage(root,cut)
        for phase in PHASES:
            if (directory/phase/'manifest.json').exists():checked(directory,phase,stage);continue
            if (directory/phase).exists():raise RuntimeError('Incomplete phase requires a new versioned attempt: '+str(directory/phase))
            command=['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run','--allocation','GPU-0370fdb0=11264',
                '--minutes','120','--reason',f'{STUDY} cut{cut} {phase}','--','pixi','run','--locked','--offline','-e','ncm','python',
                '-m',MODULE,phase,'--root',str(root),'--cut',str(cut)]
            with (directory/f'{phase}.log').open('xb') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
            if completed.returncode:raise RuntimeError(f'cut{cut} {phase} failed: see {directory/phase}.log')
            checked(directory,phase,stage)
        result=checked(directory,'train',stage);verification=checked(directory,'verify',stage)
        base.event(root/'progress.jsonl','stage_completed',cut=cut,compressed_blocks=36-cut,
            correct=result['selected']['correct'],quality_pass=result['quality_pass'],quality=result['quality'],storage=verification['storage'])
        if not result['quality_pass']:
            outcome=dict(status='quality_stop',last_accepted_cut=cut+1,failed_cut=cut,compressed_blocks=35-cut,
                selected=result['selected'],quality=result['quality'],confirmed=False)
            base.write(root/'outcome.json',outcome);return outcome
    outcome=dict(status='all_blocks_recovered' if stop_cut==0 else 'requested_prefix_completed',last_accepted_cut=stop_cut,
        compressed_blocks=36-stop_cut,heldout_and_speed_pending=True,confirmed=False)
    base.write(root/('outcome.json' if stop_cut==0 else f'through_cut_{stop_cut:02d}.json'),outcome);return outcome


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','prepare-stage','campaign',*PHASES));parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--seed-root',type=Path);parser.add_argument('--cut',type=int);parser.add_argument('--stop-cut',type=int,default=0)
    args=parser.parse_args()
    if args.phase=='prepare':result=prepare(args.root,args.seed_root)
    elif args.phase=='prepare-stage':result=prepare_stage(args.root,args.cut)
    elif args.phase=='campaign':result=campaign(args.root,stop_cut=args.stop_cut)
    else:result=run_phase(args.root,args.cut,args.phase)
    print(json.dumps(result,allow_nan=False),flush=True)
