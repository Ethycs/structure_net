"""One full response-corpus pass at fixed depth, with separate generation work.

The initializer and geometry are preserved. Training saves every early candidate;
costly native-prefix generation runs independently on a reserved evaluation card.
Only generated task evidence ranks usable recovery. MC is a retention guardrail.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

from safetensors import safe_open
import torch
from torch import nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from . import jspace_3b_usable_recovery_data as data
from . import jspace_3b_full_progressive_recovery as prior
from . import jspace_3b_ablation32_campaign as receipts
from .jspace_qwen_compact_stack import CompactQwenStack,_Bridge
from .jspace_qwen_affine_block import AffineQwenBlock

base=prior.base
read,write,sha,identity=data.read,data.write,data.sha,data.identity
ROOT=Path(__file__).resolve().parents[2]
SCHEMA='jspace.3b-usable-recovery.v1'
INITIAL_SHA='44349fede75578ebe1e2dbda4713edc86f3dee7d36ffe89dad4d0eedbc1fd6c0'
PROFILES={'plain2060':dict(prefix='GPU-8cf440df-',allocation=3584,limit=3328,prefix_mode='cpu'),
          '3060':dict(prefix='GPU-0370fdb0-',allocation=11264,limit=11008,prefix_mode='resident')}


def restore_unpacked(state,description):
    """Restore raw saved tensors into containers; never refold native weights."""
    extra=state['_extra_state'];spec=description['stack']
    if extra['packed']or extra['stripped']or extra['geometry_mode']!='bf16':raise ValueError('Unpacked saved BF16-geometry state required')
    blocks=[]
    for index,item in enumerate(spec['blocks']):
        block=AffineQwenBlock.__new__(AffineQwenBlock);nn.Module.__init__(block)
        for name in base.packed_source.artifact.BLOCK_FIELDS:setattr(block,name,item[name])
        block.packed=False;block.inference_dtype=base.packed_source.artifact.DTYPES[item['inference_dtype']]
        block.capture_mlp,block.last_mlp_hidden=False,None
        for kind in('parameters','buffers'):
            for name in item[kind]:
                value=state.get(f'blocks.{index}.{name}')
                if kind=='parameters':block.register_parameter(name,None if value is None else nn.Parameter(value.clone()))
                else:block.register_buffer(name,value)
        blocks.append(block)
    stack=CompactQwenStack.__new__(CompactQwenStack);nn.Module.__init__(stack);stack.blocks=nn.ModuleList(blocks)
    bridges=[]
    for index,item in enumerate(spec['bridges']):
        bridge=_Bridge.__new__(_Bridge);nn.Module.__init__(bridge);bridge.position_source=item['position_source']
        for name in('weight','shift'):bridge.register_buffer(name,state[f'bridges.{index}.{name}'])
        bridges.append(bridge)
    stack.bridges=nn.ModuleList(bridges);stack.geometry_mode=extra['geometry_mode'];stack.geometry_compute_dtype=None
    stack.pad_to_multiple=extra['pad_to_multiple'];stack.enabled=False;stack.checkpoint_blocks=True;stack.stripped=False
    stack.width,stack.layer_idx,stack.attention_type=blocks[0].width,blocks[0].layer_idx,blocks[0].attention_type
    stack.layer_indices=tuple(b.layer_idx for b in blocks);stack.ranks=tuple(b.rank for b in blocks)
    for index in range(len(blocks)):
        for suffix in('i','j'):
            name=f'feature_{suffix}_{index}';stack.register_buffer(name,state[name])
    stack.load_state_dict(state,strict=True)
    if not base.builder.state_equal(stack.state_dict(),state):raise ValueError('Structural raw restore changed saved state')
    return stack


def build(p,*,checkpoint=None,device='cpu',guard=None):
    path=checkpoint or p['checkpoint'];saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    description=read(Path(p['structure_export'])/'structure.json');cfg=Qwen2Config.from_dict(description['config'])
    stack=restore_unpacked(saved['stack'],description)
    E,norm,rotary=base.builder.native.load_readout_norm(p['model']['path'],device='cpu',config=cfg)
    norm.load_state_dict(saved['norm'],strict=True)
    # Keep the original run's actual rotary bits, including CPU/GPU construction rounding.
    with safe_open(Path(p['structure_export'])/'weights.safetensors',framework='pt',device='cpu')as stream:
        for name,_ in list(rotary.named_buffers()):setattr(rotary,name,stream.get_tensor('rotary.'+name))
        rotary.original_inv_freq=stream.get_tensor('rotary.original_inv_freq')
    for name in('attention_scaling','max_seq_len_cached','original_max_seq_len'):
        setattr(rotary,name,description['rotary_'+name])
    if saved.get('fixed_state')is not None and base.offload.fixed_state(stack,E,rotary)!=saved['fixed_state']:
        raise ValueError('Original teacher/readout/geometry/RoPE differs from saved checkpoint')
    if guard is not None:guard()
    return stack.to(device),norm.to(device),E.to(device),rotary.to(device)


def prepare(root,data_root,source_root,checkpoint,diagnostic_root,*,profile='plain2060'):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation requires CUDA hidden')
    root,data_root,source_root,checkpoint,diagnostic_root=map(lambda x:Path(x).resolve(),
        (root,data_root,source_root,checkpoint,diagnostic_root))
    if root.exists():raise FileExistsError(root)
    dp,records=data.verify(data_root);sp,_,_=prior.verify(source_root,live=False);old=prior.gpu_result(source_root,sp)
    if (sp['cut']!=33 or dp['cut']!=33 or sp['model']!=dp['model']or sp['corpus_fingerprint']!=dp['records_sha256']
            or dp['training']['epochs']!=1 or dp['training']['actual_supervised_tokens']!=105089
            or sha(checkpoint)!=INITIAL_SHA):raise ValueError('Declared existing3-block update46 and one full corpus pass required')
    matches=[c for c in old['candidates']if c['checkpoint']['path']==str(checkpoint)]
    if len(matches)!=1 or matches[0]['checkpoint']['sha256']!=INITIAL_SHA:raise ValueError('Initializer missing from authenticated source candidates')
    export=source_root/'run/selected/export';manifest=read(export/'manifest.json')
    inputs={str(f):sha(f)for f in(source_root/'plan.json',source_root/'run/gpu_result.json',
        source_root/'run/gpu_manifest.json',checkpoint,data_root/'plan.json',diagnostic_root/'plan.json',export/'manifest.json')}
    for name,wanted in manifest['files'].items():
        path=export/name
        if sha(path)!=wanted:raise ValueError('Original structural artifact changed')
        inputs[str(path)]=wanted
    sources=data.cache.prior.source.inherited.local_import_closure([Path(__file__),data.PROTOCOL,
        ROOT/'docs/07 - Status Reports/jspace/2026-11-16_qwen3b-usable-recovery-resource-amendment.md',
        ROOT/'docs/05 - Standards/NAL-STD-DISTILLATION-v1.md',ROOT/'tests/structure_net/test_jspace_3b_usable_recovery.py',
        ROOT/'pyproject.toml',ROOT/'pixi.lock',ROOT/'src/structure_net_resources/client.py',
        ROOT/'experiments/neural_architecture_lab/distillation_recovery.py',
        *[ROOT/'src/neural_architecture_lab'/f'{name}.py'for name in
            ('distillation_data','distillation_training','distillation_recovery','core','local_scheduler')]])
    cfg=base.training.ResponseConfig(learning_rate=2e-5,max_supervised_tokens=dp['training']['actual_supervised_tokens'])
    p=dict(schema=SCHEMA,data_root=str(data_root),data_fingerprint=dp['fingerprint'],source_root=str(source_root),
        source_fingerprint=sp['fingerprint'],checkpoint=str(checkpoint),checkpoint_sha256=INITIAL_SHA,
        structure_export=str(export),diagnostic_root=str(diagnostic_root),model=dp['model'],cut=33,
        corpus=dp['corpus'],corpus_fingerprint=dp['records_sha256'],response_cache_root=dp['response_cache_root'],
        full_data_root=dp['full_data_root'],configuration=asdict(cfg),training=dp['training'],checkpoints=dp['checkpoints'],
        generation_indices=dp['generation_indices'],generation_eos_token_ids=dp['generation_eos_token_ids'],
        max_new_tokens=128,profile=profile,resource_profiles=PROFILES,allowed_evaluation_profiles=list(PROFILES),backend='packed_torch',
        judge=asdict(base.judge.JudgeConfig(workers=1,expected_returned_model='codex_sdk/gpt-5.6-sol')),
        judge_lock=str(Path(dp['full_data_root']).parent/'.cpu_judge.lock'),seed=dp['training']['seed'],
        input_sha256=inputs,input_identity={f:identity(f)for f in inputs},dependency_identity=dp['dependency_identity'],
        source_sha256={str(f):sha(f)for f in sources},
        selection='Generation eligibility; completed useful tasks, reliable generation, lower DEV NLL, earlier checkpoint; MC only guardrail',
        automatic_extra_epochs=False,stage_B_implemented=False,resume_supported=False,confirmed=False)
    if profile not in PROFILES:raise ValueError('Unknown declared resource profile')
    p['fingerprint']=data.digest(p);root.mkdir(parents=True);write(root/'plan.json',p)
    for name,wanted in p['source_sha256'].items():
        target=root/'producer'/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
        if sha(target)!=wanted:raise ValueError('Recovery source archive differs')
    return dict(status='prepared',fingerprint=p['fingerprint'],training_updates=len(p['training']['windows']),response_tokens=105089)


def verify(root,*,live=True):
    root=Path(root).resolve();p=read(root/'plan.json');body=dict(p);fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or data.digest(body)!=fp:raise ValueError('Recovery plan changed')
    for path,old in{**p['dependency_identity'],**p['input_identity']}.items():
        if identity(path)!=old:raise ValueError('Recovery dependency changed: '+path)
    for path,wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=wanted or live and sha(path)!=wanted:raise ValueError('Recovery producer changed: '+path)
    dp,records=data.verify(p['data_root'],live=False)
    if dp['fingerprint']!=p['data_fingerprint']or p['training']!=dp['training']:raise ValueError('Recovery corpus/order differs')
    return p,dp,records


def diagnostic_admission(p):
    from . import jspace_3b_generation_path_diagnostic as diagnostic
    result=diagnostic.check(p['diagnostic_root']);plan=read(Path(p['diagnostic_root'])/'plan.json')
    receipt=Path(p['diagnostic_root'])/'training_admission.json';admission=read(receipt)
    if (result.get('status')!='completed' or plan.get('checkpoint_sha256')!=p['checkpoint_sha256']
            or any(result.get('systems',{}).get(k)is not True for k in
                ('packed_reload_exact','generation_entrypoint_exact','shared_original_prefix','shared_original_tied_embedding'))
            or admission.get('status')!='approved' or admission.get('reviewed')is not True
            or admission.get('checkpoint_sha256')!=p['checkpoint_sha256'] or admission.get('backend')!=p['backend']
            or admission.get('diagnostic_result_sha256')!=sha(Path(p['diagnostic_root'])/'run/result.json')):
        raise ValueError('Chosen initializer needs matching exact systems checks and authenticated reviewed path admission')
    return dict(admission_sha256=sha(receipt),diagnostic_result_sha256=admission['diagnostic_result_sha256'])


def reservation(profile):
    specification=PROFILES[profile];lease,guard=base.cache.reservation(specification['limit'])
    card,amount=next(iter(lease['allocation'].items()))
    if (len(lease['allocation'])!=1 or amount!=specification['allocation']
            or not card.startswith(specification['prefix'])):raise ValueError('Actual GPU allocation differs from declared recovery profile')
    return lease,guard


def identities(p):return dict(study=p['fingerprint'],data=p['data_fingerprint'],teacher=p['model']['revision'],initializer=p['checkpoint_sha256'])


def pack(stack,norm,E,rotary,p,guard):
    result=base.packed_source.PackedResponseModel(base.builder.packed_clone(stack),
        base.cpu_module_clone(norm).to(E.device),E.detach(),base.cpu_module_clone(rotary).to(E.device),
        config=base.builder.native._config(p['model']['path']),guard=guard)
    result.stack.enabled=False;result.inference_backend='packed_bf16_torch'
    return result


def pilot_updates(trainer,windows):
    before=trainer.parameter_fingerprints();rows=[trainer.train_window(window)for window in windows]
    changed=trainer.change_report(before)
    masters=[any(row['optimizer']['master_changed_by_parameter'][i]for row in rows)for i in range(len(trainer.parameters))]
    cast_exact=all(torch.equal(p.detach().cpu().reshape(-1),trainer.optimizer.state[p]['master'].to(p.dtype).reshape(-1))
        for p in trainer.parameters)
    finite=all(math.isfinite(r['optimizer']['unclipped_gradient_norm'])for r in rows)
    if not all(changed['bf16_changed_by_block'])or not cast_exact or not finite:
        raise RuntimeError('Pilot failed finite gradients, actual BF16-block changes or exact master casts')
    names=[name for name,p in trainer.model.named_parameters()if p.requires_grad]
    return dict(updates=rows,change=changed,master_cast_exact=cast_exact,
        unchanged_master_parameters=[name for name,changed in zip(names,masters)if not changed],finite_gradients=finite)


def pilot(root):
    root=Path(root).resolve();p,_,records=verify(root);diagnostic=diagnostic_admission(p);out=root/'pilot'
    if out.exists():raise FileExistsError(out)
    out.mkdir();dataset=trainer=None
    try:
        lease,guard=reservation(p['profile']);torch.set_num_threads(2);dataset=data.RecoveryDataset(p['data_root'])
        with base.execution.arithmetic_policy():
            stack,norm,E,rotary=build(p,device='cuda:0',guard=guard)
            initial=base.initializer_fingerprint(stack,norm,E,rotary)
            from .jspace_3b_response_campaign import pilot_windows
            windows=pilot_windows(records,p['training']['order'])
            cfg=base.training.ResponseConfig(**{**p['configuration'],'max_supervised_tokens':sum(len(records[i].response_ids)for w in windows for i in w)})
            trainer=base.training.ResponseTrainer(stack,norm,E,rotary,dataset,teacher_id=p['model']['revision'],config=cfg,guard=guard)
            updates=pilot_updates(trainer,windows);cp=trainer.save_checkpoint(out/'discarded.pt',identities=identities(p))
            trainer.close();trainer=None
            model=pack(stack,norm,E,rotary,p,guard);batch=dataset.batch([windows[0][0]])
            with torch.inference_mode():reference=model.readout(model.hidden_forward(batch)[:,-1]).cpu()
            export=base.packed_source.export_packed_snapshot(model,out/'export',identities=dict(identities(p),discarded=True))
            del model;gc.collect();torch.cuda.empty_cache()
            model,reload=base.packed_source.load_packed_snapshot(out/'export',device='cuda:0',guard=guard);model.stack.enabled=False
            with torch.inference_mode():actual=model.readout(model.hidden_forward(batch)[:,-1]).cpu()
            if not torch.equal(reference,actual):raise ValueError('Fresh packed pilot reload changed vocabulary logits')
            fresh=build(p,device='cpu')
            if base.initializer_fingerprint(*fresh)!=initial:raise ValueError('Scientific initializer would inherit pilot mutations')
            result=dict(status='completed',fingerprint=p['fingerprint'],passed=True,discarded=True,initializer=initial,
                updates=updates,fresh_initializer=True,packed_reload_exact=True,checkpoint=cp,packed_export=export,
                diagnostic_admission=diagnostic,lease_id=lease['id'],peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                training_seconds_extrapolation=sum(r['train_window_seconds']for r in updates['updates'])/len(windows)*len(p['training']['windows']),
                timing_scope='Representative pilot extrapolation excludes generation and evaluation',confirmed=False)
            write(out/'result.json',result);receipts._manifest(out,p,[f for f in out.rglob('*')if f.is_file()]);return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)));raise
    finally:
        if trainer is not None:trainer.close()
        if dataset is not None:dataset.close()


@torch.inference_mode()
def score_candidate(trainer,dataset,mc_dataset,p,out,checkpoint,label):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);model=pack(trainer.stack,trainer.norm,trainer.readout,trainer.rotary,p,trainer.guard)
    export=base.packed_source.export_packed_snapshot(model,out/'export',identities=dict(identities(p),checkpoint=checkpoint['sha256']))
    first=dataset.batch([dataset.development[0]]);reference=model.readout(model.hidden_forward(first)[:,-1]).cpu()
    del model;gc.collect();torch.cuda.empty_cache()
    model,reload=base.packed_source.load_packed_snapshot(out/'export',device=trainer.device,guard=trainer.guard);model.stack.enabled=False
    actual=model.readout(model.hidden_forward(first)[:,-1]).cpu()
    if not torch.equal(reference,actual):raise ValueError('Candidate packed reload changed full-vocabulary output')
    teacher=base.training.CachedHiddenTeacher(model.embedding,teacher_id=p['model']['revision'],guard=trainer.guard)
    by_id={r.example_id:i for i,r in enumerate(dataset.records)}
    response=base.evaluation.response_losses(model,(dataset.records[i]for i in dataset.development),pad_token_id=dataset.pad_token_id,
        teacher_provider=teacher,token_chunk_size=8,prepare_batch=lambda b:dataset.batch([by_id[n]for n in b['example_ids']]))
    mc=base.packed_mc(model,mc_dataset)
    result=dict(status='completed',fingerprint=p['fingerprint'],label=label,checkpoint=checkpoint,update=trainer.updates,
        response_tokens=trainer.supervised_tokens,nll=response['nll'],teacher_kl=response['kl'],mc=mc['metrics'],mc_eligibility=mc['eligibility'],
        packed_export=export,packed_reload_exact=True,generation_status='pending',backend='packed_torch',qualified=None)
    write(out/'development.json',response);write(out/'multiple_choice.json',mc);write(out/'result.json',result)
    receipts._manifest(out,p,[f for f in out.rglob('*')if f.is_file()]);return result


def run(root):
    root=Path(root).resolve();p,dp,records=verify(root);diagnostic=diagnostic_admission(p);proof=receipts._completed(root/'pilot',p)
    if not proof.get('passed')or not proof.get('discarded'):raise ValueError('Fresh discarded pilot required')
    if proof.get('diagnostic_admission')!=diagnostic:raise ValueError('Reviewed diagnostic changed since the pilot')
    out=root/'run'
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'checkpoints').mkdir();dataset=mc_dataset=trainer=None;started=time.monotonic()
    try:
        lease,guard=reservation(p['profile']);torch.set_num_threads(2)
        dataset=data.RecoveryDataset(p['data_root']);mc_dataset=data.MCDataset(p['data_root'])
        with base.execution.arithmetic_policy():
            stack,norm,E,rotary=build(p,device='cuda:0',guard=guard)
            if base.initializer_fingerprint(stack,norm,E,rotary)!=proof['initializer']:raise ValueError('Fresh science differs from pilot initializer')
            trainer=base.training.ResponseTrainer(stack,norm,E,rotary,dataset,teacher_id=p['model']['revision'],
                config=base.training.ResponseConfig(**p['configuration']),guard=guard)
            before=trainer.parameter_fingerprints();seen=[];candidates=[];training_seconds=0.
            def record():
                label=f'update_{trainer.updates:04d}';cp=trainer.save_checkpoint(out/'checkpoints'/(label+'.pt'),identities=identities(p))
                row=score_candidate(trainer,dataset,mc_dataset,p,out/'candidates'/label,cp,label)
                row['coverage']=data.coverage_report(dp,seen);candidates.append(row)
                base._event(out,'candidate_saved_generation_pending',label=label,nll=row['nll'],response_tokens=trainer.supervised_tokens)
            record();milestones={r['update']for r in p['checkpoints']}
            for window in p['training']['windows']:
                row=trainer.train_window(window);seen.extend(window);training_seconds+=row['train_window_seconds']
                base._event(out,'full_corpus_recovery',**row)
                if trainer.updates in milestones:record()
            coverage=data.coverage_report(dp,seen);change=trainer.change_report(before)
            if not coverage['planned_run_completed']or not all(change['bf16_changed_by_block']):raise ValueError('Full-pass coverage or intended block updates failed')
            result=dict(status='completed',fingerprint=p['fingerprint'],systems_passed=True,coverage=coverage,candidates=candidates,
                provisional_checkpoint=min(candidates,key=lambda c:(c['nll'],c['update']))['checkpoint'],
                selection_status='Generation pending; NLL candidate is provisional only',qualified=None,
                all_early_checkpoints_retained=True,joint_change=change,training_seconds=training_seconds,
                elapsed_seconds=time.monotonic()-started,lease_id=lease['id'],peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                automatic_extra_epochs=False,resume_supported=False,stage_B_implemented=False,confirmed=False)
            write(out/'result.json',result);receipts._manifest(out,p,[f for f in out.rglob('*')if f.is_file()]);return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),partial_artifacts_preserved=True));raise
    finally:
        if trainer is not None:trainer.close()
        if dataset is not None:dataset.close()
        if mc_dataset is not None:mc_dataset.close()


def candidate(root,p,label):
    if not label.startswith('update_')or not label[7:].isdigit():raise ValueError('Explicit saved candidate label required')
    return receipts._completed(Path(root)/'run/candidates'/label,p)


@torch.inference_mode()
def generate(root,label,*,profile=None):
    root=Path(root).resolve();p,_,records=verify(root);row=candidate(root,p,label);profile=profile or p['profile']
    if profile not in p['allowed_evaluation_profiles']:raise ValueError('Evaluation resource profile was not declared')
    out=root/'generation'/label
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);lease,guard=reservation(profile);torch.set_num_threads(2)
    try:
        with base.execution.arithmetic_policy():
            model,reload=base.packed_source.load_packed_snapshot(row['packed_export']['directory'],device='cuda:0',guard=guard)
            model.stack.enabled=False;model.inference_backend='packed_bf16_torch'
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(p['model']['path'],local_files_only=True,trust_remote_code=False)
            contexts={r['example_id']:r for r in map(json.loads,(Path(p['corpus'])/'text.jsonl').read_text().splitlines())}
            prefix_mode=p['resource_profiles'][profile]['prefix_mode']
            with base.packed_source.attach_prefix(model,p['model']['path'],mode=prefix_mode)as prefix:
                reference=base.packed_evaluation.packed_probe(model,records[p['generation_indices'][0]].prompt_ids)
                replay=base.packed_evaluation.packed_probe(model,records[p['generation_indices'][0]].prompt_ids,
                    prefixes=[r['input_ids']for r in reference]);pathproof=base.packed_evaluation.reload_parity(reference,replay)
                started=time.monotonic();canaries=base.evaluation.evaluate_canaries(model,tokenizer,device='cuda:0',
                    eos_token_ids=p['generation_eos_token_ids'],max_new_tokens=128)
                write(out/'canaries.json',canaries);rows=[]
                for i in p['generation_indices']:
                    record=records[i];result=base.evaluation.generate_record(model,record,tokenizer,device='cuda:0',
                        source_text=contexts[record.example_id]['context'],eos_token_ids=p['generation_eos_token_ids'],max_new_tokens=128)
                    rows.append(result)
                    with(out/'responses.jsonl').open('a')as stream:stream.write(json.dumps(result)+'\n')
                seconds=time.monotonic()-started
            result=dict(status='completed',fingerprint=p['fingerprint'],label=label,checkpoint=row['checkpoint'],
                canaries=canaries,corpus=rows,generation_seconds=seconds,profile=profile,prefix=prefix,
                backend='packed_torch',reload=reload,full_vocabulary_replay=pathproof,lease_id=lease['id'],
                prefix_mode=prefix_mode,cross_hardware_evaluation=profile!=p['profile'],
                timing_scope='Full-prefix recompute including declared native-prefix transfer costs; no speedup claim',confirmed=False)
            write(out/'contexts.json',{r['example_id']:contexts[r['example_id']]for r in rows});write(out/'result.json',result)
            receipts._manifest(out,p,[f for f in out.rglob('*')if f.is_file()]);return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error)));raise


def quality_rank(row):
    return (-row['useful_completed_tasks'],-row['reliable_generations'],row['nll'],row['update'])


def select_quality(rows):
    known=[r for r in rows if r.get('evidence_available')]
    eligible=[r for r in known if r.get('qualified')is True]
    if not known:return dict(status='provisional_unavailable',qualified=None,selected=None)
    chosen=min(eligible or known,key=quality_rank)
    return dict(status='exploratory_quality_pass'if eligible else'exploratory_unqualified',qualified=bool(eligible),selected=chosen)


def judge(root,label):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU judgment requires CUDA hidden')
    root=Path(root).resolve();p,_,_=verify(root);row=candidate(root,p,label);actual=receipts._completed(root/'generation'/label,p)
    out=root/'judgment'/label
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    with Path(p['judge_lock']).open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            native=read(Path(p['response_cache_root'])/'native/generation.json');contexts=read(root/'generation'/label/'contexts.json')
            executable=base.task_metrics.grade_generated_tasks(native['corpus'],actual['corpus'],contexts,maximum_drop_pp=5.,minimum_student_rate=0.)
            write(out/'executable.json',executable)
            semantic=base.judge.judge_pairs(native['corpus'],actual['corpus'],out/'semantic',config=base.judge.JudgeConfig(**p['judge']),
                contexts={k:dict(prompt=v['prompt_text'],source=v['context'],reference=v['response'],task=v['task'])for k,v in contexts.items()})
            quality=prior.task_probe.quality_summary(native,actual,semantic,executable)
            valid=[j for j in semantic['judgments']if j.get('status')=='completed']
            tasks=sum(j['student']['task_success']and j['student']['generation_success']for j in valid)
            reliable=sum(j['student']['generation_success']for j in valid)
            available=quality['evidence_available'];qualified=bool(row['mc_eligibility']['passed']and quality['diagnostic_generation_retention_pass'])if available else None
            result=dict(status='completed',fingerprint=p['fingerprint'],label=label,checkpoint=row['checkpoint'],
                evidence_available=available,qualified=qualified,useful_completed_tasks=tasks,reliable_generations=reliable,
                quality=quality,mc_guardrail=row['mc_eligibility'],nll=row['nll'],update=row['update'],confirmed=False)
        except Exception as error:
            result=dict(status='completed',fingerprint=p['fingerprint'],label=label,evidence_available=False,
                qualified=None,error_type=type(error).__name__,error=str(error),nll=row['nll'],update=row['update'],confirmed=False)
        write(out/'result.json',result);receipts._manifest(out,p,[f for f in out.rglob('*')if f.is_file()]);return result


def check(root):
    root=Path(root);p,_,_=verify(root,live=False);result=dict(status='prepared',fingerprint=p['fingerprint'])
    if(root/'pilot/manifest.json').exists():result=receipts._completed(root/'pilot',p)
    if(root/'run/manifest.json').exists():
        result=receipts._completed(root/'run',p);judgments=[];pending=[]
        for row in result['candidates']:
            path=root/'judgment'/row['label']
            if(path/'manifest.json').exists():judgments.append(receipts._completed(path,p))
            else:pending.append(row['label'])
        result=dict(result,task_selection=select_quality(judgments),generation_or_judgment_pending=pending,
            selection_final=not pending,confirmed=False)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('prepare','pilot','run','generate','judge','check'))
    for name in('root','data-root','source-root','checkpoint','diagnostic-root'):parser.add_argument('--'+name,type=Path,required=name=='root')
    parser.add_argument('--label');parser.add_argument('--profile',choices=PROFILES);args=parser.parse_args()
    if args.phase=='prepare':
        if any(getattr(args,k)is None for k in('data_root','source_root','checkpoint','diagnostic_root')):parser.error('All source/data/diagnostic inputs required')
        result=prepare(args.root,args.data_root,args.source_root,args.checkpoint,args.diagnostic_root,profile=args.profile or'plain2060')
    elif args.phase in('generate','judge'):
        if not args.label:parser.error('Explicit candidate --label required')
        result=generate(args.root,args.label,profile=args.profile)if args.phase=='generate'else judge(args.root,args.label)
    else:result=globals()[args.phase](args.root)
    print(json.dumps({k:result[k]for k in('status','fingerprint','passed','qualified','label')if k in result}))


if __name__=='__main__':main()
