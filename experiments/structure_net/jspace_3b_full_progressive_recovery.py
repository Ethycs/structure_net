"""Complete progressive compression with honest, nonblocking task qualification.

Every successful systems/training stage publishes its best observed continuation
even when task retention fails. NLL is diagnostic. GPU execution and immutable
CPU task judging are separate, permitting the next stage to fit while judging.
Only this new study's explicitly temporary checkpoint/export files are pruned.
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

import torch

from . import jspace_3b_rebuild_recovery as base
from . import jspace_3b_rebuild_task_probe as task_probe
from . import jspace_3b_full_progressive_data as data
from . import jspace_3b_full_progressive_fit as fitting

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/07 - Status Reports/jspace/2026-11-16_qwen3b-full-progressive-compression.md'
SCHEMA='jspace.3b-full-progressive-recovery.v1'
read,write,sha,identity=base.read,base.write,base.sha,base.identity
digest=base.cache.data.digest


def _source_files():
    return base.cache.prior.source.inherited.local_import_closure([
        Path(__file__),PROTOCOL,base.STANDARD,
        ROOT/'tests/structure_net/test_jspace_3b_full_progressive_recovery.py'])


def prepare(root,data_root,fit_root,cut):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation requires CUDA hidden')
    root,data_root,fit_root=map(lambda p:Path(p).resolve(),(root,data_root,fit_root))
    if root.exists():raise FileExistsError(root)
    if type(cut)is not int or not 0<=cut<=34:raise ValueError('Full continuation cuts34 through0 required')
    dp,records=data.verify(data_root,live=False)
    fp=fitting.verify(fit_root,live=False);completed=fitting.check(fit_root)
    if (completed.get('status')!='completed' or fp['data_fingerprint']!=dp['fingerprint']
            or fp['cut']!=cut or fp['model']!=dp['model'] or Path(fp['data_root']).resolve()!=data_root):
        raise ValueError('Completed fresh fit must belong to this full cache/native model/cut')
    parent=fp['parent']
    if parent is None:raise ValueError('Every stage here requires an explicitly admitted continuation parent')
    receipt_path=Path(completed['fit_receipt']).resolve();receipt=read(receipt_path)
    if (receipt.get('teacher_fingerprint')!=fp['teacher_fingerprint']
            or receipt.get('parent_checkpoint_sha256')!=parent['checkpoint_sha256']
            or not receipt.get('native_seam_bit_exact') or not receipt.get('saved_fit_compilation_passed')):
        raise ValueError('Fit identity, compiled state, or original teacher seam failed')
    inputs={str(data_root/'plan.json'):sha(data_root/'plan.json'),str(data_root/'native/manifest.json'):sha(data_root/'native/manifest.json'),
        str(fit_root/'plan.json'):sha(fit_root/'plan.json'),str(fit_root/'result.json'):sha(fit_root/'result.json'),
        str(receipt_path):sha(receipt_path)}
    inputs.update(fp['input_sha256'])
    # The fitter's completed output identities authenticate its immutable large
    # captures. Content pins are retained for the ordinary five fitted files.
    fits=dict(parent['fits']);fits[str(cut)]=receipt['fit_directory']
    if set(fits)!={str(i) for i in range(cut,36)}:raise ValueError('Complete contiguous fit map required')
    for directory in fits.values():
        for name in ('fit.pt','fit.json','relation.pt','relations.json','interface.json'):
            path=Path(directory)/name;inputs[str(path)]=sha(path)
    if sha(parent['checkpoint'])!=parent['checkpoint_sha256']:raise ValueError('Continuation checkpoint changed')
    inputs[parent['checkpoint']]=parent['checkpoint_sha256']
    p=dict(schema=SCHEMA,cut=cut,data_root=str(data_root),data_fingerprint=dp['fingerprint'],
        corpus=dp['corpus'],corpus_fingerprint=dp['records_sha256'],model=dp['model'],
        fit_root=str(receipt_path.parent),fit_campaign_root=str(fit_root),fits=fits,parent=parent,
        local_configuration=asdict(base.LocalConfig()),local_alignment_indices=dp['local_alignment_indices'],
        local_alignment_windows=dp['local_alignment_windows'],training=dp['training'],checkpoints=dp['recovery_checkpoints'],
        configuration=asdict(base.training.ResponseConfig(max_supervised_tokens=dp['recovery_response_tokens'])),
        distillation_teacher='Original native final-normalized hidden and original frozen tied full-vocabulary readout',
        functional_basis_path='Native new entrance followed by frozen compact continuation suffix; not a full-native-model logit Jacobian',
        generation_indices=dp['generation_indices'],generation_eos_token_ids=dp['generation_eos_token_ids'],
        max_new_tokens=128,max_length=416,prefix_mode='resident',seed=dp['training']['seed'],
        judge=asdict(base.judge.JudgeConfig(workers=1,expected_returned_model='codex_sdk/gpt-5.6-sol')),
        gates=dict(maximum_drop_pp=5.,maximum_family_drop_pp=10.,minimum_generation_rate=0.),
        nll_gate=False,quality_failure_stops_progression=False,
        selection='Highest packed MC correct, lowest response NLL, fewest response tokens, earliest update',
        retention=dict(unpacked_checkpoints=3,packed_exports=1,temporary_exports='per-candidate disposable scratch'),
        input_sha256=inputs,input_identity={path:identity(path) for path in inputs},
        dependency_identity={**dp['input_identity'],**fp['input_identity']},
        source_sha256={str(path):sha(path) for path in sorted(_source_files())},
        allocation_mib=11264,limit_mib=11008,required_discarded_pilot=True,confirmed=False)
    p['fingerprint']=digest(p);root.mkdir(parents=True);write(root/'plan.json',p)
    for path,wanted in p['source_sha256'].items():
        target=root/'producer'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(target)!=wanted:raise ValueError('Full recovery producer archive differs')
    return dict(status='prepared',fingerprint=p['fingerprint'],cut=cut,blocks=36-cut,
        updates=len(p['training']['windows']),response_tokens=p['training']['actual_supervised_tokens'])


def verify(root,*,live=True):
    root=Path(root).resolve();p=read(root/'plan.json');body=dict(p);fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or digest(body)!=fp:raise ValueError('Full progressive plan changed')
    for path,old in {**p['dependency_identity'],**p['input_identity']}.items():
        if identity(path)!=old:raise ValueError('Full progressive input changed: '+path)
    for path,wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=wanted or live and sha(path)!=wanted:
            raise ValueError('Full progressive producer changed: '+path)
    dp,records=data.verify(p['data_root'],live=False)
    if (p['data_fingerprint']!=dp['fingerprint'] or p['training']!=dp['training']
            or p['checkpoints']!=dp['recovery_checkpoints']):raise ValueError('Frozen full recovery data/order differs')
    return p,dp,records


def rank_candidate(candidate):
    if (type(candidate['mc_correct'])is not int or candidate['mc_count']<1
            or not 0<=candidate['mc_correct']<=candidate['mc_count'] or not math.isfinite(candidate['nll'])):
        raise ValueError('Finite measured candidate quality required for continuation ranking')
    return (-candidate['mc_correct'],candidate['nll'],candidate['response_tokens'],candidate['update'])


def select(candidates):
    if not candidates:raise ValueError('At least one actual packed candidate required')
    return min(candidates,key=rank_candidate)


def retain_best_three(directory,candidates):
    """Delete only this attempt's declared provisional checkpoint files."""
    directory=Path(directory).resolve();checkpoint_root=directory/'checkpoints'
    keep={c['checkpoint']['path'] for c in sorted(candidates,key=rank_candidate)[:3]}
    deleted=[]
    for candidate in candidates:
        checkpoint=candidate['checkpoint'];path=Path(checkpoint['path']).resolve()
        if str(path)in keep or not path.exists():continue
        if path.parent!=checkpoint_root or sha(path)!=checkpoint['sha256']:
            raise ValueError('Refusing to prune a nonowned or modified provisional checkpoint')
        path.unlink();deleted.append(dict(label=candidate['label'],path=str(path),sha256=checkpoint['sha256'],
            reason='Prospective best-three retention; metrics and identity preserved'))
    if deleted:
        with (directory/'pruned_checkpoints.jsonl').open('a') as stream:
            for row in deleted:stream.write(json.dumps(row,sort_keys=True)+'\n')
    return sorted(keep)


def _reservation(p,limit_mib):
    if limit_mib!=p['limit_mib']:raise ValueError('Allocator differs from frozen full recovery plan')
    lease,guard=base.cache.reservation(limit_mib)
    if next(iter(lease['allocation'].values()))!=p['allocation_mib']:
        raise ValueError('Declared full recovery reservation required')
    return lease,guard


def _ledger(root,p,result,parameter_count):
    """NAL records systems-complete continuation separately from task verdicts."""
    selected=result['selected']
    experiment=base.nal.Experiment(id=f"fullprogressive.cut{p['cut']}.{p['fingerprint'][:12]}",
        hypothesis_id='JSPACE-QWEN3B-FULL-PROGRESSIVE-01',name='Full progressive response recovery stage',
        parameters=dict(fingerprint=p['fingerprint'],cut=p['cut'],quality_pending=True),seed=p['seed'])
    record=base.nal.ExperimentResult(experiment_id=experiment.id,hypothesis_id=experiment.hypothesis_id,
        metrics=dict(mc_accuracy=selected['mc_accuracy'],response_nll=selected['nll'],
            compressed_blocks=36-p['cut']),primary_metric=selected['mc_accuracy'],
        model_architecture=[p['cut'],36-p['cut'],1024],model_parameters=parameter_count,
        training_time=result['training_seconds'],model_checkpoint=result['continuation_checkpoint'],
        observations=['Systems/training/generation completed; task qualification is a separate pending record',
            'MC/NLL selected exploratory continuation, independent of quality pass',
            'Parameter count covers compact trainable suffix and norm; native prefix/readout counted in storage receipt'],
        status=base.nal.ExperimentStatus.COMPLETED)
    return str(base.nal.ExperimentResultLedger(root).save(experiment,record,attempts=1,device_id=0))


def pilot(root,*,limit_mib=11008):
    root=Path(root).resolve();p,_,records=verify(root);directory=root/'pilot'
    if directory.exists():raise FileExistsError(directory)
    dataset=None;lease=None;started=time.monotonic()
    try:
        lease,guard=_reservation(p,limit_mib);torch.set_num_threads(2);torch.manual_seed(p['seed'])
        dataset=data.RebuildDataset(p['data_root'],cut=p['cut'])
        with base.execution.arithmetic_policy():
            stack,norm,E,rotary,_=base.build_student(p,device='cuda:0',guard=guard)
            from .jspace_3b_response_campaign import pilot_windows
            indices=pilot_windows(records,p['training']['order'],updates=1)[0]
            config=base.training.ResponseConfig(**{**p['configuration'],
                'max_supervised_tokens':sum(len(records[i].response_ids)for i in indices)})
            result=base.systems_pilot(stack,norm,E,rotary,dataset,
                local_indices=p['local_alignment_indices'],joint_indices=indices,
                local_config=base.LocalConfig(**p['local_configuration']),response_config=config,
                teacher_id=p['model']['revision'],model_config=base.builder.native._config(p['model']['path']),
                directory=directory/'systems',guard=guard)
        receipt=dict(status='completed',passed=True,discarded=True,fingerprint=p['fingerprint'],
            initializer=result['initializer'],systems_result_sha256=sha(directory/'systems/result.json'),
            lease_id=lease['id'],allocation=lease['allocation'],elapsed_seconds=time.monotonic()-started,
            disposable_payloads_deleted=True,
            discarded_checkpoint_sha256=result['checkpoint']['sha256'],
            discarded_packed_state_sha256=result['packed_export']['packed_state_sha256'])
        # These are explicitly disposable systems-pilot weights, not scientific
        # candidates. Preserve their tested identities and complete receipts.
        checkpoint=Path(result['checkpoint']['path']).resolve()
        if checkpoint.parent!=(directory/'systems').resolve() or sha(checkpoint)!=result['checkpoint']['sha256']:
            raise ValueError('Disposable pilot checkpoint ownership or identity differs')
        checkpoint.unlink();shutil.rmtree(directory/'systems/export')
        write(directory/'result.json',receipt);return receipt
    except BaseException as error:
        directory.mkdir(parents=True,exist_ok=True)
        write(directory/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],
            error_type=type(error).__name__,error=str(error),lease=lease,requires_diagnosis=True))
        raise
    finally:
        if dataset is not None:dataset.close()


@torch.inference_mode()
def score_candidate(stack,norm,E,rotary,dataset,mc_dataset,p,dp,directory,checkpoint,*,label,update,tokens,guard):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    scratch=directory.parents[1]/'scratch'/label;scratch.mkdir(parents=True,exist_ok=False)
    device=E.device;cfg=base.builder.native._config(p['model']['path'])
    model=base.packed_source.PackedResponseModel(base.builder.packed_clone(stack,device=device),
        base.cpu_module_clone(norm).to(device),E.detach(),base.cpu_module_clone(rotary).to(device),
        config=cfg,max_length=p.get('max_length',416),guard=guard)
    export=base.packed_source.export_packed_snapshot(model,scratch/'export',
        identities=dict(base._identities(p),checkpoint_sha256=checkpoint['sha256'],label=label))
    batch=dataset.batch([dataset.development[0]])
    reference=model.readout(model.hidden_forward(batch)[:,-1]).cpu()
    del model;gc.collect();torch.cuda.empty_cache()
    model,reload=base.packed_source.load_packed_snapshot(scratch/'export',device=device,guard=guard)
    actual=model.readout(model.hidden_forward(batch)[:,-1]).cpu()
    if not torch.equal(reference,actual) or export['packed_state_sha256']!=reload['packed_state_sha256']:
        raise RuntimeError('Actual packed export/reload replay differs')
    by_id={r.example_id:i for i,r in enumerate(dataset.records)}
    teacher=base.training.CachedHiddenTeacher(model.embedding,teacher_id=p['model']['revision'],guard=guard)
    response=base.evaluation.response_losses(model,(dataset.records[i]for i in dataset.development),
        pad_token_id=dataset.pad_token_id,max_length=p.get('max_length',416),teacher_provider=teacher,token_chunk_size=8,
        prepare_batch=lambda b:dataset.batch([by_id[n]for n in b['example_ids']]))
    mc=base.packed_mc(model,mc_dataset)
    native=read(Path(dp['response_cache_root'])/'native/development.json')
    row=dict(label=label,update=update,response_tokens=tokens,checkpoint=checkpoint,nll=response['nll'],
        teacher_kl=response['kl'],native_nll=native['nll'],nll_ratio=response['nll']/native['nll'],
        nll_is_diagnostic=True,mc_correct=mc['metrics']['correct'],mc_count=mc['metrics']['count'],
        mc_accuracy=mc['metrics']['accuracy'],mc_eligibility=mc['eligibility'],
        packed_state_sha256=export['packed_state_sha256'],packed_export=export,packed_reload=reload,
        full_vocabulary_same_packed_replay=True,generation_status='not_selected_yet',confirmed=False,
        checkpoint_role='Provisional candidate, subject to declared best-three retention',
        packed_artifact_retained=False,packed_artifact_policy='Disposable authenticated per-candidate scratch')
    write(directory/'development.json',response);write(directory/'multiple_choice.json',mc);write(directory/'result.json',row)
    del model,teacher,batch,reference,actual;gc.collect();torch.cuda.empty_cache()
    # Only newly created scratch exports are removed; permanent evidence and
    # old studies are never passed to this cleanup path.
    shutil.rmtree(scratch)
    return row


def _difference(actual,reference):
    if actual.shape!=reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('Finite equal-shaped seam-control tensors required')
    difference=actual.float()-reference.float()
    denominator=reference.float().square().mean().sqrt().clamp_min(1e-20)
    return dict(bit_exact=torch.equal(actual,reference),elements=actual.numel(),
        max_absolute_difference=float(difference.abs().max()),
        relative_rms_difference=float(difference.square().mean().sqrt()/denominator))


@torch.inference_mode()
def prefix_cache_control(model,batch,*,prompt_tokens):
    """Same-length exact systems gate plus separate causal-prefix diagnostics.

    Cached native boundaries were computed for prompt plus teacher-forced
    response. Changing the sequence length can change BF16 GEMM/reduction
    rounding, so the cropped-cache comparison is descriptive, not a gate.
    """
    device=model.embedding.device
    ids=batch['input_ids'].to(device);valid=batch['attention_mask'].to(device).bool()
    reference=batch['boundary_hidden'].detach().to(model.embedding)
    if (ids.ndim!=2 or len(ids)!=1 or not valid.all()
            or not 1<=prompt_tokens<ids.shape[1] or reference.shape!=(*ids.shape,model.stack.width)):
        raise ValueError('One complete unpadded cached DEV record with shorter prompt required')
    def boundary(tokens,mask):
        if model.cut==0:return torch.nn.functional.embedding(tokens,model.embedding)
        if model.native_prefix is None:raise ValueError('Actual native prefix must be attached for the seam control')
        return model.native_prefix(tokens,mask)
    actual=boundary(ids,valid)
    full_hidden=_difference(actual,reference)
    live_logits=model.next_logits(ids)
    cached_logits=model.readout(model.hidden_forward(dict(input_ids=ids,attention_mask=valid,
                                                          boundary_hidden=reference))[:,-1])[0]
    full_logits=_difference(live_logits,cached_logits)
    prompt_ids=ids[:,:prompt_tokens];prompt_valid=valid[:,:prompt_tokens]
    prompt_hidden=boundary(prompt_ids,prompt_valid)
    cropped=reference[:,:prompt_tokens]
    prompt_live=model.next_logits(prompt_ids)
    prompt_cached=model.readout(model.hidden_forward(dict(input_ids=prompt_ids,attention_mask=prompt_valid,
                                                          boundary_hidden=cropped))[:,-1])[0]
    return dict(passed=full_hidden['bit_exact'] and full_logits['bit_exact'],cut=model.cut,
        full_sequence_tokens=ids.shape[1],prompt_tokens=prompt_tokens,
        full_sequence_boundary=full_hidden,full_sequence_same_packed_logits=full_logits,
        full_sequence_greedy_equal=int(live_logits.argmax())==int(cached_logits.argmax()),
        prompt_versus_cropped_cache=dict(hidden=_difference(prompt_hidden,cropped),
            logits=_difference(prompt_live,prompt_cached),
            greedy_equal=int(prompt_live.argmax())==int(prompt_cached.argmax()),hard_gate=False,
            scope='Different sequence lengths; numerical differences are recorded without attributing task failure'),
        execution='Original native prefix'if model.cut else 'Original tied embedding lookup',
        scope='Systems check against authenticated native cache; not model/task accuracy')


@torch.inference_mode()
def generate_selected(p,records,candidate,directory,guard):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    # Reconstruct the exact selected checkpoint, not the final live weights.
    stack,norm,E,old_rope,proof=base.builder.build_stack(p['model']['path'],
        {int(k):v for k,v in p['fits'].items()},p['cut'],previous=candidate['checkpoint']['path'],device='cpu',trainable=False)
    del old_rope
    cfg=base.builder.native._config(p['model']['path'])
    rotary=base.model_source.partial.training_rotary(cfg,'cuda:0',guard=guard)
    model=base.packed_source.PackedResponseModel(base.builder.packed_clone(stack,device='cuda:0'),
        norm.to('cuda:0'),E.to('cuda:0'),rotary,config=cfg,guard=guard)
    del stack,norm,E,rotary;gc.collect()
    export=base.packed_source.export_packed_snapshot(model,directory/'export',
        identities=dict(base._identities(p),checkpoint_sha256=candidate['checkpoint']['sha256'],label=candidate['label']))
    if export['packed_state_sha256']!=candidate['packed_state_sha256']:
        raise RuntimeError('Selected reconstruction/packing differs from the actually scored candidate')
    del model;gc.collect();torch.cuda.empty_cache()
    model,reload=base.packed_source.load_packed_snapshot(directory/'export',device='cuda:0',guard=guard)
    if reload['packed_state_sha256']!=candidate['packed_state_sha256']:raise RuntimeError('Selected packed reload differs')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(p['model']['path'],local_files_only=True,trust_remote_code=False)
    texts={r['example_id']:r for r in map(json.loads,(Path(p['corpus'])/'text.jsonl').read_text().splitlines())}
    with base.packed_source.attach_prefix(model,p['model']['path'],mode=p['prefix_mode']) as prefix:
        first=p['generation_indices'][0]
        control_dataset=data.RebuildDataset(p['data_root'],cut=p['cut'])
        try:
            control=prefix_cache_control(model,control_dataset.batch([first]),prompt_tokens=len(records[first].prompt_ids))
        finally:control_dataset.close()
        write(directory/'prefix_cache_control.json',control)
        if not control['passed']:
            raise RuntimeError('Identical-token native boundary or same-packed generation/cache seam differs; diagnose before progression')
        started=time.monotonic()
        canaries=base.evaluation.evaluate_canaries(model,tokenizer,device='cuda:0',
            eos_token_ids=p['generation_eos_token_ids'],max_new_tokens=p['max_new_tokens'])
        write(directory/'canaries.json',canaries);rows=[]
        for index in p['generation_indices']:
            record=records[index]
            row=base.evaluation.generate_record(model,record,tokenizer,device='cuda:0',
                source_text=texts[record.example_id]['context'],eos_token_ids=p['generation_eos_token_ids'],
                max_new_tokens=p['max_new_tokens'])
            rows.append(row)
            with (directory/'responses.jsonl').open('a')as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
            base._event(directory,'selected_generation',completed=len(rows),total=len(p['generation_indices']))
        seconds=time.monotonic()-started
        warm=base.packed_evaluation.warm_timing(model,records[p['generation_indices'][0]].prompt_ids,warmup_calls=1,repeats=3)
    tokens=sum(len(r['output_ids'])for r in [*canaries['records'],*rows])
    result=dict(canaries=canaries,corpus=rows,use_cache=False,generation_seconds=seconds,
        output_tokens_including_eos=tokens,cachefree_generation_tokens_per_second=tokens/seconds,
        timing_scope='Actual packed full-prefix recomputation, not KV-cached deployment speed',
        native_prefix=prefix,warm_timing=warm,packed_export=export,packed_reload=reload,
        selected_checkpoint=candidate['checkpoint'],strict_selected_reconstruction=proof,
        prefix_cache_control=control,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),confirmed=False)
    write(directory/'generation.json',result)
    write(directory/'contexts.json',{r['example_id']:texts[r['example_id']]for r in rows})
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def run(root,*,limit_mib=11008):
    root=Path(root).resolve();p,dp,records=verify(root);pilot_result=base.pilot_receipt(root,p)
    directory=root/'run'
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir();(directory/'checkpoints').mkdir()
    dataset=mc_dataset=trainer=local=None;lease=None;phase='reservation';started=time.monotonic()
    try:
        lease,guard=_reservation(p,limit_mib);torch.set_num_threads(2);torch.manual_seed(p['seed'])
        dataset=data.RebuildDataset(p['data_root'],cut=p['cut']);mc_dataset=data.MCDataset(p['data_root'],cut=p['cut'])
        with base.execution.arithmetic_policy():
            stack,norm,E,rotary,inheritance=base.build_student(p,device='cuda:0',guard=guard)
            if base.initializer_fingerprint(stack,norm,E,rotary)!=pilot_result['initializer']:
                raise RuntimeError('Fresh scientific initializer differs from discarded pilot initializer')
            write(directory/'inheritance.json',inheritance)
            storage=base.builder.inventory(p['model']['path'],stack,norm,rotary)
            parameter_count=sum(v.numel()for v in (*stack.parameters(),*norm.parameters()))
            write(directory/'storage.json',storage);candidates=[];training_seconds=0.
            def score(checkpoint,label,update,tokens):
                base._event(directory,'packed_candidate',label=label,update=update,response_tokens=tokens)
                row=score_candidate(stack,norm,E,rotary,dataset,mc_dataset,p,dp,
                    directory/'candidates'/label,checkpoint,label=label,update=update,tokens=tokens,guard=guard)
                candidates.append(row);retain_best_three(directory,candidates)
            phase='raw_compiled_control'
            score(base.save_raw(directory/'checkpoints/prefit.pt',stack,norm,p),'prefit',-1,0)
            phase='local_prealignment'
            local=base.LocalTrainer(stack,norm,E,rotary,dataset,p['local_alignment_indices'],
                config=base.LocalConfig(**p['local_configuration']),guard=guard)
            for indices in p['local_alignment_windows']:
                row=local.train_window(indices);training_seconds+=row['train_window_seconds'];base._event(directory,phase,**row)
            local_result=local.finish();local.close();local=None;write(directory/'local_alignment.json',local_result)
            phase='joint_recovery'
            trainer=base.training.ResponseTrainer(stack,norm,E,rotary,dataset,teacher_id=p['model']['revision'],
                config=base.training.ResponseConfig(**p['configuration']),guard=guard)
            before=trainer.parameter_fingerprints()
            checkpoint=trainer.save_checkpoint(directory/'checkpoints/update_0000.pt',identities=base._identities(p))
            score(checkpoint,'update_0000',0,0);milestones={r['update']for r in p['checkpoints']}
            for indices in p['training']['windows']:
                phase='joint_recovery';row=trainer.train_window(indices);training_seconds+=row['train_window_seconds']
                base._event(directory,phase,**row)
                if trainer.updates in milestones:
                    phase='checkpoint'
                    checkpoint=trainer.save_checkpoint(directory/f'checkpoints/update_{trainer.updates:04d}.pt',identities=base._identities(p))
                    score(checkpoint,f'update_{trainer.updates:04d}',trainer.updates,trainer.supervised_tokens)
            change=trainer.change_report(before)
            if (not all(change['bf16_changed_by_block']) or trainer.visited!=set(p['training']['order'])
                    or trainer.supervised_tokens!=p['training']['actual_supervised_tokens']):
                raise RuntimeError('Actual full-suffix updates or one-pass token/membership budget failed')
            trainer.close();trainer=None
            selected=select(candidates);retained=retain_best_three(directory,candidates)
            for candidate in candidates:
                candidate['checkpoint_retained']=candidate['checkpoint']['path']in retained
            # Free live train/optimizer state before restoring the chosen model.
            del stack,norm,E,rotary;gc.collect();torch.cuda.empty_cache()
            phase='selected_generation'
            generation=generate_selected(p,records,selected,directory/'selected',guard)
            result=dict(status='completed',fingerprint=p['fingerprint'],cut=p['cut'],compressed_blocks=36-p['cut'],
                all36_compressed=p['cut']==0,systems_passed=True,quality_status='pending',relative_retention_pass=None,
                continuation_checkpoint=selected['checkpoint']['path'],continuation_checkpoint_sha256=selected['checkpoint']['sha256'],
                continuation_is_qualified=False,continuation_policy=p['selection'],selected=selected,candidates=candidates,
                retained_checkpoints=retained,local_alignment=local_result,joint_change=change,storage=storage,
                packed_export=generation['packed_export'],packed_state_sha256=generation['packed_export']['packed_state_sha256'],
                generation_seconds=generation['generation_seconds'],response_tokens=p['training']['actual_supervised_tokens'],
                training_seconds=training_seconds,elapsed_seconds=time.monotonic()-started,lease_id=lease['id'],
                allocation=lease['allocation'],allocator_mib=limit_mib,confirmed=False,
                quality_failure_stops_progression=False,task_judgment='Separate immutable CPU result pending')
            result['distillation_teacher']=p['distillation_teacher']
            result['functional_basis_path']=p['functional_basis_path']
            result['nal_ledger']=_ledger(root,p,result,parameter_count)
            write(directory/'gpu_result.json',result)
            artifacts={str(path.relative_to(directory)):sha(path)for path in directory.rglob('*.json')}
            write(directory/'gpu_manifest.json',dict(status='completed',fingerprint=p['fingerprint'],artifacts=artifacts,
                checkpoints={path:sha(path)for path in retained},packed_files=generation['packed_export']['files']))
            return result
    except BaseException as error:
        write(directory/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],phase=phase,
            error_type=type(error).__name__,error=str(error),lease=lease,requires_diagnosis=True,
            compressed_goal_complete=False,artifacts_preserved=True))
        raise
    finally:
        if local is not None:local.close()
        if trainer is not None:trainer.close()
        if dataset is not None:dataset.close()
        if mc_dataset is not None:mc_dataset.close()


def gpu_result(root,p):
    directory=Path(root)/'run';m=read(directory/'gpu_manifest.json');r=read(directory/'gpu_result.json')
    if (m.get('status')!='completed' or m.get('fingerprint')!=p['fingerprint']
            or r.get('fingerprint')!=p['fingerprint'] or r.get('status')!='completed' or not r.get('systems_passed')):
        raise ValueError('Completed systems/training/generation receipt required')
    for name,wanted in m['artifacts'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path)!=wanted:raise ValueError('GPU evidence changed')
    checkpoint=r['continuation_checkpoint']
    if m['checkpoints'].get(checkpoint)!=r['continuation_checkpoint_sha256'] or sha(checkpoint)!=r['continuation_checkpoint_sha256']:
        raise ValueError('Continuation checkpoint changed')
    return r


def judge(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU grading requires CUDA hidden')
    root=Path(root).resolve();p,dp,_=verify(root);r=gpu_result(root,p);directory=root/'run';out=directory/'task_judge'
    if out.exists():raise FileExistsError(out)
    out.mkdir();started=time.monotonic()
    try:
        actual=read(directory/'selected/generation.json');contexts=read(directory/'selected/contexts.json')
        native=read(Path(dp['response_cache_root'])/'native/generation.json')
        executable=base.task_metrics.grade_generated_tasks(native['corpus'],actual['corpus'],contexts,
            maximum_drop_pp=5.,minimum_student_rate=0.)
        write(out/'executable.json',executable)
        semantic=base.judge.judge_pairs(native['corpus'],actual['corpus'],out/'semantic',
            config=base.judge.JudgeConfig(**p['judge']),contexts={key:dict(prompt=v['prompt_text'],source=v['context'],
                reference=v['response'],task=v['task'])for key,v in contexts.items()})
        generation=task_probe.quality_summary(native,actual,semantic,executable)
        mc=r['selected']['mc_eligibility']['passed']
        qualified=(bool(mc and generation['diagnostic_generation_retention_pass'])
                   if generation['evidence_available'] else None)
        result=dict(status='completed',fingerprint=p['fingerprint'],cut=p['cut'],compressed_blocks=36-p['cut'],
            qualified=qualified,relative_retention_pass=qualified,quality_status=('unavailable'if qualified is None else 'pass'if qualified else 'fail'),
            generation_quality=generation,mc_retention_pass=mc,nll_diagnostic=r['selected']['nll'],nll_gate=False,
            continuation_checkpoint=r['continuation_checkpoint'],continuation_checkpoint_sha256=r['continuation_checkpoint_sha256'],
            continuation_unchanged=True,quality_failure_stops_progression=False,elapsed_seconds=time.monotonic()-started,confirmed=False)
        write(directory/'task_result.json',result)
        write(directory/'task_manifest.json',dict(status='completed',fingerprint=p['fingerprint'],
            gpu_manifest_sha256=sha(directory/'gpu_manifest.json'),artifacts={
                **{str(path.relative_to(directory)):sha(path)for path in out.rglob('*')if path.is_file()},
                'task_result.json':sha(directory/'task_result.json')}))
        return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],error_type=type(error).__name__,
            error=str(error),quality_status='unavailable',continuation_unchanged=True))
        raise


def check(root):
    root=Path(root).resolve();p,_,_=verify(root,live=False)
    if (root/'run/gpu_manifest.json').is_file():
        result=gpu_result(root,p)
        if (root/'run/task_manifest.json').is_file():
            m=read(root/'run/task_manifest.json')
            if m['gpu_manifest_sha256']!=sha(root/'run/gpu_manifest.json'):raise ValueError('GPU receipt changed before judging')
            for name,wanted in m['artifacts'].items():
                path=(root/'run'/name).resolve()
                if not path.is_relative_to(root/'run') or sha(path)!=wanted:raise ValueError('Task evidence changed')
            result=dict(result,task_result=read(root/'run/task_result.json'))
        return result
    for name in ('run/failure.json','pilot/failure.json'):
        if (root/name).is_file():return read(root/name)
    return dict(status='prepared',fingerprint=p['fingerprint'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','pilot','run','judge','check'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path);parser.add_argument('--fit-root',type=Path)
    parser.add_argument('--cut',type=int);parser.add_argument('--limit-mib',type=int,default=11008);args=parser.parse_args()
    if args.phase=='prepare':
        if any(x is None for x in (args.data_root,args.fit_root,args.cut)):parser.error('prepare requires --data-root --fit-root --cut')
        result=prepare(args.root,args.data_root,args.fit_root,args.cut)
    elif args.phase=='pilot':result=pilot(args.root,limit_mib=args.limit_mib)
    elif args.phase=='run':result=run(args.root,limit_mib=args.limit_mib)
    elif args.phase=='judge':result=judge(args.root)
    else:result=check(args.root)
    print(json.dumps({k:result[k]for k in ('status','fingerprint','cut','compressed_blocks','quality_status','continuation_checkpoint')if k in result}),flush=True)


if __name__=='__main__':main()
