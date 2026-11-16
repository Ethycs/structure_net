"""Independent task probe of an existing packed progressive checkpoint.

The original NLL-based rejection remains unchanged. GPU generation and CPU
judging are separate commands, so external judging need not hold a GPU lease.
This diagnostic cannot qualify a parent for another compression stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import shutil
import time

import torch

from . import jspace_3b_rebuild_recovery as recovery
from . import jspace_3b_response_eval as evaluation
from . import jspace_3b_response_judge as judge_api
from . import jspace_3b_response_task_metrics as task_metrics
from . import jspace_3b_packed_response_model as packed
from . import jspace_3b_packed_response_evaluate as timing

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/07 - Status Reports/jspace/2026-11-16_qwen3b-progressive-task-probe.md'
SCHEMA='jspace.3b-progressive-task-probe.v1'
read,write,sha,identity=recovery.read,recovery.write,recovery.sha,recovery.identity
digest=recovery.cache.data.digest


def admit_export(stage_root,label):
    """Authenticate the completed stage, chosen candidate, and packed payload."""
    if not re.fullmatch(r'(?:prefit|update_[0-9]{4})',label):
        raise ValueError('Explicit stage candidate label required')
    stage_root=Path(stage_root).resolve()
    source,dp,records=recovery.verify(stage_root,live=False)
    manifest=read(stage_root/'run/manifest.json');result=read(stage_root/'run/result.json')
    if (manifest.get('status')!='completed' or result.get('status')!='completed'
            or manifest.get('fingerprint')!=source['fingerprint'] or result.get('fingerprint')!=source['fingerprint']):
        raise ValueError('Completed matching source stage required')
    inputs={str(stage_root/'plan.json'):sha(stage_root/'plan.json'),
            str(stage_root/'run/manifest.json'):sha(stage_root/'run/manifest.json')}
    names=['result.json',*[f'evaluations/{label}/{name}' for name in
        ('result.json','development.json','multiple_choice.json','reload_parity.json','export_receipt.json',
         'reload_receipt.json','export/manifest.json','export/structure.json')]]
    for name in names:
        path=stage_root/'run'/name
        if manifest['artifacts'].get(name)!=sha(path):
            raise ValueError('Completed source artifact changed: '+name)
        inputs[str(path)]=sha(path)
    candidate=read(stage_root/'run/evaluations'/label/'result.json')
    if not any(c==candidate for c in result['candidates']) or candidate['label']!=label:
        raise ValueError('Candidate is absent from the completed stage result')
    export=(stage_root/'run/evaluations'/label/'export').resolve()
    if Path(candidate['packed_artifact_path']).resolve()!=export:
        raise ValueError('Packed export path differs from the admitted stage candidate')
    checkpoint=Path(candidate['checkpoint']['path']).resolve()
    if (checkpoint.parent!=stage_root/'run/checkpoints'
            or manifest['checkpoints'].get(str(checkpoint))!=candidate['checkpoint']['sha256']
            or sha(checkpoint)!=candidate['checkpoint']['sha256']):
        raise ValueError('Source checkpoint identity differs')
    inputs[str(checkpoint)]=candidate['checkpoint']['sha256']
    export_manifest=read(export/'manifest.json')
    expected=candidate['packed_export']
    if (export_manifest.get('packed') is not True
            or export_manifest.get('schema')!=packed.SCHEMA
            or export_manifest.get('packed_state_sha256')!=expected['packed_state_sha256']
            or sha(export/'manifest.json')!=expected['manifest_sha256']
            or manifest['packed_exports'].get(label)!=expected['manifest_sha256']
            or export_manifest.get('identities',{}).get('study')!=source['fingerprint']
            or export_manifest['identities'].get('checkpoint_sha256')!=candidate['checkpoint']['sha256']
            or export_manifest['identities'].get('teacher')!=source['model']['revision']
            or not candidate.get('packed_reload_passed')
            or not read(export.parent/'reload_parity.json').get('passed')):
        raise ValueError('Packed export provenance or existing same-packed replay is invalid')
    if set(export_manifest['files'])!={'structure.json','weights.safetensors'}:
        raise ValueError('Unexpected packed snapshot payloads')
    for name,wanted in export_manifest['files'].items():
        if sha(export/name)!=wanted:raise ValueError('Packed payload changed')
        inputs[str(export/name)]=wanted
    return source,dp,records,candidate,inputs


def prepare(root,stage_root,*,label='update_0046'):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation requires CUDA hidden')
    root=Path(root).resolve();stage_root=Path(stage_root).resolve()
    if root.exists():raise FileExistsError(root)
    source,dp,records,candidate,inputs=admit_export(stage_root,label)
    selected=[records[i] for i in source['generation_indices']]
    if (len(selected)!=16 or any(r.split!='development' for r in selected)
            or sorted(Counter(r.provenance['source_family'] for r in selected).values())!=[2]*8):
        raise ValueError('Original two-per-family 16-case DEV generation selection required')
    native_root=Path(dp['response_cache_root'])/'native'
    native_manifest=read(native_root/'manifest.json')
    native_plan=read(Path(dp['response_cache_root'])/'plan.json')
    if native_manifest.get('status')!='completed' or native_manifest.get('fingerprint')!=native_plan['fingerprint']:
        raise ValueError('Original native response controls are incomplete')
    for name in ('generation.json','development.json'):
        path=native_root/name
        if native_manifest['artifacts'].get(name)!=sha(path):raise ValueError('Native response control changed')
        inputs[str(path)]=sha(path)
    inputs[str(native_root/'manifest.json')]=sha(native_root/'manifest.json')
    texts_path=Path(dp['corpus'])/'text.jsonl'
    inputs[str(texts_path)]=sha(texts_path)
    texts={row['example_id']:row for row in map(json.loads,texts_path.read_text().splitlines())}
    contexts={r.example_id:texts[r.example_id] for r in selected}
    native_generation=read(native_root/'generation.json')
    if [r['example_id'] for r in native_generation['corpus']]!=[r.example_id for r in selected]:
        raise ValueError('Native generation cases differ from the prospective selection')
    # Original model/tokenizer/geometry dependencies have already been admitted
    # by recovery.verify; retain their immutable identities without bulk rereads.
    dependency_identity={**native_plan['input_identity'],**dp['input_identity'],**source['input_identity']}
    files=recovery.cache.prior.source.inherited.local_import_closure([
        Path(__file__),PROTOCOL,ROOT/'tests/structure_net/test_jspace_3b_rebuild_task_probe.py'])
    p=dict(schema=SCHEMA,stage_root=str(stage_root),source_fingerprint=source['fingerprint'],label=label,
        checkpoint=candidate['checkpoint'],packed_export=str(Path(candidate['packed_artifact_path']).resolve()),
        packed_state_sha256=candidate['packed_export']['packed_state_sha256'],model=source['model'],
        data_root=source['data_root'],data_fingerprint=source['data_fingerprint'],
        native_generation=str(native_root/'generation.json'),native_development=str(native_root/'development.json'),
        generation_indices=source['generation_indices'],generation_eos_token_ids=source['generation_eos_token_ids'],
        generation_example_ids=[r.example_id for r in selected],max_new_tokens=128,max_length=416,
        contexts=contexts,prefix_mode='resident',judge={**source['judge'],'workers':1},
        gates=dict(maximum_drop_pp=5.,minimum_generation_rate=0.),
        original_classification=dict(eligibility=candidate['eligibility'],cheap_gate=candidate['cheap_gate'],
            nll=candidate['nll'],mc_accuracy=candidate['mc_accuracy'],grading_status=candidate['grading_status']),
        selection_reason='Explicit authorized generation probe of update46; not a new checkpoint search',
        nll_shortcircuit=False,automatic_promotion=False,confirmed=False,allocation_mib=11264,limit_mib=11008,
        input_sha256=inputs,input_identity={path:identity(path) for path in inputs},
        dependency_identity=dependency_identity,source_sha256={str(path):sha(path) for path in sorted(files)})
    p['fingerprint']=digest(p);root.mkdir(parents=True);write(root/'plan.json',p)
    for path,wanted in p['source_sha256'].items():
        target=root/'producer'/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if sha(target)!=wanted:raise ValueError('Probe producer archive differs')
    return dict(status='prepared',fingerprint=p['fingerprint'],label=label,generation_cases=22,
                source_classification_unchanged=True)


def verify(root,*,live=True):
    root=Path(root);p=read(root/'plan.json');body=dict(p);fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or digest(body)!=fp:raise ValueError('Task probe plan changed')
    for path,old in {**p['dependency_identity'],**p['input_identity']}.items():
        if identity(path)!=old:raise ValueError('Task probe admitted input changed: '+path)
    for path,wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=wanted or live and sha(path)!=wanted:
            raise ValueError('Task probe producer changed: '+path)
    return p


def _event(directory,phase,**values):
    recovery._event(directory,phase,**values)


def run(root,*,limit_mib=11008):
    root=Path(root).resolve();p=verify(root);directory=root/'generation'
    if limit_mib!=p['limit_mib']:raise ValueError('Allocator differs from the prospective probe')
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir();lease=None;started=time.monotonic()
    try:
        lease,guard=recovery.cache.reservation(limit_mib)
        if next(iter(lease['allocation'].values()))!=p['allocation_mib']:
            raise ValueError('Probe requires its declared GPU allocation')
        torch.set_num_threads(2)
        _,records=recovery.data_source.verify(p['data_root'],live=False)
        with recovery.execution.arithmetic_policy():
            model,receipt=packed.load_packed_snapshot(p['packed_export'],device='cuda:0',guard=guard)
            if receipt['packed_state_sha256']!=p['packed_state_sha256']:
                raise ValueError('Probe reloaded a different packed state')
            write(directory/'reload_receipt.json',receipt)
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(p['model']['path'],local_files_only=True,trust_remote_code=False)
            _event(directory,'generation_start',label=p['label'],cases=22)
            with packed.attach_prefix(model,p['model']['path'],mode=p['prefix_mode']) as prefix:
                begin=time.monotonic()
                canaries=evaluation.evaluate_canaries(model,tokenizer,device='cuda:0',max_new_tokens=p['max_new_tokens'],
                                                      eos_token_ids=p['generation_eos_token_ids'])
                write(directory/'canaries.json',canaries)
                rows=[]
                for index in p['generation_indices']:
                    record=records[index]
                    row=evaluation.generate_record(model,record,tokenizer,device='cuda:0',
                        source_text=p['contexts'][record.example_id]['context'],max_new_tokens=p['max_new_tokens'],
                        eos_token_ids=p['generation_eos_token_ids'],max_length=p['max_length'])
                    rows.append(row)
                    with (directory/'responses.jsonl').open('a') as stream:
                        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
                    _event(directory,'generation',completed=len(rows),total=len(p['generation_indices']))
                duration=time.monotonic()-begin
                warm=timing.warm_timing(model,records[p['generation_indices'][0]].prompt_ids,warmup_calls=1,repeats=3)
            tokens=sum(len(r['output_ids']) for r in [*canaries['records'],*rows])
            result=dict(status='completed',fingerprint=p['fingerprint'],canaries=canaries,corpus=rows,
                execution='Existing packed artifact freshly structurally reloaded',use_cache=False,
                generation_seconds=duration,output_tokens_including_eos=tokens,
                cachefree_generation_tokens_per_second=tokens/duration,native_prefix=prefix,warm_timing=warm,
                timing_scope='Warm cache-free whole-prefix recomputation; not KV-cached deployment decode',
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),inference_backend=model.inference_backend,
                lease_id=lease['id'],allocation=lease['allocation'],allocator_mib=limit_mib,
                elapsed_seconds=time.monotonic()-started,source_classification_unchanged=True,
                no_training=True,automatic_promotion=False,confirmed=False)
            write(directory/'result.json',result)
            write(directory/'manifest.json',dict(status='completed',fingerprint=p['fingerprint'],
                artifacts={path.name:sha(path) for path in directory.iterdir() if path.is_file()}))
            del model;gc.collect();torch.cuda.empty_cache()
            return dict(status='generation_completed',fingerprint=p['fingerprint'],
                        canary_successes=canaries['successes'],canary_count=len(canaries['records']),
                        corpus_cases=len(rows),cachefree_generation_tokens_per_second=tokens/duration,
                        next_phase='judge on CPU with CUDA hidden; GPU lease may now end')
    except BaseException as error:
        write(directory/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],
            error_type=type(error).__name__,error=str(error),lease=lease,
            completed_raw_responses_preserved=True,automatic_retry=False))
        raise


def generation_result(root,p):
    directory=Path(root)/'generation';manifest=read(directory/'manifest.json')
    if manifest.get('status')!='completed' or manifest.get('fingerprint')!=p['fingerprint']:
        raise ValueError('Completed GPU generation receipt required')
    for name,wanted in manifest['artifacts'].items():
        if Path(name).name!=name or sha(directory/name)!=wanted:raise ValueError('Generated evidence changed')
    result=read(directory/'result.json')
    if (result.get('status')!='completed' or result.get('fingerprint')!=p['fingerprint']
            or [r['example_id'] for r in result['corpus']]!=p['generation_example_ids']):
        raise ValueError('Generated identities differ from the frozen probe')
    return result


def quality_summary(native,actual,semantic,executable,*,maximum_drop_pp=5.):
    """Keep availability distinct from failure and from the original NLL gate."""
    canary=evaluation.generation_gate(native['canaries'],actual['canaries'],
        maximum_drop_pp=maximum_drop_pp,minimum_student_rate=0.)
    gate=evaluation.semantic_gate(native['corpus'],actual['corpus'],semantic['judgments'],
        judge_provenance=semantic['judge_provenance'],maximum_drop_pp=maximum_drop_pp,minimum_student_rate=0.)
    valid=[r for r in semantic['judgments'] if r.get('status')=='completed' and all(
        type(r.get(arm,{}).get(key)) is bool for arm in ('native','student')
        for key in ('task_success','generation_success'))]
    observed={key:dict(native=sum(r['native'][key] for r in valid),student=sum(r['student'][key] for r in valid),
        count=len(valid),complete=len(valid)==len(semantic['judgments'])) for key in ('task_success','generation_success')}
    available=not gate.get('missing_or_failed_judgments',False) and executable['available']
    passed=canary['passed'] and gate['passed'] and executable['passed']
    return dict(status='complete' if available else 'incomplete_evidence',evidence_available=available,
        diagnostic_generation_retention_pass=passed if available else None,
        canary_gate=canary,semantic_gate=gate,semantic_observed=observed,executable_gates=executable['gates'],
        automatic_promotion=False,original_stage_eligibility_unchanged=True,confirmed=False,
        scope='Small paired DEV generation diagnostic; no population accuracy or continuation approval')


def judge(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU judge phase requires CUDA hidden')
    root=Path(root).resolve();p=verify(root);directory=root/'judgment'
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir();started=time.monotonic()
    try:
        actual=generation_result(root,p);native=read(p['native_generation'])
        executable=task_metrics.grade_generated_tasks(native['corpus'],actual['corpus'],p['contexts'],
            maximum_drop_pp=p['gates']['maximum_drop_pp'],minimum_student_rate=0.)
        write(directory/'executable.json',executable)
        contexts={key:dict(prompt=row['prompt_text'],source=row['context'],reference=row['response'],task=row['task'])
                  for key,row in p['contexts'].items()}
        _event(directory,'semantic_judge',pairs=len(actual['corpus']),workers=1)
        semantic=judge_api.judge_pairs(native['corpus'],actual['corpus'],directory/'semantic',
                                       config=judge_api.JudgeConfig(**p['judge']),contexts=contexts)
        quality=quality_summary(native,actual,semantic,executable,maximum_drop_pp=p['gates']['maximum_drop_pp'])
        result=dict(status='completed' if quality['evidence_available'] else 'completed_with_unavailable_evidence',
            fingerprint=p['fingerprint'],label=p['label'],checkpoint=p['checkpoint'],
            packed_state_sha256=p['packed_state_sha256'],quality=quality,
            original_classification=p['original_classification'],original_classification_unchanged=True,
            generation=dict(canary_successes=actual['canaries']['successes'],canary_count=len(actual['canaries']['records']),
                corpus_cases=len(actual['corpus']),seconds=actual['generation_seconds'],
                cachefree_generation_tokens_per_second=actual['cachefree_generation_tokens_per_second']),
            semantic_status=semantic['status'],elapsed_seconds=time.monotonic()-started,
            no_training=True,automatic_promotion=False,confirmed=False)
        write(directory/'result.json',result)
        write(directory/'manifest.json',dict(status='completed',fingerprint=p['fingerprint'],
            generation_manifest_sha256=sha(root/'generation/manifest.json'),
            artifacts={str(path.relative_to(directory)):sha(path) for path in directory.rglob('*') if path.is_file()}))
        return result
    except BaseException as error:
        write(directory/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],
            error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise


def check(root):
    root=Path(root).resolve();p=verify(root,live=False)
    if (root/'judgment/manifest.json').is_file():
        m=read(root/'judgment/manifest.json')
        if (m.get('status')!='completed' or m.get('fingerprint')!=p['fingerprint']
                or m['generation_manifest_sha256']!=sha(root/'generation/manifest.json')):
            raise ValueError('Judgment completion receipt differs')
        for name,wanted in m['artifacts'].items():
            path=(root/'judgment'/name).resolve()
            if not path.is_relative_to(root/'judgment') or sha(path)!=wanted:raise ValueError('Judgment evidence changed')
        return read(root/'judgment/result.json')
    for phase in ('generation','judgment'):
        if (root/phase/'failure.json').is_file():return read(root/phase/'failure.json')
    if (root/'generation/manifest.json').is_file():
        generation_result(root,p);return dict(status='generation_completed',fingerprint=p['fingerprint'])
    return dict(status='prepared',fingerprint=p['fingerprint'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','run','judge','check'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage-root',type=Path);parser.add_argument('--label',default='update_0046')
    parser.add_argument('--limit-mib',type=int,default=11008);args=parser.parse_args()
    if args.phase=='prepare':
        if args.stage_root is None:parser.error('prepare requires --stage-root')
        result=prepare(args.root,args.stage_root,label=args.label)
    elif args.phase=='run':result=run(args.root,limit_mib=args.limit_mib)
    elif args.phase=='judge':result=judge(args.root)
    else:result=check(args.root)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
