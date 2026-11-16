"""Actual packed all36 evaluation, independent from basis/weight selection.

Initial compilation controls measure the searched-to-folded numerical change.
Final controls always generate before a separate CPU judge. NLL is diagnostic;
unavailable judgments stay unavailable. All timings use full-prefix recompute.
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

import torch

from . import jspace_3b_ablation32_campaign as campaign
from . import jspace_3b_ablation32_compile as compiler
from . import jspace_3b_ablation32_data as data
from . import jspace_3b_ablation32_distill as distill
from . import jspace_3b_packed_response_model as packed
from . import jspace_3b_packed_response_evaluate as timing
from . import jspace_3b_rebuild_recovery as base
from . import jspace_3b_rebuild_task_probe as probe
from . import jspace_qwen3b_progressive_model as inventory

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='jspace.ablation32-evaluation.v1'
read,write,sha,identity=data.read,data.write,data.sha,data.identity


def selected_blocks(result,*,arm,compiled_root,model_revision,data_fingerprint,layers=36):
    if result is None:return []
    if (result.get('status')!='completed' or result.get('arm')!=arm or result.get('layers')!=layers
            or Path(result.get('compiled_root','')).resolve()!=Path(compiled_root).resolve()
            or result.get('model_revision')!=model_revision or result.get('data_fingerprint')!=data_fingerprint):
        raise ValueError('Completed matching all-layer arm result required')
    blocks=result.get('selected_blocks',[])
    if [b.get('layer')for b in blocks]!=list(range(layers)):
        raise ValueError('Every selected block is required exactly once in native order')
    for block in blocks:
        if sha(block['checkpoint'])!=block['sha256']:raise ValueError('Selected block checkpoint changed')
        value=torch.load(block['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        if (value.get('schema')!='jspace.ablation32-stage-checkpoint.v1' or value.get('layer')!=block['layer']
                or not isinstance(value.get('block'),dict) or not value['block']
                or value.get('norm')is not None and block['layer']!=layers-1):
            raise ValueError('Selected block payload or final norm belongs to another stage')
    return blocks


def prepare(root,campaign_root,compiled_root,arm_result=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Evaluation preparation requires CUDA hidden')
    root,campaign_root,compiled_root=map(lambda x:Path(x).resolve(),(root,campaign_root,compiled_root))
    if root.exists():raise FileExistsError(root)
    cp,dp,records=campaign.verify(campaign_root,live=False)
    campaign._completed(campaign_root/'search',cp)
    cm=read(compiled_root/'manifest.json');geometry=read(Path(cm['geometry_root'])/'manifest.json')
    if (cm.get('status')!='completed' or cm.get('layers')!=36 or cm.get('arm')not in('learned','random')
            or cm.get('model_path')!=cp['model']['path'] or sha(compiled_root/'compiled.pt')!=cm['checkpoint_sha256']
            or sha(Path(cm['geometry_root'])/'manifest.json')!=cm['geometry_manifest_sha256']
            or geometry.get('identities')!=campaign._identities(cp)):
        raise ValueError('Compiled model must derive from this complete search campaign')
    result_path=Path(arm_result).resolve()if isinstance(arm_result,(str,Path))else None
    result=read(result_path)if result_path else arm_result
    blocks=selected_blocks(result,arm=cm['arm'],compiled_root=compiled_root,
        model_revision=dp['model']['revision'],data_fingerprint=dp['fingerprint'])
    stage_inputs=[]
    if result is not None:
        final_root=Path(blocks[-1]['checkpoint']).resolve().parents[2]
        authenticated=distill.check(final_root)
        if any(authenticated.get(k)!=result.get(k)for k in
                ('status','arm','layers','compiled_root','selected_blocks','model_revision','data_fingerprint')):
            raise ValueError('Final arm result differs from its authenticated completed stage')
        stage_inputs.extend([final_root/'run/result.json',final_root/'run/manifest.json'])
        for block in blocks:
            stage_root=Path(block['checkpoint']).resolve().parents[2];stage=read(stage_root/'plan.json')
            body=dict(stage);fingerprint=body.pop('fingerprint')
            if (data.cache.data.digest(body)!=fingerprint or stage.get('campaign_fingerprint')!=cp['fingerprint']
                    or stage.get('data_fingerprint')!=dp['fingerprint'] or stage.get('model')!=dp['model']
                    or stage.get('compiled_root')!=str(compiled_root)or stage.get('layer')!=block['layer']
                    or stage.get('arm')!=cm['arm']):raise ValueError('Selected stage plan belongs to another study')
            value=torch.load(block['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
            expected=dict(fingerprint=fingerprint,arm=cm['arm'],layer=block['layer'],compiled_root=str(compiled_root))
            if value.get('identities')!=expected:raise ValueError('Selected block is not bound to its authenticated stage')
            stage_inputs.append(stage_root/'plan.json')
    reference=geometry['projection_reference'];native=Path(dp['response_cache_root'])/'native'
    native_manifest=read(native/'manifest.json')
    paths=[campaign_root/'plan.json',compiled_root/'manifest.json',compiled_root/'compiled.pt',
        Path(cm['geometry_root'])/'manifest.json',Path(reference['path']),native/'manifest.json',
        native/'development.json',native/'generation.json',Path(dp['corpus'])/'text.jsonl',
        *[Path(b['checkpoint'])for b in blocks],*stage_inputs]
    if result_path:paths.append(result_path)
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Searched model reference changed')
    for name in ('development.json','generation.json'):
        if sha(native/name)!=native_manifest['artifacts'][name]:raise ValueError('Original native control changed')
    dev=[i for i,r in enumerate(records)if r.split=='development']
    if len(dev)!=128 or len(dp['generation_indices'])!=16:raise ValueError('The fixed DEV128/generation16 inventory is required')
    sources=data.cache.prior.source.inherited.local_import_closure([
        Path(__file__),data.PROTOCOL,ROOT/'tests/structure_net/test_jspace_3b_ablation32_evaluate.py'])
    p=dict(schema=SCHEMA,campaign_root=str(campaign_root),campaign_fingerprint=cp['fingerprint'],
        data_root=cp['data_root'],data_fingerprint=dp['fingerprint'],model=dp['model'],
        corpus=dp['corpus'],full_data_root=dp['full_data_root'],response_cache_root=dp['response_cache_root'],
        compiled_root=str(compiled_root),arm=cm['arm'],mode='initial_compiled'if result is None else'final_distilled',
        arm_result=result,selected_blocks=blocks,projection_reference=reference,
        development_indices=dev,generation_indices=dp['generation_indices'],
        generation_eos_token_ids=dp['generation_eos_token_ids'],max_new_tokens=128,
        judge=asdict(base.judge.JudgeConfig(workers=1,expected_returned_model='codex_sdk/gpt-5.6-sol')),
        judge_lock=str(Path(cp['data_root']).parent/'.cpu_judge.lock'),allocation_mib=7680,limit_mib=7424,
        input_sha256={str(f):sha(f)for f in paths},input_identity={str(f):identity(f)for f in paths},
        dependency_identity=dp['input_identity'],source_sha256={str(f):sha(f)for f in sources},
        nll_gate=False,confirmed=False)
    p['fingerprint']=data.cache.data.digest(p);root.mkdir(parents=True);write(root/'plan.json',p)
    for source,wanted in p['source_sha256'].items():
        target=root/'producer'/Path(source).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if sha(target)!=wanted:raise ValueError('Evaluation source archive differs')
    return dict(status='prepared',fingerprint=p['fingerprint'],mode=p['mode'],arm=p['arm'])


def verify(root,*,live=True):
    root=Path(root).resolve();p=read(root/'plan.json');body=dict(p);fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or data.cache.data.digest(body)!=fp:raise ValueError('Evaluation plan changed')
    for name,old in {**p['dependency_identity'],**p['input_identity']}.items():
        if identity(name)!=old:raise ValueError('Evaluation input changed: '+name)
    for name,wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(name).relative_to(ROOT))!=wanted or live and sha(name)!=wanted:
            raise ValueError('Evaluation producer changed: '+name)
    for name,wanted in p['input_sha256'].items():
        if Path(name).suffix in('.json','.jsonl')and sha(name)!=wanted:raise ValueError('Evaluation metadata changed')
    dp,records=data.verify(p['data_root'],live=False)
    if dp['fingerprint']!=p['data_fingerprint']or dp['model']!=p['model']:raise ValueError('Evaluation teacher/data differs')
    return p,dp,records


def build_model(p,*,device='cpu',guard=None):
    stack,receipt=compiler.load_compiled(p['compiled_root'])
    cfg=compiler.native._config(p['model']['path'])
    E,norm,rotary=compiler.native.load_readout_norm(p['model']['path'],device='cpu',config=cfg)
    for block in selected_blocks(p.get('arm_result'),arm=p['arm'],compiled_root=p['compiled_root'],
            model_revision=p['model']['revision'],data_fingerprint=p['data_fingerprint'],layers=cfg.num_hidden_layers):
        compiler.restore_selected_block(stack,block['checkpoint'],expected_sha256=block['sha256'],norm=norm)
    compact=compiler.pack_preserving_geometry(stack);del stack;gc.collect()
    if guard is not None:guard()
    model=packed.PackedResponseModel(compact.to(device),norm.to(device),E.to(device),rotary.to(device),
        config=cfg,max_length=416,guard=guard)
    if model.cut!=0:raise ValueError('Evaluation must execute all blocks from original embeddings')
    return model


class EmbeddingOnlyMC:
    """Keep MC provenance while forcing actual cut0 embedding execution."""
    def __init__(self,source):self.source=source
    def __getattr__(self,name):return getattr(self.source,name)
    def batch(self,indices):
        batch=dict(self.source.batch(indices));batch.pop('boundary_hidden',None);return batch


def native_header_bytes(model_path):
    total=0;seen=set()
    for path in sorted(Path(model_path).glob('*.safetensors')):
        with path.open('rb')as stream:header=json.loads(stream.read(int.from_bytes(stream.read(8),'little')))
        for name,item in header.items():
            if name=='__metadata__'or name=='lm_head.weight'or name in seen:continue
            if item['dtype']!='BF16':raise ValueError('Expected pinned original BF16 storage')
            seen.add(name);total+=math.prod(item['shape'])*2
    if not {'model.embed_tokens.weight','model.norm.weight'}<=seen:raise ValueError('Incomplete original tensor headers')
    return total


def storage_report(model,model_path,export):
    actual=inventory.tensor_bytes(model);native=native_header_bytes(model_path)
    rotary=inventory.tensor_bytes(model.rotary)
    native+=rotary
    return dict(packed_resident_tensor_bytes=actual,native_resident_tensor_bytes=native,
        reduction_fraction=1-actual/native,serialized_bytes=export['serialized_bytes'],
        fixed_geometry_bytes=sum(v.numel()*v.element_size()for v in model.stack.buffers()if v.is_floating_point()),
        tied_embedding_copies=1,geometry_precision=model.stack.geometry_mode,
        scope='Actual packed tensor storages, including FP32 geometry; original BF16 headers plus RoPE, no KV or activations')


def difference(actual,expected):
    a,b=actual.float().cpu(),expected.float().cpu()
    if a.shape!=b.shape or not torch.isfinite(a).all()or not torch.isfinite(b).all():raise ValueError('Aligned finite reference outputs required')
    delta=a-b
    return dict(bit_exact=torch.equal(actual.cpu(),expected.cpu()),max_abs=float(delta.abs().max()),
        relative_rms=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30)))


@torch.inference_mode()
def projection_comparison(model,reference,records,selection_indices):
    rows=[]
    for item in reference['records']:
        index=item['index'];record=records[index]
        if (index not in selection_indices or record.example_id!=item['example_id']
                or record.group_id!=item['group_id'] or record.split!='train'
                or item['input_ids'].tolist()!=[list(record.input_ids)]):
            raise ValueError('Projection reference tokens or selection identities differ')
        hidden=model.hidden_forward(dict(input_ids=item['input_ids']))
        pos=item['last_prediction_position'];logits=model.readout(hidden[:,pos]).cpu()
        rows.append(dict(index=index,example_id=record.example_id,family=item['family'],
            hidden=difference(hidden,item['normalized_hidden']),logits=difference(logits,item['last_prediction_logits']),
            same_last_greedy=int(logits.argmax())==int(item['last_prediction_logits'].argmax())))
    return dict(records=rows,count=len(rows),last_greedy_agreement=sum(r['same_last_greedy']for r in rows)/len(rows),
        checkpoint=reference['checkpoint'],diagnostic_only=True,hard_numerical_gate=False,
        scope='Actual searched projected native versus packed untrained compiled execution; includes BF16/folding rounding')


class FixedTimingSeed:
    """Fix only warm_timing's untimed prefix-extension choice; all timed calls are real."""
    def __init__(self,model,token):self.model,self.token,self.first=model,token,True;self.embedding=model.embedding
    def next_logits(self,ids):
        if self.first:
            self.first=False;value=torch.zeros(self.embedding.shape[0],device=self.embedding.device);value[self.token]=1;return value
        return self.model.next_logits(ids)


def matched_warm(model,prompt_ids,token):
    result=timing.warm_timing(FixedTimingSeed(model,token),prompt_ids,warmup_calls=2,repeats=5)
    return dict(result,fixed_extension_token=token,prompt_ids=list(prompt_ids),identical_timed_inputs=True)


def _reservation(limit_mib):
    if limit_mib!=7424:raise ValueError('Fixed evaluation allocator cap7424MiB required')
    lease,guard=data.cache.reservation(limit_mib)
    if (len(lease['allocation'])!=1 or not next(iter(lease['allocation'])).startswith(campaign.TEACHER_PREFIX)
            or next(iter(lease['allocation'].values()))!=7680):raise ValueError('Single reserved2060SUPER allocation7680MiB required')
    return lease,guard


@torch.inference_mode()
def run(root,*,limit_mib=7424):
    root=Path(root).resolve();p,dp,records=verify(root);out=root/'run'
    if out.exists():raise FileExistsError(out)
    out.mkdir();started=time.monotonic();dataset=mc_dataset=None;lease=None
    try:
        lease,guard=_reservation(limit_mib);torch.set_num_threads(2)
        dataset=data.SearchDataset(p['data_root'],require_anchors=False)
        mc_dataset=data.banks.MCDataset(p['full_data_root'],cut=0)
        prompt=records[p['generation_indices'][0]].prompt_ids
        with base.model_source.execution.arithmetic_policy():
            model=build_model(p,device='cuda:0',guard=guard)
            export=packed.export_packed_snapshot(model,out/'export',identities=dict(evaluation=p['fingerprint'],arm=p['arm']))
            before=timing.packed_probe(model,prompt);storage=storage_report(model,p['model']['path'],export)
            del model;gc.collect();torch.cuda.empty_cache();guard()
            model,reload=packed.load_packed_snapshot(out/'export',device='cuda:0',guard=guard)
            after=timing.packed_probe(model,prompt,prefixes=[r['input_ids']for r in before])
            parity=timing.reload_parity(before,after);write(out/'reload.json',dict(parity,export=export,reload=reload))
            teacher=base.training.CachedHiddenTeacher(model.embedding,teacher_id=p['model']['revision'],guard=guard)
            by_id={r.example_id:i for i,r in enumerate(records)}
            response=base.evaluation.response_losses(model,(records[i]for i in p['development_indices']),
                pad_token_id=dataset.pad_token_id,max_length=416,teacher_provider=teacher,token_chunk_size=8,
                prepare_batch=lambda b:dataset.batch([by_id[name]for name in b['example_ids']]))
            write(out/'development.json',response)
            mc=base.packed_mc(model,EmbeddingOnlyMC(mc_dataset));write(out/'multiple_choice.json',mc)
            if p['mode']=='initial_compiled':
                reference=torch.load(p['projection_reference']['path'],map_location='cpu',weights_only=True)
                control=projection_comparison(model,reference,records,dp['selection_indices'])
                write(out/'projection_comparison.json',control);del reference
            else:
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(p['model']['path'],local_files_only=True,trust_remote_code=False)
                contexts={row['example_id']:row for row in map(json.loads,(Path(p['corpus'])/'text.jsonl').read_text().splitlines())}
                begin=time.monotonic();canaries=base.evaluation.evaluate_canaries(model,tokenizer,device='cuda:0',
                    eos_token_ids=p['generation_eos_token_ids'],max_new_tokens=128)
                write(out/'canaries.json',canaries);generated=[]
                for index in p['generation_indices']:
                    record=records[index];row=base.evaluation.generate_record(model,record,tokenizer,device='cuda:0',
                        source_text=contexts[record.example_id]['context'],eos_token_ids=p['generation_eos_token_ids'],max_new_tokens=128)
                    generated.append(row)
                    with(out/'responses.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
                write(out/'generation.json',dict(canaries=canaries,corpus=generated,use_cache=False,
                    generation_seconds=time.monotonic()-begin,timing_scope='Full-prefix recomputation; no KV decode claim'))
                write(out/'contexts.json',{r['example_id']:contexts[r['example_id']]for r in generated})
            extension=int(before[0]['logits'].argmax());compact_time=matched_warm(model,prompt,extension)
            packed_peak=torch.cuda.max_memory_reserved();del model,teacher,before,after;gc.collect();torch.cuda.empty_cache();guard()
            if torch.cuda.memory_allocated()>16*1024**2:raise MemoryError('Student GPU storage remains before native timing')
            torch.cuda.reset_peak_memory_stats();native=base.model_source.execution.load_native_model(p['model']['path'],device='cuda:0',attention='eager',guard=guard)
            adapter=data.cache.NativeModel(native,guard);adapter.embedding=native.model.embed_tokens.weight
            native_time=matched_warm(adapter,prompt,extension);native_peak=torch.cuda.max_memory_reserved()
            del adapter,native;gc.collect();torch.cuda.empty_cache()
        times=dict(packed=compact_time,native=native_time,matched_same_device=True,native_attention='eager',
            speedups={k:native_time[k]['median_ms']/compact_time[k]['median_ms']for k in
                ('full_prompt_next_logit','extended_prefix_next_logit')},scope='Warm full-prefix recomputation, not KV-cached TOK/S')
        write(out/'timing.json',times);write(out/'storage.json',storage)
        native_response=read(Path(p['response_cache_root'])/'native/development.json')
        result=dict(status='completed',fingerprint=p['fingerprint'],arm=p['arm'],mode=p['mode'],layers=36,
            systems_passed=True,packed_reload_exact=parity['passed'],packed_export=export,
            response_nll=response['nll'],native_response_nll=native_response['nll'],nll_gate=False,
            mc=mc['metrics'],mc_eligibility=mc['eligibility'],storage=storage,timing=times,
            quality_status='not_assessed_initial_control'if p['mode']=='initial_compiled'else'pending',
            qualified=None,lease_id=lease['id'],peak_reserved_bytes=dict(packed=packed_peak,native=native_peak),
            elapsed_seconds=time.monotonic()-started,confirmed=False)
        write(out/'result.json',result);campaign._manifest(out,p,[f for f in out.rglob('*')if f.is_file()]);return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],error_type=type(error).__name__,
            error=str(error),lease=lease,partial_artifacts_preserved=True));raise
    finally:
        if dataset is not None:dataset.close()
        if mc_dataset is not None:mc_dataset.close()


def judge(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU judging requires CUDA hidden')
    root=Path(root).resolve();p,_,_=verify(root);gpu=campaign._completed(root/'run',p)
    if p['mode']!='final_distilled':raise ValueError('Initial compilation is a numerical control, not final generation qualification')
    out=root/'judge'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    with Path(p['judge_lock']).open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            actual=read(root/'run/generation.json');contexts=read(root/'run/contexts.json')
            native=read(Path(p['response_cache_root'])/'native/generation.json')
            executable=base.task_metrics.grade_generated_tasks(native['corpus'],actual['corpus'],contexts,
                maximum_drop_pp=5.,minimum_student_rate=0.);write(out/'executable.json',executable)
            semantic=base.judge.judge_pairs(native['corpus'],actual['corpus'],out/'semantic',
                config=base.judge.JudgeConfig(**p['judge']),contexts={key:dict(prompt=v['prompt_text'],source=v['context'],
                    reference=v['response'],task=v['task'])for key,v in contexts.items()})
            quality=probe.quality_summary(native,actual,semantic,executable)
            qualified=bool(gpu['mc_eligibility']['passed']and quality['diagnostic_generation_retention_pass'])if quality['evidence_available']else None
            result=dict(status='completed',fingerprint=p['fingerprint'],arm=p['arm'],layers=36,
                qualified=qualified,relative_retention_pass=qualified,quality_status='unavailable'if qualified is None else'pass'if qualified else'fail',
                generation_quality=quality,mc_retention_pass=gpu['mc_eligibility']['passed'],
                response_nll_diagnostic=gpu['response_nll'],nll_gate=False,confirmed=False)
        except Exception as error:
            result=dict(status='completed',fingerprint=p['fingerprint'],arm=p['arm'],layers=36,qualified=None,
                relative_retention_pass=None,quality_status='unavailable',error_type=type(error).__name__,error=str(error),confirmed=False)
        write(out/'result.json',result);campaign._manifest(out,p,[f for f in out.rglob('*')if f.is_file()]);return result


def check(root):
    p,_,_=verify(root,live=False);result=dict(status='prepared',fingerprint=p['fingerprint'],mode=p['mode'])
    if(Path(root)/'run/manifest.json').exists():result=campaign._completed(Path(root)/'run',p)
    if(Path(root)/'judge/manifest.json').exists():result=campaign._completed(Path(root)/'judge',p)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('prepare','run','judge','check'))
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--campaign-root',type=Path)
    parser.add_argument('--compiled-root',type=Path);parser.add_argument('--arm-result',type=Path)
    parser.add_argument('--limit-mib',type=int,default=7424);args=parser.parse_args()
    if args.phase=='prepare':
        if args.campaign_root is None or args.compiled_root is None:parser.error('Campaign and compiled arm roots required')
        result=prepare(args.root,args.campaign_root,args.compiled_root,args.arm_result)
    elif args.phase=='run':result=run(args.root,limit_mib=args.limit_mib)
    elif args.phase=='judge':result=judge(args.root)
    else:result=check(args.root)
    print(json.dumps({k:result[k]for k in('status','fingerprint','mode','arm','qualified','quality_status')if k in result}))


if __name__=='__main__':main()
