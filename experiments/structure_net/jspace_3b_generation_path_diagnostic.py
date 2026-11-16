"""Fixed-history native/unpacked/packed generation-path audit, without training.

Original prefix weights stay on CPU and execute one layer at a time on the
reserved GPU. All compared suffixes receive the SAME actual native boundary.
Turing explicitly lacks the Ampere BF16 Triton path; stored Ampere choices are
a cross-device diagnostic, never proof of a Triton defect.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from . import jspace_3b_full_progressive_recovery as recovery
from . import jspace_3b_packed_response_model as packed
from . import jspace_3b_response_train as training

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='jspace.generation-path-diagnostic.v1'
read,write,sha,identity=recovery.read,recovery.write,recovery.sha,recovery.identity
digest=recovery.digest
STEPS=(0,1,2,4,8,16)


def histories(student,native):
    """Four predeclared tasks and fixed continuation lengths, plus native EOS."""
    chosen=[next(r for r in student['canaries']['records']if r['id']==name)
            for name in ('addition','replace','invoice')]
    chosen.append(next(r for r in student['corpus']if r['example_id'].startswith('winogrande:')
        and r['example_id'].endswith(':1336')))
    original={r.get('id',r.get('example_id')):r for r in [*native['canaries']['records'],*native['corpus']]}
    result=[]
    for record in chosen:
        key=record.get('id',record.get('example_id'));teacher=original[key]
        if record['input_ids']!=teacher['input_ids']:raise ValueError('Native/student prompt IDs differ')
        for step in STEPS:
            if step>=len(record['output_ids']):continue
            prefix=record['input_ids']+record['output_ids'][:step]
            result.append(dict(case=key,history='stored_student',step=step,input_ids=prefix,
                stored_next_token=record['output_ids'][step],original_prompt_ids=record['input_ids']))
        if teacher['stop_reason']!='eos':raise ValueError('Chosen native EOS sanity examples must have terminated')
        result.append(dict(case=key,history='native_completion',step=len(teacher['output_ids'])-1,
            input_ids=teacher['input_ids']+teacher['output_ids'][:-1],stored_next_token=teacher['output_ids'][-1],
            original_prompt_ids=record['input_ids']))
    if any(not r['input_ids']or len(r['input_ids'])>416 for r in result):raise ValueError('Exact prefixes must fit the original envelope')
    return result


def logit_summary(logits,eos_ids,*,expected=None,top_k=5):
    value=logits.detach().float().cpu()
    if value.ndim!=1 or not torch.isfinite(value).all():raise ValueError('Finite complete vocabulary required')
    logp=value.log_softmax(-1);top=value.topk(min(top_k,len(value)))
    def token_row(token):
        return dict(token_id=int(token),logit=float(value[token]),probability=float(logp[token].exp()),
            rank=int((value>value[token]).sum())+1)
    return dict(vocabulary=len(value),greedy_token=int(value.argmax()),greedy_is_eos=int(value.argmax())in eos_ids,
        top_two_logit_margin=float(top.values[0]-top.values[1])if len(top.values)>1 else None,
        eos=[token_row(t)for t in eos_ids],eos_mass=float(logp[list(eos_ids)].exp().sum()),
        expected=token_row(expected)if expected is not None else None,
        top=[token_row(t)for t in top.indices.tolist()])


def compare_logits(actual,reference):
    a,b=actual.detach().float().cpu(),reference.detach().float().cpu()
    if a.ndim!=1 or a.shape!=b.shape or not(torch.isfinite(a).all()and torch.isfinite(b).all()):
        raise ValueError('Aligned finite full-vocabulary vectors required')
    delta=a-b;lp,lq=b.log_softmax(-1),a.log_softmax(-1)
    return dict(bit_exact=torch.equal(actual.cpu(),reference.cpu()),greedy_equal=int(a.argmax())==int(b.argmax()),
        maximum_absolute_error=float(delta.abs().max()),relative_rms_error=float(delta.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30)),
        reference_to_actual_kl=float((lp.exp()*(lp-lq)).sum()))


def prepare(root,recovery_root,checkpoint,*,profile='plain2060'):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation requires CUDA hidden')
    root,recovery_root,checkpoint=map(lambda p:Path(p).resolve(),(root,recovery_root,checkpoint))
    if root.exists():raise FileExistsError(root)
    p,dp,records=recovery.verify(recovery_root,live=False);completed=recovery.gpu_result(recovery_root,p)
    candidates=[r for r in completed['candidates']if Path(r['checkpoint']['path']).resolve()==checkpoint]
    if len(candidates)!=1:raise ValueError('One retained measured candidate checkpoint required')
    candidate=candidates[0]
    if sha(checkpoint)!=candidate['checkpoint']['sha256']:raise ValueError('Candidate weights changed')
    if profile not in ('plain2060','3060'):raise ValueError('Explicit physical resource profile required')
    allocation,limit,gpu=(3584,3328,'GPU-8cf440df-')if profile=='plain2060'else(11264,11008,'GPU-')
    selected=recovery_root/'run/selected';response_root=Path(dp['response_cache_root'])
    student=read(selected/'generation.json');native=read(response_root/'native/generation.json')
    snapshots=histories(student,native)
    inputs={str(recovery_root/'plan.json'):sha(recovery_root/'plan.json'),
        str(recovery_root/'run/gpu_manifest.json'):sha(recovery_root/'run/gpu_manifest.json'),
        str(recovery_root/'run/gpu_result.json'):sha(recovery_root/'run/gpu_result.json'),
        str(selected/'generation.json'):sha(selected/'generation.json'),str(checkpoint):sha(checkpoint),
        str(response_root/'native/generation.json'):sha(response_root/'native/generation.json')}
    for path in (selected/'export').iterdir():
        if path.is_file():inputs[str(path)]=sha(path)
    sources=recovery._source_files()|recovery.base.cache.prior.source.inherited.local_import_closure([
        Path(__file__),ROOT/'tests/structure_net/test_jspace_3b_generation_path_diagnostic.py'])
    plan=dict(schema=SCHEMA,recovery_root=str(recovery_root),recovery_fingerprint=p['fingerprint'],
        source_plan=p,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),candidate=candidate,
        cut=p['cut'],model=p['model'],data_root=p['data_root'],histories=snapshots,
        stored_generation_checkpoint_sha256=student['selected_checkpoint']['sha256'],
        same_checkpoint_as_stored_generation=sha(checkpoint)==student['selected_checkpoint']['sha256'],
        rotary_export=str(selected/'export'),generation_eos_token_ids=p['generation_eos_token_ids'],
        native_control_index=p['generation_indices'][0],profile=profile,allocation_mib=allocation,limit_mib=limit,gpu_prefix=gpu,
        prefix_mode='cpu',prefix_math='Original BF16 CUDA layerwise execution; CPU-resident weights',
        original_report_changed=False,training=False,hard_teacher_student_equality_gate=False,
        input_sha256=inputs,input_identity={path:identity(path)for path in inputs},
        dependency_identity={**p['input_identity'],**p['dependency_identity']},
        source_sha256={str(path):sha(path)for path in sorted(sources)},confirmed=False)
    plan['fingerprint']=digest(plan);root.mkdir(parents=True);write(root/'plan.json',plan)
    for path,wanted in plan['source_sha256'].items():
        target=root/'producer'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if sha(target)!=wanted:raise ValueError('Diagnostic source archive differs')
    return dict(status='prepared',fingerprint=plan['fingerprint'],histories=len(snapshots),profile=profile)


def verify(root,*,live=True):
    root=Path(root).resolve();p=read(root/'plan.json');body=dict(p);fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or digest(body)!=fp:raise ValueError('Diagnostic plan changed')
    for path,old in {**p['dependency_identity'],**p['input_identity']}.items():
        if identity(path)!=old:raise ValueError('Diagnostic input changed: '+path)
    for path,wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=wanted or live and sha(path)!=wanted:
            raise ValueError('Diagnostic producer changed: '+path)
    return p


def _reservation(p):
    lease,guard=recovery.base.cache.reservation(p['limit_mib']);cards=list(lease['allocation'])
    if (len(cards)!=1 or lease['allocation'][cards[0]]!=p['allocation_mib']
            or not cards[0].startswith(p['gpu_prefix'])or os.environ.get('CUDA_VISIBLE_DEVICES')!=cards[0]):
        raise ValueError('Physical GPU or reservation differs from diagnostic profile')
    if p['profile']=='3060'and '3060'not in torch.cuda.get_device_name(0):raise ValueError('3060 profile requires the3060')
    return lease,guard


def build_models(p,*,device,guard,out):
    original=p['source_plan'];builder=recovery.base.builder
    stack,norm,E,old_rope,proof=builder.build_stack(p['model']['path'],
        {int(k):v for k,v in original['fits'].items()},p['cut'],previous=p['checkpoint'],device='cpu',trainable=False)
    del old_rope
    template,_=packed.load_packed_snapshot(p['rotary_export'],device='cpu')
    if not torch.equal(E,template.embedding):raise ValueError('Original tied vocabulary differs from exported model')
    rotary=deepcopy(template.rotary);del template;gc.collect()
    cfg=builder.native._config(p['model']['path'])
    exported_model=packed.PackedResponseModel(builder.packed_clone(stack,device='cpu'),deepcopy(norm),E,deepcopy(rotary),config=cfg)
    export=packed.export_packed_snapshot(exported_model,out/'export',identities=dict(study=p['fingerprint'],checkpoint=p['checkpoint_sha256']))
    if export['packed_state_sha256']!=p['candidate']['packed_state_sha256']:
        raise RuntimeError('Repacked candidate differs from its originally measured packed state')
    del exported_model;gc.collect()
    compact,reload=packed.load_packed_snapshot(out/'export',device='cpu')
    guard();shared_E=E.to(device);unpacked=training.ResponseModel(stack.to(device),norm.to(device),shared_E,
        rotary.to(device),max_length=416,expected_layers=cfg.num_hidden_layers,guard=guard).eval().requires_grad_(False)
    compact.embedding=shared_E;compact.guard=guard;compact.to(device)
    compact.stack.enabled=False;compact.inference_backend='packed_bf16_torch'
    native_blocks=[builder.native.load_native_block(p['model']['path'],i,device=device,config=cfg)
        for i in range(p['cut'],cfg.num_hidden_layers)]
    _,native_norm,native_rope=builder.native.load_readout_norm(p['model']['path'],device='cpu',config=cfg)
    native_norm,native_rope=native_norm.to(device),native_rope.to(device)
    return unpacked,compact,native_blocks,native_norm,native_rope,dict(export=export,reload=reload,source=proof)


@torch.no_grad()
def original_suffix(boundary,ids,blocks,norm,rotary):
    pos=torch.arange(ids.shape[1],device=ids.device)[None].expand_as(ids)
    allowed=torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).tril()[None,None]
    mask=torch.zeros(allowed.shape,device=ids.device,dtype=boundary.dtype).masked_fill(~allowed,torch.finfo(boundary.dtype).min)
    hidden=boundary;rope=rotary(hidden,pos)
    for block in blocks:
        hidden=block(hidden,attention_mask=mask,position_ids=pos,position_embeddings=rope,cache_position=pos[0],use_cache=False)[0]
    return norm(hidden)


@torch.inference_mode()
def run(root):
    root=Path(root).resolve();p=verify(root);out=root/'run'
    if out.exists():raise FileExistsError('Preserve previous diagnostic evidence')
    out.mkdir();started=time.monotonic();lease=None;dataset=None
    try:
        lease,guard=_reservation(p);torch.set_num_threads(2)
        with recovery.base.model_source.execution.arithmetic_policy():
            u,s,blocks,norm,rope,construction=build_models(p,device='cuda:0',guard=guard,out=out)
            triton_available=torch.cuda.get_device_capability(0)[0]>=8
            dataset=recovery.data.RebuildDataset(p['data_root'],cut=p['cut'])
            rows=[]
            with packed.attach_prefix(u,p['model']['path'],mode='cpu')as prefix:
                control_batch=dataset.batch([p['native_control_index']]);ids=control_batch['input_ids'].to('cuda:0')
                boundary=u.native_prefix(ids,torch.ones_like(ids,dtype=torch.bool))
                teacher=original_suffix(boundary,ids,blocks,norm,rope)
                control=dict(boundary=recovery._difference(boundary,control_batch['boundary_hidden'].to(boundary)),
                    normalized_teacher=recovery._difference(teacher,control_batch['teacher_hidden'].to(teacher)),
                    scope='Same original mathematics versus recorded3060cache; cross-GPU arithmetic is diagnostic, not bit-exact admission')
                write(out/'native_control.json',control);del boundary,teacher,control_batch
                for index,item in enumerate(p['histories']):
                    guard();ids=torch.tensor([item['input_ids']],device='cuda:0')
                    boundary=u.native_prefix(ids,torch.ones_like(ids,dtype=torch.bool))
                    batch=dict(input_ids=ids,boundary_hidden=boundary)
                    native=F.linear(original_suffix(boundary,ids,blocks,norm,rope)[:,-1],u.embedding)[0]
                    logits={'native':native.detach().cpu(),'unpacked':u.readout(u.hidden_forward(batch)[:,-1])[0].cpu()}
                    s.stack.enabled=False;logits['packed_torch']=s.readout(s.hidden_forward(batch)[:,-1])[0].cpu()
                    if triton_available:
                        s.stack.enabled=True;logits['packed_triton']=s.readout(s.hidden_forward(batch)[:,-1])[0].cpu();s.stack.enabled=False
                    # Independently invoke the production next_logits entrypoint
                    # with a fresh streamed native prefix on the first history.
                    entrypoint=None
                    if index==0:
                        entrypoint=compare_logits(u.next_logits(ids),logits['unpacked'])
                        s.native_prefix=u.native_prefix
                        try:entrypoint['packed']=compare_logits(s.next_logits(ids),logits['packed_torch'])
                        finally:s.native_prefix=None
                        if not entrypoint['bit_exact']or not entrypoint['packed']['bit_exact']:
                            raise RuntimeError('Generation entrypoint and identical-boundary suffix disagree')
                    comparison={f'{name}_vs_native':compare_logits(v,logits['native'])for name,v in logits.items()if name!='native'}
                    comparison['packed_torch_vs_unpacked']=compare_logits(logits['packed_torch'],logits['unpacked'])
                    if triton_available:comparison['packed_triton_vs_torch']=compare_logits(logits['packed_triton'],logits['packed_torch'])
                    row=dict(index=index,**item,paths={name:logit_summary(v,p['generation_eos_token_ids'],expected=item['stored_next_token'])for name,v in logits.items()},
                        comparisons=comparison,production_entrypoint=entrypoint,
                        stored_student_same_checkpoint=p['same_checkpoint_as_stored_generation'],
                        stored_choice_comparison_scope='Stored3060Triton vs currentdevice; any discrepancy confounds hardware and backend')
                    raw=out/f'logits_{index:02d}.pt';torch.save(dict(input_ids=item['input_ids'],logits=logits),raw)
                    row['raw_logits']=dict(path=str(raw),sha256=sha(raw));rows.append(row)
                    with(out/'records.jsonl').open('a')as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    print(json.dumps(dict(phase='fixed_history_audit',completed=index+1,total=len(p['histories']),
                        case=item['case'],history=item['history'],step=item['step'],seconds=time.monotonic()-started)),flush=True)
            result=dict(schema=SCHEMA,status='completed',fingerprint=p['fingerprint'],checkpoint_sha256=p['checkpoint_sha256'],
                systems=dict(packed_reload_exact=construction['reload']['packed_state_exact'],native_control=control,
                    generation_entrypoint_exact=True,shared_original_prefix=True,shared_original_tied_embedding=True),
                triton=dict(available=triton_available,reason=None if triton_available else'Ampere BF16 kernels unavailable onTuring'),
                original_prefix=prefix,construction=construction,records=rows,record_count=len(rows),
                torch_unpacked_greedy_agreement=sum(r['comparisons']['packed_torch_vs_unpacked']['greedy_equal']for r in rows)/len(rows),
                native_completion_eos={name:sum(r['paths'][name]['greedy_is_eos']for r in rows if r['history']=='native_completion')
                    for name in rows[0]['paths']},native_completion_count=sum(r['history']=='native_completion'for r in rows),
                same_checkpoint_as_stored_generation=p['same_checkpoint_as_stored_generation'],
                training_admission='Requires separate authenticated root review; no numerical heuristic silently admits recovery',
                quality_claim='Diagnostic fixed histories, not task accuracy or recovered usability',
                lease_id=lease['id'],allocation=lease['allocation'],gpu_name=torch.cuda.get_device_name(0),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-started,confirmed=False)
            write(out/'result.json',result);write(out/'manifest.json',dict(status='completed',fingerprint=p['fingerprint'],
                artifacts={str(path.relative_to(out)):sha(path)for path in out.rglob('*')if path.is_file()}));return result
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],lease=lease,
            error_type=type(error).__name__,error=str(error),requires_diagnosis=True,partial_evidence_preserved=True));raise
    finally:
        if dataset is not None:dataset.close()


def check(root):
    root=Path(root).resolve();p=verify(root,live=False)
    if(root/'run/manifest.json').exists():
        manifest=read(root/'run/manifest.json')
        if manifest['fingerprint']!=p['fingerprint']or manifest['status']!='completed':raise ValueError('Diagnostic receipt differs')
        for name,wanted in manifest['artifacts'].items():
            path=(root/'run'/name).resolve()
            if not path.is_relative_to(root/'run')or sha(path)!=wanted:raise ValueError('Diagnostic evidence changed')
        return read(root/'run/result.json')
    if(root/'run/failure.json').exists():return read(root/'run/failure.json')
    return dict(status='prepared_or_active',fingerprint=p['fingerprint'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','run','check'))
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--recovery-root',type=Path)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--profile',choices=('plain2060','3060'),default='plain2060')
    args=parser.parse_args()
    if args.phase=='prepare':
        if args.recovery_root is None or args.checkpoint is None:parser.error('Preparation requires recovery root and checkpoint')
        result=prepare(args.root,args.recovery_root,args.checkpoint,profile=args.profile)
    elif args.phase=='run':result=run(args.root)
    else:result=check(args.root)
    print(json.dumps({k:result[k]for k in ('status','fingerprint','histories','record_count','torch_unpacked_greedy_agreement','seconds')if k in result}),flush=True)


if __name__=='__main__':main()
