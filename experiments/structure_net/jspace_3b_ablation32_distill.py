"""Bounded sequential fitting of the upfront ABLATION32 compact model.

Each stage owns one block and its disposable actual-student-prefix cache.
Earlier selected blocks are immutable; future compact blocks stay on CPU.
The original native suffix supplies lookahead gradients, never new targets.
Only three current-block states are retained, and epoch zero can win selection.
An external controller acquires the declared single SUPER reservation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from . import jspace_3b_ablation32_campaign as campaign
from . import jspace_3b_ablation32_compile as compiler
from . import jspace_3b_ablation32_data as data

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='jspace.ablation32-distillation-stage.v1'
read,write,sha,identity=campaign.read,campaign.write,campaign.sha,campaign.identity
digest=campaign.digest


def source_files():
    return campaign.source_files()|data.cache.prior.source.inherited.local_import_closure([
        Path(__file__),data.PROTOCOL,ROOT/'tests/structure_net/test_jspace_3b_ablation32_distill.py',
        ROOT/'tests/structure_net/test_jspace_3b_ablation32_compile.py'])


def rank_candidate(candidate):
    value=candidate['selection']['objective']
    if not math.isfinite(value)or type(candidate['epoch'])is not int or candidate['epoch']<0:
        raise ValueError('Finite measured selection objective and explicit epoch required')
    return value,candidate['epoch']


def epoch_windows(indices,epoch,*,seed=1001,accumulation=256):
    if epoch<1 or accumulation<1 or len(set(indices))!=len(indices):raise ValueError('Distinct epoch inventory required')
    order=list(indices);random.Random(seed+epoch).shuffle(order)
    return [order[i:i+accumulation]for i in range(0,len(order),accumulation)]


def _parent(parent_root,*,layer,arm,campaign_fingerprint,compiled_root):
    if layer==0:
        if parent_root is not None:raise ValueError('First stage starts from the upfront compiled bank')
        return [],{}
    if parent_root is None:raise ValueError('Later stages require the selected adjacent student prefix')
    parent_root=Path(parent_root).resolve();pp=verify(parent_root,live=False)[0];result=check(parent_root)
    if (result.get('status')!='completed' or pp['layer']!=layer-1 or pp['arm']!=arm
            or pp['campaign_fingerprint']!=campaign_fingerprint or pp['compiled_root']!=str(compiled_root)):
        raise ValueError('Completed same-arm adjacent selected stage required')
    rows=result['selected_blocks']
    if [r['layer']for r in rows]!=list(range(layer)):raise ValueError('Incomplete selected student prefix')
    inputs={str(parent_root/'plan.json'):sha(parent_root/'plan.json'),
        str(parent_root/'run/result.json'):sha(parent_root/'run/result.json'),
        str(parent_root/'run/manifest.json'):sha(parent_root/'run/manifest.json')}
    for row in rows:
        if sha(row['checkpoint'])!=row['sha256']:raise ValueError('Selected earlier block changed')
        inputs[row['checkpoint']]=row['sha256']
    return rows,inputs


def prepare(root,campaign_root,arm,layer,parent_root=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation requires CUDA hidden')
    if arm not in ('learned','random')or type(layer)is not int or not 0<=layer<36:
        raise ValueError('Explicit paired arm and original layer0..35 required')
    root,campaign_root=Path(root).resolve(),Path(campaign_root).resolve()
    if root.exists():raise FileExistsError(root)
    cp,dp,records=campaign.verify(campaign_root,live=False)
    done=campaign._completed(campaign_root/'compiled',cp)
    if not done.get('both_arms_all_layers_compiled'):raise ValueError('Both entire models must be compiled before fitting')
    compiled_root=campaign_root/'compiled'/arm;manifest=read(compiled_root/'manifest.json')
    if (manifest.get('status')!='completed' or manifest['layers']!=36 or manifest['arm']!=arm
            or not manifest['compiled_before_any_distillation'] or manifest['additional_head_or_mlp_approximation']):
        raise ValueError('Complete residual-only upfront bank required')
    parents,inputs=_parent(parent_root,layer=layer,arm=arm,
        campaign_fingerprint=cp['fingerprint'],compiled_root=compiled_root)
    inputs.update({str(campaign_root/'plan.json'):sha(campaign_root/'plan.json'),
        str(campaign_root/'compiled/result.json'):sha(campaign_root/'compiled/result.json'),
        str(campaign_root/'compiled/manifest.json'):sha(campaign_root/'compiled/manifest.json'),
        str(compiled_root/'manifest.json'):sha(compiled_root/'manifest.json'),
        str(compiled_root/'compiled.pt'):manifest['checkpoint_sha256'],
        str(Path(cp['data_root'])/'plan.json'):sha(Path(cp['data_root'])/'plan.json')})
    # Hash the bank once at admission; immutable identities avoid repeated scans.
    if sha(compiled_root/'compiled.pt')!=manifest['checkpoint_sha256']:raise ValueError('Compiled bank changed')
    train=dp['distillation_indices'];selection=dp['selection_indices']
    longest=min(train,key=lambda i:(-len(records[i].input_ids),records[i].example_id))
    response=min(train,key=lambda i:(-len(records[i].response_ids),records[i].example_id))
    pilot_indices=list(dict.fromkeys([longest,response]))
    plan=dict(schema=SCHEMA,campaign_root=str(campaign_root),campaign_fingerprint=cp['fingerprint'],
        data_root=cp['data_root'],data_fingerprint=dp['fingerprint'],records_sha256=dp['records_sha256'],
        model=cp['model'],arm=arm,layer=layer,layers=36,compiled_root=str(compiled_root),
        parent_root=str(Path(parent_root).resolve())if parent_root is not None else None,selected_prefix=parents,
        configuration=asdict(compiler.DistillationConfig()),training_indices=train,selection_indices=selection,
        pilot_indices=pilot_indices,seed=1001,train_final_norm=layer==35,
        prefix_cache='Disposable BF16 actual selected student coordinates; exact record/token and prefix-state binding',
        train_target='Original native block-exit coordinates plus original native full-response/EOS teacher distribution',
        native_lookahead='Unmodified frozen native blocks after the current block; original norm except private student norm at last stage',
        selection='Minimum exposed TRAIN selection coordinateMSE+T2teacherKL, then earliest epoch including zero',
        retention=dict(block_checkpoints=3,full_stack_snapshots=0,packed_exports=0,
            scratch_prefix='Delete only owned completed-stage cache after selected replay'),
        required_discarded_pilot=True,allocation_mib=7680,limit_mib=7424,gpu_prefix=campaign.TEACHER_PREFIX,
        input_sha256=inputs,input_identity={path:identity(path)for path in inputs},
        dependency_identity={**dp['input_identity'],**cp['input_identity']},
        source_sha256={str(path):sha(path)for path in sorted(source_files())},confirmed=False)
    plan['fingerprint']=digest(plan);root.mkdir(parents=True);write(root/'plan.json',plan)
    for path,wanted in plan['source_sha256'].items():
        dest=root/'producer'/Path(path).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        if sha(dest)!=wanted:raise ValueError('Stage source archive differs')
    return dict(status='prepared',fingerprint=plan['fingerprint'],arm=arm,layer=layer,
        epochs=plan['configuration']['epochs'],training_examples=len(train),selection_examples=len(selection))


def verify(root,*,live=True):
    root=Path(root).resolve();p=read(root/'plan.json');body=dict(p);fp=body.pop('fingerprint')
    if p.get('schema')!=SCHEMA or digest(body)!=fp:raise ValueError('Distillation plan changed')
    for path,old in {**p['dependency_identity'],**p['input_identity']}.items():
        if identity(path)!=old:raise ValueError('Stage input changed: '+path)
    for path,wanted in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=wanted or live and sha(path)!=wanted:
            raise ValueError('Stage producer changed: '+path)
    dp,records=data.verify(p['data_root'],live=False)
    if (dp['fingerprint']!=p['data_fingerprint'] or dp['distillation_indices']!=p['training_indices']
            or dp['selection_indices']!=p['selection_indices'] or dp['records_sha256']!=p['records_sha256']):
        raise ValueError('Original teacher or disjoint TRAIN inventories changed')
    return p,dp,records


def reservation(p,limit_mib):
    if limit_mib!=p['limit_mib']:raise ValueError('Allocator differs from prepared stage')
    lease,guard=data.cache.reservation(limit_mib)
    allocation=lease['allocation'];cards=list(allocation)
    if (len(cards)!=1 or not cards[0].startswith(p['gpu_prefix']) or allocation[cards[0]]!=p['allocation_mib']
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=cards[0]):
        raise ValueError('Distillation requires the reserved single RTX2060 SUPER')
    return lease,guard


def build_stage(p,*,device='cpu',guard=None):
    """Only selected prefix/current block and the native remaining suffix move."""
    if str(device).startswith('cuda'):
        if guard is None:raise ValueError('Reservation guard required before CUDA construction')
        guard()
    stack,_=compiler.load_compiled(p['compiled_root'])
    for row in p['selected_prefix']:
        compiler.restore_selected_block(stack,row['checkpoint'],expected_sha256=row['sha256'])
    cfg=compiler.native._config(p['model']['path'])
    E,norm,rotary=compiler.native.load_readout_norm(p['model']['path'],device='cpu',dtype=torch.bfloat16,config=cfg)
    E=E.to(device);norm=norm.to(device);rotary=rotary.to(device)
    layer=p['layer']
    for i in range(layer+1):stack.blocks[i].to(device)
    for i in range(layer):stack.bridges[i].to(device)
    native_blocks=[]
    for i in range(layer+1,p['layers']):
        if guard is not None:guard()
        native_blocks.append(compiler.native.load_native_block(p['model']['path'],i,device=device,dtype=torch.bfloat16,config=cfg))
    lookahead=compiler.FrozenNativeLookahead(native_blocks,norm,E,rotary,start=layer+1,expected_layers=p['layers'])
    stage=compiler.SequentialStage(stack,E,rotary,lookahead,layer=layer,guard=guard,train_final_norm=p['train_final_norm'])
    return stage


class PrefixCache:
    """One immutable stage's actual frozen compressed-prefix coordinates."""
    def __init__(self,directory,stage,dataset,indices,*,create=True,progress=lambda **_:None):
        self.directory=Path(directory);self.stage=stage;self.dataset=dataset;self.indices=set(indices)
        self.path=self.directory/'coordinates.npy';rows=dataset.rows
        if len(self.indices)!=len(indices):raise ValueError('Distinct cache records required')
        shape=(dataset.plan['total_tokens'],stage.block.rank)
        if create:
            self.directory.mkdir(parents=True,exist_ok=False)
            bank=np.lib.format.open_memmap(self.path,mode='w+',dtype=np.int16,shape=shape)
            started=time.monotonic()
            try:
                for ordinal,index in enumerate(indices):
                    b=dataset.batch([index]);z=stage.prefix_coordinates(b)
                    if z.dtype!=torch.bfloat16 or not torch.isfinite(z).all():raise ValueError('Finite BF16 actual prefix coordinates required')
                    row=rows[index];bank[row['offset']:row['stop']]=z[0].contiguous().cpu().view(torch.int16).numpy()
                    if (ordinal+1)%64==0 or ordinal+1==len(indices):
                        bank.flush();progress(completed=ordinal+1,total=len(indices),seconds=time.monotonic()-started)
            finally:bank.flush();bank._mmap.close()
            write(self.directory/'manifest.json',dict(status='completed',prefix_fingerprint=stage.prefix_fingerprint,
                indices=list(indices),records_sha256=dataset.plan['records_sha256'],rows=rows,
                shape=list(shape),sha256=sha(self.path),identity=identity(self.path),seconds=time.monotonic()-started))
        manifest=read(self.directory/'manifest.json')
        if (manifest.get('status')!='completed' or manifest['prefix_fingerprint']!=stage.prefix_fingerprint
                or manifest['indices']!=list(indices) or manifest['records_sha256']!=dataset.plan['records_sha256']
                or manifest['rows']!=rows or manifest['shape']!=list(shape) or manifest['identity']!=identity(self.path)):
            raise ValueError('Prefix cache is not bound to this frozen student and exact records')
        self.bank=np.load(self.path,mmap_mode='r');self.manifest=manifest

    def __call__(self,index):
        if index not in self.indices:raise ValueError('Record outside cached TRAIN and selection inventory')
        row=self.dataset.rows[index];z=torch.from_numpy(np.array(self.bank[row['offset']:row['stop']],copy=True)).view(torch.bfloat16)
        return dict(student_input_coordinates=z[None],student_prefix_fingerprint=self.stage.prefix_fingerprint,
            student_prefix_input_ids=torch.tensor([self.dataset.records[index].input_ids],dtype=torch.long))

    def close(self):
        if not self.bank._mmap.closed:self.bank._mmap.close()

    def discard(self):
        self.close()
        if self.path.parent!=self.directory or identity(self.path)!=self.manifest['identity']:
            raise ValueError('Refusing to remove a changed/nonowned scratch cache')
        self.path.unlink()


def _progress(out,phase,**values):campaign._progress(out,phase,**values)


def _initial(trainer):
    if trainer.updates or trainer.optimizer.state:raise ValueError('Fresh zero-update optimizer required')
    return dict(parameters=compiler.offload.fingerprint([(str(i),p)for i,p in enumerate(trainer.parameters)]),
        fixed=trainer.fixed,prefix=trainer.stage.prefix_fingerprint)


def pilot_update_proof(trainer,before,update):
    """Zero/sub-ULP gradients may leave individual BF16 tensors unchanged."""
    names=[f'block.{name}'for name,_ in trainer.stage.block.named_parameters()]
    if trainer.stage.train_final_norm:names.extend(f'norm.{name}'for name,_ in trainer.stage.lookahead.norm.named_parameters())
    if len(before)!=len(names)or len(names)!=len(trainer.parameters):raise ValueError('Pilot parameter inventory differs')
    optimizer=update['optimizer']
    if (optimizer['updated_scalars']!=sum(p.numel()for p in trainer.parameters)
            or not math.isfinite(optimizer['unclipped_gradient_norm'])):
        raise RuntimeError('Every intended parameter must have a finite received gradient')
    rows=[]
    for name,old,parameter in zip(names,before,trainer.parameters):
        expected=trainer.optimizer.state[parameter]['master'].to(dtype=parameter.dtype,device='cpu')
        actual=parameter.detach().cpu()
        if not torch.equal(actual,expected):raise RuntimeError('BF16 parameter does not equal its updated FP32 master cast')
        rows.append(dict(name=name,storage_changed=not torch.equal(old,actual),
            expected_storage_change=not torch.equal(old,expected),master_cast_exact=True))
    if not any(row['storage_changed']for row in rows if row['name'].startswith('block.')):
        raise RuntimeError('Pilot did not change any actual current-block BF16 weights')
    return dict(all_intended_gradients_received_and_finite=True,current_block_storage_changed=True,
        parameters=rows,unchanged_tensors=[r['name']for r in rows if not r['storage_changed']],
        interpretation='Individual zero/sub-ULP updates are reported; changed FP32 masters must cast exactly into stored weights')


def _dataset(p,stage,*,cache_dir=None,progress=lambda **_:None):
    source=data.SearchDataset(p['data_root'])
    dataset=compiler.DistillationDataset(source,p['layer'])
    cached=None
    if cache_dir is not None:
        indices=[*p['training_indices'],*p['selection_indices']]
        cached=PrefixCache(cache_dir,stage,dataset,indices,progress=progress)
        dataset.prefix_provider=cached
    return source,dataset,cached


def _trainer(p,stage,dataset):
    return compiler.StageTrainer(stage,dataset,teacher_id=p['model']['revision'],
        train_indices=p['training_indices'],selection_indices=p['selection_indices'],
        config=compiler.DistillationConfig(**p['configuration']))


def _phase_manifest(out,p,result,files):
    write(out/'result.json',result)
    campaign._manifest(out,p,[out/'result.json',out/'progress.jsonl',*files])


def _failure(root,phase,p,started,lease,error):
    path=root/f'{phase}_failure.json'
    if not path.exists():write(path,dict(status='failed',fingerprint=p['fingerprint'],lease=lease,
        error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started,
        partial_artifacts_preserved=True,requires_diagnosis=True,automatic_retry_allowed=False))


def pilot(root,*,limit_mib=7424):
    root=Path(root).resolve();p,_,_=verify(root);out=root/'pilot'
    if out.exists():raise FileExistsError('Preserve previous pilot; diagnose and prepare a new attempt')
    out.mkdir();started=time.monotonic();lease=stage=trainer=source=dataset=None
    try:
        lease,guard=reservation(p,limit_mib);torch.set_num_threads(2);torch.manual_seed(p['seed'])
        _progress(out,'load',layer=p['layer'],arm=p['arm'])
        with data.cache.execution.arithmetic_policy():
            stage=build_stage(p,device='cuda:0',guard=guard);source,dataset,_=_dataset(p,stage)
            trainer=_trainer(p,stage,dataset);initial=_initial(trainer)
            before=[v.detach().cpu().clone()for v in trainer.parameters]
            trainer.begin_epoch(1);update=trainer.train_window(p['pilot_indices'])
            update_proof=pilot_update_proof(trainer,before,update)
            if trainer.fixed_signature()!=initial['fixed']:
                raise RuntimeError('Pilot changed frozen model state')
            cp=trainer.save_checkpoint(out/'current_block.pt',identities={'fingerprint':p['fingerprint'],'discarded':True})
            batch=dataset.batch([p['pilot_indices'][0]])
            with torch.no_grad():expected=stage.lookahead.readout(stage.forward(batch)[1][:,-1]).detach().cpu()
            trainer.close();trainer=None
            compiler.restore_selected_block(stage.stack,cp['path'],expected_sha256=cp['sha256'],norm=stage.lookahead.norm)
            with torch.no_grad():actual=stage.lookahead.readout(stage.forward(batch)[1][:,-1]).detach().cpu()
            if not torch.equal(actual,expected):raise RuntimeError('Saved current block full-vocabulary replay changed')
            del stage;stage=None;gc.collect();torch.cuda.empty_cache()
            stage=build_stage(p,device='cuda:0',guard=guard);trainer=_trainer(p,stage,dataset)
            if _initial(trainer)!=initial:raise RuntimeError('Fresh scientific initialization differs after pilot')
            guard();result=dict(status='completed',fingerprint=p['fingerprint'],passed=True,discarded=True,
                fresh_initializer_proved=True,initializer=initial,current_block_update_proof=update_proof,
                fixed_teacher_prefix_geometry_unchanged=True,full_vocabulary_checkpoint_replay_exact=True,
                update=update,lease_id=lease['id'],allocation=lease['allocation'],
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-started,
                scientific_updates=0,confirmed=False)
            _progress(out,'pilot_completed',seconds=result['seconds'])
            _phase_manifest(out,p,result,[Path(cp['path'])]);return result
    except BaseException as error:_failure(root,'pilot',p,started,lease,error);raise
    finally:
        if trainer is not None:trainer.close()
        if dataset is not None:dataset.close()
        if source is not None:source.close()


def retain_best_three(out,candidates):
    directory=(Path(out)/'checkpoints').resolve();keep={c['checkpoint']['path']for c in sorted(candidates,key=rank_candidate)[:3]}
    removed=[]
    for row in candidates:
        cp=row['checkpoint'];path=Path(cp['path']).resolve()
        if str(path)in keep or not path.exists():continue
        if path.parent!=directory or sha(path)!=cp['sha256']:raise ValueError('Refusing to prune changed/nonowned block checkpoint')
        path.unlink();removed.append(dict(epoch=row['epoch'],checkpoint=str(path),sha256=cp['sha256']))
    return removed


def train_epochs(p,trainer,out,*,progress):
    """Small testable lifecycle; no additional epochs or hidden joint phase."""
    out=Path(out);cpdir=out/'checkpoints';cpdir.mkdir(exist_ok=False)
    history=[];pruned=[]
    for epoch in range(trainer.config.epochs+1):
        if epoch:
            trainer.begin_epoch(epoch)
            for window in epoch_windows(p['training_indices'],epoch,seed=p['seed'],accumulation=trainer.config.accumulation_examples):
                progress('window_start',epoch=epoch,next_update=trainer.updates+1,cases=len(window))
                progress('update',**trainer.train_window(window))
        score=trainer.development()
        cp=trainer.save_checkpoint(cpdir/f'epoch_{epoch:02d}.pt',identities={'fingerprint':p['fingerprint'],
            'arm':p['arm'],'layer':p['layer'],'compiled_root':p['compiled_root']})
        row=dict(epoch=epoch,selection=score,checkpoint=cp,updates=trainer.updates);history.append(row)
        progress('selection',**row)
        deleted=retain_best_three(out,history);pruned.extend(deleted)
        if deleted:progress('retention',removed=deleted)
    best=min(history,key=rank_candidate)
    trainer.close()
    compiler.restore_selected_block(trainer.stage.stack,best['checkpoint']['path'],
        expected_sha256=best['checkpoint']['sha256'],norm=trainer.stage.lookahead.norm)
    replay=trainer.development()
    if replay!=best['selection']:raise RuntimeError('Selected early checkpoint objective did not replay exactly')
    return dict(history=history,pruned=pruned,selected=best,selected_replay_exact=True,
        epochs=trainer.config.epochs,updates=trainer.updates)


def run(root,*,limit_mib=7424):
    root=Path(root).resolve();p,_,_=verify(root);proof=campaign._completed(root/'pilot',p)
    if not all(proof.get(k)is True for k in ('passed','discarded','fresh_initializer_proved',
        'fixed_teacher_prefix_geometry_unchanged','full_vocabulary_checkpoint_replay_exact')):
        raise ValueError('Matching discarded systems pilot required')
    out=root/'run'
    if out.exists():raise FileExistsError('Preserve previous run; diagnose and prepare a new attempt')
    out.mkdir();started=time.monotonic();lease=stage=trainer=source=dataset=cached=None
    try:
        lease,guard=reservation(p,limit_mib);torch.set_num_threads(2);torch.manual_seed(p['seed'])
        _progress(out,'load',arm=p['arm'],layer=p['layer'])
        with data.cache.execution.arithmetic_policy():
            stage=build_stage(p,device='cuda:0',guard=guard)
            source,dataset,cached=_dataset(p,stage,cache_dir=out/'prefix_scratch',
                progress=lambda **v:_progress(out,'prefix_cache',**v))
            trainer=_trainer(p,stage,dataset)
            if _initial(trainer)!=proof['initializer']:raise RuntimeError('Science did not start from fresh initial state')
            actual=train_epochs(p,trainer,out,progress=lambda phase,**v:_progress(out,phase,**v))
            if trainer.fixed_signature()!=trainer.fixed:raise RuntimeError('Frozen stage state changed')
            chosen=actual['selected']['checkpoint'];rows=[*p['selected_prefix'],
                dict(layer=p['layer'],checkpoint=chosen['path'],sha256=chosen['sha256'])]
            guard();result=dict(schema=SCHEMA,status='completed',fingerprint=p['fingerprint'],arm=p['arm'],
                layer=p['layer'],layers=len(rows),compiled_root=p['compiled_root'],selected_blocks=rows,
                model_revision=p['model']['revision'],data_fingerprint=p['data_fingerprint'],
                configuration=p['configuration'],**actual,private_final_norm_fitted=p['train_final_norm'],
                original_teacher_unchanged=True,frozen_student_prefix_used=True,future_compact_blocks_trained=False,
                task_accuracy_measured=False,selection_scope='Exposed TRAIN selection, not final task accuracy',
                lease_id=lease['id'],allocation=lease['allocation'],gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                seconds=time.monotonic()-started,confirmed=False)
            cached.discard();result['owned_prefix_scratch_removed']=True
            _progress(out,'stage_completed',selected_epoch=actual['selected']['epoch'],seconds=result['seconds'])
            retained=[Path(c['checkpoint']['path'])for c in actual['history']if Path(c['checkpoint']['path']).exists()]
            _phase_manifest(out,p,result,[*retained,out/'prefix_scratch/manifest.json']);return result
    except BaseException as error:_failure(root,'run',p,started,lease,error);raise
    finally:
        if trainer is not None:trainer.close()
        if cached is not None:cached.close()
        if dataset is not None:dataset.close()
        if source is not None:source.close()


def check(root):
    root=Path(root).resolve();p,_,_=verify(root,live=False)
    if (root/'run/manifest.json').exists():return campaign._completed(root/'run',p)
    for phase in ('run','pilot'):
        if (root/f'{phase}_failure.json').exists():return read(root/f'{phase}_failure.json')
    if (root/'pilot/manifest.json').exists():return dict(status='pilot_passed',fingerprint=p['fingerprint'],pilot=campaign._completed(root/'pilot',p))
    return dict(status='prepared_or_active',fingerprint=p['fingerprint'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','pilot','run','check'));parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--campaign-root',type=Path);parser.add_argument('--parent-root',type=Path)
    parser.add_argument('--arm',choices=('learned','random'));parser.add_argument('--layer',type=int)
    parser.add_argument('--limit-mib',type=int,default=7424);args=parser.parse_args()
    if args.phase=='prepare':
        if args.campaign_root is None or args.arm is None or args.layer is None:parser.error('Preparation needs campaign, arm and layer')
        result=prepare(args.root,args.campaign_root,args.arm,args.layer,args.parent_root)
    elif args.phase=='pilot':result=pilot(args.root,limit_mib=args.limit_mib)
    elif args.phase=='run':result=run(args.root,limit_mib=args.limit_mib)
    else:result=check(args.root)
    print(json.dumps({k:result[k]for k in ('status','fingerprint','arm','layer','layers','epochs','updates','seconds')if k in result}),flush=True)


if __name__=='__main__':main()
