"""Frozen one-block Phi recovery lifecycle; every worker requires an RTX3060 lease."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_phi_data as data
from . import jspace_phi_fit as fitting
from . import jspace_phi_native as native
from . import jspace_phi_train as training
from . import jspace_phi_weights as weights
from .jspace_recovery_data import ROOT, digest, read, sha, write

SCHEMA='jspace.phi-recovery.v1'
PHASES=('pilot','train','verify')
CONFIG=training.PhiTrainConfig()
PROTOCOL=ROOT/'docs/07 - Status Reports/jspace/2026-11-15_phi4-single-block-pilot.md'


def file_identity(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def prepare(root,native_root,fit_root,data_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU preparation requires CUDA hidden')
    root,native_root,fit_root,data_root=map(lambda p:Path(p).resolve(),(root,native_root,fit_root,data_root))
    if root.exists():raise FileExistsError(root)
    dp=data.verify(data_root,live=True);np=native.verify_plan(native_root)
    nm=native.checked(native_root);fp=fitting.verify(fit_root)
    if (np['model']!=dp['model'] or np['data_fingerprint']!=dp['fingerprint'] or
            Path(np['data_root']).resolve()!=data_root or fp['native_fingerprint']!=np['fingerprint'] or
            fp['data_fingerprint']!=dp['fingerprint'] or Path(fp['native_root']).resolve()!=native_root or
            Path(fp['data_root']).resolve()!=data_root or Path(fp['model']).resolve()!=Path(dp['model']['path'])):
        raise ValueError('Native, fit, model and data identities must be the same study')
    inputs={}
    def pin(path,expected=None):
        path=str(Path(path).resolve());value=sha(path)
        if expected is not None and value!=expected:raise ValueError('Admitted input changed: '+path)
        inputs[path]=value
    for path in (native_root/'plan.json',native_root/'native/manifest.json',fit_root/'plan.json',
                 data_root/'plan.json',data_root/'manifest.json',data_root/'pilot_responses.jsonl'):
        pin(path)
    for name,expected in nm['artifacts'].items():pin(native_root/'native'/name,expected)
    receipts={}
    for phase in fitting.PHASES:
        receipt=fitting.checked(fit_root,phase,fp);path=fit_root/(phase+'_receipt.json');pin(path)
        receipts[phase]=inputs[str(path)]
        for name,expected in receipt['artifacts'].items():pin(fit_root/name,expected)
    for name,expected in dp['model']['files'].items():
        # data.verify already checked every model payload above.
        inputs[str((Path(dp['model']['path'])/name).resolve())]=expected
    files=fitting.local_import_closure([Path(__file__),PROTOCOL,
        ROOT/'tests/structure_net/test_jspace_phi_recovery.py',
        ROOT/'tests/structure_net/test_jspace_phi_train.py',
        Path(training.nal.__file__),Path(training.response_data.__file__),Path(training.response_training.__file__)])
    p=dict(schema=SCHEMA,model=dp['model'],native_root=str(native_root),fit_root=str(fit_root),data_root=str(data_root),
        native_fingerprint=np['fingerprint'],fit_fingerprint=fp['fingerprint'],data_fingerprint=dp['fingerprint'],
        fit_receipts=receipts,label_ids=dp['label_ids'],config=asdict(CONFIG),phases=list(PHASES),
        layer=39,compressed_blocks=1,total_blocks=40,allocation_mib=11264,allocator_mib=11008,
        maximum_native_net_losses=3,selection='highest DEV correct then earliest epoch; initializer eligible',
        input_sha256=inputs,input_identity={path:file_identity(path) for path in inputs},
        source_sha256={str(path):sha(path) for path in sorted(files)},
        environment=dict(torch=torch.__version__,transformers=__import__('transformers').__version__),
        confirmed=False,generation_recovery=False,speed_measured=False,full40=False,
        scope='One final block; decoded GPTQ source in BF16;128 TRAIN/64 reused DEV; no independent quality claim')
    p['fingerprint']=digest(p);root.mkdir(parents=True)
    for path,expected in p['source_sha256'].items():
        dest=root/'producer'/Path(path).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        if sha(dest)!=expected:raise ValueError('Producer archive changed')
    write(root/'plan.json',p)
    return dict(status='prepared',fingerprint=p['fingerprint'],root=str(root),confirmed=False)


def verify_plan(root,*,live=True):
    root=Path(root).resolve();p=read(root/'plan.json')
    if (p.get('schema')!=SCHEMA or p.get('fingerprint')!=digest({k:v for k,v in p.items() if k!='fingerprint'}) or
            p.get('config')!=asdict(CONFIG) or p.get('phases')!=list(PHASES) or
            (p.get('layer'),p.get('compressed_blocks'),p.get('total_blocks'))!=(39,1,40) or
            p.get('allocation_mib')!=11264 or p.get('allocator_mib')!=11008 or p.get('maximum_native_net_losses')!=3 or
            any(p.get(k)is not False for k in ('confirmed','generation_recovery','speed_measured','full40')) or
            set(p['input_identity'])!=set(p['input_sha256'])):raise ValueError('Frozen Phi recovery policy changed')
    for path,identity in p['input_identity'].items():
        if file_identity(path)!=identity:raise ValueError('Immutable recovery input changed: '+path)
    for path,expected in p['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT))!=expected or live and sha(path)!=expected:
            raise ValueError('Recovery producer/archive changed')
    if p['environment']!={'torch':torch.__version__,'transformers':__import__('transformers').__version__}:
        raise ValueError('Numerical environment changed')
    return p


def native_control(report,baseline):
    reference=[r for r in baseline['rows'] if r['split']=='development'];rows=report['records']
    if len(rows)!=64 or len(reference)!=64:raise ValueError('All64 original native DEV controls required')
    for actual,expected in zip(rows,reference):
        if (actual['identity']!=expected['identity'] or actual['family']!=expected['family'] or actual['answer']!=expected['answer'] or
                actual['native_prediction']!=expected['prediction'] or actual['native_choice_logits']!=expected['label_logits']):
            raise ValueError('Original prompt-only native control differs')
    count=sum(r['prediction']==r['answer'] for r in reference)
    if report['native_correct']!=count:raise ValueError('Native correct count changed')
    return dict(passed=True,count=64,correct=count,predictions_exact=True,logits_exact=True)


def quality(report):
    rows=report['records']
    if (report.get('count')!=64 or len(rows)!=64 or len({r['identity'] for r in rows})!=64 or
            Counter(r['family'] for r in rows)!=dict.fromkeys(data.FAMILIES,16) or
            report['correct']!=sum(r['prediction']==r['answer'] for r in rows) or
            report['native_correct']!=sum(r['native_prediction']==r['answer'] for r in rows) or
            any(not math.isfinite(x) for r in rows for x in (*r['choice_logits'],*r['native_choice_logits']))):
        raise ValueError('Complete finite paired64 DEV report required')
    return dict(passed=report['correct']>=report['native_correct']-3,correct=report['correct'],native_correct=report['native_correct'],
        count=64,maximum_native_net_losses=3,accuracy_drop=(report['native_correct']-report['correct'])/64,
        scope='exploratory reused DEV screen',confirmed=False)


def resources(m):
    allocation=m.get('allocation',{});cards=list(allocation);settings=m.get('allocator',{})
    if (len(cards)!=1 or not cards[0].startswith('GPU-0370fdb0') or allocation[cards[0]]!=11264 or
            settings.get('allocator_limit_bytes')!=11008*1024**2 or settings.get('lease_bytes')!=11264*1024**2 or
            settings.get('visible_gpu')!=cards[0] or not 0<m.get('peak_reserved_bytes',0)<=11008*1024**2):
        raise ValueError('RTX3060 reservation/cap/peak proof required')


def effective_change(value,*,required=False):
    before=value.get('initial_sha256_by_name',{});after=value.get('current_sha256_by_name',{})
    if not before or set(before)!=set(after):raise ValueError('Complete effective BF16 parameter hashes required')
    changed={name:before[name]!=after[name] for name in before};count=sum(changed.values())
    if (any(not isinstance(h,str) or len(h)!=64 for h in (*before.values(),*after.values())) or
            value.get('changed_by_name')!=changed or value.get('changed_tensor_count')!=count or
            value.get('total_tensor_count')!=len(before) or value.get('changed')!=(count>0) or required and count==0):
        raise ValueError('Effective BF16 changes differ from parameter evidence')
    return value


def _training_evidence(out):
    directory=Path(out)/'recovery';m=read(directory/'manifest.json');r=read(directory/'result.json')
    expected={'result.json','updates.json'}|{f'epoch_{i:02d}.pt' for i in range(4)}|{f'development_{i:02d}.json' for i in range(4)}
    if m.get('status')!='completed' or m.get('schema')!=training.SCHEMA or set(m.get('artifacts',{}))!=expected:
        raise ValueError('All four actual epoch checkpoints/reports required')
    for name,expected_sha in m['artifacts'].items():
        if sha(directory/name)!=expected_sha:raise ValueError('Training artifact changed')
    history=r['history']
    if (r.get('status')!='completed' or r.get('schema')!=training.SCHEMA or r.get('config')!=asdict(CONFIG) or
            [row['epoch'] for row in history]!=list(range(4)) or [row['updates'] for row in history]!=[0,32,64,96] or
            r.get('updates')!=96 or r.get('supervised_tokens')!=768 or
            r.get('selected')!=min(history,key=training.selection_key) or not r.get('fixed_state_unchanged') or
            r.get('internal_reload')!={'passed':True,'strict_state_load':True,'records_and_metrics_exact':True}):
        raise ValueError('Complete96-update budget/earliest-accuracy winner required')
    for row in history:
        if (row['checkpoint']!=f"epoch_{row['epoch']:02d}.pt" or row['report']!=f"development_{row['epoch']:02d}.json" or
                row['checkpoint_sha256']!=m['artifacts'][row['checkpoint']] or row['report_sha256']!=m['artifacts'][row['report']]):
            raise ValueError('Epoch checkpoint/report linkage changed')
        report=read(directory/row['report']);quality(report)
        if row['correct']!=report['correct'] or row['ce']!=report['ce']:raise ValueError('Epoch summary differs')
        effective_change(row['effective_parameter_change'])
        if row['epoch']==0 and row['effective_parameter_change']['changed']:raise ValueError('Initializer cannot already be changed')
    if r['metrics']!=read(directory/r['selected']['report']):raise ValueError('Selected metrics differ')
    if r['selected_effective_parameter_change']!=r['selected']['effective_parameter_change']:
        raise ValueError('Selected effective parameter changes differ')
    return r


def checked(root,phase,plan=None):
    root=Path(root).resolve();p=plan or verify_plan(root,live=False)
    if phase not in PHASES:raise ValueError('Unknown Phi recovery phase')
    out=root/phase;m=read(out/'manifest.json');r=read(out/'result.json')
    if (m.get('status')!='completed' or m.get('phase')!=phase or m.get('fingerprint')!=p['fingerprint'] or
            r.get('status')!='completed' or m.get('artifacts',{}).get('result.json')!=sha(out/'result.json')):
        raise ValueError('Matching completed recovery phase required')
    resources(m)
    for name,expected in m['artifacts'].items():
        path=(out/name).resolve()
        if not path.is_relative_to(out) or sha(path)!=expected:raise ValueError('Phase artifact changed')
    dependencies=() if phase=='pilot' else ('pilot',) if phase=='train' else ('train',)
    if m.get('dependencies')!={k:sha(root/k/'manifest.json') for k in dependencies}:raise ValueError('Prior phase proof changed')
    if phase=='pilot':
        if (not r.get('passed') or not r.get('discarded') or not r.get('systems_only') or
                not r.get('fixed_state_unchanged') or r.get('master_before')==r.get('master_after') or
                r.get('bf16_parameters_changed')is not True or
                r.get('initializer_sha256')!=m['artifacts'].get('initializer.pt')):raise ValueError('Disposable real-update pilot required')
        effective_change(r['effective_parameter_change'],required=True)
        if native_control(read(out/'initial_development.json'),read(Path(p['native_root'])/'native/baseline.json'))!=r['native_control']:
            raise ValueError('Pilot native reference differs')
    elif phase=='train':
        actual=_training_evidence(out)
        if actual!=r:raise ValueError('Outer training result differs from actual trainer')
        pilot=checked(root,'pilot',p)
        if (r['identities']!=pilot['identities'] or r['fixed']!=pilot['fixed'] or r['normalization']!=pilot['normalization'] or
                read(out/'recovery/development_00.json')!=read(root/'pilot/initial_development.json')):
            raise ValueError('Training did not start from authenticated pilot initializer')
    else:
        prior=checked(root,'train',p)
        if (not r.get('passed') or not r.get('strict_state_load') or not r.get('records_and_metrics_exact') or
                not r.get('fixed_state_unchanged') or r.get('checkpoint_sha256')!=prior['selected']['checkpoint_sha256'] or
                r.get('selected_epoch')!=prior['selected']['epoch'] or r.get('metrics')!=prior['metrics'] or
                r.get('effective_parameter_change')!=prior['selected_effective_parameter_change'] or
                r.get('quality')!=quality(prior['metrics']) or r.get('training_result_sha256')!=sha(root/'train/recovery/result.json') or
                any(r.get(k)is not False for k in ('confirmed','generation_recovery','speed_measured','full40'))):
            raise ValueError('Fresh selected checkpoint or quality scope differs')
        if r['native_control']!=native_control(r['metrics'],read(Path(p['native_root'])/'native/baseline.json')):
            raise ValueError('Fresh original native control differs')
    return r


def build_trainer(p,device,guard):
    guard()
    dataset=training.PhiTrainingData(p['native_root'],Path(p['data_root'])/'pilot_responses.jsonl',
        expected_fingerprint=p['native_fingerprint'],label_ids=p['label_ids'])
    try:
        original=weights.load_native_block(p['model']['path'],39,device,guard=guard)
        native_bytes=sum(v.numel()*v.element_size() for v in original.parameters())
        block=fitting.build_block(p['fit_root'],original,trainable=True)
        del original;gc.collect()
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()
        head,norm,rotary=weights.load_readout_norm(p['model']['path'],device,guard=guard)
        trainer=training.PhiTrainer(block,norm,head,rotary,dataset,config=CONFIG,guard=guard,
            identities=dict(study=p['fingerprint'],native=p['native_fingerprint'],fit=p['fit_fingerprint']),
            native_block_bytes=native_bytes)
        return trainer,dataset
    except BaseException:dataset.close();raise


def run(root,phase):
    root=Path(root).resolve();p=verify_plan(root)
    if phase not in PHASES:raise ValueError('Unknown Phi recovery phase')
    previous={k:checked(root,k,p) for k in (() if phase=='pilot' else ('pilot',) if phase=='train' else ('train',))}
    out=root/phase
    if out.exists():raise FileExistsError(out)
    out.mkdir();lease=settings=None;dataset=None;started=time.monotonic()
    try:
        lease,settings,guard=native.reservation()  # Always before any CUDA model/tensor allocation.
        torch.cuda.set_device(0);torch.set_num_threads(4);torch.manual_seed(CONFIG.seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        trainer,dataset=build_trainer(p,'cuda:0',guard)
        def event(value):native.event(out,'training_progress',**value)
        if phase=='pilot':
            initial=trainer.development();control=native_control(initial,read(Path(p['native_root'])/'native/baseline.json'))
            write(out/'initial_development.json',initial);initial_sha=trainer._save(out/'initializer.pt',0)
            longest=sorted(dataset.training,key=lambda i:(-len(dataset.records[i].input_ids),i))[:4]
            result=dict(status='completed',**trainer.pilot(longest),initializer_sha256=initial_sha,
                identities=trainer.identities,fixed=trainer.fixed,normalization=trainer.normalization,
                native_control=control,pilot_indices=longest,confirmed=False)
            if not result['passed']:raise ValueError('Discarded real-update pilot failed')
        elif phase=='train':
            pilot=previous['pilot'];training.load_selected(trainer,root/'pilot/initializer.pt',pilot['initializer_sha256'])
            if trainer.development()!=read(root/'pilot/initial_development.json'):raise ValueError('Fresh initializer replay differs before updates')
            result=trainer.fit(out/'recovery',event=event)
        else:
            result=training.verify_checkpoint(trainer,root/'train/recovery')
            result.update(quality=quality(result['metrics']),
                native_control=native_control(result['metrics'],read(Path(p['native_root'])/'native/baseline.json')),
                generation_recovery=False,speed_measured=False,full40=False)
        guard();write(out/'result.json',result)
        resource=dict(lease_id=lease['id'],allocation=lease['allocation'],allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        resources(resource)
        write(out/'manifest.json',dict(status='completed',phase=phase,fingerprint=p['fingerprint'],
            dependencies={k:sha(root/k/'manifest.json') for k in previous},
            artifacts={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file()},
            elapsed_seconds=time.monotonic()-started,**resource))
        return checked(root,phase,p)
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',phase=phase,fingerprint=p['fingerprint'],
            error=f'{type(error).__name__}: {error}',lease=lease,allocator=settings,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if dataset is not None:dataset.close()
        gc.collect()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','run','check'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--native-root',type=Path)
    parser.add_argument('--fit-root',type=Path);parser.add_argument('--data-root',type=Path)
    parser.add_argument('--phase',choices=PHASES)
    args=parser.parse_args()
    if args.command=='prepare':
        if any(v is None for v in (args.native_root,args.fit_root,args.data_root)):parser.error('prepare requires native/fit/data roots')
        result=prepare(args.root,args.native_root,args.fit_root,args.data_root)
    else:
        if args.phase is None:parser.error('run/check requires phase')
        result=run(args.root,args.phase) if args.command=='run' else checked(args.root,args.phase)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
