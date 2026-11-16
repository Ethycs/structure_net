"""CPU-only continuation of one frozen Phi native/capture/recovery pilot.

Waits for the already launched native worker. Every later GPU phase gets a
separate ordinary reservation; no retries, implicit adoption or full40 claim.
An interrupted launch without an exit receipt requires operator inspection.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='jspace.phi-campaign.v1'
PIXI='/home/rabbit/.pixi/bin/pixi'
ALLOCATION='GPU-0370fdb0=11264'
MINUTES=90
POLL_SECONDS=15
STEPS=('fit_prepare','fit_capture','fit_basis','fit_interface','fit_relation',
       'recovery_prepare','recovery_pilot','recovery_train','recovery_verify')
MODULES={name:'experiments.structure_net.jspace_phi_'+name for name in ('native','fit','recovery')}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')


def require_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU campaign requires CUDA hidden')


def environment():
    value=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PIXI_CACHE_DIR='/tmp/structure-net-pixi-cache',
        UV_CACHE_DIR='/tmp/structure-net-uv-cache')
    value.pop('IC_LEASE',None);value.pop('STRUCTURENET_GPU_LEASE',None)
    return value


def apis():return {key:importlib.import_module(module) for key,module in MODULES.items()}


def event(root,**value):
    value=dict(value,recorded_at=datetime.now(timezone.utc).isoformat())
    encoded=json.dumps(value,sort_keys=True,allow_nan=False)
    with (Path(root)/'events.jsonl').open('a') as stream:stream.write(encoded+'\n')
    print(encoded,flush=True)


@contextmanager
def exclusive(root):
    with (Path(root)/'controller.lock').open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Phi campaign controller holds the lock') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def prepare(root,native_root,*,fit_root=None,recovery_root=None):
    require_cpu();root,native_root=Path(root).resolve(),Path(native_root).resolve()
    fit_root=Path(fit_root or native_root/'fit').resolve()
    recovery_root=Path(recovery_root or native_root/'recovery').resolve()
    if root.exists():raise FileExistsError(root)
    if len({root,native_root,fit_root,recovery_root})!=4 or root.is_relative_to(fit_root) or root.is_relative_to(recovery_root):
        raise ValueError('Distinct controller/native/fit/recovery roots required')
    api=apis();native=api['native'].verify_plan(native_root,live=True)
    pointer=native_root/'native_launch_receipt.json';launch_pointer=read(pointer)
    launch_root=Path(launch_pointer['path']).resolve();started_path=launch_root/'started.json';started=read(started_path)
    plan_sha=sha(native_root/'plan.json')
    if (launch_pointer['plan_sha256']!=plan_sha or started['plan_sha256']!=plan_sha or
            started['fingerprint']!=native['fingerprint'] or Path(started['root']).resolve()!=native_root or
            started.get('automatic_retries')!=0):
        raise ValueError('Native launch does not match frozen native plan')
    command=started['command']
    if (command[:4]!=['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run'] or
            command.count('--allocation')!=1 or command[command.index('--allocation')+1]!=ALLOCATION or
            command[command.index('--minutes')+1]!=str(MINUTES) or MODULES['native'] not in command):
        raise ValueError('Unexpected native launch reservation/worker')
    begin=datetime.fromisoformat(started['started_at'])
    if begin.tzinfo is None:raise ValueError('Native start must carry timezone')
    extras=[Path(__file__),Path(api['fit'].__file__),Path(api['recovery'].__file__),
        ROOT/'pyproject.toml',ROOT/'pixi.lock',ROOT/'src/structure_net_resources/client.py',
        ROOT/'src/neural_architecture_lab/distillation_data.py',
        ROOT/'src/neural_architecture_lab/distillation_training.py',
        ROOT/'experiments/neural_architecture_lab/distillation_recovery.py']
    extras += [ROOT/f'tests/structure_net/test_jspace_phi_{name}.py' for name in
               ('campaign','native','fit','recovery','train','capture','affine_block','weights','data')]
    extras += [Path(name) for name in native['source_sha256']]
    files=api['fit'].local_import_closure(extras)
    pins={str(path):sha(path) for path in (native_root/'plan.json',native_root/'items.json',pointer,started_path)}
    p=dict(schema=SCHEMA,native_root=str(native_root),fit_root=str(fit_root),recovery_root=str(recovery_root),
        data_root=native['data_root'],native_fingerprint=native['fingerprint'],native_launch_root=str(launch_root),
        native_deadline=begin.timestamp()+MINUTES*60,steps=list(STEPS),allocation=ALLOCATION,lease_minutes=MINUTES,
        poll_seconds=POLL_SECONDS,automatic_retries=0,input_sha256=pins,
        source_sha256={str(path):sha(path) for path in sorted(files)},
        confirmed=False,full40=False,generation_recovery=False,
        scope='One block39 pilot on decoded quantized-source Phi-4; no speed or generation recovery claim')
    p['fingerprint']=digest(p);root.mkdir(parents=True)
    for name,expected in p['source_sha256'].items():
        target=root/'producer'/Path(name).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,target)
        if sha(target)!=expected:raise ValueError('Campaign producer archive differs')
    write(root/'plan.json',p)
    return dict(status='prepared',fingerprint=p['fingerprint'],steps=p['steps'])


def verify_plan(root):
    root=Path(root);p=read(root/'plan.json')
    if (p.get('schema')!=SCHEMA or p.get('fingerprint')!=digest({k:v for k,v in p.items() if k!='fingerprint'}) or
            p.get('steps')!=list(STEPS) or p.get('allocation')!=ALLOCATION or p.get('lease_minutes')!=MINUTES or
            p.get('automatic_retries')!=0 or p.get('poll_seconds')!=POLL_SECONDS or
            any(p.get(k)is not False for k in ('confirmed','full40','generation_recovery'))):
        raise ValueError('Frozen Phi campaign policy changed')
    for name,expected in p['input_sha256'].items():
        if sha(name)!=expected:raise ValueError('Frozen native input/launch changed')
    for name,expected in p['source_sha256'].items():
        if sha(name)!=expected or sha(root/'producer'/Path(name).relative_to(ROOT))!=expected:
            raise ValueError('Campaign source/archive changed')
    return p


def wait_native(root,p,*,api=None,now=time.time,sleep=time.sleep):
    api=api or apis();native_root=Path(p['native_root']);launch=Path(p['native_launch_root'])
    while True:
        if (native_root/'native/failure.json').exists() or (launch/'launch_error.json').exists():
            raise RuntimeError('Native worker/launcher failed; no automatic retry')
        exit_path=launch/'exit.json';manifest=native_root/'native/manifest.json'
        if exit_path.exists():
            exited=read(exit_path)
            if exited.get('returncode')!=0 or exited.get('status')!='worker_exited':
                raise RuntimeError('Native launcher exit failure; no automatic retry')
            if not manifest.exists():raise RuntimeError('Native launcher succeeded without completed manifest')
            if (exited.get('fingerprint')!=p['native_fingerprint'] or exited.get('log_sha256')!=sha(launch/'output.log') or
                    exited.get('native_manifest_sha256')!=sha(manifest)):
                raise ValueError('Native launch exit/log/manifest proof changed')
            proof=api['native'].checked(native_root)
            if proof['fingerprint']!=p['native_fingerprint']:raise ValueError('Completed native fingerprint changed')
            value=dict(status='verified',native_fingerprint=p['native_fingerprint'],
                manifest_sha256=sha(manifest),exit_sha256=sha(exit_path),log_sha256=sha(launch/'output.log'))
            target=Path(root)/'native_verified.json'
            if target.exists():
                if read(target)!=value:raise ValueError('Previously admitted native completion changed')
            else:write(target,value)
            return value
        if now()>=p['native_deadline']:raise TimeoutError('Native completion exceeded its90-minute launch window')
        event(root,kind='waiting_for_native',manifest_present=manifest.exists())
        sleep(min(POLL_SECONDS,max(0,p['native_deadline']-now())))


def phase_command(p,step):
    if step not in STEPS:raise ValueError('Unknown Phi campaign step')
    kind,phase=step.split('_',1);target=p[kind+'_root']
    inner=[PIXI,'run','-e','ncm','--locked','--no-install','--offline','python','-m',MODULES[kind]]
    if phase=='prepare':
        inner+=['prepare','--root',target,'--native-root',p['native_root'],'--data-root',p['data_root']]
        if kind=='recovery':inner+=['--fit-root',p['fit_root']]
        return inner
    return ['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run','--allocation',ALLOCATION,
        '--minutes',str(MINUTES),'--reason',f'Authorized Phi-4 block39 {kind} {phase}; independent GPU work untouched',
        '--',*inner,'run','--root',target,'--phase',phase]


def artifact_path(p,step):
    kind,phase=step.split('_',1);root=Path(p[kind+'_root'])
    return root/'plan.json' if phase=='prepare' else root/(phase+'_receipt.json') if kind=='fit' else root/phase/'manifest.json'


def admit_step(p,step,api):
    kind,phase=step.split('_',1);root=Path(p[kind+'_root']);worker=api[kind]
    worker_plan=worker.verify(root) if kind=='fit' else worker.verify_plan(root)
    if (Path(worker_plan['native_root']).resolve()!=Path(p['native_root']).resolve() or
            Path(worker_plan['data_root']).resolve()!=Path(p['data_root']).resolve() or
            worker_plan['native_fingerprint']!=p['native_fingerprint']):
        raise ValueError('Worker native/data provenance differs from this campaign')
    if kind=='recovery':
        fitted=api['fit'].verify(p['fit_root'])
        if (Path(worker_plan['fit_root']).resolve()!=Path(p['fit_root']).resolve() or
                worker_plan['fit_fingerprint']!=fitted['fingerprint']):
            raise ValueError('Recovery fitting provenance differs from this campaign')
    if phase!='prepare':worker.checked(root,phase,worker_plan)
    return dict(artifact=str(artifact_path(p,step)),artifact_sha256=sha(artifact_path(p,step)),
                fingerprint=worker_plan['fingerprint'])


def incomplete_output(p,step):
    kind,phase=step.split('_',1);root=Path(p[kind+'_root'])
    if phase=='prepare':return root.exists()
    if kind=='recovery':return (root/phase).exists()
    names={'capture':['capture'],'basis':['basis'],
        'interface':['basis/block_39/'+n for n in ('interface.json','compiled_dev.npy','output_weight.pt')],
        'relation':['basis/block_39/'+n for n in ('relation.pt','relations.json','rank_spectrum.pt','rank_diagnostic.json')]}
    return (root/(phase+'_failure.json')).exists() or any((root/n).exists() for n in names[phase])


def execute_step(root,p,step,api,*,executor=subprocess.run):
    root=Path(root);directory=root/'commands'/step;artifact=artifact_path(p,step)
    if directory.exists():
        if not (directory/'exit.json').exists():raise RuntimeError('Prior launch has no exit receipt; inspect before resuming')
        previous=read(directory/'exit.json')
        if previous.get('returncode')!=0 or previous.get('status')!='verified':
            raise RuntimeError('Prior command failed or lacks verified artifacts; no automatic retry')
        if (previous.get('started_sha256')!=sha(directory/'started.json') or
                previous.get('log_sha256')!=sha(directory/'output.log') or
                read(directory/'started.json').get('command')!=phase_command(p,step)):
            raise ValueError('Prior command/log lineage changed')
        proof=admit_step(p,step,api)
        if previous.get('proof')!=proof:raise ValueError('Previously completed worker artifact changed')
        return proof
    if artifact.exists():
        proof=admit_step(p,step,api)
        # This is an explicit external completion admission, never a fabricated
        # launch/exit. Revalidation on every restart is mandatory.
        target=root/(step+'_external.json')
        if target.exists() and read(target)!=proof:raise ValueError('External completed phase changed')
        if not target.exists():write(target,proof)
        return proof
    if incomplete_output(p,step):
        raise RuntimeError('Incomplete worker output exists; no duplicate launch')
    directory.mkdir(parents=True);command=phase_command(p,step)
    write(directory/'started.json',dict(step=step,command=command,fingerprint=p['fingerprint'],
        automatic_retries=0,started_at=datetime.now(timezone.utc).isoformat()))
    event(root,kind='command_started',step=step,command=command,log=str(directory/'output.log'))
    result=None;failure=None;proof=None
    try:
        with (directory/'output.log').open('xb') as output:
            result=executor(command,cwd=ROOT,env=environment(),stdout=output,stderr=subprocess.STDOUT,check=False)
        if result.returncode!=0:raise RuntimeError(f'Phi command {step} exited {result.returncode}; no automatic retry')
        proof=admit_step(p,step,api)
        return proof
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}';raise
    finally:
        value=dict(status='verified' if failure is None and proof is not None else 'failed',step=step,
            returncode=None if result is None else result.returncode,proof=proof,error=failure,
            started_sha256=sha(directory/'started.json'),log_sha256=sha(directory/'output.log'),
            ended_at=datetime.now(timezone.utc).isoformat())
        write(directory/'exit.json',value)
        event(root,kind='command_finished',step=step,status=value['status'],returncode=value['returncode'])


def run(root,*,executor=subprocess.run,now=time.time,sleep=time.sleep):
    require_cpu();root=Path(root).resolve();p=verify_plan(root)
    with exclusive(root):
        if (root/'failure.json').exists():raise RuntimeError('Campaign failure preserved; no automatic retry')
        api=apis()
        try:
            native=wait_native(root,p,api=api,now=now,sleep=sleep)
            verify_plan(root);proofs={}
            for step in STEPS:
                verify_plan(root)
                proofs[step]=execute_step(root,p,step,api,executor=executor)
            value=dict(status='completed',schema=SCHEMA,fingerprint=p['fingerprint'],native=native,steps=proofs,
                confirmed=False,full40=False,generation_recovery=False,
                scope=p['scope'],verification_result=str(Path(p['recovery_root'])/'verify/result.json'))
            if (root/'result.json').exists():
                if read(root/'result.json')!=value:raise ValueError('Campaign completed result changed')
            else:write(root/'result.json',value)
            event(root,kind='completed',verification_result=value['verification_result'])
            return value
        except BaseException as error:
            if not (root/'failure.json').exists():write(root/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],
                error=f'{type(error).__name__}: {error}',recorded_at=datetime.now(timezone.utc).isoformat()))
            raise


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--root',required=True);prep.add_argument('--native-root',required=True)
    prep.add_argument('--fit-root');prep.add_argument('--recovery-root')
    launch=commands.add_parser('run');launch.add_argument('--root',required=True)
    args=parser.parse_args();value=prepare(args.root,args.native_root,fit_root=args.fit_root,recovery_root=args.recovery_root) if args.command=='prepare' else run(args.root)
    print(json.dumps(value,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
