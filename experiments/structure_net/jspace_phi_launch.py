"""One append-only, reservation-backed Phi native launch; no automatic retry.

The worker authenticates scientific inputs and output artifacts. This stdlib
wrapper records the command/process/manifest lineage and never initializes CUDA.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'experiments.structure_net.jspace_phi_native'
ALLOCATION = 'GPU-0370fdb0=11264'
MINUTES = 90


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def command(root):
    return ['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run',
        '--allocation',ALLOCATION,'--minutes',str(MINUTES),'--reason',
        'Authorized Phi-4 decoded-source native prefix pilot; independent2060 untouched',
        '--','/home/rabbit/.pixi/bin/pixi','run','-e','ncm','--locked','--no-install','--offline',
        'python','-m',MODULE,'run','--root',str(Path(root).resolve())]


def environment():
    value=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PIXI_CACHE_DIR='/tmp/structure-net-pixi-cache',UV_CACHE_DIR='/tmp/structure-net-uv-cache')
    value.pop('IC_LEASE',None)
    value.pop('STRUCTURENET_GPU_LEASE',None)
    return value


@contextmanager
def exclusive(root):
    with (Path(root)/'native_launcher.lock').open('a+') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Phi native launcher owns this root') from None
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def launch(root,receipt_root,*,executor=subprocess.run):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Launch wrapper must start with CUDA hidden')
    root,receipt_root=Path(root).resolve(),Path(receipt_root).resolve()
    with exclusive(root):
        plan=read(root/'plan.json');fingerprint=plan['fingerprint']
        if (root/'native').exists():raise FileExistsError('Native phase already exists; inspect it instead of relaunching')
        if (root/'native_launch_receipt.json').exists():raise FileExistsError('Prior launch preserved; inspect before a new attempt')
        receipt_root.mkdir(parents=True,exist_ok=False)
        # The stable pointer prevents an ambiguous interrupted launch from being
        # duplicated under a different receipt-directory name.
        write(root/'native_launch_receipt.json',dict(path=str(receipt_root),plan_sha256=sha(root/'plan.json')))
        argv=command(root)
        write(receipt_root/'started.json',dict(command=argv,root=str(root),fingerprint=fingerprint,
            plan_sha256=sha(root/'plan.json'),launcher_sha256=sha(__file__),
            started_at=datetime.now(timezone.utc).isoformat(),automatic_retries=0))
        with (receipt_root/'output.log').open('xb') as output:
            process=executor(argv,cwd=ROOT,env=environment(),stdout=output,stderr=subprocess.STDOUT,check=False)
        result=dict(status='worker_exited',returncode=process.returncode,fingerprint=fingerprint,
            log_sha256=sha(receipt_root/'output.log'),ended_at=datetime.now(timezone.utc).isoformat(),
            confirmed=False,scientific_artifacts_verified=False)
        if process.returncode==0:
            manifest=root/'native/manifest.json'
            if manifest.exists():result['native_manifest_sha256']=sha(manifest)
            else:result['status']='worker_exit_without_manifest'
        write(receipt_root/'exit.json',result)
        if process.returncode!=0 or result['status']!='worker_exited':
            raise RuntimeError('Native launch failed/incomplete; preserved receipt: '+str(receipt_root))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--receipt-root',type=Path);parser.add_argument('--print-command',action='store_true')
    args=parser.parse_args()
    if args.print_command:print(shlex.join(command(args.root)))
    elif args.receipt_root is None:parser.error('Launch requires a new --receipt-root')
    else:print(json.dumps(launch(args.root,args.receipt_root),sort_keys=True,allow_nan=False))
