"""Prepare, start and reversibly stop the three-GPU research Triton model."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import requests

from experiments.word_games.common import read, sha, write
from experiments.word_games import target_70tps as target
from experiments.word_games.free_gpu_70tps import CONFIGS
from experiments.word_games.mtp_no_context_70tps import CONFIGS as MTP_CONFIGS
from experiments.word_games.gpu_services import ModelLease, SERVICES
from experiments.word_games.native_speed import GPU, IMAGE as NATIVE_IMAGE
from experiments.word_games.moe_placement import SUPER
from experiments.word_games.three_gpu_experts import SMALL
from experiments.word_games.prepare_qwen38_moe import WEIGHTS, SHA256

MODEL = 'word-games-qwen38-native'
CONTAINER = 'word-games-triton'
URL = 'http://127.0.0.1:8020'
IMAGE = 'sha256:ea2c80aa8efcf2b1578db2ab869ea1978a243b8327d5e4df9e95cabaaf74cc4f'
ROOT = Path('.pixi/word-games-triton')
SYSROOT = Path('.pixi/word-games-triton-native/sysroot')
SOURCE = Path(__file__).parent
PROFILES = {**target.FOLLOWUPS, **target.BATCH_MTP, **CONFIGS,
            **{k: v for k, v in MTP_CONFIGS.items() if k != 'control'}}
RESIDENT_CONTAINERS = ('triton', 'triton-ocr')


def weight_identity():
    stat = WEIGHTS.stat()
    return {k: getattr(stat, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def save_state(value):
    # This is a mutable runtime pointer, not append-only experiment evidence.
    path = ROOT / 'state.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def api(action):
    with requests.post(URL + f'/v2/repository/models/{MODEL}/{action}', json={}, timeout=300,
                       headers={'Connection': 'close'}) as r:
        body = r.text
        r.raise_for_status()
        return {'status': r.status_code, 'body': body}


def prepare(profile):
    if ROOT.exists():
        raise ValueError('deployment exists; preserve it rather than overwrite')
    identity = weight_identity()
    if sha(WEIGHTS) != SHA256 or weight_identity() != identity:
        raise ValueError('checkpoint changed')
    manifest = read(target.cpu.BUILD)['runtimes']['candidate']['hashes']
    if any(sha(target.cpu.RUNTIME / n) != v for n, v in manifest.items()):
        raise ValueError('native runtime changed')
    if not (SYSROOT / 'ld-linux-x86-64.so.2').exists():
        SYSROOT.mkdir(parents=True, exist_ok=True)
        export = 'word-games-native-libraries-' + str(os.getpid())
        subprocess.run(['docker', 'create', '--name', export, NATIVE_IMAGE], check=True, capture_output=True)
        try:
            subprocess.run(['docker', 'cp', export + ':/usr/lib/x86_64-linux-gnu/.', str(SYSROOT)], check=True)
        finally:
            subprocess.run(['docker', 'rm', export], check=True, capture_output=True)
    config = PROFILES[profile]
    args = target.command(config, None, 18080, CONTAINER)
    args = args[args.index(NATIVE_IMAGE) + 1:]
    for flag, value in (('--host', '127.0.0.1'), ('--port', '18080'), ('--verbosity', '4')):
        args[args.index(flag) + 1] = value
    modeldir = ROOT / 'models' / MODEL
    (modeldir / '1').mkdir(parents=True)
    (ROOT / 'runtime').mkdir()
    for source, dest in (('model.py', '1/model.py'), ('config.pbtxt', 'config.pbtxt')):
        shutil.copyfile(SOURCE / source, modeldir / dest)
    write(modeldir / '1/runtime.json', {'profile': profile, 'native_args': args})
    write(ROOT / 'manifest.json', {'profile': profile, 'config': asdict(config),
        'weights': str(WEIGHTS.resolve()), 'weights_sha256': SHA256,
        'weight_size': WEIGHTS.stat().st_size, 'native_runtime_hashes': manifest,
        'verified_weight_identity': identity,
        'triton_image': IMAGE, 'native_sysroot_image': NATIVE_IMAGE,
        'sysroot_hashes': {p.name: sha(p) for p in sorted(SYSROOT.iterdir()) if p.is_file() and not p.is_symlink()},
        'repository_hashes': {str(p.relative_to(ROOT)): sha(p) for p in sorted(modeldir.rglob('*')) if p.is_file()}})
    print('Prepared', MODEL, profile, flush=True)


def command():
    devices = ','.join((GPU, SUPER, SMALL))
    cache = Path('.pixi/word-games-cuda-cache').resolve()
    cache.mkdir(exist_ok=True)
    cmd = ['docker', 'run', '-d', '--name', CONTAINER, '--gpus', '"device=' + devices + '"',
           '--shm-size', '1g', '--stop-timeout', '60',
           '-e', 'CUDA_VISIBLE_DEVICES=' + devices, '-e', 'CUDA_CACHE_PATH=/cuda-cache',
           '-e', 'CUDA_CACHE_MAXSIZE=2147483648',
           '--entrypoint', '/opt/tritonserver/bin/tritonserver']
    for host, guest in ((8020, 8000), (8021, 8001), (8022, 8002)):
        cmd += ['-p', f'127.0.0.1:{host}:{guest}']
    for host, guest, mode in ((ROOT / 'models', '/models', 'ro'), (ROOT / 'runtime', '/runtime', 'rw'),
            (SYSROOT, '/native-sysroot', 'ro'), (target.cpu.RUNTIME, '/llama', 'ro'),
            (WEIGHTS, '/model.gguf', 'ro'), (Path('/usr/local/cuda/lib64'), '/cuda', 'ro'),
            (cache, '/cuda-cache', 'rw')):
        cmd += ['-v', f'{host.resolve()}:{guest}:{mode}']
    return cmd + [IMAGE, '--model-repository=/models', '--backend-directory=/opt/tritonserver/backends',
                  '--model-control-mode=explicit', '--exit-timeout-secs=45', '--log-verbose=0',
                  '--backend-config=python,stub-timeout-seconds=300']


def restore(evidence):
    restore_containers(evidence)
    lease = ModelLease(evidence / 'services')
    lease.before = read(lease.out / 'before.json')
    lease.attempted = list(SERVICES)
    lease.restore()


def restore_containers(evidence):
    states = read(evidence / 'resident_containers.json')
    for name, was_running in states.items():
        if was_running:
            subprocess.run(['docker', 'start', name], check=True, capture_output=True)
    for s in SERVICES:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if requests.get(s.url + '/v2/health/ready', timeout=3).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError('restored server readiness: ' + s.url)


def stop():
    state = read(ROOT / 'state.json')
    evidence = Path(state['evidence'])
    # Stop the owning container even if the model endpoint is unhealthy.
    unload = None
    try:
        unload = api('unload')
    except requests.RequestException as exc:
        unload = {'error': repr(exc)}
    stopped = subprocess.run(['docker', 'stop', '--time', '60', CONTAINER], capture_output=True, text=True)
    if stopped.returncode:
        raise RuntimeError('could not stop dedicated Triton container: ' + stopped.stderr)
    subprocess.run(['docker', 'rm', CONTAINER], check=True, capture_output=True)
    restore(evidence)
    write(evidence / 'shutdown.json', {'unload': unload, 'old_models_restored': True})
    save_state({**state, 'running': False})
    print('Stopped WORD_GAMES; Qwen3-8B and OCR restored', flush=True)


def start(evidence):
    manifest = read(ROOT / 'manifest.json')
    for file, expected in manifest['repository_hashes'].items():
        if sha(ROOT / file) != expected:
            raise ValueError('prepared repository changed')
    for file, expected in manifest['native_runtime_hashes'].items():
        if sha(target.cpu.RUNTIME / file) != expected:
            raise ValueError('native runtime changed')
    for file, expected in manifest['sysroot_hashes'].items():
        if sha(SYSROOT / file) != expected:
            raise ValueError('isolated native libraries changed')
    # Reuse the full-file hash only while inode, size and both timestamps match.
    # A touched/replaced checkpoint requires a fresh content verification.
    identity = weight_identity()
    cached_hash = identity == manifest.get('verified_weight_identity')
    if not cached_hash and (sha(WEIGHTS) != manifest['weights_sha256'] or weight_identity() != identity):
        raise ValueError('checkpoint changed')
    evidence.mkdir(parents=True, exist_ok=False)
    write(evidence / 'manifest.json', manifest)
    write(evidence / 'weight_check.json', {'hash_receipt_reused': cached_hash, 'identity': identity,
                                          'sha256': manifest['weights_sha256']})
    write(evidence / 'source_hashes.json', {str(p): sha(p) for p in SOURCE.glob('*') if p.is_file()})
    (evidence / 'producer_source').mkdir()
    for p in SOURCE.glob('*'):
        if p.is_file():
            shutil.copyfile(p, evidence / 'producer_source' / p.name)
    states = {name: subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                                            text=True).strip() == 'true' for name in RESIDENT_CONTAINERS}
    write(evidence / 'resident_containers.json', states)
    lease = ModelLease(evidence / 'services')
    started = False
    try:
        lease.acquire()
        # Release the old servers' CUDA contexts as well as their model weights.
        for name in RESIDENT_CONTAINERS:
            subprocess.run(['docker', 'stop', '--time', '60', name], check=True, capture_output=True)
        cmd = command()
        write(evidence / 'container_command.json', cmd)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        started = True
        write(evidence / 'container.json', {'id': result.stdout.strip()})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if requests.get(URL + '/v2/health/ready', timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError('Triton startup')
        write(evidence / 'load.json', api('load'))
        requests.get(URL + f'/v2/models/{MODEL}/ready', timeout=10).raise_for_status()
        save_state({'evidence': str(evidence), 'running': True})
        print('READY', URL, MODEL, manifest['profile'], flush=True)
    except BaseException as exc:
        write(evidence / 'failure.json', {'error': repr(exc),
              'response': exc.response.text if isinstance(exc, requests.HTTPError) and exc.response is not None else None})
        if started:
            logs = subprocess.run(['docker', 'logs', CONTAINER], capture_output=True, text=True)
            (evidence / 'failed_triton.log').write_text(logs.stdout + logs.stderr)
            subprocess.run(['docker', 'stop', '--time', '60', CONTAINER], capture_output=True)
            subprocess.run(['docker', 'rm', CONTAINER], capture_output=True)
        if lease.attempted:
            restore_containers(evidence)
            lease.restore()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='action', required=True)
    a = commands.add_parser('prepare')
    a.add_argument('--profile', required=True, choices=PROFILES)
    a = commands.add_parser('start')
    a.add_argument('--evidence', required=True, type=Path)
    commands.add_parser('stop')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.profile)
    elif a.action == 'start':
        start(a.evidence)
    else:
        stop()
