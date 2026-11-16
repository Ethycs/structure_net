"""CPU supervisor for the authorized, gate-preserving 15-to-36 block ladder.

Workers execute only inside inference-ctl reservations. This schedules existing
study-specific phases; it is not an additional general experiment scheduler.
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
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'jspace.3b-progressive-ladder.v1'
PROTOCOL = ROOT / 'docs/07 - Status Reports/jspace/2026-11-15_qwen3b-progressive-to-full.md'
MODULE = 'experiments.structure_net.jspace_3b_progressive_stage'
PHASES = ('boundary', 'capture', 'fit', 'parentfeatures', 'pilot', 'prealign', 'train', 'verify')
PIXI = '/home/rabbit/.pixi/bin/pixi'
SINGLE = 'GPU-0370fdb0=11264'
DUAL = SINGLE + ',GPU-7b1db3cb=7168'
GIB = 1024**3
STAGE_GIB, HEADROOM_GIB = 40, 64
RETRY_DELAYS = (5, 15)


def strict(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(strict(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(strict(value) + '\n')


def stage_api():
    from . import jspace_3b_progressive_stage
    return jspace_3b_progressive_stage


def cpu_environment():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PIXI_CACHE_DIR='/tmp/structure-net-pixi-cache', UV_CACHE_DIR='/tmp/structure-net-uv-cache')
    env.pop('IC_LEASE', None)
    return env


def require_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Start the CPU controller with CUDA_VISIBLE_DEVICES empty')


def capacity(path, remaining_stages, *, disk_usage=shutil.disk_usage):
    if type(remaining_stages) is not int or remaining_stages < 0:
        raise ValueError('Nonnegative remaining stage count required')
    anchor = Path(path).resolve()
    while not anchor.exists():
        anchor = anchor.parent
    required = (remaining_stages * STAGE_GIB + HEADROOM_GIB) * GIB
    free = disk_usage(anchor).free
    if free < required:
        raise RuntimeError(f'Insufficient storage: {free} free bytes, {required} forecast bytes required')
    return dict(filesystem_path=str(anchor), free_bytes=free, forecast_bytes=required,
                remaining_stages=remaining_stages, per_stage_gib=STAGE_GIB, headroom_gib=HEADROOM_GIB)


def prepare(root, parent_root, *, target_blocks=36):
    require_cpu()
    root, parent_root = Path(root).resolve(), Path(parent_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    api = stage_api()
    parent, verification, trained = api.admit_parent(parent_root)
    start = 36 - parent['cut']
    if not 15 <= start < target_blocks <= 36:
        raise ValueError('Continue a verified 15-or-more block model toward at most36')
    space = capacity(root, target_blocks-start)
    sources = api.source.inherited.local_import_closure([
        Path(__file__), Path(api.__file__), PROTOCOL,
        ROOT / 'tests/structure_net/test_jspace_3b_progressive_ladder.py',
        ROOT / 'tests/structure_net/test_jspace_3b_progressive_stage.py',
        ROOT / 'tests/structure_net/test_jspace_3b_progressive_capture.py'])
    evidence = {str((parent_root / relative).resolve()): sha(parent_root / relative)
                for relative in ('plan.json', 'verify/manifest.json', 'verify/result.json',
                                 'train/manifest.json', 'train/result.json')}
    plan = dict(schema=SCHEMA, parent_root=str(parent_root), initial_blocks=start,
        target_blocks=target_blocks, blocks=list(range(start+1, target_blocks+1)),
        initial_checkpoint=verification['review_checkpoint'],
        initial_checkpoint_sha256=verification['review_checkpoint_sha256'],
        initial_development=trained['candidate']['metrics'],
        seed_evidence_sha256=evidence,
        phases=list(PHASES), single_allocation=SINGLE, dual_allocation=DUAL,
        lease_minutes=90, retry_delays_seconds=list(RETRY_DELAYS),
        stage_gib=STAGE_GIB, headroom_gib=HEADROOM_GIB, preparation_storage=space,
        source_sha256={str(p): sha(p) for p in sorted(sources)},
        selection='original-native DEV gates; exact verification; previous passing checkpoint fallback',
        quality_scope='exploratory short multiple-choice continuation; no generation recovery claim',
        confirmed=False)
    plan['fingerprint'] = digest(plan)
    root.mkdir(parents=True)
    for path, expected in plan['source_sha256'].items():
        output = root / 'producer' / Path(path).relative_to(ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
        if sha(output) != expected:
            raise ValueError('Archived producing source changed')
    write(root / 'plan.json', plan)
    return dict(status='prepared', root=str(root), fingerprint=plan['fingerprint'],
                initial_blocks=start, target_blocks=target_blocks, storage=space, confirmed=False)


def verify_plan(root, *, live=True):
    root = Path(root).resolve()
    p = read(root / 'plan.json')
    payload = dict(p); fingerprint = payload.pop('fingerprint')
    if (p.get('schema') != SCHEMA or digest(payload) != fingerprint or p.get('confirmed') is not False
            or p.get('phases') != list(PHASES) or p.get('single_allocation') != SINGLE
            or p.get('dual_allocation') != DUAL or p.get('lease_minutes') != 90
            or p.get('retry_delays_seconds') != list(RETRY_DELAYS)
            or p.get('stage_gib') != STAGE_GIB or p.get('headroom_gib') != HEADROOM_GIB
            or not 15 <= p['initial_blocks'] < p['target_blocks'] <= 36
            or p['blocks'] != list(range(p['initial_blocks']+1, p['target_blocks']+1))):
        raise ValueError('Frozen progression policy changed')
    for path, expected in p['source_sha256'].items():
        archive = root / 'producer' / Path(path).relative_to(ROOT)
        if sha(archive) != expected or live and sha(path) != expected:
            raise ValueError('Producing source changed: ' + path)
    for path, expected in p['seed_evidence_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Seed model evidence changed: ' + path)
    return p


def phase_command(stage_root, phase):
    if phase not in PHASES:
        raise ValueError('Unknown progressive phase')
    allocation = DUAL if phase in ('pilot', 'train') else SINGLE
    return ['/usr/bin/python3', '/usr/local/bin/inference-ctl', 'lease', 'run',
        '--allocation', allocation, '--minutes', '90', '--reason',
        f'Authorized Qwen3B progressive {Path(stage_root).name} {phase}; independent2060 untouched',
        '--', PIXI, 'run', '--locked', '--offline', '-e', 'ncm', 'python', '-m', MODULE,
        phase, '--root', str(Path(stage_root).resolve())]


def event(root, **value):
    value = dict(value, recorded_at=datetime.now(timezone.utc).isoformat())
    with (Path(root) / 'events.jsonl').open('a') as stream:
        stream.write(strict(value) + '\n')
    print(strict(value), flush=True)


@contextmanager
def exclusive(root):
    with (Path(root) / 'controller.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another progressive controller owns this campaign') from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def transient_prelaunch(log_path, phase_path):
    if Path(phase_path).exists():
        return False
    with Path(log_path).open('rb') as stream:
        stream.seek(max(0, Path(log_path).stat().st_size-8192))
        tail = stream.read().decode(errors='replace')
    return any(s in tail for s in ("can't reach Infisical", 'No route to host',
                                  'Connection refused', 'timed out'))


def execute_phase(root, stage_root, phase, *, executor=subprocess.run, sleep=time.sleep):
    """Retry only known transient failures before the worker created its phase."""
    root, stage_root = Path(root), Path(stage_root)
    for attempt in range(1, len(RETRY_DELAYS)+2):
        directory = root / 'commands' / stage_root.name / phase / f'attempt_{attempt:02d}'
        output = directory / 'output.log'
        if directory.exists():
            if not (directory / 'exit.json').exists():
                raise RuntimeError('Unfinished prior launch: inspect its live lease/process before resuming')
            previous = read(directory / 'exit.json')
            if previous['returncode'] == 0:
                raise RuntimeError('Successful command lacks an admitted phase; inspect its artifacts')
            if not transient_prelaunch(output, stage_root / phase):
                raise RuntimeError('Failed worker attempt preserved; no automatic rerun')
        else:
            directory.mkdir(parents=True)
            command = phase_command(stage_root, phase)
            write(directory / 'started.json', dict(command=command,
                started_at=datetime.now(timezone.utc).isoformat()))
            event(root, kind='phase_started', stage=stage_root.name, phase=phase,
                  attempt=attempt, log=str(output.resolve()))
            with output.open('xb') as stream:
                result = executor(command, cwd=ROOT, env=cpu_environment(),
                                  stdout=stream, stderr=subprocess.STDOUT, check=False)
            write(directory / 'exit.json', dict(returncode=result.returncode,
                log_sha256=sha(output), ended_at=datetime.now(timezone.utc).isoformat()))
            if result.returncode == 0:
                return
            if not transient_prelaunch(output, stage_root / phase):
                raise RuntimeError(f'Worker failed with status{result.returncode}; see {output}')
        if attempt <= len(RETRY_DELAYS):
            event(root, kind='prelaunch_retry', stage=stage_root.name, phase=phase,
                  attempt=attempt, delay_seconds=RETRY_DELAYS[attempt-1])
            sleep(RETRY_DELAYS[attempt-1])
    raise RuntimeError('Transient prelaunch retry budget exhausted')


def stage_record(directory, parent_root, blocks, result, trained):
    expected = blocks if result['development_passed'] else blocks-1
    if (result.get('passed') is not True or result.get('compressed_blocks') != expected
            or result.get('promoted') is not False or result.get('confirmed') is not False
            or result.get('full36') is not (blocks == 36 and result['development_passed'])):
        raise ValueError('Stage verification does not establish its declared structural extent')
    return dict(stage_root=str(Path(directory).resolve()), parent_root=str(Path(parent_root).resolve()),
        attempted_blocks=blocks, compressed_blocks=expected,
        checkpoint=result['review_checkpoint'], checkpoint_sha256=result['review_checkpoint_sha256'],
        development_passed=result['development_passed'], attempted_development=trained['candidate']['metrics'],
        verify_manifest_sha256=sha(Path(directory)/'verify/manifest.json'),
        verify_result_sha256=sha(Path(directory)/'verify/result.json'),
        confirmed=False, speed_measured=False)


def finish(root, p, status, rows, last, **detail):
    value = dict(schema=SCHEMA, fingerprint=p['fingerprint'], status=status,
        target_blocks=p['target_blocks'], completed_stages=len(rows), stages=rows,
        last_passing=last, confirmed=False, **detail)
    write(Path(root)/'campaign_results.json', value)
    event(root, kind='campaign_terminal', status=status, compressed_blocks=last['compressed_blocks'])
    return value


def run(root):
    require_cpu()
    root = Path(root).resolve()
    with exclusive(root):
        p = verify_plan(root)
        if (root/'campaign_results.json').exists():
            return status(root)
        api = stage_api()
        rows = []
        parent_root = Path(p['parent_root'])
        last = dict(compressed_blocks=p['initial_blocks'], checkpoint=p['initial_checkpoint'],
                    checkpoint_sha256=p['initial_checkpoint_sha256'], development=p['initial_development'])
        try:
            for blocks in p['blocks']:
                verify_plan(root)
                directory = root/'stages'/f'blocks_{blocks:02d}'
                if not directory.exists():
                    storage = capacity(root, p['target_blocks']-blocks+1)
                    event(root, kind='stage_preparing', compressed_blocks=blocks, storage=storage)
                    api.prepare(directory, parent_root)
                elif not (directory/'plan.json').exists():
                    raise RuntimeError('Partial stage preparation preserved; inspect before another attempt')
                step = api.verify(directory)
                if step['cut'] != 36-blocks or Path(step['parent_root']).resolve() != parent_root.resolve():
                    raise ValueError('Progressive stage skips a layer or changes its parent')
                for phase in PHASES:
                    phase_path = directory/phase
                    if (phase_path/'manifest.json').exists():
                        api.checked(directory, phase, step)
                        event(root, kind='phase_reused', stage=directory.name, phase=phase)
                        continue
                    if phase_path.exists():
                        raise RuntimeError('Incomplete phase preserved; no blind duplicate launch')
                    execute_phase(root, directory, phase)
                    api.checked(directory, phase, step)
                    event(root, kind='phase_completed', stage=directory.name, phase=phase)
                result = api.checked(directory, 'verify', step)
                trained = api.checked(directory, 'train', step)
                row = stage_record(directory, parent_root, blocks, result, trained)
                record_path = directory/'ladder_record.json'
                if record_path.exists():
                    if read(record_path) != row:
                        raise ValueError('Completed ladder record changed')
                else:
                    write(record_path, row)
                rows.append(row)
                if not row['development_passed']:
                    return finish(root, p, 'stopped_development_gate', rows, last)
                last = dict(compressed_blocks=blocks, checkpoint=row['checkpoint'],
                    checkpoint_sha256=row['checkpoint_sha256'], development=row['attempted_development'])
                event(root, kind='stage_verified', **row)
                parent_root = directory
            final = 'structurally_full' if p['target_blocks'] == 36 else 'target_depth_reached'
            return finish(root, p, final, rows, last)
        except Exception as error:
            finish(root, p, 'execution_failed', rows, last, error=f'{type(error).__name__}: {error}')
            raise


def status(root):
    root = Path(root).resolve()
    p = verify_plan(root, live=False)
    rows = []
    for blocks in p['blocks']:
        path = root/'stages'/f'blocks_{blocks:02d}'/'ladder_record.json'
        if not path.exists():
            break
        row = read(path)
        if (row.get('attempted_blocks') != blocks or row.get('confirmed') is not False
                or sha(path.parent/'verify/manifest.json') != row['verify_manifest_sha256']
                or sha(path.parent/'verify/result.json') != row['verify_result_sha256']):
            raise ValueError('Ladder record is not bound to its verification')
        rows.append(row)
    if (root/'campaign_results.json').exists():
        result = read(root/'campaign_results.json')
        if result['fingerprint'] != p['fingerprint'] or result['stages'] != rows or result.get('confirmed') is not False:
            raise ValueError('Campaign result differs from completed stage evidence')
        return result
    last_event = None
    events = root/'events.jsonl'
    if events.exists():
        with events.open('rb') as stream:
            stream.seek(max(0, events.stat().st_size-16384))
            lines = stream.read().splitlines()
        if lines:
            last_event = json.loads(lines[-1])
    active = False
    if (root/'controller.lock').exists():
        with (root/'controller.lock').open('a+') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return dict(schema=SCHEMA, status='running' if active else ('interrupted' if last_event else 'prepared'),
        fingerprint=p['fingerprint'], initial_blocks=p['initial_blocks'], target_blocks=p['target_blocks'],
        completed_stages=len(rows), stages=rows, last_event=last_event, confirmed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'status'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path)
    parser.add_argument('--target-blocks', type=int, default=36)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.parent_root is None:
            parser.error('prepare requires --parent-root')
        result = prepare(args.root, args.parent_root, target_blocks=args.target_blocks)
    elif args.phase == 'run':
        result = run(args.root)
    else:
        result = status(args.root)
    print(strict(result), flush=True)


if __name__ == '__main__':
    main()
