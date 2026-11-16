"""Serial, reservation-managed matched attention retry after terminal parent evidence.

The controller freezes its producer before waiting, never fills gaps in parent
GPU work, and never retries a producer. Quality failure is not systems failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from . import jspace_3b_usable_recovery as parent
from .jspace_3b_full_progressive_controller import run_process

ROOT = parent.ROOT
PIXI = '/home/rabbit/.pixi/bin/pixi'
MODULE = 'experiments.structure_net.jspace_3b_attention_retry'
SCHEMA = 'jspace.3b-attention-retry-controller.v1'
ALLOCATION = 'GPU-8cf440df=3584'
ARMS = ('softmax', 'quadratic')
LABELS = ('update_0000', 'update_0071')
PARENT_LABELS = tuple(f'update_{i:04d}' for i in (0, 71, 134, 199, 278))
MINUTES = dict(pilot=90, run=180, generate=180)
LEASE_WAIT_SECONDS = 300
read, write, sha = parent.read, parent.write, parent.sha


def retry_worker():
    from . import jspace_3b_attention_retry
    return jspace_3b_attention_retry


def parent_ready(root):
    """Terminal controller plus authenticated five-way evidence; no NLL fallback."""
    root = Path(root)
    control = root.parent/'controller_v1'
    if (control/'failure.json').exists():
        return dict(status='blocked', reason='parent_controller_failed')
    if not (control/'result.json').exists() or not (control/'manifest.json').exists():
        return dict(status='pending', reason='parent_controller_not_terminal')
    result = read(control/'result.json')
    if result.get('status') != 'completed' or result.get('all_judgments_available') is not True:
        return dict(status='blocked', reason='parent_failed_or_judgments_unavailable')
    try:
        cp = read(control/'plan.json')
        parent.receipts._completed(control, cp)
        wp, _, _ = parent.verify(root, live=False)
        trained = parent.receipts._completed(root/'run', wp)
        if tuple(row['label'] for row in trained['candidates']) != PARENT_LABELS:
            raise ValueError('Parent must expose exactly five frozen candidates')
        for label in PARENT_LABELS:
            generated = parent.receipts._completed(root/'generation'/label, wp)
            judged = parent.receipts._completed(root/'judgment'/label, wp)
            if generated.get('label') != label or judged.get('label') != label:
                raise ValueError('Parent evidence label differs')
            if judged.get('evidence_available') is not True:
                return dict(status='blocked', reason='parent_judgment_unavailable', label=label)
        selected = parent.check(root)
        if not selected.get('selection_final') or selected.get('generation_or_judgment_pending'):
            return dict(status='blocked', reason='parent_selection_not_final')
        return dict(status='ready', selection=selected['task_selection'])
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return dict(status='blocked', reason='parent_terminal_evidence_invalid', error=str(error))


def wait_parent(root, seconds, verify, *, clock=time.monotonic, sleeper=time.sleep):
    deadline = clock() + seconds
    while True:
        verify()
        result = parent_ready(root)
        if result['status'] != 'pending':
            return result
        remaining = deadline - clock()
        if remaining <= 0:
            return dict(result, reason='parent_wait_expired')
        sleeper(min(30, remaining))


class Controller:
    def __init__(self, root, plan):
        self.root = Path(root)
        self.directory = self.root/'controller_v1'
        self.plan = plan
        self.worker_plan = None
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PIXI_CACHE_DIR='/tmp/structure-net-pixi-cache',
            UV_CACHE_DIR='/tmp/structure-net-uv-cache')

    def verify(self):
        for path, wanted in self.plan['source_sha256'].items():
            archive = self.directory/'producer'/Path(path).relative_to(ROOT)
            if sha(path) != wanted or sha(archive) != wanted:
                raise ValueError('Frozen controller producer changed: '+path)
        if self.worker_plan is not None:
            verified = retry_worker().verify(self.root, live=True)
            if verified['fingerprint'] != self.worker_plan['fingerprint']:
                raise ValueError('Worker plan changed')

    def event(self, phase, **values):
        parent.receipts._progress(self.directory, phase, **values)

    def command(self, phase, arm, label=None):
        self.verify()
        command = [PIXI, 'run', '--locked', '--offline', '-e', 'ncm', 'python', '-m', MODULE,
            phase, '--root', str(self.root), '--arm', arm]
        if label is not None:
            command += ['--label', label]
        if phase in MINUTES:
            command = ['/usr/bin/python3', '/usr/local/bin/inference-ctl', 'lease', 'run',
                '--allocation', ALLOCATION, '--minutes', str(MINUTES[phase]),
                '--wait', str(self.plan['lease_wait_seconds']), '--reason',
                'Matched one-block attention retry '+arm+' '+phase, '--', *command]
        elif phase != 'judge':
            raise ValueError('Unplanned worker phase')
        name = '_'.join(filter(None, (phase, arm, label)))
        log = self.directory/(name+'.log')
        self.event('launch', worker_phase=phase, arm=arm, label=label, command=command)
        # Reuse process-group/lease cleanup, not invoke's prelaunch retry loop.
        with log.open('x') as stream:
            result = run_process(command, cwd=ROOT, env=self.env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=None if phase in MINUTES else 21600)
        if result.returncode:
            raise RuntimeError(f'{name} failed; no rerun; evidence preserved at {log}')

    def phase(self, phase, arm, label=None):
        if arm not in ARMS or (phase in ('generate', 'judge') and label not in LABELS):
            raise ValueError('Unplanned arm or candidate')
        self.verify()
        directory = self.root/'arms'/arm/({'judge':'judgment', 'generate':'generation'}.get(phase, phase))
        if label is not None:
            directory /= label
        if not (directory/'manifest.json').exists():
            if directory.exists():
                raise RuntimeError('Partial phase preserved; no duplicate launch: '+str(directory))
            self.command(phase, arm, label)
        result = parent.receipts._completed(directory, self.worker_plan)
        if phase == 'pilot' and not all(result.get(key) is True for key in ('passed', 'discarded', 'fresh_initializer')):
            raise ValueError('Disposable pilot did not qualify a fresh scientific start')
        if phase == 'run' and (result.get('systems_passed') is not True
                or result.get('coverage', {}).get('planned_run_completed') is not True
                or tuple(row['label'] for row in result.get('candidates', [])) != LABELS):
            raise ValueError('Scientific run lacks complete matched coverage')
        if label is not None and result.get('label') != label:
            raise ValueError('Candidate evidence label differs')
        return result


def execute(controller):
    pilots = {arm:controller.phase('pilot', arm) for arm in ARMS}
    runs, generations, judgments = {}, {}, {}
    for arm in ARMS:
        runs[arm] = controller.phase('run', arm)
        generations[arm], judgments[arm] = {}, {}
        for label in LABELS:
            generations[arm][label] = controller.phase('generate', arm, label)
            # GPU subprocess (and lease) has exited before CPU/global-lock judge.
            try:
                judgments[arm][label] = controller.phase('judge', arm, label)
            except Exception as error:
                judgments[arm][label] = dict(status='unavailable', evidence_available=False,
                    qualified=None, error_type=type(error).__name__, error=str(error))
    controller.verify()
    checked = retry_worker().check(controller.root)
    available = all(row.get('evidence_available') is True for arm in judgments.values() for row in arm.values())
    return dict(status='completed' if available else 'blocked', pilots=pilots, runs=runs,
        generations=generations, judgments=judgments, paired_quality=checked,
        all_judgments_available=available, quality_failure_stops_evaluation=False,
        automatic_extra_training=False, partial_producers_retried=False,
        exposed_dev=True, population_confirmation=False, confirmed=False)


def run(root, parent_root, wait_seconds=86400):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The controller must have CUDA hidden')
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError('wait-seconds must be finite and nonnegative')
    root, parent_root = Path(root).resolve(), Path(parent_root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent/('.'+root.name+'.attention-retry.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another retry controller is active') from error
        if root.exists():
            raise FileExistsError('Use a fresh retry study directory; partial artifacts are preserved')
        files = parent.data.cache.prior.source.inherited.local_import_closure([
            Path(__file__), ROOT/'experiments/structure_net/jspace_3b_attention_retry.py',
            ROOT/'tests/structure_net/test_jspace_3b_attention_retry_controller.py'])
        files = set(files) | {ROOT/'experiments/structure_net/jspace_3b_attention_retry_plan.json',
            ROOT/'pixi.lock', ROOT/'pyproject.toml'}
        sources = {str(path):sha(path) for path in sorted(files)}
        plan = dict(schema=SCHEMA, parent_root=str(parent_root), worker_root=str(root),
            source_sha256=sources, arms=list(ARMS), labels=list(LABELS), wait_seconds=wait_seconds,
            lease_wait_seconds=LEASE_WAIT_SECONDS,
            allocation=ALLOCATION, allocator_mib=3328, automatic_extra_training=False,
            partial_producers_retried=False, confirmed=False)
        plan['fingerprint'] = parent.data.digest(plan)
        directory = root/'controller_v1'
        directory.mkdir(parents=True)
        write(directory/'plan.json', plan)
        for path, wanted in sources.items():
            target = directory/'producer'/Path(path).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if sha(target) != wanted:
                raise ValueError('Producer archive differs')
        controller = Controller(root, plan)
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f'Controller signal {signum}')
        handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            controller.event('waiting_parent', parent_root=str(parent_root), gpu_reservation_held=False)
            readiness = wait_parent(parent_root, wait_seconds, controller.verify)
            controller.event('parent_gate', **readiness)
            if readiness['status'] != 'ready':
                result = dict(readiness, confirmed=False)
            else:
                controller.verify()
                retry_worker().prepare(root, parent_root)
                controller.worker_plan = retry_worker().verify(root, live=True)
                result = dict(execute(controller), parent_seed=readiness['selection'])
            result['fingerprint'] = plan['fingerprint']
            write(directory/'result.json', result)
            parent.receipts._manifest(directory, plan, [directory/'plan.json', directory/'result.json'])
            return result
        except BaseException as error:
            write(directory/'failure.json', dict(status='failed', fingerprint=plan['fingerprint'],
                error_type=type(error).__name__, error=str(error), partial_artifacts_preserved=True))
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=86400)
    args = parser.parse_args()
    result = run(args.root, args.parent_root, args.wait_seconds)
    print(json.dumps({key:result[key] for key in ('status', 'fingerprint')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
