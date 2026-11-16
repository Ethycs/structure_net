"""Finish one fixed-depth full-corpus recovery pass and all saved task screens.

The existing worker owns training, generation, grading and candidate selection.
This controller only schedules its frozen phases, using the spare plain2060.
Completed phases are authenticated and adopted; partial producers are preserved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from . import jspace_3b_usable_recovery as worker
from .jspace_3b_full_progressive_controller import invoke

ROOT = worker.ROOT
PIXI = '/home/rabbit/.pixi/bin/pixi'
MODULE = 'experiments.structure_net.jspace_3b_usable_recovery'
SCHEMA = 'jspace.3b-usable-recovery-controller.v1'
ALLOCATION = 'GPU-8cf440df=3584'
MINUTES = dict(pilot=90, run=180, generate=180)
read, write, sha = worker.read, worker.write, worker.sha


def candidate_labels(data_plan):
    """The initializer plus every prospective full-corpus checkpoint, exactly once."""
    checkpoints = data_plan['checkpoints']
    updates = [row['update'] for row in checkpoints]
    if (data_plan['training']['epochs'] != 1 or len(updates) != 4
            or any(type(value) is not int or value < 1 for value in updates)
            or updates != sorted(set(updates))
            or updates[-1] != len(data_plan['training']['windows'])
            or checkpoints[-1].get('full_training_pass') is not True):
        raise ValueError('One full pass and its four frozen checkpoint windows are required')
    return [f'update_{update:04d}' for update in (0, *updates)]


def execute(controller):
    """Quality never skips a saved candidate; only one CPU judge runs at a time."""
    pilot = controller.ensure_phase('pilot')
    trained = controller.ensure_phase('run')
    generations, jobs = [], []

    def safe_judge(label):
        try:
            return controller.judge(label)
        except Exception as error:
            controller.event('judgment_unavailable', label=label, error_type=type(error).__name__)
            return dict(label=label, status='unavailable', evidence_available=False,
                qualified=None, error_type=type(error).__name__, error=str(error))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='usable-recovery-judge') as pool:
        for label in controller.plan['labels']:
            generations.append(controller.generate(label))
            jobs.append(pool.submit(safe_judge, label))
        judgments = [job.result() for job in jobs]
    # The worker alone applies its frozen generated-quality selection rule.
    selected = controller.selection()
    return dict(status='completed', pilot=pilot, training=trained, generations=generations,
        judgments=judgments, worker_selection=selected,
        all_generation_phases_completed=len(generations) == len(controller.plan['labels']),
        all_judgments_available=all(row.get('evidence_available') is True for row in judgments),
        quality_failure_stops_evaluation=False, automatic_extra_training=False, confirmed=False)


class Controller:
    def __init__(self, root, plan):
        self.root, self.plan = Path(root), plan
        self.worker_root = Path(plan['worker_root'])
        self.worker_plan = read(self.worker_root/'plan.json')
        self.lock = threading.Lock()
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PIXI_CACHE_DIR='/tmp/structure-net-pixi-cache',
            UV_CACHE_DIR='/tmp/structure-net-uv-cache')

    def event(self, phase, **values):
        with self.lock:
            worker.receipts._progress(self.root, phase, **values)

    def verify(self):
        for path, wanted in self.plan['input_sha256'].items():
            if sha(path) != wanted: raise ValueError('Controller input changed: '+path)
        for path, wanted in self.plan['source_sha256'].items():
            if sha(path) != wanted or sha(self.root/'producer'/Path(path).relative_to(ROOT)) != wanted:
                raise ValueError('Controller or frozen worker producer changed: '+path)

    def command(self, phase, *, label=None, marker):
        command = [PIXI, 'run', '--locked', '--offline', '-e', 'ncm', 'python', '-m', MODULE,
                   phase, '--root', str(self.worker_root)]
        if label is not None: command += ['--label', label]
        if phase == 'generate': command += ['--profile', 'plain2060']
        if phase in MINUTES:
            command = ['/usr/bin/python3', '/usr/local/bin/inference-ctl', 'lease', 'run',
                '--allocation', ALLOCATION, '--minutes', str(MINUTES[phase]), '--reason',
                'Usable full-corpus recovery '+phase+'; other experiments remain running', '--', *command]
        elif phase != 'judge':
            raise ValueError('Only the fixed pilot, run, generation and CPU judge phases are allowed')
        invoke(command, self.root, phase+('_'+label if label else ''), env=self.env,
            markers=[marker], timeout=None if phase in MINUTES else 21600,
            event=self.event, verify=self.verify)

    def completed_or_launch(self, directory, phase, *, label=None):
        directory = Path(directory)
        self.verify()
        if not (directory/'manifest.json').exists():
            if directory.exists():
                raise RuntimeError('Active, failed or partial phase preserved; no duplicate launch: '+str(directory))
            self.command(phase, label=label, marker=directory)
        result = worker.receipts._completed(directory, self.worker_plan)
        self.event('phase_authenticated', worker_phase=phase, label=label, lease_id=result.get('lease_id'))
        return result

    def ensure_phase(self, phase):
        if phase not in ('pilot', 'run'): raise ValueError('Only pilot and scientific run are training phases')
        result = self.completed_or_launch(self.worker_root/phase, phase)
        if phase == 'pilot':
            if not result.get('passed') or not result.get('discarded') or not result.get('fresh_initializer'):
                raise ValueError('Completed disposable pilot must preserve the fresh scientific initializer')
        else:
            if (result.get('systems_passed') is not True
                    or result.get('coverage', {}).get('planned_run_completed') is not True
                    or [row['label'] for row in result['candidates']] != self.plan['labels']):
                raise ValueError('Scientific run lacks complete coverage or its declared saved candidates')
            for label in self.plan['labels']:
                worker.candidate(self.worker_root, self.worker_plan, label)
        return dict(root=str(self.worker_root/phase), result_sha256=sha(self.worker_root/phase/'result.json'),
            status='completed', coverage=result.get('coverage'), qualified=result.get('qualified'))

    def generate(self, label):
        if label not in self.plan['labels']: raise ValueError('Unplanned candidate')
        candidate = worker.candidate(self.worker_root, self.worker_plan, label)
        result = self.completed_or_launch(self.worker_root/'generation'/label, 'generate', label=label)
        if result.get('label') != label or result.get('checkpoint') != candidate['checkpoint']:
            raise ValueError('Generation used a different saved candidate')
        return dict(label=label, checkpoint=result['checkpoint'], status='completed',
            result_sha256=sha(self.worker_root/'generation'/label/'result.json'))

    def judge(self, label):
        # The worker acquires the shared full-progressive judge file lock.
        # Acquiring it here too would deadlock the subprocess.
        result = self.completed_or_launch(self.worker_root/'judgment'/label, 'judge', label=label)
        candidate = worker.candidate(self.worker_root, self.worker_plan, label)
        if result.get('label') != label or result.get('checkpoint', candidate['checkpoint']) != candidate['checkpoint']:
            raise ValueError('Judgment used a different saved candidate')
        return dict(label=label, status='completed', evidence_available=result.get('evidence_available'),
            qualified=result.get('qualified'), result_sha256=sha(self.worker_root/'judgment'/label/'result.json'))

    def selection(self):
        self.verify()
        result = worker.check(self.worker_root)
        return dict(task_selection=result['task_selection'], selection_final=result['selection_final'],
            generation_or_judgment_pending=result['generation_or_judgment_pending'], confirmed=False)


def run(root, worker_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('The controller must have CUDA hidden')
    root, worker_root = Path(root).resolve(), Path(worker_root).resolve()
    if root.exists(): raise FileExistsError('Use a fresh controller attempt directory')
    wp, dp, _ = worker.verify(worker_root)
    profile = worker.PROFILES[wp['profile']]
    if (wp['profile'] != 'plain2060' or profile['allocation'] != 3584 or profile['limit'] != 3328
            or wp.get('automatic_extra_epochs') is not False):
        raise ValueError('Frozen single-pass spare plain2060 profile is required')
    with (worker_root/'.usable-controller.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another Stage A controller is active') from error
        files = worker.data.cache.prior.source.inherited.local_import_closure([
            Path(__file__), ROOT/'tests/structure_net/test_jspace_3b_usable_recovery_controller.py',
            *map(Path, wp['source_sha256']), *map(Path, dp['source_sha256'])])
        sources = {str(path):sha(path) for path in sorted(files)}
        p = dict(schema=SCHEMA, worker_root=str(worker_root), worker_fingerprint=wp['fingerprint'],
            data_fingerprint=dp['fingerprint'], labels=candidate_labels(dp),
            input_sha256={str(path):sha(path) for path in (worker_root/'plan.json', Path(wp['data_root'])/'plan.json')},
            source_sha256=sources, allocation=ALLOCATION, allocator_mib=3328,
            phase_minutes=MINUTES, judge_workers=1, judge_lock=wp['judge_lock'],
            automatic_extra_training=False, quality_failure_stops_evaluation=False,
            partial_producers_retried=False, confirmed=False)
        p['fingerprint'] = worker.data.digest(p)
        root.mkdir(parents=True); write(root/'plan.json', p)
        for path, wanted in sources.items():
            target = root/'producer'/Path(path).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
            if sha(target) != wanted: raise ValueError('Controller archive differs')
        controller = Controller(root, p); started = time.monotonic()
        def interrupted(signum, _frame): raise KeyboardInterrupt(f'Controller signal {signum}')
        handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            result = execute(controller)
            result.update(fingerprint=p['fingerprint'], elapsed_seconds=time.monotonic()-started)
            write(root/'result.json', result)
            worker.receipts._manifest(root, p, [root/'plan.json', root/'progress.jsonl', root/'result.json'])
            return result
        except BaseException as error:
            write(root/'failure.json', dict(status='failed', fingerprint=p['fingerprint'],
                error_type=type(error).__name__, error=str(error), partial_artifacts_preserved=True))
            raise
        finally:
            for sig, handler in handlers.items(): signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker-root', type=Path, required=True)
    args = parser.parse_args(); result = run(args.root, args.worker_root)
    print(json.dumps({key:result[key] for key in ('status', 'fingerprint', 'all_generation_phases_completed',
        'all_judgments_available')}, allow_nan=False), flush=True)


if __name__ == '__main__': main()
