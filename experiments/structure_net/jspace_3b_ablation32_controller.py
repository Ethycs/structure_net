"""Run the complete paired ABLATION32 recipe through explicit reservations.

Adopt authenticated completed phases, preserve partial scientific phases, and
never cancel subsequent fitting because a measured quality score is poor.
The RTX3060 progressive study is outside this controller's resource inventory.
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

from . import jspace_3b_ablation32_campaign as campaign
from .jspace_3b_full_progressive_controller import invoke

ROOT = campaign.ROOT
PIXI = '/home/rabbit/.pixi/bin/pixi'
MODULE = 'experiments.structure_net.jspace_3b_ablation32_'
SCHEMA = 'jspace.ablation32-controller.v1'
PAIRED = 'GPU-7b1db3cb=7680,GPU-8cf440df=2048'
SINGLE = 'GPU-7b1db3cb=7680'
ARMS = ('learned', 'random')
read, write, sha = campaign.read, campaign.write, campaign.sha


def execute_recipe(controller):
    """Testable ordering: full search, both upfront banks, and both36-stage arms."""
    controller.ensure_pilot()
    controller.ensure_anchors()
    search = controller.ensure_search()
    compiled = controller.ensure_compiled()
    initial = {arm:controller.evaluate(arm, None) for arm in ARMS}
    arms = {}; jobs = {}
    def safe_judge(evaluation):
        try:
            return controller.judge(evaluation)
        except Exception as error:
            return dict(status='unavailable', quality_status='unavailable', qualified=None,
                        error_type=type(error).__name__, error=str(error), confirmed=False)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='ablation32-judge') as pool:
        for arm in ARMS:
            previous = None; stages = []
            for layer in range(36):
                current = controller.distill(arm, layer, previous)
                if current.get('arm') != arm or current.get('layer') != layer or current.get('layers') != layer+1:
                    raise ValueError('Stage receipt does not describe the next same-arm selected prefix')
                previous = current; stages.append(current)
            final = controller.evaluate(arm, previous['result_path'])
            arms[arm] = dict(stages=stages, final_gpu_evaluation=final)
            jobs[arm] = pool.submit(safe_judge, final)
        for arm, job in jobs.items():
            arms[arm]['task_judgment'] = job.result()
    return dict(status='completed', search=search, compiled=compiled, initial_controls=initial,
        arms=arms, sequential_stages_completed=sum(len(row['stages']) for row in arms.values()),
        both_models_all36_distilled=True, quality_failure_stops_progression=False, confirmed=False)


class Controller:
    def __init__(self, root, plan):
        self.root, self.plan = Path(root), plan
        self.campaign_root = Path(plan['campaign_root'])
        self.base = Path(plan['study_root'])
        self.lock = threading.Lock()
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            MKL_NUM_THREADS='2', PIXI_CACHE_DIR='/tmp/structure-net-pixi-cache', UV_CACHE_DIR='/tmp/structure-net-uv-cache')

    def event(self, phase, **values):
        with self.lock:
            campaign._progress(self.root, phase, **values)

    def verify(self):
        for path, wanted in self.plan['source_sha256'].items():
            if sha(path) != wanted or sha(self.root/'producer'/Path(path).relative_to(ROOT)) != wanted:
                raise ValueError('Controller producer changed: ' + path)
        if sha(self.campaign_root/'plan.json') != self.plan['campaign_plan_sha256']:
            raise ValueError('Prepared campaign changed')

    def command(self, label, module, phase, root, *, extra=(), allocation=None, minutes=None, markers=()):
        command = [PIXI, 'run', '--locked', '--offline', '-e', 'ncm', 'python', '-m', MODULE+module,
                   phase, '--root', str(root), *map(str, extra)]
        if allocation is not None:
            if allocation not in (PAIRED, SINGLE) or type(minutes) is not int or minutes < 1:
                raise ValueError('An explicit spare-card resource profile and lease duration are required')
            command = ['/usr/bin/python3', '/usr/local/bin/inference-ctl', 'lease', 'run',
                '--allocation', allocation, '--minutes', str(minutes), '--reason',
                'ABLATION32 '+label+'; progressive RTX3060 left untouched', '--', *command]
        # The lease guard owns GPU expiry. An outer timeout must not kill only
        # the reservation wrapper. invoke owns a process group and forwards
        # interruption gracefully; CPU phases additionally have a bounded limit.
        invoke(command, self.root, label, env=self.env, markers=markers,
               timeout=None if allocation is not None else 21600,
               event=self.event, verify=self.verify)

    def ensure_pilot(self):
        p, _, _ = campaign.verify(self.campaign_root)
        out = self.campaign_root/'pilot'
        if not (out/'manifest.json').exists():
            if out.exists(): raise RuntimeError('Partial or active pilot preserved; no duplicate launch')
            self.command('search_pilot', 'campaign', 'pilot', self.campaign_root,
                allocation=PAIRED, minutes=self.plan['pilot_minutes'], markers=[out])
        result = campaign.pilot_receipt(self.campaign_root, p)
        self.event('search_pilot_authenticated', passed=True, lease_id=result.get('lease_id'))
        return result

    def ensure_anchors(self):
        p, dp, _ = campaign.verify(self.campaign_root)
        data_root = Path(p['data_root']); out = data_root/'anchors'
        if (out/'result.json').exists(): return campaign.anchor_receipt(data_root, dp)
        if out.exists(): raise RuntimeError('Partial or active anchor preparation preserved')
        native = Path(dp['full_data_root'])/'native'
        deadline = time.monotonic()+60*self.plan['native_wait_minutes']
        while not (native/'manifest.json').exists():
            self.verify()
            if (native/'failure.json').exists():
                raise RuntimeError('Original native capture failed; no scientific anchors available')
            if time.monotonic() >= deadline:
                raise TimeoutError('Native-cache admission deadline reached without complete source banks')
            self.event('waiting_native_anchors', native_root=str(native), gpu_reserved_by_this_wait=False)
            time.sleep(30)
        self.command('native_train_anchors', 'data', 'anchors', data_root, markers=[out])
        result = campaign.anchor_receipt(data_root, dp)
        self.event('scientific_anchors_authenticated', **result)
        return result

    def ensure_search(self):
        p, _, _ = campaign.verify(self.campaign_root)
        out = self.campaign_root/'search'
        if not (out/'manifest.json').exists():
            if out.exists() or (self.campaign_root/'search_start.json').exists() or (self.campaign_root/'search_failure.json').exists():
                raise RuntimeError('Partial or failed scientific search preserved; no automatic restart')
            self.command('joint_basis_search', 'campaign', 'search', self.campaign_root,
                allocation=PAIRED, minutes=self.plan['search_minutes'],
                markers=[out, self.campaign_root/'search_start.json', self.campaign_root/'search_failure.json'])
        result = campaign._completed(out, p)
        self.event('joint_basis_search_completed', selected_epoch=result['selection_epoch'])
        return dict(root=str(out), result_sha256=sha(out/'result.json'),
                    selection_epoch=result['selection_epoch'], total_updates=result['total_updates'])

    def ensure_compiled(self):
        p, _, _ = campaign.verify(self.campaign_root)
        out = self.campaign_root/'compiled'
        if not (out/'manifest.json').exists():
            if out.exists(): raise RuntimeError('Partial compiled banks preserved')
            self.command('compile_both_full_models', 'campaign', 'compile', self.campaign_root, markers=[out])
        result = campaign._completed(out, p)
        if not result.get('both_arms_all_layers_compiled'):
            raise ValueError('Both whole compiled banks are required before distillation')
        self.event('both_arms_compiled_before_distillation')
        return dict(root=str(out), result_sha256=sha(out/'result.json'), both_arms_all_layers_compiled=True)

    def distill(self, arm, layer, previous):
        from . import jspace_3b_ablation32_distill as worker
        self.verify()
        if shutil.disk_usage(self.base).free < 64*1024**3:
            raise OSError('Need64GiB scratch headroom for the next sequential stage')
        root = self.base/'distill'/arm/f'layer_{layer:02d}'
        parent = None if previous is None else previous['root']
        label = f'{arm}_layer{layer:02d}'
        if not root.exists():
            extra = ['--campaign-root', self.campaign_root, '--arm', arm, '--layer', layer]
            if parent is not None: extra += ['--parent-root', parent]
            self.command(label+'_prepare', 'distill', 'prepare', root, extra=extra, markers=[root])
        p, _, _ = worker.verify(root)
        if (p['campaign_root'] != str(self.campaign_root) or p['arm'] != arm or p['layer'] != layer
                or p['parent_root'] != parent):
            raise ValueError('Existing stage differs from this ordered selected prefix')
        for phase in ('pilot', 'run'):
            if (root/phase/'manifest.json').exists(): continue
            if (root/phase).exists() or (root/f'{phase}_failure.json').exists():
                raise RuntimeError('Partial scientific stage preserved: '+str(root/phase))
            self.command(label+'_'+phase, 'distill', phase, root,
                extra=['--limit-mib', 7424], allocation=SINGLE,
                minutes=self.plan['pilot_minutes'] if phase == 'pilot' else self.plan['stage_minutes'],
                markers=[root/phase, root/f'{phase}_failure.json'])
        result = worker.check(root)
        if (result.get('status') != 'completed' or result['arm'] != arm or result['layer'] != layer
                or result['layers'] != layer+1 or not result.get('selected_replay_exact')
                or [row['layer'] for row in result['selected_blocks']] != list(range(layer+1))):
            raise ValueError('Completed authentic adjacent selected-block evidence required')
        self.event('sequential_stage_completed', arm=arm, layer=layer, selected_epoch=result['selected']['epoch'])
        return dict(root=str(root), result_path=str(root/'run/result.json'),
            result_sha256=sha(root/'run/result.json'), arm=arm, layer=layer, layers=layer+1,
            selected_epoch=result['selected']['epoch'])

    def evaluate(self, arm, arm_result):
        from . import jspace_3b_ablation32_evaluate as worker
        phase = 'initial' if arm_result is None else 'final'
        root = self.base/'evaluation'/arm/phase
        label = arm+'_'+phase+'_evaluation'
        if not root.exists():
            extra = ['--campaign-root', self.campaign_root, '--compiled-root', self.campaign_root/'compiled'/arm]
            if arm_result is not None: extra += ['--arm-result', arm_result]
            self.command(label+'_prepare', 'evaluate', 'prepare', root, extra=extra, markers=[root])
        p, _, _ = worker.verify(root)
        if (p['campaign_root'] != str(self.campaign_root) or p['arm'] != arm
                or p['mode'] != ('initial_compiled' if arm_result is None else 'final_distilled')):
            raise ValueError('Evaluation belongs to another paired arm or stage')
        if arm_result is not None and p['arm_result'] != read(arm_result):
            raise ValueError('Final evaluation is not bound to the selected36-block model')
        if not (root/'run/manifest.json').exists():
            if (root/'run').exists() or (root/'run_failure.json').exists():
                raise RuntimeError('Partial evaluation preserved')
            self.command(label+'_run', 'evaluate', 'run', root, extra=['--limit-mib', 7424],
                allocation=SINGLE, minutes=self.plan['evaluate_minutes'], markers=[root/'run', root/'run_failure.json'])
        result = campaign._completed(root/'run', p)
        self.event('packed_evaluation_completed', arm=arm, mode=phase, quality_status=result.get('quality_status'))
        return dict(root=str(root), arm=arm, mode=phase, result_sha256=sha(root/'run/result.json'),
                    quality_status=result.get('quality_status'), confirmed=False)

    def judge(self, evaluation):
        from . import jspace_3b_ablation32_evaluate as worker
        root = Path(evaluation['root']); p, _, _ = worker.verify(root)
        if not (root/'judge/manifest.json').exists():
            if (root/'judge').exists():
                raise RuntimeError('Partial judge attempt preserved')
            self.command(evaluation['arm']+'_final_judge', 'evaluate', 'judge', root,
                         markers=[root/'judge'])
        result = campaign._completed(root/'judge', p)
        self.event('task_judgment_completed', arm=evaluation['arm'], quality_status=result.get('quality_status'))
        return dict(root=str(root/'judge'), result_sha256=sha(root/'judge/result.json'),
                    quality_status=result.get('quality_status'), qualified=result.get('qualified'), confirmed=False)


def run(root, campaign_root, *, pilot_minutes=120, search_minutes=4320, stage_minutes=4320,
        evaluate_minutes=240, native_wait_minutes=720):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The orchestration process must have CUDA hidden')
    budgets = dict(pilot_minutes=pilot_minutes, search_minutes=search_minutes,
        stage_minutes=stage_minutes, evaluate_minutes=evaluate_minutes, native_wait_minutes=native_wait_minutes)
    if any(type(value) is not int or value <= 0 for value in budgets.values()):
        raise ValueError('Positive explicit phase admission budgets required')
    root, campaign_root = Path(root).resolve(), Path(campaign_root).resolve()
    if root.exists(): raise FileExistsError('Use a fresh controller attempt directory')
    cp, _, _ = campaign.verify(campaign_root)
    base = campaign_root.parent
    with (base/'.ablation32-controller.lock').open('a') as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another ABLATION32 controller is active') from error
        files = campaign.data.cache.prior.source.inherited.local_import_closure([
            Path(__file__), campaign.data.PROTOCOL,
            ROOT/'tests/structure_net/test_jspace_3b_ablation32_controller.py',
            ROOT/'tests/structure_net/test_jspace_3b_ablation32_distill.py',
            ROOT/'tests/structure_net/test_jspace_3b_ablation32_evaluate.py', *campaign.source_files()])
        sources = {str(path):sha(path) for path in sorted(files)}
        p = dict(schema=SCHEMA, campaign_root=str(campaign_root), campaign_fingerprint=cp['fingerprint'],
            campaign_plan_sha256=sha(campaign_root/'plan.json'), study_root=str(base), source_sha256=sources,
            arms=list(ARMS), layers=list(range(36)), phase_reservations=dict(search=PAIRED, distillation=SINGLE),
            judge_workers=1, quality_failure_stops_progression=False, progressive_3060_untouched=True,
            partial_science_automatically_retried=False, **budgets, confirmed=False)
        p['fingerprint'] = campaign.digest(p); root.mkdir(parents=True); write(root/'plan.json', p)
        for path, wanted in sources.items():
            target = root/'producer'/Path(path).relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
            if sha(target) != wanted: raise ValueError('Controller archive differs')
        controller = Controller(root, p); started = time.monotonic()
        def interrupted(signum, _frame): raise KeyboardInterrupt(f'Controller signal {signum}')
        previous_handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            result = execute_recipe(controller)
            result.update(fingerprint=p['fingerprint'], elapsed_seconds=time.monotonic()-started)
            write(root/'result.json', result)
            campaign._manifest(root, p, [root/'plan.json', root/'progress.jsonl', root/'result.json'])
            return result
        except BaseException as error:
            write(root/'failure.json', dict(status='execution_failed', fingerprint=p['fingerprint'],
                error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started,
                partial_scientific_artifacts_preserved=True, requires_diagnosis=True, confirmed=False))
            raise
        finally:
            for sig, old in previous_handlers.items(): signal.signal(sig, old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--campaign-root', required=True, type=Path)
    for name, default in (('pilot',120), ('search',4320), ('stage',4320), ('evaluate',240), ('native-wait',720)):
        parser.add_argument('--'+name+'-minutes', type=int, default=default)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key:result[key] for key in ('status','sequential_stages_completed','both_models_all36_distilled')}))
