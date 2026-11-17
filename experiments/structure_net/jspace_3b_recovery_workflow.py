"""Bounded recovery workflow: NAL CPU dispatch, externally leased GPU children.

Phase completion is execution evidence, never an accuracy or promotion claim.
No automatic training retries; adoption requires authenticated immutable outputs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.neural_architecture_lab.distillation_recovery import module

ROOT = Path(__file__).resolve().parents[2]
PIXI = '/home/rabbit/.pixi/bin/pixi'
SCHEMA = 'jspace.recovery-reset.workflow.v1'
ADAPTERS = {
    'data': 'jspace_3b_recovery_study_data',
    'context': 'jspace_3b_context_qualification',
    'diagnostics': 'jspace_3b_recovery_diagnostics',
    'runtime': 'jspace_3b_recovery_session',
    'evaluation': 'jspace_3b_recovery_study_eval',
}
QUALIFICATION = ('data_prepare', 'evaluation_prepare', 'diagnostics_prepare', 'context_prepare',
                 'context_run', 'context_pilot', 'runtime_prepare',
                 'runtime_pilot', 'diagnostics_run')


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def cpu_environment():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                PIXI_CACHE_DIR='/tmp/structure-net-pixi-cache',
                UV_CACHE_DIR='/tmp/structure-net-uv-cache')


def prepare(root, spec_path, *, allocation, minutes=360, runtime_root=None, prefix_mode='streamed'):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Workflow preparation requires CUDA hidden')
    root, spec_path = Path(root).resolve(), Path(spec_path).resolve()
    if minutes < 1 or '=' not in allocation:
        raise ValueError('Explicit reservation allocation and positive duration required')
    spec = read(spec_path)
    runtime_root = Path(runtime_root or Path(spec['artifact_root'])/'runtime').resolve()
    if not runtime_root.is_relative_to(Path(spec['artifact_root']).resolve()) or prefix_mode not in ('streamed', 'resident'):
        raise ValueError('Declare an isolated study runtime root and supported prefix residency')
    if sha(spec['initial_checkpoint']) != spec['initial_checkpoint_sha256']:
        raise ValueError('Initializer differs from the approved eight-block checkpoint')
    sources = {Path(path) for path in read(spec['teacher_plan'])['source_sha256'] if Path(path).suffix == '.py'}
    sources.update(ROOT/'experiments/structure_net'/f'{name}.py' for name in ADAPTERS.values())
    sources.update((Path(__file__), ROOT/'experiments/structure_net/jspace_3b_recovery_study_eval.py',
                    ROOT/'experiments/structure_net/jspace_3b_full_progressive_controller.py',
                    ROOT/'experiments/neural_architecture_lab/distillation_recovery.py',
                    ROOT/'pyproject.toml', ROOT/'pixi.lock'))
    sources.update(ROOT/'src/neural_architecture_lab'/f'{name}.py' for name in
                   ('core', 'runners', 'local_scheduler', 'distillation_data',
                    'distillation_training', 'distillation_recovery'))
    hashes = {str(p.resolve()): sha(p) for p in sorted(sources)}
    plan = dict(schema=SCHEMA, spec_path=str(spec_path), spec_sha256=sha(spec_path),
                study_id=spec['study_id'], seed=spec['seed'], artifact_root=spec['artifact_root'],
                allocation=allocation, minutes=minutes, source_sha256=hashes,
                runtime_root=str(runtime_root), runtime_prefix_mode=prefix_mode,
                automatic_training_retries=False, scientific_claim=False)
    plan['fingerprint'] = module('distillation_recovery').digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        raise FileExistsError('Preserve the prepared workflow; use another root for a revised specification')
    for source in sources:
        destination = root/'producer'/source.resolve().relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_new(root/'plan.json', plan)
    return plan


def verify(root):
    root = Path(root)
    plan = read(root/'plan.json')
    body = dict(plan)
    fingerprint = body.pop('fingerprint')
    if plan['schema'] != SCHEMA or module('distillation_recovery').digest(body) != fingerprint:
        raise ValueError('Workflow fingerprint mismatch')
    if sha(plan['spec_path']) != plan['spec_sha256']:
        raise ValueError('Frozen study specification changed')
    for path, expected in plan['source_sha256'].items():
        archived = root/'producer'/Path(path).relative_to(ROOT)
        if sha(path) != expected or sha(archived) != expected:
            raise ValueError('Frozen execution source changed: '+path)
    return plan


def phase_definition(plan, name):
    base = Path(plan['artifact_root'])
    parts = name.split('_')
    if name in QUALIFICATION:
        adapter, action = parts
        directory = Path(plan['runtime_root']) if adapter == 'runtime' else base/adapter
        arguments = [action, '--root', str(directory)]
        if action == 'prepare':
            arguments += ['--spec', plan['spec_path']]
        if adapter in ('runtime', 'diagnostics') and action == 'prepare':
            arguments += ['--limit-mib', '7424']
        if adapter == 'runtime' and action == 'prepare':
            arguments += ['--prefix-mode', plan['runtime_prefix_mode']]
        if adapter == 'context' and action in ('run', 'pilot'):
            arguments += ['--limit-mib', '7168']
        if action == 'prepare':
            output = directory/('manifest.json' if adapter == 'data' else 'plan.json')
        elif adapter == 'context':
            output = directory/('candidate.json' if action == 'run' else 'pilot/result.json')
        else:
            output = directory/('pilot/result.json' if action == 'pilot' else 'run/result.json')
        gpu = action in ('run', 'pilot')
    elif name.startswith(('train:', 'generate:')):
        action, arm, target = name.split(':')
        if action not in ('train', 'generate') or arm not in ('reference', 'student_history'):
            raise ValueError('Use train:ARM:TOKENS or generate:ARM:TOKENS')
        target = int(target)
        spec = read(plan['spec_path'])
        if target not in ([0] if action == 'generate' else []) + spec['milestones']:
            raise ValueError('Undeclared checkpoint target')
        adapter, directory, gpu = 'runtime', Path(plan['runtime_root']), True
        arguments = [action, '--root', str(directory), '--arm', arm, '--target-tokens', str(target)]
        branch = 'phases' if action == 'train' else 'generation'
        output = directory/'arms'/arm/branch/f'tokens_{target}'/'result.json'
    elif name.startswith('grade:'):
        _, arm, target = name.split(':')
        target = int(target)
        spec = read(plan['spec_path'])
        if arm not in ('reference', 'student_history') or target not in [0, *spec['milestones']]:
            raise ValueError('Undeclared grading checkpoint')
        adapter, directory, gpu = 'evaluation', base/'evaluation', False
        label = f"{Path(plan['runtime_root']).name}_{arm}_tokens_{target}"
        responses = Path(plan['runtime_root'])/'arms'/arm/'generation'/f'tokens_{target}'/'responses.jsonl'
        arguments = ['run', '--root', str(directory), '--responses', str(responses),
                     '--split', 'development', '--label', label]
        output = directory/'runs'/label/'result.json'
    else:
        raise ValueError('Undeclared phase: '+name)
    command = [PIXI, 'run', '--locked', '--offline', '-e', 'ncm', 'python', '-m',
               'experiments.structure_net.'+ADAPTERS[adapter], *arguments]
    if gpu:
        command = ['/usr/bin/python3', '/usr/local/bin/inference-ctl', 'lease', 'run',
                   '--allocation', plan['allocation'], '--minutes', str(plan['minutes']),
                   '--wait', '300', '--reason', plan['study_id']+' '+name, '--', *command]
    return dict(name=name, adapter=adapter, directory=str(directory), output=str(output),
                gpu=gpu, command=command)


def check_preconditions(plan, phase):
    spec = read(plan['spec_path'])
    if phase['name'].startswith(('runtime_', 'train:', 'generate:')):
        geometry = read(spec['geometry_manifest'])
        if geometry.get('qualified') is not True or geometry.get('max_length', 0) < spec['target_context']:
            raise ValueError('Required extended geometry is not qualified')
        corpus = read(spec['corpus_manifest'])
        if corpus.get('spec_sha256') != plan['spec_sha256']:
            raise ValueError('Corpus was not prepared against this specification')
    if phase['name'].startswith(('train:', 'generate:')):
        pilot = read(Path(plan['runtime_root'])/'pilot/result.json')
        if pilot.get('passed') is not True or pilot.get('discarded') is not True:
            raise ValueError('Disposable session pilot did not qualify scientific execution')
        prepared = read(Path(plan['runtime_root'])/'plan.json')
        if (prepared['profile']['native_prefix'] != plan['runtime_prefix_mode']
                or pilot.get('plan_fingerprint') != prepared['fingerprint']):
            raise ValueError('Pilot belongs to a different runtime profile')


def authenticate_receipt(path, *, expected_fingerprint=None):
    receipt = read(path)
    if receipt.get('schema') != SCHEMA+'.phase' or receipt.get('status') != 'completed':
        raise ValueError('Phase has no authenticated completion')
    if not receipt.get('output_sha256'):
        raise ValueError('Completion has no authenticated payload')
    if expected_fingerprint is not None and receipt.get('experiment_fingerprint') != expected_fingerprint:
        raise ValueError('Phase lineage differs')
    for output, expected in receipt['output_sha256'].items():
        if sha(output) != expected:
            raise ValueError('Completed phase output changed: '+output)
    return receipt

def check_adapter(phase, log_path):
    from .jspace_3b_full_progressive_controller import run_process
    command = [PIXI, 'run', '--locked', '--offline', '-e', 'ncm', 'python', '-m',
               'experiments.structure_net.'+ADAPTERS[phase['adapter']],
               'check', '--root', phase['directory']]
    with Path(log_path).open('w') as stream:
        completed = run_process(command, cwd=ROOT, env=cpu_environment(), stdout=stream,
                                stderr=subprocess.STDOUT, timeout=None)
    if completed.returncode:
        raise ValueError('Nested phase artifact authentication failed: '+str(log_path))



class PhaseWorker:
    def __init__(self, root, *, adopt=False):
        self.root, self.adopt = Path(root), adopt

    def __call__(self, experiment, device_id):
        from .jspace_3b_full_progressive_controller import run_process
        core = module('core')
        if device_id != -1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('NAL orchestration is CPU-only; child leases own GPU execution')
        plan = verify(self.root)
        phase = experiment.parameters['phase']
        check_preconditions(plan, phase)
        phase_root = self.root/'phases'/phase['name'].replace(':', '_')
        receipt_path = phase_root/'result.json'
        fingerprint = module('local_scheduler').experiment_fingerprint(experiment)
        if receipt_path.exists():
            receipt = authenticate_receipt(receipt_path, expected_fingerprint=fingerprint)
            check_adapter(phase, phase_root/'reuse_check.log')
            result = module('local_scheduler').ExperimentResultLedger(self.root).load(experiment)
            if result is None:
                raise ValueError('Phase receipt exists without matching NAL completion; explicit reconciliation required')
            return result
        if phase_root.exists():
            raise RuntimeError('Partial phase preserved; diagnose before explicit resume: '+str(phase_root))
        output = Path(phase['output'])
        phase_root.mkdir(parents=True)
        write_new(phase_root/'started.json', dict(experiment_fingerprint=fingerprint, phase=phase,
                                                 adoption=self.adopt, started_unix=time.time()))
        started = time.monotonic()
        try:
            if self.adopt:
                if not output.is_file():
                    raise ValueError('Cannot adopt a missing phase output')
                if phase['name'].split('_')[-1] != 'prepare':
                    raise ValueError('Only independently prepared CPU phases can be adopted')
                prepared = read(Path(phase['directory'])/'plan.json')
                if prepared.get('spec_sha256') != plan['spec_sha256']:
                    raise ValueError('Prepared artifact is bound to another specification')
                if phase['adapter'] == 'runtime' and prepared['profile']['native_prefix'] != plan['runtime_prefix_mode']:
                    raise ValueError('Prepared runtime profile differs from selected workflow')
            else:
                with (phase_root/'worker.log').open('x') as stream:
                    completed = run_process(phase['command'], cwd=ROOT, env=cpu_environment(),
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=None)
                if completed.returncode:
                    raise RuntimeError(f"Phase exited {completed.returncode}; log: {phase_root/'worker.log'}")
            verify(self.root)
            if not output.is_file():
                raise ValueError('Worker returned without its required output receipt')
            evidence = read(output)
            if evidence.get('status') in ('failed', 'blocked', 'pending', 'unavailable', 'execution_failed'):
                raise ValueError('Worker output is not a completed phase: '+str(evidence.get('status')))
            check_adapter(phase, phase_root/'check.log')
            elapsed = time.monotonic()-started
            receipt = dict(schema=SCHEMA+'.phase', status='completed', scientific_claim=False,
                           experiment_fingerprint=fingerprint, output_sha256={str(output):sha(output)},
                           command=phase['command'], adopted=self.adopt, wall_seconds=elapsed)
            write_new(receipt_path, receipt)
            return core.ExperimentResult(experiment_id=experiment.id, hypothesis_id=experiment.hypothesis_id,
                metrics={'phase_wall_seconds':elapsed}, primary_metric=elapsed, model_architecture=[],
                model_parameters=0, training_time=elapsed,
                observations=['Execution phase only; no accuracy/promotion claim.', str(receipt_path)])
        except BaseException as error:
            write_new(phase_root/'failure.json', dict(status='execution_failed', error_type=type(error).__name__,
                error=str(error), partial_artifacts_preserved=True, automatic_training_retry=False))
            raise


async def run_phase(root, name, *, adopt=False):
    root = Path(root)
    plan = verify(root)
    phase = phase_definition(plan, name)
    core, runners = module('core'), module('runners')
    parameters = dict(workflow_fingerprint=plan['fingerprint'], phase=phase,
                      primary_metric_type='phase_wall_seconds')
    # Resolve dependencies before dispatch; their content, not only paths, is lineage.
    dependencies = [Path(plan['spec_path'])]
    spec = read(plan['spec_path'])
    if name.startswith(('context_run', 'context_pilot', 'runtime_', 'train:', 'generate:')):
        dependencies.append(Path(spec['corpus_manifest']))
    if name.startswith(('runtime_', 'train:', 'generate:')):
        dependencies.append(Path(spec['geometry_manifest']))
    if name.startswith(('train:', 'generate:')):
        dependencies.extend(Path(plan['runtime_root'])/relative for relative in
                            ('plan.json', 'pilot/result.json'))
    if name.startswith(('generate:', 'grade:')):
        dependencies.append(Path(plan['artifact_root'])/'evaluation/plan.json')
    if name.startswith('grade:'):
        _, arm, target = name.split(':')
        dependencies.append(Path(plan['runtime_root'])/'arms'/arm/'generation'/f'tokens_{target}'/'responses.jsonl')
    parameters['input_sha256'] = {str(path):sha(path) for path in dependencies}
    experiment = core.Experiment(id=plan['study_id']+'.'+name, hypothesis_id=plan['study_id'],
        name=name, parameters=parameters, device_id=-1, seed=plan['seed'])
    config = core.LabConfig(device_ids=[-1], max_parallel_experiments=1, max_experiment_retries=0,
        resume_completed_experiments=False, auto_balance=False, enable_wandb=False,
        results_dir=str(root), verbose=False)
    result = await runners.AsyncExperimentRunner(config, worker=PhaseWorker(root, adopt=adopt)).run_experiment(experiment)
    if result.error:
        raise RuntimeError(result.error)
    return dict(status='completed', phase=name, result=experiment.id, scientific_claim=False)


def check(root):
    plan = verify(root)
    rows = []
    for path in sorted((Path(root)/'phases').glob('*/result.json')):
        receipt = authenticate_receipt(path)
        rows.append(dict(phase=path.parent.name, status=receipt['status']))
    return dict(status='prepared', fingerprint=plan['fingerprint'], phases=rows, scientific_claim=False)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('prepare', 'run-phase', 'check'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--spec')
    parser.add_argument('--allocation', default='GPU-7b1db3cb=7680')
    parser.add_argument('--minutes', type=int, default=360)
    parser.add_argument('--phase')
    parser.add_argument('--adopt', action='store_true')
    parser.add_argument('--runtime-root')
    parser.add_argument('--prefix-mode', choices=('streamed', 'resident'), default='streamed')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        if not args.spec: parser.error('--spec is required for prepare')
        result = prepare(args.root, args.spec, allocation=args.allocation, minutes=args.minutes,
                         runtime_root=args.runtime_root, prefix_mode=args.prefix_mode)
    elif args.action == 'run-phase':
        if not args.phase: parser.error('--phase is required')
        result = asyncio.run(run_phase(args.root, args.phase, adopt=args.adopt))
    else:
        result = check(args.root)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
