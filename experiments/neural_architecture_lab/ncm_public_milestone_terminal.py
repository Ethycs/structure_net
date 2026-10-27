"""Observe later public milestones without changing frozen epoch-one evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.neural_architecture_lab import ncm_public_service_terminal as first

study, data = first.study, first.data
TEST = Path('tests/structure_net/test_ncm_public_milestone_terminal.py')
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-curriculum-plan.md')
REPO = Path(__file__).resolve().parents[2]
ENVIRONMENT = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '1',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}


def locations(epoch, attempt):
    if epoch not in study.CONFIG.milestones[1:] or type(attempt) is not int or attempt < 1:
        raise ValueError('later fixed milestone and positive attempt required')
    return (f'structure-net-ncm-public-epoch{epoch}-attempt{attempt}.service',
            study.root_for(epoch, attempt),
            study.STORAGE.parent/f'supervision/epoch{epoch}_attempt{attempt}',
            data.ROOT/f'20261026_public_epoch{epoch}_attempt{attempt}_service_terminal_v1')


def validate_launch(epoch, attempt, unit, plan, launch):
    target = epoch * study.CONFIG.updates_per_epoch
    previous = study.CONFIG.milestones[study.CONFIG.milestones.index(epoch)-1]
    start = plan['expected_first_completed_update']
    resume = Path(plan['resume'])
    argv = ['/home/rabbit/.pixi/bin/pixi', 'run', '--locked', '-e', 'ncm', 'python',
            '-m', 'experiments.structure_net.ncm_public_training', '--end-epoch', str(epoch),
            '--attempt', str(attempt), '--resume', str(resume)]
    expected = {'epoch': epoch, 'attempt': attempt, 'unit': unit, 'workdir': str(REPO),
                'expected_last_completed_update': target, 'restart_policy': 'no',
                'physical_gpu': 1, 'environment': ENVIRONMENT, 'argv': argv}
    if (any(plan.get(k) != v for k, v in expected.items())
            or plan.get('same_fixed_eight_epoch_schedule') is not True
            or type(start) is not int
            or not previous * study.CONFIG.updates_per_epoch <= start < target
            or not resume.resolve().is_relative_to(study.STORAGE.resolve())):
        raise ValueError('launch plan must retain the fixed milestone, checkpoint and environment')
    expected_launch = {'unit': unit, 'expected_epoch': epoch, 'expected_attempt': attempt,
                       'expected_first_completed_update': start,
                       'expected_last_completed_update': target,
                       'expected_training_identity': plan['expected_training_identity']}
    if any(launch.get(k) != v for k, v in expected_launch.items()):
        raise ValueError('launch record and planned milestone differ')


def validate_result(value, epoch, identity):
    expected = {'status': 'public_curriculum_milestone_training_completed', 'epoch': epoch,
                'updates': epoch * study.CONFIG.updates_per_epoch, 'training_identity': identity,
                'task_accuracy': None}
    if (any(value.get(k) != v for k, v in expected.items())
            or any(value.get(k) is not True for k in ('strict_reload_passed', 'oracle_unchanged'))
            or any(value.get(k) is not False for k in ('final_test_opened', 'speed_tested'))):
        raise ValueError('successful exit requires the complete fixed milestone result')


def state_now(unit):
    raw = subprocess.run(['systemctl', '--user', 'show', unit,
                          '--property='+','.join(first.PROPERTIES)],
                         check=True, capture_output=True, text=True).stdout
    return dict(line.split('=', 1) for line in raw.splitlines())


def live_command(pid):
    return [part.decode() for part in (Path('/proc')/pid/'cmdline').read_bytes().split(b'\0') if part]


def observe(epoch, attempt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only terminal observation required')
    unit, training_root, supervision, destination = locations(epoch, attempt)
    if destination.exists():
        raise ValueError('terminal observation already exists; inspect and retain it')
    launch = data.read(supervision/'launch.json')
    plan = data.read(supervision/'launch_plan.json')
    for name in ('launch_plan.json', 'unit_snapshot.service'):
        path = supervision/name
        if launch['artifacts'].get(str(path)) != data.sha(path):
            raise ValueError('launch must bind the exact plan and service snapshot')
    data.verify(launch['artifacts']); data.verify(plan['artifacts'])
    validate_launch(epoch, attempt, unit, plan, launch)
    state = state_now(unit)
    phase = first.classify(state, launch)
    if data.sha(state['FragmentPath']) != data.sha(supervision/'unit_snapshot.service'):
        raise ValueError('service definition changed after launch')
    if phase == 'running':
        if (state['MainPID'] != launch['state']['MainPID']
                or live_command(state['MainPID']) != plan['argv']):
            raise ValueError('the live process must match the recorded trainer command')
        result = {'status': phase, 'epoch': epoch, 'attempt': attempt,
                  'training_terminal': False, 'training_success': False,
                  'service_state': state, 'terminal_artifacts_written': False, 'task_accuracy': None}
        print(json.dumps(result), flush=True)
        return result
    training, environment = study.pipeline.broad.runtime.configure()
    paths = [Path(__file__), TEST, PLAN, Path(first.__file__), first.PLAN,
             *[supervision/n for n in ('launch.json', 'launch_plan.json', 'unit_snapshot.service')],
             *[training_root/n for n in ('producer.json', 'segment.json', 'trace.jsonl', 'resume.json')]]
    proof = data.read(training_root/'producer.json')
    training.native.vendor.verify(proof)
    if proof['fingerprint'] != plan['expected_training_identity']:
        raise ValueError('service ran a different training producer')
    segment, resume = [data.read(training_root/n) for n in ('segment.json', 'resume.json')]
    if (segment['first_completed_update'] != plan['expected_first_completed_update']
            or segment['target_completed_update'] != plan['expected_last_completed_update']
            or resume['completed_updates'] != plan['expected_first_completed_update']
            or resume['path'] != plan['resume'] or resume['sha256'] != plan['resume_sha256']):
        raise ValueError('recorded continuation differs from the launch plan')
    if phase == 'terminal_success':
        value = data.read(training_root/'result.json')
        validate_result(value, epoch, plan['expected_training_identity'])
        data.verify(value['artifacts']); paths.append(training_root/'result.json')
    elif (training_root/'failure.json').exists():
        paths.append(training_root/'failure.json')
    hashes = {str(p): data.sha(p) for p in paths}
    if state_now(unit) != state:
        raise ValueError('service state changed during terminal verification')
    data.verify(hashes)
    destination.mkdir()
    data.write(destination/'producer.json', {'schema': 'nal.ncm-milestone-terminal-observer.v1',
        'epoch': epoch, 'attempt': attempt, 'unit': unit, 'environment': environment,
        'input_hashes': hashes})
    result = {'status': phase, 'epoch': epoch, 'attempt': attempt, 'training_terminal': True,
        'training_success': phase == 'terminal_success', 'service_state': state,
        'ordinary_process_exit': state['ExecMainCode'] == '1',
        'training_process_exit_code': int(state['ExecMainStatus']) if state['ExecMainCode'] == '1' else None,
        'training_process_signal': int(state['ExecMainStatus']) if state['ExecMainCode'] in ('2', '3') else None,
        'launch_tool_exit_code': launch['launch_tool_exit_code'],
        'launch_tool_status_is_not_training_status': True, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(destination/'producer.json'): data.sha(destination/'producer.json')}}
    data.write(destination/'result.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('observe',))
    parser.add_argument('--epoch', type=int, choices=study.CONFIG.milestones[1:], required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    observe(args.epoch, args.attempt)
