"""Observe the detached public trainer; a successful query is not training success."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.structure_net import ncm_public_training as study

data = study.data
UNIT = 'structure-net-ncm-public-epoch1-attempt2.service'
SUPERVISION = study.STORAGE.parent/'supervision/epoch1_attempt2'
TRAINING = study.root_for(1, 2)
ROOT = data.ROOT/'20261026_public_epoch1_attempt2_service_terminal_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-durable-supervision.md')
TEST = Path('tests/structure_net/test_ncm_public_service_terminal.py')
PROPERTIES = ('ActiveState', 'SubState', 'Result', 'MainPID', 'ExecMainPID',
              'ExecMainCode', 'ExecMainStatus', 'NRestarts', 'InvocationID', 'FragmentPath')


def classify(state, launch):
    original = launch['state']
    if (state['InvocationID'] != original['InvocationID']
            or state['ExecMainPID'] != original['ExecMainPID']
            or state['NRestarts'] != '0'):
        raise ValueError('same service invocation and no automatic restarts required')
    if state['ActiveState'] == 'active' and state['SubState'] == 'running':
        if int(state['MainPID']) <= 0 or state['ExecMainCode'] != '0':
            raise ValueError('live service process required')
        return 'running'
    if (state['ActiveState'] == 'active' and state['SubState'] == 'exited'
            and state['MainPID'] == '0' and state['Result'] == 'success'
            and state['ExecMainCode'] == '1' and state['ExecMainStatus'] == '0'):
        return 'terminal_success'
    if state['ActiveState'] == 'failed' and state['SubState'] == 'failed':
        return 'terminal_failure'
    raise ValueError('unexpected service state; inspect rather than inferring completion')


def state_now():
    text = subprocess.run(['systemctl', '--user', 'show', UNIT,
                           '--property='+','.join(PROPERTIES)],
                          capture_output=True, text=True, check=True).stdout
    return dict(line.split('=', 1) for line in text.splitlines())


def observe():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only terminal observation required')
    if ROOT.exists():
        raise ValueError('terminal observation already exists; inspect and retain it')
    launch = data.read(SUPERVISION/'launch.json')
    data.verify(launch['artifacts'])
    state = state_now()
    phase = classify(state, launch)
    if data.sha(state['FragmentPath']) != data.sha(SUPERVISION/'unit_snapshot.service'):
        raise ValueError('service definition changed after its recorded launch')
    if phase == 'running':
        command = Path('/proc')/state['MainPID']/'cmdline'
        if (not command.exists()
                or b'experiments.structure_net.ncm_public_training' not in command.read_bytes().split(b'\0')):
            raise ValueError('recorded service process is not the live trainer launcher')
        print(json.dumps({'status': phase, 'training_terminal': False, 'training_success': False,
            'service_state': state, 'terminal_artifacts_written': False, 'task_accuracy': None}), flush=True)
        return
    training, environment = study.pipeline.broad.runtime.configure()
    paths = [Path(__file__), TEST, PLAN, SUPERVISION/'launch.json', SUPERVISION/'unit_snapshot.service',
             *[TRAINING/n for n in ('producer.json', 'segment.json', 'trace.jsonl', 'resume.json')]]
    proof = data.read(TRAINING/'producer.json')
    training.native.vendor.verify(proof)
    if proof['fingerprint'] != launch['expected_training_identity']:
        raise ValueError('service ran a different training producer')
    if phase == 'terminal_success':
        value = data.read(TRAINING/'result.json')
        expected = {'status': 'public_curriculum_milestone_training_completed', 'epoch': 1,
            'updates': 1692, 'training_identity': launch['expected_training_identity'],
            'strict_reload_passed': True, 'oracle_unchanged': True,
            'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False}
        if any(value.get(k) != v for k, v in expected.items()):
            raise ValueError('ordinary successful exit requires the complete fixed training result')
        data.verify(value['artifacts'])
        paths.append(TRAINING/'result.json')
    elif (TRAINING/'failure.json').exists():
        paths.append(TRAINING/'failure.json')
    hashes = {str(p): data.sha(p) for p in paths}
    if state_now() != state:
        raise ValueError('service state changed during terminal verification')
    data.verify(hashes)
    ROOT.mkdir()
    data.write(ROOT/'producer.json', {'schema': 'nal.ncm-service-terminal-observer.v1',
        'environment': environment, 'input_hashes': hashes, 'unit': UNIT})
    result = {'status': phase, 'training_terminal': True,
        'training_success': phase == 'terminal_success', 'service_state': state,
        'ordinary_process_exit': state['ExecMainCode'] == '1',
        'training_process_exit_code': int(state['ExecMainStatus']) if state['ExecMainCode'] == '1' else None,
        'training_process_signal': int(state['ExecMainStatus']) if state['ExecMainCode'] in ('2', '3') else None,
        'launch_tool_exit_code': launch['launch_tool_exit_code'],
        'launch_tool_status_is_not_training_status': True,
        'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(ROOT/'producer.json'): data.sha(ROOT/'producer.json')}}
    data.write(ROOT/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('observe',))
    parser.parse_args()
    observe()
