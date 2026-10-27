"""Durable launch and explicit terminal observation for the two bridge arms."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.structure_net import ncm_xsum_output_distillation as study

data = study.data
REPO = Path(__file__).resolve().parents[2]
TEST = REPO/'tests/neural_architecture_lab/test_ncm_xsum_output_distillation_service.py'
ENV = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0',
       'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
PROPERTIES = ('MainPID', 'ExecMainPID', 'ActiveState', 'SubState', 'InvocationID',
              'NRestarts', 'ExecMainCode', 'ExecMainStatus', 'Result', 'FragmentPath')


def locations(arm):
    if arm not in ('mse', 'distill'): raise ValueError('fixed comparison arm required')
    return (f'structure-net-ncm-xsum-output-{arm}-v1.service',
            data.ROOT/f'20261027_xsum_output_distill_supervision_{arm}_v1', study.ROOTS['train_'+arm])


def state_now(unit):
    raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
                                  '--property='+','.join(PROPERTIES)], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines())


def classify(state, launch):
    old = launch['state']
    if (state['InvocationID'] != old['InvocationID'] or not state['InvocationID']
            or state['NRestarts'] != '0' or state['ExecMainPID'] != old['ExecMainPID']):
        raise ValueError('different invocation, process or restart count')
    if state['ActiveState'] == 'active' and state['SubState'] == 'running':
        if state['MainPID'] != old['MainPID'] or state['MainPID'] == '0': raise ValueError('live process differs')
        return 'running'
    if state['MainPID'] != '0': raise ValueError('ambiguous nonterminal process state')
    if state['ExecMainCode'] not in ('1', '2', '3'): raise ValueError('no observed terminal process status')
    success = (state['ExecMainCode'] == '1' and state['ExecMainStatus'] == '0' and state['Result'] == 'success'
               and state['ActiveState'] == 'active' and state['SubState'] == 'exited')
    return 'terminal_success' if success else 'terminal_failure'


def launch(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only supervisor required')
    unit, root, training_root = locations(arm)
    if root.exists() or training_root.exists(): raise ValueError('existing evidence forbids duplicate launch')
    cache = study.base.require(study.ROOTS['cache']); study.base.require(study.ROOTS['pilot'])
    if cache['status'] != 'teacher_cache_passed': raise ValueError('teacher cache validity failed')
    if arm == 'distill': study.base.require(study.ROOTS['train_mse'])
    for stage in ('pilot', 'cache'): study.base.verify_proof(data.read(study.ROOTS[stage]/'producer.json'))
    argv = ['/home/rabbit/.pixi/bin/pixi', 'run', '--locked', '-e', 'ncm', 'python', '-m',
            'experiments.structure_net.ncm_xsum_output_distillation', 'train_'+arm]
    inputs = [Path(__file__), TEST, Path(study.__file__), study.PLAN, study.TEST, study.ANALYSIS,
              Path('pixi.lock'), Path('pyproject.toml')]
    inputs += [study.ROOTS[s]/n for s in ('pilot', 'cache') for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    root.mkdir(); logfile = (root/'training.log').resolve()
    plan = {'arm': arm, 'unit': unit, 'argv': argv, 'workdir': str(REPO), 'environment': ENV,
            'expected_updates': 5120, 'restart_policy': 'no', 'logfile': str(logfile),
            'artifacts': {str(p): data.sha(p) for p in inputs}}
    data.write(root/'launch_plan.json', plan)
    command = ['systemd-run', '--user', '--unit='+unit, '--service-type=exec',
        '--property=RemainAfterExit=yes', '--property=Restart=no',
        '--property=StandardOutput=append:'+str(logfile), '--property=StandardError=append:'+str(logfile),
        '--working-directory='+str(REPO), *['--setenv='+k+'='+v for k, v in ENV.items()], '--', *argv]
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    state = state_now(unit)
    if state['ActiveState'] != 'active' or state['SubState'] != 'running' or state['MainPID'] == '0':
        data.write(root/'launch_failure.json', {'state': state, 'command': command, 'stdout': proc.stdout, 'stderr': proc.stderr})
        raise ValueError('expected live durable service after launch')
    (root/'unit_snapshot.service').write_bytes(Path(state['FragmentPath']).read_bytes())
    value = {'status': 'durable_service_launched_not_training_completed', 'arm': arm, 'unit': unit,
        'state': state, 'systemd_run_exit_code': proc.returncode, 'command': command,
        'artifacts': {str(p): data.sha(p) for p in (root/'launch_plan.json', root/'unit_snapshot.service')}}
    data.write(root/'launch.json', value); print(json.dumps(value))


def observe(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only observation required')
    unit, root, training_root = locations(arm)
    launch_record = data.read(root/'launch.json'); plan = data.read(root/'launch_plan.json')
    data.verify(launch_record['artifacts']); data.verify(plan['artifacts'])
    state = state_now(unit); status = classify(state, launch_record)
    if data.sha(state['FragmentPath']) != data.sha(root/'unit_snapshot.service'): raise ValueError('service definition changed')
    if status == 'running':
        argv = [x.decode() for x in (Path('/proc')/state['MainPID']/'cmdline').read_bytes().split(b'\0') if x]
        if argv != plan['argv']: raise ValueError('live command differs')
        value = {'status': status, 'training_terminal': False, 'state': state}
        print(json.dumps(value)); return
    destination = root/'terminal'
    if destination.exists(): raise ValueError('terminal observation already recorded; inspect it')
    files = [Path(__file__), TEST, root/'launch.json', root/'launch_plan.json', root/'unit_snapshot.service']
    if status == 'terminal_success':
        result = data.read(training_root/'result.json')
        if (result['status'] != 'output_distillation_training_completed' or result['arm'] != arm
                or result['updates'] != 5120 or not result['strict_reload']): raise ValueError('incomplete training result')
        data.verify(result['artifacts']); study.base.verify_proof(data.read(training_root/'producer.json'))
        files += [training_root/'result.json', *[Path(p) for p in result['artifacts']]]
    elif (training_root/'failure.json').exists(): files.append(training_root/'failure.json')
    hashes = {str(p): data.sha(p) for p in files}
    if state_now(unit) != state: raise ValueError('terminal state changed during verification')
    value = {'status': status, 'arm': arm, 'training_terminal': True,
        'training_success': status == 'terminal_success', 'state': state,
        'training_process_exit_code': int(state['ExecMainStatus']) if state['ExecMainCode'] == '1' else None,
        'training_process_signal': int(state['ExecMainStatus']) if state['ExecMainCode'] in ('2', '3') else None,
        'launch_status_is_not_training_status': True, 'artifacts': hashes}
    destination.mkdir(); data.write(destination/'result.json', value)
    print(json.dumps({k:v for k,v in value.items() if k != 'artifacts'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('launch', 'observe'))
    parser.add_argument('--arm', choices=('mse', 'distill'), required=True)
    args = parser.parse_args(); globals()[args.action](args.arm)
