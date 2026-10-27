"""Launch one fixed task-supervision arm and separately observe its terminal state."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from experiments.structure_net import ncm_isolated_task_supervision as lineage
from experiments.neural_architecture_lab import ncm_xsum_output_distillation_service as lifecycle

lineage.install()
study = lineage.training
data = study.data
ENV = dict(lifecycle.ENV)
QUALIFICATION = data.ROOT/'20261027_isolated_task_supervision_CPU_qualification_v1.json'


def locations(arm,stage):
    if arm not in study.ARMS or stage not in ('pilot','train'):raise ValueError('fixed paired arm and stage required')
    return (f'structure-net-ncm-task-supervision-{arm}-{stage}-v3.service',
        data.ROOT/f'20261027_task_supervision_supervision_{arm}_{stage}_v3',study.ROOTS[arm,stage])


def command(arm,stage):
    locations(arm,stage)
    return ['/home/rabbit/.pixi/bin/pixi','run','--locked','-e','ncm','python','-m',
        'experiments.structure_net.ncm_isolated_task_supervision','training','--stage',stage,'--arm',arm]


def valid_result(arm,stage,result):
    return (result.get('status') == ('task_supervision_lifecycle_passed' if stage == 'pilot' else 'task_supervision_training_completed')
        and result.get('arm') == arm and result.get('stage') == stage
        and result.get('updates') == (2 if stage == 'pilot' else study.CONFIG.updates)
        and result.get('strict_final_reload_passed') is True and result.get('protected_tensors_unchanged') is True
        and result.get('fresh_optimizer_masters_match_initial_weights') is True
        and (stage != 'pilot' or all(result.get(k) is True for k in
            ('gradient_parity_passed','strict_resumed_update_passed','source_only_outputs_reproduced')))
        and result.get('task_accuracy') is None and result.get('final_test_opened') is False
        and result.get('speed_tested') is False and result.get('new_judge_requests') == 0)


def cpu_only():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():raise ValueError('CPU-only supervisor required')


def launch(arm,stage):
    cpu_only();unit,root,dest = locations(arm,stage)
    if root.exists() or dest.exists() or (study.STORAGE/arm/stage).exists():raise ValueError('existing evidence forbids duplicate launch')
    qualified = data.read(QUALIFICATION)
    if qualified['actual_tool_exit_code'] != 0:raise ValueError('actual successful CPU qualification required')
    data.verify(qualified['artifacts'])
    rows,index,manifest,paths = study.prerequisites(arm,stage)
    for other in study.ARMS:
        for phase in ('pilot','train'):
            other_unit,other_root,_ = locations(other,phase)
            if (other_root/'launch.json').exists() and lifecycle.state_now(other_unit)['MainPID'] != '0':
                raise ValueError('only one task-supervision GPU worker may run')
    files = [Path(__file__),Path(lifecycle.__file__),Path(study.__file__),study.PLAN,study.EXECUTION,
        study.TEST,study.AUDITOR,QUALIFICATION,Path('pixi.lock'),Path('pyproject.toml'),*paths]
    hashes = {str(p):data.sha(p) for p in files}
    root.mkdir();log = (root/'worker.log').resolve();argv = command(arm,stage)
    plan = {'arm':arm,'stage':stage,'unit':unit,'argv':argv,'workdir':str(lifecycle.REPO),
        'environment':ENV,'restart_policy':'no','logfile':str(log),'cases':len(rows),
        'expected_updates':2 if stage == 'pilot' else study.CONFIG.updates,'artifacts':hashes}
    data.write(root/'launch_plan.json',plan)
    cmd = ['systemd-run','--user','--unit='+unit,'--service-type=exec','--property=RemainAfterExit=yes',
        '--property=Restart=no','--property=StandardOutput=append:'+str(log),
        '--property=StandardError=append:'+str(log),'--working-directory='+str(lifecycle.REPO),
        *['--setenv='+k+'='+v for k,v in ENV.items()],'--',*argv]
    process = subprocess.run(cmd,check=True,capture_output=True,text=True);state = lifecycle.state_now(unit)
    if state['ActiveState'] != 'active' or state['SubState'] != 'running' or state['MainPID'] == '0':
        data.write(root/'launch_failure.json',{'state':state,'command':cmd,'stderr':process.stderr})
        raise ValueError('live durable worker required')
    (root/'unit_snapshot.service').write_bytes(Path(state['FragmentPath']).read_bytes())
    value = {'status':'task_supervision_service_launched_not_completed','arm':arm,'stage':stage,'unit':unit,'state':state,
        'systemd_run_exit_code':process.returncode,'command':cmd,
        'artifacts':{str(p):data.sha(p) for p in (root/'launch_plan.json',root/'unit_snapshot.service')}}
    data.write(root/'launch.json',value);print(json.dumps(value),flush=True)


def observe(arm,stage):
    cpu_only();unit,root,dest = locations(arm,stage)
    launched = data.read(root/'launch.json');plan = data.read(root/'launch_plan.json')
    data.verify(launched['artifacts']);data.verify(plan['artifacts'])
    state = lifecycle.state_now(unit);status = lifecycle.classify(state,launched)
    if data.sha(state['FragmentPath']) != data.sha(root/'unit_snapshot.service'):raise ValueError('service definition changed')
    if status == 'running':
        argv = [s.decode() for s in (Path('/proc')/state['MainPID']/'cmdline').read_bytes().split(b'\0') if s]
        if argv != plan['argv']:raise ValueError('live worker command differs')
        print(json.dumps({'status':status,'worker_terminal':False,'state':state}));return
    terminal = root/'terminal'
    if terminal.exists():raise ValueError('terminal already observed; inspect existing record')
    files = [Path(__file__),Path(lifecycle.__file__),study.TEST,root/'launch.json',root/'launch_plan.json',root/'unit_snapshot.service']
    stage_status = None
    if status == 'terminal_success':
        result = data.read(dest/'result.json')
        if not valid_result(arm,stage,result):raise ValueError('complete fixed stage result required')
        stage_status = result['status'];data.verify(result['artifacts']);study.base.verify_proof(data.read(dest/'producer.json'))
        files += [dest/'result.json',*[Path(p) for p in result['artifacts']]]
    elif (dest/'failure.json').exists():files.append(dest/'failure.json')
    hashes = {str(p):data.sha(p) for p in files}
    if lifecycle.state_now(unit) != state:raise ValueError('terminal state changed during observation')
    value = {'status':status,'arm':arm,'stage':stage,'worker_terminal':True,'worker_success':status == 'terminal_success',
        'stage_result_status':stage_status,'state':state,
        'worker_process_exit_code':int(state['ExecMainStatus']) if state['ExecMainCode'] == '1' else None,
        'worker_process_signal':int(state['ExecMainStatus']) if state['ExecMainCode'] in ('2','3') else None,
        'launch_status_is_not_worker_status':True,'artifacts':hashes}
    terminal.mkdir();data.write(terminal/'result.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='artifacts'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('launch','observe'))
    parser.add_argument('--arm',choices=study.ARMS,required=True);parser.add_argument('--stage',choices=('pilot','train'),required=True)
    args = parser.parse_args();globals()[args.action](args.arm,args.stage)
