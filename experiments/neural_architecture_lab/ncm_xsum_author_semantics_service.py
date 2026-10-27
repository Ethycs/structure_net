"""Durable services for the author-rendered semantic follow-up."""
import argparse
import json
from pathlib import Path
import subprocess

from experiments.structure_net import ncm_xsum_author_semantics as study
from experiments.neural_architecture_lab import ncm_xsum_output_distillation_service as lifecycle

data = study.data
ENV = {'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}


def locations(stage):
    if stage not in ('pilot','grade'):raise ValueError('fixed judge stage required')
    return (f'structure-net-ncm-xsum-author-{stage}-v1.service',
        data.ROOT/f'20261027_xsum_author_semantics_supervision_{stage}_v1',study.ROOTS[stage])


def command(stage):
    locations(stage)
    return ['/home/rabbit/.pixi/bin/pixi','run','--locked','-e','ncm','python','-m',
            'experiments.structure_net.ncm_xsum_author_semantics',stage]


def valid_result(stage,result,inherited):
    statuses = {'pilot':('author_semantics_pilot_passed','author_semantics_pilot_failed'),
                'grade':('author_semantics_grade_completed',)}
    return (result.get('status') in statuses[stage]
        and result.get('cases_per_arm') == (4 if stage == 'pilot' else 128)
        and set(result.get('metrics',{})) == set(study.ARMS)
        and result.get('inherited_first_judgments') == inherited
        and result.get('model_training') is False and result.get('final_test_opened') is False)


def launch(stage):
    study.configure();unit,root,dest = locations(stage)
    if root.exists() or dest.exists():raise ValueError('existing evidence forbids duplicate launch')
    previous = study.ROOTS['prepare' if stage == 'pilot' else 'pilot']
    result = study.base.require(previous)
    if stage == 'grade' and result['status'] != 'author_semantics_pilot_passed':
        raise ValueError('valid pilot required')
    study.base.verify_proof(data.read(previous/'producer.json'))
    files = [Path(__file__),Path(lifecycle.__file__),Path(study.__file__),study.PLAN,study.TEST,study.ANALYSIS,
             Path('pixi.lock'),Path('pyproject.toml')]
    files += [previous/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    root.mkdir();log = (root/'worker.log').resolve();argv = command(stage)
    plan = {'stage':stage,'unit':unit,'argv':argv,'workdir':str(lifecycle.REPO),'environment':ENV,
        'restart_policy':'no','logfile':str(log),'inherited_first_judgments':result['inherited_first_judgments'],
        'artifacts':{str(p):data.sha(p) for p in files}}
    data.write(root/'launch_plan.json',plan)
    cmd = ['systemd-run','--user','--unit='+unit,'--service-type=exec','--property=RemainAfterExit=yes',
        '--property=Restart=no','--property=StandardOutput=append:'+str(log),
        '--property=StandardError=append:'+str(log),'--working-directory='+str(lifecycle.REPO),
        *['--setenv='+k+'='+v for k,v in ENV.items()],'--',*argv]
    process = subprocess.run(cmd,check=True,capture_output=True,text=True);state = lifecycle.state_now(unit)
    if state['ActiveState'] != 'active' or state['SubState'] != 'running' or state['MainPID'] == '0':
        data.write(root/'launch_failure.json',{'state':state,'command':cmd,'stderr':process.stderr})
        raise ValueError('live service required')
    (root/'unit_snapshot.service').write_bytes(Path(state['FragmentPath']).read_bytes())
    value = {'status':'judge_service_launched_not_completed','stage':stage,'unit':unit,'state':state,
        'systemd_run_exit_code':process.returncode,'command':cmd,
        'artifacts':{str(p):data.sha(p) for p in (root/'launch_plan.json',root/'unit_snapshot.service')}}
    data.write(root/'launch.json',value);print(json.dumps(value))


def observe(stage):
    study.configure();unit,root,dest = locations(stage)
    launched = data.read(root/'launch.json');plan = data.read(root/'launch_plan.json')
    data.verify(launched['artifacts']);data.verify(plan['artifacts'])
    state = lifecycle.state_now(unit);status = lifecycle.classify(state,launched)
    if data.sha(state['FragmentPath']) != data.sha(root/'unit_snapshot.service'):
        raise ValueError('service definition changed')
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
        if not valid_result(stage,result,plan['inherited_first_judgments']):
            raise ValueError('complete fixed stage result required')
        stage_status = result['status'];data.verify(result['artifacts'])
        study.base.verify_proof(data.read(dest/'producer.json'))
        files += [dest/'result.json',*[Path(p) for p in result['artifacts']]]
    elif (dest/'failure.json').exists():files.append(dest/'failure.json')
    hashes = {str(p):data.sha(p) for p in files}
    if lifecycle.state_now(unit) != state:raise ValueError('terminal state changed during observation')
    value = {'status':status,'stage':stage,'worker_terminal':True,'worker_success':status == 'terminal_success',
        'stage_result_status':stage_status,'state':state,
        'worker_process_exit_code':int(state['ExecMainStatus']) if state['ExecMainCode'] == '1' else None,
        'worker_process_signal':int(state['ExecMainStatus']) if state['ExecMainCode'] in ('2','3') else None,
        'launch_status_is_not_worker_status':True,'artifacts':hashes}
    terminal.mkdir();data.write(terminal/'result.json',value)
    print(json.dumps({k:v for k,v in value.items() if k != 'artifacts'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('launch','observe'))
    parser.add_argument('--stage',choices=('pilot','grade'),required=True)
    args = parser.parse_args();globals()[args.action](args.stage)
