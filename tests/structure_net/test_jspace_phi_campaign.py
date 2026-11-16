from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_phi_campaign as campaign


def plan(tmp_path):
    return dict(fingerprint='campaign',native_root=str(tmp_path/'native'),fit_root=str(tmp_path/'fit'),
        recovery_root=str(tmp_path/'recovery'),data_root=str(tmp_path/'data'),native_fingerprint='native',
        native_launch_root=str(tmp_path/'launch'),native_deadline=100.,scope='one block only')


def api_for(p):
    def admit(root,*args,**kwargs):
        root=Path(root)
        path=root/'plan.json' if not args else root/(args[0]+'_receipt.json') if root==Path(p['fit_root']) else root/args[0]/'manifest.json'
        value=campaign.read(path)
        if value.get('status')!='completed':raise ValueError('incomplete actual worker receipt')
        return value
    return {kind:SimpleNamespace(verify=admit,verify_plan=admit,checked=admit) for kind in ('fit','recovery')}


def publish_step(p,step):
    kind,phase=step.split('_',1);root=Path(p[kind+'_root'])
    worker_plan=dict(status='completed',fingerprint=kind+'_prepare',native_root=p['native_root'],
        data_root=p['data_root'],native_fingerprint=p['native_fingerprint'])
    if kind=='recovery':worker_plan.update(fit_root=p['fit_root'],fit_fingerprint='fit_prepare')
    if not (root/'plan.json').exists():campaign.write(root/'plan.json',worker_plan)
    if phase!='prepare':campaign.write(campaign.artifact_path(p,step),dict(status='completed',fingerprint=worker_plan['fingerprint']))


def finish_native(p):
    native=Path(p['native_root']);launch=Path(p['native_launch_root']);launch.mkdir(parents=True,exist_ok=True)
    campaign.write(native/'native/manifest.json',dict(status='completed',fingerprint='native'))
    (launch/'output.log').write_text('native complete\n')
    campaign.write(launch/'exit.json',dict(status='worker_exited',returncode=0,fingerprint='native',
        native_manifest_sha256=campaign.sha(native/'native/manifest.json'),log_sha256=campaign.sha(launch/'output.log')))


def test_commands_have_separate_normal_leases_and_cpu_preparation(tmp_path):
    p=plan(tmp_path)
    for step in campaign.STEPS:
        command=campaign.phase_command(p,step)
        assert '--locked' in command and '--no-install' in command and '--offline' in command
        if step.endswith('_prepare'):
            assert '--allocation' not in command and command[0]==campaign.PIXI
        else:
            assert command[:4]==['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run']
            assert command.count('--allocation')==1 and command[command.index('--allocation')+1]=='GPU-0370fdb0=11264'
            assert command[command.index('--minutes')+1]=='90'
            assert command[-4:] == ['--root',p[step.split('_')[0]+'_root'],'--phase',step.split('_')[1]]
    assert campaign.phase_command(p,'recovery_prepare')[-2:]==['--fit-root',p['fit_root']]


def test_waits_for_external_native_exit_then_authenticates(tmp_path):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path)
    now=[0.];sleeps=[];checked=[]
    def sleep(seconds):
        sleeps.append(seconds);now[0]+=seconds;finish_native(p)
    api={'native':SimpleNamespace(checked=lambda path:checked.append(path) or {'fingerprint':'native'})}
    value=campaign.wait_native(root,p,api=api,now=lambda:now[0],sleep=sleep)
    assert sleeps==[15] and len(checked)==1 and value['status']=='verified'
    assert campaign.read(root/'native_verified.json')==value


@pytest.mark.parametrize('kind',['worker_failure','exit_failure','log_tamper','timeout','missing_manifest'])
def test_native_failure_stops_without_launch(tmp_path,kind):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path)
    api={'native':SimpleNamespace(checked=lambda _:pytest.fail('Failed native proof reached admission'))}
    if kind=='worker_failure':campaign.write(Path(p['native_root'])/'native/failure.json',{'status':'failed'})
    elif kind=='exit_failure':campaign.write(Path(p['native_launch_root'])/'exit.json',{'status':'worker_exited','returncode':1})
    elif kind in ('log_tamper','missing_manifest'):
        finish_native(p)
        if kind=='log_tamper':(Path(p['native_launch_root'])/'output.log').write_text('tampered')
        else:(Path(p['native_root'])/'native/manifest.json').unlink()
    with pytest.raises((RuntimeError,ValueError,TimeoutError)):
        campaign.wait_native(root,p,api=api,now=lambda:101.,sleep=lambda _:pytest.fail('Should stop immediately'))


def test_successful_child_receipts_resume_without_duplicate_launch(tmp_path,monkeypatch):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path);api=api_for(p);calls=[]
    monkeypatch.setenv('IC_LEASE','must-not-leak')
    def execute(command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and 'IC_LEASE' not in kwargs['env']
        calls.append(command);kwargs['stdout'].write(b'child complete\n')
        publish_step(p,'fit_prepare');return SimpleNamespace(returncode=0)
    proof=campaign.execute_step(root,p,'fit_prepare',api,executor=execute)
    assert len(calls)==1 and campaign.read(root/'commands/fit_prepare/exit.json')['status']=='verified'
    assert campaign.execute_step(root,p,'fit_prepare',api,executor=lambda *a,**k:pytest.fail('duplicate'))==proof
    (root/'commands/fit_prepare/output.log').write_text('changed')
    with pytest.raises(ValueError,match='lineage'):campaign.execute_step(root,p,'fit_prepare',api)


def test_failed_or_interrupted_child_is_never_retried(tmp_path):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path);api=api_for(p);calls=[]
    def failed(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=7)
    with pytest.raises(RuntimeError,match='exited 7'):campaign.execute_step(root,p,'fit_prepare',api,executor=failed)
    with pytest.raises(RuntimeError,match='Prior command failed'):campaign.execute_step(root,p,'fit_prepare',api,executor=failed)
    assert len(calls)==1
    directory=root/'commands/recovery_prepare';directory.mkdir()
    with pytest.raises(RuntimeError,match='no exit receipt'):campaign.execute_step(root,p,'recovery_prepare',api,executor=failed)
    assert len(calls)==1


def test_incomplete_external_phase_stops_before_reservation(tmp_path):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path);api=api_for(p)
    partial=Path(p['fit_root'])/'basis/block_39/compiled_dev.npy';partial.parent.mkdir(parents=True);partial.write_bytes(b'partial')
    with pytest.raises(RuntimeError,match='Incomplete worker'):
        campaign.execute_step(root,p,'fit_interface',api,executor=lambda *a,**k:pytest.fail('duplicate external launch'))


def test_external_completion_is_verified_without_invented_exit(tmp_path):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path);api=api_for(p)
    publish_step(p,'fit_capture')
    value=campaign.execute_step(root,p,'fit_capture',api,executor=lambda *a,**k:pytest.fail('external phase relaunched'))
    assert (root/'fit_capture_external.json').exists() and not (root/'commands').exists()
    assert value['artifact_sha256']==campaign.sha(campaign.artifact_path(p,'fit_capture'))


@pytest.mark.parametrize('field',['native_root','data_root','native_fingerprint','fit_root','fit_fingerprint'])
def test_unrelated_completed_worker_cannot_join_campaign(tmp_path,field):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path);api=api_for(p)
    publish_step(p,'fit_prepare');publish_step(p,'recovery_prepare')
    path=Path(p['recovery_root'])/'plan.json';value=campaign.read(path);value[field]='unrelated'
    path.write_text(__import__('json').dumps(value))
    with pytest.raises(ValueError,match='provenance differs'):
        campaign.execute_step(root,p,'recovery_prepare',api,executor=lambda *a,**k:pytest.fail('should not launch'))


def test_all_nine_steps_and_final_fresh_proof_without_quality_claim(tmp_path,monkeypatch):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path);api=api_for(p);steps=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(campaign,'verify_plan',lambda _:p)
    monkeypatch.setattr(campaign,'apis',lambda:api)
    monkeypatch.setattr(campaign,'wait_native',lambda *a,**k:{'status':'verified'})
    def execute(command,**kwargs):
        step=campaign.STEPS[len(steps)];assert command==campaign.phase_command(p,step)
        steps.append(step);publish_step(p,step);return SimpleNamespace(returncode=0)
    result=campaign.run(root,executor=execute)
    assert steps==list(campaign.STEPS) and result['status']=='completed'
    assert all(result[k]is False for k in ('confirmed','full40','generation_recovery'))
    assert 'recovery_verify' in result['steps'] and not (root/'failure.json').exists()
    assert campaign.run(root,executor=lambda *a,**k:pytest.fail('duplicate completed campaign'))==result


def test_failure_is_durable_and_lock_exclusive(tmp_path,monkeypatch):
    root=tmp_path/'controller';root.mkdir();p=plan(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(campaign,'verify_plan',lambda _:p);monkeypatch.setattr(campaign,'apis',lambda:{})
    def fail(*a,**kwargs):raise RuntimeError('native failed')
    monkeypatch.setattr(campaign,'wait_native',fail)
    with pytest.raises(RuntimeError,match='native failed'):campaign.run(root)
    assert campaign.read(root/'failure.json')['status']=='failed'
    with pytest.raises(RuntimeError,match='preserved'):campaign.run(root)
    with campaign.exclusive(root):
        with pytest.raises(RuntimeError,match='holds the lock'):
            with campaign.exclusive(root):pass


def test_prepare_freezes_native_launch_source_and_archive(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    native=tmp_path/'native';native.mkdir();launch=tmp_path/'launch';launch.mkdir()
    source=Path(campaign.__file__).resolve()
    p=dict(fingerprint='native',data_root=str(tmp_path/'data'),source_sha256={str(source):campaign.sha(source)})
    campaign.write(native/'plan.json',p);campaign.write(native/'items.json',[])
    campaign.write(native/'native_launch_receipt.json',{'path':str(launch),'plan_sha256':campaign.sha(native/'plan.json')})
    command=['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run','--allocation',campaign.ALLOCATION,
             '--minutes','90','--','python','-m',campaign.MODULES['native']]
    campaign.write(launch/'started.json',dict(command=command,plan_sha256=campaign.sha(native/'plan.json'),
        fingerprint='native',root=str(native),automatic_retries=0,started_at=datetime.now(timezone.utc).isoformat()))
    api={'native':SimpleNamespace(verify_plan=lambda *a,**k:p),
         'fit':SimpleNamespace(__file__=source,local_import_closure=lambda _: {source}),
         'recovery':SimpleNamespace(__file__=source)}
    monkeypatch.setattr(campaign,'apis',lambda:api)
    root=tmp_path/'controller';result=campaign.prepare(root,native)
    verified=campaign.verify_plan(root)
    assert verified['automatic_retries']==0 and result['fingerprint']==verified['fingerprint']
    (root/'producer'/source.relative_to(campaign.ROOT)).write_text('tampered')
    with pytest.raises(ValueError,match='source/archive'):campaign.verify_plan(root)


def test_cuda_hidden_before_import_or_prepare(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-test')
    monkeypatch.setattr(campaign,'apis',lambda:pytest.fail('import before CPU admission'))
    with pytest.raises(ValueError,match='CUDA hidden'):campaign.prepare(tmp_path/'controller',tmp_path/'native')
