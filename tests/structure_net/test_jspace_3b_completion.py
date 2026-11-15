"""No Torch/CUDA dependency: durable sequencing and reservation boundaries."""
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_3b_completion as runner


def setup(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    campaign=tmp_path/'campaign';campaign.mkdir();runner.write(campaign/'plan.json',{'fingerprint':'campaign'})
    state=tmp_path/'state';final=tmp_path/'final'
    runner.prepare(state,campaign,final)
    return state,campaign,final


def test_launch_plan_is_immutable_and_poll_timeout_bounded(tmp_path,monkeypatch):
    state,campaign,final=setup(tmp_path,monkeypatch)
    plan=runner.verify(state)
    assert plan['poll_seconds']==15 and plan['stale_seconds']==10800
    assert runner.observe(plan,100,now=10899) is None
    outcome=runner.observe(plan,100,now=10900)
    assert outcome['status']=='stalled' and outcome['quality_failure'] is False
    value=runner.read(state/'plan.json');value['final_root']='changed'
    (state/'plan.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='configuration changed'):runner.verify(state)
    with pytest.raises(ValueError):runner.prepare(tmp_path/'another',campaign,final,stale_hours=2.9)


@pytest.mark.parametrize('kind',['quality','execution'])
def test_terminal_campaign_never_launches_heldout(tmp_path,monkeypatch,kind):
    state,campaign,_=setup(tmp_path,monkeypatch)
    if kind=='quality':runner.write(campaign/'outcome.json',{'status':'quality_stop','failed_cut':27})
    else:
        failure=campaign/'stages/block_27/train';failure.mkdir(parents=True)
        runner.write(failure/'failure.json',{'status':'failed','error':'bad'})
    monkeypatch.setattr(runner,'finish',lambda state:pytest.fail('final evaluation launched'))
    result=runner.watch(state)
    assert result['status']==('quality_stop' if kind=='quality' else 'campaign_execution_failure')
    assert runner.read(state/'status.json')==result
    assert int((state/'watcher.lock').read_text())>0
    assert runner.watch(state)==result


def test_lock_prevents_duplicate_watcher(tmp_path,monkeypatch):
    state,_,_=setup(tmp_path,monkeypatch)
    with (state/'watcher.lock').open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):runner.watch(state)
    assert not (state/'status.json').exists()


def test_waits_without_relaunching_progressive_and_then_finishes(tmp_path,monkeypatch):
    state,_,_=setup(tmp_path,monkeypatch);observations=iter([None,{'status':'ready'}]);sleeps=[]
    monkeypatch.setattr(runner,'observe',lambda *a,**k:next(observations))
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:sleeps.append(seconds))
    monkeypatch.setattr(runner,'finish',lambda state:{'status':'completed','confirmed':False})
    assert runner.watch(state)['status']=='completed' and sleeps==[15]


def mocked_children(state,final,monkeypatch,*,failed_phase=None):
    calls=[]
    def execute(command,**kwargs):
        calls.append((command,kwargs))
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        assert kwargs['env']['OMP_NUM_THREADS']=='1'
        if '-c' in command:return SimpleNamespace(returncode=0)
        phase=command[command.index('-m')+2]
        if phase==failed_phase:return SimpleNamespace(returncode=9)
        if phase=='prepare':
            final.mkdir();runner.write(final/'plan.json',{'fingerprint':'final','campaign_root':runner.read(state/'plan.json')['campaign']})
        elif phase=='finalize':runner.write(final/'result.json',{'status':'completed','fingerprint':'final','classification':'measured'})
        else:
            (final/phase).mkdir();runner.write(final/phase/'manifest.json',{'status':'completed','fingerprint':'final'})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',execute)
    return calls


def test_prepare_six_reserved_phases_then_cpu_finalize_and_completed_resume(tmp_path,monkeypatch):
    state,_,final=setup(tmp_path,monkeypatch);calls=mocked_children(state,final,monkeypatch)
    result=runner.finish(state)
    assert result['status']=='completed' and result['classification']=='measured'
    gpu=[c for c,k in calls if 'lease' in c]
    assert [c[c.index('-m')+2] for c in gpu]==list(runner.PHASES)
    for command in gpu:
        assert command[:6]==['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run','--allocation','GPU-0370fdb0=11264']
        assert command[command.index('--minutes')+1]=='120'
        assert command[command.index('--')+1:command.index('-m')]==['pixi','run','--locked','--offline','-e','ncm','python']
        assert command[command.index('-m')+1]==runner.MODULE
    phases=[c[c.index('-m')+2] for c,k in calls if '-m' in c]
    assert phases==['prepare',*runner.PHASES,'finalize']
    calls.clear();assert runner.finish(state)==result
    assert all('-c' in c and 'lease' not in c for c,k in calls)


def test_child_failure_stops_without_retry_or_following_phase(tmp_path,monkeypatch):
    state,_,final=setup(tmp_path,monkeypatch);calls=mocked_children(state,final,monkeypatch,failed_phase='compact_validate')
    with pytest.raises(RuntimeError,match='exited9'):runner.finish(state)
    phases=[c[c.index('-m')+2] for c,k in calls if '-m' in c]
    assert phases==['prepare','compact_export','compact_validate']
    assert not (final/'native_eager_before').exists()


def test_incomplete_phase_is_preserved_not_overwritten(tmp_path,monkeypatch):
    state,_,final=setup(tmp_path,monkeypatch);final.mkdir()
    runner.write(final/'plan.json',{'fingerprint':'final'})
    (final/'compact_export').mkdir();calls=mocked_children(state,final,monkeypatch)
    with pytest.raises(RuntimeError,match='cannot be retried'):runner.finish(state)
    assert all('-c' in c for c,k in calls)
