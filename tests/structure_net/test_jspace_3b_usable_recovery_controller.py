"""Stage A orchestration: no duplicate science, every candidate, reserved GPU work."""
from pathlib import Path
import threading

import pytest

from experiments.structure_net import jspace_3b_usable_recovery_controller as study


def data_plan():
    return dict(training={'epochs':1, 'windows':[[]]*278}, checkpoints=[
        dict(update=i, full_training_pass=i==278) for i in (71,134,199,278)])


def test_labels_are_exactly_initializer_and_all_prospective_full_pass_checkpoints():
    assert study.candidate_labels(data_plan()) == [f'update_{i:04d}' for i in (0,71,134,199,278)]
    p=data_plan();p['checkpoints'][1]['update']=71
    with pytest.raises(ValueError):study.candidate_labels(p)
    p=data_plan();p['training']['epochs']=3
    with pytest.raises(ValueError):study.candidate_labels(p)


def test_every_candidate_runs_despite_quality_failure_and_judge_overlaps_next_generation():
    labels=study.candidate_labels(data_plan());entered=threading.Event();release=threading.Event()
    class Controller:
        plan={'labels':labels}
        def __init__(self):self.generated=[];self.graded=[];self.phases=[];self.active=0;self.maximum=0
        def ensure_phase(self,phase):self.phases.append(phase);return {'status':'completed'}
        def event(self,*args,**kwargs):pass
        def generate(self,label):
            if len(self.generated)==1:
                assert entered.wait(3),'CPU judge did not begin before second generation'
                assert self.active==1;release.set()
            self.generated.append(label);return {'label':label}
        def judge(self,label):
            self.active+=1;self.maximum=max(self.maximum,self.active)
            if label==labels[0]:entered.set();assert release.wait(3)
            self.graded.append(label);self.active-=1
            return {'label':label,'evidence_available':True,'qualified':False}
        def selection(self):
            assert self.generated==labels and self.graded==labels
            return {'qualified':False,'selection_final':True}
    controller=Controller();result=study.execute(controller)
    assert controller.phases==['pilot','run'] and controller.maximum==1
    assert result['all_generation_phases_completed'] and result['all_judgments_available']
    assert result['worker_selection']['qualified'] is False
    assert result['automatic_extra_training'] is False


def test_unavailable_judge_does_not_suppress_remaining_generations_or_claim_available_evidence():
    labels=study.candidate_labels(data_plan())
    class Controller:
        plan={'labels':labels}
        def __init__(self):self.generated=[]
        def ensure_phase(self,phase):return {}
        def event(self,*args,**kwargs):pass
        def generate(self,label):self.generated.append(label);return {'label':label}
        def judge(self,label):raise ConnectionError('Judge unavailable')
        def selection(self):return {'selection_final':False,'generation_or_judgment_pending':labels}
    controller=Controller();result=study.execute(controller)
    assert controller.generated==labels and len(result['judgments'])==5
    assert not result['all_judgments_available']
    assert all(row['qualified'] is None for row in result['judgments'])


def controller_fixture(tmp_path,monkeypatch):
    source=tmp_path/'stage_a';source.mkdir();study.write(source/'plan.json',{'fingerprint':'worker'})
    controller=study.Controller(tmp_path,{'worker_root':str(source),'labels':study.candidate_labels(data_plan())})
    monkeypatch.setattr(controller,'verify',lambda:None)
    monkeypatch.setattr(controller,'event',lambda *args,**kwargs:None)
    return controller


def test_completed_pilot_is_authenticated_and_partial_run_never_relaunched(tmp_path,monkeypatch):
    controller=controller_fixture(tmp_path,monkeypatch)
    out=controller.worker_root/'pilot';out.mkdir();(out/'manifest.json').write_text('{}')
    (out/'result.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(study.worker.receipts,'_completed',lambda directory,p:(calls.append(directory)or
        {'passed':True,'discarded':True,'fresh_initializer':True}))
    monkeypatch.setattr(controller,'command',lambda *args,**kwargs:pytest.fail('Duplicated a completed or partial phase'))
    assert controller.ensure_phase('pilot')['status']=='completed' and calls==[out]
    (controller.worker_root/'run').mkdir()
    with pytest.raises(RuntimeError,match='no duplicate launch'):controller.ensure_phase('run')


def test_run_without_full_coverage_cannot_be_adopted(tmp_path,monkeypatch):
    controller=controller_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(controller,'completed_or_launch',lambda *args,**kwargs:{
        'systems_passed':True,'coverage':{'planned_run_completed':False}})
    with pytest.raises(ValueError,match='complete coverage'):controller.ensure_phase('run')


def test_gpu_commands_have_exact_plain2060_reservation_and_judge_has_no_gpu(tmp_path,monkeypatch):
    controller=controller_fixture(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(study,'invoke',lambda command,*args,**kwargs:calls.append((command,kwargs)))
    for phase in ('pilot','run','generate','judge'):
        controller.command(phase,label='update_0000' if phase in ('generate','judge') else None,marker=tmp_path/phase)
    for (command,kwargs),phase in zip(calls,('pilot','run','generate','judge')):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['markers']==[tmp_path/phase]
        if phase=='judge':assert 'inference-ctl' not in ' '.join(command)
        else:
            assert command[1]=='/usr/local/bin/inference-ctl'
            assert command[command.index('--allocation')+1]=='GPU-8cf440df=3584'
            assert command[command.index('--minutes')+1]==str(study.MINUTES[phase])
            assert kwargs['timeout'] is None
    assert calls[2][0][-2:]==['--profile','plain2060']


def test_generation_cannot_be_relabelled_as_another_checkpoint(tmp_path,monkeypatch):
    controller=controller_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(study.worker,'candidate',lambda *args:{'checkpoint':{'sha256':'expected'}})
    monkeypatch.setattr(controller,'completed_or_launch',lambda *args,**kwargs:{
        'label':'update_0000','checkpoint':{'sha256':'different'}})
    with pytest.raises(ValueError,match='different saved candidate'):controller.generate('update_0000')
