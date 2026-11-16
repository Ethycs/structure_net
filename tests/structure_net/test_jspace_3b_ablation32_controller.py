"""Complete paired-recipe scheduling and reservation boundaries."""
from pathlib import Path

import pytest

from experiments.structure_net import jspace_3b_ablation32_controller as study


class FakeController:
    def __init__(self): self.events=[]
    def ensure_pilot(self): self.events.append('pilot')
    def ensure_anchors(self): self.events.append('anchors')
    def ensure_search(self): self.events.append('search');return {'status':'completed'}
    def ensure_compiled(self): self.events.append('compile_both');return {'both':True}
    def evaluate(self, arm, arm_result):
        self.events.append(('evaluate',arm,arm_result))
        return dict(arm=arm, quality_status='fail', root=arm)
    def distill(self, arm, layer, previous):
        self.events.append(('distill',arm,layer))
        assert previous is None if layer==0 else previous['layer']==layer-1 and previous['arm']==arm
        return dict(arm=arm, layer=layer, layers=layer+1, root=f'{arm}/{layer}',
                    result_path=f'{arm}/{layer}/run/result.json', quality_status='fail')
    def judge(self, result):
        if result['arm']=='learned': raise ConnectionError('judge unavailable')
        return dict(quality_status='fail', qualified=False)


def test_both36_stage_arms_complete_despite_low_quality_and_unavailable_judge():
    controller=FakeController();result=study.execute_recipe(controller)
    assert controller.events[:4]==['pilot','anchors','search','compile_both']
    assert controller.events[4:6]==[('evaluate','learned',None),('evaluate','random',None)]
    expected=[('distill',arm,layer) for arm in study.ARMS for layer in range(36)]
    assert [event for event in controller.events if isinstance(event,tuple) and event[0]=='distill']==expected
    assert result['sequential_stages_completed']==72 and result['both_models_all36_distilled']
    assert result['arms']['learned']['task_judgment']['quality_status']=='unavailable'
    assert result['arms']['random']['task_judgment']['qualified'] is False


def test_wrong_depth_is_execution_failure_not_an_accepted_quality_failure():
    controller=FakeController()
    controller.distill=lambda arm,layer,previous:dict(arm=arm,layer=layer,layers=999)
    with pytest.raises(ValueError,match='next same-arm'):
        study.execute_recipe(controller)


def controller(tmp_path):
    root=tmp_path/'controller';root.mkdir()
    return study.Controller(root,dict(campaign_root=str(tmp_path/'campaign_v2'),study_root=str(tmp_path),
        pilot_minutes=120,search_minutes=4320,stage_minutes=4320,evaluate_minutes=240,
        source_sha256={},campaign_plan_sha256='fixture'))


def test_gpu_commands_use_only_spare_cards_and_preserve_explicit_campaign_v2(tmp_path,monkeypatch):
    instance=controller(tmp_path);calls=[]
    monkeypatch.setattr(study,'invoke',lambda command,*a,**kw:calls.append((command,kw)))
    instance.command('search','campaign','search',instance.campaign_root,allocation=study.PAIRED,minutes=4320)
    command,kwargs=calls[0]
    assert command[command.index('--allocation')+1]==study.PAIRED
    assert 'GPU-0370fdb0' not in ' '.join(command)
    assert command[-1]==str(tmp_path/'campaign_v2')
    assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and kwargs['timeout'] is None
    instance.command('stage','distill','run',tmp_path/'stage',allocation=study.SINGLE,minutes=4320,
                     extra=['--limit-mib',7424])
    assert calls[-1][0][calls[-1][0].index('--allocation')+1]==study.SINGLE
    with pytest.raises(ValueError,match='spare-card'):
        instance.command('bad','distill','run',tmp_path,allocation='GPU-0370fdb0=11008',minutes=60)


def test_partial_pilot_is_never_relaunched(tmp_path,monkeypatch):
    instance=controller(tmp_path);(instance.campaign_root/'pilot').mkdir(parents=True)
    monkeypatch.setattr(study.campaign,'verify',lambda *a:({}, {}, []))
    instance.command=lambda *a,**kw:pytest.fail('Relaunched partial pilot')
    with pytest.raises(RuntimeError,match='preserved'):
        instance.ensure_pilot()


def test_failed_or_active_search_is_never_restarted(tmp_path,monkeypatch):
    instance=controller(tmp_path);instance.campaign_root.mkdir()
    (instance.campaign_root/'search_start.json').write_text('{}')
    monkeypatch.setattr(study.campaign,'verify',lambda *a:({}, {}, []))
    instance.command=lambda *a,**kw:pytest.fail('Restarted scientific search')
    with pytest.raises(RuntimeError,match='preserved'):
        instance.ensure_search()


def test_cpu_phases_do_not_acquire_a_gpu_reservation(tmp_path,monkeypatch):
    instance=controller(tmp_path);calls=[]
    monkeypatch.setattr(study,'invoke',lambda command,*a,**kw:calls.append((command,kw)))
    instance.command('anchors','data','anchors',tmp_path/'data')
    command,kwargs=calls[0]
    assert command[0]==study.PIXI and '--allocation' not in command
    assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
