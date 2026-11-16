"""CPU-only paired retry admission, immutable scheduling, and reservation tests."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_3b_attention_retry_controller as study


def fixture_controller(tmp_path, monkeypatch):
    (tmp_path/'controller_v1').mkdir()
    controller = study.Controller(tmp_path, dict(source_sha256={}, wait_seconds=86400, lease_wait_seconds=123))
    controller.worker_plan = {'fingerprint':'frozen'}
    monkeypatch.setattr(controller, 'verify', lambda:None)
    return controller


def test_serial_schedule_both_pilots_first_and_quality_never_stops_other_arm(tmp_path, monkeypatch):
    calls = []
    class Fake:
        root = tmp_path
        def phase(self, phase, arm, label=None):
            calls.append((phase, arm, label))
            return dict(evidence_available=True, qualified=False)
        def verify(self):
            pass
    monkeypatch.setattr(study, 'retry_worker', lambda:SimpleNamespace(check=lambda root:{'qualified':False}))
    result = study.execute(Fake())
    assert calls[:2] == [('pilot','softmax',None), ('pilot','quadratic',None)]
    assert calls[2:] == [(phase,arm,label) for arm in study.ARMS
        for phase,label in [('run',None),('generate','update_0000'),('judge','update_0000'),
            ('generate','update_0071'),('judge','update_0071')]]
    assert result['status'] == 'completed'
    assert result['paired_quality']['qualified'] is False
    assert result['population_confirmation'] is False
    assert result['automatic_extra_training'] is False


def test_judge_failure_preserved_and_other_arm_still_evaluated(tmp_path, monkeypatch):
    calls = []
    class Fake:
        root = tmp_path
        def phase(self, phase, arm, label=None):
            calls.append((phase, arm, label))
            if phase == 'judge':
                raise RuntimeError('judge unavailable')
            return {}
        def verify(self):
            pass
    monkeypatch.setattr(study, 'retry_worker', lambda:SimpleNamespace(check=lambda root:{'status':'pending'}))
    result = study.execute(Fake())
    assert result['status'] == 'blocked'
    assert len([row for row in calls if row[0] == 'generate']) == 4
    assert result['all_judgments_available'] is False
    assert all(row['qualified'] is None for arm in result['judgments'].values() for row in arm.values())


def test_gpu_leases_exact_allocation_bounded_wait_and_cpu_judge(tmp_path, monkeypatch):
    controller = fixture_controller(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(study, 'run_process', lambda command, **kwargs:
        calls.append((command, kwargs)) or SimpleNamespace(returncode=0))
    for phase in ('pilot', 'run', 'generate', 'judge'):
        controller.command(phase, 'quadratic', 'update_0071' if phase in ('generate','judge') else None)
    for command, kwargs in calls[:3]:
        assert command[:4] == ['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run']
        assert command[command.index('--allocation')+1] == 'GPU-8cf440df=3584'
        assert command[command.index('--wait')+1] == '123'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert 'inference-ctl' not in ' '.join(calls[3][0])
    assert calls[3][1]['env']['CUDA_VISIBLE_DEVICES'] == ''


def test_failed_command_never_retries_even_prelaunch_conflict(tmp_path, monkeypatch):
    controller = fixture_controller(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(study, 'run_process', lambda *args, **kwargs:
        calls.append(args) or SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match='no rerun'):
        controller.command('pilot', 'softmax')
    assert len(calls) == 1
    assert (controller.directory/'pilot_softmax.log').exists()


def test_partial_phase_never_relaunches(tmp_path, monkeypatch):
    controller = fixture_controller(tmp_path, monkeypatch)
    (tmp_path/'arms/softmax/run').mkdir(parents=True)
    monkeypatch.setattr(controller, 'command', lambda *a, **k:pytest.fail('duplicate producer'))
    with pytest.raises(RuntimeError, match='no duplicate launch'):
        controller.phase('run', 'softmax')


def test_pilot_and_scientific_receipts_require_complete_qualification(tmp_path, monkeypatch):
    controller = fixture_controller(tmp_path, monkeypatch)
    for phase in ('pilot','run'):
        directory = tmp_path/'arms/softmax'/phase
        directory.mkdir(parents=True)
        (directory/'manifest.json').touch()
    monkeypatch.setattr(study.parent.receipts, '_completed', lambda *a:{'passed':True})
    with pytest.raises(ValueError, match='Disposable pilot'):
        controller.phase('pilot','softmax')
    with pytest.raises(ValueError, match='complete matched coverage'):
        controller.phase('run','softmax')


def parent_fixture(tmp_path, monkeypatch):
    root = tmp_path/'stage_a'
    root.mkdir()
    control = tmp_path/'controller_v1'
    control.mkdir()
    (control/'manifest.json').touch()
    study.write(control/'result.json', dict(status='completed', all_judgments_available=True))
    study.write(control/'plan.json', {'fingerprint':'parent-controller'})
    monkeypatch.setattr(study.parent, 'verify', lambda *a, **k:({'fingerprint':'parent'}, {}, {}))
    monkeypatch.setattr(study.parent, 'check', lambda *a:dict(selection_final=True,
        generation_or_judgment_pending=[], task_selection={'qualified':False,'label':'update_0071'}))
    authenticated = []
    def completed(directory, plan):
        directory = Path(directory)
        authenticated.append(directory)
        if directory.name == 'run':
            return {'candidates':[{'label':label} for label in study.PARENT_LABELS]}
        return dict(label=directory.name, evidence_available=True)
    monkeypatch.setattr(study.parent.receipts, '_completed', completed)
    return root, control, authenticated, completed


def test_parent_requires_all_ten_receipts_but_unqualified_seed_is_admitted(tmp_path, monkeypatch):
    root, _, authenticated, _ = parent_fixture(tmp_path, monkeypatch)
    result = study.parent_ready(root)
    assert result['status'] == 'ready'
    assert result['selection']['qualified'] is False
    assert [path.name for path in authenticated if path.parent.name == 'generation'] == list(study.PARENT_LABELS)
    assert [path.name for path in authenticated if path.parent.name == 'judgment'] == list(study.PARENT_LABELS)


@pytest.mark.parametrize('kind', ['missing','unavailable'])
def test_terminal_parent_missing_or_unavailable_evidence_blocks(tmp_path, monkeypatch, kind):
    root, _, _, completed = parent_fixture(tmp_path, monkeypatch)
    def altered(directory, plan):
        if Path(directory) == root/'judgment/update_0278':
            if kind == 'missing':
                raise FileNotFoundError('missing receipt')
            return dict(label='update_0278', evidence_available=False)
        return completed(directory, plan)
    monkeypatch.setattr(study.parent.receipts, '_completed', altered)
    assert study.parent_ready(root)['status'] == 'blocked'


def test_parent_wait_is_cpu_only_and_timeout_is_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'run_process', lambda *a, **k:pytest.fail('GPU work before parent terminal'))
    verified = []
    result = study.wait_parent(tmp_path/'stage_a', 0, lambda:verified.append(True))
    assert verified == [True]
    assert result == {'status':'pending','reason':'parent_wait_expired'}


def test_parent_failure_blocks_without_wait_or_gpu(tmp_path, monkeypatch):
    root, control, _, _ = parent_fixture(tmp_path, monkeypatch)
    study.write(control/'failure.json', {'status':'failed'})
    monkeypatch.setattr(study, 'run_process', lambda *a, **k:pytest.fail('GPU launch'))
    result = study.wait_parent(root, 86400, lambda:None,
        sleeper=lambda seconds:pytest.fail('wait after terminal failure'))
    assert result['status'] == 'blocked'


def test_source_or_archive_mutation_rejected_before_phase(tmp_path, monkeypatch):
    source = tmp_path/'source.py'
    source.write_text('original')
    archive = tmp_path/'study/controller_v1/producer/source.py'
    archive.parent.mkdir(parents=True)
    archive.write_text('original')
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    controller = study.Controller(tmp_path/'study', {'source_sha256':{str(source):study.sha(source)}})
    controller.verify()
    archive.write_text('mutated')
    with pytest.raises(ValueError, match='producer changed'):
        controller.verify()


def test_cuda_visibility_and_existing_study_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError, match='CUDA hidden'):
        study.run(tmp_path/'study', tmp_path/'parent')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    root = tmp_path/'study'
    root.mkdir()
    with pytest.raises(FileExistsError, match='fresh retry'):
        study.run(root, tmp_path/'parent')


def test_parent_result_without_terminal_manifest_cannot_launch_gpu(tmp_path, monkeypatch):
    root, control, _, _ = parent_fixture(tmp_path, monkeypatch)
    (control/'manifest.json').unlink()
    monkeypatch.setattr(study, 'run_process', lambda *a, **k:pytest.fail('premature GPU work'))
    assert study.wait_parent(root, 0, lambda:None)['status'] == 'pending'
