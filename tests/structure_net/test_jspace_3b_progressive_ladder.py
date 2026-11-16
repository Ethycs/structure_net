from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_3b_progressive_ladder as ladder


def parent_fixture(tmp_path):
    parent = tmp_path/'seed'
    parent.mkdir()
    checkpoint = parent/'selected.pt'; checkpoint.write_bytes(b'unchanged model fixture')
    metrics = dict(count=256, correct=189, accuracy=189/256)
    verification = dict(passed=True, development_passed=True, promoted=False, confirmed=False,
        compressed_blocks=15, full36=False, review_checkpoint=str(checkpoint),
        review_checkpoint_sha256=ladder.sha(checkpoint))
    trained = dict(candidate=dict(metrics=metrics))
    ladder.write(parent/'plan.json', dict(cut=21))
    for name, value in [('verify/result.json', verification), ('train/result.json', trained),
                         ('verify/manifest.json', {}), ('train/manifest.json', {})]:
        ladder.write(parent/name, value)
    return parent


class FakeStage:
    __file__ = ladder.__file__
    source = SimpleNamespace(inherited=SimpleNamespace(
        local_import_closure=lambda paths: {Path(ladder.__file__)}))

    def admit_parent(self, root):
        return (ladder.read(Path(root)/'plan.json'), ladder.read(Path(root)/'verify/result.json'),
                ladder.read(Path(root)/'train/result.json'))

    def prepare(self, root, parent):
        old = ladder.read(Path(parent)/'plan.json')
        ladder.write(Path(root)/'plan.json', dict(cut=old['cut']-1, parent_root=str(parent)))

    def verify(self, root):
        return ladder.read(Path(root)/'plan.json')

    def checked(self, root, phase, plan):
        assert (Path(root)/phase/'manifest.json').exists()
        return ladder.read(Path(root)/phase/'result.json')


def prepared(tmp_path, monkeypatch, target=17):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    api = FakeStage()
    monkeypatch.setattr(ladder, 'stage_api', lambda: api)
    monkeypatch.setattr(ladder, 'capacity', lambda *args, **kwargs: {'free_bytes': 2000*ladder.GIB})
    parent = parent_fixture(tmp_path)
    root = tmp_path/'campaign'
    ladder.prepare(root, parent, target_blocks=target)
    return root, parent


def executor_fixture(fail_depth=None):
    calls = []
    def execute(root, directory, phase):
        blocks = 36-ladder.read(directory/'plan.json')['cut']
        calls.append((blocks, phase))
        passed = blocks != fail_depth
        parent = Path(ladder.read(directory/'plan.json')['parent_root'])
        previous = ladder.read(parent/'verify/result.json')
        checkpoint = directory/'selected.pt'
        if not checkpoint.exists():
            checkpoint.write_bytes(f'compressed model{blocks}'.encode())
        value = dict(status='completed')
        if phase == 'train':
            value['candidate'] = dict(metrics=dict(count=256, correct=189 if passed else 170))
        if phase == 'verify':
            value.update(passed=True, development_passed=passed, promoted=False, confirmed=False,
                compressed_blocks=blocks if passed else blocks-1, full36=blocks==36 and passed,
                review_checkpoint=str(checkpoint) if passed else previous['review_checkpoint'],
                review_checkpoint_sha256=ladder.sha(checkpoint) if passed else previous['review_checkpoint_sha256'])
        ladder.write(directory/phase/'result.json', value)
        ladder.write(directory/phase/'manifest.json', {'result.json': ladder.sha(directory/phase/'result.json')})
    return calls, execute


def test_commands_reserve_only_intended_devices_before_starting_python():
    for phase in ladder.PHASES:
        command = ladder.phase_command('/tmp/blocks_16', phase)
        allocation = command[command.index('--allocation')+1]
        assert allocation == (ladder.DUAL if phase in ('pilot','train') else ladder.SINGLE)
        assert 'GPU-8cf440df' not in str(command)
        assert command.index('lease') < command.index(ladder.PIXI)
        assert command[command.index('--minutes')+1] == '90'
        assert '--locked' in command and '--offline' in command
    env = ladder.cpu_environment()
    assert env['CUDA_VISIBLE_DEVICES'] == '' and 'IC_LEASE' not in env


def test_disk_forecast_protects_remaining_stages_and_headroom(tmp_path):
    required = (21*40+64)*ladder.GIB
    assert ladder.capacity(tmp_path, 21, disk_usage=lambda p: SimpleNamespace(free=required))['forecast_bytes'] == required
    with pytest.raises(RuntimeError, match='Insufficient storage'):
        ladder.capacity(tmp_path, 21, disk_usage=lambda p: SimpleNamespace(free=required-1))


def test_complete_progression_preserves_lineage_and_terminal_resume(tmp_path, monkeypatch):
    root, parent = prepared(tmp_path, monkeypatch)
    calls, execute = executor_fixture()
    monkeypatch.setattr(ladder, 'execute_phase', execute)
    result = ladder.run(root)
    assert calls == [(blocks, phase) for blocks in (16,17) for phase in ladder.PHASES]
    assert result['status'] == 'target_depth_reached' and result['last_passing']['compressed_blocks'] == 17
    assert result['confirmed'] is False
    assert result['stages'][0]['parent_root'] == str(parent)
    assert result['stages'][1]['parent_root'] == str(root/'stages/blocks_16')
    count = len(calls)
    assert ladder.run(root) == result and len(calls) == count
    assert ladder.status(root) == result


def test_failed_development_stops_and_keeps_last_passing_checkpoint(tmp_path, monkeypatch):
    root, _ = prepared(tmp_path, monkeypatch, target=18)
    calls, execute = executor_fixture(fail_depth=17)
    monkeypatch.setattr(ladder, 'execute_phase', execute)
    result = ladder.run(root)
    assert result['status'] == 'stopped_development_gate'
    assert result['last_passing']['compressed_blocks'] == 16
    assert result['stages'][-1]['checkpoint'] == result['last_passing']['checkpoint']
    assert all(blocks < 18 for blocks, _ in calls)
    assert not (root/'stages/blocks_18').exists()


def test_partial_worker_and_changed_source_cannot_be_silently_resumed(tmp_path, monkeypatch):
    root, parent = prepared(tmp_path, monkeypatch, target=16)
    directory = root/'stages/blocks_16'
    FakeStage().prepare(directory, parent)
    (directory/'boundary').mkdir()
    monkeypatch.setattr(ladder, 'execute_phase', lambda *args: pytest.fail('Duplicate worker launched'))
    with pytest.raises(RuntimeError, match='Incomplete phase'):
        ladder.run(root)
    assert ladder.read(root/'campaign_results.json')['status'] == 'execution_failed'
    p = ladder.read(root/'plan.json'); source_path = next(iter(p['source_sha256']))
    archived = root/'producer'/Path(source_path).relative_to(ladder.ROOT)
    archived.write_text('changed producer')
    with pytest.raises(ValueError, match='Producing source'):
        ladder.verify_plan(root)


def test_exclusive_controller_and_interruption_status(tmp_path, monkeypatch):
    root, _ = prepared(tmp_path, monkeypatch)
    assert ladder.status(root)['status'] == 'prepared'
    with ladder.exclusive(root):
        with pytest.raises(RuntimeError, match='Another progressive controller'):
            with ladder.exclusive(root):
                pass
        assert ladder.status(root)['status'] == 'running'
        ladder.event(root, kind='phase_started', phase='boundary')
    assert ladder.status(root)['status'] == 'interrupted'


def test_only_bounded_preworker_credential_failures_retry(tmp_path):
    root = tmp_path/'campaign'; root.mkdir()
    stage = root/'stages/blocks_16'; stage.mkdir(parents=True)
    calls, sleeps = [], []
    def execute(command, **kwargs):
        calls.append(command)
        kwargs['stdout'].write(b"can't reach Infisical: No route to host\n")
        return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError, match='retry budget exhausted'):
        ladder.execute_phase(root, stage, 'boundary', executor=execute, sleep=sleeps.append)
    assert len(calls) == 3 and sleeps == [5,15]
    assert len(list(root.glob('commands/*/boundary/*/exit.json'))) == 3


def test_created_worker_artifacts_prevent_credential_retry(tmp_path):
    root = tmp_path/'campaign'; root.mkdir()
    stage = root/'stages/blocks_16'; stage.mkdir(parents=True)
    calls = []
    def execute(command, **kwargs):
        calls.append(command); (stage/'boundary').mkdir()
        kwargs['stdout'].write(b"can't reach Infisical\n")
        return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError, match='Worker failed'):
        ladder.execute_phase(root, stage, 'boundary', executor=execute,
                             sleep=lambda n: pytest.fail('Retried a started worker'))
    assert len(calls) == 1


def test_ambiguous_prior_launch_does_not_duplicate_process(tmp_path):
    root = tmp_path/'campaign'; root.mkdir()
    stage = root/'stages/blocks_16'; stage.mkdir(parents=True)
    directory = root/'commands/blocks_16/boundary/attempt_01'
    directory.mkdir(parents=True)
    ladder.write(directory/'started.json', {'command': ['existing launch']})
    with pytest.raises(RuntimeError, match='Unfinished prior launch'):
        ladder.execute_phase(root, stage, 'boundary', executor=lambda *a,**kw: pytest.fail('Duplicate launch'))


def test_structurally_full_requires_all36_verified_blocks(tmp_path):
    directory = tmp_path/'stage'; directory.mkdir()
    ladder.write(directory/'verify/manifest.json', {})
    ladder.write(directory/'verify/result.json', {})
    result = dict(passed=True, development_passed=True, compressed_blocks=36, promoted=False,
        confirmed=False, full36=True, review_checkpoint='model.pt', review_checkpoint_sha256='sha')
    trained = dict(candidate=dict(metrics={'correct':190}))
    assert ladder.stage_record(directory, tmp_path, 36, result, trained)['compressed_blocks'] == 36
    with pytest.raises(ValueError, match='structural extent'):
        ladder.stage_record(directory, tmp_path, 36, dict(result,full36=False), trained)
