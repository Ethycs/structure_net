from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.neural_architecture_lab import ncm_public_milestone_terminal as terminal


def metadata(epoch=2, attempt=1):
    unit = terminal.locations(epoch, attempt)[0]
    start = terminal.study.CONFIG.milestones[terminal.study.CONFIG.milestones.index(epoch)-1]*1692
    resume = terminal.study.STORAGE/f'epoch1_attempt2/step_{start:06d}.pt'
    plan = {'epoch': epoch, 'attempt': attempt, 'unit': unit, 'workdir': str(terminal.REPO),
            'expected_first_completed_update': start, 'expected_last_completed_update': epoch*1692,
            'expected_training_identity': 'identity', 'resume': str(resume), 'resume_sha256': 'checkpoint',
            'restart_policy': 'no', 'physical_gpu': 1, 'same_fixed_eight_epoch_schedule': True,
            'environment': dict(terminal.ENVIRONMENT), 'artifacts': {},
            'argv': ['/home/rabbit/.pixi/bin/pixi', 'run', '--locked', '-e', 'ncm', 'python',
                     '-m', 'experiments.structure_net.ncm_public_training', '--end-epoch', str(epoch),
                     '--attempt', str(attempt), '--resume', str(resume)]}
    state = {'ActiveState': 'active', 'SubState': 'running', 'Result': 'success', 'MainPID': '123',
             'ExecMainPID': '123', 'ExecMainCode': '0', 'ExecMainStatus': '0',
             'NRestarts': '0', 'InvocationID': 'fixed', 'FragmentPath': 'fragment'}
    launch = {'unit': unit, 'expected_epoch': epoch, 'expected_attempt': attempt,
              'expected_first_completed_update': start, 'expected_last_completed_update': epoch*1692,
              'expected_training_identity': 'identity', 'state': state, 'artifacts': {}}
    return unit, plan, launch


@pytest.mark.parametrize('epoch', [2, 4, 8])
def test_later_milestone_requires_its_own_launch_and_completed_update_count(epoch):
    unit, plan, launch = metadata(epoch)
    terminal.validate_launch(epoch, 1, unit, plan, launch)
    for key, value in (('expected_last_completed_update', 1692), ('epoch', 1),
                       ('expected_first_completed_update', 0), ('restart_policy', 'always')):
        changed = deepcopy(plan); changed[key] = value
        with pytest.raises(ValueError): terminal.validate_launch(epoch, 1, unit, changed, launch)
    changed = deepcopy(launch); changed['expected_epoch'] = 1
    with pytest.raises(ValueError): terminal.validate_launch(epoch, 1, unit, plan, changed)


def test_green_epoch_one_result_cannot_complete_epoch_two():
    value = {'status': 'public_curriculum_milestone_training_completed', 'epoch': 1, 'updates': 1692,
             'training_identity': 'identity', 'task_accuracy': None, 'strict_reload_passed': True,
             'oracle_unchanged': True, 'final_test_opened': False, 'speed_tested': False}
    with pytest.raises(ValueError): terminal.validate_result(value, 2, 'identity')
    value.update(epoch=2, updates=3384)
    terminal.validate_result(value, 2, 'identity')
    for change in ({'updates': 3383}, {'training_identity': 'different'},
                   {'strict_reload_passed': False}, {'task_accuracy': .95}, {'final_test_opened': True}):
        with pytest.raises(ValueError): terminal.validate_result({**value, **change}, 2, 'identity')


def test_live_success_defaults_never_write_terminal_evidence(tmp_path, monkeypatch):
    unit, plan, launch = metadata()
    supervision = tmp_path/'supervision'; supervision.mkdir()
    destination = tmp_path/'terminal'
    fragment = tmp_path/'fragment'; fragment.write_text('same service')
    (supervision/'unit_snapshot.service').write_bytes(fragment.read_bytes())
    terminal.data.write(supervision/'launch_plan.json', plan)
    launch['state']['FragmentPath'] = str(fragment)
    launch['artifacts'] = {str(supervision/n): terminal.data.sha(supervision/n)
                           for n in ('launch_plan.json', 'unit_snapshot.service')}
    terminal.data.write(supervision/'launch.json', launch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(terminal, 'locations', lambda *args: (unit, tmp_path/'training', supervision, destination))
    monkeypatch.setattr(terminal, 'state_now', lambda unit: deepcopy(launch['state']))
    monkeypatch.setattr(terminal, 'live_command', lambda pid: plan['argv'])
    def forbidden():
        raise AssertionError('a live observation must not configure terminal verification')
    monkeypatch.setattr(terminal.study.pipeline.broad, 'runtime', SimpleNamespace(configure=forbidden))
    result = terminal.observe(2, 1)
    assert not result['training_terminal'] and not result['training_success']
    assert not destination.exists()
    monkeypatch.setattr(terminal, 'live_command', lambda pid: plan['argv'][:-1]+['wrong-checkpoint'])
    with pytest.raises(ValueError): terminal.observe(2, 1)
    assert not destination.exists()


def test_epoch_one_observer_and_invalid_attempts_stay_separate():
    for epoch, attempt in ((1, 1), (3, 1), (2, 0), (2, True)):
        with pytest.raises(ValueError): terminal.locations(epoch, attempt)
