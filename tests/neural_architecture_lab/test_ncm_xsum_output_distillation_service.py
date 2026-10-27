from copy import deepcopy
import pytest

from experiments.neural_architecture_lab import ncm_xsum_output_distillation_service as service


def live():
    return dict(InvocationID='fixed', NRestarts='0', ExecMainPID='42', MainPID='42',
                ActiveState='active', SubState='running', ExecMainCode='0', ExecMainStatus='0', Result='success')


def terminal():
    return {**live(), 'MainPID': '0', 'SubState': 'exited', 'ExecMainCode': '1'}


def test_launch_success_and_running_zero_status_are_not_training_success():
    assert service.classify(live(), {'state': live()}) == 'running'
    assert service.classify(terminal(), {'state': live()}) == 'terminal_success'


@pytest.mark.parametrize('change', [dict(InvocationID='other'), dict(NRestarts='1'),
    dict(ExecMainPID='43'), dict(MainPID='44'), dict(MainPID='0', SubState='exited', ExecMainCode='0')])
def test_mismatched_or_ambiguous_state_is_rejected(change):
    with pytest.raises(ValueError): service.classify({**live(), **change}, {'state': live()})


@pytest.mark.parametrize('code,status', [('1','1'), ('2','9'), ('3','11')])
def test_failed_exit_or_signal_cannot_pass(code, status):
    value = {**terminal(), 'ExecMainCode': code, 'ExecMainStatus': status, 'ActiveState': 'failed', 'SubState': 'failed', 'Result': 'failed'}
    assert service.classify(value, {'state': live()}) == 'terminal_failure'
