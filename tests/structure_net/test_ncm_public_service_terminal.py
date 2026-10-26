from copy import deepcopy

import pytest

from experiments.neural_architecture_lab import ncm_public_service_terminal as terminal


def running():
    return {'ActiveState': 'active', 'SubState': 'running', 'Result': 'success',
        'MainPID': '123', 'ExecMainPID': '123', 'ExecMainCode': '0', 'ExecMainStatus': '0',
        'NRestarts': '0', 'InvocationID': 'fixed'}


def test_default_success_and_zero_status_do_not_establish_completion():
    state = running()
    assert terminal.classify(state, {'state': deepcopy(state)}) == 'running'


def test_success_requires_exited_process_and_matching_service_identity():
    original = running()
    state = {**original, 'MainPID': '0', 'SubState': 'exited', 'ExecMainCode': '1'}
    assert terminal.classify(state, {'state': original}) == 'terminal_success'
    for change in ({'ExecMainCode': '0'}, {'ExecMainStatus': '9'}, {'MainPID': '123'},
                   {'InvocationID': 'new'}, {'ExecMainPID': '124'}, {'NRestarts': '1'},
                   {'Result': 'exit-code'}, {'ActiveState': 'inactive'}):
        with pytest.raises(ValueError): terminal.classify({**state, **change}, {'state': original})


def test_failed_service_is_terminal_failure_even_if_a_result_file_could_exist():
    original = running()
    state = {**original, 'MainPID': '0', 'ActiveState': 'failed', 'SubState': 'failed',
        'Result': 'signal', 'ExecMainCode': '2', 'ExecMainStatus': '9'}
    assert terminal.classify(state, {'state': original}) == 'terminal_failure'


def test_missing_or_transitional_process_is_not_a_verified_wait():
    state = running()
    for change in ({'MainPID': '0'}, {'ExecMainCode': '1'},
                   {'ActiveState': 'activating', 'SubState': 'start'}):
        with pytest.raises(ValueError): terminal.classify({**state, **change}, {'state': state})
