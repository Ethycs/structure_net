import pytest

from experiments.neural_architecture_lab import ncm_isolated_judge_gateway_v2 as gateway


def test_service_must_be_original_running_invocation():
    proof = {'PID': 124, 'supervisor_PID': 123, 'invocation_id': 'original'}
    observed = {'ActiveState': 'active', 'SubState': 'running', 'NRestarts': '0',
                'InvocationID': 'original', 'MainPID': '123', 'ExecMainPID': '123'}
    gateway.validate_state(proof, observed)
    for key, value in [('MainPID', '124'), ('InvocationID', 'new'), ('NRestarts', '1'),
                       ('SubState', 'exited'), ('ExecMainPID', '124'), ('ActiveState', 'inactive')]:
        with pytest.raises(ValueError): gateway.validate_state(proof, dict(observed, **{key: value}))


def test_neutral_response_requires_exact_requested_model_and_text():
    model = gateway.MODELS[0]
    wire = {'model': model.removeprefix('codex_sdk/'), 'choices': [{'message': {'content': ' OK '}}]}
    assert gateway.response_record(model, wire)['neutral_exact_OK']
    with pytest.raises(ValueError): gateway.response_record(gateway.MODELS[1], wire)
    wire['choices'][0]['message']['content'] = 'Probably OK'
    assert not gateway.response_record(model, wire)['neutral_exact_OK']


def test_source_binding_detects_changes(tmp_path):
    path = tmp_path/'source.py'; path.write_text('initial')
    binding = {str(path): gateway.sha(path)}; gateway.verify(binding)
    path.write_text('changed')
    with pytest.raises(ValueError): gateway.verify(binding)
