"""Boundary and lifecycle checks for the real native-process adapter."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setitem(sys.modules, 'triton_python_backend_utils', SimpleNamespace())
    spec = importlib.util.spec_from_file_location('native_backend_test',
        Path('experiments/word_games_triton/model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampling_rejects_unmeasured_controls(backend):
    assert backend.sampling('{}') == 1024
    assert backend.sampling('{"max_tokens": 128}') == 128
    assert backend.sampling('{"temperature":0.0,"response_format":{"type":"json_object"},"max_tokens":128}') == 128
    for params in ({'temperature': 1}, {'seed': 4}, {'response_format': {'type': 'text'}},
                   {'max_tokens': True}, {'max_tokens': 1025}, [], {'max_tokens': 0}):
        with pytest.raises(ValueError):
            backend.sampling(json.dumps(params))


def test_finalize_waits_before_forced_termination(backend, monkeypatch):
    kill = Mock()
    monkeypatch.setattr(backend.os, 'killpg', kill)
    model = backend.TritonPythonModel()
    model.http, model.log = Mock(), Mock()
    model.process = Mock(pid=123)
    model.process.poll.return_value = None
    model.process.wait.side_effect = [subprocess.TimeoutExpired('native', 45), 0]
    model.finalize()
    assert [c.args for c in kill.call_args_list] == [(123, backend.signal.SIGTERM), (123, backend.signal.SIGKILL)]
    assert [c.kwargs for c in model.process.wait.call_args_list] == [{'timeout': 45}, {'timeout': 10}]
    model.log.close.assert_called_once()


def test_native_request_closes_connection_without_retrying(backend):
    model = backend.TritonPythonModel()
    model.http = Mock()
    model.http.getresponse.return_value.status = 200
    model.http.getresponse.return_value.read.return_value = b'{"ok":true}'
    assert model.call('POST', '/completion', {'prompt': 'x'}) == {'ok': True}
    model.http.request.assert_called_once()
    assert model.http.request.call_args.args[3]['Connection'] == 'close'
    model.http.close.assert_called_once()
    model.http.request.side_effect = BrokenPipeError('test')
    with pytest.raises(BrokenPipeError):
        model.call('POST', '/completion', {})
    assert model.http.request.call_count == 2
    assert model.http.close.call_count == 2


def test_container_exposes_only_triton_on_loopback():
    from experiments.word_games_triton.deploy import command
    args = command()
    ports = [args[i + 1] for i, arg in enumerate(args) if arg == '-p']
    assert ports == ['127.0.0.1:8020:8000', '127.0.0.1:8021:8001', '127.0.0.1:8022:8002']
    assert '--model-control-mode=explicit' in args
    assert '--load-model' not in ' '.join(args)


def test_mutable_deployment_state_can_transition_twice(monkeypatch, tmp_path):
    from experiments.word_games_triton import deploy
    monkeypatch.setattr(deploy, 'ROOT', tmp_path)
    deploy.save_state({'running': True, 'evidence': 'first'})
    deploy.save_state({'running': False, 'evidence': 'first'})
    deploy.save_state({'running': True, 'evidence': 'second'})
    assert json.loads((tmp_path / 'state.json').read_text()) == {'running': True, 'evidence': 'second'}
