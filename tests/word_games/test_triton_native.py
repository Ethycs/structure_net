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


def test_public_prefix_restores_every_request_and_never_saves_user_data(backend):
    model = backend.TritonPythonModel()
    model.prefixes = [{'name': 'sender', 'text': 'PUBLIC\n', 'tokens': [1, 2], 'filename': 'fixed.bin'}]
    model.tokenize = Mock(return_value=[1, 2, 999])
    model.call = Mock(return_value={'n_restored': 2})
    for private in ('private event one', 'unrelated secret two'):
        prompt, info = model.restore_public_prefix('PUBLIC\n' + private, 'formatted ' + private)
        assert prompt == [1, 2, 999] and info['hit'] and info['tokens'] == 2
    assert model.call.call_count == 2
    assert all(c.args == ('POST', '/slots/0?action=restore', {'filename': 'fixed.bin'})
               for c in model.call.call_args_list)
    assert model.prefixes[0]['tokens'] == [1, 2]


def test_unknown_or_mismatching_prefix_cannot_reuse_previous_state(backend):
    model = backend.TritonPythonModel()
    model.prefixes = [{'name': 'sender', 'text': 'PUBLIC\n', 'tokens': [1, 2], 'filename': 'fixed.bin'}]
    model.call = Mock()
    model.tokenize = Mock(return_value=[1, 3, 999])
    assert not model.restore_public_prefix('PRIVATE', 'formatted')[1]['hit']
    model.tokenize.assert_not_called()
    assert model.restore_public_prefix('PUBLIC\nprivate', 'formatted') == (
        'formatted', {'hit': False, 'reason': 'token_prefix_mismatch'})
    model.call.assert_not_called()
    model.tokenize.return_value = [1, 2, 999]
    model.call.return_value = {'n_restored': 1}
    with pytest.raises(ValueError, match='restore length'):
        model.restore_public_prefix('PUBLIC\nprivate', 'formatted')


def test_prefix_build_is_zero_generation_and_drops_boundary_token(backend, tmp_path):
    model = backend.TritonPythonModel()
    model.prefixes, model.prefix_dir = [], tmp_path
    model.tokenize = Mock(return_value=[1, 2, 3])
    model.call = Mock(side_effect=[{'content': '{', 'tokens_predicted': 1, 'tokens_cached': 2}, {'n_saved': 2}, {}])
    model.initialize_prefixes([{'name': 'sender', 'text': 'PUBLIC\n'}])
    body = model.call.call_args_list[0].args[2]
    assert body == {'prompt': [1, 2], 'n_predict': 0, 'cache_prompt': False, 'id_slot': 0}
    assert model.call.call_args_list[-1].args == ('POST', '/slots/0?action=erase', {})
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['public_only'] and receipt['rebuild_on_every_model_load']


def test_prefix_snapshot_rejects_any_extra_cached_token(backend, tmp_path):
    model = backend.TritonPythonModel()
    model.prefixes, model.prefix_dir = [], tmp_path
    model.tokenize = Mock(return_value=[1, 2, 3])
    model.call = Mock(return_value={'tokens_cached': 3})
    with pytest.raises(ValueError, match='unexpected tokens'):
        model.initialize_prefixes([{'name': 'sender', 'text': 'PUBLIC\n'}])
    assert model.call.call_count == 1


def test_sender_and_receiver_caches_are_distinct(backend):
    model = backend.TritonPythonModel()
    model.prefixes = [{'name': n, 'text': n + '\n', 'tokens': [1, t], 'filename': n + '.bin'}
                      for n, t in [('sender', 2), ('receiver', 3)]]
    model.tokenize = Mock(return_value=[1, 3, 9])
    model.call = Mock(return_value={'n_restored': 2})
    _, info = model.restore_public_prefix('receiver\npacket', 'formatted')
    assert info['name'] == 'receiver'
    assert model.call.call_args.args[2] == {'filename': 'receiver.bin'}
