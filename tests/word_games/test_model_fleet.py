"""Model identity, lifecycle isolation, and switches across active streams."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading
import time

import pytest

from experiments.word_games_triton import download_catalog, fleet


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manager(tmp_path, monkeypatch):
    module = load('deployments/word_games_triton/fleet/router.py')
    monkeypatch.setattr(module, 'GPU_LOCK', str(tmp_path / 'lease'))
    result = module.Manager({'models': [
        {'alias': 'a', 'triton_model': 'native-a', 'native_port': 1},
        {'alias': 'b', 'triton_model': 'native-b', 'native_port': 2}]})
    states, events = set(), []
    monkeypatch.setattr(result, 'ready', lambda name: name in states)

    def api(path, body):
        _, name, action = path.rsplit('/', 2)
        events.append((action, name))
        if action == 'load':
            assert not states, 'A second weights worker must never overlap'
            states.add(name)
        else:
            states.discard(name)
        return {}

    monkeypatch.setattr(result, 'api', api)
    result.states, result.events = states, events
    return result


def test_unknown_alias_does_not_mutate_gpu_state(manager):
    with pytest.raises(ValueError):
        with manager.lease('../../another-model'):
            pass
    assert not manager.events


def test_stopping_router_rejects_queued_work_before_loading(manager):
    manager.stopping = True
    with pytest.raises(RuntimeError, match='stopping'):
        with manager.lease('a'):
            pytest.fail('Stopping router accepted a lease')
    assert not manager.events


def test_triton_control_accepts_empty_success_body(monkeypatch):
    module = load('deployments/word_games_triton/fleet/router.py')
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *_args, **_kwargs: io.BytesIO(b''))
    assert module.Manager({'models': []}).api('/v2/repository/models/example/load', {}) == {}
    assert fleet.api('/v2/repository/models/example/unload', {}) == {}


def test_switch_waits_for_active_response_and_reuses_resident_weights(manager):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()

    def first():
        with manager.lease('a'):
            entered.set()
            assert release.wait(5)

    def second():
        waiting.set()
        with manager.lease('b'):
            return True

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(first)
        assert entered.wait(2)
        b = pool.submit(second)
        assert waiting.wait(2)
        assert not b.done() and manager.states == {'native-a'}
        release.set()
        a.result(timeout=3)
        assert b.result(timeout=3)
    with manager.lease('b'):
        assert manager.states == {'native-b'}
    assert manager.events == [('load', 'native-a'), ('unload', 'native-a'), ('load', 'native-b')]


def test_failed_load_is_not_replaced_by_another_model(manager, monkeypatch):
    api = manager.api

    def failure(path, body):
        if path.endswith('/native-b/load'):
            manager.events.append(('failed_load', 'native-b'))
            raise RuntimeError('out of memory')
        return api(path, body)

    monkeypatch.setattr(manager, 'api', failure)
    with manager.lease('a'):
        pass
    with pytest.raises(RuntimeError, match='out of memory'):
        with manager.lease('b'):
            pytest.fail('Failed model received a request')
    assert not manager.states
    assert manager.events[-1] == ('unload', 'native-b')
    with manager.lease('a'):
        assert manager.states == {'native-a'}


def test_unload_releases_gpu_models_but_keeps_control_available(manager):
    with manager.lease('a'):
        pass
    result = manager.control('unload')
    assert not any(m['ready'] for m in result['models'])
    assert len(manager.list_models()['data']) == 2


def test_public_router_retains_lease_until_sse_finishes(manager):
    module = load('deployments/word_games_triton/regular/models/word-games-qwen38-native/1/chat_proxy.py')
    release = threading.Event()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(b'data: {"first":true}\n\n')
            self.wfile.flush()
            release.wait(5)
            self.wfile.write(b'data: [DONE]\n\n')

    native = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    threading.Thread(target=native.serve_forever, daemon=True).start()
    manager.models['a']['native_port'] = native.server_port
    proxy = module.start_chat_proxy(0, host='127.0.0.1', manager=manager)
    connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=3)
    try:
        connection.request('POST', '/v1/chat/completions', json.dumps({
            'model': 'a', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True}))
        response = connection.getresponse()
        assert response.status == 200
        assert response.readline() == b'data: {"first":true}\n'
        assert manager.mutex.locked()
        with ThreadPoolExecutor(1) as pool:
            switch = pool.submit(manager.control, 'load', 'b')
            assert not switch.done() and manager.states == {'native-a'}
            release.set()
            assert b'data: [DONE]' in response.read()
            assert switch.result(timeout=3)['models'][1]['ready']
    finally:
        release.set()
        connection.close()
        proxy.shutdown()
        proxy.server_close()
        native.shutdown()
        native.server_close()


def test_all_architectures_have_unique_aliases_and_ports_and_exact_quantization():
    models = fleet.catalog()['models']
    assert len({m['alias'] for m in models}) == len(models) == 4
    assert len({m['native_port'] for m in models}) == 4
    for model in models:
        config = fleet.runtime(model)
        args = config['native_args']
        assert args[args.index('--alias') + 1] == model['alias']
        assert args[args.index('--flash-attn') + 1] == 'on'
        assert args[args.index('--cache-type-k') + 1] == 'q8_0'
        assert args[args.index('--cache-type-v') + 1] == 'q8_0'
        assert config['public_chat_port'] is None
        if model['download']:
            assert model['weights']['file'].endswith(model['quantization'] + '.gguf')
            assert len(model['weights']['revision']) == 40
            assert len(model['weights']['sha256']) == 64


def test_download_rejects_inadequate_destination_space(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(download_catalog.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match='Insufficient space'):
        download_catalog.space_check([{'weights': {'path': '/nonexistent.gguf', 'size': 100}}])


def test_status_never_restarts_a_stopped_router(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr(fleet, 'ready', lambda _name: False)
    api = Mock()
    monkeypatch.setattr(fleet, 'api', api)
    assert not fleet.control('status')['router_ready']
    api.assert_not_called()
