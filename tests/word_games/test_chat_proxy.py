"""Real HTTP checks for the public chat boundary, including unbuffered SSE."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import threading

import pytest


@pytest.fixture
def transport():
    path = Path('deployments/word_games_triton/regular/models/word-games-qwen38-native/1/chat_proxy.py')
    spec = importlib.util.spec_from_file_location('chat_proxy_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests, release = [], threading.Event()

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append((self.command, self.path, None))
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'{}')

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            requests.append((self.command, self.path, body))
            data = json.loads(body)
            if data.get('stream'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(b'data: {"first":true}\n\n')
                self.wfile.flush()
                release.wait(5)
                self.wfile.write(b'data: [DONE]\n\n')
                self.close_connection = True
            else:
                self.send_response(400 if data.get('reject') else 200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    native = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    native.daemon_threads = True
    threading.Thread(target=native.serve_forever, daemon=True).start()
    proxy = module.start_chat_proxy(0, native.server_port, host='127.0.0.1')
    try:
        yield proxy, native, requests, release
    finally:
        release.set()
        proxy.shutdown()
        proxy.server_close()
        native.shutdown()
        native.server_close()


@pytest.mark.parametrize('method,path,status', [
    ('GET', '/slots', 404), ('GET', '/props', 404), ('POST', '/completion', 404),
    ('GET', '/health', 404), ('GET', '/v1/models?redirect=/slots', 404),
    ('GET', '/v1/chat/completions', 405), ('POST', '/v1/models', 405),
])
def test_management_paths_never_reach_native(transport, method, path, status):
    proxy, _, requests, _ = transport
    connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
    connection.request(method, path, b'{}' if method == 'POST' else None)
    response = connection.getresponse()
    assert response.status == status
    response.read()
    connection.close()
    assert not requests


def test_chat_payload_and_response_are_preserved(transport):
    proxy, _, requests, _ = transport
    body = json.dumps({'messages': [
        {'role': 'system', 'content': 'Keep roles.'},
        {'role': 'user', 'content': 'Unicode: é and 日本語'},
        {'role': 'assistant', 'content': 'Previous answer'},
        {'role': 'user', 'content': 'Next turn'},
    ], 'response_format': {'type': 'json_object'}}, ensure_ascii=False).encode()
    connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
    connection.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
    response = connection.getresponse()
    assert response.status == 200 and response.read() == body
    connection.close()
    assert requests == [('POST', '/v1/chat/completions', body)]


def test_sse_first_event_arrives_before_upstream_completes(transport):
    proxy, _, _, release = transport
    connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
    connection.request('POST', '/v1/chat/completions', b'{"stream":true}')
    response = connection.getresponse()
    assert response.getheader('Content-Type') == 'text/event-stream'
    assert response.readline() == b'data: {"first":true}\n'
    assert not release.is_set()
    release.set()
    assert b'data: [DONE]' in response.read()
    connection.close()


def test_native_errors_are_not_reported_as_success(transport):
    proxy, _, _, _ = transport
    connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
    connection.request('POST', '/v1/chat/completions', b'{"reject":true}')
    response = connection.getresponse()
    assert response.status == 400 and json.loads(response.read()) == {'reject': True}
    connection.close()


def test_missing_worker_is_unavailable(transport):
    proxy, native, _, _ = transport
    native.shutdown()
    native.server_close()
    connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
    connection.request('GET', '/v1/models')
    response = connection.getresponse()
    assert response.status == 503
    assert json.loads(response.read())['error']['message'] == 'Model is unavailable'
    connection.close()
