import pytest
import http.server
import socket
import threading

from experiments.word_games_ipc.transport import DraftSession, UnixHTTP, accepted_prefix


class Worker:
    def call(self, endpoint, body):
        assert endpoint == '/completion'
        assert isinstance(body['prompt'][0], int)
        return {'tokens':[4,5,99]}, .01


def test_draft_session_blocks_control_ids_and_stale_results():
    s = DraftSession(Worker(), [1,2,3,4,5], 'vocab')
    s.begin([1,2], tokenizer_fingerprint='vocab')
    proposal = s.propose([1,2,3])
    assert s.validate(proposal,[1,2,3]) == [4,5]
    with pytest.raises(ValueError): s.validate(proposal,[1,2,4])
    s.begin([1,2], tokenizer_fingerprint='vocab')
    with pytest.raises(ValueError): s.validate(proposal,[1,2,3])
    with pytest.raises(ValueError): s.propose([1,3])
    with pytest.raises(ValueError): s.begin([1], tokenizer_fingerprint='other')


def test_reference_rejects_first_wrong_token_and_stops_at_first_mismatch():
    assert accepted_prefix([1,2,3],[4,2,3]) == 0
    assert accepted_prefix([1,2,3],[1,4,3]) == 1
    assert accepted_prefix([1,2,3],[1,2,3]) == 3


def test_idle_worker_connection_reopens_without_retrying_generation(tmp_path):
    class Server(http.server.HTTPServer):
        address_family = socket.AF_UNIX
        def server_bind(self):
            self.socket.bind(self.server_address)
            self.server_name, self.server_port = 'localhost',0
    calls = []
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def do_POST(self):
            calls.append(self.rfile.read(int(self.headers['Content-Length'])))
            body=b'{"tokens":[4,5]}'
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
            self.wfile.write(body)
            self.close_connection=True  # simulate a keep-alive socket timing out
        def log_message(self,*args): pass
    path=tmp_path/'worker.sock'
    server=Server(str(path),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    client=UnixHTTP(path)
    try:
        assert client.call('/completion',{'prompt':[1]})[0]['tokens']==[4,5]
        client.last_activity-=2
        assert client.call('/completion',{'prompt':[2]})[0]['tokens']==[4,5]
        assert len(calls)==2
    finally:
        client.close();server.shutdown();server.server_close();thread.join()
