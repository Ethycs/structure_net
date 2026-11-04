"""Expose only native OpenAI chat and model listing on the trusted gateway network."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from contextlib import ExitStack


MAX_BODY_BYTES = 4 * 1024 * 1024
ROUTES = {'/v1/chat/completions': 'POST', '/v1/models': 'GET'}


def start_chat_proxy(port, native_port=None, host='0.0.0.0', manager=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            # Never log prompts, authorization headers, or native response bodies.
            pass

        def error(self, status, message):
            payload = json.dumps({'error': {'message': message, 'type': 'proxy_error',
                                           'code': status}}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True

        def forward(self):
            if self.path not in ROUTES:
                self.error(404, 'Unknown endpoint')
                return
            if self.command != ROUTES[self.path]:
                self.error(405, 'Method not allowed')
                return
            body = None
            model = None
            if self.command == 'GET' and manager is not None:
                payload = json.dumps(manager.list_models()).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.command == 'POST':
                if self.headers.get('Transfer-Encoding'):
                    self.error(400, 'Use Content-Length for request bodies')
                    return
                lengths = self.headers.get_all('Content-Length') or []
                if len(lengths) != 1:
                    self.error(411, 'One Content-Length is required')
                    return
                try:
                    length = int(lengths[0])
                except ValueError:
                    self.error(400, 'Invalid Content-Length')
                    return
                if not 0 <= length <= MAX_BODY_BYTES:
                    self.error(413, 'Request body is too large')
                    return
                self.connection.settimeout(30)
                body = self.rfile.read(length)
                if len(body) != length:
                    self.error(400, 'Incomplete request body')
                    return
                if manager is not None:
                    try:
                        data = json.loads(body)
                        model = data.get('model') if isinstance(data, dict) else None
                        manager.validate(model)
                        if not isinstance(data.get('messages'), list) or not data['messages']:
                            raise ValueError('A nonempty messages array is required')
                    except (ValueError, TypeError):
                        self.error(400, 'Unknown model or invalid chat request')
                        return
            upstream = None
            started = False
            try:
                # The lease spans the full response, including SSE. A model
                # switch can never unload a worker serving another request.
                with ExitStack() as stack:
                    port = stack.enter_context(manager.lease(model)) if manager else native_port
                    upstream = http.client.HTTPConnection('127.0.0.1', port, timeout=300)
                    upstream.request(self.command, self.path, body,
                                     {'Content-Type': 'application/json', 'Connection': 'close'})
                    response = upstream.getresponse()
                    disconnected = False
                    try:
                        self.send_response(response.status)
                        self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                        self.send_header('Cache-Control', 'no-store')
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        started = True
                    except OSError:
                        disconnected = True
                    try:
                        while chunk := response.read1(65536):
                            if not disconnected:
                                try:
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                                except OSError:
                                    # Drain generation while retaining its lease,
                                    # even if the downstream client disconnects.
                                    disconnected = True
                    finally:
                        upstream.close()
            except (OSError, http.client.HTTPException, RuntimeError):
                if not started:
                    self.error(503, 'Model is unavailable')
            finally:
                if upstream:
                    upstream.close()
                self.close_connection = True

        do_GET = forward
        do_POST = forward

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name='word-games-chat-proxy', daemon=True)
    thread.start()
    return server
