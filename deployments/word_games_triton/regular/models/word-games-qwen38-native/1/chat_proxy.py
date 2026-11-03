"""Expose only native OpenAI chat and model listing on the trusted gateway network."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading


MAX_BODY_BYTES = 4 * 1024 * 1024
ROUTES = {'/v1/chat/completions': 'POST', '/v1/models': 'GET'}


def start_chat_proxy(port, native_port, host='0.0.0.0'):
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
            upstream = http.client.HTTPConnection('127.0.0.1', native_port, timeout=300)
            started = False
            try:
                upstream.request(self.command, self.path, body,
                                 {'Content-Type': 'application/json', 'Connection': 'close'})
                response = upstream.getresponse()
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.end_headers()
                started = True
                # read1 forwards currently available bytes instead of buffering
                # the entire completion or waiting for a fixed-size SSE block.
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException):
                if not started:
                    self.error(503, 'Model is unavailable')
            finally:
                upstream.close()
                self.close_connection = True

        do_GET = forward
        do_POST = forward

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name='word-games-chat-proxy', daemon=True)
    thread.start()
    return server
