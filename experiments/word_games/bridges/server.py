"""CPU-only HTTP lifecycle; outbound inference uses each caller's supplied key."""
import argparse
import json
import os
import re
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .client import Client, InvalidGeneration, ModelUnavailable
from .service import Conflict, Service


MAX_BYTES = 128 * 1024


def make_server(address, service, client_factory):
    workers = threading.BoundedSemaphore(8)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            # No prompts, keys, capabilities, source paths or request bodies.
            pass

        def send_json(self, status, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(raw)
            self.close_connection = True

        def body(self):
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('unsupported_transfer_encoding')
            lengths = self.headers.get_all('Content-Length') or []
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise ValueError('content_length_required')
            size = int(lengths[0])
            if not 0 < size <= MAX_BYTES:
                raise ValueError('request_body_size')
            self.connection.settimeout(30)
            data = self.rfile.read(size)
            if len(data) != size:
                raise ValueError('incomplete_body')
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError('request_object_required')
            return value

        def handle_request(self):
            if self.command == 'GET' and self.path == '/health':
                self.send_json(200, {'status': 'ok', 'service': 'word-games-bridges',
                                    'version': '1.0.0', 'profile': 'reservations-v4',
                                    'model': 'qwen3.8-35b-a3b',
                                    'model_readiness': 'not_checked_no_startup_load'})
                return
            if not workers.acquire(blocking=False):
                self.send_json(503, {'error': 'bridge_busy'})
                return
            try:
                auth = self.headers.get_all('Authorization') or []
                if len(auth) != 1 or not auth[0].startswith('Bearer ') or len(auth[0]) > 4096:
                    raise PermissionError('bearer_required')
                key = auth[0][7:]
                client = client_factory(key)
                caller = client.caller(key)
                if self.command == 'POST' and self.path == '/v1/sessions':
                    result = service.create(client, caller, self.body())
                    self.send_json(201, result)
                    return
                match = re.fullmatch(r'/v1/sessions/([0-9a-f]{32})/(doer|controller)(?:/([a-z]+))?', self.path)
                if not match:
                    self.send_json(404, {'error': 'route_not_found'})
                    return
                sid, role, action = match.groups()
                capabilities = self.headers.get_all('X-Bridge-Capability') or []
                if len(capabilities) != 1:
                    raise PermissionError('session_capability_required')
                if self.command == 'GET' and action is None:
                    result = service.inspect(sid, capabilities[0], role)
                elif self.command == 'POST' and action:
                    result = service.mutate(sid, capabilities[0], role, action, self.body(), client)
                else:
                    self.send_json(405, {'error': 'method_not_allowed'})
                    return
                self.send_json(200, result)
            except PermissionError:
                self.send_json(403, {'error': 'access_denied'})
            except Conflict:
                self.send_json(409, {'error': 'session_conflict'})
            except ModelUnavailable:
                self.send_json(503, {'error': 'model_unavailable', 'retry_automatic': False})
            except InvalidGeneration:
                self.send_json(502, {'error': 'invalid_model_generation', 'state_committed': False})
            except (ValueError, KeyError, TypeError, IndexError):
                # Do not disclose source-side validation details through either view.
                self.send_json(422, {'error': 'invalid_request_or_proposal', 'state_committed': False})
            except (BrokenPipeError, ConnectionResetError):
                pass  # A completed durable commit remains queryable/replayable.
            except Exception:
                self.send_json(500, {'error': 'internal_error'})
            finally:
                workers.release()

        do_GET = handle_request
        do_POST = handle_request

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True

        def get_request(self):
            conn, address = super().get_request()
            conn.settimeout(30)
            return conn, address

    return Server(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='10.158.218.141')
    parser.add_argument('--port', type=int, default=18092)
    parser.add_argument('--state', default='/var/lib/word-games-bridges/bridges.sqlite')
    parser.add_argument('--ca', default='/etc/word-games-bridges/ca.crt')
    parser.add_argument('--api-base', default='https://central-dev.zt:4000/v1')
    args = parser.parse_args()
    os.umask(0o077)
    service = Service(args.state)
    server = make_server((args.bind, args.port), service,
                         lambda key: Client(lambda: key, base=args.api_base, ca=args.ca))
    def stop(*_args):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()  # Drain handlers before closing durable state.
        service.db.close()


if __name__ == '__main__':
    main()
