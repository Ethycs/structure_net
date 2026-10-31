"""Persistent Unix-domain HTTP and prefix-bound draft proposals."""
import http.client
import json
import socket
import time
import uuid

from experiments.word_games.common import digest


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path, timeout=240):
        super().__init__('localhost', timeout=timeout)
        self.path = str(path)
        self.sent_bytes = 0
        self.received_bytes = 0
        self.last_activity = None

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect(self.path)
        except Exception:
            self.close()
            raise

    def call(self, endpoint, body=None):
        # Native HTTP workers close idle keep-alive connections. A draft can
        # leave the target idle while it compiles or generates. Reconnect
        # proactively; never retry a possibly executed generation POST.
        if self.last_activity is not None and time.monotonic() - self.last_activity > 1:
            self.close()
        data = None if body is None else json.dumps(body, separators=(',', ':'), allow_nan=False).encode()
        start = time.perf_counter()
        try:
            self.request('GET' if body is None else 'POST', endpoint, body=data,
                         headers={'Content-Type':'application/json'})
            response = self.getresponse()
            wire = response.read()
            self.last_activity = time.monotonic()
            elapsed = time.perf_counter() - start
            self.sent_bytes += len(data or b'')
            self.received_bytes += len(wire)
            if response.status != 200:
                raise RuntimeError(f'worker HTTP {response.status}: {wire[:200]!r}')
            return json.loads(wire), elapsed
        except Exception:
            self.close()
            raise


def accepted_prefix(proposed, verified):
    """Reference greedy check. Verified IDs must come from the target worker."""
    count = 0
    for a,b in zip(proposed, verified):
        if a != b: break
        count += 1
    return count


class DraftSession:
    """Bind each response to an authorized request and confirmed prefix.

    The native worker receives integer IDs directly. Only ordinary candidate
    IDs are returned. This does not insert unverified candidates into target KV.
    """
    def __init__(self, worker, ordinary_ids, tokenizer_fingerprint):
        self.worker = worker
        self.ordinary = frozenset(ordinary_ids)
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.request_id = None
        self.prefix = None

    def begin(self, prefix, *, tokenizer_fingerprint):
        if tokenizer_fingerprint != self.tokenizer_fingerprint:
            raise ValueError('tokenizer identity mismatch')
        if not prefix or len(prefix) > 8000 or any(type(i) is not int or i < 0 for i in prefix):
            raise ValueError('invalid prefix')
        self.request_id = uuid.uuid4().hex
        self.prefix = tuple(prefix)

    def propose(self, prefix, count=3):
        if self.prefix is None or tuple(prefix[:len(self.prefix)]) != self.prefix:
            raise ValueError('confirmed prefix changed')
        if not 1 <= count <= 8 or len(prefix) + count > 8192:
            raise ValueError('draft bounds')
        response, elapsed = self.worker.call('/completion', {
            'prompt': list(prefix), 'n_predict': count, 'temperature':0, 'seed':73,
            'repeat_penalty':1.0, 'cache_prompt':False, 'return_tokens':True,
            'stream':False})
        tokens = []
        for token in response['tokens']:
            if type(token) is not int or token not in self.ordinary: break
            tokens.append(token)
        if len(tokens) > count:
            raise ValueError('worker exceeded draft bound')
        return {'request_id':self.request_id, 'prefix_sha256':digest(list(prefix)),
                'tokenizer_fingerprint':self.tokenizer_fingerprint, 'tokens':tokens,
                'elapsed_seconds':elapsed}

    def validate(self, proposal, prefix):
        if (proposal['request_id'] != self.request_id or proposal['prefix_sha256'] != digest(list(prefix))
                or proposal['tokenizer_fingerprint'] != self.tokenizer_fingerprint):
            raise ValueError('stale or mismatched draft response')
        if any(type(t) is not int or t not in self.ordinary for t in proposal['tokens']):
            raise ValueError('nonordinary draft token')
        return proposal['tokens']
