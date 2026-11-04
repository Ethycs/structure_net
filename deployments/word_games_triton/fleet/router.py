"""Serialize GPU leases and use explicit Triton control for on-demand models."""
from contextlib import contextmanager
import fcntl
import json
import threading
import time
import urllib.error
import urllib.request


GPU_LOCK = '/tmp/word-games-model-gpus.lock'


class Manager:
    def __init__(self, config, base='http://127.0.0.1:8000'):
        self.models = {m['alias']: m for m in config['models']}
        self.base = base
        self.mutex = threading.Lock()
        self.stopping = False

    def validate(self, alias):
        if not isinstance(alias, str) or alias not in self.models:
            raise ValueError('Unknown model')

    def api(self, path, body=None):
        request = urllib.request.Request(self.base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=540) as response:
            data = response.read()
            return json.loads(data) if data else {}

    def ready(self, name):
        try:
            with urllib.request.urlopen(self.base + f'/v2/models/{name}/ready', timeout=5) as response:
                return response.status == 200
        except urllib.error.HTTPError as error:
            if error.code in (400, 404, 503):
                return False
            raise

    def list_models(self):
        return {'object': 'list', 'data': [
            {'id': alias, 'object': 'model', 'created': 0, 'owned_by': 'word-games-triton'}
            for alias in self.models]}

    def wait_released(self):
        deadline = time.monotonic() + 60
        with open(GPU_LOCK, 'a') as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    return
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Previous native worker has not released its GPU lease')
                    time.sleep(.1)

    def select(self, alias=None):
        # Called only while holding mutex. Never touch models outside this catalog.
        if alias is not None:
            self.validate(alias)
        selected = self.models[alias]['triton_model'] if alias is not None else None
        for entry in self.models.values():
            name = entry['triton_model']
            if name != selected and self.ready(name):
                print(json.dumps({'event': 'unload', 'model': name}), flush=True)
                self.api(f'/v2/repository/models/{name}/unload', {})
                self.wait_released()
        if selected and not self.ready(selected):
            self.wait_released()
            try:
                print(json.dumps({'event': 'load', 'model': selected}), flush=True)
                self.api(f'/v2/repository/models/{selected}/load', {})
                if not self.ready(selected):
                    raise RuntimeError('Triton did not make requested model ready')
            except Exception:
                # Failed loads must never silently fall back to another model.
                self.api(f'/v2/repository/models/{selected}/unload', {})
                raise

    @contextmanager
    def lease(self, alias):
        self.validate(alias)
        if not self.mutex.acquire(timeout=540):
            raise TimeoutError('GPU request queue timed out')
        try:
            if self.stopping:
                raise RuntimeError('Router is stopping')
            self.select(alias)
            yield self.models[alias]['native_port']
        finally:
            self.mutex.release()

    def control(self, action, alias=None):
        if action not in ('status', 'load', 'unload'):
            raise ValueError('Unknown control action')
        if action == 'load':
            with self.lease(alias):
                return self.status()
        with self.mutex:
            if action == 'unload':
                self.select(None)
            return self.status()

    def status(self):
        return {'models': [{**{k: m[k] for k in ('alias', 'triton_model', 'native_port')},
                            'ready': self.ready(m['triton_model'])} for m in self.models.values()],
                'policy': 'one GPU model at a time; switch after active generation completes'}
