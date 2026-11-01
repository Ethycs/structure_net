"""Reversible, explicit-model-control leases for the two resident GPU models."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

import requests

from .common import write
from .native_speed import GPU
from .moe_placement import SUPER
from .three_gpu_experts import SMALL


@dataclass(frozen=True)
class Service:
    name: str
    url: str
    gpu: str
    minimum_free_mib: int


SERVICES = (Service('qwen3-8b-gguf', 'http://127.0.0.1:8000', SMALL, 5200),
            Service('glm-ocr', 'http://127.0.0.1:8010', SUPER, 7300))


class ModelLease:
    def __init__(self, out, services=SERVICES):
        self.out = Path(out)
        self.services = services
        self.before, self.attempted = {}, []
        self.restored = False

    def acquire(self):
        self.out.mkdir(parents=True, exist_ok=False)
        for s in self.services:
            ready = requests.get(s.url + f'/v2/models/{s.name}/ready', timeout=10)
            if ready.status_code != 200:
                raise RuntimeError(s.name + ' must be ready before this measured lease')
            config = requests.get(s.url + f'/v2/models/{s.name}/config', timeout=10)
            config.raise_for_status()
            self.before[s.name] = config.json()
        write(self.out / 'before.json', self.before)
        for s in self.services:
            self.attempted.append(s)
            r = requests.post(s.url + f'/v2/repository/models/{s.name}/unload', json={}, timeout=120)
            write(self.out / (s.name + '_unload.json'), {'status': r.status_code, 'body': r.text})
            r.raise_for_status()
        samples, stable = [], 0
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                           '--format=csv,noheader,nounits'], text=True)
            free = {a.strip(): int(b.strip()) for a, b in (line.split(',') for line in raw.strip().splitlines())}
            samples.append({'monotonic': time.monotonic(), 'free_mib': free})
            stable = stable + 1 if all(free[s.gpu] >= s.minimum_free_mib for s in self.services) else 0
            if stable >= 2:
                write(self.out / 'release.json', {'released': True, 'samples': samples})
                return
            time.sleep(1)
        write(self.out / 'release.json', {'released': False, 'samples': samples})
        raise RuntimeError('resident model CUDA allocations did not release')

    def restore(self):
        results = {}
        # Try every restoration even if another service fails.
        for s in self.attempted:
            try:
                r = requests.post(s.url + f'/v2/repository/models/{s.name}/load', json={}, timeout=240)
                r.raise_for_status()
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if requests.get(s.url + f'/v2/models/{s.name}/ready', timeout=5).status_code == 200:
                        break
                    time.sleep(1)
                else:
                    raise TimeoutError('model readiness')
                c = requests.get(s.url + f'/v2/models/{s.name}/config', timeout=10)
                c.raise_for_status()
                if c.json() != self.before[s.name]:
                    raise ValueError('service configuration changed')
                results[s.name] = {'restored': True}
            except Exception as exc:
                results[s.name] = {'restored': False, 'error': repr(exc)}
        self.restored = bool(self.attempted) and all(r['restored'] for r in results.values())
        write(self.out / 'restored.json', {'all_restored': self.restored, 'services': results})
        if not self.restored:
            raise RuntimeError('service restoration failed: ' + repr(results))
