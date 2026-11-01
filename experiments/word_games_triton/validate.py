"""Measure the loaded Triton path and verify unload/reload releases GPU memory."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import requests

from experiments.word_games.common import read, sha, write
from experiments.word_games.cpu_expert_elimination import receiver_accuracy
from experiments.word_games.native_speed import aggregate, output_tokens
from experiments.word_games.qwen_runtime import REQUESTS, REQUESTS_SHA, memory
from experiments.word_games.three_gpu_experts import Monitor
from experiments.word_games import three_gpu_experts as telemetry
from .deploy import MODEL, CONTAINER, ROOT, URL, api


class Client:
    def __init__(self):
        self.http = requests.Session()
        self.rows = []

    def complete(self, prompt, path, max_tokens=1024):
        body = {'inputs': [
            {'name': 'text_input', 'shape': [1], 'datatype': 'BYTES', 'data': [prompt]},
            {'name': 'sampling_parameters', 'shape': [1], 'datatype': 'BYTES',
             'data': [json.dumps({'max_tokens': max_tokens})]}]}
        write(path / 'request.json', body)
        start = time.perf_counter()
        r = self.http.post(URL + f'/v2/models/{MODEL}/infer', json=body, timeout=240)
        if not r.ok:
            write(path / 'response.json', {'http_status': r.status_code, 'error': r.text,
                                         'elapsed_seconds': time.perf_counter() - start})
        r.raise_for_status()
        raw = r.json()
        elapsed = time.perf_counter() - start
        if raw.get('model_name') != MODEL:
            raise ValueError('response came from another model')
        outputs = {item['name']: item['data'][0] for item in raw['outputs']}
        result = json.loads(outputs['generation_info'])
        result['content'] = outputs['text_output']
        parsed = json.loads(result['content'])
        if not isinstance(parsed, dict) or result.get('truncated') or result['stop_type'] == 'limit':
            raise ValueError('invalid or truncated generated object')
        row = {'elapsed_seconds': elapsed, 'response': result, 'parsed': parsed}
        write(path / 'response.json', row)
        self.rows.append(row)
        return result['content']


def run(out, reference):
    out.mkdir(parents=True, exist_ok=False)
    if sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('frozen input changed')
    manifest = read(ROOT / 'manifest.json')
    expected = read(reference / 'placement.json')
    if {k: v for k, v in manifest['config'].items() if k != 'name'} != {
            k: v for k, v in expected['config'].items() if k != 'name'}:
        raise ValueError('reference configuration differs from deployment')
    producer = {'manifest': manifest, 'requests_sha256': REQUESTS_SHA, 'reference': str(reference),
               'source_hashes': {str(p): sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()},
               'timing': 'Triton inference HTTP round trip including template application and native completion',
               'profile': manifest['profile'], 'weights_changed': False}
    write(out / 'producer.json', producer)
    # The original servers are stopped during deployment; monitor this server.
    telemetry.TRITON = URL
    monitor = Monitor(out, CONTAINER)
    monitor.start()
    client, passes = Client(), []
    try:
        for repeat in (1, 2):
            root = out / f'full_{repeat}'
            logfile = max((ROOT / 'runtime').glob('native-*.log'), key=lambda p: p.stat().st_mtime_ns)
            log = logfile.read_text()
            buffers = {d: float(v) for d, v in re.findall(
                r'load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB', log)}
            if set(buffers) != set(expected['model_buffers_mib']) or any(
                    abs(buffers[k] - v) > .03 for k, v in expected['model_buffers_mib'].items()):
                raise ValueError('Triton native weight placement differs from measured reference')
            if ('pipeline parallelism enabled' in log) != expected['pipeline'] or 'retrying without pipeline' in log:
                raise ValueError('native pipeline status changed')
            write(root / 'placement.json', {'model_buffers_mib': buffers, 'pipeline': expected['pipeline'],
                                           'native_log': str(logfile), 'reference_sha256': sha(reference / 'placement.json')})
            client.complete('Return exactly {"ready":true} as JSON.', root / 'warmup', 24)
            client.rows.clear()
            for item in read(REQUESTS):
                if monitor.error:
                    raise RuntimeError(monitor.error)
                client.complete(item['prompt'], root / 'runs' / item['id'], item['max_tokens'])
                row = client.rows[-1]
                print(json.dumps({'repeat': repeat, 'id': item['id'], 'seconds': row['elapsed_seconds'],
                                  'tokens': output_tokens(row['response'])}), flush=True)
            quality = receiver_accuracy(root)
            write(root / 'quality.json', quality)
            differences = []
            for item in read(REQUESTS):
                actual = read(root / 'runs' / item['id'] / 'response.json')
                previous = read(reference / 'runs' / item['id'] / 'response.json')
                if actual['parsed'] != previous['parsed']:
                    differences.append(item['id'])
            passes.append({'repeat': repeat, 'totals': aggregate(client.rows),
                           'quality': quality, 'different_parsed_outputs_from_native': differences})
            if monitor.error:
                raise RuntimeError(monitor.error)
            if repeat == 1:
                monitor.close()
                before = memory()
                unloaded = api('unload')
                # Triton must no longer advertise this model as ready.
                ready = requests.get(URL + f'/v2/models/{MODEL}/ready', timeout=10)
                if ready.status_code == 200:
                    raise ValueError('unloaded model still ready')
                released = None
                for _ in range(30):
                    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                                   '--format=csv,noheader,nounits'], text=True)
                    used = {a.strip(): int(b.strip()) for a, b in (line.split(',') for line in raw.strip().splitlines())}
                    if all(v < 1000 for v in used.values()):
                        released = used
                        break
                    time.sleep(1)
                if released is None:
                    raise ValueError('unload did not release model GPU allocations')
                write(out / 'unload.json', {'receipt': unloaded, 'memory_before': before, 'released_used_mib': released})
                write(out / 'reload.json', api('load'))
                requests.get(URL + f'/v2/models/{MODEL}/ready', timeout=10).raise_for_status()
                (out / 'after_reload').mkdir()
                monitor = Monitor(out / 'after_reload', CONTAINER)
                monitor.start()
        tokens = sum(p['totals']['output_tokens'] for p in passes)
        seconds = sum(p['totals']['request_seconds'] for p in passes)
        if any(sha(p) != h for p, h in producer['source_hashes'].items()):
            raise ValueError('adapter source changed during validation')
        write(out / 'result.json', {'completed': True, 'passes': passes, 'tok_s': tokens / seconds,
            'output_tokens': tokens, 'request_seconds': seconds, 'model_ready': True,
            'unload_release_reload_verified': True, 'population_accuracy_established': False,
            'response_hashes': {str(p): sha(p) for p in sorted(out.glob('full_*/runs/**/response.json'))}})
        print('Triton full-request tok/s:', tokens / seconds, flush=True)
    finally:
        monitor.close()
        log = subprocess.run(['docker', 'logs', CONTAINER], capture_output=True, text=True)
        (out / 'triton.log').write_text(log.stdout + log.stderr)
        for p in (ROOT / 'runtime').glob('native-*.log'):
            (out / p.name).write_bytes(p.read_bytes())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    a = p.parse_args()
    run(a.output, a.reference)
