"""Compare CPU expert removal, reduced CPU threads and selective compression."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import requests

from . import three_gpu_experts as base
from .common import PACKAGE, digest, read, sha, write
from .prepare_expert_compression import PLAN, SELECTED
from .prepare_qwen38_moe import WEIGHTS, SHA256
from .check_qwen_tokenizers import gguf_metadata
from .qwen38_moe_runtime import NativeQwen38, validate_metadata
from .qwen_runtime import REQUESTS, REQUESTS_SHA, memory
from .metaphor import Metaphor

HYPOTHESIS = 'word-games-cpu-expert-elimination-v1'
SCHEMA = 'word-games.cpu-expert-elimination.v1'
RUNTIME = Path('.pixi/word-games-sampling-v2/candidate')
BUILD = Path('data/experiments/word_games/20261031_sampling_output_build_v4/result.json')
ORIGINAL_COMMAND = base.command


@dataclass(frozen=True)
class Arm:
    name: str
    threads: int
    compressed: bool = False

    @property
    def groups(self):
        result = {'CUDA0': list(range(18, 23)), 'CUDA1': list(range(8)) + [23],
                  'CUDA2': list(range(8, 18)), 'CPU': []}
        if self.name == 'packed38':
            result['CPU'] = [24, 25]
        elif self.compressed:
            result['CUDA0'] += [24, 25]
        else:
            result['CUDA0'] += [25]
            result['CUDA2'] += [24]
        return result


ARMS = {a.name: a for a in (Arm('packed38', 12), Arm('gpu40', 12),
                           Arm('gpu40_cpu1', 1), Arm('q2_3060', 1, True))}


def command(arm, compressed, port, name):
    cmd = ORIGINAL_COMMAND(10, port, name)
    cmd[cmd.index(f'{base.BINARY.resolve()}:/llama:ro')] = f'{RUNTIME.resolve()}:/llama:ro'
    if arm.compressed:
        cmd[cmd.index(f'{WEIGHTS.resolve()}:/model.gguf:ro')] = f'{compressed.resolve()}:/model.gguf:ro'
    for flag in ('--threads', '--threads-batch'):
        cmd[cmd.index(flag) + 1] = str(arm.threads)
    cmd[cmd.index('--override-tensor') + 1] = ','.join(
        '^blk\\.(' + '|'.join(map(str, layers)) + ')\\.ffn_.*_exps\\.weight$=' + device
        for device, layers in arm.groups.items() if layers)
    return cmd


def verify(log, arm):
    layers = {int(i): dev for i, dev in re.findall(r'layer\s+(\d+) assigned to device ([^,\s]+)', log)}
    if layers != {i: 'CUDA0' for i in range(42)}:
        raise ValueError('attention/layer placement changed')
    found = {}
    for i, part, device in re.findall(
        r'tensor blk\.(\d+)\.ffn_(\w+)_exps\.weight \([^)]*\) buffer type overridden to (\S+)', log
    ):
        i = int(i)
        actual = 'CPU' if device.startswith('CPU') or device == 'CUDA_Host' else device
        expected = next((d for d, ids in arm.groups.items() if i in ids), None)
        if actual != expected or (i, part) in found:
            raise ValueError('expert projection misplacement or duplicate')
        found[i, part] = actual
    if set(found) != {(i, part) for i in range(26) for part in ('up', 'gate', 'down')}:
        raise ValueError('incomplete placement')
    return {'verified': True, 'groups': arm.groups, 'cpu_banks': len(arm.groups['CPU']),
            'gpu_banks': 40 - len(arm.groups['CPU']), 'threads': arm.threads}


def process_cpu(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    status = Path(f'/proc/{pid}/status').read_text()
    return {'cpu_seconds': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
            'threads': int(fields[17]),
            'rss_kib': int(re.search(r'^VmRSS:\s+(\d+) kB', status, re.M).group(1))}


def cell(out, arm, phase, producer, metadata, compressed, port):
    original = base.command, base.verify, base.NativeQwen38, base.HYPOTHESIS_ID
    container = {}

    def make_command(extra, selected_port, name):
        container['name'] = name
        return command(arm, compressed, selected_port, name)

    class MeasuredClient(NativeQwen38):
        pid = None

        def complete(self, prompt, path, *, max_tokens=1024):
            if self.pid is None:
                self.pid = int(subprocess.check_output(
                    ['docker', 'inspect', '--format', '{{.State.Pid}}', container['name']], text=True))
            before = process_cpu(self.pid)
            text = super().complete(prompt, path, max_tokens=max_tokens)
            after = process_cpu(self.pid)
            write(path / 'cpu.json', {'before': before, 'after': after,
                                     'cpu_seconds': after['cpu_seconds'] - before['cpu_seconds']})
            return text

    base.command, base.verify, base.NativeQwen38 = make_command, lambda log, _: verify(log, arm), MeasuredClient
    base.HYPOTHESIS_ID = HYPOTHESIS
    try:
        result = base.run_cell(out, 10, phase, producer, metadata, port)
    finally:
        base.command, base.verify, base.NativeQwen38, base.HYPOTHESIS_ID = original
    cpu = [read(p) for p in (out / 'runs').rglob('cpu.json')]
    extra = {'arm': arm.name, 'cpu_seconds': sum(r['cpu_seconds'] for r in cpu),
             'cpu_measured_calls': len(cpu), 'cpu_scope': 'server process, including template handling',
             'max_observed_threads': max((r['after']['threads'] for r in cpu), default=0),
             'max_observed_rss_kib': max((r['after']['rss_kib'] for r in cpu), default=0)}
    write(out / 'cpu_totals.json', extra)
    return {**result, **extra}


def receiver_accuracy(root):
    inputs = {r['id']: r for r in read(REQUESTS)}
    cases = []
    for p in sorted((root / 'runs/cases').glob('*/receiver/response.json')):
        name = str(p.parent.relative_to(root / 'runs'))
        packet = json.loads(inputs[name]['prompt'].split('AUTHORIZED PACKET\n', 1)[1])
        expected = sorted(({'slot': e['placement']['slot'], 'entity_id': e['entity_id'],
                            'return_by': e['placement']['return_by']}
                           for e in packet['entities'] if e.get('placement')),
                          key=lambda e: (e['slot'], e['entity_id']))
        answer = read(p)['parsed']
        correct = answer.get('decision') == packet['decision'] and answer.get('occupied_slots') == expected
        cases.append({'id': name, 'correct': correct, 'expected_decision': packet['decision'],
                      'expected_occupied_slots': expected, 'response_sha256': sha(p)})
    selector = root / 'runs/metaphor/response.json'
    try:
        Metaphor(**read(selector)['parsed'])
        valid = True
    except (TypeError, ValueError):
        valid = False
    return {'receiver_correct': sum(c['correct'] for c in cases), 'receiver_cases': len(cases),
            'cases': cases, 'selector_valid': valid, 'semantic_summary_judge': 'not yet reviewed',
            'scope': 'fixed authorized packets, not end-to-end live bridge accuracy'}


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    prep = read(args.preparation / 'result.json')
    compressed = Path(prep['weights'])
    if (not prep['completed'] or prep['lossless'] or set(prep['changed_tensors']) != SELECTED
            or prep['original_sha256'] != SHA256):
        raise ValueError('verified selective compression required')
    print('Checking original/derived checkpoints before service pause', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        original_sha, compressed_sha = pool.map(sha, (WEIGHTS, compressed))
    if original_sha != SHA256 or compressed_sha != prep['weights_sha256'] or sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('weight or request identity mismatch')
    for name, h in read(BUILD)['runtimes']['candidate']['hashes'].items():
        if sha(RUNTIME / name) != h:
            raise ValueError('runtime identity mismatch')
    metadata = gguf_metadata(WEIGHTS)
    validate_metadata(metadata)
    sources = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [PLAN, Path('pyproject.toml'), Path('pixi.lock')]
    bindings = {str(p): sha(p) for p in sources}
    producer = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, 'source_hashes': bindings,
                'original_sha256': SHA256, 'compressed_sha256': prep['weights_sha256'],
                'preparation_sha256': sha(args.preparation / 'result.json'),
                'runtime_manifest_sha256': sha(BUILD), 'requests_sha256': REQUESTS_SHA,
                'arms': {k: {'threads': a.threads, 'compressed': a.compressed, 'groups': a.groups} for k, a in ARMS.items()},
                'full_order': [list(ARMS), list(reversed(ARMS))]}
    producer['fingerprint'] = digest(producer)
    write(out / 'producer.json', producer)
    for i, p in enumerate(sources):
        dest = out / 'producer_source' / f'{i}_{p.name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(p.read_bytes())
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    url, service = base.TRITON, base.SERVICE
    before = requests.get(url + f'/v2/models/{service}/config', timeout=10)
    before.raise_for_status()
    requests.get('http://127.0.0.1:8010/v2/health/ready', timeout=5).raise_for_status()
    write(out / 'service_before.json', {'config': before.json(), 'gpu_memory': memory()})
    results, advanced = [], []
    attempted = restored = False
    error = None
    try:
        attempted = True
        response = requests.post(url + f'/v2/repository/models/{service}/unload', json={}, timeout=120)
        write(out / 'unload.json', {'status': response.status_code, 'body': response.text})
        response.raise_for_status()
        base.wait_for_release(out)
        for name, arm in ARMS.items():
            r = cell(out / ('pilot_' + name), arm, 'pilot', producer, metadata, compressed, args.port)
            results.append({'repeat': 0, **r})
            if not r['completed'] and 'readiness' in str(r['error']):
                raise RuntimeError('resident service health failure')
            if name == 'packed38':
                if not r['completed']:
                    raise RuntimeError('baseline pilot failed')
                baseline_rate = r['totals']['output_tokens_per_second']
            passed = r['completed'] and r['totals']['output_tokens_per_second'] >= .9 * baseline_rate
            write(out / (name + '_gate.json'), {'advance': passed, 'relative_rate':
                r['totals']['output_tokens_per_second'] / baseline_rate if r['completed'] else None})
            if passed:
                advanced.append(name)
        for repeat, order in enumerate((advanced, list(reversed(advanced))), 1):
            for name in order:
                path = out / f'full_{repeat}_{name}'
                r = cell(path, ARMS[name], 'full', producer, metadata, compressed, args.port)
                results.append({'repeat': repeat, **r})
                if not r['completed']:
                    raise RuntimeError(f'full replay failed: {name}')
                quality = receiver_accuracy(path)
                if quality['receiver_cases'] != 12:
                    raise ValueError('twelve receiver checks required')
                write(path / 'quality.json', quality)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            if attempted:
                response = requests.post(url + f'/v2/repository/models/{service}/load', json={}, timeout=240)
                write(out / 'reload.json', {'status': response.status_code, 'body': response.text})
                response.raise_for_status()
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if requests.get(url + f'/v2/models/{service}/ready', timeout=5).status_code == 200:
                        restored = True
                        break
                    time.sleep(1)
                after = requests.get(url + f'/v2/models/{service}/config', timeout=10)
                after.raise_for_status()
                restored = restored and after.json() == before.json()
                ocr_ready = requests.get('http://127.0.0.1:8010/v2/health/ready', timeout=5).status_code == 200
                write(out / 'service_restored.json', {'ready_and_config_equal': restored,
                    'ocr_ready': ocr_ready, 'gpu_memory': memory()})
                if not ocr_ready:
                    raise RuntimeError('OCR service not ready')
                print('Qwen3-8B restored:', restored, flush=True)
        finally:
            write(out / 'campaign_results.json', {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS,
                'fingerprint': producer['fingerprint'], 'results': results, 'advanced': advanced,
                'error': error, 'service_restored': restored, 'confirmed': False,
                'classification': 'exploratory_systems_comparison'})
    if not restored:
        raise RuntimeError('service restoration failed')
    if any(sha(p) != h for p, h in bindings.items()):
        raise ValueError('producer changed during campaign')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18088)
    run(parser.parse_args())
