"""Matched control/candidate packed38 replay for sampling-output acquisition."""
import argparse
from pathlib import Path
import re
import signal
import time
import requests
from . import three_gpu_experts as base
from .common import PACKAGE, digest, read, sha, write
from .prepare_qwen38_moe import WEIGHTS, SHA256, SIZE
from .qwen_runtime import REQUESTS, REQUESTS_SHA, memory
from .check_qwen_tokenizers import gguf_metadata
from .qwen38_moe_runtime import validate_metadata
from experiments.neural_architecture_lab.run_word_games_packed_experts import packed_command, packed_verify

PLAN = Path('docs/07 - Status Reports/2026-10-31_word-games-sampling-output-once-plan.md')
HYPOTHESIS = 'word-games-sampling-output-once-v1'


def compare(left, right):
    a = {str(p.relative_to(left/'runs')): read(p) for p in (left/'runs').rglob('response.json')}
    b = {str(p.relative_to(right/'runs')): read(p) for p in (right/'runs').rglob('response.json')}
    mismatches = []
    for name in sorted(a.keys() | b.keys()):
        if name not in a or name not in b:
            mismatches.append({'request': name, 'field': 'missing'}); continue
        for key in ('tokens', 'content', 'stop_type', 'stopping_word', 'truncated', 'tokens_predicted'):
            if a[name]['response'][key] != b[name]['response'][key]:
                mismatches.append({'request': name, 'field': key})
        if a[name]['parsed'] != b[name]['parsed']:
            mismatches.append({'request': name, 'field': 'parsed'})
    return {'equal': bool(a) and not mismatches, 'requests': len(a), 'mismatches': mismatches}


def cell(out, arm, phase, producer, metadata, args):
    original_command, original_verify = base.command, base.verify
    original_hypothesis = base.HYPOTHESIS_ID
    def command(extra, port, name):
        cmd = packed_command(extra, port, name)
        cmd[cmd.index(f'{base.BINARY.resolve()}:/llama:ro')] = f'{(args.root/arm).resolve()}:/llama:ro'
        pos = cmd.index(base.IMAGE)
        cmd[pos:pos] = ['-v', f'{(args.root/"count-sync.so").resolve()}:/count-sync.so:ro',
                        '-e', 'LD_PRELOAD=/count-sync.so']
        return cmd
    base.command, base.verify = command, packed_verify
    base.HYPOTHESIS_ID = HYPOTHESIS
    try:
        result = base.run_cell(out, 10, phase, producer, metadata, args.port)
    finally:
        base.command, base.verify = original_command, original_verify
        base.HYPOTHESIS_ID = original_hypothesis
    counts = [int(n) for n in re.findall(r'WORD_GAMES_SCHED_SYNCHRONIZE_CALLS=(\d+)', (out/'server.log').read_text())]
    write(out/'sync_counts.json', {'process_counts': counts, 'total': sum(counts),
                                  'scope': 'entire process including load, warmup, requests, shutdown'})
    if not counts or not sum(counts): raise ValueError('synchronization instrumentation inactive')
    return {'arm': arm, **result, 'scheduler_sync_calls': sum(counts)}


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if not read(args.native_tests/'result.json')['passed']: raise ValueError('native checks required')
    build = read(args.build_manifest)
    for arm in ('control', 'candidate'):
        if any(sha(args.root/arm/name) != h for name, h in build['runtimes'][arm]['hashes'].items()):
            raise ValueError('runtime identity mismatch')
    if sha(args.root/'count-sync.so') != read(args.native_tests/'result.json')['counter_library_sha256']:
        raise ValueError('counter identity mismatch')
    print('Checking frozen model and request identities before service pause', flush=True)
    if WEIGHTS.stat().st_size != SIZE or sha(WEIGHTS) != SHA256 or sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('model/request identity mismatch')
    metadata = gguf_metadata(WEIGHTS)
    validate_metadata(metadata)
    sources = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources += [PLAN, Path('experiments/neural_architecture_lab/run_word_games_packed_experts.py'),
                Path('pyproject.toml'), Path('pixi.lock')]
    bindings = {str(p): sha(p) for p in sources}
    producer = {'hypothesis_id': HYPOTHESIS, 'source_hashes': bindings,
                'weights_sha256': SHA256, 'requests_sha256': REQUESTS_SHA,
                'build_manifest_sha256': sha(args.build_manifest),
                'native_tests_sha256': sha(args.native_tests/'result.json'),
                'counter_sha256': sha(args.root/'count-sync.so'), 'order': ['control', 'candidate']}
    producer['fingerprint'] = digest(producer)
    write(out/'producer.json', producer)
    for i, p in enumerate(sources):
        dest = out/'producer_source'/f'{i}_{p.name}'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(p.read_bytes())
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    url = base.TRITON
    service = base.SERVICE
    before = requests.get(url+f'/v2/models/{service}/config', timeout=10)
    before.raise_for_status()
    requests.get('http://127.0.0.1:8010/v2/health/ready', timeout=5).raise_for_status()
    write(out/'service_before.json', {'config': before.json(), 'gpu_memory': memory()})
    results, comparisons = [], {}
    attempted = restored = False
    error = None
    try:
        attempted = True
        response = requests.post(url+f'/v2/repository/models/{service}/unload', json={}, timeout=120)
        write(out/'unload.json', {'status': response.status_code, 'body': response.text})
        response.raise_for_status()
        base.wait_for_release(out)
        for phase in ('pilot', 'full'):
            for arm in ('control', 'candidate'):
                result = cell(out/f'{phase}_{arm}', arm, phase, producer, metadata, args)
                results.append(result)
                if not result['completed']: raise RuntimeError(result['error'])
            parity = compare(out/f'{phase}_control', out/f'{phase}_candidate')
            comparisons[phase] = parity
            write(out/f'{phase}_parity.json', parity)
            expected = len(base.PILOT) if phase == 'pilot' else len(read(REQUESTS))
            if not parity['equal'] or parity['requests'] != expected:
                raise ValueError(f'{phase} output mismatch or missing requests')
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            if attempted:
                response = requests.post(url+f'/v2/repository/models/{service}/load', json={}, timeout=240)
                write(out/'reload.json', {'status': response.status_code, 'body': response.text})
                response.raise_for_status()
                deadline = time.monotonic()+180
                while time.monotonic() < deadline:
                    if requests.get(url+f'/v2/models/{service}/ready', timeout=5).status_code == 200:
                        restored = True
                        break
                    time.sleep(1)
                after = requests.get(url+f'/v2/models/{service}/config', timeout=10)
                after.raise_for_status()
                restored = restored and after.json() == before.json()
                ocr_ready = requests.get('http://127.0.0.1:8010/v2/health/ready', timeout=5).status_code == 200
                write(out/'service_restored.json', {'ready_and_config_equal': restored,
                                                   'ocr_ready': ocr_ready, 'gpu_memory': memory()})
                if not ocr_ready: raise RuntimeError('OCR service not ready after replay')
                print('Qwen3-8B restored: '+str(restored), flush=True)
        finally:
            write(out/'campaign_results.json', {'hypothesis_id': HYPOTHESIS, 'results': results,
                  'parity': comparisons, 'error': error, 'service_restored': restored,
                  'confirmed': False, 'classification': 'exploratory_fixed_suite_systems_regression'})
    if not restored: raise RuntimeError('service restoration failed')
    if any(sha(p) != h for p, h in bindings.items()): raise ValueError('producer changed during replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--build-manifest', type=Path, required=True)
    parser.add_argument('--native-tests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18088)
    run(parser.parse_args())
