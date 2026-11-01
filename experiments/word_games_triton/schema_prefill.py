"""Public prefix snapshot pilot and matched Triton throughput replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import requests

from experiments.word_games.agents import SENDER_CONTRACT, RECEIVER_CONTRACT
from experiments.word_games.common import read, sha, write
from experiments.word_games.cpu_expert_elimination import receiver_accuracy
from experiments.word_games.native_speed import aggregate, output_tokens
from experiments.word_games.qwen_runtime import REQUESTS, REQUESTS_SHA
from experiments.word_games import three_gpu_experts as telemetry
from . import deploy
from .validate import Client

HYPOTHESIS_ID = 'word-games-schema-prefill-v1'
PLAN = Path('docs/07 - Status Reports/2026-11-01_word-games-schema-prefill-plan.md')
REFERENCE = Path('data/experiments/word_games/20261031_triton_native_validation_v2/full_1')
PREFIXES = [{'name': 'sender', 'text': SENDER_CONTRACT + '\nCURRENT INPUT\n'},
            {'name': 'receiver', 'text': RECEIVER_CONTRACT + '\nAUTHORIZED PACKET\n'}]


def replace_json(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def configure(cached, original, out):
    write(out / 'unload.json', deploy.api('unload'))
    modeldir = deploy.ROOT / 'models' / deploy.MODEL / '1'
    shutil.copyfile(deploy.SOURCE / 'model.py', modeldir / 'model.py')
    runtime = read(original / 'models' / deploy.MODEL / '1/runtime.json')
    runtime['public_prefixes'] = PREFIXES if cached else []
    replace_json(modeldir / 'runtime.json', runtime)
    manifest = read(original / 'manifest.json')
    manifest['public_prefixes_enabled'] = cached
    manifest['repository_hashes'] = {str(p.relative_to(deploy.ROOT)): sha(p)
        for p in sorted((deploy.ROOT / 'models').rglob('*')) if p.is_file()}
    replace_json(deploy.ROOT / 'manifest.json', manifest)
    write(out / 'manifest.json', manifest)
    write(out / 'load.json', deploy.api('load'))
    requests.get(deploy.URL + f'/v2/models/{deploy.MODEL}/ready', timeout=10).raise_for_status()
    receipts = []
    for path in (deploy.ROOT / 'runtime').glob('public-prefixes-*/receipt.json'):
        receipt = read(path)
        receipt['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                            for p in path.parent.iterdir() if p.name != 'receipt.json'}
        receipts.append(receipt)
    if cached and len(receipts) != 1:
        raise ValueError('exactly one fresh prefix snapshot directory required')
    write(out / 'prefix_build.json', receipts)


def validate_row(item, row, cached):
    expected = next((p['name'] for p in PREFIXES if item['prompt'].startswith(p['text'])), None)
    info = row['response']['public_prefix']
    if info['hit'] != bool(cached and expected):
        raise ValueError('unexpected prefix hit/miss')
    if info['hit']:
        if info['name'] != expected or row['response']['timings']['cache_n'] != info['tokens']:
            raise ValueError('native did not reuse exactly the public prefix')


def run(out, pilot=False):
    out.mkdir(parents=True, exist_ok=False)
    if sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('frozen inputs changed')
    manifest = read(deploy.ROOT / 'manifest.json')
    if deploy.weight_identity() != manifest['verified_weight_identity']:
        raise ValueError('weight identity changed; fresh full hash required')
    for p, h in manifest['repository_hashes'].items():
        if sha(deploy.ROOT / p) != h:
            raise ValueError('prepared repository fingerprint changed')
    for p, h in manifest['native_runtime_hashes'].items():
        if sha(deploy.target.cpu.RUNTIME / p) != h:
            raise ValueError('native binary changed')
    sources = [*Path('experiments/word_games_triton').glob('*.py'),
               *PLAN.parent.glob('2026-11-01_word-games-schema-prefill-*.md'), Path('experiments/word_games/agents.py'),
               Path('experiments/word_games/common.py'), Path('experiments/word_games/native_speed.py'),
               Path('experiments/word_games/cpu_expert_elimination.py'), Path('pixi.lock')]
    bindings = {str(p): sha(p) for p in sources}
    write(out / 'producer.json', {'hypothesis_id': HYPOTHESIS_ID, 'source_hashes': bindings,
          'requests_sha256': REQUESTS_SHA, 'manifest': manifest, 'pilot': pilot,
          'order': ['cached'] if pilot else ['control', 'cached', 'cached', 'control']})
    for p in sources:
        dest = out / 'producer_source' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    original = out / 'original_deployment'
    shutil.copytree(deploy.ROOT / 'models', original / 'models')
    shutil.copyfile(deploy.ROOT / 'manifest.json', original / 'manifest.json')
    inputs = read(REQUESTS)
    if pilot:
        sender = next(i for i in inputs if i['prompt'].startswith(PREFIXES[0]['text']))
        receiver = next(i for i in inputs if i['prompt'].startswith(PREFIXES[1]['text']))
        other = {'prompt': 'Return exactly {"unrelated":true} as JSON.', 'max_tokens': 24}
        inputs = [{**item, 'id': f'pilot_{n}'} for n, item in
                  enumerate([sender, receiver, other, sender, receiver])]
    passes, error = [], None
    try:
        for index, cached in enumerate([True] if pilot else [False, True, True, False], 1):
            root = out / f'{index}_{"cached" if cached else "control"}'
            root.mkdir()
            configure(cached, original, root)
            telemetry.TRITON = deploy.URL
            monitor = telemetry.Monitor(root, deploy.CONTAINER)
            monitor.start()
            client = Client()
            try:
                client.complete('Return exactly {"ready":true} as JSON.', root / 'warmup', 24)
                client.rows.clear()
                for item in inputs:
                    if monitor.error:
                        raise RuntimeError(monitor.error)
                    client.complete(item['prompt'], root / 'runs' / item['id'], item['max_tokens'])
                    row = client.rows[-1]
                    validate_row(item, row, cached)
                    print(json.dumps({'pass': index, 'cached': cached, 'id': item['id'],
                        'seconds': row['elapsed_seconds'], 'tokens': output_tokens(row['response']),
                        'cache': row['response']['public_prefix']}), flush=True)
                if monitor.error:
                    raise RuntimeError(monitor.error)
            finally:
                monitor.close()
                log = max((deploy.ROOT / 'runtime').glob('native-*.log'), key=lambda p: p.stat().st_mtime_ns)
                shutil.copyfile(log, root / 'native.log')
            result = {'cached': cached, 'totals': aggregate(client.rows),
                      'prefill_seconds': sum(r['response']['timings']['prompt_ms'] for r in client.rows) / 1000,
                      'restore_seconds': sum(r['response']['public_prefix'].get('restore_seconds', 0) for r in client.rows),
                      'cache_hits': sum(r['response']['public_prefix']['hit'] for r in client.rows),
                      'cache_tokens': sum(r['response']['public_prefix'].get('tokens', 0) for r in client.rows)}
            if not pilot:
                result['quality'] = receiver_accuracy(root)
                result['different_from_prior'] = [i['id'] for i in inputs if
                    read(root / 'runs' / i['id'] / 'response.json')['parsed'] !=
                    read(REFERENCE / 'runs' / i['id'] / 'response.json')['parsed']]
            write(root / 'result.json', result)
            passes.append({'root': str(root), **result})
        if any(sha(p) != h for p, h in bindings.items()):
            raise ValueError('producer source changed during measurement')
    except BaseException as exc:
        error = repr(exc)
        write(out / 'failure.json', {'error': error, 'http_body':
              exc.response.text if isinstance(exc, requests.HTTPError) and exc.response is not None else None})
        logs = subprocess.run(['docker', 'logs', '--tail', '100', deploy.CONTAINER], capture_output=True, text=True)
        (out / 'failure_triton.log').write_text(logs.stdout + logs.stderr)
        raise
    finally:
        # Always leave the known, previously validated serving state in place.
        running = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Running}}', deploy.CONTAINER], text=True).strip()
        if running != 'true':
            subprocess.run(['docker', 'start', deploy.CONTAINER], check=True, capture_output=True)
            time.sleep(3)
        deploy.api('unload')
        for p in (original / 'models').rglob('*'):
            if p.is_file():
                shutil.copyfile(p, deploy.ROOT / p.relative_to(original))
        shutil.copyfile(original / 'manifest.json', deploy.ROOT / 'manifest.json')
        restored = deploy.api('load')
        requests.get(deploy.URL + f'/v2/models/{deploy.MODEL}/ready', timeout=10).raise_for_status()
        write(out / 'restore.json', {'receipt': restored, 'original_repository_restored': True})
        write(out / 'result.json', {'hypothesis_id': HYPOTHESIS_ID, 'completed': error is None,
            'error': error, 'pilot': pilot, 'passes': passes, 'original_repository_restored': True,
            'response_hashes': {str(p): sha(p) for p in sorted(out.glob('*/runs/**/response.json'))}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    run(a.output, a.pilot)
