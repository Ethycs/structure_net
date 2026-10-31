"""Active Qwen 14B / 3B native runtime and bounded WORD_GAMES evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import requests

from .common import PACKAGE, digest, read, sha, write
from .native_speed import BINARY, GPU, IMAGE, PILOT, NativePhi4, aggregate, output_tokens
from .qwen_pair import HYPOTHESIS, MODEL, PAIRS, PLAN, directory

DRAFT_GPU = 'GPU-7b1db3cb-ec9c-beb3-e7db-826c2543edf9'
REQUESTS = Path('data/experiments/word_games/20261031_speculative_inputs_v1/requests.json')
REQUESTS_SHA = 'f9830c9480f8a4694630a06102cfb33e9e16517faec22e4f9f59888137fd53cb'
CHECK = Path('data/experiments/word_games/20261031_qwen_tokenizers_v1/build/campaign_results.json')


class NativeQwen(NativePhi4):
    def format_prompt(self, prompt):
        # Let the pinned server apply the GGUF's own template, including its
        # default Qwen system message. Phi's im_sep marker is never inserted.
        r = self.http.post(self.url + '/apply-template', json={
            'messages': [{'role': 'user', 'content': prompt}], 'add_generation_prompt': True}, timeout=30)
        r.raise_for_status()
        value = r.json()['prompt']
        expected = ('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
                    '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n')
        if value != expected:
            raise ValueError('unexpected Qwen chat template')
        return value


def server_command(args, name):
    gpus = GPU + ',' + DRAFT_GPU if args.draft else GPU
    cmd = ['docker', 'run', '--rm', '--name', name, '--gpus', '"device=' + gpus + '"',
           '--user', f'{os.getuid()}:{os.getgid()}', '-p', f'127.0.0.1:{args.port}:8080',
           '-v', f'{BINARY.resolve()}:/llama:ro', '-v', '/usr/local/cuda/lib64:/cuda:ro',
           '-v', f'{directory("target").resolve()}:/target:ro',
           '-e', 'CUDA_VISIBLE_DEVICES=' + gpus, '-e', 'CUDA_DEVICE_ORDER=PCI_BUS_ID']
    if args.draft:
        cmd += ['-v', f'{directory("draft").resolve()}:/draft:ro']
    cmd += [IMAGE, '-m', '/target/' + PAIRS['target']['files'][0][0], '--alias', MODEL,
            '--host', '0.0.0.0', '--port', '8080', '--ctx-size', '8192', '--parallel', '1',
            '--fit', 'off', '--device', 'CUDA0', '--split-mode', 'none', '--main-gpu', '0',
            '--n-gpu-layers', '99', '--flash-attn', 'on', '--cache-type-k', 'q8_0',
            '--cache-type-v', 'q8_0', '--batch-size', '512', '--ubatch-size', '512',
            '--threads', '4', '--cache-ram', '0', '--no-cache-prompt', '--jinja',
            '--spec-type', 'draft-simple' if args.draft else 'none']
    if args.draft:
        cmd += ['--spec-draft-model', '/draft/' + PAIRS['draft']['files'][0][0],
                '--spec-draft-device', 'CUDA1', '--spec-draft-ngl', '99',
                '--spec-draft-n-max', str(args.draft), '--spec-draft-threads', '4',
                '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0']
    return cmd


def memory():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.free', '--format=csv'], text=True)


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if sha(REQUESTS) != REQUESTS_SHA:
        raise ValueError('frozen request corpus changed')
    check = read(args.tokenizer_check)
    if not check['compatible'] or check['classification'] != 'shared_token_ids_verified' or check['weights'] != json.loads(json.dumps(PAIRS)):
        raise ValueError('verified exact Qwen tokenizer pair required')
    print('Verifying pinned Qwen checkpoint hashes', flush=True)
    for role, config in PAIRS.items():
        for filename, size, expected in config['files']:
            p = directory(role) / filename
            if p.stat().st_size != size or sha(p) != expected:
                raise ValueError('checkpoint changed: ' + str(p))
    sources = [p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    sources += [PLAN, Path('pyproject.toml'), Path('pixi.lock')]
    bindings = {str(p): sha(p) for p in sources}
    fixture_hashes = {}
    if args.paired_fixture_root:
        if args.stage != 'live':
            raise ValueError('paired fixtures require live stage')
        fixture_hashes = {str(p): sha(p) for p in sorted((args.paired_fixture_root / 'quality/cases').glob('*/context.json'))}
        if len(fixture_hashes) != 12:
            raise ValueError('twelve paired fixture contexts required')
    name = f'word-games-qwen-{os.getpid()}'
    cmd = server_command(args, name)
    config = {'hypothesis_id': HYPOTHESIS, 'model': MODEL, 'models': PAIRS, 'stage': args.stage,
              'draft_length': args.draft, 'command': cmd, 'requests_sha256': REQUESTS_SHA,
              'tokenizer_check_sha256': sha(args.tokenizer_check), 'source_hashes': bindings,
              'paired_fixture_hashes': fixture_hashes, 'cross_request_cache': False,
              'binary_hashes': {p.name: sha(p) for p in BINARY.glob('*') if p.is_file() and not p.is_symlink()},
              'gpu_before': memory()}
    write(out / 'producer.json', {**config, 'fingerprint': digest(config)})
    for i,p in enumerate(sources):
        target = out / 'producer_source' / f'{i}_{p.name}'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    client, process = NativeQwen(args.port), None
    start = time.perf_counter()
    try:
        with (out / 'server.log').open('x') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Qwen server exited; see server.log')
                try:
                    if client.http.get(client.url + '/health', timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Qwen server startup')
                time.sleep(.5)
            props = client.http.get(client.url + '/props', timeout=10).json()
            write(out / 'runtime.json', {'load_seconds': time.perf_counter()-start,
                  'properties': props, 'gpu_loaded': memory()})
            # b11960 trims the template file's final newline on load. Compare
            # its source modulo that normalization, and check rendered bytes
            # exactly in format_prompt on every call.
            if props.get('chat_template', '').rstrip('\n') != check['chat_template'].rstrip('\n'):
                raise ValueError('runtime chat template differs from checked GGUF')
            client.complete('Return exactly {"ready":true} as JSON.', out / 'warmup', max_tokens=24)
            if client.rows[-1]['parsed'] != {'ready': True}:
                raise ValueError('Qwen JSON lifecycle failed')
            client.rows.clear()
            identity = {'model': MODEL, 'ready': True, 'weights': PAIRS, 'draft_length': args.draft,
                        'native_fingerprint': digest(config), 'backend': 'llama.cpp b11960 native'}
            write(out / 'identity.json', identity)
            if args.stage == 'live':
                from .native_quality import run_quality
                quality = run_quality(out / 'quality', out / 'identity.json', client, args.paired_fixture_root,
                                      expected_model=MODEL, hypothesis_id=HYPOTHESIS, plan=PLAN)
                records = [read(p) for p in (out / 'quality/cases').glob('*/result.json')]
                write(out / 'quality_summary.json', {'metrics': quality['metrics'],
                      'joint_case_success': sum(all(r[k] for k in ('sender_operation_match','boundary_outcome_match','receiver_state_correct')) for r in records),
                      'cases': len(records), 'semantic_review_completed': False})
            else:
                inputs = read(REQUESTS)
                chosen = [next(r for r in inputs if r['id'] == name) for name in PILOT] if args.stage == 'pilot' else inputs
                for r in chosen:
                    client.complete(r['prompt'], out / 'runs' / r['id'], max_tokens=r['max_tokens'])
                    if not isinstance(client.rows[-1]['parsed'], dict) or client.rows[-1]['response'].get('stop_type') == 'limit':
                        raise ValueError('invalid or truncated JSON in pilot/replay')
                    print(json.dumps({'id': r['id'], 'seconds': client.rows[-1]['elapsed_seconds'],
                                      'tokens': output_tokens(client.rows[-1]['response'])}), flush=True)
            if any(sha(p) != h for p,h in {**bindings, **fixture_hashes}.items()):
                raise ValueError('producing source or fixtures changed')
            result = {'hypothesis_id': HYPOTHESIS, 'model': MODEL, 'stage': args.stage, 'confirmed': False,
                      'classification': 'systems_lifecycle_only_not_quality_evidence' if args.stage == 'pilot' else 'completed_exploratory',
                      'fingerprint': digest(config), 'totals': aggregate(client.rows),
                      'artifacts': {str(p): sha(p) for p in sorted(out.rglob('response.json'))}}
            write(out / 'campaign_results.json', result)
            print(json.dumps(result['totals']), flush=True)
    except Exception as exc:
        write(out / 'interruption.json', {'error': repr(exc), 'completed_calls': len(client.rows)})
        raise
    finally:
        if process:
            subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True)
            process.wait(timeout=30)
        write(out / 'gpu_after.json', {'memory': memory()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['pilot','full','live'], default='live')
    p.add_argument('--draft', type=int, choices=[0,3], default=0)
    p.add_argument('--paired-fixture-root', type=Path)
    p.add_argument('--tokenizer-check', type=Path, default=CHECK)
    p.add_argument('--port', type=int, default=18085)
    run(p.parse_args())


if __name__ == '__main__':
    main()
