"""Isolated, nonpersistent gateway availability experiment; no semantic cases."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time

import requests

BASE = Path('data/experiments/ncm_paper_regime')
SERVICE = BASE/'20261028_isolated_judge_gateway_service_v2'
ROOT = BASE/'20261028_isolated_judge_gateway_probe_v2'
REPO = Path('/home/rabbit/llm_endpoint')
CLI = REPO/'.pixi/envs/default/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex'
UNIT = 'structure-net-ncm-isolated-judge-gateway-v2.service'
ORIGIN = 'http://127.0.0.1:8092'
MODELS = ('codex_sdk/gpt-5.6-terra', 'codex_sdk/gpt-5.6-sol')
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-isolated-judge-gateway-v2.md')
TEST = Path('tests/structure_net/test_ncm_isolated_judge_gateway_v2.py')
SETTINGS = {'LLM_ENDPOINT_CODEX_PERSISTENT': '0', 'LLM_ENDPOINT_CLAUDE_POOL': '0',
            'LLM_ENDPOINT_CLAUDE_POOL_PREWARM': '', 'LLM_ENDPOINT_MAX_WORKERS': '1',
            'LLM_ENDPOINT_SLOT_WAIT_S': '5'}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def verify(mapping):
    for p, h in mapping.items():
        if sha(p) != h: raise ValueError('gateway source or evidence changed: '+p)


def state():
    result = subprocess.run(['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'SubState',
        '-p', 'MainPID', '-p', 'ExecMainPID', '-p', 'InvocationID', '-p', 'NRestarts'],
        capture_output=True, text=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def validate_state(proof, observed):
    if (observed.get('ActiveState') != 'active' or observed.get('SubState') != 'running'
            or observed.get('NRestarts') != '0' or observed.get('InvocationID') != proof['invocation_id']
            or observed.get('MainPID') != str(proof['supervisor_PID'])
            or observed.get('ExecMainPID') != str(proof['supervisor_PID'])):
        raise ValueError('exact original isolated service invocation must be running')


def credential():
    proof = json.loads((SERVICE/'producer.json').read_text())
    verify(proof['source_hashes']); validate_state(proof, state())
    proc = Path('/proc')/str(proof['PID'])
    command = proc.joinpath('cmdline').read_bytes().decode().split('\0')
    if ('llm-endpoint' not in command or '--host' not in command or '--port' not in command
            or command[command.index('--host')+1] != '127.0.0.1'
            or command[command.index('--port')+1] != '8092'):
        raise ValueError('isolated gateway launch command differs')
    environment = dict(s.split('=', 1) for s in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in s)
    if any(environment.get(k) != v for k, v in SETTINGS.items()):
        raise ValueError('nonpersistent single-worker configuration changed')
    key = environment.get('LLM_ENDPOINT_API_KEY')
    if not key: raise ValueError('isolated service credential missing')
    return key


def serve():
    if SERVICE.exists() or not os.environ.get('INVOCATION_ID'):
        raise ValueError('fresh supervised service required')
    with socket.socket() as sock: sock.bind(('127.0.0.1', 8092))
    files = [Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(),
             Path('docs/07 - Status Reports/2026-10-28_ncm-isolated-judge-gateway.md').resolve(),
             REPO/'pixi.toml', REPO/'pixi.lock', REPO/'pyproject.toml', *sorted((REPO/'llm_endpoint').rglob('*.py'))]
    cli = str(CLI)
    if not CLI.is_file(): raise ValueError('existing authenticated completion CLI required')
    files.append(CLI.resolve())
    SERVICE.mkdir(); (SERVICE/'producer_source').mkdir()
    for i, p in enumerate(files):
        if p.suffix in ('.py', '.md', '.toml'): (SERVICE/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'ncm-isolated-judge-gateway-v2', 'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'PID': os.getpid(), 'supervisor_PID': int(state()['MainPID']),
        'invocation_id': os.environ['INVOCATION_ID'], 'origin': ORIGIN, 'unit': UNIT,
        'environment_settings': SETTINGS, 'source_hashes': {str(p): sha(p) for p in files},
        'credential_persisted': False, 'existing_services_modified': False, 'semantic_requests': 0}
    write(SERVICE/'producer.json', proof)
    environment = dict(os.environ, **SETTINGS, LLM_ENDPOINT_API_KEY=secrets.token_urlsafe(48),
                       SERVANT_CODEX_EXECUTABLE=cli)
    command = ['/home/rabbit/.pixi/bin/pixi', 'run', '--manifest-path', str(REPO/'pixi.toml'), '--locked',
               'llm-endpoint', '--host', '127.0.0.1', '--port', '8092']
    os.execvpe(command[0], command, environment)


def response_record(model, wire):
    if wire.get('model') not in (model, model.removeprefix('codex_sdk/')):
        raise ValueError('unexpected returned model')
    content = wire['choices'][0]['message']['content']
    return {'model': model, 'returned_model': wire['model'],
            'neutral_exact_OK': isinstance(content, str) and content.strip() == 'OK'}


def probe():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU/network-only neutral probe required')
    key = credential(); ROOT.mkdir()
    proof = {'origin': ORIGIN, 'models': MODELS, 'prompt': 'Reply with exactly OK.',
        'attempts_per_model': 1, 'backend_timeout_seconds': 30, 'client_timeout_seconds': 40,
        'max_tokens': 256, 'temperature': 0, 'semantic_cases_used': 0,
        'source_hashes': {str(p): sha(p) for p in (Path(__file__), PLAN, TEST)},
        'service_producer_sha256': sha(SERVICE/'producer.json')}
    write(ROOT/'producer.json', proof)
    client = requests.Session(); client.trust_env = False
    client.headers['Authorization'] = 'Bearer '+key
    health = client.get(ORIGIN+'/healthz', timeout=5); health.raise_for_status()
    write(ROOT/'health.json', health.json())
    results = []
    for i, model in enumerate(MODELS):
        payload = {'model': model, 'messages': [{'role': 'user', 'content': proof['prompt']}],
                   'temperature': 0, 'max_tokens': 256, 'timeout': 30}
        write(ROOT/f'request_{i}.json', payload); started = time.monotonic()
        try:
            response = client.post(ORIGIN+'/v1/chat/completions', json=payload, timeout=(5,40))
            write(ROOT/f'response_{i}.json', {'http_status': response.status_code,
                  'body': response.text.replace(key, '[REDACTED]')})
            response.raise_for_status(); row = response_record(model, response.json())
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as error:
            row = {'model': model, 'neutral_exact_OK': False, 'error_type': type(error).__name__}
        row['elapsed_seconds'] = time.monotonic()-started; results.append(row)
        print(json.dumps(row), flush=True)
    client.close(); credential(); verify(proof['source_hashes'])
    write(ROOT/'result.json', {'status': 'isolated_neutral_probe_completed', 'results': results,
        'new_generation_requests': 2, 'new_semantic_judge_calls': 0, 'judge_qualified': False,
        'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=('serve', 'probe'))
    args = parser.parse_args(); serve() if args.stage == 'serve' else probe()
