"""Neutral availability checks of the already running local completion gateway."""
import datetime
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path('data/experiments/ncm_paper_regime/20261028_local_judge_transport_probe_v1')
SERVICE_PID = 1050599
ORIGIN = 'http://127.0.0.1:8091'
MODELS = ('codex_sdk/gpt-5.6-terra', 'codex_sdk/gpt-5.6-sol')


def write(path, value):
    with path.open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def run():
    if ROOT.exists(): raise ValueError('fresh neutral availability probe required')
    process = Path('/proc')/str(SERVICE_PID)
    command = process.joinpath('cmdline').read_bytes().decode().split('\0')
    if ('llm-endpoint' not in command or '--host' not in command or '--port' not in command
            or command[command.index('--host')+1] != '127.0.0.1'
            or command[command.index('--port')+1] != '8091'):
        raise ValueError('exact already-running loopback gateway required')
    # Use only this gateway's configured credential; never serialize its value.
    environment = dict(s.split('=', 1) for s in process.joinpath('environ').read_bytes().decode().split('\0') if '=' in s)
    key = environment.get('LLM_ENDPOINT_API_KEY')
    if not key: raise ValueError('local service credential unavailable')
    ROOT.mkdir()
    proof = {'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': sha(__file__), 'models': MODELS, 'service_PID': SERVICE_PID,
        'origin': ORIGIN, 'prompt': 'Reply with exactly OK.', 'max_tokens': 256,
        'temperature': 0, 'timeout_seconds': [5, 30], 'attempts_per_model': 1,
        'semantic_cases_used': 0, 'judge_qualified': False, 'credential_persisted': False}
    write(ROOT/'producer.json', proof)
    client = requests.Session(); client.trust_env = False
    client.headers['Authorization'] = 'Bearer '+key
    health = client.get(ORIGIN+'/healthz', timeout=(5, 5)); health.raise_for_status()
    write(ROOT/'health.json', health.json())
    models = client.get(ORIGIN+'/v1/models', timeout=(5, 5)); models.raise_for_status()
    advertised = models.json(); write(ROOT/'models.json', advertised)
    if not set(MODELS) <= {r['id'] for r in advertised['data']}:
        raise ValueError('both exact original model names must be advertised')
    results = []
    for index, model in enumerate(MODELS):
        payload = {'model': model, 'messages': [{'role': 'user', 'content': proof['prompt']}],
            'temperature': 0, 'max_tokens': 256}
        write(ROOT/f'request_{index}.json', payload)
        started = time.monotonic()
        try:
            response = client.post(ORIGIN+'/v1/chat/completions', json=payload, timeout=(5, 30))
            body = response.text.replace(key, '[REDACTED]')
            write(ROOT/f'response_{index}.json', {'status_code': response.status_code, 'body': body})
            parsed = response.json() if response.ok else {}
            choices = parsed.get('choices') or [{}]
            content = (choices[0].get('message') or {}).get('content')
            record = {'model': model, 'status_code': response.status_code,
                'received_nonempty_completion': isinstance(content, str) and bool(content.strip()),
                'neutral_exact_OK': isinstance(content, str) and content.strip() == 'OK',
                'elapsed_seconds': time.monotonic()-started}
        except requests.RequestException as error:
            record = {'model': model, 'error_type': type(error).__name__,
                'received_nonempty_completion': False, 'neutral_exact_OK': False,
                'elapsed_seconds': time.monotonic()-started}
            write(ROOT/f'response_{index}.json', record)
        results.append(record); print(json.dumps(record), flush=True)
    if sha(__file__) != proof['source_sha256']: raise ValueError('probe source changed during execution')
    write(ROOT/'result.json', {'status': 'neutral_local_transport_probe_completed', 'results': results,
        'new_generation_requests': len(results), 'new_semantic_judge_calls': 0, 'judge_qualified': False,
        'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()}})


if __name__ == '__main__': run()
