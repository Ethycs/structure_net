"""One nongrading availability probe; never retry or touch evaluation history."""
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path('data/experiments/ncm_paper_regime/20261028_judge_transport_probe_v1')
MODEL = 'codex_sdk/gpt-5.6-terra'
ENDPOINT = 'https://central-dev.zt:4000/v1/chat/completions'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for raw in iter(lambda: f.read(8*1024**2), b''): h.update(raw)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def main():
    if ROOT.exists(): raise ValueError('fresh diagnostic root required; no retry')
    payload = {'model': MODEL, 'temperature': 0, 'max_tokens': 256,
        'messages': [{'role': 'user', 'content': 'This is a transport availability check. Reply with the single word OK.'}]}
    proof = {'schema': 'nal.ncm-judge-transport-probe.v1', 'endpoint': ENDPOINT, 'request': payload,
        'attempts': 1, 'connect_timeout_seconds': 5, 'read_timeout_seconds': 30,
        'semantic_grading': False, 'evaluation_history_changed': False,
        'source_hashes': {str(p): sha(p) for p in (Path(__file__), Path('pixi.lock'), Path('pyproject.toml'))}}
    ROOT.mkdir(); (ROOT/'producer.py').write_bytes(Path(__file__).read_bytes()); write(ROOT/'producer.json', proof)
    key = next(x.split('=', 1)[1].strip().strip('"').strip("'") for x in Path('.env').read_text().splitlines()
               if x.startswith('LITELLM_KEY='))
    started = time.monotonic(); result = {'schema': proof['schema'], 'model': MODEL, 'external_generation_requests': 1,
        'new_semantic_judge_calls': 0, 'task_accuracy': None, 'final_test_opened': False, 'valid_response': False}
    try:
        response = requests.post(ENDPOINT, headers={'Authorization': 'Bearer '+key}, json=payload,
            verify='/home/rabbit/.config/lakefs/caddy-root.crt', timeout=(5, 30))
        result['http_status'] = response.status_code
        raw = response.json(); write(ROOT/'response.json', raw); response.raise_for_status()
        result['returned_model'] = raw.get('model')
        result['valid_response'] = (raw.get('model') == MODEL
            and raw.get('choices', [{}])[0].get('message', {}).get('content', '').strip() == 'OK')
    except Exception as error: result['error_type'] = type(error).__name__
    result['elapsed_seconds'] = time.monotonic()-started
    result['artifacts'] = {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()}
    write(ROOT/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__': main()
