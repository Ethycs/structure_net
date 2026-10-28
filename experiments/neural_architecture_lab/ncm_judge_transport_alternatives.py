"""One neutral availability request per advertised alternate route; no grading."""
import json
from pathlib import Path
import time

import requests

from experiments.neural_architecture_lab import ncm_judge_transport_probe as first

ROOT = first.ROOT.parent/'20261028_judge_transport_alternatives_v1'
MODELS = ('codex_sdk/gpt-5.6-sol', 'claude_code/claude-sonnet-4-6', 'claude-sonnet-5')


def main():
    if ROOT.exists(): raise ValueError('fresh nongrading diagnostic required')
    ROOT.mkdir(); (ROOT/'producer.py').write_bytes(Path(__file__).read_bytes())
    first.write(ROOT/'producer.json', {'schema': 'nal.ncm-judge-transport-alternatives.v1',
        'models': MODELS, 'attempts_per_model': 1, 'read_timeout_seconds': 30, 'connect_timeout_seconds': 5,
        'semantic_grading': False, 'evaluation_history_changed': False,
        'selection_reason': 'Terra neutral30s probe timed out; routes were advertised by authenticated GET/v1/models',
        'source_hashes': {str(p): first.sha(p) for p in (Path(__file__), Path(first.__file__), Path('pixi.lock'), Path('pyproject.toml'))},
        'input_hashes': {str(first.ROOT/'result.json'): first.sha(first.ROOT/'result.json')}})
    key = next(x.split('=', 1)[1].strip().strip('"').strip("'") for x in Path('.env').read_text().splitlines()
               if x.startswith('LITELLM_KEY='))
    outcomes = []
    for i, model in enumerate(MODELS):
        payload = {'model': model, 'temperature': 0, 'max_tokens': 256,
            'messages': [{'role': 'user', 'content': 'This is a transport availability check. Reply with the single word OK.'}]}
        first.write(ROOT/f'request_{i}.json', payload)
        started = time.monotonic(); result = {'model': model, 'valid_response': False}
        try:
            response = requests.post(first.ENDPOINT, headers={'Authorization': 'Bearer '+key}, json=payload,
                verify='/home/rabbit/.config/lakefs/caddy-root.crt', timeout=(5, 30))
            result['http_status'] = response.status_code
            raw = response.json(); first.write(ROOT/f'response_{i}.json', raw); response.raise_for_status()
            result['returned_model'] = raw.get('model')
            result['valid_response'] = (raw.get('model') == model
                and raw.get('choices', [{}])[0].get('message', {}).get('content', '').strip() == 'OK')
        except Exception as error: result['error_type'] = type(error).__name__
        result['elapsed_seconds'] = time.monotonic()-started
        first.write(ROOT/f'outcome_{i}.json', result); outcomes.append(result)
        print(json.dumps(result), flush=True)
    result = {'schema': 'nal.ncm-judge-transport-alternatives.v1', 'outcomes': outcomes,
        'external_generation_requests': len(MODELS), 'new_semantic_judge_calls': 0, 'task_accuracy': None,
        'final_test_opened': False, 'replacement_judge_qualified': False,
        'artifacts': {str(p): first.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    first.write(ROOT/'result.json', result)


if __name__ == '__main__': main()
