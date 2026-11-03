"""Verify native/LiteLLM chat transport; these are systems checks, not quality scores."""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.request


NATIVE_MODEL = 'word-games-empero-qwen38-35b-a3b'
CA = Path('/etc/docker/certs.d/10.158.218.193:5000/ca.crt')
CASES = [
    ('history', {
        'messages': [
            {'role': 'system', 'content': 'Answer the last question with only a lowercase color word.'},
            {'role': 'user', 'content': 'Remember my color: violet.'},
            {'role': 'assistant', 'content': 'I will remember violet.'},
            {'role': 'user', 'content': 'What color did I ask you to remember?'},
        ],
    }),
    ('json', {
        'messages': [{'role': 'user', 'content': 'Compute 17 + 25. Return a JSON object with only the key sum and its numeric value.'}],
        'response_format': {'type': 'json_object'},
    }),
    ('stream', {
        'messages': [{'role': 'user', 'content': 'Reply with exactly the two words: silver meadow'}],
        'stream': True, 'stream_options': {'include_usage': True},
    }),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', default='https://10.158.218.193:4000/v1')
    parser.add_argument('--model', default='qwen3.8-35b-a3b')
    parser.add_argument('--native', action='store_true', help='Direct upstream: no gateway key or fallback controls')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    headers = {'Content-Type': 'application/json'}
    context = ssl.create_default_context(cafile=str(CA) if CA.exists() else None)
    if not args.native:
        key = next(line.split('=', 1)[1].strip().strip('"').strip("'")
                   for line in Path('.env').read_text().splitlines() if line.startswith('LITELLM_KEY='))
        headers['Authorization'] = 'Bearer ' + key
    results = []
    for name, settings in CASES:
        body = {'model': args.model, 'temperature': 0, 'seed': 73, 'max_tokens': 64, **settings}
        if not args.native:
            body.update(fallbacks=[], num_retries=0)
        request = urllib.request.Request(args.api_base.rstrip('/') + '/chat/completions',
                                         data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        try:
            response = urllib.request.urlopen(request, context=context, timeout=300)
        except urllib.error.HTTPError as error:
            failure = {'case': name, 'status': error.code, 'passed': False,
                       'error': error.read().decode(errors='replace')}
            (args.output / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
            raise SystemExit(f'Chat endpoint returned HTTP {error.code}; see failure.json') from None
        with response:
            record = {'case': name, 'request': body, 'status': response.status,
                      'headers': {k: v for k, v in response.headers.items()
                                  if k.lower() in ('content-type', 'x-litellm-model-id', 'x-litellm-api-base')}}
            if settings.get('stream'):
                events, done = [], False
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        done = True
                        break
                    events.append({'seconds': time.monotonic() - start, 'data': json.loads(data)})
                content = ''.join(choice.get('delta', {}).get('content') or ''
                                  for event in events for choice in event['data'].get('choices', []))
                models = {e['data'].get('model') for e in events if e['data'].get('model')}
                record.update(events=events, done=done, content=content,
                              passed=done and content.strip().strip('.').lower() == 'silver meadow'
                              and NATIVE_MODEL in models)
            else:
                data = json.load(response)
                content = data['choices'][0]['message']['content']
                correct = (content.strip().strip('.').lower() == 'violet' if name == 'history'
                           else json.loads(content) == {'sum': 42})
                record.update(response=data, content=content,
                              passed=correct and data.get('model') == NATIVE_MODEL
                              and data.get('usage', {}).get('completion_tokens', 0) > 0)
        record['seconds'] = time.monotonic() - start
        (args.output / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        results.append({'case': name, 'passed': record['passed'], 'seconds': record['seconds']})
        print(json.dumps(results[-1]), flush=True)
    summary = {'classification': 'systems_lifecycle_only_not_quality_evidence',
               'api_base': args.api_base, 'model': args.model, 'native_model': NATIVE_MODEL,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'results': results, 'passed': all(r['passed'] for r in results)}
    (args.output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not summary['passed']:
        raise SystemExit('Chat transport check failed; inspect saved responses')


if __name__ == '__main__':
    main()
