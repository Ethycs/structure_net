"""Isolated native-backend diagnostics; run inside the existing Triton container."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request


CASES = [
    ('addition', 'What is 2 + 2? Answer with one number.', '4'),
    ('capital', 'What is the capital of France? Answer with one word.', 'paris'),
    ('week', 'How many days are in a week? Answer with one number.', '7'),
    ('sum_json', 'Compute 17 + 25. Return a JSON object with only the key sum and its numeric value.', {'sum': 42}),
    ('unit', 'Convert 5 kilometers to meters. Answer with one number, without commas.', '5000'),
    ('name', 'Mira owns a red bicycle. Who owns the bicycle? Answer with only the name.', 'mira'),
    ('copy', 'Copy this code exactly: ZX47', 'zx47'),
    ('reverse', 'Reverse the letters abc. Answer with only the reversed letters.', 'cba'),
    ('order', 'Which is larger: 12 or 7? Answer with one number.', '12'),
    ('color', 'My favorite color is violet. What is my favorite color? Answer with one word.', 'violet'),
    ('subtract', 'What is 10 minus 3? Answer with one number.', '7'),
    ('extract', 'Extract the number of apples: Lea bought 6 apples and 2 pears. Answer with one number.', '6'),
]


def grade(content, expected):
    if isinstance(expected, dict):
        try:
            matched = json.loads(content) == expected
        except ValueError:
            matched = False
        return {'answer_present': matched, 'exact_format': matched}
    text = content.strip().lower().rstrip('.')
    # Keep format compliance distinct from recovering the requested fact.
    matched = bool(re.search(r'(?<!\w)' + re.escape(expected) + r'(?!\w)', text))
    # Presence is a diagnostic aid, not a semantic quality judge. Audit raw answers.
    return {'answer_present': matched, 'exact_format': text == expected}


def run(config):
    base = Path('/models/word-games-dolphin/1')
    runtime = json.loads((base / 'runtime.json').read_text())
    assets = base / 'runtime-assets'
    native, sysroot, cuda = (assets / key for key in ('native', 'sysroot', 'cuda'))
    out = Path('/tmp/word-games-dolphin-debug') / config['cell']
    out.mkdir(parents=True, exist_ok=False)
    args = runtime['native_args'][:]
    changes = {'-m': config['weights'], '--port': '18089', '--ctx-size': '2048',
               '--batch-size': '256', '--ubatch-size': '256', '--verbosity': '4'}
    if config['backend'] == 'cpu':
        changes.update({'--device': 'none', '--n-gpu-layers': '0', '--flash-attn': 'off',
                        '--cache-type-k': 'f16', '--cache-type-v': 'f16',
                        '--threads': '8', '--threads-batch': '8'})
    changes.update(config.get('overrides', {}))
    for flag, value in changes.items():
        args[args.index(flag) + 1] = value
    env = dict(os.environ)
    env.pop('LD_PRELOAD', None)
    env.pop('LD_LIBRARY_PATH', None)
    env['CUDA_VISIBLE_DEVICES'] = '' if config['backend'] == 'cpu' else runtime['gpu_uuids'][0]
    env['CUDA_CACHE_PATH'] = '/tmp/word-games-dolphin/cuda-cache'
    env['CUDA_CACHE_MAXSIZE'] = '2147483648'
    command = [str(sysroot / 'ld-linux-x86-64.so.2'), '--library-path',
               ':'.join(map(str, (sysroot, native, cuda))) + ':/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu',
               str(native / 'llama-server'), *args]
    (out / 'producer.json').write_text(json.dumps({
        'config': config, 'command': command, 'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'],
        'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')

    def call(path, data=None, timeout=300):
        request = urllib.request.Request('http://127.0.0.1:18089' + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    process = None
    gpu_lock = None
    rows = []
    started = time.monotonic()
    try:
        if config['backend'] == 'gpu':
            gpu_lock = open('/tmp/word-games-model-gpus.lock', 'a')
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (out / 'native.log').open('x') as log:
            process = subprocess.Popen(command, cwd=native, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 480
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Native process exited during load')
                try:
                    call('/health', timeout=1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Native load exceeded 480 seconds')
                    time.sleep(.25)
            load_seconds = time.monotonic() - started
            for name, prompt, expected in CASES[:config.get('cases', len(CASES))]:
                body = {'model': 'dolphin-x1-trinity-nano', 'messages': [
                    {'role': 'system', 'content': 'You are Dolphin, a helpful assistant.'},
                    {'role': 'user', 'content': prompt}],
                    'temperature': 0, 'seed': 73, 'max_tokens': config.get('max_tokens', 64)}
                if isinstance(expected, dict):
                    body['response_format'] = {'type': 'json_object'}
                begin = time.monotonic()
                result = call('/v1/chat/completions', body)
                content = result['choices'][0]['message']['content']
                row = {'id': name, 'request': body, 'response': result,
                       'seconds': time.monotonic() - begin, 'grade': grade(content, expected)}
                rows.append(row)
                (out / (name + '.json')).write_text(json.dumps(row, indent=2) + '\n')
                print(json.dumps({'id': name, 'content': content, 'grade': row['grade']}), flush=True)
            summary = {'cell': config['cell'], 'load_seconds': load_seconds, 'cases': len(rows),
                       'answers_present': sum(r['grade']['answer_present'] for r in rows),
                       'exact_format_correct': sum(r['grade']['exact_format'] for r in rows),
                       'classification': 'exploratory_smoke_suite_not_general_accuracy'}
            (out / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(summary), flush=True)
    except Exception as exc:
        (out / 'failure.json').write_text(json.dumps({'error': repr(exc), 'completed': len(rows)}) + '\n')
        raise
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if gpu_lock is not None:
            gpu_lock.close()


if __name__ == '__main__':
    run(json.load(sys.stdin))
