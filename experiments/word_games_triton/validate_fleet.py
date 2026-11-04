"""Live per-model chat, Triton contract, memory and unload receipts."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from . import fleet


def memory():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv'], text=True)


def run(alias, output, allow_explanatory_text=False):
    model = fleet.entry(alias)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'memory_before.csv').write_text(memory())
    start = time.monotonic()
    test = subprocess.run([sys.executable, '-m', 'experiments.word_games_triton.check_chat_route',
        '--native', '--api-base', 'http://10.158.218.141:18080/v1',
        '--model', model['alias'], '--output', str(output / 'chat'),
        *(['--allow-explanatory-text'] if allow_explanatory_text else [])], capture_output=True, text=True)
    (output / 'chat_check.log').write_text(test.stdout + test.stderr)
    state = fleet.control('status')
    (output / 'memory_loaded.csv').write_text(memory())
    (output / 'state_loaded.json').write_text(json.dumps(state, indent=2) + '\n')
    elapsed = time.monotonic() - start
    result = {'alias': alias, 'chat_passed': test.returncode == 0,
              'chat_seconds_including_load': elapsed, 'classification': 'systems_checks_not_task_accuracy'}
    try:
        if test.returncode:
            raise RuntimeError('Chat check failed; inspect ' + str(output / 'chat_check.log'))
        assert [m['alias'] for m in state['models'] if m['ready']] == [model['alias']]
        response = fleet.api(f"/v2/models/{model['triton_model']}/infer", {
            'inputs': [
                {'name': 'text_input', 'shape': [1], 'datatype': 'BYTES',
                 'data': ['Compute 17 + 25. Return a JSON object with only the key sum and its numeric value.']},
                {'name': 'sampling_parameters', 'shape': [1], 'datatype': 'BYTES',
                 'data': [json.dumps({'max_tokens': 64})]}]})
        (output / 'triton_inference.json').write_text(json.dumps(response, indent=2) + '\n')
        content = next(o['data'][0] for o in response['outputs'] if o['name'] == 'text_output')
        result['triton_contract_passed'] = json.loads(content) == {'sum': 42}
        name = model['triton_model']
        assert re.fullmatch(r'[a-z0-9-]+', name)
        native_log = subprocess.check_output(['docker', 'exec', 'triton',
            '/opt/pixi-gemma4/.pixi/envs/default/bin/python', '-c',
            'from pathlib import Path; import sys; '
            'print(max((Path("/tmp") / sys.argv[1]).glob("native-*.log"), '
            'key=lambda p: p.stat().st_mtime_ns).read_text())', name], text=True)
        (output / 'native.log').write_text(native_log)
        result['offload_lines'] = [line for line in native_log.splitlines()
                                  if 'offloaded' in line or 'buffer size' in line]
    finally:
        start = time.monotonic()
        unloaded = fleet.control('unload')
        result['unload_seconds'] = time.monotonic() - start
        result['unloaded'] = not any(m['ready'] for m in unloaded['models'])
        (output / 'memory_unloaded.csv').write_text(memory())
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result.get('triton_contract_passed'):
        raise RuntimeError('Triton inference check failed')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-explanatory-text', action='store_true')
    args = parser.parse_args()
    run(args.model, args.output, args.allow_explanatory_text)
