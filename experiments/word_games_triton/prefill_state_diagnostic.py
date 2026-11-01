"""Post-replay diagnostic: split prefill without serializing sequence state."""
import json
from pathlib import Path
import subprocess

from experiments.word_games.common import read, sha, write
from experiments.word_games.qwen_runtime import REQUESTS
from .schema_prefill import PREFIXES
from .deploy import CONTAINER, ROOT

CODE = r'''
import json, sys, urllib.request
data = json.load(sys.stdin)
def call(path, body):
    request = urllib.request.Request('http://127.0.0.1:18080' + path,
        json.dumps(body).encode(), headers={'Content-Type':'application/json','Connection':'close'})
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)
def tokenize(content):
    return call('/tokenize', {'content':content,'add_special':True,'parse_special':True})['tokens']
formatted = call('/apply-template', {'messages':[{'role':'user','content':data['item']['prompt']}],
    'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})['prompt']
base = {'prompt':formatted,'n_predict':data['item']['max_tokens'],'temperature':0,'seed':73,
        'repeat_penalty':1.0,'cache_prompt':False,'id_slot':0,'json_schema':{'type':'object'},
        'stream':False,'return_tokens':True}
whole = call('/completion',base)
tokens = tokenize('<|im_start|>user\n' + data['prefix'])[:-1]
full = tokenize(formatted)
assert full[:len(tokens)] == tokens
prefill = call('/completion',{'prompt':tokens,'n_predict':0,'cache_prompt':False,'id_slot':0})
assert prefill['tokens_cached'] == len(tokens)
split = call('/completion',{**base,'prompt':full,'cache_prompt':True})
assert split['timings']['cache_n'] == len(tokens)
# Leave only harmless warmup text in the native slot. Ordinary Triton requests
# already disable reuse, so the deployed serving behavior is unchanged.
call('/completion',{**base,'prompt':'Return exactly {"ready":true} as JSON.','n_predict':24})
print(json.dumps({'whole':whole,'prefill':prefill,'split_without_save_restore':split}))
'''


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    item = next(i for i in read(REQUESTS) if i['id'] == 'cases/outside_destination/sender/turn_01')
    write(out / 'producer.json', {'purpose': 'post-outcome mechanism diagnostic, not speed evidence',
        'source_sha256': sha(__file__), 'manifest': read(ROOT / 'manifest.json'), 'request': item})
    (out / 'producer.py').write_bytes(Path(__file__).read_bytes())
    result = subprocess.run(['docker', 'exec', '-i', CONTAINER,
        '/opt/pixi-gemma4/.pixi/envs/default/bin/python', '-c', CODE],
        input=json.dumps({'item': item, 'prefix': PREFIXES[0]['text']}), capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    root = Path('data/experiments/word_games/20261101_schema_prefill_full_v1')
    control = read(root / '1_control/runs' / item['id'] / 'response.json')['parsed']
    cached = read(root / '2_cached/runs' / item['id'] / 'response.json')['parsed']
    data['whole_matches_control'] = json.loads(data['whole']['content']) == control
    data['split_matches_cached'] = json.loads(data['split_without_save_restore']['content']) == cached
    write(out / 'result.json', data)
    print({k: data[k] for k in ('whole_matches_control', 'split_matches_cached')})


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
