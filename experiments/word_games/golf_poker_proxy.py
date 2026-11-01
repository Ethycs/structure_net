"""Exploratory golf-through-poker experiment using fresh proxy model calls."""
import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import time

import requests

from .common import digest, read, sha, write

ENDPOINT = 'https://central-dev.zt:4000/v1/chat/completions'
MODEL = 'claude-sonnet-5'
STATES = ('tee', 'fairway', 'approach', 'rough', 'bunker', 'green', 'holed')
ACTIONS = ('driver', 'iron', 'wedge', 'punch', 'putt', 'relief', 'bump')
SEED = 731104


def reference_counts(texts):
    """Visible-text counts with the already deployed Qwen GGUF tokenizer."""
    code = """import json,sys,urllib.request
out=[]
for text in json.load(sys.stdin):
 req=urllib.request.Request('http://127.0.0.1:18080/tokenize',data=json.dumps({'content':text,'add_special':False,'parse_special':False}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as r: out.append(len(json.load(r)['tokens']))
json.dump(out,sys.stdout)
"""
    result = subprocess.run(['docker','exec','-i','word-games-triton',
                             '/opt/pixi-gemma4/.pixi/envs/default/bin/python','-c',code],
                            input=json.dumps(texts),text=True,capture_output=True,check=True,timeout=60)
    return json.loads(result.stdout)


def course(number):
    """Explicit toy golf rules, not a claim to simulate regulation golf."""
    rng = random.Random(SEED + number)
    drive, recovery, sand, putting = (rng.choice(v) for v in
        ((3, 5, 7, 8), (3, 5, 7), (6, 8, 9), (5, 7, 9)))
    pitch = rng.choice((6, 8, 10))
    def row(*items):
        return [list(x) for x in items if x[1]]
    rules = {
        'tee': {
            'driver': row(('approach', drive, 0), ('rough', 8-drive, 0), ('tee', 2, 1)),
            'iron': row(('fairway', 9, 0), ('rough', 1, 0))},
        'fairway': {
            'driver': row(('green', 2, 0), ('approach', 4, 0), ('rough', 2, 0),
                          ('bunker', 1, 0), ('fairway', 1, 1)),
            'iron': row(('green', 2, 0), ('approach', 7, 0), ('rough', 1, 0)),
            'wedge': row(('approach', 10, 0))},
        'approach': {
            'wedge': row(('green', pitch, 0), ('bunker', 10-pitch, 0)),
            'bump': row(('green', 7, 0), ('approach', 3, 0))},
        'rough': {
            'punch': row(('approach', 10, 0)),
            'wedge': row(('green', recovery, 0), ('bunker', 2, 0), ('rough', 8-recovery, 0)),
            'relief': row(('approach', 10, 1))},
        'bunker': {
            'wedge': row(('green', sand, 0), ('bunker', 10-sand, 0)),
            'relief': row(('approach', 10, 1))},
        'green': {'putt': row(('holed', putting, 0), ('green', 10-putting, 0))},
        'holed': {}}
    return {'number': number, 'par': 5, 'rules': rules}


def oracle(rules):
    values = dict.fromkeys(STATES, 0.0)
    for _ in range(10000):
        q = {s: {a: sum(w/10 * (1+p+values[n]) for n,w,p in outcomes)
                 for a,outcomes in choices.items()} for s,choices in rules.items()}
        new = {s: min(v.values()) if v else 0.0 for s,v in q.items()}
        if max(abs(new[s]-values[s]) for s in STATES) < 1e-11:
            return new, q
        values = new
    raise ValueError('oracle failed to converge')


def validate_mapping(mapping):
    if set(mapping) != {'states', 'actions'}:
        raise ValueError('mapping must have only states/actions')
    for kind, expected in (('states', STATES), ('actions', ACTIONS)):
        part = mapping[kind]
        if not isinstance(part, dict) or set(part) != set(expected):
            raise ValueError('mapping is not total')
        labels = list(part.values())
        if any(not isinstance(x, str) or not x or len(x)>48 or
               any(c not in 'abcdefghijklmnopqrstuvwxyz_' for c in x) for x in labels):
            raise ValueError('mapping labels must be short lowercase identifiers')
        if len(set(labels)) != len(labels):
            raise ValueError('mapping is not injective')
        if any(any(word in label.split('_') for word in STATES+ACTIONS+('golf', 'stroke', 'hole'))
               for label in labels):
            raise ValueError('source vocabulary in target labels')
    return mapping


def translate_rules(rules, mapping):
    sm, am = mapping['states'], mapping['actions']
    return {sm[s]: {am[a]: [[sm[n], w, p] for n,w,p in rows]
            for a,rows in choices.items()} for s,choices in rules.items()}


def advance(state, action):
    choices = state['course']['rules'][state['state']]
    if action not in choices:
        raise ValueError('illegal action')
    # Common random numbers per hole and move, independent of arm and action.
    u = random.Random(SEED*100000 + state['course']['number']*100 + state['moves']).random()*10
    cumulative = 0
    for nxt, weight, penalty in choices[action]:
        cumulative += weight
        if u < cumulative:
            state['state'] = nxt
            state['strokes'] += 1+penalty
            state['moves'] += 1
            return {'next': nxt, 'penalty': penalty}
    raise AssertionError('invalid probability mass')


def parsed_object(content):
    text = content.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('response is not an object')
    return result


class Proxy:
    def __init__(self, out, model):
        self.out, self.model, self.rows = out, model, []
        key = next(x.split('=', 1)[1].strip().strip('"').strip("'")
                   for x in Path('.env').read_text().splitlines() if x.startswith('LITELLM_KEY='))
        self.http = requests.Session()
        self.http.headers['Authorization'] = 'Bearer '+key
        self.http.verify = '/home/rabbit/.config/lakefs/caddy-root.crt'

    def call(self, name, prompt, limit=2400, saved_call=None):
        payload = {'model': self.model, 'temperature': 0, 'max_tokens': limit,
                   'messages': [{'role': 'user', 'content': prompt}]}
        root = self.out/'calls'/name
        write(root/'request.json', payload)
        started = time.perf_counter()
        try:
            if saved_call is None:
                response = self.http.post(ENDPOINT, json=payload, timeout=(10, 180))
                elapsed = time.perf_counter()-started
                raw = response.json()
                status = response.status_code
            else:
                if read(saved_call/'request.json') != payload:
                    raise ValueError('saved mapping request differs')
                raw = read(saved_call/'response.json')
                elapsed = read(saved_call/'error.json')['seconds']
                status = 200
            write(root/'response.json', raw)
            if status != 200:
                raise ValueError('proxy HTTP '+str(status))
            choice = raw['choices'][0]
            usage = raw['usage']
            if raw.get('model') != self.model:
                raise ValueError('returned model alias changed')
            inputs, tokens = reference_counts([prompt, choice['message']['content']])
            row = {'name': name, 'seconds': elapsed, 'completion_tokens': tokens,
                   'prompt_tokens': inputs, 'usage': usage,
                   'token_measurement': 'deployed Qwen GGUF visible-text reference; not native provider usage',
                   'saved_call': str(saved_call) if saved_call else None,
                   'requested_model': self.model, 'returned_model': raw.get('model'),
                   'finish_reason': choice.get('finish_reason'),
                   'response_sha256': sha(root/'response.json')}
            self.rows.append(row)
            write(root/'timing.json', row)
            print(json.dumps(row), flush=True)
            if choice.get('finish_reason') != 'stop':
                raise ValueError('generation did not end normally')
            return parsed_object(choice['message']['content'])
        except Exception as exc:
            write(root/'error.json', {'error_type': type(exc).__name__, 'message': str(exc)[:400],
                                      'seconds': time.perf_counter()-started})
            raise


def prompt_for(states, mapping, arm):
    poker = arm == 'poker'
    entity, position, ledger = ('hand', 'street', 'chips_spent') if poker else ('hole', 'lie', 'strokes')
    material = []
    for state in states:
        material.append({entity: state['course']['number'], position: mapping['states'][state['state']],
                         ledger: state['strokes'], 'transition_table': translate_rules(state['course']['rules'], mapping)})
    intro = ('Play this custom poker-themed game. Minimize chips spent to settle every hand. '
             'Poker is the operational vocabulary; conventional winning-hand rankings or maximizing winnings '
             'do not override these supplied house rules. ' if poker else
             'Play this simplified golf game. Minimize strokes to finish every hole. '
             'Use this explicit probabilistic simulation, not unprovided real-world golf mechanics. ')
    return intro + (
        'Each move costs 1, plus the listed penalty. Each outcome is [next_position, weight_out_of_10, '
        'extra_penalty]. Outcome weights sum to 10. A position with no moves is terminal. '
        'Transitions are stationary and independent. Choose one currently legal move per active '
        'entry to minimize expected total remaining cost, accounting for future moves and penalties. '
        'An outside host executes all transitions and maintains the authoritative ledger. '
        'Do not invent outcomes or execute multiple moves. Return only JSON: '
        '{"decisions":[{"id":integer,"move":"exact_move_label","cost_so_far":integer,'
        '"reason":"one brief justification of 35-55 words, including the relevant tradeoff"}]}. '
        'Include exactly one decision per entry. cost_so_far must copy the authoritative current ledger. '
        'The explanations are observable decision justifications, not a request for hidden reasoning.\n'
    ) + json.dumps(material, separators=(',', ':'))


def run(out, model, target, mapping_call=None):
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    config = {'study': 'word-games-golf-poker-proxy-v1', 'exploratory': True,
              'endpoint': ENDPOINT, 'model': model, 'seed': SEED,
              'target_generated_tokens': target, 'stop_at_complete_paired_blocks': True,
              'initial_holes': 9, 'additional_holes_per_block': 3, 'max_holes_per_arm': 27,
              'max_moves_per_hole': 12, 'max_requests': 70, 'retries': 0,
              'source_sha256': sha(source), 'pixi_lock_sha256': sha('pixi.lock'),
              'plan_sha256': sha('docs/07 - Status Reports/2026-11-01_word-games-golf-poker-proxy-plan.md'),
              'instrumentation_amendment_sha256': sha('docs/07 - Status Reports/2026-11-01_word-games-golf-poker-proxy-token-amendment.md'),
              'token_measurement': 'Qwen-token-equivalent visible text; provider usage retained separately',
              'reference_tokenizer_manifest_sha256': sha('.pixi/word-games-triton/manifest.json'),
              'saved_mapping_call': str(mapping_call) if mapping_call else None,
              'timing': 'nonstreaming HTTP round trip including proxy, prefill, generation and response',
              'source_engine': 'new small stochastic golf fixture; not the V2 placement Engine',
              'local_serving_configuration_changed': False}
    write(out/'producer.json', config)
    (out/'producer.py').write_bytes(source.read_bytes())
    courses = [course(i) for i in range(1, 28)]
    write(out/'courses.json', courses)
    values = {c['number']: oracle(c['rules']) for c in courses}
    write(out/'oracle.json', {n: {'values': v, 'action_values': q} for n,(v,q) in values.items()})
    client = Proxy(out, model)
    decisions, finished, failed, mapping, error = [], [], [], None, None
    started = time.perf_counter()
    try:
        prompt = ('Create an operational metaphor that allows an agent to play a simplified golf game '
                  'using only poker vocabulary. The host preserves the exact transition probabilities '
                  'and costs. Strokes become chips spent, holes become hands, and the objective remains '
                  'MINIMIZE cost until terminal, not maximize winnings. You choose the poker names; '
                  'do not write code or alter rules. Return only JSON with keys states and actions, '
                  'each mapping EVERY supplied source label to a distinct lowercase poker-themed '
                  'identifier containing only a-z and underscores, up to 48 characters. Avoid all '
                  'source labels and words golf/stroke/hole in the target identifiers. '
                  'States: '+json.dumps(STATES)+'. Actions: '+json.dumps(ACTIONS)+'. '
                  'holed is terminal; tee is initial; the other states are intermediate positions. '
                  'driver is risky long play; iron is controlled advance; wedge is short attack; '
                  'punch is conservative recovery; putt is terminal attempt; relief is costly escape; '
                  'bump is conservative short advance. Mapping only, no commentary.')
        mapping = validate_mapping(client.call('000_mapping', prompt, 1200, saved_call=mapping_call))
        write(out/'mapping.json', mapping)
        identity = {'states': {s:s for s in STATES}, 'actions': {a:a for a in ACTIONS}}
        number, block = 1, 0
        while number <= 27:
            count = 9 if block == 0 else 3
            batch = courses[number-1:number-1+count]
            arms = {arm: [{'course': c, 'state': 'tee', 'strokes': 0, 'moves': 0}
                           for c in batch] for arm in ('golf', 'poker')}
            wave = 0
            while any(arms.values()):
                order = ('golf', 'poker') if (wave+block)%2 == 0 else ('poker', 'golf')
                for arm in order:
                    active = arms[arm]
                    if not active:
                        continue
                    if len(client.rows) >= 70:
                        raise ValueError('request safety cap')
                    codec = mapping if arm == 'poker' else identity
                    request_name = f'{len(client.rows):03d}_{arm}_block{block}_wave{wave}'
                    proposal = client.call(request_name, prompt_for(active, codec, arm))
                    proposed = proposal.get('decisions')
                    expected = {s['course']['number'] for s in active}
                    if (not isinstance(proposed, list) or len(proposed) != len(active) or
                        any(not isinstance(d,dict) or type(d.get('id')) is not int for d in proposed) or
                        {d['id'] for d in proposed} != expected):
                        raise ValueError('invalid decision batch')
                    by_id = {d['id']: d for d in proposed}
                    reverse = {v:k for k,v in codec['actions'].items()}
                    remaining = []
                    for state in active:
                        n, position = state['course']['number'], state['state']
                        proposal = by_id[n]
                        action = reverse.get(proposal.get('move'))
                        legal = action in state['course']['rules'][position]
                        v, q = values[n]
                        row = {'arm': arm, 'hole': n, 'before': position, 'move_number': state['moves']+1,
                               'proposal': proposal, 'source_action': action, 'legal': legal,
                               'ledger_correct': type(proposal.get('cost_so_far')) is int and
                                                 proposal['cost_so_far'] == state['strokes'],
                               'nontrivial': len(q[position])>1, 'request': request_name}
                        if legal:
                            regret = q[position][action]-v[position]
                            row.update(expected_excess_strokes=regret, optimal=regret<1e-7,
                                       transition=advance(state, action), strokes_after=state['strokes'])
                        else:
                            row.update(expected_excess_strokes=None, optimal=False)
                        decisions.append(row)
                        if state['state'] == 'holed':
                            finished.append({'arm': arm, 'hole': n, 'strokes': state['strokes'],
                                             'moves': state['moves'], 'par': state['course']['par'],
                                             'oracle_expected_strokes': v['tee']})
                        elif not legal or state['moves'] >= 12:
                            failed.append({'arm': arm, 'hole': n, 'state': state['state'],
                                           'strokes': state['strokes'],
                                           'reason': 'illegal_move' if not legal else 'move_cap'})
                        else:
                            remaining.append(state)
                    write(out/'events'/f'{request_name}.json', [r for r in decisions if r['request']==request_name])
                    arms[arm] = remaining
                wave += 1
            number += count
            block += 1
            write(out/'blocks'/f'{block}.json', {'holes_per_arm': number-1,
                  'completion_tokens': sum(r['completion_tokens'] for r in client.rows),
                  'completed_holes': len(finished), 'failed_holes': len(failed)})
            if sum(r['completion_tokens'] for r in client.rows) >= target*.9:
                break
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:400]}
    wall = time.perf_counter()-started
    summary = {}
    for arm in ('golf', 'poker'):
        ds = [d for d in decisions if d['arm']==arm]
        nontrivial = [d for d in ds if d['nontrivial']]
        complete = [h for h in finished if h['arm']==arm]
        calls = [r for r in client.rows if f'_{arm}_' in r['name']]
        tokens = sum(r['completion_tokens'] for r in calls)
        seconds = sum(r['seconds'] for r in calls)
        summary[arm] = {'decisions': len(ds), 'legal': sum(d['legal'] for d in ds),
                        'ledger_correct': sum(d['ledger_correct'] for d in ds),
                        'optimal': sum(d['optimal'] for d in ds),
                        'nontrivial_decisions': len(nontrivial),
                        'nontrivial_optimal': sum(d['optimal'] for d in nontrivial),
                        'completed_holes': len(complete),
                        'failed_holes': len([h for h in failed if h['arm']==arm]),
                        'total_strokes_completed_holes': sum(h['strokes'] for h in complete),
                        'total_par_completed_holes': sum(h['par'] for h in complete),
                        'oracle_expected_strokes_completed_holes': sum(h['oracle_expected_strokes'] for h in complete),
                        'expected_excess_strokes_all_decisions': sum(d['expected_excess_strokes'] or 0 for d in ds),
                        'completion_tokens': tokens, 'request_seconds': seconds,
                        'full_request_tok_s': tokens/seconds if seconds else None}
    tokens = sum(r['completion_tokens'] for r in client.rows)
    seconds = sum(r['seconds'] for r in client.rows)
    result = {'study': config['study'], 'completed': error is None, 'error': error,
              'summary': summary, 'completion_tokens': tokens,
              'prompt_tokens': sum(r['prompt_tokens'] for r in client.rows),
              'request_seconds': seconds, 'experiment_wall_seconds': wall,
              'full_request_tok_s': tokens/seconds if seconds else None,
              'requested_model': model, 'returned_models': sorted({str(r['returned_model']) for r in client.rows}),
              'requests': len(client.rows), 'model_generated_mapping': mapping,
              'token_measurement': config['token_measurement'],
              'provider_reported_completion_tokens': sum(r['usage']['completion_tokens'] for r in client.rows),
              'decisions': decisions, 'finished': finished, 'failed': failed,
              'source_unchanged': sha(source)==config['source_sha256'],
              'population_accuracy_established': False,
              'secrets_in_synthetic_inputs': False,
              'v3_codec_integrated': False,
              'artifacts': {str(p.relative_to(out)): sha(p) for p in out.rglob('*.json')}}
    write(out/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('decisions','artifacts','finished','model_generated_mapping')}), flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--target-generated-tokens', type=int, default=10000)
    parser.add_argument('--mapping-call', type=Path)
    args = parser.parse_args()
    run(args.output, args.model, args.target_generated_tokens, args.mapping_call)
