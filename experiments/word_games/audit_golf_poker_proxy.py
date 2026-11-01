"""Independently audit the frozen toy-game proxy trajectory and token receipts."""
import argparse
import json
from pathlib import Path
import random
import statistics

import numpy as np

from .common import read, sha, write


def independent_oracle(rules):
    states = [s for s, choices in rules.items() if choices]
    index = {s:i for i,s in enumerate(states)}
    policy = {s:next(iter(rules[s])) for s in states}
    for _ in range(100):
        matrix, cost = np.eye(len(states)), np.ones(len(states))
        for s,i in index.items():
            for nxt, weight, penalty in rules[s][policy[s]]:
                cost[i] += weight/10*penalty
                if nxt in index:
                    matrix[i,index[nxt]] -= weight/10
        solution = np.linalg.solve(matrix, cost)
        values = {s:float(solution[i]) for s,i in index.items()}
        values.update({s:0.0 for s in rules if s not in index})
        scores = {s:{a:sum(weight/10*(1+penalty+values[nxt]) for nxt,weight,penalty in outcomes)
                     for a,outcomes in rules[s].items()} for s in states}
        updated = {s:min(scores[s], key=scores[s].get) for s in states}
        if updated == policy:
            return values, scores
        policy = updated
    raise AssertionError('policy iteration did not converge')


def audit(result_path, output):
    result, root = read(result_path), result_path.parent
    for path, expected in result['artifacts'].items():
        assert sha(root/path) == expected, path
    config = read(root/'producer.json')
    assert sha(root/'producer.py') == config['source_sha256']
    courses = {c['number']:c for c in read(root/'courses.json')}
    stored_oracle = read(root/'oracle.json')
    independent = {n:independent_oracle(c['rules']) for n,c in courses.items()}
    difference = max(abs(v-stored_oracle[str(n)]['values'][s])
                     for n,(values,_) in independent.items() for s,v in values.items())
    assert difference < 1e-8
    mapping = read(root/'mapping.json')
    ledger, seen = {}, set()
    for row in result['decisions']:
        n, arm = row['hole'], row['arm']
        key = arm, n
        state, strokes, moves = ledger.get(key, ('tee', 0, 0))
        assert row['before'] == state and row['move_number'] == moves+1
        assert row['ledger_correct'] == (row['proposal'].get('cost_so_far') == strokes)
        sm = mapping['states'] if arm == 'poker' else {s:s for s in courses[n]['rules']}
        am = mapping['actions'] if arm == 'poker' else {a:a for actions in courses[n]['rules'].values() for a in actions}
        request = read(root/'calls'/row['request']/'request.json')
        material = json.loads(request['messages'][0]['content'].rsplit('\n',1)[1])
        entity, position, amount = ('hand','street','chips_spent') if arm=='poker' else ('hole','lie','strokes')
        entry = next(item for item in material if item[entity]==n)
        assert entry[position] == sm[state] and entry[amount] == strokes
        expected_table = {sm[s]:{am[a]:[[sm[t],w,p] for t,w,p in outcomes]
                                for a,outcomes in actions.items()} for s,actions in courses[n]['rules'].items()}
        assert entry['transition_table'] == expected_table
        action = row['source_action']
        assert row['legal'] == (action in courses[n]['rules'][state])
        if row['legal']:
            values, q = independent[n]
            regret = q[state][action]-values[state]
            assert abs(row['expected_excess_strokes']-regret)<1e-8
            assert row['optimal'] == (regret<1e-7)
            draw = random.Random(config['seed']*100000+n*100+moves).random()*10
            cumulative = 0
            for nxt,weight,penalty in courses[n]['rules'][state][action]:
                cumulative += weight
                if draw < cumulative:
                    break
            assert row['transition'] == {'next':nxt,'penalty':penalty}
            strokes += 1+penalty
            assert strokes == row['strokes_after']
            ledger[key] = nxt,strokes,moves+1
        seen.add(key)
    for hole in result['finished']:
        assert ledger[hole['arm'],hole['hole']] == ('holed',hole['strokes'],hole['moves'])
    rows = [read(p) for p in sorted((root/'calls').glob('*/timing.json'))]
    from .golf_poker_proxy import reference_counts
    texts = []
    for row in rows:
        call = root/'calls'/row['name']
        texts.extend((read(call/'request.json')['messages'][0]['content'],
                      read(call/'response.json')['choices'][0]['message']['content']))
    recounted = reference_counts(texts)
    for i,row in enumerate(rows):
        assert recounted[2*i:2*i+2] == [row['prompt_tokens'],row['completion_tokens']]
    assert sum(x['completion_tokens'] for x in rows) == result['completion_tokens']
    assert abs(sum(x['seconds'] for x in rows)-result['request_seconds'])<1e-8
    assert sha('.pixi/word-games-triton/manifest.json') == config['reference_tokenizer_manifest_sha256']
    arms = {}
    for arm in ('golf','poker'):
        ds = [d for d in result['decisions'] if d['arm']==arm]
        fs = [f for f in result['finished'] if f['arm']==arm]
        difficult = [d for d in ds if d['nontrivial']]
        arms[arm] = {'optimal_percent':100*sum(d['optimal'] for d in ds)/len(ds),
                     'nontrivial_optimal_percent':100*sum(d['optimal'] for d in difficult)/len(difficult),
                     'legal_percent':100*sum(d['legal'] for d in ds)/len(ds),
                     'ledger_percent':100*sum(d['ledger_correct'] for d in ds)/len(ds),
                     'strokes_relative_to_par':sum(f['strokes']-f['par'] for f in fs),
                     'completed_holes':len(fs)}
    failures = [d for d in result['decisions'] if not(d['legal'] and d['optimal'] and d['ledger_correct'])]
    write(output, {'campaign':str(result_path),'campaign_sha256':sha(result_path),
                   'audit_source_sha256':sha(__file__),'status':'independent_audit_passed',
                   'oracle_method':'linear-system policy iteration, compared with producer value iteration',
                   'oracle_max_difference':difference,'verified_decisions':len(result['decisions']),
                   'verified_hole_starts':len(seen),'arms':arms,'nonoptimal_or_invalid_decisions':failures,
                   'median_request_seconds':statistics.median(r['seconds'] for r in rows),
                   'max_request_seconds':max(r['seconds'] for r in rows),
                   'token_measurement':result['token_measurement'],
                   'native_provider_tokens_available':False,
                   'all_visible_reference_token_counts_recomputed':True,
                   'local_serving_manifest_unchanged':True})
    print(json.dumps({'audit':'passed','oracle_max_difference':difference,'arms':arms,
                      'issues':len(failures)},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    audit(args.result,args.output)
