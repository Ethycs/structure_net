"""Independent lookup/move/return probes after the recorded workflow failure."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import requests

from experiments.word_games.common import read, sha, write
from experiments.word_games.backend import vendor
from experiments.word_games.metaphor import select
from experiments.word_games.native_speed import aggregate
from experiments.word_games.qwen_runtime import memory
from .normal_operation import evaluate, context_for, expected_state
from .validate import Client
from . import deploy

PLAN = Path('docs/07 - Status Reports/2026-11-01_word-games-normal-operation-followup.md')


def fixtures():
    reserve = {'op': 'reserve', 'loan_id': 'loan:bronze', 'gallery_id': 'gal:north',
               'slot': 'N1', 'due': '2027-01-10', 'approval_id': None}
    for i, name in enumerate(('lookup_title', 'move_placed', 'return_last')):
        engine = vendor().Engine.fixture()
        if name == 'lookup_title':
            engine.loans['loan:clock']['title'] = 'Copper Timekeeper'
            engine.rebuild_indexes()
            source = 'Reserve Copper Timekeeper in North Gallery N2 through 2027-01-10, with no approval.'
            operation = {**reserve, 'loan_id': 'loan:clock', 'slot': 'N2'}
        else:
            setup = context_for(engine, {'id': 'fixture'})
            seed = {**reserve, 'slot': 'N1' if name == 'move_placed' else 'N2'}
            engine.apply(vendor().proposal(setup, [seed], 'fixture'), expected_context_id=setup['context_id'])
            if name == 'move_placed':
                source = 'Move the existing loan:bronze display from N1 to N2 in North Gallery, keeping its return date 2027-01-10 and no approval.'
                operation = {'op': 'transfer', 'loan_id': 'loan:bronze', 'to_gallery_id': 'gal:north',
                             'slot': 'N2', 'due': '2027-01-10', 'approval_id': None}
            else:
                source = 'End the existing loan:bronze display and return it to home custody, leaving its home slot unoccupied.'
                operation = {'op': 'cancel', 'loan_id': 'loan:bronze'}
        yield engine, {'id': name, 'source': source, 'expected_operations': [operation],
                       'expected_decision': 'applied', 'stale': False, 'canary': f'WG_NORMAL_FOLLOWUP_PRIVATE_{i}_d81c2f'}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(deploy.ROOT / 'manifest.json')
    if manifest['profile'] != 'free_whole1024' or deploy.weight_identity() != manifest['verified_weight_identity']:
        raise ValueError('expected serving model required')
    runtime = read(deploy.ROOT / 'models' / deploy.MODEL / '1/runtime.json')
    if runtime.get('public_prefixes'):
        raise ValueError('uncached baseline required')
    for f, h in manifest['repository_hashes'].items():
        if sha(deploy.ROOT / f) != h:
            raise ValueError('prepared repository changed')
    sources = sorted([*Path('experiments/word_games').rglob('*.py'), *Path('experiments/word_games_triton').glob('*.py'), PLAN])
    bindings = {str(p): sha(p) for p in sources}
    prepared = list(fixtures())
    for engine, case in prepared:
        expected_state(engine, case)
    write(out / 'producer.json', {'hypothesis_id': 'word-games-normal-operation-followup-v1',
        'source_hashes': bindings, 'manifest': manifest, 'cases': [c for e, c in prepared],
        'purpose': 'independent functionality coverage; does not amend failed main workflow'})
    for p in sources:
        dst = out / 'producer_source' / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    client, results, error = Client(), [], None
    try:
        metaphor = select(client, out / 'metaphor')
        write(out / 'chosen_metaphor.json', asdict(metaphor))
        for engine, case in prepared:
            current_memory = memory()
            write(out / 'cases' / case['id'] / 'memory.json', current_memory)
            if any(int(line.rsplit(',', 1)[1].strip().split()[0]) < 384 for line in current_memory.strip().splitlines()[1:]):
                raise RuntimeError('GPU headroom below 384 MiB')
            result = evaluate(client, engine, context_for(engine, case), case, out / 'cases' / case['id'], metaphor)
            results.append(result)
            print(json.dumps({'id': case['id'], 'joint': result['joint_structured_success'],
                              'sender_turns': result['sender']['turns']}), flush=True)
        if any(sha(p) != h for p, h in bindings.items()):
            raise ValueError('producing source changed')
    except BaseException as exc:
        error = repr(exc)
        write(out / 'failure.json', {'error': error})
        raise
    finally:
        ready = requests.get(deploy.URL + f'/v2/models/{deploy.MODEL}/ready', timeout=10).status_code == 200
        write(out / 'campaign_results.json', {'hypothesis_id': 'word-games-normal-operation-followup-v1',
            'completed': error is None, 'error': error, 'results': results,
            'joint_structured_successes': sum(r['joint_structured_success'] for r in results),
            'completed_events': len(results), 'planned_events': 3,
            'totals': aggregate(client.rows) if client.rows else None, 'serving_ready': ready,
            'manifest_unchanged': manifest == read(deploy.ROOT / 'manifest.json'),
            'response_hashes': {str(p): sha(p) for p in out.rglob('response.json') if 'producer_source' not in str(p)}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    run(p.parse_args().output)
