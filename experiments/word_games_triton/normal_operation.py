"""Run ordinary live WORD_GAMES tasks and a dependent workflow on Triton."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import requests

from experiments.word_games.backend import vendor
from experiments.word_games.campaign import evaluate_case, deterministic_checks, paired_checks
from experiments.word_games.common import read, sha, write, canonical
from experiments.word_games.firewall import POLICY, exchange
from experiments.word_games.metaphor import select, Metaphor
from experiments.word_games.native_speed import aggregate
from experiments.word_games.qwen_runtime import memory
from experiments.word_games.scenarios import cases, setup
from . import deploy
from .validate import Client

HYPOTHESIS_ID = 'word-games-normal-operation-v1'
PLAN = Path('docs/07 - Status Reports/2026-11-01_word-games-normal-operation-plan.md')
INDEPENDENT = ('normal_reserve', 'changed_due', 'approved_annotation', 'cancel')


def workflow():
    reserve = {'op': 'reserve', 'loan_id': 'loan:bronze', 'gallery_id': 'gal:north',
               'slot': 'N1', 'due': '2027-01-10', 'approval_id': None}
    transfer = {k: v for k, v in reserve.items() if k != 'gallery_id'}
    transfer.update(op='transfer', to_gallery_id='gal:north', due='2027-01-11')
    items = [
        ('reserve_heron', 'Reserve Bronze Heron (loan:bronze) in North Gallery N1 through 2027-01-10, with no approval.', reserve),
        ('reserve_clock', 'Reserve Bronze Clock (loan:clock) in North Gallery N2 through 2027-01-10, with no approval.',
         {**reserve, 'loan_id': 'loan:clock', 'slot': 'N2'}),
        ('extend_heron', 'Extend the current Bronze Heron (loan:bronze) reservation through 2027-01-11, keeping North Gallery N1 and no approval.', transfer),
        ('annotate_heron', 'Append exactly "Condition checked" to loan:bronze.',
         {'op': 'annotate', 'loan_id': 'loan:bronze', 'text': 'Condition checked'}),
        ('return_clock', 'End the existing loan:clock display and return it to home custody, leaving its home slot unoccupied.',
         {'op': 'cancel', 'loan_id': 'loan:clock'}),
        ('move_heron', 'Move the existing loan:bronze display from N1 to the now-free N2 in North Gallery. Keep its return date 2027-01-11 and no approval.',
         {**transfer, 'slot': 'N2'}),
        ('return_heron', 'End the existing loan:bronze display and return it to home custody, leaving its home slot unoccupied.',
         {'op': 'cancel', 'loan_id': 'loan:bronze'}),
    ]
    return [{'id': name, 'source': source, 'expected_operations': [operation],
             'expected_decision': 'applied', 'stale': False,
             'canary': f'WG_NORMAL_PRIVATE_{i}_8bd13e46'} for i, (name, source, operation) in enumerate(items)]


def context_for(engine, case):
    return engine.issue_context(actor='curator', scope=['gal:north'], loan_ids=list(engine.loans),
                                allowed_ops=list(POLICY.operations), event_id=case['id'])


def expected_state(engine, case):
    oracle = vendor().Engine.fixture()
    for name, value in engine.snapshot().items():
        setattr(oracle, name, copy.deepcopy(value))
    oracle.rebuild_indexes()
    context = context_for(oracle, case)
    oracle.apply(vendor().proposal(context, case['expected_operations'], case['id']),
                 expected_context_id=context['context_id'])
    return oracle.snapshot()


def check_expected_workflow():
    engine = vendor().Engine.fixture()
    metaphor = Metaphor('library', 'book', 'branch', 'shelf', 'shelve', 'return', 'permit', 'due date')
    for case in workflow():
        context = context_for(engine, case)
        packet, _ = exchange(engine, vendor().proposal(context, case['expected_operations'], case['id']),
                             context['context_id'], metaphor)
        if packet['decision'] != 'applied':
            raise ValueError('invalid expected workflow: ' + case['id'])
    if engine.revision != 7 or any(l['reservation'] for l in engine.loans.values()):
        raise ValueError('workflow did not return to empty occupancy')
    return {'steps': 7, 'final_revision': engine.revision, 'all_expected_operations_valid': True}


def evaluate(client, engine, context, case, out, metaphor):
    expected = expected_state(engine, case)
    write(out / 'expected_state.json', expected)
    # evaluate_case owns its output directory, so place the oracle beside it.
    result = evaluate_case(client, case, out / 'live', metaphor, engine_context=(engine, context))
    result['backend_state_correct'] = engine.snapshot() == expected
    result['joint_structured_success'] = (result['sender_operation_match'] and result['boundary_outcome_match']
        and result['receiver_state_correct'] and result['backend_state_correct'] and result['invariants']
        and not any(result[k] for k in ('canary_in_receiver_input', 'canary_in_receiver_output', 'forbidden_source_id_in_packet')))
    write(out / 'result.json', result)
    return result


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(deploy.ROOT / 'manifest.json')
    runtime = read(deploy.ROOT / 'models' / deploy.MODEL / '1/runtime.json')
    if runtime.get('public_prefixes') or manifest['profile'] != 'free_whole1024':
        raise ValueError('validated uncached serving profile required')
    if deploy.weight_identity() != manifest['verified_weight_identity']:
        raise ValueError('weight identity changed')
    for f, h in manifest['repository_hashes'].items():
        if sha(deploy.ROOT / f) != h:
            raise ValueError('serving repository changed')
    for f, h in manifest['native_runtime_hashes'].items():
        if sha(deploy.target.cpu.RUNTIME / f) != h:
            raise ValueError('native runtime changed')
    requests.get(deploy.URL + f'/v2/models/{deploy.MODEL}/ready', timeout=10).raise_for_status()
    sources = sorted([*Path('experiments/word_games').rglob('*.py'), *Path('experiments/word_games_triton').glob('*.py'),
                      PLAN, Path('experiments/word_games/vendor/provenance.json'), Path('pixi.lock')])
    bindings = {str(p): sha(p) for p in sources}
    selected = [next(c for c in cases() if c['id'] == name) for name in INDEPENDENT]
    write(out / 'producer.json', {'hypothesis_id': HYPOTHESIS_ID, 'source_hashes': bindings,
        'manifest': manifest, 'runtime': runtime, 'independent_cases': selected,
        'workflow': workflow(), 'policy': asdict(POLICY), 'fresh_live_calls': True,
        'original_config_changed': False, 'expected_answers_visible_to_model': False})
    for p in sources:
        target = out / 'producer_source' / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write(out / 'workflow_preflight.json', check_expected_workflow())
    client, independent, sequential = Client(), [], []
    error, selected_metaphor = None, None
    started = time.monotonic()
    try:
        selected_metaphor = select(client, out / 'metaphor')
        write(out / 'chosen_metaphor.json', asdict(selected_metaphor))
        write(out / 'deterministic_checks.json', deterministic_checks(selected_metaphor))
        write(out / 'paired_private_checks.json', paired_checks(selected_metaphor))
        for group, tasks in [('independent', selected), ('workflow', workflow())]:
            engine = vendor().Engine.fixture() if group == 'workflow' else None
            for case in tasks:
                current_memory = memory()
                write(out / group / case['id'] / 'memory.json', current_memory)
                if any(int(line.rsplit(',', 1)[1].strip().split()[0]) < 384
                       for line in current_memory.strip().splitlines()[1:]):
                    raise RuntimeError('GPU headroom below 384 MiB')
                if group == 'independent':
                    engine, context = setup(case)
                else:
                    context = context_for(engine, case)
                result = evaluate(client, engine, context, case, out / group / case['id'], selected_metaphor)
                (independent if group == 'independent' else sequential).append(result)
                print(json.dumps({'group': group, 'id': case['id'], 'joint': result['joint_structured_success'],
                    'sender': result['sender_operation_match'], 'state': result['backend_state_correct'],
                    'receiver': result['receiver_state_correct'], 'decision': result['decision']}), flush=True)
                if group == 'workflow' and not result['joint_structured_success']:
                    break
        if any(sha(p) != h for p, h in bindings.items()):
            raise ValueError('producing source changed')
    except BaseException as exc:
        error = repr(exc)
        write(out / 'failure.json', {'error': error})
        raise
    finally:
        ready = requests.get(deploy.URL + f'/v2/models/{deploy.MODEL}/ready', timeout=10).status_code == 200
        all_results = independent + sequential
        write(out / 'campaign_results.json', {'hypothesis_id': HYPOTHESIS_ID, 'completed': error is None,
            'error': error, 'planned_events': 11, 'completed_events': len(all_results),
            'joint_structured_successes': sum(r['joint_structured_success'] for r in all_results),
            'independent': independent, 'workflow': sequential,
            'unexecuted_workflow_events': [c['id'] for c in workflow()[len(sequential):]],
            'model_selected_metaphor': asdict(selected_metaphor) if selected_metaphor else None,
            'totals': aggregate(client.rows) if client.rows else None,
            'wall_seconds': time.monotonic() - started, 'serving_ready': ready,
            'manifest_unchanged': manifest == read(deploy.ROOT / 'manifest.json'),
            'confirmed': False, 'population_accuracy_established': False, 'independent_semantic_judge': False,
            'response_hashes': {str(p): sha(p) for p in sorted(out.rglob('response.json')) if 'producer_source' not in str(p)}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
