"""Source-grounded review of fixed saved labels; never student task accuracy."""
import argparse
from copy import deepcopy
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import sys

from experiments.structure_net import ncm_recovered_label_sample as sample
from experiments.structure_net import ncm_task_supervision_development as history
from experiments.structure_net import ncm_xsum_semantic_probe as protocol

data, base, q = protocol.data, protocol.base, protocol.q
first, qualified = protocol.first, protocol.qualified
ROOTS = {s: data.ROOT/f'20261027_recovered_label_review_{s}_v1' for s in ('prepare', 'pilot', 'grade', 'audit')}
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-label-review.md')
TEST = Path('tests/structure_net/test_ncm_recovered_label_review.py')
HYPOTHESIS = 'ncm-recovered-saved-label-review-v1'
PILOT_INDICES = (0, 1, 64, 65)
ROLES = ('secondary', 'primary')


def packets_for(cases):
    return [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts']['saved']}], role='secondary') for r in cases]


def paired_metrics(cases, scores):
    n = len(cases)
    if not n or set(scores) != set(ROLES) or any([s['id'] for s in scores[r]] != [c['id'] for c in cases] for r in ROLES):
        raise ValueError('both complete ordered judge populations required')
    values = {r: protocol.metric(scores[r]) for r in ROLES}
    both_valid = sum(a['valid'] and b['valid'] for a, b in zip(scores['secondary'], scores['primary']))
    passed = sum(a['passed'] is True and b['passed'] is True for a, b in zip(scores['secondary'], scores['primary']))
    known_fail = sum(a['passed'] is False or b['passed'] is False for a, b in zip(scores['secondary'], scores['primary']))
    unresolved = n-passed-known_fail
    disagreement = sum(a['valid'] and b['valid'] and a['passed'] != b['passed'] for a, b in zip(scores['secondary'], scores['primary']))
    claims = {}
    for role in ROLES:
        counts, digits = Counter(), Counter()
        for row in scores[role]:
            if row['valid']:
                for claim in row['evaluations'][0]['claims']:
                    counts[claim['verdict']] += 1
                    if re.search(r'\d', claim['claim']):
                        digits[claim['verdict']] += 1
        claims[role] = {'all_claims': dict(counts), 'claims_containing_digits': dict(digits)}
    return {'cases': n, 'per_judge': values, 'both_valid': both_valid, 'joint_passed': passed,
        'joint_known_failed': known_fail, 'joint_unresolved': unresolved,
        'joint_pass_rate': passed/n if both_valid == n else None,
        'joint_all_case_lower_bound': passed/n, 'joint_all_case_upper_bound': (passed+unresolved)/n,
        'pass_disagreements_among_both_valid': disagreement, 'claim_verdicts': claims}


def summarize(cases, scores):
    value = {'overall': paired_metrics(cases, scores), 'strata': {}}
    for stratum in ('long', 'short'):
        positions = [i for i, row in enumerate(cases) if row['stratum'] == stratum]
        value['strata'][stratum] = paired_metrics([cases[i] for i in positions],
            {role: [scores[role][i] for i in positions] for role in ROLES})
    return value


def configure():
    protocol.configure(); qualified.require_judges()
    if q.original.parse is not q.parse:
        raise ValueError('qualified conservative-failure parser must be active')


def receipt(root):
    value = data.read(root/'actual_tool_receipt.json')
    if value['actual_tool_exit_code'] != 0 or value['artifacts'].get(str(root/'result.json')) != data.sha(root/'result.json'):
        raise ValueError('actual successful artifact-bound parent required')
    data.verify(value['artifacts'])
    return data.read(root/'result.json')


def freeze(root, stage, paths, inherited):
    if root.exists():
        raise ValueError('fresh append-only review stage required')
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(Path('experiments').resolve()):
                files.add(path)
    inputs = [*paths, *qualified.judge_inputs(), Path('pixi.lock'), Path('pyproject.toml')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, path in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema': 'nal.ncm-recovered-label-review.v1', 'hypothesis_id': HYPOTHESIS, 'stage': stage,
        'config': {'judges': dict(q.JUDGES), 'rubric': q.RUBRIC, 'cases': 128, 'pilot_indices': list(PILOT_INDICES),
            'workers': 4, 'inherited_first_judgments': inherited, 'both_judges_unconditional': True,
            'label_quality_only': True, 'model_accuracy': None, 'training_admitted': False},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in dict.fromkeys(inputs)}}
    proof['fingerprint'] = base.fingerprint(proof); data.write(root/'producer.json', proof)
    return proof


def finish(root, proof, value):
    base.verify_proof(proof)
    artifacts = {str(p): data.sha(p) for p in root.rglob('*') if p.is_file() and 'producer_source' not in p.parts}
    result = {'schema': proof['schema'], 'hypothesis_id': HYPOTHESIS, **value,
        'application_task_accuracy': None, 'model_training': False, 'final_test_opened': False,
        'speed_tested': False, 'training_admitted': False, 'artifacts': artifacts}
    data.write(root/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


def prepare():
    configure(); state = receipt(sample.ROOT)
    if state['status'] != 'fixed_saved_label_review_sample_prepared' or state['cases'] != 128:
        raise ValueError('fixed full sample required')
    selected = data.read(sample.ROOT/'cases.json')
    cases = [{**r, 'pair_id': r['id'], 'id': r['review_case_id'], 'texts': {'saved': r['target_text']}} for r in selected]
    if (len(cases) != 128 or Counter(r['stratum'] for r in cases) != {'long': 64, 'short': 64}
            or len({r['source_component'] for r in cases}) != 128):
        raise ValueError('unchanged balanced independent source components required')
    _, _, _, _, cache, paths = history.current_history()
    packets = packets_for(cases)
    uncertain = {r['key'] for r in data.read(protocol.FRONTIER) if r['status'] == 'unknown_inflight_or_unstarted'}
    if {first.packet_key(p, r) for p in packets for r in ROLES} & uncertain:
        raise ValueError('uncertain prior requests cannot receive new budgets')
    inputs = [*paths, protocol.FRONTIER, *[sample.ROOT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'cases.json', 'selection.json')]]
    root = ROOTS['prepare']; proof = freeze(root, 'prepare', inputs, len(cache))
    data.write(root/'cases.json', cases); data.write(root/'exact_packets.json', packets)
    data.write(root/'judgment_cache.json', cache)
    finish(root, proof, {'status': 'recovered_label_review_prepared', 'cases': 128,
        'inherited_first_judgments': len(cache), 'new_judge_requests': 0})


def prepared():
    state = receipt(ROOTS['prepare']); base.verify_proof(data.read(ROOTS['prepare']/'producer.json'))
    cases = data.read(ROOTS['prepare']/'cases.json'); packets = data.read(ROOTS['prepare']/'exact_packets.json')
    cache = data.read(ROOTS['prepare']/'judgment_cache.json')
    if state['status'] != 'recovered_label_review_prepared' or len(cache) != state['inherited_first_judgments'] or packets != packets_for(cases):
        raise ValueError('fixed complete packets and inherited firsts required')
    qualified.audit_cache(cache)
    return cases, packets, cache


def grade(pilot):
    configure(); cases, packets, inherited = prepared(); cache = deepcopy(inherited)
    paths = [ROOTS['prepare']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'cases.json', 'exact_packets.json', 'judgment_cache.json')]
    if pilot:
        cases, packets = [cases[i] for i in PILOT_INDICES], [packets[i] for i in PILOT_INDICES]
    else:
        checked = receipt(ROOTS['pilot']); base.verify_proof(data.read(ROOTS['pilot']/'producer.json'))
        if checked['status'] != 'recovered_label_judge_pilot_passed':
            raise ValueError('eight valid pilot outcomes required')
        previous = data.read(ROOTS['pilot']/'judgment_cache.json'); protocol.preserve(inherited, previous)
        for key, entry in previous.items():
            first.remember(cache, key, entry)
        paths += [ROOTS['pilot']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'judgment_cache.json')]
    root = ROOTS['pilot' if pilot else 'grade']; proof = freeze(root, 'pilot' if pilot else 'grade', paths, len(inherited))
    data.write(root/'exact_packets.json', packets)
    request_metrics = {}
    with (root/'new_firsts.jsonl').open('x') as journal:
        for role in ROLES:
            request_metrics[role] = protocol.grade_arm(root, cases, packets, cache, 'saved', role, journal)
    scores = {role: data.read(root/f'{role}_saved_scores.json') for role in ROLES}
    metrics = summarize(cases, scores)
    protocol.preserve(inherited, cache); data.write(root/'judgment_cache.json', cache)
    status = ('recovered_label_judge_pilot_passed' if metrics['overall']['both_valid'] == 4 else 'recovered_label_judge_pilot_failed') if pilot else 'recovered_saved_labels_graded'
    finish(root, proof, {'status': status, 'cases': len(cases), 'inherited_first_judgments': len(inherited),
        'canonical_first_judgments': len(cache), 'metrics': metrics, 'requests': request_metrics,
        'new_judge_requests': sum(v['new_requests'] for v in request_metrics.values())})


def audit():
    configure(); cases, packets, inherited = prepared()
    state = receipt(ROOTS['grade']); receipt(ROOTS['pilot'])
    for stage in ('pilot', 'grade'):
        base.verify_proof(data.read(ROOTS[stage]/'producer.json'))
    cache = data.read(ROOTS['grade']/'judgment_cache.json')
    protocol.preserve(inherited, cache); protocol.preserve(data.read(ROOTS['pilot']/'judgment_cache.json'), cache)
    expected = {first.packet_key(p, r) for p in packets for r in ROLES}
    if set(cache) != set(inherited) | expected or len(cache) != state['canonical_first_judgments']:
        raise ValueError('exact union of fixed requests and all inherited firsts required')
    qualified.audit_cache(cache)
    scores = {role: protocol.score_rows(cases, packets, cache, 'saved', role) for role in ROLES}
    for role in ROLES:
        if scores[role] != data.read(ROOTS['grade']/f'{role}_saved_scores.json'):
            raise ValueError('saved score replay differs')
    metrics = summarize(cases, scores)
    if metrics != state['metrics']:
        raise ValueError('complete paired metric replay differs')
    paths = [ROOTS[s]/n for s in ('prepare', 'pilot', 'grade') for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [ROOTS['grade']/n for n in ('exact_packets.json', 'judgment_cache.json', 'primary_saved_scores.json', 'secondary_saved_scores.json')]
    root = ROOTS['audit']; proof = freeze(root, 'raw_firsts_and_metric_replay', paths, len(inherited))
    finish(root, proof, {'status': 'recovered_saved_label_firsts_and_metrics_replayed', 'cases': 128,
        'inherited_first_judgments': len(inherited), 'canonical_first_judgments': len(cache),
        'new_judge_requests': 0, 'metrics': metrics})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=tuple(ROOTS)); args = parser.parse_args()
    root = ROOTS[args.stage]; existed = root.exists()
    try:
        with (data.ROOT/'recovered_label_review.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.stage == 'prepare': prepare()
            elif args.stage == 'audit': audit()
            else: grade(args.stage == 'pilot')
    except Exception as error:
        if root.exists() and not existed:
            data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
