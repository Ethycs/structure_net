"""Replay model-aware first judgments and the unchanged same-case64 label gate."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_sol64_bert_roundtrip as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def transitions(raw, current):
    if (len(raw) != 64 or len(current) != 64 or len({r['id'] for r in raw}) != 64
            or [r['id'] for r in raw] != [r['id'] for r in current]):
        raise ValueError('paired complete64 raw/rendered cases required')
    return {'cases': 64, 'raw_passed': sum(r['passed'] is True for r in raw),
        'rendered_passed': sum(r['passed'] is True for r in current),
        'retained_passes': sum(a['passed'] is True and b['passed'] is True for a, b in zip(raw, current)),
        'lost_passes': sum(a['passed'] is True and b['passed'] is False for a, b in zip(raw, current)),
        'gained_passes': sum(a['passed'] is False and b['passed'] is True for a, b in zip(raw, current)),
        'all_valid': all(a['valid'] and b['valid'] for a, b in zip(raw, current))}


def finalize():
    run.configure()
    q = run.q
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError('fresh Sol teacher semantic aggregate required')
    for root in (run.PREP, run.PILOT, run.GRADE):
        run.native.vendor.terminal(root)
        run.native.vendor.verify(q.read(root / 'producer.json'))
    cases = run.expected_cases()
    packets = {'labels': [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts']['labels']}]) for r in cases]}
    proof = {'public_file_sha256': run.reference.PUBLIC_SHA256, 'exact_sources': run.reference.verify_public(cases)}
    if (q.read(run.PREP / 'cases.json') != cases or q.read(run.PREP / 'packets.json') != packets
            or q.read(run.PREP / 'public_proof.json') != proof
            or q.read(run.PREP / 'split_proof.json') != run.split_guard(cases)):
        raise ValueError('fixed64 first-label/source/packet bindings differ')
    old = run.lineage.base_cache()
    pilot = q.read(run.PILOT / 'judgment_cache.json')
    cache = q.read(run.GRADE / 'judgment_cache.json')
    if any(pilot.get(k) != v for k, v in old.items()) or any(cache.get(k) != v for k, v in pilot.items()):
        raise ValueError('preceding or pilot first changed')
    run.lineage.audit_cache(cache)
    # Historical raw attempt bytes must match the previous registered aggregate.
    old_artifacts = q.read(run.lineage.CAMPAIGN)['artifacts']
    for entry in old.values():
        for path in entry['row']['attempts']:
            if old_artifacts.get(path) != run.file_hash(path):
                raise ValueError('historical registered raw request changed')

    def replay(root, role, count, decisions):
        expected = replay_scores(cases[:count], {'labels': packets['labels'][:count]}, decisions, 'labels', role)
        metric = {**run.metric(expected), 'judge': q.JUDGES[role]}
        if (q.read(root / f'{role}_labels_case_scores.json') != expected
                or q.read(root / f'{role}_labels_result.json') != metric):
            raise ValueError('first-response score or label-only metric replay differs')
        return expected, metric

    pilot_scores, _ = replay(run.PILOT, 'secondary', 4, pilot)
    if not all(r['valid'] for r in pilot_scores):
        raise ValueError('four actual valid BERT rendering pilot judgments required')
    secondary, secondary_metric = replay(run.GRADE, 'secondary', 64, cache)
    admitted = run.oracle.qualifies(secondary_metric)
    metrics, joint = {'secondary': secondary_metric}, None
    if admitted:
        primary, metrics['primary'] = replay(run.GRADE, 'primary', 64, cache)
        # Deliberately compute the intersection here from raw first scores.
        other = {r['id']: r for r in primary}
        if len(other) != 64 or set(other) != {r['id'] for r in secondary}:
            raise ValueError('same complete64 judge population required')
        valid = all(r['valid'] and other[r['id']]['valid'] for r in secondary)
        count = sum(r['passed'] is True and other[r['id']]['passed'] is True for r in secondary)
        joint = {'cases': 64, 'joint_passed': count, 'all_valid': valid, 'qualified': valid and count >= 61}
    elif (run.GRADE / 'primary_labels_case_scores.json').exists():
        raise ValueError('primary ran before secondary admission')
    qualified = admitted and joint['qualified']
    classification = 'fixed64_BERT_roundtrip_semantically_qualified' if qualified else 'fixed64_BERT_roundtrip_quality_failed'
    status = q.read(run.GRADE / 'result.json')
    if (status['status'] != classification or status['metrics'] != metrics or status['joint'] != joint
            or status['representation_qualified'] != bool(qualified) or status['primary_admitted'] != admitted
            or status['student_task_accuracy'] is not None or status['student_trained']):
        raise ValueError('locked qualification result differs')
    allowed = {run.reference.first.packet_key(p, role) for p in packets['labels']
        for role in (('secondary', 'primary') if admitted else ('secondary',))}
    pilot_allowed = {run.reference.first.packet_key(p, 'secondary') for p in packets['labels'][:4]}
    if set(cache) - set(old) - allowed or set(pilot) - set(old) - pilot_allowed:
        raise ValueError('judgments outside registered fixed64 scope appeared')
    for root, current, preceding in ((run.PILOT, pilot, old), (run.GRADE, cache, pilot)):
        state = q.read(root / 'result.json')
        if state['canonical_first_judgments'] != len(current) or state['inherited_first_judgments'] != len(preceding):
            raise ValueError('canonical first counts differ')
    paths = [p for root in (run.PREP, run.PILOT, run.GRADE) for p in root.rglob('*.json')
        if 'producer_source' not in p.parts]
    artifacts = {str(p): run.file_hash(p) for p in paths}
    artifacts[str(run.lineage.CACHE)] = run.lineage.CACHE_SHA
    artifacts[str(Path(__file__))] = run.file_hash(__file__)
    for entry in cache.values():
        for path in entry['row']['attempts']:
            artifacts[path] = run.file_hash(path)
    paired = {}
    raw_cases, raw_packets = q.read(run.teacher.PREP / 'cases.json'), q.read(run.teacher.PREP / 'packets.json')
    for role in metrics:
        raw = replay_scores(raw_cases, raw_packets, old, 'labels', role)
        if q.read(run.teacher.GRADE / f'{role}_labels_case_scores.json') != raw:
            raise ValueError('inherited raw control does not match its first judgments')
        current = q.read(run.GRADE / f'{role}_labels_case_scores.json')
        paired[role] = transitions(raw, current)
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'classification': classification,
        'representation_qualified': bool(qualified), 'primary_admitted': admitted, 'metrics': metrics, 'joint': joint,
        'canonical_first_judgments': len(cache), 'component_semantics_only': True,
        'student_task_accuracy': None, 'student_trained': False, 'confirmed': False, 'paired_raw_to_rendered': paired,
        'final_test_opened': False, 'speed_tested': False, 'artifacts': artifacts}
    run.RESULTS.mkdir()
    path = run.RESULTS / 'campaign_results.json'
    run.write_json(path, campaign)
    rows = ['| Judge | Semantic label passes | Valid |', '| --- | ---: | ---: |']
    for role, metric in metrics.items():
        rows.append(f"| {role} ({metric['judge']}) | {metric['passed']}/64 | {metric['valid']}/64 |")
    run.REPORT.write_text('# Native BERT rendering of64 new teacher summaries\n\n'
        f'Hypothesis `{run.HYPOTHESIS}`. Classification **{classification}**.\n\n'
        f'Campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n'
        'Every original first target was tokenized and rendered with the pinned native BERT tokenizer. '
        'No neural model, clipping, repair, candidate filtering or semantic reroll was used. '
        'The original raw summaries qualified62/64 jointly; their existing first judgments are reused.\n\n'
        + '\n'.join(rows) + '\n\n'
        + f'Same-case joint result: `{joint}`. Paired transitions: `{paired}`.\n\n'
        + f'All3350 preceding first decisions remain unchanged; current canonical cache: {len(cache)}.\n\n'
        'Judges assess complete-source faithfulness, salience and readability with the128 storage-token cap; '
        'character differences and exact-summary matches are not the semantic endpoint. '
        'These64 additional TRAIN-designated sources were gradient-disjoint from the original690 and128 DEV. '
        'This is a component rendering result, not source-only student accuracy or a ceiling on a source-conditioned decoder. '
        'The two automated judges share a provider; this is not independent human grading. '
        'No final source was opened or hashed; student task accuracy and speed remain unmeasured.\n')
    print(json.dumps({'classification': classification, 'representation_qualified': bool(qualified),
        'canonical_first_judgments': len(cache), 'campaign_sha256': run.file_hash(path)}), flush=True)


def register():
    path = run.RESULTS / 'campaign_results.json'
    state = run.q.read(path)
    register_records(path, run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        'name': 'Native BERT text-rendering semantics on64 new teacher summaries',
        'description': 'BERT tokenization/rendering of all64 original new teacher summaries, paired unchanged raw controls, unchanged128-token semantic rubric and same-case dual-judge61gate; no model training or source-only student result.',
        'classification': state['classification'], 'representation_qualified': state['representation_qualified'],
        'component_semantics_only': True, 'canonical_first_judgments': state['canonical_first_judgments'],
        'student_task_accuracy_measured': False}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('finalize', 'register'))
    args = parser.parse_args()
    {'finalize': finalize, 'register': register}[args.stage]()
