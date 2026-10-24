"""Replay exact source packets and first judgments for matched source attention."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_sol64_grounded_oracle_semantics as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.finalize_ncm_sol64_bert_roundtrip import transitions
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def finalize():
    run.configure()
    run.lineage.require_judges()
    q = run.q
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError('fresh codec semantic aggregate/report required')
    for root in (run.PREP, run.PILOT, run.GRADE):
        run.native.vendor.terminal(root)
    cases = run.require_numeric()
    packets = {arm: [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts'][arm]}]) for r in cases] for arm in run.ARMS}
    proof = {'public_file_sha256': run.reference.PUBLIC_SHA256, 'exact_sources': run.reference.verify_public(cases)}
    if (q.read(run.PREP / 'cases.json') != cases or q.read(run.PREP / 'packets.json') != packets
            or q.read(run.PREP / 'public_proof.json') != proof):
        raise ValueError('complete original source/reconstruction packets differ')
    old = run.lineage.base_cache()
    pilot, cache = q.read(run.PILOT / 'judgment_cache.json'), q.read(run.GRADE / 'judgment_cache.json')
    if any(pilot.get(k) != v for k, v in old.items()) or any(cache.get(k) != v for k, v in pilot.items()):
        raise ValueError('a historical or pilot first judgment changed')
    run.lineage.audit_cache(cache)
    old_artifacts = q.read(run.lineage.CAMPAIGN)['artifacts']
    for entry in old.values():
        for path in entry['row']['attempts']:
            if old_artifacts.get(path) != run.file_hash(path):
                raise ValueError('a historical raw request changed')

    def replay(root, arm, role, count, decisions):
        scores = replay_scores(cases[:count], {arm: packets[arm][:count]}, decisions, arm, role)
        value = {**run.metric(scores), 'judge': q.JUDGES[role]}
        if (q.read(root / f'{role}_{arm}_case_scores.json') != scores
                or q.read(root / f'{role}_{arm}_result.json') != value):
            raise ValueError('first-response semantic score or metric differs')
        return scores, value

    metrics, joints, admitted, tables = {}, {}, {}, {}
    allowed, pilot_allowed = set(), set()
    for arm in run.ARMS:
        probes, _ = replay(run.PILOT, arm, 'secondary', 4, pilot)
        if not all(r['valid'] for r in probes):
            raise ValueError('every pilot judgment must be schema-valid')
        scores, value = replay(run.GRADE, arm, 'secondary', 64, cache)
        metrics[arm], tables[arm, 'secondary'] = {'secondary': value}, scores
        admitted[arm] = value['valid'] == 64 and value['passed'] >= 61
        joints[arm] = None
        if admitted[arm]:
            primary, metrics[arm]['primary'] = replay(run.GRADE, arm, 'primary', 64, cache)
            tables[arm, 'primary'] = primary
            if [r['id'] for r in scores] != [r['id'] for r in primary]:
                raise ValueError('same ordered64 cases required for joint gate')
            valid = all(a['valid'] and b['valid'] for a, b in zip(scores, primary))
            count = sum(a['passed'] is True and b['passed'] is True for a, b in zip(scores, primary))
            joints[arm] = {'cases': 64, 'joint_passed': count, 'all_valid': valid, 'qualified': valid and count >= 61}
        elif (run.GRADE / f'primary_{arm}_case_scores.json').exists():
            raise ValueError('primary graded an unadmitted codec arm')
        allowed |= {run.reference.first.packet_key(p, role) for p in packets[arm]
            for role in (('secondary', 'primary') if admitted[arm] else ('secondary',))}
        pilot_allowed |= {run.reference.first.packet_key(p, 'secondary') for p in packets[arm][:4]}
    if set(cache) - set(old) - allowed or set(pilot) - set(old) - pilot_allowed:
        raise ValueError('new judgments outside the registered two-arm scope appeared')
    qualified = [arm for arm in run.ARMS if joints[arm] is not None and joints[arm]['qualified']]
    classification = 'fixed64_grounded_oracle_semantically_qualified' if 'source_on' in qualified else 'fixed64_grounded_oracle_quality_failed'
    state = q.read(run.GRADE / 'result.json')
    if (state['status'] != classification or state['metrics'] != metrics or state['joint'] != joints
            or state['primary_admitted'] != admitted or state['qualified_codecs'] != qualified
            or state['student_task_accuracy'] is not None or state['student_trained']):
        raise ValueError('fixed complete-summary component decision differs')
    for root, current, prior in ((run.PILOT, pilot, old), (run.GRADE, cache, pilot)):
        result = q.read(root / 'result.json')
        if result['canonical_first_judgments'] != len(current) or result['inherited_first_judgments'] != len(prior):
            raise ValueError('preserved first counts differ')
    paired = {}
    for arm in run.ARMS:
        paired[arm] = {}
        for role in metrics[arm]:
            controls = {}
            for name, control in (('raw', run.rendering.teacher), ('bert', run.rendering)):
                path = control.GRADE / f'{role}_labels_case_scores.json'
                if not path.exists():
                    controls[name] = None
                    continue
                control_cases, control_packets = q.read(control.PREP / 'cases.json'), q.read(control.PREP / 'packets.json')
                scores = replay_scores(control_cases, control_packets, old, 'labels', role)
                if q.read(path) != scores:
                    raise ValueError('inherited raw/BERT control changed')
                controls[name] = transitions(scores, tables[arm, role])
            paired[arm][role] = controls
    source_attention_transitions = {
        role: transitions(tables['source_off', role], tables['source_on', role])
        if ('source_off', role) in tables and ('source_on', role) in tables else None
        for role in ('secondary', 'primary')}
    artifacts = {str(Path(__file__)): run.file_hash(__file__), str(run.lineage.CACHE): run.lineage.CACHE_SHA}
    for root in (run.PREP, run.PILOT, run.GRADE, run.NUMERIC_AUDIT):
        for path in root.rglob('*.json'):
            if 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    for entry in cache.values():
        for path in entry['row']['attempts']:
            artifacts[path] = run.file_hash(path)
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'classification': classification, 'metrics': metrics,
        'joint': joints, 'primary_admitted': admitted, 'qualified_codecs': qualified,
        'paired_controls_to_codec': paired, 'source_off_to_on_transitions': source_attention_transitions,
        'canonical_first_judgments': len(cache),
        'component_semantics_only': True, 'oracle_target_codes': True, 'source_prediction': False,
        'student_task_accuracy': None, 'confirmed': False, 'final_test_opened': False,
        'speed_tested': False, 'artifacts': artifacts}
    run.RESULTS.mkdir()
    path = run.RESULTS / 'campaign_results.json'
    run.write_json(path, campaign)
    table = ['| Codec | Judge | Whole-summary passes | Valid |', '| --- | --- | ---: | ---: |']
    for arm in run.ARMS:
        for role, value in metrics[arm].items():
            table.append(f"| {arm} | {role} | {value['passed']}/64 | {value['valid']}/64 |")
    run.REPORT.write_text('# Same-decoder source-attention oracle comparison\n\n'
        f'Hypothesis `{run.HYPOTHESIS}`. Classification **{classification}**.\n\n'
        f'Campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n'
        'Both arms received the same oracle codes and used the SAME fully trained native655 word decoder; '
        'only source attention was disabled or enabled. Original sources, complete128-position outputs '
        'and the native renderer were retained without truncation or repair. The separate '
        'numeric workflow used the pinned old native655 checkpoint on GPU0 and passed an independent '
        'CPU checkpoint-binding and rendering audit.\n\n'
        + '\n'.join(table) + '\n\n'
        + f'Same-case joint gates: `{joints}`. Qualified arms: `{qualified}`.\n\n'
        + f'Paired raw/BERT control transitions: `{paired}`.\n\n'
        + f'Matched source-off to source-on transitions: `{source_attention_transitions}`.\n\n'
        + f'Canonical first judgments retained: {len(cache)}.\n\n'
        'Semantic judges accept paraphrases and assess full-source faithfulness, salience, readability '
        'and the128 storage-token cap. High word-ID agreement does not establish this endpoint. '
        'These64 additional TRAIN-designated sources were excluded from previous gradients. The checkpoint is the older completed native655 model; the new FP16 conditioning candidate is a separate trial. '
        'This component result is not source-only summarization or a ceiling on a source-conditioned decoder. '
        'The two automated judges share a provider and are not independent human annotation. '
        'No sealed final source was opened or hashed; the95% student goal and speed remain unestablished.\n')
    print(json.dumps({'classification': classification, 'qualified_codecs': qualified,
        'canonical_first_judgments': len(cache), 'campaign_sha256': run.file_hash(path)}), flush=True)


def register():
    path = run.RESULTS / 'campaign_results.json'
    state = run.q.read(path)
    register_records(path, run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        'name': 'Same-word-decoder source-attention oracle control on64 summaries',
        'description': 'All64 unchanged first targets and identical oracle codes, SAME frozen native655 decoder weights with source attention off/on; complete-source semantic judgments, same-case61gate, all prior raw/BERT firsts retained; not source-only student accuracy.',
        'classification': state['classification'], 'component_semantics_only': True,
        'canonical_first_judgments': state['canonical_first_judgments'], 'student_task_accuracy_measured': False}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('finalize', 'register'))
    args = parser.parse_args()
    {'finalize': finalize, 'register': register}[args.stage]()
