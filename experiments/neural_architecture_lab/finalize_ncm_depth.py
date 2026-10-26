"""Replay complete-summary semantic judgments for the matched NCM depth study."""
import argparse
import json
from pathlib import Path
import sys

from experiments.structure_net import ncm_depth_development as development
from experiments.neural_architecture_lab.finalize_ncm_broad_generator import classify
from experiments.neural_architecture_lab.finalize_ncm_onpolicy_residual import paired_passes

study = development.study
BASE = study.broad.runtime.ROOT
ROOT = BASE/'20261026_ncm_depth_results_v1'
ARMS = {d: BASE/f'20261026_ncm_depth_{d}_semantic_replay_v1' for d in (12, 24)}
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-depth-reporting-plan.md')
REPORT = Path('docs/08 - Analysis/2026-10-26_ncm-depth.md')
HYPOTHESIS = 'ncm-generator-depth-v1'
TEST = Path('tests/structure_net/test_ncm_depth_finalizer.py')
REGISTRAR = Path('experiments/neural_architecture_lab/register_ncm_depth.py')


def source_files():
    paths = {Path(__file__).resolve(), TEST.resolve(), REGISTRAR.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(Path('experiments').resolve()):
                paths.add(path)
    return sorted(paths)


def selected_candidate(arms):
    if set(arms) != {'12', '24'}:
        raise ValueError('both matched depths required')
    qualified = []
    for depth, value in arms.items():
        expected = classify(value['valid'], value['passed'], value['primary_admitted'], value['joint'])
        if value['classification'] != expected or value['DEV_qualified'] != (expected == 'DEV_qualified'):
            raise ValueError('candidate qualification differs from fixed semantic gate')
        accuracy = value['passed']/128 if value['valid'] == 128 else None
        if value['task_accuracy'] != accuracy:
            raise ValueError('semantic accuracy must retain the full128 denominator')
        if value['DEV_qualified']:
            accuracy = sum(v['joint_passed'] for v in value['joint'].values())/128
            if value['joint_task_accuracy'] != accuracy:
                raise ValueError('joint accuracy differs from same-case decisions')
            qualified.append(depth)
    return max(qualified, key=lambda d: (arms[d]['joint_task_accuracy'], d == '12')) if qualified else None


def comparison_classification(arms):
    candidate = selected_candidate(arms)
    unresolved = any(v['classification'] == 'DEV_transport_invalid' for v in arms.values())
    return ('DEV_candidate_qualified' if candidate else 'DEV_comparison_transport_invalid' if unresolved
            else 'DEV_semantic_partial_progress' if any(v['passed'] for v in arms.values())
            else 'DEV_both_semantic_quality_failed')


def paired_decisions(control, deeper):
    value = paired_passes(control, deeper)
    value['control_only'] = value.pop('fixed_only')
    value['deeper_only'] = value.pop('residual_only')
    value['accuracy_difference_depth24_minus_depth12'] = value.pop('accuracy_difference_residual_minus_fixed')
    return value


def receipt(root, required=('producer.json', 'result.json')):
    value = study.broad.catalog.read(root/'actual_tool_receipt.json')
    if value['actual_tool_exit_code'] != 0:
        raise ValueError('actual successful tool completion required: '+str(root))
    if any(value['artifacts'].get(str(root/name)) != study.broad.file_hash(root/name) for name in required):
        raise ValueError('actual receipt must bind each required artifact')
    for path, expected in value['artifacts'].items():
        if study.broad.file_hash(path) != expected:
            raise ValueError('tool-receipted artifact changed')


def replay(depth):
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_native_development import replay_outputs
    destination = ARMS[depth]
    if destination.exists():
        raise ValueError('fresh append-only semantic replay required')
    training, evaluation = development.configure(depth)
    q, old, lineage = training.q, evaluation.old, evaluation.lineage
    lineage.configure()
    train_root, roots = study.ROOTS[depth, 'train'], development.roots(depth)
    audit = development.audit_training(training, depth)
    trained = q.read(train_root/'result.json')
    environment = q.read(train_root/'producer.json')['environment']
    pilot_train = q.read(study.ROOTS[depth, 'pilot']/'result.json')
    if (trained['frozen_hashes'] != pilot_train['frozen_hashes'] or not pilot_train['strict_resume_passed']
            or trained['initial_model_sha256'] != pilot_train['initial_model_sha256']
            or not trained['initial_predictions_identical'] or trained['task_accuracy'] is not None
            or trained['final_test_opened']):
        raise ValueError('parent parity, frozen tensors or training boundaries differ')
    for root in (train_root, *roots.values()):
        receipt(root)
        training.preparation.completed(root)
        proof = q.read(root/'producer.json')
        if proof['fingerprint'] != q.digest({k: v for k, v in proof.items() if k != 'fingerprint'}):
            raise ValueError('producer fingerprint differs')
        if proof['environment'] != environment or proof['depth'] != depth:
            raise ValueError('matched runtime or selected depth differs')
    cohorts, manifest, split = evaluation.sources()
    rows = [r for c in old.COHORTS for r in cohorts[c]]
    outputs = q.read(roots['outputs']/'outputs.json')
    probes = q.read(roots['inference_pilot']/'outputs.json')
    tokenizer = training.native.tokenizer()
    replay_outputs(outputs, rows, manifest, tokenizer)
    selected_probes = old.inference_probes(cohorts, manifest)
    replay_outputs(probes, selected_probes, manifest, tokenizer)
    by_id = {r['id']: r for r in outputs}
    if any(by_id[r['id']] != r for r in probes):
        raise ValueError('full predictions differ from inference pilot')
    for stage, count in (('inference_pilot', len(selected_probes)), ('outputs', 128)):
        result = q.read(roots[stage]/'result.json')
        if (result['model_sha256'] != trained['final_model_sha256'] or result['cases'] != count
                or result['outputs_sha256'] != study.broad.file_hash(roots[stage]/'outputs.json')):
            raise ValueError('selected checkpoint or complete output binding differs')
    cases = {c: [{**r, 'texts': {'student': by_id[r['id']]['text']}} for r in cohorts[c]] for c in old.COHORTS}
    packets = {c: {'student': [q.packet(r['id'], r['source'],
        [{'candidate_id': 'summary', 'text': r['texts']['student']}]) for r in cases[c]]} for c in old.COHORTS}
    inherited = lineage.base_cache()
    cache, pilot_cache = [q.read(roots[s]/'judgment_cache.json') for s in ('grade', 'judge_pilot')]
    allowed_pilot = {old.oracle.first.packet_key(p, 'secondary') for c in old.COHORTS
                     for p in packets[c]['student'][:2]}
    development.previous.preserve_firsts(inherited, pilot_cache, allowed_new=allowed_pilot)
    development.previous.preserve_firsts(pilot_cache, cache)
    lineage.audit_cache(cache)
    for stage in ('judge_pilot', 'grade'):
        proof = q.read(roots[stage]/'producer.json')
        result = q.read(roots[stage]/'result.json')
        if (q.read(roots[stage]/'cases.json') != cases or q.read(roots[stage]/'exact_packets.json') != packets
                or proof['depth'] != depth or proof['config']['rubric'] != q.RUBRIC
                or proof['config']['judges'] != q.JUDGES or proof['config']['inherited_firsts'] != len(inherited)
                or result['producer_fingerprint'] != proof['fingerprint']):
            raise ValueError('source-only judge packets, rubric or first lineage differ')
    scores, metrics = {}, {c: {} for c in old.COHORTS}
    pilot_metrics = {c: {} for c in old.COHORTS}

    def decisions(stage, cohort, role, count, firsts):
        expected = replay_scores(cases[cohort][:count], {'student': packets[cohort]['student'][:count]},
                                 firsts, 'student', role)
        directory = roots[stage]/cohort
        reported = q.read(directory/f'{role}_student_result.json')
        if (q.read(directory/f'{role}_student_case_scores.json') != expected
                or any(reported.get(k) != v for k, v in old.task_metric(expected).items())):
            raise ValueError('raw first judgments and semantic metrics differ')
        return expected, reported

    for cohort in old.COHORTS:
        pilot_scores, pilot_metrics[cohort]['secondary'] = decisions('judge_pilot', cohort, 'secondary', 2, pilot_cache)
        if not all(r['valid'] for r in pilot_scores):
            raise ValueError('valid four-case judge pilot required')
        scores[cohort], metrics[cohort]['secondary'] = decisions('grade', cohort, 'secondary', 64, cache)
    pilot_result = q.read(roots['judge_pilot']/'result.json')
    if (not pilot_result['valid_pilot'] or pilot_result['metrics'] != pilot_metrics
            or pilot_result['primary_admitted'] or pilot_result['joint'] is not None
            or pilot_result['DEV_qualified'] or pilot_result['task_accuracy'] is not None
            or pilot_result['canonical_first_judgments'] != len(pilot_cache)
            or pilot_result['inherited_first_judgments'] != len(inherited)):
        raise ValueError('judge pilot aggregate or lineage differs')
    admitted = old.secondary_admitted({c: metrics[c]['secondary'] for c in old.COHORTS})
    joints = None
    if admitted:
        joints = {}
        for c in old.COHORTS:
            primary, metrics[c]['primary'] = decisions('grade', c, 'primary', 64, cache)
            joints[c] = old.joint_gate(scores[c], primary)
    elif any((roots['grade']/c/'primary_student_case_scores.json').exists() for c in old.COHORTS):
        raise ValueError('primary judge ran without admission')
    roles = ('secondary', 'primary') if admitted else ('secondary',)
    allowed = {old.oracle.first.packet_key(p, role) for c in old.COHORTS
               for p in packets[c]['student'] for role in roles}
    development.previous.preserve_firsts(inherited, cache, allowed_new=allowed)
    valid = sum(metrics[c]['secondary']['valid'] for c in old.COHORTS)
    passed = sum(metrics[c]['secondary']['passed'] for c in old.COHORTS)
    classification = classify(valid, passed, admitted, joints)
    expected = {'metrics': metrics, 'primary_admitted': admitted, 'joint': joints,
        'DEV_qualified': classification == 'DEV_qualified', 'task_accuracy': passed/128 if valid == 128 else None,
        'canonical_first_judgments': len(cache), 'inherited_first_judgments': len(inherited),
        'final_test_opened': False, 'speed_tested': False, 'metric': 'complete_summary_semantic_pass',
        'task_accuracy_judge': q.JUDGES['secondary']}
    grade = q.read(roots['grade']/'result.json')
    if any(grade.get(k) != v for k, v in expected.items()):
        raise ValueError('reported full DEV aggregate or qualification differs')
    paths = [*source_files(), PLAN, study.PLAN, study.ROOTS[depth, 'pilot']/'result.json']
    paths += [train_root/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'trace.jsonl')]
    for root in roots.values():
        paths += [p for p in root.rglob('*.json') if 'producer_source' not in p.parts]
    value = {'schema': 'nal.ncm-depth-semantic-replay.v1', 'hypothesis_id': HYPOTHESIS,
        'depth': depth, 'classification': classification, 'cases': 128, 'valid': valid, 'passed': passed, **expected,
        'joint_task_accuracy': sum(v['joint_passed'] for v in joints.values())/128
            if joints and all(v['all_valid'] for v in joints.values()) else None,
        'training_audit': audit, 'training_seconds': trained['training_seconds'],
        'trainable_parameters': trained['trainable_parameters'], 'peak_allocated_bytes': trained['peak_allocated_bytes'],
        'checkpoint_sha256': trained['checkpoint_sha256'], 'frozen_hashes': trained['frozen_hashes'],
        'split_proof': split, 'artifacts': {str(p): study.broad.file_hash(p) for p in dict.fromkeys(paths)}}
    destination.mkdir()
    study.broad.write_json(destination/'scores.json', scores)
    study.broad.write_json(destination/'result.json', value)
    print(json.dumps({k: value[k] for k in ('depth', 'classification', 'cases', 'valid', 'passed', 'task_accuracy')}), flush=True)


def aggregate():
    if ROOT.exists() or REPORT.exists():
        raise ValueError('fresh append-only depth aggregate and report required')
    read, sha = study.broad.catalog.read, study.broad.file_hash
    arms, scores, paths = {}, {}, [*source_files(), PLAN, study.PLAN]
    for depth in (12, 24):
        root = ARMS[depth]
        receipt(root, ('result.json', 'scores.json'))
        value = read(root/'result.json')
        for path, expected in value['artifacts'].items():
            if sha(path) != expected:
                raise ValueError('replayed semantic artifact changed')
        arms[str(depth)] = {k: v for k, v in value.items() if k != 'artifacts'}
        scores[str(depth)] = read(root/'scores.json')
        paths += [root/n for n in ('result.json', 'scores.json', 'actual_tool_receipt.json')]
    if (arms['12']['inherited_first_judgments'] != 4179
            or arms['24']['inherited_first_judgments'] != arms['12']['canonical_first_judgments']
            or arms['12']['split_proof'] != arms['24']['split_proof']
            or arms['12']['frozen_hashes'] != arms['24']['frozen_hashes']):
        raise ValueError('matched split, frozen tensors or complete sibling first lineage differs')
    caches = [read(development.roots(d)['grade']/'judgment_cache.json') for d in (12, 24)]
    development.previous.preserve_firsts(caches[0], caches[1])
    orders = []
    for depth in (12, 24):
        trace = [json.loads(line) for line in (study.ROOTS[depth, 'train']/'trace.jsonl').read_text().splitlines()]
        orders.append([[(m['ids'], m['changes']) for m in r['microbatches']] for r in trace])
    if orders[0] != orders[1] or len(orders[0]) != 1024:
        raise ValueError('matched data and corruption schedules differ')
    cohorts = tuple(scores['12'])
    if set(cohorts) != set(scores['24']):
        raise ValueError('paired cohorts differ')
    paired = {c: paired_decisions(scores['12'][c], scores['24'][c]) for c in cohorts}
    paired['all128'] = paired_decisions([r for c in cohorts for r in scores['12'][c]],
                                       [r for c in cohorts for r in scores['24'][c]])
    candidate = selected_candidate(arms)
    unresolved = any(v['classification'] == 'DEV_transport_invalid' for v in arms.values())
    classification = comparison_classification(arms)
    campaign = {'schema': 'nal.ncm-depth-results.v1', 'hypothesis_id': HYPOTHESIS,
        'classification': classification, 'conformance': 'EXPLORATORY_UNDERPOWERED', 'arms': arms,
        'paired_secondary_decisions': paired, 'selected_candidate': candidate,
        'comparison_has_unresolved_judgments': unresolved, 'cases_per_arm': 128,
        'independent_training_seeds': 1, 'same_parent_checkpoint': study.parent.PARENT_SHA,
        'same_sampling': True, 'identity_initialized_extra_blocks': True,
        'source_only': True, 'task_metric': 'complete_summary_semantic_pass',
        'inherited_first_judgments': 4179, 'canonical_first_judgments': len(caches[1]),
        'all_historical_and_pilot_firsts_preserved': True, 'final_test_opened': False,
        'final_task_goal_achieved': False, 'speed_tested': False,
        'artifacts': {str(p): sha(p) for p in dict.fromkeys(paths)}}
    ROOT.mkdir()
    study.broad.write_json(ROOT/'campaign_results.json', campaign)
    lines = ['# NCM generator depth: 12 versus 24 blocks', '',
        f'Classification: `{classification}`. One matched training seed; final320 remains sealed.', '',
        '| Generator blocks | Semantic DEV passes | Valid judgments | Trainable parameters | Training minutes | Peak GPU GB |',
        '|---|---:|---:|---:|---:|---:|']
    for depth, arm in arms.items():
        passes = f"{arm['passed']}/128 ({100*arm['task_accuracy']:.2f}%)" if arm['task_accuracy'] is not None else 'Unresolved'
        lines.append(f"| {depth} | {passes} | {arm['valid']}/128 | {arm['trainable_parameters']:,} | {arm['training_seconds']/60:.2f} | {arm['peak_allocated_bytes']/1e9:.3f} |")
    lines += ['', 'Accuracy uses complete-summary semantic judgments with accepted paraphrases, not exact reference matching.',
        'Each arm used all 1,376 reviewed TRAIN pairs, 1,024 updates, four examples per update and the same sampling/corruption schedule.',
        'The codec, word-source attention and Llama body remained frozen. The deeper model used identity-initialized added residual blocks.',
        '', 'Paired Terra decisions: `'+json.dumps(paired['all128'], sort_keys=True)+'`.',
        'Selected candidate for the separately governed final protocol: `'+str(candidate)+'`.',
        'DEV qualification is not final accuracy. Inference speed was not measured.', '',
        'Detailed cohort metrics, same-case joint gates and artifact hashes are in `'+str(ROOT/'campaign_results.json')+'`.']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: campaign[k] for k in ('classification', 'selected_candidate', 'paired_secondary_decisions')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('replay', 'aggregate'))
    parser.add_argument('--depth', type=int, choices=(12, 24))
    args = parser.parse_args()
    if args.stage == 'replay':
        if args.depth is None:
            parser.error('replay requires --depth')
        replay(args.depth)
    else:
        aggregate()
