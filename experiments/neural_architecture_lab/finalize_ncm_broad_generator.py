"""Replay every broader-generator DEV output, semantic first and decision gate."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_broad_generator as study
from experiments.structure_net import audit_ncm_broad_generator as training_audit
from experiments.structure_net import ncm_broad_generator_development as development

RESULTS = study.runtime.ROOT / '20261025_broad_generator_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-broad-generator.md')
HYPOTHESIS = study.HYPOTHESIS_ID
ROOTS = development.ROOTS


def classify(valid, passed, admitted, joints):
    """Whole-summary semantic evidence; invalid transport cannot become accuracy."""
    if type(valid) is not int or type(passed) is not int or not 0 <= passed <= valid <= 128:
        raise ValueError('complete128 denominator and valid pass counts required')
    if admitted:
        if valid != 128 or passed < 122 or joints is None or len(joints) != 2:
            raise ValueError('two admitted cohort joint decisions required')
        for value in joints.values():
            if (value['cases'] != 64 or not 0 <= value['joint_passed'] <= 64
                    or value['qualified'] != (value['all_valid'] and value['joint_passed'] >= 61)):
                raise ValueError('same-case joint threshold differs')
        if not all(v['all_valid'] for v in joints.values()): return 'DEV_transport_invalid'
        if all(v['qualified'] for v in joints.values()): return 'DEV_qualified'
    elif joints is not None:
        raise ValueError('joint judges require admission')
    if valid != 128: return 'DEV_transport_invalid'
    return 'DEV_semantic_partial_progress' if passed else 'DEV_semantic_quality_failed'


def finalize():
    training, evaluation, environment = development.configure()
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_native_development import replay_outputs
    q, old, lineage = training.q, evaluation.old, evaluation.lineage
    lineage.configure()
    if RESULTS.exists():
        raise ValueError('Fresh append-only aggregate root required')
    for root in (study.RUNS['train'], *ROOTS.values()):
        training.preparation.completed(root)
    train_audit = training_audit.DEST
    if q.read(train_audit / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('Completed independent training audit required')
    audited = q.read(train_audit / 'campaign_results.json')
    if (audited['status'] != 'broad_generator_training_independently_replayed'
            or audited['cases'] != 1376 or audited['updates'] != study.CONFIG.updates
            or audited['auditor_sha256'] != study.file_hash(training_audit.__file__)):
        raise ValueError('complete unchanged broader-generator training audit required')
    for path, digest in audited['artifacts'].items():
        if training.file_hash(Path(path)) != digest:
            raise ValueError('Audited training artifact changed: ' + path)
    cohorts, manifest, split = evaluation.sources()
    rows = [r for cohort in old.COHORTS for r in cohorts[cohort]]
    outputs, pilot = [q.read(ROOTS[s] / 'outputs.json') for s in ('outputs', 'inference_pilot')]
    tokenizer = training.native.tokenizer()
    replay_outputs(outputs, rows, manifest, tokenizer)
    replay_outputs(pilot, old.inference_probes(cohorts, manifest), manifest, tokenizer)
    by_id = {r['id']: r for r in outputs}
    if any(by_id[r['id']] != r for r in pilot):
        raise ValueError('Full predictions differ from systems pilot')
    trained = q.read(study.RUNS['train'] / 'result.json')
    if (trained['updates'] != study.CONFIG.updates or trained['TRAIN_cases'] != 1376
            or not trained['oracle_unchanged'] or not trained['strict_reload_passed']):
        raise ValueError('full budget and unchanged reconstruction path required')
    for stage in ('inference_pilot', 'outputs'):
        result = q.read(ROOTS[stage] / 'result.json')
        if (result['model_sha256'] != trained['final_model_sha256']
                or result['outputs_sha256'] != training.file_hash(ROOTS[stage] / 'outputs.json')
                or q.read(ROOTS[stage] / 'producer.json')['environment'] != environment):
            raise ValueError('Fixed checkpoint, output or runtime identity differs')
    cases = {c: [{**r, 'texts': {'student': by_id[r['id']]['text']}} for r in cohorts[c]]
             for c in old.COHORTS}
    packets = {c: {'student': [q.packet(r['id'], r['source'],
        [{'candidate_id': 'summary', 'text': r['texts']['student']}]) for r in cases[c]]}
        for c in old.COHORTS}
    inherited = lineage.base_cache()
    cache = q.read(ROOTS['grade'] / 'judgment_cache.json')
    pilot_cache = q.read(ROOTS['judge_pilot'] / 'judgment_cache.json')
    if (any(pilot_cache.get(k) != v for k, v in inherited.items())
            or any(cache.get(k) != v for k, v in pilot_cache.items())):
        raise ValueError('Canonical or pilot first judgments changed')
    lineage.audit_cache(cache)
    for stage in ('judge_pilot', 'grade'):
        if (q.read(ROOTS[stage] / 'cases.json') != cases
                or q.read(ROOTS[stage] / 'exact_packets.json') != packets):
            raise ValueError('Judged packets do not match complete source-only predictions')
    metrics, scores = {c: {} for c in old.COHORTS}, {}

    def replay(stage, cohort, role, count, decisions):
        expected = replay_scores(cases[cohort][:count],
            {'student': packets[cohort]['student'][:count]}, decisions, 'student', role)
        directory = ROOTS[stage] / cohort
        if q.read(directory / f'{role}_student_case_scores.json') != expected:
            raise ValueError('Raw first-judgment score replay differs')
        metric = old.task_metric(expected)
        reported = q.read(directory / f'{role}_student_result.json')
        if any(reported.get(k) != v for k, v in metric.items()):
            raise ValueError('Reported semantic metric differs')
        return expected, reported

    for cohort in old.COHORTS:
        pilot_scores, _ = replay('judge_pilot', cohort, 'secondary', 2, pilot_cache)
        if not all(r['valid'] for r in pilot_scores):
            raise ValueError('Valid four-case judge pilot required')
        scores[cohort, 'secondary'], metrics[cohort]['secondary'] = replay(
            'grade', cohort, 'secondary', 64, cache)
    admitted = old.secondary_admitted({c: metrics[c]['secondary'] for c in old.COHORTS})
    joints = None
    if admitted:
        joints = {}
        for cohort in old.COHORTS:
            scores[cohort, 'primary'], metrics[cohort]['primary'] = replay(
                'grade', cohort, 'primary', 64, cache)
            joints[cohort] = old.joint_gate(scores[cohort, 'secondary'], scores[cohort, 'primary'])
    elif any((ROOTS['grade'] / c / 'primary_student_case_scores.json').exists() for c in old.COHORTS):
        raise ValueError('Primary judge ran without admission')
    allowed = {old.oracle.first.packet_key(p, role) for c in old.COHORTS
               for p in packets[c]['student']
               for role in (('secondary', 'primary') if admitted else ('secondary',))}
    if set(cache) - set(inherited) - allowed:
        raise ValueError('Unexpected new judgments outside the fixed DEV scope')
    qualified = admitted and all(joints[c]['qualified'] for c in old.COHORTS)
    valid = sum(metrics[c]['secondary']['valid'] for c in old.COHORTS)
    passed = sum(metrics[c]['secondary']['passed'] for c in old.COHORTS)
    accuracy = passed / 128 if valid == 128 else None
    result = q.read(ROOTS['grade'] / 'result.json')
    if (result['metrics'] != metrics or result['primary_admitted'] != admitted
            or result['joint'] != joints or result['DEV_qualified'] != qualified
            or result['task_accuracy'] != accuracy
            or result['canonical_first_judgments'] != len(cache)
            or result['inherited_first_judgments'] != len(inherited)
            or result['final_test_opened'] or result['speed_tested']):
        raise ValueError('Aggregate accuracy, denominator, lineage or gate differs')
    paths = [Path(__file__), train_audit / 'campaign_results.json', train_audit / 'actual_tool_receipt.json']
    for root in ROOTS.values():
        paths += [p for p in root.rglob('*.json') if 'producer_source' not in p.parts]
    classification = classify(valid, passed, admitted, joints)
    joint_accuracy = (sum(v['joint_passed'] for v in joints.values()) / 128
                      if joints and all(v['all_valid'] for v in joints.values()) else None)
    campaign = {'schema': 'nal.ncm-broad-generator-results.v1', 'hypothesis_id': HYPOTHESIS,
        'classification': classification, 'training_updates': study.CONFIG.updates, 'training_cases': 1376,
        'joint_task_accuracy': joint_accuracy, 'frozen_reconstruction_preserved': True,
        'conformance': 'EXPLORATORY_UNDERPOWERED', 'final_task_goal_achieved': False,
        'cases': 128, 'valid': valid, 'passed': passed, 'task_accuracy': accuracy,
        'task_accuracy_judge': q.JUDGES['secondary'], 'metric': 'complete_summary_semantic_pass',
        'wilson_95': q.wilson(passed, 128) if valid == 128 else None,
        'metrics': metrics, 'joint': joints, 'primary_admitted': admitted, 'DEV_qualified': qualified,
        'source_only': True, 'split_proof': split, 'canonical_first_judgments': len(cache),
        'inherited_first_judgments': len(inherited), 'final_test_opened': False, 'speed_tested': False,
        'checkpoint_sha256': trained['checkpoint_sha256'],
        'artifacts': {str(p): training.file_hash(p) for p in paths}}
    RESULTS.mkdir()
    training.write_json(RESULTS / 'campaign_results.json', campaign)
    print(json.dumps({k: v for k, v in campaign.items() if k not in ('artifacts', 'split_proof', 'metrics')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.parse_args()
    finalize()
