"""Recover a completed judge pilot by replay; never recreate lost process status.

The original session's terminal response was unavailable after compaction. This
is an offline artifact audit, not another judge request or a replacement score.
Its observed exit can authorize continuation, while the original exit remains
explicitly unknown in the recovery receipt.
"""
import json
from pathlib import Path

from experiments.structure_net import ncm_broad_generator_development as dev
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores

DEST = dev.study.runtime.ROOT / '20261025_broad_generator_judge_pilot_recovery_v1'


def main():
    if DEST.exists():
        raise ValueError('fresh append-only recovery audit required')
    training, evaluation, environment = dev.configure()
    q, old, lineage = training.q, evaluation.old, evaluation.lineage
    lineage.configure()
    root = dev.ROOTS['judge_pilot']
    proof = q.read(root / 'producer.json')
    training.native.vendor.verify(proof)
    if proof['environment'] != environment:
        raise ValueError('pilot runtime identity changed')
    training.preparation.completed(dev.ROOTS['outputs'])
    cohorts, _, _ = evaluation.sources()
    outputs = q.read(dev.ROOTS['outputs'] / 'outputs.json')
    state = q.read(dev.ROOTS['outputs'] / 'result.json')
    if dev.study.file_hash(dev.ROOTS['outputs'] / 'outputs.json') != state['outputs_sha256']:
        raise ValueError('source-only outputs changed')
    rows = [r for c in old.COHORTS for r in cohorts[c]]
    if [r['id'] for r in rows] != [r['id'] for r in outputs]:
        raise ValueError('ordered complete source-only outputs required')
    by_id = {r['id']: r for r in outputs}
    cases = {c: [{**r, 'texts': {'student': by_id[r['id']]['text']}}
                 for r in cohorts[c]] for c in old.COHORTS}
    packets = {c: {'student': [q.packet(r['id'], r['source'],
        [{'candidate_id': 'summary', 'text': r['texts']['student']}])
        for r in cases[c]]} for c in old.COHORTS}
    if q.read(root / 'cases.json') != cases or q.read(root / 'exact_packets.json') != packets:
        raise ValueError('pilot packet/source binding differs')
    inherited = lineage.base_cache()
    cache = q.read(root / 'judgment_cache.json')
    if any(cache.get(k) != v for k, v in inherited.items()):
        raise ValueError('historical first judgment changed')
    allowed = {old.oracle.first.packet_key(p, 'secondary')
               for c in old.COHORTS for p in packets[c]['student'][:2]}
    if set(cache) != set(inherited) | allowed:
        raise ValueError('pilot must contain exactly the four fixed first judgments')
    lineage.audit_cache(cache)
    metrics = {}
    for c in old.COHORTS:
        scores = replay_scores(cases[c][:2], {'student': packets[c]['student'][:2]},
                               cache, 'student', 'secondary')
        if q.read(root / c / 'secondary_student_case_scores.json') != scores:
            raise ValueError('pilot score replay differs')
        measured = old.task_metric(scores)
        reported = q.read(root / c / 'secondary_student_result.json')
        if measured['valid'] != 2 or any(reported.get(k) != v for k, v in measured.items()):
            raise ValueError('valid pilot metrics required')
        metrics[c] = {'secondary': reported}
    result = q.read(root / 'result.json')
    expected = {'schema': proof['schema'], 'status': 'systems_judge_pilot_valid',
        'metrics': metrics, 'valid_pilot': True, 'primary_admitted': False,
        'joint': None, 'DEV_qualified': False, 'canonical_first_judgments': len(cache),
        'inherited_first_judgments': len(inherited), 'task_accuracy': None,
        'task_accuracy_judge': q.JUDGES['secondary'],
        'metric': 'complete_summary_semantic_pass', 'final_test_opened': False,
        'speed_tested': False, 'producer_fingerprint': proof['fingerprint']}
    if result != expected:
        raise ValueError('complete pilot result replay differs')
    paths = [Path(__file__), *sorted(root.rglob('*.json'))]
    audit = {'status': 'completed_pilot_independently_replayed',
        'original_session_id': 25910, 'original_process_exit_code': None,
        'reason': 'terminal response unavailable; session no longer present',
        'valid_pilot_cases': 4, 'passed_pilot_cases': sum(
            metrics[c]['secondary']['passed'] for c in old.COHORTS),
        'new_judge_requests': 0, 'prior_and_pilot_firsts_preserved': True,
        'artifacts': {str(p): dev.study.file_hash(p) for p in paths}}
    DEST.mkdir()
    dev.study.write_json(DEST / 'result.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
