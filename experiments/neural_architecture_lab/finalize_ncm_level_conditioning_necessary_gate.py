"""Derive a necessary quality failure without inventing semantic judgments."""
import json
from pathlib import Path

from experiments.structure_net import ncm_level_conditioning_numeric as run

HYPOTHESIS = 'ncm-level-conditioning-native655-necessary-quality-failure-v1'
DEST = run.ROOT / '20261025_level_conditioning_necessary_quality_results_v1'
AUDIT = run.ROOT / '20261024_level_conditioning_numeric_audit_v1'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-level-conditioning-necessary-quality-failure.md')


def bounds(outputs):
    cohorts = {}
    if len(outputs) != 128 or len({r['id'] for r in outputs}) != 128:
        raise ValueError('all 128 distinct original predictions required')
    for name in run.source.COHORTS:
        rows = [r for r in outputs if r['split'] == name]
        if len(rows) != 64:
            raise ValueError('both complete original 64-case cohorts required')
        empty = [r['id'] for r in rows if not r['text'].strip()]
        cohorts[name] = {'cases': 64, 'empty_ids': empty, 'empty': len(empty),
            'maximum_possible_passes': 64 - len(empty),
            'maximum_possible_rate': (64 - len(empty)) / 64,
            'required_same_case_passes': 61,
            'necessary_gate_possible': 64 - len(empty) >= 61}
    nonempty = sum(c['maximum_possible_passes'] for c in cohorts.values())
    return {'cohorts': cohorts, 'cases': 128, 'empty': 128 - nonempty,
        'maximum_possible_passes': nonempty, 'maximum_possible_rate': nonempty / 128,
        'actual_semantic_accuracy': None,
        'necessary_quality_gate_possible': all(c['necessary_gate_possible'] for c in cohorts.values())}


def main():
    if DEST.exists():
        raise ValueError('fresh append-only finalization required')
    if (run.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or run.q.read(AUDIT / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('actual complete inference and independent CPU audit required')
    audit = run.q.read(AUDIT / 'result.json')
    if audit['status'] != 'all128_new_source_only_state_feature_code_and_rendering_bindings_replayed_semantics_pending':
        raise ValueError('wrong independent source-only audit')
    for path, expected in audit['artifacts'].items():
        if run.file_hash(path) != expected:
            raise ValueError('audited evidence changed: ' + path)
    outputs = run.q.read(run.OUTPUTS / 'outputs.json')
    result = bounds(outputs)
    for name, c in result['cohorts'].items():
        if c['maximum_possible_passes'] != audit['diagnostics_not_task_accuracy'][name]['nonempty']:
            raise ValueError('independent audited nonempty count differs')
    # An empty string necessarily fails the unchanged original complete-summary rubric.
    optimistic_grade = {'faithful': True, 'salient': True, 'readable': True}
    if run.q.passes(optimistic_grade, '', 0):
        raise ValueError('the original task rule no longer rejects empty summaries')
    if result['necessary_quality_gate_possible']:
        raise ValueError('this finalizer cannot classify a possibly passing candidate')
    paths = [Path(__file__), REPORT, run.OUTPUTS / 'outputs.json', run.OUTPUTS / 'result.json',
        run.WORKFLOW / 'actual_tool_receipt.json', AUDIT / 'result.json',
        AUDIT / 'actual_tool_receipt.json', Path(run.q.__file__),
        Path(run.q.passes.__code__.co_filename)]
    artifacts = {str(p): run.file_hash(p) for p in dict.fromkeys(paths)}
    DEST.mkdir()
    campaign = {'hypothesis_id': HYPOTHESIS,
        'status': 'completed_necessary_quality_failure_semantic_accuracy_unmeasured',
        'classification': 'cannot_meet_95_percent_complete_summary_quality',
        'analysis': 'retrospective deterministic upper bound from unchanged nonempty task requirement',
        **result, 'checkpoint_sha256': audit['checkpoint_sha256'], 'model_sha256': audit['model_sha256'],
        'nonempty_does_not_mean_semantic_pass': True, 'judgments_fabricated': False,
        'judge_requests': 0, 'semantic_cache_untouched': True, 'full_semantic_study_completed': False,
        'final_test_opened': False, 'speed_tested': False, 'artifacts': artifacts}
    run.write_json(DEST / 'campaign_results.json', campaign)
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(DEST / 'campaign_results.json', REPORT, DEST, {HYPOTHESIS: {
        'description': 'The fixed 1024-update FP16 native655 NCM checkpoint cannot reach the unchanged 95% complete-summary gate: 10 of all 128 archived source-only outputs are empty.',
        'classification': campaign['classification'],
        'actual_semantic_accuracy_measured': False, 'empty_summaries': result['empty'],
        'maximum_possible_semantic_accuracy': result['maximum_possible_rate'],
        'nonempty_does_not_mean_semantic_pass': True}})
    print(json.dumps({k: v for k, v in campaign.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
