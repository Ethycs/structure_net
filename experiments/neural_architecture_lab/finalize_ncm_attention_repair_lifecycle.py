"""Register the audited repair lifecycle as systems evidence only."""
import json
from pathlib import Path

from experiments.neural_architecture_lab import audit_ncm_attention_repair_lifecycle as audit
from experiments.neural_architecture_lab.ncm_result_registry import register_records

run = audit.run
DEST = run.ROOT / '20261025_attention_code_repair_lifecycle_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-attention-code-repair-lifecycle.md')


def main():
    if DEST.exists() or run.q.read(audit.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('fresh finalization after actual independent CPU audit0 required')
    result = run.q.read(audit.DEST / 'result.json')
    if result['status'] != 'attention_repair_CPU_state_trace_and_TRAIN_bindings_replayed_not_quality':
        raise ValueError('complete independent state/binding audit required')
    for path, digest in result['artifacts'].items():
        if run.file_hash(path) != digest:
            raise ValueError('completed lifecycle evidence changed: ' + path)
    paths = [Path(__file__), REPORT, audit.DEST / 'result.json', audit.DEST / 'actual_tool_receipt.json']
    DEST.mkdir()
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'status': 'completed_two_update_TRAIN_systems_check',
        'classification': 'repair_gradients_frozen_geometry_and_strict_reload_passed_quality_unmeasured',
        'trainable_parameters': result['trainable_parameters'], 'updates': 2, 'cases': 2,
        'changed_hard_code_positions': result['changed_hard_code_positions'],
        'task_accuracy': None, 'quality_improvement_established': False,
        'final_test_opened': False, 'speed_tested': False,
        'artifacts': {**result['artifacts'], **{str(p): run.file_hash(p) for p in paths}}}
    run.write_json(DEST / 'campaign_results.json', campaign)
    register_records(DEST / 'campaign_results.json', REPORT, DEST, {run.HYPOTHESIS: {
        'description': 'A separate 3.78M-parameter source-attention NCM code repair module completed two TRAIN updates with finite source gradients, frozen codec and strict state/output reload. No hard code changed and quality is unmeasured.',
        'classification': campaign['classification'], 'systems_only': True,
        'task_accuracy_measured': False, 'quality_improvement_established': False}})
    print(json.dumps({k: v for k, v in campaign.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
