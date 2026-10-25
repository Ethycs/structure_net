"""Register the completed two-case privileged diagnostic without a quality claim."""
import json
from pathlib import Path

from experiments.neural_architecture_lab import audit_ncm_oracle_prefix_diagnostic as audit
from experiments.neural_architecture_lab.ncm_result_registry import register_records

run = audit.run
DEST = run.ROOT / '20261025_level_conditioning_oracle_prefix_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-oracle-prefix-diagnostic.md')


def main():
    if DEST.exists() or run.q.read(audit.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('fresh finalization after actual independent audit 0 required')
    result = run.q.read(audit.DEST / 'result.json')
    if result['status'] != 'oracle_prefix_bindings_counts_and_rendering_replayed_not_task_accuracy':
        raise ValueError('complete independent oracle-prefix audit required')
    for path, digest in result['artifacts'].items():
        if run.component.file_hash(path) != digest:
            raise ValueError('diagnostic evidence changed: ' + path)
    run.native.vendor.verify(run.q.read(run.DEST / 'producer.json'))
    paths = [Path(__file__), REPORT, audit.DEST / 'result.json', audit.DEST / 'actual_tool_receipt.json']
    artifacts = {**result['artifacts'], **{str(p): run.component.file_hash(p) for p in paths}}
    DEST.mkdir()
    campaign = {'hypothesis_id': run.HYPOTHESIS, 'status': 'completed_two_case_exploratory_diagnostic',
        'classification': 'root_only_correction_insufficient_full_oracle_decoding_substantially_improved',
        'cases': 2, 'task_accuracy': None, 'oracle_target_information': True,
        'semantic_judgments': 0, 'final_test_opened': False, 'speed_tested': False,
        'diagnostics': result['diagnostics'], 'artifacts': artifacts}
    run.component.write_json(DEST / 'campaign_results.json', campaign)
    register_records(DEST / 'campaign_results.json', REPORT, DEST, {run.HYPOTHESIS: {
        'description': 'On two predeclared diagnostic sources, full oracle codes substantially improved frozen NCM decoding, while correcting only the root did not fix either output; this is not semantic task accuracy.',
        'classification': campaign['classification'], 'cases': 2, 'oracle_target_information': True,
        'task_accuracy_measured': False}})
    print(json.dumps({'status': campaign['status'], 'cases': 2, 'task_accuracy': None}), flush=True)


if __name__ == '__main__':
    main()
