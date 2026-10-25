"""Finalize preserved collection failures without fabricating quality metrics."""
import json
from pathlib import Path

from experiments.neural_architecture_lab import inventory_ncm_grounded_semantic_stop as semantics
from experiments.neural_architecture_lab import audit_ncm_sol_teacher144_partial as teacher
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT, q = semantics.run.ROOT, semantics.run.q
DEST = ROOT / '20261025_interrupted_collection_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-interrupted-collections.md')
HYPOTHESES = ('ncm-grounded-semantic-transport-stop-v1', 'ncm-sol-teacher144-outage-firsts-v1')


def main():
    if DEST.exists() or REPORT.exists():
        raise ValueError('fresh failure finalization required')
    inputs = [Path(__file__), *[root / name for root, name in (
        (semantics.DEST, 'campaign_results.json'), (semantics.DEST, 'actual_tool_receipt.json'),
        (semantics.DEST, 'judgment_cache.json'), (semantics.DEST, 'request_frontier.json'),
        (teacher.DEST, 'result.json'), (teacher.DEST, 'actual_tool_receipt.json'),
        (teacher.DEST, 'first_candidates.json'), (teacher.DEST, 'unattempted_ids.json'),
        (teacher.DEST, 'exhausted_firsts.json'))]]
    for root in (semantics.DEST, teacher.DEST):
        if q.read(root / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
            raise ValueError('actual independent replay completion required')
        semantics.run.native.vendor.verify(q.read(root / 'producer.json'))
    sem = q.read(inputs[1])
    teach = q.read(teacher.DEST / 'result.json')
    for state in (sem, teach):
        for path, digest in state['artifacts'].items():
            if file_hash(path) != digest:
                raise ValueError('completed collection evidence changed')
    if (sem['classification'] != 'transport_invalid_semantics_unestablished'
            or sem['canonical_first_judgments'] != 3539 or sem['unknown_request_keys'] != 3
            or teach['observed'] != 144 or teach['invalid_firsts'] != 9):
        raise ValueError('complete preserved failure populations required')
    p = semantics.run.native.vendor.freeze(DEST, 'completed_interrupted_collection_failure_ledgers', inputs,
        {'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False,
         'new_requests': 0, 'canonical_judgment_cache': str(semantics.DEST / 'judgment_cache.json')})
    campaign = {k: v for k, v in sem.items() if k != 'artifacts'}
    campaign.update(teacher_partial={k: v for k, v in teach.items() if k != 'artifacts'},
        new_model_training=False, speed_tested=False,
        artifacts={**sem['artifacts'], **teach['artifacts'], **{str(path): file_hash(path) for path in inputs},
            str(DEST / 'producer.json'): file_hash(DEST / 'producer.json')})
    write_json(DEST / 'campaign_results.json', campaign)
    REPORT.write_text('# Interrupted NCM collections: failures and allowances preserved\n\n'
        'The source-attention oracle semantic study remains transport-invalid. Its actual workflow exit was 1 '
        'and the grade worker exited -15. Independent raw replay recovered three exhausted firsts missing '
        'from the stopped cache: the new ledger contains 3,539 first judgments, preserving all 3,523 ancestors, '
        'four valid pilot additions and twelve invalid transport additions. Three further request folders have '
        'no saved attempts. Each retains a possible unsaved call and at most two future attempts. No request '
        'was repeated and no aggregate semantic accuracy or source-attention qualification is established.\n\n'
        'The separate teacher continuation also remains failed. All 144 drained first pipelines and 303 raw '
        'attempts replayed: 135 syntax-valid/capacity-fitting targets, one exhausted syntax failure and eight '
        'exhausted service failures. Its 8,487 untouched IDs are enumerated. Across prior collections, all '
        '1,513 observed firsts, including 73 invalids, remain preserved. Target syntax/length validity is not '
        'semantic acceptance or student accuracy; no new training data was admitted.\n\n'
        'Ten focused replay tests passed. Both independent audits actually exited 0; their source workflows '
        'did not. The latest main student still has ten empty summaries among 128 source-only DEV cases '
        '(an upper bound of 118/128, not measured accuracy), so it cannot meet 95% on that cohort. Sealed '
        'final sources remain unopened, and no speedup is claimed.\n\n'
        'The canonical cache in this failure ledger is '
        f'`{semantics.DEST / "judgment_cache.json"}`. New evaluation plans must bind it explicitly after '
        'registration and independent backend readback; completed producers and their guards stay unchanged.\n')
    semantics.run.native.vendor.verify(p)
    register_records(DEST / 'campaign_results.json', REPORT, DEST, {
        HYPOTHESES[0]: {'name': 'Stopped source-attention semantic first-judgment ledger',
            'description': 'Raw replay recovered all twelve transport-invalid firsts and preserves three uncertain request allowances. All 3539 firsts retained; incomplete oracle semantic study, no accuracy or source-attention qualification.',
            'classification': campaign['classification'], 'canonical_first_judgments': 3539,
            'student_task_accuracy_measured': False},
        HYPOTHESES[1]: {'name': 'Stopped teacher continuation: 144 first pipelines',
            'description': 'Independent replay of 144 drained teacher firsts and 303 raw attempts preserves one syntax and eight service failures. All 8487 untouched IDs enumerated; no semantic qualification or training admission.',
            'classification': 'partial_teacher_firsts_replayed_with_failures_preserved',
            'data_preparation_only': True, 'student_task_accuracy_measured': False}})
    print(json.dumps({'status': 'interrupted_collection_ledgers_finalized_and_registered',
        'canonical_first_judgments': 3539, 'teacher_observed_firsts': 144, 'student_task_accuracy': None}), flush=True)


if __name__ == '__main__':
    main()
