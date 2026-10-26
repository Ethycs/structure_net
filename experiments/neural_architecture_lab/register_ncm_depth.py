"""Register the audited exploratory depth result and independently read it back."""
import argparse
import json
from pathlib import Path

from experiments.neural_architecture_lab import finalize_ncm_depth as final

ROOT, REPORT, HYPOTHESIS = final.ROOT, final.REPORT, final.HYPOTHESIS
CAMPAIGN = ROOT/'campaign_results.json'
read, sha = final.study.broad.catalog.read, final.study.broad.file_hash


def record_for(result):
    if (result['hypothesis_id'] != HYPOTHESIS or result['cases_per_arm'] != 128
            or result['independent_training_seeds'] != 1 or result['inherited_first_judgments'] != 4179
            or result['canonical_first_judgments'] < 4179 or not result['same_sampling']
            or result['same_parent_checkpoint'] != final.study.parent.PARENT_SHA
            or not result['identity_initialized_extra_blocks'] or not result['source_only']
            or not result['all_historical_and_pilot_firsts_preserved'] or result['final_test_opened']
            or result['speed_tested'] or result['final_task_goal_achieved']
            or result['selected_candidate'] != final.selected_candidate(result['arms'])
            or result['classification'] != final.comparison_classification(result['arms'])
            or result['comparison_has_unresolved_judgments'] != any(
                v['classification'] == 'DEV_transport_invalid' for v in result['arms'].values())):
        raise ValueError('complete audited depth DEV comparison required; final goal remains unproven')
    record = {'name': 'NCM generator depth: 12 versus 24 transformer blocks',
        'classification': result['classification'], 'parent_hypothesis_id': 'ncm-broad-generator-v1',
        'paired_DEV_cases': 128, 'matched_arms': 2, 'training_updates_per_arm': 1024, 'training_cases': 1376,
        'canonical_first_judgments': result['canonical_first_judgments'],
        'comparison_has_unresolved_judgments': result['comparison_has_unresolved_judgments'],
        'selected_candidate': result['selected_candidate'] or 'none',
        'final_task_accuracy_measured': False, 'source_only': True, 'frozen_reconstruction_preserved': True}
    descriptions = []
    for depth, arm in result['arms'].items():
        if (arm['depth'] != int(depth) or arm['cases'] != 128
                or arm['training_audit']['updates'] != 1024 or arm['training_audit']['distinct_TRAIN_cases'] != 1376
                or arm['final_test_opened'] or arm['speed_tested']):
            raise ValueError('complete matched checkpoints and DEV denominators required')
        for key in ('valid', 'passed', 'DEV_qualified', 'classification', 'trainable_parameters'):
            record['depth'+depth+'_'+key] = arm[key]
        for key in ('task_accuracy', 'joint_task_accuracy'):
            if arm[key] is not None:
                record['depth'+depth+'_'+key] = arm[key]
        descriptions.append(f"depth{depth}: Terra {arm['passed']} passes, {arm['valid']}/128 valid; joint qualification {arm['DEV_qualified']}")
    record['description'] = ('Matched 1,024-update continuations from BROAD on all1,376 reviewed TRAIN pairs. '
        'The intervention adds12 identity-initialized residual blocks;12 code levels and frozen codec/body are unchanged. '
        + '; '.join(descriptions) + '. All historical first judgments preserved. One training seed; final320 and inference speed remain unmeasured.')
    return record


def main(stage):
    final.receipt(ROOT, ('campaign_results.json',))
    result = read(CAMPAIGN)
    record = record_for(result)
    for path, expected in result['artifacts'].items():
        if sha(path) != expected:
            raise ValueError('completed depth evidence changed')
    import numpy as np
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(CAMPAIGN, REPORT, ROOT, {HYPOTHESIS: record})
        print(json.dumps({'status': 'registered_and_read_back', 'hypothesis': HYPOTHESIS}), flush=True)
        return
    import chromadb
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    registered = read(ROOT/(HYPOTHESIS+'_registry_readback.json'))
    stored = registered['record']
    exact = collection.get(ids=[HYPOTHESIS], include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']], n_results=10)
    local = read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json'))
    if (any(stored.get(k) != v for k, v in record.items()) or exact['ids'] != [HYPOTHESIS]
            or any(exact['metadatas'][0].get(k) != v for k, v in stored.items())
            or local != registered or HYPOTHESIS not in semantic['ids'][0]
            or stored['campaign_sha256'] != sha(CAMPAIGN) or stored['report_sha256'] != sha(REPORT)):
        raise ValueError('independent exact, semantic or JSON registry readback differs')
    value = {'status': 'independent_exact_and_semantic_registry_readback_passed',
        'hypothesis_id': HYPOTHESIS, 'classification': result['classification'],
        'campaign_sha256': sha(CAMPAIGN), 'report_sha256': sha(REPORT), 'registrar_sha256': sha(__file__)}
    with (ROOT/'backend_exact_readback.json').open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('register', 'readback'))
    main(parser.parse_args().stage)
