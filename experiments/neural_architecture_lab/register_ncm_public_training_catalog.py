"""Register audited public-data admission without claiming factual/task quality."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_public_training_catalog as catalog

data, DEST, HYPOTHESIS = catalog.data, catalog.AUDIT, catalog.HYPOTHESIS
CAMPAIGN = DEST/'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-26_ncm-public-training-catalog.md')


def record_for(result):
    counts = result['counts']
    required = ('all_original_rows_replayed', 'all_candidate_tokens_recomputed',
                'all_public_development_text_overlaps_recomputed', 'all_byte_offsets_replayed',
                'all_remaining_pairs_retained', 'training_data_admitted')
    if (result['hypothesis_id'] != HYPOTHESIS or not all(result[key] for key in required)
            or result['classification'] != 'full_public_paired_catalog_admitted_for_pretraining'
            or result['admission_scope'] != 'public_reference_conditional_pretraining_only'
            or counts['original_rows'] != 204045 or counts['original_candidates'] != 163051
            or counts['retained_pairs']+counts['additional_excluded_rows'] != 163051
            or counts['prior_screen_exclusions']+163051 != 204045
            or result['public_development_cases'] != 320 or result['protected_application_target_cases'] != 64
            or result['existing_TRAIN_target_cases'] != 1376 or result['strings_rewritten_or_truncated']
            or result['label_factual_quality_verified'] or result['model_training'] or result['new_judge_requests']
            or result['task_accuracy'] is not None or result['final_test_opened']
            or result['source_disjointness_established'] or result['sealed_final_full_text_overlap_audit_completed']
            or result['public_validation_or_test_payload_read']):
        raise ValueError('complete data-only pretraining admission boundaries required')
    return {'name': 'Full public XSum conditional pretraining catalog with protected development controls',
        'classification': result['classification'], 'admitted_public_pairs': counts['retained_pairs'],
        'protected_public_development_cases': 320, 'training_data_admitted': True,
        'label_factual_quality_verified': False, 'model_training': False, 'task_accuracy_measured': False,
        'source_disjointness_established': False,
        'description': f"Admitted all{counts['retained_pairs']} remaining public XSum TRAIN pairs after the original "
            f"163051-candidate screen and{counts['additional_excluded_rows']} further exclusions. Protected320 public "
            'codec diagnostic articles and equivalent targets,64 reserved application target frames and1376 current '
            'TRAIN targets. Replayed every original pairing, candidate token count, public-development text overlap, '
            'native target frame, deduplication decision and UTF8 byte offset. References are unverified public '
            'pretraining labels. No model training, new semantic judgment or final source access occurred. '
            'This is not a95percent complete-summary accuracy result.'}


def main(stage):
    data.completed(DEST)
    result = data.read(CAMPAIGN)
    data.verify(result['artifacts'])
    record = record_for(result)
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(CAMPAIGN, REPORT, DEST, {HYPOTHESIS: record})
        print(json.dumps({'status': 'registered_and_read_back', 'hypothesis': HYPOTHESIS}))
        return
    import chromadb
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    receipt = data.read(DEST/(HYPOTHESIS+'_registry_readback.json'))
    stored = receipt['record']
    exact = collection.get(ids=[HYPOTHESIS], include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']], n_results=10)
    if (any(stored.get(k) != v for k, v in record.items()) or exact['ids'] != [HYPOTHESIS]
            or any(exact['metadatas'][0].get(k) != v for k, v in stored.items())
            or data.read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json')) != receipt
            or HYPOTHESIS not in semantic['ids'][0] or stored['campaign_sha256'] != data.sha(CAMPAIGN)
            or stored['report_sha256'] != data.sha(REPORT)):
        raise ValueError('independent backend/JSON readback differs')
    value = {'status': 'independent_registry_readback_passed', 'hypothesis_id': HYPOTHESIS,
        'campaign_sha256': data.sha(CAMPAIGN), 'report_sha256': data.sha(REPORT)}
    data.write(DEST/'backend_exact_readback.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('register', 'readback'))
    main(parser.parse_args().stage)
