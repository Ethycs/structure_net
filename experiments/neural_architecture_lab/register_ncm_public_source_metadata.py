"""Register exhaustive source tokenization replay without a model quality claim."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import audit_ncm_public_source_metadata as audit

data, DEST = audit.data, audit.ROOTS['full']
HYPOTHESIS = audit.producer.HYPOTHESIS
CAMPAIGN = DEST/'result.json'
REPORT = Path('docs/08 - Analysis/2026-10-26_ncm-public-source-metadata.md')


def record_for(value):
    if (value['hypothesis_id'] != HYPOTHESIS or value['status'] != 'all_public_full_source_metadata_replayed'
            or value['cases'] != 162353 or value['source_positions'] != 91344717
            or value['native_prompt_positions'] != 125616400 or value['max_source_positions'] != 4124
            or value['max_native_prompt_positions'] != 2275 or value['pilot_repetitions_exact'] != 9
            or not all(value[k] for k in ('all_records_replayed', 'all_tokenization_and_alignment_recomputed',
                                         'original_TRAIN_geometry_equal', 'full_metadata_audited'))
            or any(value[k] for k in ('model_training', 'body_loaded', 'features_computed',
                'target_fields_in_body_input', 'final_test_opened', 'independent_tokenization_implementation'))
            or value['new_judge_requests'] or value['task_accuracy'] is not None):
        raise ValueError('complete data-only source metadata audit required')
    return {'name': 'All public source prompts and alignments replayed exactly',
        'classification': value['status'], 'source_cases': value['cases'],
        'source_positions': value['source_positions'], 'native_prompt_positions': value['native_prompt_positions'],
        'full_metadata_audited': True, 'model_training': False, 'task_accuracy_measured': False,
        'description': 'Independently replayed all162353 admitted public XSum source records, every native '
            'Llama prompt token, GPT2 source token and character-overlap weight. Preserved the original '
            '23Oct2026 prompt date and full text. Reproduced9 public probes and2 original TRAIN controls. '
            'Verified complete ordered byte indexes and original source bindings. This reuses the original '
            'tokenization implementation in a separate execution. Maximum source4124, native prompt2275; '
            'complete FP32 feature estimate1122443882496 bytes. No features, training or quality measurement.'}


def main(stage):
    value = data.completed(DEST)
    record = record_for(value)
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(CAMPAIGN, REPORT, DEST, {HYPOTHESIS: record})
        print(json.dumps({'status': 'registered_and_read_back', 'hypothesis_id': HYPOTHESIS}))
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
        raise ValueError('independent metadata registry readback differs')
    result = {'status': 'independent_registry_readback_passed', 'hypothesis_id': HYPOTHESIS,
              'campaign_sha256': data.sha(CAMPAIGN), 'report_sha256': data.sha(REPORT)}
    data.write(DEST/'backend_exact_readback.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('register', 'readback'))
    main(parser.parse_args().stage)
