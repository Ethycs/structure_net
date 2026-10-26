"""Register the example batching systems result separately from model quality."""
import argparse
import json
from pathlib import Path

from experiments.neural_architecture_lab.finalize_ncm_example_batching import DEST, HYPOTHESIS, file_hash, read

REPORT = Path('docs/08 - Analysis/2026-10-26_ncm-example-batching.md')
CAMPAIGN = DEST/'campaign_results.json'


def record_for(result):
    if (result['hypothesis_id'] != HYPOTHESIS or result['default_backend_numerically_qualified']
            or result['math_reference_numerically_qualified'] or not result['weights_unchanged']
            or result['optimizer_steps'] or result['new_judge_requests'] or result['task_accuracy'] is not None
            or result['final_test_opened'] or result['application_speed_tested']
            or not result['systems_lifecycle_only_not_quality_evidence']
            or result['training_optimizer_lifecycle_qualified']):
        raise ValueError('unchanged-weight systems evidence with preserved failed variants required')
    return {'name': 'NCM complete-example batching numerical and resource diagnostic',
        'classification': result['classification'], 'systems_lifecycle_only_not_quality_evidence': True,
        'default_backend_numerically_qualified': False, 'math_reference_numerically_qualified': False,
        'serial_codec_numerically_qualified': result['serial_codec_numerically_qualified'],
        'largest_resource_qualified_batch': max(result['resource_qualified_sizes'], default=0),
        'weights_unchanged': True, 'optimizer_steps': 0, 'final_task_accuracy_measured': False,
        'description': 'Matched shortest/longest previously admitted TRAIN examples compare separate '
            'and batched FP32 gradients. Default and MATH attention backends both failed fixed tolerances '
            'despite equal code argmaxes. A third diagnostic preserves per-example frozen-codec preparation. '
            'Its qualification is '+str(result['serial_codec_numerically_qualified'])+'. '
            'Recorded resource-qualified sizes are '+str(result['resource_qualified_sizes'])+'. '
            'No optimizer step, semantic judgment, application speed test or final source access occurred. '
            'This does not qualify the95percent complete-summary task goal.'}


def main(stage):
    if read(DEST/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed audit required')
    result = read(CAMPAIGN)
    for name, expected in result['artifacts'].items():
        if file_hash(name) != expected:
            raise ValueError('completed evidence changed')
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
    receipt = read(DEST/(HYPOTHESIS+'_registry_readback.json'))
    stored = receipt['record']
    exact = collection.get(ids=[HYPOTHESIS], include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']], n_results=10)
    if (any(stored.get(k) != v for k, v in record.items()) or exact['ids'] != [HYPOTHESIS]
            or any(exact['metadatas'][0].get(k) != v for k, v in stored.items())
            or read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json')) != receipt
            or HYPOTHESIS not in semantic['ids'][0] or stored['campaign_sha256'] != file_hash(CAMPAIGN)
            or stored['report_sha256'] != file_hash(REPORT)):
        raise ValueError('independent backend and JSON readback differ')
    result = {'status': 'independent_registry_readback_passed', 'hypothesis_id': HYPOTHESIS,
        'campaign_sha256': file_hash(CAMPAIGN), 'report_sha256': file_hash(REPORT)}
    with (DEST/'backend_exact_readback.json').open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('register', 'readback'))
    main(parser.parse_args().stage)
