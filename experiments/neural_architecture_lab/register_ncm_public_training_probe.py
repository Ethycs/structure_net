"""Register the fixed TRAIN diagnostic without implying semantic task success."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_public_training_probe as probe

data, DEST, HYPOTHESIS = probe.data, probe.ROOTS['results'], probe.HYPOTHESIS
CAMPAIGN = DEST/'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-public-training-probe.md')


def record_for(result):
    scores = result['scores']
    if (result['hypothesis_id'] != HYPOTHESIS or result['task_accuracy'] is not None
            or result['model_updates'] != 0 or result['new_judge_requests'] != 0
            or result['final_test_opened'] or set(scores) != {'BROAD', 'epoch2'}
            or any(set(conditions) != {'matched', 'shuffled'} for conditions in scores.values())
            or any(v['code_positions'] != 16*641 for c in scores.values() for v in c.values())):
        raise ValueError('complete unchanged sixteen-case TRAIN diagnostic required')
    delta = scores['epoch2']['matched']['rollout_correct_codes']-scores['BROAD']['matched']['rollout_correct_codes']
    expected = 'TRAIN_rollout_'+('improved' if delta > 0 else 'decreased' if delta < 0 else 'unchanged')
    if result['classification'] != expected or result['matched_rollout_correct_change'] != delta:
        raise ValueError('paired TRAIN change and classification differ')
    return {'name': 'Public NCM epoch2 fixed TRAIN-control diagnostic',
        'classification': expected, 'task_accuracy_measured': False,
        'matched_rollout_correct_change': delta, 'TRAIN_controls': 16,
        'description': 'Compared BROAD and public epoch2 on the original sixteen TRAIN controls, '
            'with matched and cyclically shuffled sources. Independently replayed teacher-forced '
            f'and rollout code counts; matched rollout changed by {delta} of 10256 codes. '
            'This diagnostic uses training examples and establishes no semantic accuracy, '
            'generalization or speed improvement. Both completed semantic DEV milestones remain '
            '0/128. No model update or new semantic judgment occurred; final test stayed sealed.'}


def main(stage):
    probe.receipt(DEST, ('campaign_results.json',))
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
