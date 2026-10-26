"""Register independently replayed public NCM milestones without final claims."""
import argparse
import json
from pathlib import Path

from experiments.neural_architecture_lab import finalize_ncm_public_curriculum as final

data = final.data


def record_for(result):
    epoch = result['epoch']
    hypothesis = f'ncm-public-curriculum-epoch{epoch}-v1'
    milestones = result['milestones']
    if (result['hypothesis_id'] != hypothesis or result['parent_hypothesis_id'] != 'ncm-public-curriculum-v1'
            or result['cases_per_milestone'] != 128 or result['independent_training_trajectories'] != 1
            or milestones[-1]['epoch'] != epoch or milestones[-1]['attempt'] != result['attempt']
            or result['classification'] != milestones[-1]['classification']
            or result['selected_candidate_epoch'] != final.selection(milestones)
            or result['canonical_first_judgments'] != milestones[-1]['canonical_first_judgments']
            or result['final_test_opened'] or result['final_task_goal_achieved'] or result['speed_tested']):
        raise ValueError('verified milestone result required; final goal remains unmeasured')
    current = milestones[-1]
    for row in milestones:
        if (row['cases'] != 128 or not row['source_only'] or row['final_test_opened']
                or row['final_task_goal_achieved'] or row['speed_tested']
                or row['training_updates'] != row['epoch']*final.study.CONFIG.updates_per_epoch
                or row['public_presentations'] != row['epoch']*final.study.CONFIG.public_cases
                or row['anchor_presentations'] != row['epoch']*final.study.CONFIG.anchors_per_epoch):
            raise ValueError('complete fixed training exposure and DEV-only evidence required')
    record = {'name': f'Public NCM curriculum epoch{epoch}', 'classification': result['classification'],
        'parent_hypothesis_id': 'ncm-public-curriculum-v1', 'epoch': epoch, 'attempt': result['attempt'],
        'milestones_completed': len(milestones), 'valid': current['valid'], 'passed': current['passed'],
        'DEV_cases': 128, 'DEV_qualified': current['DEV_qualified'],
        'training_updates': current['training_updates'], 'public_presentations': current['public_presentations'],
        'anchor_presentations': current['anchor_presentations'], 'canonical_first_judgments': result['canonical_first_judgments'],
        'selected_candidate_epoch': result['selected_candidate_epoch'] or 0,
        'final_task_accuracy_measured': False, 'source_only': True,
        'description': f"Frozen-body NCM public curriculum epoch{epoch}: Terra {current['passed']} complete-summary passes; "
            f"{current['valid']}/128 valid judgments; same-case joint DEV qualification {current['DEV_qualified']}. "
            'Every committed training segment and semantic first response was replayed. '
            'One continuing trajectory; final320 and inference speed remain unmeasured.'}
    for key in ('task_accuracy', 'joint_task_accuracy'):
        if current[key] is not None: record[key] = current[key]
    return record


def main(stage, epoch):
    root, report = final.roots(epoch)['results'], final.report_path(epoch)
    campaign = root/'campaign_results.json'
    final.receipt(root, ('campaign_results.json',))
    value = data.read(campaign); data.verify(value['artifacts'])
    record = record_for(value); hypothesis = value['hypothesis_id']
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(campaign, report, root, {hypothesis: record})
        print(json.dumps({'status': 'registered_and_read_back', 'hypothesis_id': hypothesis}), flush=True)
        return
    import chromadb
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    registered = data.read(root/(hypothesis+'_registry_readback.json'))
    stored = registered['record']
    exact = collection.get(ids=[hypothesis], include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']], n_results=10)
    local = data.read(Path('data/hypothesis_registry')/(hypothesis+'.json'))
    if (any(stored.get(k) != v for k, v in record.items()) or exact['ids'] != [hypothesis]
            or any(exact['metadatas'][0].get(k) != v for k, v in stored.items())
            or local != registered or hypothesis not in semantic['ids'][0]
            or stored['campaign_sha256'] != data.sha(campaign) or stored['report_sha256'] != data.sha(report)):
        raise ValueError('independent exact, semantic or JSON registry readback differs')
    result = {'status': 'independent_exact_and_semantic_registry_readback_passed',
        'hypothesis_id': hypothesis, 'classification': value['classification'],
        'campaign_sha256': data.sha(campaign), 'report_sha256': data.sha(report), 'registrar_sha256': data.sha(__file__)}
    data.write(root/'backend_exact_readback.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('register', 'readback'))
    parser.add_argument('--epoch', type=int, choices=final.study.CONFIG.milestones, required=True)
    args = parser.parse_args()
    main(args.stage, args.epoch)
