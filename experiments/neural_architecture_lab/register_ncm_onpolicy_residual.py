"""Register the measured matched NCM comparison and verify independent readback."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('data/experiments/ncm_paper_regime/20261025_onpolicy_residual_results_v1')
CAMPAIGN = ROOT/'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-onpolicy-residual.md')
HYPOTHESIS = 'ncm-onpolicy-residual-v1'
ARMS = ('fixed_targets','residual_targets')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def record_for(result):
    if (result['hypothesis_id'] != HYPOTHESIS or set(result['arms']) != set(ARMS)
            or result['cases_per_arm'] != 128 or result['independent_training_seeds'] != 1
            or result['inherited_first_judgments'] != 3923 or result['canonical_first_judgments'] < 3923
            or not result['same_initialization'] or not result['same_sampling'] or not result['source_only']
            or not result['all_historical_and_pilot_firsts_preserved']
            or result['final_test_opened'] or result['speed_tested'] or result['final_task_goal_achieved']):
        raise ValueError('complete audited two-arm DEV comparison required; final goal remains unproven')
    record = {'name':'NCM learning from generated histories with residual code correction',
        'classification':result['classification'],'parent_hypothesis_id':'ncm-broad-generator-v1',
        'paired_DEV_cases':128,'matched_arms':2,'training_updates_per_arm':1024,'training_cases':1376,
        'source_only':True,'all128_cases_retained_per_arm':True,
        'canonical_first_judgments':result['canonical_first_judgments'],
        'comparison_has_unresolved_judgments':result['comparison_has_unresolved_judgments'],
        'selected_candidate':result['selected_candidate'] or 'none',
        'final_task_accuracy_measured':False,'frozen_reconstruction_preserved':True}
    descriptions = []
    for arm in ARMS:
        value = result['arms'][arm]
        if (value['arm'] != arm or value['cases'] != 128 or value['training_updates'] != 1024
                or value['training_cases'] != 1376 or not value['frozen_reconstruction_preserved']
                or value['final_test_opened'] or value['speed_tested']):
            raise ValueError('both fixed-final checkpoints and complete128 denominators required')
        for key in ('valid','passed','DEV_qualified','classification'):
            record[arm+'_'+key] = value[key]
        for key in ('task_accuracy','joint_task_accuracy'):
            if value[key] is not None: record[arm+'_'+key] = value[key]
        descriptions.append(f"{arm}: Terra {value['passed']} passes, {value['valid']}/128 valid; "
                            f"joint DEV qualification {value['DEV_qualified']}")
    record['description'] = ('Matched 1,024-update continuations from the same frozen-codec BROAD model on '
        '1,376 complete-summary TRAIN pairs. Both see their own current generated histories; only residual_targets '
        'recomputes later labels to compensate for historical prediction errors. '+ '; '.join(descriptions)+
        '. All prior first judgments retained. One matched training seed; sealed final320 and speed evaluation remain closed.')
    return record


def main(stage):
    result = read(CAMPAIGN)
    if read(ROOT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed aggregate receipt required')
    record = record_for(result)
    for path,digest in result['artifacts'].items():
        if sha(path) != digest: raise ValueError('completed artifact changed: '+path)
    # Compatibility with the existing registry environment; no model runtime is loaded.
    import numpy as np
    if not hasattr(np,'NaN'): np.NaN = np.nan
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(CAMPAIGN,REPORT,ROOT,{HYPOTHESIS:record})
        print(json.dumps({'status':'registered_and_read_back','hypothesis':HYPOTHESIS}))
        return
    import chromadb
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    receipt = read(ROOT/(HYPOTHESIS+'_registry_readback.json'))
    stored = receipt['record']
    exact = collection.get(ids=[HYPOTHESIS],include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']],n_results=10)
    local = read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json'))
    if (any(stored.get(k) != v for k,v in record.items())
            or exact['ids'] != [HYPOTHESIS] or any(exact['metadatas'][0].get(k) != v for k,v in stored.items())
            or local != receipt or HYPOTHESIS not in semantic['ids'][0]
            or stored['campaign_sha256'] != sha(CAMPAIGN) or stored['report_sha256'] != sha(REPORT)):
        raise ValueError('independent registry exact/semantic/JSON/hash readback failed')
    value = {'status':'independent_exact_and_semantic_registry_readback_passed','hypothesis_id':HYPOTHESIS,
        'classification':result['classification'],'DEV_cases_per_arm':128,
        'selected_candidate':result['selected_candidate'],'campaign_sha256':sha(CAMPAIGN),
        'report_sha256':sha(REPORT),'registrar_sha256':sha(__file__)}
    with (ROOT/'backend_exact_readback.json').open('x') as output:
        json.dump(value,output,indent=2,allow_nan=False)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('register','readback'))
    main(parser.parse_args().stage)
