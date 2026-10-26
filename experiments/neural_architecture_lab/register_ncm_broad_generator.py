"""Register stabilized broader-generator evidence and verify a fresh readback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if not hasattr(np, 'NaN'): np.NaN = np.nan

ROOT = Path('data/experiments/ncm_paper_regime/20261025_broad_generator_results_v1')
CAMPAIGN = ROOT/'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-broad-generator.md')
HYPOTHESIS = 'ncm-broad-generator-v1'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def main(stage):
    result = read(CAMPAIGN)
    if (read(ROOT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or result['hypothesis_id'] != HYPOTHESIS or result['cases'] != 128
            or result['training_updates'] != 4096 or result['training_cases'] != 1376
            or result['final_test_opened'] or result['speed_tested'] or result['final_task_goal_achieved']):
        raise ValueError('completed broad-generator DEV evidence required; final goal remains unproven')
    for path, digest in result['artifacts'].items():
        if sha(path) != digest: raise ValueError('completed artifact changed: '+path)
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        record = {'name':'Broader complete-summary NCM generator with frozen reconstruction',
                  'description':f"Fixed 4,096 updates on 1,376 complete-summary teacher pairs; all 128 source-disjoint DEV outputs. "
                      f"Terra valid {result['valid']}/128, with {result['passed']} passes. "
                      f"Joint DEV qualification: {result['DEV_qualified']}. Codec and word-source path unchanged; "
                      'sealed final 320 cases and speed evaluation remain closed.',
                  'classification':result['classification'],'DEV_qualified':result['DEV_qualified'],
                  'DEV_cases':128,'DEV_valid':result['valid'],'DEV_passed':result['passed'],
                  'source_only':True,'canonical_first_judgments':result['canonical_first_judgments'],
                  'all128_cases_retained':True,'final_task_accuracy_measured':False}
        if result['task_accuracy'] is not None: record['DEV_accuracy'] = result['task_accuracy']
        if result['joint_task_accuracy'] is not None: record['DEV_joint_accuracy'] = result['joint_task_accuracy']
        register_records(CAMPAIGN,REPORT,ROOT,{HYPOTHESIS:record})
        print(json.dumps({'status':'registered_and_read_back','hypothesis':HYPOTHESIS}))
        return
    import chromadb
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    receipt = read(ROOT/(HYPOTHESIS+'_registry_readback.json'))
    record = receipt['record']
    exact = collection.get(ids=[HYPOTHESIS],include=['metadatas'])
    semantic = collection.query(query_texts=[record['description']],n_results=10)
    local = read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json'))
    if (exact['ids'] != [HYPOTHESIS] or any(exact['metadatas'][0].get(k) != v for k,v in record.items())
            or local != receipt or HYPOTHESIS not in semantic['ids'][0]
            or record['campaign_sha256'] != sha(CAMPAIGN) or record['report_sha256'] != sha(REPORT)):
        raise ValueError('independent registry exact/semantic/JSON/hash readback failed')
    value = {'status':'independent_exact_and_semantic_registry_readback_passed',
             'hypothesis_id':HYPOTHESIS,'classification':result['classification'],
             'DEV_cases':128,'DEV_valid':result['valid'],'DEV_passed':result['passed'],
             'campaign_sha256':sha(CAMPAIGN),'report_sha256':sha(REPORT),'registrar_sha256':sha(__file__)}
    with (ROOT/'backend_exact_readback.json').open('x') as output:
        json.dump(value,output,indent=2,allow_nan=False)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('register','readback'))
    main(parser.parse_args().stage)
