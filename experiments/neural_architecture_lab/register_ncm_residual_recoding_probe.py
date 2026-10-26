"""Conservatively register the reference-assisted residual capacity diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

ROOT = Path('data/experiments/ncm_paper_regime/20261025_residual_recoding_probe_results_v1')
CAMPAIGN = ROOT / 'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-residual-recoding-probe.md')
HYPOTHESIS = 'ncm-residual-recoding-probe-v1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main(stage):
    result = read(CAMPAIGN)
    if (read(ROOT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or result['hypothesis_id'] != HYPOTHESIS or result['cases'] != 16
            or result['comparisons'] != 256 or result['readouts'] != 512
            or result['task_accuracy'] is not None or result['source_only_inference']
            or not result['oracle_target_access'] or result['final_task_goal_achieved']):
        raise ValueError('complete oracle diagnostic required; task accuracy is unmeasured')
    for p, expected in result['artifacts'].items():
        if sha(p) != expected:
            raise ValueError('completed diagnostic artifact changed: ' + p)
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        record = {'name':'Frozen NCM residual recoding capacity on generated prefixes',
            'description':'Reference-assisted diagnostic on sixteen fixed TRAIN summaries. Requantizing the '
                'remaining three levels after nine generated levels raises active-token reconstruction '
                'from 62.84% to 81.20%, with all sixteen cases improving. No training or semantic accuracy '
                'improvement established; the latest source-only DEV result remains zero out of128.',
            'classification':result['classification'],'TRAIN_cases':16,'comparisons':256,
            'oracle_target_access':True,'source_only_inference':False,
            'task_accuracy_measured':False,'final_task_goal_achieved':False}
        register_records(CAMPAIGN, REPORT, ROOT, {HYPOTHESIS:record})
        print(json.dumps({'status':'registered_and_read_back','hypothesis_id':HYPOTHESIS}))
        return
    import chromadb
    c = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    receipt = read(ROOT/(HYPOTHESIS+'_registry_readback.json'))
    record = receipt['record']
    exact = c.get(ids=[HYPOTHESIS],include=['metadatas'])
    semantic = c.query(query_texts=[record['description']],n_results=10)
    if (exact['ids'] != [HYPOTHESIS] or HYPOTHESIS not in semantic['ids'][0]
            or any(exact['metadatas'][0].get(k) != v for k,v in record.items())
            or read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json')) != receipt
            or record['campaign_sha256'] != sha(CAMPAIGN) or record['report_sha256'] != sha(REPORT)):
        raise ValueError('independent exact/semantic/JSON/hash readback failed')
    value = {'status':'independent_exact_and_semantic_registry_readback_passed',
        'hypothesis_id':HYPOTHESIS,'classification':result['classification'],
        'campaign_sha256':sha(CAMPAIGN),'report_sha256':sha(REPORT),'registrar_sha256':sha(__file__)}
    with (ROOT/'backend_exact_readback.json').open('x') as out:
        json.dump(value,out,indent=2,allow_nan=False)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('register','readback'))
    main(parser.parse_args().stage)
