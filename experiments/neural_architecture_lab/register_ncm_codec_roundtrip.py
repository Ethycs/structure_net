"""Conservative local registration of the completed codec component diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

ROOT = Path('data/experiments/ncm_paper_regime/20261025_codec_roundtrip_results_v1')
CAMPAIGN = ROOT/'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-codec-roundtrip.md')
ID = 'ncm-summary-codec-roundtrip-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(stage):
    value=read(CAMPAIGN)
    if (value['outputs']!=400 or value['cases']!=80 or value['source_only_task_accuracy'] is not None
            or read(ROOT/'actual_tool_receipt.json')['actual_tool_exit_code']!=0):
        raise ValueError('complete independently replayed component diagnostic required')
    for path, expected in value['artifacts'].items():
        if sha(Path(path))!=expected:raise ValueError('artifact changed')
    if stage=='register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(CAMPAIGN,REPORT,ROOT,{ID:{
            'name':'NCM summary codec round trip: representation, decoder and source attention',
            'description':'Five fixed arms on 64 reserved summaries plus 16 TRAIN controls, with correct target codes. '
                'Repaired codec alone: 99.29% content-token agreement and 45/64 exact native-rendered summaries. '
                'Latest adapted decoder: 98.56% alone, 92.65% with source attention and 3/64 exact. '
                'All400 outputs independently replayed; no source-only task accuracy or speed claim.',
            'classification':value['classification'],'component_cases':64,'TRAIN_control_cases':16,
            'oracle_target_codes':True,'source_only':False,'semantic_accuracy_measured':False,
            'repaired_exact':45,'adapted_with_source_exact':3,'reconstruction_gate_passed':False}})
        print(json.dumps({'status':'registered_and_read_back','hypothesis':ID}))
        return
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    collection=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db')).hypotheses_collection
    expected=read(ROOT/(ID+'_registry_readback.json'))['record']
    actual=collection.get(ids=[ID],include=['metadatas'])
    semantic=collection.query(query_texts=[expected['description']],n_results=10)
    if (actual['ids']!=[ID] or any(actual['metadatas'][0].get(k)!=v for k,v in expected.items())
            or read(Path('data/hypothesis_registry')/(ID+'.json'))['record']!=expected
            or ID not in semantic['ids'][0] or expected['report_sha256']!=sha(REPORT)
            or expected['campaign_sha256']!=sha(CAMPAIGN)):
        raise ValueError('independent registry readback differs')
    result={'status':'independent_exact_and_semantic_registry_readback_passed',
            'hypothesis_id':ID,'campaign_sha256':sha(CAMPAIGN),'report_sha256':sha(REPORT),
            'registrar_sha256':sha(Path(__file__))}
    with (ROOT/'backend_exact_readback.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('register','readback'))
    main(p.parse_args().stage)
