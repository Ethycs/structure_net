"""Register and independently read back the completed decoder-adaptation study.

Metadata only; uses the existing default Pixi environment for Chroma.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# Compatibility with installed Chroma 0.4; no environment/package mutation.
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

ROOT = Path('data/experiments/ncm_paper_regime/20261025_generated_code_decoder_results_v1')
CAMPAIGN = ROOT / 'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-generated-code-decoder.md')
HYPOTHESIS = 'ncm-generated-code-decoder-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(stage):
    result = read(CAMPAIGN)
    if (read(ROOT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or result['hypothesis_id'] != HYPOTHESIS or result['valid'] != 128
            or result['passed'] != 0 or result['classification'] != 'DEV_semantic_quality_failed'
            or result['final_test_opened'] or result['speed_tested']):
        raise ValueError('Expected independently finalized, complete failed DEV study')
    for path, digest in result['artifacts'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Completed artifact changed: '+path)
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(CAMPAIGN, REPORT, ROOT, {HYPOTHESIS: {
            'name': 'NCM existing decoder adapted to generated codes: complete semantic DEV',
            'description': 'Fixed 1024 decoder-only updates on 732 engineering summaries using actual '
                'frozen-generator codes. TRAIN word agreement rose from 36.88% to 42.38%, but '
                'all 128 source-disjoint complete-summary judgments failed under Terra. '
                'Frozen tensors and checkpoint reload independently verified; final and speed gates closed.',
            'classification': result['classification'], 'DEV_qualified': False,
            'DEV_cases': 128, 'DEV_passed': 0, 'DEV_accuracy': 0., 'source_only': True,
            'canonical_first_judgments': result['canonical_first_judgments'],
            'all128_cases_retained': True, 'final_task_accuracy_measured': False}})
        print(json.dumps({'status': 'registered_and_read_back', 'hypothesis': HYPOTHESIS}))
        return
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    collection = StandardizedLogger(LoggingConfig(
        project_name='structure_net_completed_hypotheses', enable_wandb=False, auto_upload=False,
        enable_chromadb=True, chromadb_path='data/chroma_db')).hypotheses_collection
    if collection is None:
        raise ValueError('Registry unavailable')
    record = read(ROOT/(HYPOTHESIS+'_registry_readback.json'))['record']
    exact = collection.get(ids=[HYPOTHESIS], include=['metadatas'])
    semantic = collection.query(query_texts=[record['description']], n_results=10)
    local = read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json'))
    if (exact['ids'] != [HYPOTHESIS] or any(exact['metadatas'][0].get(k) != v for k, v in record.items())
            or local['record'] != record or HYPOTHESIS not in semantic['ids'][0]
            or record['campaign_sha256'] != sha(CAMPAIGN) or record['report_sha256'] != sha(REPORT)):
        raise ValueError('Independent exact, semantic or file readback failed')
    value = {'status': 'independent_exact_and_semantic_registry_readback_passed',
        'hypothesis_id': HYPOTHESIS, 'classification': result['classification'],
        'DEV_passed': 0, 'DEV_cases': 128, 'campaign_sha256': sha(CAMPAIGN),
        'report_sha256': sha(REPORT), 'registrar_sha256': sha(Path(__file__)),
        'compatibility': 'process-local np.NaN alias for existing Chroma; model environment unchanged'}
    with (ROOT/'backend_exact_readback.json').open('x') as f:
        json.dump(value, f, indent=2)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('register', 'readback'))
    main(parser.parse_args().stage)
