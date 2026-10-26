"""Conservatively register the frozen public-summary codec diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('data/experiments/ncm_paper_regime/20261026_public_xsum_codec_results_v2')
CAMPAIGN = ROOT/'campaign_results.json'
REPORT = Path('docs/08 - Analysis/2026-10-26_ncm-public-xsum-codec.md')
HYPOTHESIS = 'ncm-public-xsum-codec-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def record_for(result):
    if (result['hypothesis_id'] != HYPOTHESIS or result['outputs'] != 672
            or result['counts'] != {'representative': 256, 'long_stress': 32, 'digit_stress': 32, 'TRAIN_control': 16}
            or not result['all_raw_metrics_replayed'] or not result['all_prior_TRAIN_controls_equal']
            or not result['pilot_full_equal'] or not result['selection_replayed_from_163051_candidates']
            or not result['no_source_attention_or_generator'] or not result['target_aware_diagnostic']
            or result['model_training'] or result['new_judge_requests'] or result['final_task_goal_achieved']
            or result['source_only_task_accuracy'] is not None or result['training_data_admitted']):
        raise ValueError('complete component-only diagnostic required')
    record = {'name': 'Frozen codec reconstruction on candidate public XSum training summaries',
        'classification': result['classification'], 'representative_cases': 256,
        'stress_cases': 64, 'TRAIN_controls': 16, 'matched_codecs': 2,
        'target_aware_diagnostic': True, 'final_task_accuracy_measured': False,
        'model_training': False, 'training_data_admitted': False}
    descriptions = []
    for arm in ('released', 'repaired'):
        metric = result['metrics']['representative'][arm]
        if metric['cases'] != 256: raise ValueError('full representative denominators required')
        for key in ('exact_rendered', 'content_matches', 'content_tokens', 'has_digits', 'digit_cases_preserved', 'descriptive_reconstruction_gate'):
            record[arm+'_'+key] = metric[key]
        descriptions.append(f"{arm}: {metric['exact_rendered']}/256 exact, "
            f"{metric['content_matches']}/{metric['content_tokens']} content tokens, "
            f"{metric['digit_cases_preserved']}/{metric['has_digits']} digit sequences")
    record['description'] = ('Fixed target-aware reconstruction of unchanged public TRAIN summaries, '
        'comparing released and repaired codecs without source attention or a generator. '+ '; '.join(descriptions)+
        '. Full pilot, control, tensor and metric bindings replayed. This does not measure source-only summarization accuracy, '
        'verify reference factuality, or admit a larger training run. Final320 remains sealed.')
    return record


def main(stage):
    if read(ROOT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0: raise ValueError('completed aggregate required')
    result = read(CAMPAIGN)
    for name, expected in result['artifacts'].items():
        if sha(name) != expected: raise ValueError('evidence changed: '+name)
    record = record_for(result)
    import numpy as np
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    if stage == 'register':
        from experiments.neural_architecture_lab.ncm_result_registry import register_records
        register_records(CAMPAIGN, REPORT, ROOT, {HYPOTHESIS: record})
        print(json.dumps({'status': 'registered_and_read_back', 'hypothesis': HYPOTHESIS}))
        return
    import chromadb
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    receipt = read(ROOT/(HYPOTHESIS+'_registry_readback.json'))
    stored = receipt['record']
    exact = collection.get(ids=[HYPOTHESIS], include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']], n_results=10)
    if (any(stored.get(k) != v for k, v in record.items()) or exact['ids'] != [HYPOTHESIS]
            or any(exact['metadatas'][0].get(k) != v for k, v in stored.items())
            or read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json')) != receipt
            or HYPOTHESIS not in semantic['ids'][0] or stored['campaign_sha256'] != sha(CAMPAIGN)
            or stored['report_sha256'] != sha(REPORT)):
        raise ValueError('independent registry readback differs')
    value = {'status': 'independent_registry_readback_passed', 'hypothesis_id': HYPOTHESIS,
        'classification': result['classification'], 'campaign_sha256': sha(CAMPAIGN), 'report_sha256': sha(REPORT)}
    with (ROOT/'backend_exact_readback.json').open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('register', 'readback'))
    main(parser.parse_args().stage)
