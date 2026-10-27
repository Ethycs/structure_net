"""Report a public TRAIN diagnostic without treating agreement as task accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_public_pair_fit_probe as study

data = study.data
ROOT = data.ROOT/'20261027_public_pair_fit_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-public-pair-fit.md')


def record_for(value):
    if (value['hypothesis_id'] != study.HYPOTHESIS or value['cases'] != 32
            or value['task_accuracy'] is not None or value['model_updates'] != 0
            or value['new_judge_requests'] != 0 or value['final_test_opened']
            or value['semantic_generalization_established']
            or value['undertraining_causally_established']):
        raise ValueError('read-only exploratory TRAIN diagnostic required')
    scores = value['scores']
    if (set(scores) != {'BROAD','epoch2'}
            or any(set(modes) != {'matched','shuffled'} for modes in scores.values())
            or any(s['code_positions'] != 32*641 for modes in scores.values() for s in modes.values())):
        raise ValueError('both checkpoints and source conditions on all 32 cases required')
    delta = scores['epoch2']['matched']['rollout_correct_codes']-scores['BROAD']['matched']['rollout_correct_codes']
    return {'name':'NCM public TRAIN pair learning diagnostic',
        'classification':'public_TRAIN_rollout_'+('improved' if delta>0 else 'decreased' if delta<0 else 'unchanged'),
        'cases':32,'matched_correct_code_change':delta,'task_accuracy_measured':False,
        'description':f'Fixed hash-selected 32 public TRAIN pairs; BROAD versus epoch2, matched and cyclically shuffled sources. Matched rollout changed by {delta} of 20512 codes. Teacher-forced, generated-code and native-word agreement are exact representation diagnostics, not semantic accuracy. No generalization, causal undertraining or speed claim. No model updates or new judgments; final test remained sealed.'}


def report():
    audit = study.ROOTS['audit']
    study.reference.receipt(audit)
    value = data.completed(audit)
    if value['status'] != 'public_pair_fit_independently_replayed':
        raise ValueError('successful independent audit required')
    value = {**value,'scores':data.read(audit/'scores.json')}
    record = record_for(value)
    if ROOT.exists() or REPORT.exists():raise ValueError('fresh report and aggregate required')
    paths = [Path(__file__),audit/'producer.json',audit/'result.json',audit/'actual_tool_receipt.json',
        audit/'scores.json',*[study.ROOTS[s]/n for s in ('pilot','run') for n in ('producer.json','result.json','actual_tool_receipt.json')]]
    campaign = {**value,'classification':record['classification'],'report':str(REPORT),
        'artifacts':{str(p):data.sha(p) for p in paths}}
    ROOT.mkdir();data.write(ROOT/'campaign_results.json',campaign)
    lines = ['# Fit on public summarization training pairs','',
        f"Classification: `{record['classification']}`. Hypothesis: `{study.HYPOTHESIS}`.",
        f"Aggregate: `{ROOT/'campaign_results.json'}`; SHA256 `{data.sha(ROOT/'campaign_results.json')}`.",'',
        '| Checkpoint | Source | Teacher code agreement | Generated code agreement | Native word agreement |',
        '|---|---|---:|---:|---:|']
    for arm,modes in value['scores'].items():
        for mode,s in modes.items():
            lines.append(f"| {arm} | {mode} | {s['teacher_code_agreement']:.2%} | {s['rollout_code_agreement']:.2%} | {s['native_word_agreement']:.2%} |")
    lines += ['',
        'The sample contains 32 public TRAIN pairs selected by a fixed hash from all 162,353 admitted pairs before inspecting outputs. Each public pair had two presentations by epoch2. The earlier 16-case TRAIN result instead measured frequently repeated application anchors. These populations differ, so their score difference does not isolate repetition as a cause.',
        'Each checkpoint generated from complete source features alone. Teacher forcing was a separate diagnostic. Shuffled conditions cyclically permuted sources while retaining targets. The two-case pilot included the longest selected source, repeated exactly under both checkpoints, and reproduced in the full run. CPU replay checked selection, source pairing, raw counts and rendered words.',
        'Exact reference code or word agreement is not semantic summary accuracy: valid alternative summaries may disagree, and public references are not individually verified factual gold. This small training sample cannot establish generalization, prove that additional training will solve the failure, or qualify the 95% goal. Application semantic DEV remains 0/128 at the last completed evaluation.',
        'Locked Pixi, physical GPU2 RTX2060SUPER, FP32 inference, TF32 off; one source feature matrix streamed at a time. Model tensors and both ongoing training service identities remained unchanged. No gradients, model updates, new judges, final-test access or speed measurement occurred.']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'classification':record['classification'],'report':str(REPORT)}))


def checked():
    receipt = data.read(ROOT/'actual_tool_receipt.json')
    if (receipt['actual_tool_exit_code'] != 0
            or any(receipt['artifacts'].get(str(p)) != data.sha(p) for p in (ROOT/'campaign_results.json',REPORT))):
        raise ValueError('actual report execution receipt required')
    data.verify(receipt['artifacts'])
    value = data.read(ROOT/'campaign_results.json');data.verify(value['artifacts'])
    record = record_for(value)
    if value['classification'] != record['classification']:raise ValueError('classification changed')
    return record


def register():
    record = checked()
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(ROOT/'campaign_results.json',REPORT,ROOT,{study.HYPOTHESIS:record})
    print(json.dumps({'status':'registered_and_read_back','hypothesis':study.HYPOTHESIS}))


def readback():
    record = checked()
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    h = study.HYPOTHESIS;receipt = data.read(ROOT/(h+'_registry_readback.json'));stored = receipt['record']
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[h],include=['metadatas'])
    semantic = collection.query(query_texts=[stored['description']],n_results=10)
    if (any(stored.get(k) != v for k,v in record.items()) or exact['ids'] != [h]
            or any(exact['metadatas'][0].get(k) != v for k,v in stored.items())
            or h not in semantic['ids'][0] or stored['campaign_sha256'] != data.sha(ROOT/'campaign_results.json')
            or stored['report_sha256'] != data.sha(REPORT)
            or data.read(Path('data/hypothesis_registry')/(h+'.json')) != receipt):
        raise ValueError('independent registry readback differs')
    value = {'status':'independent_registry_readback_passed','hypothesis_id':h,
        'campaign_sha256':data.sha(ROOT/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(ROOT/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('report','register','readback'))
    globals()[parser.parse_args().stage]()
