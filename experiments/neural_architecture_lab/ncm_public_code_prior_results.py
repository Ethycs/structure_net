"""Conservative reporting and registry entry for source-blind code counts."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_public_code_prior as study

data,ROOT,REPORT=study.data,study.ROOTS['results'],study.REPORT


def record_for(value):
    if (value['hypothesis_id']!=study.HYPOTHESIS or value['fitted']!=162321
            or value['task_accuracy'] is not None or value['model_updates']!=0
            or value['new_judge_requests']!=0 or value['final_test_opened']
            or not value['independent_counter_replay']):
        raise ValueError('complete independently replayed exploratory count analysis required')
    measured=value['measurements'];scores=measured['scores']
    if (set(scores)!={'positional_mode','BROAD_matched','BROAD_shuffled','epoch2_matched','epoch2_shuffled'}
            or any(s['positions']!=32*641 for s in scores.values()) or len(measured['levels'])!=12):
        raise ValueError('all fixed cases, source conditions and levels required')
    delta=scores['positional_mode']['correct']-scores['epoch2_matched']['correct']
    expected='source_blind_mode_'+('matches_or_exceeds' if delta>=0 else 'below')+'_epoch2_code_agreement'
    if value['classification']!=expected or measured['classification']!=expected:
        raise ValueError('count comparison and classification differ')
    return {'name':'Public NCM positional code-prior diagnostic','classification':expected,
        'task_accuracy_measured':False,'fitted_public_TRAIN_cases':162321,'diagnostic_TRAIN_cases':32,
        'source_blind_minus_epoch2_correct_codes':delta,
        'description':f'Fit a source-blind modal code per position on 162321 public TRAIN hierarchies, excluding the fixed 32 diagnostic examples. On those examples the baseline minus epoch2 matched-source agreement is {delta} of 20512 codes. Independently replay all counts with Counters. Report per-level entropy and nonmodal agreement. No model update, semantic accuracy, speed or final-test claim; an independently selected modal hierarchy need not be coherent text.'}


def report():
    audit=study.ROOTS['audit'];study.probe.reference.receipt(audit);value=data.completed(audit)
    if value['status']!='public_code_prior_independently_replayed':raise ValueError('complete count audit required')
    record=record_for(value)
    if ROOT.exists() or REPORT.exists():raise ValueError('fresh report required')
    paths=[Path(__file__),audit/'producer.json',audit/'result.json',audit/'actual_tool_receipt.json',
        *[study.ROOTS['run']/n for n in ('producer.json','result.json','actual_tool_receipt.json','counts.npz','coverage.json')]]
    ROOT.mkdir();data.write(ROOT/'campaign_results.json',{**value,'artifacts':{str(p):data.sha(p) for p in paths}})
    lines=['# Public code agreement against a source-blind baseline','',
        f"Classification: `{record['classification']}`. Hypothesis: `{study.HYPOTHESIS}`.",
        f"Aggregate: `{ROOT/'campaign_results.json'}`; SHA256 `{data.sha(ROOT/'campaign_results.json')}`.",'',
        '| Predictor | Code agreement | Agreement on nonmodal reference codes | Nonmodal correct / positions |',
        '|---|---:|---:|---:|']
    for name,s in value['measurements']['scores'].items():
        ratio='N/A' if s['nonmodal_agreement'] is None else f"{s['nonmodal_agreement']:.2%}"
        lines.append(f"| {name} | {s['agreement']:.2%} | {ratio} | {s['nonmodal_correct']} / {s['nonmodal_positions']} |")
    lines+=['','| Level | Code positions | Mean positional entropy (bits) | Mean occupied codes | Mean training mode share |',
        '|---:|---:|---:|---:|---:|']
    for i,level in enumerate(value['measurements']['levels']):
        lines.append(f"| {i} | {level['length']} | {level['mean_position_entropy_bits']:.3f} | {level['mean_occupied_codes_per_position']:.1f} | {level['mean_mode_fraction']:.2%} |")
    lines+=['',
        'The baseline uses the most frequent code separately at each of 641 positions, breaking ties by the lowest code ID. It fits all 162,321 admitted public TRAIN examples outside the fixed 32-case probe. Source content is not an input to its predictions, and it runs no neural model. The model predictions are the exact saved BROAD and epoch2 matched/shuffled outputs from that probe.',
        'Nonmodal agreement counts only reference positions whose code differs from the fitted positional mode. Its denominator is fixed across predictors. These are exact-code diagnostics, not semantic summary scores. A modal hierarchy assembled from different examples need not decode to coherent text. Frequent codes cannot automatically be identified with padding, and positional entropy does not measure semantic information.',
        'This exploratory follow-up was specified after observing epoch2 matched/shuffled agreement of 32.85%/37.25%. All labels and TRAIN index bindings were checked. The 256-record lifecycle pilot reproduced in the full pass. An independent full pass using Python Counters reproduced every NumPy count, then replayed all reported measurements. It reuses the existing 32-case TRAIN artifact but opens no new source corpus, source feature matrices, DEV or final-test payloads.',
        'The ongoing public curriculum and matched application-data training studies remained unchanged. No GPU, model training, new semantic judgments or speed measurement was involved. This result cannot establish that more data or a particular new loss will solve the task. Application semantic DEV remains 0/128; the 95% complete-summary goal is unqualified.']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'classification':record['classification'],'report':str(REPORT)}))


def checked():
    receipt=data.read(ROOT/'actual_tool_receipt.json')
    if (receipt['actual_tool_exit_code']!=0 or any(receipt['artifacts'].get(str(p))!=data.sha(p)
            for p in (ROOT/'campaign_results.json',REPORT))):raise ValueError('actual report receipt required')
    data.verify(receipt['artifacts']);value=data.read(ROOT/'campaign_results.json');data.verify(value['artifacts'])
    return record_for(value)


def register():
    record=checked()
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(ROOT/'campaign_results.json',REPORT,ROOT,{study.HYPOTHESIS:record})
    print(json.dumps({'status':'registered_and_read_back','hypothesis':study.HYPOTHESIS}))


def readback():
    expected=checked()
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    h=study.HYPOTHESIS;receipt=data.read(ROOT/(h+'_registry_readback.json'));record=receipt['record']
    collection=chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact=collection.get(ids=[h],include=['metadatas']);semantic=collection.query(query_texts=[record['description']],n_results=10)
    if (any(record.get(k)!=v for k,v in expected.items()) or exact['ids']!=[h]
            or any(exact['metadatas'][0].get(k)!=v for k,v in record.items()) or h not in semantic['ids'][0]
            or record['campaign_sha256']!=data.sha(ROOT/'campaign_results.json') or record['report_sha256']!=data.sha(REPORT)
            or data.read(Path('data/hypothesis_registry')/(h+'.json'))!=receipt):raise ValueError('independent registry replay differs')
    value={'status':'independent_registry_readback_passed','hypothesis_id':h,
        'campaign_sha256':data.sha(ROOT/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(ROOT/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('report','register','readback'))
    globals()[parser.parse_args().stage]()
