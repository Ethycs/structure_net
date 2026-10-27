"""Report target-aware code interventions; never reinterpret them as task accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_oracle_prefix as study

data,ROOT = study.data,study.ROOTS['results']
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-oracle-prefix.md')


def report():
    audit = study.ROOTS['audit'];study.controls.receipt(audit,('result.json',))
    value = data.completed(audit);study.prior.verify(data.read(study.ROOTS['run']/'producer.json'))
    if value['status'] != 'all320_oracle_prefix_outputs_replayed':raise ValueError('complete independent replay required')
    if ROOT.exists() or REPORT.exists():raise ValueError('fresh report required')
    lines = ['# Correct early codes, then generate the remaining hierarchy','',
        f"Classification: `{value['classification']}`. This intervention supplies known target codes; it does not measure source-only summarization accuracy.",'',
        '| Correct codes supplied | Remaining-code agreement | Content-token agreement | Exact native texts | Digit sequences preserved |',
        '|---:|---:|---:|---:|---:|']
    for arm in study.ARMS:
        v = value['scores'][arm]
        remaining = 'No generation remains' if v['remaining_code_agreement'] is None else f"{v['remaining_code_agreement']:.2%}"
        lines.append(f"| {v['injected_codes_per_case']} | {remaining} | {v['content_agreement']:.2%} | {v['exact_native_texts']}/64 | {v['digit_sequences_preserved']}/{v['digit_cases']} |")
    lines += ['','Changes versus the uncorrected baseline on the same remaining positions:','',
        '| Correct codes supplied | Remaining-code gain | Content-token gain | Joint descriptive gate |',
        '|---:|---:|---:|---|']
    for arm,v in value['effects'].items():
        lines.append(f"| {study.ARMS[arm]} | {100*v['remaining_code_gain']:.2f} pp | {100*v['content_gain']:.2f} pp | {v['joint_gate_passed']} |")
    lines += ['',
        'The fixed joint gate requires at least10 percentage points of remaining-code improvement and20 percentage points of content-token improvement with at most25 injected codes. Injected positions never count as generated successes. The all641-code reconstruction control cannot pass this gate.',
        'All64 previously reserved and inspected component sources were used in their original order, with the same epoch2 checkpoint and full original Llama features. Correct prefixes contain1,9 or25 codes at complete level boundaries; later target codes never enter that generation call. Different valid summaries can have different codes, so these exact comparisons are component diagnostics.',
        'Every no-injection output exactly reproduced the prior source-only codes, words and text. Every all641-code output exactly reproduced the repaired-parent source-attention codec control. The two-case pilot repeated all conditions exactly and matched the full run. An independent CPU audit re-rendered raw words and replayed all320 outputs, metrics, controls and gates.',
        'The run used locked Pixi ncm, physicalGPU0 RTX3060, FP32, TF32 off and a256MiB feature cache within4.9GB peak allocation. Full model hashes remained unchanged; no gradients, training, new semantic judgments, final-test access or speed measurement occurred. The epoch4 trainer continued unchanged on the RTX2060.',
        'Application semantic DEV remains0/128. Supplying correct codes does not show that a learned model can predict them, and this experiment cannot qualify the95% goal.',
        f"Raw evidence: `{study.ROOTS['run']}`."]
    ROOT.mkdir();REPORT.write_text('\n'.join(lines)+'\n')
    paths = [audit/'result.json',audit/'actual_tool_receipt.json',study.ROOTS['run']/'result.json',
             study.ROOTS['run']/'actual_tool_receipt.json',REPORT,Path(__file__)]
    data.write(ROOT/'campaign_results.json',{**value,'report':str(REPORT),
        'artifacts':{str(p):data.sha(p) for p in paths}})
    print(json.dumps({'classification':value['classification'],'report':str(REPORT)}))


def register():
    p = ROOT/'campaign_results.json';receipt = data.read(ROOT/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(p)) != data.sha(p):raise ValueError('actual report receipt required')
    data.verify(receipt['artifacts']);value = data.read(p);data.verify(value['artifacts'])
    record = {'name':'Reserved NCM early oracle-code intervention','classification':value['classification'],
        'application_final_accuracy_measured':False,'target_aware_diagnostic':True,'cases':64,
        'passing_interventions':','.join(value['passing_interventions']),
        'description':'Frozen public epoch2 NCM on all64 previously reserved component sources. Inject0,1,9,25 or641 true codes; generate only the remaining hierarchy. Exclude injected positions from generated-code metrics. Replay320 outputs and exact source-only/oracle reconstruction controls. Target-aware component diagnostic, not learned correction, task accuracy or speed evidence.'}
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(p,REPORT,ROOT,{study.HYPOTHESIS:record});print(json.dumps({'status':'registered_and_read_back'}))


def readback():
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    h = study.HYPOTHESIS;receipt = data.read(ROOT/(h+'_registry_readback.json'));record = receipt['record']
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[h],include=['metadatas']);semantic = collection.query(query_texts=[record['description']],n_results=10)
    if (exact['ids'] != [h] or any(exact['metadatas'][0].get(k) != v for k,v in record.items()) or h not in semantic['ids'][0]
            or record['campaign_sha256'] != data.sha(ROOT/'campaign_results.json') or record['report_sha256'] != data.sha(REPORT)
            or record['application_final_accuracy_measured'] or not record['target_aware_diagnostic']
            or data.read(Path('data/hypothesis_registry')/(h+'.json')) != receipt):raise ValueError('independent registry readback differs')
    value = {'status':'independent_registry_readback_passed','hypothesis_id':h,
        'campaign_sha256':data.sha(ROOT/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(ROOT/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('report','register','readback'))
    globals()[parser.parse_args().stage]()
