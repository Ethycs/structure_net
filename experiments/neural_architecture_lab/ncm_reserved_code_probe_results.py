"""Publish the reserved code diagnostic without treating it as task accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_reserved_code_probe as study

data = study.data
ROOT = study.ROOTS['results']
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-reserved-code-probe.md')


def report():
    audit = study.ROOTS['audit']
    study.controls.receipt(audit,('result.json',))
    result = data.completed(audit)
    if (result['status'] != 'reserved_code_raw_counts_replayed'
            or result['classification'] != study.classify(result['scores']['matched'])
            or any(v['cases'] != 64 for v in result['scores'].values())):
        raise ValueError('complete independently replayed64-source diagnostic required')
    study.verify(data.read(study.ROOTS['run']/'producer.json'))
    if ROOT.exists() or REPORT.exists():raise ValueError('fresh report required')
    lines = ['# Reserved summary codes: teacher forcing versus free generation','',
        f"Classification: `{result['classification']}`. Application semantic DEV remains0/128; this diagnostic does not measure summary accuracy.",'',
        '| Source condition | Teacher code agreement | Rollout code agreement | Gap | Root agreement | Native-word agreement |',
        '|---|---:|---:|---:|---:|---:|']
    for name,v in result['scores'].items():
        lines.append(f"| {name} | {v['teacher_code_agreement']:.2%} | {v['rollout_code_agreement']:.2%} | {v['teacher_rollout_gap']:.2%} | {v['root_agreement']:.2%} | {v['native_word_agreement']:.2%} |")
    lines += ['',
        'All64 previously reserved component summaries were included in their original order. The public epoch2 checkpoint, original Llama features, codec and source adapters remained fixed. Teacher forcing supplies correct earlier code levels; free generation receives only source features. Shuffling cyclically replaces the source while retaining the original target for diagnostic scoring.',
        'Teacher and rollout root predictions agreed for every condition. The frozen codec reproduced every original641-code target exactly. The first/longest-source pilot repeated exactly and matched the full run. Model hashes remained unchanged, no gradients were created, and GPU allocation stayed within4.9GB.',
        'Source identities were checked against all1376 reviewed and162353 public TRAIN pairs. Reviewed TRAIN text overlap was rechecked; the independently audited public full-text screen was reused only after verifying its binding to the exact64 component source file. These sources have been inspected in earlier component experiments and are not a sealed test.',
        'The descriptive exposure-error pattern requires at least90% teacher code agreement and a gap of at least20 percentage points. Teacher agreement below90% is classified as weak even with correct earlier codes. These thresholds diagnose an error pattern; they do not identify a unique cause. Different valid summaries may have different codes, and native-word agreement includes non-padding special tokens.',
        'Inference used physicalGPU0 RTX3060, FP32, TF32 off and locked Pixi ncm while the unchanged epoch4 curriculum ran on the RTX2060. No training, semantic judge requests, sealed-final access or speed measurement occurred.',
        f"Detailed per-level counts and raw outputs: `{study.ROOTS['run']}`."]
    ROOT.mkdir();REPORT.write_text('\n'.join(lines)+'\n')
    paths = [audit/'result.json',audit/'actual_tool_receipt.json',study.ROOTS['run']/'result.json',
        study.ROOTS['run']/'actual_tool_receipt.json',REPORT,Path(__file__)]
    value = {**result,'hypothesis_id':study.HYPOTHESIS,'report':str(REPORT),
        'artifacts':{str(p):data.sha(p) for p in paths}}
    data.write(ROOT/'campaign_results.json',value)
    print(json.dumps({'classification':value['classification'],'report':str(REPORT)}))


def register():
    p = ROOT/'campaign_results.json';receipt = data.read(ROOT/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(p)) != data.sha(p):
        raise ValueError('actual successful report receipt required')
    data.verify(receipt['artifacts']);v = data.read(p);data.verify(v['artifacts'])
    record = {'name':'Reserved-source NCM teacher forcing and rollout diagnosis',
        'classification':v['classification'],'application_final_accuracy_measured':False,
        'cases':64,'teacher_code_agreement':v['scores']['matched']['teacher_code_agreement'],
        'rollout_code_agreement':v['scores']['matched']['rollout_code_agreement'],
        'description':'Frozen public epoch2 NCM diagnostic on all64 reserved application component summaries, with source-matched and cyclically shuffled conditions. Raw teacher and rollout codes independently replayed. Previously inspected component data, not sealed task accuracy; no training, new judges, final-test access or speed measurement.'}
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(p,REPORT,ROOT,{study.HYPOTHESIS:record})
    print(json.dumps({'status':'registered_and_read_back'}))


def readback():
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    h = study.HYPOTHESIS;receipt = data.read(ROOT/(h+'_registry_readback.json'));record = receipt['record']
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[h],include=['metadatas'])
    semantic = collection.query(query_texts=[record['description']],n_results=10)
    if (exact['ids'] != [h] or any(exact['metadatas'][0].get(k) != v for k,v in record.items())
            or h not in semantic['ids'][0] or record['campaign_sha256'] != data.sha(ROOT/'campaign_results.json')
            or record['report_sha256'] != data.sha(REPORT) or record['application_final_accuracy_measured']
            or data.read(Path('data/hypothesis_registry')/(h+'.json')) != receipt):
        raise ValueError('independent registry readback differs')
    value = {'status':'independent_registry_readback_passed','hypothesis_id':h,
        'campaign_sha256':data.sha(ROOT/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(ROOT/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('report','register','readback'))
    globals()[parser.parse_args().stage]()
