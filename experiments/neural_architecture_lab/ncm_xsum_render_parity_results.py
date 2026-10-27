"""Publish the measured render mismatch without claiming corrected quality."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_xsum_render_parity as study

data, ROOT = study.data, study.ROOT
HYPOTHESIS = 'ncm-xsum-render-parity-v1'
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-xsum-render-parity.md')


def report():
    state = study.probe.base.require(ROOT)
    data.verify(data.read(ROOT/'producer.json')['source_and_input_hashes'])
    comparison = data.read(ROOT/'comparison.json')
    counts = {a:sum(r['arm'] == a and r['changed'] for r in comparison) for a in study.probe.ARMS}
    if (len(comparison) != 512 or len({(r['id'],r['arm']) for r in comparison}) != 512
            or counts != state['changed_text_counts'] or state['status'] != 'all512_author_text_decodes_match'
            or state['native_render_only_pass_upper_bound'] != state['unchanged_native_passes']+counts['native']):
        raise ValueError('complete render mismatch aggregate differs')
    campaign = ROOT/'campaign_results.json'
    if campaign.exists() or REPORT.exists():
        raise ValueError('fresh measured report required')
    lines = ['# XSum rendering mismatch: PAD is not the end token','',
        'Classification: `author_render_mismatch_verified`. This is a rendering diagnostic; corrected semantic accuracy remains unmeasured.','',
        'The frozen four-arm semantic probe used the local decoder that stops at either EOS or PAD. The pinned author script stops at EOS and removes PAD tokens. The existing author-compatible local renderer matched the exact extracted author function on all512 saved word sequences. No generation or training was repeated.','',
        '| Arm | Changed texts out of128 |','|---|---:|',
        *[f'| {a} | {counts[a]} |' for a in study.probe.ARMS],'',
        f"The completed native arm had{state['original_native_passes']}/128 complete-summary passes under the EOS-or-PAD policy. All128 raw first judgments were independently replayed and valid. All{state['unchanged_native_passes']} passes have unchanged text. Even if every changed native text passes, retaining unchanged first judgments limits this correction alone to{state['native_render_only_pass_upper_bound']}/128 ({state['native_render_only_pass_upper_bound']/128:.2%}). This upper bound is not a measured corrected score.",'',
        'At observation, the other arms were still being graded. Their incomplete outcomes were not used here. The old experiment and its first judgments remain intact. Corrected texts and exact packets are saved for a separately governed follow-up that must inherit the completed cache.','',
        'The128 sources are the reused public XSum TRAIN probe, excluded from adapter fitting but possibly seen by the released teacher. This is not application final accuracy. No final320 source was opened; no speed was measured.','',
        f'Raw evidence: `{ROOT}/result.json`.']
    REPORT.write_text('\n'.join(lines)+'\n')
    value = {**state,'hypothesis_id':HYPOTHESIS,'classification':'author_render_mismatch_verified',
        'report':str(REPORT),'artifacts':{str(p):data.sha(p) for p in
            (ROOT/'result.json',ROOT/'actual_tool_receipt.json',ROOT/'comparison.json',Path(__file__),REPORT)}}
    data.write(campaign,value)
    print(json.dumps({'status':'render_mismatch_reported','campaign_sha256':data.sha(campaign),'report':str(REPORT)}))


def register():
    campaign = ROOT/'campaign_results.json';receipt = data.read(ROOT/'report_actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(campaign)) != data.sha(campaign):
        raise ValueError('actual report receipt required')
    data.verify(receipt['artifacts']);v = data.read(campaign);data.verify(v['artifacts'])
    record = {'name':'Exact author rendering of frozen XSum probe tokens','classification':v['classification'],
        'public_probe_cases':128,'decoded_arm_cases':512,'native_changed_texts':v['changed_text_counts']['native'],
        'native_render_only_pass_upper_bound':v['native_render_only_pass_upper_bound'],
        'corrected_semantic_accuracy_measured':False,'application_final_accuracy_measured':False,
        'description':'All512 frozen raw word sequences matched the pinned author EOS-only decoder. The previous EOS-or-PAD renderer altered16 native,18 initial,22 MSE and17 distilled texts. Native original first judgments replayed as3/128 passes; unchanged firsts limit a rendering-only correction to at most19/128. This is an upper bound, not measured corrected semantic accuracy, application final accuracy or speed evidence.'}
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(campaign,REPORT,ROOT,{HYPOTHESIS:record})
    print(json.dumps({'status':'registered_and_read_back','hypothesis_id':HYPOTHESIS}))


def readback():
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    record = data.read(ROOT/(HYPOTHESIS+'_registry_readback.json'))['record']
    collection = chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact = collection.get(ids=[HYPOTHESIS],include=['metadatas'])
    semantic = collection.query(query_texts=[record['description']],n_results=10)
    if (exact['ids'] != [HYPOTHESIS] or any(exact['metadatas'][0].get(k) != v for k,v in record.items())
            or HYPOTHESIS not in semantic['ids'][0]
            or record['campaign_sha256'] != data.sha(ROOT/'campaign_results.json')
            or record['report_sha256'] != data.sha(REPORT)
            or data.read(Path('data/hypothesis_registry')/(HYPOTHESIS+'.json')) != data.read(ROOT/(HYPOTHESIS+'_registry_readback.json'))):
        raise ValueError('independent registry readback differs')
    value = {'status':'independent_registry_readback_passed','hypothesis_id':HYPOTHESIS,
        'campaign_sha256':data.sha(ROOT/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(ROOT/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('report','register','readback'))
    globals()[parser.parse_args().stage]()
