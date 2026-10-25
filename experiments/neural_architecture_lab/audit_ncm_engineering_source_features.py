"""Replay every engineering FP32 payload after actual frozen-body extraction."""
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_engineering_source_features as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = run.ROOT / '20261025_engineering_source_features_audit_v1'


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only feature audit required')
    if run.original.q.read(run.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actually completed whole feature workflow required')
    index = run.inputs()
    for root in (run.PILOT,run.CACHE):run.native.vendor.terminal(root)
    manifest = run.original.q.read(run.CACHE/'manifest.json')
    pilot = run.original.q.read(run.PILOT/'manifest.json')
    if len(manifest)!=888 or set(manifest)!={r['id'] for r in index}:
        raise ValueError('all888 complete source payloads required')
    raw_bytes=stored_bytes=0
    for i,source in enumerate(index,1):
        if file_hash(source['path'])!=source['sha256']:raise ValueError('source geometry changed')
        row=run.original.q.read(source['path'])
        entry=manifest[row['id']]
        features=run.original.load_feature(entry,row)
        if (entry['raw_bytes']!=features.numel()*4
                or entry['stored_bytes']!=Path(entry['header_path']).stat().st_size+Path(entry['payload_path']).stat().st_size
                or run.original.q.read(run.CACHE/'entries'/(row['id']+'.json'))!=entry
                or row['id'] in pilot and entry!=pilot[row['id']]):
            raise ValueError('source/entry/pilot/raw/stored byte bindings differ')
        raw_bytes+=entry['raw_bytes'];stored_bytes+=entry['stored_bytes']
        if i%64==0 or i==888:print(json.dumps({'engineering_features_audited':i,'total':888}),flush=True)
    state=run.original.q.read(run.CACHE/'result.json')
    if (state['raw_bytes']!=raw_bytes or state['stored_bytes']!=stored_bytes
            or state['manifest_sha256']!=file_hash(run.CACHE/'manifest.json')
            or state['original_body_hash_unchanged']!=run.original.base.EXPECTED_BODY
            or not state['all_FP32_byte_roundtrips_exact'] or state['original_MLPs_updated']):
        raise ValueError('complete actual frozen-body feature state differs')
    inputs=[Path(__file__),run.CACHE/'manifest.json',run.CACHE/'result.json',
        run.WORKFLOW/'actual_tool_receipt.json',run.metadata.DEST/'index.json',
        *[root/'producer.json' for root in (run.WORKFLOW,run.PILOT,run.CACHE)]]
    inputs += [Path(r['path']) for r in index]
    inputs += [Path(v[k]) for v in manifest.values() for k in ('header_path','payload_path')]
    DEST.mkdir()
    result={'status':'all888_engineering_feature_payloads_and_source_bindings_replayed',
        'cases':888,'raw_bytes':raw_bytes,'stored_bytes':stored_bytes,
        'original_body_hash_unchanged':state['original_body_hash_unchanged'],
        'FP32_bytes_preserved':True,'original_MLPs_updated':False,'task_accuracy':None,
        'final_test_opened':False,'artifacts':{str(p):file_hash(p) for p in inputs}}
    write_json(DEST/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':main()
