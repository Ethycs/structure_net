"""Recover unchanged stored labels for already cached TRAIN source features."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('data/experiments/ncm_paper_regime')
DEST=ROOT/'20261025_cached_stored_summary_join_v1'
INVENTORY=ROOT/'20261025_shared_summary_native128_inventory_v1'
SOURCES=ROOT/'20261024_additional10000_training_sources_v1/cases.json'
ORIGINAL=Path('data/experiments/ncm_memory/20261022_data_v1/train.jsonl')
FEATURES=ROOT/'20261024_additional10000_source_features_v1'
INDEX=ROOT/'20261024_additional10000_source_metadata_v1/index.json'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def bind(source, original, metadata):
    if (source['source']!=original['source'] or source['source_sha256']!=original['source_sha256']
            or source['source_sha256']!=metadata['source_sha256']
            or hashlib.sha256(source['source'].encode()).hexdigest()!=source['source_sha256']
            or original['id']!=metadata['id'] or original['split']!='train'
            or original['target_sha256']!=metadata['target_sha256']
            or hashlib.sha256(original['target'].encode()).hexdigest()!=original['target_sha256']):
        raise ValueError('exact source bytes and unchanged stored target identity required')
    return {**source,'target_text':original['target'],'target_sha256':original['target_sha256'],
        'stored_label_id':original['id'],'original_line_number':metadata['original_line_number'],
        'target_origin':'unchanged_stored_summary','semantic_qualified':False,
        'stored_tokens':metadata['target_tokens'],'native_lexical_tokens':metadata['native_target_tokens'],
        'original_source_id':original['source_id'],'original_group_id':original['group_id']}


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('fresh CPU-only data inventory required')
    read=lambda p:json.loads(p.read_text())
    state=read(INVENTORY/'result.json')
    eligible=INVENTORY/'eligible_train_metadata.jsonl'
    if (file_hash(eligible)!=state['eligible_metadata_sha256'] or file_hash(ORIGINAL)!=state['original_train_sha256']
            or read(INVENTORY/'actual_tool_receipt.json')['actual_tool_exit_code']!=0):
        raise ValueError('unchanged completed metadata exclusion inventory required')
    sources=read(SOURCES)
    if len(sources)!=10000 or len({r['source_component'] for r in sources})!=10000:
        raise ValueError('all10000 reserved distinct TRAIN components required')
    first={}
    with eligible.open() as f:
        for line in f:
            row=json.loads(line);first.setdefault(row['source_sha256'],row)
    selected={r['id']:first[r['source_sha256']] for r in sources if r['source_sha256'] in first}
    line_numbers={r['original_line_number'] for r in selected.values()}
    originals={}
    with ORIGINAL.open() as f:
        for number,line in enumerate(f,1):
            if number in line_numbers:originals[number]=json.loads(line)
    if set(originals)!=line_numbers:raise ValueError('all referenced original lines required')
    cases=[bind(r,originals[selected[r['id']]['original_line_number']],selected[r['id']])
           for r in sources if r['id'] in selected]
    manifest=read(FEATURES/'manifest.json');index=read(INDEX)
    feature_state=read(FEATURES/'result.json')
    if len(manifest)!=10000 or file_hash(FEATURES/'manifest.json')!=feature_state['manifest_sha256']:
        raise ValueError('unchanged full10000 feature manifest required')
    by_id={r['id']:r for r in index}
    for row in cases:
        if row['id'] not in manifest or by_id[row['id']]['source_sha256']!=row['source_sha256']:
            raise ValueError('cached source feature identity differs')
    inputs=[Path(__file__),SOURCES,ORIGINAL,eligible,INVENTORY/'result.json',INDEX,
        FEATURES/'manifest.json',FEATURES/'result.json']
    DEST.mkdir()
    write=lambda name,value:(DEST/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('cases.json',cases)
    write('feature_manifest.json',{r['id']:manifest[r['id']] for r in cases})
    write('source_index.json',[by_id[r['id']] for r in cases])
    result={'status':'unchanged_stored_summaries_joined_to_cached_TRAIN_sources',
        'candidate_pairs':len(cases),'source_components':len({r['source_component'] for r in cases}),
        'source_length_bins':dict(Counter(str(r['bin']) for r in cases)),
        'target_selection':'first eligible physical original TRAIN row for each exact source hash',
        'source_bytes_equal':True,'stored_targets_unchanged':True,
        'inherited_metadata_exclusions':state['protected_metadata'],
        'full_text_near_duplicate_audit_completed':False,'feature_payloads_replayed_here':False,
        'training_admitted':False,'new_teacher_requests':0,'new_model_training':False,
        'semantic_accuracy_measured':False,'sealed_final_files_opened':False,
        'input_hashes':{str(p):file_hash(p) for p in inputs},
        'artifact_hashes':{str(DEST/n):file_hash(DEST/n) for n in ('cases.json','feature_manifest.json','source_index.json')}}
    write('result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','artifact_hashes')}))


if __name__=='__main__':main()
