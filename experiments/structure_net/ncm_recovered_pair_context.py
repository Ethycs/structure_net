"""Inventory exact original context joins for the guarded recovered pair pool."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3

BASE = Path('data/experiments/ncm_paper_regime')
ROOT = BASE/'20261027_recovered_pair_context_v1'
POOL = BASE/'20261027_recovered_pair_training_pool_v1'
POOL_SHA = '4ea297064370d4b9cfa40c170a7b79db4d65c4a6991fc33f223852e399296003'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-pair-context.md')
DATABASES = [BASE/name/'pairs.sqlite' for name in ('20261027_available_pair_supplement_v1','20261027_available_pair_deep_join_v2')]
FAST = BASE/'20261027_available_pair_census_fast_v1/pairs.sqlite'
OBJECTS = Path('/data2/structure_net/ncm_available_pairs_20261027_v1/objects')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def text_sha(text):return hashlib.sha256(text.encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def connect(path):return sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True)


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('fresh CPU-only inventory required')
    source=POOL/'review_candidates_train.jsonl'
    if sha(source)!=POOL_SHA:raise ValueError('fixed complete guarded pool required')
    receipt=read(POOL/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code']!=0:raise ValueError('completed guarded-pool worker required')
    for p,value in receipt['artifacts'].items():
        if sha(p)!=value:raise ValueError('pool receipt artifact changed')
    paths=[source,POOL/'producer.json',POOL/'result.json',POOL/'actual_tool_receipt.json',*DATABASES,FAST,Path('pixi.lock'),Path('pyproject.toml')]
    proof={'schema':'nal.ncm-recovered-pair-context.v1','source_hashes':{str(p):sha(p) for p in (Path(__file__),PLAN)},
        'input_hashes':{str(p):sha(p) for p in paths},'cases':104503,'training_admitted':False,'new_requests':0,'full_original_prompts_recovered':False}
    ROOT.mkdir();write(ROOT/'producer.json',proof)
    pool={}
    db=sqlite3.connect(ROOT/'contexts.sqlite')
    db.executescript('CREATE TABLE pairs(id TEXT PRIMARY KEY,source_sha TEXT,target_sha TEXT,component TEXT,substring INTEGER);'
        'CREATE TABLE contexts(pair_id TEXT,role TEXT,created_at TEXT,identity TEXT,request_sha TEXT,origin TEXT,record TEXT,provenance TEXT,PRIMARY KEY(pair_id,origin,record));')
    with source.open() as f:
        for line in f:
            row=json.loads(line)
            if row['training_admitted'] or row['split']!='TRAIN_review_candidate' or row['id'] in pool:raise ValueError('unique unadmitted candidates required')
            entry=(row['source_sha256'],row['target_sha256'],row['source_component'],int(row['target_text'] in row['source']))
            pool[row['id']]=entry;db.execute('INSERT INTO pairs VALUES(?,?,?,?,?)',(row['id'],*entry))
    if len(pool)!=104503:raise ValueError('complete candidate population required')
    counts=Counter();objects={}
    def add(pair,role,date,identity,request,origin,record,provenance):
        if any(v is not None and not isinstance(v,str) for v in (role,date,identity,request)):raise ValueError('literal string metadata required')
        db.execute('INSERT OR IGNORE INTO contexts VALUES(?,?,?,?,?,?,?,?)',
            (pair,role,date,identity,request,origin,record,json.dumps(provenance,sort_keys=True)))
    for path in DATABASES:
        source_db=connect(path)
        query='SELECT r.pair_id,l.id,l.kind,l.target,l.pointers,l.metadata FROM label_results r JOIN labels l ON l.id=r.id WHERE r.pair_id IS NOT NULL'
        for pair,identity,kind,target,raw_pointers,raw_meta in source_db.execute(query):
            if pair not in pool:continue
            counts['matching_label_records']+=1
            if text_sha(target)!=pool[pair][1]:raise ValueError('label target differs for same pair')
            pointers=json.loads(raw_pointers);meta=json.loads(raw_meta)
            if not isinstance(pointers,list) or not isinstance(meta,dict):raise ValueError('original label schema differs')
            if kind!='raw' or len(pointers)!=1 or not isinstance(pointers[0],dict) or pointers[0].get('span_text_sha256')!=pool[pair][0]:
                counts['label_records_without_direct_single_span_match']+=1;continue
            pointer=pointers[0]
            add(pair,pointer.get('role'),pointer.get('created_at'),meta.get('summarizer_identity'),None,str(path),identity,
                {'label_id':identity,'source_span':pointer,'format':meta.get('format'),'metadata_summary_matches_target':meta.get('summary')==target})
            counts['direct_single_span_label_records']+=1
        source_db.close();db.commit()
        print(json.dumps({'database':str(path),'direct_records':counts['direct_single_span_label_records']}),flush=True)
    source_db=connect(FAST)
    for pair,file_sha,source_sha,target_sha in source_db.execute('SELECT pair_id,file_sha,source_sha,target_sha FROM cached_labels WHERE pair_id IS NOT NULL'):
        if pair not in pool:continue
        counts['matching_cache_records']+=1
        if (source_sha,target_sha)!=pool[pair][:2]:raise ValueError('cache source/target binding differs')
        path=OBJECTS/file_sha[:2]/file_sha
        if sha(path)!=file_sha:raise ValueError('original cache object differs')
        obj=read(path)
        if (not isinstance(obj,dict) or obj.get('text_sha256')!=source_sha or not isinstance(obj.get('summary'),str)
                or text_sha(obj['summary'])!=target_sha):
            counts['cache_records_without_exact_body_summary_schema']+=1;continue
        objects[str(path)]=file_sha
        add(pair,obj.get('role'),obj.get('created_at'),obj.get('summarizer_identity'),obj.get('request_sha256'),str(FAST),file_sha,
            {'object':str(path),'sha256':file_sha,'schema_keys':sorted(obj)})
        counts['direct_cache_records']+=1
    source_db.close();db.commit()
    db.execute('CREATE INDEX context_pair ON contexts(pair_id)');db.commit()
    totals={};exported=0
    with (ROOT/'context_records.jsonl').open('x') as output:
        for pair,source_sha,target_sha,component,substring in db.execute('SELECT * FROM pairs ORDER BY id'):
            contexts=[{'role':r[0],'created_at':r[1],'summarizer_identity':r[2],'request_sha256':r[3],'origin':r[4],'record':r[5],'provenance':json.loads(r[6])}
                for r in db.execute('SELECT role,created_at,identity,request_sha,origin,record,provenance FROM contexts WHERE pair_id=? ORDER BY origin,record',(pair,))]
            group='substring' if substring else 'non_substring';counter=totals.setdefault(group,Counter());counter['pairs']+=1
            roles={r['role'] for r in contexts if r['role'] is not None};dates={r['created_at'] for r in contexts if r['created_at'] is not None}
            identities={r['summarizer_identity'] for r in contexts if r['summarizer_identity'] is not None}
            requests={r['request_sha256'] for r in contexts if r['request_sha256'] is not None}
            for key,flag in [('any_context',bool(contexts)),('role_available',bool(roles)),('one_observed_role',len(roles)==1),
                    ('conflicting_roles',len(roles)>1),('date_available',bool(dates)),('conflicting_dates',len(dates)>1),
                    ('identity_fingerprint_available',bool(identities)),('request_hash_available',bool(requests))]:counter[key]+=int(flag)
            if contexts:
                record={'id':pair,'source_sha256':source_sha,'target_sha256':target_sha,'source_component':component,
                    'exact_substring_target':bool(substring),'contexts':contexts,'training_admitted':False,
                    'full_original_prompt_recovered':False,'semantic_label_quality_checked':False}
                output.write(json.dumps(record,allow_nan=False)+'\n');exported+=1
    db.close();write(ROOT/'object_bindings.json',objects)
    for collection in (proof['source_hashes'],proof['input_hashes']):
        for path,expected in collection.items():
            if sha(path)!=expected:raise ValueError('inventory input changed')
    result={'schema':proof['schema'],'status':'exact_pair_context_inventory_completed','pairs':len(pool),'pairs_with_context':exported,
        'counts':dict(counts),'strata':{k:dict(v) for k,v in totals.items()},'original_cache_objects':len(objects),
        'training_admitted':False,'full_original_prompts_recovered':False,'semantic_label_quality_checked':False,
        'new_requests':0,'task_accuracy':None,'final_test_opened':False,
        'artifacts':{str(p):sha(p) for p in ROOT.iterdir() if p.is_file()}}
    write(ROOT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(__doc__).parse_args();existed=ROOT.exists()
    try:run()
    except Exception as error:
        if ROOT.exists() and not existed:write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
