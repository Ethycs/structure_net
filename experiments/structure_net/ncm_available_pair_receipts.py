"""Recover summary inputs from receipts and join content-addressed caches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from experiments.structure_net import ncm_available_pair_census as census

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_census_fast_v1')
archive=census.archive


def request_shape(value):
    if not isinstance(value,dict):return None
    nested=value.get('request')
    messages=value.get('messages') or (nested.get('messages') if isinstance(nested,dict) else None)
    if not isinstance(messages,list):return None
    users=[m.get('content') for m in messages if isinstance(m,dict) and m.get('role')=='user']
    if len(users)!=1 or not isinstance(users[0],str):return None
    try:body=json.loads(users[0])
    except ValueError:return None
    if not isinstance(body,dict) or not set(body)<={'fragments','kind','user_spine','max_output_tokens'}:
        return None
    fragments=body.get('fragments')
    if not isinstance(fragments,list) or not fragments or not all(isinstance(f,dict) for f in fragments):return None
    if all(isinstance(f.get('fragment'),str) and isinstance(f.get('label'),str) for f in fragments):
        if len({f['label'] for f in fragments})!=len(fragments):return None
        kind='raw'
    elif all(isinstance(f.get('summary'),str) for f in fragments):kind='merge'
    else:return None
    return kind,messages,fragments


def raw_pairs(fragments,content):
    value=json.loads(content)
    if not isinstance(value,dict) or not isinstance(value.get('atoms'),list):
        raise ValueError('response is not a complete atom list')
    atoms=value['atoms']
    if len(atoms)!=len(fragments) or any(not isinstance(a,dict) for a in atoms):
        raise ValueError('atom count differs from request')
    out=[]
    for f,a in zip(fragments,atoms,strict=True):
        if (not f['fragment'].strip() or a.get('label')!=f['label']
                or not isinstance(a.get('summary'),str) or not a['summary'].strip()):
            raise ValueError('atom label/order or summary invalid')
        quotes=a.get('support')
        if quotes is not None and (not isinstance(quotes,list) or any(not isinstance(q,str) or q not in f['fragment'] for q in quotes)):
            raise ValueError('support quote outside its own fragment')
        out.append((f['label'],f['fragment'],a['summary']))
    return out


def connect():
    census.ROOT=ROOT;census.STORAGE=ROOT
    db=census.open_census()
    db.executescript('''
      CREATE TABLE IF NOT EXISTS verified_inputs(sha TEXT PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS requests(file_sha TEXT PRIMARY KEY, kind TEXT, messages TEXT, fragments TEXT);
      CREATE TABLE IF NOT EXISTS request_keys(key TEXT,file_sha TEXT,PRIMARY KEY(key,file_sha));
      CREATE TABLE IF NOT EXISTS responses(file_sha TEXT PRIMARY KEY,request_key TEXT,messages_key TEXT,content TEXT,finish TEXT);
      CREATE TABLE IF NOT EXISTS receipt_pairs(response_sha TEXT,request_sha TEXT,kind TEXT,exact_prompt_sha TEXT,output_sha TEXT,PRIMARY KEY(response_sha,request_sha));
    ''')
    db.execute("INSERT OR IGNORE INTO verified_inputs SELECT DISTINCT input_sha FROM pairs WHERE input_kind='raw'")
    db.commit();return db


def sources():
    # Tag only actual source text/chunks. Never treat a target-only text match
    # as proof that a cached source has been recovered.
    db=connect();db.close();original=census.text
    def source_text(connection,value):
        h,n=original(connection,value)
        connection.execute('INSERT OR IGNORE INTO verified_inputs VALUES(?)',(h,))
        return h,n
    census.text=source_text
    try:census.sources()
    finally:census.text=original


def index():
    if archive.read(ROOT/'payload_pack_result.json')['status']!='all_indexed_JSON_payloads_packed':
        raise ValueError('all published JSON payloads required')
    archive.write(ROOT/'receipt_index_producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'census_sha256':archive.sha(Path(census.__file__)),'exact_fragment_labels_required':True,
        'non_summary_questions_and_judgments_not_admitted':True})
    db=connect();payloads=census.readonly(ROOT/'payloads.sqlite');counts=Counter();schemas=Counter()
    for i,(h,raw) in enumerate(payloads.execute('SELECT sha,raw FROM payloads'),1):
        try:value=json.loads(raw)
        except (ValueError,UnicodeError):counts['not_valid_JSON']+=1;continue
        if not isinstance(value,dict):counts['not_object']+=1;continue
        schemas[','.join(sorted(value))]+=1
        shape=request_shape(value)
        if shape:
            kind,messages,fragments=shape
            encoded=json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':'))
            db.execute('INSERT OR IGNORE INTO requests VALUES(?,?,?,?)',(h,kind,encoded,json.dumps(fragments,ensure_ascii=False)))
            keys={h,census.digest(encoded),census.digest(encoded+'\n')}
            for key in ('request_sha256','messages_sha256'):
                if isinstance(value.get(key),str) and len(value[key])==64:keys.add(value[key])
            db.executemany('INSERT OR IGNORE INTO request_keys VALUES(?,?)',[(k,h) for k in keys])
            if kind=='raw':
                for f in fragments:
                    sh,_=census.text(db,f['fragment']);db.execute('INSERT OR IGNORE INTO verified_inputs VALUES(?)',(sh,))
            counts['summary_requests_'+kind]+=1
        content=value.get('content',value.get('completion'))
        if isinstance(content,str) and any(k in value for k in ('request_sha256','messages_sha256','request_journal_sha256')):
            db.execute('INSERT OR IGNORE INTO responses VALUES(?,?,?,?,?)',(h,
                value.get('request_sha256',value.get('request_journal_sha256')),value.get('messages_sha256'),
                content,value.get('finish_reason')))
            counts['response_records']+=1
        if i%10000==0:db.commit();print(json.dumps({'JSON_payloads_scanned':i,'counts':counts}),flush=True)
    db.commit();archive.write(ROOT/'receipt_index_result.json',{'status':'all_packed_JSON_receipts_classified','counts':dict(counts)})
    archive.write(ROOT/'receipt_schema_counts.json',dict(schemas));db.close();payloads.close()


def unique_request(db,*keys):
    found={}
    for key in keys:
        if not isinstance(key,str):continue
        for row in db.execute('SELECT r.file_sha,r.kind,r.messages,r.fragments FROM requests r JOIN request_keys k ON k.file_sha=r.file_sha WHERE k.key=?',(key,)):
            found[row[0]]=row
    # Several wrappers may contain the exact same model input. Different
    # inputs sharing a claimed key are unresolved, never arbitrarily selected.
    distinct={(r[1],r[2],r[3]) for r in found.values()}
    if len(distinct)!=1:return None
    return found[sorted(found)[0]]


def join():
    if not (ROOT/'receipt_index_result.json').exists() or not (ROOT/'sources_result.json').exists():
        raise ValueError('complete source and receipt indexes required')
    db=connect();counts=Counter()
    for h,key,message_key,content,finish in db.execute('SELECT * FROM responses'):
        req=unique_request(db,key,message_key)
        if req is None:counts['response_without_unique_summary_input']+=1;continue
        if finish not in ('stop','eos','eos_token'):
            counts['noncomplete_finish']+=1;continue
        rh,kind,messages,encoded=req;fragments=json.loads(encoded)
        try:
            if kind=='raw':
                candidates=raw_pairs(fragments,content)
                for label,source,target in candidates:
                    census.pair(db,'response:'+h,label,'raw_response_joined',None,'raw',source,target)
                    counts['raw_response_fragment_pairs']+=1
            else:
                value=json.loads(content)
                if not isinstance(value,dict):raise ValueError('merge response is not an object')
                target=value.get('summary')
                if not isinstance(target,str) or not target.strip():raise ValueError('missing merge summary')
                source=json.dumps([f['summary'] for f in fragments],ensure_ascii=False,separators=(',',':'))
                census.pair(db,'response:'+h,'merge','merge_response_joined',None,'children',source,target)
                counts['merge_response_pairs']+=1
            db.execute('INSERT INTO receipt_pairs VALUES(?,?,?,?,?)',(h,rh,kind,census.digest(messages),census.digest(content)))
        except (ValueError,KeyError,TypeError):counts['summary_response_format_or_support_invalid']+=1
    db.commit()
    idx=census.readonly(archive.STORAGE/'index.sqlite');payloads=census.readonly(ROOT/'payloads.sqlite')
    rows=idx.execute("SELECT category,sha256 FROM artifacts WHERE category IN ('atomic_cache','merge_cache') GROUP BY category,sha256")
    for i,(category,h) in enumerate(rows,1):
        raw=payloads.execute('SELECT raw FROM payloads WHERE sha=?',(h,)).fetchone()
        if raw is None:raise ValueError('complete cache payload required')
        value=json.loads(raw[0])
        if not isinstance(value,dict):
            counts[category+'_not_object']+=1
            db.execute('INSERT INTO cached_labels VALUES(?,?,?,?,?,?,?)',(h,category,None,None,None,None,type(value).__name__))
            continue
        target=value.get('summary');sh=value.get('text_sha256');key=value.get('request_sha256')
        pid=None;th=None
        if isinstance(target,str) and target.strip():
            th,_=census.text(db,target)
            if category=='atomic_cache':
                found=db.execute('SELECT t.text FROM texts t JOIN verified_inputs v ON v.sha=t.sha WHERE t.sha=?',(sh,)).fetchone()
                if found:
                    quotes=value.get('support',[])
                    if isinstance(quotes,list) and all(isinstance(q,str) and q in found[0] for q in quotes):
                        pid=census.pair(db,'cache:'+h,h,'atomic_cache_joined',None,'raw',found[0],target)
                        counts['atomic_cache_source_verified']+=1
                    else:counts['atomic_cache_invalid_support']+=1
                else:counts['atomic_cache_missing_source']+=1
            else:
                req=unique_request(db,key)
                if req is not None and req[1]=='merge':
                    source=json.dumps([f['summary'] for f in json.loads(req[3])],ensure_ascii=False,separators=(',',':'))
                    pid=census.pair(db,'cache:'+h,h,'merge_cache_joined',None,'children',source,target)
                    counts['merge_cache_input_recovered']+=1
                else:counts['merge_cache_missing_input']+=1
        else:counts[category+'_missing_summary']+=1
        db.execute('INSERT INTO cached_labels VALUES(?,?,?,?,?,?,?)',(h,category,sh,th,key,pid,','.join(sorted(value))))
        if i%10000==0:db.commit();print(json.dumps({'cache_contents_joined':i,'counts':counts}),flush=True)
    db.commit();archive.write(ROOT/'receipt_join_result.json',{'status':'all_indexed_cache_and_receipt_joins_attempted',
        'counts':dict(counts),'training_admitted':False,'semantic_quality_reviewed':False})
    db.close();idx.close();payloads.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('sources','index','join'))
    globals()[parser.parse_args().stage]()
