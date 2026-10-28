"""Recover pointer-bound labels omitted by the original filename/schema index.

Works in a new database. Raw inputs must come from an explicit source record;
matching a summary-only text is never accepted as source evidence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from experiments.structure_net import ncm_available_pair_census as census
from experiments.structure_net import ncm_available_pair_extra_labels as extra
from experiments.structure_net import ncm_available_pair_receipts as receipts

archive=census.archive
BASE=receipts.ROOT
ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_supplement_v1')
PACKAGE=Path('data/experiments/ncm_paper_regime/20261027_available_pair_source_package_v1')


def resolve_pointer(pointer, lookup):
    whole=lookup(pointer['turn_text_sha256'])
    if whole is not None:
        start,end=pointer['start_char'],pointer['end_char']
        if census.digest(whole)!=pointer['turn_text_sha256'] or not 0<=start<end<=len(whole):
            raise ValueError('whole source hash or bounds mismatch')
        fragment=whole[start:end]
        if census.digest(fragment)!=pointer['span_text_sha256']:
            raise ValueError('source span hash mismatch')
        return fragment,'whole_turn_and_span_verified'
    fragment=lookup(pointer['span_text_sha256'])
    if fragment is not None:
        if census.digest(fragment)!=pointer['span_text_sha256'] or len(fragment)!=pointer['end_char']-pointer['start_char']:
            raise ValueError('standalone source span hash or length mismatch')
        return fragment,'span_content_verified_without_whole_turn'
    raise ValueError('source content unavailable')


def add_source(db,value,origin):
    if isinstance(value,str) and value.strip():
        db.execute('INSERT OR IGNORE INTO source_texts VALUES(?,?,?)',(census.digest(value),value,origin))


def add_label(db,row,origin,record,kind='raw'):
    found=extra.label(row)
    if found:
        key,_,target,pointers=found
        if row.get('child_section_ids'):kind='raw_structural_conversion'
        db.execute('INSERT OR IGNORE INTO labels VALUES(?,?,?,?,?)',
                   (key,kind,target,json.dumps(pointers,ensure_ascii=False),json.dumps(row,ensure_ascii=False)))
        db.execute('INSERT OR IGNORE INTO label_locations VALUES(?,?,?)',(origin,str(record),key))
        return True
    return False


def ingest_body(db,value,origin,expected=None):
    if not isinstance(value,dict) or set(value)!={'turns'} or not isinstance(value['turns'],list):return False
    if not value['turns'] or any(not isinstance(t,dict) or set(t)!={'role','text'}
        or t['role'] not in ('user','assistant','system') or not isinstance(t['text'],str) or not t['text']
        for t in value['turns']):return False
    if expected is not None and extra.identity(value)!=expected:raise ValueError('body identity mismatch')
    for turn in value['turns']:add_source(db,turn['text'],origin)
    return True


def collect(db,base,counts):
    package=census.readonly(PACKAGE/'sources.sqlite')
    for sha,value in package.execute('SELECT sha,text FROM source_texts'):
        if census.digest(value)!=sha:raise ValueError('source package content mismatch')
        add_source(db,value,'original_data_package')
    package.close();db.commit()
    idx=census.readonly(archive.STORAGE/'index.sqlite')
    # Body databases were explicitly marked unsupported by the original pass.
    seen=set()
    for origin,result in base.execute("SELECT origin,result FROM processed WHERE result LIKE '%summary_schema_error%'"):
        bundle,path=origin.removeprefix('archive:').split(':',1)
        sha=idx.execute('SELECT sha256 FROM artifacts WHERE bundle=? AND path=?',(bundle,path)).fetchone()[0]
        if sha in seen:continue
        seen.add(sha);source=census.readonly(archive.object_path(sha))
        names={r[1] for r in source.execute('PRAGMA table_info(bodies)')}
        if {'body_sha256','body_json'}<=names:
            for body_hash,raw in source.execute('SELECT body_sha256,body_json FROM bodies'):
                if not ingest_body(db,json.loads(raw),'body_database:'+sha,body_hash):
                    raise ValueError('unsupported source body')
                counts['source_body_records']+=1
        elif {'body_sha256','summaries_json','summary_sha256'}<=names:
            for body_hash,raw,summary_hash in source.execute('SELECT body_sha256,summaries_json,summary_sha256 FROM bodies'):
                value=json.loads(raw)
                if extra.identity(value)!=summary_hash:raise ValueError('summary body identity mismatch')
                for number,row in enumerate(value):
                    if not add_label(db,row,'summary_body_database:'+sha,body_hash+':'+str(number)):
                        counts['unhandled_summary_body_row']+=1
                    else:counts['summary_body_label_occurrences']+=1
        else:counts['remaining_unsupported_database_schema']+=1
        source.close();db.commit();print(json.dumps({'body_database':sha,'counts':counts}),flush=True)
    # Archived atomic rows that lacked their local source can join globally.
    for (origin,) in base.execute("SELECT DISTINCT origin FROM issues WHERE kind='atomic' AND reason='missing source turn'"):
        bundle,path=origin.removeprefix('archive:').split(':',1)
        sha=idx.execute('SELECT sha256 FROM artifacts WHERE bundle=? AND path=?',(bundle,path)).fetchone()[0]
        missing={r[0] for r in base.execute("SELECT record FROM issues WHERE origin=? AND kind='atomic'",(origin,))}
        for kind,row in census.sections(archive.object_path(sha)):
            if kind=='atomic' and row.get('section_id') in missing:
                counts['missing_local_source_label_occurrences']+=int(add_label(db,row,origin,row['section_id']))
        db.commit()
    idx.close()
    labels=census.readonly(extra.ROOT/'labels.sqlite')
    for file_sha,record,key,metadata in labels.execute('SELECT * FROM locations'):
        counts['extra_label_occurrences']+=int(add_label(db,json.loads(metadata),'extra:'+file_sha,record))
    for file_sha,record,kind,messages,fragments,keys in labels.execute('SELECT * FROM requests'):
        request_id=file_sha+':'+record
        db.execute('INSERT INTO requests VALUES(?,?,?,?)',(request_id,kind,messages,fragments))
        db.executemany('INSERT OR IGNORE INTO request_keys VALUES(?,?)',[(k,request_id) for k in json.loads(keys)])
        if kind=='raw':
            for f in json.loads(fragments):add_source(db,f['fragment'],'extra_request:'+request_id)
    labels.close();db.commit()
    candidates=census.readonly(BASE/'extra_payloads.sqlite')
    for sha,raw in candidates.execute('SELECT sha,raw FROM candidates'):
        try:values=[json.loads(raw)]
        except (ValueError,UnicodeError):
            try:values=[json.loads(line) for line in raw.splitlines() if line.strip()]
            except (ValueError,UnicodeError):continue
        for value in values:
            if ingest_body(db,value,'extra_body:'+sha):counts['extra_source_bodies']+=1
    candidates.close();db.commit()


def join(db,base,counts):
    def lookup(sha):
        row=db.execute('SELECT text FROM source_texts WHERE sha=?',(sha,)).fetchone()
        if row is None:row=base.execute('SELECT t.text FROM texts t JOIN verified_inputs v ON v.sha=t.sha WHERE t.sha=?',(sha,)).fetchone()
        return row[0] if row else None
    for i,(key,kind,target,encoded,metadata) in enumerate(db.execute('SELECT * FROM labels'),1):
        try:
            fragments=[];proofs=[]
            for p in json.loads(encoded):
                fragment,proof=resolve_pointer(p,lookup);fragments.append(fragment);proofs.append(proof)
            source='\n'.join(fragments);row=json.loads(metadata)
            support=row.get('support',[])
            if not isinstance(support,list) or any(not isinstance(q,str) or q not in source for q in support):
                raise ValueError('support quote outside source')
            pid=census.pair(db,'pointer_label',key,'pointer_joined',None,kind,source,target)
            db.execute('INSERT INTO label_results VALUES(?,?,?)',(key,pid,json.dumps(proofs)))
            counts['pointer_labels_joined']+=1
        except (KeyError,ValueError,TypeError) as error:
            reason=str(error) if isinstance(error,ValueError) else type(error).__name__
            db.execute('INSERT INTO label_results VALUES(?,?,?)',(key,None,reason));counts['pointer_unresolved_'+reason]+=1
        if i%10000==0:db.commit();print(json.dumps({'pointer_labels':i,'counts':counts}),flush=True)
    db.commit()
    payloads=census.readonly(BASE/'payloads.sqlite')
    for file_sha,category,source_sha,target_sha,request_sha in base.execute('SELECT file_sha,category,source_sha,target_sha,request_sha FROM cached_labels WHERE pair_id IS NULL AND target_sha IS NOT NULL'):
        value=json.loads(payloads.execute('SELECT raw FROM payloads WHERE sha=?',(file_sha,)).fetchone()[0])
        if category=='atomic_cache':
            source=lookup(source_sha)
            support=value.get('support',[])
            if source is not None and isinstance(support,list) and all(isinstance(q,str) and q in source for q in support):
                census.pair(db,'cache:'+file_sha,file_sha,'additional_atomic_cache_join',None,'raw',source,value['summary'])
                counts['additional_atomic_cache_joins']+=1
        elif category=='merge_cache':
            req=receipts.unique_request(db,request_sha)
            if req and req[1]=='merge':
                source=json.dumps([f['summary'] for f in json.loads(req[3])],ensure_ascii=False,separators=(',',':'))
                census.pair(db,'cache:'+file_sha,file_sha,'additional_merge_cache_join',None,'children',source,value['summary'])
                counts['additional_merge_cache_joins']+=1
    payloads.close()
    for sha,key,message_key,content,finish in base.execute('SELECT * FROM responses r WHERE NOT EXISTS(SELECT 1 FROM receipt_pairs p WHERE p.response_sha=r.file_sha)'):
        req=receipts.unique_request(db,key,message_key)
        if req is None or finish not in ('stop','eos','eos_token'):continue
        try:
            if req[1]=='raw':
                for label,source,target in receipts.raw_pairs(json.loads(req[3]),content):
                    census.pair(db,'response:'+sha,label,'additional_response_join',None,'raw',source,target)
                    counts['additional_raw_response_pairs']+=1
            else:
                value=json.loads(content);target=value.get('summary')
                if not isinstance(target,str) or not target.strip():raise ValueError('empty merge')
                source=json.dumps([f['summary'] for f in json.loads(req[3])],ensure_ascii=False,separators=(',',':'))
                census.pair(db,'response:'+sha,'merge','additional_response_join',None,'children',source,target)
                counts['additional_merge_response_pairs']+=1
        except (ValueError,KeyError,TypeError):counts['additional_invalid_responses']+=1
    package=census.readonly(PACKAGE/'sources.sqlite')
    for ident,source,target,has_images in package.execute('SELECT id,source,target,has_image_metadata FROM session_pairs'):
        census.pair(db,'LOC0MO',str(ident),'stored_session_summary',None,'locomo_image_context' if has_images else 'raw',source,target)
    package.close();db.commit()


def main():
    if ROOT.exists():raise ValueError('fresh supplemental census required')
    for path in (BASE/'finish_result.json',extra.ROOT/'result.json',PACKAGE/'result.json'):
        if not path.exists():raise ValueError('completed prerequisite required: '+str(path))
    ROOT.mkdir();archive.write(ROOT/'producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'base_finish_sha256':archive.sha(BASE/'finish_result.json'),'extra_labels_sha256':archive.sha(extra.ROOT/'result.json'),
        'source_package_sha256':archive.sha(PACKAGE/'result.json'),'training_admitted':False,'new_model_calls':0})
    census.STORAGE=ROOT;db=census.open_census();base=census.readonly(BASE/'pairs.sqlite')
    db.executescript('''
      CREATE TABLE source_texts(sha TEXT PRIMARY KEY,text TEXT,origin TEXT);
      CREATE TABLE labels(id TEXT PRIMARY KEY,kind TEXT,target TEXT,pointers TEXT,metadata TEXT);
      CREATE TABLE label_locations(origin TEXT,record TEXT,label_id TEXT,PRIMARY KEY(origin,record));
      CREATE TABLE label_results(id TEXT PRIMARY KEY,pair_id TEXT,proof TEXT);
      CREATE TABLE requests(file_sha TEXT PRIMARY KEY,kind TEXT,messages TEXT,fragments TEXT);
      CREATE TABLE request_keys(key TEXT,file_sha TEXT,PRIMARY KEY(key,file_sha));
    ''')
    db.executemany('INSERT INTO requests VALUES(?,?,?,?)',base.execute('SELECT * FROM requests'))
    db.executemany('INSERT INTO request_keys VALUES(?,?)',base.execute('SELECT * FROM request_keys'))
    counts=Counter();collect(db,base,counts);join(db,base,counts)
    value={'status':'supplemental_body_and_pointer_joins_completed','counts':dict(counts),
        'distinct_pointer_targets':db.execute('SELECT COUNT(*) FROM labels').fetchone()[0],
        'unresolved_pointer_targets':db.execute('SELECT COUNT(*) FROM label_results WHERE pair_id IS NULL').fetchone()[0],
        'supplement_pairs_before_cross_corpus_dedup':db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0],
        'training_admitted':False,'semantic_quality_reviewed':False}
    archive.write(ROOT/'result.json',value);db.close();base.close();print(json.dumps(value),flush=True)


if __name__=='__main__':argparse.ArgumentParser(__doc__).parse_args();main()
