"""Count recoverable stored pairs across all core and archived summary stores.

Counts are exact-text candidates, not semantic labels or training admission.
Only aggregate counts, hashes and provenance leave this disk-backed inventory.
"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import tiktoken

from experiments.structure_net import ncm_available_pair_archive as archive

ROOT, STORAGE = archive.ROOT, archive.STORAGE
CORE = Path('data/experiments/ncm_memory/20261022_shared_v1')
ENCODING = tiktoken.get_encoding('cl100k_base')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)


def open_census():
    db = sqlite3.connect(STORAGE/'pairs.sqlite',timeout=60)
    db.executescript('''
      PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS texts(sha TEXT PRIMARY KEY, text TEXT, tokens INTEGER);
      CREATE TABLE IF NOT EXISTS pairs(id TEXT PRIMARY KEY, input_kind TEXT,
        input_sha TEXT, target_sha TEXT, input_tokens INTEGER, target_tokens INTEGER,
        exact_copy INTEGER);
      CREATE INDEX IF NOT EXISTS pair_input ON pairs(input_kind,input_sha);
      CREATE TABLE IF NOT EXISTS occurrences(origin TEXT, record TEXT, kind TEXT,
        pair_id TEXT, source_id TEXT, PRIMARY KEY(origin,record,kind));
      CREATE INDEX IF NOT EXISTS occurrence_pair ON occurrences(pair_id);
      CREATE TABLE IF NOT EXISTS processed(origin TEXT PRIMARY KEY, result TEXT);
      CREATE TABLE IF NOT EXISTS issues(origin TEXT, record TEXT, kind TEXT, reason TEXT,
        PRIMARY KEY(origin,record,kind));
      CREATE TABLE IF NOT EXISTS cached_labels(file_sha TEXT PRIMARY KEY, category TEXT,
        source_sha TEXT, target_sha TEXT, request_sha TEXT, pair_id TEXT, schema_keys TEXT);
    ''')
    return db


def text(db, value):
    h = digest(value)
    row = db.execute('SELECT tokens FROM texts WHERE sha=?',(h,)).fetchone()
    if row is None:
        n = len(ENCODING.encode(value,disallowed_special=()))
        db.execute('INSERT INTO texts VALUES(?,?,?)',(h,value,n))
    else:
        n = row[0]
    return h,n


def pair(db, origin, record, kind, source_id, input_kind, source, target):
    if not source.strip() or not target.strip():
        raise ValueError('empty source or target')
    sh,st = text(db,source); th,tt = text(db,target)
    key = digest(json.dumps([input_kind,sh,th],separators=(',',':')))
    db.execute('INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?,?)',
               (key,input_kind,sh,th,st,tt,int(source==target)))
    db.execute('INSERT INTO occurrences VALUES(?,?,?,?,?)',
               (origin,record,kind,key,source_id))
    return key


def sections(path):
    with closing(readonly(path)) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'sections' in tables:
            for kind,sid,payload in db.execute('SELECT kind,sid,payload FROM sections ORDER BY kind,sid'):
                row=json.loads(payload)
                if sid != row['section_id']:
                    raise ValueError('section identity mismatch')
                yield kind,row
        elif 'snapshot' in tables:
            found = db.execute('SELECT payload FROM snapshot WHERE id=1').fetchone()
            if not found:
                return
            payload=json.loads(found[0])
            for key,kind in [('atomic_sections','atomic'),('hierarchy_sections','hierarchy'),('sections','parents')]:
                for row in payload.get(key,[]):
                    yield kind,row
        else:
            raise ValueError('unsupported summary schema: '+','.join(sorted(tables)))


def scan_store(db, origin, summary, source):
    prior=db.execute('SELECT result FROM processed WHERE origin=?',(origin,)).fetchone()
    if prior:
        return json.loads(prior[0])
    counts=Counter();errors=Counter();turns={}
    if source is not None:
        with closing(readonly(source)) as raw:
            names={r[1] for r in raw.execute('PRAGMA table_info(turns)')}
            if {'turn_id','text','role','source_id'} <= names:
                for tid,value,role,sid in raw.execute('SELECT turn_id,text,role,source_id FROM turns'):
                    h,_=text(db,value);turns[tid]=(value,role,sid,h)
                    counts['source_turns']+=1
    try:
        records=list(sections(summary))
    except Exception as error:
        result={'counts':dict(counts),'errors':{'summary_schema_error':1},'error_type':type(error).__name__}
        db.execute('INSERT INTO processed VALUES(?,?)',(origin,json.dumps(result)));db.commit()
        return result
    lookup={r['section_id']:r for _,r in records}
    for kind,row in records:
        counts['sections_'+kind]+=1;sid=row.get('section_id','unknown')
        try:
            target=row['summary']
            if not isinstance(target,str) or not target.strip():
                raise ValueError('missing text summary')
            evidence=[]
            try:
                for span in row.get('spans',[]):
                    if span['turn_id'] not in turns:
                        raise ValueError('missing source turn')
                    value,role,source_id,h=turns[span['turn_id']]
                    start,end=span['start_char'],span['end_char']
                    if not (h==span['turn_text_sha256'] and role==span['role']
                            and source_id==span['source_id'] and 0<=start<end<=len(value)):
                        raise ValueError('source bounds, hash or attribution mismatch')
                    fragment=value[start:end]
                    if digest(fragment)!=span['span_text_sha256']:
                        raise ValueError('source fragment hash mismatch')
                    text(db,fragment);evidence.append(fragment)
            except (ValueError,KeyError,TypeError):
                if kind=='atomic':
                    raise
                # A child-summary input can be recovered without raw source
                # evidence. Keep that provenance limitation explicit.
                counts['merge_raw_evidence_unverified']+=1
            if kind=='atomic':
                if not evidence:
                    raise ValueError('no atomic source spans')
                source_text='\n'.join(evidence)
                classification='verbatim_bypass_candidate' if target==source_text else 'raw_summary_candidate'
                pair(db,origin,sid,classification,row.get('source_id'),'raw',source_text,target)
                counts[classification]+=1
            else:
                children=row.get('child_section_ids',[])
                if not children or any(c not in lookup for c in children):
                    raise ValueError('structural summary without complete children')
                source_text=json.dumps([lookup[c]['summary'] for c in children],ensure_ascii=False,separators=(',',':'))
                pair(db,origin,sid,'hierarchy_merge_candidate',row.get('source_id'),'children',source_text,target)
                counts['hierarchy_merge_candidate']+=1
        except (KeyError,ValueError,TypeError,sqlite3.IntegrityError) as error:
            reason=str(error) if isinstance(error,ValueError) else type(error).__name__
            errors[reason]+=1
            db.execute('INSERT OR IGNORE INTO issues VALUES(?,?,?,?)',(origin,sid,kind,reason))
    result={'counts':dict(counts),'errors':dict(errors),'summary_path':str(summary),'source_path':str(source) if source else None}
    db.execute('INSERT INTO processed VALUES(?,?)',(origin,json.dumps(result)));db.commit()
    return result


def stage_start(stage):
    path=ROOT/(stage+'_producer.json')
    if path.exists():
        proof=archive.read(path)
        if proof['producer_sha256']!=archive.sha(Path(__file__)):
            raise ValueError('census producer changed after stage began')
    else:
        archive.write(path,{'producer':str(Path(__file__)),'producer_sha256':archive.sha(Path(__file__)),
                           'stage':stage,'semantics_reviewed':False,'training_admitted':False,
                           'final_evaluation_files_opened':False,'new_model_calls':0})
    return open_census()


def core():
    db=stage_start('core');catalog=archive.read(CORE/'source/research_data/longmemeval-s-500-stores-20261021/catalog.json')
    total=Counter();errors=Counter();started=time.monotonic()
    for i,item in enumerate(catalog['questions'],1):
        store=CORE/'restored'/item['store']
        r=scan_store(db,'core:'+item['store'],store/'native-spine-live-v1.sqlite',store/'memory.db')
        total.update(r['counts']);errors.update(r['errors'])
        if i%25==0:
            print(json.dumps({'core_stores':i,'of':len(catalog['questions']),'counts':total,'errors':errors,
                              'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    value={'status':'all_core_stores_counted','stores':len(catalog['questions']),
           'counts':dict(total),'errors':dict(errors),'elapsed_seconds':time.monotonic()-started}
    archive.write(ROOT/'core_result.json',value);db.close();print(json.dumps(value),flush=True)


def stores():
    restored=archive.read(ROOT/'restore_result.json')
    if restored['status']!='all_indexed_summary_artifacts_restored':
        raise ValueError('complete archive payload restoration required')
    db=stage_start('stores');idx=readonly(STORAGE/'index.sqlite')
    sources={(b,str(Path(p).parent)):h for b,p,h in idx.execute(
        "SELECT bundle,path,sha256 FROM artifacts WHERE category='source_database'")}
    rows=list(idx.execute("SELECT bundle,path,sha256 FROM artifacts WHERE category='summary_database' ORDER BY bundle,path"))
    total=Counter();errors=Counter()
    for i,(bundle,path,h) in enumerate(rows,1):
        source=sources.get((bundle,str(Path(path).parent)))
        r=scan_store(db,'archive:'+bundle+':'+path,archive.object_path(h),archive.object_path(source) if source else None)
        total.update(r['counts']);errors.update(r['errors'])
        if i%10==0:
            print(json.dumps({'archived_summary_databases':i,'of':len(rows),'counts':total,'errors':errors}),flush=True)
    archive.write(ROOT/'stores_result.json',{'status':'all_archived_summary_databases_counted',
        'databases':len(rows),'counts':dict(total),'errors':dict(errors)})
    db.close();idx.close()


def sources():
    if archive.read(ROOT/'restore_result.json')['status']!='all_indexed_summary_artifacts_restored':
        raise ValueError('complete archive payload restoration required')
    db=stage_start('sources');idx=readonly(STORAGE/'index.sqlite');total=Counter();failures=[]
    rows=list(idx.execute("SELECT DISTINCT sha256 FROM artifacts WHERE category='source_database' ORDER BY sha256"))
    for i,(h,) in enumerate(rows,1):
        origin='source_database:'+h
        prior=db.execute('SELECT result FROM processed WHERE origin=?',(origin,)).fetchone()
        if prior:
            total.update(json.loads(prior[0])['counts']);continue
        counts=Counter()
        try:
            with closing(readonly(archive.object_path(h))) as raw:
                tables={r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table in ('turns','chunks'):
                    if table not in tables or 'text' not in {r[1] for r in raw.execute('PRAGMA table_info('+table+')')}:
                        continue
                    for (value,) in raw.execute('SELECT text FROM '+table):
                        if isinstance(value,str):
                            text(db,value);counts[table]+=1
        except sqlite3.DatabaseError as error:
            failures.append({'sha256':h,'type':type(error).__name__})
        db.execute('INSERT INTO processed VALUES(?,?)',(origin,json.dumps({'counts':dict(counts)})))
        db.commit();total.update(counts)
        print(json.dumps({'unique_source_databases':i,'of':len(rows),'text_records':sum(total.values())}),flush=True)
    archive.write(ROOT/'sources_result.json',{'status':'all_source_databases_scanned','databases':len(rows),
                  'counts':dict(total),'failures':failures});db.close();idx.close()


def caches():
    if not all((ROOT/(s+'_result.json')).exists() for s in ('core','stores','sources')):
        raise ValueError('all core and archive source fragments must be indexed first')
    db=stage_start('caches');idx=readonly(STORAGE/'index.sqlite');counts=Counter();schemas=Counter()
    rows=idx.execute("SELECT category,sha256 FROM artifacts WHERE category IN ('atomic_cache','merge_cache') GROUP BY category,sha256 ORDER BY category,sha256")
    for i,(category,h) in enumerate(rows,1):
        if db.execute('SELECT 1 FROM cached_labels WHERE file_sha=?',(h,)).fetchone():
            continue
        row=archive.read(archive.object_path(h)); keys=','.join(sorted(row));schemas[category+':'+keys]+=1
        target=row.get('summary');source=row.get('text_sha256');request=row.get('request_sha256');pid=None;th=None
        if isinstance(target,str) and target.strip():
            th,_=text(db,target)
            found=db.execute('SELECT text FROM texts WHERE sha=?',(source,)).fetchone() if source else None
            if category=='atomic_cache' and found:
                pid=pair(db,'cache:'+h,h,'atomic_cache_joined',None,'raw',found[0],target)
                counts['atomic_content_joined']+=1
            else:
                counts[category+'_source_unresolved']+=1
        else:
            counts[category+'_summary_not_text']+=1
        db.execute('INSERT INTO cached_labels VALUES(?,?,?,?,?,?,?)',(h,category,source,th,request,pid,keys))
        if i%5000==0:
            db.commit();print(json.dumps({'cache_files':i,'counts':counts}),flush=True)
    db.commit();archive.write(ROOT/'caches_result.json',{'status':'unique_cache_contents_counted',
        'counts':dict(counts),'schemas':dict(schemas)});db.close();idx.close()


def report():
    db=readonly(STORAGE/'pairs.sqlite')
    value={'counts_are_unreviewed_candidates_not_qualified_training_pairs':True,
           'distinct_pairs':db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0],
           'by_input_kind':[], 'occurrences_by_kind':dict(db.execute('SELECT kind,COUNT(*) FROM occurrences GROUP BY kind')),
           'issues_by_reason':dict(db.execute('SELECT reason,COUNT(*) FROM issues GROUP BY reason')),
           'completed_stores':db.execute('SELECT COUNT(*) FROM processed').fetchone()[0]}
    for kind in ('raw','children'):
        row=db.execute('''SELECT COUNT(*),COUNT(DISTINCT input_sha),SUM(exact_copy),
            SUM(input_tokens BETWEEN 192 AND 2048 AND target_tokens BETWEEN 1 AND 128 AND exact_copy=0)
            FROM pairs WHERE input_kind=?''',(kind,)).fetchone()
        value['by_input_kind'].append(dict(zip(('kind','pairs','distinct_inputs','exact_copies','application_length_fit_noncopy'),(kind,*row))))
    db.close();print(json.dumps(value,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('core','stores','sources','caches','report'))
    globals()[parser.parse_args().stage]()
