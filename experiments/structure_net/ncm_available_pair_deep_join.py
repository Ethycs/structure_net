"""Join deeply embedded summaries and archived SQLite views to real sources."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3

from experiments.structure_net import ncm_available_pair_census as census
from experiments.structure_net import ncm_available_pair_supplement as supplement
from experiments.structure_net import ncm_available_pair_receipts as receipts

archive=census.archive
BASE=receipts.ROOT
SCAN=Path('data/experiments/ncm_paper_regime/20261027_available_pair_deep_scan_v1')
ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_deep_join_v1')


def record(db,row,origin,number,counts):
    if not isinstance(row,dict):return
    if any(s in str(row.get('status','')).lower() for s in ('invalid','failed','error')):
        counts['failed_records']+=1;return
    if set(row)=={'role','text'} and row.get('role') in ('user','assistant','system'):
        supplement.add_source(db,row['text'],origin);counts['source_turn_records']+=1
    shape=receipts.request_shape(row)
    if shape:
        kind,messages,fragments=shape
        encoded=json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        ident=census.digest(encoded)
        db.execute('INSERT OR IGNORE INTO requests VALUES(?,?,?,?)',(ident,kind,encoded,json.dumps(fragments,ensure_ascii=False)))
        keys={ident,census.digest(encoded+'\n')}
        for key in ('request_sha256','messages_sha256'):
            if isinstance(row.get(key),str):keys.add(row[key])
        db.executemany('INSERT OR IGNORE INTO request_keys VALUES(?,?)',[(k,ident) for k in keys])
        if kind=='raw':
            for f in fragments:supplement.add_source(db,f['fragment'],origin)
        counts['deep_requests_'+kind]+=1
    target=row.get('summary')
    if not isinstance(target,str) or not target.strip():return
    if supplement.add_label(db,row,origin,number):counts['pointer_label_occurrences']+=1
    sid=row.get('section_id')
    if isinstance(sid,str):
        db.execute('INSERT OR IGNORE INTO section_targets VALUES(?,?,?)',(sid,census.digest(target),target))
        children=row.get('child_section_ids')
        if isinstance(children,list) and children:
            db.execute('INSERT OR IGNORE INTO structural_labels VALUES(?,?,?,?)',
                       (census.digest(json.dumps([children,target],ensure_ascii=False)),sid,json.dumps(children),target))
    source_hash=row.get('text_sha256',row.get('span_text_sha256'))
    if isinstance(source_hash,str) and len(source_hash)==64:
        raw=json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        db.execute('INSERT OR IGNORE INTO hash_labels VALUES(?,?,?,?)',(census.digest(raw),source_hash,target,raw))
        counts['hash_label_occurrences']+=1
    for key in ('source_text','input_text'):
        if isinstance(row.get(key),str) and row[key].strip():
            census.pair(db,origin,str(number)+':'+key,'explicit_source_summary',None,'raw',row[key],target)
            counts['explicit_source_summary_records']+=1;break


def databases(db,scan,counts):
    seen=set();schemas=Counter();out=[]
    rows=scan.execute("SELECT bundle,path,sha,local_path FROM files WHERE local_path IS NOT NULL AND path NOT LIKE '%-wal'")
    for number,(bundle,path,sha,local_path) in enumerate(rows,1):
        wal=scan.execute('SELECT sha,local_path FROM files WHERE bundle=? AND path=?',(bundle,path+'-wal')).fetchone()
        identity=(sha,wal[0] if wal else None)
        if identity in seen:counts['duplicate_SQLite_views']+=1;continue
        seen.add(identity);original=Path(local_path)
        if archive.sha(original)!=sha:raise ValueError('restored database changed')
        if wal:
            view=ROOT/'wal_views'/sha;view.mkdir(parents=True)
            main=view/'source.db';shutil.copyfile(original,main)
            if archive.sha(wal[1])!=wal[0]:raise ValueError('restored WAL changed')
            shutil.copyfile(wal[1],str(main)+'-wal')
            source=census.readonly(main);snapshot=view/'snapshot.sqlite';dest=sqlite3.connect(snapshot)
            source.backup(dest);dest.close();source.close();source=census.readonly(snapshot)
            out.append({'main_sha256':sha,'wal_sha256':wal[0],'snapshot_sha256':archive.sha(snapshot)})
        else:source=census.readonly(original)
        tables={r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        schemas[','.join(sorted(tables))]+=1
        for table in ('turns','chunks','events'):
            if table not in tables:continue
            names={r[1] for r in source.execute('PRAGMA table_info('+table+')')}
            if 'text' in names:
                for (text,) in source.execute('SELECT text FROM '+table):
                    supplement.add_source(db,text,'deep_database:'+sha+':'+table);counts['database_source_text_records']+=1
        if 'sections' in tables:
            names={r[1] for r in source.execute('PRAGMA table_info(sections)')}
            if {'kind','sid','payload'}<=names:
                for sid,payload in source.execute('SELECT sid,payload FROM sections'):
                    record(db,json.loads(payload),'deep_database:'+sha,sid,counts)
        if 'snapshot' in tables:
            names={r[1] for r in source.execute('PRAGMA table_info(snapshot)')}
            if 'payload' in names:
                for (payload,) in source.execute('SELECT payload FROM snapshot'):
                    value=json.loads(payload)
                    for key in ('atomic_sections','hierarchy_sections','sections'):
                        for row in value.get(key,[]):record(db,row,'deep_database:'+sha,key+':'+row.get('section_id',''),counts)
        if 'bodies' in tables:
            names={r[1] for r in source.execute('PRAGMA table_info(bodies)')}
            if {'body_sha256','body_json'}<=names:
                for h,raw in source.execute('SELECT body_sha256,body_json FROM bodies'):
                    if not supplement.ingest_body(db,json.loads(raw),'deep_database:'+sha,h):
                        counts['unsupported_body_record']+=1
            if 'summaries_json' in names:
                for h,raw in source.execute('SELECT body_sha256,summaries_json FROM bodies'):
                    for i,row in enumerate(json.loads(raw)):record(db,row,'deep_database:'+sha,h+':'+str(i),counts)
        source.close();db.commit()
        print(json.dumps({'deep_database_files':number,'counts':counts}),flush=True)
    archive.write(ROOT/'database_views.json',{'schema_counts':dict(schemas),'WAL_snapshots':out,'unique_views':len(seen)})


def remaining_joins(db,base,counts):
    def lookup(sha):
        row=db.execute('SELECT text FROM source_texts WHERE sha=?',(sha,)).fetchone()
        if row is None:row=base.execute('SELECT t.text FROM texts t JOIN verified_inputs v ON v.sha=t.sha WHERE t.sha=?',(sha,)).fetchone()
        return row[0] if row else None
    for key,sha,target,raw in db.execute('SELECT * FROM hash_labels'):
        source=lookup(sha);row=json.loads(raw);support=row.get('support',[])
        if source is None:counts['hash_label_missing_source']+=1;continue
        if not isinstance(support,list) or any(not isinstance(q,str) or q not in source for q in support):
            counts['hash_label_bad_support']+=1;continue
        census.pair(db,'deep_hash_label',key,'source_hash_joined',None,'raw',source,target)
        counts['hash_labels_joined']+=1
    for key,sid,encoded,target in db.execute('SELECT * FROM structural_labels'):
        values=[]
        for child in json.loads(encoded):
            options=db.execute('SELECT target FROM section_targets WHERE sid=?',(child,)).fetchall()
            if len(options)!=1:break
            values.append(options[0][0])
        else:
            census.pair(db,'deep_structural_label',key,'ordered_children_joined',None,'children',
                       json.dumps(values,ensure_ascii=False,separators=(',',':')),target)
            counts['structural_labels_joined']+=1;continue
        counts['structural_labels_missing_or_ambiguous_children']+=1
    db.commit()


def main():
    if ROOT.exists():raise ValueError('fresh deep join required')
    for path in (SCAN/'result.json',supplement.ROOT/'result.json'):
        if not path.exists():raise ValueError('completed prerequisite required')
    ROOT.mkdir();archive.write(ROOT/'producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'supplement_helper_sha256':archive.sha(Path(supplement.__file__)),'scan_result_sha256':archive.sha(SCAN/'result.json'),
        'supplement_result_sha256':archive.sha(supplement.ROOT/'result.json'),'new_model_calls':0,'training_admitted':False})
    census.STORAGE=ROOT;db=census.open_census();base=census.readonly(BASE/'pairs.sqlite')
    previous=census.readonly(supplement.ROOT/'pairs.sqlite');scan=census.readonly(SCAN/'records.sqlite')
    db.executescript('''
      CREATE TABLE source_texts(sha TEXT PRIMARY KEY,text TEXT,origin TEXT);
      CREATE TABLE labels(id TEXT PRIMARY KEY,kind TEXT,target TEXT,pointers TEXT,metadata TEXT);
      CREATE TABLE label_locations(origin TEXT,record TEXT,label_id TEXT,PRIMARY KEY(origin,record));
      CREATE TABLE label_results(id TEXT PRIMARY KEY,pair_id TEXT,proof TEXT);
      CREATE TABLE requests(file_sha TEXT PRIMARY KEY,kind TEXT,messages TEXT,fragments TEXT);
      CREATE TABLE request_keys(key TEXT,file_sha TEXT,PRIMARY KEY(key,file_sha));
      CREATE TABLE hash_labels(id TEXT PRIMARY KEY,source_sha TEXT,target TEXT,metadata TEXT);
      CREATE TABLE section_targets(sid TEXT,target_sha TEXT,target TEXT,PRIMARY KEY(sid,target_sha));
      CREATE TABLE structural_labels(id TEXT PRIMARY KEY,sid TEXT,children TEXT,target TEXT);
    ''')
    for table,n in [('source_texts',3),('requests',4),('request_keys',2)]:
        db.executemany('INSERT INTO '+table+' VALUES('+','.join('?'*n)+')',previous.execute('SELECT * FROM '+table))
    previous.close();db.commit();counts=Counter();databases(db,scan,counts)
    for i,(sha,raw) in enumerate(scan.execute('SELECT sha,raw FROM records'),1):
        record(db,json.loads(raw),'deep_record:'+sha,'root',counts)
        if i%25000==0:db.commit();print(json.dumps({'deep_records':i,'counts':counts}),flush=True)
    db.commit();supplement.join(db,base,counts);remaining_joins(db,base,counts)
    value={'status':'deep_JSON_and_SQLite_source_joins_completed','counts':dict(counts),
        'distinct_pointer_labels':db.execute('SELECT COUNT(*) FROM labels').fetchone()[0],
        'unresolved_pointer_labels':db.execute('SELECT COUNT(*) FROM label_results WHERE pair_id IS NULL').fetchone()[0],
        'distinct_pairs_before_cross_corpus_dedup':db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0],
        'training_admitted':False,'semantic_quality_reviewed':False}
    archive.write(ROOT/'result.json',value);db.close();base.close();scan.close();print(json.dumps(value),flush=True)


if __name__=='__main__':argparse.ArgumentParser(__doc__).parse_args();main()
