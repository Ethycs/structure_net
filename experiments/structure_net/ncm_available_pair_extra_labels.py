"""Inventory labels and real summary requests in previously unindexed JSON.

This stage records source pointers and preserves missing-input labels. It does
not admit them as recovered pairs, change training, or read QA fields as labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.structure_net import ncm_available_pair_archive as archive
from experiments.structure_net import ncm_available_pair_receipts as receipts

SOURCE=receipts.ROOT
ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_extra_labels_v1')


def identity(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def label(row):
    if not isinstance(row,dict) or not isinstance(row.get('summary'),str) or not row['summary'].strip():return None
    target=row['summary']
    if isinstance(row.get('spans'),list) and row['spans']:
        pointers=row['spans'];kind='section_spans'
    elif isinstance(row.get('pointer'),dict):pointers=[row['pointer']];kind='summary_pointer'
    else:return None
    keys=('span_text_sha256','turn_text_sha256','start_char','end_char')
    if any(not isinstance(p,dict) or not all(k in p for k in keys)
           or not isinstance(p['span_text_sha256'],str) or len(p['span_text_sha256'])!=64
           or not isinstance(p['turn_text_sha256'],str) or len(p['turn_text_sha256'])!=64
           or type(p['start_char']) is not int or type(p['end_char']) is not int
           or not 0<=p['start_char']<p['end_char'] for p in pointers):return None
    content_key=identity([[{k:p[k] for k in keys} for p in pointers],target])
    return content_key,kind,target,pointers


def extract_rows(value):
    if isinstance(value,list):
        for i,row in enumerate(value):yield str(i),row
        return
    if not isinstance(value,dict):return
    if label(value):yield 'root',value
    for key in ('atoms','summaries','atomic_sections','hierarchy_sections','sections'):
        rows=value.get(key)
        if isinstance(rows,list):
            for i,row in enumerate(rows):yield key+':'+str(i),row
        elif isinstance(rows,dict):
            for name,row in rows.items():yield key+':'+str(name),row


def main():
    if ROOT.exists():raise ValueError('fresh unindexed-label census required')
    result=archive.read(SOURCE/'extra_scan_result.json')
    if result['bundles']!=798:raise ValueError('complete all-bundle discovery required')
    ROOT.mkdir();archive.write(ROOT/'producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'discovery_result_sha256':archive.sha(SOURCE/'extra_scan_result.json'),
        'source_pointer_labels_not_yet_admitted_as_pairs':True,'new_model_calls':0})
    src=sqlite3.connect((SOURCE/'extra_payloads.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(ROOT/'labels.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE labels(id TEXT PRIMARY KEY,kind TEXT,target TEXT,pointers TEXT);
      CREATE TABLE locations(file_sha TEXT,record TEXT,label_id TEXT,metadata TEXT,PRIMARY KEY(file_sha,record));
      CREATE TABLE requests(file_sha TEXT,record TEXT,kind TEXT,messages TEXT,fragments TEXT,keys TEXT,PRIMARY KEY(file_sha,record));
      CREATE TABLE unresolved(file_sha TEXT,record TEXT,kind TEXT,metadata TEXT,PRIMARY KEY(file_sha,record));
    ''')
    counts=Counter();schemas=Counter()
    for i,(h,raw) in enumerate(src.execute('SELECT sha,raw FROM candidates'),1):
        try:values=[json.loads(raw)]
        except (ValueError,UnicodeError):
            try:values=[json.loads(line) for line in raw.splitlines() if line.strip()]
            except (ValueError,UnicodeError):counts['invalid_payload']+=1;continue
        for number,value in enumerate(values):
            counts['JSON_records']+=1
            if isinstance(value,dict) and any(k in value for k in ('question','qa','prediction','answers','reference_answer')):
                counts['question_answer_or_evaluation_record']+=1;continue
            if isinstance(value,dict) and any(s in str(value.get('status','')).lower() for s in ('invalid','failed','error')):
                counts['failed_or_invalid_record']+=1;continue
            request=receipts.request_shape(value)
            if request:
                kind,messages,fragments=request
                encoded=json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':'))
                keys={h,hashlib.sha256(encoded.encode()).hexdigest(),hashlib.sha256((encoded+'\n').encode()).hexdigest()}
                for key in ('request_sha256','messages_sha256'):
                    if isinstance(value.get(key),str) and len(value[key])==64:keys.add(value[key])
                db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?)',(h,str(number),kind,encoded,json.dumps(fragments,ensure_ascii=False),json.dumps(sorted(keys))))
                counts['summary_requests_'+kind]+=1
            for name,row in extract_rows(value):
                record=str(number)+':'+name;found=label(row)
                if found:
                    key,kind,target,pointers=found
                    db.execute('INSERT OR IGNORE INTO labels VALUES(?,?,?,?)',(key,kind,target,json.dumps(pointers,ensure_ascii=False)))
                    db.execute('INSERT INTO locations VALUES(?,?,?,?)',(h,record,key,json.dumps(row,ensure_ascii=False)))
                    counts['pointer_label_occurrences']+=1
                elif isinstance(row,str) and row.strip():
                    db.execute('INSERT INTO unresolved VALUES(?,?,?,?)',(h,record,'summary_text_without_source_pointer',json.dumps(row,ensure_ascii=False)))
                    counts['summary_text_without_source_pointer']+=1
                elif isinstance(row,dict):
                    shape=','.join(sorted(row));schemas[shape]+=1
                    db.execute('INSERT INTO unresolved VALUES(?,?,?,?)',(h,record,'unhandled_record_schema',json.dumps(row,ensure_ascii=False)))
                    counts['unhandled_record_schema']+=1
            if isinstance(value,dict) and not request and not any(k in value for k in ('atoms','summaries','atomic_sections','hierarchy_sections','sections','summary')):
                counts['other_metadata_or_source_only_record']+=1
        if i%5000==0:db.commit();print(json.dumps({'extra_payloads':i,'counts':counts}),flush=True)
    db.commit()
    value={'status':'all_unindexed_candidate_payloads_label_pointers_classified',
        'payloads':i,'counts':dict(counts),'distinct_pointer_target_records':db.execute('SELECT COUNT(*) FROM labels').fetchone()[0],
        'unresolved_rows':db.execute('SELECT COUNT(*) FROM unresolved').fetchone()[0],
        'unhandled_row_schemas':dict(schemas),'source_joins_completed':False,'training_admitted':False}
    archive.write(ROOT/'result.json',value);db.close();src.close();print(json.dumps(value),flush=True)


if __name__=='__main__':argparse.ArgumentParser(__doc__).parse_args();main()
