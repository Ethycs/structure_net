"""Recover the separate batched hierarchy-job protocol and prompt-keyed caches."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sqlite3

from experiments.structure_net import ncm_available_pair_census as census
from experiments.structure_net import ncm_available_pair_receipts as receipts

archive=census.archive
ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_batch_merges_v1')
BASE=receipts.ROOT
CONTRACT=census.CORE/'source/src/memory_condense/search/spine_summary.py'


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def job_body(value):
    if not isinstance(value,dict) or not set(value)<={'kind','fragments','user_spine','max_output_tokens','max_prompt_tokens','label'}:return None
    fragments=value.get('fragments')
    if value.get('kind') not in ('user_spine','attached_context') or not isinstance(fragments,list) or not fragments:return None
    if any(not isinstance(f,dict) or set(f)!={'role','transcript_date','summary'}
           or not isinstance(f['summary'],str) or not f['summary'].strip() for f in fragments):return None
    if type(value.get('max_output_tokens')) is not int or value['max_output_tokens']<1:return None
    if value.get('user_spine') is not None and not isinstance(value['user_spine'],str):return None
    return {k:value.get(k) for k in ('kind','fragments','user_spine','max_output_tokens')}


def batch_outputs(jobs,content):
    value=json.loads(content);rows=value.get('summaries') if isinstance(value,dict) else None
    if not isinstance(rows,list) or len(rows)!=len(jobs):raise ValueError('merge job count mismatch')
    output=[]
    for i,(job,row) in enumerate(zip(jobs,rows,strict=True)):
        if not isinstance(row,dict) or row.get('label')!='S'+str(i) or not isinstance(row.get('summary'),str) or not row['summary'].strip():
            raise ValueError('merge job label/order mismatch')
        if len(census.ENCODING.encode(row['summary'],disallowed_special=()))>job['max_output_tokens']:
            raise ValueError('merge target exceeds its own job cap')
        output.append(row['summary'])
    return output


def scan(db,counts,system):
    def admit_job(value):
        body=job_body(value)
        if body is None:return None
        messages=[{'role':'system','content':system},{'role':'user','content':canonical(body)}]
        key=census.digest(canonical(messages))
        db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?)',(key,canonical(body),canonical([f['summary'] for f in body['fragments']])))
        return key
    for table,path in [('payloads',BASE/'payloads.sqlite'),('candidates',BASE/'extra_payloads.sqlite')]:
        source=census.readonly(path)
        for i,(sha,raw) in enumerate(source.execute('SELECT sha,raw FROM '+table),1):
            try:values=[json.loads(raw)]
            except (ValueError,UnicodeError):
                try:values=[json.loads(line) for line in raw.splitlines() if line.strip()]
                except (ValueError,UnicodeError):continue
            for ordinal,value in enumerate(values):
                if not isinstance(value,dict):continue
                if any(k in value for k in ('question','prediction','answers','qa')):continue
                nested=value.get('request');messages=value.get('messages') or (nested.get('messages') if isinstance(nested,dict) else None)
                if isinstance(messages,list):
                    users=[m.get('content') for m in messages if isinstance(m,dict) and m.get('role')=='user']
                    if len(users)==1 and isinstance(users[0],str):
                        try:body=json.loads(users[0])
                        except ValueError:body=None
                        if isinstance(body,dict) and set(body)=={'jobs'} and isinstance(body['jobs'],list) and body['jobs']:
                            jobs=body['jobs'];keys=[admit_job(j) for j in jobs]
                            if all(keys) and all(j.get('label')=='S'+str(n) for n,j in enumerate(jobs)):
                                ident=sha+':'+str(ordinal);encoded=canonical(messages)
                                db.execute('INSERT OR IGNORE INTO requests VALUES(?,?,?)',(ident,encoded,json.dumps(keys)))
                                references={sha,census.digest(encoded),census.digest(encoded+'\n')}
                                for k in ('request_sha256','messages_sha256'):
                                    if isinstance(value.get(k),str):references.add(value[k])
                                db.executemany('INSERT OR IGNORE INTO request_keys VALUES(?,?)',[(k,ident) for k in references])
                                counts['batch_merge_requests']+=1
                        elif job_body(body):admit_job(body)
                # Saved typed jobs bind the same single-job prompt hash as the
                # original compiler, including role/date/spine conditioning.
                for key in ('job','jobs'):
                    values2=value.get(key)
                    if key=='job':values2=[values2]
                    if isinstance(values2,list):
                        for job in values2:admit_job(job)
                if value.get('valid') is False or any(s in str(value.get('status','')).lower() for s in ('invalid','failed','error')):continue
                summaries=value.get('summaries')
                if isinstance(summaries,dict) and 'atoms_sha256' in value:
                    for key,target in summaries.items():
                        if isinstance(key,str) and len(key)==64 and isinstance(target,str) and target.strip():
                            db.execute('INSERT OR IGNORE INTO labels VALUES(?,?,?,?,?)',(sha,str(ordinal)+':'+key,key,target,'prompt_keyed_summary_map'))
                            counts['summary_map_occurrences']+=1
                if (isinstance(value.get('summary'),str) and value['summary'].strip()
                        and not any(k in value for k in ('text_sha256','span_text_sha256','support'))):
                    for key in ('request_sha256','original_job_sha256','prompt_sha256'):
                        if isinstance(value.get(key),str) and len(value[key])==64:
                            db.execute('INSERT OR IGNORE INTO labels VALUES(?,?,?,?,?)',(sha,str(ordinal)+':'+key,value[key],value['summary'],'job_keyed_summary'))
            if i%10000==0:db.commit();print(json.dumps({'payload_table':table,'scanned':i,'counts':counts}),flush=True)
        db.commit();source.close()


def main():
    if ROOT.exists():raise ValueError('fresh batch census required')
    ROOT.mkdir();system=None
    for node in ast.parse(CONTRACT.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SPINE_SUMMARY_SYSTEM' for t in node.targets):system=ast.literal_eval(node.value)
    if not isinstance(system,str):raise ValueError('original merge system contract unavailable')
    archive.write(ROOT/'producer.json',{'producer_sha256':archive.sha(Path(__file__)),'original_contract_sha256':archive.sha(CONTRACT),
        'input_conditioning_retained_in_job_metadata':True,'pair_identity_normalizes_to_ordered_summary_texts':True,'new_model_calls':0,'training_admitted':False})
    census.STORAGE=ROOT;db=census.open_census()
    db.executescript('''
      CREATE TABLE jobs(key TEXT PRIMARY KEY,body TEXT,source TEXT);
      CREATE TABLE requests(id TEXT PRIMARY KEY,messages TEXT,jobs TEXT);
      CREATE TABLE request_keys(key TEXT,id TEXT,PRIMARY KEY(key,id));
      CREATE TABLE labels(file_sha TEXT,record TEXT,job_key TEXT,target TEXT,kind TEXT,PRIMARY KEY(file_sha,record));
      CREATE TABLE outcomes(origin TEXT,record TEXT,pair_id TEXT,reason TEXT,PRIMARY KEY(origin,record));
    ''')
    counts=Counter();scan(db,counts,system)
    for file_sha,record,key,target,kind in db.execute('SELECT * FROM labels'):
        job=db.execute('SELECT body,source FROM jobs WHERE key=?',(key,)).fetchone()
        if job is None:counts[kind+'_missing_job']+=1;continue
        body=json.loads(job[0])
        if len(census.ENCODING.encode(target,disallowed_special=()))>body['max_output_tokens']:
            counts[kind+'_over_original_cap']+=1;continue
        pid=census.pair(db,'batch_cache:'+file_sha,record,kind,None,'children',job[1],target)
        db.execute('INSERT INTO outcomes VALUES(?,?,?,?)',(file_sha,record,pid,'job_hash_verified'));counts[kind+'_joined']+=1
    base=census.readonly(BASE/'pairs.sqlite')
    for sha,key,messages_key,content,finish in base.execute('SELECT * FROM responses'):
        if finish not in ('stop','eos','eos_token'):continue
        found={}
        for reference in (key,messages_key):
            if reference:
                for ident,messages,encoded in db.execute('SELECT r.id,r.messages,r.jobs FROM requests r JOIN request_keys k ON k.id=r.id WHERE k.key=?',(reference,)):found[(messages,encoded)]=encoded
        if len(found)!=1:continue
        keys=json.loads(next(iter(found.values())));jobs=[json.loads(db.execute('SELECT body FROM jobs WHERE key=?',(k,)).fetchone()[0]) for k in keys]
        try:targets=batch_outputs(jobs,content)
        except (ValueError,KeyError,TypeError):counts['invalid_batch_response']+=1;continue
        for i,(key,target) in enumerate(zip(keys,targets,strict=True)):
            source=db.execute('SELECT source FROM jobs WHERE key=?',(key,)).fetchone()[0]
            census.pair(db,'batch_response:'+sha,str(i),'validated_batch_merge',None,'children',source,target);counts['batch_response_pairs']+=1
    db.commit();value={'status':'batch_merge_job_and_summary_cache_census_completed','counts':dict(counts),
        'distinct_jobs':db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],
        'distinct_pairs':db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0],'training_admitted':False}
    archive.write(ROOT/'result.json',value);db.close();base.close();print(json.dumps(value),flush=True)


if __name__=='__main__':argparse.ArgumentParser(__doc__).parse_args();main()
