"""Inventory the six original source-data files without using QA labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import tiktoken

from experiments.structure_net import ncm_available_pair_archive as archive

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_source_package_v1')
SOURCE=Path('data/experiments/ncm_memory/20261022_shared_v1/source')
ENCODING=tiktoken.get_encoding('cl100k_base')


def stream_array(path):
    """Decode one top-level array item at a time, retaining no prior records."""
    decoder=json.JSONDecoder();buffer='';started=False;need_value=True;ended=False;may_be_empty=True
    with Path(path).open() as stream:
        eof=False
        while True:
            if not eof:
                part=stream.read(1024*1024);eof=not part;buffer+=part
            while True:
                buffer=buffer.lstrip()
                if not started:
                    if not buffer:break
                    if buffer[0]!='[':raise ValueError('top-level array required')
                    buffer=buffer[1:];started=True;continue
                if ended:
                    if buffer:raise ValueError('extra data after array')
                    break
                if not buffer:break
                if need_value:
                    if buffer[0]==']':
                        if not may_be_empty:raise ValueError('trailing array comma')
                        buffer=buffer[1:];ended=True;continue
                    try:value,end=decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        if eof:raise
                        break
                    yield value;buffer=buffer[end:];need_value=False;may_be_empty=False
                else:
                    if buffer[0]==',':buffer=buffer[1:];need_value=True
                    elif buffer[0]==']':buffer=buffer[1:];ended=True
                    else:raise ValueError('array separator required')
            if eof:
                if not ended or buffer.strip():raise ValueError('incomplete JSON array')
                return


def fragments(value,cap=2048):
    tokens=ENCODING.encode(value,disallowed_special=())
    if len(tokens)<=cap:
        if value:yield value
        return
    pieces=[ENCODING.decode_single_token_bytes(t) for t in tokens];start=position=0
    while start<len(tokens):
        end=min(start+cap,len(tokens))
        while end>start:
            try:
                part=b''.join(pieces[start:end]).decode('utf-8')
                if len(ENCODING.encode(part,disallowed_special=()))<=cap:break
            except UnicodeDecodeError:pass
            end-=1
        if end==start:raise ValueError('fragment cannot hold one Unicode character')
        if value[position:position+len(part)]!=part:raise ValueError('fragment text changed')
        yield part;position+=len(part);start=end
    if position!=len(value):raise ValueError('fragmentation lost source text')


def put(db,value,kind,origin):
    if not isinstance(value,str) or not value:return
    h=hashlib.sha256(value.encode()).hexdigest()
    db.execute('INSERT OR IGNORE INTO source_texts VALUES(?,?,?)',(h,value,len(ENCODING.encode(value,disallowed_special=()))))
    db.execute('INSERT OR IGNORE INTO source_occurrences VALUES(?,?,?)',(origin,kind,h))
    if kind=='turn':
        for fragment in fragments(value):put(db,fragment,'fragment',origin)


def main():
    if ROOT.exists():raise ValueError('fresh source-package census required')
    ROOT.mkdir()
    index_path=SOURCE/'.dvc/cache/files/md5/57/e6647cf62ed14ebb881b610603f05f.dir'
    index_bytes=index_path.read_bytes()
    if hashlib.md5(index_bytes).hexdigest()!='57e6647cf62ed14ebb881b610603f05f':
        raise ValueError('original source DVC directory binding differs')
    index=json.loads(index_bytes);verified=[]
    for row in index:
        path=SOURCE/'data'/row['relpath'];md5=hashlib.md5();sha=hashlib.sha256();size=0
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):
                md5.update(chunk);sha.update(chunk);size+=len(chunk)
        if md5.hexdigest()!=row['md5']:raise ValueError('source payload DVC binding differs')
        verified.append({'path':str(path),'md5':row['md5'],'sha256':sha.hexdigest(),'bytes':size})
    archive.write(ROOT/'producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'DVC_directory_MD5_verified':True,'files':verified,'new_model_calls':0,
        'QA_fields_used_as_training_inputs_or_targets':False,'training_admitted':False})
    db=sqlite3.connect(ROOT/'sources.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE source_texts(sha TEXT PRIMARY KEY,text TEXT,tokens INTEGER);
      CREATE TABLE source_occurrences(origin TEXT,kind TEXT,sha TEXT,PRIMARY KEY(origin,kind,sha));
      CREATE TABLE session_pairs(id TEXT PRIMARY KEY,source_sha TEXT,target_sha TEXT,source TEXT,target TEXT,input_tokens INTEGER,target_tokens INTEGER,origin TEXT,has_image_metadata INTEGER);
    ''')
    counts={}
    for name in ('longmemeval_s_cleaned.json','longmemeval_oracle.json'):
        local=Counter()
        for ordinal,row in enumerate(stream_array(SOURCE/'data'/name)):
            local['records']+=1
            sessions=row['haystack_sessions'];ids=row['haystack_session_ids']
            if len(sessions)!=len(ids):raise ValueError('source session identities differ')
            for sid,turns in zip(ids,sessions,strict=True):
                local['session_occurrences']+=1
                for i,turn in enumerate(turns):
                    value=turn.get('content',turn.get('text'))
                    if not isinstance(value,str):raise ValueError('text-only source turn required')
                    put(db,value,'turn',name+':'+str(sid)+':'+str(i));local['turn_occurrences']+=1
            if ordinal%25==0:db.commit()
        counts[name]=dict(local);db.commit();print(json.dumps({'file':name,'counts':local}),flush=True)
    local=Counter()
    for ordinal,row in enumerate(stream_array(SOURCE/'data/locomo10.json')):
        conversation=row['conversation'];summaries=row.get('session_summary',{})
        for key,turns in conversation.items():
            if not key.startswith('session_') or not isinstance(turns,list):continue
            local['sessions']+=1;parts=[];has_image=False
            for i,turn in enumerate(turns):
                value=turn['text'];put(db,value,'turn','locomo:'+str(ordinal)+':'+key+':'+str(i))
                local['turns']+=1;parts.append(str(turn['speaker'])+': '+value)
                if any(k in turn for k in ('img_url','image_url','blip_caption','img_caption')):
                    local['turns_with_image_metadata']+=1;has_image=True
            target=summaries.get(key+'_summary')
            if isinstance(target,str) and target.strip():
                source='\n'.join(parts);sh=hashlib.sha256(source.encode()).hexdigest();th=hashlib.sha256(target.encode()).hexdigest()
                h=hashlib.sha256((sh+':'+th).encode()).hexdigest()
                db.execute('INSERT OR IGNORE INTO session_pairs VALUES(?,?,?,?,?,?,?,?,?)',(h,sh,th,source,target,
                    len(ENCODING.encode(source,disallowed_special=())),len(ENCODING.encode(target,disallowed_special=())),str(ordinal)+':'+key,int(has_image)))
                local['stored_session_summaries']+=1
        db.commit()
    counts['locomo10.json']=dict(local)
    local=Counter()
    for i,row in enumerate(stream_array(SOURCE/'data/build-session-8f7f7561.turns.json')):
        if not isinstance(row,list) or len(row)!=3 or row[0] not in ('user','assistant','system'):
            raise ValueError('fixed build-turn triplet schema differs')
        put(db,row[2],'turn','build:'+str(i));local[row[0]]+=1
    counts['build-session-8f7f7561.turns.json']=dict(local)
    # The snapshot and Markdown are alternate raw transcript sources, not
    # stored source-summary pairs. Count their records separately; do not
    # invent a target or claim their source overlap has been fully resolved.
    kinds=Counter()
    with (SOURCE/'data/build-session-8f7f7561.snapshot.jsonl').open() as stream:
        for line in stream:
            if line.strip():value=json.loads(line);kinds[str(value.get('type'))]+=1
    counts['build-session-8f7f7561.snapshot.jsonl']={'record_types':dict(kinds),'stored_summary_pairs_identified':0}
    counts['codex-transcript-2026-08-15-to-2026-08-23.md']={'kind':'raw_markdown_transcript','stored_summary_pairs_identified':0}
    db.commit()
    eligible=db.execute("SELECT COUNT(DISTINCT o.sha) FROM source_occurrences o JOIN source_texts t ON t.sha=o.sha WHERE o.kind='fragment' AND t.tokens BETWEEN 192 AND 2048").fetchone()[0]
    paired=db.execute('SELECT COUNT(*),SUM(input_tokens BETWEEN 192 AND 2048 AND target_tokens BETWEEN 1 AND 128),SUM(input_tokens BETWEEN 192 AND 2048 AND target_tokens BETWEEN 1 AND 128 AND has_image_metadata=0) FROM session_pairs').fetchone()
    result={'status':'all_six_original_data_files_accounted_for','file_counts':counts,
        'distinct_source_fragment_contents':db.execute("SELECT COUNT(DISTINCT sha) FROM source_occurrences WHERE kind='fragment'").fetchone()[0],
        'distinct_192_to2048_token_source_fragments':eligible,'stored_session_summary_pairs':paired[0],
        'session_summary_pairs_with_application_lengths':paired[1],
        'text_only_session_summary_pairs_with_application_lengths':paired[2],
        'QA_labels_used':False,'raw_sources_are_not_automatically_labeled_pairs':True,
        'cross_archive_overlap_not_yet_reconciled':True,'snapshot_and_markdown_source_overlap_pending':True,
        'image_metadata_in_LOC0MO_not_represented_in_text_pair':True,'training_admitted':False}
    archive.write(ROOT/'result.json',result);db.close();print(json.dumps(result),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(__doc__).parse_args();main()
