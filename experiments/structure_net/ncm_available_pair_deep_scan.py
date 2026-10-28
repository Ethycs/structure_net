"""Stream remaining JSON and restore unindexed SQLite/WAL artifacts.

Run with the separate locked pair_census_tools Pixi manifest. Large containers
are never accumulated; their individual summary/source records are visited.
All counts are inventory candidates, not training labels or semantic scores.
"""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import time

import ijson

from experiments.structure_net import ncm_available_pair_archive as archive
from experiments.structure_net import ncm_available_pair_discovery as discovery

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_deep_scan_v1')
BASE=discovery.ROOT
CAP=8*1024**2
SKIP_ARRAYS={'vector','vectors','scores','similarities','embedding','embeddings','weights','token_ids','input_ids'}


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def relevant(value):
    if not isinstance(value,dict):return False
    if isinstance(value.get('summary'),str) and any(k in value for k in
        ('pointer','spans','text_sha256','span_text_sha256','request_sha256','request','request_identity','source_text','input_text')):return True
    if set(value)=={'role','text'} and value.get('role') in ('user','assistant','system') and isinstance(value.get('text'),str):return True
    if isinstance(value.get('messages'),list) and not any(k in value for k in ('question','prediction','answers','qa')):return True
    return False


def walk(value,path,emit):
    if isinstance(value,dict):
        if relevant(value):emit(path,value)
        for key,item in value.items():
            if key in SKIP_ARRAYS:continue
            if key in ('index_json','atomic_index_json') and isinstance(item,str):
                try:walk(json.loads(item),path+'.'+key+':decoded',emit)
                except (ValueError,RecursionError):emit(path+'.'+key+':invalid_embedded_JSON',{'_inventory_error':'invalid embedded JSON'})
            elif isinstance(item,(dict,list)):walk(item,path+'.'+key,emit)
    elif isinstance(value,list):
        for i,item in enumerate(value):
            if isinstance(item,(dict,list)):walk(item,path+'.'+str(i),emit)


def stream_objects(source,emit,counts):
    """SAX parse with bounded container assembly and individual object callbacks."""
    stack=[];ordinal=0
    def append(value,size):
        if not stack:return
        frame=stack[-1]
        if frame['value'] is None:return
        frame['size']+=size
        if frame['size']>CAP:
            frame['value']=None;counts['oversized_containers_streamed']+=1;return
        if frame['kind']=='map':frame['value'][frame['key']]=value
        else:frame['value'].append(value)
    for prefix,event,value in ijson.parse(source,use_float=True):
        if event in ('start_map','start_array'):
            skip=prefix.rsplit('.',1)[-1] in SKIP_ARRAYS
            stack.append({'kind':'map' if event=='start_map' else 'array',
                'value':None if skip else ({} if event=='start_map' else []),'size':2,'key':None})
        elif event=='map_key':
            stack[-1]['key']=value
            stack[-1]['size']+=len(value)
        elif event in ('end_map','end_array'):
            frame=stack.pop();v=frame['value'];size=frame['size']
            if frame['kind']=='map' and v is not None:
                ordinal+=1
                if relevant(v):emit(prefix+':object'+str(ordinal),v)
            append(v,size)
        elif event in ('string','number','boolean','null'):
            if event=='string' and stack and stack[-1]['kind']=='map' and stack[-1]['key'] in ('index_json','atomic_index_json'):
                # Process an encoded index even when its parent exceeds CAP.
                try:
                    stream_objects(BytesIO(value.encode()),lambda p,v:emit(prefix+':decoded.'+p,v),counts)
                except (ValueError,ijson.JSONError):counts['invalid_embedded_JSON']+=1
                append(None,16)
            else:append(value,len(value) if isinstance(value,str) else 16)
    if stack:raise ValueError('unterminated streamed JSON')
    counts['streamed_objects']+=ordinal


def main():
    if ROOT.exists():raise ValueError('fresh deep inventory root required')
    result=archive.read(BASE/'extra_scan_result.json')
    if result['bundles']!=798:raise ValueError('full discovery required')
    ROOT.mkdir();(ROOT/'files').mkdir()
    archive.write(ROOT/'producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'manifest_sha256':archive.sha(Path('experiments/structure_net/pair_census_tools/pixi.toml')),
        'lock_sha256':archive.sha(Path('experiments/structure_net/pair_census_tools/pixi.lock')),
        'ijson_version':ijson.__version__,'ijson_backend':ijson.backend,
        'container_cap_bytes_proxy':CAP,'no_model_calls':True,'training_admitted':False})
    source=sqlite3.connect((BASE/'extra_payloads.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(ROOT/'records.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE records(sha TEXT PRIMARY KEY,raw TEXT);
      CREATE TABLE locations(bundle TEXT,path TEXT,json_path TEXT,record_sha TEXT);
      CREATE TABLE files(bundle TEXT,path TEXT,classification TEXT,sha TEXT,bytes INTEGER,local_path TEXT,PRIMARY KEY(bundle,path));
      CREATE TABLE schemas(keys TEXT PRIMARY KEY,count INTEGER);
      CREATE TABLE bundles(bundle TEXT PRIMARY KEY,counts TEXT);
    ''')
    counts=Counter();schemas=Counter();started=time.monotonic()
    for number,row in enumerate(discovery.rows(),1):
        bundle=row['bundle']
        chosen={p:(kind,h,n) for p,kind,h,n in source.execute('''SELECT path,classification,sha,bytes FROM files
          WHERE bundle=? AND (classification IN ('other_JSON','large_JSON_unparsed')
          OR (extension IN ('.sqlite','.db','.sqlite3','.db-wal','.sqlite-wal') AND classification!='already_indexed'))''',(bundle,))}
        # Include exact main files alongside archived WALs, even if main was indexed.
        for path in list(chosen):
            if path.endswith('-wal'):
                found=source.execute('SELECT classification,sha,bytes FROM files WHERE bundle=? AND path=?',(bundle,path[:-4])).fetchone()
                if found:chosen[path[:-4]]=found
        if not chosen:
            db.execute('INSERT INTO bundles VALUES(?,?)',(bundle,'{}'));db.commit();continue
        local=Counter();seen=set();tarpath=archive.STORAGE/'bundles'/(row['archive_sha256']+'.tar.zst')
        with subprocess.Popen(['zstd','-dc',str(tarpath)],stdout=subprocess.PIPE) as child:
            try:
                with tarfile.open(fileobj=child.stdout,mode='r|') as tar:
                    for member in tar:
                        path=member.name.removeprefix('./')
                        if not member.isfile() or path not in chosen:continue
                        kind,expected,size=chosen[path];seen.add(path)
                        if member.size!=size:raise ValueError('archive member size changed')
                        def emit(json_path,value):
                            raw=canonical(value);h=hashlib.sha256(raw.encode()).hexdigest()
                            db.execute('INSERT OR IGNORE INTO records VALUES(?,?)',(h,raw))
                            db.execute('INSERT INTO locations VALUES(?,?,?,?)',(bundle,path,json_path,h))
                            local['candidate_record_occurrences']+=1
                        if Path(path).suffix in ('.sqlite','.db','.sqlite3','.db-wal','.sqlite-wal'):
                            dest=ROOT/'files'/bundle.replace('/','__')/path
                            if '..' in Path(path).parts or Path(path).is_absolute():raise ValueError('unsafe archive path')
                            dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
                            with tar.extractfile(member) as inp,dest.open('xb') as out:
                                while block:=inp.read(4*1024**2):h.update(block);out.write(block)
                            digest=h.hexdigest();local['SQLite_and_WAL_files_restored']+=1
                            db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',(bundle,path,kind,digest,size,str(dest)))
                        elif kind=='large_JSON_unparsed':
                            with tar.extractfile(member) as inp:stream_objects(inp,emit,local)
                            local['large_JSON_fully_streamed']+=1
                            db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',(bundle,path,'large_JSON_streamed',None,size,None))
                        else:
                            with tar.extractfile(member) as inp:raw=inp.read()
                            digest=hashlib.sha256(raw).hexdigest()
                            if digest!=expected:raise ValueError('small JSON content changed')
                            try:values=[json.loads(raw)] if Path(path).suffix=='.json' else [json.loads(line) for line in raw.splitlines() if line.strip()]
                            except (ValueError,UnicodeError):local['invalid_JSON']+=1;continue
                            for i,value in enumerate(values):
                                if isinstance(value,dict):schemas[','.join(sorted(value))]+=1
                                walk(value,str(i),emit)
                            local['other_JSON_inspected']+=1
                        if sum(local.values())%5000==0:db.commit()
                child.stdout.close()
                if child.wait()!=0:raise ValueError('decompression failure')
            finally:
                if child.poll() is None:child.kill();child.wait()
        if seen!=set(chosen):raise ValueError('selected archive members missing')
        db.execute('INSERT INTO bundles VALUES(?,?)',(bundle,json.dumps(local)));db.commit();counts.update(local)
        print(json.dumps({'bundles_accounted_for':number,'of':798,'counts':counts}),flush=True)
    db.executemany('INSERT INTO schemas VALUES(?,?)',schemas.items());db.commit()
    value={'status':'all_remaining_JSON_and_SQLite_artifacts_inspected','bundles':798,'counts':dict(counts),
        'unique_candidate_records':db.execute('SELECT COUNT(*) FROM records').fetchone()[0],
        'elapsed_seconds':time.monotonic()-started,'source_joins_complete':False,'training_admitted':False}
    archive.write(ROOT/'result.json',value);db.close();source.close();print(json.dumps(value),flush=True)


if __name__=='__main__':argparse.ArgumentParser(__doc__).parse_args();main()
