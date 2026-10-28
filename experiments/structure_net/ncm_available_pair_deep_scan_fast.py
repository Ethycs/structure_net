"""Resume completed archive checkpoints with a native numeric-array prefilter."""
import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile
import threading
import time

from experiments.structure_net import ncm_available_pair_deep_scan as slow

archive=slow.archive
ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_deep_scan_v2')
BINARY=Path('data/experiments/ncm_paper_regime/20261027_available_pair_numeric_filter_build_v1/numeric_arrays')
ORIGINAL_STREAM=slow.stream_objects


def stream(source,emit,counts):
    child=subprocess.Popen([str(BINARY)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    errors=[]
    def feed():
        try:
            shutil.copyfileobj(source,child.stdin,1024**2);child.stdin.close()
        except BaseException as error:errors.append(error)
    writer=threading.Thread(target=feed,daemon=True);writer.start()
    try:
        ORIGINAL_STREAM(child.stdout,emit,counts);child.stdout.close();status=child.wait();writer.join()
        diagnostic=child.stderr.read();child.stderr.close()
        if status or errors:raise ValueError('native JSON filter failed: '+diagnostic.decode()[:1000])
        stats=json.loads(diagnostic)
        counts['numeric_metadata_arrays_elided']+=stats['numeric_metadata_arrays_elided']
        counts['numeric_metadata_values_elided']+=stats['number_values_elided']
    finally:
        if child.poll() is None:child.kill();child.wait()
        writer.join(timeout=5)


def qualify():
    db=sqlite3.connect((slow.ROOT/'records.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    records=[json.loads(r[0]) for r in db.execute("SELECT raw FROM records WHERE json_extract(raw,'$.summary') IS NOT NULL LIMIT 20")];db.close()
    data=json.dumps({'numeric_metadata':[0.123456789]*2000000,'records':records},ensure_ascii=False).encode()
    original=[];filtered=[];a=Counter();b=Counter()
    start=time.monotonic();ORIGINAL_STREAM(BytesIO(data),lambda p,v:original.append(slow.canonical(v)),a);old_time=time.monotonic()-start
    start=time.monotonic();stream(BytesIO(data),lambda p,v:filtered.append(slow.canonical(v)),b);new_time=time.monotonic()-start
    if original!=filtered:raise ValueError('candidate record content or order changed')
    value={'status':'native_prefilter_preserves_actual_summary_records','actual_summary_records':len(records),
        'emitted_records':len(original),'input_bytes':len(data),'original_seconds':old_time,'filtered_seconds':new_time,
        'speedup_on_numeric_fixture':old_time/new_time,'native_counts':dict(b),
        'producer_sha256':archive.sha(Path(__file__)),'binary_sha256':archive.sha(BINARY)}
    archive.write(BINARY.parent/'integration_check.json',value);print(json.dumps(value),flush=True)


def main():
    if ROOT.exists():raise ValueError('fresh resumed inventory root required')
    state=subprocess.check_output(['systemctl','--user','show','structure-net-ncm-available-pair-deep-scan-v1.service',
        '-p','MainPID,ExecMainPID,ExecMainStatus,InvocationID,SubState'],text=True)
    stopped=archive.read(Path('data/experiments/ncm_paper_regime/20261027_available_pair_parser_supersession_v1.json'))
    prior_state=stopped['before']['structure-net-ncm-available-pair-deep-scan-v1.service']
    if ('MainPID=0\n' not in state or stopped['actual_stop_command_exit']!=0
            or 'ExecMainPID=1479589' not in prior_state or 'InvocationID=69ccdc7a588a4b469ab6e982eddd6741' not in prior_state):
        raise ValueError('original parser must be deliberately stopped with its identity preserved')
    check=archive.read(BINARY.parent/'integration_check.json')
    if check['status']!='native_prefilter_preserves_actual_summary_records' or check['binary_sha256']!=archive.sha(BINARY):raise ValueError('qualified binary required')
    ROOT.mkdir();prior=sqlite3.connect((slow.ROOT/'records.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(ROOT/'records.sqlite');prior.backup(db);prior.close()
    completed={r[0] for r in db.execute('SELECT bundle FROM bundles')}
    db.execute('CREATE INDEX IF NOT EXISTS location_record_hash ON locations(record_sha)')
    db.execute('DELETE FROM files WHERE bundle NOT IN (SELECT bundle FROM bundles)')
    db.execute('DELETE FROM locations WHERE bundle NOT IN (SELECT bundle FROM bundles)')
    db.execute('DELETE FROM records WHERE NOT EXISTS(SELECT 1 FROM locations l WHERE l.record_sha=records.sha)');db.commit()
    rows=[r for r in slow.discovery.rows() if r['bundle'] not in completed]
    archive.write(ROOT/'producer.json',{'producer_sha256':archive.sha(Path(__file__)),'prior_parser_state':state,
        'prior_identity_before_manual_stop':prior_state,'manual_stop_receipt':stopped,
        'prior_completed_bundles':len(completed),'remaining_bundles':[r['bundle'] for r in rows],
        'prior_committed_database_sha256':archive.sha(slow.ROOT/'records.sqlite'),
        'prior_WAL_sha256':archive.sha(slow.ROOT/'records.sqlite-wal') if (slow.ROOT/'records.sqlite-wal').exists() else None,
        'filter_checks_sha256':archive.sha(BINARY.parent/'checks.json'),'integration_check_sha256':archive.sha(BINARY.parent/'integration_check.json'),
        'binary_sha256':archive.sha(BINARY),'numeric_only_noninput_arrays_elided':True,
        'protected_input_fields':['child_section_ids','spans','fragments','messages','support','turns'],
        'original_full_archive_bytes_preserved':True,'training_admitted':False})
    source=sqlite3.connect((slow.BASE/'extra_payloads.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    totals=Counter();schemas=Counter()
    for encoded, in db.execute('SELECT counts FROM bundles'):totals.update(json.loads(encoded))
    slow.stream_objects=stream;started=time.monotonic()
    for number,row in enumerate(rows,len(completed)+1):
        name=row['bundle'];local=Counter();seen=set()
        selected={p:(kind,h,size) for p,kind,h,size in source.execute("SELECT path,classification,sha,bytes FROM files WHERE bundle=? AND classification IN ('other_JSON','large_JSON_unparsed')",(name,))}
        missing_db=source.execute("SELECT COUNT(*) FROM files WHERE bundle=? AND classification!='already_indexed' AND extension IN ('.db','.sqlite','.sqlite3','.db-wal','.sqlite-wal')",(name,)).fetchone()[0]
        if missing_db:raise ValueError('remaining database restoration needs explicit handling')
        with subprocess.Popen(['zstd','-dc',str(archive.STORAGE/'bundles'/(row['archive_sha256']+'.tar.zst'))],stdout=subprocess.PIPE) as child:
            try:
                with tarfile.open(fileobj=child.stdout,mode='r|') as tar:
                    for member in tar:
                        path=member.name.removeprefix('./')
                        if not member.isfile() or path not in selected:continue
                        kind,expected,size=selected[path];seen.add(path)
                        if member.size!=size:raise ValueError('archive member size differs')
                        def emit(json_path,value):
                            raw=slow.canonical(value);h=hashlib.sha256(raw.encode()).hexdigest()
                            db.execute('INSERT OR IGNORE INTO records VALUES(?,?)',(h,raw))
                            db.execute('INSERT INTO locations VALUES(?,?,?,?)',(name,path,json_path,h));local['candidate_record_occurrences']+=1
                        if kind=='large_JSON_unparsed':
                            with tar.extractfile(member) as inp:stream(inp,emit,local)
                            local['large_JSON_fully_streamed']+=1
                            db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',(name,path,'large_JSON_streamed',None,size,None));db.commit()
                            print(json.dumps({'large_files_completed_in_remaining_bundles':local['large_JSON_fully_streamed'],
                                'file_bytes':size,'counts':local}),flush=True)
                        else:
                            with tar.extractfile(member) as inp:raw=inp.read()
                            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('small JSON changed')
                            try:values=[json.loads(raw)] if Path(path).suffix=='.json' else [json.loads(x) for x in raw.splitlines() if x.strip()]
                            except (ValueError,UnicodeError):local['invalid_JSON']+=1;continue
                            for i,value in enumerate(values):
                                if isinstance(value,dict):schemas[','.join(sorted(value))]+=1
                                slow.walk(value,str(i),emit)
                            local['other_JSON_inspected']+=1
                child.stdout.close()
                if child.wait():raise ValueError('decompression failed')
            finally:
                if child.poll() is None:child.kill();child.wait()
        if seen!=set(selected):raise ValueError('archive members missing')
        db.execute('INSERT INTO bundles VALUES(?,?)',(name,json.dumps(local)));db.commit();totals.update(local)
        print(json.dumps({'bundles_accounted_for':number,'of':798,'counts':totals}),flush=True)
    if db.execute('SELECT COUNT(*) FROM bundles').fetchone()[0]!=798 or totals['large_JSON_fully_streamed']!=67 or totals['other_JSON_inspected']!=650729:
        raise ValueError('full archive coverage does not reconcile')
    db.executemany('INSERT OR REPLACE INTO schemas VALUES(?,?)',schemas.items());db.commit()
    result={'status':'all_remaining_JSON_and_SQLite_artifacts_inspected','bundles':798,'counts':dict(totals),
        'unique_candidate_records':db.execute('SELECT COUNT(*) FROM records').fetchone()[0],
        'elapsed_seconds_for_resumed_bundles':time.monotonic()-started,'completed_original_bundles':len(completed),
        'schema_frequency_scope':'remaining bundles only; full schema counts retained by first discovery',
        'source_joins_complete':False,'training_admitted':False}
    archive.write(ROOT/'result.json',result);db.close();source.close();print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('qualify','run'));a=p.parse_args()
    qualify() if a.stage=='qualify' else main()
