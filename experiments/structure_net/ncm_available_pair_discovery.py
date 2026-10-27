"""Account for archived files omitted by the sender's filename-based index."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import time

from experiments.structure_net import ncm_available_pair_archive as archive

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_census_fast_v1')
FIELDS={'messages','turns','body','atoms','summaries','atomic_sections','hierarchy_sections',
        'sections','source_text','input_text','output_text','source','target','fragments'}


def rows():
    index={r['relpath']:r['md5'] for r in archive.read(archive.HANDOFF/'metadata_v2/research_data.dir.json')}
    return [{**r,'dvc_md5':index['reusable-research-20261021/'+r['bundle']+'/data.tar.zst']}
            for r in archive.read(archive.HANDOFF/'metadata_v2/archive/catalog.json')['results']]


def restore_extra():
    conn=sqlite3.connect(archive.STORAGE/'index.sqlite')
    old={r[0] for r in conn.execute('SELECT bundle FROM bundles')};conn.close()
    selected=[r for r in rows() if r['bundle'] not in old];failures=[];complete=0
    archive.write(ROOT/'extra_restore_producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'archive_helper_sha256':archive.sha(Path(archive.__file__)),'bundles':len(selected),
        'compressed_bytes':sum(r['output_sizes']['data.tar.zst'] for r in selected)})
    with ThreadPoolExecutor(max_workers=2) as pool:
        work={pool.submit(archive.restore_one,r):r['bundle'] for r in selected}
        for future in as_completed(work):
            try:future.result();complete+=1
            except Exception as error:failures.append({'bundle':work[future],'type':type(error).__name__})
            if (complete+len(failures))%25==0:
                print(json.dumps({'extra_bundles_restored':complete,'of':len(selected),'failures':len(failures)}),flush=True)
    archive.write(ROOT/'extra_restore_result.json',{'status':'all_extra_bundles_restored' if not failures else 'incomplete_extra_restore',
        'bundles':len(selected),'completed':complete,'failures':failures})
    if failures:raise ValueError('additional archive restoration incomplete')


def interesting(value):
    if isinstance(value,dict):return bool(set(value)&FIELDS)
    if isinstance(value,list):return any(isinstance(r,dict) and set(r)&FIELDS for r in value[:20])
    return False


def scan_extra():
    # Every bundle has already passed the immutable archive checks before use.
    for p,status in ((archive.ROOT/'restore_result.json','all_indexed_summary_artifacts_restored'),
                     (ROOT/'extra_restore_result.json','all_extra_bundles_restored')):
        if archive.read(p)['status']!=status:raise ValueError('all798 bundles required')
    archive.write(ROOT/'extra_scan_producer.json',{'producer_sha256':archive.sha(Path(__file__)),
        'scope':'all archive member names; nonindexed JSON/JSONL payload schema discovery',
        'candidate_fields':sorted(FIELDS),'maximum_single_JSON_bytes':64*1024**2,
        'large_unparsed_payloads_preserved_in_report':True,'training_admitted':False})
    idx=sqlite3.connect(archive.STORAGE/'index.sqlite');db=sqlite3.connect(ROOT/'extra_payloads.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE files(bundle TEXT,path TEXT,bytes INTEGER,extension TEXT,classification TEXT,sha TEXT,PRIMARY KEY(bundle,path));
      CREATE TABLE candidates(sha TEXT PRIMARY KEY,raw BLOB);
      CREATE TABLE bundle_results(bundle TEXT PRIMARY KEY,result TEXT);
    ''')
    totals=Counter();schemas=Counter();oversized=[];started=time.monotonic()
    for i,row in enumerate(rows(),1):
        name=row['bundle'];known={r[0] for r in idx.execute('SELECT path FROM artifacts WHERE bundle=?',(name,))}
        path=archive.STORAGE/'bundles'/(row['archive_sha256']+'.tar.zst');local=Counter()
        with subprocess.Popen(['zstd','-dc',str(path)],stdout=subprocess.PIPE) as child:
            try:
                with tarfile.open(fileobj=child.stdout,mode='r|') as tar:
                    for member in tar:
                        if not member.isfile():continue
                        p=member.name.removeprefix('./');suffix=Path(p).suffix.lower();classification='non_JSON';h=None
                        if p in known:classification='already_indexed'
                        elif suffix in ('.json','.jsonl','.ndjson'):
                            if member.size>64*1024**2:
                                classification='large_JSON_unparsed';oversized.append({'bundle':name,'path':p,'bytes':member.size})
                            else:
                                with tar.extractfile(member) as source:raw=source.read()
                                h=hashlib.sha256(raw).hexdigest()
                                try:
                                    values=[json.loads(raw)] if suffix=='.json' else [json.loads(line) for line in raw.splitlines() if line.strip()]
                                    relevant=any(interesting(v) for v in values)
                                    classification='candidate_JSON' if relevant else 'other_JSON'
                                    for v in values:
                                        if isinstance(v,dict):schemas[','.join(sorted(v))]+=1
                                    if relevant:db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?)',(h,raw))
                                except (ValueError,UnicodeError):classification='invalid_JSON'
                                del raw
                        local[classification]+=1
                        db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)',(name,p,member.size,suffix,classification,h))
                child.stdout.close()
                if child.wait()!=0:raise ValueError('extra discovery decompression failed')
            finally:
                if child.poll() is None:child.kill();child.wait()
        db.execute('INSERT INTO bundle_results VALUES(?,?)',(name,json.dumps(dict(local))));db.commit();totals.update(local)
        if i%25==0:print(json.dumps({'all_archive_bundles_scanned':i,'of':798,'counts':totals}),flush=True)
    result={'status':'all_archive_member_names_and_small_unindexed_JSON_schemas_counted','bundles':798,
        'counts':dict(totals),'unique_extra_candidate_payloads':db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
        'unparsed_large_JSON':oversized,'all_possible_pairs_established':False,
        'elapsed_seconds':time.monotonic()-started}
    archive.write(ROOT/'extra_scan_result.json',result);archive.write(ROOT/'extra_schema_counts.json',dict(schemas))
    db.close();idx.close();print(json.dumps({k:v for k,v in result.items() if k!='unparsed_large_JSON'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('restore_extra','scan_extra'))
    globals()[parser.parse_args().stage]()
