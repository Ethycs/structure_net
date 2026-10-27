"""Stream archived JSON into an SSD SQLite index for exhaustive pair joins."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import time

from experiments.structure_net import ncm_available_pair_archive as archive

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_census_fast_v1')
CATEGORIES=('atomic_cache','merge_cache','possible_request_response_provenance','other_summary_candidate')


def run():
    proof=ROOT/'payload_pack_producer.json'
    if proof.exists():
        if archive.read(proof)['producer_sha256']!=archive.sha(Path(__file__)):
            raise ValueError('payload pack producer changed')
    else:
        archive.write(proof,{'producer_sha256':archive.sha(Path(__file__)),
            'index_sha256':archive.sha(archive.STORAGE/'index.sqlite'),'categories':CATEGORIES})
    idx=sqlite3.connect((archive.STORAGE/'index.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db=sqlite3.connect(ROOT/'payloads.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS payloads(sha TEXT PRIMARY KEY,raw BLOB);
        CREATE TABLE IF NOT EXISTS packed(bundle TEXT PRIMARY KEY, records INTEGER);
    ''')
    db.execute('PRAGMA cache_size=-65536')
    rows=[json.loads(r[0]) for r in idx.execute('SELECT metadata FROM bundles')]
    pending={r['bundle']:r for r in sorted(rows,key=lambda r:r['output_sizes']['data.tar.zst'])}
    complete={r[0] for r in db.execute('SELECT bundle FROM packed')}
    for name in complete:pending.pop(name)
    started=time.monotonic()
    while pending:
        progressed=False
        for name,row in list(pending.items()):
            receipt=archive.ROOT/'restored'/(name.split('/')[-1]+'.json')
            if not receipt.exists():continue
            expected={p:(size,h) for p,size,h,category in idx.execute(
                'SELECT path,bytes,sha256,category FROM artifacts WHERE bundle=?',(name,)) if category in CATEGORIES}
            remaining={p for p,(_,h) in expected.items() if db.execute('SELECT 1 FROM payloads WHERE sha=?',(h,)).fetchone() is None}
            found=set()
            if remaining:
                path=archive.STORAGE/'bundles'/(row['archive_sha256']+'.tar.zst')
                with subprocess.Popen(['zstd','-dc',str(path)],stdout=subprocess.PIPE) as child:
                    try:
                        with tarfile.open(fileobj=child.stdout,mode='r|') as tar:
                            for member in tar:
                                p=member.name.removeprefix('./')
                                if p not in remaining:continue
                                size,h=expected[p]
                                if not member.isfile() or member.size!=size or p in found:
                                    raise ValueError('selected JSON payload geometry differs')
                                with tar.extractfile(member) as f:raw=f.read()
                                if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=h:
                                    raise ValueError('selected JSON payload SHA256 differs')
                                db.execute('INSERT OR IGNORE INTO payloads VALUES(?,?)',(h,raw));found.add(p)
                                del raw
                        child.stdout.close()
                        if child.wait()!=0:raise ValueError('full bundle stream failed')
                    finally:
                        if child.poll() is None:child.kill();child.wait()
                if found!=remaining:raise ValueError('indexed payload absent from archive')
            db.execute('INSERT INTO packed VALUES(?,?)',(name,len(expected)));db.commit()
            pending.pop(name);complete.add(name);progressed=True
            print(json.dumps({'packed_bundles':len(complete),'total':len(rows),'JSON_files':len(expected),
                              'bundle':name,'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
        if not progressed:
            state=subprocess.check_output(['systemctl','--user','show','structure-net-ncm-available-pair-archive-v1.service',
                '--property=ActiveState,SubState'],text=True)
            if 'ActiveState=active' not in state and pending:
                raise ValueError('archive restorer terminal with unpacked bundles')
            time.sleep(5)
    expected=idx.execute("SELECT COUNT(DISTINCT sha256) FROM artifacts WHERE category IN (?,?,?,?)",CATEGORIES).fetchone()[0]
    actual=db.execute('SELECT COUNT(*) FROM payloads').fetchone()[0]
    if expected!=actual:raise ValueError('full unique JSON payload denominator differs')
    archive.write(ROOT/'payload_pack_result.json',{'status':'all_indexed_JSON_payloads_packed',
        'unique_JSON_payloads':actual,'bundles':len(rows),'all_payload_SHA256_verified':True,
        'elapsed_seconds':time.monotonic()-started,'pair_count_established':False})
    db.close();idx.close()


if __name__=='__main__':run()
