"""Restore the entire published summary-artifact inventory for a pair census.

No models, labels, evaluation queries or training-set admission. Payloads are
content-addressed on /data2; every selected member is checked against the
published SHA256 and every downloaded bundle against both DVC MD5 and SHA256.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
import datetime
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
import ssl
import subprocess
import tarfile
import threading
import time
import urllib.request


HANDOFF = Path('data/experiments/ncm_memory/20261025_broad_handoff_v1')
ROOT = Path('data/experiments/ncm_paper_regime/20261027_available_pair_census_v1')
STORAGE = Path('/data2/structure_net/ncm_available_pairs_20261027_v1')
COMMIT = 'ded03c25a12e56d620710d2bff334ba5beed923584783c460084eb5922a8270a'
LOCK = threading.Lock()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def object_path(digest):
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('valid content SHA256 required')
    return STORAGE/'objects'/digest[:2]/digest


def index():
    if ROOT.exists() or STORAGE.exists():
        raise ValueError('fresh census roots required')
    ROOT.mkdir(); STORAGE.mkdir(parents=True)
    manifest = read(HANDOFF/'handoff/SHA256.json')
    source = HANDOFF/'handoff/summary-artifacts.jsonl.zst'
    if sha(source) != manifest[source.name]['sha256']:
        raise ValueError('published full artifact inventory differs')
    catalog = read(HANDOFF/'metadata_v2/archive/catalog.json')['results']
    dvc = {r['relpath']: r['md5'] for r in read(HANDOFF/'metadata_v2/research_data.dir.json')}
    conn = sqlite3.connect(STORAGE/'index.sqlite')
    conn.executescript('''
      CREATE TABLE artifacts(bundle TEXT, path TEXT, category TEXT, bytes INTEGER,
                             sha256 TEXT, PRIMARY KEY(bundle,path));
      CREATE INDEX artifact_hash ON artifacts(sha256);
      CREATE TABLE bundles(bundle TEXT PRIMARY KEY, metadata TEXT);
    ''')
    counts = Counter()
    with subprocess.Popen(['zstd', '-dc', str(source)], stdout=subprocess.PIPE, text=True) as child:
        for line in child.stdout:
            r = json.loads(line)
            conn.execute('INSERT INTO artifacts VALUES(?,?,?,?,?)',
                         (r['bundle'], r['path'], r['category'], r['bytes'], r['sha256']))
            counts[r['category']] += 1
        if child.wait() != 0:
            raise ValueError('full compressed metadata stream failed')
    chosen = {r[0] for r in conn.execute('SELECT DISTINCT bundle FROM artifacts')}
    for row in catalog:
        if row['bundle'] in chosen:
            row = {**row, 'dvc_md5': dvc['reusable-research-20261021/'+row['bundle']+'/data.tar.zst']}
            conn.execute('INSERT INTO bundles VALUES(?,?)', (row['bundle'], json.dumps(row)))
    conn.commit()
    totals = {k: {'files': n, 'unique_content_hashes': conn.execute(
        'SELECT COUNT(DISTINCT sha256) FROM artifacts WHERE category=?', (k,)).fetchone()[0]}
        for k, n in counts.items()}
    conn.close()
    proof = {'status': 'complete_published_artifact_index_ready', 'lakefs_commit': COMMIT,
             'archive_bundles_total': len(catalog), 'selected_bundles': len(chosen),
             'selected_compressed_bytes': sum(r['output_sizes']['data.tar.zst'] for r in catalog if r['bundle'] in chosen),
             'categories': totals, 'pair_count_established': False,
             'training_admitted': False, 'new_model_calls': 0,
             'artifacts': {str(p): sha(p) for p in [Path(__file__), source,
                  HANDOFF/'metadata_v2/archive/catalog.json', HANDOFF/'metadata_v2/research_data.dir.json', STORAGE/'index.sqlite']}}
    write(ROOT/'index_result.json', proof)
    print(json.dumps({k:v for k,v in proof.items() if k != 'artifacts'}), flush=True)


class Client:
    """Small read-only S3 SigV4 client; credentials never enter argv or reports."""
    def __init__(self):
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read('data/experiments/ncm_memory/20261022_shared_v1/source/.dvc/config.local')
        remote = next(cfg[s] for s in cfg.sections() if 'lakefs' in s)
        self.access = remote['access_key_id'].strip().strip('\"').strip("'")
        self.secret = remote['secret_access_key'].strip().strip('\"').strip("'")
        self.context = ssl.create_default_context(cafile='/home/rabbit/.config/lakefs/caddy-root.crt')

    def get(self, md5):
        host = 'central-dev.zt:8000'
        path = '/memory-condense/'+COMMIT+'/files/md5/'+md5[:2]+'/'+md5[2:]
        now = datetime.datetime.now(datetime.timezone.utc)
        stamp, day = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y%m%d')
        payload = hashlib.sha256(b'').hexdigest()
        headers = {'host': host, 'x-amz-content-sha256': payload, 'x-amz-date': stamp}
        names = ';'.join(sorted(headers))
        canonical = 'GET\n'+path+'\n\n'+''.join(k+':'+headers[k]+'\n' for k in sorted(headers))+'\n'+names+'\n'+payload
        scope = day+'/us-east-1/s3/aws4_request'
        message = 'AWS4-HMAC-SHA256\n'+stamp+'\n'+scope+'\n'+hashlib.sha256(canonical.encode()).hexdigest()
        key = ('AWS4'+self.secret).encode()
        for part in (day, 'us-east-1', 's3', 'aws4_request'):
            key = hmac.new(key, part.encode(), hashlib.sha256).digest()
        signature = hmac.new(key, message.encode(), hashlib.sha256).hexdigest()
        headers['Authorization'] = 'AWS4-HMAC-SHA256 Credential='+self.access+'/'+scope+', SignedHeaders='+names+', Signature='+signature
        return urllib.request.urlopen(urllib.request.Request('https://'+host+path, headers=headers),
                                      context=self.context, timeout=60)


def restore_one(row):
    name = row['bundle'].split('/')[-1]
    receipt = ROOT/'restored'/(name+'.json')
    if receipt.exists():
        value = read(receipt)
        if value['archive_sha256'] != row['archive_sha256']:
            raise ValueError('completed bundle binding changed')
        return {**value, 'reused_completed_restore': True}
    archive = STORAGE/'bundles'/(row['archive_sha256']+'.tar.zst')
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        client = Client()
        for attempt in range(1,4):
            temp = archive.with_suffix('.attempt'+str(attempt)+'.partial')
            if temp.exists():
                continue
            try:
                md5, digest, size = hashlib.md5(), hashlib.sha256(), 0
                with client.get(row['dvc_md5']) as response, temp.open('xb') as out:
                    while block := response.read(8*1024**2):
                        out.write(block); md5.update(block); digest.update(block); size += len(block)
                if (size != row['output_sizes']['data.tar.zst'] or md5.hexdigest() != row['dvc_md5']
                        or digest.hexdigest() != row['archive_sha256']):
                    raise ValueError('bundle size, DVC MD5 or SHA256 mismatch')
                temp.rename(archive)
                write(ROOT/'downloads'/(name+'.json'), {'bundle':row['bundle'], 'bytes':size,
                      'md5':md5.hexdigest(), 'sha256':digest.hexdigest(), 'lakefs_commit':COMMIT})
                break
            except Exception as error:
                write(ROOT/'download_failures'/(name+'-'+str(attempt)+'.json'),
                      {'bundle':row['bundle'], 'attempt':attempt, 'type':type(error).__name__,
                       'HTTP_status':getattr(error,'code',None)})
                if attempt == 3:
                    raise
        if not archive.exists():
            raise ValueError('no verified downloaded bundle; earlier failed attempts preserved')
    elif sha(archive) != row['archive_sha256']:
        raise ValueError('existing archive SHA256 differs')
    conn = sqlite3.connect((STORAGE/'index.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    wanted = {p:(size,h) for p,size,h in conn.execute(
        'SELECT path,bytes,sha256 FROM artifacts WHERE bundle=?', (row['bundle'],))}
    conn.close()
    seen = set(); new_objects = 0; bytes_verified = 0
    with subprocess.Popen(['zstd','-dc',str(archive)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
        try:
            with tarfile.open(fileobj=child.stdout,mode='r|') as tar:
                for member in tar:
                    path = member.name.removeprefix('./')
                    if path not in wanted:
                        continue
                    size, expected = wanted[path]
                    if path in seen or not member.isfile() or member.size != size:
                        raise ValueError('selected archive member type, duplication or size differs')
                    destination = object_path(expected)
                    if destination.exists():
                        if destination.stat().st_size != size:
                            raise ValueError('content-addressed payload size changed')
                        # This exact content was previously streamed and verified.
                        seen.add(path); bytes_verified += size
                        continue
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    temporary = destination.with_name(expected+'.'+name+'.partial')
                    digest = hashlib.sha256(); count = 0
                    with tar.extractfile(member) as source, temporary.open('xb') as out:
                        while block := source.read(8*1024**2):
                            out.write(block); digest.update(block); count += len(block)
                    if count != size or digest.hexdigest() != expected:
                        raise ValueError('selected member SHA256 or size differs')
                    with LOCK:
                        if destination.exists():
                            temporary.unlink()
                        else:
                            temporary.rename(destination); new_objects += 1
                    seen.add(path); bytes_verified += size
            child.stdout.close()
            if child.wait() != 0:
                raise ValueError('archive decompression failed')
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
    if seen != set(wanted):
        raise ValueError('not every declared summary artifact found in bundle')
    value = {'bundle':row['bundle'], 'archive_sha256':row['archive_sha256'],
             'selected_members':len(seen), 'selected_bytes':bytes_verified,
             'new_content_objects':new_objects, 'all_selected_members_available':True}
    write(receipt,value)
    return value


def restore():
    proof = read(ROOT/'index_result.json')
    if proof['artifacts'][str(Path(__file__))] != sha(Path(__file__)):
        raise ValueError('frozen inventory producer changed')
    conn = sqlite3.connect((STORAGE/'index.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    rows = [json.loads(r[0]) for r in conn.execute('SELECT metadata FROM bundles')]
    conn.close()
    rows.sort(key=lambda r:r['output_sizes']['data.tar.zst'])
    started = time.monotonic(); failures=[]; done=0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(restore_one,r):r['bundle'] for r in rows}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result=future.result(); done+=1
                print(json.dumps({'completed_bundles':done,'total':len(rows),'bundle':name,
                                  'members':result['selected_members'],'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
            except Exception as error:
                failures.append({'bundle':name,'type':type(error).__name__,'message':str(error)[:200]})
                print(json.dumps({'restore_error':failures[-1]}),flush=True)
    result={'status':'all_indexed_summary_artifacts_restored' if not failures else 'incomplete_restore',
            'completed_bundles':done,'selected_bundles':len(rows),'failures':failures,
            'elapsed_seconds':time.monotonic()-started,'pair_count_established':False,'training_admitted':False}
    write(ROOT/'restore_result.json',result)
    if failures:
        raise ValueError('incomplete archive restore; preserved explicit failures')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('index','restore'))
    globals()[parser.parse_args().stage]()
