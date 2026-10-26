"""Inventory shared archive metadata without opening any source/summary payload."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('data/experiments/ncm_memory/20261025_broad_handoff_v1')
DEST = Path('data/experiments/ncm_paper_regime/20261025_handoff_routing_inventory_v1')
EXPECTED = {'atomic_cache':253366,'merge_cache':13843,'source_database':158,'summary_database':324}


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def route(source_root):
    # Routing hints deliberately do not claim source independence.
    if 'longmemeval' in source_root.lower(): return 'longmemeval_named_root'
    if 'engineering' in source_root.lower(): return 'engineering_named_root'
    return 'unclassified_root'


def main():
    if DEST.exists(): raise ValueError('fresh metadata-only inventory required')
    manifest=read(ROOT/'handoff/SHA256.json')
    selected=['source-families.json','summary-artifacts.jsonl.zst','engineering-shard-verification.json']
    for name in selected:
        p=ROOT/'handoff'/name
        if sha(p)!=manifest[name]['sha256'] or p.stat().st_size!=manifest[name]['bytes']:
            raise ValueError('received handoff metadata changed')
    metadata=read(ROOT/'metadata_v2/restore_receipt.json')
    needed={ROOT/'metadata_v2'/n for n in ('handoff/HANDOFF.json','archive/catalog.json','archive/coverage.json')}
    receipts={Path(r['path']).resolve():r for r in metadata['objects']}
    for p in needed:
        receipt=receipts[p.resolve()]
        if sha(p)!=receipt['sha256'] or p.stat().st_size!=receipt['bytes']:
            raise ValueError('received archive metadata changed')
    families=read(ROOT/'handoff/source-families.json')['stores']
    catalog=read(ROOT/'metadata_v2/archive/catalog.json')['results']
    bundle_roots={r['bundle']:r['source'] for r in catalog}
    if len(bundle_roots)!=len(catalog): raise ValueError('unique archive bundle roots required')
    totals=Counter(); unique=defaultdict(set); grouped=defaultdict(Counter)
    engineering_sources=set(); source_metadata=set(); physical=0
    with subprocess.Popen(['zstd','-dc',str(ROOT/'handoff/summary-artifacts.jsonl.zst')],
                          stdout=subprocess.PIPE,text=True) as child:
        for line in child.stdout:
            row=json.loads(line); physical+=1
            category=row['category']; kind=route(bundle_roots[row['bundle']])
            totals[category]+=1; unique[category].add(row['sha256']); grouped[kind][category]+=1
            if category=='source_database':
                key=(row['bundle'],row['path'],row['sha256'])
                source_metadata.add(key)
                if kind=='engineering_named_root': engineering_sources.add(key)
        if child.wait()!=0: raise ValueError('complete metadata index required')
    if any(totals[k]!=v for k,v in EXPECTED.items()):
        raise ValueError('complete declared archive denominator differs')
    if source_metadata!={(r['bundle'],r['path'],r['sha256']) for r in families}:
        raise ValueError('source-family/index bindings differ')
    handoff=read(ROOT/'metadata_v2/handoff/HANDOFF.json')
    verification=read(ROOT/'handoff/engineering-shard-verification.json')
    coverage=read(ROOT/'metadata_v2/archive/coverage.json')
    # The sender declares25 engineering stores. This is metadata coverage,
    # not a claim that every source component/target has been admitted.
    if len(engineering_sources)!=handoff['engineering_shard']['stores']:
        raise ValueError('engineering export source-store count differs')
    roots=defaultdict(list)
    for row in families:
        roots[row['source_root']].append(row)
    by_root=[{'source_root':name,'routing_hint':route(name),'source_databases':len(values),
              'unique_source_database_hashes':len({r['sha256'] for r in values}),
              'source_database_bytes':sum(r['bytes'] for r in values),
              'source_independence_verified':False}
             for name,values in sorted(roots.items())]
    result={'status':'shared_archive_metadata_routing_inventoried_no_training_admission',
            'physical_index_records':physical,'physical_categories':dict(totals),
            'unique_file_hashes_by_category':{k:len(v) for k,v in unique.items()},
            'path_routing_counts':{k:dict(v) for k,v in grouped.items()},
            'source_databases':len(families),'source_database_roots':len(roots),
            'engineering_source_databases':len(engineering_sources),
            'existing_engineering_export':handoff['engineering_shard'],
            'all_engineering_named_source_store_count_covered_by_existing_export':True,
            'unclassified_sources_are_not_verified_non_LME':True,
            'archive_compressed_bytes':coverage['archive_bytes'],
            'archived_source_bytes':coverage['source_bytes'],
            'source_or_summary_payloads_opened':0,'new_downloads':0,'new_requests':0,
            'new_training_pairs_admitted':0,'model_training':False,'task_accuracy':None,
            'sealed_final_source_text_opened_or_hashed':False,
            'limitation':'File hashes count unique files, not independent sources or qualified complete-summary labels. Path hints cannot certify non-LongMemEval provenance.'}
    paths=[Path(__file__),ROOT/'handoff/SHA256.json',ROOT/'metadata_v2/restore_receipt.json',
           *[ROOT/'handoff'/n for n in selected],*sorted(needed)]
    result['artifacts']={str(p):sha(p) for p in paths}
    DEST.mkdir()
    for name,value in [('result.json',result),('source_roots.json',by_root)]:
        (DEST/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (DEST/'producer.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','path_routing_counts')}))


if __name__=='__main__':main()
