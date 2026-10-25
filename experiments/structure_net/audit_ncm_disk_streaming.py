"""Replay original TRAIN payloads with the existing bounded lossless disk loader."""
import hashlib
import json
from pathlib import Path
import resource
import time

from experiments.structure_net import ncm_sequential_disk_training as launch

run = launch.training
ROOT = run.ROOT / '20261025_sequential_full732_disk_replay_v1'


def audit():
    run.preparation.completed(run.base.DATA)
    rows = run.q.read(run.base.DATA / 'cases.json')
    index = run.q.read(run.base.DATA / 'source_index.json')
    manifest = run.q.read(run.base.DATA / 'feature_manifest.json')
    if len(rows) != 732:
        raise ValueError('complete unchanged TRAIN population required')
    sources = [Path(__file__),Path(launch.__file__),launch.PLAN,
        Path(run.preparation.streaming.__file__),
        Path(run.preparation.streaming.original.__file__),
        Path(run.preparation.streaming.original.codec.__file__)]
    inputs = [run.base.DATA / n for n in ('cases.json','source_index.json',
        'feature_manifest.json','result.json','actual_tool_receipt.json')]
    ROOT.mkdir(); (ROOT / 'producer_source').mkdir()
    for i,path in enumerate(sources):
        (ROOT / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema':'nal.ncm-existing-disk-loader-replay.v1',
        'config':{'feature_cache_bytes':launch.CACHE_BYTES,'TRAIN_cases':732,
            'systems_lifecycle_only_not_quality_evidence':True},
        'source_hashes':{str(p):run.file_hash(p) for p in sources},
        'input_hashes':{str(p):run.file_hash(p) for p in inputs}}
    proof['fingerprint'] = run.q.digest(proof)
    run.write_json(ROOT / 'producer.json',proof)
    cache = run.preparation.streaming.FeatureCache(index,manifest,max_bytes=launch.CACHE_BYTES)
    recorded = []; started = time.monotonic()
    for row in rows:
        features = cache.get(row)
        digest = hashlib.sha256(memoryview(features.numpy()).cast('B')).hexdigest()
        header = run.q.read(manifest[row['id']]['header_path'])
        if digest != header['raw_sha256'] or cache.resident_bytes > launch.CACHE_BYTES:
            raise ValueError('original raw bytes or feature-cache bound differs')
        recorded.append({'id':row['id'],'raw_sha256':digest,'shape':list(features.shape),
            'bytes':features.numel()*features.element_size()})
        if len(recorded)%64 == 0:
            value = {'loaded':len(recorded),'elapsed_seconds':time.monotonic()-started,
                'feature_cache':cache.metrics(),'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            run.write_json(ROOT / 'progress.json',value)
            print(json.dumps(value),flush=True)
    first_pass_seconds = time.monotonic()-started
    hashes = {r['id']:r['raw_sha256'] for r in recorded}
    lookup = {r['id']:r for r in rows}
    longest = max(index,key=lambda r:r['source_positions'])['id']
    for identity in dict.fromkeys((rows[0]['id'],longest)):
        features = cache.get(lookup[identity])
        if hashlib.sha256(memoryview(features.numpy()).cast('B')).hexdigest() != hashes[identity]:
            raise ValueError('reloaded source bytes differ')
    run.write_json(ROOT / 'source_byte_checks.json',recorded)
    run.native.vendor.verify(proof)
    result = {'status':'all732_original_FP32_payloads_streamed_with_bounded_cache',
        'cases':len(recorded),'total_feature_bytes':sum(r['bytes'] for r in recorded),
        'raw_bytes_exact':True,'first_and_longest_reload_exact':True,
        'feature_cache':cache.metrics(),'first_pass_seconds':first_pass_seconds,
        'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'byte_checks_sha256':run.file_hash(ROOT / 'source_byte_checks.json'),
        'producer_fingerprint':proof['fingerprint'],'task_accuracy':None,
        'final_test_opened':False,'speed_tested':False}
    run.write_json(ROOT / 'result.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    audit()
