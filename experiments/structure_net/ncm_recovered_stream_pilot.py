"""Qualify source/context/code joins using all completed feature-pilot sources.

CPU-only I/O diagnostic, no training admission or optimizer/model/judge calls.
The full immutable pair/code indexes are used; the explicitly selected subset
contains every pair referring to the 15 already audited source-feature records.
Compare serial and two-batch-prefetched reads, including repeated pair indices
and a restarted suffix, without changing presentation order or numeric bytes.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from experiments.structure_net import ncm_recovered_training_stream as stream
from experiments.structure_net import ncm_public_curriculum as schedule
from experiments.structure_net.ncm_indexed_jsonl import IndexedJSONL

BASE = Path('data/experiments/ncm_paper_regime')
ROOT = BASE/'20261028_recovered_training_stream_pilot_v1'
CATALOG = BASE/'20261028_recovered_paraphrase_catalog_v1'
META = BASE/'20261028_recovered_paraphrase_metadata_full_v1'
CODES = BASE/'20261028_recovered_paraphrase_codes_full_v1'
FEATURES = BASE/'20261028_recovered_paraphrase_features_pilot_v1'
TEST = Path('tests/structure_net/test_ncm_recovered_training_stream.py')
CODEC = '9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32'
BODY = '75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c'


def read(path): return json.loads(path.read_text())
def lines(path):
    with path.open() as f: return list(map(json.loads, f))


def digest(rows):
    value = hashlib.sha256()
    for row in rows:
        value.update(json.dumps({k: v for k, v in row.items() if k != 'features'}, sort_keys=True).encode())
        matrix = row['features']
        value.update(str(tuple(matrix.shape)).encode()); value.update(matrix.numpy().tobytes())
    return value.hexdigest()


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only consumer pilot required')
    hashes = {}
    def add(values):
        for p, h in values.items():
            key = str(Path(p).resolve())
            if key in hashes and hashes[key] != h: raise ValueError('inconsistent parent hash binding')
            hashes[key] = h
    for root in (CATALOG, META, CODES, FEATURES):
        receipt, result = (read(root/n) for n in ('actual_tool_receipt.json', 'result.json'))
        if receipt['actual_tool_exit_code'] != 0: raise ValueError('completed actual parent receipt required')
        add(receipt['artifacts']); add(result['artifacts'])
        add({str(root/n): stream.file_hash(root/n) for n in ('actual_tool_receipt.json', 'result.json')})
    audit = read(FEATURES/'audit_actual_tool_receipt.json')
    if audit['actual_tool_exit_code'] != 0 or not read(FEATURES/'audit_result.json')['all_disk_records_replayed']:
        raise ValueError('complete successful source-feature disk replay required')
    add(audit['artifacts'])
    add({str(FEATURES/n): stream.file_hash(FEATURES/n) for n in ('audit_result.json', 'audit_actual_tool_receipt.json')})
    for p, h in hashes.items():
        if stream.file_hash(p) != h: raise ValueError('parent receipt artifact changed: ' + p)
    code_result, feature_result = read(CODES/'result.json'), read(FEATURES/'result.json')
    if (code_result['frames'] != 43894 or code_result['codec_sha256'] != CODEC
            or feature_result['cases'] != 15 or feature_result['body_sha256'] != BODY):
        raise ValueError('complete fixed target corpus and source pilot identities required')
    pair_path = CATALOG/'pairs.jsonl'
    pairs = IndexedJSONL(pair_path, lines(CATALOG/'pairs_index.jsonl'), hashes[str(pair_path.resolve())])
    entries = lines(Path(feature_result['storage'])/'index.jsonl')
    source_ids = {r['id'] for r in entries}
    selected = [i for i in range(len(pairs)) if pairs[i]['source_record_id'] in source_ids]
    codes = stream.CodeStore(code_result['binary'], lines(Path(code_result['index'])),
        expected_sha256=hashes[str(Path(code_result['binary']).resolve())], codec_sha256=CODEC)
    dataset = stream.RecoveredTrainingDataset(pairs, codes, lines(META/'index.jsonl'), entries,
        body_sha256=BODY, codec_sha256=CODEC, pair_positions=selected)
    inputs = {str(p): stream.file_hash(p) for p in (CATALOG/'pairs_index.jsonl', META/'index.jsonl',
        Path(code_result['index']), Path(feature_result['storage'])/'index.jsonl', Path('pixi.lock'), Path('pyproject.toml'))}
    source_files = (Path(__file__), Path(stream.__file__), Path(schedule.__file__),
        Path('experiments/structure_net/ncm_indexed_jsonl.py'), Path(stream.features.__file__),
        Path(stream.features.codec.__file__), Path('experiments/structure_net/ncm_memory_data.py'), TEST)
    proof = {'schema': 'nal.ncm-recovered-training-stream-pilot.v1',
        'config': {'source_population': 15, 'microbatch': 4, 'prefetch_batches': 2, 'CPU_only': True,
            'maximum_RSS_bytes': 4*1024**3, 'feature_cache': False, 'model_calls': 0, 'training_admitted': False},
        'source_hashes': {str(p): stream.file_hash(p) for p in source_files},
        'input_hashes': {**hashes, **inputs}}
    ROOT.mkdir(); (ROOT/'producer_source').mkdir()
    def write(name, value):
        with (ROOT/name).open('x') as f: json.dump(value, f, indent=2); f.write('\n')
    for i, p in enumerate(source_files): (ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    write('producer.json', proof)
    write('selection.json', {'catalog_pair_positions': selected, 'pairs': [r['id'] for r in dataset.joins],
                             'source_records': sorted(source_ids)})
    order = list(reversed(range(len(dataset)))) + [0, len(dataset)-1, 0]
    examples = [schedule.Example('recovered', index, 1, position, bool(position%2), 271142+position)
                for position, index in enumerate(order)]
    requests = [(i//4, min(4, len(examples)-i), examples[i:i+4]) for i in range(0, len(examples), 4)]
    def fetch(values):
        return [{**row, 'presentation': asdict(e)} for row, e in
                zip(dataset.__getitems__([e.index for e in values]), values, strict=True)]
    serial = []; positions = 0; started = time.monotonic()
    for _, _, values in requests:
        rows = fetch(values); serial.append(digest(rows))
        positions += sum(r['features'].shape[0] for r in rows)
        del rows
    serial_seconds = time.monotonic()-started
    started = time.monotonic(); replayed = []
    for _, _, _, rows in schedule.prefetch(requests, fetch, capacity=2):
        replayed.append(digest(rows)); del rows
    prefetch_seconds = time.monotonic()-started
    if serial != replayed: raise ValueError('prefetch changed complete feature/target bytes or presentation order')
    cut = len(requests)//2
    resumed = [digest(rows) for _, _, _, rows in schedule.prefetch(requests[cut:], fetch, capacity=2)]
    if resumed != serial[cut:]: raise ValueError('resumed I/O suffix differs')
    for collection in (proof['source_hashes'], inputs):
        for p, h in collection.items():
            if stream.file_hash(p) != h: raise ValueError('consumer input changed')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss > proof['config']['maximum_RSS_bytes']: raise ValueError('bounded host memory ceiling exceeded')
    write('result.json', {'schema': proof['schema'], 'status': 'all_selected_source_context_target_joins_replayed',
        'catalog_pairs': len(pairs), 'selected_pairs': len(dataset), 'source_contexts': len(source_ids),
        'presentations': len(order), 'source_positions_read': positions, 'batches': len(requests),
        'serial_and_prefetched_bytes_identical': True, 'resumed_suffix_identical': True,
        'serial_wall_seconds_including_hashing': serial_seconds,
        'prefetch_wall_seconds_including_hashing': prefetch_seconds,
        'timing_scope': 'single CPU I/O diagnostic with warm filesystem cache; not inference or training speed',
        'peak_RSS_bytes': rss, 'task_accuracy': None, 'model_calls': 0, 'new_judge_calls': 0,
        'final_test_opened': False, 'training_admitted': False,
        'artifacts': {str(p): stream.file_hash(p) for p in ROOT.iterdir() if p.is_file()}})
    pairs.close()
    print(json.dumps({k: v for k, v in read(ROOT/'result.json').items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed:
            with (ROOT/'failure.json').open('x') as f: json.dump({'type': type(error).__name__, 'message': str(error)}, f)
        raise
