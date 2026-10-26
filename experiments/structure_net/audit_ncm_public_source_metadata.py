"""Exhaustive bounded CPU replay of original public source tokenization."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import resource
import time

from experiments.structure_net import ncm_public_source_metadata as producer

data, catalog = producer.data, producer.catalog
ROOTS = {s: data.ROOT/f'20261026_public_source_metadata_{s}_audit_v1' for s in producer.ROOTS}
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-source-metadata-audit-plan.md')
TEST = Path('tests/structure_net/test_ncm_public_source_metadata_audit.py')
SOURCE_FIELDS = ('id', 'source_id', 'group_id', 'source_component', 'source_sha256', 'source',
                 'spans', 'split', 'source_cl100k_tokens', 'source_truncated', 'public_provenance')


def read_record(stream, entry):
    stream.seek(entry['offset_bytes'])
    payload = stream.read(entry['length_bytes'])
    if len(payload) != entry['length_bytes'] or hashlib.sha256(payload).hexdigest() != entry['record_sha256']:
        raise ValueError('indexed record bytes differ')
    row = json.loads(payload)
    if row['id'] != entry['id']:
        raise ValueError('indexed record identity differs')
    return row


def verify_record(row, entry, pair, position, expected, forbidden):
    if (row != expected or forbidden & row.keys() or row['id'] != pair['id']
            or entry['catalog_position'] != position or row['source_truncated']
            or any(row[k] != pair[k] for k in SOURCE_FIELDS)
            or any(entry[k] != row[k] for k in ('id', 'source_sha256', 'source_component',
                'source_frame', 'alignment_sha256', 'native_prompt_sha256'))
            or entry['source_positions'] != len(row['source_ids'])
            or entry['native_prompt_positions'] != len(row['context_ids'])
            or not 1 <= len(row['source_ids']) <= row['source_frame'] <= 8192
            or len(row['alignment']) != len(row['source_ids'])):
        raise ValueError('source-only full metadata replay or binding differs')
    for values in row['alignment']:
        if (not values or any(type(i) is not int or not 0 <= i < len(row['context_ids'])
                or not math.isfinite(w) or w <= 0 for i, w in values)
                or abs(sum(w for _, w in values)-1) > 1e-6):
            raise ValueError('finite normalized complete source alignment required')


def tasks(path, entries, positions, batch_size=128):
    offset = count = 0
    batch = []
    with Path(path).open() as stream:
        for line, position in itertools.zip_longest(stream, positions):
            if line is None or position is None:
                raise ValueError('complete ordered source index required')
            entry = json.loads(line)
            if (entry['offset_bytes'] != offset or entry['length_bytes'] <= 0
                    or entry['catalog_position'] != position or entry['id'] != entries[position]['id']):
                raise ValueError('contiguous ordered byte index required')
            offset += entry['length_bytes']; count += 1
            batch.append((position, entry, entries[position]))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
    if count != len(positions) or offset != Path(path).with_name('metadata.jsonl').stat().st_size:
        raise ValueError('complete metadata payload and population required')


def initialize(stage, pilot):
    global _metadata, _tokens, _pairs, _saved, _pilot
    from experiments.structure_net import ncm_pixi
    ncm_pixi.configure()
    from experiments.structure_net import ncm_additional_source_metadata as metadata
    _metadata, _tokens = metadata, metadata.tokenizers()
    _pairs = (catalog.DEST/'training.jsonl').open('rb')
    _saved = (producer.ROOTS[stage]/'metadata.jsonl').open('rb')
    _pilot = pilot


def replay(batch):
    result = {'cases': 0, 'source_positions': 0, 'native_prompt_positions': 0,
              'max_source_positions': 0, 'max_native_prompt_positions': 0, 'pilot_repetitions_exact': 0}
    for position, entry, pair_entry in batch:
        pair, row = read_record(_pairs, pair_entry), read_record(_saved, entry)
        source = {key: pair[key] for key in SOURCE_FIELDS}
        expected = _metadata.prepare_one(source, *_tokens)
        verify_record(row, entry, pair, position, expected, _metadata.FORBIDDEN)
        if row['id'] in _pilot:
            if row != _pilot[row['id']]:
                raise ValueError('public pilot geometry differs')
            result['pilot_repetitions_exact'] += 1
        result['cases'] += 1
        for key in ('source_positions', 'native_prompt_positions'):
            result[key] += entry[key]
            if entry[key] > result['max_'+key]:
                result['max_'+key] = entry[key]
                result['max_'+key+'_id'] = row['id']
    result['worker_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return result


def bounded_results(pool, jobs, pending_limit=8):
    pending = deque()
    for job in jobs:
        pending.append(pool.submit(replay, job))
        if len(pending) == pending_limit:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def audit(stage):
    root, source = ROOTS[stage], producer.ROOTS[stage]
    if root.exists():
        raise ValueError('fresh append-only source metadata audit required')
    original = data.completed(source)
    if stage == 'full' and not data.completed(ROOTS['pilot'])['all_records_replayed']:
        raise ValueError('actual completed pilot audit required')
    proof = data.read(source/'producer.json')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    for i, (name, expected) in enumerate(sorted(proof['source_hashes'].items())):
        if data.sha(source/'producer_source'/f'{i}_{Path(name).name}') != expected:
            raise ValueError('producer source snapshot differs')
    training, environment, metadata, records = producer.ready()
    parity = metadata.parity(*metadata.tokenizers())
    if parity != data.read(source/'original_TRAIN_parity.json'):
        raise ValueError('original TRAIN geometry controls differ')
    positions = producer.pilot_positions(records.entries) if stage == 'pilot' else range(len(records))
    with (producer.ROOTS['pilot']/'metadata.jsonl').open() as stream:
        pilot = {r['id']: r for r in map(json.loads, stream)}
    paths = [p for p in source.iterdir() if p.is_file()]
    paths += [Path(__file__), PLAN, TEST, Path('pyproject.toml'), Path('pixi.lock')]
    if stage == 'full':
        paths += [ROOTS['pilot']/name for name in ('result.json', 'actual_tool_receipt.json')]
    binding = {str(p): data.sha(p) for p in paths}
    root.mkdir()
    data.write(root/'producer.json', {'input_and_source_hashes': binding,
        'source_producer': str(source/'producer.json'), 'stage': stage, 'workers': 1 if stage == 'pilot' else 4,
        'records_per_task': 128, 'maximum_pending_tasks': 8, 'environment': environment})
    started = time.monotonic()
    totals = {'cases': 0, 'source_positions': 0, 'native_prompt_positions': 0,
        'max_source_positions': 0, 'max_native_prompt_positions': 0, 'pilot_repetitions_exact': 0,
        'worker_peak_RSS_bytes': 0}
    with ProcessPoolExecutor(max_workers=1 if stage == 'pilot' else 4,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(stage, pilot)) as pool:
        for batch in bounded_results(pool, tasks(source/'index.jsonl', records.entries, positions)):
            for key in ('cases', 'source_positions', 'native_prompt_positions', 'pilot_repetitions_exact'):
                totals[key] += batch[key]
            for key in ('max_source_positions', 'max_native_prompt_positions'):
                if batch[key] > totals[key]:
                    totals[key] = batch[key]; totals[key+'_id'] = batch[key+'_id']
            totals['worker_peak_RSS_bytes'] = max(totals['worker_peak_RSS_bytes'], batch['worker_peak_RSS_bytes'])
            if totals['cases'] % 4096 == 0 or totals['cases'] == len(positions):
                print(json.dumps({'replayed': totals['cases'], 'total': len(positions),
                                  'elapsed_seconds': time.monotonic()-started}), flush=True)
    records.close()
    for key in ('cases', 'source_positions', 'native_prompt_positions', 'max_source_positions', 'max_native_prompt_positions'):
        if totals[key] != original[key]:
            raise ValueError('full metadata population/counts differ')
    if totals['pilot_repetitions_exact'] != 9:
        raise ValueError('all nine public pilot controls required')
    data.verify(binding); data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    value = {'schema': 'nal.ncm-public-source-metadata-audit.v1', 'hypothesis_id': producer.HYPOTHESIS,
        'status': 'all_public_'+stage+'_source_metadata_replayed', 'stage': stage,
        **totals, 'all_records_replayed': True, 'all_tokenization_and_alignment_recomputed': True,
        'independent_tokenization_implementation': False, 'original_TRAIN_geometry_equal': True,
        'estimated_full_FP32_feature_bytes': totals['source_positions']*3072*4,
        'elapsed_seconds': time.monotonic()-started,
        'parent_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        **producer.BOUNDARIES, 'full_metadata_audited': stage == 'full',
        'artifacts': {**binding, str(root/'producer.json'): data.sha(root/'producer.json')}}
    data.write(root/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=tuple(ROOTS))
    stage = parser.parse_args().stage
    existed = ROOTS[stage].exists()
    try:
        audit(stage)
    except Exception as error:
        if not existed and ROOTS[stage].exists():
            data.write(ROOTS[stage]/'failure.json', {'exception': type(error).__name__, 'message': str(error)})
        raise
