"""Extract full original-body FP32 features once into bounded lossless disk shards."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import torch

from experiments.structure_net import ncm_public_source_metadata as metadata
from experiments.structure_net import audit_ncm_public_source_metadata as audited
from experiments.structure_net import ncm_public_feature_pilot as prior
from experiments.structure_net import ncm_packed_feature_store as store

data, original = metadata.data, prior.original
ROOTS = {s: data.ROOT/f'20261026_public_feature_store_{s}_v1' for s in ('pilot', 'full')}
STORAGE = Path('/data2/structure_net/ncm_public_features_20261026_v1')
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-feature-store-plan.md')
TESTS = [Path('tests/structure_net/test_ncm_packed_feature_store.py'),
         Path('tests/structure_net/test_ncm_public_feature_store.py')]
HYPOTHESIS = 'ncm-public-feature-store-v1'
BOUNDARIES = {'model_training': False, 'new_judge_requests': 0, 'task_accuracy': None,
              'final_test_opened': False, 'features_admitted_for_training': False}


def selected_positions(entries, original_ids):
    lookup = {r['id']: i for i, r in enumerate(entries)}
    if len(lookup) != len(entries) or len(original_ids) != 9:
        raise ValueError('complete unique index and original nine probes required')
    selected = [lookup[k] for k in original_ids]
    for key in ('source_positions', 'native_prompt_positions'):
        selected += sorted(range(len(entries)), key=lambda i: (-entries[i][key], entries[i]['id']))[:4]
    return list(dict.fromkeys(selected))


def ready():
    from experiments.structure_net import ncm_pixi
    _, environment = ncm_pixi.configure()
    state = data.completed(audited.ROOTS['full'])
    if not state['full_metadata_audited'] or not state['all_records_replayed'] or state['cases'] != 162353:
        raise ValueError('completed exhaustive full source metadata audit required')
    source = metadata.ROOTS['full']
    proof = data.read(source/'producer.json')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    with (source/'index.jsonl').open() as stream:
        entries = list(map(json.loads, stream))
    expected = data.read(source/'result.json')['artifacts'][str(source/'metadata.jsonl')]
    records = metadata.indexed.IndexedJSONL(source/'metadata.jsonl', entries, expected)
    passed = data.completed(prior.ROOT)
    if passed['cases'] != 9 or not passed['all_repeated_extractions_exact']:
        raise ValueError('original nine-source feature pilot required')
    old_manifest = data.read(prior.ROOT/'manifest.json')
    old_hashes = {k: data.read(v['header_path'])['raw_sha256'] for k, v in old_manifest.items()}
    positions = selected_positions(entries, list(old_manifest))
    return environment, records, positions, old_hashes


def encode_verified(features, binding):
    header, payload = store.codec.encode(features, 'byte_shuffled')
    restored = store.codec.decode(header, payload)
    if not torch.equal(features.contiguous().view(torch.uint8), restored.contiguous().view(torch.uint8)):
        raise ValueError('encoded public feature bytes differ')
    header['binding'] = binding
    return header, payload


def freeze(root, stage, environment, selection):
    files = {Path(__file__).resolve(), PLAN.resolve(), *[p.resolve() for p in TESTS]}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):
                files.add(p)
    paths = [audited.ROOTS['full']/name for name in ('result.json', 'actual_tool_receipt.json')]
    paths += [metadata.ROOTS['full']/name for name in ('result.json', 'actual_tool_receipt.json', 'index.jsonl')]
    paths += [prior.ROOT/name for name in ('result.json', 'actual_tool_receipt.json', 'manifest.json')]
    paths += [p for p in original.base.MODEL_PATH.iterdir() if p.is_file()]
    paths += [Path('pyproject.toml'), Path('pixi.lock')]
    if stage == 'full':
        paths += [ROOTS['pilot']/name for name in ('result.json', 'actual_tool_receipt.json', 'audit_result.json',
                                                 'audit_actual_tool_receipt.json', 'raw_hashes.json')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-public-feature-store.v1', 'hypothesis_id': HYPOTHESIS,
        'stage': stage, 'environment': environment, 'selection': selection,
        'config': {'physical_GPU': 0, 'body_precision': 'BF16', 'features': 'complete_FP32',
            'TF32': False, 'encoder_threads': 4, 'pending_limit': 8, 'shard_bytes': 256*1024**2,
            'minimum_free_disk_bytes': 100*1024**3, 'storage': str(STORAGE/stage),
            'maximum_GPU_allocated_bytes': 8_600_000_000, **BOUNDARIES},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in paths}}
    data.write(root/'producer.json', proof)
    return proof


def run(stage):
    root, storage = ROOTS[stage], STORAGE/stage
    if root.exists() or storage.exists():
        raise ValueError('fresh evidence and feature storage roots required')
    environment, records, pilot_positions, expected_hashes = ready()
    original.metadata.old_training.require_gpu()
    if stage == 'full':
        pilot = data.completed(ROOTS['pilot'])
        receipt = data.read(ROOTS['pilot']/'audit_actual_tool_receipt.json')
        data.verify(receipt['artifacts'])
        if receipt['actual_tool_exit_code'] != 0 or not data.read(ROOTS['pilot']/'audit_result.json')['all_disk_records_replayed']:
            raise ValueError('actual passed pilot disk replay required')
        if pilot['status'] != 'public_feature_store_pilot_completed':
            raise ValueError('actual passed extraction/concurrency pilot required')
        expected_hashes.update(data.read(ROOTS['pilot']/'raw_hashes.json'))
    positions = pilot_positions if stage == 'pilot' else range(len(records))
    selected = [{'catalog_position': p, **{k: records.entries[p][k] for k in
        ('id', 'source_positions', 'native_prompt_positions')}} for p in pilot_positions]
    STORAGE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(STORAGE).free < 100*1024**3:
        raise ValueError('feature storage requires 100GiB free headroom')
    proof = freeze(root, stage, environment, selected)
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(original.base.MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    if parent.config.num_hidden_layers != 28 or original.base.module_hash(parent.model) != original.base.EXPECTED_BODY:
        raise ValueError('original unchanged 28-layer Llama body required')
    torch.cuda.reset_peak_memory_stats()
    data.write(root/'original_TRAIN_parity.json', original.reference_parity(parent))
    writer = store.Writer(storage)
    pending = deque()
    counters = {'cases': 0, 'source_positions': 0, 'raw_bytes': 0, 'payload_bytes': 0,
                'prior_raw_hash_controls': 0, 'repeated_extractions': 0}
    raw_hashes = {}
    started = time.monotonic()

    def consume():
        entry, future, extract_seconds = pending.popleft()
        header, payload = future.result()
        identity = entry['id']
        if identity in expected_hashes:
            if header['raw_sha256'] != expected_hashes[identity]:
                raise ValueError('prior public feature bytes differ')
            counters['prior_raw_hash_controls'] += 1
        writer.append(header, payload, metadata_record_sha256=entry['record_sha256'])
        if stage == 'pilot':
            raw_hashes[identity] = header['raw_sha256']
        counters['cases'] += 1; counters['source_positions'] += entry['source_positions']
        counters['raw_bytes'] += header['raw_bytes']; counters['payload_bytes'] += len(payload)
        if stage == 'pilot' or counters['cases'] % 128 == 0:
            value = {**counters, 'total': len(positions), 'elapsed_seconds': time.monotonic()-started,
                'last_extraction_seconds': extract_seconds, 'last_id': identity,
                'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), 'pending_matrices': len(pending)}
            data.write(root/'progress.json', value)
            print(json.dumps(value), flush=True)
        if counters['cases'] % 1024 == 0 and shutil.disk_usage(STORAGE).free < 100*1024**3:
            raise ValueError('feature disk headroom exhausted')

    with ThreadPoolExecutor(max_workers=4) as pool, torch.no_grad():
        for position in positions:
            row = records[position]
            if original.metadata.FORBIDDEN & row.keys():
                raise ValueError('target fields in feature extraction input')
            tick = time.perf_counter()
            features = original.extract(parent, row)
            elapsed = time.perf_counter()-tick
            if stage == 'pilot':
                again = original.extract(parent, row)
                if not torch.equal(features.view(torch.uint8), again.view(torch.uint8)):
                    raise ValueError('repeated original feature extraction differs')
                counters['repeated_extractions'] += 1
                del again
            if torch.cuda.max_memory_allocated() > 8_600_000_000:
                raise ValueError('original body allocation budget exceeded')
            pending.append((records.entries[position], pool.submit(encode_verified, features, original.binding(row)), elapsed))
            del features
            if len(pending) == 8:
                consume()
        while pending:
            consume()
    shards = writer.close(); records.close()
    if counters['cases'] != len(positions) or counters['prior_raw_hash_controls'] != len(expected_hashes):
        raise ValueError('complete population and all prior raw hash controls required')
    if original.base.module_hash(parent.model) != original.base.EXPECTED_BODY or any(p.grad is not None for p in parent.parameters()):
        raise ValueError('frozen original body changed or received gradients')
    data.write(root/'shards.json', shards)
    data.write(root/'raw_hashes.json', raw_hashes)
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    artifacts = {str(p): data.sha(p) for p in root.iterdir() if p.is_file()}
    artifacts[str(storage/'index.jsonl')] = data.sha(storage/'index.jsonl')
    artifacts.update({r['path']: r['sha256'] for r in shards})
    value = {'status': 'public_feature_store_'+stage+'_completed', 'hypothesis_id': HYPOTHESIS,
        'stage': stage, **counters, 'shards': len(shards), 'storage': str(storage),
        'body_sha256': original.base.EXPECTED_BODY, 'all_encoded_roundtrips_exact': True,
        'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'elapsed_seconds': time.monotonic()-started, 'artifacts': artifacts, **BOUNDARIES}
    data.write(root/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


def audit(stage):
    root = ROOTS[stage]
    if (root/'audit_result.json').exists():
        raise ValueError('fresh disk replay required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only disk replay required')
    original_result = data.completed(root)
    proof = data.read(root/'producer.json')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    _, records, selected, _ = ready()
    positions = selected if stage == 'pilot' else range(len(records))
    counters = {'cases': 0, 'source_positions': 0, 'raw_bytes': 0, 'payload_bytes': 0}
    coverage = {}
    pending = deque()
    started = time.monotonic()

    def replay(entry, row, expected):
        features = store.load(entry, original.binding(row), source_positions=len(row['source_ids']),
                              metadata_record_sha256=expected)
        return len(features), features.numel()*4, entry['length_bytes']

    def consume():
        positions_count, raw, payload = pending.popleft().result()
        counters['cases'] += 1; counters['source_positions'] += positions_count
        counters['raw_bytes'] += raw; counters['payload_bytes'] += payload
        if counters['cases'] % 1024 == 0:
            print(json.dumps({'disk_records_replayed': counters['cases'], 'total': len(positions),
                              'elapsed_seconds': time.monotonic()-started}), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool, (STORAGE/stage/'index.jsonl').open() as stream:
        for position in positions:
            entry = json.loads(stream.readline())
            row = records[position]
            if entry['offset_bytes'] != coverage.get(entry['path'], 0):
                raise ValueError('complete contiguous shard coverage required')
            coverage[entry['path']] = entry['offset_bytes']+entry['length_bytes']
            pending.append(pool.submit(replay, entry, row, records.entries[position]['record_sha256']))
            if len(pending) == 8:
                consume()
        if stream.read(1):
            raise ValueError('unaccounted disk feature records')
        while pending:
            consume()
    shards = data.read(root/'shards.json')
    if coverage != {r['path']: r['bytes'] for r in shards} or any(counters[k] != original_result[k] for k in counters):
        raise ValueError('complete shard/feature population accounting differs')
    records.close()
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    value = {'status': 'public_feature_store_'+stage+'_disk_replay_passed', **counters,
        'all_disk_records_replayed': True, 'all_shards_completely_covered': True,
        'independent_GPU_reexecution': False, 'elapsed_seconds': time.monotonic()-started, **BOUNDARIES,
        'artifacts': {str(root/'result.json'): data.sha(root/'result.json'),
                      str(root/'producer.json'): data.sha(root/'producer.json')}}
    data.write(root/'audit_result.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('extract', 'audit'))
    parser.add_argument('stage', choices=tuple(ROOTS))
    args = parser.parse_args()
    existed = ROOTS[args.stage].exists()
    try:
        (run if args.command == 'extract' else audit)(args.stage)
    except Exception as error:
        if args.command == 'extract' and not existed and ROOTS[args.stage].exists():
            data.write(ROOTS[args.stage]/'failure.json', {'exception': type(error).__name__, 'message': str(error), **BOUNDARIES})
        raise
