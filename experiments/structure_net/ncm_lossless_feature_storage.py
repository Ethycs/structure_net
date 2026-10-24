"""Fixed16 TRAIN feature storage control; no model or task-accuracy claim."""
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import torch

from experiments.structure_net import ncm_fp32_feature_codec as codec
from experiments.structure_net import ncm_lm1b_native_capacity as capacity
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, training, q, ROOT = capacity.native, capacity.training, capacity.q, capacity.ROOT
HYPOTHESIS = 'ncm-lossless-FP32-source-feature-storage-control-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-lossless-feature-storage-plan.md')
TEST = Path('tests/structure_net/test_ncm_fp32_feature_codec.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_lossless_feature_storage.py')
DEST = ROOT / '20261024_lossless_FP32_feature_storage_control_v1'
QUOTAS = {0: 4, 1: 3, 2: 3, 3: 3, 4: 3}


def choose(rows):
    rows = [r for r in rows if r['target_origin'] == 'original_approved_target_unchanged']
    selected = []
    for band, count in QUOTAS.items():
        lo, hi = capacity.BANDS[band]
        options = sorted((r for r in rows if lo <= r['source_cl100k_tokens'] <= hi),
            key=lambda r: (text_hash(HYPOTHESIS + ':source:' + r['id']), r['id']))
        if len(options) < count:
            raise ValueError('complete fixed16 TRAIN feature source quotas required')
        selected.extend({**r, 'storage_control_band': band} for r in options[:count])
    return selected


def producing(selected, manifest):
    files = [Path(__file__), Path(codec.__file__), PLAN, TEST, AUDITOR, Path(capacity.__file__)]
    inputs = [training.DATA / 'cases.json', training.DATA / 'manifest.json']
    inputs += [Path(manifest[r['id']]['path']) for r in selected]
    p = native.vendor.freeze(DEST, 'fixed16_lossless_FP32_feature_storage', [*files, *inputs], {
        'hypothesis_id': HYPOTHESIS, 'CPU_only': True, 'TRAIN_only': True, 'source_cases': 16,
        'quotas': QUOTAS, 'feature_dtype': 'FP32 unchanged', 'compression': 'zlib1 interleaved vsbyte-shuffled',
        'bitwise_reconstruction_required': True, 'model_weights_loaded_or_changed': False,
        'model_accuracy_measured': False, 'model_speed_measured': False, 'task_accuracy': None,
        'canonical_firsts_changed': False, 'final_sources_opened_or_hashed': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (DEST / 'producer_source' / f'lossless_feature_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(DEST / 'producer.json', p)
    return p


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['layout']].append(row)
    return {layout: {'cases': len(values), 'raw_bytes': sum(v['raw_bytes'] for v in values),
        'stored_bytes': sum(v['stored_bytes'] for v in values),
        'stored_over_raw': sum(v['stored_bytes'] for v in values) / sum(v['raw_bytes'] for v in values),
        'bitwise_reconstruction': all(v['bitwise_reconstruction'] for v in values),
        'by_band': {str(b): {'raw_bytes': sum(v['raw_bytes'] for v in values if v['band'] == b),
            'stored_bytes': sum(v['stored_bytes'] for v in values if v['band'] == b)} for b in range(5)}}
        for layout, values in grouped.items()}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized() or DEST.exists():
        raise ValueError('fresh CPU-only lossless feature storage control required')
    selected = choose(q.read(training.DATA / 'cases.json'))
    manifest = q.read(training.DATA / 'manifest.json')
    p = producing(selected, manifest)
    (DEST / 'payloads').mkdir()
    rows = []
    for row in selected:
        entry = manifest[row['id']]
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('original source feature changed')
        value = torch.load(entry['path'], map_location='cpu', weights_only=True, mmap=True)
        features = value['features']
        if (value['id'] != row['id'] or value['source_sha256'] != row['source_sha256']
                or features.dtype != torch.float32 or features.shape != (entry['source_tokens'], 3072)):
            raise ValueError('complete original FP32 source feature identity differs')
        for layout in codec.LAYOUTS:
            start = time.perf_counter()
            header, payload = codec.encode(features, layout)
            encode_seconds = time.perf_counter() - start
            stem = DEST / 'payloads' / (row['id'] + '.' + layout)
            header_path, blob_path = stem.with_suffix(stem.suffix + '.json'), stem.with_suffix(stem.suffix + '.zlib')
            write_json(header_path, header)
            blob_path.write_bytes(payload)
            start = time.perf_counter()
            restored = codec.decode(q.read(header_path), blob_path.read_bytes())
            decode_seconds = time.perf_counter() - start
            exact = torch.equal(restored.view(torch.uint8), features.contiguous().view(torch.uint8))
            if not exact:
                raise ValueError('lossless FP32 feature byte reconstruction failed')
            rows.append({'id': row['id'], 'band': row['storage_control_band'], 'layout': layout,
                'input_path': entry['path'], 'input_sha256': entry['sha256'], 'source_sha256': row['source_sha256'],
                'source_positions': len(features), 'header_path': str(header_path), 'header_sha256': file_hash(header_path),
                'payload_path': str(blob_path), 'payload_sha256': file_hash(blob_path), 'raw_bytes': header['raw_bytes'],
                'stored_bytes': header_path.stat().st_size + blob_path.stat().st_size,
                'bitwise_reconstruction': exact, 'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds})
    write_json(DEST / 'measurements.json', rows)
    result = {'status': 'fixed16_FP32_feature_storage_roundtrip_completed_independent_audit_pending',
        'metrics': aggregate(rows), 'source_cases': 16, 'original_features_or_model_weights_changed': False,
        'task_accuracy': None, 'model_speed_measured': False, 'all_final_sources_closed': True}
    native.vendor.verify(p)
    write_json(DEST / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
