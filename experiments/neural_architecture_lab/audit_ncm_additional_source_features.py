"""CPU replay of every full-precision source cache payload, binding and manifest."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_additional_source_features as run

DEST = run.ROOT / '20261024_additional10000_source_features_audit_v1'


def main():
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('fresh CPU feature audit and actual completed whole extraction required')
    index = run.require_readiness()
    for root in (run.PILOT, run.FEATURES):
        run.native.vendor.terminal(root)
    manifest = run.q.read(run.FEATURES / 'manifest.json')
    pilot = run.q.read(run.PILOT / 'manifest.json')
    if set(manifest) != {r['id'] for r in index} or len(manifest) != 10000 or set(pilot) != set(run.probes(index)):
        raise ValueError('every original10000source and all metadata pilot cases required')
    raw_bytes, stored_bytes = 0, 0
    for i, source in enumerate(index, 1):
        if run.file_hash(source['path']) != source['sha256']:
            raise ValueError('source metadata changed')
        row = run.q.read(source['path'])
        entry = manifest[row['id']]
        features = run.load_feature(entry, row)
        if (entry['raw_bytes'] != features.numel() * 4
                or entry['stored_bytes'] != Path(entry['header_path']).stat().st_size + Path(entry['payload_path']).stat().st_size
                or run.q.read(run.FEATURES / 'entries' / (row['id'] + '.json')) != entry):
            raise ValueError('raw/stored bytes or source manifest entry differ')
        if row['id'] in pilot and entry != pilot[row['id']]:
            raise ValueError('first feature pilot must be reused exactly')
        raw_bytes += entry['raw_bytes']
        stored_bytes += entry['stored_bytes']
        if i % 128 == 0:
            print(json.dumps({'audited_source_features': i, 'cases': 10000}), flush=True)
    for root, count in ((run.PILOT, len(pilot)), (run.FEATURES, 10000)):
        result = run.q.read(root / 'result.json')
        if (result['cases'] != count or result['original_body_hash_unchanged'] != run.base.EXPECTED_BODY
                or not result['all_source_positions_retained'] or not result['all_FP32_byte_roundtrips_exact']
                or result['peak_allocated_bytes'] > 8_600_000_000
                or result['manifest_sha256'] != run.file_hash(root / 'manifest.json')):
            raise ValueError('actual original-body extraction/shape/storage/memory state differs')
    state = run.q.read(run.FEATURES / 'result.json')
    if state['raw_bytes'] != raw_bytes or state['stored_bytes'] != stored_bytes:
        raise ValueError('all10000source byte totals differ')
    artifacts = {str(Path(__file__)): run.file_hash(Path(__file__))}
    for root in (run.WORKFLOW, run.PILOT, run.FEATURES):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)] = run.file_hash(path)
    result = {'status': 'all10000_lossless_FP32_source_cache_payloads_and_bindings_replayed', 'cases': 10000,
        'raw_bytes': raw_bytes, 'stored_bytes': stored_bytes, 'FP32_bytes_preserved': True,
        'original_body_hash_unchanged': run.base.EXPECTED_BODY, 'task_accuracy': None,
        'model_speed_measured': False, 'all_final_sources_closed': True, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
