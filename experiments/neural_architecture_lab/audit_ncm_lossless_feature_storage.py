"""Decode every saved lossless feature payload and compare actual source bytes."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_lossless_feature_storage as run

DEST = run.ROOT / '20261024_lossless_FP32_feature_storage_audit_v1'


def main():
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.DEST / 'actual_tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('fresh independent CPU lossless storage audit and actual benchmark completion required')
    run.native.vendor.verify(run.q.read(run.DEST / 'producer.json'))
    cases = run.choose(run.q.read(run.training.DATA / 'cases.json'))
    selected = {r['id']: r for r in cases}
    rows = run.q.read(run.DEST / 'measurements.json')
    if len(rows) != 32 or {(r['id'], r['layout']) for r in rows} != {(r['id'], l) for r in cases for l in run.codec.LAYOUTS}:
        raise ValueError('all16original TRAIN sources and both lossless layouts required')
    for row in rows:
        for path, key in (('input_path', 'input_sha256'), ('header_path', 'header_sha256'), ('payload_path', 'payload_sha256')):
            if run.file_hash(row[path]) != row[key]:
                raise ValueError('actual original feature/header/payload hash differs')
        value = torch.load(row['input_path'], map_location='cpu', weights_only=True, mmap=True)
        original = value['features']
        header = run.q.read(row['header_path'])
        restored = run.codec.decode(header, Path(row['payload_path']).read_bytes())
        if (value['source_sha256'] != selected[row['id']]['source_sha256']
                or header['layout'] != row['layout'] or row['band'] != selected[row['id']]['storage_control_band']
                or not torch.equal(restored.view(torch.uint8), original.contiguous().view(torch.uint8))
                or not row['bitwise_reconstruction'] or row['source_positions'] != len(original)
                or row['raw_bytes'] != original.numel() * 4
                or row['stored_bytes'] != Path(row['header_path']).stat().st_size + Path(row['payload_path']).stat().st_size):
            raise ValueError('complete actual source feature bytes or storage accounting differs')
    metrics = run.aggregate(rows)
    if run.q.read(run.DEST / 'result.json')['metrics'] != metrics:
        raise ValueError('lossless storage aggregates differ')
    artifacts = {str(Path(__file__)): run.file_hash(Path(__file__))}
    for path in run.DEST.rglob('*'):
        if path.is_file() and 'producer_source' not in path.parts:
            artifacts[str(path)] = run.file_hash(path)
    result = {'status': 'all32_payloads_bitwise_FP32_reconstruction_and_storage_replayed', 'source_cases': 16,
        'metrics': metrics, 'original_features_or_model_weights_changed': False, 'task_accuracy': None,
        'model_speed_measured': False, 'all_final_sources_closed': True, 'artifacts': artifacts}
    DEST.mkdir()
    run.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
