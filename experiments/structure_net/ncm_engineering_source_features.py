"""Extract unchanged frozen-body features for the verified engineering sources."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from experiments.structure_net import ncm_engineering_source_metadata as metadata
from experiments.structure_net import ncm_additional_source_features as original
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, ROOT = metadata.native, metadata.ROOT
MODULE = 'experiments.structure_net.ncm_engineering_source_features'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-engineering-source-features-plan.md')
WORKFLOW, PILOT, CACHE = [ROOT / ('20261025_engineering_source_features_' + name + '_v1')
    for name in ('workflow', 'pilot', 'cache')]


def inputs():
    if metadata.data.old.reservation.read(metadata.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual complete engineering source geometry required')
    native.vendor.verify(original.q.read(metadata.DEST / 'producer.json'))
    state = original.q.read(metadata.DEST / 'result.json')
    index = original.q.read(metadata.DEST / 'index.json')
    if (state['cases'] != 888 or state['partitions'] != {'train': 732, 'test': 156}
            or state['index_sha256'] != file_hash(metadata.DEST / 'index.json')
            or len(index) != 888 or len({r['id'] for r in index}) != 888):
        raise ValueError('all unchanged source-only engineering inputs required')
    return index


def probes(index):
    result = [index[0]['id']]
    for field in ('source_positions', 'native_prompt_positions'):
        result.append(min(index, key=lambda r: (-r[field], r['id']))['id'])
    return list(dict.fromkeys(result))


def producing(root, stage, rows, extra=()):
    files = [Path(__file__), PLAN, Path(metadata.__file__), metadata.PLAN, Path(original.__file__),
        Path(original.codec.__file__), Path(original.base.__file__), Path(original.base.source.__file__)]
    files += [Path(original.base.source.module_hash.__code__.co_filename)] if hasattr(original.base.source, 'module_hash') else []
    paths = [metadata.DEST / name for name in ('producer.json', 'result.json', 'index.json',
        'actual_tool_receipt.json', 'original_training_geometry_parity.json')]
    paths += [original.base.MODEL_PATH / name for name in ('config.json', 'model.safetensors.index.json',
        'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors')]
    paths += [Path(row['path']) for row in rows]
    paths += list(extra)
    root.mkdir()
    (root / 'producer_source').mkdir()
    for i, path in enumerate(dict.fromkeys(files)):
        (root / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    p = {'schema': 'nal.ncm-engineering-source-features.v1', 'stage': stage,
        'source_hashes': {str(path): file_hash(path) for path in files},
        'input_hashes': {str(path): file_hash(path) for path in paths},
        'source_scope': 'verified engineering sources; TRAIN and engineering heldout features only',
        'config': {'cases': len(rows), 'physical_GPU': 0, 'parent_layers': 28,
            'original_body_sha256': original.base.EXPECTED_BODY, 'body_precision': 'BF16unchanged',
            'all_parent_MLPs_frozen': True, 'aligned_features': 'FP32unchanged',
            'feature_storage': 'lossless byte-shuffled zlib1', 'full_source_retained': True,
            'source_only': True, 'targets_or_codes_in_parent_arguments': False,
            'date_string': '23 Oct 2026', 'TF32': False, 'peak_limit_bytes': 8_600_000_000,
            'minimum_free_disk_bytes': 8 * 1024**3, 'task_accuracy': None,
            'model_training': False, 'speed_tested': False, 'sealed_final_sources_opened_or_hashed': False}}
    write_json(root / 'producer.json', p)
    return p


def cache(pilot=False):
    native.vendor.corpus.quality.dependencies()
    original.metadata.old_training.require_gpu()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('same physical GPU0 required for exact original-body features')
    index = inputs()
    retained, extra = {}, []
    if not pilot:
        native.vendor.terminal(PILOT)
        retained = original.q.read(PILOT / 'manifest.json')
        extra = [PILOT / n for n in ('producer.json', 'result.json', 'manifest.json', 'worker_terminal_receipt.json')]
    chosen = [r for r in index if r['id'] in probes(index)] if pilot else index
    root = PILOT if pilot else CACHE
    p = producing(root, 'real_geometry_parity_and_lossless_pilot' if pilot else 'all888_frozen_engineering_source_features', chosen, extra)
    (root / 'features').mkdir()
    (root / 'entries').mkdir()
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(original.base.MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    if parent.config.num_hidden_layers != 28 or original.base.module_hash(parent.model) != original.base.EXPECTED_BODY:
        raise ValueError('unchanged original 28-layer frozen body required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    parity = original.reference_parity(parent)
    manifest = {}
    for i, entry in enumerate(chosen, 1):
        if shutil.disk_usage(ROOT).free < 8 * 1024**3:
            raise ValueError('8GiB disk floor reached; no truncation or precision fallback')
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('source metadata changed')
        row = original.q.read(entry['path'])
        if metadata.geometry.FORBIDDEN & row.keys() or row['source_truncated']:
            raise ValueError('complete source-only feature input required')
        if row['id'] in retained:
            value = retained[row['id']]
            original.load_feature(value, row)
        else:
            features = original.extract(parent, row)
            if pilot and not torch.equal(features.view(torch.uint8), original.extract(parent, row).contiguous().view(torch.uint8)):
                raise ValueError('frozen feature extraction must repeat exactly')
            header, payload = original.codec.encode(features, 'byte_shuffled')
            header['binding'] = original.binding(row)
            header_path = root / 'features' / (row['id'] + '.json')
            payload_path = root / 'features' / (row['id'] + '.zlib')
            payload_path.write_bytes(payload)
            write_json(header_path, header)
            value = {'header_path': str(header_path), 'header_sha256': file_hash(header_path),
                'payload_path': str(payload_path), 'payload_sha256': file_hash(payload_path),
                'source_tokens': len(features), 'raw_bytes': features.numel() * 4,
                'stored_bytes': header_path.stat().st_size + payload_path.stat().st_size}
            restored = original.load_feature(value, row)
            if not torch.equal(features.view(torch.uint8), restored.view(torch.uint8)):
                raise ValueError('lossless storage changed original FP32 feature bytes')
        manifest[row['id']] = value
        write_json(root / 'entries' / (row['id'] + '.json'), value)
        if pilot or i % 16 == 0 or i == len(chosen):
            write_json(root / 'progress.json', {'cached': i, 'cases': len(chosen), 'FP32_bytes_preserved': True})
            print(json.dumps({'cached_engineering_source_features': i, 'cases': len(chosen)}), flush=True)
    if (original.base.module_hash(parent.model) != original.base.EXPECTED_BODY
            or any(v.grad is not None for v in parent.parameters())):
        raise ValueError('original parent received gradients or changed')
    peak = torch.cuda.max_memory_allocated()
    if peak > 8_600_000_000:
        raise ValueError('feature extraction exceeds original GPU capacity limit')
    write_json(root / 'reference_parity.json', parity)
    write_json(root / 'manifest.json', manifest)
    native.vendor.verify(p)
    result = {'status': 'engineering_source_feature_parity_pilot_passed' if pilot else 'all888_engineering_frozen_body_features_cached_losslessly',
        'cases': len(manifest), 'manifest_sha256': file_hash(root / 'manifest.json'),
        'original_body_hash_unchanged': original.base.EXPECTED_BODY, 'all_FP32_byte_roundtrips_exact': True,
        'raw_bytes': sum(r['raw_bytes'] for r in manifest.values()),
        'stored_bytes': sum(r['stored_bytes'] for r in manifest.values()), 'peak_allocated_bytes': peak,
        'original_MLPs_updated': False, 'model_training': False, 'task_accuracy': None,
        'sealed_final_sources_opened_or_hashed': False}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)


def workflow():
    native.vendor.corpus.quality.dependencies()
    index = inputs()
    if any(root.exists() for root in (WORKFLOW, PILOT, CACHE)):
        raise ValueError('fresh append-only engineering feature roots required')
    p = producing(WORKFLOW, 'supervise_engineering_frozen_source_features', index)
    for stage, root in (('pilot', PILOT), ('cache', CACHE)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'child_pid': child.pid, 'observer_pid': os.getpid(),
            'observer': 'subprocess.wait', 'command': command}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        if code:
            raise RuntimeError('engineering frozen feature stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'engineering_source_feature_workflow_completed', 'cases': 888})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('workflow', 'pilot', 'cache'))
    args = parser.parse_args()
    {'workflow': workflow, 'pilot': lambda: cache(True), 'cache': cache}[args.stage]()
