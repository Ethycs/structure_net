"""Read-only systems diagnostic after native655 failed all128 DEV summaries."""
import argparse
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_additional_source_features as features

metadata, development, q = features.metadata, features.development, features.q
base, native = features.base, features.native
ROOT = features.ROOT / '20261024_native655_feature_pipeline_check_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-native-feature-pipeline-check.md')
TEST = Path('tests/structure_net/test_ncm_native_feature_pipeline_check.py')
FIELDS = ('id', 'source_id', 'group_id', 'source_sha256', 'source', 'spans', 'split', 'source_cl100k_tokens')


def populations():
    features.require_readiness()
    train = q.read(development.training.DATA / 'cases.json')
    train_manifest = q.read(development.training.DATA / 'manifest.json')
    parts = q.read(metadata.pinned.METADATA / 'cases.json')
    manifest = q.read(development.FEATURES / 'manifest.json')
    return [('train', train, train_manifest)] + [(name, parts[name], manifest) for name in development.COHORTS]


def select(rows, manifest):
    longest = min(rows[1:], key=lambda r: (-manifest[r['id']]['source_tokens'], r['id']))
    return [rows[0], longest]


def stats(x):
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 3072 or not torch.isfinite(x).all():
        raise ValueError('complete finite FP32 features required')
    return {'source_positions': len(x), 'mean': float(x.mean()), 'RMS': float(x.square().mean().sqrt()),
        'centered_RMS': float((x - x.mean(0)).square().mean().sqrt()),
        'mean_position_L2': float(x.norm(dim=-1).mean()),
        'adjacent_duplicate_fraction': float((x[1:] == x[:-1]).all(-1).float().mean()) if len(x) > 1 else 0.}


def compare(before, after):
    if before.shape != after.shape or before.dtype != torch.float32 or after.dtype != torch.float32:
        raise ValueError('same complete FP32 feature geometry required')
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('nonfinite pipeline features')
    delta = after - before
    return {'FP32_bytes_exact': torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8)),
        'different_values': int(torch.count_nonzero(delta)), 'max_absolute_error': float(delta.abs().max()),
        'RMS_error': float(delta.square().mean().sqrt())}


def cached(row, entry):
    if features.file_hash(entry['path']) != entry['sha256']:
        raise ValueError('original source cache changed')
    value = torch.load(entry['path'], map_location='cpu', weights_only=True, mmap=True)
    for key in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256'):
        if value[key] != row[key]:
            raise ValueError('cached source binding differs:' + key)
    if value['features'].shape != (entry['source_tokens'], 3072):
        raise ValueError('cached full-source shape differs')
    return value['features']


def freeze(root, stage, extra=()):
    paths = [Path(__file__), PLAN, TEST, Path(features.__file__), Path(metadata.__file__),
        Path(metadata.pinned.__file__), Path(metadata.development.__file__), Path(base.__file__), Path(base.source.__file__),
        development.training.DATA / 'cases.json', development.training.DATA / 'manifest.json',
        metadata.pinned.METADATA / 'cases.json', development.FEATURES / 'manifest.json',
        development.RESULTS / 'campaign_results.json', development.RESULTS / 'backend_exact_readback.json',
        development.WORKFLOW / 'tool_terminal_receipt.json', *extra]
    p = native.vendor.freeze(root, stage, paths, {'systems_readiness_only_not_quality_evidence': True,
        'prior_native655_DEV_secondary_passes': 0, 'prior_native655_DEV_cases': 128,
        'unchanged_checkpoint': True, 'new_semantic_judgments': False, 'final_sources_opened_or_hashed': False,
        'same_original_body_BF16_and_FP32_character_alignment': True, 'TF32': False,
        'selection': 'first plus longest distinct source in TRAIN, shortDEV and longDEV',
        'all_source_positions_retained': True, 'task_accuracy': None, 'speed_measured': False})
    for i, path in enumerate(paths[:9]):
        p['source_hashes'][str(path)] = features.file_hash(path)
        (root / 'producer_source' / f'pipeline_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    features.write_json(root / 'producer.json', p)
    return p


def cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only source metadata/cache audit required')
    parts = populations()
    p = freeze(ROOT / 'cpu', 'all783_source_metadata_and_cached_feature_statistics')
    tokenizers = metadata.tokenizers()
    records = []
    for cohort, rows, manifest in parts:
        for row in rows:
            rebuilt = metadata.prepare_one({k: row[k] for k in FIELDS}, *tokenizers)
            differences = [k for k in metadata.GEOMETRY if row[k] != rebuilt[k]]
            entry = manifest[row['id']]
            records.append({'id': row['id'], 'cohort': cohort, 'metadata_differences': differences,
                'feature_path': entry['path'], 'feature_sha256': entry['sha256'], **stats(cached(row, entry))})
        print(json.dumps({'checked_cohort': cohort, 'source_cases': len(rows)}), flush=True)
    native.vendor.verify(p)
    features.write_json(ROOT / 'cpu/cases.json', records)
    features.write_json(ROOT / 'cpu/result.json', {'status': 'source_pipeline_CPU_diagnostic_completed',
        'source_cases': len(records), 'metadata_mismatches': sum(bool(r['metadata_differences']) for r in records),
        'cases_sha256': features.file_hash(ROOT / 'cpu/cases.json'), 'task_accuracy': None})


@torch.no_grad()
def gpu():
    features.metadata.old_training.require_gpu()
    if q.read(ROOT / 'cpu/actual_tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('actual completed CPU diagnostic required')
    parts = populations()
    paths = [base.MODEL_PATH / name for name in ('config.json', 'model.safetensors.index.json',
        'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors')]
    p = freeze(ROOT / 'gpu', 'recompute_first_longest_TRAIN_shortDEV_longDEV_features',
        [ROOT / 'cpu/result.json', ROOT / 'cpu/cases.json', *paths])
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(base.MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    if base.module_hash(parent.model) != base.EXPECTED_BODY or parent.config.num_hidden_layers != 28:
        raise ValueError('original frozen28layer body differs')
    torch.cuda.reset_peak_memory_stats()
    records = []
    for cohort, rows, manifest in parts:
        for row in select(rows, manifest):
            entry = manifest[row['id']]
            after = features.extract(parent, row)
            path = ROOT / 'gpu' / (row['id'] + '.pt')
            torch.save({'features': after}, path)
            record = {'id': row['id'], 'cohort': cohort, 'reference_path': entry['path'],
                'reference_sha256': entry['sha256'], 'recomputed_path': str(path),
                'recomputed_sha256': features.file_hash(path), **compare(cached(row, entry), after)}
            records.append(record)
            features.write_json(ROOT / 'gpu/progress.json', records)
            print(json.dumps(record), flush=True)
    peak = torch.cuda.max_memory_allocated()
    if peak > 8_600_000_000 or base.module_hash(parent.model) != base.EXPECTED_BODY:
        raise ValueError('original body or8.6GBmemory contract differs')
    if any(v.grad is not None for v in parent.parameters()):
        raise ValueError('frozen body received gradients')
    native.vendor.verify(p)
    features.write_json(ROOT / 'gpu/result.json', {'status': 'source_pipeline_GPU_diagnostic_completed',
        'cases': records, 'all_FP32_bytes_exact': all(r['FP32_bytes_exact'] for r in records),
        'original_body_sha256_unchanged': base.EXPECTED_BODY, 'peak_allocated_bytes': peak, 'task_accuracy': None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('cpu', 'gpu'))
    args = parser.parse_args()
    {'cpu': cpu, 'gpu': gpu}[args.stage]()
