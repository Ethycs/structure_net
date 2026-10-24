"""Full original-body FP32features with audited lossless disk storage, after DEV."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from experiments.structure_net import ncm_additional_source_metadata as metadata
from experiments.structure_net import ncm_lossless_feature_storage as storage
from experiments.structure_net import ncm_lm1b_native_development as development
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT, base = metadata.native, metadata.q, metadata.ROOT, metadata.base
codec = storage.codec
HYPOTHESIS = 'ncm-additional10000-frozen-body-lossless-FP32-features-v1'
MODULE = 'experiments.structure_net.ncm_additional_source_features'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-additional-source-features-plan.md')
TEST = Path('tests/structure_net/test_ncm_additional_source_features.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_additional_source_features.py')
WORKFLOW = ROOT / '20261024_additional10000_source_features_workflow_v1'
PILOT = ROOT / '20261024_additional10000_source_features_pilot_v1'
FEATURES = ROOT / '20261024_additional10000_source_features_v1'
METADATA_AUDIT = ROOT / '20261024_additional10000_source_metadata_audit_v1'
STORAGE_AUDIT = ROOT / '20261024_lossless_FP32_feature_storage_audit_v1'
INDEX_SHA = '72db66dd94fb8b062ebe9cce830abba0b3b8a50e3297a3d4fca406396f409f16'


def require_readiness():
    if (q.read(metadata.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0
            or q.read(METADATA_AUDIT / 'actual_tool_terminal_receipt.json')['exit_code'] != 0
            or file_hash(metadata.DATA / 'index.json') != INDEX_SHA
            or q.read(STORAGE_AUDIT / 'actual_tool_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('actual completed source metadata and lossless storage audits required')
    audit = q.read(METADATA_AUDIT / 'result.json')
    if audit['cases'] != 10000 or audit['artifacts'].get(str(metadata.DATA / 'index.json')) != INDEX_SHA:
        raise ValueError('complete audited10000source index required')
    native.vendor.verify(q.read(metadata.DATA / 'producer.json'))
    native.vendor.verify(q.read(storage.DEST / 'producer.json'))
    value = q.read(STORAGE_AUDIT / 'result.json')['metrics']['byte_shuffled']
    if value['cases'] != 16 or not value['bitwise_reconstruction']:
        raise ValueError('audited bitwise FP32 cache codec required')
    path = development.RESULTS / 'campaign_results.json'
    record = q.read(development.RESULTS / (development.HYPOTHESIS + '_registry_readback.json'))
    result = q.read(path)
    if (q.read(development.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0
            or record['record']['campaign_sha256'] != file_hash(path) or record['exact']['campaign_sha256'] != file_hash(path)
            or q.read(development.RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'):
        raise ValueError('actual completed native655DEV and independent registry readback required before GPU reuse')
    if result['DEV_qualified'] or result['classification'] != 'full128_source_only_DEV_quality_failed':
        raise ValueError('large feature extraction not admitted: final qualification takes priority after successful DEV')
    return q.read(metadata.DATA / 'index.json')


def probes(index):
    selected = [index[0]['id']]
    for field in ('native_prompt_positions', 'source_positions'):
        selected.append(min(index, key=lambda r: (-r[field], r['id']))['id'])
    result = list(dict.fromkeys(selected))
    if not 2 <= len(result) <= 3:
        raise ValueError('first/longest-native/longest-GPT2 source probes required')
    return result


def binding(row):
    return {**{k: row[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
        'original_body_sha256': base.EXPECTED_BODY}


def load_feature(entry, row):
    if (file_hash(entry['header_path']) != entry['header_sha256']
            or file_hash(entry['payload_path']) != entry['payload_sha256']):
        raise ValueError('lossless source feature header/payload changed')
    header = q.read(entry['header_path'])
    if header['binding'] != binding(row):
        raise ValueError('source/prompt/alignment/original-body feature binding differs')
    features = codec.decode(header, Path(entry['payload_path']).read_bytes())
    if (features.dtype != torch.float32 or features.shape != (len(row['source_ids']), 3072)
            or entry['source_tokens'] != len(features) or not torch.isfinite(features).all()):
        raise ValueError('complete finite original FP32 source features required')
    return features


@torch.no_grad()
def extract(parent, row):
    ids = torch.tensor([row['context_ids']], dtype=torch.long, device='cuda')
    hidden = base.source.frozen_parent_features(parent, ids, torch.ones_like(ids))[0]
    features = base.align(hidden, row['alignment'])
    if features.dtype != torch.float32 or features.shape != (len(row['source_ids']), 3072) or not torch.isfinite(features).all():
        raise ValueError('complete FP32 character-aligned original-body features required')
    return features


def reference_parity(parent):
    rows = q.read(metadata.old_training.DATA / 'cases.json')
    manifest = q.read(metadata.old_training.DATA / 'manifest.json')
    selected = [rows[0], max(rows, key=lambda r: manifest[r['id']]['source_tokens'])]
    result = []
    for row in selected:
        entry = manifest[row['id']]
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('original TRAIN reference feature changed')
        before = torch.load(entry['path'], map_location='cpu', weights_only=True, mmap=True)['features']
        after = extract(parent, row)
        if not torch.equal(before.view(torch.uint8), after.contiguous().view(torch.uint8)):
            raise ValueError('recomputed original first/longest TRAIN features differ in bytes')
        result.append({'id': row['id'], 'reference_sha256': entry['sha256'], 'FP32_bytes_exact': True})
    return result


def producing(root, stage, inputs=(), extra=None):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(metadata.__file__), metadata.PLAN, Path(codec.__file__),
        Path(base.__file__), Path(base.source.__file__), Path(development.__file__)]
    common = [metadata.DATA / 'producer.json', metadata.DATA / 'index.json', metadata.WORKFLOW / 'tool_terminal_receipt.json',
        METADATA_AUDIT / 'result.json', METADATA_AUDIT / 'actual_tool_terminal_receipt.json',
        STORAGE_AUDIT / 'result.json', STORAGE_AUDIT / 'actual_tool_terminal_receipt.json',
        development.RESULTS / 'campaign_results.json', development.RESULTS / (development.HYPOTHESIS + '_registry_readback.json'),
        development.RESULTS / 'backend_exact_readback.json', development.WORKFLOW / 'tool_terminal_receipt.json',
        metadata.old_training.DATA / 'cases.json', metadata.old_training.DATA / 'manifest.json']
    common += [base.MODEL_PATH / n for n in ('config.json', 'model.safetensors.index.json',
        'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors')]
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {**(extra or {}), 'hypothesis_id': HYPOTHESIS,
        'original_body_hash': base.EXPECTED_BODY, 'body_layers': 28, 'body_precision': 'BF16unchanged',
        'original_body_and_MLPs_updated': False, 'aligned_feature_precision': 'FP32unchanged',
        'feature_storage': 'lossless byte-shuffled zlib1, not quantized', 'full_source_retained': True,
        'source_only': True, 'new_teacher_labels_read': False, 'date': '23 Oct 2026',
        'GPU_reuse_after_native655DEV': True, 'DEV_success_redirects_to_final_qualification': True,
        'physical_GPU': 0, 'TF32': False, 'peak_limit_bytes': 8_600_000_000,
        'minimum_free_disk_bytes': 8 * 1024**3, 'canonical_firsts_changed': False,
        'task_accuracy': None, 'model_speed_measured': False, 'all_final_sources_closed': True})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'additional_feature_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def cache(pilot=False):
    index = require_readiness()
    metadata.old_training.require_gpu()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('same original physicalGPU0 BF16body required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    retained = {}
    extra = []
    if not pilot:
        native.vendor.terminal(PILOT)
        if not q.read(PILOT / 'result.json')['all_FP32_byte_roundtrips_exact']:
            raise ValueError('actual repeated extraction/lossless original-feature parity pilot required')
        retained = q.read(PILOT / 'manifest.json')
        extra += [PILOT / n for n in ('manifest.json', 'result.json', 'reference_parity.json', 'worker_terminal_receipt.json')]
    ids = set(probes(index))
    chosen = [r for r in index if r['id'] in ids] if pilot else index
    root = PILOT if pilot else FEATURES
    p = producing(root, 'first_and_both_longest_source_feature_pilot' if pilot else 'all10000_original_body_FP32_source_features',
        [Path(r['path']) for r in chosen] + extra, {'pilot': pilot, 'cases': len(chosen)})
    (root / 'features').mkdir()
    (root / 'entries').mkdir()
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(base.MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    if parent.config.num_hidden_layers != 28 or base.module_hash(parent.model) != base.EXPECTED_BODY:
        raise ValueError('unchanged original28layer body required')
    torch.cuda.reset_peak_memory_stats()
    parity = reference_parity(parent)
    manifest = {}
    for i, source in enumerate(chosen, 1):
        if shutil.disk_usage(ROOT).free < 8 * 1024**3:
            raise ValueError('reserved8GiBdisk floor reached; no source truncation or dtype fallback')
        if file_hash(source['path']) != source['sha256']:
            raise ValueError('prepared source metadata changed')
        row = q.read(source['path'])
        if row['id'] in retained:
            entry = retained[row['id']]
            load_feature(entry, row)
        else:
            features = extract(parent, row)
            if pilot and not torch.equal(features.view(torch.uint8), extract(parent, row).contiguous().view(torch.uint8)):
                raise ValueError('original frozen body feature extraction is not exactly repeatable')
            header, payload = codec.encode(features, 'byte_shuffled')
            header['binding'] = binding(row)
            header_path = root / 'features' / (row['id'] + '.json')
            payload_path = root / 'features' / (row['id'] + '.zlib')
            payload_path.write_bytes(payload)
            write_json(header_path, header)
            entry = {'header_path': str(header_path), 'header_sha256': file_hash(header_path),
                'payload_path': str(payload_path), 'payload_sha256': file_hash(payload_path),
                'source_tokens': len(features), 'raw_bytes': features.numel() * 4,
                'stored_bytes': header_path.stat().st_size + payload_path.stat().st_size}
            restored = load_feature(entry, row)
            if not torch.equal(features.view(torch.uint8), restored.view(torch.uint8)):
                raise ValueError('actual stored features differ from original FP32 extraction bytes')
        manifest[row['id']] = entry
        write_json(root / 'entries' / (row['id'] + '.json'), entry)
        if i % 16 == 0 or pilot or i == len(chosen):
            write_json(root / 'progress.json', {'cached': i, 'cases': len(chosen), 'FP32_bytes_preserved': True})
            print(json.dumps({'cached_source_features': i, 'cases': len(chosen)}), flush=True)
    if base.module_hash(parent.model) != base.EXPECTED_BODY or any(v.grad is not None for v in parent.parameters()):
        raise ValueError('original body changed or received gradients')
    peak = torch.cuda.max_memory_allocated()
    if peak > 8_600_000_000:
        raise ValueError('feature extraction exceeds original8.6GBplan')
    write_json(root / 'reference_parity.json', parity)
    write_json(root / 'manifest.json', manifest)
    native.vendor.verify(p)
    result = {'status': 'original_body_feature_pilot_passed' if pilot else 'all10000_original_FP32_features_cached_losslessly',
        'cases': len(manifest), 'raw_bytes': sum(v['raw_bytes'] for v in manifest.values()),
        'stored_bytes': sum(v['stored_bytes'] for v in manifest.values()), 'peak_allocated_bytes': peak,
        'original_body_hash_unchanged': base.EXPECTED_BODY, 'all_source_positions_retained': True,
        'all_FP32_byte_roundtrips_exact': True, 'manifest_sha256': file_hash(root / 'manifest.json'),
        'task_accuracy': None, 'model_speed_measured': False, 'all_final_sources_closed': True}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)


def workflow():
    require_readiness()
    if any(p.exists() for p in (WORKFLOW, PILOT, FEATURES)):
        raise ValueError('fresh additional source feature roots required')
    p = producing(WORKFLOW, 'supervise_full_original_body_lossless_FP32_cache')
    receipts = []
    for stage, root in (('pilot', PILOT), ('cache', FEATURES)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'child_pid': child.pid, 'observer_pid': os.getpid(), 'observer': 'subprocess.wait', 'command': command}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('additional original-body feature stage failed:' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'all10000_lossless_FP32_source_feature_workflow_completed', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot', 'cache', 'workflow'))
    args = parser.parse_args()
    {'pilot': lambda: cache(True), 'cache': cache, 'workflow': workflow}[args.stage]()
