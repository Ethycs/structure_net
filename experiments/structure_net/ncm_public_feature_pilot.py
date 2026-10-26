"""Verify original frozen-body FP32 features for the nine public pipeline probes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from experiments.structure_net import ncm_public_source_metadata as preparation
from experiments.structure_net import ncm_additional_source_features as original

data = preparation.data
ROOT = data.ROOT/'20261026_public_feature_pilot_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-feature-pilot-plan.md')
HYPOTHESIS = 'ncm-public-original-body-feature-pilot-v1'


def freeze(environment, sources):
    paths = [preparation.ROOTS['pilot']/name for name in
             ('metadata.jsonl', 'index.jsonl', 'result.json', 'actual_tool_receipt.json')]
    paths += [original.metadata.old_training.DATA/name for name in ('cases.json', 'manifest.json')]
    paths += [p for p in original.base.MODEL_PATH.iterdir() if p.is_file()]
    paths += [Path('pyproject.toml'), Path('pixi.lock')]
    files = {Path(__file__).resolve(), PLAN.resolve(),
        Path('tests/structure_net/test_ncm_fp32_feature_codec.py').resolve(),
        Path('tests/structure_net/test_ncm_public_source_metadata.py').resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()): files.add(p)
    ROOT.mkdir(); (ROOT/'producer_source').mkdir(); (ROOT/'features').mkdir()
    for i, p in enumerate(sorted(files)):
        (ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    value = {'schema': 'nal.ncm-public-feature-pilot.v1', 'hypothesis_id': HYPOTHESIS,
        'environment': environment, 'config': {'source_ids': [r['id'] for r in sources], 'physical_GPU': 0,
            'body_precision': 'BF16', 'aligned_feature_precision': 'FP32', 'TF32': False,
            'all_source_positions_retained': True, 'maximum_allocation_bytes': 8_600_000_000,
            'expected_body_sha256': original.base.EXPECTED_BODY,
            'systems_lifecycle_only_not_quality_evidence': True},
        'input_hashes': {str(p): data.sha(p) for p in paths},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)}}
    data.write(ROOT/'producer.json', value)
    return value


def run():
    if ROOT.exists(): raise ValueError('fresh original-body feature pilot root required')
    from experiments.structure_net import ncm_pixi
    _, environment = ncm_pixi.configure()
    original.metadata.old_training.require_gpu()
    state = data.completed(preparation.ROOTS['pilot'])
    if not state['original_TRAIN_geometry_equal'] or state['cases'] != 9:
        raise ValueError('completed nine-source metadata pilot required')
    source_proof = data.read(preparation.ROOTS['pilot']/'producer.json')
    data.verify(source_proof['input_hashes']); data.verify(source_proof['source_hashes'])
    with (preparation.ROOTS['pilot']/'metadata.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    if (len(rows) != 9 or any(original.metadata.FORBIDDEN & r.keys() for r in rows)
            or any(r['source_truncated'] for r in rows)):
        raise ValueError('complete source-only public probes required')
    proof = freeze(environment, rows)
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(original.base.MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    if parent.config.num_hidden_layers != 28 or original.base.module_hash(parent.model) != original.base.EXPECTED_BODY:
        raise ValueError('unchanged original28-layer body required')
    torch.cuda.reset_peak_memory_stats()
    parity = original.reference_parity(parent)
    data.write(ROOT/'original_TRAIN_feature_parity.json', parity)
    manifest, measurements = {}, []
    with torch.no_grad():
        for row in rows:
            torch.cuda.synchronize(); started = time.perf_counter()
            features = original.extract(parent, row)
            torch.cuda.synchronize(); extraction_seconds = time.perf_counter()-started
            again = original.extract(parent, row)
            if not torch.equal(features.contiguous().view(torch.uint8), again.contiguous().view(torch.uint8)):
                raise ValueError('public source extraction is not bitwise repeatable')
            started = time.perf_counter()
            header, payload = original.codec.encode(features, 'byte_shuffled')
            header['binding'] = original.binding(row)
            header_path, payload_path = ROOT/'features'/(row['id']+'.json'), ROOT/'features'/(row['id']+'.zlib')
            data.write(header_path, header); payload_path.write_bytes(payload)
            entry = {'header_path': str(header_path), 'header_sha256': data.sha(header_path),
                'payload_path': str(payload_path), 'payload_sha256': data.sha(payload_path),
                'source_tokens': len(features), 'raw_bytes': features.numel()*4,
                'stored_bytes': header_path.stat().st_size+payload_path.stat().st_size}
            restored = original.load_feature(entry, row)
            if not torch.equal(features.contiguous().view(torch.uint8), restored.contiguous().view(torch.uint8)):
                raise ValueError('lossless serialized public feature differs')
            measurement = {'id': row['id'], 'source_positions': len(features),
                'extraction_seconds': extraction_seconds, 'encode_write_verify_seconds': time.perf_counter()-started,
                'repeated_extraction_bytes_equal': True, 'lossless_roundtrip_bytes_equal': True,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            measurements.append(measurement); manifest[row['id']] = entry
            data.write(ROOT/f"measurement_{len(measurements)}.json", measurement)
            print(json.dumps(measurement), flush=True)
    unchanged = original.base.module_hash(parent.model) == original.base.EXPECTED_BODY
    if not unchanged or any(p.grad is not None for p in parent.parameters()):
        raise ValueError('body changed or received gradients')
    peak = torch.cuda.max_memory_allocated()
    if peak > 8_600_000_000: raise ValueError('original8.6GBbody memory budget exceeded')
    data.write(ROOT/'manifest.json', manifest)
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    result = {'status': 'nine_public_original_body_feature_controls_passed', 'hypothesis_id': HYPOTHESIS,
        'cases': len(rows), 'original_TRAIN_reference_cases_exact': len(parity),
        'all_repeated_extractions_exact': True, 'all_lossless_roundtrips_exact': True,
        'body_hash_unchanged': original.base.EXPECTED_BODY, 'peak_allocated_bytes': peak,
        'raw_feature_bytes': sum(v['raw_bytes'] for v in manifest.values()),
        'stored_feature_bytes': sum(v['stored_bytes'] for v in manifest.values()),
        'model_training': False, 'new_judge_requests': 0, 'task_accuracy': None,
        'whole_corpus_feature_pipeline_qualified': False, 'final_test_opened': False,
        'systems_lifecycle_only_not_quality_evidence': True,
        'artifacts': {str(p): data.sha(p) for p in ROOT.rglob('*') if p.is_file()}}
    data.write(ROOT/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('run',))
    parser.parse_args()
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if not existed and ROOT.exists():
            data.write(ROOT/'failure.json', {'exception': type(error).__name__, 'message': str(error), 'task_accuracy': None})
        raise
