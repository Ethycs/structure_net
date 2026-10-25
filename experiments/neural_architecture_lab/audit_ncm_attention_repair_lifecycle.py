"""Independent CPU state and provenance replay for the repair systems check."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_attention_repair_lifecycle as run

DEST = run.ROOT / '20261025_attention_code_repair_lifecycle_audit_v1'


def main():
    if (DEST.exists() or torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or run.q.read(run.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('fresh CPU audit after actual completed CUDA worker required')
    producer = run.q.read(run.DEST / 'producer.json')
    run.native.vendor.verify(producer)
    result = run.q.read(run.DEST / 'result.json')
    if (result['status'] != 'attention_code_repair_two_TRAIN_updates_and_strict_reload_passed'
            or result['updates'] != 2 or result['cases'] != 2 or not result['all_cases_TRAIN_only']
            or not result['frozen_codec_unchanged'] or not result['strict_model_optimizer_RNG_and_hard_output_reload']
            or result['peak_allocated_bytes'] > 3_000_000_000 or result['task_accuracy'] is not None
            or result['final_test_opened'] or result['speed_tested']
            or run.file_hash(run.DEST / 'model.pt') != result['checkpoint_sha256']
            or run.file_hash(run.DEST / 'outputs.json') != result['outputs_sha256']):
        raise ValueError('complete frozen-codebook lifecycle scope differs')
    saved = torch.load(run.DEST / 'model.pt', map_location='cpu', weights_only=True)
    model = run.repair.AttentionCodeRepair()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    if (saved['config'] != asdict(run.repair.Config()) or saved['step'] != 2
            or saved['producer_fingerprint'] != producer['fingerprint']
            or run.native.vendor.module_hash(model) != result['initial_model_sha256']):
        raise ValueError('exact initialized repair configuration or producer differs')
    model.load_state_dict(saved['model'], strict=True)
    if (run.native.vendor.module_hash(model) != result['final_model_sha256']
            or sum(p.numel() for p in model.parameters()) != result['trainable_parameters']
            or result['trainable_parameters'] != 3778049
            or torch.equal(initial['delta.weight'], saved['model']['delta.weight'])
            or torch.equal(initial['source_kv.weight'], saved['model']['source_kv.weight'])
            or not saved['optimizer']['state']
            or any(float(v['step']) != 2 for v in saved['optimizer']['state'].values())
            or saved['CPU_rng'].dtype != torch.uint8 or len(saved['CUDA_rng']) != 1):
        raise ValueError('changed head/source weights, all optimizer steps or saved RNG differs')
    trace = run.q.read(run.DEST / 'trace.json')
    if ([r['step'] for r in trace] != [1, 2] or trace[-1]['source_gradient_L1'] <= 0
            or any(len(r['losses']) != 2 or not math.isfinite(r['gradient_norm']) for r in trace)
            or any(not math.isfinite(v) or v < 0 for r in trace for loss in r['losses'] for v in loss.values())):
        raise ValueError('finite two-update trace and actual source-attention gradients required')
    # Independently select the pair from the original archived TRAIN population.
    rows = run.q.read(run.original.training.DATA / 'cases.json')
    manifest = run.q.read(run.original.training.DATA / 'manifest.json')
    predictions = run.q.read(run.original.training.TRAIN / 'diagnostic_outputs.json')
    ids = [predictions[0]['id'], min(predictions, key=lambda r: (-manifest[r['id']]['source_tokens'], r['id']))['id']]
    before = run.q.read(run.DEST / 'initial_outputs.json')
    after = run.q.read(run.DEST / 'outputs.json')
    by_id = {r['id']: r for r in predictions}
    if ([r['id'] for r in before] != ids or [r['id'] for r in after] != ids
            or any(r['codes'] != by_id[r['id']]['codes'] for r in before)
            or not set(ids) <= {r['id'] for r in rows}
            or any(len(r['codes']) != 641 or any(type(c) is not int or not 0 <= c < 8192 for c in r['codes']) for r in after)):
        raise ValueError('selected TRAIN pair, identity initialization or full outputs differ')
    changed = [sum(a != b for a, b in zip(x['codes'], y['codes'])) for x, y in zip(before, after)]
    paths = [Path(__file__), *[run.DEST / n for n in ('producer.json', 'result.json',
        'actual_tool_receipt.json', 'model.pt', 'trace.json', 'initial_outputs.json', 'outputs.json', 'selection.json')]]
    DEST.mkdir()
    audit = {'status': 'attention_repair_CPU_state_trace_and_TRAIN_bindings_replayed_not_quality',
        'trainable_parameters': 3778049, 'updates': 2, 'cases': 2,
        'changed_hard_code_positions': changed, 'task_accuracy': None,
        'quality_improvement_established': False, 'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(p): run.file_hash(p) for p in paths}}
    run.write_json(DEST / 'result.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
