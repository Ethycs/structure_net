"""Localize failure in the frozen new conditional NCM using oracle prefixes."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_level_conditioning_checkpoint as checkpoint
from experiments.structure_net import ncm_sol64_grounded_oracle as component

native, q, ROOT = checkpoint.native, checkpoint.q, checkpoint.ROOT
HYPOTHESIS = 'ncm-level-conditioning-oracle-prefix-diagnostic-v1'
DEST = ROOT / '20261025_level_conditioning_oracle_prefix_diagnostic_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-oracle-prefix-diagnostic-plan.md')
TEST = Path('tests/structure_net/test_ncm_oracle_prefix_diagnostic.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_oracle_prefix_diagnostic.py')
FORCED_LEVELS = (0, 1, 3, 6, 12)


@contextmanager
def forced_sampler(generator, oracle, count):
    levels = tuple(generator.levels)
    if (not 0 <= count <= len(levels) or oracle.shape != (1, sum(levels))
            or '_sample_level' in generator.__dict__ or generator.l0_ar):
        raise ValueError('clean non-autoregressive native root and complete oracle hierarchy required')
    original = generator._sample_level
    trace, offset = [], 0

    def sample(logits, temperature, top_k, top_p):
        nonlocal offset
        i = len(trace)
        if (i >= len(levels) or logits.shape != (1, levels[i], 8192)
                or temperature != 0. or top_k != 0 or top_p != 1.):
            raise ValueError('one greedy call per complete native level required')
        target = oracle[:, offset:offset + levels[i]]
        ce = F.cross_entropy(logits.float().flatten(0, 1), target.flatten())
        trace.append({'level': i, 'positions': levels[i], 'forced': i < count,
            'oracle_cross_entropy': float(ce)})
        offset += levels[i]
        return target.clone() if i < count else original(logits, temperature, top_k, top_p)

    generator._sample_level = sample
    try:
        yield trace
        if len(trace) != len(levels) or offset != sum(levels):
            raise ValueError('incomplete native hierarchy was generated')
    finally:
        del generator.__dict__['_sample_level']


@torch.no_grad()
def evaluate(model, row, tokenizer):
    metadata = q.read(row['source_metadata']['path'])
    features = component.features.load_feature(row['feature'], metadata).unsqueeze(0).cuda()
    mask = torch.ones(features.shape[:2], device='cuda', dtype=torch.bool)
    oracle = torch.tensor([row['oracle_codes']], device='cuda', dtype=torch.long)
    words = torch.tensor([row['native_word_ids']], device='cuda', dtype=torch.long)
    active = words != 0
    oracle_latent = model.codec.vq.idx_to_z_q(oracle)
    ordinary = model.generate(features, mask)
    generator = model.concepts.generator
    arms = {}
    for count in FORCED_LEVELS:
        with forced_sampler(generator, oracle, count) as trace:
            codes = model.concepts.generate_codes(features, mask)
        cut = sum(generator.levels[:count])
        if not torch.equal(codes[:, :cut], oracle[:, :cut]):
            raise ValueError('forced oracle prefix not retained')
        logits = model.word_logits(features, mask, codes)
        raw = logits.argmax(-1)
        if count == 0 and (not torch.equal(codes, ordinary['codes'])
                or not torch.equal(raw, ordinary['raw_word_ids'])):
            raise ValueError('zero-forcing differs from ordinary first prediction')
        latent = model.codec.vq.idx_to_z_q(codes)
        arms[str(count)] = {'forced_levels': count, 'forced_positions': cut,
            'codes': codes.cpu().tolist()[0], 'level_trace': trace,
            'oracle_latent_MSE': float((latent - oracle_latent).square().mean()),
            'active_word_matches': int(((raw == words) & active).sum()),
            'active_words': int(active.sum()),
            'word_cross_entropy': float(F.cross_entropy(logits.float().flatten(0, 1), words.flatten(), ignore_index=0)),
            **component.numeric.decode(logits, tokenizer)}
    source_off = component.numeric.decode(model.codec.decoder(oracle_latent), tokenizer)
    if (model._word_bound is not None or model.concepts._bound is not None
            or '_sample_level' in generator.__dict__):
        raise ValueError('temporary source or sampling context leaked')
    return {'id': row['id'], 'source_sha256': row['source_sha256'],
        'source_positions': features.shape[1], 'feature_payload_sha256': row['feature']['payload_sha256'],
        'oracle_codes': row['oracle_codes'], 'oracle_word_ids': row['native_word_ids'],
        'arms': arms, 'oracle_decode_without_source': source_off}


def main():
    native.vendor.corpus.quality.dependencies()
    checkpoint.training.require_gpu()
    if DEST.exists():
        raise ValueError('fresh append-only diagnostic root required')
    rows = component.inputs()
    longest = min(rows, key=lambda r: (-r['feature']['source_tokens'], r['id']))
    selected = [rows[0], longest]
    if selected[0]['id'] == selected[1]['id']:
        raise ValueError('distinct predeclared first and longest-source cases required')
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(checkpoint.__file__),
        Path(component.__file__), Path(component.numeric.__file__),
        Path(component.features.__file__), Path(component.features.codec.__file__),
        Path(checkpoint.training.lifecycle.head.__file__),
        Path(checkpoint.training.lifecycle.head.concepts.__file__),
        *sorted(native.vendor.VENDOR.rglob('*.py'))]
    inputs = [checkpoint.training.TRAIN / n for n in ('model.pt', 'result.json', 'worker_terminal_receipt.json')]
    inputs += [checkpoint.TRAIN_AUDIT / n for n in ('result.json', 'actual_tool_terminal_receipt.json')]
    inputs += [component.numeric.OUTPUTS / 'outputs.json', component.NUMERIC_AUDIT / 'result.json',
        component.FEATURE_AUDIT / 'result.json', component.features.FEATURES / 'manifest.json']
    for row in selected:
        inputs += [Path(row['source_metadata']['path']), *[Path(row['feature'][k]) for k in ('header_path', 'payload_path')]]
    producer = native.vendor.freeze(DEST, 'frozen_new_checkpoint_oracle_prefix_diagnostic', [*files, *inputs], {
        'hypothesis_id': HYPOTHESIS, 'selected_ids': [r['id'] for r in selected],
        'forced_levels': FORCED_LEVELS, 'all_positions_retained': True,
        'physical_GPU': 2, 'precision': 'FP32', 'TF32': False, 'peak_limit_bytes': 3_000_000_000,
        'oracle_target_information': True, 'task_accuracy': None, 'semantic_judge_requests': 0,
        'final_test_opened': False, 'speed_tested': False, 'training': False})
    for i, path in enumerate(files):
        (DEST / 'producer_source' / f'oracle_prefix_{i}_{path.name}').write_bytes(path.read_bytes())
    checkpoint_state = None
    model = None
    try:
        model, checkpoint_state = checkpoint.load_selected()
        before = native.vendor.module_hash(model)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = native.tokenizer()
        outputs = []
        for row in selected:
            outputs.append(evaluate(model, row, tokenizer))
            component.write_json(DEST / 'outputs.json', outputs)
            print(json.dumps({'diagnostic_cases_completed': len(outputs)}), flush=True)
        peak = torch.cuda.max_memory_allocated()
        if (native.vendor.module_hash(model) != before or any(p.grad is not None for p in model.parameters())
                or peak > 3_000_000_000):
            raise ValueError('frozen model, no-gradient or memory contract failed')
    finally:
        if model is not None:
            model.close()
    native.vendor.verify(producer)
    result = {'status': 'two_case_oracle_prefix_diagnostic_completed_not_task_accuracy',
        'hypothesis_id': HYPOTHESIS, 'cases': 2, 'forced_levels': FORCED_LEVELS,
        'checkpoint_sha256': checkpoint_state['checkpoint_sha256'], 'model_sha256': before,
        'model_unchanged': True, 'source_contexts_restored': True, 'peak_allocated_bytes': peak,
        'outputs_sha256': component.file_hash(DEST / 'outputs.json'),
        'oracle_target_information': True, 'task_accuracy': None,
        'final_test_opened': False, 'speed_tested': False}
    component.write_json(DEST / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
