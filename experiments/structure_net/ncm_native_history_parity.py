"""Measure packed and sequential NCM paths at identical histories; no training."""
from pathlib import Path
import json
import os

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_sol64_grounded_oracle as checkpoint
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = checkpoint.native, checkpoint.q, checkpoint.ROOT
DEST = ROOT / '20261024_native655_history_path_parity_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-native-history-parity-plan.md')
TEST = Path('tests/structure_net/test_ncm_native_history_parity.py')
HYPOTHESIS = 'ncm-native655-history-path-parity-v1'


@torch.no_grad()
def forced_logits(head, source, mask, codes):
    generator = head.generator
    levels = tuple(generator.levels)
    if codes.shape != (len(source), sum(levels)):
        raise ValueError('all native oracle levels required')
    had_sampler = '_sample_level' in generator.__dict__
    previous = generator.__dict__.get('_sample_level')
    captured, offset = [], 0

    def supply(logits, temperature, top_k, top_p):
        nonlocal offset
        i = len(captured)
        if i >= len(levels) or logits.shape[:2] != (len(source), levels[i]):
            raise ValueError('native generation level order or shape changed')
        captured.append(logits.detach().cpu())
        value = codes[:, offset:offset + levels[i]].clone()
        offset += levels[i]
        return value

    generator._sample_level = supply
    try:
        prefix = torch.full((len(source), 1), 101, dtype=torch.long, device=source.device)
        with head.source_context(source, mask):
            actual = generator.generate(None, prefix, temperature=0., top_k=0, top_p=1., use_kv_cache=False, cfg_scale=0.)
        if len(captured) != len(levels) or not torch.equal(actual, codes):
            raise ValueError('complete identical oracle code history was not preserved')
        return torch.cat(captured, dim=1)
    finally:
        if had_sampler:
            generator._sample_level = previous
        else:
            del generator._sample_level


def metrics(values, levels):
    packed, sequential = values['packed'], values['sequential']
    gold, history, rollout = values['oracle_codes'], values['generated_history_logits'], values['free_codes']
    result, offset = [], 0
    for width in levels:
        a, b = packed[:, offset:offset + width], sequential[:, offset:offset + width]
        clean = gold[:, offset:offset + width]
        noisy = history[:, offset:offset + width]
        result.append({'positions': width, 'max_abs_logit_difference': float((a-b).abs().max()),
            'allclose_1e4': torch.allclose(a, b, rtol=1e-4, atol=1e-4),
            'packed_sequential_argmax_disagreements': int((a.argmax(-1) != b.argmax(-1)).sum()),
            'clean_history_code_CE': float(F.cross_entropy(a.flatten(0, 1), clean.flatten())),
            'generated_history_code_CE': float(F.cross_entropy(noisy.flatten(0, 1), clean.flatten())),
            'clean_history_correct_codes': int((a.argmax(-1) == clean).sum()),
            'free_correct_codes': int((rollout[:, offset:offset+width] == clean).sum())})
        offset += width
    return {'levels': result, 'all_levels_aligned_1e4': all(r['allclose_1e4'] for r in result),
        'root_clean_generated_max_abs': float((packed[:, :levels[0]] - history[:, :levels[0]]).abs().max()),
        'source_swap_root_max_abs': float((packed[:, :levels[0]] - values['swapped_source_logits'][:, :levels[0]]).abs().max()),
        'source_swap_argmax_changes': int((packed.argmax(-1) != values['swapped_source_logits'].argmax(-1)).sum())}


def main():
    checkpoint.configure()
    if (DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '1'
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or not torch.cuda.is_available()
            or '2060' not in torch.cuda.get_device_name() or 'SUPER' in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 2_000_000_000):
        raise ValueError('fresh two-TRAIN-case diagnostic on physical1 RTX2060 with2GBfree required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = checkpoint.selected
    rows = q.read(selected.training.DATA / 'cases.json')
    manifest = q.read(selected.training.DATA / 'manifest.json')
    longest = min(rows, key=lambda r: (-manifest[r['id']]['source_tokens'], r['id']))
    chosen = [rows[0], longest]
    if len({r['id'] for r in chosen}) != 2:
        raise ValueError('distinct first/longest TRAIN cases required')
    code_path = selected.training.training.CACHE / 'outputs.json'
    codes = {r['id']: r['codes'] for r in q.read(code_path)}
    files = [Path(__file__), PLAN, TEST, Path(checkpoint.__file__), checkpoint.PLAN,
        Path(selected.__file__), Path(selected.head.__file__), Path(selected.head.concepts.__file__),
        *sorted(native.vendor.VENDOR.rglob('*.py'))]
    inputs = [selected.training.DATA / 'cases.json', selected.training.DATA / 'manifest.json', code_path,
        selected.training.TRAIN / 'model.pt', selected.training.TRAIN / 'result.json',
        selected.TRAIN_AUDIT / 'result.json', selected.TRAIN_AUDIT / 'actual_tool_terminal_receipt.json',
        *[Path(manifest[r['id']]['path']) for r in chosen]]
    p = native.vendor.freeze(DEST, 'systems_two_TRAIN_history_path_diagnostic', [*files, *inputs], {
        'hypothesis_id': HYPOTHESIS, 'checkpoint_sha256': checkpoint.CHECKPOINT_SHA,
        'TRAIN_case_ids': [r['id'] for r in chosen], 'model_training': False,
        'oracle_sampler_only_for_logit_comparison': True, 'physical_GPU': 1,
        'precision': 'FP32', 'TF32': False, 'peak_limit_bytes': 2_000_000_000,
        'task_accuracy': None, 'external_judgments': False, 'final_test_opened': False})
    model, _ = checkpoint.load_selected('cuda')
    before = native.vendor.module_hash(model)
    source = []
    for row in chosen:
        entry = manifest[row['id']]
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('original TRAIN feature changed')
        value = torch.load(entry['path'], map_location='cpu', weights_only=True)
        if (value['features'].dtype != torch.float32
                or value['features'].shape != (entry['source_tokens'], 3072)
                or any(value[k] != row[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256'))):
            raise ValueError('complete original FP32 TRAIN source features required')
        source.append(value['features'].unsqueeze(0).cuda())
    torch.cuda.reset_peak_memory_stats()
    values, reports = {}, {}
    try:
        for i, row in enumerate(chosen):
            f, other = source[i], source[1-i]
            mask = torch.ones(f.shape[:2], dtype=torch.bool, device='cuda')
            other_mask = torch.ones(other.shape[:2], dtype=torch.bool, device='cuda')
            gold = torch.tensor([codes[row['id']]], dtype=torch.long, device='cuda')
            with torch.no_grad():
                packed = model.concepts.teacher_logits(f, mask, gold).cpu()
                sequential = forced_logits(model.concepts, f, mask, gold)
                free = model.concepts.generate_codes(f, mask)
                history = model.concepts.teacher_logits(f, mask, free).cpu()
                swapped = model.concepts.teacher_logits(other, other_mask, gold).cpu()
            raw = {'packed': packed, 'sequential': sequential, 'oracle_codes': gold.cpu(),
                'free_codes': free.cpu(), 'generated_history_logits': history, 'swapped_source_logits': swapped}
            if any(not torch.isfinite(v).all() for v in raw.values()):
                raise ValueError('all diagnostic logits and codes must be finite')
            values[row['id']], reports[row['id']] = raw, metrics(raw, tuple(native.LEVELS))
        peak = torch.cuda.max_memory_allocated()
        if peak > 2_000_000_000 or native.vendor.module_hash(model) != before or any(v.grad is not None for v in model.parameters()):
            raise ValueError('frozen model or2GB diagnostic memory budget differs')
    finally:
        model.close()
    torch.save(values, DEST / 'logits.pt')
    native.vendor.verify(p)
    write_json(DEST / 'result.json', {'status': 'two_TRAIN_history_path_diagnostic_recorded',
        'reports': reports, 'model_unchanged': True, 'model_sha256': before, 'peak_allocated_bytes': peak,
        'logits_sha256': file_hash(DEST / 'logits.pt'), 'source_only_task_accuracy': None,
        'final_test_opened': False, 'no_training_or_semantic_judgments': True})
    print(json.dumps(q.read(DEST / 'result.json')), flush=True)


if __name__ == '__main__':
    main()
