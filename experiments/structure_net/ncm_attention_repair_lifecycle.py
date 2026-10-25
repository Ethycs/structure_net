"""Two real-geometry TRAIN updates for attention repair; not a quality experiment."""
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_attention_code_repair as repair
from experiments.structure_net import ncm_level_conditioning_checkpoint as original
from experiments.structure_net import ncm_lm1b_word_gradient as geometry
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = original.native, original.q, original.ROOT
HYPOTHESIS = 'ncm-source-attention-code-repair-lifecycle-v1'
DEST = ROOT / '20261025_attention_code_repair_lifecycle_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-attention-code-repair-plan.md')
TEST = Path('tests/structure_net/test_ncm_attention_code_repair.py')
SEED = 130142


def metadata():
    audit = q.read(original.TRAIN_AUDIT / 'result.json')
    if (q.read(original.TRAIN_AUDIT / 'actual_tool_terminal_receipt.json')['exit_code'] != 0
            or q.read(original.training.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('actual completed original training and independent audit required')
    for path, expected in audit['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('audited original TRAIN evidence changed: ' + path)
    state = q.read(original.training.TRAIN / 'result.json')
    if (state['checkpoint_sha256'] != '3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec'
            or not state['strict_reload_passed'] or state['updates'] != 1024):
        raise ValueError('fixed new final checkpoint required')
    rows = q.read(original.training.DATA / 'cases.json')
    manifest = q.read(original.training.DATA / 'manifest.json')
    predicted = q.read(original.training.TRAIN / 'diagnostic_outputs.json')
    codes = {r['id']: r['codes'] for r in q.read(original.training.training.CACHE / 'outputs.json')}
    by_id = {r['id']: r for r in rows}
    if (len(rows) != 655 or len(predicted) != 16 or len({r['id'] for r in predicted}) != 16
            or not {r['id'] for r in predicted} <= by_id.keys()):
        raise ValueError('exact completed TRAIN diagnostic subset required')
    selected = [predicted[0], min(predicted, key=lambda r: (-manifest[r['id']]['source_tokens'], r['id']))]
    if selected[0]['id'] == selected[1]['id']:
        raise ValueError('distinct first and longest-source TRAIN probes required')
    result = []
    for value in selected:
        row, entry = by_id[value['id']], manifest[value['id']]
        if (row['target_origin'] != 'original_approved_target_unchanged'
                or value['source_sha256'] != row['source_sha256']
                or value['feature_sha256'] != entry['sha256']
                or value['source_positions'] != entry['source_tokens']
                or len(value['codes']) != 641 or len(codes[value['id']]) != 641
                or file_hash(entry['path']) != entry['sha256']):
            raise ValueError('unaltered original target, cached prediction and full source required')
        result.append({'row': row, 'feature': entry, 'prediction': value, 'oracle_codes': codes[value['id']]})
    return result


def require_gpu():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or not torch.cuda.is_available() or '2060 SUPER' not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 3_500_000_000):
        raise ValueError('physical GPU2 RTX2060SUPER with3.5GBfree required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_features(item):
    row, entry = item['row'], item['feature']
    value = torch.load(entry['path'], map_location='cpu', weights_only=True)
    if (any(value[k] != row[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256'))
            or value['features'].shape != (entry['source_tokens'], 3072)
            or value['features'].dtype != torch.float32 or not torch.isfinite(value['features']).all()):
        raise ValueError('full aligned original FP32 features required')
    features = value['features'].unsqueeze(0).cuda().requires_grad_(True)
    return features, torch.ones(features.shape[:2], dtype=torch.bool, device='cuda')


@torch.no_grad()
def outputs(model, items, tensors, books):
    model.eval()
    return [{'id': item['row']['id'], 'codes': model.repair(codes, features, mask, books).cpu().tolist()[0]}
        for item, (codes, features, mask, _) in zip(items, tensors)]


def main():
    native.vendor.corpus.quality.dependencies()
    require_gpu()
    if DEST.exists():
        raise ValueError('fresh append-only lifecycle root required')
    items = metadata()
    files = [Path(__file__), PLAN, TEST, Path(repair.__file__), Path(geometry.__file__),
        Path(original.__file__), Path(native.__file__), *sorted(native.vendor.VENDOR.rglob('*.py'))]
    inputs = [original.TRAIN_AUDIT / n for n in ('result.json', 'actual_tool_terminal_receipt.json')]
    inputs += [original.training.TRAIN / n for n in ('result.json', 'diagnostic_outputs.json')]
    inputs += [original.training.DATA / n for n in ('cases.json', 'manifest.json')]
    inputs += [original.training.training.CACHE / 'outputs.json',
        native.RELEASE / 'files/lm1b/tokenizer.pt', native.RELEASE / 'files/lm1b/small/config.yaml']
    inputs += [Path(item['feature']['path']) for item in items]
    c = repair.Config()
    producer = native.vendor.freeze(DEST, 'two_TRAIN_updates_with_frozen_native_codebooks', [*files, *inputs], {
        'hypothesis_id': HYPOTHESIS, 'config': asdict(c), 'seed': SEED, 'updates': 2,
        'microbatch': 1, 'accumulation': 2, 'learning_rate': 1e-4,
        'optimizer': {'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01},
        'loss': 'equal-level code CE plus0.05 full soft-latent MSE', 'physical_GPU': 2,
        'precision': 'FP32', 'TF32': False, 'peak_limit_bytes': 3_000_000_000,
        'selected_ids': [i['row']['id'] for i in items], 'all_cases_TRAIN_only': True,
        'NCM_generator_loaded': False, 'parent_loaded': False, 'codec_frozen': True,
        'source_features_detached': True, 'actual_generator_errors': True,
        'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False})
    for i, path in enumerate(files):
        (DEST / 'producer_source' / f'repair_lifecycle_{i}_{path.name}').write_bytes(path.read_bytes())
    write_json(DEST / 'selection.json', {'ids': [i['row']['id'] for i in items],
        'selection': 'first and longest source among16 original final TRAIN diagnostics'})
    codec = native.load_codec()
    if (native.vendor.module_hash(codec.encoder) != original.training.lifecycle.ENCODER
            or native.vendor.module_hash(codec.vq) != original.training.lifecycle.VQ):
        raise ValueError('same protected native encoder and VQ geometry required')
    codec_hash = native.vendor.module_hash(codec)
    books = [e.weight for e in codec.vq.embeddings]
    tensors = []
    for item in items:
        features, mask = load_features(item)
        tensors.append((torch.tensor([item['prediction']['codes']], dtype=torch.long, device='cuda'),
            features, mask, torch.tensor([item['oracle_codes']], dtype=torch.long, device='cuda')))
    model = repair.AttentionCodeRepair(c).cuda()
    initial_hash = native.vendor.module_hash(model)
    initial = outputs(model, items, tensors, books)
    if [r['codes'] for r in initial] != [i['prediction']['codes'] for i in items]:
        raise ValueError('initial repair must exactly preserve every actual input code')
    write_json(DEST / 'initial_outputs.json', initial)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trace = []
    torch.cuda.reset_peak_memory_stats()
    for step in (1, 2):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for codes, features, mask, target in tensors:
            logits = model(codes, features, mask, books)
            ce = repair.level_loss(logits, target, c.levels)
            soft = geometry.soft_latent(codec, torch.cat(logits, dim=1))
            desired = codec.vq.idx_to_z_q(target)
            mse = (soft - desired).square().mean()
            loss = (ce + .05 * mse) / 2
            if not torch.isfinite(loss):
                raise ValueError('nonfinite real-geometry attention repair objective')
            loss.backward()
            losses.append({'code_CE': float(ce.detach()), 'soft_latent_MSE': float(mse.detach())})
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        source_gradient = float(model.source_kv.weight.grad.abs().sum())
        if (not torch.isfinite(norm) or model.delta.weight.grad.abs().sum() <= 0
                or step == 2 and source_gradient <= 0
                or any(f.grad is not None for _, f, _, _ in tensors)
                or any(v.grad is not None for v in codec.parameters())):
            raise ValueError('finite corrector gradients and protected input/books required')
        opt.step()
        trace.append({'step': step, 'losses': losses, 'gradient_norm': float(norm), 'source_gradient_L1': source_gradient})
        write_json(DEST / 'trace.json', trace)
        print(json.dumps(trace[-1]), flush=True)
    final_hash = native.vendor.module_hash(model)
    final = outputs(model, items, tensors, books)
    saved = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer': opt.state_dict(), 'CPU_rng': torch.get_rng_state(),
        'CUDA_rng': torch.cuda.get_rng_state_all(), 'config': asdict(c), 'step': 2,
        'producer_fingerprint': producer['fingerprint']}
    torch.save(saved, DEST / 'model.pt')
    del model, opt, saved, logits, soft, ce, mse, loss
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(DEST / 'model.pt', map_location='cpu', weights_only=True)
    model = repair.AttentionCodeRepair(c).cuda()
    model.load_state_dict(saved['model'], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['CPU_rng'])
    torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    same = original.training.lifecycle.same
    if (outputs(model, items, tensors, books) != final
            or native.vendor.module_hash(model) != final_hash
            or not same(model.state_dict(), saved['model']) or not same(opt.state_dict(), saved['optimizer'])
            or not torch.equal(torch.get_rng_state(), saved['CPU_rng'])
            or not same(torch.cuda.get_rng_state_all(), saved['CUDA_rng'])
            or native.vendor.module_hash(codec) != codec_hash or final_hash == initial_hash):
        raise ValueError('strict model/optimizer/RNG/output reload or frozen codec failed')
    peak = torch.cuda.max_memory_allocated()
    if peak > 3_000_000_000:
        raise ValueError('repair lifecycle exceeds3GB allocation limit')
    write_json(DEST / 'outputs.json', final)
    native.vendor.verify(producer)
    result = {'status': 'attention_code_repair_two_TRAIN_updates_and_strict_reload_passed',
        'hypothesis_id': HYPOTHESIS, 'updates': 2, 'cases': 2,
        'trainable_parameters': sum(v.numel() for v in model.parameters()),
        'initial_model_sha256': initial_hash, 'final_model_sha256': final_hash,
        'frozen_codec_sha256': codec_hash, 'frozen_codec_unchanged': True,
        'strict_model_optimizer_RNG_and_hard_output_reload': True,
        'checkpoint_sha256': file_hash(DEST / 'model.pt'), 'outputs_sha256': file_hash(DEST / 'outputs.json'),
        'peak_allocated_bytes': peak, 'all_cases_TRAIN_only': True,
        'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False}
    write_json(DEST / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
