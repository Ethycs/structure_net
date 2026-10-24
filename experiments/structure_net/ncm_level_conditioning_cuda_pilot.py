"""Two-update FP16 systems pilot on spare GPU2; not summary-quality evidence."""
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_native_source_training as previous
from experiments.structure_net import ncm_level_conditioning_objective as objective
from experiments.structure_net.ncm_lm1b_decoder_repair import same

capacity, lifecycle, native = previous.capacity, previous.lifecycle, previous.native
q, ROOT = previous.q, previous.ROOT
DEST = ROOT / '20261024_level_conditioning_FP16_GPU2_pilot_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-level-conditioning-cuda-pilot.md')
TESTS = [Path('tests/structure_net/test_ncm_level_conditioning_objective.py'),
    Path('tests/structure_net/test_ncm_level_conditioning_corruption.py'),
    Path('tests/structure_net/test_ncm_level_conditioning_cuda_pilot.py')]
WIDTH = 3777


def selected():
    rows = q.read(previous.DATA / 'cases.json')
    manifest = q.read(previous.DATA / 'manifest.json')
    result = [rows[i] for i in q.read(previous.DATA / 'selection.json')['pilot_indices']]
    codes = {r['id']: r['codes'] for r in q.read(previous.training.CACHE / 'outputs.json')}
    if len(result) != 2 or len({r['id'] for r in result}) != 2:
        raise ValueError('original distinct first/longest TRAIN sources required')
    tensors = previous.training.load_features(result, manifest)
    return result, manifest, codes, tensors


def batch(rows, index, codes, features):
    row = rows[index]
    source = features[row['id']]
    if len(source) > WIDTH:
        raise ValueError('complete source exceeds fixed systems frame')
    value = F.pad(source, (0, 0, 0, WIDTH - len(source))).unsqueeze(0).cuda()
    mask = (torch.arange(WIDTH) < len(source)).unsqueeze(0).cuda()
    return (value, mask, torch.tensor([codes[row['id']]], device='cuda', dtype=torch.long),
        torch.tensor([row['native_word_ids']], device='cuda', dtype=torch.long))


@torch.no_grad()
def outputs(model, rows, codes, features):
    model.eval()
    result = []
    for index in range(2):
        x, mask, _, _ = batch(rows, index, codes, features)
        result.append({k: v.cpu().tolist() for k, v in model.generate(x, mask).items()})
    return result


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or not torch.cuda.is_available() or '2060 SUPER' not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 5_000_000_000):
        raise ValueError('physicalGPU2 RTX2060SUPER with5GBfree required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    native.vendor.terminal(previous.DATA)
    rows, manifest, codes, features = selected()
    files = [Path(__file__), PLAN, *TESTS, Path(objective.__file__), Path(objective.corruption.__file__),
        Path(objective.words.__file__), Path(capacity.__file__), Path(lifecycle.mechanism.__file__),
        Path(lifecycle.head.__file__), Path(lifecycle.head.concepts.__file__), Path(previous.training.__file__),
        Path(same.__code__.co_filename), *sorted(native.vendor.VENDOR.rglob('*.py'))]
    inputs = [previous.DATA / n for n in ('cases.json', 'manifest.json', 'selection.json')]
    inputs += [previous.training.CACHE / 'outputs.json', lifecycle.WARM / 'model.pt', lifecycle.head.DOMAIN_CHECKPOINT,
        *[Path(manifest[r['id']]['path']) for r in rows],
        *[native.RELEASE / 'files' / name for name in native.FILES], native.RELEASE / 'files/lm1b/small/config.yaml']
    p = native.vendor.freeze(DEST, 'two_update_FP16_level_conditioning_systems_check', inputs + files,
        {'systems_lifecycle_only_not_quality_evidence': True, 'source_cases': 2, 'updates': 2,
            'microbatch': 1, 'accumulation': 2, 'padded_source_positions': WIDTH,
            'padding_not_source_clipping': True, 'clean_targets_unchanged': True,
            'trainable_parameters': 127732736, 'original_initializer': capacity.INITIAL,
            'new_random_weights': False, 'noise_rates': [.75, .25], 'noise_includes_generated_root': True,
            'noise_endpoints_our_choice_not_exact_author_training_recipe': True,
            'precision': 'FP32weights/FP16training/FP32source_only_hard_generation',
            'scaler_initial': 128., 'scaler_growth_interval': 2000, 'TF32': False,
            'optimizer': 'AdamW1e-5,.9/.95,eps1e-8,wd.01,clip1', 'neighbor_seed': 124142,
            'dropout_seed': 125142, 'physical_GPU': 2, 'peak_limit_bytes': 4_900_000_000,
            'GPU0_body_cache_job_interrupted': False, 'Llama_body_loaded_or_trained': False,
            'pilot_not_quality_training_initializer': True, 'task_accuracy': None,
            'new_semantic_judgments': False, 'final_test_opened': False, 'speed_tested': False})
    for index, path in enumerate(files):
        p['source_hashes'][str(path)] = previous.lifecycle.file_hash(path)
        (DEST / 'producer_source' / f'level_pilot_{index}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    lifecycle.write_json(DEST / 'producer.json', p)
    torch.cuda.reset_peak_memory_stats()
    model = capacity.fresh()
    initial = lifecycle.trainable_hash(model)
    initial_native = {'generator': native.vendor.module_hash(model.concepts.generator),
        'word_decoder': native.vendor.module_hash(model.codec.decoder)}
    initial_outputs = outputs(model, rows, codes, features)
    lifecycle.write_json(DEST / 'initial_outputs.json', initial_outputs)
    root_before = model.concepts.generator.l0_queries.detach().cpu().clone()
    opt = lifecycle.optimizer(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    neighbor = torch.Generator().manual_seed(124142)
    torch.manual_seed(125142)
    torch.cuda.manual_seed_all(125142)
    trace = []
    for step in (1, 2):
        model.train()
        opt.zero_grad(set_to_none=True)
        micro = []
        for index in (0, 1):
            x, mask, c, w = batch(rows, index, codes, features)
            result = objective.backward_microbatch(model, x, mask, c, w, neighbor,
                accumulation=2, self_history=index == 0, dtype=torch.float16, scaler=scaler)
            micro.append({'id': rows[index]['id'], 'valid_source_positions': int(mask.sum()), 'losses': result})
        scaler.unscale_(opt)
        params = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for _, v in params):
            raise ValueError('FP16pilot nonfinite unscaled gradients; no silent precision/scale retry')
        active = {group: any(n.startswith(group) and v.grad is not None and bool(v.grad.abs().sum() > 0)
            for n, v in params) for group in lifecycle.GROUPS}
        if (not all(active.values()) or any(v.grad is not None
                for module in (model.codec.encoder, model.codec.vq) for v in module.parameters())):
            raise ValueError('native root/generator/decoder/source gradient or frozen-geometry contract differs')
        norm = torch.nn.utils.clip_grad_norm_([v for _, v in params], 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite FP16pilot norm')
        scaler.step(opt)
        scaler.update()
        trace.append({'step': step, 'gradient_norm': float(norm), 'active_groups': active,
            'scaler': scaler.state_dict(), 'microbatches': micro})
        lifecycle.write_json(DEST / 'trace.json', trace)
        print(json.dumps({'updates': step, 'FP16_scale': scaler.get_scale(), 'gradient_norm': float(norm)}), flush=True)
    del x, mask, c, w, params
    expected = outputs(model, rows, codes, features)
    final = lifecycle.trainable_hash(model)
    native_after = {'generator': native.vendor.module_hash(model.concepts.generator),
        'word_decoder': native.vendor.module_hash(model.codec.decoder)}
    if (initial != capacity.INITIAL or final == initial or any(native_after[k] == initial_native[k] for k in native_after)
            or torch.equal(root_before, model.concepts.generator.l0_queries.detach().cpu())
            or lifecycle.protected(model) != {'encoder': lifecycle.ENCODER, 'VQ': lifecycle.VQ}):
        raise ValueError('actual native/attention learning or protected geometry differs')
    saved = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'step': 2,
        'CPU_rng': torch.get_rng_state(), 'CUDA_rng': torch.cuda.get_rng_state_all(),
        'neighbor_rng': neighbor.get_state(), 'producer_fingerprint': p['fingerprint']}
    torch.save(saved, DEST / 'model.pt')
    model.close()
    del model, opt, scaler, saved
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(DEST / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model = capacity.fresh()
    model.load_state_dict(saved['model'], strict=True)
    opt = lifecycle.optimizer(model)
    opt.load_state_dict(saved['optimizer'])
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    scaler.load_state_dict(saved['scaler'])
    torch.set_rng_state(saved['CPU_rng'])
    torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    neighbor.set_state(saved['neighbor_rng'])
    if (outputs(model, rows, codes, features) != expected or not same(model.state_dict(), saved['model'])
            or not same(opt.state_dict(), saved['optimizer']) or scaler.state_dict() != saved['scaler']
            or not same(torch.get_rng_state(), saved['CPU_rng']) or not same(torch.cuda.get_rng_state_all(), saved['CUDA_rng'])
            or not same(neighbor.get_state(), saved['neighbor_rng']) or lifecycle.trainable_hash(model) != final
            or lifecycle.protected(model) != {'encoder': lifecycle.ENCODER, 'VQ': lifecycle.VQ}):
        raise ValueError('strict model/optimizer/scaler/RNG/source-only generation reload failed')
    peak = torch.cuda.max_memory_allocated()
    if peak > 4_900_000_000:
        raise ValueError('FP16systems pilot exceeds4.9GBplan')
    lifecycle.write_json(DEST / 'reload_outputs.json', expected)
    native.vendor.verify(p)
    result = {'status': 'FP16_level_conditioning_two_update_lifecycle_passed', 'updates': 2,
        'initial_trainable_sha256': initial, 'final_trainable_sha256': final,
        'trainable_parameters': 127732736, 'native_after': native_after,
        'checkpoint_sha256': lifecycle.file_hash(DEST / 'model.pt'), 'peak_allocated_bytes': peak,
        'scaler': scaler.state_dict(), 'strict_model_optimizer_scaler_rng_hard_output_reload': True,
        'generated_root_updated': True, 'protected_codec_hashes': lifecycle.protected(model),
        'FP16_variant_not_main_BF16_quality_run': True, 'task_accuracy': None,
        'speed_tested': False, 'final_test_opened': False, 'producer_fingerprint': p['fingerprint']}
    lifecycle.write_json(DEST / 'result.json', result)
    model.close()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
