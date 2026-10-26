"""Matched reviewed-corpus NCM depth pilots and fixed-budget training."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import sys
import time
import types

import torch

from experiments.structure_net import ncm_depth
from experiments.structure_net import ncm_onpolicy_residual_training as parent
from experiments.structure_net import ncm_public_batch4_deterministic as deterministic
from experiments.structure_net import ncm_example_batching as batching
from experiments.structure_net import ncm_trainable_example_batching as serial

broad = parent.broad
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-depth-plan.md')
TEST = Path('tests/structure_net/test_ncm_depth.py')
ROOTS = {(d, s): broad.runtime.ROOT/f'20261026_ncm_depth_{d}_{s}_v1'
         for d in (12, 24) for s in ('pilot', 'train')}


@dataclass(frozen=True)
class Config:
    updates: int = 1024
    accumulation: int = 4
    microbatch: int = 2
    warmup: int = 64
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    sampler_seed: int = 241142
    corruption_seed: int = 242142
    dropout_seed: int = 243142
    peak_limit_bytes: int = 5_200_000_000
    host_feature_cache_bytes: int = 256*1024**2


CONFIG = Config()


def memory(root, phase):
    torch.cuda.synchronize()
    value = {'phase': phase, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'allocated_bytes': torch.cuda.memory_allocated()}
    with (root/'memory.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    if value['peak_allocated_bytes'] > CONFIG.peak_limit_bytes:
        raise ValueError('fixed depth experiment GPU allocation limit exceeded')
    return value


def freeze(root, depth, stage, training, environment):
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    paths = [parent.PARENT/n for n in ('model.pt', 'result.json', 'actual_tool_receipt.json')]
    paths += [broad.RUNS['prepare']/n for n in ('cases.json', 'selection.json', 'oracle_outputs.json',
                                              'producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [broad.catalog.DEST/n for n in ('source_index.json', 'feature_manifest.json')]
    paths += [Path('pyproject.toml'), Path('pixi.lock')]
    if stage == 'train':
        paths += [ROOTS[d, 'pilot']/n for d in (12, 24)
                  for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-depth.v1', 'hypothesis_id': 'ncm-generator-depth-v1',
             'stage': stage, 'depth': depth, 'conformance': 'EXPLORATORY_UNDERPOWERED',
             'config': asdict(CONFIG), 'environment': environment,
             'source_hashes': {str(p): broad.file_hash(p) for p in sorted(files)},
             'input_hashes': {str(p): broad.file_hash(p) for p in paths}}
    proof['fingerprint'] = training.q.digest(proof)
    broad.write_json(root/'producer.json', proof)
    return proof


def save(path, model, opt, scaler, sampler, noise, permutation, step, proof):
    if path.exists():
        raise ValueError('checkpoint overwrite forbidden')
    value = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
             'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
             'sampler_rng': sampler.get_state(), 'noise_rng': noise.get_state(),
             'CPU_rng': torch.get_rng_state(), 'CUDA_rng': torch.cuda.get_rng_state_all(),
             'permutation': list(permutation), 'step': step, 'depth': proof['depth'],
             'producer_fingerprint': proof['fingerprint']}
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary); temporary.rename(path)


def restore(path, model, opt, scaler, sampler, noise, permutation):
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng']); noise.set_state(saved['noise_rng'])
    permutation[:] = saved['permutation']
    torch.set_rng_state(saved['CPU_rng']); torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    return saved


@torch.no_grad()
def identity_controls(model, rows, loader, training):
    broad.mechanism.configure(model, False)
    results = []
    for row in rows:
        f, m, gold = training.prior.tensors(row, loader)
        logits = serial.teacher_logits(model.concepts, f, m, gold).cpu()
        codes = broad.mechanism.generate(model, f, m)
        words = model.word_logits(f, m, codes).argmax(-1)
        results.append((logits, codes.cpu(), words.cpu()))
    return results


def update(model, rows, loader, opt, scaler, noise, step, info):
    broad.mechanism.configure(model, True)
    opt.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    rate = broad.rate(step, CONFIG)
    for group in opt.param_groups:
        group['lr'] = rate
    records = []
    for offset in (0, 2):
        chosen = rows[offset:offset+2]
        features, mask, gold = batching.collate([loader.get(r) for r in chosen],
            [r['oracle_codes'] for r in chosen], device='cuda')
        histories, changes = batching.histories(gold, model.concepts.generator.levels, 8192,
                                               noise, [False, True])
        with torch.autocast('cuda', dtype=torch.float16):
            logits = serial.teacher_logits(model.concepts, features, mask, histories)
        loss = batching.weighted_loss(logits, gold, model.concepts.generator.levels, examples_in_update=4)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        scaler.scale(loss).backward()
        records.append({'ids': [r['id'] for r in chosen], 'changes': changes,
                        'weighted_loss': float(loss.detach())})
        del features, mask, gold, histories, logits, loss
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    if not torch.isfinite(norm) or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError('nonfinite or protected gradient')
    root = model.concepts.generator.l0_queries.grad
    if root is None or not root.abs().sum():
        raise ValueError('root queries must learn')
    if step == 1:
        for index in info['inserted']:
            block = model.concepts.generator.transformer.blocks[index]
            for projection in (block.attn.c_proj, block.mlp.c_proj):
                if projection.weight.grad is None or not projection.weight.grad.abs().sum():
                    raise ValueError('added residual output projection has no gradient')
    previous = scaler.get_scale()
    scaler.step(opt); scaler.update(); broad.offload.assert_cpu_state(opt)
    if scaler.get_scale() < previous:
        raise ValueError('skipped optimizer update')
    opt.zero_grad(set_to_none=True)
    return {'step': step, 'lr': rate, 'gradient_norm': float(norm), 'scale': scaler.get_scale(),
            'microbatches': records, 'mean_CE': sum(r['weighted_loss'] for r in records)}


def run(depth, stage):
    root = ROOTS[depth, stage]
    if root.exists():
        raise ValueError('fresh append-only depth run required')
    deterministic.configure_runtime()
    training, environment = broad.runtime.configure()
    broad.offload.require_rtx2060()
    training.preparation.completed(broad.RUNS['prepare'])
    for path, expected in training.q.read(broad.RUNS['prepare']/'result.json')['artifacts'].items():
        if broad.file_hash(path) != expected:
            raise ValueError('reviewed TRAIN data changed')
    if stage == 'train':
        for d in (12, 24):
            training.preparation.completed(ROOTS[d, 'pilot'])
            result = training.q.read(ROOTS[d, 'pilot']/'result.json')
            if not result['strict_resume_passed'] or not result['initial_predictions_identical']:
                raise ValueError('both depth lifecycle pilots must pass')
    rows = training.q.read(broad.RUNS['prepare']/'cases.json')
    selection = training.q.read(broad.RUNS['prepare']/'selection.json')
    lookup = {r['id']: r for r in rows}
    if len(rows) != 1376 or len(lookup) != 1376:
        raise ValueError('all 1376 reviewed TRAIN examples required')
    index, manifest = [training.q.read(broad.catalog.DEST/n) for n in ('source_index.json', 'feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=CONFIG.host_feature_cache_bytes)
    print(json.dumps({'depth': depth, 'stage': stage, 'status': 'loading_BROAD_parent'}), flush=True)
    model = parent.load_initial(training)
    model.concepts.teacher_logits = types.MethodType(serial.teacher_logits, model.concepts)
    frozen = training.prior.protected(model)
    control_ids = list(dict.fromkeys([selection['first_id'], selection['longest_source_id'],
                                     *selection['diagnostic_ids']]))[:4]
    controls = [lookup[i] for i in control_ids]
    if len(controls) != 4:
        raise ValueError('four distinct pilot TRAIN controls required')
    before = identity_controls(model, controls, loader, training)
    info = ncm_depth.expand(model, depth)
    after = identity_controls(model, controls, loader, training)
    same = training.preparation.checkpoint.training.lifecycle.same
    if not same(before, after) or training.prior.protected(model) != frozen:
        raise ValueError('depth expansion changed initial predictions or frozen tensors')
    del before, after
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    proof = freeze(root, depth, stage, training, environment)
    broad.write_json(root/'architecture.json', {**info, 'initial_predictions_identical': True,
        'unchanged_code_levels': list(model.concepts.generator.levels)})
    initial_hash = training.native.vendor.module_hash(model)
    broad.mechanism.configure(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params)
    opt = broad.offload.CPUAdamW(params, lr=CONFIG.max_lr, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    sampler = torch.Generator().manual_seed(CONFIG.sampler_seed)
    noise = torch.Generator().manual_seed(CONFIG.corruption_seed)
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    permutation = []
    updates = 2 if stage == 'pilot' else CONFIG.updates
    print(json.dumps({'depth': depth, 'stage': stage, 'trainable_parameters': count,
                      'initial_predictions_identical': True, 'updates': updates}), flush=True)
    started = time.monotonic()
    with (root/'trace.jsonl').open('x') as trace:
        for step in range(1, updates+1):
            chosen = []
            for i in range(4):
                if stage == 'pilot':
                    chosen.append(controls[i])
                else:
                    if not permutation:
                        permutation.extend(torch.randperm(len(rows), generator=sampler).tolist())
                    chosen.append(rows[permutation.pop()])
            metric = update(model, chosen, loader, opt, scaler, noise, step, info)
            peak = memory(root, 'update_'+str(step))['peak_allocated_bytes']
            record = {**metric, 'depth': depth, 'elapsed_seconds': time.monotonic()-started, 'peak_bytes': peak}
            trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush()
            if step <= 2 or step % 16 == 0:
                broad.write_json(root/'progress.json', record)
                print(json.dumps({k: v for k, v in record.items() if k != 'microbatches'}), flush=True)
            if (stage == 'pilot' and step == 1) or (stage == 'train' and step % 256 == 0 and step != updates):
                save(root/f'checkpoint_{step}.pt', model, opt, scaler, sampler, noise, permutation, step, proof)
    elapsed = time.monotonic()-started
    final_hash = training.native.vendor.module_hash(model)
    if training.prior.protected(model) != frozen or initial_hash == final_hash:
        raise ValueError('frozen tensors changed or trainable model did not update')
    save(root/'model.pt', model, opt, scaler, sampler, noise, permutation, updates, proof)
    tokenizer = training.native.tokenizer()
    diagnostics = training.prior.diagnostics(model, controls, loader, tokenizer)
    broad.write_json(root/'outputs.json', diagnostics)
    resumed = False
    if stage == 'pilot':
        restore(root/'checkpoint_1.pt', model, opt, scaler, sampler, noise, permutation)
        replay = update(model, chosen, loader, opt, scaler, noise, 2, info)
        broad.write_json(root/'resumed_update_2.json', replay)
        expected = torch.load(root/'model.pt', map_location='cpu', weights_only=True, mmap=True)
        comparisons = {'model': same(model.state_dict(), expected['model']),
            'optimizer': same(opt.state_dict(), expected['optimizer']),
            'scaler': same(scaler.state_dict(), expected['scaler']),
            'sampler_rng': torch.equal(sampler.get_state(), expected['sampler_rng']),
            'noise_rng': torch.equal(noise.get_state(), expected['noise_rng']),
            'CPU_rng': torch.equal(torch.get_rng_state(), expected['CPU_rng']),
            'CUDA_rng': same(torch.cuda.get_rng_state_all(), expected['CUDA_rng']),
            'permutation': permutation == expected['permutation'], 'update_metrics': replay == metric}
        del expected
        broad.write_json(root/'resumed_state_equality.json', comparisons)
        if not all(comparisons.values()):
            raise ValueError('exact second-update resume failed')
        resumed = True
    restore(root/'model.pt', model, opt, scaler, sampler, noise, permutation)
    if (training.native.vendor.module_hash(model) != final_hash
            or training.prior.diagnostics(model, controls, loader, tokenizer) != diagnostics
            or training.prior.protected(model) != frozen):
        raise ValueError('strict reload or frozen tensor equality failed')
    peak = memory(root, 'strict_reload')['peak_allocated_bytes']
    training.native.vendor.verify(proof)
    result = {'schema': 'nal.ncm-depth.v1', 'status': 'depth_'+stage+'_completed_semantic_quality_pending',
        'depth': depth, 'updates': updates, 'TRAIN_cases': len(rows), 'trainable_parameters': count,
        'initial_predictions_identical': True, 'strict_reload_passed': True, 'strict_resume_passed': resumed,
        'frozen_hashes': frozen, 'initial_model_sha256': initial_hash, 'final_model_sha256': final_hash,
        'checkpoint_sha256': broad.file_hash(root/'model.pt'), 'training_seconds': elapsed,
        'peak_allocated_bytes': peak, 'TRAIN_controls': training.prior.summarize(diagnostics),
        'feature_cache': loader.metrics(), 'task_accuracy': None, 'final_test_opened': False,
        'producer_fingerprint': proof['fingerprint']}
    broad.write_json(root/'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('depth', type=int, choices=(12, 24))
    parser.add_argument('stage', choices=('pilot', 'train'))
    args = parser.parse_args()
    root = ROOTS[args.depth, args.stage]
    existed = root.exists()
    try:
        run(args.depth, args.stage)
    except Exception as error:
        if not existed and root.exists():
            broad.write_json(root/'failure.json', {'exception': type(error).__name__, 'message': str(error),
                                                  'task_accuracy': None})
        raise
