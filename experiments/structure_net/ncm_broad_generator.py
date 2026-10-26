"""Train the source-conditioned generator on all guarded complete-summary firsts."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch

from experiments.structure_net import ncm_pixi as runtime
from experiments.structure_net import ncm_generated_code_decoder as parent
from experiments.structure_net import ncm_broad_teacher_catalog as catalog
from experiments.structure_net import ncm_packed_sequential_objective as objective
from experiments.structure_net import ncm_sequential_levels as mechanism
from experiments.structure_net import ncm_cpu_adamw as offload
from experiments.structure_net.ncm_memory_data import file_hash, write_json

SCHEMA_VERSION = 'nal.ncm-broad-generator.v1'
HYPOTHESIS_ID = 'ncm-broad-generator-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-broad-generator-plan.md')
TEST = Path('tests/structure_net/test_ncm_broad_generator.py')
RUNS = {s: runtime.ROOT / ('20261025_broad_generator_' + s + '_v1')
        for s in ('prepare', 'pilot', 'train')}
CASES_SHA = 'cd3d2a8a1fe0d398d1ca951e0df382d05c47cefe4d07892107b23d6e1c7e5914'


@dataclass(frozen=True)
class Config:
    updates: int = 4096
    accumulation: int = 4
    warmup: int = 128
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    sampler_seed: int = 181142
    corruption_seed: int = 182142
    dropout_seed: int = 183142
    peak_limit_bytes: int = 4_900_000_000
    host_feature_cache_bytes: int = 256 * 1024**2
    precision: str = 'FP32weights_FP16training_FP32evaluation'
    optimizer: str = 'CPU_FP32_AdamW_betas0.9_0.95_eps1e-8_wd0.01_clip1'
    TF32: bool = False
    physical_GPU: int = 1
    codec_and_word_source_attention_frozen: bool = True
    final_test_opened: bool = False
    speed_tested: bool = False


CONFIG = Config()


def rate(step, config=CONFIG):
    if not 1 <= step <= config.updates or not 0 < config.warmup < config.updates:
        raise ValueError('step/warmup outside fixed budget')
    if step <= config.warmup:
        return config.max_lr * step / config.warmup
    return config.min_lr + .5 * (config.max_lr-config.min_lr) * (
        1 + math.cos(math.pi*(step-config.warmup)/(config.updates-config.warmup)))


def select_controls(rows, index):
    by_id = {r['id']: r for r in rows}
    lengths = {r['id']: r['source_positions'] for r in index}
    if len(by_id) != len(rows) or set(by_id) != set(lengths) or len(rows) < 16:
        raise ValueError('complete unique source bindings and at least16 rows required')
    longest = max(by_id, key=lambda k: (lengths[k], k))
    selected = list(dict.fromkeys([*sorted(by_id)[:15], longest]))
    selected += [k for k in sorted(by_id) if k not in selected][:16-len(selected)]
    return {'diagnostic_ids': selected, 'longest_source_id': longest,
            'first_id': rows[0]['id'],
            'longest_target_id': max(rows, key=lambda r: (r['native_lexical_tokens'], r['id']))['id']}


def bind_codes(row, codes):
    if (row['split'] != 'TRAIN' or len(codes) != 641
            or any(type(x) is not int or not 0 <= x < 8192 for x in codes)):
        raise ValueError('TRAIN-only complete integer hierarchy required')
    if len(row['native_word_ids']) != 128 or catalog.text_hash(row['target_text']) != row['target_sha256']:
        raise ValueError('unaltered full target frame/text required')
    return {**row, 'oracle_codes': codes}


def freeze(root, stage, training, environment, extra=()):
    files = {Path(__file__), PLAN, TEST}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and (
                    path.is_relative_to(Path('experiments').resolve())
                    or path.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(path)
    files.update(training.native.vendor.VENDOR.rglob('*.py'))
    inputs = [catalog.DEST / n for n in ('cases.json', 'source_index.json', 'feature_manifest.json', 'result.json')]
    inputs += [catalog.AUDIT / n for n in ('result.json', 'actual_tool_receipt.json')]
    inputs += [parent.PARENT / n for n in ('model.pt', 'result.json', 'actual_tool_receipt.json')]
    inputs += [Path('pyproject.toml'), Path('pixi.lock'), *extra]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, path in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema': SCHEMA_VERSION, 'hypothesis_id': HYPOTHESIS_ID,
             'conformance': 'EXPLORATORY_UNDERPOWERED', 'stage': stage,
             'config': asdict(CONFIG), 'parent_checkpoint_sha256': parent.PARENT_SHA,
             'actual_updates': {'prepare': 0, 'pilot': 2, 'train': CONFIG.updates}[stage],
             'systems_lifecycle_only_not_quality_evidence': stage == 'pilot',
             'source_hashes': {str(p): file_hash(p) for p in sorted(files)},
             'input_hashes': {str(p): file_hash(p) for p in dict.fromkeys(inputs)},
             'environment': environment}
    proof['fingerprint'] = training.q.digest(proof)
    write_json(root/'producer.json', proof)
    return proof


def memory(root, phase):
    torch.cuda.synchronize()
    row = {'phase': phase, 'allocated_bytes': torch.cuda.memory_allocated(),
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    with (root/'memory.jsonl').open('a') as output:
        output.write(json.dumps(row, allow_nan=False)+'\n')
    if row['peak_allocated_bytes'] > CONFIG.peak_limit_bytes:
        raise ValueError('whole lifecycle CUDA allocation budget exceeded')
    return row


@torch.no_grad()
def oracle(model, rows, loader):
    model.eval().requires_grad_(False)
    out = []
    for row in rows:
        codes = torch.tensor([row['oracle_codes']], device='cuda')
        f = loader.get(row).unsqueeze(0).cuda()
        mask = torch.ones(f.shape[:2], device=f.device, dtype=torch.bool)
        bare = model.codec.decoder(model.codec.vq.idx_to_z_q(codes)).argmax(-1)[0].cpu()
        bound = model.word_logits(f, mask, codes).argmax(-1)[0].cpu()
        gold = torch.tensor(row['native_word_ids'])
        out.append({'id': row['id'], 'codec_only_raw_ids': bare.tolist(),
                    'with_source_raw_ids': bound.tolist(), 'active_words': int((gold != 0).sum()),
                    'codec_only_matches': int(((bare == gold) & (gold != 0)).sum()),
                    'with_source_matches': int(((bound == gold) & (gold != 0)).sum())})
    return out


def prepare(training, environment):
    if file_hash(catalog.DEST/'cases.json') != CASES_SHA:
        raise ValueError('fixed complete-summary catalog changed')
    training.preparation.completed(catalog.AUDIT)
    audit = catalog.read(catalog.AUDIT/'result.json')
    if audit['cases'] != 1376 or audit['status'] != 'all_catalog_feature_payloads_replayed':
        raise ValueError('all1376 original feature payloads must be audited')
    state = catalog.read(catalog.DEST/'result.json')
    for path, expected in state['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('catalog identity changed')
    rows, index, manifest = [catalog.read(catalog.DEST/n) for n in ('cases.json', 'source_index.json', 'feature_manifest.json')]
    if len(rows) != 1376:
        raise ValueError('entire admitted TRAIN catalog required')
    selection = select_controls(rows, index)
    root = RUNS['prepare']
    model = parent.load_parent(training)
    proof = freeze(root, 'prepare', training, environment)
    model.eval().requires_grad_(False)
    tokenizer = training.native.tokenizer()
    encoded = []
    repeated = {selection[k] for k in ('first_id', 'longest_source_id', 'longest_target_id')}
    with torch.no_grad():
        for row in rows:
            lexical = tokenizer.encode(row['target_text'], add_special_tokens=False)
            expected = [101, *lexical, 101] + [0] * (126-len(lexical))
            if row['native_word_ids'] != expected or not 0 < len(lexical) <= 126:
                raise ValueError('complete native target frame changed')
            gold = torch.tensor([expected], device='cuda')
            codes = model.codec.encode(gold)
            if row['id'] in repeated and not torch.equal(codes, model.codec.encode(gold)):
                raise ValueError('target encoding is not repeatable')
            encoded.append(bind_codes(row, codes[0].cpu().tolist()))
            if len(encoded) % 128 == 0:
                print(json.dumps({'target_codes_prepared': len(encoded), 'total': len(rows)}), flush=True)
    loader = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=CONFIG.host_feature_cache_bytes)
    clean = oracle(model, encoded, loader)
    if training.native.vendor.module_hash(model) != parent.MODEL_SHA:
        raise ValueError('target preparation changed parent tensors')
    for name, value in [('cases.json', encoded), ('selection.json', selection), ('oracle_outputs.json', clean)]:
        write_json(root/name, value)
    memory(root, 'prepared_all_codes_and_oracles')
    training.native.vendor.verify(proof)
    result = {'schema': SCHEMA_VERSION, 'status': 'all1376_targets_and_oracles_prepared',
              'TRAIN_cases': len(encoded), 'code_positions': len(encoded)*641,
              'cases_sha256': file_hash(root/'cases.json'),
              'artifacts': {str(root/n): file_hash(root/n) for n in ('cases.json', 'selection.json', 'oracle_outputs.json', 'producer.json')},
              'feature_cache': loader.metrics(), 'task_accuracy': None, 'model_training': False,
              'parent_model_sha256': parent.MODEL_SHA, 'final_test_opened': False}
    write_json(root/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)
    model.close()


def train(stage, training, environment):
    training.preparation.completed(RUNS['prepare'])
    ready = catalog.read(RUNS['prepare']/'result.json')
    for path, expected in ready['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('prepared TRAIN targets changed')
    extra = [RUNS['prepare']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'cases.json', 'oracle_outputs.json', 'selection.json')]
    if stage == 'train':
        training.preparation.completed(RUNS['pilot'])
        pilot = catalog.read(RUNS['pilot']/'result.json')
        if not pilot['strict_reload_passed'] or not pilot['gradient_parity_passed'] or not pilot['oracle_unchanged']:
            raise ValueError('complete passing lifecycle/parity/oracle pilot required')
        extra += [RUNS['pilot']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    rows = catalog.read(RUNS['prepare']/'cases.json')
    selection = catalog.read(RUNS['prepare']/'selection.json')
    lookup = {r['id']: r for r in rows}
    controls = [lookup[i] for i in selection['diagnostic_ids']]
    if len(rows) != 1376 or len(lookup) != 1376:
        raise ValueError('all1376 TRAIN cases required')
    index, manifest = [catalog.read(catalog.DEST/n) for n in ('source_index.json', 'feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=CONFIG.host_feature_cache_bytes)
    root = RUNS[stage]
    model = parent.load_parent(training)
    proof = freeze(root, stage, training, environment, extra)
    memory(root, 'model_loaded')
    frozen = training.prior.protected(model)
    parity = []
    if stage == 'pilot':
        for key, noisy in (('first_id', False), ('longest_source_id', True)):
            f, m, gold = training.prior.tensors(lookup[selection[key]], loader)
            parity.append(objective.gradient_parity(model, f, m, gold, noisy))
            memory(root, 'gradient_parity_'+key)
        write_json(root/'gradient_parity.json', parity)
    tokenizer = training.native.tokenizer()
    initial = training.prior.diagnostics(model, controls, loader, tokenizer)
    write_json(root/'initial_outputs.json', initial)
    if stage == 'train' and initial != catalog.read(RUNS['pilot']/'initial_outputs.json'):
        raise ValueError('pilot/full initial inference differs')
    mechanism.configure(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = offload.CPUAdamW(params, lr=CONFIG.max_lr, betas=(.9,.95), eps=1e-8, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    sampler = torch.Generator().manual_seed(CONFIG.sampler_seed)
    noise = torch.Generator().manual_seed(CONFIG.corruption_seed)
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    permutation = []
    updates = 2 if stage == 'pilot' else CONFIG.updates
    started = time.monotonic()
    with (root/'trace.jsonl').open('x') as trace:
        for step in range(1, updates+1):
            mechanism.configure(model, True); opt.zero_grad(set_to_none=True)
            for group in opt.param_groups: group['lr'] = rate(step)
            metrics = []
            for micro in range(CONFIG.accumulation):
                if not permutation:
                    permutation = torch.randperm(len(rows), generator=sampler).tolist()
                row = (lookup[selection['longest_source_id' if micro%2 else 'first_id']]
                       if stage == 'pilot' else rows[permutation.pop()])
                f, m, gold = training.prior.tensors(row, loader)
                metric = objective.backward_example(model, f, m, gold, noise, scaler,
                                                     CONFIG.accumulation, bool(micro%2))
                metrics.append({'id': row['id'], **metric})
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm) or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError('nonfinite or frozen gradient')
            if step == 1 and (model.concepts.generator.l0_queries.grad is None
                             or not model.concepts.generator.l0_queries.grad.abs().sum()):
                raise ValueError('root queries must learn')
            old_scale = scaler.get_scale()
            scaler.step(opt); scaler.update(); offload.assert_cpu_state(opt)
            if scaler.get_scale() < old_scale:
                raise ValueError('skipped optimizer update is not a completed update')
            peak = memory(root, 'update_'+str(step))['peak_allocated_bytes']
            record = {'step': step, 'updates': updates, 'lr': rate(step), 'gradient_norm': float(norm),
                      'metrics': metrics, 'mean_CE': sum(x['mean_CE'] for x in metrics)/len(metrics),
                      'elapsed_seconds': time.monotonic()-started, 'peak_bytes': peak}
            trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush()
            if step <= 2 or step%16 == 0:
                write_json(root/'progress.json', record)
                print(json.dumps({k:v for k,v in record.items() if k != 'metrics'}), flush=True)
    elapsed = time.monotonic()-started
    opt.zero_grad(set_to_none=True)
    final_hash = training.native.vendor.module_hash(model)
    if training.prior.protected(model) != frozen or final_hash == parent.MODEL_SHA:
        raise ValueError('protected tensors changed or generator did not update')
    training.prior.save(root/'model.pt', model, opt, scaler, sampler, noise, updates, proof)
    write_json(root/'sampler_remaining.json', permutation)
    write_json(root/'checkpoint_ready.json', {'step': updates, 'model_sha256': final_hash,
               'checkpoint_sha256': file_hash(root/'model.pt'), 'training_seconds': elapsed,
               'producer_fingerprint': proof['fingerprint'], 'evaluation_pending': True})
    final = training.prior.diagnostics(model, controls if stage == 'pilot' else rows, loader, tokenizer)
    write_json(root/'outputs.json', final)
    shuffled = training.prior.diagnostics(model, controls, loader, tokenizer, shuffle=True)
    write_json(root/'shuffled_source_outputs.json', shuffled)
    clean = oracle(model, controls, loader)
    original = {r['id']:r for r in catalog.read(RUNS['prepare']/'oracle_outputs.json')}
    if clean != [original[r['id']] for r in controls]:
        raise ValueError('clean reconstruction changed')
    write_json(root/'oracle_outputs.json', clean)
    del opt; gc.collect(); torch.cuda.empty_cache()
    saved = torch.load(root/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    opt = offload.CPUAdamW(params, lr=CONFIG.max_lr, betas=(.9,.95), eps=1e-8, weight_decay=.01)
    opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng']); noise.set_state(saved['noise_rng'])
    torch.set_rng_state(saved['CPU_rng']); torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    same = training.preparation.checkpoint.training.lifecycle.same
    expected = {r['id']:r for r in final}
    if (not same(model.state_dict(), saved['model']) or not same(opt.state_dict(), saved['optimizer'])
            or not same(scaler.state_dict(), saved['scaler'])
            or not torch.equal(sampler.get_state(), saved['sampler_rng'])
            or not torch.equal(noise.get_state(), saved['noise_rng'])
            or not torch.equal(torch.get_rng_state(), saved['CPU_rng'])
            or not same(torch.cuda.get_rng_state_all(), saved['CUDA_rng'])
            or permutation != catalog.read(root/'sampler_remaining.json')
            or training.prior.diagnostics(model, controls, loader, tokenizer) != [expected[r['id']] for r in controls]
            or training.prior.protected(model) != frozen
            or training.native.vendor.module_hash(model) != final_hash):
        raise ValueError('strict checkpoint/optimizer/scaler/RNG/output reload failed')
    peak = memory(root, 'strict_reload')['peak_allocated_bytes']
    training.native.vendor.verify(proof)
    result = {'schema': SCHEMA_VERSION, 'hypothesis_id': HYPOTHESIS_ID,
              'status': 'broad_generator_'+stage+'_completed_semantic_quality_pending',
              'updates': updates, 'TRAIN_cases': len(rows), 'trainable_parameters': sum(p.numel() for p in params),
              'initial_controls': training.prior.summarize(initial), 'final_TRAIN': training.prior.summarize(final),
              'shuffled_source16': training.prior.summarize(shuffled), 'strict_reload_passed': True,
              'gradient_parity_passed': all(r['passed'] for r in parity) if stage == 'pilot' else True,
              'oracle_unchanged': True, 'frozen_hashes': frozen, 'initial_model_sha256': parent.MODEL_SHA,
              'final_model_sha256': final_hash, 'checkpoint_sha256': file_hash(root/'model.pt'),
              'producer_fingerprint': proof['fingerprint'], 'training_seconds': elapsed,
              'peak_allocated_bytes': peak, 'feature_cache': loader.metrics(),
              'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False}
    write_json(root/'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True); model.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=tuple(RUNS))
    stage = parser.parse_args().stage
    root = RUNS[stage]
    if root.exists(): raise ValueError('fresh append-only artifact root required')
    training, environment = runtime.configure()
    offload.require_rtx2060()
    torch.cuda.reset_peak_memory_stats()
    try:
        prepare(training, environment) if stage == 'prepare' else train(stage, training, environment)
    except Exception as error:
        if root.exists():
            write_json(root/'failure.json', {'exception': type(error).__name__, 'message': str(error), 'task_accuracy': None})
        raise


if __name__ == '__main__': main()
