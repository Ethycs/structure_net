"""Decoder-only follow-up to the fixed sequential engineering TRAIN control."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_sequential_engineering as prior
from experiments.structure_net.ncm_memory_data import file_hash, write_json

base, native, preparation, q, ROOT = prior.base, prior.native, prior.preparation, prior.q, prior.ROOT
PARENT = prior.RUNS['capacity']
PARENT_SHA = 'd27af43e7bb8f8a80d81e2c19c51e12ddb8bcc08154dd0540303862ecd81f841'
MODEL_SHA = 'ad2f1eb8de12b8abfc34ac228aa3b452a09a6cd78cdaf48eb3222163b8dcbb7f'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-sequential-decoder-plan.md')
TEST = Path('tests/structure_net/test_ncm_sequential_decoder.py')
RUNS = {s: ROOT / ('20261025_sequential_decoder_' + s + '_v1') for s in ('pilot', 'capacity', 'train')}
CONFIG = dict(accumulation=4, max_lr=1e-4, min_lr=1e-5, sampler_seed=151142,
    dropout_seed=152142, precision='FP32weights_FP16training_FP32evaluation',
    physical_GPU=2, TF32=False, peak_limit_bytes=4_900_000_000,
    only_decoder_trainable=True, encoder_VQ_generator_source_adapters_frozen=True,
    parent_Llama_absent_and_frozen=True, pad_id=0, EOS_id=101,
    loss='per_example_native_word_CE_ignore_PAD0', checkpoint_selection='fixed_final',
    oracle_codes_in_training_only=True, task_accuracy=None, final_test_opened=False, speed_tested=False)


def configure(model, training):
    model.eval().requires_grad_(False)
    model.codec.decoder.requires_grad_(training).train(training)
    return model


def protected(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('concepts.codec.decoder.'):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def word_loss(logits, words):
    if logits.shape[:2] != words.shape or not (words != 0).any():
        raise ValueError('nonempty aligned native word frames required')
    return F.cross_entropy(logits.float().flatten(0, 1), words.flatten(), ignore_index=0)


def rate(step, updates):
    if not 1 <= step <= updates:
        raise ValueError('step outside fixed budget')
    warmup = min(32, max(1, updates // 8))
    if step <= warmup:
        return CONFIG['max_lr'] * step / warmup
    return CONFIG['min_lr'] + .5 * (CONFIG['max_lr'] - CONFIG['min_lr']) * (1 + math.cos(math.pi * (step-warmup)/(updates-warmup)))


def load_parent():
    preparation.completed(PARENT)
    if file_hash(PARENT / 'model.pt') != PARENT_SHA:
        raise ValueError('exact sequential initializer required')
    model = base.load_renderer()
    saved = torch.load(PARENT / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True)
    if native.vendor.module_hash(model) != MODEL_SHA:
        raise ValueError('full sequential tensors changed')
    return configure(model, False)


@torch.no_grad()
def evaluate(model, rows, loader, tokenizer, source_only=False):
    configure(model, False)
    outputs = []
    for row in rows:
        features, mask, gold = prior.tensors(row, loader)
        codes = prior.mechanism.generate(model, features, mask) if source_only else gold
        raw = model.word_logits(features, mask, codes).argmax(-1)
        words = torch.tensor([row['native_word_ids']], device=raw.device)
        active = words != 0
        ids = raw.cpu().tolist()[0]
        decoded = preparation.rendering.decode(ids, tokenizer)
        reference = preparation.rendering.decode(row['native_word_ids'], tokenizer)
        outputs.append({'id': row['id'], 'source_sha256': row['source_sha256'],
            'target_sha256': row['target_sha256'], 'reference_text': reference['text'],
            'target_text': row['target_text'], 'codes': codes.cpu().tolist()[0],
            'raw_word_ids': ids, **decoded,
            'active_word_matches': int(((raw == words) & active).sum()), 'active_words': int(active.sum()),
            'rendered_exact': decoded['text'] == reference['text'],
            'source_only': source_only, 'task_accuracy': None})
    return outputs


def summarize(outputs):
    total = sum(r['active_words'] for r in outputs)
    matched = sum(r['active_word_matches'] for r in outputs)
    return {'cases': len(outputs), 'active_word_matches': matched, 'active_words': total,
        'active_word_agreement': matched / total, 'rendered_exact_cases': sum(r['rendered_exact'] for r in outputs),
        'empty_outputs': sum(not r['text'] for r in outputs)}


def capacity_pass(oracle, source):
    return (oracle['cases'] == source['cases'] == 16
        and oracle['active_word_agreement'] == 1 and oracle['rendered_exact_cases'] == 16
        and source['rendered_exact_cases'] == 16 and not source['empty_outputs'])


def freeze(root, stage, updates):
    files = {Path(__file__), PLAN, TEST}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(native.vendor.VENDOR.resolve())):
                files.add(p)
    files.update(native.vendor.VENDOR.rglob('*.py'))
    inputs = [base.DATA / n for n in ('cases.json', 'source_index.json', 'feature_manifest.json', 'selection.json', 'result.json', 'actual_tool_receipt.json')]
    inputs += [PARENT / n for n in ('model.pt', 'result.json', 'producer.json', 'actual_tool_receipt.json')]
    if stage != 'pilot':
        dependency = RUNS['pilot' if stage == 'capacity' else 'capacity']
        inputs += [dependency / n for n in ('result.json', 'actual_tool_receipt.json', 'producer.json')]
    root.mkdir()
    (root / 'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root / 'producer_source' / f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-sequential-decoder.v1', 'hypothesis_id': 'ncm-sequential-decoder-v1',
        'conformance': 'EXPLORATORY', 'stage': stage, 'config': {**CONFIG, 'updates': updates,
            'train_cases': 732 if stage == 'train' else 16, 'initialization': 'same_sequential_parent_each_stage'},
        'source_hashes': {str(p): file_hash(p) for p in sorted(files)},
        'input_hashes': {str(p): file_hash(p) for p in inputs},
        'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()}}
    proof['fingerprint'] = q.digest(proof)
    write_json(root / 'producer.json', proof)
    return proof


def save(path, model, opt, scaler, sampler, permutation, step, proof):
    if path.exists():
        raise ValueError('checkpoint overwrite forbidden')
    torch.save({'decoder': {k: v.detach().cpu().clone() for k, v in model.codec.decoder.state_dict().items()},
        'parent_checkpoint_sha256': PARENT_SHA, 'protected_sha256': protected(model),
        'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'sampler_rng': sampler.get_state(),
        'sampler_permutation_remaining': permutation.copy(), 'CPU_rng': torch.get_rng_state(),
        'CUDA_rng': torch.cuda.get_rng_state_all(), 'step': step, 'producer_fingerprint': proof['fingerprint']}, path)


def run(stage):
    native.vendor.corpus.quality.dependencies()
    preparation.checkpoint.training.require_gpu()
    preparation.completed(base.DATA)
    if stage != 'pilot':
        dependency = RUNS['pilot' if stage == 'capacity' else 'capacity']
        preparation.completed(dependency)
        outcome = q.read(dependency / 'result.json')
        if not outcome['strict_reload_passed'] or (stage == 'train' and not outcome['capacity_pass']):
            raise ValueError('completed decoder lifecycle and applicable control gate required')
    if file_hash(base.DATA / 'cases.json') != q.read(base.DATA / 'result.json')['cases_sha256']:
        raise ValueError('TRAIN data changed')
    all_rows = q.read(base.DATA / 'cases.json')
    lookup = {r['id']: r for r in all_rows}
    controls = [lookup[i] for i in q.read(base.DATA / 'selection.json')['diagnostic_ids']]
    rows = all_rows if stage == 'train' else controls
    if len(rows) != (732 if stage == 'train' else 16):
        raise ValueError('fixed TRAIN selection required')
    index, manifest = q.read(base.DATA / 'source_index.json'), q.read(base.DATA / 'feature_manifest.json')
    loader = preparation.streaming.FeatureCache(index, manifest, max_bytes=1024**3)
    lengths = {r['id']: r['source_positions'] for r in index}
    longest = max(range(len(rows)), key=lambda i: lengths[rows[i]['id']])
    updates = {'pilot': 2, 'capacity': 256, 'train': 1024}[stage]
    root = RUNS[stage]
    if root.exists():
        raise ValueError('fresh decoder study root required')
    model = load_parent()
    proof = freeze(root, stage, updates)
    frozen, initial_hash = protected(model), native.vendor.module_hash(model.codec.decoder)
    tokenizer = native.tokenizer()
    initial = evaluate(model, controls, loader, tokenizer)
    source_before = evaluate(model, controls, loader, tokenizer, True)
    write_json(root / 'initial_oracle_outputs.json', initial)
    write_json(root / 'initial_source_outputs.json', source_before)
    configure(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    if {id(p) for p in params} != {id(p) for p in model.codec.decoder.parameters()}:
        raise ValueError('only decoder parameter identities may update')
    opt = torch.optim.AdamW(params, lr=CONFIG['max_lr'], betas=(.9, .95), eps=1e-8, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    sampler = torch.Generator().manual_seed(CONFIG['sampler_seed'])
    torch.manual_seed(CONFIG['dropout_seed']); torch.cuda.manual_seed_all(CONFIG['dropout_seed'])
    permutation = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (root / 'trace.jsonl').open('x') as trace:
        for step in range(1, updates + 1):
            configure(model, True)
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group['lr'] = rate(step, 256 if stage == 'pilot' else updates)
            metrics = []
            for micro in range(CONFIG['accumulation']):
                if not permutation:
                    permutation = torch.randperm(len(rows), generator=sampler).tolist()
                i = ([0, longest, 0, longest][micro] if stage == 'pilot' else permutation.pop())
                features, mask, gold = prior.tensors(rows[i], loader)
                words = torch.tensor([rows[i]['native_word_ids']], device=features.device)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model.word_logits(features, mask, gold)
                    loss = word_loss(logits, words)
                scaler.scale(loss / CONFIG['accumulation']).backward()
                metrics.append({'id': rows[i]['id'], 'word_CE': float(loss.detach())})
                del logits, loss
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite decoder gradients')
            if step == 1 and (model.codec.decoder.output_layer.weight.grad is None or not model.codec.decoder.output_layer.weight.grad.abs().sum()):
                raise ValueError('decoder output projection must receive gradients')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError('frozen parameter received gradients')
            scaler.step(opt); scaler.update()
            peak = torch.cuda.max_memory_allocated()
            if peak > CONFIG['peak_limit_bytes']:
                raise ValueError('decoder exceeded4.9GB allocation')
            record = {'step': step, 'updates': updates, 'lr': opt.param_groups[0]['lr'],
                'word_CE': sum(r['word_CE'] for r in metrics) / len(metrics),
                'gradient_norm': float(norm), 'examples': metrics, 'elapsed_seconds': time.monotonic()-started, 'peak_bytes': peak}
            trace.write(json.dumps(record, allow_nan=False) + '\n'); trace.flush()
            if step <= 2 or step % 16 == 0:
                write_json(root / 'progress.json', record)
                print(json.dumps({k: v for k, v in record.items() if k != 'examples'}), flush=True)
    training_seconds = time.monotonic() - started
    final = evaluate(model, rows, loader, tokenizer)
    source_after = evaluate(model, controls, loader, tokenizer, True)
    if [r['codes'] for r in source_before] != [r['codes'] for r in source_after]:
        raise ValueError('frozen source-only generator changed')
    final_hash = native.vendor.module_hash(model.codec.decoder)
    if protected(model) != frozen or initial_hash == final_hash:
        raise ValueError('protected tensors changed or decoder did not learn')
    save(root / 'model.pt', model, opt, scaler, sampler, permutation, updates, proof)
    del opt
    gc.collect(); torch.cuda.empty_cache()
    saved = torch.load(root / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    # Restore into a fresh full parent to verify the compositional checkpoint.
    model.close(); del model
    gc.collect(); torch.cuda.empty_cache()
    model = load_parent()
    model.codec.decoder.load_state_dict(saved['decoder'], strict=True)
    configure(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CONFIG['max_lr'], betas=(.9, .95), eps=1e-8, weight_decay=.01)
    opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng']); torch.set_rng_state(saved['CPU_rng']); torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    same = preparation.checkpoint.training.lifecycle.same
    if (not same(model.codec.decoder.state_dict(), saved['decoder']) or not same(opt.state_dict(), saved['optimizer'])
            or not same(scaler.state_dict(), saved['scaler']) or not torch.equal(sampler.get_state(), saved['sampler_rng'])
            or not torch.equal(torch.get_rng_state(), saved['CPU_rng']) or not same(torch.cuda.get_rng_state_all(), saved['CUDA_rng'])
            or permutation != saved['sampler_permutation_remaining'] or protected(model) != frozen
            or evaluate(model, controls, loader, tokenizer, True) != source_after
            or native.vendor.module_hash(model.codec.decoder) != final_hash):
        raise ValueError('strict fresh model/optimizer/scaler/RNG/output reload failed')
    write_json(root / 'oracle_outputs.json', final)
    write_json(root / 'source_outputs.json', source_after)
    native.vendor.verify(proof)
    oracle_summary, source_summary = summarize(final), summarize(source_after)
    result = {'schema': proof['schema'], 'status': 'decoder_' + stage + '_completed_TRAIN_control_only',
        'hypothesis_id': proof['hypothesis_id'], 'updates': updates, 'TRAIN_cases': len(rows),
        'trainable_parameters': sum(p.numel() for p in params), 'initial_oracle16': summarize(initial),
        'final_oracle': oracle_summary, 'source_only16': source_summary,
        'strict_reload_passed': True, 'generator_outputs_unchanged': True,
        'protected_sha256': frozen, 'initial_decoder_sha256': initial_hash, 'final_decoder_sha256': final_hash,
        'checkpoint_sha256': file_hash(root / 'model.pt'), 'parent_checkpoint_sha256': PARENT_SHA,
        'producer_fingerprint': proof['fingerprint'], 'training_seconds': training_seconds,
        'peak_allocated_bytes': peak, 'capacity_pass': capacity_pass(oracle_summary, source_summary),
        'full_decoder_reconstruction_pass': stage == 'train' and oracle_summary['active_word_agreement'] >= .99
            and oracle_summary['rendered_exact_cases'] / len(rows) >= .95 and not oracle_summary['empty_outputs'],
        'stored_targets_semantically_reviewed': False, 'task_accuracy': None,
        'final_test_opened': False, 'speed_tested': False}
    write_json(root / 'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=tuple(RUNS))
    run(parser.parse_args().stage)
