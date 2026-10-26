"""Shared mixed-corpus I/O, optimizer updates and exact training-state recovery."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import time

import torch

from experiments.structure_net import ncm_public_curriculum as curriculum
from experiments.structure_net import ncm_example_batching as batching
from experiments.structure_net import ncm_sequential_levels as mechanism


class MixedSources:
    """All mutable JSONL/cache access belongs to a single prefetch worker."""
    def __init__(self, public, anchors, anchor_cache):
        self.public, self.anchors, self.anchor_cache = public, anchors, anchor_cache
        public_ids = {r['id'] for r in public.pairs.entries}
        anchor_ids = {r['id'] for r in anchors}
        if (len(anchor_ids) != len(anchors) or public_ids & anchor_ids
                or any(r['split'] != 'TRAIN' or len(r['oracle_codes']) != 641
                       or len(r['native_word_ids']) != 128 for r in anchors)):
            raise ValueError('distinct complete TRAIN populations required')

    def fetch(self, examples):
        if not examples or any(e.population not in ('public', 'anchor') for e in examples):
            raise ValueError('nonempty bound mixed-population request required')
        public = [e.index for e in examples if e.population == 'public']
        values = iter(self.public.__getitems__(public))
        result = []
        for e in examples:
            if e.population == 'public':
                row = next(values)
            else:
                if not 0 <= e.index < len(self.anchors):
                    raise IndexError('valid reviewed application index required')
                original = self.anchors[e.index]
                row = {**{k: original[k] for k in ('id', 'source_sha256', 'native_word_ids', 'oracle_codes')},
                       'features': self.anchor_cache.get(original)}
            result.append({**row, 'presentation': asdict(e)})
        return result


def histories(gold, examples, levels):
    if len(examples) != len(gold):
        raise ValueError('exactly one deterministic noise stream per example required')
    values, changes = [], []
    for i, example in enumerate(examples):
        rng = torch.Generator().manual_seed(example.corruption_seed)
        value, count = batching.objective.history(gold[i:i+1], levels, 8192, rng, example.noisy)
        values.append(value); changes.append(count)
    return torch.cat(values), changes


def batch_digest(rows):
    """Pipeline diagnostic binds full FP32 bytes and exact native labels."""
    import json
    digest = hashlib.sha256()
    for row in rows:
        meta = {k: row[k] for k in ('id', 'source_sha256', 'native_word_ids', 'oracle_codes', 'presentation')}
        digest.update(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
        value = row['features'].detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def update(model, batches, opt, scaler, *, step, config=curriculum.CONFIG):
    """Consume one complete effective update; prefetch cannot commit progress."""
    mechanism.configure(model, True)
    opt.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for group in opt.param_groups:
        group['lr'] = curriculum.rate(step, config)
    levels = model.concepts.generator.levels
    expected_epoch, expected_update = divmod(step - 1, config.updates_per_epoch)
    expected_epoch += 1
    expected_size = min(config.effective_examples, config.presentations_per_epoch -
                        expected_update * config.effective_examples)
    count = 0
    metrics, gradient_time, wait_time = [], 0., 0.
    while count < expected_size:
        before = time.monotonic()
        index, denominator, examples, rows = next(batches)
        wait_time += time.monotonic() - before
        expected_positions = list(range(expected_update * config.effective_examples + count,
                                        expected_update * config.effective_examples + count + len(rows)))
        if (index != expected_update or denominator != expected_size or len(examples) != len(rows)
                or len(rows) != min(config.microbatch, expected_size - count)
                or any(e.epoch != expected_epoch for e in examples)
                or [e.position for e in examples] != expected_positions
                or any(row['presentation'] != asdict(e) for e, row in zip(examples, rows))):
            raise ValueError('exact completed-update position and actual loss denominator required')
        torch.cuda.synchronize()
        before = time.monotonic()
        features, mask, gold = batching.collate([r['features'] for r in rows],
                                               [r['oracle_codes'] for r in rows], device='cuda')
        prior, changes = histories(gold, examples, levels)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model.concepts.teacher_logits(features, mask, prior)
        loss = batching.weighted_loss(logits, gold, levels, examples_in_update=expected_size)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite complete mixed-corpus training loss')
        scaler.scale(loss).backward()
        metrics.append({'ids': [r['id'] for r in rows], 'positions': expected_positions,
                        'populations': [e.population for e in examples],
                        'noisy': [e.noisy for e in examples], 'history_changes': changes,
                        'weighted_CE': float(loss.detach())})
        count += len(rows)
        del features, mask, gold, prior, logits, loss, rows
        torch.cuda.synchronize()
        gradient_time += time.monotonic() - before
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    root = model.concepts.generator.l0_queries
    if (not torch.isfinite(norm) or root.grad is None or not root.grad.abs().sum()
            or any(p.grad is not None for p in model.parameters() if not p.requires_grad)):
        raise ValueError('finite trainable gradients, learning root, and frozen protected state required')
    previous = scaler.get_scale()
    before = time.monotonic()
    scaler.step(opt); scaler.update()
    if scaler.get_scale() < previous:
        raise ValueError('skipped optimizer update')
    torch.cuda.synchronize()
    optimizer_time = time.monotonic() - before
    opt.zero_grad(set_to_none=True)
    return {'step': step, 'epoch': expected_epoch, 'examples': count,
            'lr': curriculum.rate(step, config), 'weighted_CE': sum(r['weighted_CE'] for r in metrics),
            'gradient_norm': float(norm), 'scale': scaler.get_scale(), 'microbatches': metrics,
            'disk_wait_seconds': wait_time, 'gradient_seconds': gradient_time,
            'optimizer_seconds': optimizer_time}


def snapshot(model, opt, scaler, *, step, fingerprint, config=curriculum.CONFIG):
    return {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
            'CPU_rng': torch.get_rng_state(), 'CUDA_rng': torch.cuda.get_rng_state_all(),
            'completed_updates': step, 'producer_fingerprint': fingerprint,
            'curriculum': asdict(config)}


def save(path, model, opt, scaler, *, step, fingerprint, config=curriculum.CONFIG):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    if path.exists() or temporary.exists():
        raise ValueError('checkpoint evidence cannot be overwritten')
    value = snapshot(model, opt, scaler, step=step, fingerprint=fingerprint, config=config)
    with temporary.open('xb') as output:
        torch.save(value, output)
        output.flush(); os.fsync(output.fileno())
    temporary.rename(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def restore(saved, model, opt, scaler, *, fingerprint, config=curriculum.CONFIG):
    if (saved['producer_fingerprint'] != fingerprint or saved['curriculum'] != asdict(config)
            or type(saved['completed_updates']) is not int
            or not 0 <= saved['completed_updates'] <= config.updates):
        raise ValueError('exact training producer, curriculum and completed update required')
    model.load_state_dict(saved['model'], strict=True)
    opt.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
    torch.set_rng_state(saved['CPU_rng']); torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    return saved['completed_updates']
