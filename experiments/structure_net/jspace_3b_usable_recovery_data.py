"""Full admitted-corpus recovery orders over existing immutable native banks.

Stage A defaults to one pass through every TRAIN example. Existing DEV examples
remain exposed monitoring/selection data, not new confirmation. Length buckets
reorder optimizer windows only: the current worker still executes independent
singleton microbatches, with no concatenation, truncation or invented EOS.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil

from . import jspace_3b_full_progressive_data as banks

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/07 - Status Reports/jspace/2026-11-16_qwen3b-usable-full-corpus-recovery.md'
SCHEMA = 'jspace.3b-usable-full-corpus-data.v1'
cache = banks.cache
read, write, sha, identity = cache.read, cache.write, cache.sha, cache.identity
digest = cache.data.digest


def admit(records):
    """Keep all unique examples, including distinct examples sharing a TRAIN group."""
    fingerprint = cache.data.dataset_fingerprint(records)
    train, dev, inventory = [], [], []
    for index, record in enumerate(records):
        if record.split not in ('train', 'development'):
            raise ValueError('Only explicitly admitted TRAIN and exposed DEV records are permitted')
        if (record.finish_reason != 'eos' or record.eos_token_id is None
                or record.response_ids[-1] != record.eos_token_id):
            raise ValueError('Every complete response must contain its observed terminal EOS')
        if len(record.prompt_ids) > 288 or len(record.response_ids) > 128 or len(record.input_ids) > 416:
            raise ValueError('Oversize record rejected; full-corpus recovery never truncates')
        family = record.provenance.get('source_family')
        if not isinstance(family, str) or not family:
            raise ValueError('Every record requires an explicit source family')
        (train if record.split == 'train' else dev).append(index)
        inventory.append(dict(index=index, example_id=record.example_id, group_id=record.group_id,
            split=record.split, family=family, input_tokens=len(record.input_ids),
            response_tokens=len(record.response_ids), eos_token_id=record.eos_token_id))
    if not train or not dev:
        raise ValueError('Nonempty TRAIN and separate exposed DEV inventories required')
    train_groups = {records[i].group_id for i in train}
    dev_groups = {records[i].group_id for i in dev}
    if train_groups & dev_groups:
        raise ValueError('TRAIN and DEV source groups overlap')
    return dict(records_sha256=fingerprint, training_indices=train, development_indices=dev,
        record_inventory=inventory, training_examples=len(train), training_groups=len(train_groups),
        development_examples=len(dev), development_groups=len(dev_groups),
        train_dev_group_overlap=0,
        within_train_group_policy='Retain all distinct admitted examples; groups may contain multiple examples')


def _counts(inventory, indices):
    families = Counter(); tokens = Counter()
    for index in indices:
        row = inventory[index]
        families[row['family']] += 1
        tokens[row['family']] += row['response_tokens']
    return dict(examples=len(indices), response_tokens=sum(tokens.values()),
        input_tokens=sum(inventory[i]['input_tokens'] for i in indices),
        family_examples=dict(sorted(families.items())), family_response_tokens=dict(sorted(tokens.items())))


def epoch_plan(records, admitted, epoch, *, seed, accumulation_examples=16, buffer_size=256):
    """Global shuffle followed by bounded NAL length buckets; no replacement."""
    if (type(epoch) is not int or epoch < 1 or type(seed) is not int
            or type(accumulation_examples) is not int or accumulation_examples < 1
            or type(buffer_size) is not int or buffer_size < accumulation_examples
            or buffer_size % accumulation_examples):
        raise ValueError('Positive epoch and divisible bounded bucket/window sizes required')
    train = list(admitted['training_indices'])
    random.Random(seed+epoch).shuffle(train)
    lookup = {records[i].example_id:i for i in train}
    grouped = cache.data.length_buckets((records[i] for i in train), batch_size=accumulation_examples,
        max_length=416, buffer_size=buffer_size, seed=seed+epoch+1_000_003)
    windows = [[lookup[record.example_id] for record in group] for group in grouped]
    order = [index for window in windows for index in window]
    if (len(order) != len(train) or len(set(order)) != len(train)
            or set(order) != set(train) or any(len(window) > accumulation_examples for window in windows)):
        raise ValueError('Epoch omitted, repeated or substituted an admitted TRAIN example')
    return dict(epoch=epoch, order=order, windows=windows, optimizer_updates=len(windows),
        **_counts(admitted['record_inventory'], order), source_groups=admitted['training_groups'],
        complete_training_pass=True, seed=seed+epoch)


def training_plan(records, *, epochs=1, seed=20261117, accumulation_examples=16, buffer_size=256,
                  thresholds=(25_000, 50_000, 75_000)):
    if type(epochs) is not int or epochs < 1:
        raise ValueError('A positive explicit number of full passes is required')
    thresholds = tuple(thresholds)
    if (not thresholds or any(type(v) is not int or v <= 0 for v in thresholds)
            or tuple(sorted(set(thresholds))) != thresholds):
        raise ValueError('Strictly increasing positive response-token checkpoint thresholds required')
    admitted = admit(records)
    plans = [epoch_plan(records, admitted, epoch, seed=seed, accumulation_examples=accumulation_examples,
                        buffer_size=buffer_size) for epoch in range(1, epochs+1)]
    windows, order, checkpoints, window_inventory = [], [], [], []
    total_tokens = total_inputs = 0
    for ep in plans:
        epoch_tokens = 0; pending = list(thresholds)
        for local_update, indices in enumerate(ep['windows'], 1):
            counts = _counts(admitted['record_inventory'], indices)
            total_tokens += counts['response_tokens']; epoch_tokens += counts['response_tokens']
            total_inputs += counts['input_tokens']; windows.append(indices); order.extend(indices)
            complete = local_update == len(ep['windows'])
            row = dict(update=len(windows), epoch=ep['epoch'], epoch_update=local_update,
                examples=len(indices), examples_seen=len(order), supervised_tokens=counts['response_tokens'],
                cumulative_supervised_tokens=total_tokens, epoch_supervised_tokens=epoch_tokens,
                input_tokens=counts['input_tokens'], full_training_pass=complete)
            window_inventory.append(row)
            crossed = [value for value in pending if epoch_tokens >= value]
            pending = [value for value in pending if value not in crossed]
            if crossed or complete:
                checkpoints.append(dict(update=len(windows), epoch=ep['epoch'], examples_seen=len(order),
                    supervised_tokens=total_tokens, epoch_supervised_tokens=epoch_tokens,
                    crossed_thresholds=crossed, full_training_pass=complete,
                    early_candidate_allowed=True))
        if epoch_tokens != ep['response_tokens']:
            raise ValueError('Full-pass response-token inventory differs')
    return dict(**admitted, epochs=plans, checkpoints=checkpoints,
        training=dict(windows=windows, order=order, window_inventory=window_inventory,
            accumulation_examples=accumulation_examples, microbatch_examples=1, epochs=epochs,
            examples=len(order), unique_examples=admitted['training_examples'],
            source_groups=admitted['training_groups'], actual_supervised_tokens=total_tokens,
            input_tokens=total_inputs, nominal_supervised_tokens=total_tokens,
            family_examples=plans[0]['family_examples'], family_response_tokens=plans[0]['family_response_tokens'],
            family_counts_scope='Per full pass; unique examples, not repeated epoch presentations',
            seed=seed, buffer_size=buffer_size,
            ordering='Seeded global TRAIN shuffle, bounded NAL length buckets, no replacement within epoch',
            packing=False, speedup_claim=False,
            complete_response_eos=True, additional_epochs_require_new_prepared_plan=True),
        coverage_policy='Evaluate early candidates honestly; final run receipt must state actual full-pass coverage',
        validation_scope='Existing exposed DEV monitoring/selection only; not independent confirmation')


def coverage_report(plan, seen_indices):
    """Validate actual observed order and distinguish presentations from new data."""
    seen = list(seen_indices); expected = plan['training']['order']
    if any(type(index) is not int for index in seen) or len(seen) > len(expected) or seen != expected[:len(seen)]:
        raise ValueError('Observed training cases do not match the frozen no-replacement schedule')
    inventory = plan['record_inventory']; unique = set(seen)
    actual = _counts(inventory, seen); distinct = _counts(inventory, sorted(unique))
    boundaries = []; total = 0
    for epoch in plan['epochs']:
        total += len(epoch['order']); boundaries.append(total)
    return dict(presentations=len(seen), unique_examples=len(unique),
        unique_source_groups=len({inventory[i]['group_id'] for i in unique}),
        examples_available=plan['training_examples'], remaining_unique_examples=plan['training_examples']-len(unique),
        fraction_of_unique_training_examples=len(unique)/plan['training_examples'],
        supervised_response_tokens=actual['response_tokens'], input_tokens=actual['input_tokens'],
        family_presentations=actual['family_examples'], family_unique_examples=distinct['family_examples'],
        family_response_tokens=actual['family_response_tokens'],
        complete_epochs=sum(len(seen) >= boundary for boundary in boundaries),
        completed_optimizer_windows=sum(row['examples_seen'] <= len(seen)
                                        for row in plan['training']['window_inventory']),
        all_training_examples_seen=len(unique) == plan['training_examples'],
        planned_run_completed=len(seen) == len(expected),
        developmental_monitoring_is_exposed=True, independent_test_claim=False)


def source_files():
    return cache.prior.source.inherited.local_import_closure([
        Path(__file__), PROTOCOL, ROOT/'pyproject.toml', ROOT/'pixi.lock',
        ROOT/'tests/structure_net/test_jspace_3b_usable_recovery_data.py',
        ROOT/'experiments/neural_architecture_lab/distillation_recovery.py',
        ROOT/'src/neural_architecture_lab/distillation_data.py'])


def prepare(root, full_data_root, *, cut=33, epochs=1, seed=20261117,
            accumulation_examples=16, buffer_size=256):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Recovery data preparation requires CUDA hidden')
    root, full_data_root = Path(root).resolve(), Path(full_data_root).resolve()
    if root.exists(): raise FileExistsError(root)
    native_plan, records = banks.verify(full_data_root, live=False)
    if type(cut) is not int or cut not in native_plan['cuts']:
        raise ValueError('The fixed-depth boundary must exist in the original native inventory')
    native = full_data_root/'native'
    banks._manifest_files(native, native_plan['fingerprint'], hash_payloads=False)
    result = read(native/'result.json')
    if not all(result.get(key) is True for key in ('full_teacher_bit_exact', 'teacher_unchanged',
        'existing_boundary_bit_exact', 'seed_boundary35_bit_exact')):
        raise ValueError('Completed original native numerical controls required')
    schedule = training_plan(records, epochs=epochs, seed=seed,
        accumulation_examples=accumulation_examples, buffer_size=buffer_size)
    if schedule['training_examples'] != 4441 or schedule['development_examples'] != 128:
        raise ValueError('This protocol pins the complete4441TRAIN/128DEV admitted population')
    if schedule['records_sha256'] != native_plan['records_sha256']:
        raise ValueError('Native token-row inventory differs from full recovery corpus')
    response = Path(native_plan['response_cache_root'])
    manifest_paths = [full_data_root/'plan.json', native/'manifest.json', native/'result.json',
                      response/'plan.json', response/'native/manifest.json']
    inputs = {str(path):sha(path) for path in manifest_paths}
    reference_paths = [native/f'boundary_{cut:02d}.npy', native/f'boundary_{cut+1:02d}.npy',
        native/f'mc_boundary_{cut:02d}.npy', response/'native/teacher_hidden.npy']
    sources = {str(path):sha(path) for path in sorted(source_files())}
    plan = dict(schema=SCHEMA, full_data_root=str(full_data_root), full_data_fingerprint=native_plan['fingerprint'],
        response_cache_root=str(response), response_cache_fingerprint=native_plan['response_cache_fingerprint'],
        model=native_plan['model'], cut=cut, corpus=native_plan['corpus'], rows=native_plan['rows'],
        width=native_plan['width'], vocabulary=native_plan['vocabulary'], total_tokens=native_plan['total_tokens'],
        max_length=416, pad_token_id=151643,
        generation_indices=native_plan['generation_indices'], generation_eos_token_ids=native_plan['generation_eos_token_ids'],
        generation_max_tokens=128, **schedule,
        input_sha256=inputs, input_identity={path:identity(path) for path in inputs},
        dependency_identity={**native_plan['input_identity'], **{str(path):identity(path) for path in reference_paths}},
        source_sha256=sources, bank_copy_bytes=0, full_bank_rehash=False,
        original_geometry_changed=False, confirmed=False)
    if (any(i not in plan['development_indices'] for i in plan['generation_indices'])
            or len(plan['generation_indices']) != 16):
        raise ValueError('Existing balanced16 generation probes must remain exposed DEV-only')
    generation_families = Counter(plan['record_inventory'][i]['family'] for i in plan['generation_indices'])
    if len(generation_families) != 8 or set(generation_families.values()) != {2}:
        raise ValueError('Generated-quality monitoring requires two existing DEV examples per family')
    plan['generation_family_examples'] = dict(sorted(generation_families.items()))
    plan['fingerprint'] = digest(plan)
    root.mkdir(parents=True); write(root/'plan.json', plan)
    for source, wanted in sources.items():
        target = root/'producer'/Path(source).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if sha(target) != wanted: raise ValueError('Full-corpus data producer archive differs')
    return dict(status='prepared', fingerprint=plan['fingerprint'], cut=cut, epochs=epochs,
        examples=plan['training_examples'], groups=plan['training_groups'],
        response_tokens=plan['training']['actual_supervised_tokens'],
        updates=len(plan['training']['windows']), bank_copy_bytes=0)


def verify(root, *, live=True):
    root = Path(root).resolve(); plan = read(root/'plan.json'); body = dict(plan); fp = body.pop('fingerprint')
    if plan.get('schema') != SCHEMA or digest(body) != fp:
        raise ValueError('Full-corpus recovery data plan changed')
    for path, old in {**plan['dependency_identity'], **plan['input_identity']}.items():
        if identity(path) != old: raise ValueError('Recovery source/cache identity changed: '+path)
    for path, wanted in plan['input_sha256'].items():
        if sha(path) != wanted: raise ValueError('Recovery input manifest changed')
    for path, wanted in plan['source_sha256'].items():
        if sha(root/'producer'/Path(path).relative_to(ROOT)) != wanted or live and sha(path) != wanted:
            raise ValueError('Full-corpus recovery data producer changed')
    native, records = banks.verify(plan['full_data_root'], live=False)
    settings = plan['training']
    expected = training_plan(records, epochs=settings['epochs'], seed=settings['seed'],
        accumulation_examples=settings['accumulation_examples'], buffer_size=settings['buffer_size'])
    if (native['fingerprint'] != plan['full_data_fingerprint'] or native['model'] != plan['model']
            or any(plan[key] != value for key, value in expected.items())):
        raise ValueError('Frozen full-pass recovery schedule or native teacher differs')
    return plan, records


class RecoveryDataset:
    """Existing exact boundary/teacher banks; no data repacking or new masks."""
    def __init__(self, root):
        self.root = Path(root).resolve(); self.plan, _ = verify(root, live=False)
        self.native = banks.RebuildDataset(self.plan['full_data_root'], cut=self.plan['cut'])
        self.records = self.native.records; self.rows = self.native.rows
        self.training = self.training_indices = self.plan['training_indices']
        self.development = self.development_indices = self.plan['development_indices']
        self.width, self.cut, self.pad_token_id = self.native.width, self.native.cut, self.native.pad_token_id
        self.max_length = self.plan['max_length']; self.items = self.native.items

    def batch(self, indices, **kwargs):
        return self.native.batch(indices, **kwargs)

    def close(self): self.native.close()


class MCDataset:
    """Existing exposed MC controls, never returned as a training population."""
    def __init__(self, root):
        self.root = Path(root).resolve(); self.plan, _ = verify(root, live=False)
        self.native = banks.MCDataset(self.plan['full_data_root'], cut=self.plan['cut'])
        for name in ('items', 'label_ids', 'native_records', 'native_correct', 'training', 'development',
                     'width', 'cut', 'pad_token_id'):
            setattr(self, name, getattr(self.native, name))

    def batch(self, indices, **kwargs): return self.native.batch(indices, **kwargs)
    def close(self): self.native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'check'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--full-data-root', type=Path)
    parser.add_argument('--cut', type=int, default=33); parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--accumulation-examples', type=int, default=16)
    parser.add_argument('--buffer-size', type=int, default=256)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.full_data_root is None: parser.error('--full-data-root is required')
        result = prepare(args.root, args.full_data_root, cut=args.cut, epochs=args.epochs, seed=args.seed,
            accumulation_examples=args.accumulation_examples, buffer_size=args.buffer_size)
    else:
        p, _ = verify(args.root, live=False)
        result = dict(status='prepared', fingerprint=p['fingerprint'], cut=p['cut'],
            examples=p['training_examples'], groups=p['training_groups'], epochs=p['training']['epochs'],
            response_tokens=p['training']['actual_supervised_tokens'], checkpoints=p['checkpoints'])
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__': main()
