"""Full-coverage decoder adaptation, with frozen representations and strict restart."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_recovered_codec_codes as prepared
from experiments.structure_net import ncm_lm1b_decoder_repair as old

data, runtime = prepared.data, prepared.runtime
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-decoder-training.md')
TEST = Path('tests/structure_net/test_ncm_recovered_decoder_training.py')
QUALIFICATION = data.ROOT/'20261027_recovered_decoder_CPU_v1.json'
ROOTS = {s: data.ROOT/f'20261027_recovered_decoder_{s}_v1' for s in ('pilot', 'train')}
STORAGE = prepared.STORAGE.parent
SCHEMA = 'nal.ncm-recovered-decoder-training.v1'
HYPOTHESIS = 'ncm-recovered-decoder-domain-v1'


@dataclass(frozen=True)
class Config:
    frames: int = 99367
    epochs: int = 2
    domain_batch: int = 32
    retention_batch: int = 8
    microbatch: int = 4
    warmup: int = 128
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    domain_weight: float = .8
    domain_seed: int = 93142
    retention_seed: int = 94142
    augmentation_seed: int = 95142
    dropout_seed: int = 96142
    peak_limit_bytes: int = 4_500_000_000

    @property
    def steps_per_epoch(self): return math.ceil(self.frames/self.domain_batch)

    @property
    def updates(self): return self.epochs*self.steps_per_epoch


CONFIG = Config()


def domain_orders(config=CONFIG):
    return [torch.randperm(config.frames, generator=torch.Generator().manual_seed(config.domain_seed+e))
            for e in range(config.epochs)]


def positions(step, orders, config=CONFIG):
    if not 1 <= step <= config.updates: raise ValueError('outside fixed exposure')
    epoch, batch = divmod(step-1, config.steps_per_epoch)
    return orders[epoch][batch*config.domain_batch:(batch+1)*config.domain_batch].tolist()


def rate(step, config=CONFIG):
    if not 1 <= step <= config.updates: raise ValueError('outside fixed schedule')
    if step <= config.warmup: return config.max_lr*step/config.warmup
    fraction = (step-config.warmup)/(config.updates-config.warmup)
    return config.min_lr+(config.max_lr-config.min_lr)*.5*(1+math.cos(math.pi*fraction))


def per_example_loss(logits, words):
    values = F.cross_entropy(logits.float().transpose(1, 2), words, ignore_index=0, reduction='none')
    counts = words.ne(0).sum(-1)
    if not counts.gt(0).all(): raise ValueError('nonempty native frames required')
    return values.sum(-1)/counts


def require(root):
    receipt = data.read(root/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 0: raise ValueError('actual successful execution required')
    result = data.read(root/'result.json')
    if 'artifacts' in result: data.verify(result['artifacts'])
    return result


def load_initial(training):
    codec = training.native.load_codec().float().eval().requires_grad_(False)
    frozen = old.frozen_hash(codec)
    saved = torch.load(runtime.codec_source.BROAD/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    codec.load_state_dict(runtime.codec_source.codec_state(saved['model'], codec.state_dict()), strict=True)
    del saved
    if training.native.vendor.module_hash(codec) != runtime.codec_source.CODEC_SHA or old.frozen_hash(codec) != frozen:
        raise ValueError('repaired decoder with exact released representation required')
    return old.decoder_training(codec), frozen


def optimizer_for(codec):
    return torch.optim.AdamW(codec.decoder.parameters(), lr=CONFIG.max_lr, betas=(.9, .95), eps=1e-8, weight_decay=.01)


def checkpoint(path, codec, optimizer, retention, augmentation, step, frozen, proof):
    if path.exists(): raise ValueError('checkpoint is append-only')
    saved = {'step': step, 'config': asdict(CONFIG),
        'decoder': {k: v.detach().cpu() for k, v in codec.decoder.state_dict().items()},
        'optimizer': optimizer.state_dict(), 'sampler_rng': retention.get_state(),
        'augmentation_rng': augmentation.get_state(), 'CPU_rng': torch.get_rng_state(),
        'CUDA_rng': torch.cuda.get_rng_state_all(), 'frozen_state_sha256': frozen,
        'producer_fingerprint': proof['fingerprint']}
    temporary = path.with_suffix('.tmp'); torch.save(saved, temporary); temporary.replace(path)


def restore(path, training, proof):
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved['producer_fingerprint'] != proof['fingerprint'] or saved['config'] != asdict(CONFIG): raise ValueError('checkpoint producer/config differs')
    codec, frozen = load_initial(training); codec.decoder.load_state_dict(saved['decoder'], strict=True)
    if frozen != saved['frozen_state_sha256']: raise ValueError('frozen state differs')
    optimizer = optimizer_for(codec); optimizer.load_state_dict(saved['optimizer'])
    retention = torch.Generator(); retention.set_state(saved['sampler_rng'])
    augmentation = torch.Generator(); augmentation.set_state(saved['augmentation_rng'])
    torch.set_rng_state(saved['CPU_rng']); torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    return codec, optimizer, retention, augmentation, frozen


def snapshot(codec, optimizer, retention, augmentation):
    def copy(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k: copy(v) for k, v in value.items()}
        if isinstance(value, list): return [copy(v) for v in value]
        if isinstance(value, tuple): return tuple(copy(v) for v in value)
        return value
    return copy({'decoder': codec.decoder.state_dict(), 'optimizer': optimizer.state_dict(),
        'retention_rng': retention.get_state(), 'augmentation_rng': augmentation.get_state(),
        'CPU_rng': torch.get_rng_state(), 'CUDA_rng': torch.cuda.get_rng_state_all()})


def update(step, selected, codec, optimizer, retention, augmentation, frames, codes, author):
    if codec.encoder.training or codec.vq.training or not codec.decoder.training: raise ValueError('decoder-only training mode required')
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups: group['lr'] = rate(step)
    domain_loss, retention_loss = 0., 0.
    for start in range(0, len(selected), CONFIG.microbatch):
        ids = selected[start:start+CONFIG.microbatch]
        words = torch.from_numpy(np.array(frames[ids], dtype=np.int64)).cuda()
        code = torch.from_numpy(np.array(codes[ids], dtype=np.int64)).cuda()
        with torch.no_grad(): latent = codec.vq.idx_to_z_q(code)
        logits = codec.decoder(latent.detach()); losses = per_example_loss(logits, words)
        loss = CONFIG.domain_weight*losses.sum()/len(selected)
        if not torch.isfinite(loss): raise ValueError('nonfinite domain loss')
        loss.backward(); domain_loss += float(loss.detach())
    retained = torch.randint(len(author), (CONFIG.retention_batch,), generator=retention).tolist()
    augmented = [old.augment(author[i], augmentation) for i in retained]
    for start in range(0, CONFIG.retention_batch, CONFIG.microbatch):
        words = torch.stack([v[0] for v in augmented[start:start+CONFIG.microbatch]]).cuda()
        with torch.no_grad(): latent = codec.vq.idx_to_z_q(codec.encode(words))
        logits = codec.decoder(latent.detach()); losses = per_example_loss(logits, words)
        loss = (1-CONFIG.domain_weight)*losses.sum()/CONFIG.retention_batch
        if not torch.isfinite(loss): raise ValueError('nonfinite retention loss')
        loss.backward(); retention_loss += float(loss.detach())
    if any(v.grad is not None for n, v in codec.named_parameters() if not n.startswith('decoder.')):
        raise ValueError('protected parameter received gradients')
    norm = torch.nn.utils.clip_grad_norm_(codec.decoder.parameters(), 1., error_if_nonfinite=True)
    if not norm > 0: raise ValueError('nonzero decoder gradients required')
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if torch.cuda.max_memory_allocated() > CONFIG.peak_limit_bytes: raise ValueError('own memory cap exceeded')
    return {'step': step, 'lr': rate(step), 'domain_positions': selected, 'retention_positions': retained,
        'augmentation': [v[1] for v in augmented], 'weighted_domain_CE': domain_loss,
        'weighted_retention_CE': retention_loss, 'gradient_norm': float(norm)}


def run(stage):
    root = ROOTS[stage]; storage = STORAGE/stage
    if root.exists() or storage.exists(): raise ValueError('fresh append-only training attempt required')
    qualified = data.read(QUALIFICATION); data.verify(qualified['artifacts'])
    if qualified['actual_tool_exit_code'] != 0 or qualified['tests_passed'] != 3: raise ValueError('successful CPU contracts required')
    cached = require(prepared.ROOT); require(prepared.corpus.ROOT)
    if cached['status'] != 'all99367_fixed_codes_encoded_and_read_back' or cached['frames'] != CONFIG.frames:
        raise ValueError('complete code cache required')
    if stage == 'train':
        pilot = require(ROOTS['pilot'])
        if not pilot['strict_resumed_update_passed']: raise ValueError('actual optimizer/restart pilot required')
    training, environment = runtime.codec_source.prior.runtime()
    runtime.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER'
            or torch.cuda.mem_get_info()[0] < CONFIG.peak_limit_bytes): raise ValueError('GPU2 and memory headroom required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    before_services = runtime.lifecycle.services()
    paths = [QUALIFICATION, PLAN, TEST, Path('pixi.lock'), Path('pyproject.toml'), old.DATA/'frames.pt', old.DATA/'result.json',
             old.DATA/'producer.json', old.INPUTS/'manifest.json', runtime.codec_source.BROAD/'model.pt']
    for source in (prepared.ROOT, prepared.corpus.ROOT): paths += [source/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [Path(cached['binary']), Path(cached['index']), prepared.corpus.ROOT/'frames.uint16.bin', prepared.corpus.ROOT/'index.jsonl']
    if stage == 'train': paths += [ROOTS['pilot']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    if data.sha(runtime.codec_source.BROAD/'model.pt') != runtime.codec_source.CHECKPOINT_SHA: raise ValueError('parent bytes differ')
    if data.sha(old.DATA/'frames.pt') != data.read(old.DATA/'result.json')['frames_sha256']: raise ValueError('retention corpus differs')
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(training.native.vendor.VENDOR.resolve())): files.add(p)
    root.mkdir(); storage.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)): (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, 'stage': stage, 'config': asdict(CONFIG),
        'environment': environment, 'updates': 2 if stage == 'pilot' else CONFIG.updates,
        'initial_codec_sha256': runtime.codec_source.CODEC_SHA,
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)}, 'input_hashes': {str(p): data.sha(p) for p in paths}}
    proof['fingerprint'] = training.q.digest(proof); data.write(root/'producer.json', proof)
    frames = np.memmap(prepared.corpus.ROOT/'frames.uint16.bin', dtype='<u2', mode='r', shape=(CONFIG.frames, 128))
    codes = np.memmap(cached['binary'], dtype='<u2', mode='r', shape=(CONFIG.frames, 641))
    author = torch.load(old.DATA/'frames.pt', map_location='cpu', weights_only=True, mmap=True)
    if author.shape != (16384, 128): raise ValueError('complete author TRAIN retention population required')
    orders = domain_orders(); torch.save(orders, root/'domain_orders.pt')
    torch.cuda.reset_peak_memory_stats(); codec, frozen = load_initial(training)
    controls = data.read(prepared.ROOT/'pilot.json')['positions']
    for start in range(0, len(controls), 8):
        ids = controls[start:start+8]
        with torch.no_grad(): actual = codec.encode(torch.from_numpy(np.array(frames[ids], dtype=np.int64)).cuda())
        if not np.array_equal(actual.cpu().numpy(), codes[ids]): raise ValueError('cached/repaired encoder control mismatch')
    del actual
    optimizer = optimizer_for(codec)
    retention = torch.Generator().manual_seed(CONFIG.retention_seed)
    augmentation = torch.Generator().manual_seed(CONFIG.augmentation_seed)
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    started = time.monotonic(); updates = 2 if stage == 'pilot' else CONFIG.updates; count = 0
    with (root/'trace.jsonl').open('x') as trace:
        for step in range(1, updates+1):
            selected = positions(step, orders)
            if stage == 'pilot' and step == 2: selected = controls
            record = update(step, selected, codec, optimizer, retention, augmentation, frames, codes, author)
            count += len(selected); record['elapsed_seconds'] = time.monotonic()-started
            trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush()
            if stage == 'pilot' or step % 32 == 0:
                print(json.dumps({k: record[k] for k in ('step', 'weighted_domain_CE', 'weighted_retention_CE', 'elapsed_seconds')}), flush=True)
            if stage == 'train' and (step % 1024 == 0 or step % CONFIG.steps_per_epoch == 0):
                checkpoint(storage/f'step{step}.pt', codec, optimizer, retention, augmentation, step, frozen, proof)
    final_path = storage/f'step{updates}.pt'
    if not final_path.exists(): checkpoint(final_path, codec, optimizer, retention, augmentation, updates, frozen, proof)
    if old.frozen_hash(codec) != frozen or training.native.vendor.module_hash(codec) == runtime.codec_source.CODEC_SHA:
        raise ValueError('frozen representation changed or decoder did not update')
    if stage == 'pilot':
        expected_record = update(3, positions(3, orders), codec, optimizer, retention, augmentation, frames, codes, author)
    expected = snapshot(codec, optimizer, retention, augmentation)
    del codec, optimizer; gc.collect(); torch.cuda.empty_cache()
    codec, optimizer, retention, augmentation, reloaded_frozen = restore(final_path, training, proof)
    if stage == 'pilot':
        observed_record = update(3, positions(3, orders), codec, optimizer, retention, augmentation, frames, codes, author)
        if observed_record != expected_record: raise ValueError('resumed third-update losses/selection differ')
        data.write(root/'restart_update.json', {'uninterrupted': expected_record, 'resumed': observed_record})
    if not old.same(expected, snapshot(codec, optimizer, retention, augmentation)) or reloaded_frozen != frozen or old.frozen_hash(codec) != frozen:
        raise ValueError('strict model/optimizer/RNG restart differs')
    if stage == 'train' and count != CONFIG.frames*CONFIG.epochs: raise ValueError('incomplete population exposure')
    after_services = runtime.lifecycle.services()
    if any(after_services[u]['ExecMainPID'] != before_services[u]['ExecMainPID'] for u in before_services): raise ValueError('other worker process changed')
    data.write(root/'services.json', {'before': before_services, 'after': after_services})
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    result = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS,
        'status': 'decoder_training_restart_pilot_passed' if stage == 'pilot' else 'full_recovered_decoder_training_completed',
        'updates': updates, 'domain_presentations': count, 'retention_presentations': updates*CONFIG.retention_batch,
        'strict_resumed_update_passed': stage == 'pilot', 'strict_reload_passed': True, 'frozen_state_sha256': frozen,
        'checkpoint': str(final_path), 'checkpoint_sha256': data.sha(final_path), 'initial_codec_sha256': runtime.codec_source.CODEC_SHA,
        'final_codec_sha256': training.native.vendor.module_hash(codec) if stage == 'train' else None,
        'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic()-started,
        'systems_lifecycle_only_not_quality_evidence': stage == 'pilot', 'semantic_pair_training_admitted': False,
        'task_accuracy': None, 'final_test_opened': False, 'speed_gain_measured': False,
        'artifacts': {str(p): data.sha(p) for p in [root/'producer.json', root/'trace.jsonl', root/'domain_orders.pt', root/'services.json', final_path]}}
    data.write(root/'result.json', result); print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=('pilot', 'train'))
    args = parser.parse_args(); existed = ROOTS[args.stage].exists()
    try: run(args.stage)
    except Exception as error:
        if ROOTS[args.stage].exists() and not existed: data.write(ROOTS[args.stage]/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
