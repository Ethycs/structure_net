"""Fixed FP16 decoder-only continuation on the expanded source-disjoint corpus."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_expanded_codec_corpus as corpus
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = corpus.native, corpus.q, corpus.ROOT
domain, lifecycle = corpus.domain, corpus.domain.base
warm = corpus.component.grounded
MODULE = 'experiments.structure_net.ncm_expanded_codec_training'
HYPOTHESIS = 'ncm-expanded9936-FP16-decoder-only-continuation-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-expanded-codec-training-plan.md')
TEST = Path('tests/structure_net/test_ncm_expanded_codec_training.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_expanded_codec_training.py')
CORPUS_AUDIT = ROOT / '20261024_expanded9936_codec_corpus_audit_v1'
WORKFLOW, PILOT, TRAIN = (ROOT / ('20261024_expanded9936_codec_training_' + n + '_v1')
    for n in ('workflow', 'pilot', 'train'))
WARM_DECODER_SHA = '74d6bb3fc88e2314722c3fcbb9f8e2cb62803bfc076c1f7806d33db0a6988fca'


@dataclass(frozen=True)
class Config:
    updates: int = 1024
    pilot_updates: int = 2
    microbatch: int = 4
    accumulation: int = 8
    warmup: int = 64
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    author_probability: float = .2
    full_frame_probability: float = .2
    sampler_seed: int = 137142
    augmentation_seed: int = 138142
    dropout_seed: int = 139142


def rate(step):
    c = Config()
    if not 1 <= step <= c.updates:
        raise ValueError('outside fixed1024 decoder budget')
    if step <= c.warmup:
        return c.max_lr * step / c.warmup
    t = (step-c.warmup)/(c.updates-c.warmup)
    return c.min_lr + .5*(c.max_lr-c.min_lr)*(1+math.cos(math.pi*t))


def require_corpus():
    if q.read(corpus.DATA / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0 or q.read(CORPUS_AUDIT / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual complete corpus and independent CPU replay required')
    result = q.read(CORPUS_AUDIT / 'result.json')
    if (result['status'] != 'all9936_sources_complete_native_frames_and_disjointness_replayed'
            or result['source_cases'] != 9936 or not result['component64_excluded']
            or not result['permanent_DEV_disjoint'] or result['summary_targets_used']):
        raise ValueError('complete source-disjoint raw text corpus required')
    for path, expected in result['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('audited raw-text corpus changed')
    if file_hash(warm.DOMAIN_CHECKPOINT) != warm.DOMAIN_SHA:
        raise ValueError('fixed original domain decoder checkpoint required')


def load_codec(device='cpu'):
    if file_hash(native.RELEASE / 'files/lm1b/tokenizer.pt') != native.FILES['lm1b/tokenizer.pt']:
        raise ValueError('author native codec changed')
    OmegaConf, VQVAE, _ = native.vendor.vendor_classes()
    cfg = OmegaConf.load(native.RELEASE / 'files/lm1b/small/config.yaml')
    codec = VQVAE(**cfg.tokenizer.vqvae)
    released = torch.load(native.RELEASE / 'files/lm1b/tokenizer.pt', map_location='cpu', weights_only=True, mmap=True)
    codec.load_state_dict(released['model'], strict=True)
    saved = torch.load(warm.DOMAIN_CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    codec.decoder.load_state_dict(saved['decoder'], strict=True)
    if native.vendor.module_hash(codec.decoder) != WARM_DECODER_SHA or tuple(codec.vq.levels) != tuple(native.LEVELS):
        raise ValueError('same fixed domain decoder and native code geometry required')
    codec.eval().requires_grad_(False)
    codec.decoder.requires_grad_(True)
    return codec.to(device).float()


def augment(frame, rng):
    if torch.rand((), generator=rng).item() < Config().full_frame_probability:
        return frame.clone(), {'mode': 'full'}
    lexical = [v for v in frame.tolist()[1:-1] if v not in (0, 101, 102)]
    if not lexical:
        raise ValueError('nonempty lexical TRAIN frame required')
    length = int(torch.randint(1, len(lexical)+1, (), generator=rng))
    return torch.tensor([101, *lexical[:length], 101] + [0]*(126-length)), {'mode': 'prefix', 'length': length}


def sample(raw, author, index, sampler, augmentation):
    frames, descriptors = [], []
    for _ in range(Config().microbatch):
        if torch.rand((), generator=sampler).item() < Config().author_probability:
            row = int(torch.randint(len(author), (), generator=sampler))
            frame, info = author[row], {'kind': 'author_TRAIN', 'row': row}
        else:
            source = int(torch.randint(len(index), (), generator=sampler))
            entry = index[source]
            window = int(torch.randint(entry['frame_count'], (), generator=sampler))
            row = entry['frame_start'] + window
            frame, info = raw[row], {'kind': 'raw_source', 'source_index': source, 'window': window, 'row': row}
        value, mode = augment(frame, augmentation)
        frames.append(value)
        descriptors.append({**info, **mode, 'target_sha256': q.digest(value.tolist())})
    return torch.stack(frames), descriptors


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(corpus.__file__), corpus.PLAN,
        Path(lifecycle.__file__), Path(warm.__file__), *sorted(native.vendor.VENDOR.rglob('*.py'))]
    common = [CORPUS_AUDIT / 'result.json', CORPUS_AUDIT / 'actual_tool_receipt.json',
        corpus.DATA / 'producer.json', corpus.DATA / 'result.json', corpus.DATA / 'frames.pt',
        corpus.DATA / 'index.json', corpus.DATA / 'split_proof.json', domain.AUTHOR_DATA / 'frames.pt',
        domain.AUTHOR_DATA / 'result.json', warm.DOMAIN_CHECKPOINT, warm.DOMAIN_CHECKPOINT.parent / 'result.json',
        native.RELEASE / 'files/lm1b/tokenizer.pt', native.RELEASE / 'files/lm1b/small/config.yaml']
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), {
        **asdict(Config()), 'hypothesis_id': HYPOTHESIS, 'warm_checkpoint_sha256': warm.DOMAIN_SHA,
        'warm_decoder_sha256': WARM_DECODER_SHA, 'train_source_families': 9936,
        'component64_excluded': True, 'permanent_DEV_disjoint': True,
        'objective': 'word CE ignore PAD0, full native FP32 oracle codes; no code corruption or fine-level dropout',
        'mixture': '20percent author TRAIN /80percent uniform source then uniform complete window',
        'length_augmentation': '20percent original fullframe /80percent uniformly chosen lexical prefix',
        'codec_encoder_VQ_and_EMA_frozen': True, 'decoder_only_trainable': True,
        'parent_and_NCM_generator_absent': True, 'all_parent_MLPs_frozen': True,
        'new_random_trainable_weights': False, 'physical_GPU': 1, 'TF32': False,
        'precision': 'FP32weights/FP32encoder_codes/FP16decoder_training/FP32hardinference',
        'scaler': 'initial128,growthinterval2000', 'peak_limit_bytes': 2_000_000_000,
        'checkpoint_selection': 'fixed final1024 only; no DEV selection',
        'oracle_target_codes': True, 'source_only_summary_accuracy': None,
        'canonical_firsts_changed': False, 'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'expanded_codec_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


@torch.no_grad()
def probes(codec, raw, author, index):
    longest = min(index, key=lambda r: (-r['lexical_tokens'], r['id']))
    inputs = torch.stack([raw[0], raw[longest['frame_start']], author[0]]).cuda()
    mode = codec.decoder.training
    codec.decoder.eval()
    with torch.autocast('cuda', enabled=False):
        codes = codec.encode(inputs)
        words = codec.decoder(codec.vq.idx_to_z_q(codes)).argmax(-1)
    codec.decoder.train(mode)
    return {'codes': codes.cpu().tolist(), 'words': words.cpu().tolist()}


def optimizer_for(codec):
    return torch.optim.AdamW(codec.decoder.parameters(), lr=Config().max_lr,
        betas=(.9, .95), eps=1e-8, weight_decay=.01)


def train(pilot=False):
    corpus.component.configure()
    require_corpus()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or not torch.cuda.is_available() or '2060' not in torch.cuda.get_device_name()
            or 'SUPER' in torch.cuda.get_device_name() or torch.cuda.mem_get_info()[0] < 2_000_000_000):
        raise ValueError('physical1 RTX2060 with2GBfree required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    extra = []
    if not pilot:
        native.vendor.terminal(PILOT)
        if q.read(PILOT / 'result.json')['status'] != 'expanded_codec_FP16_training_reload_pilot_passed':
            raise ValueError('actual pilot required before full continuation')
        extra = [PILOT / n for n in ('result.json', 'initial_probes.json', 'worker_terminal_receipt.json')]
    root = PILOT if pilot else TRAIN
    p = producing(root, 'two_full_accumulation_update_lifecycle' if pilot else 'fixed1024_expanded_decoder_continuation', extra)
    raw = torch.load(corpus.DATA / 'frames.pt', map_location='cpu', weights_only=True)
    author = torch.load(domain.AUTHOR_DATA / 'frames.pt', map_location='cpu', weights_only=True)
    index = q.read(corpus.DATA / 'index.json')
    codec = load_codec('cuda')
    initial_hash = native.vendor.module_hash(codec.decoder)
    protected = lifecycle.frozen_hash(codec)
    initial = probes(codec, raw, author, index)
    if not pilot and initial != q.read(PILOT / 'initial_probes.json'):
        raise ValueError('full continuation must start at original domain decoder, not pilot')
    write_json(root / 'initial_probes.json', initial)
    torch.manual_seed(Config().dropout_seed)
    torch.cuda.manual_seed_all(Config().dropout_seed)
    sampler = torch.Generator().manual_seed(Config().sampler_seed)
    augmentation = torch.Generator().manual_seed(Config().augmentation_seed)
    optimizer = optimizer_for(codec)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    codec.decoder.train()
    torch.cuda.reset_peak_memory_stats()
    updates = Config().pilot_updates if pilot else Config().updates
    with (root / 'trace.jsonl').open('x') as trace:
        for step in range(1, updates+1):
            if codec.encoder.training or codec.vq.training or not codec.decoder.training:
                raise ValueError('only native decoder may train')
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = rate(step)
            losses, samples = [], []
            for _ in range(Config().accumulation):
                inputs, descriptors = sample(raw, author, index, sampler, augmentation)
                inputs = inputs.cuda()
                with torch.no_grad(), torch.autocast('cuda', enabled=False):
                    latent = codec.vq.idx_to_z_q(codec.encode(inputs))
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = codec.decoder(latent.detach())
                loss = F.cross_entropy(logits.float().flatten(0, 1), inputs.flatten(), ignore_index=0)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite native decoder word CE')
                scaler.scale(loss/Config().accumulation).backward()
                losses.append(loss.item())
                samples.extend(descriptors)
            if any(v.grad is not None for n, v in codec.named_parameters() if not n.startswith('decoder.')):
                raise ValueError('protected encoder/VQ received gradients')
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(codec.decoder.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite decoder gradients; no skipped-update fallback')
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() != 128.:
                raise ValueError('FP16 scaler skipped or changed an update')
            trace.write(json.dumps({'step': step, 'lr': rate(step), 'loss': sum(losses)/len(losses),
                'gradient_norm': norm.item(), 'scaler': scaler.state_dict(), 'samples': samples}, allow_nan=False)+'\n')
            trace.flush()
            if pilot or step % 32 == 0:
                print(json.dumps({'decoder_step': step, 'updates': updates, 'word_CE': sum(losses)/len(losses)}), flush=True)
    if lifecycle.frozen_hash(codec) != protected or native.vendor.module_hash(codec.decoder) == initial_hash:
        raise ValueError('protected state changed or decoder did not update')
    outputs = probes(codec, raw, author, index)
    final_hash = native.vendor.module_hash(codec.decoder)
    state = {'step': updates, 'config': asdict(Config()), 'producer_fingerprint': p['fingerprint'],
        'decoder': {k: v.detach().cpu().clone() for k, v in codec.decoder.state_dict().items()},
        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'sampler_rng': sampler.get_state(),
        'augmentation_rng': augmentation.get_state(), 'CPU_rng': torch.get_rng_state(),
        'CUDA_rng': torch.cuda.get_rng_state_all(), 'protected_sha256': protected}
    torch.save(state, root / 'model.pt')
    saved = torch.load(root / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    del state, codec, optimizer, logits, loss, latent
    gc.collect()
    torch.cuda.empty_cache()
    codec = load_codec('cuda')
    codec.decoder.load_state_dict(saved['decoder'], strict=True)
    optimizer = optimizer_for(codec)
    optimizer.load_state_dict(saved['optimizer'])
    scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng'])
    augmentation.set_state(saved['augmentation_rng'])
    torch.set_rng_state(saved['CPU_rng'])
    torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    if (native.vendor.module_hash(codec.decoder) != final_hash or lifecycle.frozen_hash(codec) != protected
            or probes(codec, raw, author, index) != outputs
            or not lifecycle.same(optimizer.state_dict(), saved['optimizer'])
            or scaler.state_dict() != saved['scaler'] or not torch.equal(sampler.get_state(), saved['sampler_rng'])
            or not torch.equal(augmentation.get_state(), saved['augmentation_rng'])
            or not torch.equal(torch.get_rng_state(), saved['CPU_rng'])
            or not lifecycle.same(torch.cuda.get_rng_state_all(), saved['CUDA_rng'])):
        raise ValueError('strict decoder/optimizer/scaler/RNG/hard-output reload failed')
    peak = torch.cuda.max_memory_allocated()
    if peak > 2_000_000_000:
        raise ValueError('expanded decoder training exceeds2GB allocation budget')
    write_json(root / 'reload_probes.json', outputs)
    native.vendor.verify(p)
    result = {'status': 'expanded_codec_FP16_training_reload_pilot_passed' if pilot else 'expanded_codec_FP16_fixed1024_trained_semantics_pending',
        'updates': updates, 'initial_decoder_sha256': initial_hash, 'final_decoder_sha256': final_hash,
        'protected_sha256_unchanged': protected, 'decoder_trainable_parameters': sum(v.numel() for v in codec.decoder.parameters()),
        'exact_reload': True, 'peak_allocated_bytes': peak, 'scaler': saved['scaler'],
        'checkpoint_sha256': file_hash(root / 'model.pt'), 'component64_excluded': True,
        'permanent_DEV_disjoint': True, 'source_only_summary_accuracy': None,
        'canonical_semantic_cache_changed': False, 'final_test_opened': False, 'speed_tested': False}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)


def workflow():
    corpus.component.configure()
    require_corpus()
    if any(root.exists() for root in (WORKFLOW, PILOT, TRAIN)):
        raise ValueError('fresh expanded decoder workflow required')
    p = producing(WORKFLOW, 'supervise_decoder_only_expanded_corpus_continuation')
    receipts = []
    for stage, root in (('pilot', PILOT), ('train', TRAIN)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'})
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'child_pid': child.pid, 'observer_pid': os.getpid(), 'observer': 'subprocess.wait', 'command': command}
        write_json(WORKFLOW / (stage+'_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('expanded decoder stage failed: '+stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'expanded_decoder_training_completed_independent_audit_pending', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot', 'train', 'workflow'))
    args = parser.parse_args()
    {'pilot': lambda: train(True), 'train': train, 'workflow': workflow}[args.stage]()
