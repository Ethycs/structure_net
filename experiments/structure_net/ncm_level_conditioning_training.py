"""Exploratory FP16native655 adaptation with revised level conditioning on GPU2."""
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

from experiments.structure_net import ncm_lm1b_native_capacity as capacity
from experiments.structure_net import ncm_level_conditioning_objective as objective
from experiments.structure_net import ncm_level_conditioning_cuda_pilot as systems
from experiments.structure_net import ncm_raw_feature_lru as streaming

lifecycle, training, native, q, ROOT = capacity.lifecycle, capacity.training, capacity.native, capacity.q, capacity.ROOT
HYPOTHESIS = 'ncm-level-conditioning-FP16-native655-source-training-v1'
MODULE = 'experiments.structure_net.ncm_level_conditioning_training'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-level-conditioning-native655-training-plan.md')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_level_conditioning_training.py')
TEST = Path('tests/structure_net/test_ncm_level_conditioning_training.py')
WORKFLOW = ROOT / '20261024_level_conditioning_FP16_native655_training_workflow_v1'
DATA = ROOT / '20261024_level_conditioning_FP16_native655_training_data_v1'
PILOT = ROOT / '20261024_level_conditioning_FP16_native655_training_pilot_v1'
TRAIN = ROOT / '20261024_level_conditioning_FP16_native655_training_train_v1'
CAPACITY_AUDIT = ROOT / '20261024_lm1b_native16_capacity_audit_v1'
CAPACITY_CHECKPOINT_SHA = 'a7941cb921662c04294ab5267aebeaad3ba1d17e8f140fb095e430316d542003'


@dataclass(frozen=True)
class Config:
    updates: int = 1024
    microbatch: int = 1
    accumulation: int = 16
    warmup: int = 64
    max_lr: float = 5e-5
    min_lr: float = 5e-6
    sampler_seed: int = 126142
    neighbor_seed: int = 127142
    dropout_seed: int = 128142


def rate(step):
    c = Config()
    if not 1 <= step <= c.updates:
        raise ValueError('outside fixed1024 native adaptation budget')
    if step <= c.warmup:
        return c.max_lr * step / c.warmup
    ratio = (step - c.warmup) / (c.updates - c.warmup)
    return c.min_lr + .5 * (c.max_lr - c.min_lr) * (1 + math.cos(math.pi * ratio))


def require_capacity():
    capacity.require_systems()
    if (q.read(CAPACITY_AUDIT / 'actual_tool_terminal_receipt.json')['exit_code'] != 0
            or q.read(capacity.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('actual CPU audited completed native16 control required')
    result = q.read(capacity.TRAIN / 'result.json')
    if (result['checkpoint_sha256'] != CAPACITY_CHECKPOINT_SHA or result['updates'] != 256
            or result['word_ID_fit_diagnostic_only'] != 1. or not result['strict_reload_passed']):
        raise ValueError('native TRAIN learnability proof differs')
    for path, expected in q.read(CAPACITY_AUDIT / 'result.json')['artifacts'].items():
        if lifecycle.file_hash(path) != expected:
            raise ValueError('audited native16 evidence changed:' + path)
    if q.read(systems.DEST / 'actual_tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('actual spareGPU2 FP16objective pilot required')
    state = q.read(systems.DEST / 'result.json')
    native.vendor.verify(q.read(systems.DEST / 'producer.json'))
    if (state['status'] != 'FP16_level_conditioning_two_update_lifecycle_passed'
            or state['updates'] != 2 or state['peak_allocated_bytes'] > 4_900_000_000
            or not state['strict_model_optimizer_scaler_rng_hard_output_reload']
            or state['initial_trainable_sha256'] != capacity.INITIAL
            or lifecycle.file_hash(systems.DEST / 'model.pt') != state['checkpoint_sha256']):
        raise ValueError('actual FP16native root/decoder/noise/reload/memory pilot differs')


def require_gpu():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or not torch.cuda.is_available() or '2060 SUPER' not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 5_000_000_000):
        raise ValueError('physicalGPU2 RTX2060SUPER with5GBfree required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def attention_hash(model):
    return q.digest({name: native.vendor.module_hash(module) for name, module in (
        ('concept_memory', model.concepts.source_memory), ('concept_adapters', model.concepts.adapters),
        ('word_memory', model.word_memory), ('word_adapters', model.word_adapters))})


def choose(rows):
    result = [r for r in rows if r['target_origin'] == 'original_approved_target_unchanged']
    if len(result) != 655 or len({r['id'] for r in result}) != 655:
        raise ValueError('all655 unchanged original targets required;35revised labels excluded')
    return result


def split_guard(rows, development):
    parts = [rows, development['short_development'], development['long_diagnostic']]
    if [len(part) for part in parts] != [655, 64, 64]:
        raise ValueError('complete655/64/64 populations required')
    proof = {}
    for key in ('id', 'source_id', 'group_id', 'source_sha256'):
        sets = [{r[key] for r in part} for part in parts]
        if (any(len(s) != len(p) for s, p in zip(sets, parts))
                or any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))):
            raise ValueError('native655 TRAIN/DEV identity/session/group/source overlap')
        proof[key] = {'train_unique': 655, 'short_unique': 64, 'long_unique': 64, 'overlap': 0}
    return proof


def producing(root, stage, inputs=(), extra=None):
    files = [Path(__file__), PLAN, AUDITOR, TEST, Path(capacity.__file__), capacity.PLAN, capacity.AUDITOR,
        Path(lifecycle.__file__), Path(lifecycle.mechanism.__file__), Path(lifecycle.objective.__file__),
        Path(lifecycle.objective.original.__file__), Path(lifecycle.head.__file__), Path(training.__file__),
        Path(objective.__file__), Path(objective.corruption.__file__), Path(streaming.__file__), Path(systems.__file__), systems.PLAN]
    common = [CAPACITY_AUDIT / 'result.json', CAPACITY_AUDIT / 'actual_tool_terminal_receipt.json',
        capacity.WORKFLOW / 'tool_terminal_receipt.json', capacity.TRAIN / 'result.json', lifecycle.WARM / 'model.pt',
        lifecycle.WARM / 'result.json', lifecycle.WARM_WORKFLOW / 'tool_terminal_receipt.json', training.CACHE / 'outputs.json']
    common += [native.RELEASE / 'files' / name for name in native.FILES]
    common += [native.RELEASE / 'files/lm1b/small/config.yaml',
        *sorted(p for p in (native.INVENTORY / 'bert-tokenizer').iterdir() if p.is_file())]
    common += [systems.DEST / n for n in ('producer.json', 'result.json', 'actual_tool_terminal_receipt.json')]
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {**asdict(Config()), **(extra or {}),
        'hypothesis_id': HYPOTHESIS, 'train_cases': 655, 'original_revised35_excluded': True,
        'labels_not_all_semantically_qualified': True, 'source_disjoint_task_accuracy': None,
        'native_trainable_parameters': 127732736, 'encoder_and_VQ_frozen': True, 'Llama_body_loaded': False,
        'warm_checkpoint_sha256': lifecycle.WARM_SHA, 'initial_trainable_sha256': capacity.INITIAL,
        'native16_or_pilot_weights_reused': False, 'new_random_trainable_weights': False,
        'precision': 'FP32weights/FP16training/FP32hardinference', 'TF32': False, 'peak_limit_bytes': 4_900_000_000,
        'physical_GPU': 2, 'host_feature_cache_budget_bytes': 512 * 1024**2,
        'sampler': 'independent torch.randint655 micro1,accum16 with replacement',
        'loss': 'clean.5C+.7W,noisy.5C+.3W dividedaccum;firstmicro eachupdate self.25C+1W unscaled viahard-forward wordgradient',
        'noise': 'fixed linearly decreasing level rates.75to.25;generatedrooteligible;targetsintact',
        'FP16scaler': 'initial128,growthinterval2000', 'not_exact_author_noise_endpoints': True,
        'comparison_confounded_by_noise_precision_microbatch_and_seed': True,
        'checkpoint_selection': 'fixed final1024 only',
        'no_DEV_or_final_model_selection': True, 'all_final_sources_closed': True, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = lifecycle.file_hash(path)
        (root / 'producer_source' / f'native655_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    lifecycle.write_json(root / 'producer.json', p)
    return p


def prepare():
    require_capacity()
    rows = choose(q.read(training.DATA / 'cases.json'))
    manifest = q.read(training.DATA / 'manifest.json')
    original_proof = split_guard(rows, q.read(training.DEV))
    p = producing(DATA, 'all655_original_TRAIN_only', [training.DATA / 'cases.json', training.DATA / 'manifest.json', training.DEV])
    longest = min(range(655), key=lambda i: (-manifest[rows[i]['id']]['source_tokens'], rows[i]['id']))
    if longest == 0:
        raise ValueError('distinct first/longest source pilot required')
    ids = {r['id'] for r in rows}
    diagnostic_ids = [r['id'] for r in q.read(capacity.DATA / 'cases.json')]
    if len(diagnostic_ids) != 16 or not set(diagnostic_ids) <= ids:
        raise ValueError('same fixed16 TRAIN-only reload diagnostic required')
    lifecycle.write_json(DATA / 'cases.json', rows)
    lifecycle.write_json(DATA / 'manifest.json', {r['id']: manifest[r['id']] for r in rows})
    lifecycle.write_json(DATA / 'selection.json', {'pilot_indices': [0, longest], 'diagnostic_ids': diagnostic_ids})
    lifecycle.write_json(DATA / 'split_proof.json', original_proof)
    native.vendor.verify(p)
    lifecycle.write_json(DATA / 'result.json', {'status': 'all655_original_TRAIN_prepared', 'cases': 655,
        'task_accuracy': None, 'all_final_sources_closed': True})


def save(path, model, opt, scaler, sampler, neighbor, step, producer):
    torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
        'sampler_rng': sampler.get_state(), 'neighbor_rng': neighbor.get_state(),
        'CPU_rng': torch.get_rng_state(), 'CUDA_rng': torch.cuda.get_rng_state_all(), 'step': step,
        'config': asdict(Config()), 'producer_fingerprint': producer['fingerprint']}, path)


def train(pilot=False):
    require_gpu()
    native.vendor.corpus.quality.dependencies()
    native.vendor.terminal(DATA)
    if not pilot:
        native.vendor.terminal(PILOT)
        if not q.read(PILOT / 'result.json')['strict_reload_passed']:
            raise ValueError('actual full-native655 data-path pilot required')
    rows, manifest, selection = q.read(DATA / 'cases.json'), q.read(DATA / 'manifest.json'), q.read(DATA / 'selection.json')
    codes = {r['id']: r['codes'] for r in q.read(training.CACHE / 'outputs.json')}
    root = PILOT if pilot else TRAIN
    p = producing(root, 'native655_two_update_pilot' if pilot else 'native655_fixed1024_training',
        [DATA / n for n in ('cases.json', 'manifest.json', 'selection.json', 'split_proof.json', 'worker_terminal_receipt.json')]
        + [Path(manifest[r['id']]['path']) for r in rows], {'pilot': pilot, 'actual_updates': 2 if pilot else 1024,
            'actual_accumulation': 2 if pilot else 16})
    features = streaming.RawFeatureCache(rows, manifest)
    model = capacity.fresh()
    diagnostics = [{r['id']: r for r in rows}[identity] for identity in selection['diagnostic_ids']]
    initial = capacity.outputs(model, diagnostics, manifest)
    if not pilot and initial != q.read(PILOT / 'initial_outputs.json'):
        raise ValueError('larger run must reset original initializer, not pilot/native16 weights')
    lifecycle.write_json(root / 'initial_outputs.json', initial)
    sampler = torch.Generator().manual_seed(Config().sampler_seed)
    neighbor = torch.Generator().manual_seed(Config().neighbor_seed)
    torch.manual_seed(Config().dropout_seed)
    torch.cuda.manual_seed_all(Config().dropout_seed)
    opt = lifecycle.optimizer(model)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    updates, accumulation = (2, 2) if pilot else (1024, 16)
    root_before = model.concepts.generator.l0_queries.detach().cpu().clone()
    initial_native = {'generator': native.vendor.module_hash(model.concepts.generator),
        'word_decoder': native.vendor.module_hash(model.codec.decoder), 'attention': attention_hash(model)}
    torch.cuda.reset_peak_memory_stats()
    with (root / 'trace.jsonl').open('x') as stream:
        for step in range(1, updates + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            lr = Config().max_lr if pilot else rate(step)
            for group in opt.param_groups:
                group['lr'] = lr
            micro = []
            for index in range(accumulation):
                indices = [selection['pilot_indices'][index % 2]] if pilot else torch.randint(655, (1,), generator=sampler).tolist()
                f, m, c, w = training.batch(rows, indices, features, codes)
                loss = objective.backward_microbatch(model, f, m, c, w, neighbor, accumulation=accumulation,
                    self_history=index == 0, dtype=torch.float16, scaler=scaler)
                micro.append({'indices': indices, 'case_ids': [rows[i]['id'] for i in indices], 'source_positions': f.shape[1], 'losses': loss})
            if (any(v.grad is not None for module in (model.codec.encoder, model.codec.vq) for v in module.parameters())
                    or model.concepts.generator.l0_queries.grad is None
                    or not torch.count_nonzero(model.concepts.generator.l0_queries.grad)):
                raise ValueError('native root must receive gradients and frozen geometry must not')
            scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise ValueError('nonfinite full-native655 gradient')
            scaler.step(opt)
            scaler.update()
            stream.write(json.dumps({'step': step, 'lr': lr, 'gradient_norm': float(norm),
                'scaler': scaler.state_dict(), 'microbatches': micro}, allow_nan=False) + '\n')
            stream.flush()
            if step % 16 == 0 or pilot:
                print(json.dumps({'step': step, 'updates': updates, 'self_word_loss': micro[0]['losses']['self_word_loss']}), flush=True)
    del group, f, m, c, w
    final_hash = lifecycle.trainable_hash(model)
    native_hashes = {'generator': native.vendor.module_hash(model.concepts.generator), 'word_decoder': native.vendor.module_hash(model.codec.decoder)}
    root_updated = not torch.equal(root_before, model.concepts.generator.l0_queries.detach().cpu())
    if (final_hash == capacity.INITIAL or not root_updated
            or any(native_hashes[k] == initial_native[k] for k in native_hashes)
            or attention_hash(model) == initial_native['attention']
            or lifecycle.protected(model) != {'encoder': lifecycle.ENCODER, 'VQ': lifecycle.VQ}):
        raise ValueError('actual full-native learning or frozen geometry differs')
    before_reload = capacity.outputs(model, diagnostics, manifest)
    save(root / 'model.pt', model, opt, scaler, sampler, neighbor, updates, p)
    model.close()
    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(root / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model = capacity.fresh()
    model.load_state_dict(saved['model'], strict=True)
    opt = lifecycle.optimizer(model)
    opt.load_state_dict(saved['optimizer'])
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    scaler.load_state_dict(saved['scaler'])
    sampler.set_state(saved['sampler_rng'])
    neighbor.set_state(saved['neighbor_rng'])
    torch.set_rng_state(saved['CPU_rng'])
    torch.cuda.set_rng_state_all(saved['CUDA_rng'])
    if (capacity.outputs(model, diagnostics, manifest) != before_reload or not lifecycle.same(model.state_dict(), saved['model'])
            or not lifecycle.same(opt.state_dict(), saved['optimizer']) or scaler.state_dict() != saved['scaler']
            or lifecycle.trainable_hash(model) != final_hash
            or not torch.equal(sampler.get_state(), saved['sampler_rng']) or not torch.equal(neighbor.get_state(), saved['neighbor_rng'])
            or not torch.equal(torch.get_rng_state(), saved['CPU_rng']) or not lifecycle.same(torch.cuda.get_rng_state_all(), saved['CUDA_rng'])
            or lifecycle.protected(model) != {'encoder': lifecycle.ENCODER, 'VQ': lifecycle.VQ}):
        raise ValueError('strict full-native655 fullstate/optimizer/allRNG/source-only reload failed')
    peak = torch.cuda.max_memory_allocated()
    if peak > 4_900_000_000:
        raise ValueError('FP16native655 exceeds4.9GB frozen budget')
    lifecycle.write_json(root / 'diagnostic_outputs.json', before_reload)
    native.vendor.verify(p)
    result = {'status': 'level_conditioning_FP16_native655_training_completed_source_only_DEV_pending', 'updates': updates, 'train_cases': 655,
        'trainable_parameters': 127732736, 'initial_trainable_sha256': capacity.INITIAL, 'final_trainable_sha256': final_hash,
        'initial_attention_sha256': initial_native['attention'], 'final_attention_sha256': attention_hash(model),
        'native16_or_pilot_weights_reused': False, 'strict_reload_passed': True, 'root_queries_updated': root_updated,
        'protected_codec_hashes': lifecycle.protected(model), 'peak_allocated_bytes': peak, 'final_native_hashes': native_hashes,
        'checkpoint_sha256': lifecycle.file_hash(root / 'model.pt'), 'source_disjoint_task_accuracy': None,
        'feature_cache': features.metrics(), 'scaler': scaler.state_dict(), 'physical_GPU': 2,
        'precision': 'FP32weights/FP16training/FP32hardinference',
        'all_final_sources_closed': True, 'speed_tested': False, 'producer_fingerprint': p['fingerprint']}
    lifecycle.write_json(root / 'result.json', result)
    model.close()
    print(json.dumps(result), flush=True)


def workflow():
    require_capacity()
    if any(p.exists() for p in (WORKFLOW, DATA, PILOT, TRAIN)):
        raise ValueError('fresh full-native655 training roots required')
    p = producing(WORKFLOW, 'supervise_native655_fixed1024_adaptation')
    receipts = []
    for stage, root, gpu in (('prepare', DATA, ''), ('pilot', PILOT, '2'), ('train', TRAIN, '2')):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
        lifecycle.write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'command': command, 'child_pid': child.pid, 'observer_pid': os.getpid(), 'observer': 'subprocess.wait'}
        lifecycle.write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            lifecycle.write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('full-native655 stage failed:' + stage)
    native.vendor.verify(p)
    lifecycle.write_json(WORKFLOW / 'result.json', {'status': 'level_conditioning_FP16_native655_training_workflow_completed_source_only_DEV_pending', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'train', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: train(True), 'train': train, 'workflow': workflow}[args.stage]()
