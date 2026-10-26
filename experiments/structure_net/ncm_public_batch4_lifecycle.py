"""Qualify real public extremes, four-example gradients and resumed updates."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import sys
import types

import torch

from experiments.structure_net import ncm_example_batching as batching
from experiments.structure_net import ncm_example_batching_probe as probe
from experiments.structure_net import ncm_trainable_example_batching as serial
from experiments.structure_net import ncm_lm1b_native_finetuning as native_reference
from experiments.structure_net import ncm_public_feature_store as public

parent, broad, data = probe.parent, probe.broad, public.data
ROOT = data.ROOT/'20261026_public_batch4_lifecycle_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-batch4-lifecycle-plan.md')
AMENDMENT = Path('docs/07 - Status Reports/2026-10-26_ncm-public-batch4-root-gradient-repair.md')
TEST = Path('tests/structure_net/test_ncm_public_batch4_lifecycle.py')


@dataclass(frozen=True)
class Config:
    microbatch: int = 4
    effective_examples: int = 8
    updates: int = 2
    sampler_seed: int = 221142
    noise_seed: int = 222142
    dropout_seed: int = 223142
    gradient_relative_tolerance: float = 5e-5
    gradient_absolute_tolerance: float = 1e-4
    peak_limit_bytes: int = 4_900_000_000
    physical_GPU: int = 1
    TF32: bool = False
    systems_lifecycle_only_not_quality_evidence: bool = True


CONFIG = Config()


def select(entries):
    if len(entries) < 4 or len({r['id'] for r in entries}) != len(entries):
        raise ValueError('distinct public pilot feature records required')
    chosen = []
    for criterion in (lambda r: r['source_positions'], lambda r: abs(r['source_positions']-768),
                      lambda r: -r['native_prompt_positions'], lambda r: -r['source_positions']):
        row = min((r for r in entries if r['id'] not in chosen), key=lambda r: (criterion(r), r['id']))
        chosen.append(row['id'])
    if (max(r['source_positions'] for r in entries if r['id'] in chosen) != max(r['source_positions'] for r in entries)
            or max(r['native_prompt_positions'] for r in entries if r['id'] in chosen) != max(r['native_prompt_positions'] for r in entries)):
        raise ValueError('both measured public length extremes required')
    return chosen


def memory(phase):
    torch.cuda.synchronize()
    value = {'phase': phase, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'allocated_bytes': torch.cuda.memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    with (ROOT/'memory.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    if value['peak_allocated_bytes'] > CONFIG.peak_limit_bytes:
        raise ValueError('public batch4 whole-lifecycle allocation budget exceeded')
    return value


def parity(model, values, codes, noisy):
    broad.mechanism.configure(model, False)
    model.zero_grad(set_to_none=True)
    levels = model.concepts.generator.levels
    gold = torch.tensor(codes, dtype=torch.long)
    histories, changes = batching.histories(gold, levels, 8192,
        torch.Generator().manual_seed(CONFIG.noise_seed), [noisy]*4)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    outputs = []
    model.concepts.teacher_logits = types.MethodType(native_reference.teacher_logits, model.concepts)
    for i in range(4):
        features, mask, target = batching.collate([values[i]], [codes[i]], device='cuda')
        logits = model.concepts.teacher_logits(features, mask, histories[i:i+1].cuda())
        batching.weighted_loss(logits, target, levels, examples_in_update=4).backward()
        outputs.append(logits.detach().cpu())
        del features, mask, target, logits
    expected = {n: p.grad.detach().cpu().clone() if p.grad is not None else None for n, p in params.items()}
    root_name = next(n for n, p in params.items() if p is model.concepts.generator.l0_queries)
    if expected[root_name] is None or not expected[root_name].abs().sum():
        raise ValueError('original native reference must train learned root queries')
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    model.concepts.teacher_logits = types.MethodType(serial.teacher_logits, model.concepts)
    features, mask, target = batching.collate(values, codes, device='cuda')
    logits = model.concepts.teacher_logits(features, mask, histories.cuda())
    batching.weighted_loss(logits, target, levels, examples_in_update=4).backward()
    actual, reference = logits.detach().cpu(), torch.cat(outputs)
    result = probe.gradient_comparison(expected, params)
    result.update(noisy_history=noisy, history_changes=changes,
        logits_allclose=torch.allclose(actual, reference, rtol=1e-4, atol=1e-4),
        max_abs_logit_difference=float((actual-reference).abs().max()),
        argmax_disagreements=int((actual.argmax(-1) != reference.argmax(-1)).sum()),
        trainable_parameters_compared=sum(p.numel() for p in params.values()))
    result['passed'] = (result['gradient_presence_equal'] and result['finite'] and result['logits_allclose']
        and result['relative_gradient_L2_error'] <= CONFIG.gradient_relative_tolerance
        and result['max_abs_gradient_difference'] <= CONFIG.gradient_absolute_tolerance
        and result['argmax_disagreements'] == 0)
    model.zero_grad(set_to_none=True)
    data.write(ROOT/('parity_noisy.json' if noisy else 'parity_clean.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'parameters'}), flush=True)
    memory('parity_noisy' if noisy else 'parity_clean')
    if not result['passed']:
        raise ValueError('four-example FP32 gradient/logit parity failed')


def save(path, model, opt, scaler, sampler, noise, step, proof):
    if path.exists():
        raise ValueError('checkpoint overwrite forbidden')
    value = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'sampler_rng': sampler.get_state(),
        'noise_rng': noise.get_state(), 'CPU_rng': torch.get_rng_state(), 'CUDA_rng': torch.cuda.get_rng_state_all(),
        'step': step, 'producer_fingerprint': proof['fingerprint']}
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary); temporary.rename(path)


def update(model, values, codes, opt, scaler, sampler, noise):
    broad.mechanism.configure(model, True)
    opt.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    records = []
    for micro in range(2):
        order = torch.randperm(4, generator=sampler).tolist()
        features, mask, target = batching.collate([values[i] for i in order], [codes[i] for i in order], device='cuda')
        histories, changes = batching.histories(target, model.concepts.generator.levels, 8192, noise,
                                                [False, True, False, True])
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model.concepts.teacher_logits(features, mask, histories)
        loss = batching.weighted_loss(logits, target, model.concepts.generator.levels, examples_in_update=8)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite four-example loss')
        scaler.scale(loss).backward()
        records.append({'order': order, 'changes': changes, 'weighted_loss': float(loss.detach())})
        del features, mask, target, histories, logits, loss
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    if not torch.isfinite(norm) or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError('nonfinite or protected parameter gradient')
    if model.concepts.generator.l0_queries.grad is None or not model.concepts.generator.l0_queries.grad.abs().sum():
        raise ValueError('coarsest root queries must receive training gradients')
    previous = scaler.get_scale()
    scaler.step(opt); scaler.update(); broad.offload.assert_cpu_state(opt)
    if scaler.get_scale() < previous:
        raise ValueError('optimizer update skipped')
    opt.zero_grad(set_to_none=True)
    return {'microbatches': records, 'gradient_norm': float(norm), 'scale': scaler.get_scale()}


@torch.no_grad()
def outputs(model, values):
    broad.mechanism.configure(model, False)
    result = []
    for value in values:
        features = value.unsqueeze(0).cuda()
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device='cuda')
        codes = broad.mechanism.generate(model, features, mask)
        words = model.word_logits(features, mask, codes).argmax(-1)
        result.append({'codes': codes.cpu().tolist()[0], 'raw_word_ids': words.cpu().tolist()[0]})
    return result


def freeze(training, environment, identities, extra):
    files = {Path(__file__).resolve(), PLAN.resolve(), AMENDMENT.resolve(), TEST.resolve(),
             Path('tests/structure_net/test_ncm_trainable_example_batching.py').resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    paths = [parent.PARENT/name for name in ('model.pt', 'result.json', 'actual_tool_receipt.json')]
    paths += [public.ROOTS['pilot']/name for name in ('result.json', 'actual_tool_receipt.json', 'audit_result.json', 'audit_actual_tool_receipt.json')]
    paths += [Path('pyproject.toml'), Path('pixi.lock'), *extra]
    ROOT.mkdir(); (ROOT/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-public-batch4-lifecycle.v1', 'hypothesis_id': 'ncm-public-batch4-lifecycle-v1',
        'config': asdict(CONFIG), 'selection_ids': identities, 'environment': environment,
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in paths}}
    proof['fingerprint'] = training.q.digest(proof)
    data.write(ROOT/'producer.json', proof)
    return proof


def run():
    if ROOT.exists():
        raise ValueError('fresh append-only public batch4 lifecycle root required')
    from experiments.structure_net import ncm_public_code_targets as targets
    data.completed(targets.ROOTS['full'])
    target_receipt = data.read(targets.ROOTS['full']/'audit_actual_tool_receipt.json')
    data.verify(target_receipt['artifacts'])
    target_audit = data.read(targets.ROOTS['full']/'audit_result.json')
    if (target_receipt['actual_tool_exit_code'] != 0 or not target_audit['all_bindings_replayed']
            or not target_audit['targets_admitted_for_training'] or target_audit['cases'] != 162353):
        raise ValueError('actual exhaustive public code-target binding audit required')
    training, environment = broad.runtime.configure()
    broad.offload.require_rtx2060()
    data.completed(public.ROOTS['pilot'])
    receipt = data.read(public.ROOTS['pilot']/'audit_actual_tool_receipt.json')
    data.verify(receipt['artifacts'])
    if receipt['actual_tool_exit_code'] != 0 or not data.read(public.ROOTS['pilot']/'audit_result.json')['all_disk_records_replayed']:
        raise ValueError('actual passed public feature disk pilot required')
    source = public.metadata.ROOTS['full']
    with (source/'index.jsonl').open() as stream:
        all_meta = {r['id']: r for r in map(json.loads, stream)}
    with (public.STORAGE/'pilot/index.jsonl').open() as stream:
        packed = {r['id']: r for r in map(json.loads, stream)}
    chosen = select([all_meta[k] for k in packed])
    with (targets.ROOTS['full']/'index.jsonl').open() as stream:
        target_entries = {r['id']: r for r in map(json.loads, stream) if r['id'] in chosen}
    with (targets.ROOTS['full']/'codes.jsonl').open('rb') as stream:
        saved_labels = {k: public.audited.read_record(stream, target_entries[k]) for k in chosen}
    _, _, pairs = targets.ready()
    rows = [pairs[all_meta[k]['catalog_position']] for k in chosen]
    values = [public.store.load(packed[k], {**{f: all_meta[k][f] for f in
        ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
        'original_body_sha256': public.original.base.EXPECTED_BODY}, source_positions=all_meta[k]['source_positions'],
        metadata_record_sha256=all_meta[k]['record_sha256']) for k in chosen]
    pairs.close()
    model = parent.load_initial(training)
    model.concepts.teacher_logits = types.MethodType(serial.teacher_logits, model.concepts)
    broad.mechanism.configure(model, False)
    frozen = training.prior.protected(model)
    with torch.no_grad():
        codes = [model.codec.encode(torch.tensor([r['native_word_ids']], device='cuda'))[0].cpu().tolist() for r in rows]
    if any(targets.bind(row, code) != saved_labels[row['id']] for row, code in zip(rows, codes)):
        raise ValueError('full public label corpus differs from selected initializer codec')
    proof = freeze(training, environment, chosen, [source/'index.jsonl', public.STORAGE/'pilot/index.jsonl',
        *[targets.ROOTS['full']/name for name in ('result.json', 'actual_tool_receipt.json', 'index.jsonl', 'codes.jsonl',
                                               'audit_result.json', 'audit_actual_tool_receipt.json')]])
    data.write(ROOT/'cases.json', [{'id': r['id'], 'source_positions': len(v), 'native_word_ids': r['native_word_ids'],
                                  'codes': c} for r, v, c in zip(rows, values, codes)])
    torch.cuda.reset_peak_memory_stats()
    for noisy in (False, True):
        parity(model, values, codes, noisy)
    broad.mechanism.configure(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 105323136:
        raise ValueError('unchanged complete trainable generator parameter set required')
    opt = broad.offload.CPUAdamW(params, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    sampler = torch.Generator().manual_seed(CONFIG.sampler_seed)
    noise = torch.Generator().manual_seed(CONFIG.noise_seed)
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    for step in (1, 2):
        record = update(model, values, codes, opt, scaler, sampler, noise)
        memory('update_'+str(step)); data.write(ROOT/f'update_{step}.json', record)
        save(ROOT/f'step_{step}.pt', model, opt, scaler, sampler, noise, step, proof)
        if training.prior.protected(model) != frozen:
            raise ValueError('protected tensors changed')
    expected_outputs = outputs(model, values)
    data.write(ROOT/'outputs_after_step2.json', expected_outputs)
    del opt; gc.collect(); torch.cuda.empty_cache()
    first = torch.load(ROOT/'step_1.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(first['model'], strict=True)
    opt = broad.offload.CPUAdamW(params, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    opt.load_state_dict(first['optimizer']); scaler.load_state_dict(first['scaler'])
    sampler.set_state(first['sampler_rng']); noise.set_state(first['noise_rng'])
    torch.set_rng_state(first['CPU_rng']); torch.cuda.set_rng_state_all(first['CUDA_rng'])
    same = training.preparation.checkpoint.training.lifecycle.same
    if not same(model.state_dict(), first['model']) or not same(opt.state_dict(), first['optimizer']):
        raise ValueError('checkpoint reload differs')
    data.write(ROOT/'resumed_update_2.json', update(model, values, codes, opt, scaler, sampler, noise))
    second = torch.load(ROOT/'step_2.pt', map_location='cpu', weights_only=True, mmap=True)
    actual = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(),
        'sampler_rng': sampler.get_state(), 'noise_rng': noise.get_state(), 'CPU_rng': torch.get_rng_state(),
        'CUDA_rng': torch.cuda.get_rng_state_all()}
    equality = {k: same(v, second[k]) for k, v in actual.items()}
    data.write(ROOT/'resumed_state_equality.json', equality)
    if not all(equality.values()) or outputs(model, values) != expected_outputs:
        raise ValueError('resumed second update or source-only outputs differ')
    peak = memory('resumed_evaluation')['peak_allocated_bytes']
    if training.prior.protected(model) != frozen or training.native.vendor.module_hash(model) == parent.MODEL_SHA:
        raise ValueError('protected state changed or trainable model did not update')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    value = {'status': 'public_batch4_lifecycle_passed', 'hypothesis_id': proof['hypothesis_id'],
        'FP32_clean_and_noisy_gradient_parity_passed': True, 'strict_resumed_update_passed': True,
        'source_only_outputs_reproduced': True, 'protected_tensors_unchanged': True,
        'trainable_parameters': 105323136, 'optimizer_updates': 2, 'replayed_optimizer_updates': 1,
        'peak_GPU_allocated_bytes': peak, 'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False,
        'systems_lifecycle_only_not_quality_evidence': True,
        'artifacts': {str(p): data.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    data.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)
    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('run',))
    parser.parse_args()
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if not existed and ROOT.exists():
            data.write(ROOT/'failure.json', {'exception': type(error).__name__, 'message': str(error),
                'task_accuracy': None, 'systems_lifecycle_only_not_quality_evidence': True})
        raise
