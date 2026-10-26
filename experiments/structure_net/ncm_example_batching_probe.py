"""Qualify example batching on real TRAIN memories without updating weights."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from experiments.structure_net import ncm_example_batching as batching
from experiments.structure_net import ncm_onpolicy_residual_training as parent

broad = parent.broad
HYPOTHESIS = 'ncm-example-batching-v1'
ROOT = broad.runtime.ROOT/'20261026_example_batching_probe_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-example-batching-plan.md')
TEST = Path('tests/structure_net/test_ncm_example_batching.py')


@dataclass(frozen=True)
class Config:
    candidate_sizes: tuple = (1, 2, 4, 8)
    effective_examples: int = 8
    warmups: int = 1
    measurements: int = 3
    parity_seed: int = 212142
    corruption_seed: int = 213142
    dropout_seed: int = 214142
    loss_scale: float = 128.
    gradient_relative_tolerance: float = 5e-5
    gradient_absolute_tolerance: float = 1e-4
    logits_rtol: float = 1e-4
    logits_atol: float = 1e-4
    peak_limit_bytes: int = 4_900_000_000
    feature_cache_bytes: int = 256*1024**2
    physical_GPU: int = 1
    TF32: bool = False
    optimizer_steps: int = 0
    systems_lifecycle_only_not_quality_evidence: bool = True


CONFIG = Config()


def select(rows, index):
    by_id = {r['id']: r for r in rows}
    lengths = {r['id']: r['source_positions'] for r in index}
    if (len(rows) != 1376 or len(by_id) != len(rows) or set(by_id) != set(lengths)
            or any(r['split'] != 'TRAIN' for r in rows)):
        raise ValueError('unchanged1376 complete TRAIN examples required')
    ascending = sorted(by_id, key=lambda k: (lengths[k], k))
    descending = sorted(by_id, key=lambda k: (-lengths[k], k))
    return {'parity_ids': [ascending[0], descending[0]],
            'capacity_ids': descending[:CONFIG.effective_examples],
            'source_positions': {k: lengths[k] for k in dict.fromkeys([ascending[0], *descending[:8]])}}


def freeze(training, environment, selection):
    files = {Path(__file__).resolve(), Path(batching.__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(Path('experiments').resolve()):
                files.add(path)
    files.update(training.native.vendor.VENDOR.resolve().rglob('*.py'))
    inputs = [parent.PARENT/n for n in ('model.pt', 'result.json', 'actual_tool_receipt.json')]
    inputs += [broad.RUNS['prepare']/n for n in ('cases.json', 'result.json', 'actual_tool_receipt.json')]
    inputs += [broad.catalog.DEST/n for n in ('source_index.json', 'feature_manifest.json')]
    inputs += [Path('pyproject.toml'), Path('pixi.lock')]
    ROOT.mkdir(); (ROOT/'producer_source').mkdir()
    for i, path in enumerate(sorted(files)):
        (ROOT/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema': 'nal.ncm-example-batching.v1', 'hypothesis_id': HYPOTHESIS,
        'conformance': 'EXPLORATORY_SYSTEMS_DIAGNOSTIC', 'config': asdict(CONFIG),
        'selection': selection, 'environment': environment,
        'source_hashes': {str(p): broad.file_hash(p) for p in sorted(files)},
        'input_hashes': {str(p): broad.file_hash(p) for p in inputs}}
    proof['fingerprint'] = training.q.digest(proof)
    broad.write_json(ROOT/'producer.json', proof)
    return proof


def gradient_comparison(expected, params):
    records, error, norm, maximum = [], 0., 0., 0.
    presence_equal = finite = True
    for name, param in params.items():
        a, b = expected[name], param.grad
        if (a is None) != (b is None):
            presence_equal = False
        if a is None or b is None:
            records.append({'name': name, 'expected_present': a is not None, 'actual_present': b is not None})
            continue
        b = b.detach().cpu()
        finite = finite and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        delta = b.double() - a.double()
        e, n, m = float(delta.square().sum()), float(a.double().square().sum()), float(delta.abs().max())
        error += e; norm += n; maximum = max(maximum, m)
        records.append({'name': name, 'numel': a.numel(), 'squared_error': e,
                        'reference_squared_norm': n, 'max_absolute_error': m})
    if not finite:
        raise ValueError('nonfinite FP32 parity gradient')
    return {'gradient_presence_equal': presence_equal, 'finite': finite,
        'relative_gradient_L2_error': (error/max(norm, 1e-30))**.5,
        'max_abs_gradient_difference': maximum, 'parameters': records}


def parity(model, values, codes, *, noisy):
    broad.mechanism.configure(model, False)
    model.zero_grad(set_to_none=True)
    levels = model.concepts.generator.levels
    gold = torch.tensor(codes, dtype=torch.long)
    histories, counts = batching.histories(gold, levels, 8192,
        torch.Generator().manual_seed(CONFIG.parity_seed), [noisy]*len(values))
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    outputs = []
    for i, value in enumerate(values):
        f, m, target = batching.collate([value], [codes[i]], device='cuda')
        logits = model.concepts.teacher_logits(f, m, histories[i:i+1].cuda())
        batching.weighted_loss(logits, target, levels, examples_in_update=2).backward()
        outputs.append(logits.detach().cpu())
        del f, m, target, logits
    expected = {n: p.grad.detach().cpu().clone() if p.grad is not None else None for n, p in params.items()}
    model.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()
    f, m, target = batching.collate(values, codes, device='cuda')
    logits = model.concepts.teacher_logits(f, m, histories.cuda())
    batching.weighted_loss(logits, target, levels, examples_in_update=2).backward()
    actual, reference = logits.detach().cpu(), torch.cat(outputs)
    metrics = gradient_comparison(expected, params)
    metrics.update({'noisy_history': noisy, 'history_changes': counts,
        'logits_allclose': torch.allclose(actual, reference, rtol=CONFIG.logits_rtol, atol=CONFIG.logits_atol),
        'max_abs_logit_difference': float((actual-reference).abs().max()),
        'argmax_disagreements': int((actual.argmax(-1) != reference.argmax(-1)).sum()),
        'trainable_parameters_compared': sum(p.numel() for p in params.values()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    metrics['passed'] = (metrics['gradient_presence_equal'] and metrics['finite']
        and metrics['relative_gradient_L2_error'] <= CONFIG.gradient_relative_tolerance
        and metrics['max_abs_gradient_difference'] <= CONFIG.gradient_absolute_tolerance
        and metrics['logits_allclose'] and metrics['argmax_disagreements'] == 0)
    model.zero_grad(set_to_none=True)
    return metrics


def capacity(model, values, codes, size):
    broad.mechanism.configure(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    rng = torch.Generator().manual_seed(CONFIG.corruption_seed)
    levels = model.concepts.generator.levels
    records = []
    for iteration in range(CONFIG.warmups+CONFIG.measurements):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); started = time.perf_counter()
        losses, changes = [], []
        for start in range(0, len(values), size):
            f, m, target = batching.collate(values[start:start+size], codes[start:start+size], device='cuda')
            histories, count = batching.histories(target, levels, 8192, rng, [True]*len(target))
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model.concepts.teacher_logits(f, m, histories)
            loss = batching.weighted_loss(logits, target, levels, examples_in_update=len(values))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite batched FP16 loss')
            (loss*CONFIG.loss_scale).backward()
            losses.append(float(loss.detach())); changes.extend(count)
            del f, m, target, histories, logits, loss
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.div_(CONFIG.loss_scale)
                    if not torch.isfinite(p.grad).all():
                        raise ValueError('nonfinite batched FP16 gradient')
        torch.cuda.synchronize(); elapsed = time.perf_counter()-started
        item = {'iteration': iteration, 'warmup': iteration < CONFIG.warmups,
            'seconds': elapsed, 'loss': sum(losses), 'history_changes': changes,
            'finite_gradients': True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        records.append(item)
        broad.write_json(ROOT/f'capacity_B{size}_iteration{iteration}.json', item)
        print(json.dumps({'stage': 'capacity', 'microbatch': size, **item}), flush=True)
        if item['peak_allocated_bytes'] > CONFIG.peak_limit_bytes:
            return {'microbatch': size, 'status': 'allocation_budget_exceeded', 'iterations': records}
    times = [r['seconds'] for r in records if not r['warmup']]
    return {'microbatch': size, 'status': 'resource_check_passed', 'iterations': records,
        'median_effective_batch_seconds': statistics.median(times),
        'examples_per_second': len(values)/statistics.median(times),
        'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in records)}


def run():
    if ROOT.exists():
        raise ValueError('fresh append-only batching diagnostic root required')
    training, environment = broad.runtime.configure()
    broad.offload.require_rtx2060()
    training.preparation.completed(broad.RUNS['prepare'])
    for path, expected in training.q.read(broad.RUNS['prepare']/'result.json')['artifacts'].items():
        if broad.file_hash(path) != expected:
            raise ValueError('prepared TRAIN data changed: '+path)
    rows = training.q.read(broad.RUNS['prepare']/'cases.json')
    index, manifest = [training.q.read(broad.catalog.DEST/n) for n in ('source_index.json', 'feature_manifest.json')]
    selection = select(rows, index)
    model = parent.load_initial(training)
    proof = freeze(training, environment, selection)
    loader = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=CONFIG.feature_cache_bytes)
    lookup = {r['id']: r for r in rows}
    parity_rows = [lookup[k] for k in selection['parity_ids']]
    values = [loader.get(r) for r in parity_rows]
    codes = [r['oracle_codes'] for r in parity_rows]
    comparisons = []
    for noisy in (False, True):
        torch.cuda.reset_peak_memory_stats()
        result = parity(model, values, codes, noisy=noisy)
        broad.write_json(ROOT/('parity_noisy.json' if noisy else 'parity_clean.json'), result)
        comparisons.append(result)
        print(json.dumps({'stage': 'parity', **{k: v for k, v in result.items() if k != 'parameters'}}), flush=True)
        if not result['passed']:
            break
    resources = []
    if all(r['passed'] for r in comparisons) and len(comparisons) == 2:
        chosen = [lookup[k] for k in selection['capacity_ids']]
        values = [loader.get(r) for r in chosen]
        codes = [r['oracle_codes'] for r in chosen]
        for size in CONFIG.candidate_sizes:
            try:
                result = capacity(model, values, codes, size)
            except torch.cuda.OutOfMemoryError as error:
                result = {'microbatch': size, 'status': 'CUDA_out_of_memory', 'error': str(error),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            broad.write_json(ROOT/f'capacity_B{size}.json', result)
            resources.append(result)
            if result['status'] != 'resource_check_passed':
                break
    model.zero_grad(set_to_none=True)
    unchanged = training.native.vendor.module_hash(model) == parent.MODEL_SHA
    if not unchanged:
        raise ValueError('batching diagnostic unexpectedly changed model tensors')
    qualified = [r['microbatch'] for r in resources if r['status'] == 'resource_check_passed']
    result = {'schema': 'nal.ncm-example-batching.v1', 'hypothesis_id': HYPOTHESIS,
        'classification': 'FP32_parity_and_resource_diagnostic_completed' if len(comparisons) == 2 and all(r['passed'] for r in comparisons)
                          else 'FP32_example_batching_parity_failed',
        'config': asdict(CONFIG), 'producer_fingerprint': proof['fingerprint'],
        'parity': [{k: v for k, v in r.items() if k != 'parameters'} for r in comparisons],
        'resource_results': resources, 'resource_qualified_sizes': qualified,
        'weights_unchanged': unchanged, 'model_sha256': parent.MODEL_SHA,
        'feature_cache': loader.metrics(), 'optimizer_steps': 0, 'new_judge_requests': 0,
        'task_accuracy': None, 'final_test_opened': False, 'application_speed_tested': False,
        'systems_lifecycle_only_not_quality_evidence': True}
    broad.write_json(ROOT/'result.json', result)
    print(json.dumps({'classification': result['classification'], 'resource_qualified_sizes': qualified,
                      'weights_unchanged': unchanged, 'task_accuracy': None}), flush=True)
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
            broad.write_json(ROOT/'failure.json', {'exception': type(error).__name__, 'message': str(error),
                'task_accuracy': None, 'systems_lifecycle_only_not_quality_evidence': True})
        raise
