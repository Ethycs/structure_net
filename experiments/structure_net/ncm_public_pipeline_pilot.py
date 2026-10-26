"""Qualify the actual mixed-data effective128 pipeline before larger training."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
import types

import torch

from experiments.structure_net import ncm_public_training_bundle as bundle
from experiments.structure_net import ncm_public_batch4_lifecycle as lifecycle
from experiments.structure_net import ncm_public_training_engine as engine
from experiments.structure_net import ncm_public_curriculum as curriculum

data, broad, parent = bundle.data, lifecycle.broad, lifecycle.parent
ROOT = data.ROOT/'20261026_public_pipeline_pilot_v1'
CHECKPOINTS = Path('/data2/structure_net/ncm_public_curriculum_20261026_v1/pilot')
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-curriculum-plan.md')
TESTS = [Path('tests/structure_net')/name for name in (
    'test_ncm_public_curriculum.py', 'test_ncm_public_training_engine.py',
    'test_ncm_public_pipeline_pilot.py', 'test_ncm_public_training.py')]


def ready(training):
    qualified = data.completed(lifecycle.ROOT)
    qualification = data.read(lifecycle.ROOT/'producer.json')
    data.verify(qualification['source_hashes']); data.verify(qualification['input_hashes'])
    if (qualified['status'] != 'public_batch4_lifecycle_passed'
            or not qualified['FP32_clean_and_noisy_gradient_parity_passed']
            or not qualified['strict_resumed_update_passed']
            or not qualified['protected_tensors_unchanged']):
        raise ValueError('corrected native-reference batch4 GPU lifecycle must pass first')
    public = bundle.open_dataset()
    data.completed(broad.RUNS['prepare'])
    anchors = data.read(broad.RUNS['prepare']/'cases.json')
    if len(public) != curriculum.CONFIG.public_cases or len(anchors) != curriculum.CONFIG.anchor_cases:
        raise ValueError('complete admitted public and reviewed populations required')
    index, manifest = [data.read(broad.catalog.DEST/name) for name in ('source_index.json', 'feature_manifest.json')]
    cache = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=256*1024**2)
    return engine.MixedSources(public, anchors, cache)


def model_and_optimizer(training):
    model = parent.load_initial(training)
    model.concepts.teacher_logits = types.MethodType(lifecycle.serial.teacher_logits, model.concepts)
    broad.mechanism.configure(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 105323136:
        raise ValueError('exact complete BROAD trainable population required')
    opt = broad.offload.CPUAdamW(params, lr=curriculum.CONFIG.max_lr,
                               betas=(.9, .95), eps=1e-8, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)
    return model, opt, scaler


def freeze(training, environment, *, root=ROOT, extra=(), stage='pilot'):
    files = {Path(__file__).resolve(), PLAN.resolve(), *[p.resolve() for p in TESTS]}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    paths = [bundle.ROOT/name for name in ('manifest.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [lifecycle.ROOT/name for name in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [broad.RUNS['prepare']/name for name in ('cases.json', 'selection.json', 'oracle_outputs.json',
                                                  'result.json', 'actual_tool_receipt.json')]
    paths += [broad.catalog.DEST/name for name in ('source_index.json', 'feature_manifest.json')]
    paths += [parent.PARENT/name for name in ('model.pt', 'result.json', 'actual_tool_receipt.json')]
    paths += [Path('pyproject.toml'), Path('pixi.lock'), *extra]
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-public-curriculum-producer.v1', 'hypothesis_id': 'ncm-public-curriculum-v1',
             'stage': stage,
             'config': asdict(curriculum.CONFIG), 'environment': environment,
             'systems_lifecycle_only_not_quality_evidence': stage == 'pilot',
             'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
             'input_hashes': {str(p): data.sha(p) for p in paths}}
    proof['fingerprint'] = training.q.digest(proof)
    data.write(root/'producer.json', proof)
    return proof


def memory(phase, root=ROOT):
    torch.cuda.synchronize()
    row = {'phase': phase, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'allocated_bytes': torch.cuda.memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    with (root/'memory.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    if row['peak_allocated_bytes'] > curriculum.CONFIG.peak_limit_bytes:
        raise ValueError('effective128 whole-lifecycle GPU memory budget exceeded')
    return row


def pipeline_identity(sources, epoch):
    serial, prefetched = [], []
    before = time.monotonic()
    for index, denominator, refs in epoch.batches(stop_update=2):
        rows = sources.fetch(refs)
        serial.append({'update': index, 'denominator': denominator,
                       'positions': [r.position for r in refs], 'digest': engine.batch_digest(rows)})
    serial_seconds = time.monotonic() - before
    before = time.monotonic()
    for index, denominator, refs, rows in curriculum.prefetch(epoch.batches(stop_update=2), sources.fetch):
        prefetched.append({'update': index, 'denominator': denominator,
                           'positions': [r.position for r in refs], 'digest': engine.batch_digest(rows)})
    if serial != prefetched:
        raise ValueError('serial/prefetched complete feature or label bytes differ')
    return {'batches': serial, 'serial_seconds': serial_seconds,
            'prefetched_seconds': time.monotonic() - before, 'all_records_equal': True,
            'warm_cache_order_confounded_not_speed_claim': True}


def run():
    if ROOT.exists() or CHECKPOINTS.exists():
        raise ValueError('fresh append-only mixed pipeline pilot required')
    training, environment = broad.runtime.configure()
    broad.offload.require_rtx2060()
    sources = ready(training)
    torch.cuda.reset_peak_memory_stats()
    model, opt, scaler = model_and_optimizer(training)
    frozen = training.prior.protected(model)
    proof = freeze(training, environment)
    CHECKPOINTS.mkdir(parents=True)
    memory('model_loaded')
    epoch = curriculum.Epoch(1)
    data.write(ROOT/'pipeline_identity.json', pipeline_identity(sources, epoch))
    torch.manual_seed(curriculum.CONFIG.dropout_seed)
    torch.cuda.manual_seed_all(curriculum.CONFIG.dropout_seed)
    batches = curriculum.prefetch(epoch.batches(stop_update=2), sources.fetch,
                                  capacity=curriculum.CONFIG.prefetch_batches)
    records = []
    try:
        for step in (1, 2):
            record = engine.update(model, batches, opt, scaler, step=step)
            broad.offload.assert_cpu_state(opt)
            memory('update_'+str(step)); records.append(record)
            data.write(ROOT/f'update_{step}.json', record)
            engine.save(CHECKPOINTS/f'step_{step}.pt', model, opt, scaler,
                        step=step, fingerprint=proof['fingerprint'])
            if training.prior.protected(model) != frozen:
                raise ValueError('protected tensors changed')
    finally:
        batches.close()
    controls = sources.fetch(epoch.update(0)[:4])
    expected_outputs = lifecycle.outputs(model, [r['features'] for r in controls])
    data.write(ROOT/'outputs_after_step2.json', expected_outputs)
    first = torch.load(CHECKPOINTS/'step_1.pt', map_location='cpu', weights_only=True, mmap=True)
    engine.restore(first, model, opt, scaler, fingerprint=proof['fingerprint'])
    # Replay without the prefetcher: loading order must not perturb training RNG.
    batches = ((i, n, refs, sources.fetch(refs)) for i, n, refs in epoch.batches(1, 2))
    replay = engine.update(model, batches, opt, scaler, step=2)
    data.write(ROOT/'resumed_update_2.json', replay)
    second = torch.load(CHECKPOINTS/'step_2.pt', map_location='cpu', weights_only=True, mmap=True)
    same = training.preparation.checkpoint.training.lifecycle.same
    current = engine.snapshot(model, opt, scaler, step=2, fingerprint=proof['fingerprint'])
    equality = {key: same(value, second[key]) for key, value in current.items()}
    data.write(ROOT/'resumed_state_equality.json', equality)
    if (not all(equality.values()) or replay['microbatches'] != records[1]['microbatches']
            or lifecycle.outputs(model, [r['features'] for r in controls]) != expected_outputs):
        raise ValueError('exact effective128 resumed optimizer state or source-only outputs differ')
    if training.prior.protected(model) != frozen:
        raise ValueError('protected tensors changed during effective128 lifecycle')
    peak = memory('resumed_evaluation')['peak_allocated_bytes']
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    artifacts = [p for p in ROOT.iterdir() if p.is_file()] + list(CHECKPOINTS.glob('*.pt'))
    value = {'status': 'public_mixed_effective128_pipeline_passed', 'config': asdict(curriculum.CONFIG),
             'all_serial_prefetched_records_equal': True, 'strict_resumed_update_passed': True,
             'source_only_outputs_reproduced': True, 'protected_tensors_unchanged': True,
             'optimizer_updates': 2, 'replayed_optimizer_updates': 1,
             'effective_examples_per_update': 128, 'public_presentations': 192, 'anchor_presentations': 64,
             'peak_GPU_allocated_bytes': peak, 'anchor_cache': sources.anchor_cache.metrics(),
             'systems_lifecycle_only_not_quality_evidence': True, 'new_judge_requests': 0,
             'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False,
             'artifacts': {str(p): data.sha(p) for p in artifacts}}
    data.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)
    sources.public.pairs.close(); sources.public.codes.close(); model.close()
    del model, opt, current, first, second
    gc.collect(); torch.cuda.empty_cache()


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
                                           'task_accuracy': None})
        raise
