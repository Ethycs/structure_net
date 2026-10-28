"""Train the full recovered conversational corpus with source-balanced epochs."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch

from experiments.structure_net import ncm_recovered_training_inputs as inputs
from experiments.structure_net import ncm_recovered_training_schedule as schedule
from experiments.structure_net import ncm_task_supervision_training as task

CONFIG = schedule.CONFIG
data, pipeline, engine, broad = task.data, task.pipeline, task.engine, task.broad
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-paraphrase-training-preregistration.md')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_recovered_paraphrase_training.py')
TESTS = [Path('tests/structure_net')/n for n in ('test_ncm_recovered_training_schedule.py',
    'test_ncm_recovered_training_inputs.py', 'test_ncm_recovered_training_stream.py',
    'test_ncm_recovered_paraphrase_training_audit.py')]
ROOTS = {s: inputs.BASE/f'20261028_recovered_paraphrase_training_{s}_v1'
         for s in ('prepare', 'pilot', 'pilot_audit', 'train', 'audit')}
STORAGE = Path('/data2/structure_net/ncm_recovered_paraphrase_20261028_v1/training')


def require(proofs, stage):
    value = proofs.completed(ROOTS[stage])
    proof = proofs.read(ROOTS[stage]/'producer.json')
    proofs.verify(proof['source_hashes']); proofs.verify(proof['input_hashes'])
    if proof['config'] != asdict(CONFIG): raise ValueError('exact fixed training configuration required')
    return value


def freeze(root, stage, proofs, training, environment):
    files = {Path(__file__).resolve(), PLAN.resolve(), AUDITOR.resolve(), *[p.resolve() for p in TESTS]}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())): files.add(p)
    for p in (Path('pixi.lock'), Path('pyproject.toml')): proofs.bind(p)
    proof = {'schema': schedule.SCHEMA, 'hypothesis_id': schedule.HYPOTHESIS, 'stage': stage,
        'config': asdict(CONFIG), 'environment': environment,
        'parent_checkpoint_sha256': task.schedule.PARENT_CHECKPOINT_SHA,
        'parent_model_sha256': task.PARENT_MODEL_SHA, 'trainable_parameters': 105323136,
        'systems_lifecycle_only_not_quality_evidence': stage in ('pilot', 'pilot_audit'),
        'source_hashes': {str(p): proofs.sha(p) for p in sorted(files)},
        'input_hashes': dict(proofs.inputs)}
    proof['fingerprint'] = task.base.fingerprint(proof)
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)): (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json', proof)
    return proof


def memory(root, phase):
    torch.cuda.synchronize()
    row = {'phase': phase, 'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(),
        'GPU_allocated_bytes': torch.cuda.memory_allocated(),
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    with (root/'memory.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    if row['peak_GPU_allocated_bytes'] > CONFIG.peak_limit_bytes or row['peak_RSS_bytes'] > CONFIG.host_limit_bytes:
        raise ValueError('whole-lifecycle GPU or host-memory ceiling exceeded')
    return row


def controls_for(sources, selection):
    refs = [schedule.balancing.shared.Example(r['population'], r['index'], 1, i, False, 273142+i)
            for i, r in enumerate(selection)]
    rows = sources.fetch(refs)
    if [r['id'] for r in rows] != [r['id'] for r in selection]: raise ValueError('control selection binding differs')
    return [r['features'] for r in rows], [r['oracle_codes'] for r in rows]


def stream_parity(sources, groups):
    direct = [(i, n, [asdict(e) for e in refs], engine.batch_digest(sources.fetch(refs)))
              for i, n, refs in schedule.batches(groups, stop_update=2)]
    prefetched = [(i, n, [asdict(e) for e in refs], engine.batch_digest(rows)) for i, n, refs, rows in
        schedule.balancing.shared.prefetch(schedule.batches(groups, stop_update=2), sources.fetch,
                                          capacity=CONFIG.prefetch_batches)]
    if direct != prefetched: raise ValueError('real full-population serial and prefetched batches differ')
    return {'all_records_equal': True, 'microbatches': direct}


def prepare():
    root = ROOTS['prepare']
    if root.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('fresh CPU-only preparation required')
    training, environment = broad.runtime.configure()
    proofs = inputs.Proofs(); sources, groups, controls = inputs.ready(proofs)
    proof = freeze(root, 'prepare', proofs, training, environment)
    data.write(root/'controls.json', controls)
    data.write(root/'bindings.json', {'recovered_pairs': str(inputs.qualified.CATALOG/'pairs.jsonl'),
        'recovered_features': str(inputs.FEATURES), 'singleton_codes': str(inputs.qualified.CODES),
        'reviewed_anchors': str(inputs.BINDINGS), 'source_groups': str(schedule.balancing.ROOT/'groups.json'),
        'model_initializer': str(task.PARENT/'model.pt'), 'training_plan': str(PLAN)})
    proofs.verify(proof['source_hashes']); proofs.verify(proof['input_hashes'])
    value = {'schema': schedule.SCHEMA, 'hypothesis_id': schedule.HYPOTHESIS,
        'status': 'full_recovered_training_inputs_admitted', 'recovered_pairs': len(sources.recovered),
        'recovered_source_groups': len(groups), 'conditioned_source_features': len(sources.recovered.features),
        'reviewed_anchors': len(sources.anchors), 'all_target_codes_bound': True, 'all_source_context_joins_bound': True,
        'data_admitted': True, 'model_training': False, 'full_training_admitted': False,
        'task_accuracy': None, 'final_test_opened': False, 'new_judge_calls': 0,
        'artifacts': {str(p): proofs.sha(p) for p in root.iterdir() if p.is_file()}}
    data.write(root/'result.json', value); sources.recovered.pairs.close()
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


def optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params) != 105323136: raise ValueError('fixed complete trainable population required')
    opt = broad.offload.CPUAdamW(params, lr=CONFIG.max_lr, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    if any(not torch.equal(p.detach().cpu(), master.detach()) for p, master in zip(params, opt.masters, strict=True)):
        raise ValueError('fresh optimizer masters must match actual loaded weights')
    return opt, torch.amp.GradScaler('cuda', init_scale=128., growth_interval=2000)


def run(stage):
    root, storage = ROOTS[stage], STORAGE/stage
    if stage not in ('pilot', 'train') or root.exists() or storage.exists(): raise ValueError('fresh fitting or lifecycle root required')
    training, environment = broad.runtime.configure(); pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER'
            or torch.cuda.mem_get_info()[0] < CONFIG.peak_limit_bytes):
        raise ValueError('physical GPU2 RTX2060 SUPER with full lifecycle headroom required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    proofs = inputs.Proofs()
    if not require(proofs, 'prepare')['data_admitted']: raise ValueError('complete data admission required')
    if stage == 'train':
        pilot, audit = require(proofs, 'pilot'), require(proofs, 'pilot_audit')
        if (not all(pilot[k] for k in ('strict_resumed_update_passed', 'protected_tensors_unchanged',
                'gradient_parity_passed', 'source_only_outputs_reproduced'))
                or audit['status'] != 'recovered_training_pilot_replayed'):
            raise ValueError('full-population lifecycle and independent replay required')
    sources, groups, selection = inputs.ready(proofs)
    if selection != proofs.read(ROOTS['prepare']/'controls.json'): raise ValueError('complete-population controls changed')
    proof = freeze(root, stage, proofs, training, environment); storage.mkdir(parents=True)
    data.write(root/'controls.json', selection)
    torch.cuda.reset_peak_memory_stats(); model = task.load_initial(training)
    frozen, initial_hash = training.prior.protected(model), training.native.vendor.module_hash(model)
    values, codes = controls_for(sources, selection)
    initial_outputs = pipeline.lifecycle.outputs(model, values)
    if initial_outputs != pipeline.lifecycle.outputs(model, values): raise ValueError('source-only control outputs are not repeatable')
    data.write(root/'initial_outputs.json', initial_outputs)
    if stage == 'pilot':
        data.write(root/'stream_parity.json', stream_parity(sources, groups))
        previous = pipeline.lifecycle.ROOT; pipeline.lifecycle.ROOT = root
        try:
            for noisy in (False, True): pipeline.lifecycle.parity(model, values, codes, noisy)
        finally: pipeline.lifecycle.ROOT = previous
    if training.native.vendor.module_hash(model) != initial_hash: raise ValueError('pre-update checks altered initializer')
    broad.mechanism.configure(model, True); opt, scaler = optimizer(model)
    memory(root, 'optimizer_loaded')
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    updates = 2 if stage == 'pilot' else CONFIG.updates
    checkpoints = []; started = time.monotonic()
    batches = schedule.balancing.shared.prefetch(schedule.batches(groups, stop_update=updates), sources.fetch,
                                                capacity=CONFIG.prefetch_batches)
    try:
        with (root/'trace.jsonl').open('x') as trace:
            for step in range(1, updates+1):
                row = engine.update(model, batches, opt, scaler, step=step, config=CONFIG)
                broad.offload.assert_cpu_state(opt)
                row.update(elapsed_seconds=time.monotonic()-started, **memory(root, 'step_'+str(step)))
                trace.write(json.dumps(row, allow_nan=False)+'\n'); trace.flush()
                if stage == 'pilot' or step % CONFIG.recovery_interval == 0 or step % CONFIG.updates_per_epoch == 0:
                    if training.prior.protected(model) != frozen: raise ValueError('protected tensors changed')
                    path = storage/f'step{step}.pt'
                    engine.save(path, model, opt, scaler, step=step, fingerprint=proof['fingerprint'], config=CONFIG)
                    checkpoint = {'path': str(path), 'sha256': proofs.sha(path), 'completed_updates': step,
                                  'producer_fingerprint': proof['fingerprint']}
                    data.write(path.with_suffix('.json'), checkpoint); checkpoints.append(checkpoint)
                if stage == 'pilot' or step % 16 == 0:
                    print(json.dumps({k: v for k, v in row.items() if k != 'microbatches'}), flush=True)
    finally: batches.close()
    final_outputs = pipeline.lifecycle.outputs(model, values); data.write(root/'final_outputs.json', final_outputs)
    if stage == 'pilot':
        first = torch.load(storage/'step1.pt', map_location='cpu', weights_only=True, mmap=True)
        engine.restore(first, model, opt, scaler, fingerprint=proof['fingerprint'], config=CONFIG)
        replay_batches = ((i, n, refs, sources.fetch(refs)) for i, n, refs in schedule.batches(groups, 1, 2))
        replay = engine.update(model, replay_batches, opt, scaler, step=2, config=CONFIG)
        data.write(root/'resumed_update_2.json', replay)
        if replay['microbatches'] != row['microbatches'] or pipeline.lifecycle.outputs(model, values) != final_outputs:
            raise ValueError('resumed update inputs, gradients or source-only outputs differ')
        del first
    final = Path(checkpoints[-1]['path'])
    saved = torch.load(final, map_location='cpu', weights_only=True, mmap=True)
    snapshot = engine.snapshot(model, opt, scaler, step=updates, fingerprint=proof['fingerprint'], config=CONFIG)
    same = training.preparation.checkpoint.training.lifecycle.same
    equality = {k: same(value, saved[k]) for k, value in snapshot.items()}
    data.write(root/'final_state_equality.json', equality)
    if not all(equality.values()): raise ValueError('model/optimizer/scaler/RNG/checkpoint equality failed')
    del snapshot
    final_hash = training.native.vendor.module_hash(model)
    if engine.restore(saved, model, opt, scaler, fingerprint=proof['fingerprint'], config=CONFIG) != updates:
        raise ValueError('strict final cursor differs')
    if (training.native.vendor.module_hash(model) != final_hash or training.prior.protected(model) != frozen
            or final_hash == initial_hash): raise ValueError('trained or protected state differs after strict reload')
    peak = memory(root, 'final_verified')
    data.write(root/'checkpoints.json', checkpoints); (root/'model.pt').symlink_to(final.resolve())
    proofs.verify(proof['source_hashes']); proofs.verify(proof['input_hashes'])
    artifacts = [p for p in root.iterdir() if p.is_file()]
    artifacts += [p for row in checkpoints for p in (Path(row['path']), Path(row['path']).with_suffix('.json'))]
    result = {'schema': schedule.SCHEMA, 'hypothesis_id': schedule.HYPOTHESIS, 'stage': stage,
        'status': 'recovered_paraphrase_lifecycle_passed' if stage == 'pilot' else 'recovered_paraphrase_training_completed',
        'updates': updates, 'recovered_pairs': len(sources.recovered), 'source_groups': len(groups),
        'reviewed_anchors': len(sources.anchors), 'config': asdict(CONFIG), 'trainable_parameters': 105323136,
        'gradient_parity_passed': stage == 'pilot', 'strict_resumed_update_passed': stage == 'pilot',
        'source_only_outputs_reproduced': stage == 'pilot', 'strict_final_reload_passed': True,
        'fresh_optimizer_masters_match_initial_weights': True,
        'protected_tensors_unchanged': True, 'frozen_hashes': frozen, 'final_model_sha256': final_hash,
        'checkpoint_sha256': proofs.sha(final), 'elapsed_seconds': time.monotonic()-started, **peak,
        'anchor_cache': sources.anchor_cache.metrics(), 'task_accuracy': None, 'final_test_opened': False,
        'new_judge_calls': 0, 'speed_tested': False, 'systems_lifecycle_only_not_quality_evidence': stage == 'pilot',
        'artifacts': {str(p): proofs.sha(p) for p in artifacts}}
    data.write(root/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)
    sources.recovered.pairs.close(); model.close(); del saved, model, opt, scaler
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=('prepare', 'pilot', 'train'))
    args = parser.parse_args(); root = ROOTS[args.stage]; existed = root.exists()
    try: prepare() if args.stage == 'prepare' else run(args.stage)
    except Exception as error:
        if root.exists() and not existed:
            data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error), 'task_accuracy': None})
        raise
