"""Train complete public epochs, stopping at fixed semantic-evaluation milestones."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch

from experiments.structure_net import ncm_public_pipeline_pilot as pipeline

data, engine, curriculum = pipeline.data, pipeline.engine, pipeline.curriculum
CONFIG = curriculum.CONFIG
STORAGE = pipeline.CHECKPOINTS.parent/'train'


def root_for(epoch, attempt):
    if epoch not in CONFIG.milestones or type(attempt) is not int or attempt < 1:
        raise ValueError('fixed milestone and positive execution attempt required')
    return data.ROOT/f'20261026_public_curriculum_epoch{epoch}_attempt{attempt}_v1'


def require_previous_grade(epoch):
    previous = CONFIG.milestones[:CONFIG.milestones.index(epoch)]
    if not previous:
        return []
    root = data.ROOT/f'20261026_public_curriculum_epoch{previous[-1]}_DEV_grade_v1'
    result = data.completed(root)
    if not valid_failed_grade(result):
        raise ValueError('previous valid failed DEV milestone required; qualified model goes to final protocol')
    return [root/'result.json', root/'actual_tool_receipt.json']


def valid_failed_grade(result):
    cohorts = ('short_development', 'long_diagnostic')
    roles = ('secondary', 'primary') if result.get('primary_admitted') else ('secondary',)
    return (not result.get('DEV_qualified') and result.get('task_accuracy') is not None
            and set(result.get('metrics', {})) == set(cohorts)
            and all(result['metrics'][c].get(role, {}).get('valid') == 64
                    and result['metrics'][c][role].get('cases') == 64 for c in cohorts for role in roles))


def checkpoint(path, model, opt, scaler, step, identity):
    engine.save(path, model, opt, scaler, step=step, fingerprint=identity)
    row = {'path': str(path), 'sha256': data.sha(path), 'completed_updates': step,
           'training_identity': identity}
    data.write(path.with_suffix('.json'), row)
    return row


def resume_record(path, identity, end_epoch):
    path = Path(path)
    if not path.resolve().is_relative_to(STORAGE.resolve()):
        raise ValueError('resume only an immutable checkpoint from this full training study')
    row = data.read(path.with_suffix('.json'))
    if (row['path'] != str(path) or row['sha256'] != data.sha(path)
            or row['training_identity'] != identity
            or type(row['completed_updates']) is not int
            or not 0 < row['completed_updates'] <= end_epoch * CONFIG.updates_per_epoch):
        raise ValueError('exact committed checkpoint bytes and training schedule required')
    previous = CONFIG.milestones[:CONFIG.milestones.index(end_epoch)]
    if previous and row['completed_updates'] < previous[-1] * CONFIG.updates_per_epoch:
        raise ValueError('continuation cannot repeat a previously evaluated training segment')
    return row


def audit_trace(path, sources, start, end, config=CONFIG):
    epoch = None
    counts = {'public': 0, 'anchor': 0}
    cursor = start
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            number, local = divmod(cursor, config.updates_per_epoch)
            if epoch is None or epoch.number != number + 1:
                epoch = curriculum.Epoch(number + 1, config)
            refs = epoch.update(local)
            flat_ids = [v for batch in row['microbatches'] for v in batch['ids']]
            flat_modes = [v for batch in row['microbatches'] for v in batch['noisy']]
            flat_positions = [v for batch in row['microbatches'] for v in batch['positions']]
            flat_populations = [v for batch in row['microbatches'] for v in batch['populations']]
            identities = [(sources.public.pairs.entries if e.population == 'public' else sources.anchors)[e.index]['id']
                          for e in refs]
            if (row['step'] != cursor + 1 or row['epoch'] != number + 1 or row['examples'] != len(refs)
                    or flat_ids != identities or flat_modes != [e.noisy for e in refs]
                    or flat_positions != [e.position for e in refs]
                    or flat_populations != [e.population for e in refs]
                    or row['lr'] != curriculum.rate(cursor + 1, config)
                    or not math.isfinite(row['weighted_CE']) or not math.isfinite(row['gradient_norm'])):
                raise ValueError('recorded updates differ from complete fixed curriculum')
            for ref in refs:
                counts[ref.population] += 1
            cursor += 1
    if cursor != end:
        raise ValueError('missing or additional recorded optimizer updates')
    return {'first_completed_update': start, 'last_completed_update': end,
            'segment_updates': end-start, 'presentations': counts, 'exact_sample_and_mode_order': True,
            'trace_sha256': data.sha(path)}


def run(end_epoch, attempt, resume=None):
    root = root_for(end_epoch, attempt)
    storage = STORAGE/f'epoch{end_epoch}_attempt{attempt}'
    if root.exists() or storage.exists():
        raise ValueError('fresh append-only attempt required; resume into a new attempt')
    if end_epoch != 1 and resume is None:
        raise ValueError('later milestones continue the completed earlier checkpoint')
    prerequisite_paths = require_previous_grade(end_epoch)
    pilot = data.completed(pipeline.ROOT)
    qualification = data.read(pipeline.ROOT/'producer.json')
    data.verify(qualification['source_hashes']); data.verify(qualification['input_hashes'])
    if (pilot['status'] != 'public_mixed_effective128_pipeline_passed'
            or not pilot['strict_resumed_update_passed'] or not pilot['all_serial_prefetched_records_equal']
            or not pilot['protected_tensors_unchanged']):
        raise ValueError('real mixed-data effective128 lifecycle must pass before training')
    training, environment = pipeline.broad.runtime.configure()
    if training.q.digest(pilot['config']) != training.q.digest(asdict(CONFIG)):
        raise ValueError('same qualified effective128 curriculum required')
    pipeline.broad.offload.require_rtx2060()
    sources = pipeline.ready(training)
    torch.cuda.reset_peak_memory_stats()
    model, opt, scaler = pipeline.model_and_optimizer(training)
    frozen = training.prior.protected(model)
    extra = [pipeline.ROOT/name for name in ('result.json', 'actual_tool_receipt.json')]
    proof = pipeline.freeze(training, environment, root=root, extra=extra, stage='train')
    # Every segment/recovery uses the same data, source code and eight-epoch LR.
    identity = proof['fingerprint']
    storage.mkdir(parents=True)
    completed = 0
    torch.manual_seed(CONFIG.dropout_seed); torch.cuda.manual_seed_all(CONFIG.dropout_seed)
    if resume is not None:
        record = resume_record(resume, identity, end_epoch)
        saved = torch.load(resume, map_location='cpu', weights_only=True, mmap=True)
        completed = engine.restore(saved, model, opt, scaler, fingerprint=identity)
        if completed != record['completed_updates']:
            raise ValueError('checkpoint cursor differs from committed sidecar')
        data.write(root/'resume.json', record)
        del saved
    data.write(root/'segment.json', {'end_epoch': end_epoch, 'attempt': attempt,
        'first_completed_update': completed, 'target_completed_update': end_epoch * CONFIG.updates_per_epoch,
        'prerequisite_hashes': {str(p): data.sha(p) for p in prerequisite_paths},
        'training_identity': identity})
    first_completed = completed
    pipeline.memory('model_loaded', root)
    started = time.monotonic()
    last_checkpoint = None
    with (root/'trace.jsonl').open('x') as trace:
        while completed < end_epoch * CONFIG.updates_per_epoch:
            epoch_index, local_start = divmod(completed, CONFIG.updates_per_epoch)
            epoch = curriculum.Epoch(epoch_index + 1)
            batches = curriculum.prefetch(epoch.batches(local_start), sources.fetch,
                                          capacity=CONFIG.prefetch_batches)
            try:
                for _ in range(local_start, CONFIG.updates_per_epoch):
                    row = engine.update(model, batches, opt, scaler, step=completed + 1)
                    pipeline.broad.offload.assert_cpu_state(opt)
                    completed += 1
                    row['elapsed_seconds'] = time.monotonic() - started
                    row['peak_allocated_bytes'] = pipeline.memory('update_'+str(completed), root)['peak_allocated_bytes']
                    trace.write(json.dumps(row, allow_nan=False)+'\n'); trace.flush()
                    if completed % CONFIG.recovery_interval == 0 or completed % CONFIG.updates_per_epoch == 0:
                        trace.flush(); os.fsync(trace.fileno())
                        if training.prior.protected(model) != frozen:
                            raise ValueError('protected codec or word path changed')
                        last_checkpoint = checkpoint(storage/f'step_{completed:06d}.pt', model, opt, scaler,
                                                     completed, identity)
                    if completed <= 2 or completed % 16 == 0:
                        print(json.dumps({k: v for k, v in row.items() if k != 'microbatches'}), flush=True)
            finally:
                batches.close()
    if last_checkpoint is None:
        # Recovery may start after the last optimizer update but before result closure.
        last_checkpoint = checkpoint(storage/f'step_{completed:06d}.pt', model, opt, scaler, completed, identity)
    audit = audit_trace(root/'trace.jsonl', sources, first_completed, completed)
    data.write(root/'trace_audit.json', audit)
    final_hash = training.native.vendor.module_hash(model)
    if final_hash == pipeline.parent.MODEL_SHA or training.prior.protected(model) != frozen:
        raise ValueError('generator must change and protected paths must remain exact')
    selected = Path(last_checkpoint['path'])
    saved = torch.load(selected, map_location='cpu', weights_only=True, mmap=True)
    expected = engine.snapshot(model, opt, scaler, step=completed, fingerprint=identity)
    same = training.preparation.checkpoint.training.lifecycle.same
    if not all(same(expected[k], saved[k]) for k in expected):
        raise ValueError('saved model/optimizer/scaler/cursor/RNG state differs')
    engine.restore(saved, model, opt, scaler, fingerprint=identity)
    if training.native.vendor.module_hash(model) != final_hash:
        raise ValueError('strict final checkpoint reload differs')
    del saved, expected
    # Reuse the established source-only inference loader without copying a large checkpoint.
    (root/'model.pt').symlink_to(selected.resolve())
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    peak = pipeline.memory('final_reload', root)['peak_allocated_bytes']
    artifacts = [p for p in root.iterdir() if p.is_file()]
    if resume is not None:
        artifacts += [Path(resume), Path(resume).with_suffix('.json')]
    value = {'schema': 'nal.ncm-public-training.v1', 'hypothesis_id': 'ncm-public-curriculum-v1',
        'status': 'public_curriculum_milestone_training_completed', 'epoch': end_epoch,
        'updates': completed, 'public_cases': CONFIG.public_cases, 'reviewed_cases': CONFIG.anchor_cases,
        'public_presentations': end_epoch * CONFIG.public_cases,
        'anchor_presentations': end_epoch * CONFIG.anchors_per_epoch, 'trainable_parameters': 105323136,
        'strict_reload_passed': True, 'oracle_unchanged': True, 'frozen_hashes': frozen,
        'checkpoint_sha256': last_checkpoint['sha256'], 'final_model_sha256': final_hash,
        'training_identity': identity, 'segment_seconds': time.monotonic() - started,
        'segment_audit': audit,
        'peak_allocated_bytes': peak, 'config': asdict(CONFIG), 'task_accuracy': None,
        'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(p): data.sha(p) for p in artifacts}}
    data.write(root/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)
    sources.public.pairs.close(); sources.public.codes.close(); model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--end-epoch', type=int, choices=CONFIG.milestones, default=1)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root = root_for(args.end_epoch, args.attempt)
    existed = root.exists()
    try:
        run(args.end_epoch, args.attempt, args.resume)
    except Exception as error:
        if not existed and root.exists():
            data.write(root/'failure.json', {'exception': type(error).__name__, 'message': str(error),
                                            'task_accuracy': None})
        raise
