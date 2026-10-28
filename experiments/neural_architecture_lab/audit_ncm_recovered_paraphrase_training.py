"""Replay recovered training exposure, code noise and actual saved model state."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_recovered_paraphrase_training as study

LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
STATE_KEYS = {'model', 'optimizer', 'scaler', 'CPU_rng', 'CUDA_rng',
              'completed_updates', 'producer_fingerprint', 'curriculum'}


def replay(records, groups, resolve, start, end, config=study.CONFIG):
    if not 0 <= start < end <= config.updates: raise ValueError('nonempty committed update interval required')
    records = iter(records); epoch = None; previous_scale = None
    counts, sources, targets, anchors = Counter(), Counter(), Counter(), Counter()
    digest = hashlib.sha256(); positions = 0
    for step in range(start+1, end+1):
        try: value = next(records)
        except StopIteration as error: raise ValueError('missing completed update') from error
        number, local = divmod(step-1, config.updates_per_epoch)
        if epoch is None or epoch.number != number+1: epoch = study.schedule.balancing.Epoch(number+1, groups, config)
        refs = epoch.update(local)
        if (value['step'] != step or value['epoch'] != number+1 or value['examples'] != len(refs)
                or value['lr'] != study.schedule.balancing.shared.rate(step, config)
                or len(value['microbatches']) != math.ceil(len(refs)/config.microbatch)
                or any(not math.isfinite(value[k]) or value[k] < 0 for k in ('weighted_CE', 'gradient_norm'))
                or not math.isfinite(value['scale']) or value['scale'] <= 0
                or previous_scale is not None and value['scale'] < previous_scale):
            raise ValueError('update geometry, epoch, denominator or optimizer statistics differ')
        previous_scale = value['scale']
        for offset, micro in zip(range(0, len(refs), config.microbatch), value['microbatches'], strict=True):
            selected = refs[offset:offset+config.microbatch]; rows = [resolve(e) for e in selected]
            if any(len(r['oracle_codes']) != 641 or len(r['native_word_ids']) != 128
                   or any(type(c) is not int or not 0 <= c < 8192 for c in r['oracle_codes']) for r in rows):
                raise ValueError('complete target hierarchies required')
            _, changes = study.engine.histories(torch.tensor([r['oracle_codes'] for r in rows]), selected, LEVELS)
            expected = {'ids': [r['id'] for r in rows], 'positions': [e.position for e in selected],
                'populations': [e.population for e in selected], 'noisy': [e.noisy for e in selected],
                'history_changes': changes}
            if (any(micro[k] != v for k, v in expected.items()) or not math.isfinite(micro['weighted_CE'])
                    or micro['weighted_CE'] < 0): raise ValueError('source/target pairing, noise or microbatch loss differs')
            for e, row in zip(selected, rows, strict=True):
                counts[e.population] += 1; counts['noisy'] += e.noisy
                positions += row['source_positions']
                if e.population == 'recovered':
                    sources[row['source_sha256']] += 1; targets[row['id']] += 1
                else: anchors[row['id']] += 1
        if value['weighted_CE'] != sum(m['weighted_CE'] for m in value['microbatches']):
            raise ValueError('effective loss sum differs')
        digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
    if next(records, None) is not None: raise ValueError('extra unaccounted update')
    presentations = counts['recovered']+counts['anchor']
    return {'updates': end-start, 'presentations': presentations, 'counts': dict(counts),
        'distinct_recovered_sources': len(sources), 'distinct_recovered_targets': len(targets),
        'distinct_reviewed_anchors': len(anchors), 'source_visit_histogram': dict(Counter(sources.values())),
        'target_visit_histogram': dict(Counter(targets.values())), 'anchor_visit_histogram': dict(Counter(anchors.values())),
        'source_feature_positions_presented': positions, 'native_target_positions_presented': presentations*128,
        'target_code_positions_presented': presentations*641, 'all_sample_and_corruption_counts_replayed': True,
        'audited_records_sha256': digest.hexdigest()}


def state_hash(state, prefix=''):
    h = hashlib.sha256(); count = 0
    for name, value in sorted(state.items()):
        if not name.startswith(prefix): continue
        h.update(name[len(prefix):].encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()); count += 1
    if not count: raise ValueError('nonempty exact protected state required')
    return h.hexdigest()


def audit(stage):
    name = 'pilot_audit' if stage == 'pilot' else 'audit'
    root, source = study.ROOTS[name], study.ROOTS[stage]
    if root.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('fresh CPU-only audit required')
    proofs = study.inputs.Proofs(); result = study.require(proofs, stage)
    original_proof = proofs.read(source/'producer.json')
    limit = 2 if stage == 'pilot' else study.CONFIG.updates
    expected = 'recovered_paraphrase_lifecycle_passed' if stage == 'pilot' else 'recovered_paraphrase_training_completed'
    if (result['status'] != expected or result['updates'] != limit or result['stage'] != stage
            or result['config'] != asdict(study.CONFIG) or not result['strict_final_reload_passed']
            or not result['fresh_optimizer_masters_match_initial_weights']
            or not result['protected_tensors_unchanged'] or result['task_accuracy'] is not None
            or result['final_test_opened'] or result['new_judge_calls'] != 0 or result['speed_tested']):
        raise ValueError('complete fixed training stage required')
    sources, groups, controls = study.inputs.ready(proofs)
    if controls != proofs.read(source/'controls.json'): raise ValueError('actual complete-population controls differ')
    training, environment = study.broad.runtime.configure()
    proof = study.freeze(root, name, proofs, training, environment)
    def resolve(e):
        if e.population == 'recovered':
            row = sources.recovered.pairs[e.index]
            codes = sources.recovered.codes.get(row)
            positions = sources.recovered.metadata[row['source_record_id']]['source_positions']
        else:
            row = sources.anchors[e.index]; codes = row['oracle_codes']
            positions = sources.anchor_cache.index[row['id']]['source_positions']
        return {**row, 'oracle_codes': codes, 'source_positions': positions}
    with (source/'trace.jsonl').open() as f: observed = replay(map(json.loads, f), groups, resolve, 0, limit)
    if stage == 'train':
        prior = proofs.read(study.schedule.balancing.ROOT/'result.json')
        if (observed['presentations'] != prior['presentations'] or observed['counts'] != prior['counts']
                or observed['distinct_recovered_targets'] != 43894 or observed['distinct_recovered_sources'] != 38484
                or observed['source_visit_histogram'] != {4: 38484} or observed['distinct_reviewed_anchors'] != 2384):
            raise ValueError('full qualified source/target/anchor exposure not delivered')
    steps = [1, 2] if stage == 'pilot' else sorted(set(range(study.CONFIG.recovery_interval, limit+1, study.CONFIG.recovery_interval))
        | set(range(study.CONFIG.updates_per_epoch, limit+1, study.CONFIG.updates_per_epoch)))
    checkpoints = proofs.read(source/'checkpoints.json')
    if [r['completed_updates'] for r in checkpoints] != steps: raise ValueError('complete recovery checkpoint cadence required')
    for record in checkpoints:
        path = Path(record['path']); proofs.bind(path, record['sha256'])
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if (set(saved) != STATE_KEYS or saved['completed_updates'] != record['completed_updates']
                or saved['producer_fingerprint'] != original_proof['fingerprint']
                or record['producer_fingerprint'] != original_proof['fingerprint']
                or saved['curriculum'] != asdict(study.CONFIG) or proofs.read(path.with_suffix('.json')) != record):
            raise ValueError('checkpoint state/cursor/producer binding differs')
        if record['completed_updates'] == limit:
            if (record['sha256'] != result['checkpoint_sha256'] or proofs.sha(source/'model.pt') != record['sha256']
                    or state_hash(saved['model']) != result['final_model_sha256']):
                raise ValueError('actual final model tensors differ')
            frozen = {name: state_hash(saved['model'], prefix) for name, prefix in
                (('codec', 'concepts.codec.'), ('word_memory', 'word_memory.'), ('word_adapters', 'word_adapters.'))}
            if frozen != result['frozen_hashes'] or frozen != proofs.read(study.task.PARENT/'result.json')['frozen_hashes']:
                raise ValueError('actual saved protected tensors changed')
        del saved
    equality = proofs.read(source/'final_state_equality.json')
    if set(equality) != STATE_KEYS or not all(v is True for v in equality.values()):
        raise ValueError('complete exact saved-state comparison required')
    memory = [json.loads(s) for s in (source/'memory.jsonl').read_text().splitlines()]
    gpu = max(r.get('peak_GPU_allocated_bytes', r.get('peak_allocated_bytes', 0)) for r in memory)
    host = max(r.get('peak_RSS_bytes', 0) for r in memory)
    if (gpu != result['peak_GPU_allocated_bytes'] or host != result['peak_RSS_bytes']
            or gpu > study.CONFIG.peak_limit_bytes or host > study.CONFIG.host_limit_bytes):
        raise ValueError('recorded whole-lifecycle memory bound differs')
    if stage == 'pilot':
        for mode in ('clean', 'noisy'):
            parity = proofs.read(source/f'parity_{mode}.json')
            if not (parity['passed'] and parity['gradient_presence_equal'] and parity['finite'] and parity['logits_allclose']
                    and parity['relative_gradient_L2_error'] <= 5e-5 and parity['max_abs_gradient_difference'] <= 1e-4
                    and parity['argmax_disagreements'] == 0 and parity['trainable_parameters_compared'] == 105323136):
                raise ValueError('actual original-reference FP32 gradient parity failed')
        restarted = proofs.read(source/'resumed_update_2.json'); replay([restarted], groups, resolve, 1, 2)
        original = json.loads((source/'trace.jsonl').read_text().splitlines()[-1])
        if restarted['microbatches'] != original['microbatches']: raise ValueError('actual resumed update differs')
        direct = [(i, n, [asdict(e) for e in refs], study.engine.batch_digest(sources.fetch(refs)))
                  for i, n, refs in study.schedule.batches(groups, stop_update=2)]
        recorded = proofs.read(source/'stream_parity.json')
        if not recorded['all_records_equal'] or json.loads(json.dumps(direct)) != recorded['microbatches']:
            raise ValueError('independent streamed input replay differs')
    study.data.write(root/'replay.json', observed)
    proofs.verify(proof['source_hashes']); proofs.verify(proof['input_hashes'])
    value = {'schema': study.schedule.SCHEMA, 'hypothesis_id': study.schedule.HYPOTHESIS,
        'status': 'recovered_training_pilot_replayed' if stage == 'pilot' else 'recovered_full_training_replayed',
        'stage': stage, 'updates': limit, 'replay': observed, 'actual_saved_protected_tensors_verified': True,
        'source_checkpoint_sha256': result['checkpoint_sha256'], 'task_accuracy': None,
        'systems_lifecycle_only_not_quality_evidence': stage == 'pilot', 'training_updates_added': 0,
        'new_judge_calls': 0, 'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(p): proofs.sha(p) for p in root.iterdir() if p.is_file()}}
    study.data.write(root/'result.json', value); sources.recovered.pairs.close()
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=('pilot', 'train'))
    args = parser.parse_args(); root = study.ROOTS['pilot_audit' if args.stage == 'pilot' else 'audit']; existed = root.exists()
    try: audit(args.stage)
    except Exception as error:
        if root.exists() and not existed: study.data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
