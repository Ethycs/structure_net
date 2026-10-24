"""CPU replay of revised FP16native655 state, level corruption, sampling and AMP."""
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_level_conditioning_training as run

DEST = run.ROOT / '20261024_level_conditioning_FP16_native655_training_audit_v1'


def main():
    if DEST.exists() or torch.cuda.is_initialized() or run.q.read(run.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0:
        raise ValueError('fresh CPU audit and actual completed whole-tool exit required')
    run.native.vendor.corpus.quality.dependencies()
    run.require_capacity()
    for root in (run.DATA, run.PILOT, run.TRAIN):
        run.native.vendor.terminal(root)
    rows = run.choose(run.q.read(run.training.DATA / 'cases.json'))
    if rows != run.q.read(run.DATA / 'cases.json'):
        raise ValueError('all655 unchanged original TRAIN targets required')
    if run.q.read(run.DATA / 'split_proof.json') != run.split_guard(rows, run.q.read(run.training.DEV)):
        raise ValueError('actual source-disjoint655/64/64 populations differ')
    manifest, selected = run.q.read(run.DATA / 'manifest.json'), run.q.read(run.DATA / 'selection.json')
    codes = {r['id']: r['codes'] for r in run.q.read(run.training.CACHE / 'outputs.json')}
    results = {}
    for root, pilot in ((run.PILOT, True), (run.TRAIN, False)):
        state = run.q.read(root / 'result.json')
        saved = torch.load(root / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        updates, accumulation = (2, 2) if pilot else (1024, 16)
        if (state['updates'] != updates or state['train_cases'] != 655 or not state['strict_reload_passed']
                or state['native16_or_pilot_weights_reused'] or not state['root_queries_updated']
                or state['initial_trainable_sha256'] != run.capacity.INITIAL or state['peak_allocated_bytes'] > 4_900_000_000
                or state['checkpoint_sha256'] != run.lifecycle.file_hash(root / 'model.pt')
                or saved['step'] != updates or saved['config'] != run.asdict(run.Config())
                or saved['producer_fingerprint'] != state['producer_fingerprint']
                or any(int(v['step']) != updates for v in saved['optimizer']['state'].values())
                or saved['scaler'] != state['scaler'] or saved['scaler']['scale'] != 128.
                or saved['scaler']['_growth_tracker'] != updates
                or state['feature_cache']['peak_resident_bytes'] > 512 * 1024**2
                or state['precision'] != 'FP32weights/FP16training/FP32hardinference' or state['physical_GPU'] != 2
                or state['source_disjoint_task_accuracy'] is not None or state['speed_tested']):
            raise ValueError('actual native655 update/checkpoint/optimizer/reset/limits differ')
        sampler = torch.Generator().manual_seed(run.Config().sampler_seed)
        neighbor = torch.Generator().manual_seed(run.Config().neighbor_seed)
        count = 0
        for count, line in enumerate((root / 'trace.jsonl').read_text().splitlines(), 1):
            row = json.loads(line)
            if (row['step'] != count or row['lr'] != (run.Config().max_lr if pilot else run.rate(count))
                    or not math.isfinite(row['gradient_norm']) or len(row['microbatches']) != accumulation
                    or row['scaler']['scale'] != 128. or row['scaler']['_growth_tracker'] != count):
                raise ValueError('exact native655 update/rate/finite trace differs')
            for i, micro in enumerate(row['microbatches']):
                indices = [selected['pilot_indices'][i % 2]] if pilot else torch.randint(655, (1,), generator=sampler).tolist()
                chosen = [rows[k] for k in indices]
                if (micro['indices'] != indices or micro['case_ids'] != [r['id'] for r in chosen]
                        or micro['source_positions'] != max(manifest[r['id']]['source_tokens'] for r in chosen)):
                    raise ValueError('all sampled original source/length bindings differ')
                clean = torch.tensor([codes[r['id']] for r in chosen], dtype=torch.long)
                _, proof = run.objective.corruption.corrupt(clean, tuple(run.native.LEVELS),
                    (8192,) * len(run.native.LEVELS), neighbor)
                losses = micro['losses']
                required = {'clean_concept_loss', 'clean_word_loss', 'noisy_concept_loss', 'noisy_word_loss', 'corruption',
                    'accumulation', 'self_history_used', 'autocast_dtype', 'clean_word_weight', 'noisy_word_weight',
                    'clean_and_noisy_concept_weight'}
                if (losses['corruption'] != proof or losses['accumulation'] != accumulation
                        or losses['self_history_used'] != (i == 0) or losses['autocast_dtype'] != 'torch.float16'
                        or losses['clean_word_weight'] != .7 or losses['noisy_word_weight'] != .3
                        or losses['clean_and_noisy_concept_weight'] != .5):
                    raise ValueError('complete native655 noise stream differs')
                if i == 0:
                    required |= {'self_concept_loss', 'self_word_loss', 'source_only_history_codes', 'word_gradient_into_concepts'}
                    history = losses['source_only_history_codes']
                    if (len(history) != 1 or any(len(h) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in h) for h in history)
                            or losses['word_gradient_into_concepts'] is not True):
                        raise ValueError('complete source-only native655 histories and declared weights required')
                if set(losses) != required or any(not math.isfinite(v) or v < 0 for k, v in losses.items() if k.endswith('_loss')):
                    raise ValueError('complete finite native655 objective trace differs')
        if (count != updates or not torch.equal(sampler.get_state(), saved['sampler_rng'])
                or not torch.equal(neighbor.get_state(), saved['neighbor_rng'])):
            raise ValueError('complete sampler/noise/checkpoint RNG replay differs')
        model = run.lifecycle.mechanism.enable(run.lifecycle.head.from_domain(device='cpu'))
        warm = torch.load(run.lifecycle.WARM / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        model.load_attention_state(warm['attention'])
        if (run.lifecycle.trainable_hash(model) != run.capacity.INITIAL
                or run.attention_hash(model) != state['initial_attention_sha256']):
            raise ValueError('original warm native/attention initializer differs')
        model.load_state_dict(saved['model'], strict=True)
        if (run.lifecycle.trainable_hash(model) != state['final_trainable_sha256']
                or run.attention_hash(model) != state['final_attention_sha256']
                or state['final_attention_sha256'] == state['initial_attention_sha256']
                or run.lifecycle.protected(model) != {'encoder': run.lifecycle.ENCODER, 'VQ': run.lifecycle.VQ}
                or run.native.vendor.module_hash(model.concepts.generator) != state['final_native_hashes']['generator']
                or run.native.vendor.module_hash(model.codec.decoder) != state['final_native_hashes']['word_decoder']):
            raise ValueError('strict native655 checkpoint or protected geometry differs')
        model.close()
        del model
        results['pilot' if pilot else 'full'] = {'updates': updates, 'presentations': updates * accumulation,
            'checkpoint_sha256': state['checkpoint_sha256'], 'peak_allocated_bytes': state['peak_allocated_bytes']}
    if run.q.read(run.PILOT / 'initial_outputs.json') != run.q.read(run.TRAIN / 'initial_outputs.json'):
        raise ValueError('pilot/full initial source-only outputs must match')
    artifacts = {str(Path(__file__)): run.lifecycle.file_hash(Path(__file__))}
    for root in (run.DATA, run.PILOT, run.TRAIN, run.WORKFLOW):
        for path in root.rglob('*'):
            if path.is_file() and 'producer_source' not in path.parts:
                artifacts[str(path)] = run.lifecycle.file_hash(path)
    result = {'status': 'level_conditioning_FP16_native655_actual_trace_state_and_scaler_replayed_DEV_pending', 'results': results,
        'source_disjoint_task_accuracy': None, 'all_final_sources_closed': True, 'speed_tested': False, 'artifacts': artifacts}
    DEST.mkdir()
    run.lifecycle.write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
