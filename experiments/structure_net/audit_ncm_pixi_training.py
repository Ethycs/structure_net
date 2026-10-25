"""Replay the completed Pixi run's raw counts, schedule, state and memory."""
import json
from pathlib import Path

from experiments.structure_net import ncm_pixi as runtime


def main():
    run, environment = runtime.configure()
    root = runtime.RUNS['train']
    destination = runtime.ROOT / '20261025_sequential_full732_pixi_train_audit_v1'
    if destination.exists():
        raise ValueError('Fresh audit root required')
    run.preparation.completed(root)
    result, proof = [run.q.read(root / name) for name in ('result.json', 'producer.json')]
    if (proof['environment']['pixi'] != environment
            or proof['config']['host_feature_cache_bytes'] != 256 * 1024**2
            or result['feature_cache']['peak_resident_bytes'] > 256 * 1024**2):
        raise ValueError('Qualified Pixi runtime and bounded feature cache required')
    rows = run.q.read(run.base.DATA / 'cases.json')
    lookup = {row['id']: row for row in rows}
    outputs = run.q.read(root / 'outputs.json')
    if len(rows) != 732 or [r['id'] for r in outputs] != list(lookup):
        raise ValueError('Complete ordered732 TRAIN diagnostics required')
    totals = dict(words=0, correct_words=0, correct_codes=0, code_positions=0)
    for output in outputs:
        row = lookup[output['id']]
        words, gold = output['raw_word_ids'], row['native_word_ids']
        codes, expected_codes = output['codes'], row['oracle_codes']
        active = sum(b != 0 for b in gold)
        matched = sum(a == b for a, b in zip(words, gold) if b != 0)
        correct_codes = sum(a == b for a, b in zip(codes, expected_codes))
        if (len(words) != len(gold) or len(codes) != len(expected_codes)
                or output['source_id_used'] != output['id']
                or matched != output['active_word_matches'] or active != output['active_words']
                or correct_codes != sum(v['rollout_correct'] for v in output['levels'])):
            raise ValueError('Source binding or raw diagnostic counts differ')
        totals['words'] += active
        totals['correct_words'] += matched
        totals['correct_codes'] += correct_codes
        totals['code_positions'] += len(codes)
    if run.prior.summarize(outputs) != result['final_TRAIN']:
        raise ValueError('Reported TRAIN diagnostic differs')
    trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
    if [r['step'] for r in trace] != list(range(1, 2049)):
        raise ValueError('Fixed2048 update sequence incomplete')
    rng = run.torch.Generator().manual_seed(run.CONFIG['sampler_seed'])
    permutation, exposures = [], {row['id']: 0 for row in rows}
    for record in trace:
        if (record['lr'] != run.rate(record['step']) or len(record['metrics']) != 4
                or record['mean_CE'] != sum(r['mean_CE'] for r in record['metrics']) / 4):
            raise ValueError('Learning rate, accumulation or arithmetic differs')
        for micro, metric in enumerate(record['metrics']):
            if not permutation:
                permutation = run.torch.randperm(len(rows), generator=rng).tolist()
            identity = rows[permutation.pop()]['id']
            if metric['id'] != identity or metric['noisy_history'] != bool(micro % 2):
                raise ValueError('Fixed sampling or clean/noisy schedule differs')
            exposures[identity] += 1
    if permutation != run.q.read(root / 'sampler_remaining.json'):
        raise ValueError('Sampler remainder differs')
    memory = [json.loads(line) for line in (root / 'memory.jsonl').read_text().splitlines()]
    peak = max(r['peak_allocated_bytes'] for r in memory)
    if (peak != result['peak_allocated_bytes'] or peak > run.CONFIG['peak_limit_bytes']
            or memory[-1]['phase'] != 'strict_reload' or not result['allocation_budget_passed']
            or run.file_hash(root / 'model.pt') != result['checkpoint_sha256']
            or proof['fingerprint'] != result['producer_fingerprint']
            or not result['strict_reload_passed'] or not result['gradient_parity_passed']
            or result['task_accuracy'] is not None or result['final_test_opened'] or result['speed_tested']):
        raise ValueError('Lifecycle, checkpoint, memory or quality boundary differs')
    saved = run.torch.load(root / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    if (saved['step'] != 2048 or saved['producer_fingerprint'] != proof['fingerprint']
            or any(float(state['step']) != 2048 for state in saved['optimizer']['state'].values())):
        raise ValueError('Optimizer update counters differ')
    paths = [root / n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json',
        'model.pt', 'outputs.json', 'trace.jsonl', 'memory.jsonl', 'sampler_remaining.json')]
    value = {'status': 'complete_raw_TRAIN_schedule_checkpoint_and_memory_replayed',
        'cases': len(outputs), 'updates': 2048, 'presentations': sum(exposures.values()),
        'exposures_min': min(exposures.values()), 'exposures_max': max(exposures.values()),
        'diagnostic_totals': totals, 'peak_allocated_bytes': peak,
        'first16_mean_CE': sum(r['mean_CE'] for r in trace[:16]) / 16,
        'last16_mean_CE': sum(r['mean_CE'] for r in trace[-16:]) / 16,
        'artifacts': {str(p): run.file_hash(p) for p in paths},
        'auditor_sha256': run.file_hash(Path(__file__)), 'task_accuracy': None,
        'final_test_opened': False, 'speed_tested': False}
    destination.mkdir()
    run.write_json(destination / 'campaign_results.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
