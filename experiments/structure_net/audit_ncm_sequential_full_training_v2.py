"""Recompute full732 TRAIN diagnostics, exposure counts and memory from evidence."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_sequential_full_training_v2 as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def audit(stage):
    root = run.RUNS[stage]
    run.preparation.completed(root)
    result = run.q.read(root / 'result.json')
    proof = run.q.read(root / 'producer.json')
    rows = run.q.read(run.base.DATA / 'cases.json')
    lookup = {r['id']:r for r in rows}
    controls = run.q.read(run.base.DATA / 'selection.json')['diagnostic_ids']
    expected = list(lookup) if stage == 'train' else controls
    outputs = run.q.read(root / 'outputs.json')
    if [r['id'] for r in outputs] != expected:
        raise ValueError('complete ordered TRAIN diagnostic required')
    totals = dict(words=0, correct_words=0, correct_codes=0, code_positions=0)
    for output in outputs:
        row = lookup[output['id']]
        words, gold = output['raw_word_ids'], row['native_word_ids']
        codes, code_gold = output['codes'], row['oracle_codes']
        active = sum(b != 0 for b in gold)
        matched = sum(a == b for a,b in zip(words,gold) if b != 0)
        if (len(words) != len(gold) or len(codes) != len(code_gold)
                or output['source_id_used'] != output['id']
                or matched != output['active_word_matches'] or active != output['active_words']):
            raise ValueError('source binding or raw word counts differ')
        correct_codes = sum(a == b for a,b in zip(codes,code_gold))
        if correct_codes != sum(level['rollout_correct'] for level in output['levels']):
            raise ValueError('raw code counts differ')
        totals['words'] += active; totals['correct_words'] += matched
        totals['correct_codes'] += correct_codes; totals['code_positions'] += len(codes)
    if run.prior.summarize(outputs) != result['final_TRAIN']:
        raise ValueError('reported diagnostic differs')
    trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()]
    steps = 2 if stage == 'pilot' else 2048
    if [r['step'] for r in trace] != list(range(1,steps+1)):
        raise ValueError('fixed update sequence incomplete')
    rng = torch.Generator().manual_seed(run.CONFIG['sampler_seed'])
    permutation = []; exposures = {r['id']:0 for r in rows}
    longest = max(run.q.read(run.base.DATA / 'source_index.json'), key=lambda r:r['source_positions'])['id']
    for record in trace:
        if record['lr'] != run.rate(record['step']) or len(record['metrics']) != 4:
            raise ValueError('learning rate or accumulation changed')
        if record['mean_CE'] != sum(r['mean_CE'] for r in record['metrics'])/4:
            raise ValueError('mean cross entropy differs')
        for micro, metric in enumerate(record['metrics']):
            if not permutation:permutation = torch.randperm(len(rows),generator=rng).tolist()
            identity = [rows[0]['id'],longest,rows[0]['id'],longest][micro] if stage == 'pilot' else rows[permutation.pop()]['id']
            if metric['id'] != identity or metric['noisy_history'] != bool(micro%2):
                raise ValueError('fixed sample or corruption schedule differs')
            exposures[identity] += 1
    if permutation != run.q.read(root / 'sampler_remaining.json'):
        raise ValueError('sampler remainder differs')
    memory = [json.loads(line) for line in (root / 'memory.jsonl').read_text().splitlines()]
    peak = max(r['peak_allocated_bytes'] for r in memory)
    if (peak != result['peak_allocated_bytes'] or peak > run.CONFIG['peak_limit_bytes']
            or memory[-1]['phase'] != 'strict_reload' or not result['allocation_budget_passed']):
        raise ValueError('whole lifecycle memory budget differs')
    if (file_hash(root / 'model.pt') != result['checkpoint_sha256']
            or proof['fingerprint'] != result['producer_fingerprint']
            or not result['strict_reload_passed'] or not result['gradient_parity_passed']
            or result['task_accuracy'] is not None or result['final_test_opened'] or result['speed_tested']):
        raise ValueError('checkpoint, lifecycle or quality claim differs')
    state = torch.load(root / 'model.pt',map_location='cpu',weights_only=True,mmap=True)
    if (state['step'] != steps or state['producer_fingerprint'] != proof['fingerprint']
            or any(float(s['step']) != steps for s in state['optimizer']['state'].values())):
        raise ValueError('optimizer update counters differ')
    paths = [root / n for n in ('producer.json','result.json','actual_tool_receipt.json',
        'model.pt','outputs.json','trace.jsonl','memory.jsonl','sampler_remaining.json')]
    destination = run.ROOT / ('20261025_sequential_full732_' + stage + '_audit_v2')
    destination.mkdir()
    value = {'status':'raw_TRAIN_counts_trace_memory_and_update_counters_replayed','stage':stage,
        'cases':len(outputs),'updates':steps,'presentations':sum(exposures.values()),
        'exposures_min':min(exposures.values()),'exposures_max':max(exposures.values()),
        'diagnostic_totals':totals,'peak_allocated_bytes':peak,
        'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in memory),
        'first16_mean_CE':sum(r['mean_CE'] for r in trace[:16])/len(trace[:16]),
        'last16_mean_CE':sum(r['mean_CE'] for r in trace[-16:])/len(trace[-16:]),
        'artifacts':{str(p):file_hash(p) for p in paths},
        'auditor_sha256':file_hash(Path(__file__)),
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False}
    write_json(destination / 'campaign_results.json',value)
    print(json.dumps({k:v for k,v in value.items() if k != 'artifacts'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=tuple(run.RUNS))
    audit(parser.parse_args().stage)
