"""Hash-checked throughput and output diagnostics for the 70 tok/s study."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from experiments.word_games.common import read, sha, write, digest
from experiments.word_games.cpu_expert_elimination import receiver_accuracy
from experiments.word_games.native_speed import output_tokens
from experiments.word_games.qwen_runtime import REQUESTS, REQUESTS_SHA
from experiments.word_games.target_70tps import HYPOTHESIS
from experiments.neural_architecture_lab.analyze_word_games_cpu_experts import responses
from experiments.neural_architecture_lab.analyze_word_games_moe_placement import structural_diagnostic


def analyze(root, review, out):
    out.mkdir(parents=True, exist_ok=False)
    campaign, producer = read(root / 'campaign_results.json'), read(root / 'producer.json')
    if campaign['hypothesis_id'] != HYPOTHESIS or campaign['fingerprint'] != producer['fingerprint']:
        raise ValueError('identity mismatch')
    if campaign['error'] is not None or not campaign['service_restored']:
        raise ValueError('completed campaign with restored service required')
    if producer['fingerprint'] != digest({k: v for k, v in producer.items() if k != 'fingerprint'}):
        raise ValueError('fingerprint mismatch')
    if Counter(sha(p) for p in (root / 'producer_source').iterdir()) != Counter(producer['source_hashes'].values()):
        raise ValueError('source snapshots changed')
    if sha(REQUESTS) != REQUESTS_SHA or producer['requests_sha256'] != REQUESTS_SHA:
        raise ValueError('requests changed')
    if producer.get('first_stage') and sha(producer['first_stage']) != producer['first_stage_sha256']:
        raise ValueError('preceding stage changed')
    manual = read(review)
    for item in manual['evidence']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('reviewed evidence changed')
    for cell in campaign['results']:
        for path, expected in cell['response_hashes'].items():
            if sha(path) != expected:
                raise ValueError('response changed')
    full = [c for c in campaign['results'] if c['phase'] == 'full']
    conditions, details = {}, []
    for name in ['control', *campaign['advanced']]:
        cells = sorted((c for c in full if c['arm'] == name), key=lambda c: c['repeat'])
        if len(cells) != 2 or any(not c['completed'] or c['totals']['calls'] != 36 for c in cells):
            raise ValueError('two complete passes required')
        quality, ratios, memory, text_counts, split_counts = [], [], {}, [], []
        for c in cells:
            path = root / f'full_{c["repeat"]}_{name}'
            refpath = root / f'full_{c["repeat"]}_control'
            actual, reference = responses(path), responses(refpath)
            if len(actual) != 36 or set(actual) != set(reference):
                raise ValueError('wrong request population')
            q = receiver_accuracy(path)
            if q != read(path / 'quality.json'):
                raise ValueError('receiver grading changed')
            quality.append(q)
            text_equal = sum(actual[k]['response']['content'] == reference[k]['response']['content'] for k in actual)
            text_counts.append(text_equal)
            ratios += [(output_tokens(actual[k]['response']) / actual[k]['elapsed_seconds']) /
                       (output_tokens(reference[k]['response']) / reference[k]['elapsed_seconds']) for k in actual]
            for line in (path / 'telemetry.jsonl').read_text().splitlines():
                for row in json.loads(line)['gpus']:
                    memory[row[0]] = min(memory.get(row[0], 99999), int(row[2]))
            placement = read(path / 'placement.json')
            split_counts += [g['splits'] for g in placement['graphs']]
            details.append({'arm': name, 'repeat': c['repeat'], 'totals': c['totals'],
                 'quality': q, 'structure_vs_control': structural_diagnostic(reference, actual),
                 'text_equal': text_equal, 'cpu_seconds': c['cpu_seconds'], 'placement': placement,
                 'checkpoint_events_including_warmup': c.get('checkpoint_events_including_warmup')})
        tokens = sum(c['totals']['output_tokens'] for c in cells)
        seconds = sum(c['totals']['request_seconds'] for c in cells)
        conditions[name] = {
            'tok_s': tokens / seconds, 'target_70_reached': tokens / seconds >= 70,
            'output_tokens': tokens, 'request_seconds': seconds,
            'byte_s': sum(c['totals']['output_bytes'] for c in cells) / seconds,
            'decode_tok_s': tokens / sum(c['totals']['decode_seconds'] for c in cells),
            'prefill_seconds': sum(c['totals']['prefill_seconds'] for c in cells),
            'cpu_ms_per_output_token': sum(c['cpu_seconds'] for c in cells) / tokens * 1000,
            'max_observed_rss_mib': max(c['max_observed_rss_kib'] for c in cells) / 1024,
            'min_gpu_free_mib': memory, 'median_request_rate_ratio': median(ratios),
            'text_equal': sum(text_counts), 'comparisons': 72, 'graph_split_counts': split_counts,
            'receiver_correct': sum(q['receiver_correct'] for q in quality), 'receiver_cases': 24,
            'valid_selectors': sum(q['selector_valid'] for q in quality),
            'drafted': sum(c['totals']['drafted'] for c in cells),
            'draft_accepted': sum(c['totals']['draft_accepted'] for c in cells),
        }
    best = max(conditions, key=lambda k: conditions[k]['tok_s'])
    for c in conditions.values():
        c['throughput_ratio'] = c['tok_s'] / conditions['control']['tok_s']
        c['existing_checks_pass'] = c['receiver_correct'] == 24 and c['valid_selectors'] == 2
    checked = [k for k, c in conditions.items() if c['existing_checks_pass']]
    best_checked = max(checked, key=lambda k: conditions[k]['tok_s']) if checked else None
    classification = 'exploratory_below_70tps_quality_unestablished'
    if conditions[best]['target_70_reached']:
        classification = ('exploratory_70tps_reached_quality_unestablished' if conditions[best]['existing_checks_pass']
                          else 'exploratory_70tps_reached_existing_checks_failed')
    result = {'hypothesis_id': HYPOTHESIS, 'confirmed': False,
        'classification': classification, 'best_checked_arm': best_checked,
        'qualified_70tps_reached': any(c['target_70_reached'] and c['existing_checks_pass'] for c in conditions.values()),
        'best_arm': best, 'conditions': conditions, 'full_passes': details,
        'pilots': [c for c in campaign['results'] if c['phase'] == 'pilot'],
        'campaign': str(root / 'campaign_results.json'), 'campaign_sha256': sha(root / 'campaign_results.json'),
        'manual_review': str(review), 'manual_review_sha256': sha(review),
        'preceding_stage': producer.get('first_stage'), 'preceding_stage_sha256': producer.get('first_stage_sha256'),
        'producer_sha256': sha(root / 'producer.json'), 'analysis_source_sha256': sha(__file__),
        'service_restored': True, 'population_accuracy_established': False,
        'independent_semantic_judge': False, 'production_deployed': False}
    write(out / 'campaign_results.json', result)
    print(json.dumps({'classification': result['classification'], 'best_arm': best, 'conditions': conditions}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.root, a.review, a.output)
