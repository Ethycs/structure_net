"""Summarize CPU expert elimination without treating exact wording as accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from experiments.word_games.common import read, sha, write, digest
from experiments.word_games.native_speed import output_tokens
from experiments.word_games.cpu_expert_elimination import receiver_accuracy, REQUESTS
from experiments.neural_architecture_lab.analyze_word_games_moe_placement import structural_diagnostic


def responses(root):
    return {str(p.parent.relative_to(root / 'runs')): read(p)
            for p in (root / 'runs').rglob('response.json')}


def summarize(root, preparation, review, out):
    out.mkdir(parents=True, exist_ok=False)
    campaign, producer = read(root / 'campaign_results.json'), read(root / 'producer.json')
    if campaign['error'] is not None or not campaign['service_restored']:
        raise ValueError('complete campaign and service restoration required')
    if producer['fingerprint'] != digest({k: v for k, v in producer.items() if k != 'fingerprint'}):
        raise ValueError('producer fingerprint mismatch')
    if Counter(sha(p) for p in (root / 'producer_source').iterdir()) != Counter(producer['source_hashes'].values()):
        raise ValueError('producer snapshot mismatch')
    if sha(preparation / 'result.json') != producer['preparation_sha256']:
        raise ValueError('derived checkpoint provenance mismatch')
    if sha(REQUESTS) != producer['requests_sha256']:
        raise ValueError('request population changed')
    manual = read(review)
    for record in [*manual['receivers'], *manual['selected_sender_differences']]:
        for key in ('request', 'response'):
            if key in record and sha(record[key]) != record[key + '_sha256']:
                raise ValueError('manual review evidence changed')
    for cell in campaign['results']:
        for path, h in cell['response_hashes'].items():
            if sha(path) != h:
                raise ValueError('response changed: ' + path)
    full = [c for c in campaign['results'] if c['phase'] == 'full']
    conditions, details = {}, []
    for name in campaign['advanced']:
        cells = sorted((c for c in full if c['arm'] == name), key=lambda c: c['repeat'])
        if len(cells) != 2 or any(not c['completed'] or c['totals']['calls'] != 36 for c in cells):
            raise ValueError('two completed full passes required')
        tokens = sum(c['totals']['output_tokens'] for c in cells)
        seconds = sum(c['totals']['request_seconds'] for c in cells)
        cpu = sum(c['cpu_seconds'] for c in cells)
        quality, faster, ratios, memory = [], [], [], {}
        for c in cells:
            path = root / f'full_{c["repeat"]}_{name}'
            refpath = root / f'full_{c["repeat"]}_packed38'
            actual, reference = responses(path), responses(refpath)
            if len(actual) != 36 or set(actual) != set(reference):
                raise ValueError('request population mismatch')
            q = receiver_accuracy(path)
            if q != read(path / 'quality.json'):
                raise ValueError('saved receiver grading does not match recomputation')
            quality.append(q)
            diagnostic = structural_diagnostic(reference, actual)
            ref = next(r for r in full if r['repeat'] == c['repeat'] and r['arm'] == 'packed38')
            faster.append(c['totals']['output_tokens_per_second'] > ref['totals']['output_tokens_per_second'])
            pair_ratios = [(output_tokens(actual[k]['response']) / actual[k]['elapsed_seconds']) /
                           (output_tokens(reference[k]['response']) / reference[k]['elapsed_seconds'])
                           for k in actual]
            ratios += pair_ratios
            for line in (path / 'telemetry.jsonl').read_text().splitlines():
                for row in json.loads(line)['gpus']:
                    memory[row[0]] = min(memory.get(row[0], 99999), int(row[2]))
            details.append({'arm': name, 'repeat': c['repeat'], 'totals': c['totals'],
                'cpu_seconds': c['cpu_seconds'], 'quality': q, 'structure_vs_control': diagnostic,
                'text_equal': sum(actual[k]['response']['content'] == reference[k]['response']['content'] for k in actual),
                'median_request_rate_ratio': median(pair_ratios)})
        conditions[name] = {
            'output_tokens': tokens, 'request_seconds': seconds, 'tok_s': tokens / seconds,
            'byte_s': sum(c['totals']['output_bytes'] for c in cells) / seconds,
            'decode_tok_s': tokens / sum(c['totals']['decode_seconds'] for c in cells),
            'prefill_seconds': sum(c['totals']['prefill_seconds'] for c in cells),
            'cpu_seconds': cpu, 'cpu_ms_per_output_token': cpu / tokens * 1000,
            'max_observed_threads': max(c['max_observed_threads'] for c in cells),
            'max_observed_rss_mib': max(c['max_observed_rss_kib'] for c in cells) / 1024,
            'min_gpu_free_mib': memory, 'faster_full_passes': sum(faster),
            'median_request_rate_ratio': median(ratios),
            'receiver_correct': sum(q['receiver_correct'] for q in quality),
            'receiver_cases': sum(q['receiver_cases'] for q in quality),
            'valid_selectors': sum(q['selector_valid'] for q in quality),
        }
    control = conditions['packed38']
    for c in conditions.values():
        c['throughput_ratio'] = c['tok_s'] / control['tok_s']
        c['cpu_per_token_reduction'] = 1 - c['cpu_ms_per_output_token'] / control['cpu_ms_per_output_token']
    result = {
        'hypothesis_id': campaign['hypothesis_id'], 'confirmed': False,
        'classification': 'exploratory_speed_and_cpu_cost_measured_quality_unestablished',
        'campaign': str(root / 'campaign_results.json'), 'campaign_sha256': sha(root / 'campaign_results.json'),
        'preparation': str(preparation / 'result.json'), 'preparation_sha256': sha(preparation / 'result.json'),
        'producer_sha256': sha(root / 'producer.json'), 'analysis_source_sha256': sha(__file__),
        'manual_review': str(review), 'manual_review_sha256': sha(review),
        'service_restored': campaign['service_restored'], 'conditions': conditions, 'full_passes': details,
        'pilots': [c for c in campaign['results'] if c['phase'] == 'pilot'],
        'accuracy_scope': 'receiver state against fixed authorized packets; selector schema; summaries separately reviewed',
        'population_accuracy_established': False, 'independent_semantic_judge': False,
    }
    write(out / 'campaign_results.json', result)
    print(json.dumps(conditions, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.preparation, args.review, args.output)
