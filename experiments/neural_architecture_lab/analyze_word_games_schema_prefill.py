"""Hash-checked paired analysis of the public-prefix Triton replay."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from experiments.word_games.qwen_runtime import REQUESTS, REQUESTS_SHA


def analyze(campaign, output):
    data = read(campaign)
    if not data['completed'] or data['pilot'] or not data['original_repository_restored']:
        raise ValueError('completed full replay and restored baseline required')
    if sha(REQUESTS) != REQUESTS_SHA or len(data['passes']) != 4 or len(data['response_hashes']) != 144:
        raise ValueError('four full frozen passes required')
    for p, h in data['response_hashes'].items():
        if sha(p) != h:
            raise ValueError('response changed: ' + p)
    if [p['cached'] for p in data['passes']] != [False, True, True, False]:
        raise ValueError('ABBA order required')
    conditions = {}
    for name, cached in [('control', False), ('cached', True)]:
        passes = [p for p in data['passes'] if p['cached'] == cached]
        tokens = sum(p['totals']['output_tokens'] for p in passes)
        seconds = sum(p['totals']['request_seconds'] for p in passes)
        conditions[name] = {'output_tokens': tokens, 'request_seconds': seconds, 'tok_s': tokens / seconds,
            'prefill_seconds': sum(p['prefill_seconds'] for p in passes),
            'restore_seconds': sum(p['restore_seconds'] for p in passes),
            'cache_hits': sum(p['cache_hits'] for p in passes), 'cache_tokens': sum(p['cache_tokens'] for p in passes),
            'receiver_correct': sum(p['quality']['receiver_correct'] for p in passes),
            'receiver_cases': sum(p['quality']['receiver_cases'] for p in passes),
            'valid_selectors': sum(p['quality']['selector_valid'] for p in passes),
            'repeat_tok_s': [p['totals']['output_tokens_per_second'] for p in passes],
            'differences_from_prior': [p['different_from_prior'] for p in passes]}
    changes, structured_changes = [], []
    for c, b in [(1, 0), (2, 3)]:
        candidate, baseline = [Path(data['passes'][i]['root']) for i in (c, b)]
        for item in read(REQUESTS):
            paths = [p / 'runs' / item['id'] / 'response.json' for p in (candidate, baseline)]
            current, original = [read(p)['parsed'] for p in paths]
            if current != original:
                changes.append({'candidate_pass': c + 1, 'control_pass': b + 1, 'id': item['id'],
                                'candidate': current, 'control': original})
            # Diagnostic only: these fields control execution or report state.
            # Excluding prose here does not amend the exact-output adoption gate.
            prose = {'target_text'} if '/sender/' in item['id'] else {'summary'} if '/receiver' in item['id'] else set()
            if {k: v for k, v in current.items() if k not in prose} != {
                    k: v for k, v in original.items() if k not in prose}:
                structured_changes.append({'candidate_pass': c + 1, 'control_pass': b + 1, 'id': item['id'],
                                           'candidate': current, 'control': original})
    cached, control = conditions['cached'], conditions['control']
    checks = all(p['quality']['receiver_correct'] == 12 and p['quality']['receiver_cases'] == 12
                 and p['quality']['selector_valid'] for p in data['passes'])
    stable_control = not any(control['differences_from_prior'])
    faster = cached['tok_s'] > control['tok_s']
    parity = not changes and stable_control
    classification = ('exploratory_speedup_with_output_parity' if faster and parity and checks else
                      'exploratory_speedup_requires_quality_review' if faster and checks else
                      'exploratory_existing_checks_failed' if not checks else 'exploratory_no_speedup')
    builds = [read(Path(p['root']) / 'prefix_build.json')[0] for p in data['passes'] if p['cached']]
    saved_per_request = (control['request_seconds'] - cached['request_seconds']) / 72
    result = {'hypothesis_id': data['hypothesis_id'], 'classification': classification,
        'campaign': str(campaign), 'campaign_sha256': sha(campaign), 'conditions': conditions,
        'paired_parsed_differences': changes, 'matching_paired_outputs': 72 - len(changes),
        'structured_field_differences': structured_changes,
        'matching_structured_outputs': 72 - len(structured_changes),
        'stable_control': stable_control, 'existing_checks_pass': checks,
        'speedup_percent': (cached['tok_s'] / control['tok_s'] - 1) * 100,
        'request_time_saved_seconds': control['request_seconds'] - cached['request_seconds'],
        'prefill_time_saved_seconds': control['prefill_seconds'] - cached['prefill_seconds'],
        'target_70_reached': cached['tok_s'] >= 70, 'serving_adoption_gate': faster and parity and checks,
        'prefix_build_seconds': [b['seconds'] for b in builds],
        'prefix_bytes': [sum(f['bytes'] for f in b['files'].values()) for b in builds],
        'amortization_requests': [b['seconds'] / saved_per_request if saved_per_request > 0 else None for b in builds],
        'confirmed': False, 'population_accuracy_established': False,
        'scope': 'frozen requests and existing mechanical checks; no independent semantic judge'}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'campaign_results.json', result)
    print({k: v for k, v in result.items() if k not in ('conditions', 'paired_parsed_differences', 'structured_field_differences')})
    print({k: {n: v[n] for n in ('tok_s', 'request_seconds', 'prefill_seconds', 'restore_seconds')}
           for k, v in conditions.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.campaign, a.output)
