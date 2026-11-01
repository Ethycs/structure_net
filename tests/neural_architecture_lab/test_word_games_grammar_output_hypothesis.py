import copy

import pytest

from neural_architecture_lab.word_games_grammar_output_hypothesis import GrammarOutputEvidence, summarize


def campaign():
    cells = []
    for pair in ('full_1', 'full_2'):
        for arm in ('control', 'candidate'):
            cells.append({
                'replay': pair, 'arm': arm, 'completed': True, 'error': None,
                'totals': {'calls': 36, 'json_objects': 36, 'truncated': 0,
                           'output_tokens': 1000, 'request_seconds': 20 if arm == 'control' else 19},
                'downloads': {'raw_bytes': 1000 if arm == 'control' else 200},
            })
    return {
        'error': None, 'service_restored': True, 'results': cells,
        'parity': {p: {'equal': True, 'requests': 36, 'mismatches': []} for p in ('full_1', 'full_2')},
    }


def record(c):
    return GrammarOutputEvidence(
        hypothesis_id='word-games-grammar-on-demand-v2', **summarize(c),
        campaign='campaign.json', campaign_sha256='a' * 64,
        report='report.md', report_sha256='b' * 64,
    ).record()


def test_faster_fixed_suite_does_not_become_accuracy_or_deployment_claim():
    r = record(campaign())
    assert r['control_tok_s'] == 50
    assert r['candidate_tok_s'] == pytest.approx(2000 / 38)
    assert r['faster_pairs'] == 2
    assert r['tested'] and r['exact_replay_outputs']
    assert not r['confirmed'] and not r['population_accuracy_established'] and not r['production_deployed']


def test_one_slow_pair_does_not_support_consistent_speedup():
    c = campaign()
    c['results'][-1]['totals']['request_seconds'] = 20.1
    r = record(c)
    assert r['candidate_tok_s'] > r['control_tok_s']
    assert r['faster_pairs'] == 1
    assert r['confirmation_status'].endswith('no_consistent_speedup')


@pytest.mark.parametrize('mutation', (
    lambda c: c.update(service_restored=False),
    lambda c: c['parity']['full_2'].update(equal=False),
    lambda c: c['results'][1]['totals'].update(truncated=1),
    lambda c: c['results'][1]['totals'].update(request_seconds=float('nan')),
    lambda c: c['results'][1]['downloads'].update(raw_bytes=1000),
    lambda c: c['results'].append(copy.deepcopy(c['results'][0])),
))
def test_invalid_or_incomplete_evidence_cannot_register_as_passed(mutation):
    c = campaign()
    mutation(c)
    with pytest.raises(ValueError):
        record(c)
