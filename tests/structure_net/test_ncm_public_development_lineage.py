from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_public_development_lineage as lineage


def packet_key(packet, role):
    return packet['id']+':'+role


def write(path, value):
    path.write_text(json.dumps(value))


def grade(root, inherited, *, tag='new'):
    root.mkdir()
    packets = {c: {'student': [{'id': f'{tag}_{c}_{i}'} for i in range(64)]}
               for c in ('short', 'long')}
    cache = {**inherited, **{packet_key(p, 'secondary'): {'decision': False}
             for c in packets.values() for p in c['student']}}
    result = {'inherited_first_judgments': len(inherited), 'canonical_first_judgments': len(cache),
              'primary_admitted': False}
    write(root/'producer.json', {'schema': 'fixed_producer'})
    write(root/'result.json', result)
    write(root/'judgment_cache.json', cache)
    write(root/'exact_packets.json', packets)
    bind(root)
    return cache


def bind(root):
    write(root/'actual_tool_receipt.json', {'actual_tool_exit_code': 0,
        'artifacts': {str(root/n): lineage.data.sha(root/n) for n in ('producer.json', 'result.json')}})


@pytest.fixture
def evaluation():
    return SimpleNamespace(old=SimpleNamespace(COHORTS=('short', 'long'),
        oracle=SimpleNamespace(first=SimpleNamespace(packet_key=packet_key))))


def test_complete_extension_preserves_prior_failures_and_reuses_firsts(tmp_path, evaluation):
    inherited = {'historical': {'decision': False}, 'new_short_0:secondary': {'decision': False}}
    expected = grade(tmp_path/'grade', inherited)
    before = deepcopy(inherited)
    assert lineage.append_grade(inherited, tmp_path/'grade', evaluation) == expected
    assert inherited == before
    assert len(expected) == len(inherited)+127


@pytest.mark.parametrize('failure', [
    'changed_first', 'missing_first', 'unrelated_first', 'unadmitted_primary',
    'wrong_count', 'wrong_ancestor_count', 'missing_cohort', 'partial_cohort',
    'unbound_producer', 'failed_tool',
])
def test_rejects_lost_changed_or_unqualified_judgments(tmp_path, evaluation, failure):
    root = tmp_path/'grade'
    inherited = {'historical': {'decision': False}}
    cache = grade(root, inherited)
    result = lineage.data.read(root/'result.json')
    packets = lineage.data.read(root/'exact_packets.json')
    if failure == 'changed_first': cache['historical'] = {'decision': True}
    elif failure == 'missing_first': del cache['historical']
    elif failure == 'unrelated_first': cache['unrelated'] = {'decision': True}
    elif failure == 'unadmitted_primary': cache['new_short_0:primary'] = {'decision': True}
    elif failure == 'wrong_ancestor_count': result['inherited_first_judgments'] = 0
    elif failure == 'missing_cohort': del packets['long']
    elif failure == 'partial_cohort': packets['long']['student'].pop()
    result['canonical_first_judgments'] = len(cache)+(failure == 'wrong_count')
    write(root/'result.json', result)
    write(root/'judgment_cache.json', cache)
    write(root/'exact_packets.json', packets)
    bind(root)
    if failure in ('unbound_producer', 'failed_tool'):
        receipt = lineage.data.read(root/'actual_tool_receipt.json')
        if failure == 'unbound_producer': del receipt['artifacts'][str(root/'producer.json')]
        else: receipt['actual_tool_exit_code'] = 1
        write(root/'actual_tool_receipt.json', receipt)
    with pytest.raises(ValueError):
        lineage.append_grade(inherited, root, evaluation)


def test_depth_cache_retains_both_128_case_extensions(tmp_path, monkeypatch, evaluation):
    base, results = tmp_path/'base', tmp_path/'results'
    base.mkdir(); results.mkdir()
    inherited = {str(i): {'decision': False} for i in range(4179)}
    write(base/'judgment_cache.json', inherited)
    root12, root24 = tmp_path/'depth12', tmp_path/'depth24'
    control = grade(root12, inherited, tag='depth12')
    deeper = grade(root24, control, tag='depth24')
    write(results/'campaign_results.json', {'canonical_first_judgments': 4435, 'artifacts': {}})
    campaign_sha = lineage.data.sha(results/'campaign_results.json')
    write(results/'actual_tool_receipt.json', {'actual_tool_exit_code': 0,
        'artifacts': {str(results/'campaign_results.json'): campaign_sha}})
    monkeypatch.setattr(lineage.original, 'BASE', base)
    monkeypatch.setattr(lineage.original, 'BASE_SHA', lineage.data.sha(base/'judgment_cache.json'))
    monkeypatch.setattr(lineage, 'DEPTH_RESULTS', results)
    monkeypatch.setattr(lineage, 'DEPTH_ROOTS', (root12, root24))
    monkeypatch.setattr(lineage, 'DEPTH_CAMPAIGN_SHA', campaign_sha)
    monkeypatch.setattr(lineage, 'DEPTH_CACHE_SHA', lineage.data.sha(root24/'judgment_cache.json'))
    assert lineage.depth_cache({'0': inherited['0']}, evaluation) == deeper
    assert len(deeper) == 4435
    changed = dict(deeper)
    changed['depth12_short_0:secondary'] = {'decision': True}
    write(root24/'judgment_cache.json', changed)
    with pytest.raises(ValueError, match='changed or disappeared'):
        lineage.depth_cache({'0': inherited['0']}, evaluation)
