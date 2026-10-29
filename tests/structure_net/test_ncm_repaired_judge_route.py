from copy import deepcopy

import pytest

from experiments.structure_net import ncm_repaired_judge_route as repair


def test_only_reviewed_provider_changes_are_admitted():
    before = {p: pair[0] for p, pair in repair.EXPECTED_CHANGES.items()}
    after = {p: pair[1] for p, pair in repair.EXPECTED_CHANGES.items()}
    before['unchanged'] = after['unchanged'] = 'same'
    assert repair.check_changes(before, after) == repair.EXPECTED_CHANGES
    with pytest.raises(ValueError, match='two reviewed'):
        repair.check_changes(before, {**after, 'unchanged': 'other'})
    with pytest.raises(ValueError, match='inventory'):
        repair.check_changes(before, {**after, 'extra': 'other'})


def test_successor_proof_retains_original_prerequisites(monkeypatch):
    monkeypatch.setattr(repair.base, 'sha', lambda p: 'measured:'+str(p))
    original = {'source_hashes': {'original': 'source'}, 'input_hashes': {'warmup': 'hash'},
                'schema': 'unchanged', 'exec_fallback_allowed': False}
    before = deepcopy(original)
    result = repair.extend_producer(original)
    assert original == before
    assert result['source_hashes']['original'] == 'source'
    assert result['input_hashes']['warmup'] == 'hash'
    assert result['input_hashes'][str(repair.OLD_SERVICE/'producer.json')].startswith('measured:')
    assert result['endpoint_repair']['requalification_required']
    assert result['exec_fallback_allowed'] is False


def test_install_preserves_wire_protocol_and_uses_new_evidence(monkeypatch):
    q = repair.qualification
    original = (q.transport.request, q.transport.replay_row, q.score, dict(q.transport.MODELS))
    old_roots = dict(q.ROOTS)
    monkeypatch.setattr(repair, 'INSTALLED', False)
    monkeypatch.setattr(repair, 'require', lambda: None)
    monkeypatch.setattr(repair.gateway, 'install', lambda: None)
    for target, names in ((repair.gateway, ('SERVICE', 'ROOT', 'UNIT', 'PLAN', 'TEST', 'producer_proof')),
                          (q, ('neutral', 'ROOTS', 'PLAN', 'TEST')),
                          (q.transport, ('IDENTITY', 'credential'))):
        for name in names:
            monkeypatch.setattr(target, name, getattr(target, name))
    repair.install()
    assert (q.transport.request, q.transport.replay_row, q.score, q.transport.MODELS) == original
    assert not set(old_roots.values()) & set(q.ROOTS.values())
    assert q.transport.IDENTITY == 'ncm-persistent-isolated-SQLite-judge-transport-v1'
    assert repair.gateway.SERVICE == repair.SERVICE


def test_consumer_cannot_change_prediction_or_candidate_families():
    assert repair.consumer_arguments('task', 'grade', 'expanded') == ['grade', '--arm', 'expanded']
    assert repair.consumer_arguments('hierarchy', 'audit', 'tree') == ['hierarchy', 'audit', '--arm', 'tree']
    with pytest.raises(ValueError):
        repair.consumer_arguments('task', 'outputs', 'expanded')
    with pytest.raises(ValueError):
        repair.consumer_arguments('task', 'grade', 'replacement')
