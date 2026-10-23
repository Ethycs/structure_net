import json

import pytest

from experiments.structure_net import ncm_sealed_final_quality as final


def scores(count, failed=()):
    return [{'id': str(i), 'valid': True, 'passed': i not in failed} for i in range(count)]


def test_two_individually_95_percent_judges_do_not_imply_95_percent_consensus():
    a = scores(20, {0})
    b = scores(20, {1})
    result = final.consensus(a, b)
    assert result['joint_passed'] == 18 and result['task_accuracy'] == .9
    assert not result['qualified']
    agreed = final.consensus(a, a)
    assert agreed['joint_passed'] == 19 and agreed['qualified']
    assert json.loads(json.dumps(agreed, allow_nan=False)) == agreed
    assert final.minimum(256) == 244 and final.minimum(64) == 61


def test_invalid_judgments_stay_in_the_denominator_and_prevent_a_success_claim():
    a, b = scores(20), scores(20)
    b[0].update(valid=False, passed=None)
    result = final.consensus(a, b)
    assert result['cases'] == 20 and result['valid_both'] == 19
    assert result['joint_passed'] == 19 and result['task_accuracy'] is None
    assert not result['qualified']
    assert result['joint_accuracy_lower_bound'] == .95 and result['joint_accuracy_upper_bound'] == 1.
    assert not final.consensus(a, None)['qualified']
    b[0]['passed'] = True
    with pytest.raises(ValueError, match='validity and complete-summary'):
        final.consensus(a, b)
    b[0]['passed'] = None
    with pytest.raises(ValueError, match='same complete source order'):
        final.consensus(a, list(reversed(b)))


def test_failed_development_admission_prevents_even_a_sealed_source_read(monkeypatch):
    def deny():
        raise ValueError('development gate failed')
    def forbidden_read(path):
        raise AssertionError('must not read sealed data: '+str(path))
    monkeypatch.setattr(final, 'require_admission', deny)
    monkeypatch.setattr(final.q, 'read', forbidden_read)
    with pytest.raises(ValueError, match='development gate failed'):
        final.prepare()
