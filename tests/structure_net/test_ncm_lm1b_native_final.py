from copy import deepcopy

import pytest

from experiments.structure_net import ncm_lm1b_native_final as run


def qualified_DEV():
    return {'classification': 'full128_source_only_DEV_qualified', 'DEV_qualified': True, 'primary_admitted': True,
        'all128_cases_retained': True, 'source_prediction': True, 'oracle_target_codes': False, 'final_test_opened': False,
        'metrics': {c: {role: {'cases': 64, 'valid': 64, 'passed': 61} for role in ('secondary', 'primary')}
            for c in run.development.COHORTS},
        'joint': {c: {'cases': 64, 'all_valid': True, 'joint_passed': 61, 'qualified': True} for c in run.development.COHORTS}}


def test_actual_same_case_both_cohort_gate_required():
    state = qualified_DEV()
    run.development_gate(state)
    bad = deepcopy(state)
    bad['joint'][run.development.COHORTS[0]]['joint_passed'] = 58
    with pytest.raises(ValueError):
        run.development_gate(bad)
    for patch in ({'DEV_qualified': False}, {'all128_cases_retained': False}, {'oracle_target_codes': True},
            {'source_prediction': False}, {'final_test_opened': True}):
        with pytest.raises(ValueError):
            run.development_gate({**state, **patch})


def test_failed_admission_stops_before_any_final_source_or_metadata_access(monkeypatch):
    events = []

    def refuse():
        events.append('admission')
        raise ValueError('not admitted')

    monkeypatch.setattr(run, 'require_admission', refuse)
    monkeypatch.setattr(run.reservation, 'sealed_metadata', lambda: pytest.fail('sealed metadata accessed before admission'))
    monkeypatch.setattr(run, 'file_hash', lambda _: pytest.fail('file hashed before admission'))
    for function in (run.prepare, run.sources, run.base_cache, run.workflow):
        with pytest.raises(ValueError, match='not admitted'):
            function()
    assert events == ['admission'] * 4


def test_final_thresholds_and_same_case_joint_not_two_marginals():
    assert run.minimum(256) == 244 and run.minimum(64) == 61
    a = [{'id': str(i), 'valid': True, 'passed': i >= 12} for i in range(256)]
    b = [{'id': str(i), 'valid': True, 'passed': i not in range(12, 24)} for i in range(256)]
    assert run.consensus(a, a)['qualified']
    result = run.consensus(a, b)
    assert result['joint_passed'] == 232 and not result['qualified']
    assert not run.consensus(a, None)['qualified'] and run.consensus(a, None)['task_accuracy'] is None
