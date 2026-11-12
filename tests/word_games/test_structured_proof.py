import pytest

from experiments.word_games.structured_proof import (
    PHI, Mapping, Engine, REGIMES, alphabet, states, context, compare)


@pytest.mark.parametrize('regime', REGIMES)
def test_real_adapter_handles_commit_rejection_and_replay(regime):
    engine=Engine.fixture(); c=context(engine,regime); m=Mapping(PHI)
    for word in ([],[alphabet()[0]],[alphabet()[0],alphabet()[13]],
                 [alphabet()[18]],[alphabet()[15]]):
        checks,_=compare(engine,c,word,m)
        assert all(checks.values()), checks


def test_domain_is_all_valid_noncolliding_placements():
    fixtures=list(states())
    assert len(fixtures)==21
    for e in fixtures: e._invariants(e.loans)
    assert {x['op'] for x in alphabet()}==set(Engine.OP_FIELDS)


def test_corrupted_inverse_is_detected():
    m=Mapping(PHI); original=m.decode
    def bad(value):
        result=original(value)
        if isinstance(result,dict) and result.get('operations'):
            result['operations'][0]={'op':'cancel','loan_id':'loan:bronze'}
        return result
    m.decode=bad
    e=Engine.fixture(); c=context(e,'full')
    checks,_=compare(e,c,[alphabet()[0]],m,replay=False)
    assert not checks['result'] and not checks['complete_source_state']


def test_unrestricted_bijection_has_a_real_counterexample():
    m=Mapping(PHI)
    a={'target_id':'one'}; b={'target_id':'two'}
    assert a!=b and m.encode(a)==m.encode(b)
