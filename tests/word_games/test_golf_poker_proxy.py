import copy

import pytest

from experiments.word_games.golf_poker_proxy import (
    ACTIONS, STATES, advance, course, oracle, translate_rules, validate_mapping,
)


def test_courses_and_bellman_residual():
    for i in range(1, 28):
        c = course(i)
        v, q = oracle(c['rules'])
        assert 2 < v['tee'] < 12
        assert v['holed'] == 0
        for state, choices in c['rules'].items():
            for action, outcomes in choices.items():
                assert sum(w for _,w,_ in outcomes) == 10
                expected = sum(w/10*(1+p+v[n]) for n,w,p in outcomes)
                assert abs(expected-q[state][action]) < 1e-9
            if choices:
                assert abs(v[state]-min(q[state].values())) < 1e-9


def test_mapping_roundtrip_and_reject_alias_collision():
    mapping = {'states': dict(zip(STATES, ('ante', 'flop', 'turn', 'draw', 'blind', 'river', 'showdown'))),
               'actions': dict(zip(ACTIONS, ('raise', 'call', 'bet', 'check', 'reveal', 'fold', 'limp')))}
    validate_mapping(mapping)
    encoded = translate_rules(course(1)['rules'], mapping)
    decoded = translate_rules(encoded, {k: {v:s for s,v in d.items()} for k,d in mapping.items()})
    assert decoded == course(1)['rules']
    mapping['states']['holed'] = mapping['states']['tee']
    with pytest.raises(ValueError, match='injective'):
        validate_mapping(mapping)


def test_transition_shared_randomness_and_illegal_move_no_mutation():
    a = {'course': course(1), 'state': 'tee', 'strokes': 0, 'moves': 0}
    b = copy.deepcopy(a)
    assert advance(a, 'driver') == advance(b, 'driver')
    assert a == b
    before = copy.deepcopy(a)
    with pytest.raises(ValueError):
        advance(a, 'not_an_action')
    assert a == before
