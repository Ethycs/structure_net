from itertools import combinations

import pytest

from experiments.structure_net import jspace_layer_combination_policy as p


def score(correct=27, agreement=27, ce=2.):
    return dict(metrics=dict(correct=correct, teacher_agreement_count=agreement, choice_ce=ce))


def screen(): return {k: score() for k in p.screen_masks()}


def test_complete_pair_screen_and_both_background_enumeration():
    masks = p.screen_masks()
    assert len(masks) == 681
    assert all(p.mask_id(p.restored(pair)) in masks for pair in combinations(range(36), 2))
    assert len(p.six_groups()) == 12
    candidates = [0, 4, 9, 13, 25, 35]
    refinement = p.refinement_masks(candidates)
    assert len(refinement) == 128
    assert p.mask_id(p.compressed(candidates)) in refinement
    assert p.mask_id(p.restored(candidates)) in refinement


def test_higher_order_rescue_has_prospective_priority():
    scores = screen()
    group = p.six_groups()[7]
    scores[p.mask_id(p.restored(group))] = score(40)
    scores[p.mask_id(p.restored((2, 7)))] = score(38)
    chosen = p.select_candidates(scores)
    assert chosen['layers'] == list(group)
    assert chosen['method'].startswith('six_layer')
    scores[p.mask_id(p.restored((2, 7)))] = score(39)
    assert p.select_candidates(scores)['method'].startswith('top_two')


def test_nonadditive_pair_can_enter_despite_poor_singletons():
    scores = screen()
    scores[p.mask_id(p.restored((0,)))] = score(31)
    scores[p.mask_id(p.restored((1,)))] = score(30)
    scores[p.mask_id(p.restored((14, 27)))] = score(38)
    assert {0, 1, 14, 27} <= set(p.select_candidates(scores)['layers'])


def test_repair_is_exactly_three_and_ranked_in_actual_compact_background():
    candidates = [0, 1, 4, 10, 20, 35]
    scores = {key: score() for key in p.refinement_masks(candidates)}
    scores[p.mask_id(p.restored((4, 10, 35)))] = score(43, 45, .9)
    scores[p.mask_id(p.compressed((0, 1, 20)))] = score(60, 60, .1)
    chosen = p.selection(scores, candidates)
    assert chosen['layers'] == [4, 10, 35]
    assert chosen['rescue_correct'] == 16
    assert chosen['positive_rescue']


def test_ties_deterministic_and_incomplete_search_rejected():
    scores = screen()
    first = p.select_candidates(scores)
    assert p.select_candidates(dict(reversed(list(scores.items())))) == first
    scores.pop(next(iter(scores)))
    with pytest.raises(ValueError): p.select_candidates(scores)
    with pytest.raises(ValueError): p.mask_id([1]*36)
    with pytest.raises(ValueError): p.compressed([1, 1])


def test_paired_decisions_not_just_equal_accuracy():
    records = [dict(identity=str(i), answer=0, prediction=i%2) for i in range(64)]
    other = [dict(r, prediction=1-r['prediction']) for r in records]
    result = p.paired(dict(records=records), dict(records=other))
    assert result['gained'] == result['lost'] == 32
    with pytest.raises(ValueError): p.paired(dict(records=records), dict(records=other[::-1]))
