import pytest

from experiments.neural_architecture_lab.finalize_ncm_level_conditioning_necessary_gate import bounds


def rows(empty_short=6, empty_long=4):
    return [{'id': f'{split}-{i}', 'split': split, 'text': '  ' if i < empty else 'text'}
        for split, empty in (('short_development', empty_short), ('long_diagnostic', empty_long))
        for i in range(64)]


def test_empty_outputs_give_only_an_upper_bound():
    b = bounds(rows())
    assert b['maximum_possible_passes'] == 118
    assert b['maximum_possible_rate'] == 118 / 128
    assert b['actual_semantic_accuracy'] is None
    assert not b['necessary_quality_gate_possible']


def test_each_cohort_must_be_possible_even_when_combined_bound_is_high():
    b = bounds(rows(4, 0))
    assert b['maximum_possible_rate'] > .95
    assert not b['necessary_quality_gate_possible']
    assert bounds(rows(3, 3))['necessary_quality_gate_possible']


def test_missing_or_duplicate_predictions_are_not_silently_excluded():
    with pytest.raises(ValueError):
        bounds(rows()[:-1])
    bad = rows()
    bad[-1] = bad[0]
    with pytest.raises(ValueError):
        bounds(bad)
