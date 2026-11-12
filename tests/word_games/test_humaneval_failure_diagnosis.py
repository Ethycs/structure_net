import ast

import pytest

from experiments.word_games.diagnose_humaneval_failures import diagnostic_cases, corrected_instruction, replace_once


def test_all_arithmetic_cases_are_bounded_and_include_right_associativity():
    cases=diagnostic_cases(160)
    assert any(c['args']==[['**','**'],[2,3,2]] and c['expected']==512 for c in cases)
    assert all(type(c['expected']) is int and abs(c['expected'])<10**15 for c in cases)
    assert len({repr(c['args']) for c in cases})==len(cases)


def test_missing_suffix_intervention_changes_no_existing_instruction_prefix():
    controller='Implement odd_count(lst). In `lst` use the literal. Example ["whole"].'
    original='Implement bridge_operation(parcel_0). In `parcel_0` use the literal. Example ['
    fixed=corrected_instruction(113,controller,original)
    assert fixed.startswith(original) and fixed[len(original):]=='"whole"].'
    with pytest.raises(ValueError):corrected_instruction(113,controller,'unrelated instruction')


def test_interface_intervention_changes_only_named_signature():
    original='Implement find_zero(xs: list). Expand until a bracket is found.'
    fixed=corrected_instruction(32,'unused',original)
    assert fixed=='Implement bridge_operation(parcel_0: list). Expand until a bracket is found.'


def test_patch_anchor_must_be_unique():
    with pytest.raises(ValueError): replace_once('x x','x','y')
    with pytest.raises(ValueError): replace_once('abc','missing','replacement')


def test_diagnostic_corpora_cover_empty_and_multiple_digit_counts():
    cases=diagnostic_cases(113)
    assert len(cases)==40 and cases[0]['expected']==[]
    assert any('12nput.' in item for c in cases for item in c['expected'])
    assert len(diagnostic_cases(32))==20
