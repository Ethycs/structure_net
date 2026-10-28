import pytest

from experiments.structure_net import ncm_recovered_label_review as review


def row(name, passed, digit=False):
    return {'id': name, 'text': 'A summary', 'tokens': 2, 'valid': passed is not None,
        'passed': passed, 'evaluations': None if passed is None else [{'faithful': passed, 'salient': True,
        'readable': True, 'claims': [{'claim': 'The price is 17.' if digit else 'A supported fact.',
                                  'verdict': 'supported' if passed else 'contradicted'}]}]}


def test_joint_denominator_disagreement_and_numeric_claims():
    cases = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
    scores = {'secondary': [row('a', True), row('b', False, True), row('c', True)],
              'primary': [row('a', True), row('b', True, True), row('c', False)]}
    value = review.paired_metrics(cases, scores)
    assert value['joint_passed'] == 1 and value['joint_pass_rate'] == 1/3
    assert value['joint_known_failed'] == 2 and value['joint_unresolved'] == 0
    assert value['pass_disagreements_among_both_valid'] == 2
    assert value['claim_verdicts']['secondary']['claims_containing_digits'] == {'contradicted': 1}


def test_invalid_is_unresolved_unless_other_judge_already_failed():
    cases = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
    scores = {'secondary': [row('a', None), row('b', None), row('c', True)],
              'primary': [row('a', True), row('b', False), row('c', True)]}
    value = review.paired_metrics(cases, scores)
    assert value['joint_pass_rate'] is None and value['joint_unresolved'] == 1
    assert value['joint_all_case_lower_bound'] == 1/3 and value['joint_all_case_upper_bound'] == 2/3
    assert value['both_valid'] == 1


def test_missing_or_reordered_case_cannot_improve_denominator():
    cases = [{'id': 'a'}, {'id': 'b'}]
    with pytest.raises(ValueError, match='complete ordered'):
        review.paired_metrics(cases, {'secondary': [row('a', True), row('b', True)], 'primary': [row('b', True), row('a', True)]})
