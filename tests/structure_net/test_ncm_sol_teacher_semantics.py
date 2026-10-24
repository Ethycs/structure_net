"""Marginal judge scores must not replace same-case qualification."""
import pytest

from experiments.structure_net import ncm_sol_teacher_semantics as run


def scores(fail=(), invalid=()):
    return [{'id': str(i), 'valid': i not in invalid, 'passed': i not in fail and i not in invalid} for i in range(64)]


def test_intersection_not_two_marginal_passes():
    assert run.joint_gate(scores((0, 1, 2)), scores((0, 1, 2)))['qualified']
    result = run.joint_gate(scores((0, 1, 2)), scores((3, 4, 5)))
    assert result['joint_passed'] == 58 and not result['qualified']


def test_invalid_or_missing_case_cannot_pass():
    result = run.joint_gate(scores(invalid=(0,)), scores())
    assert result['joint_passed'] == 63 and not result['all_valid'] and not result['qualified']
    with pytest.raises(ValueError, match='same complete64'):
        run.joint_gate(scores()[:-1], scores())
    other = scores()
    other[-1]['id'] = 'different'
    with pytest.raises(ValueError, match='same complete64'):
        run.joint_gate(scores(), other)
