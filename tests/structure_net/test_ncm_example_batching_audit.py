from copy import deepcopy
import pytest

from experiments.neural_architecture_lab.finalize_ncm_example_batching import parity_gate, replay_parity


CONFIG = {'gradient_relative_tolerance': 5e-5, 'gradient_absolute_tolerance': 1e-4}


def row():
    return {'gradient_presence_equal': True, 'finite': True, 'relative_gradient_L2_error': 5e-5,
        'max_abs_gradient_difference': 1e-4, 'logits_allclose': True, 'argmax_disagreements': 0,
        'passed': True, 'parameters': [{'name': 'weight', 'squared_error': 1e-8,
            'reference_squared_norm': 4., 'max_absolute_error': 1e-4}]}


def test_joint_gate_includes_boundary_and_rejects_each_violation():
    assert parity_gate(row(), CONFIG)
    for name, value in (('gradient_presence_equal', False), ('finite', False),
                        ('relative_gradient_L2_error', 5.1e-5), ('max_abs_gradient_difference', 1.1e-4),
                        ('logits_allclose', False), ('argmax_disagreements', 1)):
        changed = {**row(), name: value}
        assert not parity_gate(changed, CONFIG)


def test_replay_detects_modified_gradient_reduction_or_claim():
    assert replay_parity(row(), CONFIG)['passed']
    for key, value in (('squared_error', 2e-8), ('reference_squared_norm', 2.), ('max_absolute_error', 2e-4)):
        changed = deepcopy(row()); changed['parameters'][0][key] = value
        with pytest.raises(ValueError, match='differs'):
            replay_parity(changed, CONFIG)
    changed = {**row(), 'passed': False}
    with pytest.raises(ValueError, match='differs'):
        replay_parity(changed, CONFIG)
