from copy import deepcopy

import pytest

from experiments.neural_architecture_lab import finalize_ncm_depth as final


def arm(passed, joint=None, valid=128):
    joints = {str(i): {'cases': 64, 'joint_passed': n, 'all_valid': True, 'qualified': n >= 61}
              for i, n in enumerate(joint)} if joint else None
    classification = final.classify(valid, passed, joints is not None, joints)
    return {'valid': valid, 'passed': passed, 'primary_admitted': joints is not None, 'joint': joints,
            'classification': classification, 'DEV_qualified': classification == 'DEV_qualified',
            'task_accuracy': passed/128 if valid == 128 else None,
            'joint_task_accuracy': sum(joint)/128 if joint else None}


@pytest.mark.parametrize('left,right,expected', [
    (arm(0), arm(1), None),
    (arm(0), arm(122, [60, 62]), None),
    (arm(122, [61, 61]), arm(122, [61, 61]), '12'),
    (arm(124, [62, 62]), arm(125, [62, 63]), '24'),
    (arm(126, [63, 63]), arm(124, [62, 62]), '12'),
    (arm(0, valid=127), arm(122, [61, 61]), '24'),
])
def test_selects_only_qualified_arms_and_smaller_depth_on_exact_tie(left, right, expected):
    assert final.selected_candidate({'12': left, '24': right}) == expected


@pytest.mark.parametrize('change', ['gate', 'accuracy', 'semantic_accuracy', 'missing'])
def test_rejects_unearned_candidate(change):
    values = deepcopy({'12': arm(122, [61, 61]), '24': arm(0)})
    if change == 'gate': values['24']['DEV_qualified'] = True
    elif change == 'accuracy': values['12']['joint_task_accuracy'] = 1.
    elif change == 'semantic_accuracy': values['24']['task_accuracy'] = .95
    else: del values['12']
    with pytest.raises(ValueError): final.selected_candidate(values)


def test_invalid_transport_has_no_task_accuracy_or_reduced_denominator():
    values = {'12': arm(0), '24': arm(100, valid=127)}
    assert final.comparison_classification(values) == 'DEV_comparison_transport_invalid'
    values['24']['task_accuracy'] = 100/127
    with pytest.raises(ValueError): final.selected_candidate(values)


def test_registry_retains_failure_and_rejects_unmeasured_final_success():
    from experiments.neural_architecture_lab import register_ncm_depth as registry
    arms = {}
    for depth in ('12', '24'):
        arms[depth] = {**arm(0), 'depth': int(depth), 'cases': 128,
            'training_audit': {'updates': 1024, 'distinct_TRAIN_cases': 1376},
            'final_test_opened': False, 'speed_tested': False, 'trainable_parameters': 123}
    value = {'hypothesis_id': final.HYPOTHESIS, 'cases_per_arm': 128, 'independent_training_seeds': 1,
        'inherited_first_judgments': 4179, 'canonical_first_judgments': 4435, 'same_sampling': True,
        'same_parent_checkpoint': final.study.parent.PARENT_SHA, 'identity_initialized_extra_blocks': True,
        'source_only': True, 'all_historical_and_pilot_firsts_preserved': True, 'final_test_opened': False,
        'speed_tested': False, 'final_task_goal_achieved': False, 'selected_candidate': None,
        'classification': 'DEV_both_semantic_quality_failed', 'comparison_has_unresolved_judgments': False,
        'arms': arms}
    record = registry.record_for(value)
    assert record['depth12_task_accuracy'] == record['depth24_task_accuracy'] == 0.
    assert not record['final_task_accuracy_measured']
    value['final_task_goal_achieved'] = True
    with pytest.raises(ValueError): registry.record_for(value)


def test_pairing_never_turns_invalid_judgments_into_accuracy():
    def rows(values):
        return [{'id': str(i), 'valid': v is not None, 'passed': v} for i, v in enumerate(values)]
    value = final.paired_decisions(rows([True, False, None]), rows([False, True, True]))
    assert value['control_only'] == value['deeper_only'] == value['unresolved'] == 1
    assert value['accuracy_difference_depth24_minus_depth12'] is None
    valid = final.paired_decisions(rows([False, True, False]), rows([True, True, False]))
    assert valid['accuracy_difference_depth24_minus_depth12'] == 1/3
    with pytest.raises(ValueError): final.paired_decisions(rows([False, True]), rows([True]))
