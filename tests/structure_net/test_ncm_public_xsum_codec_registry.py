import copy
import pytest
from experiments.neural_architecture_lab import register_ncm_public_xsum_codec as registry


def completed():
    metric = dict(cases=256, exact_rendered=244, content_matches=990, content_tokens=1000,
        has_digits=20, digit_cases_preserved=19, invalid_outputs=0, empty=0, descriptive_reconstruction_gate=True)
    return dict(hypothesis_id=registry.HYPOTHESIS, outputs=672,
        counts={'representative': 256, 'long_stress': 32, 'digit_stress': 32, 'TRAIN_control': 16},
        all_raw_metrics_replayed=True, all_prior_TRAIN_controls_equal=True, pilot_full_equal=True,
        selection_replayed_from_163051_candidates=True, no_source_attention_or_generator=True,
        target_aware_diagnostic=True, model_training=False, new_judge_requests=0,
        final_task_goal_achieved=False, source_only_task_accuracy=None, training_data_admitted=False,
        final_test_opened=False, speed_tested=False,
        classification='repaired_public_TRAIN_codec_gate_passed_task_unproven',
        metrics={'representative': {'released': copy.deepcopy(metric), 'repaired': copy.deepcopy(metric)}})


def test_reconstruction_pass_never_becomes_task_accuracy():
    record = registry.record_for(completed())
    assert record['repaired_descriptive_reconstruction_gate']
    assert not record['final_task_accuracy_measured']
    assert not record['training_data_admitted']


@pytest.mark.parametrize('changes', [{'final_task_goal_achieved': True}, {'source_only_task_accuracy': .95},
    {'training_data_admitted': True}, {'final_test_opened': True}, {'speed_tested': True}, {'outputs': 671}])
def test_component_scope_and_completeness_are_mandatory(changes):
    with pytest.raises(ValueError): registry.record_for({**completed(), **changes})


def test_stale_passing_gate_and_classification_rejected():
    result = completed()
    result['metrics']['representative']['repaired']['exact_rendered'] = 243
    with pytest.raises(ValueError, match='gate differs'): registry.record_for(result)
    result['metrics']['representative']['repaired']['descriptive_reconstruction_gate'] = False
    with pytest.raises(ValueError, match='classification'): registry.record_for(result)
    result['classification'] = 'repaired_public_TRAIN_codec_reconstruction_gate_failed'
    assert not registry.record_for(result)['repaired_descriptive_reconstruction_gate']
