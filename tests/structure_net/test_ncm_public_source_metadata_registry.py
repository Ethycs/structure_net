import pytest

from experiments.neural_architecture_lab.register_ncm_public_source_metadata import record_for, HYPOTHESIS


def test_incomplete_replay_and_quality_claims_are_rejected():
    row = {k: True for k in ('all_records_replayed', 'all_tokenization_and_alignment_recomputed',
                            'original_TRAIN_geometry_equal', 'full_metadata_audited')}
    row.update({k: False for k in ('model_training', 'body_loaded', 'features_computed',
        'target_fields_in_body_input', 'final_test_opened', 'independent_tokenization_implementation')})
    row.update(hypothesis_id=HYPOTHESIS, status='all_public_full_source_metadata_replayed', cases=162353,
        source_positions=91344717, native_prompt_positions=125616400, max_source_positions=4124,
        max_native_prompt_positions=2275, pilot_repetitions_exact=9, new_judge_requests=0, task_accuracy=None)
    assert record_for(row)['full_metadata_audited']
    for key, value in (('cases', 162352), ('all_records_replayed', False), ('pilot_repetitions_exact', 8),
            ('model_training', True), ('features_computed', True), ('task_accuracy', .95), ('final_test_opened', True)):
        with pytest.raises(ValueError): record_for({**row, key: value})
