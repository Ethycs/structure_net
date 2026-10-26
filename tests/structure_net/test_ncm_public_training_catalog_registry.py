import pytest

from experiments.neural_architecture_lab.register_ncm_public_training_catalog import record_for, HYPOTHESIS


def result():
    value = {key: True for key in ('all_original_rows_replayed', 'all_candidate_tokens_recomputed',
        'all_public_development_text_overlaps_recomputed', 'all_byte_offsets_replayed',
        'all_remaining_pairs_retained', 'training_data_admitted')}
    value.update({key: False for key in ('strings_rewritten_or_truncated', 'label_factual_quality_verified',
        'model_training', 'final_test_opened', 'source_disjointness_established',
        'sealed_final_full_text_overlap_audit_completed', 'public_validation_or_test_payload_read')})
    return {**value, 'hypothesis_id': HYPOTHESIS, 'classification': 'full_public_paired_catalog_admitted_for_pretraining',
        'admission_scope': 'public_reference_conditional_pretraining_only', 'public_development_cases': 320,
        'protected_application_target_cases': 64, 'existing_TRAIN_target_cases': 1376,
        'new_judge_requests': 0, 'task_accuracy': None,
        'counts': {'original_rows': 204045, 'original_candidates': 163051, 'retained_pairs': 162353,
                   'additional_excluded_rows': 698, 'prior_screen_exclusions': 40994}}


def test_data_admission_is_not_promoted_to_quality_or_full_source_disjointness():
    row = record_for(result())
    assert row['training_data_admitted'] and not row['task_accuracy_measured']
    for key, value in (('task_accuracy', .95), ('model_training', True), ('label_factual_quality_verified', True),
                       ('source_disjointness_established', True), ('final_test_opened', True),
                       ('all_original_rows_replayed', False), ('public_development_cases', 319)):
        with pytest.raises(ValueError): record_for({**result(), key: value})


def test_incomplete_candidate_accounting_is_not_registered():
    r = result(); r['counts']['retained_pairs'] -= 1
    with pytest.raises(ValueError): record_for(r)
