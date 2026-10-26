import copy
import hashlib
import json

import pytest

from experiments.structure_net import ncm_public_training_bundle as bundle


def stages():
    common = {'cases': 162353, 'source_positions': 91344717, 'model_training': False,
              'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False}
    return [
        {**common, 'status': 'all_public_full_source_metadata_replayed', 'full_metadata_audited': True, 'all_records_replayed': True},
        {**common, 'status': 'public_feature_store_full_completed', 'body_sha256': bundle.BODY_SHA, 'all_encoded_roundtrips_exact': True, 'prior_raw_hash_controls': 16},
        {**common, 'status': 'public_feature_store_full_disk_replay_passed', 'all_disk_records_replayed': True, 'all_shards_completely_covered': True},
        {**common, 'status': 'public_code_targets_full_completed', 'codec_sha256': bundle.CODEC_SHA, 'codec_unchanged': True, 'all_repeated_encodes_exact': True,
         'pilot_repetitions_exact': 16, 'original_TRAIN_controls_exact': 16},
        {**common, 'status': 'all_public_full_target_bindings_replayed', 'all_bindings_replayed': True, 'targets_admitted_for_training': True}]


def test_every_full_stage_and_identity_is_required():
    bundle.validate_stages(*stages())
    for index, key, value in ((0, 'cases', 162352), (1, 'body_sha256', 'wrong'),
            (2, 'all_disk_records_replayed', False), (3, 'codec_sha256', 'wrong'),
            (3, 'pilot_repetitions_exact', 15), (4, 'targets_admitted_for_training', False),
            (4, 'task_accuracy', .95), (1, 'source_positions', 91344716)):
        items = copy.deepcopy(stages()); items[index][key] = value
        with pytest.raises(ValueError): bundle.validate_stages(*items)


def test_audit_receipt_must_bind_exact_successful_producer(tmp_path):
    def write(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()
    result_path, result_hash = write('result.json', {'cases': 162353})
    audit_path, audit_hash = write('audit_result.json', {'artifacts': {result_path: result_hash}})
    write('actual_tool_receipt.json', {'actual_tool_exit_code': 0, 'artifacts': {result_path: result_hash}})
    write('audit_actual_tool_receipt.json', {'actual_tool_exit_code': 0, 'artifacts': {audit_path: audit_hash}})
    assert bundle.completed_with_audit(tmp_path)[0]['cases'] == 162353
    write('result.json', {'cases': 16})
    with pytest.raises(ValueError): bundle.completed_with_audit(tmp_path)
