import json
import time

import pytest

from experiments.structure_net import ncm_latest_early_audit as audit


def test_no_audit_before_actual_grade_receipt(tmp_path):
    (tmp_path/'result.json').write_text('{}')
    assert not audit.grade_ready(tmp_path, audit.OriginalProofs())


def test_quality_failure_is_a_completed_grade(tmp_path):
    proofs = audit.OriginalProofs()
    result = tmp_path/'result.json'; result.write_text('{"status":"DEV_quality_failed"}')
    (tmp_path/'actual_tool_receipt.json').write_text(json.dumps({
        'actual_tool_exit_code': 0, 'artifacts': {str(result): proofs.sha(result)}}))
    assert audit.grade_ready(tmp_path, proofs)


def test_failed_or_mutated_grade_is_rejected(tmp_path):
    proofs = audit.OriginalProofs()
    result = tmp_path/'result.json'; result.write_text('{}')
    receipt = tmp_path/'actual_tool_receipt.json'
    value = {'actual_tool_exit_code': 1, 'artifacts': {str(result): proofs.sha(result)}}
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError): audit.grade_ready(tmp_path, proofs)
    value['actual_tool_exit_code'] = 0; receipt.write_text(json.dumps(value))
    result.write_text('{"changed":true}')
    with pytest.raises(ValueError): audit.grade_ready(tmp_path, proofs)
    receipt.unlink(); (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError): audit.grade_ready(tmp_path, proofs)


def test_only_own_live_hashes_share_and_changes_are_detected(tmp_path, monkeypatch):
    audit.MEASURED.clear()
    from experiments.structure_net import ncm_recovered_training_inputs as original
    clock = time.time_ns(); monkeypatch.setattr(original.time, 'time_ns', lambda: clock+10_000_000_000)
    first, second = audit.AuditProofs(), audit.AuditProofs()
    assert not first.cache and first.inputs is not second.inputs
    path = tmp_path/'large'; path.write_bytes(b'a'*(1024**2))
    digest = first.bind(path)
    assert second.sha(path) == digest and second.cache is first.cache
    path.write_bytes(b'b'*(1024**2))
    assert second.sha(path) != digest
    assert not audit.OriginalProofs().cache
    audit.MEASURED.clear()


def test_audit_binding_preserves_original_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.latest.data, 'sha', lambda p: 'measured')
    value = {'config': {'metric': 'complete_summary_semantic_pass'}, 'input_hashes': {'source': 'fixed'}}
    bound = audit.bind_audit(value, tmp_path)
    assert bound['config']['metric'] == value['config']['metric']
    assert bound['input_hashes']['source'] == 'fixed'
    assert not bound['config']['independent_audit_admission']['hashes_received_from_other_processes']
    assert value == {'config': {'metric': 'complete_summary_semantic_pass'}, 'input_hashes': {'source': 'fixed'}}
