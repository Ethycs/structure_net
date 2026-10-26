"""Reject incomplete examples, bound full frames, and invalidate altered evidence."""
import json

import pytest

from experiments.structure_net import ncm_public_xsum_data as data


@pytest.mark.parametrize('source,target,native', [(192, 1, 1), (2048, 128, 126)])
def test_inclusive_application_and_native_limits(source, target, native):
    assert data.token_reasons(source, target, [200]*native, {0, 100, 101, 102}) == []


@pytest.mark.parametrize('source,target,native,reason', [
    (191, 20, 20, 'application_bypass_below192'),
    (2049, 20, 20, 'source_above2048'),
    (300, 0, 20, 'target_outside1to128'),
    (300, 129, 20, 'target_outside1to128'),
    (300, 20, 0, 'native_target_outside1to126'),
    (300, 20, 127, 'native_target_outside1to126'),
])
def test_excluded_ranges_do_not_get_truncated(source, target, native, reason):
    assert reason in data.token_reasons(source, target, [200]*native, {0, 100, 101, 102})


@pytest.mark.parametrize('special', [0, 100, 101, 102])
def test_special_or_unknown_cannot_be_training_text(special):
    assert data.token_reasons(300, 3, [200, special, 201], {0, 100, 101, 102}) == ['native_special_or_unknown']


@pytest.mark.parametrize('changes', [
    {'summary': ''}, {'summary': None}, {'document': '  '}, {'id': '１２３'},
    {'id': 123}, {'id': 'bbc:123'}, {'split': 'test'},
])
def test_incomplete_or_wrong_schema_rejected(changes):
    row = {'document': 'Original source.', 'summary': 'Full summary.', 'id': '123'}
    with pytest.raises(ValueError):
        data.validate_row({**row, **changes})


def test_completion_receipt_is_necessary_and_does_not_hide_payload_mutation(tmp_path):
    payload = tmp_path/'rows.jsonl'
    payload.write_text('original')
    data.write(tmp_path/'result.json', {'artifacts': {str(payload): data.sha(payload)}})
    with pytest.raises(FileNotFoundError):
        data.completed(tmp_path)
    data.write(tmp_path/'actual_tool_receipt.json', {'actual_tool_exit_code': 0})
    assert data.completed(tmp_path)['artifacts'][str(payload)] == data.sha(payload)
    payload.write_text('changed')
    with pytest.raises(ValueError, match='input changed'):
        data.completed(tmp_path)


def test_failed_exit_cannot_be_reused(tmp_path):
    data.write(tmp_path/'actual_tool_receipt.json', {'actual_tool_exit_code': 1})
    with pytest.raises(ValueError, match='actual successful'):
        data.completed(tmp_path)


def test_append_only_evidence(tmp_path):
    path = tmp_path/'result.json'
    data.write(path, {'value': 1})
    with pytest.raises(FileExistsError):
        data.write(path, {'value': 2})
    assert json.loads(path.read_text()) == {'value': 1}
