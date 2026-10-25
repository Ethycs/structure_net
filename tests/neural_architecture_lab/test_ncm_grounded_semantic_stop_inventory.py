import json

import pytest

from experiments.neural_architecture_lab import inventory_ncm_grounded_semantic_stop as audit


def save_attempt(root, index, valid=False):
    packet = {'case_id': 'example', 'source_lines': [], 'candidates': []}
    body = {'model': 'teacher', 'temperature': 0, 'max_tokens': 4096,
        'messages': [{'role': 'system', 'content': 'rubric'},
            {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]}
    raw = {'attempt': index, 'request': body, 'request_sha256': audit.run.q.digest(body), 'valid': valid}
    if valid:
        raw.update(http_status=200, response={'model': 'teacher', 'choices': [
            {'message': {'content': '[{"accepted": true}]'}}]})
    path = root / 'attempts' / f'example_{index}.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(raw))
    return packet, path


def reconstruct(root, packet=None):
    packet = packet or {'case_id': 'example', 'source_lines': [], 'candidates': []}
    return audit.reconstruct(packet, 'secondary', root, 'teacher', 'rubric', lambda text, _: json.loads(text))


def test_empty_and_partially_saved_folders_keep_consumed_allowance(tmp_path):
    row, state = reconstruct(tmp_path)
    assert row is None and state['maximum_future_attempts'] == 2
    packet, _ = save_attempt(tmp_path, 0)
    row, state = reconstruct(tmp_path, packet)
    assert row is None and state['maximum_future_attempts'] == 1
    save_attempt(tmp_path, 1)
    row, state = reconstruct(tmp_path, packet)
    assert row is None and state['maximum_future_attempts'] == 0


def test_complete_exhausted_raw_request_recovers_even_without_row_file(tmp_path):
    for i in range(3):
        packet, _ = save_attempt(tmp_path, i)
    row, state = reconstruct(tmp_path, packet)
    assert row['status'] == 'invalid' and len(row['attempts']) == 3
    assert 'evaluations' not in row and state['maximum_future_attempts'] == 0


def test_first_valid_reply_and_no_later_retry(tmp_path):
    packet, _ = save_attempt(tmp_path, 0, True)
    row, _ = reconstruct(tmp_path, packet)
    assert row['evaluations'] == [{'accepted': True}]
    save_attempt(tmp_path, 1)
    with pytest.raises(ValueError, match='no later attempt'):
        reconstruct(tmp_path, packet)


def test_forged_request_hash_or_validity_is_rejected(tmp_path):
    packet, path = save_attempt(tmp_path, 0, True)
    original = json.loads(path.read_text())
    raw = {**original, 'request_sha256': 'forged'}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='binding'):
        reconstruct(tmp_path, packet)
    original['valid'] = False
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='raw validity'):
        reconstruct(tmp_path, packet)
