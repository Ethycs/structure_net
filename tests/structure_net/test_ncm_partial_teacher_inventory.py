import json

import pytest

from experiments.structure_net import ncm_partial_teacher_inventory as run


def write(tmp_path, index, valid):
    folder = tmp_path / 'requests' / 'case' / 'draft'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'attempt_{index}.json').write_text(json.dumps({'valid': valid,
        'response': {'choices': [{'message': {'content': 'first'}}]}}))


def test_first_valid_observation_and_exhausted_failure_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(run.run.original, 'parse', lambda text, row: {'summary': text})
    write(tmp_path, 0, False)
    assert run.descriptor({'id': 'case'}, 'draft', tmp_path) is None
    write(tmp_path, 1, False)
    write(tmp_path, 2, True)
    result = run.descriptor({'id': 'case'}, 'draft', tmp_path)
    assert result['value'] == {'summary': 'first'} and len(result['attempts']) == 3
    write(tmp_path, 2, False)
    result = run.descriptor({'id': 'case'}, 'draft', tmp_path)
    assert result['status'] == 'invalid' and result['value'] is None


def test_later_reply_cannot_replace_earlier_valid_first(tmp_path, monkeypatch):
    monkeypatch.setattr(run.run.original, 'parse', lambda text, row: {'summary': text})
    write(tmp_path, 0, True)
    write(tmp_path, 1, False)
    with pytest.raises(ValueError, match='later attempt'):
        run.descriptor({'id': 'case'}, 'draft', tmp_path)
