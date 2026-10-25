import json

import pytest

from experiments.neural_architecture_lab import audit_ncm_sol_teacher144_partial as audit


class Encoding:
    def encode(self, text, disallowed_special=()):
        return list(text.encode())


def fixture(tmp_path, count=1, success=True):
    row = {'id': 'source', 'source': 'The source describes a frozen transformer.'}
    models = {'draft': 'fixed-teacher'}
    systems = {'draft': 'Frozen instructions'}
    lines = audit.run.previous.original.source_lines(row)
    value = {'summary': 'The passage describes a frozen transformer.',
        'support': [{'summary_span': 'a frozen transformer', 'source_lines': [lines[0]['line_id']]}]}
    body = {'model': models['draft'], 'temperature': 0, 'max_tokens': 768,
        'messages': [{'role': 'system', 'content': systems['draft']},
            {'role': 'user', 'content': json.dumps({'source_lines': lines}, ensure_ascii=False)}]}
    paths = []
    for i in range(count):
        raw = {'request': body, 'prompt_tokens': len(json.dumps(body['messages'], ensure_ascii=False).encode()) + 16,
            'valid': success, 'error_type': 'ReadTimeout'}
        if success:
            raw.update(http_status=200, response={'model': 'fixed-teacher', 'choices': [
                {'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]})
        path = tmp_path / f'attempt_{i}.json'
        path.write_text(json.dumps(raw))
        paths.append(str(path))
    observation = {'status': 'valid' if success else 'invalid', 'value': value if success else None, 'attempts': paths}
    return row, observation, models, systems, value


def test_first_valid_and_exhausted_failures_preserved(tmp_path):
    row, obs, models, systems, value = fixture(tmp_path)
    assert audit.replay_stage(row, 'draft', obs, Encoding(), models, systems) == value
    row, obs, models, systems, _ = fixture(tmp_path, 3, False)
    assert audit.replay_stage(row, 'draft', obs, Encoding(), models, systems) is None
    obs['attempts'] = obs['attempts'][:2]
    with pytest.raises(ValueError, match='exhausted allowance'):
        audit.replay_stage(row, 'draft', obs, Encoding(), models, systems)


@pytest.mark.parametrize('tamper', ['request', 'model', 'finish', 'valid'])
def test_tampering_is_rejected(tmp_path, tamper):
    row, obs, models, systems, _ = fixture(tmp_path)
    path = tmp_path / 'attempt_0.json'
    raw = json.loads(path.read_text())
    if tamper == 'request':
        raw['request']['messages'][1]['content'] = 'changed source'
    elif tamper == 'model':
        raw['response']['model'] = 'replacement-teacher'
    elif tamper == 'finish':
        raw['response']['choices'][0]['finish_reason'] = 'length'
    else:
        raw['valid'] = False
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        audit.replay_stage(row, 'draft', obs, Encoding(), models, systems)


def test_no_attempt_after_valid_first(tmp_path):
    row, obs, models, systems, _ = fixture(tmp_path, 2)
    with pytest.raises(ValueError, match='stop at the first valid'):
        audit.replay_stage(row, 'draft', obs, Encoding(), models, systems)
