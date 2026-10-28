from copy import deepcopy
import json

import pytest
import requests

from experiments.structure_net import ncm_isolated_judge_transport as local


def packet():
    return {'case_id': 'control', 'source_lines': [{'line_id': 'L0', 'text': 'The count is 7.'}],
            'candidates': [{'candidate_id': 'summary', 'text': 'The count is 8.'}]}


def wire(model='gpt-5.6-terra'):
    grade = {'candidate_id': 'summary', 'faithful': False, 'salient': False,
             'readable': True, 'claims': [], 'reason': 'The number is wrong.'}
    return {'model': model, 'choices': [{'message': {'content': json.dumps({'evaluations': [grade]})}}]}


class Response:
    status_code = 200
    def __init__(self, body): self.body = body
    def json(self): return deepcopy(self.body)
    def raise_for_status(self): pass


def test_valid_negative_is_terminal_and_wire_alias_is_preserved(tmp_path):
    (tmp_path/'attempts').mkdir(); calls = []
    def post(*args, **kwargs): calls.append(kwargs); return Response(wire())
    row = local.request(packet(), 'secondary', tmp_path, post=post, key='unit-test-only')
    assert len(calls) == len(row['attempts']) == 1 and row['status'] == 'valid'
    assert row['evaluations'][0]['faithful'] is False
    raw = local.q.read(row['attempts'][0])
    assert raw['wire_response']['model'] == 'gpt-5.6-terra'
    assert raw['response']['model'] == local.MODELS['secondary']
    assert 'unit-test-only' not in json.dumps(raw)
    local.replay_row(packet(), 'secondary', tmp_path, row)
    with pytest.raises(ValueError, match='overwrite'):
        local.request(packet(), 'secondary', tmp_path, post=post, key='unit-test-only')


def test_only_transport_or_schema_failure_can_retry(tmp_path):
    (tmp_path/'attempts').mkdir(); calls = []
    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1: raise requests.ReadTimeout()
        return Response(wire())
    row = local.request(packet(), 'secondary', tmp_path, post=post, key='unit-test-only')
    assert len(row['attempts']) == len(calls) == 2
    local.replay_row(packet(), 'secondary', tmp_path, row)


def test_exhausted_invalid_requests_and_wrong_model_cannot_qualify(tmp_path):
    (tmp_path/'attempts').mkdir()
    row = local.request(packet(), 'secondary', tmp_path,
        post=lambda *a, **k: Response(wire('gpt-5.6-sol')), key='unit-test-only')
    assert row['status'] == 'invalid' and len(row['attempts']) == 3
    local.replay_row(packet(), 'secondary', tmp_path, row)
    with pytest.raises(ValueError): local.normalize(wire('another-model'), local.MODELS['secondary'])


def test_raw_response_mapping_and_decisions_cannot_be_rewritten(tmp_path):
    (tmp_path/'attempts').mkdir()
    row = local.request(packet(), 'secondary', tmp_path,
        post=lambda *a, **k: Response(wire()), key='unit-test-only')
    path = tmp_path/'attempts/control_0.json'; raw = local.q.read(path)
    raw['response']['choices'][0]['message']['content'] = '{}'
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match='wire response'):
        local.replay_row(packet(), 'secondary', tmp_path, row)


def test_isolated_backend_deadline_precedes_client_deadline(tmp_path):
    (tmp_path/'attempts').mkdir(); calls = []
    def post(url, **kwargs): calls.append((url, kwargs)); return Response(wire())
    row = local.request(packet(), 'secondary', tmp_path, post=post, key='unit-test-only')
    url, request = calls[0]
    assert url == 'http://127.0.0.1:8092/v1/chat/completions'
    assert request['json']['timeout'] == 160 < request['timeout'] == 180
    assert request['json']['messages'][0]['content'] == local.previous.profile.RUBRIC
    local.replay_row(packet(), 'secondary', tmp_path, row)
