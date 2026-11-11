import copy
import io
import json
import threading
import urllib.error
import urllib.request

import pytest

from experiments.word_games.bridges import service as module
from experiments.word_games.bridges.client import Client, InvalidGeneration
from experiments.word_games.bridges.mapping import unique_table
from experiments.word_games.bridges.mapping import ROOT_DEPLOYMENT
from experiments.word_games.bridges.server import make_server
from experiments.word_games.bridges.vendor import capsule
from experiments.word_games.bridges.vendor.runtime import KEYS, ENUMS, PREFIXES


def test_mapping():
    return {'schema_version': 1,
            'key_map': {k: 'k_' + k for k in KEYS},
            'enum_map': {k: 'e_' + k for k in ENUMS},
            'id_prefixes': {k: 'p_' + k for k in PREFIXES},
            'target': {'name': 'Logistics', 'identity': 'Planner',
                       'objective': 'Arrange packages', 'rules': 'Use the provided actions.'}}


test_mapping.__test__ = False


class FakeClient:
    def caller(self, key):
        if key != 'caller-key':
            raise PermissionError()
        return 'test-caller'

    def generate(self, stage, prompt, record, max_tokens=1024):
        assert stage == 'goal'
        record({'stage': stage, 'prompt': prompt})
        return {'goal': prompt.split('\nGOAL\n', 1)[1]}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'generate_mapping', lambda *args: test_mapping())
    service = module.Service(tmp_path / 'bridges.sqlite')
    client = FakeClient()
    created = service.create(client, 'test-caller', {})
    yield service, client, created
    service.db.close()


def prepare(service, client, created):
    sid = created['session_id']
    return service.mutate(sid, created['doer_capability'], 'doer', 'prepare', {
        'request_id': 'prepare', 'version': 0,
        'goal': 'Reserve loan:bronze in gal:north N1 until 2027-01-10.',
        'context': {'scope': ['gal:north'], 'loan_ids': ['loan:bronze'],
                    'allowed_ops': ['reserve'], 'event_id': 'source-secret-event'}}, client)


def target_proposal(service, created):
    session = service.get(created['session_id'])
    mapper = capsule.mapping(session['capsule']['task_mapping'])
    privacy = module.Privacy(session['aliases'])
    return mapper.encode(privacy.shield({'operations': [{
        'op': 'reserve', 'loan_id': 'loan:bronze', 'gallery_id': 'gal:north',
        'slot': 'N1', 'due': '2027-01-10', 'approval_id': None}]}))


def test_controller_view_isolated_and_no_commit_authority(setup):
    service, client, created = setup
    sid = created['session_id']; prepare(service, client, created)
    public = service.inspect(sid, created['controller_capability'], 'controller')
    text = json.dumps(public)
    for secret in ['source-secret-event', 'Bronze Heron', 'loan:bronze', 'gal:north',
                   'source_goal', 'key_map', 'permissions', created['doer_capability']]:
        assert secret not in text
    with pytest.raises(PermissionError):
        service.inspect(sid, created['controller_capability'], 'doer')
    proposal = target_proposal(service, created)
    service.mutate(sid, created['controller_capability'], 'controller', 'submit', {
        'request_id': 'proposal', 'version': 1, 'proposal': proposal}, client)
    assert service.get(sid)['engine']['snapshot']['revision'] == 0
    with pytest.raises(PermissionError):
        service.mutate(sid, created['controller_capability'], 'controller', 'apply', {
            'request_id': 'attack', 'version': 2}, client)
    result = service.mutate(sid, created['doer_capability'], 'doer', 'apply', {
        'request_id': 'commit', 'version': 2}, client)
    assert result['result']['status'] == 'committed'
    assert service.get(sid)['engine']['snapshot']['loans']['loan:bronze']['reservation']['slot'] == 'N1'


def test_idempotency_stale_revision_and_restart(setup):
    service, client, created = setup
    sid = created['session_id']; prepare(service, client, created)
    proposal = target_proposal(service, created)
    body = {'request_id': 'p', 'version': 1, 'proposal': proposal}
    first = service.mutate(sid, created['controller_capability'], 'controller', 'submit', body, client)
    assert service.mutate(sid, created['controller_capability'], 'controller', 'submit', body, client) == first
    with pytest.raises(module.Conflict):
        service.mutate(sid, created['controller_capability'], 'controller', 'submit',
                       {**body, 'version': 2}, client)
    resumed = module.Service(service.path)
    result = resumed.mutate(sid, created['doer_capability'], 'doer', 'apply', {
        'request_id': 'commit', 'version': 2}, client)
    assert result['result']['revision'] == 1
    assert resumed.mutate(sid, created['doer_capability'], 'doer', 'apply', {
        'request_id': 'commit', 'version': 2}, client) == result
    assert resumed.get(sid)['engine']['snapshot']['revision'] == 1
    resumed.db.close()


def test_rejected_proposal_preserves_committed_state(setup):
    service, client, created = setup
    sid = created['session_id']; prepare(service, client, created)
    before = service.get(sid)
    proposal = target_proposal(service, created)
    proposal['k_operations'][0]['k_due'] = '2030-01-01'
    with pytest.raises(ValueError):
        service.mutate(sid, created['controller_capability'], 'controller', 'submit', {
            'request_id': 'bad', 'version': 1, 'proposal': proposal}, client)
    assert service.get(sid) == before


def test_literal_quote_handles_roundtrip():
    privacy = module.Privacy({})
    goal = privacy.goal('Append "private source text" to loan:bronze.')
    assert 'private source text' not in goal
    assert privacy.unshield(privacy.shield({'text': 'private source text'})) == {'text': 'private source text'}


def test_capsule_retranslation_preserves_checkpoint_and_rejects_mutation(setup):
    service, _, created = setup
    parent = service.get(created['session_id'])['capsule']
    child = copy.deepcopy(parent)
    child['generation'] = 1
    child['task_mapping']['target']['name'] = 'New domain'
    child['bridge_state']['active_domain'] = 'New domain'
    child['deployment']['operating_instructions'] = 'New translated instructions'
    assert capsule.validate(child, parent)['valid']
    child['carried_state']['k_revision'] += 1
    with pytest.raises(ValueError, match='carried_state_changed'):
        capsule.validate(child, parent)


def test_http_auth_role_separation_and_health(setup):
    service, client, created = setup
    server = make_server(('127.0.0.1', 0), service, lambda key: client)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    base = 'http://127.0.0.1:' + str(server.server_port)
    try:
        with urllib.request.urlopen(base + '/health') as response:
            assert json.load(response)['model_readiness'] == 'not_checked_no_startup_load'
        path = base + '/v1/sessions/' + created['session_id'] + '/doer'
        req = urllib.request.Request(path, headers={'Authorization': 'Bearer caller-key',
            'X-Bridge-Capability': created['controller_capability']})
        with pytest.raises(urllib.error.HTTPError) as ex:
            urllib.request.urlopen(req)
        assert ex.value.code == 403
        req = urllib.request.Request(path, headers={'Authorization': 'Bearer wrong',
            'X-Bridge-Capability': created['doer_capability']})
        with pytest.raises(urllib.error.HTTPError) as ex:
            urllib.request.urlopen(req)
        assert ex.value.code == 403
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_capsules_and_private_receipts_not_exposed_to_controller(setup):
    service, _, created = setup
    public = service.inspect(created['session_id'], created['controller_capability'], 'controller')
    assert set(public) == {'session_id', 'version', 'generation', 'manual', 'task', 'proposal', 'result'}


def test_context_scope_and_authorization_survive_restart(setup):
    service, client, created = setup
    sid = created['session_id']; prepare(service, client, created)
    engine = module.restore(service.get(sid)['engine'])
    context = service.get(sid)['pending']['context']
    assert engine.contexts[context['context_id']]['scope'] == ['gal:north']
    assert engine._context_ids[context['context_id']] == {'loan:bronze'}


@pytest.mark.parametrize('finish,model,deployment', [
    ('length', 'qwen3.8-35b-a3b', 'expected'),
    ('stop', 'different-model', 'expected'),
    ('stop', 'qwen3.8-35b-a3b', 'different-deployment')])
def test_generation_rejects_truncation_or_wrong_route_and_keeps_attempt(monkeypatch, finish, model, deployment):
    body = {'model': model, 'choices': [{'finish_reason': finish,
                                       'message': {'content': '{"ok":true}'}}]}
    class Response(io.BytesIO):
        headers = {'X-Litellm-Model-Id': deployment}
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: Response(json.dumps(body).encode()))
    client = Client(lambda: 'private-inference-key', deployment_id='expected')
    attempts = []
    with pytest.raises(InvalidGeneration):
        client.generate('test', 'Prompt', attempts.append)
    assert len(attempts) == 1
    assert 'private-inference-key' not in json.dumps(attempts)
    assert attempts[0]['raw_response'] == json.dumps(body)


def test_table_retries_bounded_and_invalid_output_retained():
    class RepeatingClient:
        count = 0
        def generate(self, *args, **kwargs):
            self.count += 1
            return {'a': 'duplicate', 'b': 'duplicate'}
    client = RepeatingClient(); records = []
    with pytest.raises(ValueError):
        unique_table(client, 'keys', 'prompt', records.append, ['a', 'b'])
    assert client.count == 2
    assert len(records) == 2
    assert all(row['duplicates'] == ['duplicate'] for row in records)


def test_live_controller_retry_validates_before_commit(setup):
    service, client, created = setup
    sid = created['session_id']; prepare(service, client, created)
    good = target_proposal(service, created)
    bad = copy.deepcopy(good)
    bad['k_operations'][0]['k_due'] = '2030-01-01'
    class RetryClient:
        count = 0
        def generate(self, *args, **kwargs):
            self.count += 1
            return bad if self.count == 1 else good
    retry_client = RetryClient()
    result = service.mutate(sid, created['controller_capability'], 'controller', 'solve',
                            {'request_id': 'solve', 'version': 1}, retry_client)
    assert result['proposal'] == good and retry_client.count == 2
    assert service.get(sid)['engine']['snapshot']['revision'] == 0
