from copy import deepcopy
import json

import pytest

from experiments.structure_net import ncm_local_task_development as local
from tests.structure_net.test_ncm_local_judge_transport import packet, wire, Response


def test_both_exact_judges_require_complete_local_control_admission():
    proof = {'config': {'endpoint': local.transport.ENDPOINT, 'transport_identity': local.transport.IDENTITY,
                       'rubric': local.qualification.previous.profile.RUBRIC}}
    for role, model in local.transport.MODELS.items():
        result = dict(status='local_judge_wire_firsts_and_controls_replayed', judge=model,
                      qualification_passed=True, controls=50, correct=50, cases=13, valid_requests=13)
        local.check_role(role, result, proof)
        for bad in (dict(result, correct=49), dict(result, valid_requests=12), dict(result, judge='other'),
                    dict(result, qualification_passed=False)):
            with pytest.raises(ValueError): local.check_role(role, bad, proof)


def test_proof_preserves_rubric_models_and_historical_first_count():
    proof = {'stage': 'grade', 'input_hashes': {'prior': 'digest'}, 'config': {
        'judges': deepcopy(local.transport.MODELS), 'rubric': local.qualification.previous.profile.RUBRIC,
        'inherited_first_judgments': 5647}}
    before = deepcopy(proof)
    new = local.augment_proof(proof, {'local_admission': 'verified'})
    assert new['config']['inherited_first_judgments'] == 5647
    assert new['config']['judges'] == before['config']['judges']
    assert new['config']['rubric'] == before['config']['rubric']
    assert new['input_hashes'] == {'prior': 'digest', 'local_admission': 'verified'}
    assert new['config']['judge_endpoint'] == local.transport.ENDPOINT
    with pytest.raises(ValueError): local.augment_proof(dict(proof, stage='outputs'), {})


def test_local_raw_audit_preserves_old_firsts_and_rejects_wire_tampering(tmp_path):
    root = tmp_path/'new_grade'; origin = root/'requests/secondary/key'; (origin/'attempts').mkdir(parents=True)
    row = local.transport.request(packet(), 'secondary', origin,
        post=lambda *a, **k: Response(wire()), key='unit-test-only')
    from experiments.structure_net.ncm_source_readout_grade import audit_row
    audit_row(packet(), 'secondary', row)
    cache = {'old': {'origin': str(tmp_path/'old_grade/requests/key'), 'preserved': 'historical'},
             'new': {'origin': str(origin), 'packet': packet(), 'role': 'secondary', 'row': row}}
    before = deepcopy(cache)
    assert local.audit_local_entries(cache, [root]) == 1 and cache == before
    path = origin/'attempts/control_0.json'; raw = local.q.read(path)
    raw['wire_response']['model'] = 'different-model'; path.write_text(json.dumps(raw))
    with pytest.raises(ValueError): local.audit_local_entries(cache, [root])
