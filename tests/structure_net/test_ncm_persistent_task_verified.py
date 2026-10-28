from copy import deepcopy

import pytest

from experiments.structure_net import ncm_persistent_task_verified as run


def test_policy_keeps_semantic_inputs_and_rejects_conflicting_bindings(monkeypatch):
    monkeypatch.setattr(run.data,'sha',lambda path:'measured:'+str(path))
    proof={'config':{'judges':{'secondary':'Terra','primary':'Sol'},'rubric':'fixed',
        'inherited_first_judgments':5651},'input_hashes':{'prior':'original'},'stage':'grade'}
    before=deepcopy(proof);bound=run.bind_policy(proof)
    assert proof==before and bound['stage']=='grade'
    assert all(bound['config'][k]==v for k,v in before['config'].items())
    assert bound['input_hashes']['prior']=='original'
    assert bound['config']['input_validation_policy']['first_access_full_SHA256'] is True
    assert bound['config']['input_validation_policy']['cross_process_cache'] is False
    with pytest.raises(ValueError):run.bind_policy({**proof,'input_hashes':{str(run.CPU):'conflicting'}})
