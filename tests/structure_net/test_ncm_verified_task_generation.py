from copy import deepcopy

import pytest

from experiments.structure_net import ncm_verified_task_generation as run


def test_generation_policy_preserves_model_source_split_and_semantic_contract(monkeypatch):
    monkeypatch.setattr(run.data, 'sha', lambda p: 'measured:'+str(p))
    proof = {'stage': 'outputs', 'arm': 'expanded',
             'config': {'source_only': True, 'target_fields_in_prediction': False,
                        'DEV_cases': 128, 'rubric': 'unchanged', 'final_test_opened': False},
             'input_hashes': {'checkpoint': 'fixed', 'split': 'fixed'}}
    original = deepcopy(proof)
    bound = run.bind_policy(proof)
    assert proof == original
    assert bound['stage'] == 'outputs' and bound['arm'] == 'expanded'
    assert all(bound['config'][k] == v for k, v in proof['config'].items())
    assert all(bound['input_hashes'][k] == v for k, v in proof['input_hashes'].items())
    assert bound['config']['input_validation_policy']['cross_process_cache'] is False
    assert bound['config']['input_validation_policy']['first_access_full_SHA256'] is True
    with pytest.raises(ValueError, match='conflicts'):
        run.bind_policy({**proof, 'input_hashes': {str(run.CPU): 'changed'}})
