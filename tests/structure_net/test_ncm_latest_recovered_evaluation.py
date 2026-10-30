from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_latest_recovered_evaluation as latest


def test_started_branches_cannot_disappear_from_history(tmp_path):
    paths = (tmp_path/'tree', tmp_path/'old_recovered')
    latest.require_unstarted(paths)
    paths[1].mkdir()
    with pytest.raises(ValueError, match='already started'):
        latest.require_unstarted(paths)


def test_complete_task_history_is_retained_without_claiming_tree_outcomes(monkeypatch):
    events = []
    judgments = {str(i): i for i in range(5903)}
    ancestor = SimpleNamespace(audit_cache=lambda value: events.append(('audit', value)))
    monkeypatch.setattr(latest, 'require_unstarted', lambda paths: events.append(('guard', paths)))
    def history(arm):
        assert arm == 'tree'
        return 'training', 'evaluation', 'environment', ancestor, judgments, ['all-prior-inputs']
    monkeypatch.setattr(latest.generated.hierarchy, 'current_history', history)
    result = latest.closed_history()
    assert result == ('training', 'evaluation', 'environment', judgments, ['all-prior-inputs'])
    assert events[0] == ('guard', (*latest.DEFERRED, *latest.SUPERSEDED))
    assert events[1] == ('audit', judgments)
    judgments.pop('0')
    with pytest.raises(ValueError, match='5903'):
        latest.closed_history()


def test_priority_binding_keeps_model_rubric_and_split_and_rejects_conflicts(monkeypatch):
    monkeypatch.setattr(latest.data, 'sha', lambda path: 'measured:'+str(path))
    proof = {'config': {'checkpoint_sha256': 'model', 'rubric': 'unchanged',
                        'final_test_opened': False, 'DEV_cases': 128},
             'input_hashes': {'prior': 'fixed'}}
    original = deepcopy(proof)
    bound = latest.bind_priority(proof)
    assert proof == original
    assert all(bound['config'][k] == v for k, v in proof['config'].items())
    assert bound['input_hashes']['prior'] == 'fixed'
    assert bound['config']['base_first_judgments'] == 5903
    assert not bound['config']['input_validation_policy']['cross_process_cache']
    conflict = deepcopy(proof)
    conflict['input_hashes'][str(latest.PLAN)] = 'old-unrelated-plan'
    with pytest.raises(ValueError, match='conflicts'):
        latest.bind_priority(conflict)


def test_latest_stages_never_reuse_original_or_deferred_roots():
    roots = [*latest.GENERATION.values(), latest.COMPARISON,
             *(root for stages in latest.SEMANTIC.values() for root in stages.values())]
    assert len(roots) == len(set(roots))
    assert set(roots).isdisjoint((*latest.SUPERSEDED, *latest.DEFERRED))
    assert tuple(latest.SEMANTIC) == ('new_codec_only', 'parent_with_source')
