import pytest
from types import SimpleNamespace

from experiments.structure_net import ncm_onpolicy_development as dev


def test_first_judgments_are_append_only_and_cannot_be_replaced_or_dropped():
    old = {'a':{'status':'valid','pass':False},'b':{'status':'invalid'}}
    current = {**old,'c':{'status':'valid','pass':True}}
    assert dev.preserve_firsts(old,current,expected_count=3,allowed_new={'c'}) == current
    for changed in ({'a':old['a'],'c':current['c']}, {**current,'a':{'status':'valid','pass':True}}):
        with pytest.raises(ValueError,match='changed or disappeared'):
            dev.preserve_firsts(old,changed)
    with pytest.raises(ValueError,match='outside'):
        dev.preserve_firsts(old,current,allowed_new=set())
    with pytest.raises(ValueError,match='count'):
        dev.preserve_firsts(old,current,expected_count=4)


def test_each_arm_has_distinct_fresh_pipeline_roots():
    roots = [p for stages in dev.ROOTS.values() for p in stages.values()]
    assert len(roots) == len(set(roots)) == 8
    assert all(set(stages) == set(dev.STAGES) for stages in dev.ROOTS.values())


def test_configuring_both_arms_in_one_process_keeps_original_ancestry_and_loader(monkeypatch):
    base = {str(i):{'first':i} for i in range(3923)}
    calls = []
    ancestor = SimpleNamespace(base_cache=lambda:dict(list(base.items())[:3795]),
        audit_cache=lambda cache:None, inputs=lambda:['original_ancestor'],configure=lambda:None,
        require_judges=lambda:None,uncertain_allowances=lambda:[])
    original_loader = object()
    training = SimpleNamespace(q=SimpleNamespace(read=lambda p:dict(base)),load_initial=original_loader)
    evaluation = SimpleNamespace(lineage=ancestor,sources=lambda:None)
    environment = {'immutable':'test'}
    def configure_parent():
        calls.append('parent')
        return training,evaluation,environment
    monkeypatch.setattr(dev,'_BOOT',None)
    monkeypatch.setattr(dev.previous,'configure',configure_parent)
    monkeypatch.setattr(dev.study.broad,'file_hash',lambda p:
        dev.BASE_CACHE_SHA if p == dev.BASE_CACHE else dev.study.PARENT_RESULTS_SHA)
    for arm in ('fixed_targets','residual_targets','fixed_targets'):
        t,e,env = dev.configure(arm,preflight=True)
        assert e.lineage.base_cache() == base and e.lineage.COUNT == 3923
        assert e.lineage.inputs() == ['original_ancestor',dev.BASE_CACHE,
            dev.study.PARENT_RESULTS/'campaign_results.json',dev.study.PARENT_RESULTS/'backend_exact_readback.json',
            dev.study.PARENT_RESULTS/'backend_actual_tool_receipt.json']
        assert t.load_initial is original_loader and env is environment
        assert t.RUNS == {'train':dev.study.ROOTS[arm,'train']}
    assert calls == ['parent']
