import pytest

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
