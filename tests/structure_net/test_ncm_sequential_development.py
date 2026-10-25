import pytest

from experiments.structure_net import ncm_sequential_development as run


def sources():
    keys=('id','source_id','group_id','source_sha256','source_component')
    return {c:[{**{k:f'{c}-{i}' for k in keys},'split':c} for i in range(64)] for c in run.old.COHORTS}


def test_components_and_all_source_identifiers_are_disjoint():
    cohorts=sources()
    train=[{k:'train' for k in ('id','source_id','group_id','source_sha256','source_component')}]
    assert all(r['overlap']==0 for r in run.split_guard(train,cohorts).values())
    train[0]['source_component']=cohorts[run.old.COHORTS[0]][0]['source_component']
    with pytest.raises(ValueError,match='source_component'):run.split_guard(train,cohorts)


def test_heldout_target_fields_cannot_enter_prediction():
    for field in run.old.FORBIDDEN:
        with pytest.raises(ValueError,match='source-only'):
            run.predict(None,{field:[]},{},None)


def test_incomplete_development_denominator_is_rejected():
    cohorts=sources();cohorts[run.old.COHORTS[0]].pop()
    with pytest.raises(ValueError,match='complete'):run.split_guard([],cohorts)
