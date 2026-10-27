from copy import deepcopy

import pytest

from experiments.structure_net import ncm_expanded_teacher_codes as study


def row(identity):
    return {'id':identity,'split':'TRAIN','target_text':'Unchanged summary.',
        'target_sha256':study.catalog.old.text_hash('Unchanged summary.'),'native_word_ids':[101,42,101]+[0]*125}


@pytest.mark.parametrize('bad',[[-1]*641,[8192]*641,[True]*641,[0]*640,[1.0]*641])
def test_hierarchy_requires_complete_integer_code_range(bad):
    with pytest.raises(ValueError):study.bind(row('a'),bad)


def test_target_edit_and_wrong_split_cannot_receive_codes():
    for edits in ({'target_text':'Edited.'},{'split':'DEV'},{'native_word_ids':[0]*127}):
        with pytest.raises(ValueError):study.bind({**row('a'),**edits},[1]*641)


def test_union_keeps_original_codes_and_metadata_exact():
    rows=[row('old'),row('new')];old={'old':study.bind(rows[0],[1]*641)};before=deepcopy(old)
    result=study.merge(rows,old,{'new':[2]*641})
    assert result[0]==old['old'] and old==before
    assert result[1]==study.bind(rows[1],[2]*641)
    assert 'oracle_codes' not in rows[0]


@pytest.mark.parametrize('kind',['missing','extra','overlap','duplicate_row','changed_old_target'])
def test_exact_old_new_partition_and_prior_binding_required(kind):
    rows=[row('old'),row('new')];old={'old':study.bind(rows[0],[1]*641)};fresh={'new':[2]*641}
    if kind=='missing':fresh={}
    elif kind=='extra':fresh['other']=[2]*641
    elif kind=='overlap':fresh['old']=[2]*641
    elif kind=='duplicate_row':rows.append(rows[0])
    elif kind=='changed_old_target':old['old']['target_text']='Edited.'
    with pytest.raises(ValueError):study.merge(rows,old,fresh)


def test_pilot_full_comparison_allows_only_collection_file_position_change():
    before={**study.bind(row('new'),[1]*641),'teacher_first_provenance':{'inventory':'pilot','inventory_position':0},
            'draft':{'attempts':['original/raw/attempt_0.json']}}
    after={**deepcopy(before),'teacher_first_provenance':{'inventory':'full','inventory_position':32}}
    assert study.same_first_target(before,after)
    for change in ({'oracle_codes':[2]*641},{'target_text':'Changed'},
                   {'draft':{'attempts':['replacement/raw/attempt_0.json']}}):
        assert not study.same_first_target(before,{**after,**change})
