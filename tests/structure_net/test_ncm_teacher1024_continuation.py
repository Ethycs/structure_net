from copy import deepcopy

import pytest

from experiments.structure_net import ncm_teacher1024_continuation as study
from experiments.neural_architecture_lab import ncm_teacher1024_continuation_service as service


def fixture_rows():
    rows = [{'id':str(i)} for i in range(10)]
    retained = [{**rows[0],'teacher_valid':True}]
    partial = [{**rows[i],'teacher_valid':i != 3} for i in (1,2,3)]
    return rows,retained,partial,{'1','2','3'},{'1','2','3'}


def test_partition_retains_failed_first_and_only_selects_untouched():
    rows,retained,partial,raw,saved = fixture_rows()
    observed,unused = study.partition(rows,retained,partial,raw,saved)
    assert observed == retained+partial and not observed[-1]['teacher_valid']
    assert unused == ['4','5','6','7','8','9']


@pytest.mark.parametrize('change',['duplicate','overlap','raw_only','missing_raw','missing_saved','outside'])
def test_any_unaccounted_or_retried_first_stops_continuation(change):
    rows,retained,partial,raw,saved = fixture_rows()
    if change == 'duplicate':partial.append(partial[0])
    elif change == 'overlap':retained.append(partial[0])
    elif change == 'raw_only':raw.add('4')
    elif change == 'missing_raw':raw.remove('2')
    elif change == 'missing_saved':saved.remove('1')
    else:rows.pop(3)
    with pytest.raises(ValueError):study.partition(rows,retained,partial,raw,saved)


def test_merge_preserves_complete_original_order_and_negative_record():
    rows,retained,partial,_,_ = fixture_rows()
    new = [{**r,'teacher_valid':True} for r in rows[4:]]
    values = study.combine(rows,new,partial,retained)
    assert [r['id'] for r in values] == [r['id'] for r in rows]
    assert values[3] == partial[-1]
    with pytest.raises(ValueError):study.combine(rows,new,partial,retained,partial[-1:])
    with pytest.raises(ValueError):study.combine(rows,new,retained)


def test_raw_attempts_cannot_receive_an_extra_allowance(tmp_path,monkeypatch):
    monkeypatch.setattr(study.original,'replay_candidate',lambda *a:None)
    root = tmp_path;folder = root/'requests'/'x'/'draft';folder.mkdir(parents=True)
    paths = [folder/f'attempt_{i}.json' for i in range(3)]
    for p in paths:p.write_text('{}')
    value = {'draft':{'status':'invalid','attempts':[str(p) for p in paths]},'review':None}
    assert study.replay_paths({'id':'x'},value,root,None,None) == paths
    bad = deepcopy(value);bad['draft']['attempts'].pop()
    with pytest.raises(ValueError):study.replay_paths({'id':'x'},bad,root,None,None)
    (folder/'attempt_3.json').write_text('{}')
    with pytest.raises(ValueError):study.replay_paths({'id':'x'},value,root,None,None)


def test_continuation_service_has_distinct_namespace_and_worker():
    unit,root,dest = service.locations('pilot')
    assert unit.endswith('-v3.service') and root.name.endswith('_v3')
    assert dest == study.ROOTS['pilot'] and dest != study.old.ROOTS['pilot']
    assert service.command('pilot')[-2:] == ['experiments.structure_net.ncm_teacher1024_continuation','pilot']
    value = {'status':'all1024_first_candidates_completed','cases':1024,'student_training_admitted':False,
        'canonical_semantic_cache_changed':False,'final_test_opened':False}
    assert service.valid_result('generate',value)
    assert not service.valid_result('generate',{**value,'cases':1016})
