from copy import deepcopy
import pytest

from experiments.structure_net import ncm_reserved_code_probe as p


def case(i=0):
    return {**{k:str(i) for k in p.KEYS},'split':'reserved_component','source':'source '+str(i),
        'oracle_codes':[1]*641,'native_word_ids':[101,5,101]+[0]*125}


def output(row):
    return {'id':row['id'],'source_id_used':row['id'],'task_accuracy':None,
        'codes':list(row['oracle_codes']),'teacher_codes':list(row['oracle_codes']),
        'raw_word_ids':list(row['native_word_ids']),'active_words':3,'active_word_matches':3,
        'levels':[{'positions':n,'teacher_correct':n,'rollout_correct':n,'CE':0.} for n in p.controls.LEVELS]}


def test_all64_source_frame_and_order_preserved():
    rows = [case(i) for i in range(64)]
    assert p.select(rows,list(reversed(rows))) == rows
    assert p.select([{**case(100),'split':'TRAIN_control'},*rows],rows) == rows
    for field in ('source','source_component','native_word_ids'):
        altered=deepcopy(rows);altered[0][field]=[] if field=='native_word_ids' else 'different'
        with pytest.raises(ValueError):p.select(rows,altered)
    with pytest.raises(ValueError):p.select(rows[:-1],rows)
    with pytest.raises(ValueError):p.select(rows,[*rows[:-1],rows[0]])


def test_source_identity_overlap_rejected():
    assert not any(p.identity_guard([case(1)],[case(2)])['overlaps'].values())
    for key in p.KEYS:
        row=case(2);row[key]=case(1)[key]
        with pytest.raises(ValueError):p.identity_guard([case(1)],[row])


def test_exact_raw_metric_replay_and_shuffled_binding():
    rows=[case(1),case(2)];outputs=[output(r) for r in rows]
    result=p.score(outputs,rows)
    assert result['teacher_code_agreement']==result['rollout_code_agreement']==1
    assert result['teacher_rollout_gap']==0 and result['native_word_agreement']==1
    for i,r in enumerate(outputs):r['source_id_used']=rows[1-i]['id']
    assert p.score(outputs,rows,True)==result
    with pytest.raises(ValueError):p.score(outputs,rows)


@pytest.mark.parametrize('field,value',[('task_accuracy',.95),('source_id_used','other'),('active_words',4),('active_word_matches',2)])
def test_invalid_raw_metadata_rejected(field,value):
    row=case();out=output(row);out[field]=value
    with pytest.raises(ValueError):p.score([out],[row])


def test_root_consistency_counts_and_finite_ce():
    row=case();out=output(row);out['teacher_codes'][0]=2
    with pytest.raises(ValueError,match='root'):p.score([out],[row])
    for mutate in (lambda o:o['levels'][1].update(teacher_correct=0),
                   lambda o:o['levels'][1].update(CE=float('nan')),
                   lambda o:o['codes'].append(1)):
        out=output(row);mutate(out)
        with pytest.raises(ValueError):p.score([out],[row])


def test_reserved_is_never_relabelled_train():
    row=case();row['split']='TRAIN'
    with pytest.raises(ValueError):p.score([output(row)],[row])


def test_classification_boundary_is_descriptive():
    assert p.classify({'teacher_code_agreement':.8999,'teacher_rollout_gap':.8})=='reserved_teacher_prediction_also_weak'
    assert p.classify({'teacher_code_agreement':.9,'teacher_rollout_gap':.2})=='reserved_exposure_error_pattern_supported'
    assert p.classify({'teacher_code_agreement':.95,'teacher_rollout_gap':.1999})=='reserved_exposure_error_pattern_unresolved'
