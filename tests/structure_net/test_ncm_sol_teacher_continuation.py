"""Syntax failure remains invalid; service failure still stops new submissions."""
import pytest
from experiments.structure_net import ncm_sol_teacher_continuation as run


def invalid(tmp_path,http=200,finish='stop'):
    path=tmp_path/f'raw_{http}_{finish}.json'
    run.write_json(path,{'http_status':http,'response':{'model':run.previous.teacher.MODELS['review'],
        'choices':[{'finish_reason':finish}]}})
    return {'id':'0','teacher_valid':False,'capacity_fit':False,'draft':{'status':'valid'},
        'review':{'status':'invalid','attempts':[str(path)]}}


def test_schema_invalid_is_not_reclassified_as_valid(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'WORKERS',1)
    (tmp_path/'cases').mkdir()
    first=invalid(tmp_path)
    assert run.fatal_exhaustion(first) is False
    seen=[]
    def candidate(row,*args):
        seen.append(row['id'])
        return first if row['id']=='0' else {'id':row['id'],'teacher_valid':True,'capacity_fit':False}
    values=run.candidate_jobs([{'id':str(i)} for i in range(4)],tmp_path,None,None,candidate)
    assert seen==['0','1','2','3'] and values[0]['teacher_valid'] is False
    assert not any(v['capacity_fit'] for v in values)


def test_transport_or_incomplete_reply_stops_without_replacing_first(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'WORKERS',1)
    (tmp_path/'cases').mkdir()
    first=invalid(tmp_path,http=429)
    assert run.fatal_exhaustion(first)
    assert run.fatal_exhaustion(invalid(tmp_path,finish='length'))
    seen=[]
    def candidate(row,*args):
        seen.append(row['id'])
        return first
    with pytest.raises(RuntimeError,match='inflight drained'):
        run.candidate_jobs([{'id':str(i)} for i in range(4)],tmp_path,None,None,candidate)
    assert seen==['0']
    assert run.q.read(tmp_path/'partial_firsts.json')==[first]
