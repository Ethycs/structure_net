from copy import deepcopy
import json
import threading
import time

import pytest

from experiments.structure_net import ncm_teacher1024_isolated as study


def raw(model='gpt-5.6-terra',finish='stop'):
    response={'model':model,'choices':[{'finish_reason':finish,'message':{'content':'first'}}]}
    return {'transport':study.URL,'http_status':200,'response':response,'response_text':json.dumps(response),'valid':True,
        'request':{'model':'codex_sdk/gpt-5.6-terra','messages':[]},'prompt_tokens':16}


@pytest.mark.parametrize('change',[{'model':'gpt-5.6-sol'},{'model':'codex_sdk/gpt-5.6-terra'},{'finish':'length'}])
def test_model_alias_is_exact_and_raw_response_is_not_rewritten(change):
    value=raw();before=deepcopy(value)
    assert study.valid_native_transport(value,'draft') and value==before
    assert not study.valid_native_transport(raw(**change),'draft')
    assert not study.valid_native_transport({'response':[]},'draft')


def test_native_replay_binds_complete_payload_and_first_attempt(tmp_path,monkeypatch):
    class Encoding:
        def encode(self,*a,**k):return []
    monkeypatch.setattr(study.teacher,'payload',lambda *a:raw()['request'])
    monkeypatch.setattr(study.teacher,'parse',lambda content,row:{'summary':content})
    p=tmp_path/'attempt_0.json';study.data.write(p,raw())
    observation={'status':'valid','value':{'summary':'first'},'attempts':[str(p)]}
    assert study.replay_stage({'id':'x'},'draft',observation,Encoding())=={'summary':'first'}
    changed=raw();changed['response']['model']='gpt-5.6-sol';changed['response_text']=json.dumps(changed['response'])
    p.write_text(json.dumps(changed))
    with pytest.raises(ValueError):study.replay_stage({'id':'x'},'draft',observation,Encoding())
    p.write_text(json.dumps(raw()));p1=tmp_path/'attempt_1.json';study.data.write(p1,raw())
    with pytest.raises(ValueError):study.replay_stage({'id':'x'},'draft',{**observation,'attempts':[str(p),str(p1)]},Encoding())


def test_legacy_firsts_use_the_original_raw_validator(monkeypatch):
    monkeypatch.setattr(study.data,'read',lambda p:{'request':{'unchanged':True}})
    monkeypatch.setattr(study,'OLD_REPLAY',lambda *a:'legacy-first')
    assert study.replay_stage({},'draft',{'attempts':['old']},None)=='legacy-first'


def test_failed_pipeline_stops_submissions_and_drains_all_eight(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'fatal',lambda value:True)
    started=[];lock=threading.Lock()
    def candidate(row,*a):
        with lock:started.append(row['id'])
        time.sleep(.01)
        return {**row,'teacher_valid':False}
    root=tmp_path/'batch'
    with pytest.raises(RuntimeError):study.collect_batch([{'id':str(i)} for i in range(16)],root,None,None,candidate=candidate)
    assert len(started)==8 and len(study.data.read(root/'partial_firsts.json'))==8
    assert study.data.read(root/'terminal.json')['all_inflight_drained']


def test_schema_failure_streak_is_carried_across_batches(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'fatal',lambda value:False)
    candidate=lambda row,*a:{**row,'teacher_valid':False}
    _,streak=study.collect_batch([{'id':'a'},{'id':'b'}],tmp_path/'first',None,None,candidate=candidate)
    assert streak==2
    with pytest.raises(RuntimeError):study.collect_batch([{'id':'c'}],tmp_path/'next',None,None,streak,candidate)
    assert study.data.read(tmp_path/'next'/'terminal.json')['streak']==3


def test_successful_batch_is_complete_and_never_exceeds_eight_active_workers(tmp_path):
    lock=threading.Lock();active=peak=0
    def candidate(row,*a):
        nonlocal active,peak
        with lock:active+=1;peak=max(active,peak)
        time.sleep(.01)
        with lock:active-=1
        return {**row,'teacher_valid':True}
    rows=[{'id':str(i)} for i in range(16)]
    values,streak=study.collect_batch(rows,tmp_path/'batch',None,None,candidate=candidate)
    assert [r['id'] for r in values]==[r['id'] for r in rows] and streak==0 and 1<=peak<=8 and active==0


def test_request_preserves_invalid_json_transport_response(tmp_path,monkeypatch):
    class Encoding:
        def encode(self,*a,**k):return []
    class Response:
        status_code=502;text='gateway failed before JSON'
        def json(self):raise ValueError('not JSON')
    monkeypatch.setattr(study.teacher,'payload',lambda *a:raw()['request'])
    monkeypatch.setattr(study,'credential',lambda:'test-only')
    monkeypatch.setattr(study.requests,'post',lambda *a,**k:Response())
    result=study.request({'id':'x'},'draft',tmp_path,Encoding())
    assert result['status']=='invalid' and len(result['attempts'])==3
    for p in result['attempts']:
        value=study.data.read(p)
        assert value['http_status']==502 and value['response_text']==Response.text and not value['valid']
