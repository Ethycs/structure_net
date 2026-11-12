import pytest

from experiments.word_games import controller_coding as run
from experiments.word_games.bridges.client import InvalidGeneration, ModelUnavailable


def args(tmp_path):
    return dict(arm='translated', root=tmp_path/'out', workspace=tmp_path/'workspace', key='placeholder',
        controller=object(), qwen=object(), goal='original requirement',
        example={'loans':{'loan:x':{'id':'loan:x','title':'PRIVATE TITLE'}}},
        instruction='initial instruction', privacy=run.Privacy({}), mapping=run.Mapping(run.PHI), source_manual={})


def test_controller_receives_schema_without_concrete_values():
    value={'loans':{'loan:PRIVATE_ID':{'title':'PRIVATE TITLE','restricted':True,'notes':['PRIVATE NOTE']}},'text':'PRIVATE QUERY'}
    shaped=run.shape(value)
    assert 'PRIVATE' not in repr(shaped)
    assert shaped['loans']['<identifier>']['restricted']=='boolean'


def test_translator_pause_budget_stops_before_execution(tmp_path, monkeypatch):
    revisions=[]
    monkeypatch.setattr(run,'translate',lambda *a:{'kind':'needs_simplification','instruction':'','needs':['one_step']})
    def revision(*a):
        revisions.append(a[-1]); return 'simpler instruction'
    monkeypatch.setattr(run,'interpret',revision)
    monkeypatch.setattr(run,'actor',lambda *a:pytest.fail('Executor must not run before a translation'))
    result=run.run_arm(**args(tmp_path))
    assert result['status']=='clarification_exhausted' and not result['delivery_complete']
    assert len(revisions)==2 and len(result['clarification_requests'])==3
    assert all(set(r)=={'kind','needs','stage'} for r in revisions)


def test_executor_request_returns_to_controller_and_revision_is_translated_before_same_session_resume(tmp_path, monkeypatch):
    order=[];sessions=[]
    def translate(client,root,instruction,*a):
        order.append(('translate',instruction))
        return {'kind':'instruction','instruction':'translated '+instruction,'needs':[]}
    def interpret(*a):
        order.append(('controller',a[-1])); return 'simpler instruction'
    def actor(workspace,prompt,out,key,session):
        sessions.append(session);order.append(('execute',session))
        return {'session_ids':['one-session'],'errors':[],'returncode':0,'timed_out':False,'seconds':1,
            'controller_request':{'kind':'needs_simplification','needs':['one_step'],'message':'PRIVATE PROSE'} if len(sessions)==1 else None}
    monkeypatch.setattr(run,'translate',translate);monkeypatch.setattr(run,'interpret',interpret);monkeypatch.setattr(run,'actor',actor)
    result=run.run_arm(**args(tmp_path))
    assert [x[0] for x in order]==['translate','execute','controller','translate','execute']
    assert order[2][1]=={'kind':'needs_simplification','needs':['one_step'],'stage':'executor'}
    assert order[3][1]=='simpler instruction' and sessions==[None,'one-session']
    assert result['delivery_complete'] and result['actor_invocations']==2


@pytest.mark.parametrize('names,status', [(['ContextOverflowError'],'context_or_timeout_failure'),
                                         (['InvalidInstructionFeedback'],'invalid_actor_feedback')])
def test_model_failure_is_not_retried(tmp_path, monkeypatch, names, status):
    options=args(tmp_path);options['arm']='native'
    monkeypatch.setattr(run,'actor',lambda *a:{'session_ids':[],'errors':[{'error':{'name':n}} for n in names],
                                             'returncode':0,'timed_out':False,'seconds':1})
    monkeypatch.setattr(run,'interpret',lambda *a:pytest.fail('No answer repair on failure'))
    result=run.run_arm(**options)
    assert result['status']==status and not result['delivery_complete'] and result['actor_invocations']==1


def test_transport_failure_stops_even_without_session_id(tmp_path, monkeypatch):
    options=args(tmp_path);options['arm']='native'
    monkeypatch.setattr(run,'actor',lambda *a:{'session_ids':[],'errors':[{'error':{'name':'APIError'}}],
                                             'returncode':0,'timed_out':False,'seconds':1})
    with pytest.raises(ModelUnavailable):run.run_arm(**options)


@pytest.mark.parametrize('bad', [
    {'kind':'instruction','instruction':'','needs':[]},
    {'kind':'needs_simplification','instruction':'','needs':['one_step','one_step']},
    {'kind':'needs_simplification','instruction':'','needs':['PRIVATE']},
    {'kind':'needs_simplification','instruction':'','needs':[{}]},
    {'kind':'instruction','instruction':'do it','needs':[],'source':'PRIVATE'},
])
def test_invalid_translation_never_reaches_executor(bad):
    with pytest.raises(InvalidGeneration):run.validate_translation(bad)
