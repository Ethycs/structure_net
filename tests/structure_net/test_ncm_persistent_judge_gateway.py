from copy import deepcopy

import pytest

from experiments.neural_architecture_lab import ncm_persistent_judge_gateway as gateway
from experiments.structure_net import ncm_persistent_judge_qualification as qualification


def test_persistent_launcher_preserves_rpc_arguments():
    args=['app-server','-c','mcp_servers={}']
    assert gateway.launcher.command(args)==[str(gateway.launcher.WRAPPER),*args]
    assert args==['app-server','-c','mcp_servers={}']


def test_fallback_is_rejected_before_any_execution():
    for args in ([],['exec','--model','gpt-5.6-sol','-'],['login'],['mcp','list']):
        with pytest.raises(ValueError,match='only the persistent'):gateway.launcher.command(args)


def test_live_service_invocation_and_restart_guards_remain_strict():
    proof={'invocation_id':'original','supervisor_PID':123}
    state=dict(ActiveState='active',SubState='running',NRestarts='0',InvocationID='original',MainPID='123',ExecMainPID='123')
    gateway.base.validate_state(proof,state)
    for change in ({'NRestarts':'1'},{'InvocationID':'other'},{'ExecMainPID':'124'},{'SubState':'exited'}):
        with pytest.raises(ValueError):gateway.base.validate_state(proof,{**state,**change})


def test_qualification_keeps_control_and_scoring_functions_on_distinct_roots(monkeypatch):
    original=qualification.original
    score,request,replay=original.score,original.transport.request,original.transport.replay_row
    old_roots=deepcopy(original.ROOTS); models=dict(original.transport.MODELS)
    for target,name in ((original,'neutral'),(original,'ROOTS'),(original,'PLAN'),(original,'TEST'),
                        (original.transport,'IDENTITY'),(original.transport,'credential')):
        monkeypatch.setattr(target,name,getattr(target,name))
    monkeypatch.setattr(gateway,'require',lambda:None);monkeypatch.setattr(gateway,'install',lambda:None)
    qualification.install()
    assert not set(original.ROOTS.values()) & set(old_roots.values())
    assert original.transport.MODELS==models and original.transport.IDENTITY==qualification.IDENTITY
    assert original.score is score and original.transport.request is request and original.transport.replay_row is replay
