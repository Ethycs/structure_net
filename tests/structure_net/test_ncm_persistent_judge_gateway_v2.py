import importlib.util
import sys

import pytest

from experiments.neural_architecture_lab import ncm_persistent_judge_gateway_v2 as gateway
from experiments.neural_architecture_lab import ncm_persistent_judge_launcher as original_launcher
from experiments.structure_net import ncm_persistent_judge_qualification_v2 as qualification


def launcher(monkeypatch):
    monkeypatch.setitem(sys.modules,'ncm_persistent_judge_launcher',original_launcher)
    spec=importlib.util.spec_from_file_location('isolated_sqlite_launcher',gateway.CLI)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_isolation_keeps_persistent_only_and_original_arguments(monkeypatch):
    module=launcher(monkeypatch);args=['app-server','-c','mcp_servers={}']
    value=module.command(args)
    assert value[:-2]==original_launcher.command(args)
    assert value[-2:]==['-c','sqlite_home="'+str(module.SQLITE)+'"']
    with pytest.raises(ValueError):module.command(['exec','--model','gpt-5.6-sol'])


def test_initialization_is_not_model_quality_and_requires_closed_process():
    value={'initialize_response':{'id':1,'result':{'userAgent':'test'}},'model_requests':0,'process_closed':True}
    gateway.check_warmup(value)
    for change in ({'initialize_response':None},{'model_requests':1},{'process_closed':False}):
        with pytest.raises(ValueError):gateway.check_warmup({**value,**change})


def test_qualification_retains_original_semantic_protocol(monkeypatch):
    original=qualification.original
    score,request,replay=original.score,original.transport.request,original.transport.replay_row
    roots=dict(original.ROOTS);models=dict(original.transport.MODELS)
    for target,name in ((original,'neutral'),(original,'ROOTS'),(original,'PLAN'),(original,'TEST'),
                        (original.transport,'IDENTITY'),(original.transport,'credential')):
        monkeypatch.setattr(target,name,getattr(target,name))
    monkeypatch.setattr(gateway,'require',lambda:None);monkeypatch.setattr(gateway,'install',lambda:None)
    qualification.install()
    assert not set(roots.values()) & set(original.ROOTS.values())
    assert original.score is score and original.transport.request is request and original.transport.replay_row is replay
    assert original.transport.MODELS==models and original.transport.IDENTITY==qualification.IDENTITY


def test_producer_keeps_original_and_new_prerequisites(monkeypatch):
    monkeypatch.setattr(gateway.base,'sha',lambda path:'hash:'+str(path))
    original={'source_hashes':{'old_source.py':'old_hash'}}
    proof=gateway.producer_proof(original)
    for path in (gateway.previous.CPU,gateway.CPU,gateway.WARMUP/'actual_tool_receipt.json',
                 gateway.previous.DIRECT/'actual_tool_receipt.json'):
        assert proof['input_hashes'][str(path)]=='hash:'+str(path)
    assert proof['source_hashes']['old_source.py']=='old_hash'
    assert original=={'source_hashes':{'old_source.py':'old_hash'}}
    assert proof['exec_fallback_allowed'] is False
