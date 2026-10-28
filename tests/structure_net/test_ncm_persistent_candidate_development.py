from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_persistent_candidate_development as run


def test_fixed_families_preserve_all_semantic_origins():
    assert run.select('hierarchy','tree') is run.hierarchy
    assert run.select('recovered','new_codec_only') is run.recovered
    with pytest.raises(ValueError):run.select('hierarchy','new_codec_only')
    with pytest.raises(ValueError):run.select('unknown','tree')
    roots=run.semantic_roots()
    for module in (run.consumer.task,run.hierarchy,run.recovered):
        assert all(stages[s] in roots for stages in module.ROOTS.values() for s in ('judge_pilot','grade'))


def test_configure_preserves_cache_sources_and_original_prerequisites(monkeypatch):
    calls=[];cache={'prior':{'immutable':'first'}};before=deepcopy(cache);bound={'admission':'digest'}
    ancestor=SimpleNamespace(COUNT=5651,base_cache=lambda:deepcopy(cache),inputs=lambda:['original'],
        configure=lambda:'original_configuration',require_judges=lambda:calls.append('original_admission'),
        uncertain_allowances=lambda:{'original':'budget'},audit_cache=lambda value:calls.append(('old_audit',value)))
    training=SimpleNamespace(q=run.consumer.q);evaluation=SimpleNamespace(lineage=ancestor)
    trained,environment={'checkpoint':'unchanged'},{'environment':'unchanged'}
    module=SimpleNamespace(configure=lambda arm:(training,evaluation,trained,environment))
    monkeypatch.setattr(run.consumer,'bindings',lambda:bound)
    monkeypatch.setattr(run.consumer,'audit_local_entries',lambda values,roots:calls.append(('wire_audit',values,roots)))
    monkeypatch.setattr(run.consumer.q,'request',run.consumer.q.request)
    run.install_candidate(module,bound)
    result=module.configure('tree');lineage=evaluation.lineage
    assert result==(training,evaluation,trained,environment)
    assert lineage.COUNT==5651 and lineage.base_cache()==before and cache==before
    assert lineage.configure()=='original_configuration' and lineage.uncertain_allowances()=={'original':'budget'}
    assert list(map(str,lineage.inputs()))==['original','admission']
    lineage.require_judges();lineage.audit_cache(cache)
    assert calls[0]=='original_admission' and calls[1]==('old_audit',cache)
    assert calls[2]==('wire_audit',cache,run.semantic_roots())
    assert run.consumer.q.request is run.consumer.serialized_request
    monkeypatch.setattr(run.consumer,'bindings',lambda:{'changed':'admission'})
    with pytest.raises(ValueError):lineage.require_judges()


def test_unrecognized_judge_runtime_cannot_dispatch():
    module=SimpleNamespace(configure=lambda arm:(SimpleNamespace(q=object()),None,None,None))
    run.install_candidate(module,{})
    with pytest.raises(ValueError,match='shared semantic'):module.configure('tree')


def test_producer_keeps_models_rubric_firsts_and_candidate_identity(monkeypatch):
    monkeypatch.setattr(run.data,'sha',lambda path:'measured:'+str(path))
    proof={'stage':'grade','arm':'tree','config':{'judges':dict(run.consumer.transport.MODELS),
        'rubric':run.consumer.qualification.previous.profile.RUBRIC,'inherited_first_judgments':5775,
        'source_only':True},'input_hashes':{'model':'original','prior_firsts':'unchanged'}}
    before=deepcopy(proof);value=run.augment(proof,{'admitted_transport':'verified'})
    for key,item in before['config'].items():assert value['config'][key]==item
    assert value['arm']=='tree' and value['input_hashes']['model']=='original'
    assert value['input_hashes']['prior_firsts']=='unchanged'
    assert value['config']['active_judge_requests']==1
    assert value['input_hashes'][str(run.CPU)]=='measured:'+str(run.CPU)
    with pytest.raises(ValueError):run.augment({**before,'stage':'outputs'},{})
