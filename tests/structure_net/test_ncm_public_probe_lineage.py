from copy import deepcopy

import pytest

from experiments.structure_net import ncm_public_probe_lineage as lineage
from experiments.neural_architecture_lab import ncm_public_probe_milestones as reporting


@pytest.fixture
def branch(monkeypatch):
    monkeypatch.setattr(lineage.probe.first,'packet_key',lambda p,role:p['id']+':'+role)
    packets = {arm:[{'id':f'{arm}_{i}'} for i in range(128)] for arm in lineage.probe.ARMS}
    metrics = {a:{'secondary':{'cases':128,'valid':128,'passed':0},'primary':None} for a in packets}
    inherited = {'past':{'status':'valid','decision':False},
                 'native_0:secondary':{'status':'invalid','decision':None}}
    cache = {**inherited,**{p['id']+':secondary':{'status':'valid','decision':False}
                            for ps in packets.values() for p in ps if p['id'] != 'native_0'}}
    return inherited,cache,packets,metrics


def test_retains_negative_and_invalid_firsts_and_all_arms(branch):
    inherited,cache,packets,metrics = branch
    before = deepcopy(inherited)
    result = lineage.extend(inherited,cache,packets,metrics,len(cache))
    assert result == cache and inherited == before
    assert result['past']['decision'] is False
    assert result['native_0:secondary']['status'] == 'invalid'
    assert len(result) == 513


@pytest.mark.parametrize('failure',['changed','lost','missing_new','extra','partial_arm','missing_arm',
    'unadmitted_primary','omitted_admitted_primary','wrong_count','unresolved_admission'])
def test_rejects_incomplete_or_replaced_branch(branch,failure):
    inherited,cache,packets,metrics = branch
    if failure == 'changed':cache['past']['decision'] = True; inherited = {'past':{'status':'valid','decision':False},'native_0:secondary':inherited['native_0:secondary']}
    elif failure == 'lost':del cache['past']
    elif failure == 'missing_new':del cache['distill_127:secondary']
    elif failure == 'extra':cache['other:secondary'] = {'status':'valid','decision':True}
    elif failure == 'partial_arm':packets['native'].pop()
    elif failure == 'missing_arm':del packets['native']
    elif failure == 'unadmitted_primary':metrics['native']['primary'] = {'cases':128}
    elif failure == 'omitted_admitted_primary':metrics['native']['secondary']['passed'] = 122
    elif failure == 'unresolved_admission':
        metrics['native']['secondary'].update(valid=127,passed=122)
        metrics['native']['primary'] = {'cases':128}
    with pytest.raises(ValueError):
        lineage.extend(inherited,cache,packets,metrics,len(cache)+(failure == 'wrong_count'))


def test_primary_firsts_must_cover_every_admitted_case(branch):
    inherited,cache,packets,metrics = branch
    metrics['native']['secondary']['passed'] = 122
    metrics['native']['primary'] = {'cases':128}
    cache.update({p['id']+':primary':{'status':'valid','decision':False} for p in packets['native']})
    assert lineage.extend(inherited,cache,packets,metrics,len(cache)) == cache
    del cache['native_127:primary']
    with pytest.raises(ValueError,match='every admitted'):
        lineage.extend(inherited,cache,packets,metrics,len(cache))


def test_insertion_occurs_only_after_epoch2_and_once(monkeypatch):
    monkeypatch.setattr(lineage,'INSTALLED',False)
    monkeypatch.setattr(lineage,'OLD_APPEND',lambda cache,root,evaluation:[*cache,root])
    monkeypatch.setattr(lineage,'branch_cache',lambda cache:[*cache,'probe'])
    monkeypatch.setattr(lineage,'OLD_PATHS',lambda roots:['old'])
    monkeypatch.setattr(lineage,'branch_inputs',lambda:['probe-inputs'])
    # Keep shared historical modules unchanged after this systems test.
    for key in ('append_grade','lineage_paths','PLAN','TEST','PREFLIGHT'):
        monkeypatch.setattr(lineage.original,key,getattr(lineage.original,key))
    lineage.install();first = lineage.original.append_grade;lineage.install()
    assert lineage.original.append_grade is first
    assert first([],lineage.roots(1)['grade'],None) == [lineage.roots(1)['grade']]
    assert first([],lineage.roots(2)['grade'],None) == [lineage.roots(2)['grade'],'probe']
    assert first([],lineage.roots(4)['grade'],None) == [lineage.roots(4)['grade']]
    assert lineage.original.lineage_paths([lineage.roots(2)['grade']]) == ['old','probe-inputs']
    assert lineage.original.lineage_paths([lineage.roots(1)['grade']]) == ['old']


def test_reporting_inserts_branch_count_without_pooling_other_milestones():
    assert reporting.inherited_for_milestone(1,4435,5203) == 4435
    assert reporting.inherited_for_milestone(2,4563,5203) == 4563
    assert reporting.inherited_for_milestone(4,4691,5203) == 5203
    assert reporting.inherited_for_milestone(8,5331,5203) == 5331
    with pytest.raises(ValueError):reporting.inherited_for_milestone(4,4690,5203)
    with pytest.raises(ValueError):reporting.inherited_for_milestone(4,4691,4690)


@pytest.mark.parametrize('epoch',[1,2])
def test_historical_milestones_cannot_be_reconfigured(epoch):
    with pytest.raises(ValueError,match='historical'):lineage.configure(epoch,1)
    with pytest.raises(ValueError,match='unchanged'):reporting.run('replay',epoch,1)
