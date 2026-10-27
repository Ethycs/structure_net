from copy import deepcopy
import json
import subprocess
import sys

import pytest

from experiments.structure_net import ncm_isolated_task_supervision as study


def population():
    rows = [{'id':str(i)} for i in range(1024)]
    labels = [{**r,'teacher_valid':i>=16,'capacity_fit':i>=16,'target_text':'first-'+str(i)}
              for i,r in enumerate(rows)]
    return rows,deepcopy(labels[:446]),labels,deepcopy(labels[16:24]),deepcopy(labels[446:454])


def test_complete_population_keeps_all_historical_negatives_and_both_pilots():
    values=population();result=study.validate_firsts(*values)
    assert result['outcomes']==1024 and result['preserved_firsts']==446
    assert result['total_invalid']==result['preserved_historical_invalid']==16
    assert result['teacher_valid']==result['capacity_fit']==1008
    values[2][900].update(teacher_valid=False,capacity_fit=False)
    assert study.validate_firsts(*values)['total_invalid']==17


@pytest.mark.parametrize('change',['partial','rerolled_failure','changed_success','missing_historical',
                                  'reordered','changed_old_pilot','changed_new_pilot'])
def test_partial_or_modified_first_population_is_rejected(change):
    rows,retained,labels,old,new=population()
    if change=='partial':labels.pop()
    elif change=='rerolled_failure':labels[0].update(teacher_valid=True,capacity_fit=True)
    elif change=='changed_success':labels[100]['target_text']='replacement'
    elif change=='missing_historical':retained.pop()
    elif change=='reordered':labels[0],labels[1]=labels[1],labels[0]
    elif change=='changed_old_pilot':old[0]['target_text']='replacement'
    else:new[0]['target_text']='replacement'
    with pytest.raises(ValueError):study.validate_firsts(rows,retained,labels,old,new)


def test_only_unexecuted_downstream_roots_change():
    assert study.CATALOG_ROOTS['pilot']==study.OLD_CATALOG_ROOTS['pilot']
    assert study.TARGET_ROOTS['pilot']==study.OLD_TARGET_ROOTS['pilot']
    assert study.TARGET_ROOTS['pilot_audit']==study.OLD_TARGET_ROOTS['pilot_audit']
    for s in ('pilot','pilot_audit'):
        assert study.TRAIN_ROOTS['existing',s]==study.OLD_TRAIN_ROOTS['existing',s]
        assert study.TRAIN_ROOTS['expanded',s]!=study.OLD_TRAIN_ROOTS['expanded',s]
    for a in ('existing','expanded'):
        for s in ('train','audit'):assert study.TRAIN_ROOTS[a,s]!=study.OLD_TRAIN_ROOTS[a,s]
    b=study.root_bindings()
    assert b['collection']['prepare'].endswith('prepare_v2')
    assert b['collection']['pilot'].endswith('pilot_v2')
    assert b['collection']['generate'].endswith('generate_v5')
    assert b['catalog']['prepare'].endswith('prepare_v3')
    assert b['training']['existing:pilot'].endswith('pilot_v1')
    assert b['training']['expanded:pilot'].endswith('pilot_v3')
    assert study.training.CONFIG.updates==2048 and study.training.CONFIG.presentations_per_epoch==65536


@pytest.mark.parametrize('change',[{'cases':446},{'preserved_exhausted_failures':8},
    {'status':'teacher_pilot_firsts_replayed'},{'new_requests':1}])
def test_full_raw_audit_and_all_sixteen_historical_failures_are_required(monkeypatch,change):
    value={'status':'all1024_teacher_firsts_replayed','cases':1024,'preserved_exhausted_failures':16,'new_requests':0}
    monkeypatch.setattr(study.base,'require',lambda root:deepcopy(value))
    monkeypatch.setattr(study.base,'verify_proof',lambda proof:None)
    monkeypatch.setattr(study.data,'read',lambda path:{})
    assert study.require_collection('audit')==value
    value.update(change)
    with pytest.raises(ValueError):study.require_collection('audit')


def test_consumer_installs_native_replay_without_rebinding_historical_collector():
    code='''
import json
from experiments.structure_net import ncm_isolated_task_supervision as m
before=dict(m.OLD_COLLECTION.ROOTS)
from experiments.neural_architecture_lab import ncm_isolated_task_supervision_service as s
m.catalog.collection.configure()
print(json.dumps({'historical_roots_unchanged':m.OLD_COLLECTION.ROOTS==before,
    'native_replay_installed':m.recovered.original.replay is m.recovered.replay_stage,
    'command':s.command('expanded','pilot'),'unit':s.locations('expanded','pilot')[0],
    'root':str(s.locations('expanded','pilot')[2]),'training_installed':m.training.ROOTS==m.TRAIN_ROOTS}))
'''
    process=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    value=json.loads(process.stdout.strip().splitlines()[-1])
    assert value['historical_roots_unchanged'] and value['native_replay_installed'] and value['training_installed']
    assert value['unit'].endswith('-v3.service') and value['root'].endswith('_v3')
    assert value['command'][-6:]==['experiments.structure_net.ncm_isolated_task_supervision',
        'training','--stage','pilot','--arm','expanded']
