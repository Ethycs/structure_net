from copy import deepcopy
import json
import subprocess
import sys

import pytest

from experiments.structure_net import ncm_recovered_task_supervision as study


def test_only_unexecuted_downstream_roots_change():
    assert study.CATALOG_ROOTS['pilot'] == study.OLD_CATALOG_ROOTS['pilot']
    assert study.TARGET_ROOTS['pilot'] == study.OLD_TARGET_ROOTS['pilot']
    assert study.TARGET_ROOTS['pilot_audit'] == study.OLD_TARGET_ROOTS['pilot_audit']
    for stage in ('pilot','pilot_audit'):
        assert study.TRAIN_ROOTS['existing',stage] == study.OLD_TRAIN_ROOTS['existing',stage]
        assert study.TRAIN_ROOTS['expanded',stage] != study.OLD_TRAIN_ROOTS['expanded',stage]
    for arm in ('existing','expanded'):
        for stage in ('train','audit'):assert study.TRAIN_ROOTS[arm,stage] != study.OLD_TRAIN_ROOTS[arm,stage]
    assert study.COLLECTION_ROOTS['prepare'] == study.OLD_COLLECTION_ROOTS['prepare']
    assert study.COLLECTION_ROOTS['generate'] == study.recovered.ROOTS['generate']
    assert study.OLD_COLLECTION.ROOTS == study.OLD_COLLECTION_ROOTS


@pytest.mark.parametrize('change',['partial','missing_failures','wrong_status'])
def test_partial_or_replaced_collection_cannot_pass_consumer_guard(monkeypatch,change):
    value = {'status':'all1024_teacher_firsts_replayed','cases':1024,'preserved_exhausted_failures':8}
    monkeypatch.setattr(study.base,'require',lambda root:deepcopy(value))
    monkeypatch.setattr(study.base,'verify_proof',lambda proof:None)
    monkeypatch.setattr(study.data,'read',lambda path:{})
    assert study.require_collection('audit') == value
    if change == 'partial':value['cases']=438
    elif change == 'missing_failures':value['preserved_exhausted_failures']=0
    else:value['status']='teacher_pilot_firsts_replayed'
    with pytest.raises(ValueError):study.require_collection('audit')


def test_bindings_record_complete_lineage_and_unchanged_budget():
    bindings = study.root_bindings()
    assert bindings['collection']['prepare'].endswith('prepare_v2')
    assert bindings['collection']['generate'].endswith('generate_v4')
    assert bindings['catalog']['prepare'].endswith('prepare_v2')
    assert bindings['training']['existing:pilot'].endswith('pilot_v1')
    assert bindings['training']['expanded:pilot'].endswith('pilot_v2')
    assert study.training.CONFIG.updates == 2048 and study.training.CONFIG.presentations_per_epoch == 65536


def test_service_installs_consumer_mapping_without_mutating_collector():
    code = '''
import json
from experiments.structure_net import ncm_recovered_task_supervision as m
before = dict(m.OLD_COLLECTION.ROOTS)
from experiments.neural_architecture_lab import ncm_recovered_task_supervision_service as s
print(json.dumps({'unchanged':m.OLD_COLLECTION.ROOTS == before,
    'command':s.command('expanded','pilot'),'unit':s.locations('expanded','pilot')[0],
    'root':str(s.locations('expanded','pilot')[2]),'installed':m.training.ROOTS == m.TRAIN_ROOTS}))
'''
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    value = json.loads(result.stdout.strip().splitlines()[-1])
    assert value['unchanged'] and value['installed']
    assert value['unit'].endswith('-v2.service') and value['root'].endswith('_v2')
    assert value['command'][-6:] == ['experiments.structure_net.ncm_recovered_task_supervision',
        'training','--stage','pilot','--arm','expanded']
