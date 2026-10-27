from pathlib import Path

import pytest

from experiments.structure_net import ncm_teacher1024_writer_recovery as study
from experiments.neural_architecture_lab import ncm_teacher1024_writer_recovery_service as service


def test_producer_is_complete_before_the_single_exclusive_write(tmp_path):
    source = tmp_path/'producer.py';source.write_text('pass\n')
    root = tmp_path/'stage'
    proof = {'config':{'original438_firsts_preserved':True,'original8_exhausted_failures_preserved':True},
        'source_hashes':{str(source):study.data.sha(source)},'input_hashes':{}}
    result = study.write_producer(root,proof,[source])
    assert study.data.read(root/'producer.json') == result
    study.base.verify_proof(result)
    assert next((root/'producer_source').iterdir()).read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):study.write_producer(root,proof,[source])


def test_new_service_and_roots_cannot_overwrite_failed_writer_stage():
    assert study.FAILURE.name.endswith('_v3')
    assert all(p.name.endswith('_v4') for p in study.ROOTS.values())
    unit,root,dest = service.locations('pilot')
    assert unit.endswith('-v4.service') and root.name.endswith('_v4')
    assert dest == study.ROOTS['pilot']
    assert service.command('pilot')[-2:] == ['experiments.structure_net.ncm_teacher1024_writer_recovery','pilot']
