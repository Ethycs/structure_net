from copy import deepcopy

import pytest

from experiments.structure_net import ncm_teacher1024_recovery as recovery
from experiments.neural_architecture_lab import ncm_teacher1024_recovery_service as service


def test_exact_original_training_label_accepted_without_rewriting_record():
    row = {'split':'additional_train','source':'unchanged','id':'original'}
    before = deepcopy(row)
    assert recovery.reserved_train_split(row)
    assert row == before


@pytest.mark.parametrize('value',['train','dev','test','component','final','TRAIN'])
def test_other_split_labels_rejected(value):
    assert not recovery.reserved_train_split({'split':value})


def test_new_worker_and_service_cannot_reuse_failed_roots():
    for stage in ('pilot','generate'):
        unit, supervision, root = service.locations(stage)
        assert unit.endswith('-v2.service') and str(supervision).endswith('_v2')
        assert root == recovery.ROOTS[stage] and str(root).endswith('_v2')
        assert service.command(stage)[-2:] == ['experiments.structure_net.ncm_teacher1024_recovery',stage]
