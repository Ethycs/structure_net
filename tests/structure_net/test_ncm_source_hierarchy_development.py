from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from experiments.structure_net import ncm_source_hierarchy_development as dev
from tests.structure_net.test_ncm_source_coverage import toy, forward


def records(arm):
    exposure = dict(presentations=65536, noisy_presentations=32768, distinct_sources_visited=1376)
    trained = dict(arm=arm, status='source_hierarchy_training_completed', stage='train',
        config=asdict(dev.study.CONFIG), updates=2048, cases=1376,
        trainable_parameters=dev.study.mechanism.PARAMETERS[arm], strict_final_reload_passed=True,
        protected_tensors_unchanged=True, systems_lifecycle_only_not_quality_evidence=False,
        checkpoint_sha256='fixed', measured_exposure=exposure)
    audited = dict(arm=arm, status='source_hierarchy_training_replayed',
        all_corruption_counts_replayed=True, checkpoint_sha256='fixed', **exposure)
    return trained, audited


def test_incomplete_or_different_training_cannot_be_evaluated():
    trained, audited = records('tree')
    dev.check_training(trained, audited, 'tree')
    for broken in (dict(audited, arm='dense'), dict(audited, presentations=65535),
                   dict(audited, checkpoint_sha256='other')):
        with pytest.raises(ValueError): dev.check_training(trained, broken, 'tree')
    broken = deepcopy(trained); broken['measured_exposure']['noisy_presentations'] -= 1
    with pytest.raises(ValueError): dev.check_training(broken, audited, 'tree')


@pytest.mark.parametrize('arm', ['tree', 'dense'])
def test_selected_architecture_restores_nonzero_branches_exactly(arm):
    torch.manual_seed(2810)
    original = toy(); dev.study.mechanism.install(original, arm)
    with torch.no_grad():
        for parameter in original.parameters(): parameter.add_(torch.randn_like(parameter)*.03)
    saved = deepcopy(original.state_dict())
    restored = toy(); dev.study.mechanism.install(restored, arm)
    restored.load_state_dict(saved, strict=True)
    original.eval(); restored.eval()
    x, source = torch.randn(2,4,8), torch.randn(2,17,8)
    with torch.no_grad(): assert torch.equal(forward(original,x,source), forward(restored,x,source))
    wrong = toy(); dev.study.mechanism.install(wrong, 'dense' if arm == 'tree' else 'tree')
    with pytest.raises(RuntimeError): wrong.load_state_dict(saved, strict=True)
