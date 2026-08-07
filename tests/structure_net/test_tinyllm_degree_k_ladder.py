import math

import numpy as np
import torch

from experiments.structure_net.tinyllm_degree_k_ladder import (
    DegreeKLadderConfig,
    _branch_dataset,
    _targets,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def test_degree_targets_wrap_k_times():
    phase = np.linspace(0.0, 2.0 * math.pi, 97, endpoint=False)
    for k in (1, 2, 3):
        posterior, bins = _targets(phase, k, 16)
        assert posterior.shape == (97, 16)
        assert torch.allclose(posterior.sum(1), torch.ones(97), atol=1e-6)
        assert bins.min() >= 0 and bins.max() < 16


def test_branch_dataset_has_exact_uniform_zk_fibers():
    task = CircleTaskConfig(train_samples=96)
    for k in (2, 3):
        _, labels, condition = _branch_dataset(
            task, k=k, count=96, seed=7, regime="composition"
        )
        assert np.array_equal(np.bincount(labels, minlength=k), np.full(k, len(labels) // k))
        assert condition.shape == (len(labels), 2)


def test_underpowered_ladder_config_is_explicit():
    config = DegreeKLadderConfig(
        seeds=(7,), training_steps=2, train_samples=48, batch_size=12,
        allow_underpowered=True, phase_grid_points=48, trace_phase_points=32,
    )
    assert config.degrees == (1, 2, 3)
