from dataclasses import FrozenInstanceError

import pytest
import torch

from experiments.structure_net import ncm_dense_matched_forecaster as run


def test_fixed_budget_and_seed_stream_cover_the_complete_training_population():
    cfg = run.Config()
    assert cfg.updates == 32768 and cfg.snapshot_interval == 8192
    assert len({cfg.model_seed, cfg.sampler_seed, cfg.noise_seed}) == 3
    sampler = torch.Generator().manual_seed(cfg.sampler_seed)
    counts = torch.zeros(690, dtype=torch.long)
    for _ in range(cfg.updates):
        counts[int(torch.randint(690, (1,), generator=sampler))] += 1
    assert int(counts.min()) > 0
    with pytest.raises(FrozenInstanceError):
        cfg.updates = 2


def test_coarse_prediction_history_never_receives_finest_target_codes():
    labels = torch.arange(800)[None]
    history = run.old.prediction_history(labels)
    assert torch.equal(history[:, :608], labels[:, :608])
    assert torch.count_nonzero(history[:, 608:]) == 0
    assert torch.equal(labels, torch.arange(800)[None])


def test_new_budget_has_exact_warmup_and_cosine_endpoints():
    cfg = run.Config()
    assert run.old.clean.rate(cfg.warmup, cfg) == cfg.max_lr
    assert run.old.clean.rate(cfg.updates, cfg) == cfg.min_lr
    assert run.old.clean.rate(1, cfg) == cfg.max_lr / cfg.warmup
