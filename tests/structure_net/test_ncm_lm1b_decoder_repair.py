from dataclasses import replace

import pytest
import torch
from torch import nn

from experiments.structure_net import ncm_lm1b_decoder_repair as run


def test_native_augmentation_is_deterministic_and_uses_genuine_boundaries():
    frame = torch.tensor([101] + list(range(1000, 1126)) + [101])
    a = torch.Generator().manual_seed(9); b = torch.Generator().manual_seed(9)
    shortened = replace(run.Config(), native_probability=0.)
    x, meta = run.augment(frame, a, shortened)
    y, same_meta = run.augment(frame, b, shortened)
    assert torch.equal(x, y) and meta == same_meta
    n = meta["length"]
    assert len(x) == 128 and x[0] == x[n + 1] == 101
    assert x[1:n + 1].tolist() == frame[1:n + 1].tolist() and not x[n + 2:].any()
    original, meta = run.augment(frame, a, replace(run.Config(), native_probability=1.))
    assert torch.equal(original, frame) and meta["mode"] == "native"


def test_only_decoder_parameters_receive_gradients_or_training_mode():
    codec = nn.Module(); codec.encoder = nn.Linear(4, 4); codec.vq = nn.Linear(4, 4); codec.decoder = nn.Linear(4, 8)
    run.decoder_training(codec)
    assert not codec.training and not codec.encoder.training and not codec.vq.training and codec.decoder.training
    x = codec.vq(codec.encoder(torch.randn(2, 4)))
    codec.decoder(x).square().sum().backward()
    assert all(p.grad is None for p in codec.encoder.parameters()) and all(p.grad is None for p in codec.vq.parameters())
    assert all(p.grad is not None for p in codec.decoder.parameters())


def test_fixed_schedule_and_nested_checkpoint_state_comparison():
    c = run.Config()
    assert run.rate(c.warmup) == c.max_lr and run.rate(c.updates) == c.min_lr
    with pytest.raises(ValueError): run.rate(0)
    assert run.same({"rng": [torch.arange(3)]}, {"rng": [torch.arange(3)]})
    assert not run.same({"rng": [torch.arange(3)]}, {"rng": [torch.arange(4)]})
