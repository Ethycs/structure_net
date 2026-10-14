import pytest
import torch
from dataclasses import asdict
from experiments.structure_net.llama_noise_tensor_continuation import (
    Config, balanced_hidden_loss, noisy_pair, output_objective, perturb_factors, schedule)
from experiments.structure_net.llama_neighbourhood_tensorization import digest, FactorLinear, factor_state, install_factors


def test_noise_pair_recomputes_teacher_and_clean_is_exact():
    x = torch.ones(1, 9, 8)
    old = x * 3
    calls = []
    def teacher(value):
        calls.append(value.clone())
        return value * 3
    clean_x, clean_y = noisy_pair(x, old, teacher, 0., torch.Generator().manual_seed(3))
    assert clean_x is x and torch.equal(clean_y, old) and not calls
    changed, target = noisy_pair(x, old, teacher, .05, torch.Generator().manual_seed(3))
    assert torch.equal(calls[0], changed) and torch.equal(target, changed * 3)
    assert not torch.equal(changed, x) and not target.requires_grad


def test_noise_is_rms_scaled_and_streams_separate():
    a = torch.nn.Parameter(torch.ones(100000) * 2)
    b = torch.nn.Parameter(a.detach().clone())
    r = perturb_factors([a], .01, torch.Generator().manual_seed(7))
    perturb_factors([b], .01, torch.Generator().manual_seed(7))
    assert torch.equal(a, b) and .0195 < r[0]["noise_rms"] < .0205
    saved = a.clone()
    perturb_factors([a], 0., torch.Generator().manual_seed(8))
    assert torch.equal(a, saved)
    assert schedule(64, 256, 3) == schedule(64, 256, 3)
    assert len(set(schedule(64, 256, 3)[:64])) == 64


def test_response_balancing_and_energy_floor():
    target = torch.ones(1, 10, 4, requires_grad=True)
    response_bad = torch.ones(1, 10, 4)
    response_bad[:, 7:9] += 1
    prefix_bad = torch.ones(1, 10, 4)
    prefix_bad[:, :2] += 1
    assert balanced_hidden_loss(response_bad, target, 8) > balanced_hidden_loss(prefix_bad, target, 8)
    p = response_bad.clone().requires_grad_()
    loss = balanced_hidden_loss(p, target, 8)
    loss.backward()
    assert p.grad is not None and target.grad is None
    assert torch.isfinite(balanced_hidden_loss(torch.ones(1, 3, 4), torch.zeros(1, 3, 4), 2))


def test_global_alignment_detached_teacher_and_loss_sign():
    s = torch.tensor([[[3., 0., -1.], [0., 3., -1.]]], requires_grad=True)
    t = s.detach().clone().requires_grad_()
    labels = torch.tensor([[0, 1]])
    loss, kl, ce = output_objective(s, t, labels)
    assert abs(float(kl)) < 1e-6 and ce > 0
    loss.backward()
    assert s.grad is not None and t.grad is None
    with pytest.raises(ValueError):
        output_objective(s, t, labels[:, :1])
    with pytest.raises(ValueError):
        Config(input_noise=float("nan"))


def test_configuration_fingerprint_and_exact_factor_roundtrip(tmp_path):
    assert digest(asdict(Config())) != digest(asdict(Config(seed=41015)))
    assert digest(asdict(Config())) != digest(asdict(Config(input_noise=.06)))
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(self.mlp, name, FactorLinear(torch.randn(8, 3).half(), torch.randn(3, 8).half()))
    blocks = torch.nn.ModuleList([Block(), Block()])
    path = tmp_path / "factors.pt"
    original = factor_state(blocks)
    torch.save(original, path)
    install_factors(blocks, torch.load(path, weights_only=True), "cpu")
    actual = factor_state(blocks)
    assert all(torch.equal(original[k], actual[k]) for k in original)
    assert not any(p.requires_grad for p in blocks.parameters())
