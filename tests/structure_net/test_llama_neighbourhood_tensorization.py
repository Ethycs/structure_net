from dataclasses import asdict
import io
import pytest
import torch

from experiments.structure_net.llama_neighbourhood_tensorization import (
    Config, FactorLinear, digest, normalized_error)


def test_compression_preserves_operator_shape_and_is_differentiable():
    torch.manual_seed(1)
    left, right = torch.randn(24, 3), torch.randn(3, 16)
    weight = left @ right
    factor = FactorLinear.from_weight(weight, torch.linspace(.5, 2, 16), 3, 17)
    x = torch.randn(2, 7, 16)
    torch.testing.assert_close(factor(x), x @ weight.T, atol=3e-5, rtol=1e-4)
    target = torch.randn(2, 7, 24, requires_grad=True)
    normalized_error(factor(x), target).backward()
    assert target.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in factor.parameters())
    assert sum(p.numel() for p in factor.parameters()) < weight.numel()
    buffer = io.BytesIO()
    torch.save(factor.state_dict(), buffer)
    buffer.seek(0)
    reloaded = FactorLinear(torch.zeros_like(factor.left), torch.zeros_like(factor.right))
    reloaded.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(factor(x), reloaded(x), rtol=0, atol=0)


def test_configuration_and_fingerprint_fail_closed():
    with pytest.raises(ValueError):
        Config(rank=0)
    with pytest.raises(ValueError):
        Config(lr=float("nan"))
    with pytest.raises(ValueError):
        digest({"value": float("inf")})
    assert digest(asdict(Config())) != digest(asdict(Config(rank=512)))
    with pytest.raises(ValueError):
        FactorLinear.from_weight(torch.randn(16, 16), torch.ones(16), 9, 1)


def test_initializer_seed_is_local_and_reproducible():
    torch.manual_seed(22)
    w = torch.randn(24, 16)
    before = torch.random.get_rng_state().clone()
    a = FactorLinear.from_weight(w, torch.ones(16), 3, 77)
    assert torch.equal(before, torch.random.get_rng_state())
    b = FactorLinear.from_weight(w, torch.ones(16), 3, 77)
    torch.testing.assert_close(a.left, b.left, rtol=0, atol=0)
    assert normalized_error(torch.ones(2), torch.ones(2)) == 0
