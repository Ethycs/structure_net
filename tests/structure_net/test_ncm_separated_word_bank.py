from types import SimpleNamespace
import pytest
import torch
from torch import nn

from experiments.structure_net.ncm_native_query_cache import QueryLoraLinear
from experiments.structure_net.ncm_separated_word_bank import SeparatedQueryLora, word_bank


def test_zero_word_identity_old_bank_preservation_and_word_only_gradients():
    torch.manual_seed(1)
    old = QueryLoraLinear(nn.Linear(8, 8), rank=2)
    with torch.no_grad():
        old.b.normal_()
    old.enabled = True
    module = SeparatedQueryLora(old, word_rank=2)
    inputs = torch.randn(2, 5, 8)
    backbone = SimpleNamespace(adapters=[module])
    with torch.no_grad():
        expected_coarse = old(inputs)
        assert torch.equal(module(inputs), expected_coarse)
        with word_bank(backbone):
            assert torch.equal(module(inputs), module.base(inputs))
    optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with word_bank(backbone):
            loss = module(inputs).square().mean()
        loss.backward()
        assert module.word_b.grad.abs().sum() > 0
        assert module.a.grad is None and module.b.grad is None
        assert all(p.grad is None for p in module.base.parameters())
        optimizer.step()
    with torch.no_grad():
        assert torch.equal(module(inputs), expected_coarse)
        with word_bank(backbone):
            assert not torch.equal(module(inputs), module.base(inputs))


def test_context_restores_bank_after_exception():
    old = QueryLoraLinear(nn.Linear(3, 3), rank=1)
    module = SeparatedQueryLora(old, word_rank=1)
    backbone = SimpleNamespace(adapters=[module])
    with pytest.raises(RuntimeError):
        with word_bank(backbone):
            assert module.bank == "word"
            raise RuntimeError("test")
    assert module.bank == "coarse"
