from types import SimpleNamespace
import pytest
import torch
from torch import nn

from experiments.structure_net.ncm_native_query_cache import QueryLoraLinear
from experiments.structure_net.ncm_separated_word_bank import SeparatedQueryLora, word_bank
from experiments.structure_net.ncm_quality_word_runtime import static_word_runtime


@pytest.mark.parametrize("zero", [False, True])
def test_static_word_operations_exact_and_no_repeated_zero_reductions(monkeypatch, zero):
    torch.manual_seed(3)
    module = SeparatedQueryLora(QueryLoraLinear(nn.Linear(8, 8), rank=2), word_rank=2)
    module.enabled = True
    if not zero:
        with torch.no_grad():
            module.word_b.normal_()
    backbone = SimpleNamespace(adapters=[module])
    x = torch.randn(2, 5, 8)
    with torch.no_grad(), word_bank(backbone):
        expected = module(x)
        with static_word_runtime(backbone):
            with monkeypatch.context() as patch:
                def forbidden(*args, **kwargs):
                    raise AssertionError("repeated reduction")
                patch.setattr(torch, "count_nonzero", forbidden)
                assert torch.equal(module(x), expected)
                assert torch.equal(module(x), expected)
        assert "forward" not in module.__dict__


def test_static_runtime_rejects_mutated_parameters_and_restores_method():
    module = SeparatedQueryLora(QueryLoraLinear(nn.Linear(3, 3), rank=1), word_rank=1)
    module.enabled = True
    backbone = SimpleNamespace(adapters=[module])
    with torch.no_grad(), word_bank(backbone):
        with pytest.raises(ValueError, match="changed"):
            with static_word_runtime(backbone):
                module.word_b.add_(.1)
                module(torch.ones(1, 3))
    assert "forward" not in module.__dict__
