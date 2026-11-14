from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net.jspace_packed_core import StackedCoreMLP
from experiments.structure_net.jspace_groebner_tensor import features


@pytest.mark.parametrize("removed", [[], [1, 5, 17]])
def test_dense_stack_matches_direct_neuron_substitution_and_keeps_no_removed_gate_rows(removed):
    torch.manual_seed(815)
    original = SimpleNamespace(gate_proj=nn.Linear(7, 23, bias=False).double(),
                               up_proj=nn.Linear(7, 23, bias=False).double(),
                               down_proj=nn.Linear(23, 7, bias=False).double())
    fit = {"anchors": [0, 3, 7], "mean": torch.randn(23, dtype=torch.double),
           "scale": torch.rand(23, dtype=torch.double)+1,
           "coefficients": {"2": torch.randn(23, 10, dtype=torch.double)/16}}
    packed = StackedCoreMLP(original, fit, removed)
    x = torch.randn(2, 4, 7, dtype=torch.double, requires_grad=True)
    h = F.silu(original.gate_proj(x))*original.up_proj(x)
    if removed:
        z = (h[..., fit["anchors"]]-fit["mean"][fit["anchors"]])/fit["scale"][fit["anchors"]]
        prediction = F.linear(features(z, 2), fit["coefficients"]["2"][removed])
        h = h.clone()
        h[..., removed] = prediction*fit["scale"][removed]+fit["mean"][removed]
    target = original.down_proj(h)
    actual = packed.reference(x)
    assert torch.allclose(actual, target, atol=1e-12, rtol=1e-12)
    da = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    dt = torch.autograd.grad(target.sum(), x)[0]
    assert torch.allclose(da, dt, atol=1e-12, rtol=1e-12)
    assert packed.kept == 23-len(removed)
    assert packed.padded % 16 == 0
    for start in (packed.kept, packed.padded+packed.kept):
        assert not torch.count_nonzero(packed.gate_up_weight[start:start+packed.padded-packed.kept])
