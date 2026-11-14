from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.structure_net.jspace_groebner_tensor import PolynomialResidualMLP
from experiments.structure_net.jspace_bench_covered import PackedResidualMLP


@pytest.mark.parametrize("compiled",[True,False])
def test_vectorized_tensor_features_preserve_values_and_gradients(compiled):
    torch.manual_seed(716)
    original=SimpleNamespace(gate_proj=nn.Linear(7,12).double(),up_proj=nn.Linear(7,12).double(),
                             down_proj=nn.Linear(12,7).double(),act_fn=F.silu)
    fit={"anchors":[0,2,4],"mean":torch.randn(12,dtype=torch.double),"scale":torch.rand(12,dtype=torch.double)+1,
         "coefficients":{"1":torch.randn(12,4,dtype=torch.double),"2":torch.randn(12,10,dtype=torch.double)}}
    previous=PolynomialResidualMLP(original,fit,[1,5],compiled=compiled)
    packed=PackedResidualMLP(original,fit,[1,5],compiled=compiled)
    x=torch.randn(2,3,7,dtype=torch.double,requires_grad=True)
    a,b=previous(x),packed(x)
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a.square().sum(),x,retain_graph=True)[0]
    gb=torch.autograd.grad(b.square().sum(),x)[0]
    assert torch.allclose(ga,gb,atol=1e-12,rtol=1e-12)
