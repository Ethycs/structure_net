import torch

from experiments.structure_net.ncm_factorization import BiasedFactorLinear,boundary_loss


def test_factorization_preserves_bias_exactly_and_routes_gradients_only_to_factors():
    torch.manual_seed(7)
    linear=torch.nn.Linear(24,40)
    linear.weight.data.copy_(torch.randn(40,3)@torch.randn(3,24))
    factor=BiasedFactorLinear.from_linear(linear,torch.linspace(.5,2,24),3,19)
    x=torch.randn(5,24)
    torch.testing.assert_close(factor(x),linear(x),rtol=2e-5,atol=2e-5)
    assert torch.equal(factor.bias,linear.bias) and not factor.bias.requires_grad
    loss=boundary_loss(factor(x),torch.randn(5,40,requires_grad=True))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in (factor.left,factor.right))
    assert factor.bias.grad is None


def test_factor_checkpoint_reload_is_exact_and_target_gets_no_gradient():
    torch.manual_seed(9)
    a=BiasedFactorLinear(torch.randn(8,3),torch.randn(3,12),torch.randn(8))
    b=BiasedFactorLinear(torch.zeros(8,3),torch.zeros(3,12),torch.zeros(8))
    b.load_state_dict(a.state_dict(),strict=True)
    x=torch.randn(7,12);target=torch.randn(7,8,requires_grad=True)
    assert torch.equal(a(x),b(x))
    boundary_loss(a(x),target).backward()
    assert target.grad is None
