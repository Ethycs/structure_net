import math

import pytest
import torch

from experiments.structure_net.jspace_layer_distill import (
    affine_project_native, coefficient_from_stats, full_sequence_vjp_loss,
    gradient_statistics, projected_boundary_mse, rms_readout_jacobian_loss,
)


def test_affine_projection_keeps_nonzero_anchor_and_position_rows():
    u = torch.tensor([[1.], [0.]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[4., 3.], [-2., 7.], [9., -1.]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[[5., 8.], [6., 9.]]], dtype=torch.float64, requires_grad=True)
    result = affine_project_native(y, u, a, torch.tensor([2, 0]))
    assert torch.equal(result, torch.tensor([[[5., -1.], [6., 3.]]], dtype=torch.float64))
    result.sum().backward()
    assert torch.equal(y.grad, torch.tensor([[[1., 0.], [1., 0.]]], dtype=torch.float64))
    assert u.grad is None and a.grad is None


def test_projected_loss_reports_irreducible_floor_and_detaches_teacher():
    u = torch.tensor([[1.], [0.]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([0., 3.], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[5., 9.]], dtype=torch.float64, requires_grad=True)
    student = torch.tensor([[2., 3.]], dtype=torch.float64, requires_grad=True)
    retained, floor = projected_boundary_mse(student, teacher, u, a)
    assert retained.item() == 4.5 and floor.item() == 18.
    assert torch.equal(retained + floor, (student - teacher).square().mean())
    assert not floor.requires_grad
    retained.backward()
    assert torch.equal(student.grad, torch.tensor([[-3., 0.]], dtype=torch.float64))
    assert teacher.grad is None and u.grad is None and a.grad is None


def test_vjp_loss_equals_analytic_jacobian_quadratic_with_cross_token_terms():
    student = torch.tensor([[2., 1.], [-1., 3.]], dtype=torch.float64, requires_grad=True)
    teacher = torch.zeros_like(student, requires_grad=True)
    matrix = torch.tensor([[1., 2., 3., 4.], [2., -1., -2., 1.]], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(lambda x: matrix @ x.flatten(), teacher)
    probes = jacobian.clone().requires_grad_()
    loss = full_sequence_vjp_loss(student, teacher, probes)
    expected = ((matrix @ student.flatten()) ** 2).mean()
    tokenwise_wrong = torch.einsum("ktd,td->kt", probes, student).square().sum(-1).mean()
    assert torch.equal(loss, expected) and not torch.equal(loss, tokenwise_wrong)
    loss.backward()
    assert torch.equal(student.grad, (matrix.T @ (matrix @ student.detach().flatten())).reshape_as(student))
    assert teacher.grad is None and probes.grad is None


def test_vjp_preserves_opposite_cross_token_cancellation():
    student = torch.tensor([[[1.], [-1.]]], requires_grad=True)
    teacher = torch.zeros_like(student)
    probes = torch.ones(1, 1, 2, 1)
    loss = full_sequence_vjp_loss(student, teacher, probes)
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_fp32_loss_arithmetic_under_bf16_autocast():
    student = torch.tensor([[2., 3.]], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([[1., 5.]], dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        projected = affine_project_native(teacher, torch.eye(2), torch.zeros(2))
        loss, floor = projected_boundary_mse(student, teacher, torch.eye(2), torch.zeros(2))
        vjp = full_sequence_vjp_loss(student, teacher, torch.ones(1, 1, 2, dtype=torch.bfloat16))
    assert projected.dtype == loss.dtype == floor.dtype == vjp.dtype == torch.float32
    (loss + vjp).backward()
    assert torch.isfinite(student.grad).all()


def test_gradient_statistics_matches_analytic_values_without_grad_mutation():
    p = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    unused = torch.tensor([4.], dtype=torch.float64, requires_grad=True)
    frozen = torch.tensor([8.], dtype=torch.float64)
    p.grad = torch.tensor([19., 23.], dtype=torch.float64)
    before = p.grad.clone()
    task, local = p.square().sum(), p[0] - 2 * p[1]
    stats = gradient_statistics(task, local, [p, unused, frozen])
    assert stats == pytest.approx({"task_grad_norm": math.sqrt(20), "local_grad_norm": math.sqrt(5),
                                   "grad_dot": -6., "grad_cosine": -.6})
    assert torch.equal(p.grad, before) and unused.grad is None and frozen.grad is None
    (task + .1 * local).backward()
    assert torch.allclose(p.grad, before + torch.tensor([2.1, 3.8], dtype=torch.float64))


def test_gradient_statistics_handles_constant_unused_and_zero_losses():
    p = torch.tensor([1.], requires_grad=True)
    unrelated = torch.tensor([2.], requires_grad=True)
    stats = gradient_statistics(torch.tensor(4.), unrelated.sum(), [p])
    assert stats == {"task_grad_norm": 0., "local_grad_norm": 0., "grad_dot": 0., "grad_cosine": 0.}
    zero = gradient_statistics((p * 0).sum(), p.square().sum(), [p])
    assert zero["task_grad_norm"] == zero["grad_cosine"] == 0.
    assert p.grad is None and unrelated.grad is None


def test_coefficient_uses_aggregate_rms_norms_not_mean_per_case_ratios():
    rows = [{"task_grad_norm": 3., "local_grad_norm": 1.},
            {"task_grad_norm": 4., "local_grad_norm": 2.}]
    assert coefficient_from_stats(rows) == pytest.approx(.1 * 5 / math.sqrt(5))
    assert coefficient_from_stats([{"task_grad_norm": 0., "local_grad_norm": 0.}]) == 0.
    assert coefficient_from_stats(rows, target_ratio=0) == 0.
    with pytest.raises(ValueError, match="zero local"):
        coefficient_from_stats([{"task_grad_norm": 1., "local_grad_norm": 0.}])
    with pytest.raises(ValueError, match="At least one"):
        coefficient_from_stats([])
    with pytest.raises(ValueError, match="finite"):
        coefficient_from_stats([{"task_grad_norm": math.nan, "local_grad_norm": 1.}])


def test_finite_shape_and_position_validation():
    y, u, a = torch.ones(1, 2, 2), torch.eye(2), torch.zeros(3, 2)
    with pytest.raises(ValueError, match="outside"):
        affine_project_native(y, u, a, torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="Basis must"):
        affine_project_native(y, torch.eye(3), a, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="finite"):
        full_sequence_vjp_loss(y, y * float("nan"), torch.ones(1, 1, 2, 2))
    with pytest.raises(ValueError, match="Probes must"):
        full_sequence_vjp_loss(y, y, torch.ones(1, 2, 2))
    with pytest.raises(ValueError, match="equal nonempty"):
        projected_boundary_mse(y, y[0], u, a)
    p = torch.tensor([1.], requires_grad=True)
    with pytest.raises(ValueError, match="duplicates"):
        gradient_statistics(p.sum(), p.sum(), [p, p])
    with pytest.raises(ValueError, match="finite"):
        gradient_statistics(torch.tensor(float("inf")), p.sum(), [p])


def test_nonfinite_gradient_is_rejected_even_when_loss_is_finite():
    p = torch.tensor(0., requires_grad=True)
    with pytest.raises(ValueError, match="gradient must be finite"):
        gradient_statistics(p.sqrt(), p.square(), [p])


@pytest.mark.parametrize("epsilon", [.00001, .7])
def test_rms_readout_metric_matches_full_analytic_jacobian_and_gradient(epsilon):
    torch.manual_seed(2401)
    teacher = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    student = torch.randn_like(teacher, requires_grad=True)
    gain = torch.tensor([.5, 1.3, -.7], dtype=torch.float64, requires_grad=True)
    readout = torch.randn(5, 3, dtype=torch.float64)
    gram = (readout.T @ readout).detach().requires_grad_()

    def suffix(x):
        normalized = x / (x.square().mean(-1, keepdim=True) + epsilon).sqrt()
        return (normalized * gain.detach()) @ readout.T

    jacobian = torch.autograd.functional.jacobian(suffix, teacher).reshape(10, 6)
    expected = (jacobian @ (student - teacher.detach()).flatten()).square().mean()
    loss = rms_readout_jacobian_loss(student, teacher, gain, epsilon, gram, 5)
    assert torch.allclose(loss, expected, atol=2e-14, rtol=2e-14)
    actual_gradient = torch.autograd.grad(loss, student, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, student)[0]
    assert torch.allclose(actual_gradient, expected_gradient, atol=2e-14, rtol=2e-14)
    loss.backward()
    assert teacher.grad is None and gain.grad is None and gram.grad is None


def test_rms_readout_metric_retains_nonzero_epsilon_radial_derivative():
    teacher = torch.tensor([[1., 2.]], dtype=torch.float64)
    student = (teacher * 2).requires_grad_()
    gain, gram = torch.ones(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64)
    epsilon = .4
    loss = rms_readout_jacobian_loss(student, teacher, gain, epsilon, gram, 2)
    radius = (teacher.square().mean() + epsilon).sqrt()
    expected = (teacher * epsilon / radius.pow(3)).square().mean()
    assert loss > 0 and torch.allclose(loss, expected, atol=1e-16, rtol=1e-14)
    zero_epsilon = rms_readout_jacobian_loss(student, teacher, gain, 0., gram, 2)
    assert zero_epsilon < 1e-28


def test_rms_readout_metric_fp32_autocast_and_validation():
    student = torch.tensor([[2., 3.], [1., 4.]], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.ones_like(student)
    gain, gram = torch.ones(2), torch.eye(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = rms_readout_jacobian_loss(student, teacher, gain, .1, gram, 2)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    with pytest.raises(ValueError, match="epsilon"):
        rms_readout_jacobian_loss(student, teacher, gain, -.1, gram, 2)
    with pytest.raises(ValueError, match="positive integer"):
        rms_readout_jacobian_loss(student, teacher, gain, .1, gram, 0)
    with pytest.raises(ValueError, match="native boundary width"):
        rms_readout_jacobian_loss(student, teacher, gain, .1, torch.eye(3), 2)
    with pytest.raises(ValueError, match="finite"):
        rms_readout_jacobian_loss(student, teacher, gain, .1, gram * float("nan"), 2)
