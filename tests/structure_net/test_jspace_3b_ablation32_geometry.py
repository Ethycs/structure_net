"""Projector algebra, real QR gradients, and original-width RMSNorm folding."""
import math

import pytest
import torch

from experiments.structure_net import jspace_3b_ablation32_geometry as geometry


def module(seed=220101, **kwargs):
    return geometry.DiscardedSubspace(8, 4, seed=seed, **kwargs)


def test_discarded_and_retained_spaces_are_complementary_and_keep_dc():
    result = module().export()
    d, b = result['discarded'], result['retained']
    torch.testing.assert_close(d.T @ d, torch.eye(4, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(b.T @ d, torch.zeros((4, 4), dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(b @ b.T + d @ d.T, torch.eye(8, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(b[:, 0], torch.full((8,), 1 / math.sqrt(8), dtype=torch.float64))
    x = torch.arange(24, dtype=torch.float64).reshape(3, 8)
    projected = geometry.project_with_discarded(x, d)
    torch.testing.assert_close(projected.mean(-1), x.mean(-1), atol=1e-12, rtol=0)
    torch.testing.assert_close(projected, projected @ b @ b.T, atol=1e-12, rtol=0)
    assert result['metadata']['rms_denominator_width'] == 8


def test_qr_search_has_correct_gradients_and_window_vjp_matches_direct_chain():
    direct, accumulated = module(), module()
    generator = torch.Generator().manual_seed(4)
    x = torch.randn((2, 3, 8), generator=generator, dtype=torch.float64)
    target = torch.randn(x.shape, generator=generator, dtype=torch.float64)
    direct_d = direct.discarded_basis()
    sum((geometry.project_with_discarded(x[i], direct_d) * target[i]).sum() for i in range(2)).backward()
    d = accumulated.discarded_basis()
    leaf = d.detach().requires_grad_()
    for i in range(2):
        (geometry.project_with_discarded(x[i], leaf) * target[i]).sum().backward()
    d.backward(leaf.grad)
    torch.testing.assert_close(accumulated.raw.grad, direct.raw.grad, atol=1e-11, rtol=1e-11)
    assert direct.raw.grad.norm() > 0
    # Finite-difference one actual raw QR parameter, not just the projector leaf.
    direction = torch.randn(direct.raw.shape, generator=generator, dtype=torch.float64)
    expected = (direct.raw.grad * direction).sum()
    original = direct.raw.detach().clone()
    values = []
    for sign in (-1, 1):
        with torch.no_grad():
            direct.raw.copy_(original + sign * 1e-6 * direction)
            values.append(sum((direct(x[i]) * target[i]).sum() for i in range(2)))
    torch.testing.assert_close((values[1] - values[0]) / 2e-6, expected, atol=1e-7, rtol=1e-7)


def test_export_is_detached_repeatable_and_initialization_does_not_change_global_rng():
    before = torch.random.get_rng_state().clone()
    first, second = module(), module()
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(first.raw, second.raw)
    assert torch.equal(first.retained_basis(), first.retained_basis())
    assert not first.retained_basis().requires_grad
    assert not torch.equal(first.raw, module(seed=220102).raw)
    for column in first.retained_basis()[:, 1:].T:
        assert column[column.abs().argmax()] >= 0


def test_affine_projection_and_nonuniform_rms_scale_fold_exactly():
    b = module().retained_basis()
    generator = torch.Generator().manual_seed(6)
    z = torch.randn((3, 4), generator=generator, dtype=torch.float64)
    mu = torch.randn((8,), generator=generator, dtype=torch.float64)
    weight = torch.randn((5, 8), generator=generator, dtype=torch.float64)
    scale = torch.randn((8,), generator=generator, dtype=torch.float64)
    bias = torch.randn((5,), generator=generator, dtype=torch.float64)
    x = mu + z @ b.T
    denominator = geometry.rms_denominator(z, original_width=8, eps=1e-6, retained_basis=b, anchor=mu)
    torch.testing.assert_close(denominator, (x.square().mean(-1, keepdim=True) + 1e-6).sqrt())
    folded, offset = geometry.fold_rms_linear_weights(weight, scale, b, anchor=mu)
    native = ((x / denominator) * scale) @ weight.T + bias
    compact = (z @ folded.T + offset) / denominator + bias
    torch.testing.assert_close(compact, native, atol=1e-12, rtol=1e-12)
    projected = geometry.project_with_discarded(x, module().discarded_basis(), anchor=mu)
    torch.testing.assert_close(projected, x, atol=1e-12, rtol=1e-12)
    # Width1024 RMSNorm is not an equivalent replacement for width2048 RMSNorm.
    unanchored = geometry.rms_denominator(z, original_width=8, eps=1e-6)
    assert not torch.allclose(unanchored, (z.square().mean(-1, keepdim=True) + 1e-6).sqrt())


def test_rmsnorm_does_not_have_layernorms_mean_shift_invariance():
    x = torch.tensor([[1., 2., -3., 5.]], dtype=torch.float64)
    rms = lambda value: value / (value.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    assert not torch.allclose(rms(x), rms(x - x.mean(-1, keepdim=True)))


def test_position_anchors_broadcast_without_expanding_batch_or_token_axes():
    m = module()
    b, d = m.retained_basis(), m.discarded_basis()
    generator = torch.Generator().manual_seed(8)
    z = torch.randn((2, 3, 4), generator=generator, dtype=torch.float64)
    mu = torch.randn((3, 8), generator=generator, dtype=torch.float64)
    x = mu + z @ b.T
    torch.testing.assert_close(geometry.project_with_discarded(x, d, anchor=mu), x)
    torch.testing.assert_close(geometry.project_with_discarded(x, d, anchor=mu[None]), x)
    denominator = geometry.rms_denominator(z, original_width=8, eps=1e-6, retained_basis=b, anchor=mu)
    torch.testing.assert_close(denominator, (x.square().mean(-1, keepdim=True) + 1e-6).sqrt())
    weight = torch.randn((5, 8), generator=generator, dtype=torch.float64)
    scale = torch.randn((8,), generator=generator, dtype=torch.float64)
    folded, offset = geometry.fold_rms_linear_weights(weight, scale, b, anchor=mu)
    torch.testing.assert_close((z @ folded.T + offset) / denominator, ((x / denominator) * scale) @ weight.T)
    exported = m.export(anchor=mu)
    assert exported['anchor'].shape == (3, 8) and exported['raw'].shape == (7, 4)
    with pytest.raises(ValueError, match='expand'):
        geometry.project_with_discarded(x[:1], d, anchor=torch.ones((2, 3, 8)))
    with pytest.raises(ValueError, match='broadcast'):
        geometry.project_with_discarded(x, d, anchor=torch.ones((4, 8)))


def test_full_discard_is_explicit_and_bf16_application_has_real_gradient():
    m = module(preserve_mean=False)
    assert m.raw.shape == (8, 4)
    assert m.export()['metadata']['retained_dc_column'] is None
    d = module().discarded_basis().float().detach().requires_grad_()
    x = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    output = geometry.project_with_discarded(x, d)
    output.float().square().sum().backward()
    assert output.dtype == torch.bfloat16
    assert d.grad is not None and torch.isfinite(d.grad).all() and d.grad.norm() > 0


def test_rank_deficiency_and_invalid_geometry_fail_instead_of_silent_replacement():
    m = module()
    with torch.no_grad():
        m.raw.zero_()
    with pytest.raises(ValueError, match='rank deficient'):
        m.discarded_basis()
    with pytest.raises(ValueError):
        geometry.DiscardedSubspace(8, 0, seed=1)
    with pytest.raises(ValueError, match='Anchor'):
        geometry.project_with_discarded(torch.ones(2, 8), torch.ones(8, 4), anchor=torch.ones(4))


def test_full_qwen_memory_inventory_is_not_a_gpu_fit_claim():
    inventory = geometry.memory_inventory()
    assert inventory['raw_parameters'] == 603684864
    assert inventory['raw_gradients'] == 603684864
    assert inventory['adam_first_and_second_moments'] == 1207369728
    assert inventory['qr_discarded_outputs'] == 603979776
    assert inventory['application_all_interfaces'] == 301989888
    assert inventory['application_one_interface'] == 8388608
