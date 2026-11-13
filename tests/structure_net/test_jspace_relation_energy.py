import pytest
import torch

from experiments.structure_net.jspace_relation_energy import span_energy


def test_half_energy_has_sqrt_half_norm_and_exact_orthogonal_accounting():
    result = span_energy(torch.tensor([[1., 1.], [-1., -1.]]), torch.tensor([[1., 0.]]), torch.zeros(2))
    assert result["span_rank"] == 1
    assert result["raw"]["projected_energy_fraction"] == pytest.approx(.5)
    assert result["raw"]["residual_energy_fraction"] == pytest.approx(.5)
    assert result["raw"]["pooled_relative_residual_norm"] == pytest.approx(.5**.5)
    assert result["raw"]["relative_pythagorean_error"] < 1e-12


def test_energy_depends_on_span_not_nonorthogonal_row_scaling():
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn(50, 5, generator=generator)
    rows = torch.eye(5)[:2]
    mixed = torch.tensor([[10., 3.], [1., 2.]]) @ rows
    first = span_energy(hidden, rows, torch.zeros(5))
    second = span_energy(hidden, mixed, torch.zeros(5))
    assert first["raw"]["projected_energy_fraction"] == pytest.approx(second["raw"]["projected_energy_fraction"])
    duplicated = span_energy(hidden, torch.cat((mixed, mixed[:1])), torch.zeros(5))
    assert duplicated["span_rank"] == 2
    assert duplicated["raw"]["residual_energy_fraction"] == pytest.approx(first["raw"]["residual_energy_fraction"])


def test_raw_mean_energy_and_centered_variation_are_distinct():
    hidden = torch.tensor([[100., 1.], [100., -1.]])
    result = span_energy(hidden, torch.tensor([[0., 1.]]), torch.tensor([100., 0.]))
    assert result["raw"]["projected_energy_fraction"] == pytest.approx(1/10001)
    assert result["around_training_mean"]["projected_energy_fraction"] == pytest.approx(1.)
    # A shifted heldout mean must not be fitted away by the diagnostic.
    shifted = span_energy(hidden + torch.tensor([2., 0.]), torch.tensor([[0., 1.]]), torch.tensor([100., 0.]))
    assert shifted["around_training_mean"]["projected_energy_fraction"] == pytest.approx(.2)


def test_invalid_input_and_zero_energy_are_explicit():
    with pytest.raises(ValueError, match="finite"):
        span_energy(torch.full((2, 4), float("nan")), torch.eye(4), torch.zeros(4))
    zero = span_energy(torch.zeros(2, 4), torch.eye(4)[:1], torch.zeros(4))
    assert zero["raw"]["projected_energy_fraction"] is None
