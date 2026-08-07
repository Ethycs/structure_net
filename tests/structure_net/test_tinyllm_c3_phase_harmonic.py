import math

import torch

from experiments.structure_net.tinyllm_c3_phase_harmonic import (
    allowed_harmonic_reconstructions,
    phase_dft,
    spectral_metrics,
)


def _synthetic_phase_response() -> torch.Tensor:
    theta = torch.arange(24, dtype=torch.float64) * (2.0 * math.pi / 24.0)
    constant = torch.tensor([1.2, -0.4], dtype=torch.float64)
    h3 = torch.tensor([0.5, 0.2], dtype=torch.float64)
    h6 = torch.tensor([-0.1, 0.3], dtype=torch.float64)
    return (
        constant[None]
        + torch.cos(3.0 * theta)[:, None] * h3[None]
        + torch.sin(6.0 * theta)[:, None] * h6[None]
    )


def test_c3_spectrum_has_no_forbidden_character_frequency() -> None:
    values = _synthetic_phase_response()
    coefficients = phase_dft(values)
    metrics = spectral_metrics(values, coefficients)
    assert metrics["forbidden_variation_energy_fraction"] < 1e-24
    assert metrics["maximum_deck_periodicity_relative_error"] < 1e-12
    assert metrics["theta_zero_reconstruction_relative_error"] < 1e-12


def test_nested_allowed_harmonics_reconstruct_theta_zero() -> None:
    values = _synthetic_phase_response()
    reconstructions, _ = allowed_harmonic_reconstructions(values)
    assert torch.allclose(reconstructions["phase_twirl"], torch.tensor([1.2, -0.4], dtype=torch.float64), atol=1e-12)
    expected_first = torch.tensor([1.7, -0.2], dtype=torch.float64)
    assert torch.allclose(reconstructions["allowed_prefix_3"], expected_first, atol=1e-12)
    assert torch.allclose(reconstructions["allowed_prefix_12"], values[0], atol=1e-12)
    assert torch.allclose(reconstructions["full_dft"], values[0], atol=1e-12)


def test_forbidden_frequency_is_detected() -> None:
    values = _synthetic_phase_response()
    theta = torch.arange(24, dtype=torch.float64) * (2.0 * math.pi / 24.0)
    values[:, 0] += 0.25 * torch.cos(theta)
    metrics = spectral_metrics(values, phase_dft(values))
    assert metrics["forbidden_variation_energy_fraction"] > 0.01
    assert metrics["maximum_deck_periodicity_relative_error"] > 0.01
