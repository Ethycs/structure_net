import math

import torch

from experiments.structure_net.tinyllm_irrep_fusion_ablation import (
    deck_fourier_components,
    phase_phenotype,
    reconstruct_character_phase,
    substitute_orbit_carrier,
)


def test_true_deck_phases_only_permute_reconstructed_sheets() -> None:
    torch.manual_seed(11)
    for k in (2, 3):
        sheets = torch.randn(5, k, 2, 4)
        for shift in range(k):
            rotated, imaginary = reconstruct_character_phase(
                sheets, 2.0 * math.pi * shift / k
            )
            expected = sheets.roll(-shift, dims=1)
            assert imaginary < 1e-12
            assert torch.allclose(rotated, expected, atol=1e-6)


def test_continuous_c3_phase_preserves_barycenter_and_character_energy() -> None:
    torch.manual_seed(17)
    sheets = torch.randn(7, 3, 3, 5)
    rotated, imaginary = reconstruct_character_phase(sheets, math.pi / 5.0)
    assert imaginary < 1e-12
    assert torch.allclose(rotated.mean(1), sheets.mean(1), atol=1e-6)
    before = torch.square(torch.abs(deck_fourier_components(sheets))).sum(dim=(0, 2))
    after = torch.square(torch.abs(deck_fourier_components(rotated))).sum(dim=(0, 2))
    assert torch.allclose(before, after, atol=1e-5)


def test_substituted_carrier_preserves_barycenter_zero_mean_and_norm() -> None:
    torch.manual_seed(23)
    sheets = torch.randn(9, 3, 2, 6)
    substituted = substitute_orbit_carrier(sheets)
    assert torch.allclose(substituted.mean(1), sheets.mean(1), atol=1e-6)
    original_delta = sheets - sheets.mean(1, keepdim=True)
    substituted_delta = substituted - substituted.mean(1, keepdim=True)
    assert torch.allclose(substituted_delta.mean(1), torch.zeros_like(substituted_delta[:, 0]), atol=1e-6)
    original_norm = torch.square(original_delta).sum(dim=(1, 2, 3)).sqrt()
    substituted_norm = torch.square(substituted_delta).sum(dim=(1, 2, 3)).sqrt()
    assert torch.allclose(original_norm, substituted_norm, atol=1e-5)


def test_phase_phenotype_uses_frozen_thresholds() -> None:
    assert phase_phenotype(None, 0.10, 0.25) == "degenerate"
    assert phase_phenotype(0.10, 0.10, 0.25) == "radial"
    assert phase_phenotype(0.17, 0.10, 0.25) == "mixed"
    assert phase_phenotype(0.25, 0.10, 0.25) == "finite_group_phase_sensitive"
