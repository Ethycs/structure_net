from __future__ import annotations

import math

import pytest
import torch

from experiments.structure_net import tinyllm_cyclic_writer_intertwiner as cyclic
from experiments.structure_net import tinyllm_fixed_gauge_writer_capacity as capacity


def test_primary_configuration_is_locked() -> None:
    config = cyclic.CyclicWriterIntertwinerConfig()
    assert config.seeds == (7, 29, 53)
    assert config.phase_bins == 16
    assert config.writer_order == 4
    assert config.writer_rank == 3
    assert config.random_controls == 256
    with pytest.raises(ValueError, match="fixed"):
        cyclic.CyclicWriterIntertwinerConfig(seeds=(7,))
    with pytest.raises(ValueError, match="fixed"):
        cyclic.CyclicWriterIntertwinerConfig(
            seeds=(7,), random_controls=8, allow_underpowered=False
        )


def test_cyclic_action_exactly_rotates_fourier_features() -> None:
    theta = torch.tensor([0.1, 1.7, 5.8], dtype=torch.float64)
    carrier = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    features = capacity.fourier_features(carrier, 4)
    alpha = 2.0 * math.pi / 16.0
    shifted = torch.stack(
        (torch.cos(theta + alpha), torch.sin(theta + alpha)), dim=1
    )
    expected = capacity.fourier_features(shifted, 4)
    action = cyclic.cyclic_feature_action(4, 16)
    assert torch.allclose(features @ action, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        torch.linalg.matrix_power(action, 16),
        torch.eye(9, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("harmonic", (1, 2, 3, 4))
def test_each_canonical_harmonic_subspace_is_an_exact_representation(
    harmonic: int,
) -> None:
    config = cyclic.CyclicWriterIntertwinerConfig()
    action = cyclic.cyclic_feature_action(4, 16)
    diagnostics = cyclic.writer_intertwiner_diagnostics(
        cyclic.canonical_harmonic_basis(4, harmonic),
        action,
        16,
        config.numerical_rank_rtol,
    )
    assert diagnostics["writer_rank"] == 3
    assert diagnostics["maximum_nonidentity_obstruction"] < 1e-12
    assert diagnostics["induced_closure_error"] < 1e-12


def test_harmonic_mixture_need_not_form_an_invariant_subspace() -> None:
    writer = torch.zeros(9, 3, dtype=torch.float64)
    writer[0, 0] = 1.0
    writer[2, 0] = 1.0
    writer[1, 1] = 1.0
    writer[4, 1] = 1.0
    writer[-1, 2] = 1.0
    diagnostics = cyclic.writer_intertwiner_diagnostics(
        writer, cyclic.cyclic_feature_action(4, 16), 16, 1e-10
    )
    assert diagnostics["writer_rank"] == 3
    assert diagnostics["maximum_nonidentity_obstruction"] > 0.05


def test_canonical_overlap_identifies_the_exact_harmonic_type() -> None:
    basis = cyclic.canonical_harmonic_basis(4, 3)
    diagnostics = cyclic.writer_intertwiner_diagnostics(
        basis, cyclic.cyclic_feature_action(4, 16), 16, 1e-10
    )
    overlaps = cyclic.canonical_overlaps(diagnostics["projector"], 4)
    assert overlaps["3"] == pytest.approx(1.0)
    assert overlaps["1"] == pytest.approx(1.0 / 3.0)
    assert overlaps["2"] == pytest.approx(1.0 / 3.0)
    assert overlaps["4"] == pytest.approx(1.0 / 3.0)


def test_random_controls_are_deterministic() -> None:
    action = cyclic.cyclic_feature_action(4, 16)
    first = cyclic.random_subspace_obstructions(
        action=action,
        phase_bins=16,
        feature_width=9,
        rank=3,
        count=8,
        seed=91,
        numerical_rank_rtol=1e-10,
    )
    second = cyclic.random_subspace_obstructions(
        action=action,
        phase_bins=16,
        feature_width=9,
        rank=3,
        count=8,
        seed=91,
        numerical_rank_rtol=1e-10,
    )
    assert first == second
    assert all(0.0 <= value <= 1.0 for value in first)


@pytest.mark.parametrize(
    (
        "contracts",
        "rank",
        "obstruction",
        "closure",
        "specificity",
        "expected",
        "passed",
    ),
    [
        (False, 3, True, True, True, "invalid", False),
        (True, 2, True, True, True, "degenerate_writer_image", False),
        (
            True,
            3,
            True,
            True,
            True,
            "approximate_c16_writer_representation",
            True,
        ),
        (True, 3, True, False, True, "nonclosing_induced_action", False),
        (True, 3, False, False, True, "harmonic_mixed_writer_image", False),
        (True, 3, True, True, False, "nonspecific_writer_subspace", False),
    ],
)
def test_checkpoint_classification_is_preregistered(
    contracts: bool,
    rank: int,
    obstruction: bool,
    closure: bool,
    specificity: bool,
    expected: str,
    passed: bool,
) -> None:
    assert cyclic.classify_checkpoint(
        contracts_valid=contracts,
        writer_rank=rank,
        expected_rank=3,
        obstruction_pass=obstruction,
        closure_pass=closure,
        specificity_pass=specificity,
    ) == (expected, passed)


def test_campaign_decision_requires_all_three_checkpoint_gates() -> None:
    supported = cyclic._campaign_decision(
        ["approximate_c16_writer_representation"] * 3, 3, False
    )
    assert supported["supported"] is True
    failed = cyclic._campaign_decision(["harmonic_mixed_writer_image"] * 3, 0, False)
    assert failed["supported"] is False
    assert failed["conclusion"] == "harmonic_mixed_writer_image"


def test_predecessor_campaign_and_writer_results_are_hash_locked() -> None:
    campaign, path, details = cyclic._load_predecessor(
        cyclic.CyclicWriterIntertwinerConfig()
    )
    assert cyclic._sha256(path) == cyclic.PREDECESSOR_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        cyclic.PREDECESSOR_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7, 29, 53}
