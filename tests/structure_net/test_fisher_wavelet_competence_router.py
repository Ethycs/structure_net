from __future__ import annotations

import numpy as np

from experiments.structure_net.fisher_wavelet_competence_router import (
    Config,
    build_campaign,
    competence,
    fisher_rao_distance,
    fisher_wavelet_basis,
    states,
    teacher_probabilities,
)


def test_fisher_rao_and_wavelet_geometry_contracts() -> None:
    probability = teacher_probabilities(states())
    assert fisher_rao_distance(probability[0], probability[0]) <= 1e-7
    assert fisher_rao_distance(probability[0], probability[1]) > 0.0
    basis, diagnostics = fisher_wavelet_basis(Config())
    assert basis.shape == (8, 4)
    assert diagnostics["adjacency_edges"] == 12
    assert diagnostics["minimum_weighted_degree"] > 0.0
    assert diagnostics["eigenvector_orthonormality_error"] <= 1e-10


def test_competence_is_not_nested() -> None:
    signatures = {tuple(row.astype(int)) for row in competence(states())}
    assert (1, 0, 0) in signatures
    assert (1, 0, 1) in signatures
    assert (0, 1, 0) in signatures


def test_preregistered_campaign_gates_pass() -> None:
    campaign = build_campaign()
    assert campaign["valid"] is True
    assert campaign["aggregate"]["improving_seed_count"] >= 4
    assert campaign["aggregate"]["mean_relative_loss_reduction"] >= 0.20
    assert campaign["ood"]["routed_to_c"] == campaign["ood"]["sample_count"]
    assert np.isfinite(campaign["aggregate"]["mean_wavelet_loss"])
