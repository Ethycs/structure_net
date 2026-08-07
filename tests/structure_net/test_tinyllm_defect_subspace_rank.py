from __future__ import annotations

import pytest
import torch

from experiments.structure_net.tinyllm_defect_subspace_rank import (
    DefectSubspaceRankConfig,
    fit_right_singular_basis,
    project_onto_basis,
    random_source_direction,
    select_minimum_rank,
)


def test_config_freezes_primary_seeds_and_rank_order() -> None:
    with pytest.raises(ValueError, match="fixed"):
        DefectSubspaceRankConfig(seeds=(7,))
    with pytest.raises(ValueError, match="increasing"):
        DefectSubspaceRankConfig(ranks=(1, 4, 2))
    with pytest.raises(ValueError, match="rank one"):
        DefectSubspaceRankConfig(ranks=(2, 4))
    with pytest.raises(ValueError, match="numerical rank"):
        DefectSubspaceRankConfig(numerical_rank_rtol=0.0)


def test_singular_basis_and_projection_recover_known_subspace() -> None:
    torch.manual_seed(11)
    left = torch.randn(20, 2, dtype=torch.float64)
    right, _ = torch.linalg.qr(torch.randn(7, 2, dtype=torch.float64))
    matrix = left @ right.T
    basis, energy = fit_right_singular_basis(matrix)
    reconstructed = project_onto_basis(matrix, basis, 2)
    assert basis.shape == (2, 7)
    assert torch.allclose(basis @ basis.T, torch.eye(2, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(reconstructed, matrix, atol=1e-10)
    assert float(energy[:2].sum()) == pytest.approx(1.0)


def test_random_control_is_deterministic_and_inside_source_span() -> None:
    torch.manual_seed(19)
    basis, _ = torch.linalg.qr(torch.randn(9, 4, dtype=torch.float64))
    basis = basis.T
    direction = random_source_direction(basis, 101)
    repeated = random_source_direction(basis, 101)
    projected = project_onto_basis(direction[None], basis, basis.shape[0])[0]
    assert torch.equal(direction, repeated)
    assert torch.allclose(direction, projected, atol=1e-12)
    assert float(torch.linalg.vector_norm(direction)) == pytest.approx(1.0)


def _record(preservation: float, causal: bool = True) -> dict:
    return {
        "causal_pass": causal,
        "effect_preservation": {
            "degenerate": False,
            "preserved_fraction": preservation,
        },
    }


def test_minimum_rank_requires_one_fixed_rank_in_every_cell() -> None:
    cells = [
        {"interventions": {"rank_1": _record(0.95), "rank_2": _record(0.98)}},
        {"interventions": {"rank_1": _record(0.89), "rank_2": _record(0.97)}},
    ]
    assert select_minimum_rank(cells, (1, 2), 0.90) == 2
    cells[1]["interventions"]["rank_2"] = _record(0.99, causal=False)
    assert select_minimum_rank(cells, (1, 2), 0.90) is None
