from __future__ import annotations

import pytest
import torch

from experiments.structure_net.tinyllm_defect_boundary_basis import (
    DefectBoundaryBasisConfig,
    _json_compatible,
    _seed_gates,
    aggregate,
    combine_bases,
    minimum_fixed_correction,
    project_residual_onto_normals,
    random_residual_basis,
    residualize_rows,
    select_winner_and_challenger,
)


def _record(passed: bool, preservation: float = 1.0) -> dict:
    return {
        "causal_pass": passed,
        "effect_preservation": {
            "degenerate": False,
            "preserved_fraction": preservation,
        },
    }


def _cell(cohort: str, regime: str) -> dict:
    return {
        "cohort": cohort,
        "regime": regime,
        "decomposition_relative_error": 1e-8,
        "interventions": {
            "zero": _record(False, 0.0),
            "exact": _record(True),
            "base_geometric": _record(False, 0.89),
            "boundary_plus_1": _record(True, 0.99),
            "boundary_plus_2": _record(True, 1.0),
            "boundary_plus_4": _record(True, 1.0),
            "geometric_plus_1": _record(False, 0.89),
            "geometric_plus_2": _record(True, 0.99),
            "geometric_plus_4": _record(True, 1.0),
            "shuffled_plus_1": _record(False, 0.4),
            "shuffled_plus_2": _record(False, 0.7),
            "shuffled_plus_4": _record(True, 0.99),
            "random_plus_1": _record(False, 0.1),
            "random_plus_2": _record(False, 0.2),
            "random_plus_4": _record(False, 0.3),
        },
    }


def test_config_freezes_primary_seeds_and_correction_titration() -> None:
    DefectBoundaryBasisConfig()
    with pytest.raises(ValueError, match="fixed"):
        DefectBoundaryBasisConfig(seeds=(7, 29))
    with pytest.raises(ValueError, match="starts at one"):
        DefectBoundaryBasisConfig(correction_counts=(2, 4))


def test_json_compatibility_normalizes_tuple_configuration() -> None:
    config = DefectBoundaryBasisConfig()
    normalized = _json_compatible(config.__dict__)
    assert normalized["seeds"] == [7, 29, 53]
    assert normalized["correction_counts"] == [1, 2, 4]


def test_winner_and_challenger_never_select_same_class() -> None:
    exact = torch.tensor([[0.1, 2.0, 0.3], [4.0, 1.0, 0.0]])
    base = torch.tensor([[5.0, 4.0, 3.0], [7.0, 6.0, 8.0]])
    winner, challenger = select_winner_and_challenger(exact, base)
    assert winner.tolist() == [1, 0]
    assert challenger.tolist() == [0, 2]


def test_boundary_projection_is_signed_and_row_local() -> None:
    residual = torch.tensor([[2.0, 3.0], [4.0, -2.0]], dtype=torch.float64)
    normals = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    projected = project_residual_onto_normals(residual, normals)
    assert torch.allclose(projected, torch.tensor([[2.0, 0.0], [0.0, -2.0]], dtype=torch.float64))


def test_combined_and_random_bases_stay_in_geometric_complement() -> None:
    geometric = torch.eye(8, dtype=torch.float64)
    base = geometric[:2]
    rows = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]])
    residual = residualize_rows(rows.double(), base)
    assert torch.allclose(residual[:, :2], torch.zeros(1, 2, dtype=torch.float64))
    random = random_residual_basis(geometric, 2, 3, 17)
    combined = combine_bases(base, random, 3)
    assert torch.allclose(combined @ combined.T, torch.eye(5, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(random @ base.T, torch.zeros(3, 2, dtype=torch.float64), atol=1e-12)


def test_joint_gate_requires_boundary_repair_and_specific_controls() -> None:
    cells = [
        _cell("heldout_a", "composition"),
        _cell("heldout_a", "extrapolation"),
        _cell("heldout_b", "composition"),
        _cell("heldout_b", "extrapolation"),
    ]
    failures = sorted(f"{cell['cohort']}:{cell['regime']}" for cell in cells)
    config = DefectBoundaryBasisConfig()
    gates = _seed_gates(cells, failures, True, config)
    assert gates["basis_contract"]
    assert gates["exact_endpoint_replication"]
    assert gates["predecessor_failure_replication"]
    assert gates["boundary_rank_one_sufficiency"]
    assert gates["shuffled_and_random_specificity"]
    assert gates["minimum_boundary_correction"] == 1
    assert gates["minimum_geometric_correction"] == 2
    assert gates["minimum_shuffled_correction"] == 4
    assert gates["minimum_random_correction"] is None


def test_minimum_correction_and_aggregate_are_conjunctive() -> None:
    cells = [_cell("heldout_a", "composition"), _cell("heldout_b", "extrapolation")]
    cells[1]["interventions"]["boundary_plus_1"] = _record(False, 0.99)
    assert minimum_fixed_correction(cells, "boundary", (1, 2, 4), 0.90) == 2
    gate_names = (
        "basis_contract",
        "exact_endpoint_replication",
        "predecessor_failure_replication",
        "boundary_rank_one_sufficiency",
        "shuffled_and_random_specificity",
    )
    runs = [
        {
            "seed": seed,
            "source_basis": {"base_rank": 1, "boundary_cumulative_energy": [1.0]},
            "gates": {name: not (seed == 29 and name == "boundary_rank_one_sufficiency") for name in gate_names},
        }
        for seed in (7, 29, 53)
    ]
    result = aggregate(runs)
    assert result["gate_counts"]["boundary_rank_one_sufficiency"] == 2
    assert not result["gate_passes"]["boundary_rank_one_sufficiency"]
    assert not result["confirmed"]
