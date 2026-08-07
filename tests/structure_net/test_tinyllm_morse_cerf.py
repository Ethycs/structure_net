import numpy as np
import torch

from experiments.structure_net.tinyllm_morse_cerf import (
    cyclic_symmetrize,
    landscape_metrics,
    simplex_grid,
    vertex_exact_homotopy,
)


def test_vertex_exact_homotopy_preserves_every_sheet_endpoint() -> None:
    torch.manual_seed(4)
    sheets = torch.randn(5, 3, 4)
    matrix = torch.randn(4, 4)
    update = lambda value: torch.tanh(value @ matrix)
    deltas = update(sheets.flatten(0, 1)).reshape_as(sheets)
    weights = torch.eye(3)
    expected = sheets + deltas
    for alpha in (0.0, 0.2, 0.75, 1.0):
        completed, commutator = vertex_exact_homotopy(
            sheets, deltas, weights, alpha, update
        )
        assert torch.allclose(completed.transpose(0, 1), expected, atol=2e-6)
        assert torch.all(commutator >= 0)


def test_cyclic_symmetrization_is_invariant() -> None:
    weights, _ = simplex_grid(3, subdivisions=6)
    values = np.arange(len(weights), dtype=np.float64)
    symmetric = cyclic_symmetrize(values, weights)
    lookup = {tuple(np.round(item, 12)): i for i, item in enumerate(weights)}
    for index, weight in enumerate(weights):
        rotated = lookup[tuple(np.round(np.roll(weight, 1), 12))]
        assert symmetric[index] == symmetric[rotated]


def test_interval_landscapes_distinguish_center_peak_and_basin() -> None:
    weights, integer = simplex_grid(2, interval_points=49)
    coordinate = weights[:, 0] - 0.5
    peak = -(coordinate**2)
    peak_metrics = landscape_metrics(peak, weights, integer)
    assert peak_metrics["center"]["index"] == 1
    assert not peak_metrics["mature"]

    basin = coordinate**2
    basin_metrics = landscape_metrics(basin, weights, integer)
    assert basin_metrics["center"]["index"] == 0
    assert basin_metrics["mature"]


def test_triangle_landscape_recovers_positive_center_hessian() -> None:
    weights, integer = simplex_grid(3, subdivisions=12)
    basin = np.square(weights - 1.0 / 3.0).sum(axis=1)
    metrics = landscape_metrics(basin, weights, integer)
    assert metrics["center"]["index"] == 0
    assert metrics["critical_census"]["minima"] == 1
    assert metrics["center_in_vertex_component"]
