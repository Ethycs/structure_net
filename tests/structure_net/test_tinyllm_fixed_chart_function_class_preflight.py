from __future__ import annotations

import torch

from experiments.structure_net.tinyllm_fixed_chart_function_class_preflight import (
    GRID_POINTS,
    WARP_PARAMETERS,
    build_preflight,
    monotone_endpoint_warp,
)


def test_registered_warps_are_monotone_odd_and_fixed_endpoint() -> None:
    grid = torch.linspace(-1.0, 1.0, GRID_POINTS, dtype=torch.float64)
    for parameter in WARP_PARAMETERS:
        warped = monotone_endpoint_warp(grid, parameter)
        assert torch.all(torch.diff(warped) > 0)
        assert abs(float(warped[0]) + 1.0) <= 1e-12
        assert abs(float(warped[-1]) - 1.0) <= 1e-12
        assert float(torch.max(torch.abs(warped + warped.flip(0)))) <= 1e-12
        assert float(warped.abs().max()) <= 1.0
        assert float(torch.max(torch.abs(warped - grid))) >= 0.05


def test_preflight_closes_nontrivial_exact_scalar_branch() -> None:
    record = build_preflight()
    assert record["valid"] is True
    assert record["status"] == "completed"
    assert record["classification"] == (
        "exact_physical_chart_requires_frozen_scalar_spine"
    )
    assert record["optimizer_steps_executed"] == 0
    assert record["checkpoints_loaded"] == 0
    assert all(record["source_checks"].values())
    assert record["identifiability"]["passed"] is True
    assert all(
        cell["pass"] for cell in record["analytic_positive_control"].values()
    )
    assert all(cell["pass"] for cell in record["monotone_endpoint_witnesses"])
    typed = record["typed_split"]
    assert typed["pass"] is True
    assert typed["physical_scalar_parameter_count"] == 0
    assert typed["auxiliary_parameter_count"] > 0
    assert typed["parameter_gradient_leak_count"] == 0
    assert typed["maximum_physical_state_dependence"] <= 1e-7
    assert typed["maximum_analytic_identity_error"] <= 1e-7
    assert typed["maximum_auxiliary_state_rms_difference"] >= 1e-4
