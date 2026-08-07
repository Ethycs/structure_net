import math

import pytest
import torch

from experiments.structure_net.tinyllm_orbit_radius_titration import (
    OrbitRadiusConfig,
    aggregate,
    crossing_summary,
    radial_residual_metrics,
)


def _points(flags):
    denominator = len(flags) - 1
    return [
        {"radius": index / denominator, "causal_pass": passed}
        for index, passed in enumerate(flags)
    ]


def test_config_requires_ordered_unit_radius_grid() -> None:
    OrbitRadiusConfig(seeds=(7,), degrees=(2,), allow_underpowered=True)
    with pytest.raises(ValueError, match="start at 0"):
        OrbitRadiusConfig(
            seeds=(7,), degrees=(2,), radii=(0.1, 1.0), allow_underpowered=True
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        OrbitRadiusConfig(
            seeds=(7,), degrees=(2,), radii=(0.0, 0.5, 0.5, 1.0), allow_underpowered=True
        )


def test_crossing_summary_distinguishes_single_and_reentrant_curves() -> None:
    clean = crossing_summary(_points([False, False, True, True, True]))
    assert clean == {
        "endpoint_replication": True,
        "single_crossing": True,
        "critical_radius": 0.5,
        "pass_sequence": [False, False, True, True, True],
    }
    reentrant = crossing_summary(_points([False, True, False, True, True]))
    assert reentrant["endpoint_replication"]
    assert not reentrant["single_crossing"]
    assert reentrant["critical_radius"] == 0.25
    failed_endpoint = crossing_summary(_points([True, True, True]))
    assert not failed_endpoint["endpoint_replication"]
    assert not failed_endpoint["single_crossing"]


def test_radial_residual_metrics_recognize_quadratic_path() -> None:
    full = torch.tensor([[1.0, -2.0, 3.0]])
    radius = 0.5
    metrics = radial_residual_metrics(radius**2 * full, full, radius)
    assert math.isclose(metrics["norm_ratio_to_full"], radius**2)
    assert math.isclose(metrics["cosine_to_full"], 1.0)
    assert math.isclose(metrics["normalized_squared_error_to_quadratic_chord"], 0.0)
    assert metrics["normalized_squared_error_to_linear_chord"] > 0.0
    zero = radial_residual_metrics(torch.zeros_like(full), full, 0.0)
    assert zero["cosine_to_full"] is None


def test_aggregate_keeps_degree_three_secondary() -> None:
    config = OrbitRadiusConfig(
        seeds=(7, 17, 29, 41, 53), degrees=(2, 3), allow_underpowered=False
    )
    runs = []
    for k in (2, 3):
        for seed in config.seeds:
            passed = k == 2 or seed != 7
            runs.append(
                {
                    "k": k,
                    "seed": seed,
                    "gates": {
                        "endpoint_replication": passed,
                        "single_radial_crossing": passed,
                        "shift_stable_threshold": passed,
                    },
                    "regimes": {
                        regime: {
                            "curves": {
                                path: {
                                    "crossing": {"critical_radius": 0.5}
                                }
                                for path in (
                                    "exact_orbit",
                                    "linear_chord",
                                    "quadratic_chord",
                                )
                            }
                        }
                        for regime in ("composition", "extrapolation")
                    },
                }
            )
    result = aggregate(runs, config)
    assert result["confirmed"]
    assert result["degrees"]["2"]["role"] == "primary"
    assert result["degrees"]["3"]["role"] == "secondary_diagnostic"
