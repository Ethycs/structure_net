from __future__ import annotations

import pytest
import torch

from experiments.structure_net.tinyllm_c2_fusion_radius import (
    EARLY_FRONT_SEEDS,
    FusionRadiusConfig,
    _matched_directions,
    _monotone_after_onset,
    _onset,
    _radius_key,
    _seed_gates,
    aggregate,
    symmetric_response,
)


def test_config_requires_confirmatory_seed_count_and_valid_grid() -> None:
    with pytest.raises(ValueError, match="five seeds"):
        FusionRadiusConfig(seeds=(7,))
    with pytest.raises(ValueError, match="unique, sorted"):
        FusionRadiusConfig(radius_grid=(0.0, 0.5, 0.25, 1.0))
    with pytest.raises(ValueError, match="observed and primary"):
        FusionRadiusConfig(radius_grid=(0.0, 0.5, 0.75))


def test_symmetric_response_is_even_and_quadratic_for_square_map() -> None:
    barycenter = torch.tensor([[0.2, -0.4], [0.6, 0.1]])
    deviation = torch.tensor([[0.5, -0.3], [-0.2, 0.7]])
    apply = lambda value: value + value.square()
    propagated, response = symmetric_response(barycenter, deviation, 0.5, apply)
    _, exchanged = symmetric_response(barycenter, -deviation, 0.5, apply)
    assert torch.allclose(propagated, apply(barycenter))
    assert torch.allclose(response, 0.25 * deviation.square(), atol=1e-7)
    assert torch.equal(response, exchanged)


def test_control_directions_match_per_orbit_norm() -> None:
    exact = torch.arange(1, 25, dtype=torch.float32).reshape(4, 2, 3)
    controls = _matched_directions(exact, seed=17)
    exact_norm = exact.square().sum((1, 2)).sqrt()
    for values in controls.values():
        assert torch.allclose(values.square().sum((1, 2)).sqrt(), exact_norm)
    assert not torch.equal(controls["cross_orbit_transplant"], exact)
    assert not torch.equal(controls["matched_random_symmetric"], exact)


def test_onset_and_monotonicity_use_only_primary_grid() -> None:
    config = FusionRadiusConfig(seeds=(7,), allow_underpowered=True)
    records = {
        _radius_key(radius): {"causal_pass": radius in (0.5, 0.75, 1.0)}
        for radius in config.radius_grid
    }
    assert _onset(records, config) == 0.5
    assert _monotone_after_onset(records, 0.5, config)
    records[_radius_key(0.75)]["causal_pass"] = False
    assert not _monotone_after_onset(records, 0.5, config)
    records[_radius_key(1.25)]["causal_pass"] = True
    assert _onset(records, config) == 0.5


def test_seed_gate_uses_joint_shift_onset() -> None:
    config = FusionRadiusConfig(seeds=(7,), allow_underpowered=True)
    regimes = {
        "composition": {
            "onset_radius": 0.5,
            "monotone_after_onset": True,
            "control_reproduction": {"a": False},
            "exchange_max_posterior_error": 0.0,
        },
        "extrapolation": {
            "onset_radius": 0.75,
            "monotone_after_onset": True,
            "control_reproduction": {"a": False},
            "exchange_max_posterior_error": 0.0,
        },
    }
    gates = _seed_gates(7, regimes, config)
    assert gates["early_front_locality"] is False
    assert gates["shift_stable_onset"] is True
    assert gates["monotone_causal_response"] is True


def test_aggregate_requires_exact_early_and_later_cohorts() -> None:
    config = FusionRadiusConfig()
    runs = []
    for seed in config.seeds:
        early = seed in EARLY_FRONT_SEEDS
        runs.append(
            {
                "seed": seed,
                "gates": {
                    "early_front_locality": True if early else None,
                    "later_front_finite_radius": None if early else True,
                    "shift_stable_onset": True,
                    "monotone_causal_response": True,
                    "control_specificity": True,
                    "exchange_invariance": True,
                },
                "regimes": {
                    regime: {
                        "onset_radius": 0.5 if early else 0.75,
                        "target_cut": "block_0_post_attention",
                    }
                    for regime in ("composition", "extrapolation")
                },
            }
        )
    result = aggregate(runs, config)
    assert result["confirmed"] is True
    runs[0]["gates"]["early_front_locality"] = False
    assert aggregate(runs, config)["confirmed"] is False
