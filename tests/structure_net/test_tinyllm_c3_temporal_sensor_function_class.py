from __future__ import annotations

import pytest
import torch

from experiments.structure_net import tinyllm_c3_temporal_quotient_training as stage0
from experiments.structure_net.tinyllm_c3_temporal_sensor_function_class import (
    CLASSIFICATION_PASS,
    PARAMETER_COUNT,
    PINNED_SOURCE_SHA256,
    PREREGISTRATION_SHA256,
    _source_hashes,
    analyze_gradient_route,
    analyze_witness,
    build_result,
    construct_analytic_witness,
)


@pytest.fixture(scope="module")
def witness_result() -> dict:
    return analyze_witness()


@pytest.fixture(scope="module")
def gradient_result() -> dict:
    return analyze_gradient_route()


def test_sources_and_preregistration_are_frozen() -> None:
    sources = _source_hashes()
    assert sources["preregistration"] == PREREGISTRATION_SHA256
    assert all(
        sources[name] == expected
        for name, expected in PINNED_SOURCE_SHA256.items()
    )
    assert len(sources["runner"]) == 64


def test_closed_form_gelu_witness_uses_five_nonzero_parameters() -> None:
    encoder = construct_analytic_witness(stage0.LearnedC3InvariantEncoder())
    assert sum(parameter.numel() for parameter in encoder.parameters()) == PARAMETER_COUNT
    assert sum(
        int(torch.count_nonzero(parameter)) for parameter in encoder.parameters()
    ) == 5
    grid = torch.linspace(-4.0, 4.0, 10_001)
    reconstructed = encoder.shared_map(grid[:, None])[:, 0]
    assert float((reconstructed - grid).abs().max()) <= 5e-7


def test_witness_matches_analytic_carrier_and_fixed_task_on_both_shifts(
    witness_result: dict,
) -> None:
    assert witness_result["pass"] is True
    for regime in ("composition", "extrapolation"):
        measured = witness_result["regimes"][regime]
        assert measured["carrier_maximum_absolute_error"] <= 2e-6
        assert measured["maximum_nonidentity_action_error"] <= 2e-6
        assert measured["temporal_target_correlation"] >= 0.99
        assert measured["temporal_target_rmse"] <= 0.08
        assert measured["fixed_interval_task_pass"] is True
        assert abs(measured["shuffled_target_correlation"]) <= 0.10
        assert measured["shuffled_target_rmse"] >= 0.80


def test_true_task_reaches_sensor_and_local_descent_is_restored(
    gradient_result: dict,
) -> None:
    assert gradient_result["pass"] is True
    assert gradient_result["finite"] is True
    assert gradient_result["total_gradient_norm"] >= 1e-6
    assert gradient_result["nonzero_gradient_fraction"] >= 0.90
    assert gradient_result["deck_loss_absolute_error"] <= 1e-6
    assert gradient_result["deck_gradient_maximum_absolute_error"] <= 2e-5
    assert gradient_result["diagnostic_loss_decrease"] >= 1e-4
    assert (
        gradient_result["state_sha256_before"]
        == gradient_result["state_sha256_after_restore"]
    )


def test_aggregate_verdict_licenses_only_the_conditional_campaign() -> None:
    result = build_result()
    assert result["classification"] == CLASSIFICATION_PASS
    assert result["gates"]["campaign_licensed"] is True
    assert result["gates"]["zero_optimizer_steps"] is True
    assert result["gates"]["no_tinyllm_instantiated"] is True
    assert result["gates"]["no_checkpoint_loaded"] is True
    assert result["accounting"] == {
        "optimizer_steps": 0,
        "tinyllm_models_instantiated": 0,
        "checkpoints_loaded": 0,
        "trained_parameters": 0,
        "sensor_family_parameters": 184,
        "closed_form_nonzero_witness_parameters": 5,
    }
