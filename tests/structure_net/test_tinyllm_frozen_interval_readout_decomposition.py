from __future__ import annotations

import math

import torch

from experiments.structure_net.tinyllm_frozen_interval_readout_decomposition import (
    IntervalReadoutConfig,
    aggregate_details,
    apply_ridge_map,
    deterministic_target_permutation,
    fit_ridge_map,
    fixed_interval_posterior,
    scalar_metrics,
    task_metrics,
)


def test_fixed_interval_decoder_matches_declared_generator_chart() -> None:
    scalar = torch.tensor([-0.91, -0.25, 0.0, 0.67, 0.94], dtype=torch.float64)
    posterior = fixed_interval_posterior(scalar)
    centers = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    width = 2.0 / 15.0
    expected = torch.softmax(
        -0.5 * ((centers[None] - scalar[:, None]) / width).square(), dim=-1
    )
    assert torch.equal(posterior, expected)
    assert torch.allclose(posterior.sum(1), torch.ones(len(scalar), dtype=torch.float64))


def test_standardized_ridge_recovers_affine_scalar_and_matrix_targets() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(256, 5, generator=generator, dtype=torch.float64)
    scalar_target = 0.4 * x[:, 0] - 1.2 * x[:, 2] + 0.7
    matrix_target = torch.stack((scalar_target, -2.0 * scalar_target + 0.3), dim=1)

    scalar_fit = fit_ridge_map(
        x,
        scalar_target,
        ridge_lambda=1e-12,
        standardization_floor=1e-6,
    )
    matrix_fit = fit_ridge_map(
        x,
        matrix_target,
        ridge_lambda=1e-12,
        standardization_floor=1e-6,
    )
    assert torch.max(torch.abs(apply_ridge_map(scalar_fit, x).reshape(-1) - scalar_target)) < 1e-9
    assert torch.max(torch.abs(apply_ridge_map(matrix_fit, x) - matrix_target)) < 1e-9
    assert scalar_fit.rank == 6
    assert math.isfinite(scalar_fit.condition_number)


def test_target_permutation_is_deterministic_nonidentity_and_cell_specific() -> None:
    first = deterministic_target_permutation(
        64, "d6", "analytic_calibrated", 7, 20_260_811
    )
    replay = deterministic_target_permutation(
        64, "d6", "analytic_calibrated", 7, 20_260_811
    )
    other = deterministic_target_permutation(
        64, "d10", "analytic_calibrated", 7, 20_260_811
    )
    assert torch.equal(first, replay)
    assert not torch.equal(first, torch.arange(64))
    assert not torch.equal(first, other)
    assert torch.equal(torch.sort(first).values, torch.arange(64))


def test_metrics_keep_interval_bin_error_noncircular_and_measure_pairs() -> None:
    target = torch.eye(4, dtype=torch.float64)
    posterior = torch.eye(4, dtype=torch.float64)[torch.tensor([3, 1, 2, 0])]
    measured = task_metrics(posterior, target)
    assert measured["exact_bin_accuracy"] == 0.5
    assert measured["mean_absolute_bin_error"] == 1.5

    predicted = torch.tensor([0.1, 0.2, -0.4, -0.5])
    cosine = torch.tensor([0.15, 0.15, -0.45, -0.45])
    fiber_id = torch.tensor([0, 0, 1, 1])
    scalar = scalar_metrics(predicted, cosine, fiber_id)
    assert scalar["cosine_pearson"] > 0.98
    assert scalar["mean_opposite_sheet_absolute_difference"] == pytest.approx(0.1)


def _detail(
    preset: str,
    condition: str,
    seed: int,
    *,
    typed: bool,
    untyped: bool,
    input_gauge: bool,
    shuffled: bool = False,
) -> dict:
    return {
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "source": {"source_task_adequacy_pass": seed == 7},
        "gates": {
            "validity": True,
            "arms": {
                "input_affine_gauge": {
                    "true_both_shifts": input_gauge,
                    "shuffled_both_shifts": shuffled,
                },
                "untyped_final_readout": {
                    "true_both_shifts": untyped,
                    "shuffled_both_shifts": shuffled,
                },
                "typed_interval_readout": {
                    "true_both_shifts": typed,
                    "shuffled_both_shifts": shuffled,
                },
                "frontend_typed_bypass": {
                    "true_both_shifts": True,
                    "shuffled_both_shifts": shuffled,
                },
            },
        },
    }


def test_aggregate_requires_joint_four_of_five_in_every_stratum() -> None:
    config = IntervalReadoutConfig(device="cpu")
    details = []
    for preset in config.presets:
        for condition in config.conditions:
            for index, seed in enumerate(config.seeds):
                details.append(
                    _detail(
                        preset,
                        condition,
                        seed,
                        typed=index < 4,
                        untyped=True,
                        input_gauge=False,
                    )
                )
    aggregate = aggregate_details(details, config)
    assert aggregate["valid"] is True
    assert aggregate["classification"] == "frozen_typed_interval_readout_architecture_stable"
    assert aggregate["gates"]["primary_hypothesis_pass"] is True
    assert aggregate["gates"]["stop_before_joint_retraining"] is True


def test_aggregate_rejects_marginal_pooling_and_shuffled_specificity() -> None:
    config = IntervalReadoutConfig(device="cpu")
    details = []
    for stratum_index, (preset, condition) in enumerate(
        (pair for pair in ((p, c) for p in config.presets for c in config.conditions))
    ):
        for index, seed in enumerate(config.seeds):
            # One stratum has only three true passes; the total remains large.
            typed = index < (3 if stratum_index == 3 else 5)
            details.append(
                _detail(
                    preset,
                    condition,
                    seed,
                    typed=typed,
                    untyped=False,
                    input_gauge=False,
                    shuffled=(stratum_index == 0 and index < 2),
                )
            )
    aggregate = aggregate_details(details, config)
    assert aggregate["gates"]["primary_hypothesis_pass"] is False
    assert aggregate["classification"] == "partial_frozen_interface_repair"
    assert aggregate["strata"]["d6/analytic_calibrated"]["arms"][
        "typed_interval_readout"
    ]["family_gate"] is False


# Kept local to avoid making pytest a runtime dependency of the experiment.
import pytest
