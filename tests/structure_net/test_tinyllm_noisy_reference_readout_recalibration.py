from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.structure_net import (
    tinyllm_noisy_reference_readout_recalibration as readout,
)


def test_primary_protocol_is_locked() -> None:
    config = readout.ReadoutRecalibrationConfig(device="cpu")
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.noise_sigma == 0.035
    assert config.head_ridge_ratio == 0.01
    assert config.required_seed_passes == 4
    with pytest.raises(ValueError, match="seeds are fixed"):
        replace(config, seeds=(7,))
    with pytest.raises(ValueError, match="noise is fixed"):
        replace(config, noise_sigma=0.087)
    with pytest.raises(ValueError, match="ridge ratio is fixed"):
        replace(config, head_ridge_ratio=0.1)
    assert readout._evidence_role(config) == (
        "preregistered_frozen_system_readout_intervention"
    )
    assert readout._evidence_role(
        replace(config, partial_outcome_relaunch=True)
    ) == "preregistered_mixed_pedigree_partial_analytic_exposure"


def test_linear_readout_ridge_recovers_ordered_target() -> None:
    generator = torch.Generator().manual_seed(31)
    features = torch.randn(512, 12, generator=generator)
    true_weight = torch.randn(5, 12, generator=generator) * 0.4
    probabilities = torch.softmax(features @ true_weight.T, -1)
    frozen = true_weight + 0.3 * torch.randn(5, 12, generator=generator)
    fitted, audit = readout.fit_linear_readout(
        features, probabilities, frozen, ridge_ratio=1e-5
    )
    metrics = readout.evaluate_linear_readout(fitted, features, probabilities)
    assert metrics["mean_target_cross_entropy"] < 1.50
    assert metrics["expected_cosine_correlation"] > 0.99
    assert audit["weight_displacement_frobenius"] > 0.0


def test_affine_scalar_interval_repairs_scale_and_offset() -> None:
    centers = torch.linspace(-1.0, 1.0, 9)
    target_indices = torch.arange(9).repeat(32)
    target = torch.nn.functional.one_hot(target_indices, 9).float()
    true_coordinate = target @ centers
    observed = (true_coordinate - 0.17) / 1.25
    frozen = torch.softmax(
        -80.0 * (observed[:, None] - centers[None, :]).square(), -1
    )
    parameters = readout.fit_scalar_interval(frozen, target)
    metrics = readout.evaluate_scalar_interval(parameters, frozen, target)
    assert metrics["exact_bin_accuracy"] > 0.95
    assert metrics["expected_cosine_correlation"] > 0.99


def test_classification_is_locked() -> None:
    def arms(linear: tuple[int, int], scalar: tuple[int, int], noisy=(0, 0)):
        return {
            arm: {
                "linear_primary_pass_count": linear[index],
                "scalar_primary_pass_count": scalar[index],
                "linear_noisy_repair_pass_count": noisy[index],
            }
            for index, arm in enumerate(readout.ARMS)
        }

    assert readout.classify_campaign(
        arms((5, 5), (5, 5)), valid=True, required=4
    ) == "scalar_interval_calibration_sufficient"
    assert readout.classify_campaign(
        arms((5, 5), (0, 0)), valid=True, required=4
    ) == "linear_readout_recalibration_sufficient"
    assert readout.classify_campaign(
        arms((5, 0), (0, 0)), valid=True, required=4
    ) == "arm_stratified_readout_repair"
    assert readout.classify_campaign(
        arms((0, 0), (0, 0), noisy=(5, 0)), valid=True, required=4
    ) == "condition_specific_readout_tradeoff"
    assert readout.classify_campaign(
        arms((0, 0), (0, 0)), valid=True, required=4
    ) == "probe_readout_gap_persists"
    assert readout.classify_campaign(
        arms((5, 5), (5, 5)), valid=False, required=4
    ) == "invalid"


def test_locked_orientation_source_replays() -> None:
    config = readout.ReadoutRecalibrationConfig(
        seeds=(7,),
        required_seed_passes=1,
        device="cpu",
        allow_underpowered=True,
    )
    campaign, path, entries, task, source_config = readout._load_sources(config)
    assert readout._sha256(path) == readout.SOURCE_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == readout.SOURCE_IMPLEMENTATION_SHA256
    assert campaign["artifacts"]["noise_arrays_sha256"] == readout.SOURCE_NOISE_SHA256
    assert ("analytic_calibrated", 7) in entries
    assert task.phase_bins == 16
    assert source_config.seeds == (7, 17, 29, 41, 53)


def test_accuracy_gate_uses_both_shifted_regimes() -> None:
    evaluations = {
        "composition": {"exact_bin_accuracy": 0.72},
        "extrapolation": {"exact_bin_accuracy": 0.61},
    }
    baseline = {"composition": 0.74, "extrapolation": 0.63}
    assert readout._accuracy_gate(evaluations, baseline, 0.03)
    evaluations["extrapolation"]["exact_bin_accuracy"] = 0.59
    assert not readout._accuracy_gate(evaluations, baseline, 0.03)
