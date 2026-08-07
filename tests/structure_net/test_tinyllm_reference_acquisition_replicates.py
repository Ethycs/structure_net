from dataclasses import replace
import inspect
import math

import pytest
import torch

from experiments.structure_net.tinyllm_calibrated_frontend_causal import (
    CalibratedDataset,
)
from experiments.structure_net.tinyllm_reference_acquisition_replicates import (
    COUNTS,
    PRIMARY_SEEDS,
    ReferenceAcquisitionConfig,
    analytic_circular_mean,
    apply_equivariant_moment_denoiser,
    build_repeat_noise,
    charge_one_moment_features,
    classify_campaign,
    fit_equivariant_moment_denoiser,
    misgrouped_circular_mean,
    packet_from_orientation,
    repeated_observations,
    scaling_slope,
    task_gate,
)


def _minimal_dataset(calibration: torch.Tensor, fiber_ids: torch.Tensor) -> CalibratedDataset:
    class Fiber:
        fiber_id = fiber_ids

    class Paired:
        fiber = Fiber()

    return CalibratedDataset(paired=Paired(), calibration=calibration)  # type: ignore[arg-type]


def test_primary_configuration_is_locked() -> None:
    config = ReferenceAcquisitionConfig()
    assert config.seeds == PRIMARY_SEEDS
    assert config.repeat_counts == COUNTS
    assert config.sigma_radians == 0.175
    with pytest.raises(ValueError, match="primary seeds"):
        replace(config, repeat_counts=(1, 4, 16))
    shakedown = ReferenceAcquisitionConfig(
        seeds=(7,),
        repeat_counts=(1, 4),
        required_seed_passes=1,
        control_pass_ceiling=1,
        allow_underpowered=True,
    )
    assert shakedown.allow_underpowered is True


def test_repeat_noise_preserves_source_first_draw_and_pair_sharing() -> None:
    fiber_ids = torch.tensor((0, 0, 1, 1, 2, 2))
    calibration = torch.tensor(
        ((1.0, 0.0),) * len(fiber_ids), dtype=torch.float32
    )
    dataset = _minimal_dataset(calibration, fiber_ids)
    source = torch.tensor((0.2, 0.2, -0.5, -0.5, 1.1, 1.1))
    repeats = build_repeat_noise(
        dataset, source, split_seed=123, maximum_repeats=8
    )
    assert torch.equal(repeats[:, 0], source.double())
    assert repeats.shape == (6, 8)
    assert torch.equal(repeats[0], repeats[1])
    assert torch.equal(repeats[2], repeats[3])
    assert torch.equal(repeats[4], repeats[5])


def test_circular_mean_reduces_balanced_angular_error() -> None:
    calibration = torch.tensor(
        ((1.0, 0.0), (0.0, 1.0)), dtype=torch.float64
    )
    noise = torch.tensor(
        ((1.0, -1.0, 1.0, -1.0), (0.8, -0.8, 0.8, -0.8)),
        dtype=torch.float64,
    )
    observations = repeated_observations(calibration, noise, 0.2)
    single = analytic_circular_mean(observations, 1)
    averaged = analytic_circular_mean(observations, 4)
    target = torch.complex(calibration[:, 0], calibration[:, 1])
    assert torch.angle(averaged * target.conj()).abs().max() < 1e-12
    assert torch.angle(single * target.conj()).abs().mean() > 0.15
    packet = packet_from_orientation(calibration.float(), averaged)
    assert torch.allclose(packet[:, :2].double(), calibration, atol=1e-7)


def test_moment_features_are_rotation_equivariant_and_reflection_compatible() -> None:
    angles = torch.tensor(
        ((0.1, 0.2, -0.1, 0.0), (1.0, 1.1, 0.9, 1.2)),
        dtype=torch.float64,
    )
    observations = torch.polar(torch.ones_like(angles), angles)
    features = charge_one_moment_features(observations, 4)
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(0.73))
    rotated = charge_one_moment_features(observations * rotation, 4)
    reflected = charge_one_moment_features(observations.conj(), 4)
    assert torch.allclose(rotated, features * rotation, atol=1e-10)
    assert torch.allclose(reflected, features.conj(), atol=1e-10)


def test_learned_denoiser_has_three_real_parameters_and_preserves_equivariance() -> None:
    generator = torch.Generator().manual_seed(9)
    clean_angles = torch.linspace(-math.pi, math.pi, 64)
    clean = torch.stack((clean_angles.cos(), clean_angles.sin()), 1)
    noise = torch.randn((64, 16), generator=generator, dtype=torch.float64)
    observations = repeated_observations(clean, noise, 0.175)
    coefficients, fit = fit_equivariant_moment_denoiser(
        observations, clean, 16, 0.01
    )
    prediction = apply_equivariant_moment_denoiser(
        observations, 16, coefficients
    )
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(-0.41))
    rotated = apply_equivariant_moment_denoiser(
        observations * rotation, 16, coefficients
    )
    assert coefficients.dtype == torch.float64
    assert coefficients.shape == (3,)
    assert fit["parameter_count"] == 3
    assert torch.allclose(rotated, prediction * rotation, atol=1e-10)


def test_misgrouped_control_destroys_same_reference_membership() -> None:
    base = torch.tensor((0.0, math.pi / 2, math.pi, -math.pi / 2))
    unique = torch.polar(torch.ones(4, 4), base[:, None].expand(4, 4))
    observations = unique.repeat_interleave(2, dim=0)
    fiber_ids = torch.arange(4).repeat_interleave(2)
    correct = analytic_circular_mean(observations, 4)
    control = misgrouped_circular_mean(observations, fiber_ids, 4)
    assert not torch.allclose(control, correct)
    assert torch.equal(control[0], control[1])


def test_scaling_and_task_gates_are_exact() -> None:
    assert scaling_slope((1, 4, 16, 64), (0.2, 0.1, 0.05, 0.025)) == pytest.approx(-0.5)
    clean = {"composition": 0.8, "extrapolation": 0.7}
    evaluations = {
        "composition": {"task": {"exact_bin_accuracy": 0.771}},
        "extrapolation": {"task": {"exact_bin_accuracy": 0.68}},
    }
    passed, losses = task_gate(clean, evaluations, 0.03)
    assert passed is True
    assert losses == pytest.approx({"composition": 0.029, "extrapolation": 0.02})
    evaluations["composition"]["task"]["exact_bin_accuracy"] = 0.769
    assert task_gate(clean, evaluations, 0.03)[0] is False


def test_locked_classification_is_arm_aware() -> None:
    both = {condition: 5 for condition in (
        "analytic_calibrated", "learned_calibrated_equivariant"
    )}
    analytic_only = {
        "analytic_calibrated": 5,
        "learned_calibrated_equivariant": 3,
    }
    assert classify_campaign(
        valid=True, analytic_pass=both, learned_pass=both, required=4
    ) == "acquisition_precision_sufficient"
    assert classify_campaign(
        valid=True, analytic_pass=both, learned_pass=analytic_only, required=4
    ) == "analytic_reference_averaging_sufficient"
    assert classify_campaign(
        valid=False, analytic_pass=both, learned_pass=both, required=4
    ) == "invalid"


def test_denoiser_fit_has_no_task_label_or_optimizer_interface() -> None:
    signature = inspect.signature(fit_equivariant_moment_denoiser)
    assert tuple(signature.parameters) == (
        "observations",
        "clean_orientation",
        "count",
        "ridge_ratio",
    )
    source = inspect.getsource(fit_equivariant_moment_denoiser)
    assert "target_posteriors" not in source
    assert "target_bins" not in source
    assert "torch.optim" not in source
