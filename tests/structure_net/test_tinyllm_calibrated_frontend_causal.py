import math

import numpy as np
import torch

from experiments.structure_net.tinyllm_calibrated_frontend_causal import (
    CONDITIONS,
    CUTS,
    AnalyticCalibratedCanonicalizer,
    CalibratedEquivariantEncoder,
    CalibratedFrontendConfig,
    CalibratedTinyLLM,
    _experiments,
    aggregate_details,
    generate_calibrated_dataset,
    identifiability_contract,
)
from experiments.structure_net.tinyllm_invariant_frontend_causal import (
    decode_sensor_tokens,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _tinyllm_config,
)
from structure_net.components.models import TinyLLMModel


def _small_config(**overrides):
    values = dict(
        preset="tiny",
        seeds=(7,),
        training_steps=2,
        train_samples=32,
        batch_size=8,
        probe_train_samples=64,
        probe_validation_samples=32,
        probe_test_samples=32,
        probe_steps=20,
        allow_underpowered=True,
    )
    values.update(overrides)
    return CalibratedFrontendConfig(**values)


def test_identifiability_contract_breaks_every_target_changing_gauge_pair():
    record = identifiability_contract()
    assert record["passed"] is True
    assert record["checked_gauge_pairs"] == 12_096
    assert record["target_changing_pairs"] == 12_096
    assert record["violations"] == 0
    assert record["maximum_old_observation_sensor_error"] < 2e-15
    assert record["minimum_target_changing_calibration_distance"] > 0.04


def test_calibration_packet_contains_nuisance_only_and_aligns_with_examples():
    task = CircleTaskConfig(train_samples=32)
    dataset = generate_calibrated_dataset(
        task, sample_count=32, seed=117, shuffle=False
    )
    calibration = dataset.calibration
    assert calibration.shape == (32, 8)
    assert torch.allclose(calibration[:, :2].norm(dim=1), torch.ones(32), atol=1e-6)
    assert torch.equal(calibration[:, 2].sign(), dataset.paired.circle.directions)
    assert not torch.allclose(calibration[:, 0], dataset.paired.fiber.cosine)


def test_analytic_positive_control_retains_cosine_on_all_declared_regimes():
    task = CircleTaskConfig()
    canonicalizer = AnalyticCalibratedCanonicalizer(task)
    for index, regime in enumerate(("interpolation", "composition", "extrapolation")):
        dataset = generate_calibrated_dataset(
            task,
            sample_count=1_024,
            seed=307 + index * 1_009,
            regime=regime,
            shuffle=False,
        )
        sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
        estimate = canonicalizer(sensor, dataset.calibration).squeeze(1).numpy()
        target = dataset.paired.fiber.cosine.numpy()
        assert np.corrcoef(estimate, target)[0, 1] > 0.98


def test_learned_encoder_is_invariant_to_joint_acquisition_transform():
    torch.manual_seed(23)
    batch, steps = 13, 8
    encoder = CalibratedEquivariantEncoder(steps, vector_channels=6).eval()
    history = torch.arange(steps, dtype=torch.float32) / (steps - 1) - 1.0
    orientation_angle = torch.linspace(-0.3, 0.3, batch)
    orientation = torch.stack(
        (orientation_angle.cos(), orientation_angle.sin()), dim=1
    )
    amplitude = torch.linspace(0.7, 1.3, batch)[:, None]
    offset = torch.randn(batch, 2) * 0.1
    drift = torch.randn(batch, 2) * 0.1
    corrected = torch.randn(batch, steps, 2)
    sensor = torch.zeros(batch, steps, 3)
    sensor[..., :2] = (
        amplitude[:, None, :] * corrected
        + offset[:, None, :]
        + drift[:, None, :] * history[None, :, None]
    )
    sensor[..., 2] = torch.randn(batch, steps)
    calibration = torch.cat(
        (
            orientation,
            torch.linspace(-0.5, 0.5, batch)[:, None],
            amplitude,
            offset,
            drift,
        ),
        dim=1,
    )
    angle = 0.71
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    scale = 1.6
    extra_offset = torch.tensor([0.31, -0.17])
    extra_drift = torch.tensor([-0.05, 0.08])
    transformed = sensor.clone()
    transformed[..., :2] = (
        scale * torch.einsum("ij,btj->bti", rotation, sensor[..., :2])
        + extra_offset[None, None, :]
        + extra_drift[None, None, :] * history[None, :, None]
    )
    transformed[..., 2] = torch.randn_like(transformed[..., 2]) * 11.0
    transformed_calibration = calibration.clone()
    transformed_calibration[:, :2] = torch.einsum(
        "ij,bj->bi", rotation, orientation
    )
    transformed_calibration[:, 3:4] = scale * amplitude
    transformed_calibration[:, 4:6] = (
        scale * torch.einsum("ij,bj->bi", rotation, offset)
        + extra_offset[None, :]
    )
    transformed_calibration[:, 6:8] = (
        scale * torch.einsum("ij,bj->bi", rotation, drift)
        + extra_drift[None, :]
    )
    assert torch.allclose(
        encoder(sensor, calibration),
        encoder(transformed, transformed_calibration),
        atol=3e-5,
        rtol=3e-5,
    )


def test_all_frontends_run_with_matched_tiny_transformer():
    task = CircleTaskConfig(train_samples=32)
    config = _small_config()
    dataset = generate_calibrated_dataset(task, sample_count=8, seed=41)
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    for condition in CONDITIONS:
        base = _tinyllm_config("tiny", task, 7)
        model = TinyLLMModel(
            type(base)(**{**base.__dict__, "block_size": task.sequence_length + 1})
        )
        system = CalibratedTinyLLM(model, condition, task, config)
        cuts = system.forward_cuts(
            dataset.paired.circle.input_ids, sensor, dataset.calibration
        )
        assert set(cuts) == set(CUTS)
        assert cuts["frontend"].shape[0] == 8
        assert cuts["full"].shape == (8, model.config.n_embd)


def test_campaign_crosses_three_arms_and_five_seeds(tmp_path):
    config = CalibratedFrontendConfig()
    experiments = _experiments(config, CircleTaskConfig(), tmp_path)
    assert len(experiments) == 15
    assert {item.parameters["condition"] for item in experiments} == set(CONDITIONS)
    assert all(item.parameters["identifiability_contract"]["passed"] for item in experiments)


def _detail(condition, seed, failed_cell=None):
    cuts = {}
    for cut in CUTS:
        evaluations = {}
        for regime in ("in_distribution", "composition", "extrapolation"):
            passed = failed_cell != (cut, regime)
            evaluations[regime] = {
                "cosine_pearson": 0.94 if passed else 0.89,
                "balanced_accuracy": 0.51,
                "conditional_log_loss_gain_over_cosine_only": 0.01,
            }
        cuts[cut] = {"probe": {"evaluations": evaluations}}
    return {"condition": condition, "seed": seed, "analysis": {"cuts": cuts}}


def test_joint_gate_requires_same_four_seeds_across_both_cuts_and_shifts():
    config = CalibratedFrontendConfig()
    details = []
    for condition in CONDITIONS:
        for seed in config.seeds:
            failure = None
            if condition == "learned_calibrated_equivariant" and seed in (7, 17):
                failure = ("frontend", "extrapolation")
            details.append(_detail(condition, seed, failure))
    aggregate = aggregate_details(details, config)
    learned = aggregate["arms"]["learned_calibrated_equivariant"]
    assert learned["joint_pass_count"] == 3
    assert learned["success"] is False
