import math

import numpy as np
import torch

from experiments.structure_net.tinyllm_invariant_frontend_causal import (
    CUTS,
    CONDITIONS,
    AnalyticInvariantCanonicalizer,
    InvariantFrontendConfig,
    LearnedEquivariantEncoder,
    _experiments,
    aggregate_details,
    decode_sensor_tokens,
)
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    generate_paired_dataset,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def _small_config(**overrides):
    values = dict(
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
    return InvariantFrontendConfig(**values)


def test_decode_sensor_tokens_returns_declared_bin_centers():
    task = CircleTaskConfig(train_samples=32)
    paired = generate_paired_dataset(
        task, sample_count=32, seed=91, support="N3", shuffle=False
    )
    decoded = decode_sensor_tokens(paired.circle.input_ids, task)
    assert decoded.shape == (32, task.sensor_steps, 3)
    spacing = 2.0 * task.quantization_limit / task.value_bins
    grid_position = (decoded + task.quantization_limit) / spacing - 0.5
    assert torch.allclose(grid_position, grid_position.round(), atol=1e-5)


def test_analytic_canonicalizer_recovers_clean_future_cosine():
    task = CircleTaskConfig(train_samples=32)
    config = _small_config()
    canonicalizer = AnalyticInvariantCanonicalizer(task, config)
    generator = np.random.default_rng(31)
    count = 24
    future = generator.uniform(0.3, 2.8, count)
    direction = generator.choice(np.array([-1.0, 1.0]), count)
    speed = generator.uniform(0.12, 0.58, count)
    current = future - direction * task.future_delta
    history = np.arange(task.sensor_steps) - (task.sensor_steps - 1)
    angles = current[:, None] + direction[:, None] * speed[:, None] * history
    amplitude = generator.uniform(0.7, 1.4, (count, 1, 1))
    planar = amplitude * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    offset = generator.uniform(-0.3, 0.3, (count, 1, 2))
    drift = generator.uniform(-0.2, 0.2, (count, 1, 2))
    planar += offset + drift * history[None, :, None]
    nuisance_z = generator.normal(size=(count, task.sensor_steps, 1))
    sensor = torch.tensor(np.concatenate((planar, nuisance_z), axis=-1), dtype=torch.float32)
    estimate = canonicalizer(sensor).squeeze(1).numpy()
    assert np.corrcoef(estimate, np.cos(future))[0, 1] > 0.995
    assert np.sqrt(np.mean(np.square(estimate - np.cos(future)))) < 0.06


def test_learned_encoder_vector_relation_is_structural():
    torch.manual_seed(17)
    encoder = LearnedEquivariantEncoder(sensor_steps=8, vector_channels=6).eval()
    sensor = torch.randn(11, 8, 3)
    angle = 0.73
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    history = torch.arange(8, dtype=torch.float32) - 7
    transformed = sensor.clone()
    transformed[..., :2] = (
        1.7 * torch.einsum("ij,btj->bti", rotation, sensor[..., :2])
        + torch.tensor([0.4, -0.2])[None, None, :]
        + history[None, :, None] * torch.tensor([0.03, 0.07])[None, None, :]
    )
    transformed[..., 2] = torch.randn_like(transformed[..., 2]) * 9.0
    expected = torch.einsum("ij,bj->bi", rotation, encoder.vector(sensor))
    assert torch.allclose(encoder.vector(transformed), expected, atol=2e-5, rtol=2e-5)


def test_learned_encoder_receives_task_gradients():
    encoder = LearnedEquivariantEncoder(sensor_steps=8, vector_channels=6)
    loss = encoder(torch.randn(9, 8, 3)).square().mean()
    loss.backward()
    assert encoder.temporal_weights.grad is not None
    assert torch.isfinite(encoder.temporal_weights.grad).all()


def test_campaign_has_three_arms_crossed_with_all_seeds(tmp_path):
    config = _small_config(seeds=(7, 17, 29, 41, 53), allow_underpowered=False)
    task = CircleTaskConfig()
    experiments = _experiments(config, task, tmp_path)
    assert len(experiments) == 15
    assert {item.parameters["condition"] for item in experiments} == set(CONDITIONS)


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


def test_joint_gate_requires_same_four_seeds_across_every_cut_and_shift():
    config = _small_config(seeds=(7, 17, 29, 41, 53), allow_underpowered=False)
    details = []
    for condition in CONDITIONS:
        for seed in config.seeds:
            failure = None
            if condition == "learned_equivariant" and seed in (7, 17):
                failure = ("frontend", "extrapolation")
            details.append(_detail(condition, seed, failure))
    aggregate = aggregate_details(details, config)
    learned = aggregate["arms"]["learned_equivariant"]
    assert learned["joint_pass_count"] == 3
    assert learned["success"] is False
