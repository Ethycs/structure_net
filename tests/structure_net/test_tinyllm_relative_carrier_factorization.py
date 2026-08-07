import math

import torch

from experiments.structure_net.tinyllm_io_correspondence import generate_relation_fiber_dataset
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from experiments.structure_net.tinyllm_relative_carrier_factorization import (
    AnalyticRelativeVectorCarrier,
    EquivariantVectorEncoder,
    RelativeCarrierConfig,
    _relative_vector,
)
from experiments.structure_net.tinyllm_invariant_frontend_causal import decode_sensor_tokens


def test_relative_target_is_complete_unit_vector():
    task = CircleTaskConfig(train_samples=32)
    dataset = generate_relation_fiber_dataset(
        task, target_kind="relative", sample_count=32, seed=19
    )
    vector = _relative_vector(dataset)
    assert vector.shape == (32, 2)
    assert torch.allclose(vector.norm(dim=-1), torch.ones(32), atol=1e-6)


def test_analytic_relative_carrier_is_rotation_equivariant():
    task = CircleTaskConfig(train_samples=64)
    dataset = generate_relation_fiber_dataset(
        task, target_kind="relative", sample_count=64, seed=29
    )
    sensor = decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    carrier = AnalyticRelativeVectorCarrier(task)
    angle = 0.43
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    transformed = sensor.clone()
    transformed[..., :2] = torch.einsum("ij,btj->bti", rotation, sensor[..., :2])
    actual = carrier(transformed)
    expected = torch.einsum("ij,bj->bi", rotation, carrier(sensor))
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_learned_carrier_gradient_is_vector_supervision_only():
    torch.manual_seed(7)
    carrier = EquivariantVectorEncoder(8, 4)
    sensor = torch.randn(12, 8, 3)
    target = torch.nn.functional.normalize(torch.randn(12, 2), dim=-1)
    loss = torch.square(carrier(sensor) - target).mean()
    loss.backward()
    assert carrier.base.temporal_weights.grad is not None
    assert torch.isfinite(carrier.base.temporal_weights.grad).all()


def test_shakedown_configuration_allows_one_seed():
    config = RelativeCarrierConfig(
        seeds=(7,), training_steps=2, carrier_steps=2,
        train_samples=32, batch_size=8, allow_underpowered=True,
    )
    assert config.seeds == (7,)
