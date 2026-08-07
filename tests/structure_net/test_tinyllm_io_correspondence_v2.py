import math

import torch

from experiments.structure_net.tinyllm_io_correspondence_v2 import (
    EquivariantVectorEncoder,
    SCHEMA_VERSION,
    _implementation_digest,
)


def test_corrected_schema_is_append_only_revision():
    assert SCHEMA_VERSION == "nal.tinyllm-io-correspondence.v1.1"
    assert len(_implementation_digest()) == 64


def test_complete_vector_encoder_is_exactly_rotation_equivariant():
    torch.manual_seed(17)
    encoder = EquivariantVectorEncoder(sensor_steps=8, vector_channels=4).eval()
    sensor = torch.randn(12, 8, 3)
    angle = 0.71
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=sensor.dtype,
    )
    transformed = sensor.clone()
    transformed[..., :2] = torch.einsum("ij,btj->bti", rotation, sensor[..., :2])
    with torch.no_grad():
        original = encoder(sensor)
        actual = encoder(transformed)
        expected = torch.einsum("ij,bj->bi", rotation, original)
    assert original.shape == (12, 2)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
