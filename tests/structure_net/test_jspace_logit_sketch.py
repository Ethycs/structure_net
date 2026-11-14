"""Exact small-model checks of full-output sketching and normalization."""
from itertools import product
import math
from unittest.mock import patch

import pytest
import torch

from experiments.structure_net.jspace_logit_sketch import (
    gram_from_input_vjps,
    normalization_metadata,
    project_sign_probe_groups,
    project_sign_probes,
    subspace_energy_fraction,
)


def enumerated_projections(weight, positions, chunk):
    """Exercise chunked projection with every possible Rademacher sign vector."""
    vocabulary = weight.shape[0]
    signs = torch.tensor(
        list(product((-1.0, 1.0), repeat=positions * vocabulary)),
        dtype=weight.dtype,
    ).reshape(-1, positions, vocabulary)
    calls = []
    cursor = 0

    def randint(low, high, size, *, device, dtype, generator):
        nonlocal cursor
        assert (low, high) == (0, 2)
        assert size[:2] == signs.shape[:2]
        count = size[-1]
        assert count <= chunk
        calls.append(size)
        value = ((signs[..., cursor:cursor + count] + 1) / 2).to(device=device, dtype=dtype)
        cursor += count
        return value

    with patch("experiments.structure_net.jspace_logit_sketch.torch.randint", side_effect=randint):
        projected = project_sign_probes(
            weight, positions, len(signs), torch.Generator().manual_seed(17), chunk
        )
    assert cursor == vocabulary
    assert len(calls) == math.ceil(vocabulary / chunk)
    torch.testing.assert_close(
        projected, (signs @ weight) / math.sqrt(positions * vocabulary),
        rtol=1e-14, atol=1e-14,
    )
    return projected


def test_exhaustive_signs_match_full_jacobian_with_real_rmsnorm_autograd():
    # Two different output positions, each influenced by both input positions.
    # The RMS denominator stays inside autograd, rather than being frozen.
    x = torch.tensor([[0.2, -0.7], [1.1, 0.4]], dtype=torch.float64, requires_grad=True)
    mixing = torch.tensor([
        [0.8, 0.1, -0.4, 0.3],
        [-0.2, 0.7, 0.5, -0.1],
        [0.6, -0.8, 0.2, 0.9],
        [0.1, 0.4, -0.7, 0.6],
    ], dtype=torch.float64)
    gain = torch.tensor([0.8, 1.3], dtype=torch.float64)
    weight = torch.tensor([[1.0, -0.4], [0.3, 0.9]], dtype=torch.float64)

    def normalized_hidden(value):
        hidden = torch.tanh(mixing @ value.flatten()).reshape(2, 2)
        return gain * hidden / (hidden.square().mean(-1, keepdim=True) + 0.07).sqrt()

    normalized = normalized_hidden(x)
    projected = enumerated_projections(weight, positions=2, chunk=1)
    vjps = torch.stack([
        torch.autograd.grad((normalized * probe).sum(), x, retain_graph=True)[0]
        for probe in projected
    ])
    actual = gram_from_input_vjps(vjps)
    explicit = torch.autograd.functional.jacobian(
        lambda value: normalized_hidden(value) @ weight.T, x
    ).reshape(4, 2, 2)
    expected = torch.einsum("otd,ote->de", explicit, explicit) / 4
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)

    # Freezing RMSNorm's denominator must be detectable by this test problem.
    hidden = torch.tanh(mixing @ x.flatten()).reshape(2, 2)
    frozen = gain * hidden / (hidden.square().mean(-1, keepdim=True) + 0.07).sqrt().detach()
    wrong_vjps = torch.stack([
        torch.autograd.grad((frozen * probe).sum(), x, retain_graph=True)[0]
        for probe in projected
    ])
    assert not torch.allclose(actual, gram_from_input_vjps(wrong_vjps), rtol=1e-3, atol=1e-3)


def test_full_output_sketch_retains_direction_missed_by_one_contrast():
    weight = torch.eye(2, dtype=torch.float64)
    projected = enumerated_projections(weight, positions=1, chunk=2)
    full = gram_from_input_vjps(projected)
    torch.testing.assert_close(full, torch.eye(2, dtype=torch.float64) / 2)
    contrast = torch.tensor([1.0, -1.0], dtype=torch.float64)
    common_direction = torch.ones(2, dtype=torch.float64)
    assert common_direction @ (contrast[:, None] * contrast[None, :]) @ common_direction == 0
    assert common_direction @ full @ common_direction > 0


def test_seeded_groups_and_examples_draw_fresh_probes():
    weight = torch.arange(35, dtype=torch.float32).reshape(7, 5) / 13
    gen = torch.Generator().manual_seed(861)
    first = project_sign_probe_groups(weight, 3, 4, gen, vocab_chunk_size=3)
    second = project_sign_probe_groups(weight, 3, 4, gen, vocab_chunk_size=3)
    assert first.shape == (2, 4, 3, 5)
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first, second)
    assert not torch.equal(first[0, :, 0], first[0, :, 1])
    replay = torch.Generator().manual_seed(861)
    torch.testing.assert_close(first, project_sign_probe_groups(weight, 3, 4, replay, 3), rtol=0, atol=0)
    torch.testing.assert_close(second, project_sign_probe_groups(weight, 3, 4, replay, 3), rtol=0, atol=0)


def test_projection_uses_float_accumulation_and_frozen_readout():
    weight = torch.arange(15, dtype=torch.bfloat16).reshape(5, 3).requires_grad_(True)
    result = project_sign_probes(weight, 2, 3, torch.Generator().manual_seed(2), 2)
    assert result.dtype == torch.float32
    assert result.shape == (3, 2, 3)
    assert not result.requires_grad
    assert weight.grad is None


def test_gram_sums_input_positions_but_averages_probes():
    vjps = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [-1.0, 1.0]]])
    expected = sum(torch.outer(row, row) for probe in vjps for row in probe) / 2
    torch.testing.assert_close(gram_from_input_vjps(vjps), expected)
    repeated = torch.cat((vjps, vjps), dim=0)
    torch.testing.assert_close(gram_from_input_vjps(repeated), expected)
    twice_the_positions = torch.cat((vjps, vjps), dim=1)
    torch.testing.assert_close(gram_from_input_vjps(twice_the_positions), 2 * expected)


def test_energy_fraction_and_normalization_metadata():
    gram = torch.diag(torch.tensor([9.0, 3.0, 0.0]))
    basis = torch.eye(3)
    torch.testing.assert_close(subspace_energy_fraction(gram, basis[:, :1]), torch.tensor(0.75))
    torch.testing.assert_close(subspace_energy_fraction(gram, basis), torch.tensor(1.0))
    meta = normalization_metadata(vocabulary_size=7, output_positions=3, probes=5)
    assert meta["output_components"] == 21
    assert meta["probe_scale"] == pytest.approx(1 / math.sqrt(21))
    assert meta["probe_reduction"] == "mean"
    assert meta["input_position_reduction"] == "sum"


@pytest.mark.parametrize("positions,probes,chunk", [(0, 2, 3), (2, 0, 3), (2, 2, 0), (True, 2, 3)])
def test_invalid_probe_dimensions_are_rejected(positions, probes, chunk):
    with pytest.raises(ValueError):
        project_sign_probes(torch.ones(3, 2), positions, probes, torch.Generator(), chunk)


def test_invalid_gram_inputs_are_rejected():
    with pytest.raises(ValueError):
        gram_from_input_vjps(torch.ones(2, 3))
    with pytest.raises(FloatingPointError):
        gram_from_input_vjps(torch.full((2, 3, 4), float("nan")))
    with pytest.raises(ValueError):
        subspace_energy_fraction(torch.zeros(2, 2), torch.eye(2))
