"""Unbiased channel-Gramian sketches of a frozen full-vocabulary readout.

For P selected output positions, V vocabulary entries and input states h_t,
let J_t be the Jacobian of all P*V raw logits with respect to h_t. Independent
Rademacher signs with scale 1/sqrt(P*V) give

    E[sum_t g_t g_t.T] = sum_t J_t.T J_t / (P*V).

The vocabulary projection is performed before autograd. The caller must retain
the actual final RMSNorm (and the decoder block) in the differentiated graph.
These helpers never construct a vocabulary-by-input-state Jacobian or cache a
probe bank. Reuse an advancing generator across examples; do not reseed every
example to the same state.
"""
from __future__ import annotations

import math

import torch


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _floating_matrix(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def normalization_metadata(
    vocabulary_size: int, output_positions: int, probes: int
) -> dict[str, int | float | str]:
    """Describe the normalization; input positions are summed, not averaged."""
    _positive_int(vocabulary_size, "vocabulary_size")
    _positive_int(output_positions, "output_positions")
    _positive_int(probes, "probes")
    components = vocabulary_size * output_positions
    return {
        "output_metric": "raw full-vocabulary logits at selected output positions",
        "estimator": "independent Rademacher VJP outer products",
        "vocabulary_size": vocabulary_size,
        "output_positions": output_positions,
        "output_components": components,
        "probe_scale": 1.0 / math.sqrt(components),
        "probes": probes,
        "probe_reduction": "mean",
        "input_position_reduction": "sum",
        "expected_gramian": "sum_t J_t.T @ J_t / (output_positions * vocabulary_size)",
    }


@torch.no_grad()
def project_sign_probes(
    weight: torch.Tensor,
    output_positions: int,
    probes: int,
    generator: torch.Generator,
    vocab_chunk_size: int = 4096,
) -> torch.Tensor:
    """Return fresh projected signs [probes, positions, hidden_width].

    ``weight`` is the frozen readout [vocabulary, hidden_width]. Each element of
    the conceptual [probes, positions, vocabulary] sign tensor is independent;
    vocabulary chunks bound both that tensor and the readout dtype-conversion
    temporary. Accumulation uses FP32, or FP64 when the readout is FP64.

    A fixed seed and chunk configuration reproduce the result. Changing chunk
    sizes may change the random draws' layout, while preserving the estimator.
    The supplied generator must match the weight device, as required by Torch.
    """
    _floating_matrix(weight, "weight")
    _positive_int(output_positions, "output_positions")
    _positive_int(probes, "probes")
    _positive_int(vocab_chunk_size, "vocab_chunk_size")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicitly seeded torch.Generator")
    vocabulary, width = weight.shape
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    projected = torch.zeros(
        (probes, output_positions, width), device=weight.device, dtype=dtype
    )
    for start in range(0, vocabulary, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, vocabulary)
        signs = torch.randint(
            0, 2, (probes, output_positions, stop - start),
            device=weight.device, dtype=torch.int8, generator=generator,
        ).to(dtype)
        signs.mul_(2).sub_(1)
        projected.add_(signs @ weight[start:stop].to(dtype))
    return projected.mul_(1.0 / math.sqrt(output_positions * vocabulary))


@torch.no_grad()
def project_sign_probe_groups(
    weight: torch.Tensor,
    output_positions: int,
    probes: int,
    generator: torch.Generator,
    vocab_chunk_size: int = 4096,
    *,
    groups: int = 2,
) -> torch.Tensor:
    """Return [groups, probes, positions, width] from independent fresh draws.

    Use two groups for a subspace-stability diagnostic. The generator advances
    between groups and calls, so neither groups nor examples reuse a probe bank.
    """
    _positive_int(groups, "groups")
    return torch.stack([
        project_sign_probes(
            weight, output_positions, probes, generator, vocab_chunk_size
        )
        for _ in range(groups)
    ])


def gram_from_input_vjps(input_vjps: torch.Tensor) -> torch.Tensor:
    """Sum input-position outer products and average probes from [K,L,d].

    The VJPs must already include the output normalization used by
    :func:`project_sign_probes`. This function does not divide by L and does not
    add another vocabulary/position normalization.
    """
    if not isinstance(input_vjps, torch.Tensor):
        raise TypeError("input_vjps must be a tensor")
    if input_vjps.ndim != 3 or min(input_vjps.shape) == 0:
        raise ValueError("input_vjps must have nonempty [probes, input_positions, width] shape")
    if not input_vjps.is_floating_point():
        raise TypeError("input_vjps must have a floating-point dtype")
    if not torch.isfinite(input_vjps).all():
        raise FloatingPointError("input_vjps contains nonfinite values")
    dtype = torch.float64 if input_vjps.dtype == torch.float64 else torch.float32
    rows = input_vjps.to(dtype).reshape(-1, input_vjps.shape[-1])
    return (rows.T @ rows) / input_vjps.shape[0]


def subspace_energy_fraction(gram: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return tr(U.T G U)/tr(G), for a PSD G and orthonormal columns of U.

    The caller verifies orthonormality once when constructing the basis. This
    helper deliberately does not clamp the fraction, which would hide invalid
    inputs or numerical problems. Evaluate a basis from one probe group against
    the other group's Gramian to inspect stability without reusing its fit.
    """
    _floating_matrix(gram, "gram")
    _floating_matrix(basis, "basis")
    if gram.shape[0] != gram.shape[1] or basis.shape[0] != gram.shape[0]:
        raise ValueError("gram must be square and match the basis input width")
    if basis.shape[1] > basis.shape[0] or gram.device != basis.device:
        raise ValueError("basis must fit the input width and share the Gramian device")
    dtype = torch.float64 if torch.float64 in (gram.dtype, basis.dtype) else torch.float32
    g, u = gram.to(dtype), basis.to(dtype)
    total = g.trace()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("Gramian trace must be finite and positive")
    captured = (u * (g @ u)).sum()
    if not torch.isfinite(captured):
        raise FloatingPointError("Captured subspace energy is nonfinite")
    return captured / total
