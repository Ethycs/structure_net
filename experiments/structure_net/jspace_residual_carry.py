"""Keep the full residual stream while computing rank-reduced Qwen updates.

This diagnostic architecture does not replace the native network exactly. The
incoming QKV and SwiGLU numerators still approximate the omitted component with
the fitted, fixed mean shifts. The actual input complement is carried forward
unchanged, however, and its dynamic energy enters BOTH RMS denominators.
Neither native attention/MLP weights nor a parallel native branch are retained.

For an orthonormal U, write h = U z + e with e orthogonal to U. Attention and
MLP updates stay in U, so e and ||e||^2 are unchanged within this block. The
returned value is h + U(z_out-z), not the fixed-anchor reconstruction. Different
blocks can therefore have different bases without a destructive bridge.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_quadratic_qwen import QuadraticQwenBlock, forward_reduced


def dynamic_complement_energy(hidden, coordinates):
    """Differentiable ||h||²-||Uᵀh||², with roundoff clipped at zero.

    The compiled block's basis must be orthonormal. Its original constructor
    validates that condition; quantizing that basis changes this identity and
    is outside this FP32/FP64 diagnostic's contract.
    """
    if (hidden.ndim != 3 or coordinates.ndim != 3
            or hidden.shape[:2] != coordinates.shape[:2]
            or hidden.dtype not in (torch.float32, torch.float64)
            or coordinates.dtype != hidden.dtype or coordinates.device != hidden.device):
        raise ValueError("Matching FP32/FP64 hidden states and coordinates are required")
    return (hidden.square().sum(-1, keepdim=True)
            - coordinates.square().sum(-1, keepdim=True)).clamp_min(0.)


class _DynamicEnergyView:
    """A per-call view, never a mutation or monkeypatch of the owned module."""
    def __init__(self, block, energy):
        self._block = block
        self.anchor_energy = energy

    def __getattr__(self, name):
        return getattr(self._block, name)

    def _rows(self, table, positions, batch, length):
        if table is self.anchor_energy:
            return table
        return self._block._rows(table, positions, batch, length)

    @property
    def last_mlp_hidden(self):
        return self._block.last_mlp_hidden

    @last_mlp_hidden.setter
    def last_mlp_hidden(self, value):
        self._block.last_mlp_hidden = value


class ResidualCarryBlock(nn.Module):
    """Own a quadratic/softmax compact branch and preserve its input residual.

    Deepcopy ``wrapped`` first if the caller needs an unchanged control. The
    standard result is a tensor; requesting attention weights returns a pair.
    Cache and query-subset execution are deliberately unsupported in this first
    diagnostic. Global or position-specific fixed mean shifts are supported.
    """
    def __init__(self, wrapped: QuadraticQwenBlock):
        super().__init__()
        if not isinstance(wrapped, QuadraticQwenBlock):
            raise TypeError("Expected a QuadraticQwenBlock")
        basis = wrapped.affine.basis
        if basis is None or basis.dtype not in (torch.float32, torch.float64):
            raise ValueError("Residual carry needs an unstripped FP32/FP64 orthonormal basis")
        self.wrapped = wrapped
        self.train(wrapped.training)

    def __getattr__(self, name):
        if name == "forward_coordinates":
            # This wrapper deliberately requires the native-width residual.
            # Advertising its child's reduced entry point would bypass carry.
            raise AttributeError(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("wrapped"), name)

    @property
    def affine(self):
        return self.wrapped.affine

    @property
    def block(self):
        return self.affine

    def get_extra_state(self):
        return {"version": 1, "residual": "full_input_plus_compact_update",
                "rms": "dynamic_complement_energy", "incoming_shifts": "fixed_mean_approximation"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Residual-carry architecture metadata mismatch")

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, query_indices=None, **kwargs):
        block = self.affine
        if past_key_value is not None or use_cache or kwargs.get("past_key_values") is not None:
            raise ValueError("Residual-carry diagnostic does not support a KV cache")
        if query_indices is not None:
            raise ValueError("Residual-carry diagnostic requires all input queries")
        if (hidden_states.ndim != 3 or hidden_states.shape[-1] != block.width
                or hidden_states.dtype != block.basis.dtype
                or hidden_states.dtype not in (torch.float32, torch.float64)):
            raise ValueError("Input must have native width and the FP32/FP64 basis dtype")
        positions = block._positions(hidden_states, cache_position, position_ids, None)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            coordinates = F.linear(hidden_states, block.basis.T)
            energy = dynamic_complement_energy(hidden_states, coordinates)
            view = _DynamicEnergyView(block, energy)
            result, probabilities = forward_reduced(view, coordinates, positions,
                mode=self.wrapped.mode, metric=self.wrapped.metric,
                attention_mask=attention_mask, cache_position=cache_position,
                position_ids=position_ids, position_embeddings=position_embeddings)
            output = hidden_states + F.linear(result-coordinates, block.basis)
        return (output, probabilities) if output_attentions else output

    def pack(self, dtype=None, buffer_dtype=None):
        self.wrapped.pack(dtype=dtype, buffer_dtype=buffer_dtype)
        return self.eval().requires_grad_(False)
