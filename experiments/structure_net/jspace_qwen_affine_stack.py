"""Execute two compiled Qwen blocks through one fixed affine coordinate bridge.

The frozen single-block compiler remains the source of all learned weights,
normalization, RoPE, cache and polynomial helpers. Only the residual execution
is composed here: encode once, bridge reduced coordinates, reconstruct once.
The bridge is algebraically equivalent to an intermediate native boundary in
real arithmetic; eliminating that boundary also eliminates its BF16 rounding,
so deployed accuracy must be measured after packing.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_compact_model import arithmetic_dtype
from .jspace_qwen_affine_block import AffineQwenBlock


def forward_reduced(block, coordinates, positions, *, attention_mask=None,
                    past_key_value=None, cache_position=None, position_embeddings=None):
    """Run an existing compiled block on its own r-wide residual coordinates.

    Positions have already passed the block's position-policy checks. Returns
    (coordinates, attention probabilities), with no native reconstruction.
    """
    if position_embeddings is None:
        raise ValueError("Qwen rotary position_embeddings are required")
    if coordinates.ndim != 3 or coordinates.shape[-1] != block.rank:
        raise ValueError("Reduced state must have shape [batch, sequence, block rank]")
    if block.capture_mlp and coordinates.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Diagnostic MLP capture must be disabled before CUDA graph capture")
    batch, length, _ = coordinates.shape
    activation_dtype = coordinates.dtype
    compute_dtype = arithmetic_dtype(coordinates)
    energy = block._rows(block.anchor_energy, positions, batch, length)
    z = coordinates
    denominator = block.rms_denominator(z, energy, block.input_eps)
    shift = block._shift(block.qkv_shift, block.qkv_offset, positions, batch, length, compute_dtype)
    qkv = ((block._linear(z, block.qkv_weight).to(compute_dtype) + shift) / denominator
           + block.qkv_bias.to(compute_dtype)).to(activation_dtype)
    q, k, v = qkv.split((block.nq * block.head_dim, block.nkv * block.head_dim,
                         block.nkv * block.head_dim), -1)
    q = q.view(batch, length, block.nq, block.head_dim).transpose(1, 2)
    k = k.view(batch, length, block.nkv, block.head_dim).transpose(1, 2)
    v = v.view(batch, length, block.nkv, block.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = block._rotary(q, k, cos, sin)
    if past_key_value is not None:
        k, v = block._cache(past_key_value, k, v, cache_position, cos, sin)
    if block.group != 1:
        k = k.repeat_interleave(block.group, 1)
        v = v.repeat_interleave(block.group, 1)
    scores = (q @ k.transpose(-1, -2)) * block.scale
    if attention_mask is not None:
        scores = scores + attention_mask[..., :length, :k.shape[-2]]
    elif length > 1:
        causal = torch.ones(length, k.shape[-2], device=z.device, dtype=torch.bool).triu(1 + k.shape[-2] - length)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    probabilities = scores.to(compute_dtype).softmax(-1).to(activation_dtype)
    probabilities = F.dropout(probabilities, p=block.attention_dropout, training=block.training)
    mixed = (probabilities @ v).transpose(1, 2).reshape(batch, length, -1)
    update = (block._linear(mixed, block.attention_output).to(compute_dtype)
              + block.attention_output_bias.to(compute_dtype)).to(activation_dtype)
    z = z + update
    denominator = block.rms_denominator(z, energy, block.post_eps)
    shift = block._shift(block.gate_up_shift, block.gate_up_offset, positions, batch, length, compute_dtype)
    gate, up = ((block._linear(z, block.gate_up_weight).to(compute_dtype) + shift)
                / denominator).to(activation_dtype).chunk(2, -1)
    hidden = F.silu(gate) * up
    block.last_mlp_hidden = hidden if block.capture_mlp else None
    return z + block._mlp_update(hidden, activation_dtype, compute_dtype), probabilities


class AffineQwenStack(nn.Module):
    """Own two adjacent AffineQwenBlocks and preserve their native cache slots.

    Replace the two corresponding decoder-list entries with this single entry;
    leave the original model config/cache layer count unchanged. The ordinary
    forward returns (native_hidden,). With output_attentions=True its second
    element is the tuple of the two constituent attention tensors. Therefore
    HF attention collection contains one stack entry holding two tensors,
    rather than two ordinary layer entries; default attention collection is off.

    Position tables must have equal lengths, or one must be global. Two
    non-global tables must use the same explicit position policy. Fixed bridge
    buffers remain in FP32/FP64 even when learned weights/tables pack to BF16.
    """

    def __init__(self, first, second):
        super().__init__()
        if not isinstance(first, AffineQwenBlock) or not isinstance(second, AffineQwenBlock) or first is second:
            raise ValueError("Stack requires two distinct compiled AffineQwenBlocks")
        if first.width != second.width or first.native_head_dim != second.native_head_dim:
            raise ValueError("Stack blocks must share native residual and rotary head widths")
        if second.layer_idx != first.layer_idx + 1:
            raise ValueError("Stack blocks must retain adjacent original layer indices")
        if first.attention_type != second.attention_type:
            raise ValueError("Stack blocks must share an attention-mask type")
        if first.basis.device != second.basis.device or first.basis.dtype != second.basis.dtype:
            raise ValueError("Fixed coordinate bases must share device and arithmetic dtype")
        if first.position_count != second.position_count and min(first.position_count, second.position_count) != 1:
            raise ValueError("Anchor tables must have equal position counts or a global row")
        if min(first.position_count, second.position_count) > 1 and first.position_policy != second.position_policy:
            raise ValueError("Position-specific blocks must share the same position policy")
        self.blocks = nn.ModuleList([first, second])
        self.width = first.width
        self.ranks = (first.rank, second.rank)
        self.layer_idx = first.layer_idx
        self.layer_indices = (first.layer_idx, second.layer_idx)
        self.attention_type = first.attention_type
        self.position_count = max(first.position_count, second.position_count)
        self._bridge_position_block = 0 if first.position_count > 1 else 1
        with torch.no_grad(), torch.autocast(device_type=first.basis.device.type, enabled=False):
            self.register_buffer("bridge_weight", (second.basis.T @ first.basis).detach().contiguous())
            anchors = first.anchor.expand(self.position_count, -1) - second.anchor.expand(self.position_count, -1)
            self.register_buffer("bridge_shift", (anchors @ second.basis).detach().contiguous())
        self.training = first.training

    @property
    def packed(self):
        return all(block.packed for block in self.blocks)

    def compact_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def bridge(self, coordinates, positions):
        """Map z1 into z2 without constructing the native-width residual."""
        batch, length = coordinates.shape[:2]
        shift = self.blocks[0]._rows(self.bridge_shift, positions, batch, length)
        with torch.autocast(device_type=coordinates.device.type, enabled=False):
            return (F.linear(coordinates.to(self.bridge_weight.dtype), self.bridge_weight)
                    + shift).to(coordinates.dtype)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        if position_embeddings is None:
            raise ValueError("Qwen rotary position_embeddings are required")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise ValueError("Stack input must have the original native residual width")
        positions = [block._positions(hidden_states, cache_position, position_ids, past_key_value)
                     for block in self.blocks]
        z = self.blocks[0].encode(hidden_states)
        attentions = []
        for index, block in enumerate(self.blocks):
            if index:
                z = self.bridge(z, positions[self._bridge_position_block])
            z, probabilities = forward_reduced(block, z, positions[index], attention_mask=attention_mask,
                past_key_value=past_key_value, cache_position=cache_position, position_embeddings=position_embeddings)
            if output_attentions:
                attentions.append(probabilities)
        last = self.blocks[-1]
        anchor = last._rows(last.anchor, positions[-1], *hidden_states.shape[:2])
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            output = (F.linear(z.to(last.basis.dtype), last.basis) + anchor).to(hidden_states.dtype)
        return (output, tuple(attentions)) if output_attentions else (output,)

    def pack(self, dtype=None, buffer_dtype=None):
        for block in self.blocks:
            block.pack(dtype=dtype, buffer_dtype=buffer_dtype)
        return self
