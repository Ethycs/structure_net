"""Causal normalized quadratic attention in reduced-coordinate Qwen blocks.

This is a separate architectural experiment, not a change to the frozen affine
or Groebner campaigns. The native SwiGLU width is retained. Both attention arms
use the same Q/K/V projections, GQA expansion, RoPE, causal support, arithmetic
precision and residual path; only their normalization kernel differs.

The quadratic kernel is p(s)=1+s+s²/2. It is strictly positive but not monotonic
over all real scores and is not an exact softmax replacement. Recovery and held-
out evaluation are necessary. The implementation forms the Q×K score matrix;
it does not claim linear attention complexity or introduce feature expansions.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .jspace_compact_model import arithmetic_dtype, rotate_half
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack


def _mode(mode):
    if mode not in ("quadratic", "softmax"):
        raise ValueError("Attention mode must be quadratic or softmax")
    return mode


def _allowed(scores, attention_mask, query_positions):
    batch, _, queries, keys = scores.shape
    if query_positions is None:
        # The default describes uncached attention or an appended dynamic-cache
        # chunk. A static cache requires explicit absolute query positions.
        positions = torch.arange(keys-queries, keys, device=scores.device)[None]
    else:
        positions = query_positions
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("Absolute query positions must be integer tensors")
        if positions.ndim == 1:
            positions = positions[None]
        if positions.ndim != 2 or positions.shape[-1] != queries or positions.shape[0] not in (1, batch):
            raise ValueError("Query positions must broadcast to [batch, query length]")
        positions = positions.to(scores.device)
    allowed = torch.arange(keys, device=scores.device)[None, None, None, :] <= positions[:, None, :, None]
    if attention_mask is not None:
        if attention_mask.ndim not in (2, 3, 4) or attention_mask.shape[-1] < keys:
            raise ValueError("Mask must be [Q,K], [B,Q,K], or [B,H,Q,K]")
        if attention_mask.shape[-2] not in (1, queries):
            raise ValueError("Mask query dimension must be 1 or the current query length")
        mask = attention_mask[..., :keys].to(scores.device)
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.dtype != torch.bool:
            if not mask.is_floating_point():
                raise ValueError("Attention mask must be boolean or an additive floating mask")
            # Qwen's additive mask is zero on allowed entries and negative on
            # excluded entries. Arbitrary additive score biases are not part
            # of this API: only exactly zero means allowed.
            mask = mask == 0
        allowed = allowed & mask
    return allowed


def normalized_causal_attention(q, k, v, *, scale, attention_mask=None, query_positions=None,
                                mode="quadratic", dropout_p=0., training=False, stable=False):
    """Return (weighted values, probabilities) for head-expanded Q/K/V.

    Inputs are [batch, heads, tokens, head_width]. Bool masks use True=allowed;
    floating masks use 0=allowed and nonzero=excluded. Causality always applies.
    The mask is applied to polynomial weights, never by squaring -inf scores.
    All-masked rows return zero weights/output and finite zero gradients.

    The default deployed path evaluates literal 1+s+.5*s², then normalizes.
    It adds no maximum reduction or score rescaling. Callers must reject
    nonfinite outputs when scores overflow this FP32/FP64 arithmetic.

    Explicit stable=True instead evaluates .5*((s/t + 1/t)**2 + (1/t)**2),
    t=max(1,max(abs(s))). Its common t^-2 cancels during normalization. That
    diagnostic variant avoids finite-score square overflow but has extra work;
    the block/stack wrappers use the fast direct default.
    """
    _mode(mode)
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[:2] != k.shape[:2]
            or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1] or not k.shape[-2]):
        raise ValueError("Expected compatible head-expanded [B,H,T,D] Q/K/V")
    if not all(x.is_floating_point() and x.device == q.device and x.dtype == q.dtype for x in (q, k, v)):
        raise ValueError("Q/K/V must have one floating dtype and device")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Native attention scale must be finite and positive")
    if not 0 <= dropout_p < 1:
        raise ValueError("Invalid attention dropout")
    dtype = arithmetic_dtype(q)
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (q.to(dtype) @ k.to(dtype).transpose(-1, -2))*scale
        allowed = _allowed(scores, attention_mask, query_positions)
        if mode == "quadratic":
            # Mask before evaluation to avoid squaring an excluded sentinel,
            # then zero excluded polynomial weights (p(0) itself is nonzero).
            safe = torch.where(allowed, scores, 0.)
            if stable:
                magnitude = safe.abs().amax(-1, keepdim=True).clamp_min(1.).detach()
                inverse = magnitude.reciprocal()
                polynomial = .5*((safe/magnitude+inverse).square()+inverse.square())
            else:
                polynomial = 1+safe+.5*safe.square()
            weights = torch.where(allowed, polynomial, 0.)
            denominator = weights.sum(-1, keepdim=True)
            probabilities = weights/denominator.clamp_min(torch.finfo(dtype).tiny)
        else:
            has_keys = allowed.any(-1, keepdim=True)
            masked = scores.masked_fill(~allowed, -torch.inf)
            safe = torch.where(has_keys, masked, 0.)
            probabilities = torch.where(allowed, safe.softmax(-1), 0.)
    probabilities = F.dropout(probabilities.to(q.dtype), p=dropout_p, training=training)
    return probabilities @ v, probabilities


class RopePairMetric(nn.Module):
    """Positive metric shared within a KV group and constant on each RoPE pair.

    M=diag(exp(log_m), exp(log_m)) in Qwen's half-split pair ordering. Applying
    sqrt(M) to both Q and K commutes with the selected RoPE rotations, preserves
    shared GQA keys, and can be folded into incoming affine Q/K projections.
    Arbitrary dense or query-specific symmetric metrics are not implied.
    """
    def __init__(self, nkv, head_dim, *, device=None, dtype=torch.float32):
        super().__init__()
        if nkv <= 0 or head_dim <= 0 or head_dim % 2:
            raise ValueError("Metric requires positive KV heads and complete rotary pairs")
        self.nkv, self.head_dim = nkv, head_dim
        self.log_metric = nn.Parameter(torch.zeros(nkv, head_dim//2, device=device, dtype=dtype))

    def pair_scale(self):
        return (.5*self.log_metric).exp()

    def forward(self, q, k):
        if k.shape[1] != self.nkv or q.shape[1] % self.nkv or q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError("Metric and GQA dimensions differ")
        half = self.pair_scale()
        scale = torch.cat((half, half), -1)
        query = scale.repeat_interleave(q.shape[1]//self.nkv, 0)
        return q*query[None, :, None].to(q.dtype), k*scale[None, :, None].to(k.dtype)

    @torch.no_grad()
    def fold_into(self, block):
        if block.nkv != self.nkv or block.head_dim != self.head_dim:
            raise ValueError("Metric and affine block dimensions differ")
        half = self.pair_scale()
        if not torch.isfinite(half).all() or (half <= 0).any():
            raise ValueError("Cannot deploy a nonfinite or underflowed Q/K metric")
        key = torch.cat((half, half), -1)
        query = key.repeat_interleave(block.group, 0)
        rows = torch.cat((query.reshape(-1), key.reshape(-1), torch.ones_like(key).reshape(-1)))
        block.qkv_weight.mul_(rows[:, None].to(block.qkv_weight.dtype))
        block.qkv_shift.mul_(rows[None].to(block.qkv_shift.dtype))
        block.qkv_bias.mul_(rows.to(block.qkv_bias.dtype))
        if block.qkv_offset is not None:
            block.qkv_offset.mul_(rows.to(block.qkv_offset.dtype))
        return block


def _check_block(block):
    if not isinstance(block, AffineQwenBlock):
        raise TypeError("Expected an AffineQwenBlock")
    if block.has_polynomial or block.polynomial_feature_count or block.intermediate_size != block.native_intermediate_size:
        raise ValueError("Quadratic-attention experiment retains the complete native SwiGLU; no polynomial MLP")


def forward_reduced(block, coordinates, positions, *, mode="quadratic", metric=None, attention_mask=None,
                    past_key_value=None, cache_position=None, position_ids=None, position_embeddings=None,
                    query_indices=None):
    """Differentiable reduced block execution without native reconstruction."""
    if position_embeddings is None or coordinates.ndim != 3 or coordinates.shape[-1] != block.rank:
        raise ValueError("Reduced [B,T,rank] coordinates and native rotary embeddings are required")
    if past_key_value is not None and cache_position is None and position_ids is None:
        raise ValueError("Cached attention requires explicit absolute positions, including static caches")
    if block.capture_mlp and coordinates.is_cuda and torch.cuda.is_current_stream_capturing():
        raise ValueError("Disable diagnostic MLP capture before graph capture")
    batch, length = coordinates.shape[:2]
    activation_dtype, dtype = coordinates.dtype, arithmetic_dtype(coordinates)
    energy = block._rows(block.anchor_energy, positions, batch, length)
    z = coordinates
    denominator = block.rms_denominator(z, energy, block.input_eps)
    qwidth = block.nq*block.head_dim
    query_positions = cache_position if cache_position is not None else position_ids
    mask = attention_mask.get(block.attention_type) if isinstance(attention_mask, dict) else attention_mask
    if query_indices is None:
        shift = block._shift(block.qkv_shift, block.qkv_offset, positions, batch, length, dtype)
        qkv = ((block._linear(z, block.qkv_weight).to(dtype)+shift)/denominator+block.qkv_bias.to(dtype)).to(activation_dtype)
        q, k, v = qkv.split((qwidth, block.nkv*block.head_dim, block.nkv*block.head_dim), -1)
        query_length = length
    else:
        if query_indices.ndim != 1 or query_indices.dtype != torch.long or not len(query_indices):
            raise ValueError("query_indices must be a nonempty 1D LongTensor")
        query_length = len(query_indices)
        zq = z.index_select(1, query_indices)
        full_positions = positions
        positions = None if positions is None else positions.index_select(-1, query_indices)
        qshift = block._shift(block.qkv_shift[:, :qwidth], None if block.qkv_offset is None else block.qkv_offset[:qwidth],
                              positions, batch, query_length, dtype)
        kvshift = block._shift(block.qkv_shift[:, qwidth:], None if block.qkv_offset is None else block.qkv_offset[qwidth:],
                               full_positions, batch, length, dtype)
        q = ((block._linear(zq, block.qkv_weight[:qwidth]).to(dtype)+qshift)/denominator.index_select(1, query_indices)
             +block.qkv_bias[:qwidth].to(dtype)).to(activation_dtype)
        kv = ((block._linear(z, block.qkv_weight[qwidth:]).to(dtype)+kvshift)/denominator
              +block.qkv_bias[qwidth:].to(dtype)).to(activation_dtype)
        k, v = kv.chunk(2, -1)
        query_positions = query_indices if query_positions is None else query_positions.index_select(-1, query_indices)
        if mask is not None and mask.shape[-2] == length:
            mask = mask.index_select(-2, query_indices)
        z, energy = zq, energy.index_select(1, query_indices)
    q = q.reshape(batch, query_length, block.nq, block.head_dim).transpose(1, 2)
    k = k.reshape(batch, length, block.nkv, block.head_dim).transpose(1, 2)
    v = v.reshape(batch, length, block.nkv, block.head_dim).transpose(1, 2)
    if metric is not None:
        q, k = metric(q, k)
    cos, sin = position_embeddings
    if query_indices is None:
        q, k = block._rotary(q, k, cos, sin)
    else:
        qc, qs = cos.index_select(-2, query_indices), sin.index_select(-2, query_indices)
        if block.has_head_projection:
            qc, qs = qc[..., block.query_rope_columns].transpose(1, 2), qs[..., block.query_rope_columns].transpose(1, 2)
            kc, ks = cos[..., block.key_rope_columns].transpose(1, 2), sin[..., block.key_rope_columns].transpose(1, 2)
        else:
            qc, qs, kc, ks = qc[:, None], qs[:, None], cos[:, None], sin[:, None]
        q, k = q*qc+rotate_half(q)*qs, k*kc+rotate_half(k)*ks
    if past_key_value is not None:
        k, v = block._cache(past_key_value, k, v, cache_position, cos, sin)
    if block.group != 1:
        k, v = k.repeat_interleave(block.group, 1), v.repeat_interleave(block.group, 1)
    mixed, probabilities = normalized_causal_attention(q, k, v, scale=block.scale, attention_mask=mask,
        query_positions=query_positions,
        mode=mode, dropout_p=block.attention_dropout, training=block.training)
    mixed = mixed.transpose(1, 2).reshape(batch, query_length, -1)
    update = (block._linear(mixed, block.attention_output).to(dtype)+block.attention_output_bias.to(dtype)).to(activation_dtype)
    z = z+update
    denominator = block.rms_denominator(z, energy, block.post_eps)
    shift = block._shift(block.gate_up_shift, block.gate_up_offset, positions, batch, query_length, dtype)
    gate, up = ((block._linear(z, block.gate_up_weight).to(dtype)+shift)/denominator).to(activation_dtype).chunk(2, -1)
    hidden = F.silu(gate)*up
    if block.capture_mlp:
        block.last_mlp_hidden = hidden
    if block.mlp_output.shape[-1] != hidden.shape[-1]:
        hidden = F.pad(hidden, (0, block.mlp_output.shape[-1]-hidden.shape[-1]))
    return z+block._linear(hidden, block.mlp_output).to(activation_dtype), probabilities


class QuadraticQwenBlock(nn.Module):
    """Own a compiled affine block; deepcopy it first to preserve a control."""
    def __init__(self, block, *, mode="quadratic", learn_metric=False):
        super().__init__(); _check_block(block)
        self.original, self.mode = block, _mode(mode)
        self.metric = RopePairMetric(block.nkv, block.head_dim, device=block.qkv_weight.device,
            dtype=arithmetic_dtype(block.qkv_weight)) if learn_metric else None
        self.train(block.training)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("original"), name)

    def get_extra_state(self):
        return dict(version=1, mode=self.mode, metric_active=self.metric is not None)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Construct the same attention mode and metric packing before reload")

    @property
    def affine(self):
        return self.original

    @property
    def block(self):
        return self.original

    def compact_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward_coordinates(self, coordinates, attention_mask=None, position_ids=None, past_key_value=None,
                            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,
                            query_indices=None):
        block = self.original
        positions = block._positions(coordinates, cache_position, position_ids, past_key_value)
        output, probabilities = forward_reduced(block, coordinates, positions, mode=self.mode, metric=self.metric,
            attention_mask=attention_mask, past_key_value=past_key_value, cache_position=cache_position,
            position_ids=position_ids, position_embeddings=position_embeddings, query_indices=query_indices)
        return (output, probabilities) if output_attentions else output

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,
                query_indices=None, **kwargs):
        block = self.original
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != block.width:
            raise ValueError("Block input requires native-width hidden states")
        positions = block._positions(hidden_states, cache_position, position_ids, past_key_value)
        z, probabilities = self.forward_coordinates(block.encode(hidden_states), attention_mask=attention_mask,
            past_key_value=past_key_value, cache_position=cache_position, position_ids=position_ids,
            position_embeddings=position_embeddings, output_attentions=True, query_indices=query_indices)
        if query_indices is not None and positions is not None:
            positions = positions.index_select(-1, query_indices)
        anchor = block._rows(block.anchor, positions, *z.shape[:2])
        with torch.autocast(device_type=z.device.type, enabled=False):
            output = (F.linear(z.to(block.basis.dtype), block.basis)+anchor).to(hidden_states.dtype)
        return (output, probabilities) if output_attentions else (output,)

    def pack(self, dtype=None, buffer_dtype=None):
        if self.metric is not None:
            self.metric.fold_into(self.original); self.metric = None
        basis = self.original.basis
        if basis is not None and basis.dtype in (torch.float16, torch.bfloat16) and not self.original.packed:
            self.original.basis = basis.float()
        try:
            self.original.pack(dtype=dtype, buffer_dtype=buffer_dtype)
        finally:
            self.original.basis = basis
        return self.eval().requires_grad_(False)


class QuadraticQwenStack(nn.Module):
    """Retain an arbitrary contiguous stack's direct reduced-coordinate bridges."""
    def __init__(self, stack, *, mode=None, learn_metric=False, geometry_mode="fp32", pad_to_multiple=128):
        super().__init__()
        preserved_metrics = None
        if isinstance(stack, (list, tuple, nn.ModuleList)):
            wrappers = list(stack)
            if not wrappers or any(not isinstance(b, QuadraticQwenBlock) for b in wrappers):
                raise TypeError("Stack list must contain QuadraticQwenBlock wrappers")
            modes = {b.mode for b in wrappers}
            if len(modes) != 1 or mode is not None and mode not in modes:
                raise ValueError("Every wrapped block must use the same attention mode")
            if learn_metric:
                raise ValueError("Wrapper-list construction preserves existing metrics; configure them on the blocks")
            mode = wrappers[0].mode
            preserved_metrics = [b.metric for b in wrappers]
            stack = CompactQwenStack([b.block for b in wrappers], geometry_mode=geometry_mode,
                                      pad_to_multiple=pad_to_multiple)
        if not isinstance(stack, CompactQwenStack):
            raise TypeError("Expected a CompactQwenStack")
        for block in stack.blocks:
            _check_block(block)
        self.original, self.mode = stack, _mode(mode or "quadratic")
        self.metrics = nn.ModuleList(preserved_metrics if preserved_metrics is not None else
            [RopePairMetric(b.nkv, b.head_dim, device=b.qkv_weight.device,
             dtype=arithmetic_dtype(b.qkv_weight)) if learn_metric else None for b in stack.blocks])
        self.train(stack.training)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("original"), name)

    def get_extra_state(self):
        return dict(version=1, mode=self.mode, metric_active=[m is not None for m in self.metrics])

    def compact_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Construct the same attention mode and metric packing before reload")

    def forward_coordinates(self, coordinates, attention_mask=None, position_ids=None, past_key_value=None,
                            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,
                            query_indices=None):
        stack = self.original
        if coordinates.ndim != 3 or coordinates.shape[-1] != stack.blocks[0].rank or position_embeddings is None:
            raise ValueError("Stack requires first-block coordinates and rotary embeddings")
        if stack.geometry_compute_dtype is not None and coordinates.dtype != stack.geometry_compute_dtype:
            raise ValueError("Explicit worker geometry requires matching activation dtype")
        checked = stack.checkpoint_blocks and torch.is_grad_enabled()
        if checked and (past_key_value is not None or use_cache or output_attentions or any(b.capture_mlp for b in stack.blocks)):
            raise ValueError("Activation checkpointing requires uncached execution without captures")
        positions = [b._positions(coordinates, cache_position, position_ids, past_key_value) for b in stack.blocks]
        z = coordinates
        attentions = []
        for i, block in enumerate(stack.blocks):
            if i:
                bridge = stack.bridges[i-1]
                shift = block._rows(bridge.shift, positions[i-1+bridge.position_source], *z.shape[:2])
                z = stack._geometry(z, bridge.weight, shift)
            def execute(value, index=i, current=block):
                return forward_reduced(current, value, positions[index], mode=self.mode, metric=self.metrics[index],
                    attention_mask=attention_mask, past_key_value=past_key_value, cache_position=cache_position,
                    position_ids=position_ids, position_embeddings=position_embeddings,
                    query_indices=query_indices if index == len(stack.blocks)-1 else None)
            if checked:
                z = checkpoint(lambda value, run=execute: run(value)[0], z, use_reentrant=False)
            else:
                z, probabilities = execute(z)
                if output_attentions:
                    attentions.append(probabilities)
        return (z, tuple(attentions)) if output_attentions else z

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,
                query_indices=None, **kwargs):
        stack = self.original
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != stack.width:
            raise ValueError("Stack input requires native-width hidden states")
        z = stack._geometry(hidden_states, stack.blocks[0].basis.T)
        result = self.forward_coordinates(z, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings, query_indices=query_indices)
        z, attentions = result if output_attentions else (result, None)
        last = stack.blocks[-1]
        positions = last._positions(hidden_states, cache_position, position_ids, past_key_value)
        if query_indices is not None and positions is not None:
            positions = positions.index_select(-1, query_indices)
        anchor = last._rows(last.anchor, positions, *z.shape[:2])
        output = stack._geometry(z, last.basis, anchor)
        return (output, attentions) if output_attentions else (output,)

    def pack(self, dtype=None, buffer_dtype=None, *, strip_unused=True):
        for i, (block, metric) in enumerate(zip(self.original.blocks, self.metrics)):
            if metric is not None:
                metric.fold_into(block); self.metrics[i] = None
        self.original.pack(dtype=dtype, buffer_dtype=buffer_dtype, strip_unused=strip_unused)
        return self.eval().requires_grad_(False)


class FoldedReadout(nn.Module):
    """Full-vocabulary affine readout without reconstructing native hidden states.

    For a fixed global anchor a and basis U, logits are (Bz+c)/r, where
    B=E diag(gain) U, c=E diag(gain) a and
    r²=(zᵀ(UᵀU)z+2(Uᵀa)ᵀz+aᵀa)/native_width+epsilon.
    Gram/cross terms are retained even if the supplied basis is quantized or
    nonorthogonal. B and c may recover jointly; geometry remains fixed.
    This is an identity in real arithmetic, not a BF16 rounding identity.
    """
    def __init__(self, readout, basis, anchor, gain, epsilon, *, trainable=True, parameter_dtype=torch.float32):
        super().__init__()
        if (readout.ndim != 2 or basis.ndim != 2 or readout.shape[1] != basis.shape[0]
                or gain.shape != (basis.shape[0],) or not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError("Invalid native readout, basis, gain or RMS epsilon")
        if anchor.ndim == 2 and anchor.shape[0] == 1:
            anchor = anchor[0]
        if anchor.shape != (basis.shape[0],):
            raise ValueError("FoldedReadout requires a global [d] or [1,d] affine anchor")
        if any(t.device != basis.device or not t.is_floating_point() for t in (readout, anchor, gain)):
            raise ValueError("Readout geometry must use one device and floating tensors")
        if not all(torch.isfinite(t).all() for t in (readout, basis, anchor, gain)):
            raise ValueError("Readout inputs must be finite")
        dtype = torch.float64 if basis.dtype == torch.float64 else torch.float32
        with torch.no_grad(), torch.autocast(device_type=basis.device.type, enabled=False):
            u, a = basis.detach().to(dtype), anchor.detach().to(dtype)
            weighted = readout.detach().to(dtype)*gain.detach().to(dtype)[None]
            self.weight = nn.Parameter((weighted@u).to(parameter_dtype).contiguous(), requires_grad=trainable)
            self.bias = nn.Parameter((weighted@a).to(parameter_dtype).contiguous(), requires_grad=trainable)
            self.register_buffer("gram", (u.T@u).contiguous())
            self.register_buffer("cross", (u.T@a).contiguous())
            self.register_buffer("anchor_energy", a.square().sum())
        self.native_width, self.rank, self.epsilon = basis.shape[0], basis.shape[1], float(epsilon)

    def denominator(self, coordinates):
        if coordinates.shape[-1] != self.rank:
            raise ValueError("Readout coordinates have the wrong rank")
        with torch.autocast(device_type=coordinates.device.type, enabled=False):
            z = coordinates.to(self.gram.dtype)
            energy = (F.linear(z, self.gram)*z).sum(-1, keepdim=True)+2*(z*self.cross).sum(-1, keepdim=True)+self.anchor_energy
            # Gram expansion is nonnegative in real arithmetic; guard small
            # cancellation roundoff near h=0 without changing the RMS epsilon.
            return (energy.clamp_min(0)/self.native_width+self.epsilon).sqrt()

    def forward(self, coordinates, position_ids=None):
        del position_ids  # Global anchor: all absolute positions use this head.
        numerator = F.linear(coordinates.to(self.weight.dtype), self.weight, self.bias)
        return numerator.to(self.gram.dtype)/self.denominator(coordinates)

    def get_extra_state(self):
        return dict(version=1, native_width=self.native_width, rank=self.rank, epsilon=self.epsilon, anchor="global")

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Folded readout geometry metadata differs")
