"""Optional fused inference and recoverable geometry for compiled Qwen blocks.

This module never changes the frozen compiler. Wrappers OWN AND MODIFY their
input module: callers must deepcopy the baseline first. ``geometry_mode=bf16``
replaces basis/bridge buffers, deliberately changing arithmetic for subsequent
joint recovery. Anchors, energies and affine bridge shifts remain FP32. No
duplicate FP32/BF16 geometry is retained. Padding changes packed GEMM shapes.

The differentiable Torch path defines the new structure during adaptation.
Optional Triton kernels preserve explicit BF16 cast boundaries but still need
GPU parity checks (especially libdevice SiLU) before accepting a deployment.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_compact_model import arithmetic_dtype
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_affine_stack import AffineQwenStack, forward_reduced

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = libdevice = None


if triton is not None:
    @triton.jit
    def _qkv_kernel(RAW, SHIFT, DENOMINATOR, BIAS, POSITIONS, OUT,
                    WIDTH: tl.constexpr, POSITIONS_COUNT: tl.constexpr,
                    GLOBAL: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        position = 0 if GLOBAL else tl.load(POSITIONS + row)
        valid = (col < WIDTH) & (position >= 0) & (position < POSITIONS_COUNT)
        raw = tl.load(RAW + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        shift = tl.load(SHIFT + position * WIDTH + col, valid, float("nan")).to(tl.float32)
        denominator = tl.load(DENOMINATOR + row).to(tl.float32)
        bias = tl.load(BIAS + col, col < WIDTH, 0).to(tl.float32)
        out = tl.div_rn(raw + shift, denominator) + bias
        tl.store(OUT + row * WIDTH + col, out.to(tl.bfloat16), col < WIDTH)

    @triton.jit
    def _hidden_at(RAW, SHIFT, DENOMINATOR, row, position, col, mask,
                   KEEP: tl.constexpr, POSITIONS_COUNT: tl.constexpr):
        valid = mask & (position >= 0) & (position < POSITIONS_COUNT)
        gate = tl.load(RAW + row * (2 * KEEP) + col, mask, 0).to(tl.float32)
        up = tl.load(RAW + row * (2 * KEEP) + KEEP + col, mask, 0).to(tl.float32)
        gs = tl.load(SHIFT + position * (2 * KEEP) + col, valid, float("nan")).to(tl.float32)
        us = tl.load(SHIFT + position * (2 * KEEP) + KEEP + col, valid, float("nan")).to(tl.float32)
        denominator = tl.load(DENOMINATOR + row).to(tl.float32)
        gate = tl.div_rn(gate + gs, denominator).to(tl.bfloat16).to(tl.float32)
        up = tl.div_rn(up + us, denominator).to(tl.bfloat16).to(tl.float32)
        # Separate SiLU rounding and multiplication rounding, matching eager
        # BF16 F.silu(gate) * up. Avoid approximate tl.sigmoid/exp and FP32 FMA.
        silu = tl.div_rn(gate, 1. + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
        return (silu * up).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _features_kernel(RAW, SHIFT, DENOMINATOR, POSITIONS, OUT, ANCHORS, MEAN, SCALE, FI, FJ,
                         KEEP: tl.constexpr, FEATURES: tl.constexpr, ANCHOR_COUNT: tl.constexpr,
                         PADDED: tl.constexpr, POSITIONS_COUNT: tl.constexpr,
                         GLOBAL: tl.constexpr, HAS_POLYNOMIAL: tl.constexpr,
                         KEEP_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
        row, block = tl.program_id(0), tl.program_id(1)
        position = 0 if GLOBAL else tl.load(POSITIONS + row)
        if block < KEEP_BLOCKS:
            col = block * BLOCK + tl.arange(0, BLOCK)
            hidden = _hidden_at(RAW, SHIFT, DENOMINATOR, row, position, col, col < KEEP,
                                KEEP, POSITIONS_COUNT)
            tl.store(OUT + row * PADDED + col, hidden.to(tl.bfloat16), col < KEEP)
        else:
            f = (block - KEEP_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
            value = tl.full((BLOCK,), 0., tl.float32)
            if HAS_POLYNOMIAL:
                valid = (f > 0) & (f < FEATURES)
                ai = tl.load(FI + f, valid, 0)
                aj = tl.load(FJ + f, valid, 0)
                ci = tl.load(ANCHORS + ai, valid, 0)
                cj = tl.load(ANCHORS + aj, valid, 0)
                hi = _hidden_at(RAW, SHIFT, DENOMINATOR, row, position, ci, valid, KEEP, POSITIONS_COUNT)
                hj = _hidden_at(RAW, SHIFT, DENOMINATOR, row, position, cj, valid, KEEP, POSITIONS_COUNT)
                mi, mj = tl.load(MEAN + ai, valid, 0), tl.load(MEAN + aj, valid, 0)
                si, sj = tl.load(SCALE + ai, valid, 1), tl.load(SCALE + aj, valid, 1)
                zi, zj = tl.div_rn(hi - mi, si), tl.div_rn(hj - mj, sj)
                value = tl.where(f == 0, 1., tl.where(f <= ANCHOR_COUNT, zi, zi * zj))
                value = tl.where(f < FEATURES, value, 0.)
            tl.store(OUT + row * PADDED + KEEP + f, value.to(tl.bfloat16), f < PADDED - KEEP)


def affine_qkv_reference(raw, shift, denominator, bias, activation_dtype=None):
    """Torch reference, with separate FP32 add/divide/add and one final cast."""
    dtype = arithmetic_dtype(raw)
    return ((raw.to(dtype) + shift.to(dtype)) / denominator.to(dtype)
            + bias.to(dtype)).to(activation_dtype or raw.dtype)


def swiglu_polynomial_reference(raw, shift, denominator, block, *, padded_width=None, activation_dtype=None):
    """Torch reference for retained SwiGLU, graph features and optional zero padding."""
    dtype = arithmetic_dtype(raw)
    gate, up = ((raw.to(dtype) + shift.to(dtype)) / denominator.to(dtype)).to(
        activation_dtype or raw.dtype).chunk(2, -1)
    hidden = F.silu(gate) * up
    if block.has_polynomial:
        hidden = torch.cat((hidden, block._polynomial_features(hidden).to(hidden.dtype)), -1)
    padded_width = padded_width or hidden.shape[-1]
    if padded_width < hidden.shape[-1]:
        raise ValueError("Padding cannot remove native or polynomial features")
    return F.pad(hidden, (0, padded_width - hidden.shape[-1])) if padded_width != hidden.shape[-1] else hidden


def affine_qkv_post(raw, denominator, block, positions, *, use_triton=False, activation_dtype=None):
    batch, length = raw.shape[:2]
    if not use_triton or triton is None or not raw.is_cuda:
        shift = block._shift(block.qkv_shift, block.qkv_offset, positions, batch, length, arithmetic_dtype(raw))
        return affine_qkv_reference(raw, shift, denominator, block.qkv_bias, activation_dtype)
    if not block.packed or raw.dtype != torch.bfloat16 or raw.requires_grad or torch.is_grad_enabled():
        raise ValueError("Triton postprocessing requires packed BF16 inference under no_grad")
    out = torch.empty_like(raw)
    indices = raw if positions is None else positions.reshape(-1).contiguous()
    width = raw.shape[-1]
    _qkv_kernel[(batch * length, triton.cdiv(width, 256))](
        raw, block.qkv_shift, denominator, block.qkv_bias, indices, out,
        width, block.position_count, positions is None, 256, num_warps=4, enable_fp_fusion=False)
    return out


def gate_swiglu_features(raw, denominator, block, positions, *, feature_i=None, feature_j=None,
                         padded_width=None, use_triton=False, activation_dtype=None):
    batch, length = raw.shape[:2]
    logical = block.intermediate_size + block.polynomial_feature_count
    padded_width = padded_width or logical
    if not use_triton or triton is None or not raw.is_cuda:
        shift = block._shift(block.gate_up_shift, block.gate_up_offset, positions, batch, length, arithmetic_dtype(raw))
        return swiglu_polynomial_reference(raw, shift, denominator, block, padded_width=padded_width,
                                           activation_dtype=activation_dtype)
    if not block.packed or raw.dtype != torch.bfloat16 or raw.requires_grad or torch.is_grad_enabled():
        raise ValueError("Triton postprocessing requires packed BF16 inference under no_grad")
    if padded_width < logical:
        raise ValueError("Padding cannot remove features")
    if block.has_polynomial and (feature_i is None or feature_j is None):
        raise ValueError("Precomputed feature index maps are required for fused polynomial inference")
    keep = block.intermediate_size
    out = torch.empty((*raw.shape[:-1], padded_width), dtype=raw.dtype, device=raw.device)
    indices = raw if positions is None else positions.reshape(-1).contiguous()
    keep_blocks = triton.cdiv(keep, 256)
    maps = (block.polynomial_anchor_positions, block.polynomial_anchor_mean, block.polynomial_anchor_scale,
            feature_i, feature_j) if block.has_polynomial else (raw,) * 5
    _features_kernel[(batch * length, keep_blocks + triton.cdiv(padded_width - keep, 256))](
        raw, block.gate_up_shift, denominator, indices, out, *maps,
        keep, block.polynomial_feature_count, len(block.polynomial_anchor_positions) if block.has_polynomial else 0,
        padded_width, block.position_count, positions is None, block.has_polynomial,
        keep_blocks, 256, num_warps=4, enable_fp_fusion=False)
    return out


def _run_reduced(block, z, positions, *, attention_mask, past_key_value, cache_position,
                 position_embeddings, fuse_qkv, fuse_mlp, padded_width, feature_i, feature_j):
    logical = block.intermediate_size + block.polynomial_feature_count
    if not fuse_qkv and not fuse_mlp and padded_width == logical:
        return forward_reduced(block, z, positions, attention_mask=attention_mask,
            past_key_value=past_key_value, cache_position=cache_position, position_embeddings=position_embeddings)
    if not block.packed:
        raise ValueError("Fused or padded reduced execution requires a packed block")
    if block.capture_mlp:
        raise ValueError("Disable diagnostic activation capture before fused/padded execution")
    batch, length = z.shape[:2]
    activation_dtype, compute_dtype = z.dtype, arithmetic_dtype(z)
    energy = block._rows(block.anchor_energy, positions, batch, length)
    denominator = block.rms_denominator(z, energy, block.input_eps)
    raw = block._linear(z, block.qkv_weight)
    qkv = affine_qkv_post(raw, denominator, block, positions, use_triton=fuse_qkv,
                          activation_dtype=activation_dtype)
    q, k, v = qkv.split((block.nq * block.head_dim, block.nkv * block.head_dim, block.nkv * block.head_dim), -1)
    q = q.view(batch, length, block.nq, block.head_dim).transpose(1, 2)
    k = k.view(batch, length, block.nkv, block.head_dim).transpose(1, 2)
    v = v.view(batch, length, block.nkv, block.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = block._rotary(q, k, cos, sin)
    if past_key_value is not None:
        k, v = block._cache(past_key_value, k, v, cache_position, cos, sin)
    if block.group != 1:
        k, v = k.repeat_interleave(block.group, 1), v.repeat_interleave(block.group, 1)
    scores = (q @ k.transpose(-1, -2)) * block.scale
    if attention_mask is not None:
        scores = scores + attention_mask[..., :length, :k.shape[-2]]
    elif length > 1:
        mask = torch.ones(length, k.shape[-2], device=z.device, dtype=torch.bool).triu(1 + k.shape[-2] - length)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probabilities = F.dropout(scores.to(compute_dtype).softmax(-1).to(activation_dtype),
                               p=block.attention_dropout, training=block.training)
    mixed = (probabilities @ v).transpose(1, 2).reshape(batch, length, -1)
    update = (block._linear(mixed, block.attention_output).to(compute_dtype)
              + block.attention_output_bias.to(compute_dtype)).to(activation_dtype)
    z = z + update
    denominator = block.rms_denominator(z, energy, block.post_eps)
    features = gate_swiglu_features(block._linear(z, block.gate_up_weight), denominator, block, positions,
        feature_i=feature_i, feature_j=feature_j, padded_width=padded_width, use_triton=fuse_mlp,
        activation_dtype=activation_dtype)
    update = block._linear(features, block.mlp_output)
    if block.has_polynomial:
        update = update.to(compute_dtype) + block.polynomial_bias.to(compute_dtype)
    return z + update.to(activation_dtype), probabilities


class _FastAffine(nn.Module):
    def __init__(self, original, *, geometry_mode="fp32", pad_to_multiple=0,
                 enabled=True, fuse_qkv=True, fuse_mlp=True):
        super().__init__()
        if geometry_mode not in ("fp32", "bf16"):
            raise ValueError("geometry_mode must be fp32 (control) or bf16 (recovered approximation)")
        if not isinstance(pad_to_multiple, int) or pad_to_multiple < 0:
            raise ValueError("pad_to_multiple must be a nonnegative integer")
        self.original = original
        self.geometry_mode, self.pad_to_multiple = geometry_mode, pad_to_multiple
        self.enabled, self.fuse_qkv, self.fuse_mlp = enabled, fuse_qkv, fuse_mlp
        self.training = original.training
        for index, block in enumerate(self.blocks):
            if geometry_mode == "fp32" and block.basis.dtype not in (torch.float32, torch.float64):
                raise ValueError("FP32 control requires the original arithmetic-precision geometry")
            if geometry_mode == "bf16":
                if block.anchor.dtype != torch.float32 or block.anchor_energy.dtype != torch.float32:
                    raise ValueError("BF16 geometry requires FP32 anchors and energy buffers")
                block.basis = block.basis.to(torch.bfloat16)
            if block.has_polynomial:
                count = len(block.polynomial_anchor_positions)
                prefix = torch.cat((torch.zeros(1, dtype=torch.long, device=block.basis.device),
                                    torch.arange(count, device=block.basis.device)))
                fi = torch.cat((prefix, block.polynomial_pair_i))
                fj = torch.cat((prefix, block.polynomial_pair_j))
            else:
                fi = fj = torch.empty(0, dtype=torch.long, device=block.basis.device)
            self.register_buffer(f"feature_i_{index}", fi)
            self.register_buffer(f"feature_j_{index}", fj)
            self._pad(block)
        if isinstance(original, AffineQwenStack) and geometry_mode == "bf16":
            original.bridge_weight = original.bridge_weight.to(torch.bfloat16)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("original"), name)

    @property
    def blocks(self):
        return self.original.blocks if isinstance(self.original, AffineQwenStack) else (self.original,)

    def _pad(self, block):
        if not block.packed:
            return
        logical = block.intermediate_size + block.polynomial_feature_count
        target = ((logical + self.pad_to_multiple - 1) // self.pad_to_multiple * self.pad_to_multiple
                  if self.pad_to_multiple else logical)
        if block.mlp_output.shape[1] > target:
            raise ValueError("Existing packed output width exceeds requested padding")
        if block.mlp_output.shape[1] < target:
            block.mlp_output = nn.Parameter(F.pad(block.mlp_output.detach(),
                (0, target - block.mlp_output.shape[1])).contiguous(), requires_grad=False)

    def compact_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def pack(self, dtype=None, buffer_dtype=None):
        for block in self.blocks:
            if not block.packed:
                # The frozen packer uses basis.dtype as its arithmetic dtype.
                # Temporarily restoring only that dtype preserves FP32 offset
                # folding and polynomial bias; no duplicate stays resident.
                basis = block.basis
                if basis.dtype == torch.bfloat16:
                    block.basis = basis.float()
                try:
                    block.pack(dtype=dtype, buffer_dtype=buffer_dtype)
                finally:
                    block.basis = basis
            self._pad(block)
        return self

    def _can_fuse(self, hidden):
        return (self.enabled and triton is not None and hidden.is_cuda and hidden.dtype == torch.bfloat16
                and not torch.is_grad_enabled() and not self.training
                and all(block.packed and not block.training and not block.capture_mlp
                        and block.qkv_weight.dtype == block.gate_up_weight.dtype == block.mlp_output.dtype == torch.bfloat16
                        and block.qkv_shift.dtype in (torch.float32, torch.bfloat16)
                        and block.gate_up_shift.dtype in (torch.float32, torch.bfloat16)
                        and (not block.has_polynomial or block.polynomial_anchor_mean.dtype == torch.float32)
                        for block in self.blocks))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        can_fuse = self._can_fuse(hidden_states)
        padded = any(block.packed and block.mlp_output.shape[1] != block.intermediate_size + block.polynomial_feature_count
                     for block in self.blocks)
        if not can_fuse and not padded:
            return self.original(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        if position_embeddings is None:
            raise ValueError("Qwen rotary position_embeddings are required")
        positions = [block._positions(hidden_states, cache_position, position_ids, past_key_value) for block in self.blocks]
        z = self.blocks[0].encode(hidden_states)
        attentions = []
        for index, block in enumerate(self.blocks):
            if index:
                z = self.original.bridge(z, positions[self.original._bridge_position_block])
            z, probabilities = _run_reduced(block, z, positions[index], attention_mask=attention_mask,
                past_key_value=past_key_value, cache_position=cache_position, position_embeddings=position_embeddings,
                fuse_qkv=can_fuse and self.fuse_qkv, fuse_mlp=can_fuse and self.fuse_mlp,
                padded_width=block.mlp_output.shape[1] if block.packed else block.intermediate_size + block.polynomial_feature_count,
                feature_i=getattr(self, f"feature_i_{index}"), feature_j=getattr(self, f"feature_j_{index}"))
            if output_attentions:
                attentions.append(probabilities)
        last = self.blocks[-1]
        anchor = last._rows(last.anchor, positions[-1], *hidden_states.shape[:2])
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            output = (F.linear(z.to(last.basis.dtype), last.basis) + anchor).to(hidden_states.dtype)
        if not output_attentions:
            return (output,)
        return output, tuple(attentions) if isinstance(self.original, AffineQwenStack) else attentions[0]


class FastAffineQwenBlock(_FastAffine):
    def __init__(self, block, **kwargs):
        if not isinstance(block, AffineQwenBlock):
            raise TypeError("Expected a compiled AffineQwenBlock")
        super().__init__(block, **kwargs)

    @classmethod
    def from_packed(cls, block, **kwargs):
        if not block.packed:
            raise ValueError("from_packed requires a packed block")
        return cls(block, **kwargs)


class FastAffineQwenStack(_FastAffine):
    def __init__(self, stack, **kwargs):
        if not isinstance(stack, AffineQwenStack):
            raise TypeError("Expected a compiled AffineQwenStack")
        super().__init__(stack, **kwargs)

    @classmethod
    def from_packed(cls, stack, **kwargs):
        if not stack.packed:
            raise ValueError("from_packed requires a packed stack")
        return cls(stack, **kwargs)
