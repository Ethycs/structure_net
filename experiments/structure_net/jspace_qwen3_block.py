"""Qwen3 compact affine blocks, stacks and selective local loading.

Q/K are projected and RMS-normalized at the ORIGINAL head width, then complete
RoPE pairs are selected. V/O, residual coordinates and graph-polynomial MLPs
use the existing compiler algebra. This is intentionally not the BF16 Qwen2
Triton path. Parameters/activations explicitly use FP32 or BF16; affine geometry remains
FP32. FP32 controls promote official BF16 tensors without inventing source bits.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding,
)

from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack
from .jspace_qwen_parallel_model import _selected_tensors


def _fp32(dtype):
    if dtype != torch.float32:
        raise ValueError("Qwen3 geometry requires FP32")


def _precision(dtype):
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Qwen3 execution supports only FP32 or BF16")


def _execution(value):
    _precision(value.dtype)
    if torch.is_autocast_enabled(value.device.type):
        raise ValueError("Autocast is incompatible with explicit Qwen3 execution precision")


def load_config(model_path, config=None):
    value = deepcopy(config) if config is not None else AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    if value.model_type != "qwen3":
        raise ValueError("Expected the official dense Qwen3 architecture")
    if value.use_sliding_window or any(t != "full_attention" for t in value.layer_types):
        raise ValueError("This port requires full causal attention in every layer")
    value._attn_implementation = "eager"
    return value


def load_native_block(model_path, layer, device="cpu", dtype=torch.bfloat16, *, config=None):
    _precision(dtype); config = load_config(model_path, config)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < config.num_hidden_layers:
        raise ValueError("Layer index outside model config")
    with torch.device("meta"):
        block = Qwen3DecoderLayer(config, layer_idx=layer)
    prefix = f"model.layers.{layer}."
    values = _selected_tensors(model_path, [prefix + k for k in block.state_dict()], device, dtype)
    block.load_state_dict({k[len(prefix):]: v for k, v in values.items()}, strict=True, assign=True)
    return block.eval().requires_grad_(False)


def load_embedding(model_path, device="cpu", dtype=torch.bfloat16, *, config=None):
    _precision(dtype); load_config(model_path, config)
    return _selected_tensors(model_path, ["model.embed_tokens.weight"], device, dtype)[
        "model.embed_tokens.weight"].requires_grad_(False)


def load_rotary(model_path, device="cpu", dtype=torch.bfloat16, *, config=None):
    _precision(dtype); config = load_config(model_path, config)
    return Qwen3RotaryEmbedding(config=config, device=device).float().eval()


def load_readout_norm(model_path, device="cpu", dtype=torch.bfloat16, *, config=None):
    """Return native output weight, final norm and RoPE; do not tie input E."""
    _precision(dtype); config = load_config(model_path, config)
    key = "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    values = _selected_tensors(model_path, [key, "model.norm.weight"], device, dtype)
    norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=dtype)
    norm.load_state_dict({"weight": values["model.norm.weight"]}, strict=True)
    rotary = Qwen3RotaryEmbedding(config=config, device=device).float().eval()
    return values[key].requires_grad_(False), norm.eval().requires_grad_(False), rotary


class Qwen3AffineBlock(AffineQwenBlock):
    """Native-width QK RMSNorm followed by optional compact paired attention.

    Full residual rank/full heads/no polynomial removal is an algebraic native
    control (FP32 roundoff remains). Reduced rank/heads/polynomial relations
    are explicit approximations requiring recovery and task evaluation.
    """
    def __init__(self, original, basis, mean_by_position, *, trainable=True,
                 parameter_dtype=None, position_policy="error", learn_offsets=True,
                 buffer_dtype=torch.float32, head_projection=None, capture_mlp=False):
        parameter_dtype = parameter_dtype or original.self_attn.q_proj.weight.dtype
        _precision(parameter_dtype); _fp32(buffer_dtype)
        _fp32(basis.dtype); _fp32(mean_by_position.dtype)
        if original.self_attn.config.model_type != "qwen3":
            raise ValueError("Qwen3AffineBlock requires a native Qwen3 decoder")
        for tensor in list(original.parameters()) + list(original.buffers()):
            if tensor.is_floating_point(): _precision(tensor.dtype)
        if original.attention_type != "full_attention":
            raise ValueError("Only full causal attention is supported")
        # The old compiler may fold the affine projections, but it must NOT
        # select Q/K rows before Qwen3's nonlinear native-width normalization.
        super().__init__(original, basis, mean_by_position, trainable=trainable,
                         parameter_dtype=torch.float32, position_policy=position_policy,
                         learn_offsets=learn_offsets, buffer_dtype=buffer_dtype,
                         head_projection=None, capture_mlp=capture_mlp)
        attn = original.self_attn
        self.q_norm_eps, self.k_norm_eps = attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon
        self.q_norm_weight = nn.Parameter(attn.q_norm.weight.detach().to(parameter_dtype).clone(), requires_grad=trainable)
        self.k_norm_weight = nn.Parameter(attn.k_norm.weight.detach().to(parameter_dtype).clone(), requires_grad=trainable)
        if head_projection is not None:
            d, device = self.native_head_dim, self.basis.device
            pairs = head_projection["rope_pairs"].detach().to(device=device)
            values = head_projection["value_basis"].detach().to(device=device, dtype=torch.float32)
            if pairs.ndim != 2 or pairs.shape[0] != self.nkv or pairs.dtype not in (torch.int32, torch.int64):
                raise ValueError("RoPE pairs must be integer [KV heads, retained pairs]")
            if d % 2 or not 0 < pairs.shape[1] <= d // 2 or (pairs < 0).any() or (pairs >= d // 2).any():
                raise ValueError("RoPE pairs outside native half-width")
            if any(len(row.unique()) != len(row) for row in pairs):
                raise ValueError("RoPE pairs must be distinct")
            self.head_dim = pairs.shape[1] * 2
            eye = torch.eye(self.head_dim, device=device).expand(self.nkv, -1, -1)
            if values.shape != (self.nkv, d, self.head_dim) or not torch.isfinite(values).all():
                raise ValueError("Value basis must be finite [KV heads, native width, retained width]")
            if not torch.allclose(values.transpose(-1, -2) @ values, eye, atol=2e-4, rtol=2e-4):
                raise ValueError("Value basis columns must be orthonormal")
            self.key_rope_columns = torch.cat((pairs, pairs + d // 2), -1).long()
            self.query_rope_columns = self.key_rope_columns.repeat_interleave(self.group, 0)
            self.value_basis = values.contiguous()
            full_qk = (self.nq + self.nkv) * d
            for name in ("qkv_weight", "qkv_bias", "qkv_offset"):
                source = getattr(self, name)
                if source is not None:
                    tail = [values[g].T @ source.detach()[full_qk + g*d:full_qk + (g+1)*d].float()
                            for g in range(self.nkv)]
                    value = torch.cat((source.detach()[:full_qk].float(), *tail), 0).to(parameter_dtype)
                    setattr(self, name, nn.Parameter(value.contiguous(), requires_grad=trainable))
            self.qkv_shift = torch.cat((self.qkv_shift[:, :full_qk], *[
                self.qkv_shift[:, full_qk + g*d:full_qk + (g+1)*d] @ values[g]
                for g in range(self.nkv)]), -1).contiguous()
            output = torch.cat([self.attention_output.detach()[:, h*d:(h+1)*d].float() @ values[h // self.group]
                                for h in range(self.nq)], -1)
            self.attention_output = nn.Parameter(output.to(parameter_dtype).contiguous(), requires_grad=trainable)
            self.has_head_projection, self.compact_head = True, self.head_dim < d
        # Fold native BF16 weights and geometry in FP32, then round each final
        # compiled parameter once. Geometry/offset tables remain FP32.
        for parameter in self.parameters():
            parameter.data = parameter.data.to(parameter_dtype)
        self.inference_dtype = parameter_dtype

    @staticmethod
    def _head_norm(value, gain, eps):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + eps)
        return normalized.to(value.dtype) * gain

    def _normalized_selected_qk(self, q, k):
        q = self._head_norm(q, self.q_norm_weight, self.q_norm_eps)
        k = self._head_norm(k, self.k_norm_weight, self.k_norm_eps)
        if self.has_head_projection:
            q = torch.gather(q, -1, self.query_rope_columns[None, :, None, :].expand(*q.shape[:-1], self.head_dim))
            k = torch.gather(k, -1, self.key_rope_columns[None, :, None, :].expand(*k.shape[:-1], self.head_dim))
        return q, k

    def _mlp_update(self, hidden, activation_dtype, compute_dtype):
        # A stack may align packed polynomial GEMM columns with zero padding.
        if self.packed:
            value = torch.cat((hidden, self._polynomial_features(hidden).to(hidden.dtype)), -1) if self.has_polynomial else hidden
            value = F.pad(value, (0, self.mlp_output.shape[1] - value.shape[-1]))
            output = F.linear(value.to(self.mlp_output.dtype), self.mlp_output)
            return (output.float() + self.polynomial_bias).to(activation_dtype) if self.has_polynomial else output.to(activation_dtype)
        return super()._mlp_update(hidden, activation_dtype, compute_dtype)

    def forward_reduced(self, z, positions=None, *, attention_mask=None, past_key_value=None,
                        cache_position=None, position_embeddings=None):
        _execution(z)
        if z.ndim != 3 or z.shape[-1] != self.rank or position_embeddings is None:
            raise ValueError("Compact [B,L,rank] input and native RoPE are required")
        batch, length = z.shape[:2]
        activation_dtype = z.dtype
        energy = self._rows(self.anchor_energy, positions, batch, length)
        denominator = self.rms_denominator(z, energy, self.input_eps)
        shift = self._shift(self.qkv_shift, self.qkv_offset, positions, batch, length, torch.float32)
        qkv = ((F.linear(z.to(self.qkv_weight.dtype), self.qkv_weight).float() + shift)
               / denominator + self.qkv_bias.float()).to(activation_dtype)
        q, k, v = qkv.split((self.nq * self.native_head_dim, self.nkv * self.native_head_dim,
                             self.nkv * self.head_dim), -1)
        q = q.view(batch, length, self.nq, self.native_head_dim).transpose(1, 2)
        k = k.view(batch, length, self.nkv, self.native_head_dim).transpose(1, 2)
        v = v.view(batch, length, self.nkv, self.head_dim).transpose(1, 2)
        q, k = self._normalized_selected_qk(q, k)
        cos, sin = position_embeddings
        if cos.dtype != activation_dtype or sin.dtype != activation_dtype:
            raise ValueError("RoPE tables must match activation precision")
        q, k = self._rotary(q, k, cos, sin)
        if past_key_value is not None:
            k, v = self._cache(past_key_value, k, v, cache_position, cos, sin)
        if self.group != 1:
            k, v = k.repeat_interleave(self.group, 1), v.repeat_interleave(self.group, 1)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask[..., :length, :k.shape[-2]]
        elif length > 1:
            mask = torch.ones(length, k.shape[-2], device=z.device, dtype=torch.bool).triu(1 + k.shape[-2] - length)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probabilities = F.dropout(scores.float().softmax(-1).to(activation_dtype), p=self.attention_dropout, training=self.training)
        mixed = (probabilities @ v).transpose(1, 2).reshape(batch, length, -1)
        update = (F.linear(mixed.to(self.attention_output.dtype), self.attention_output).float()
                  + self.attention_output_bias.float()).to(activation_dtype)
        z = z + update
        denominator = self.rms_denominator(z, energy, self.post_eps)
        shift = self._shift(self.gate_up_shift, self.gate_up_offset, positions, batch, length, torch.float32)
        gate, up = ((F.linear(z.to(self.gate_up_weight.dtype), self.gate_up_weight).float() + shift)
                    / denominator).to(activation_dtype).chunk(2, -1)
        hidden = F.silu(gate) * up
        self.last_mlp_hidden = hidden if self.capture_mlp else None
        return z + self._mlp_update(hidden, activation_dtype, torch.float32), probabilities

    def forward_coordinates(self, coordinates, attention_mask=None, position_ids=None,
                            past_key_value=None, cache_position=None, position_embeddings=None, **kwargs):
        positions = self._positions(coordinates, cache_position, position_ids, past_key_value)
        return self.forward_reduced(coordinates, positions, attention_mask=attention_mask,
            past_key_value=past_key_value, cache_position=cache_position, position_embeddings=position_embeddings)[0]

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        _execution(hidden_states)
        positions = self._positions(hidden_states, cache_position, position_ids, past_key_value)
        z, probabilities = self.forward_reduced(self.encode(hidden_states), positions,
            attention_mask=attention_mask, past_key_value=past_key_value, cache_position=cache_position,
            position_embeddings=position_embeddings)
        output = (F.linear(z.float(), self.basis) + self._rows(self.anchor, positions, *z.shape[:2])).to(hidden_states.dtype)
        return (output, probabilities) if output_attentions else (output,)

    def pack(self, dtype=None, buffer_dtype=torch.float32):
        dtype = dtype or self.inference_dtype
        _precision(dtype); _fp32(buffer_dtype)
        return super().pack(dtype=dtype, buffer_dtype=buffer_dtype)


class Qwen3CompactStack(CompactQwenStack):
    """Encode-once suffix with FP32 geometry; never uses old Qwen2 fused kernels."""
    def __init__(self, blocks, *, checkpoint_blocks=False, pad_to_multiple=128):
        blocks = list(blocks)
        if not blocks or any(not isinstance(b, Qwen3AffineBlock) for b in blocks):
            raise ValueError("Expected Qwen3AffineBlocks")
        super().__init__(blocks, geometry_mode="fp32", enabled=False,
                         checkpoint_blocks=checkpoint_blocks, pad_to_multiple=pad_to_multiple)

    @property
    def rank(self):
        if len(set(self.ranks)) != 1:
            raise ValueError("This stack has varying ranks; use ranks instead")
        return self.ranks[0]

    def _can_fuse(self, hidden):
        return False

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        _execution(hidden_states)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width or position_embeddings is None:
            raise ValueError("Native [B,L,width] input and native RoPE are required")
        checkpointing = self.checkpoint_blocks and torch.is_grad_enabled()
        if checkpointing and (past_key_value is not None or use_cache or output_attentions or any(b.capture_mlp for b in self.blocks)):
            raise ValueError("Checkpointing requires uncached execution without diagnostic capture")
        positions = [b._positions(hidden_states, cache_position, position_ids, past_key_value) for b in self.blocks]
        z = F.linear(hidden_states.float(), self.blocks[0].basis.T).to(hidden_states.dtype)
        attentions = []
        for i, block in enumerate(self.blocks):
            if i:
                bridge = self.bridges[i - 1]
                shift = block._rows(bridge.shift, positions[i - 1 + bridge.position_source], *z.shape[:2])
                z = (F.linear(z.float(), bridge.weight) + shift).to(hidden_states.dtype)
            def run(value, current=block, index=i):
                return current.forward_reduced(value, positions[index], attention_mask=attention_mask,
                    past_key_value=past_key_value, cache_position=cache_position, position_embeddings=position_embeddings)
            if checkpointing:
                def coordinates_only(value, execute=run): return execute(value)[0]
                z = checkpoint(coordinates_only, z, use_reentrant=False)
            else:
                z, probabilities = run(z)
                if output_attentions: attentions.append(probabilities)
        last = self.blocks[-1]
        output = (F.linear(z.float(), last.basis) + last._rows(last.anchor, positions[-1], *z.shape[:2])).to(hidden_states.dtype)
        return (output, tuple(attentions)) if output_attentions else (output,)

    def pack(self, *, dtype=None, buffer_dtype=torch.float32, strip_unused=True):
        dtype = dtype or self.blocks[0].inference_dtype
        _precision(dtype); _fp32(buffer_dtype)
        return super().pack(dtype=dtype, buffer_dtype=buffer_dtype, strip_unused=strip_unused)

    def get_extra_state(self):
        return {**super().get_extra_state(), "architecture": "qwen3_native_qknorm",
                "parameter_dtypes": tuple(str(b.qkv_weight.dtype) for b in self.blocks),
                "qk_native_widths": tuple(b.native_head_dim for b in self.blocks)}
