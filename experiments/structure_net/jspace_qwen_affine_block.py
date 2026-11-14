"""Compile one Qwen decoder block into a fixed affine channel subspace.

The prefix and tied vocabulary readout remain native.  Within this block, the
residual state is z and represents a[position] + U z. An optional head
projection retains complete RoPE frequency pairs in Q/K and folds an
uncentered orthonormal value basis into V and O. GQA, causal token coverage,
original attention score scaling are preserved. An optional polynomial stage
compiles selected SwiGLU channels through a supplied degree-two graph basis;
without that stage the native intermediate width remains unchanged.

All folding identities hold in real arithmetic.  BF16 matrix multiplication
and optional shift-table quantization introduce their usual rounding errors;
the full-rank control and post-pack evaluation must measure them.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_compact_model import arithmetic_dtype, rope, rotate_half


class AffineQwenBlock(nn.Module):
    """Qwen-compatible block with an r-wide affine residual state.

    ``basis`` is [native_width, rank], with orthonormal columns.
    ``mean_by_position`` is [P, native_width]; P=1 denotes a global anchor.
    For P>1, positions must be within the supplied table. ``position_policy``
    may explicitly request clamping; the default index-select raises on an
    unsupported position without a data-dependent host check in the forward.

    Basis, anchors and their energies remain in arithmetic precision, even
    after ``pack``. ``buffer_dtype`` only controls the large folded shift
    tables. The learned numerator offsets are shared over positions, avoiding
    free learned parameters for each position in the calibration table.

    Optional ``head_projection`` contains ``rope_pairs`` [KV heads, m] and
    ``value_basis`` [KV heads, native head width, 2m]. Each KV group uses the
    same Q/K frequency selection for its associated query heads. Values are
    projected without centering, so no discarded mean is silently omitted.
    """

    def __init__(self, original, basis, mean_by_position, *, trainable=True,
                 parameter_dtype=None, position_policy="error",
                 learn_offsets=True, buffer_dtype=None, head_projection=None,
                 capture_mlp=False):
        super().__init__()
        attention = original.self_attn
        config = attention.config
        reference = attention.q_proj.weight
        device = reference.device
        compute_dtype = arithmetic_dtype(reference)
        self.inference_dtype = reference.dtype
        parameter_dtype = parameter_dtype or reference.dtype
        shift_dtype = buffer_dtype or compute_dtype
        self.width = reference.shape[1]
        if basis.ndim != 2 or basis.shape[0] != self.width:
            raise ValueError("Basis must have shape [native width, rank]")
        self.rank = basis.shape[1]
        if not 0 < self.rank <= self.width:
            raise ValueError("Rank must lie in [1, native width]")
        if mean_by_position.ndim != 2 or mean_by_position.shape[1] != self.width or not len(mean_by_position):
            raise ValueError("Mean table must have shape [positive positions, native width]")
        if position_policy not in ("error", "clamp"):
            raise ValueError("Position policy must be error or explicitly clamp")
        if config.hidden_act != "silu":
            raise ValueError("This compiler preserves Qwen SwiGLU and requires hidden_act=silu")
        u = basis.detach().to(device=device, dtype=compute_dtype).contiguous()
        mean = mean_by_position.detach().to(device=device, dtype=compute_dtype)
        if not torch.isfinite(u).all() or not torch.isfinite(mean).all():
            raise ValueError("Basis and means must be finite")
        tolerance = 1e-10 if compute_dtype == torch.float64 else 2e-4
        if not torch.allclose(u.T @ u, torch.eye(self.rank, device=device, dtype=compute_dtype),
                              atol=tolerance, rtol=tolerance):
            raise ValueError("Basis columns must be orthonormal")
        # With a complete basis the mathematical anchor is exactly zero. Avoid
        # storing a position table of numerical projection roundoff.
        anchor = mean - (mean @ u) @ u.T if self.rank < self.width else mean.new_zeros(1, self.width)
        self.position_count = len(anchor)
        self.position_policy = position_policy
        self.attention_type = original.attention_type
        self.layer_idx = attention.layer_idx
        self.native_head_dim = attention.head_dim
        self.head_dim = self.native_head_dim
        self.nq = config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.group = self.nq // self.nkv
        self.scale = attention.scaling
        self.attention_dropout = attention.attention_dropout
        self.input_eps = original.input_layernorm.variance_epsilon
        self.post_eps = original.post_attention_layernorm.variance_epsilon
        self.intermediate_size = original.mlp.gate_proj.out_features
        self.native_intermediate_size = self.intermediate_size
        self.packed = False
        self.has_polynomial = False
        self.polynomial_feature_count = 0
        self.capture_mlp = bool(capture_mlp)
        self.last_mlp_hidden = None
        self.register_buffer("basis", u)
        self.register_buffer("anchor", anchor.contiguous())
        self.register_buffer("anchor_energy", anchor.square().sum(-1, keepdim=True))

        def parameter(value, requires_grad=trainable):
            return nn.Parameter(value.detach().to(parameter_dtype).contiguous(), requires_grad=requires_grad)

        def incoming(weight, gain):
            weighted = weight.detach().to(compute_dtype) * gain.detach().to(compute_dtype)[None]
            return weighted @ u, anchor @ weighted.T

        qkv_weights = torch.cat([p.weight.detach() for p in
                                 (attention.q_proj, attention.k_proj, attention.v_proj)])
        folded_qkv, shift = incoming(qkv_weights, original.input_layernorm.weight)
        bias = torch.cat([p.bias.detach() if p.bias is not None else p.weight.new_zeros(p.out_features)
                          for p in (attention.q_proj, attention.k_proj, attention.v_proj)]).to(compute_dtype)
        output = u.T @ attention.o_proj.weight.detach().to(compute_dtype)
        self.has_head_projection = head_projection is not None
        self.register_buffer("query_rope_columns", None)
        self.register_buffer("key_rope_columns", None)
        self.register_buffer("value_basis", None)
        if head_projection is not None:
            d = self.native_head_dim
            pairs = head_projection["rope_pairs"].detach().to(device=device)
            s = head_projection["value_basis"].detach().to(device=device, dtype=compute_dtype)
            if pairs.ndim != 2 or pairs.shape[0] != self.nkv or pairs.dtype not in (torch.int32, torch.int64):
                raise ValueError("RoPE pairs must be integer indices [KV heads, retained pairs]")
            if not 0 < pairs.shape[1] <= d // 2 or d % 2:
                raise ValueError("RoPE projection must retain complete native frequency pairs")
            if (pairs < 0).any() or (pairs >= d // 2).any() or any(len(row.unique()) != len(row) for row in pairs):
                raise ValueError("RoPE pair indices must be distinct and within the native half-width")
            self.head_dim = 2 * pairs.shape[1]
            if s.shape != (self.nkv, d, self.head_dim) or not torch.isfinite(s).all():
                raise ValueError("Value basis must be finite [KV heads, native width, retained width]")
            identity = torch.eye(self.head_dim, device=device, dtype=compute_dtype).expand(self.nkv, -1, -1)
            if not torch.allclose(s.transpose(-1, -2) @ s, identity, atol=tolerance, rtol=tolerance):
                raise ValueError("Value basis columns must be orthonormal within each KV group")
            columns = torch.cat((pairs, pairs + d // 2), -1).long()
            self.key_rope_columns = columns
            self.query_rope_columns = columns.repeat_interleave(self.group, 0)
            self.value_basis = s.contiguous()
            query_rows = torch.cat([self.query_rope_columns[h] + h * d for h in range(self.nq)])
            key_rows = torch.cat([columns[g] + (self.nq + g) * d for g in range(self.nkv)])
            qk_rows = torch.cat((query_rows, key_rows))
            value_weights = []
            value_shifts = []
            value_biases = []
            for g in range(self.nkv):
                sl = slice((self.nq + self.nkv + g) * d, (self.nq + self.nkv + g + 1) * d)
                value_weights.append(s[g].T @ folded_qkv[sl])
                value_shifts.append(shift[:, sl] @ s[g])
                value_biases.append(s[g].T @ bias[sl])
            folded_qkv = torch.cat((folded_qkv[qk_rows], *value_weights), 0)
            shift = torch.cat((shift[:, qk_rows], *value_shifts), -1)
            bias = torch.cat((bias[qk_rows], *value_biases), 0)
            output = torch.cat([output[:, h*d:(h+1)*d] @ s[h // self.group]
                                for h in range(self.nq)], -1)
        self.compact_head = self.head_dim < self.native_head_dim
        self.qkv_weight = parameter(folded_qkv)
        self.register_buffer("qkv_shift", shift.to(shift_dtype).contiguous())
        self.qkv_bias = parameter(bias)
        self.attention_output = parameter(output)
        if attention.o_proj.bias is None:
            self.register_buffer("attention_output_bias", u.new_zeros(self.rank))
        else:
            self.attention_output_bias = parameter(u.T @ attention.o_proj.bias.detach().to(compute_dtype))
        mlp = original.mlp
        if any(p.bias is not None for p in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)):
            raise ValueError("Qwen SwiGLU projections must be bias-free")
        gate_up_weight, shift = incoming(torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)),
                                         original.post_attention_layernorm.weight)
        self.gate_up_weight = parameter(gate_up_weight)
        self.register_buffer("gate_up_shift", shift.to(shift_dtype).contiguous())
        self.mlp_output = parameter(u.T @ mlp.down_proj.weight.detach().to(compute_dtype))
        if learn_offsets:
            self.qkv_offset = parameter(u.new_zeros(folded_qkv.shape[0]))
            self.gate_up_offset = parameter(u.new_zeros(2 * self.intermediate_size))
        else:
            self.register_parameter("qkv_offset", None)
            self.register_parameter("gate_up_offset", None)
        self.register_parameter("polynomial_output", None)
        self.register_parameter("polynomial_bias", None)
        for name in ("polynomial_anchor_positions", "polynomial_anchor_mean", "polynomial_anchor_scale",
                     "polynomial_pair_i", "polynomial_pair_j", "polynomial_kept", "polynomial_removed"):
            self.register_buffer(name, None)
        self.train(original.training)

    def compact_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_capture_mlp(self, enabled):
        """Enable a diagnostic activation reference, clearing any prior graph.

        ``last_mlp_hidden`` is the actual SiLU(gate)*up tensor, including its
        autograd connection when gradients are enabled. After polynomial
        compilation it contains only retained native channels. This hook is
        excluded from state_dict and must be disabled before CUDA capture.
        """
        self.capture_mlp = bool(enabled)
        self.last_mlp_hidden = None
        return self

    def compile_polynomial(self, fit, removed):
        """Compile fitted graph substitutions for selected native MLP channels.

        ``coefficients['2']`` predicts standardized activations from constant,
        linear and combinations-with-replacement quadratic anchor features.
        Native activation rows for eliminated channels are physically removed.
        The graph identity is algebraic; approximation error of the fitted
        neural relation must be measured separately by the experiment.
        """
        if self.packed or self.has_polynomial:
            raise ValueError("Polynomial compilation requires a fresh unpacked block")
        n = self.native_intermediate_size
        anchors = [int(v) for v in fit["anchors"]]
        removed = [int(v) for v in removed]
        if not anchors or len(set(anchors)) != len(anchors) or any(v < 0 or v >= n for v in anchors):
            raise ValueError("Polynomial anchors must be distinct native channel indices")
        if not removed or len(set(removed)) != len(removed) or any(v < 0 or v >= n for v in removed):
            raise ValueError("Removed channels must be distinct native channel indices")
        if set(anchors) & set(removed):
            raise ValueError("Polynomial anchors cannot be removed")
        feature_count = 1 + len(anchors) + len(anchors) * (len(anchors) + 1) // 2
        device, dtype = self.mlp_output.device, arithmetic_dtype(self.mlp_output)
        mean = torch.as_tensor(fit["mean"], device=device, dtype=dtype).detach()
        scale = torch.as_tensor(fit["scale"], device=device, dtype=dtype).detach()
        coefficient = torch.as_tensor(fit["coefficients"]["2"], device=device, dtype=dtype).detach()
        if mean.shape != (n,) or scale.shape != (n,) or coefficient.shape != (n, feature_count):
            raise ValueError("Polynomial fit shapes do not match native channels and quadratic features")
        if not all(torch.isfinite(v).all() for v in (mean, scale, coefficient)) or (scale <= 0).any():
            raise ValueError("Polynomial fit must be finite with positive scales")
        removed_set = set(removed)
        kept = [i for i in range(n) if i not in removed_set]
        rows = kept + [i + n for i in kept]
        parameter_dtype = self.mlp_output.dtype
        trainable = self.mlp_output.requires_grad
        weight = self.mlp_output.detach().to(dtype)
        eliminated = weight[:, removed]
        polynomial_output = (eliminated * scale[removed]) @ coefficient[removed]
        polynomial_bias = eliminated @ mean[removed]
        self.polynomial_output = nn.Parameter(polynomial_output.to(parameter_dtype).contiguous(), requires_grad=trainable)
        self.polynomial_bias = nn.Parameter(polynomial_bias.contiguous(), requires_grad=trainable)
        self.mlp_output = nn.Parameter(self.mlp_output.detach()[:, kept].contiguous(), requires_grad=trainable)
        self.gate_up_weight = nn.Parameter(self.gate_up_weight.detach()[rows].contiguous(),
                                           requires_grad=self.gate_up_weight.requires_grad)
        self.gate_up_shift = self.gate_up_shift[:, rows].contiguous()
        if self.gate_up_offset is not None:
            self.gate_up_offset = nn.Parameter(self.gate_up_offset.detach()[rows].contiguous(),
                                               requires_grad=self.gate_up_offset.requires_grad)
        indices = {original: position for position, original in enumerate(kept)}
        self.polynomial_anchor_positions = torch.tensor([indices[i] for i in anchors], device=device)
        self.polynomial_anchor_mean = mean[anchors].contiguous()
        self.polynomial_anchor_scale = scale[anchors].contiguous()
        pairs = torch.triu_indices(len(anchors), len(anchors), device=device)
        self.polynomial_pair_i, self.polynomial_pair_j = pairs[0], pairs[1]
        self.polynomial_kept = torch.tensor(kept, device=device)
        self.polynomial_removed = torch.tensor(removed, device=device)
        self.intermediate_size = len(kept)
        self.polynomial_feature_count = feature_count
        self.has_polynomial = True
        self.last_mlp_hidden = None
        return self

    def _polynomial_features(self, hidden):
        z = (hidden[..., self.polynomial_anchor_positions].to(self.polynomial_anchor_mean.dtype)
             - self.polynomial_anchor_mean) / self.polynomial_anchor_scale
        return torch.cat((torch.ones_like(z[..., :1]), z,
                          z[..., self.polynomial_pair_i] * z[..., self.polynomial_pair_j]), -1)

    def _mlp_update(self, hidden, activation_dtype, compute_dtype):
        if not self.has_polynomial:
            return self._linear(hidden, self.mlp_output).to(activation_dtype)
        phi = self._polynomial_features(hidden)
        if self.packed:
            value = self._linear(torch.cat((hidden, phi.to(hidden.dtype)), -1), self.mlp_output).to(compute_dtype)
        else:
            value = self._linear(hidden, self.mlp_output).to(compute_dtype)
            value = value + self._linear(phi, self.polynomial_output).to(compute_dtype)
        return (value + self.polynomial_bias.to(compute_dtype)).to(activation_dtype)

    def _positions(self, hidden_states, cache_position, position_ids, past_key_value):
        batch, length = hidden_states.shape[:2]
        if self.position_count == 1:
            return None
        if cache_position is not None:
            positions = cache_position
        elif position_ids is not None:
            positions = position_ids
        elif past_key_value is None:
            positions = torch.arange(length, device=hidden_states.device)
        else:
            raise ValueError("Cached position-specific anchors require explicit absolute positions")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("Positions must be integer tensors")
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.ndim != 2 or positions.shape[-1] != length or positions.shape[0] not in (1, batch):
            raise ValueError("Position shape must broadcast to [batch, sequence]")
        positions = positions.to(device=hidden_states.device, dtype=torch.long).expand(batch, length)
        if self.position_policy == "clamp":
            positions = positions.clamp(0, self.position_count - 1)
        return positions

    @staticmethod
    def _rows(table, positions, batch, length):
        if positions is None:
            return table[:1].unsqueeze(0).expand(batch, length, table.shape[-1])
        return table.index_select(0, positions.reshape(-1)).view(batch, length, table.shape[-1])

    def encode(self, hidden_states):
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return F.linear(hidden_states.to(self.basis.dtype), self.basis.T).to(hidden_states.dtype)

    def rms_denominator(self, coordinates, anchor_energy, eps):
        # Original width is essential: dividing by rank would rescale the model.
        energy = coordinates.to(arithmetic_dtype(coordinates)).square().sum(-1, keepdim=True)
        return ((energy + anchor_energy.to(energy.dtype)) / self.width + eps).sqrt()

    @staticmethod
    def _linear(value, weight):
        return F.linear(value.to(weight.dtype), weight)

    def _shift(self, table, offset, positions, batch, length, dtype):
        shift = self._rows(table, positions, batch, length).to(dtype)
        return shift if offset is None else shift + offset.to(dtype)

    def _rotary(self, q, k, cos, sin):
        if not self.has_head_projection:
            return rope(q, k, cos, sin)
        kcos = cos[..., self.key_rope_columns].transpose(1, 2)
        ksin = sin[..., self.key_rope_columns].transpose(1, 2)
        # Broadcast each KV group's frequencies over its query heads without
        # materializing a duplicate cosine table for every query head.
        grouped = q.reshape(q.shape[0], self.nkv, self.group, q.shape[2], self.head_dim)
        q = (grouped * kcos.unsqueeze(2) + rotate_half(grouped) * ksin.unsqueeze(2)).reshape_as(q)
        return q, k * kcos + rotate_half(k) * ksin

    def _cache(self, cache, k, v, cache_position, cos, sin):
        # StaticCache is allocated from the unchanged native model config. Pad
        # only its storage. Slice immediately after update so both attention
        # matmuls still use the compact head width. DynamicCache stores compact
        # heads directly and needs no padding.
        storage_width = self.head_dim
        existing = getattr(cache, "key_cache", ())
        if len(existing) > self.layer_idx and isinstance(existing[self.layer_idx], torch.Tensor):
            item = existing[self.layer_idx]
            if item.numel():
                storage_width = item.shape[-1]
        if storage_width not in (self.head_dim, self.native_head_dim):
            raise ValueError("Cache head width is incompatible with this projection")
        if storage_width > self.head_dim:
            k = F.pad(k, (0, storage_width - self.head_dim))
            v = F.pad(v, (0, storage_width - self.head_dim))
        k, v = cache.update(k, v, self.layer_idx,
                            {"cache_position": cache_position, "cos": cos, "sin": sin})
        return k[..., :self.head_dim], v[..., :self.head_dim]

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        if position_embeddings is None:
            raise ValueError("Qwen rotary position_embeddings are required")
        if self.capture_mlp and hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Diagnostic MLP capture must be disabled before CUDA graph capture")
        batch, length, _ = hidden_states.shape
        activation_dtype = hidden_states.dtype
        compute_dtype = arithmetic_dtype(hidden_states)
        positions = self._positions(hidden_states, cache_position, position_ids, past_key_value)
        energy = self._rows(self.anchor_energy, positions, batch, length)
        z = self.encode(hidden_states)
        denominator = self.rms_denominator(z, energy, self.input_eps)
        shift = self._shift(self.qkv_shift, self.qkv_offset, positions, batch, length, compute_dtype)
        qkv = ((self._linear(z, self.qkv_weight).to(compute_dtype) + shift) / denominator
               + self.qkv_bias.to(compute_dtype)).to(activation_dtype)
        q, k, v = qkv.split((self.nq * self.head_dim, self.nkv * self.head_dim,
                             self.nkv * self.head_dim), -1)
        q = q.view(batch, length, self.nq, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.nkv, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.nkv, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = self._rotary(q, k, cos, sin)
        if past_key_value is not None:
            k, v = self._cache(past_key_value, k, v, cache_position, cos, sin)
        if self.group != 1:
            k = k.repeat_interleave(self.group, 1)
            v = v.repeat_interleave(self.group, 1)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask[..., :length, :k.shape[-2]]
        elif length > 1:
            causal = torch.ones(length, k.shape[-2], device=z.device, dtype=torch.bool).triu(1 + k.shape[-2] - length)
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probabilities = scores.to(compute_dtype).softmax(-1).to(activation_dtype)
        probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
        mixed = (probabilities @ v).transpose(1, 2).reshape(batch, length, -1)
        update = (self._linear(mixed, self.attention_output).to(compute_dtype)
                  + self.attention_output_bias.to(compute_dtype)).to(activation_dtype)
        z = z + update
        denominator = self.rms_denominator(z, energy, self.post_eps)
        shift = self._shift(self.gate_up_shift, self.gate_up_offset, positions, batch, length, compute_dtype)
        gate, up = ((self._linear(z, self.gate_up_weight).to(compute_dtype) + shift)
                    / denominator).to(activation_dtype).chunk(2, -1)
        hidden = F.silu(gate) * up
        self.last_mlp_hidden = hidden if self.capture_mlp else None
        z = z + self._mlp_update(hidden, activation_dtype, compute_dtype)
        anchor = self._rows(self.anchor, positions, batch, length)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            output = (F.linear(z.to(self.basis.dtype), self.basis) + anchor).to(activation_dtype)
        return (output, probabilities) if output_attentions else (output,)

    def pack(self, dtype=None, buffer_dtype=None):
        """Fold shared offsets into tables, optionally quantize, then freeze.

        With unchanged parameter/table dtypes the offset folding preserves the
        same arithmetic expression. Passing BF16 explicitly quantizes the
        learned weights and/or shift tables and requires a fresh quality check.
        Basis, anchors and anchor energies remain in arithmetic precision.
        Polynomial columns are concatenated into the native output GEMM; that
        algebraic rearrangement can change floating-point accumulation order.
        """
        if self.packed:
            return self
        dtype = dtype or self.inference_dtype
        for table_name, offset_name in (("qkv_shift", "qkv_offset"), ("gate_up_shift", "gate_up_offset")):
            table = getattr(self, table_name)
            offset = getattr(self, offset_name)
            if offset is not None:
                table = table.to(self.basis.dtype) + offset.detach().to(self.basis.dtype)
            if buffer_dtype is not None:
                table = table.to(buffer_dtype)
            setattr(self, table_name, table.contiguous())
            setattr(self, offset_name, None)
        if self.has_polynomial:
            self.mlp_output = nn.Parameter(torch.cat((self.mlp_output.detach().to(dtype),
                                                      self.polynomial_output.detach().to(dtype)), -1), requires_grad=False)
            self.polynomial_output = None
        for name, parameter in self.named_parameters():
            parameter.data = parameter.data.to(self.basis.dtype if name == "polynomial_bias" else dtype)
            parameter.requires_grad_(False)
        self.set_capture_mlp(False)
        self.packed = True
        return self
