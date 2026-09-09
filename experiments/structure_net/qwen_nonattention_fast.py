"""Frozen Qwen prefix execution variants with official attention unchanged."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net.qwen_nonattention_pilot import int8_linear


@dataclass(frozen=True)
class Execution:
    name: str
    int8_threshold: float | None = None
    preserve_down: bool = False


VARIANTS = (Execution("fused_fp16"), Execution("int8_mlp6", 6),
            Execution("int8_mlp3", 3), Execution("int8_gate_up6", 6, True))


def rms(x, weight, eps):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def front(x, packed_weight, input_weight, query_weight, key_weight, cos, sin, heads, kvheads, dim, eps):
    b, n, _ = x.shape
    packed = F.linear(rms(x, input_weight, eps), packed_weight)
    query, key, value = packed.split((heads*dim, kvheads*dim, kvheads*dim), dim=-1)
    query = rms(query.reshape(b, n, heads, dim), query_weight, eps).transpose(1, 2)
    key = rms(key.reshape(b, n, kvheads, dim), key_weight, eps).transpose(1, 2)
    query, key = q.apply_rotary_pos_emb(query, key, cos, sin)
    return query, key, value.reshape(b, n, kvheads, dim).transpose(1, 2)


def add_norm(x, projected, weight, eps):
    residual = x + projected
    return residual, rms(residual, weight, eps)


def mlp(x, gate_weight, up_weight, down_weight):
    return F.linear(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight), down_weight)


def readout(attended, output_weight, segment_size):
    b, h, n, d = attended.shape
    heads = attended.transpose(1, 2).float().reshape(b, n//segment_size, segment_size, h*d).mean(2)
    # A linear output projection commutes with pooling in exact arithmetic.
    # Casting pooled inputs changes FP16 rounding, so fidelity is measured.
    ov = F.linear(heads.to(output_weight.dtype), output_weight).float()
    return {"heads": heads, "ov": ov, "changes": 1-F.cosine_similarity(ov[:, 1:], ov[:, :-1], dim=-1)}


class QuantizedMLP(nn.Module):
    def __init__(self, original, threshold, preserve_down):
        super().__init__()
        self.gate_proj = int8_linear(original.gate_proj, threshold)
        self.up_proj = int8_linear(original.up_proj, threshold)
        self.down_proj = original.down_proj if preserve_down else int8_linear(original.down_proj, threshold)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FastPrefix(nn.Module):
    def __init__(self, original, execution=VARIANTS[0], compile_kernels=True):
        super().__init__()
        if any(p.requires_grad for p in original.parameters()):
            raise ValueError("original model must be frozen")
        self.original = original
        self.execution = execution
        self.packed_names = []
        for index, layer in enumerate(original.layers):
            a = layer.self_attn
            name = f"packed_qkv_{index}"
            self.register_buffer(name, torch.cat((a.q_proj.weight, a.k_proj.weight, a.v_proj.weight), 0))
            self.packed_names.append(name)
        self.quantized_mlps = nn.ModuleList()
        if execution.int8_threshold is not None:
            self.quantized_mlps.extend(QuantizedMLP(layer.mlp, execution.int8_threshold, execution.preserve_down) for layer in original.layers[:5])
        def prepare(fn):
            return torch.compile(fn, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs") if compile_kernels else fn
        self.front_fn = prepare(front)
        self.add_norm_fn = prepare(add_norm)
        self.mlp_fn = prepare(mlp)
        self.readout_fn = prepare(readout)

    def forward(self, ids, setting=q.MoBASetting(1024, 2, "all"), segment_size=128):
        if ids.ndim != 2 or ids.shape[1] % segment_size:
            raise ValueError("require complete pooling segments")
        original = self.original
        b, n = ids.shape
        x = original.embed_tokens(ids)
        positions = torch.arange(n, device=ids.device)[None, :]
        cos, sin = original.rotary_emb(x, positions)
        cu = torch.arange(b+1, device=ids.device, dtype=torch.int32) * n
        config = original.config
        for index, layer in enumerate(original.layers):
            a = layer.self_attn
            query, key, value = self.front_fn(x, getattr(self, self.packed_names[index]), layer.input_layernorm.weight,
                                             a.q_norm.weight, a.k_norm.weight, cos, sin, config.num_attention_heads,
                                             config.num_key_value_heads, a.head_dim, config.rms_norm_eps)
            attended = q.moba_attention(query, key, value, setting, cu, 32) if setting.enabled(index, n) else q.flash_call(query, key, value)
            if index == 5:
                return self.readout_fn(attended, a.o_proj.weight, segment_size)
            projected = F.linear(attended.transpose(1, 2).reshape(b, n, -1), a.o_proj.weight)
            residual, normalized = self.add_norm_fn(x, projected, layer.post_attention_layernorm.weight, config.rms_norm_eps)
            del query, key, value, attended, projected, x
            if self.execution.int8_threshold is None:
                output = self.mlp_fn(normalized, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)
            else:
                output = self.quantized_mlps[index](normalized)
            x = residual + output
            del residual, normalized, output
        raise RuntimeError("missing sixth-layer readout")


def fidelity_gate(metrics, approximate):
    cosine, relative, pearson = (.995, .05, .995) if approximate else (.9999, .005, .999)
    return bool(metrics["heads_cosine_mean"] >= cosine and metrics["ov_cosine_mean"] >= cosine
                and metrics["heads_relative_l2"] <= relative and metrics["ov_relative_l2"] <= relative
                and metrics["change_pearson"] is not None and metrics["change_pearson"] >= pearson
                and metrics["top_quarter_change_f1"] >= .95)
