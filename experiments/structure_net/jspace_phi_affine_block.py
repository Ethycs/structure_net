"""Selective dense Phi-4 loading and its affine residual/head/MLP compiler.

Phi-4 uses the installed Phi3 decoder: contiguous fused Q/K/V and gate/up,
full half-split RoPE, GQA, two RMSNorms and two residual additions. Non-owning
views expose those same weights to the existing algebra; no native block is
retained by the compiled block. Input embedding and output head stay UNTIED.

Only the unscaled, full-RoPE, zero-dropout Phi-4 geometry is supported. Tiny
Phi3 configurations with this algebra are useful controls; validate_phi4_config
additionally checks the official 14B dimensions. Quantized checkpoints require
a separate, explicit dequantization path and are rejected by these loaders.
Callers own GPU reservation admission. No loader creates a full language model.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
from transformers import AutoConfig
from transformers.models.phi3.modeling_phi3 import (
    Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding,
)

from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack
from .jspace_qwen_parallel_model import _selected_tensors


def validate_config(config):
    """Validate supported decoder algebra, without allocating model weights."""
    if config.model_type != "phi3" or config.hidden_act != "silu":
        raise ValueError("Expected the Phi3/Phi-4 SiLU decoder")
    if getattr(config, "partial_rotary_factor", 1.0) != 1.0 or config.rope_scaling is not None:
        raise ValueError("Only full, unscaled Phi-4 RoPE is supported")
    if config.sliding_window is not None:
        raise ValueError("Sliding-window attention is not supported")
    if any(getattr(config, name, 0.0) != 0.0 for name in ("resid_pdrop", "embd_pdrop", "attention_dropout")):
        raise ValueError("Phi-4 compilation requires zero dropout")
    if getattr(config, "attention_bias", False) or config.tie_word_embeddings:
        raise ValueError("Expected bias-free Phi-4 attention and untied embedding/readout")
    width, nq, nkv = config.hidden_size, config.num_attention_heads, config.num_key_value_heads
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in
           (width, nq, nkv, config.intermediate_size, config.num_hidden_layers)):
        raise ValueError("Positive integer Phi dimensions are required")
    if width % nq or nq % nkv or (width // nq) % 2:
        raise ValueError("Phi-4 requires integral GQA groups and even full head width")
    if getattr(config, "head_dim", width // nq) != width // nq:
        raise ValueError("Unexpected Phi head width")
    return {"model_type": "phi3", "hidden_size": width,
            "intermediate_size": config.intermediate_size, "num_attention_heads": nq,
            "num_key_value_heads": nkv, "head_dim": width // nq,
            "rms_norm_eps": config.rms_norm_eps, "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_position_embeddings,
            "partial_rotary_factor": 1.0, "rope_scaling": None,
            "tie_word_embeddings": False}


def validate_phi4_config(config):
    """Check the actual microsoft/phi-4 architectural profile, not Phi-4-mini."""
    result = validate_config(config)
    expected = dict(hidden_size=5120, intermediate_size=17920, num_hidden_layers=40,
                    num_attention_heads=40, num_key_value_heads=10, vocab_size=100352,
                    rms_norm_eps=1e-5, rope_theta=250000, max_position_embeddings=16384,
                    original_max_position_embeddings=16384)
    if any(getattr(config, key, None) != value for key, value in expected.items()):
        raise ValueError("Configuration does not match the official Phi-4 14B profile")
    if getattr(config, "architectures", None) != ["Phi3ForCausalLM"]:
        raise ValueError("Expected Phi3ForCausalLM architecture")
    return result


def load_config(model_path, config=None):
    value = deepcopy(config) if config is not None else AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    validate_config(value)
    if getattr(value, "quantization_config", None):
        raise ValueError("Dense floating weights are required; GPTQ is not silently dequantized")
    value._attn_implementation = "eager"
    return value


def _precision(dtype):
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Phi execution supports FP32, BF16 or FP16")


def load_native_block(model_path, layer, device="cpu", dtype=torch.bfloat16, *, config=None):
    _precision(dtype); config = load_config(model_path, config)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < config.num_hidden_layers:
        raise ValueError("Layer index outside Phi config")
    with torch.device("meta"):
        block = Phi3DecoderLayer(config, layer_idx=layer)
    prefix = f"model.layers.{layer}."
    values = _selected_tensors(model_path, [prefix + key for key in block.state_dict()], device, dtype)
    block.load_state_dict({key[len(prefix):]: value for key, value in values.items()}, strict=True, assign=True)
    return block.eval().requires_grad_(False)


def load_embedding(model_path, device="cpu", dtype=torch.bfloat16, *, config=None):
    _precision(dtype); load_config(model_path, config)
    return _selected_tensors(model_path, ["model.embed_tokens.weight"], device, dtype)[
        "model.embed_tokens.weight"].requires_grad_(False)


def load_rotary(model_path, device="cpu", *, config=None):
    """Use native Phi frequencies and constructor device; never Qwen RoPE."""
    return Phi3RotaryEmbedding(load_config(model_path, config), device=device).float().eval()


def load_readout_norm(model_path, device="cpu", dtype=torch.bfloat16, *, config=None):
    """Return independent output weight, final RMSNorm and native Phi RoPE."""
    _precision(dtype); config = load_config(model_path, config)
    values = _selected_tensors(model_path, ["lm_head.weight", "model.norm.weight"], device, dtype)
    norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=dtype)
    norm.load_state_dict({"weight": values["model.norm.weight"]}, strict=True)
    return (values["lm_head.weight"].requires_grad_(False), norm.eval().requires_grad_(False),
            Phi3RotaryEmbedding(config, device=device).float().eval())


def compatible_view(original):
    """Zero-copy split views of actual Phi fused weights; original is untouched."""
    if not isinstance(original, Phi3DecoderLayer):
        raise ValueError("Expected an installed Phi3DecoderLayer")
    config = original.config
    validate_config(config)
    a, mlp = original.self_attn, original.mlp
    width, dim = config.hidden_size, a.head_dim
    qrows, kvrows = config.num_attention_heads * dim, config.num_key_value_heads * dim
    expected = ((a.qkv_proj, (qrows + 2 * kvrows, width)), (a.o_proj, (width, width)),
                (mlp.gate_up_proj, (2 * config.intermediate_size, width)),
                (mlp.down_proj, (width, config.intermediate_size)))
    for projection, shape in expected:
        if not isinstance(projection, torch.nn.Linear) or projection.bias is not None or tuple(projection.weight.shape) != shape:
            raise ValueError("Unexpected fused Phi projection shape or bias")
        _precision(projection.weight.dtype)
        if projection.weight.is_meta:
            raise ValueError("Native Phi weights must be materialized before compilation")
    if any(module.p != 0 for module in (original.resid_attn_dropout, original.resid_mlp_dropout)):
        raise ValueError("Native residual dropout must be zero")

    def rows(projection, start, count):
        return SimpleNamespace(weight=projection.weight.narrow(0, start, count), bias=None, out_features=count)

    attention = SimpleNamespace(config=config, layer_idx=a.layer_idx, head_dim=dim,
        scaling=a.scaling, attention_dropout=a.attention_dropout,
        q_proj=rows(a.qkv_proj, 0, qrows), k_proj=rows(a.qkv_proj, qrows, kvrows),
        v_proj=rows(a.qkv_proj, qrows + kvrows, kvrows), o_proj=a.o_proj)
    return SimpleNamespace(self_attn=attention, input_layernorm=original.input_layernorm,
        post_attention_layernorm=original.post_attention_layernorm, attention_type="full_attention",
        training=original.training, mlp=SimpleNamespace(
            gate_proj=rows(mlp.gate_up_proj, 0, config.intermediate_size),
            up_proj=rows(mlp.gate_up_proj, config.intermediate_size, config.intermediate_size),
            down_proj=mlp.down_proj))


class AffinePhiBlock(AffineQwenBlock):
    """Phi-4 algebra compiled through shared affine/head/polynomial operations.

    Full residual rank/full heads/no polynomial is the native control in real
    arithmetic. Low precision folding and packing require measured parity;
    reduced dimensions and polynomial substitution are explicit approximations.
    External position_embeddings must come from the matching native Phi RoPE.
    """
    def __init__(self, original, basis, mean_by_position, **kwargs):
        self.phi_geometry = validate_config(original.config)
        super().__init__(compatible_view(original), basis, mean_by_position, **kwargs)

    def get_extra_state(self):
        return {"schema": "jspace.phi-affine.v1", "geometry": deepcopy(self.phi_geometry)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Phi checkpoint architecture/rotary geometry differs")


class CompactPhiStack(CompactQwenStack):
    """Contiguous Phi affine blocks; shared bridge/cache/packing implementation."""
    def __init__(self, blocks, **kwargs):
        blocks = list(blocks)
        if not blocks or any(not isinstance(block, AffinePhiBlock) for block in blocks):
            raise ValueError("CompactPhiStack requires Phi affine blocks")
        if any(block.phi_geometry != blocks[0].phi_geometry for block in blocks):
            raise ValueError("Phi stack blocks must share the native architecture and RoPE")
        super().__init__(blocks, **kwargs)

    def get_extra_state(self):
        return {**super().get_extra_state(), "architecture": "phi4_full_rotary",
                "phi_geometry": deepcopy(self.blocks[0].phi_geometry)}
