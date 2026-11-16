"""An explicit attention-only intervention on the recovered compact Qwen stack.

The softmax control delegates to the existing Torch executor. The intervention
uses the same native-dtype QK matmul and original scale, replacing only selected
layers' softmax with normalized 1+s+s²/2. Existing polynomial MLPs, affine
coordinates, RoPE, GQA, retained heads and cast boundaries remain in place.

This module does not launch training or claim recovery/speed. Its snapshot uses
the existing packed tensor layout under a distinct schema, requiring this
module's loader so attention mode cannot silently disappear.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .jspace_compact_model import arithmetic_dtype
from .jspace_qwen_compact_stack import CompactQwenStack
from .jspace_qwen_affine_fast import _run_reduced, affine_qkv_post, gate_swiglu_features
from .jspace_quadratic_qwen import _allowed
from . import jspace_3b_packed_response_model as packed

SCHEMA = "jspace.attention-retry-packed.v1"
ATTENTION_SCHEMA = "jspace.attention-retry.v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _settings(mode, layers, available):
    layers = tuple(layers)
    if mode not in ("softmax", "quadratic"):
        raise ValueError("Attention mode must be softmax or quadratic")
    if (not layers or any(type(layer) is not int for layer in layers)
            or tuple(sorted(set(layers))) != layers or not set(layers) <= set(available)):
        raise ValueError("Declare sorted distinct absolute target layers inside this stack")
    return dict(schema=ATTENTION_SCHEMA, mode=mode, layers=list(layers),
        score_arithmetic="original_activation_dtype_QK_and_native_scale",
        normalization_arithmetic="fp64_for_fp64_otherwise_fp32",
        executor="torch", polynomial="1+s+s^2/2", stable_rescaling=False)


def quadratic_probabilities(scores, *, attention_mask=None, query_positions=None):
    """Normalize native-rounded scores; never square excluded mask sentinels.

Boolean masks mean allowed; additive masks use zero for allowed and nonzero
for excluded. Causality is enforced independently. Entirely excluded rows
return finite zero values and zero gradients. Direct polynomial evaluation has
the same finite-score-square range limitation as the original quadratic study;
there is no hidden clipping, temperature change or stability rescaling.
"""
    if scores.ndim != 4 or not scores.is_floating_point() or not scores.shape[-1]:
        raise ValueError("Scores require floating [batch, heads, queries, keys]")
    allowed = _allowed(scores, attention_mask, query_positions)
    with torch.autocast(device_type=scores.device.type, enabled=False):
        safe = torch.where(allowed, scores.to(arithmetic_dtype(scores)), 0.)
        weights = torch.where(allowed, 1 + safe + .5 * safe.square(), 0.)
        return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


def quadratic_reduced(block, z, positions, *, attention_mask=None, past_key_value=None,
                      cache_position=None, position_ids=None, position_embeddings=None,
                      padded_width=None, feature_i=None, feature_j=None):
    """Existing reduced Torch algebra, with only attention normalization changed."""
    if z.ndim != 3 or z.shape[-1] != block.rank or position_embeddings is None:
        raise ValueError("Reduced coordinates and native rotary embeddings required")
    logical = block.intermediate_size + block.polynomial_feature_count
    padded_width = logical if padded_width is None else padded_width
    padded = padded_width != logical
    if padded and (not block.packed or padded_width < logical or block.capture_mlp):
        raise ValueError("Padded execution requires packed weights without activation capture")
    batch, length = z.shape[:2]
    activation_dtype, compute_dtype = z.dtype, arithmetic_dtype(z)
    energy = block._rows(block.anchor_energy, positions, batch, length)
    denominator = block.rms_denominator(z, energy, block.input_eps)
    raw = block._linear(z, block.qkv_weight)
    if padded:
        qkv = affine_qkv_post(raw, denominator, block, positions, use_triton=False,
                             activation_dtype=activation_dtype)
    else:
        shift = block._shift(block.qkv_shift, block.qkv_offset, positions, batch, length, compute_dtype)
        qkv = ((raw.to(compute_dtype) + shift) / denominator
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
        k, v = k.repeat_interleave(block.group, 1), v.repeat_interleave(block.group, 1)
    # Preserve the production BF16 matmul and BF16 scaling before promotion.
    # The old normalized_causal_attention promotes Q/K before multiplying and
    # therefore is not an arithmetic-matched control for this recovered stack.
    scores = (q @ k.transpose(-1, -2)) * block.scale
    probabilities = quadratic_probabilities(scores, attention_mask=attention_mask,
        query_positions=cache_position if cache_position is not None else position_ids).to(activation_dtype)
    probabilities = F.dropout(probabilities, p=block.attention_dropout, training=block.training)
    mixed = (probabilities @ v).transpose(1, 2).reshape(batch, length, -1)
    update = (block._linear(mixed, block.attention_output).to(compute_dtype)
              + block.attention_output_bias.to(compute_dtype)).to(activation_dtype)
    z = z + update
    denominator = block.rms_denominator(z, energy, block.post_eps)
    raw = block._linear(z, block.gate_up_weight)
    if padded:
        features = gate_swiglu_features(raw, denominator, block, positions,
            feature_i=feature_i, feature_j=feature_j, padded_width=padded_width,
            use_triton=False, activation_dtype=activation_dtype)
        update = block._linear(features, block.mlp_output)
        if block.has_polynomial:
            update = update.to(compute_dtype) + block.polynomial_bias.to(compute_dtype)
        return z + update.to(activation_dtype), probabilities
    shift = block._shift(block.gate_up_shift, block.gate_up_offset, positions, batch, length, compute_dtype)
    gate, up = ((raw.to(compute_dtype) + shift) / denominator).to(activation_dtype).chunk(2, -1)
    hidden = F.silu(gate) * up
    block.last_mlp_hidden = hidden if block.capture_mlp else None
    return z + block._mlp_update(hidden, activation_dtype, compute_dtype), probabilities


class AttentionRetryStack(CompactQwenStack):
    """Same registered weights/buffers, with explicit per-layer attention mode."""

    @classmethod
    def from_stack(cls, stack, *, mode="quadratic", layers=(35,), copy=True):
        if type(stack) is not CompactQwenStack:
            raise TypeError("Adopt an ordinary CompactQwenStack; do not nest retry wrappers")
        settings = _settings(mode, layers, stack.layer_indices)
        # No constructor/refolding/bridge recomputation. copy=False transfers
        # ownership and is intended for a freshly restored private CPU stack.
        owned = deepcopy(stack) if copy else stack
        result = cls.__new__(cls)
        result.__dict__.update(owned.__dict__)
        result.attention_retry = settings
        result.enabled = False
        return result

    def _can_fuse(self, hidden):
        # Parent PackedResponseModel may toggle enabled on Ampere. Both arms
        # stay on the identical Torch executor regardless of that switch.
        return False

    def get_extra_state(self):
        return dict(super().get_extra_state(), attention_retry=deepcopy(self.attention_retry))

    def set_extra_state(self, state):
        if _json(state) != _json(self.get_extra_state()):
            raise ValueError("Attention retry mode/layers or compact structure differ")

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        if past_key_value is not None or use_cache:
            raise ValueError("Attention retry currently qualifies cache-free execution only")
        if self.attention_retry["mode"] == "softmax":
            return super().forward(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width or position_embeddings is None:
            raise ValueError("Native [batch, sequence, width] input and rotary embeddings required")
        if self.geometry_compute_dtype is not None and hidden_states.dtype != self.geometry_compute_dtype:
            raise ValueError("Explicit worker compute requires matching activation dtype")
        checkpointing = self.checkpoint_blocks and torch.is_grad_enabled()
        if checkpointing and (past_key_value is not None or use_cache or output_attentions or any(b.capture_mlp for b in self.blocks)):
            raise ValueError("Activation checkpointing requires uncached execution without attention/MLP capture")
        positions = [b._positions(hidden_states, cache_position, position_ids, past_key_value) for b in self.blocks]
        z = self._geometry(hidden_states, self.blocks[0].basis.T)
        attentions = []
        for i, block in enumerate(self.blocks):
            if i:
                bridge = self.bridges[i - 1]
                p = positions[i - 1 + bridge.position_source]
                shift = block._rows(bridge.shift, p, *z.shape[:2])
                z = self._geometry(z, bridge.weight, shift)

            def run(value, index=i, current=block):
                common = dict(attention_mask=attention_mask, past_key_value=past_key_value,
                    cache_position=cache_position, position_embeddings=position_embeddings,
                    padded_width=current.mlp_output.shape[1] if current.packed else current.intermediate_size + current.polynomial_feature_count,
                    feature_i=getattr(self, f"feature_i_{index}"), feature_j=getattr(self, f"feature_j_{index}"))
                if current.layer_idx in self.attention_retry["layers"]:
                    return quadratic_reduced(current, value, positions[index], position_ids=position_ids, **common)
                return _run_reduced(current, value, positions[index], fuse_qkv=False, fuse_mlp=False, **common)

            if checkpointing:
                def coordinates_only(value, execute=run):
                    return execute(value)[0]
                z = checkpoint(coordinates_only, z, use_reentrant=False)
            else:
                z, probabilities = run(z)
                if output_attentions:
                    attentions.append(probabilities)
        last = self.blocks[-1]
        anchor = last._rows(last.anchor, positions[-1], *hidden_states.shape[:2])
        output = self._geometry(z, last.basis, anchor)
        return (output, tuple(attentions)) if output_attentions else (output,)


def restore_unpacked(state, description):
    """Strict raw checkpoint reload without recompiling any fitted layer."""
    from .jspace_3b_usable_recovery import restore_unpacked as restore
    extra = dict(state["_extra_state"])
    metadata = extra.pop("attention_retry", None)
    if metadata is None:
        raise ValueError("Raw checkpoint must explicitly declare retry attention")
    ordinary = restore(dict(state, _extra_state=extra), description)
    result = AttentionRetryStack.from_stack(ordinary, mode=metadata["mode"], layers=metadata["layers"], copy=False)
    result.load_state_dict(state, strict=True)
    return result


class PackedAttentionRetryModel(packed.PackedResponseModel):
    def __init__(self, stack, norm, embedding, rotary, **kwargs):
        if not isinstance(stack, AttentionRetryStack):
            raise TypeError("Packed retry requires an explicit AttentionRetryStack")
        super().__init__(stack, norm, embedding, rotary, **kwargs)
        self.stack.enabled = False
        self.inference_backend = "packed_bf16_torch_attention_retry"


def _carrier(model):
    """A temporary serialization view; shares immutable tensors, never forwards."""
    if not isinstance(model.stack, AttentionRetryStack) or model.native_prefix is not None:
        raise ValueError("Detach the native prefix and export an explicit retry model")
    stack = CompactQwenStack.__new__(CompactQwenStack)
    stack.__dict__.update(model.stack.__dict__)
    view = packed.PackedResponseModel.__new__(packed.PackedResponseModel)
    view.__dict__.update(model.__dict__)
    # Copy module registry before substitution, preserving the original model.
    view._modules = dict(model._modules)
    view.stack = stack
    view.inference_backend = "tensor_carrier_only_attention_defined_by_outer_manifest"
    return view


def _bound_digest(metadata, tensor_state):
    return hashlib.sha256(_json(dict(attention_retry=metadata, tensor_state=tensor_state)).encode()).hexdigest()


def export_packed_snapshot(model, directory, *, identities):
    """Export one tensor copy with mandatory mode/layer metadata; append-only."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    view = _carrier(model)
    settings = model.stack.attention_retry
    if settings != _settings(settings["mode"], settings["layers"], model.stack.layer_indices):
        raise ValueError("Unrecognized attention execution policy")
    model._guard()
    state = packed.packed_state_digest(view)
    description = dict(packed._describe(view), schema=SCHEMA, attention_retry=deepcopy(settings))
    tensors = {"stack."+name:value.detach().cpu().contiguous().clone()
               for name, value in view.stack.state_dict().items() if isinstance(value, torch.Tensor)}
    tensors["norm.weight"] = model.norm.weight.detach().cpu().contiguous().clone()
    tensors["embedding"] = model.embedding.detach().cpu().contiguous().clone()
    tensors.update({"rotary."+name:value.detach().cpu().contiguous().clone()
                    for name, value in model.rotary.named_buffers()})
    tensors["rotary.original_inv_freq"] = model.rotary.original_inv_freq.detach().cpu().contiguous().clone()
    description["tensor_specs"] = packed.artifact._tensor_specs(tensors)
    directory.mkdir(parents=True)
    packed._write(directory / "structure.json", description)
    packed.save_file(tensors, directory / "weights.safetensors", metadata={"format":"pt", "schema":SCHEMA})
    del tensors
    files = {name:packed.artifact.sha(directory/name) for name in ("structure.json", "weights.safetensors")}
    receipt = dict(schema=SCHEMA, attention_retry=deepcopy(settings), identities=dict(identities),
        files=files, cut=model.cut, packed=True, tensor_state_sha256=state["packed_state_sha256"],
        packed_state_sha256=_bound_digest(settings, state["packed_state_sha256"]),
        tensor_bytes=state["tensor_bytes"], embedding_copies=1, inference_backend=model.inference_backend,
        scope="Attention mode and target layers are required executable state")
    packed._write(directory / "manifest.json", receipt)
    total = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    return dict(receipt, directory=str(directory.resolve()), serialized_bytes=total,
                manifest_sha256=packed.artifact.sha(directory / "manifest.json"))


def load_packed_snapshot(directory, device="cpu", guard=None):
    device = packed.source.execution._guard(device, guard)
    directory = Path(directory)
    receipt = json.loads((directory / "manifest.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("packed") is not True:
        raise ValueError("Explicit attention-retry snapshot required")
    if set(receipt["files"]) != {"structure.json", "weights.safetensors"}:
        raise ValueError("Unexpected attention-retry artifact inventory")
    for name, wanted in receipt["files"].items():
        if packed.artifact.sha(directory/name) != wanted:
            raise ValueError("Packed attention-retry artifact changed: " + name)
    description = json.loads((directory / "structure.json").read_text())
    settings = receipt["attention_retry"]
    if description.get("schema") != SCHEMA or description.get("attention_retry") != settings:
        raise ValueError("Packed mode, layers or tensor identity changed")
    with packed.safe_open(directory / "weights.safetensors", framework="pt", device="cpu") as stream:
        if stream.metadata().get("schema") != SCHEMA:
            raise ValueError("Packed tensor-file execution schema differs")
        tensors = {name:stream.get_tensor(name) for name in stream.keys()}
    if packed.artifact._tensor_specs(tensors) != description["tensor_specs"]:
        raise ValueError("Packed tensor shape/dtype/byte inventory differs")
    ordinary = packed._restore_stack(description["stack"], tensors, device)
    if settings != _settings(settings["mode"], settings["layers"], ordinary.layer_indices):
        raise ValueError("Unrecognized attention execution policy")
    stack = AttentionRetryStack.from_stack(ordinary, mode=settings["mode"], layers=settings["layers"], copy=False)
    config = packed.Qwen2Config.from_dict(description["config"])
    norm = packed.Qwen2RMSNorm(config.hidden_size, eps=description["norm_epsilon"])
    norm.weight = nn.Parameter(tensors["norm.weight"].to(device), requires_grad=False)
    rotary = packed.source.partial.training_rotary(config, "cpu")
    expected = {"rotary."+name for name, _ in rotary.named_buffers()} | {"rotary.original_inv_freq"}
    if {name for name in tensors if name.startswith("rotary.")} != expected:
        raise ValueError("Saved rotary buffer registration differs")
    for name, _ in list(rotary.named_buffers()):
        setattr(rotary, name, tensors["rotary."+name].to(device))
    rotary.original_inv_freq = tensors["rotary.original_inv_freq"].to(device)
    rotary.attention_scaling = description["rotary_attention_scaling"]
    rotary.max_seq_len_cached = description["rotary_max_seq_len_cached"]
    rotary.original_max_seq_len = description["rotary_original_max_seq_len"]
    model = PackedAttentionRetryModel(stack, norm, tensors["embedding"].to(device), rotary,
        config=config, max_length=description["max_length"], guard=guard)
    state = packed.packed_state_digest(_carrier(model))
    if (state["packed_state_sha256"] != receipt["tensor_state_sha256"]
            or _bound_digest(settings, state["packed_state_sha256"]) != receipt["packed_state_sha256"]
            or state["tensor_bytes"] != receipt["tensor_bytes"] or model.cut != receipt["cut"]):
        raise ValueError("Fresh packed retry reload changed executable state")
    return model, dict(receipt, directory=str(directory.resolve()), strict_state_load=True,
                      packed_state_exact=True, inference_backend=model.inference_backend)
