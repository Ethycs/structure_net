"""Standalone packed Qwen artifacts, independent of dense weights and fit files.

Version 1 supports the standard tied-embedding Qwen2/SwiGLU architecture and a
complete, packed, stripped 36-block CompactQwenStack. All executable structure
is explicit JSON metadata; tensor data uses safetensors. Loading never invokes
the original compiler or accesses an original checkpoint. Native rotary config
is retained, but the artifact's calibrated position capacity is a separate hard
limit: this package does not restore the native model's long context.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RMSNorm, Qwen2RotaryEmbedding

from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack, _Bridge


SCHEMA = "jspace.qwen-allblocks-artifact.v1"
PACKED_SCHEMA = "jspace.qwen-allblocks-packed.v1"
DTYPES = {str(v).removeprefix("torch."): v for v in
          (torch.float64, torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.int32)}
BLOCK_FIELDS = ("width", "rank", "position_count", "position_policy", "attention_type", "layer_idx",
    "native_head_dim", "head_dim", "nq", "nkv", "group", "scale", "attention_dropout", "input_eps", "post_eps",
    "intermediate_size", "native_intermediate_size", "packed", "has_polynomial", "polynomial_feature_count",
    "has_head_projection", "compact_head")
PARAMETERS = ("qkv_weight", "qkv_bias", "attention_output", "gate_up_weight", "mlp_output",
              "qkv_offset", "gate_up_offset", "polynomial_output", "polynomial_bias")
BUFFERS = ("basis", "anchor", "anchor_energy", "query_rope_columns", "key_rope_columns", "value_basis",
           "qkv_shift", "gate_up_shift", "polynomial_anchor_positions", "polynomial_anchor_mean",
           "polynomial_anchor_scale", "polynomial_pair_i", "polynomial_pair_j", "polynomial_kept", "polynomial_removed")
LOCAL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "vocab.txt",
               "tokenizer.model", "spiece.model", "chat_template.jinja")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def _dtype(value):
    name = str(value).removeprefix("torch.")
    if name not in DTYPES:
        raise ValueError("Unsupported tensor dtype: " + name)
    return name


def _config(path):
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if (config.model_type != "qwen2" or config.num_hidden_layers != 36 or not config.tie_word_embeddings
            or config.hidden_act != "silu" or config.use_sliding_window
            or any(kind != "full_attention" for kind in getattr(config, "layer_types", ()) or ())):
        raise ValueError("Artifact v1 requires standard tied Qwen2, 36 layers, SwiGLU, and full attention")
    head = getattr(config, "head_dim", config.hidden_size//config.num_attention_heads)
    if config.hidden_size != head*config.num_attention_heads or head % 2 or config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("Unsupported Qwen2 head geometry")
    if getattr(config, "rope_scaling", None) not in (None, {"rope_type": "default"}, {"type": "default"}):
        raise ValueError("Artifact v1 requires native default RoPE; scaling variants need a new validated schema")
    config._attn_implementation = "eager"
    return config


def _tensor_specs(tensors):
    return {name: dict(shape=list(t.shape), dtype=_dtype(t.dtype), bytes=t.numel()*t.element_size())
            for name, t in sorted(tensors.items())}


def resident_tensor_bytes(model):
    seen, total = set(), 0
    for value in (*model.parameters(), *model.buffers()):
        if id(value) not in seen:
            seen.add(id(value)); total += value.numel()*value.element_size()
    return total


def describe_stack(stack):
    if (not isinstance(stack, CompactQwenStack) or stack.layer_indices != tuple(range(36))
            or not stack.packed or not stack.stripped or stack.geometry_compute_dtype is not None):
        raise ValueError("Export requires a packed, stripped 36-block deployment stack with no worker override")
    blocks = []
    for block in stack.blocks:
        if block.training or block.capture_mlp:
            raise ValueError("Export requires evaluation mode without diagnostic activation capture")
        item = {name: getattr(block, name) for name in BLOCK_FIELDS}
        item["inference_dtype"] = _dtype(block.inference_dtype)
        item["parameters"] = list(block._parameters)
        item["buffers"] = list(block._buffers)
        blocks.append(item)
    return dict(geometry_mode=stack.geometry_mode, pad_to_multiple=stack.pad_to_multiple,
                enabled=stack.enabled, blocks=blocks,
                bridges=[dict(position_source=b.position_source) for b in stack.bridges])


def _metadata_from_state(state, config):
    """Standard-Qwen2 v1 reconstruction contract, serialized before any load.

    Shapes encode retained widths and position tables; the native config encodes
    original RMS epsilon, scale, GQA, activation and rotary semantics. Unsupported
    alternatives are rejected rather than guessed. Real exports additionally
    require the frozen evaluator's schema and exact campaign fingerprint.
    """
    extra = state.get("_extra_state", {})
    if (extra.get("version") != 1 or tuple(extra.get("layer_indices", ())) != tuple(range(36))
            or extra.get("packed") is not True or extra.get("stripped") is not True):
        raise ValueError("Expected complete packed/stripped CompactQwenStack v1 metadata")
    blocks = []
    native_head = getattr(config, "head_dim", config.hidden_size//config.num_attention_heads)
    for i in range(36):
        prefix = f"blocks.{i}."
        values = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
        rank = values["qkv_weight"].shape[1]
        head_dim = values["qkv_weight"].shape[0]//(config.num_attention_heads+2*config.num_key_value_heads)
        projected = "key_rope_columns" in values
        polynomial = "polynomial_anchor_positions" in values
        anchors = len(values["polynomial_anchor_positions"]) if polynomial else 0
        blocks.append(dict(width=config.hidden_size, rank=rank, position_count=len(values["anchor_energy"]),
            position_policy="error", attention_type="full_attention", layer_idx=i, native_head_dim=native_head,
            head_dim=head_dim, nq=config.num_attention_heads, nkv=config.num_key_value_heads,
            group=config.num_attention_heads//config.num_key_value_heads, scale=native_head**-.5,
            attention_dropout=config.attention_dropout, input_eps=config.rms_norm_eps, post_eps=config.rms_norm_eps,
            intermediate_size=values["gate_up_weight"].shape[0]//2, native_intermediate_size=config.intermediate_size,
            packed=True, has_polynomial=polynomial, polynomial_feature_count=1+anchors+anchors*(anchors+1)//2 if polynomial else 0,
            has_head_projection=projected, compact_head=head_dim<native_head,
            inference_dtype=_dtype(values["qkv_weight"].dtype), parameters=list(PARAMETERS),
            buffers=[*BUFFERS, "attention_output_bias"]))
    if tuple(extra.get("ranks", ())) != tuple(b["rank"] for b in blocks):
        raise ValueError("Packed stack rank metadata disagrees with tensor shapes")
    return dict(geometry_mode=extra["geometry_mode"], pad_to_multiple=extra["pad_to_multiple"], enabled=True,
                blocks=blocks, bridges=[dict(position_source=0 if a["position_count"] > 1 else 1)
                                       for a in blocks[:-1]])


def _validate(metadata, tensors, config):
    """Validate the complete runtime shape contract before constructing shells."""
    stack = metadata["stack"]
    blocks, bridges = stack["blocks"], stack["bridges"]
    if len(blocks) != 36 or len(bridges) != 35 or [b["layer_idx"] for b in blocks] != list(range(36)):
        raise ValueError("Artifact must preserve all 36 original cache indices")
    if stack["geometry_mode"] not in ("fp32", "bf16", "fp16") or not isinstance(stack["enabled"], bool):
        raise ValueError("Invalid geometry mode or fusion setting")
    padding = stack["pad_to_multiple"]
    capacity = metadata["position_capacity"]
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("Invalid packed alignment")
    if (not isinstance(capacity, int) or not 2 <= capacity <= min(416, config.max_position_embeddings)
            or not any(b["position_count"] == capacity for b in blocks)):
        raise ValueError("Position capacity must match a bounded calibrated table of 2..416 positions")
    expected = set()

    def tensor(name, shape, *, integer=False):
        expected.add(name)
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError("Missing/wrong tensor shape: " + name)
        if integer and value.dtype != torch.int64 or not integer and not value.is_floating_point():
            raise ValueError("Wrong tensor dtype: " + name)
        if not integer and not torch.isfinite(value).all():
            raise ValueError("Nonfinite tensor: " + name)
        return value

    native_head = getattr(config, "head_dim", config.hidden_size//config.num_attention_heads)
    for i, block in enumerate(blocks):
        prefix = f"stack.blocks.{i}."
        if set(block) != set(BLOCK_FIELDS) | {"inference_dtype", "parameters", "buffers"}:
            raise ValueError("Unknown or missing runtime block attributes")
        rank, head, count, keep = (block[k] for k in ("rank", "head_dim", "position_count", "intermediate_size"))
        if not (0 < rank <= config.hidden_size and 0 < head <= native_head and head % 2 == 0
                and count in (1, capacity) and 0 < keep <= config.intermediate_size):
            raise ValueError("Invalid compact rank, head, position or MLP dimensions")
        fixed = dict(width=config.hidden_size, native_head_dim=native_head, nq=config.num_attention_heads,
            nkv=config.num_key_value_heads, group=config.num_attention_heads//config.num_key_value_heads,
            native_intermediate_size=config.intermediate_size, position_policy="error", attention_type="full_attention",
            packed=True, compact_head=head < native_head, scale=native_head**-.5,
            input_eps=config.rms_norm_eps, post_eps=config.rms_norm_eps, attention_dropout=config.attention_dropout)
        if any(block[k] != v for k, v in fixed.items()) or block["inference_dtype"] not in DTYPES:
            raise ValueError("Block metadata disagrees with standard native Qwen2 semantics")
        parameter_names, buffer_names = set(block["parameters"]), set(block["buffers"])
        if (parameter_names & buffer_names or parameter_names - {"attention_output_bias"} != set(PARAMETERS)
                or buffer_names - {"attention_output_bias"} != set(BUFFERS)
                or "attention_output_bias" not in parameter_names | buffer_names):
            raise ValueError("Invalid parameter/buffer registration contract")
        qkv = (block["nq"]+2*block["nkv"])*head
        tensor(prefix+"qkv_weight", (qkv, rank)); tensor(prefix+"qkv_bias", (qkv,))
        tensor(prefix+"qkv_shift", (count, qkv)); tensor(prefix+"anchor_energy", (count, 1))
        tensor(prefix+"attention_output", (rank, block["nq"]*head)); tensor(prefix+"attention_output_bias", (rank,))
        tensor(prefix+"gate_up_weight", (2*keep, rank)); tensor(prefix+"gate_up_shift", (count, 2*keep))
        if i in (0, 35):
            tensor(prefix+"basis", (block["width"], rank))
        if i == 35:
            tensor(prefix+"anchor", (count, block["width"]))
        if block["has_head_projection"]:
            k = tensor(prefix+"key_rope_columns", (block["nkv"], head), integer=True)
            q = tensor(prefix+"query_rope_columns", (block["nq"], head), integer=True)
            tensor(prefix+"value_basis", (block["nkv"], native_head, head))
            if (not torch.equal(k[:, head//2:], k[:, :head//2]+native_head//2)
                    or (k[:, :head//2] < 0).any() or (k[:, :head//2] >= native_head//2).any()
                    or any(len(row.unique()) != len(row) for row in k)
                    or not torch.equal(q, k.repeat_interleave(block["group"], 0))):
                raise ValueError("Invalid paired RoPE selection or GQA group mapping")
        elif head != native_head:
            raise ValueError("Reduced heads require explicit paired RoPE metadata")
        features = block["polynomial_feature_count"]
        if block["has_polynomial"]:
            positions = tensors.get(prefix+"polynomial_anchor_positions")
            if positions is None or positions.ndim != 1 or not len(positions):
                raise ValueError("Missing polynomial anchors")
            anchors = len(positions)
            if features != 1+anchors+anchors*(anchors+1)//2:
                raise ValueError("Invalid quadratic feature count")
            tensor(prefix+"polynomial_anchor_positions", (anchors,), integer=True)
            tensor(prefix+"polynomial_anchor_mean", (anchors,))
            scale = tensor(prefix+"polynomial_anchor_scale", (anchors,))
            pi = tensor(prefix+"polynomial_pair_i", (anchors*(anchors+1)//2,), integer=True)
            pj = tensor(prefix+"polynomial_pair_j", pi.shape, integer=True)
            pairs = torch.triu_indices(anchors, anchors)
            kept = tensor(prefix+"polynomial_kept", (keep,), integer=True)
            removed = tensor(prefix+"polynomial_removed", (config.intermediate_size-keep,), integer=True)
            if ((positions < 0).any() or (positions >= keep).any() or len(positions.unique()) != anchors
                    or (scale <= 0).any() or not torch.equal(pi.cpu(), pairs[0]) or not torch.equal(pj.cpu(), pairs[1])
                    or not torch.equal(torch.cat((kept, removed)).sort().values.cpu(), torch.arange(config.intermediate_size))):
                raise ValueError("Invalid polynomial indices/scales")
            tensor(prefix+"polynomial_bias", (rank,))
            prefix_indices = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(anchors)))
            fi, fj = torch.cat((prefix_indices, pairs[0])), torch.cat((prefix_indices, pairs[1]))
        else:
            if features != 0 or keep != config.intermediate_size:
                raise ValueError("Unexplained MLP channel removal")
            fi = fj = torch.empty(0, dtype=torch.long)
        for name, wanted in (("i", fi), ("j", fj)):
            actual = tensor(f"stack.feature_{name}_{i}", wanted.shape, integer=True)
            if not torch.equal(actual.cpu(), wanted):
                raise ValueError("Fused polynomial feature indices changed")
        logical = keep+features
        padded = (logical+padding-1)//padding*padding if padding else logical
        output = tensor(prefix+"mlp_output", (rank, padded))
        if torch.count_nonzero(output[:, logical:]):
            raise ValueError("Packed alignment columns must be zero")
        if i < 35:
            following = blocks[i+1]
            expected_source = 0 if count > 1 else 1
            if bridges[i] != {"position_source": expected_source}:
                raise ValueError("Invalid bridge position source")
            tensor(f"stack.bridges.{i}.weight", (following["rank"], rank))
            tensor(f"stack.bridges.{i}.shift", (max(count, following["position_count"]), following["rank"]))
    tensor("embedding.weight", (config.vocab_size, config.hidden_size))
    tensor("norm.weight", (config.hidden_size,))
    if set(tensors) != expected:
        raise ValueError("Unexpected tensor state (including unstripped geometry or unfused offsets)")
    if metadata["tensor_specs"] != _tensor_specs(tensors):
        raise ValueError("Tensor inventory/dtype/byte metadata differs")
    return sum(v.numel()*v.element_size() for v in tensors.values())


def _shell_stack(description, tensors, device):
    """Populate explicit registrations without allocating native matrices."""
    stack = CompactQwenStack.__new__(CompactQwenStack); nn.Module.__init__(stack)
    blocks = []
    for i, item in enumerate(description["blocks"]):
        block = AffineQwenBlock.__new__(AffineQwenBlock); nn.Module.__init__(block)
        for name in BLOCK_FIELDS:
            setattr(block, name, item[name])
        block.inference_dtype = DTYPES[item["inference_dtype"]]
        block.capture_mlp, block.last_mlp_hidden = False, None
        for kind in ("parameters", "buffers"):
            for name in item[kind]:
                tensor = tensors.get(f"stack.blocks.{i}.{name}")
                value = None if tensor is None else tensor.to(device)
                if kind == "parameters":
                    block.register_parameter(name, None if value is None else nn.Parameter(value, requires_grad=False))
                else:
                    block.register_buffer(name, value)
        blocks.append(block)
    stack.blocks = nn.ModuleList(blocks)
    bridges = []
    for i, item in enumerate(description["bridges"]):
        bridge = _Bridge.__new__(_Bridge); nn.Module.__init__(bridge)
        bridge.position_source = item["position_source"]
        for name in ("weight", "shift"):
            bridge.register_buffer(name, tensors[f"stack.bridges.{i}.{name}"].to(device))
        bridges.append(bridge)
    stack.bridges = nn.ModuleList(bridges)
    stack.geometry_mode, stack.geometry_compute_dtype = description["geometry_mode"], None
    stack.pad_to_multiple, stack.enabled = description["pad_to_multiple"], description["enabled"]
    stack.checkpoint_blocks, stack.stripped = False, True
    stack.width, stack.layer_idx, stack.attention_type = blocks[0].width, 0, blocks[0].attention_type
    stack.layer_indices, stack.ranks = tuple(range(36)), tuple(b.rank for b in blocks)
    for i in range(36):
        for name in ("i", "j"):
            stack.register_buffer(f"feature_{name}_{i}", tensors[f"stack.feature_{name}_{i}"].to(device))
    return stack.eval().requires_grad_(False)


def _bound_prefill(module, args, kwargs):
    ids = kwargs.get("input_ids", args[0] if args else None)
    embeddings = kwargs.get("inputs_embeds")
    length = ids.shape[-1] if ids is not None else embeddings.shape[-2] if embeddings is not None else 0
    if length > module.compressed_position_capacity:
        raise ValueError("Prompt exceeds the artifact's calibrated position capacity")
    # Fixed per-position gathers enforce the same bound for CUDA cache positions
    # without adding synchronization or kernels to graph replay.
    for name in ("position_ids", "cache_position"):
        value = kwargs.get(name)
        if value is not None and value.device.type == "cpu" and value.numel():
            if value.min() < 0 or value.max() >= module.compressed_position_capacity:
                raise ValueError("Position exceeds the artifact's calibrated position capacity")


class ArtifactQwenForCausalLM(Qwen2ForCausalLM):
    def generate(self, inputs=None, *args, **kwargs):
        ids = inputs if inputs is not None else kwargs.get("input_ids")
        embeds = kwargs.get("inputs_embeds")
        length = ids.shape[-1] if ids is not None else embeds.shape[-2] if embeds is not None else 0
        generation = kwargs.get("generation_config") or self.generation_config
        new_tokens = kwargs.get("max_new_tokens", generation.max_new_tokens)
        maximum = length+new_tokens if new_tokens is not None else kwargs.get("max_length", generation.max_length)
        if maximum > self.compressed_position_capacity:
            raise ValueError("Requested generation exceeds the artifact's calibrated position capacity")
        return super().generate(inputs, *args, **kwargs)


def _model(config, generation, stack, tensors, capacity, device):
    with torch.device("meta"):
        model = ArtifactQwenForCausalLM(config)
        norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    model.model.layers = nn.ModuleList([stack])
    tied = nn.Parameter(tensors["embedding.weight"].to(device), requires_grad=False)
    model.model.embed_tokens.weight = tied; model.lm_head.weight = tied
    norm.weight = nn.Parameter(tensors["norm.weight"].to(device), requires_grad=False)
    model.model.norm = norm
    model.model.rotary_emb = Qwen2RotaryEmbedding(config, device=device)
    model.generation_config = generation
    model.compressed_position_capacity = model.model.compressed_position_capacity = capacity
    model.model.register_forward_pre_hook(_bound_prefill, with_kwargs=True)
    model.eval().requires_grad_(False)
    if any(v.is_meta for v in (*model.parameters(), *model.buffers())):
        raise ValueError("Standalone reconstruction left uninitialized meta tensors")
    return model


def _embedding(native_source, config):
    path = Path(native_source)
    index = path/"model.safetensors.index.json"
    key = "model.embed_tokens.weight"
    name = json.loads(index.read_text())["weight_map"][key] if index.exists() else "model.safetensors"
    with safe_open(path/name, framework="pt", device="cpu") as stream:
        tensor = stream.get_tensor(key)
    if tensor.shape != (config.vocab_size, config.hidden_size):
        raise ValueError("Native tied embedding dimensions changed")
    return tensor


def _export(destination, state, norm, description, native_source, source_fingerprint, *, embedding=None,
            position_capacity=None, provenance=None):
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError("A source campaign fingerprint is required")
    destination, native_source = Path(destination), Path(native_source)
    if destination.exists():
        raise FileExistsError(destination)
    config = _config(native_source)
    generation = GenerationConfig.from_pretrained(str(native_source), local_files_only=True)
    embedding = _embedding(native_source, config) if embedding is None else embedding.detach()
    tensors = {"stack."+name: value.detach().cpu().contiguous() for name, value in state.items() if isinstance(value, torch.Tensor)}
    tensors.update({"norm."+name: value.detach().cpu().contiguous() for name, value in norm.items()})
    tensors["embedding.weight"] = embedding.cpu().contiguous()
    counts = [b["position_count"] for b in description["blocks"] if b["position_count"] > 1]
    capacity = position_capacity or min(counts or [416, config.max_position_embeddings])
    metadata = dict(schema=SCHEMA, source_fingerprint=source_fingerprint, stack=description,
                    position_capacity=capacity, tensor_specs=_tensor_specs(tensors),
                    architecture="standard tied Qwen2; full attention; native default RoPE; packed quadratic residual stack")
    tensor_size = _validate(metadata, tensors, config)
    destination.mkdir(parents=True)
    copied = []
    for name in LOCAL_FILES:
        if (native_source/name).is_file():
            shutil.copy2(native_source/name, destination/name); copied.append(name)
    for path in sorted((native_source/"chat_templates").glob("*.jinja")):
        target = destination/"chat_templates"/path.name
        target.parent.mkdir(exist_ok=True); shutil.copy2(path, target); copied.append(str(target.relative_to(destination)))
    if "tokenizer_config.json" not in copied:
        raise ValueError("Native source has no local tokenizer configuration")
    # Exercise only the copied files; this also detects missing tokenizer assets.
    AutoTokenizer.from_pretrained(str(destination), local_files_only=True, trust_remote_code=False)
    _write(destination/"structure.json", metadata)
    save_file(tensors, destination/"weights.safetensors", metadata={"format": "pt", "schema": SCHEMA})
    rotary = Qwen2RotaryEmbedding(config, device="cpu")
    rotary_bytes = sum(t.numel()*t.element_size() for t in rotary.buffers())
    files = [*copied, "structure.json", "weights.safetensors"]
    from . import jspace_qwen_affine_block, jspace_qwen_compact_stack, jspace_qwen_affine_fast, jspace_compact_model, jspace_qwen_affine_stack
    sources = [Path(__file__), *(Path(m.__file__) for m in (jspace_qwen_affine_block, jspace_qwen_compact_stack,
               jspace_qwen_affine_fast, jspace_compact_model, jspace_qwen_affine_stack))]
    manifest = dict(schema=SCHEMA, source_fingerprint=source_fingerprint, position_capacity=capacity,
        native_config_max_position_embeddings=config.max_position_embeddings,
        context_scope="Only positions below position_capacity are supported; native long context is not claimed",
        files={name: sha(destination/name) for name in sorted(files)},
        tensor_payload_bytes=tensor_size, regenerated_rotary_buffer_bytes=rotary_bytes,
        resident_tensor_bytes=tensor_size+rotary_bytes, embedding_copies=1,
        payload_file_bytes=sum((destination/name).stat().st_size for name in files),
        runtime_source_sha256={p.name: sha(p) for p in sources},
        versions={"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        provenance=provenance or {})
    _write(destination/"manifest.json", manifest)
    return {**manifest, "total_file_bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())}


def export_artifact(destination, packed_stack, norm, native_source, source_fingerprint, *, embedding=None,
                    position_capacity=None):
    """Export a live packed stack, preserving explicit runtime registrations."""
    config = _config(native_source)
    if not isinstance(norm, Qwen2RMSNorm) or norm.variance_epsilon != config.rms_norm_eps:
        raise ValueError("Export requires the native Qwen2 RMSNorm semantics")
    return _export(destination, packed_stack.state_dict(), norm.state_dict(), describe_stack(packed_stack),
        native_source, source_fingerprint, embedding=embedding, position_capacity=position_capacity,
        provenance={"kind": "live_packed_stack"})


def export_packed_checkpoint(destination, packed_path, native_source, *, source_fingerprint, position_capacity=416):
    """Export evaluator output by reading only native E plus local config/assets."""
    value = torch.load(packed_path, map_location="cpu", weights_only=True, mmap=True)
    if value.get("schema") != PACKED_SCHEMA or value.get("fingerprint") != source_fingerprint:
        raise ValueError("Frozen evaluator export schema/fingerprint mismatch")
    config = _config(native_source)
    description = _metadata_from_state(value["stack"], config)
    # The frozen evaluator loads native Qwen at the deployed parameter dtype.
    # HF checkpoint storage may instead be FP32/FP16, so reproduce that cast.
    embedding = _embedding(native_source, config)
    original_embedding_dtype = _dtype(embedding.dtype)
    deployment_dtype = value["stack"]["blocks.0.qkv_weight"].dtype
    embedding = embedding.to(deployment_dtype)
    return _export(destination, value["stack"], value["norm"], description, native_source, source_fingerprint,
        position_capacity=position_capacity, embedding=embedding,
        provenance={"kind": "frozen_evaluator_packed_checkpoint", "packed_sha256": sha(packed_path),
                    "native_embedding_storage_dtype": original_embedding_dtype,
                    "embedding_deployment_dtype": _dtype(deployment_dtype),
                    "metadata_contract": "standard Qwen2 + CompactQwenStack v1 state; original error position policy"})


def _verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported standalone artifact schema")
    mandatory = {"config.json", "generation_config.json", "tokenizer_config.json", "structure.json", "weights.safetensors"}
    if not mandatory.issubset(manifest["files"]):
        raise ValueError("Manifest must hash every required artifact component")
    for name, expected in manifest["files"].items():
        path = directory/name
        if not path.resolve().is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError("Artifact file/hash mismatch: " + name)
    return manifest


def load_artifact(directory, *, device="cpu", load_tokenizer=True):
    """Load only files inside the artifact and installed, frozen runtime code."""
    directory = Path(directory)
    manifest = _verify(directory)
    metadata = json.loads((directory/"structure.json").read_text())
    if (metadata.get("schema") != SCHEMA or metadata["source_fingerprint"] != manifest["source_fingerprint"]
            or metadata["position_capacity"] != manifest["position_capacity"]):
        raise ValueError("Artifact metadata provenance mismatch")
    config = _config(directory)
    generation = GenerationConfig.from_pretrained(str(directory), local_files_only=True)
    with safe_open(directory/"weights.safetensors", framework="pt", device="cpu") as stream:
        tensors = {key: stream.get_tensor(key) for key in stream.keys()}
    payload_bytes = _validate(metadata, tensors, config)
    if payload_bytes != manifest["tensor_payload_bytes"]:
        raise ValueError("Artifact tensor byte accounting mismatch")
    stack = _shell_stack(metadata["stack"], tensors, device)
    model = _model(config, generation, stack, tensors, metadata["position_capacity"], device)
    if resident_tensor_bytes(model) != manifest["resident_tensor_bytes"]:
        raise ValueError("Standalone resident state size changed")
    model.artifact_manifest = manifest
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False) if load_tokenizer else None
    return model, tokenizer


@contextmanager
def _campaign_protocol(campaign_root):
    """Select recorded protocol only for campaign checks; loading stays separate."""
    from . import jspace_qwen_parallel as campaign
    schema = json.loads((Path(campaign_root)/"plan.json").read_text())["schema"]
    if schema == "jspace.qwen-parallel.v1":
        yield campaign
    elif schema == "jspace.qwen-parallel.v2":
        from .jspace_allblocks_v2 import protocol_context
        with protocol_context():
            yield campaign
    else:
        raise ValueError("Unsupported artifact source campaign schema: " + str(schema))


def _completed_campaign(campaign_root):
    root = Path(campaign_root)
    with _campaign_protocol(root) as campaign:
        plan = campaign.verify(root)  # No dependency on original native weights.
        campaign.checked(root, root/"evaluation")
    completed = json.loads((root/"campaign_results.json").read_text())
    if completed["status"] != "completed" or completed["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Primary timings/evaluation must complete before artifact processing")
    return plan


def export_campaign(campaign_root, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Artifact export is CPU-only; set CUDA_VISIBLE_DEVICES='' ")
    root = Path(campaign_root)
    plan = _completed_campaign(root)
    return export_packed_checkpoint(output, root/"evaluation/packed.pt", plan["model"]["path"],
        source_fingerprint=plan["fingerprint"], position_capacity=416)


def validate_campaign(campaign_root, output):
    """Reserved post-timing parity only; no new quality selection or training."""
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_evaluate import evaluation_cases, runtime_cases, score_cases, tensor_bytes
    from .jspace_qwen_affine import graph_decoder

    root, output = Path(campaign_root), Path(output)
    if (output/"validation.json").exists() or (output/"validation_logits.json").exists():
        raise FileExistsError("Inspect existing standalone validation before another attempt")
    lease = wait_ready([0])
    guard = lambda: wait_ready([0])
    plan = _completed_campaign(root)
    torch.set_num_threads(4); torch.manual_seed(plan["config"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_artifact(output, device="cuda:0")
    if model.artifact_manifest["source_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Artifact belongs to a different primary campaign")
    tasks = json.loads((root/"tasks.json").read_text())
    records = score_cases(model, evaluation_cases(tasks), plan["label_ids"], guard=guard,
        event=lambda **kw: print(json.dumps({"kind": "standalone_score", **kw}), flush=True))
    _write(output/"validation_logits.json", records)
    reference = json.loads((root/"evaluation/compact_fresh.json").read_text())
    decoder = graph_decoder(model); capture_seconds = decoder.capture()
    outputs, static_exact = [], []
    with torch.inference_mode():
        for case in runtime_cases(tasks):
            guard()
            ids = torch.tensor([case["input_ids"]], device="cuda:0")
            eager, graph = decoder.generate(ids, False), decoder.generate(ids, True)
            static_exact.append(torch.equal(eager, graph)); outputs.append(graph.cpu().tolist())
    expected_graph = json.loads((root/"evaluation/compact_timing.json").read_text())["generated_tokens"]["cuda_graph"]
    _write(output/"validation_graph.json", {"outputs": outputs, "graph_static_exact": static_exact})
    guard(); peak = torch.cuda.max_memory_reserved()
    if peak > sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("Artifact parity validation exceeded its GPU reservation")
    result = dict(schema=SCHEMA, source_fingerprint=plan["fingerprint"], status="completed",
        exact_512_logit_records=records == reference, exact_eight_graph32_outputs=outputs == expected_graph,
        graph_static_exact=all(static_exact), resident_tensor_bytes=tensor_bytes(model),
        graph_capture_seconds=capture_seconds, peak_reserved_bytes=peak, lease_id=lease["id"],
        passed=records == reference and outputs == expected_graph and all(static_exact),
        inputs={str(p): sha(p) for p in (output/"manifest.json", root/"evaluation/compact_fresh.json",
               root/"evaluation/compact_timing.json", root/"tasks.json")},
        artifacts={p.name: sha(p) for p in (output/"validation_logits.json", output/"validation_graph.json")},
        scope="Standalone artifact parity after primary timings; no checkpoint selection or quality reinterpretation")
    _write(output/"validation.json", result)
    del decoder, model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "validate"))
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (export_campaign if args.stage == "export" else validate_campaign)(args.campaign, args.output)
    print(json.dumps(result, allow_nan=False), flush=True)
    if args.stage == "validate" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
