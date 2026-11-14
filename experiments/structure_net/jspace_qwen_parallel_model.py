"""Selective local Qwen loading and warm-start assembly for grouped recovery.

Workers load individual native decoder blocks and one shared readout, not a
duplicate full language model per block. Only build_teacher loads a full model.
All checkpoint access is local; GPU callers must already hold their reservation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from safetensors import safe_open
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from .jspace_lens import load_small_qwen
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack


def group_fit_path(root, layer):
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < 36:
        raise ValueError("Qwen3B layer must be an integer in [0, 35]")
    return Path(root) / "groups" / f"group_{(35-layer)//6:02d}" / f"block_{layer:02d}"


def recovery_checkpoint_path(root, previous_group):
    if isinstance(previous_group, int) and not isinstance(previous_group, bool):
        if not 0 <= previous_group < 6:
            raise ValueError("Recovery group must lie in [0, 5]")
        return Path(root) / "groups" / f"group_{previous_group:02d}" / "recovered_unpacked.pt"
    if previous_group is None:
        raise ValueError("A previous recovery group is required")
    path = Path(previous_group)
    if not path.is_absolute():
        path = Path(root) / path
    return path if path.suffix == ".pt" else path / "recovered_unpacked.pt"


def _load(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)


def load_recovered_checkpoint(root, previous_group):
    state = _load(recovery_checkpoint_path(root, previous_group))
    if not isinstance(state, dict) or not isinstance(state.get("stack"), dict) or not isinstance(state.get("norm"), dict):
        raise ValueError("Recovery checkpoint requires stack and norm state dictionaries")
    extra = state["stack"].get("_extra_state")
    if not isinstance(extra, dict) or extra.get("packed") is not False or extra.get("stripped") is not False:
        raise ValueError("Recovery requires an unpacked, unstripped CompactQwenStack checkpoint")
    return state


def state_dict_cpu(module):
    """Clone tensor state and preserve/deepcopy non-tensor get_extra_state data."""
    return {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for name, value in module.state_dict().items()}


def _config(model_path, config=None):
    value = deepcopy(config) if config is not None else AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    if value.model_type != "qwen2":
        raise ValueError("This loader supports the pinned Qwen2/Qwen2.5 decoder architecture")
    value._attn_implementation = "eager"
    return value


def _key_map(model_path):
    model_path = Path(model_path)
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    checkpoint = model_path / "model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError("Expected a local model.safetensors or its shard index")
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        return {name: checkpoint.name for name in handle.keys()}


def _selected_tensors(model_path, names, device, dtype):
    index, groups = _key_map(model_path), {}
    for name in names:
        if name not in index:
            raise KeyError("Missing local checkpoint tensor: " + name)
        groups.setdefault(index[name], []).append(name)
    result = {}
    for filename, keys in groups.items():
        with safe_open(Path(model_path) / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                result[key] = handle.get_tensor(key).to(device=device, dtype=dtype)
    return result


def load_native_block(model_path, layer, device="cpu", dtype=torch.bfloat16, *, config=None):
    """Instantiate only one decoder block on meta, then assign selected shards."""
    config = _config(model_path, config)
    if not 0 <= layer < config.num_hidden_layers:
        raise ValueError("Native layer index outside model config")
    with torch.device("meta"):
        block = Qwen2DecoderLayer(config, layer_idx=layer)
    prefix = f"model.layers.{layer}."
    names = [prefix + key for key in block.state_dict()]
    values = _selected_tensors(model_path, names, device, dtype)
    block.load_state_dict({key[len(prefix):]: value for key, value in values.items()}, strict=True, assign=True)
    return block.eval().requires_grad_(False)


def load_readout_norm(model_path, device="cpu", dtype=torch.bfloat16, *, config=None):
    """Return (fixed readout Tensor E, native RMSNorm, native rotary module)."""
    config = _config(model_path, config)
    key = "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    values = _selected_tensors(model_path, [key, "model.norm.weight"], device, dtype)
    norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=dtype)
    norm.load_state_dict({"weight": values["model.norm.weight"]}, strict=True)
    rotary = Qwen2RotaryEmbedding(config=config, device=device).eval()
    return values[key].requires_grad_(False), norm.eval().requires_grad_(False), rotary


def _settings(root, model_path, config, rank):
    plan_path = Path(root) / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    model_path = Path(model_path or plan.get("model", {}).get("path", ""))
    if not (model_path / "config.json").exists() and config is None:
        raise ValueError("A local model path/config is required")
    config = _config(model_path, config)
    rank = rank if rank is not None else plan.get("config", {}).get("rank", 1024)
    if not isinstance(rank, int) or not 0 < rank <= config.hidden_size:
        raise ValueError("Campaign rank must lie within the native residual width")
    return model_path, config, rank


def _fitted_basis(fit, width, rank, device):
    basis = fit["basis"]
    if basis.ndim != 2 or basis.shape[0] != width or basis.shape[1] < rank:
        raise ValueError("Fitted basis does not contain the requested native width/rank")
    return basis[:, :rank].to(device=device, dtype=torch.float32)


def build_stack(root, cut, trainable, previous_group=None, *, device="cpu", dtype=torch.bfloat16,
                parameter_dtype=torch.bfloat16, model_path=None, config=None, rank=None,
                geometry_compute_dtype=None, checkpoint_blocks=False):
    """Compile every fitted block in a suffix and preserve accepted old weights.

    Defaults: BF16 trainable parameters, fixed BF16 geometry, 128-column packed
    alignment. Source loading dtype can be FP16 on a Turing worker; use explicit
    geometry_compute_dtype=FP16 and FP16 autocast there. Position tables remain
    FP32 until deployment packing. The previous norm is available separately via
    load_recovered_checkpoint; no new readout adapter is introduced.
    """
    model_path, config, rank = _settings(root, model_path, config, rank)
    if not isinstance(cut, int) or not 0 <= cut < config.num_hidden_layers <= 36:
        raise ValueError("Cut must define a nonempty native suffix within 36 layers")
    previous = load_recovered_checkpoint(root, previous_group) if previous_group is not None else None
    old_layers = list(previous["stack"]["_extra_state"]["layer_indices"]) if previous else []
    if old_layers and (old_layers != list(range(old_layers[0], config.num_hidden_layers)) or cut > old_layers[0]):
        raise ValueError("Previous accepted model must be a contained contiguous suffix")
    old_positions = {layer: i for i, layer in enumerate(old_layers)}
    blocks = []
    for layer in range(cut, config.num_hidden_layers):
        directory = group_fit_path(root, layer)
        fit, relation = _load(directory / "fit.pt"), _load(directory / "relation.pt")
        native = load_native_block(model_path, layer, device=device, dtype=dtype, config=config)
        u = _fitted_basis(fit, config.hidden_size, rank, device)
        block = AffineQwenBlock(native, u.clone(), fit["means"].to(device=device, dtype=torch.float32), trainable=trainable,
            parameter_dtype=parameter_dtype, head_projection={"rope_pairs": fit["rope_pairs"].to(device),
                                                             "value_basis": fit["value_basis"].to(device=device, dtype=torch.float32)})
        block.compile_polynomial(relation, relation["removed"])
        if layer in old_positions:
            prefix = f"blocks.{old_positions[layer]}."
            saved = {name[len(prefix):]: value for name, value in previous["stack"].items() if name.startswith(prefix)}
            block.load_state_dict(saved, strict=True)
            # Old state stores BF16 deployed geometry. Reconstruct the original
            # fitted FP32 U before deriving new fixed inter-layer bridges.
            block.basis = u
        blocks.append(block)
        del native, fit, relation
    stack = CompactQwenStack(blocks, geometry_mode="bf16", geometry_compute_dtype=geometry_compute_dtype,
                             pad_to_multiple=128, checkpoint_blocks=checkpoint_blocks)
    if not trainable:
        stack.pack(dtype=dtype, buffer_dtype=dtype, strip_unused=True).eval().requires_grad_(False)
    return stack


def build_teacher(root, previous_group=None, *, device="cuda:0", dtype=torch.bfloat16,
                  model_path=None, config=None, rank=None, model_loader=None, geometry_compute_dtype=None):
    """Load one full local teacher, replace its accepted suffix, preserve tied E."""
    model_path, config, rank = _settings(root, model_path, config, rank)
    model, tokenizer = (model_loader or load_small_qwen)(str(model_path), device=device, dtype=dtype)
    if previous_group is not None:
        recovered = load_recovered_checkpoint(root, previous_group)
        cut = recovered["stack"]["_extra_state"]["layer_indices"][0]
        # Release the old native suffix before allocating its replacement.
        model.model.layers = nn.ModuleList(list(model.model.layers[:cut]))
        stack = build_stack(root, cut, False, previous_group, device=device, dtype=dtype,
            model_path=model_path, config=config, rank=rank, geometry_compute_dtype=geometry_compute_dtype)
        model.model.layers.append(stack)
        model.model.norm.load_state_dict(recovered["norm"], strict=True)
    model.eval().requires_grad_(False)
    if config.tie_word_embeddings and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("Teacher assembly lost the native embedding/readout tie")
    return model, tokenizer


def systems_stack(model_path, source_fit, source_relation, blocks=36, rank=1024, *, device="cpu",
                  dtype=torch.bfloat16, parameter_dtype=torch.bfloat16, config=None,
                  geometry_compute_dtype=None, checkpoint_blocks=True, stabilized=False):
    """Shape-equivalent repeated-final-block stack for memory/backward pilots ONLY.

    Compile one real native final block, then copy that small compiled module.
    No full model is allocated. Optional residual scaling 1/blocks reduces the
    risk of exploding repeated polynomial dynamics without changing dimensions;
    either initialization is explicitly unsuitable for task-quality conclusions.
    """
    config = _config(model_path, config)
    if not isinstance(blocks, int) or not 1 <= blocks <= config.num_hidden_layers:
        raise ValueError("Systems block count must fit the native cache configuration")
    if not isinstance(rank, int) or not 0 < rank <= config.hidden_size:
        raise ValueError("Systems rank must lie within the native residual width")
    fit = _load(source_fit) if isinstance(source_fit, (str, Path)) else source_fit
    relation = _load(source_relation) if isinstance(source_relation, (str, Path)) else source_relation
    native = load_native_block(model_path, config.num_hidden_layers - 1, device=device, dtype=dtype, config=config)
    prototype = AffineQwenBlock(native, _fitted_basis(fit, config.hidden_size, rank, device),
        fit["means"].to(device=device, dtype=torch.float32), parameter_dtype=parameter_dtype,
        head_projection={"rope_pairs": fit["rope_pairs"].to(device),
                         "value_basis": fit["value_basis"].to(device=device, dtype=torch.float32)})
    prototype.compile_polynomial(relation, relation["removed"])
    scale = 1. / blocks if stabilized else 1.
    if stabilized:
        with torch.no_grad():
            for name in ("attention_output", "mlp_output", "polynomial_output", "polynomial_bias"):
                parameter = getattr(prototype, name)
                if parameter is not None:
                    parameter.mul_(scale)
    compiled = [prototype] + [deepcopy(prototype) for _ in range(blocks - 1)]
    for i, block in enumerate(compiled):
        block.layer_idx = i
    stack = CompactQwenStack(compiled, geometry_mode="bf16", geometry_compute_dtype=geometry_compute_dtype,
                             pad_to_multiple=128, checkpoint_blocks=checkpoint_blocks)
    stack.systems_only = True
    stack.systems_metadata = {"scope": "systems-only repeated final block; no task-quality evidence",
        "repeated_native_layer": config.num_hidden_layers - 1, "blocks": blocks, "rank": rank,
        "stabilized": stabilized, "residual_scale": scale}
    return stack
