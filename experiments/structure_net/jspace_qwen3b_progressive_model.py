"""Explicit-fit assembly for the existing Qwen2.5-3B progressive recipe.

This is the original affine/head/polynomial compiler, not the Qwen3-8B port.
The caller authenticates fit/checkpoint files and holds any GPU reservation.
Accepted blocks and interior bridges are retained exactly when adding one
preceding block. Only its entrance bridge is newly calculated from FP32 fits.
"""
from collections.abc import Mapping
from pathlib import Path
import json
import math

import torch

from . import jspace_qwen_parallel_model as native
from .jspace_incremental_suffix import preserve_inheritance, state_equal
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack
from .jspace_qwen_parallel_train import cpu_module_clone


def load_previous(previous):
    """Read an unpacked selected checkpoint without changing its tensors."""
    value = native._load(previous) if isinstance(previous, (str, Path)) else previous
    if not isinstance(value, Mapping) or not isinstance(value.get("stack"), Mapping) or not isinstance(value.get("norm"), Mapping):
        raise ValueError("Previous checkpoint requires stack and norm states")
    extra = value["stack"].get("_extra_state", {})
    if (extra.get("packed") is not False or extra.get("stripped") is not False or
            extra.get("geometry_mode") != "bf16" or extra.get("pad_to_multiple") != 128):
        raise ValueError("Previous stack must use unpacked BF16 geometry and 128-column alignment")
    layers = extra.get("layer_indices")
    if not isinstance(layers, (tuple, list)) or not layers or any(type(i) is not int for i in layers):
        raise ValueError("Previous stack needs explicit contiguous layer indices")
    if list(layers) != list(range(layers[0], layers[-1] + 1)):
        raise ValueError("Previous stack is not contiguous")
    return value


def _mapping(fit_paths, cut, count):
    if not isinstance(fit_paths, Mapping) or any(type(k) is not int for k in fit_paths):
        raise ValueError("Fit paths must map integer native layer indices to directories")
    missing = set(range(cut, count)) - set(fit_paths)
    if missing:
        raise ValueError(f"Missing explicit fits for layers {sorted(missing)}")
    return {layer: Path(fit_paths[layer]) for layer in range(cut, count)}


def build_stack(model_path, fit_paths, cut, previous=None, *, device="cpu", trainable=True,
                rank=1024, dtype=torch.bfloat16, checkpoint_blocks=True, config=None,
                native_loader=None, readout_loader=None):
    """Return ``(unpacked_stack, norm, tied_E, rotary, inheritance)``.

    ``previous`` accepts a selected.pt path or its {stack,norm} mapping. The
    parent may have the same cut (strict reconstruction), or cut+1 (one-block
    expansion). ``trainable=False`` freezes parameters but retains input
    autograd; packing is separate and explicit via :func:`packed_clone`.
    Fit directories each contain fit.pt and relation.pt, with no group-number
    assumptions. Native/readout loaders are injectable for real tiny fixtures.
    """
    cfg = native._config(model_path, config)
    if not cfg.tie_word_embeddings:
        raise ValueError("The pinned 3B recipe requires one tied embedding/readout")
    if type(cut) is not int or not 0 <= cut < cfg.num_hidden_layers:
        raise ValueError("Cut must define a nonempty native suffix")
    if type(rank) is not int or not 0 < rank <= cfg.hidden_size or dtype != torch.bfloat16:
        raise ValueError("This progressive recipe requires valid rank and original BF16 parameters")
    paths = _mapping(fit_paths, cut, cfg.num_hidden_layers)
    old = load_previous(previous) if previous is not None else None
    old_layers = list(old["stack"]["_extra_state"]["layer_indices"]) if old else []
    if old_layers and (old_layers[-1] != cfg.num_hidden_layers - 1 or old_layers[0] not in (cut, cut + 1)):
        raise ValueError("Previous model must be the same suffix or exactly one block shorter")
    positions = {layer: i for i, layer in enumerate(old_layers)}
    load_block = native_loader or native.load_native_block
    blocks = []
    for layer, directory in paths.items():
        fit, relation = native._load(directory / "fit.pt"), native._load(directory / "relation.pt")
        original = load_block(model_path, layer, device=device, dtype=dtype, config=cfg)
        basis = native._fitted_basis(fit, cfg.hidden_size, rank, device)
        block = AffineQwenBlock(original, basis.clone(), fit["means"].to(device=device, dtype=torch.float32),
            trainable=trainable, parameter_dtype=dtype,
            head_projection={"rope_pairs": fit["rope_pairs"].to(device),
                             "value_basis": fit["value_basis"].to(device=device, dtype=torch.float32)})
        block.compile_polynomial(relation, relation["removed"])
        if layer in positions:
            prefix = f"blocks.{positions[layer]}."
            saved = {key[len(prefix):]: value for key, value in old["stack"].items() if key.startswith(prefix)}
            # A wrong fit must not silently change an inherited basis/bridge.
            if not torch.equal(saved.get("basis", torch.empty(0)).cpu(), basis.bfloat16().cpu()):
                raise ValueError(f"Inherited fitted basis mismatch at layer {layer}")
            block.load_state_dict(saved, strict=True)
            block.basis = basis  # Construct new bridges before BF16 storage rounding.
        blocks.append(block)
        del original, fit, relation
    stack = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=128,
                             checkpoint_blocks=checkpoint_blocks)
    readout, norm, rotary = (readout_loader or native.load_readout_norm)(
        model_path, device=device, dtype=dtype, config=cfg)
    if old and old_layers[0] == cut:
        stack.load_state_dict(old["stack"], strict=True)
        norm.load_state_dict(old["norm"], strict=True)
        if not state_equal(stack.state_dict(), old["stack"]) or not state_equal(norm.state_dict(), old["norm"]):
            raise ValueError("Same-cut strict reconstruction changed parent state")
        inheritance = dict(passed=True, unchanged_layers=old_layers,
            interior_bridges_preserved=len(stack.bridges), new_layer=None, new_bridge=None, norm_exact=True)
    elif old:
        inheritance = preserve_inheritance(stack, norm, old)
    else:
        inheritance = dict(passed=True, unchanged_layers=[], interior_bridges_preserved=0,
            new_layer=None, new_bridge=None, norm_exact=True, native_initialization=True)
    stack.eval().requires_grad_(trainable)
    norm.eval().requires_grad_(trainable)
    readout.requires_grad_(False)
    rotary.eval().requires_grad_(False)
    return stack, norm, readout, rotary, inheritance


def packed_clone(stack, *, device=None):
    """Copy directly via CPU, preserving live state; return frozen packed stack."""
    destination = device if device is not None else next(stack.parameters()).device
    result = cpu_module_clone(stack)
    result.checkpoint_blocks = False
    return result.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
                       strip_unused=True).to(destination).eval().requires_grad_(False)


def tensor_bytes(*modules):
    """Count allocated tensor storages once, including fixed geometry buffers."""
    storages = {}
    for module in modules:
        values = [module] if isinstance(module, torch.Tensor) else [*module.parameters(), *module.buffers()]
        for value in values:
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def inventory(model_path, stack, norm, rotary, *, cut=None, config=None):
    """Actual packed suffix plus BF16 native prefix and tied E, including cut0.

    Native tensor sizes come from safetensors headers, without loading native
    weights. This is storage accounting, not activation/KV memory or speed.
    """
    cfg = native._config(model_path, config)
    cut = stack.layer_indices[0] if cut is None else cut
    if not cfg.tie_word_embeddings or type(cut) is not int or list(stack.layer_indices) != list(range(cut, cfg.num_hidden_layers)):
        raise ValueError("Inventory requires the actual tied-E contiguous suffix")
    suffix = tensor_bytes(packed_clone(stack, device="cpu"), cpu_module_clone(norm))
    rotation = tensor_bytes(rotary)
    prefix = embedding = native_total = norm_bytes = 0
    seen, layers, prefix_layers = set(), set(), set()
    for path in sorted(Path(model_path).glob("*.safetensors")):
        with path.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(length))
        for name, value in header.items():
            if name == "__metadata__" or name in seen or name == "lm_head.weight":
                continue
            seen.add(name)
            if value["dtype"] != "BF16":
                raise ValueError("Pinned native tensor storage must be BF16")
            size = math.prod(value["shape"]) * 2
            native_total += size
            if name == "model.embed_tokens.weight":
                embedding += size
            elif name == "model.norm.weight":
                norm_bytes += size
            elif name.startswith("model.layers."):
                layer = int(name.split(".")[2]); layers.add(layer)
                if layer < cut:
                    prefix += size; prefix_layers.add(layer)
            else:
                raise ValueError("Unaccounted native model tensor: " + name)
    if (not embedding or not norm_bytes or layers != set(range(cfg.num_hidden_layers)) or
            prefix_layers != set(range(cut))):
        raise ValueError("Incomplete native prefix/embedding/norm inventory")
    full = prefix + embedding + suffix + rotation
    baseline = native_total + rotation
    return dict(cut=cut, compressed_layers=len(stack.blocks), packed_suffix_and_norm_bytes=suffix,
        frozen_native_prefix_bytes=prefix, tied_embedding_readout_bytes=embedding, rotary_bytes=rotation,
        full_model_tensor_bytes=full, native_full_tensor_bytes=baseline,
        native_treated_suffix_and_norm_bytes=native_total-prefix-embedding,
        full_model_reduction_fraction=1-full/baseline,
        accounting="BF16 native prefix and tied E once; actual packed BF16 suffix/norm with all fixed geometry; no KV/activations")
