"""Packed snapshot inference and independent structural export/reload.

No training producer is changed. All quality callbacks run the packed stack;
reload parity compares the same packed executor, not an unpacked approximation.
Native prefix offload is explicit and reports its transfer/time cost.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from . import jspace_3b_response_model as source
from .jspace_3b_response_train import ResponseModel
from . import jspace_allblocks_artifact as artifact
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_compact_stack import CompactQwenStack, _Bridge

SCHEMA = "jspace.packed-response-snapshot.v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path, value):
    with Path(path).open("x") as stream:
        stream.write(_json(value)+"\n")


class PackedResponseModel(ResponseModel):
    """Response-model interface over a frozen packed/stripped inference stack."""
    def __init__(self, stack, norm, embedding, rotary, *, config, max_length=416, guard=None):
        nn.Module.__init__(self)
        if (not isinstance(stack, CompactQwenStack) or not stack.packed or not stack.stripped
                or not stack.layer_indices
                or stack.layer_indices != tuple(range(stack.layer_indices[0], config.num_hidden_layers))):
            raise ValueError("A complete packed/stripped contiguous suffix is required")
        if (embedding.shape != (config.vocab_size, config.hidden_size)
                or embedding.dtype != torch.bfloat16 or not config.tie_word_embeddings
                or stack.width != config.hidden_size):
            raise ValueError("Original tied BF16 embedding/readout and native width required")
        if any(b.position_policy != "error" or b.position_count > 1 and max_length > b.position_count
               for b in stack.blocks):
            raise ValueError("Packed response exceeds the fitted positional envelope")
        self.stack, self.norm, self.rotary = stack, norm, rotary
        self.register_buffer("embedding", embedding.detach())
        self.max_length, self.expected_layers = max_length, config.num_hidden_layers
        self.guard, self.native_prefix = guard, None
        self.cut, self.config = stack.layer_indices[0], deepcopy(config)
        self.stack.checkpoint_blocks = False
        self.eval().requires_grad_(False)
        device = self._guard()
        # BF16 Triton pointwise kernels require Ampere. Turing keeps the packed
        # matrices and executes their Torch fallback at the same BF16 precision.
        self.stack.enabled = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        self.inference_backend = "packed_bf16_triton" if self.stack.enabled else "packed_bf16_torch"


def build_packed_student(arm, source_plan, checkpoint, device="cpu", guard=None):
    device = source.execution._guard(device, guard)
    p = json.loads(Path(source_plan).read_text()) if isinstance(source_plan, (str, Path)) else source_plan
    stack, norm, embedding, old_rotary, proof = source.build_student(arm, p, checkpoint, "cpu", guard)
    config = source.builder.native._config(p["model"]["path"])
    packed = source.builder.packed_clone(stack, device="cpu")
    del stack, old_rotary
    gc.collect()
    rotary = source.partial.training_rotary(config, device, guard=guard)
    model = PackedResponseModel(packed.to(device), norm.to(device), embedding.to(device), rotary,
                                config=config, max_length=416, guard=guard)
    header = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    proof = dict(proof, source_checkpoint_schema=header.get("schema", "legacy stack/norm"),
        source_checkpoint_identities=header.get("identities"), packed=True, stripped=True,
        inference_backend=model.inference_backend,
        scope="Saved weights packed before quality testing; no pack-versus-unpacked quality assertion")
    del header
    return model, proof


def _describe(model):
    if not model.stack.packed or not model.stack.stripped or model.native_prefix is not None:
        raise ValueError("Export only a frozen packed suffix with temporary native prefix detached")
    blocks = []
    for block in model.stack.blocks:
        if block.training or block.capture_mlp:
            raise ValueError("Packed export requires evaluation without diagnostic activation capture")
        blocks.append(dict(**{key:getattr(block, key) for key in artifact.BLOCK_FIELDS},
            inference_dtype=artifact._dtype(block.inference_dtype), parameters=list(block._parameters),
            buffers=list(block._buffers)))
    return dict(schema=SCHEMA, config=model.config.to_dict(), max_length=model.max_length,
        cut=model.cut, expected_layers=model.expected_layers,
        norm_epsilon=model.norm.variance_epsilon,
        rotary_attention_scaling=float(model.rotary.attention_scaling),
        rotary_max_seq_len_cached=int(model.rotary.max_seq_len_cached),
        rotary_original_max_seq_len=int(model.rotary.original_max_seq_len),
        stack=dict(blocks=blocks, bridges=[dict(position_source=b.position_source) for b in model.stack.bridges],
            geometry_mode=model.stack.geometry_mode, pad_to_multiple=model.stack.pad_to_multiple,
            extra_state=model.stack.get_extra_state()),
        execution_policy="BF16 packed; sm80+ Triton otherwise Torch; last-row-only generation readout",
        prefix_scope="Native prefix weights are external for cut>0; full native prefix recomputation is required")


def packed_state_digest(model):
    description = _describe(model)
    # Hash registered names, as state_dict/export do. Equal logical entries can
    # alias one Tensor before export and have independent storage after reload
    # (the non-polynomial feature_i/feature_j empty buffers are one example).
    # Storage identity is not part of the frozen inference-state contract.
    values = [(name, value) for name, value in model.named_parameters(remove_duplicate=False)]
    values.extend((name, value) for name, value in model.named_buffers(remove_duplicate=False))
    values.append(("rotary.original_inv_freq", model.rotary.original_inv_freq))
    tensor = source.offload.fingerprint(sorted(values))
    return dict(packed_state_sha256=hashlib.sha256((_json(description)+tensor["sha256"]).encode()).hexdigest(),
                tensor_sha256=tensor["sha256"], tensor_bytes=tensor["tensor_bytes"])


def export_packed_snapshot(model, directory, *, identities):
    directory = Path(directory)
    if directory.exists(): raise FileExistsError(directory)
    model._guard()
    description = _describe(model)
    state = packed_state_digest(model)
    tensors = {"stack."+name:value.detach().cpu().contiguous().clone()
               for name, value in model.stack.state_dict().items() if isinstance(value, torch.Tensor)}
    tensors["norm.weight"] = model.norm.weight.detach().cpu().contiguous().clone()
    tensors["embedding"] = model.embedding.detach().cpu().contiguous().clone()
    tensors.update({"rotary."+name:value.detach().cpu().contiguous().clone()
                    for name, value in model.rotary.named_buffers()})
    tensors["rotary.original_inv_freq"] = model.rotary.original_inv_freq.detach().cpu().contiguous().clone()
    description["tensor_specs"] = artifact._tensor_specs(tensors)
    directory.mkdir(parents=True)
    _write(directory/"structure.json", description)
    save_file(tensors, directory/"weights.safetensors", metadata={"format":"pt", "schema":SCHEMA})
    del tensors
    files = {name:artifact.sha(directory/name) for name in ("structure.json", "weights.safetensors")}
    receipt = dict(schema=SCHEMA, packed=True, cut=model.cut, identities=dict(identities), files=files,
        serialized_bytes=sum((directory/name).stat().st_size for name in files),
        embedding_copies=1, source_inference_backend=model.inference_backend, **state)
    _write(directory/"manifest.json", receipt)
    return dict(receipt, directory=str(directory.resolve()), manifest_sha256=artifact.sha(directory/"manifest.json"))


def _restore_stack(description, tensors, device):
    blocks = []
    for index, item in enumerate(description["blocks"]):
        if (set(item) != set(artifact.BLOCK_FIELDS)|{"inference_dtype", "parameters", "buffers"}
                or item["packed"] is not True
                or set(item["parameters"]) - {"attention_output_bias"} != set(artifact.PARAMETERS)
                or set(item["buffers"]) - {"attention_output_bias"} != set(artifact.BUFFERS)):
            raise ValueError("Unsupported packed block registration/attribute contract")
        block = AffineQwenBlock.__new__(AffineQwenBlock); nn.Module.__init__(block)
        for key in artifact.BLOCK_FIELDS: setattr(block, key, item[key])
        block.inference_dtype = artifact.DTYPES[item["inference_dtype"]]
        block.capture_mlp, block.last_mlp_hidden = False, None
        for kind in ("parameters", "buffers"):
            for name in item[kind]:
                tensor = tensors.get(f"stack.blocks.{index}.{name}")
                value = None if tensor is None else tensor.to(device)
                if kind == "parameters":
                    block.register_parameter(name, None if value is None else nn.Parameter(value, requires_grad=False))
                else: block.register_buffer(name, value)
        blocks.append(block)
    stack = CompactQwenStack.__new__(CompactQwenStack); nn.Module.__init__(stack)
    stack.blocks = nn.ModuleList(blocks)
    bridges = []
    for index, item in enumerate(description["bridges"]):
        bridge = _Bridge.__new__(_Bridge); nn.Module.__init__(bridge)
        bridge.position_source = item["position_source"]
        for name in ("weight", "shift"):
            bridge.register_buffer(name, tensors[f"stack.bridges.{index}.{name}"].to(device))
        bridges.append(bridge)
    stack.bridges = nn.ModuleList(bridges)
    stack.geometry_mode, stack.geometry_compute_dtype = description["geometry_mode"], None
    stack.pad_to_multiple, stack.enabled = description["pad_to_multiple"], False
    stack.checkpoint_blocks, stack.stripped = False, True
    stack.width, stack.layer_idx, stack.attention_type = blocks[0].width, blocks[0].layer_idx, blocks[0].attention_type
    stack.layer_indices, stack.ranks = tuple(b.layer_idx for b in blocks), tuple(b.rank for b in blocks)
    for i in range(len(blocks)):
        for suffix in ("i", "j"):
            stack.register_buffer(f"feature_{suffix}_{i}", tensors[f"stack.feature_{suffix}_{i}"].to(device))
    if _json(stack.get_extra_state()) != _json(description["extra_state"]):
        raise ValueError("Packed structural metadata differs from reconstructed stack")
    state = {name[len("stack."):]:value for name, value in tensors.items() if name.startswith("stack.")}
    state["_extra_state"] = stack.get_extra_state()
    stack.load_state_dict(state, strict=True)
    return stack.eval().requires_grad_(False)


def load_packed_snapshot(directory, device="cpu", guard=None):
    device = source.execution._guard(device, guard)
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("packed") is not True:
        raise ValueError("Expected a completed packed snapshot manifest")
    if set(manifest["files"]) != {"structure.json", "weights.safetensors"}:
        raise ValueError("Unexpected packed snapshot files")
    for name, digest in manifest["files"].items():
        if artifact.sha(directory/name) != digest: raise ValueError("Packed snapshot changed: "+name)
    description = json.loads((directory/"structure.json").read_text())
    if description.get("schema") != SCHEMA: raise ValueError("Packed structural schema differs")
    with safe_open(directory/"weights.safetensors", framework="pt", device="cpu") as stream:
        tensors = {name:stream.get_tensor(name) for name in stream.keys()}
    if artifact._tensor_specs(tensors) != description["tensor_specs"]:
        raise ValueError("Saved packed tensor shape/dtype/byte inventory differs")
    config = Qwen2Config.from_dict(description["config"])
    stack = _restore_stack(description["stack"], tensors, device)
    norm = Qwen2RMSNorm(config.hidden_size, eps=description["norm_epsilon"])
    norm.weight = nn.Parameter(tensors["norm.weight"].to(device), requires_grad=False)
    rotary = source.partial.training_rotary(config, "cpu")
    expected_rotary = {"rotary."+name for name, _ in rotary.named_buffers()} | {"rotary.original_inv_freq"}
    if {name for name in tensors if name.startswith("rotary.")} != expected_rotary:
        raise ValueError("Actual saved rotary buffer registration differs")
    for name, _ in list(rotary.named_buffers()): setattr(rotary, name, tensors["rotary."+name].to(device))
    rotary.original_inv_freq = tensors["rotary.original_inv_freq"].to(device)
    rotary.attention_scaling = description["rotary_attention_scaling"]
    rotary.max_seq_len_cached = description["rotary_max_seq_len_cached"]
    rotary.original_max_seq_len = description["rotary_original_max_seq_len"]
    model = PackedResponseModel(stack, norm, tensors["embedding"].to(device), rotary,
        config=config, max_length=description["max_length"], guard=guard)
    if model.cut != manifest["cut"] or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Packed depth or frozen state changed on reload")
    actual = packed_state_digest(model)
    if any(actual[key] != manifest[key] for key in actual):
        raise ValueError("Fresh packed structural reload changed actual parameter/buffer bits")
    return model, dict(manifest, directory=str(directory.resolve()), strict_state_load=True,
        packed_state_exact=True, inference_backend=model.inference_backend)


def _synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


@contextmanager
def attach_prefix(model, model_path, *, mode="resident"):
    """Temporary real native prefix; CPU mode streams weights, GPU computes.

    Both modes share the existing embedding. Offload avoids a full native
    prefix in VRAM and records every weight transfer. Timings between modes do
    not isolate a packing speedup and must not be compared as though they do.
    """
    if mode not in {"resident", "cpu"}: raise ValueError("Choose resident or explicitly CPU-offloaded prefix")
    if model.native_prefix is not None: raise ValueError("A native prefix is already attached")
    device = model.embedding.device; model._guard()
    started = time.perf_counter()
    stats = dict(mode=mode, cut=model.cut, calls=0, prefix_seconds=0., host_to_device_bytes=0,
        setup_seconds=0., shared_embedding=True, compute_device=str(device),
        scope="Prefix wall time and transfers are included in full-prompt recomputation timing")
    if model.cut == 0:
        stats["setup_seconds"] = time.perf_counter()-started
        yield stats
        return
    if mode == "resident":
        with source.attach_native_prefix(model, model_path, device, model.guard):
            original = model.native_prefix
            def timed(ids, valid):
                _synchronize(device); begin = time.perf_counter()
                value = original(ids, valid)
                _synchronize(device)
                stats["prefix_seconds"] += time.perf_counter()-begin; stats["calls"] += 1
                return value
            model.native_prefix = timed
            stats["setup_seconds"] = time.perf_counter()-started
            try: yield stats
            finally: model.native_prefix = original
        return
    config = source.builder.native._config(model_path)
    blocks = source._prefix_layers(model_path, model.cut, config, torch.device("cpu"), None)
    rotary = source.partial.training_rotary(config, "cpu").to(device)
    originals = [[(value, value.detach()) for value in (*block.parameters(), *block.buffers())] for block in blocks]
    @torch.no_grad()
    def prefix(ids, valid):
        model._guard(); _synchronize(device); begin = time.perf_counter()
        hidden = F.embedding(ids, model.embedding)
        positions = torch.arange(ids.shape[1], device=device)[None].expand_as(ids)
        allowed = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=device).tril()[None, None]
        allowed = allowed & valid[:, None, None, :]
        attention = torch.zeros(allowed.shape, dtype=hidden.dtype, device=device)
        attention.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        rope = rotary(hidden, positions)
        for block, values in zip(blocks, originals):
            model._guard()
            try:
                for value, original in values:
                    value.data = original.to(device)
                    if device.type == "cuda": stats["host_to_device_bytes"] += original.numel()*original.element_size()
                hidden = block(hidden, attention_mask=attention, position_ids=positions,
                    position_embeddings=rope, cache_position=positions[0], use_cache=False)[0]
            finally:
                # Restore original CPU storage without copying unchanged GPU
                # weights back to host. The frozen prefix owns no optimizer.
                for value, original in values: value.data = original
        _synchronize(device)
        stats["prefix_seconds"] += time.perf_counter()-begin; stats["calls"] += 1
        return hidden
    model.native_prefix = prefix
    stats["setup_seconds"] = time.perf_counter()-started
    try: yield stats
    finally:
        model.native_prefix = None
        del prefix, originals, blocks, rotary
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


@torch.no_grad()
def packed_old_mc_evaluate(arm, source_plan, model, guard=None):
    """Old MC DEV scored with the packed stack, never the training executor."""
    p = json.loads(Path(source_plan).read_text()) if isinstance(source_plan, (str, Path)) else source_plan
    source.progressive.source.check_plan(p)
    if arm == "progressive" and p.get("schema") == source.progressive.SCHEMA:
        dataset = source.progressive._dataset(p)
    elif arm == "all36" and p.get("schema") == source.joint.SCHEMA:
        dataset = source.joint.dataset(p)
    else: raise ValueError("Packed MC arm and source plan disagree")
    try:
        rows = []
        for index in dataset.development:
            model._guard()
            if guard is not None: guard()
            batch = dataset.batch([index], device=model.embedding.device, dtype=torch.bfloat16)
            output = model.stack(batch["hidden"], attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                position_embeddings=model.rotary(batch["hidden"], batch["position_ids"]), use_cache=False)[0]
            rows.append(source.joint.cache.sanity._record(dataset.items[index], model.norm(output[:, -1]),
                                                         model.embedding, p["label_ids"]))
        cases = [c for c in dataset.items if c["split"] == "development"]
        if len(cases) != 256 or p["native_correct"] != 195:
            raise ValueError("Expected the exposed 256-case MC development set")
        correct = sum(r["prediction"] == r["answer"] for r in rows)
        families = {f:dict(count=sum(r["family"] == f for r in rows),
            correct=sum(r["family"] == f and r["prediction"] == r["answer"] for r in rows))
            for f in sorted({r["family"] for r in rows})}
        return dict(metrics=dict(count=len(rows), correct=correct, accuracy=correct/len(rows), by_family=families),
            records=rows, native_correct=p["native_correct"], native_count=len(cases),
            eligibility=source.joint.broad.eligibility(rows, p["native_records"], cases), packed=True,
            inference_backend=model.inference_backend,
            scope="Packed exposed MC DEV retention; not independent confirmation")
    finally: dataset.close()
