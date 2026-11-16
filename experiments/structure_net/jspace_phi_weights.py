"""Local Phi-4 GPTQ-v1 decoding, bounded to one projection/block at a time.

This exposes *decoded quantized-source* weights, not the official BF16 model.
Packing reference: AutoGPTQ/AutoGPTQ ``qlinear_cuda.py`` pack/forward, where
qweight packs eight input rows and qzeros packs eight (zero_point - 1) columns.
https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda.py
No custom CUDA kernel, model download, or whole-model constructor is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

from safetensors import safe_open
import torch
from transformers import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding

SOURCE_KIND = "gptq_v1_int4_decoded_quantized_source"
REFERENCE_URL = "https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda.py"
DTYPE_BYTES = {"I32": 4, "F16": 2, "BF16": 2, "F32": 4}


def _device(device, guard):
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or reserved CUDA devices are supported")
    if device.type == "cuda":
        if guard is None:
            raise ValueError("A live reservation guard is required before CUDA access")
        guard()
    return device


def _dtype(dtype):
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("Decoded weights require FP16, BF16, or FP32")


def load_config(model_path, config=None):
    raw = json.loads((Path(model_path) / "config.json").read_text())
    q = raw.get("quantization_config", {})
    if (raw.get("model_type") != "phi3" or q.get("quant_method") != "gptq"
            or q.get("bits") != 4 or q.get("group_size") != 128
            or q.get("desc_act") is not False or q.get("checkpoint_format", "gptq") != "gptq"):
        raise ValueError("Only Phi3/Phi-4 GPTQ-v1 int4 group128 without act-order is supported")
    value = Phi3Config.from_dict(raw)
    if config is not None:
        for name in ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "vocab_size", "rms_norm_eps",
                     "rope_theta", "rope_scaling", "max_position_embeddings", "tie_word_embeddings"):
            if getattr(config, name) != getattr(value, name):
                raise ValueError(f"Supplied config differs from pinned Phi source: {name}")
        value = deepcopy(config)
    if value.tie_word_embeddings:
        raise ValueError("This Phi-4 loader requires the actual untied readout")
    value._attn_implementation = "eager"
    return value


def _index(model_path):
    root = Path(model_path).resolve()
    value = json.loads((root / "model.safetensors.index.json").read_text())
    mapping = value["weight_map"]
    for filename in set(mapping.values()):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Shard index must name local safetensors files")
        if not (root / filename).resolve().is_relative_to(root):
            raise ValueError("Shard resolves outside the authenticated model directory")
    return mapping


def _read(model_path, names):
    root, mapping, groups = Path(model_path), _index(model_path), {}
    for name in names:
        groups.setdefault(mapping[name], []).append(name)
    result = {}
    for filename, keys in groups.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                result[key] = handle.get_tensor(key)
    return result


def dequantize_gptq(qweight, scales, qzeros, g_idx, *, dtype=torch.bfloat16,
                    group_size=128, chunk_rows=128):
    """Decode [input/8,output] packed weights to contiguous [output,input].

    Products round to the stored scale dtype first, matching AutoGPTQ's eager
    tensor fallback; the requested execution dtype is a separate final cast.
    Chunking bounds temporary unpacked/FP32 buffers, not the final matrix.
    """
    _dtype(dtype)
    if (group_size != 128 or not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool)
            or chunk_rows < 8 or chunk_rows % 8):
        raise ValueError("Use group128 and a positive multiple-of-eight input chunk")
    if any(x.device.type != "cpu" for x in (qweight, scales, qzeros, g_idx)):
        raise ValueError("The bounded decoder operates on CPU tensors only")
    if qweight.ndim != 2 or qweight.dtype != torch.int32:
        raise ValueError("qweight must be a packed I32 matrix")
    n, m = qweight.shape[0] * 8, qweight.shape[1]
    if n == 0 or m == 0 or n % group_size or m % 8:
        raise ValueError("Unsupported packed dimensions")
    if (scales.shape != (n // group_size, m) or scales.dtype not in (torch.float16, torch.bfloat16)
            or qzeros.shape != (n // group_size, m // 8) or qzeros.dtype != torch.int32
            or g_idx.shape != (n,) or g_idx.dtype != torch.int32):
        raise ValueError("GPTQ scales/zero/group shapes or dtypes disagree")
    if not torch.equal(g_idx, torch.arange(n, dtype=torch.int32) // group_size):
        raise ValueError("g_idx is not the declared contiguous group128/no-act-order map")
    # The local AutoRound/GPTQ source uses signed scales. The published decoder
    # multiplies them literally; imposing positivity would change/reject it.
    if not torch.isfinite(scales).all():
        raise ValueError("GPTQ scales must be finite")
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    zeros = ((qzeros.unsqueeze(-1) >> shifts) & 15).reshape(scales.shape) + 1
    output = torch.empty((m, n), dtype=dtype)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        codes = ((qweight[start//8:end//8, None, :] >> shifts[None, :, None]) & 15).reshape(end-start, m)
        groups = g_idx[start:end].long()
        decoded = (codes - zeros[groups]).float() * scales[groups].float()
        decoded = decoded.to(scales.dtype)
        if not torch.isfinite(decoded).all():
            raise ValueError("Decoded GPTQ matrix is not finite in source precision")
        output[:, start:end].copy_(decoded.to(dtype).T)
    return output


def load_projection(model_path, name, *, dtype=torch.bfloat16, device="cpu", guard=None,
                    chunk_rows=128):
    device = _device(device, guard); _dtype(dtype); load_config(model_path)
    keys = [name + "." + k for k in ("qweight", "scales", "qzeros", "g_idx")]
    values = _read(model_path, keys)
    decoded = dequantize_gptq(*(values[key] for key in keys), dtype=dtype, chunk_rows=chunk_rows)
    if guard is not None:
        guard()
    return decoded.to(device).requires_grad_(False)


def load_native_block(model_path, layer, device="cpu", dtype=torch.bfloat16, *,
                      config=None, guard=None, chunk_rows=128):
    device = _device(device, guard); _dtype(dtype); config = load_config(model_path, config)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < config.num_hidden_layers:
        raise ValueError("Phi layer index outside the model")
    with torch.device("meta"):
        block = Phi3DecoderLayer(config, layer_idx=layer)
    prefix, state = f"model.layers.{layer}.", {}
    for key, expected in block.state_dict().items():
        name = prefix + key
        if key.endswith("_proj.weight"):
            value = load_projection(model_path, name.removesuffix(".weight"), device=device,
                                    dtype=dtype, guard=guard, chunk_rows=chunk_rows)
        else:
            value = _read(model_path, [name])[name]
            if value.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not torch.isfinite(value).all():
                raise ValueError("Native Phi tensor must be finite floating point")
            if guard is not None:
                guard()
            value = value.to(device=device, dtype=dtype)
        if value.shape != expected.shape:
            raise ValueError(f"Phi native tensor shape mismatch: {key}")
        state[key] = value
    block.load_state_dict(state, strict=True, assign=True)
    block.source_kind = SOURCE_KIND
    return block.eval().requires_grad_(False)


def _dense(model_path, key, device, dtype, guard):
    device = _device(device, guard); _dtype(dtype)
    value = _read(model_path, [key])[key]
    if value.dtype != torch.float16:
        raise ValueError("Pinned Phi GPTQ unquantized tensors must use stored FP16")
    if guard is not None:
        guard()
    return value.to(device=device, dtype=dtype).requires_grad_(False)


def load_embedding(model_path, device="cpu", dtype=torch.bfloat16, *, config=None, guard=None):
    config = load_config(model_path, config)
    value = _dense(model_path, "model.embed_tokens.weight", device, dtype, guard)
    if value.shape != (config.vocab_size, config.hidden_size):
        raise ValueError("Phi embedding shape mismatch")
    return value


def load_rotary(model_path, device="cpu", dtype=torch.bfloat16, *, config=None, guard=None):
    device = _device(device, guard); _dtype(dtype); config = load_config(model_path, config)
    # Construct frequencies on the intended execution device. Keep FP32 buffers.
    return Phi3RotaryEmbedding(config=config, device=device).eval().requires_grad_(False)


def load_readout_norm(model_path, device="cpu", dtype=torch.bfloat16, *, config=None, guard=None):
    device = _device(device, guard); _dtype(dtype); config = load_config(model_path, config)
    head = _dense(model_path, "lm_head.weight", device, dtype, guard)
    gain = _dense(model_path, "model.norm.weight", device, dtype, guard)
    if head.shape != (config.vocab_size, config.hidden_size) or gain.shape != (config.hidden_size,):
        raise ValueError("Phi untied readout/norm shape mismatch")
    with torch.device("meta"):
        norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.load_state_dict({"weight": gain}, strict=True, assign=True)
    rotary = load_rotary(model_path, device, dtype, config=config, guard=guard)
    return head, norm.eval().requires_grad_(False), rotary


def inspect_checkpoint(model_path):
    """Read only JSON/safetensor headers: payload inventory, NOT a SHA verification."""
    root = Path(model_path).resolve(); config = load_config(root); mapping = _index(root)
    tensors, files = {}, {}
    for filename in sorted(set(mapping.values())):
        path = root / filename
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 16 * 1024 * 1024:
                raise ValueError("Unexpectedly large safetensors header")
            header = json.loads(stream.read(length))
        files[filename] = dict(file_bytes=path.stat().st_size, header_bytes=8+length)
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if mapping.get(key) != filename or key in tensors:
                raise ValueError("Tensor headers disagree with shard index")
            size = value["data_offsets"][1] - value["data_offsets"][0]
            tensors[key] = dict(shape=value["shape"], dtype=value["dtype"], payload_bytes=size)
    if set(tensors) != set(mapping):
        raise ValueError("Shard index references absent tensors")
    layers = []
    for layer in range(config.num_hidden_layers):
        selected = {k:v for k,v in tensors.items() if k.startswith(f"model.layers.{layer}.")}
        dense_numel = sum(v["shape"][0]*8*v["shape"][1] for k,v in selected.items() if k.endswith(".qweight"))
        dense_numel += sum(v["payload_bytes"] // DTYPE_BYTES[v["dtype"]] for k,v in selected.items() if k.endswith(".weight"))
        layers.append(dict(layer=layer, quantized_payload_bytes=sum(v["payload_bytes"] for v in selected.values()),
                           decoded_parameter_count=dense_numel, decoded_fp16_bf16_bytes=2*dense_numel,
                           decoded_fp32_bytes=4*dense_numel))
    dense_rest = sum(v["payload_bytes"] // DTYPE_BYTES[v["dtype"]] for k,v in tensors.items()
                     if not k.startswith("model.layers.") and k.endswith(".weight"))
    total = dense_rest + sum(v["decoded_parameter_count"] for v in layers)
    metadata = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "generation_config.json"):
        if (root/name).exists():
            metadata[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    return dict(source_kind=SOURCE_KIND, source_precision="GPTQ-v1 int4 linears; FP16 scales/norms/untied embedding/head",
                path=str(root), reference_url=REFERENCE_URL, metadata_sha256=metadata, files=files,
                quantized_payload_bytes=sum(v["payload_bytes"] for v in tensors.values()),
                shard_file_bytes=sum(v["file_bytes"] for v in files.values()), layers=layers,
                dense_parameter_count=total, decoded_fp16_bf16_bytes=2*total, decoded_fp32_bytes=4*total,
                embedding=tensors["model.embed_tokens.weight"], readout=tensors["lm_head.weight"],
                payload_hashes_verified=False)
