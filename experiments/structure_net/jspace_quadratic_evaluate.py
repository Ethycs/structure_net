"""One-block-at-a-time FP32 scoring with ephemeral disk-backed token states.

Logical bank I/O is reported separately from model execution; these counters do
not claim physical device I/O because the OS may cache mapped pages. The caller
owns the input embedding/readout and must make loaders release any external GPU
references. This evaluator never retains a list of decoder blocks or gradients.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_quadratic_qwen import FoldedReadout


def _close(bank, *, flush=False):
    if bank is not None:
        if flush:
            bank.flush()
        bank._mmap.close()


def _bank(path, shape):
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing scratch bank: " + str(path))
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reader(head, label_ids, device):
    """Stage only selected output rows; never copy a full vocabulary to GPU."""
    if isinstance(head, FoldedReadout):
        indices = torch.tensor(label_ids, device=head.weight.device)
        weight = head.weight.detach().index_select(0, indices).to(device=device, dtype=torch.float32)
        bias = head.bias.detach().index_select(0, indices).to(device=device, dtype=torch.float32)
        gram = head.gram.detach().to(device=device, dtype=torch.float32)
        cross = head.cross.detach().to(device=device, dtype=torch.float32)
        energy = head.anchor_energy.detach().to(device=device, dtype=torch.float32)
        width, eps = head.native_width, head.epsilon
        def score(hidden):
            variance = ((F.linear(hidden, gram)*hidden).sum(-1, keepdim=True)
                        +2*(hidden*cross).sum(-1, keepdim=True)+energy).clamp_min(0)/width
            return F.linear(hidden, weight, bias)/(variance+eps).sqrt()
        return score
    if isinstance(head, tuple) and len(head) == 2:
        norm, embedding = head
        if not hasattr(norm, "variance_epsilon") or not hasattr(norm, "weight"):
            raise TypeError("Native readout tuple must contain Qwen RMSNorm and a vocabulary matrix")
        indices = torch.tensor(label_ids, device=embedding.device)
        weight = embedding.detach().index_select(0, indices).to(device=device, dtype=torch.float32)
        gain = norm.weight.detach().to(device=device, dtype=torch.float32)
        eps = norm.variance_epsilon
        def score(hidden):
            normalized = hidden*torch.rsqrt(hidden.square().mean(-1, keepdim=True)+eps)
            return F.linear(gain*normalized, weight)
        return score
    if callable(head):
        # Explicit callback contract, useful for other reference heads. The
        # callback returns [batch, len(label_ids)] logits, not full vocabulary.
        return lambda hidden: head(hidden, label_ids)
    raise TypeError("Readout must be FoldedReadout, (QwenRMSNorm,E), or score_labels(hidden,label_ids) callback")


def _validate_cases(cases, embedding, label_ids):
    if not cases or embedding.ndim != 2 or not embedding.is_floating_point():
        raise ValueError("Expected nonempty cases and a floating [vocabulary,width] embedding")
    if embedding.device.type != "cpu":
        raise ValueError("Keep the input embedding on CPU for one-block GPU streaming")
    if not label_ids or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in label_ids):
        raise ValueError("label_ids must be nonnegative vocabulary indices")
    identities = [case["identity"] for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate evaluation case identities")
    lengths = []
    for case in cases:
        tokens, choices, answer = case["input_ids"], case["choices"], case["answer"]
        if not tokens or any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < len(embedding) for t in tokens):
            raise ValueError("Input token IDs must be nonempty and within the embedding vocabulary")
        if (not 0 < len(choices) <= len(label_ids) or isinstance(answer, bool)
                or not isinstance(answer, int) or not 0 <= answer < len(choices)):
            raise ValueError("Invalid answer choices or answer index")
        lengths.append(len(tokens))
    return lengths


@torch.inference_mode()
def stream_score(cases, *, input_embedding, block_loader, final_readout, label_ids, scratch_dir,
                 rotary=None, position_embeddings=None, layer_count=36, bridge_loader=None,
                 device="cpu", guard=lambda: None, microbatch=4, progress=lambda **kw: None):
    """Score cases in original order using two alternating FP32 token banks.

    ``block_loader(i)`` returns a frozen native HF decoder block or a compact
    wrapper exposing ``forward_coordinates``. Only that block is retained until
    its full bank pass completes. ``bridge_loader(i)`` optionally returns fixed
    ``(weight, shift)`` applied after block i (never after the final block).
    Bridge weights follow F.linear's [next_width,current_width] convention;
    shifts may be global [width]/[1,width] or position tables [P,width].

    Supply exactly one of ``rotary`` or ``position_embeddings``; both have the
    callable convention (hidden, position_ids) -> (cos,sin). ``input_embedding``
    is a CPU native E or folded E_in tensor. A native final readout is (norm,E),
    a compact head is FoldedReadout, or a custom callback accepts (hidden,ids).
    Only selected label rows are staged for the supplied standard heads.

    ``scratch_dir`` must not exist: this function owns it and removes its banks
    on success or error. No caller files are recursively deleted. Final-layer
    storage contains only each case's last valid row. Returns records and I/O,
    execution-time and shape diagnostics; outputs are not saved in scratch.
    """
    device, path = torch.device(device), Path(scratch_dir)
    if (isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 1
            or isinstance(microbatch, bool) or not isinstance(microbatch, int) or microbatch < 1):
        raise ValueError("layer_count and microbatch must be positive integers")
    if (rotary is None) == (position_embeddings is None):
        raise ValueError("Supply exactly one rotary/position_embeddings callback")
    positional = rotary if rotary is not None else position_embeddings
    embedding = input_embedding.weight if isinstance(input_embedding, torch.nn.Embedding) else input_embedding
    if isinstance(label_ids, torch.Tensor):
        if label_ids.ndim != 1:
            raise ValueError("label_ids must be one-dimensional")
        labels = label_ids.detach().cpu().tolist()
    else:
        labels = list(label_ids)
    lengths = _validate_cases(cases, embedding, labels)
    offsets = np.cumsum([0, *lengths], dtype=np.int64)
    total_tokens = int(offsets[-1])
    path.mkdir(parents=True, exist_ok=False)
    bank_paths = [path/"tokens_0.npy", path/"tokens_1.npy"]
    source = destination = None
    started = time.perf_counter()
    metrics = dict(cases=len(cases), total_tokens=total_tokens, dtype="float32", device=str(device),
                   microbatch=microbatch, max_live_decoder_blocks=1, layers=[],
                   logical_read_bytes=0, logical_write_bytes=0, max_scratch_bytes=0,
                   scope="Mapped logical I/O; OS page cache and physical I/O are not measured")
    try:
        guard()
        begin = time.perf_counter()
        source = _bank(bank_paths[0], (total_tokens, embedding.shape[1]))
        for i, case in enumerate(cases):
            guard()
            values = embedding.detach().index_select(0, torch.tensor(case["input_ids"])).float()
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite input embedding rows")
            source[offsets[i]:offsets[i+1]] = values.numpy()
        source.flush()
        metrics["initial_embedding_seconds"] = time.perf_counter()-begin
        metrics["logical_write_bytes"] += total_tokens*embedding.shape[1]*4
        metrics["max_scratch_bytes"] = bank_paths[0].stat().st_size
        current = 0
        for layer in range(layer_count):
            guard()
            start_layer = time.perf_counter()
            weight = shift = values = selected = tensor = None
            block = block_loader(layer)
            if not isinstance(block, torch.nn.Module):
                raise TypeError("block_loader must return a decoder module")
            block.eval().requires_grad_(False)
            for tensor in (*block.parameters(), *block.buffers()):
                if tensor.device != device or tensor.is_floating_point() and tensor.dtype != torch.float32:
                    raise ValueError("Streaming blocks must already be on the selected device in FP32")
            compact = callable(getattr(block, "forward_coordinates", None))
            bridge = None
            if bridge_loader is not None and layer < layer_count-1:
                weight, shift = bridge_loader(layer)
                weight, shift = weight.detach().to(device=device, dtype=torch.float32), shift.detach().to(device=device, dtype=torch.float32)
                if weight.ndim != 2 or shift.ndim not in (1, 2) or shift.shape[-1] != weight.shape[0]:
                    raise ValueError("Invalid reduced bridge dimensions")
                bridge = (weight, shift)
            destination_path = bank_paths[1-current]
            row_count = len(cases) if layer == layer_count-1 else total_tokens
            row = dict(layer=layer, compact=compact, input_shape=list(source.shape), output_shape=None,
                       bridge=bridge is not None, logical_read_bytes=0, logical_write_bytes=0,
                       read_seconds=0., compute_seconds=0., write_seconds=0., host_to_device_bytes=0,
                       parameter_bytes=sum(p.numel()*p.element_size() for p in block.parameters()),
                       buffer_bytes=sum(b.numel()*b.element_size() for b in block.buffers()))
            for begin_case in range(0, len(cases), microbatch):
                guard()
                indices = list(range(begin_case, min(begin_case+microbatch, len(cases))))
                longest = max(lengths[i] for i in indices)
                begin = time.perf_counter()
                host = np.zeros((len(indices), longest, source.shape[1]), dtype=np.float32)
                for j, i in enumerate(indices):
                    host[j, :lengths[i]] = source[offsets[i]:offsets[i+1]]
                row["read_seconds"] += time.perf_counter()-begin
                row["logical_read_bytes"] += sum(lengths[i] for i in indices)*source.shape[1]*4
                hidden = torch.from_numpy(host).to(device)
                row["host_to_device_bytes"] += host.nbytes if device.type != "cpu" else 0
                positions = torch.arange(longest, device=device)
                position_ids = positions[None].expand(len(indices), -1)
                key_valid = positions[None] < torch.tensor([lengths[i] for i in indices], device=device)[:, None]
                causal = positions[None, :] <= positions[:, None]
                allowed = key_valid[:, None, None, :] & causal[None, None]
                mask = torch.zeros(allowed.shape, device=device, dtype=torch.float32).masked_fill_(~allowed, torch.finfo(torch.float32).min)
                embeddings = positional(hidden, position_ids)
                _synchronize(device); begin = time.perf_counter()
                kwargs = dict(attention_mask=mask, position_ids=position_ids, cache_position=positions,
                              position_embeddings=embeddings, use_cache=False)
                result = block.forward_coordinates(hidden, **kwargs) if compact else block(hidden, **kwargs)
                result = result[0] if isinstance(result, tuple) else result
                if result.ndim != 3 or result.shape[:2] != hidden.shape[:2] or result.dtype != torch.float32:
                    raise ValueError("Decoder must return all query rows as FP32 [batch,length,width]")
                if bridge is not None:
                    weight, shift = bridge
                    if result.shape[-1] != weight.shape[1]:
                        raise ValueError("Bridge input width differs from block output")
                    if shift.ndim == 2 and len(shift) != 1:
                        if len(shift) < longest:
                            raise ValueError("Bridge position table is shorter than this batch")
                        shift = shift[:longest]
                    result = F.linear(result, weight)+shift
                _synchronize(device); row["compute_seconds"] += time.perf_counter()-begin
                if not torch.isfinite(result).all():
                    raise FloatingPointError(f"Nonfinite layer {layer} outputs")
                if destination is None:
                    destination = _bank(destination_path, (row_count, result.shape[-1]))
                    row["output_shape"] = list(destination.shape)
                    metrics["max_scratch_bytes"] = max(metrics["max_scratch_bytes"],
                        sum(p.stat().st_size for p in bank_paths if p.exists()))
                begin = time.perf_counter()
                if layer == layer_count-1:
                    selected = result[torch.arange(len(indices), device=device),
                        torch.tensor([lengths[i]-1 for i in indices], device=device)]
                    destination[indices] = selected.cpu().numpy()
                    row["logical_write_bytes"] += selected.numel()*4
                else:
                    values = result.cpu().numpy()
                    for j, i in enumerate(indices):
                        destination[offsets[i]:offsets[i+1]] = values[j, :lengths[i]]
                    row["logical_write_bytes"] += sum(lengths[i] for i in indices)*result.shape[-1]*4
                row["write_seconds"] += time.perf_counter()-begin
                del hidden, result, embeddings, host
                progress(kind="batch", layer=layer, completed=min(begin_case+microbatch, len(cases)), total=len(cases))
            begin = time.perf_counter(); destination.flush(); row["write_seconds"] += time.perf_counter()-begin
            _close(source); source = None
            bank_paths[current].unlink()
            source, destination = destination, None
            current = 1-current
            row["total_seconds"] = time.perf_counter()-start_layer
            metrics["logical_read_bytes"] += row["logical_read_bytes"]
            metrics["logical_write_bytes"] += row["logical_write_bytes"]
            metrics["layers"].append(row)
            del block, bridge
            # Variables from the last batch must not retain any decoder output
            # or bridge tensors while the next block is loaded.
            del values, selected, weight, shift, tensor
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            guard(); progress(kind="layer", **row)
        readout_start = time.perf_counter()
        reader = _reader(final_readout, labels, device)
        records = []
        for begin_case in range(0, len(cases), microbatch):
            guard()
            end = min(begin_case+microbatch, len(cases))
            host = np.array(source[begin_case:end], copy=True)
            metrics["logical_read_bytes"] += host.nbytes
            logits = reader(torch.from_numpy(host).to(device)).float()
            if logits.shape != (end-begin_case, len(labels)) or not torch.isfinite(logits).all():
                raise ValueError("Readout must return finite selected-label logits [batch,len(label_ids)]")
            rows = logits.cpu().tolist()
            for offset, case in enumerate(cases[begin_case:end]):
                values = rows[offset][:len(case["choices"])]
                records.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
                                    prediction=max(range(len(values)), key=values.__getitem__), logits=values))
        metrics["final_readout_seconds"] = time.perf_counter()-readout_start
        metrics["total_seconds"] = time.perf_counter()-started
        metrics["scratch_banks_removed"] = True
        return {"records": records, "metrics": metrics}
    finally:
        _close(source); _close(destination)
        for bank_path in bank_paths:
            if bank_path.exists():
                bank_path.unlink()
        # Never recursively remove unexpected user files placed in this folder.
        if path.exists() and not any(path.iterdir()):
            path.rmdir()


def paired_summary(native, compact, *, maximum_accuracy_drop=.05):
    """Aggregate five-point quality gate; family losses remain diagnostics."""
    from .jspace_qwen_parallel_evaluate import paired_quality
    if not math.isfinite(maximum_accuracy_drop) or not 0 <= maximum_accuracy_drop <= 1:
        raise ValueError("Accuracy drop budget must lie in [0,1]")
    quality = paired_quality(native, compact)
    return dict(quality=quality, maximum_accuracy_drop=maximum_accuracy_drop,
                aggregate_quality_pass=quality["overall"]["accuracy_drop"] <= maximum_accuracy_drop+1e-12,
                family_quality_is_gate=False)
