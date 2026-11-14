"""Grouped native-block calibration through a fixed accepted Qwen teacher.

One forward and one VJP per output probe serve every requested new block.
All context-token MLP derivatives are retained as LOSSLESS BF16 bits. The
caller owns reservations, frozen teacher identity, original-teacher targets,
and subsequent independent basis/graph fitting; no model weights are changed.
"""
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_logit_sketch import project_sign_probe_groups, normalization_metadata


def write_bfloat16_bits(array, index, value):
    """Write the actual backward/forward BF16 values, never quantize FP32."""
    if value.dtype != torch.bfloat16 or array.dtype != np.dtype("int16"):
        raise ValueError("Lossless BF16-bit storage requires an already-bfloat16 tensor and int16 destination")
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite capture tensor")
    array[index] = value.detach().contiguous().view(torch.int16).cpu().numpy()


class StreamedBFloat16JacobianRows:
    """Read-only BF16-bit .npy rows exposed as exact FP32 values to the fitter."""
    def __init__(self, path, *, device="cpu", max_rows=256):
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        self.path, self.device, self.max_rows = Path(path).resolve(), torch.device(device), max_rows
        value = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if not isinstance(value, np.memmap) or value.ndim != 2 or min(value.shape) <= 0 or value.dtype != np.dtype("int16") or not value.dtype.isnative:
            if isinstance(value, np.memmap): value._mmap.close()
            raise ValueError("Expected nonempty native-endian int16 BF16-bit matrix")
        self._matrix, self.shape, self.ndim, self.dtype = value, tuple(value.shape), 2, torch.float32
        self.read_calls = self.rows_read = self.bytes_read = self.maximum_read_rows = 0

    def __len__(self): return self.shape[0]
    def is_floating_point(self): return True
    @property
    def readonly(self): return True

    def __getitem__(self, index):
        if self._matrix is None: raise ValueError("Jacobian source is closed")
        if not isinstance(index, slice) or index.step not in (None, 1):
            raise TypeError("Only contiguous row slices are supported")
        start, stop, _ = index.indices(len(self)); count = max(stop - start, 0)
        if count > self.max_rows: raise ValueError("Requested slice exceeds max_rows")
        bits = torch.from_numpy(np.array(self._matrix[start:stop], copy=True, order="C"))
        result = bits.view(torch.bfloat16).float().to(self.device, non_blocking=False)
        self.read_calls += 1; self.rows_read += count; self.bytes_read += count * self.shape[1] * 2
        self.maximum_read_rows = max(self.maximum_read_rows, count)
        return result

    def metadata(self):
        return {"path": str(self.path), "shape": list(self.shape), "dtype": "torch.float32", "storage": "lossless BF16 bits in int16",
                "device": str(self.device), "max_rows": self.max_rows, "readonly": True,
                "read_calls": self.read_calls, "rows_read": self.rows_read, "bytes_read": self.bytes_read,
                "maximum_read_rows": self.maximum_read_rows, "transfer": "owned bounded BF16-bit copy, exact FP32 conversion, blocking transfer"}

    def close(self):
        if self._matrix is not None: self._matrix._mmap.close(); self._matrix = None
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def _length(case):
    return len(case["input_ids"]) if "input_ids" in case else int(case["length"])


def estimate_group_bytes(cases, calibration_indices, block_shapes, *, probe_groups=2, probes_per_group=8,
                         activation_positions=16, vocabulary_size=151936):
    """Payload bytes only; excludes small .npy headers and moment-statistic files."""
    lengths = [_length(c) for c in cases]; calibration = list(calibration_indices)
    tokens, derivative_rows = sum(lengths), sum(lengths[i] for i in calibration) * probe_groups * probes_per_group
    sampled = (len(calibration) + sum(c["split"] == "development" for c in cases)) * activation_positions
    positions = len(cases[0]["loss_positions"])
    blocks = {str(i): {"boundary": tokens * s["width"] * 2, "mlp_activations": sampled * s["intermediate"] * 2,
                       "mlp_jacobian": derivative_rows * s["intermediate"] * 2} for i, s in block_shapes.items()}
    logits = len(cases) * positions * vocabulary_size * 2
    return {"blocks": blocks, "accepted_logits": logits, "total": logits + sum(sum(v.values()) for v in blocks.values()),
            "jacobian_rows_per_block": derivative_rows, "context_tokens": tokens,
            "calibration_context_tokens": sum(lengths[i] for i in calibration), "storage": "already-BF16 values; no extra quantization"}


class GroupedTeacherCapture:
    """Capture one fixed accepted teacher in groups of at most six new blocks.

    ``forward()`` must return [1,L,d] final NORMALIZED hidden states from the
    accepted teacher, e.g. model.model(..., use_cache=False).last_hidden_state.
    It may start from a cached frozen prefix. The first requested native block
    input becomes the independent variable; its complete accepted suffix stays
    in the graph. All requested new blocks must expose native Qwen Q/K/V and
    down_proj modules. Already-compact suffix blocks require no hooks.
    """
    def __init__(self, root, model, layer_ids, cases, calibration_indices, *, teacher_fingerprint,
                 probe_groups=2, probes_per_group=8, activation_positions=16, position_capacity=None,
                 seed=20861, max_blocks=6, guard=None):
        self.root, self.model = Path(root), model
        self.layers = tuple(sorted(layer_ids)); self.cases = list(cases); self.calibration = tuple(sorted(calibration_indices))
        if self.root.exists(): raise FileExistsError(self.root)
        if not self.layers or len(set(self.layers)) != len(self.layers) or len(self.layers) > max_blocks or max_blocks > 6:
            raise ValueError("Request one unique group of at most six blocks")
        if not self.cases or not self.calibration or len(set(self.calibration)) != len(self.calibration):
            raise ValueError("Nonempty cases and unique calibration indices required")
        if any(i < 0 or i >= len(self.cases) or self.cases[i]["split"] != "train" for i in self.calibration):
            raise ValueError("Jacobian calibration may use training cases only")
        if any(c["split"] not in ("train", "development") for c in self.cases):
            raise ValueError("Held-out evaluation is forbidden in calibration capture")
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in (probe_groups, probes_per_group, activation_positions)):
            raise ValueError("Positive integer probe and activation counts required")
        if not teacher_fingerprint: raise ValueError("Fixed accepted teacher fingerprint required")
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Accepted teacher must be eval() with all parameters frozen")
        self.fixed_tensors = list(model.parameters()) + list(model.buffers())
        if any(t.is_inference() for t in self.fixed_tensors):
            raise ValueError("VJP teacher requires normal tensors, not inference-mode tensors")
        self.versions = [(id(t), t._version) for t in self.fixed_tensors]
        self.readout = model.lm_head.weight; self.device = self.readout.device; self.guard = guard
        if self.device.type == "cuda":
            if guard is None: raise ValueError("CUDA capture requires the caller's reservation guard")
            guard()
        self.groups, self.probes, self.positions = probe_groups, probes_per_group, activation_positions
        self.lengths = [_length(c) for c in self.cases]
        self.output_positions = len(self.cases[0]["loss_positions"])
        if self.output_positions < 1 or any(_length(c) < 1 or len(c["loss_positions"]) != self.output_positions or
                any(p < 0 or p >= _length(c) for p in c["loss_positions"]) for c in self.cases):
            raise ValueError("Invalid aligned output positions")
        self.capacity = position_capacity or max(self.lengths)
        if self.capacity < max(self.lengths): raise ValueError("Position capacity is too short")
        self.offsets = np.cumsum([0] + self.lengths).tolist()
        self.development = tuple(i for i, c in enumerate(self.cases) if c["split"] == "development")
        if not self.development: raise ValueError("Native development activation rows required")
        self.cal_rows = {i: row for row, i in enumerate(self.calibration)}
        self.dev_rows = {i: row for row, i in enumerate(self.development)}
        self.jac_offsets = {}; cursor = 0
        for i in self.calibration:
            self.jac_offsets[i] = cursor; cursor += self.lengths[i] * self.groups * self.probes
        self.shapes, self.modules = {}, {}
        for i in self.layers:
            block = model.model.layers[i]
            if not hasattr(block, "self_attn") or not hasattr(block, "mlp"):
                raise ValueError("Requested new blocks must expose native Qwen projections")
            attention = block.self_attn; config = attention.config
            width, intermediate = attention.q_proj.in_features, block.mlp.down_proj.in_features
            heads, kv, head_dim = config.num_attention_heads, config.num_key_value_heads, attention.head_dim
            if heads % kv or head_dim % 2: raise ValueError("Unsupported GQA/RoPE dimensions")
            self.shapes[i] = {"width": width, "intermediate": intermediate, "heads": heads, "kv_heads": kv, "head_dim": head_dim}
            self.modules[i] = block
        self.estimate = estimate_group_bytes(self.cases, self.calibration, self.shapes, probe_groups=self.groups,
            probes_per_group=self.probes, activation_positions=self.positions, vocabulary_size=len(self.readout))
        self.root.mkdir(parents=True); self.arrays, self.statistics = {}, {}
        self.logits = np.lib.format.open_memmap(self.root / "accepted_teacher_logits.npy", mode="w+", dtype=np.int16,
            shape=(len(self.cases), self.output_positions, len(self.readout)))
        for i, spec in self.shapes.items():
            path = self.root / f"block_{i:02d}"; path.mkdir(); d, m = spec["width"], spec["intermediate"]
            shapes = {"boundary": (self.offsets[-1], d), "mlp_calibration": (len(self.calibration) * self.positions, m),
                      "mlp_development": (len(self.development) * self.positions, m), "mlp_jacobian_bf16": (cursor, m)}
            self.arrays[i] = {name: np.lib.format.open_memmap(path / (name + ".npy"), mode="w+", dtype=np.int16, shape=shape)
                              for name, shape in shapes.items()}
            kv, h = spec["kv_heads"], spec["head_dim"]; pairs = kv * h // 2
            zero = lambda *shape: torch.zeros(*shape, device=self.device, dtype=torch.float32)
            self.statistics[i] = {"input_grams": zero(self.groups, d, d), "mean_sums": zero(self.capacity, d),
                "position_counts": zero(self.capacity), "pair_scores": zero(self.groups, kv, h // 2),
                "mask_grams": zero(self.groups, pairs, pairs), "value_moments": zero(kv, h, h), "value_means": zero(kv, h)}
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.seen, self.order, self.failed, self.finished = set(), [], False, False
        self.metadata = {"teacher_fingerprint": teacher_fingerprint, "layers": list(self.layers), "seed": seed,
            "probe_groups": self.groups, "probes_per_group": self.probes, "activation_positions": self.positions,
            "calibration_indices": list(self.calibration), "development_indices": list(self.development),
            "row_order": "calibration case index, independent probe group, probe, ALL context token positions",
            "storage": "raw BF16 bits in int16; backward dtype checked before every write; no quantization",
            "teacher_execution": "grad mode enabled for every forward; only calibration inputs require gradients; consistent differentiable Torch suffix path",
            "normalization": normalization_metadata(len(self.readout), self.output_positions, self.probes), "estimated_payload_bytes": self.estimate,
            "original_teacher_targets": "separate caller-owned cache; never overwritten or relabeled as accepted teacher"}

    def _check_teacher(self):
        current = list(self.model.parameters()) + list(self.model.buffers())
        if self.model.training or any(p.requires_grad for p in self.model.parameters()) or [(id(t), t._version) for t in current] != self.versions:
            raise ValueError("Accepted teacher changed during grouped capture")
        if self.guard is not None: self.guard()

    def capture_case(self, index, forward):
        if self.failed or self.finished: raise ValueError("Capture is failed or already finished")
        if index in self.seen or not 0 <= index < len(self.cases): raise ValueError("Invalid or duplicate case index")
        self._check_teacher(); calibration = index in self.cal_rows
        if calibration and torch.is_inference_mode_enabled(): raise ValueError("Jacobian capture cannot run inside inference_mode")
        captured, handles = {i: {} for i in self.layers}, []
        def boundary_hook(module, args, layer):
            value = args[0]
            if layer == self.layers[0]:
                value = value.detach().clone().requires_grad_(True) if calibration else value.detach()
            captured[layer]["input"] = value
            return (value, *args[1:])
        try:
            for layer, block in self.modules.items():
                handles.append(block.register_forward_pre_hook(lambda m, a, i=layer: boundary_hook(m, a, i)))
                for kind in ("q", "k", "v"):
                    handles.append(getattr(block.self_attn, kind + "_proj").register_forward_hook(
                        lambda m, a, out, i=layer, key=kind: captured[i].__setitem__(key, out)))
                handles.append(block.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=layer: captured[i].__setitem__("mlp", a[0])))
            # Fast suffixes select their Torch path using grad mode. Keep that
            # path identical for cache-only and VJP cases; frozen parameters
            # and a detached input avoid constructing the cache-only graph.
            with torch.enable_grad():
                output = forward()
                if output.shape != (1, self.lengths[index], self.readout.shape[1]): raise ValueError("forward must return normalized [1,L,d] teacher states")
                if calibration and not output.requires_grad: raise ValueError("Accepted suffix/final RMSNorm was detached")
            for handle in handles: handle.remove()
            handles.clear()
            self._record_forward(index, captured, output)
            if calibration:
                with torch.enable_grad(): self._record_vjps(index, captured, output)
            self._check_teacher(); self.seen.add(index); self.order.append(index)
        except BaseException:
            self.failed = True
            raise
        finally:
            for handle in handles: handle.remove()

    def _record_forward(self, index, captured, output):
        case, length = self.cases[index], self.lengths[index]
        with torch.no_grad():
            selected = output[0, case["loss_positions"]]
            write_bfloat16_bits(self.logits, index, F.linear(selected.to(self.readout.dtype), self.readout))
            sample = torch.linspace(0, length - 1, self.positions, device=self.device).long()
            for layer, values in captured.items():
                if set(values) != {"input", "q", "k", "v", "mlp"}: raise ValueError("Requested teacher block was not executed")
                arrays, stats, spec = self.arrays[layer], self.statistics[layer], self.shapes[layer]
                write_bfloat16_bits(arrays["boundary"], slice(self.offsets[index], self.offsets[index + 1]), values["input"][0])
                if case["split"] == "train":
                    stats["mean_sums"][:length].add_(values["input"][0].float()); stats["position_counts"][:length].add_(1)
                if index in self.cal_rows or index in self.dev_rows:
                    name, row = ("mlp_calibration", self.cal_rows[index]) if index in self.cal_rows else ("mlp_development", self.dev_rows[index])
                    write_bfloat16_bits(arrays[name], slice(row * self.positions, (row + 1) * self.positions), values["mlp"][0, sample])
                if index in self.cal_rows:
                    value = values["v"][0].float().reshape(length, spec["kv_heads"], spec["head_dim"]).transpose(0, 1)
                    stats["value_moments"].add_(value.transpose(1, 2) @ value / length); stats["value_means"].add_(value.mean(1))

    def _record_vjps(self, index, captured, output):
        normalized = output[0, self.cases[index]["loss_positions"]]
        vectors = project_sign_probe_groups(self.readout, self.output_positions, self.probes, self.generator, groups=self.groups)
        targets = [captured[i][key] for i in self.layers for key in ("input", "q", "k", "mlp")]
        length = self.lengths[index]
        for group in range(self.groups):
            for probe in range(self.probes):
                scalar = (normalized.to(vectors.dtype) * vectors[group, probe]).sum()
                gradients = torch.autograd.grad(scalar, targets, retain_graph=not (group == self.groups - 1 and probe == self.probes - 1))
                for offset, layer in enumerate(self.layers):
                    gx, gq, gk, gm = gradients[4 * offset:4 * offset + 4]
                    row = self.jac_offsets[index] + (group * self.probes + probe) * length
                    write_bfloat16_bits(self.arrays[layer]["mlp_jacobian_bf16"], slice(row, row + length), gm[0])
                    stats, spec = self.statistics[layer], self.shapes[layer]
                    x = gx[0].detach().float()
                    if not torch.isfinite(x).all(): raise FloatingPointError("Nonfinite input VJP")
                    stats["input_grams"][group].addmm_(x.T, x, alpha=1 / self.probes)
                    kv, h = spec["kv_heads"], spec["head_dim"]; sharing = spec["heads"] // kv
                    q = (captured[layer]["q"].detach().float() * gq.float())[0].reshape(length, kv, sharing, h).sum((0, 2))
                    k = (captured[layer]["k"].detach().float() * gk.float())[0].reshape(length, kv, h).sum(0)
                    mask = q[:, :h // 2] + q[:, h // 2:] + k[:, :h // 2] + k[:, h // 2:]
                    stats["pair_scores"][group].add_(mask.square() / self.probes)
                    flat = mask.flatten(); stats["mask_grams"][group].add_(flat[:, None] * flat[None, :] / self.probes)

    def finish(self):
        if self.failed or self.finished or len(self.seen) != len(self.cases): raise ValueError("Only a complete successful capture can finish")
        self._check_teacher(); self.logits.flush(); blocks = {}
        for layer, stats in self.statistics.items():
            counts = stats["position_counts"]; means = stats["mean_sums"] / counts[:, None].clamp_min(1)
            means[counts == 0] = stats["mean_sums"].sum(0) / counts.sum()
            result = {name: value / len(self.calibration) for name, value in stats.items()
                      if name not in ("mean_sums", "position_counts")}
            result.update(means=means, position_counts=counts, counts=counts,
                          grams=result["input_grams"], input_gram=result["input_grams"].mean(0))
            if any(not torch.isfinite(v).all() for v in result.values()): raise FloatingPointError("Nonfinite captured statistics")
            path = self.root / f"block_{layer:02d}"
            torch.save({k: v.detach().cpu() for k, v in result.items()}, path / "statistics.pt")
            for array in self.arrays[layer].values(): array.flush()
            blocks[str(layer)] = {"statistics": str(path / "statistics.pt"),
                **{name: str(path / (name + ".npy")) for name in self.arrays[layer]}, "shape": self.shapes[layer]}
        items = [{**case, "offset": self.offsets[i], "stop": self.offsets[i + 1],
                  "activation_positions": torch.linspace(0, self.lengths[i] - 1, self.positions).long().tolist(),
                  "jacobian_row_offset": self.jac_offsets.get(i)} for i, case in enumerate(self.cases)]
        metadata = {**self.metadata, "status": "completed", "capture_order": self.order, "blocks": blocks,
                    "calibration_families": [self.cases[i]["family"] for i in self.calibration for _ in range(self.positions)],
                    "development_families": [self.cases[i]["family"] for i in self.development for _ in range(self.positions)],
                    "accepted_teacher_logits": str(self.root / "accepted_teacher_logits.npy")}
        for name, value in (("items.json", {"items": items}), ("metadata.json", metadata)):
            with (self.root / name).open("x") as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        self.finished = True
        return metadata

    def close(self):
        for array in [self.logits, *(a for arrays in self.arrays.values() for a in arrays.values())]:
            if array._mmap is not None and not array._mmap.closed: array.flush(); array._mmap.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
