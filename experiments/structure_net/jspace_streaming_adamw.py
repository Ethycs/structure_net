"""Bounded-transfer AdamW with persistent FP32 CPU masters and moments.

This extends the existing NCM CPU-master approach to low-precision model
parameters without retaining all host gradients or rounding masters at resume.
Device weights/gradients stay in their declared dtype. CPU AdamW arithmetic is
not promised to match a CUDA AdamW trajectory bit for bit. The implementation
uses synchronous transfers deliberately: staging-buffer ownership is explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


class StreamingCPUAdamW(torch.optim.Optimizer):
    """One parameter group; accumulation remains the caller's responsibility.

    ``chunk_bytes`` bounds one FP32 work chunk (default 32 MiB). Two FP32 work
    buffers plus one staging buffer per parameter dtype may be live. Optimizer state
    itself occupies 12 bytes per trainable scalar on CPU. CUDA norm reduction
    may temporarily cast one gradient chunk to FP32. No CUDA availability probe
    or GPU allocation is performed for a CPU-only instance.

    Call ``step(max_grad_norm=...)`` after the desired accumulation. Nonfinite
    gradients raise before any parameter/moment update. For explicit FP16 loss
    scaling pass ``grad_scale``; this is not a dynamic GradScaler replacement.
    Checkpoint directories store master/m/v chunks, never a second full state.
    """

    SCHEMA = "jspace.streaming-cpu-adamw.v1"

    def __init__(self, params, *, lr=3e-5, betas=(.9, .999), eps=1e-8,
                 weight_decay=0., chunk_bytes=32 * 1024**2, max_cpu_state_bytes=None,
                 pin_memory=None):
        params = list(params)
        if not params or len({id(p) for p in params}) != len(params):
            raise ValueError("Distinct nonempty trainable parameter list required")
        if any(not p.requires_grad or not p.is_contiguous() or p.numel() == 0 or
               p.dtype not in (torch.float32, torch.bfloat16, torch.float16) for p in params):
            raise ValueError("Trainable contiguous FP32/BF16/FP16 parameters required")
        if not isinstance(chunk_bytes, int) or chunk_bytes < 4:
            raise ValueError("chunk_bytes must hold at least one FP32 scalar")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._validate_hyperparameters(defaults)
        state_bytes = 12 * sum(p.numel() for p in params)
        if max_cpu_state_bytes is not None and state_bytes > max_cpu_state_bytes:
            raise MemoryError(f"FP32 CPU optimizer state requires {state_bytes} bytes")
        super().__init__(params, defaults)
        self._parameters = tuple(params)
        self.chunk_elements = min(chunk_bytes // 4, max(p.numel() for p in params))
        self.chunk_bytes = self.chunk_elements * 4
        self._dtype_elements = {dtype: min(self.chunk_elements, max(p.numel() for p in params if p.dtype == dtype))
                               for dtype in {p.dtype for p in params}}
        # Metadata inspection does not initialize CUDA in a CPU-only caller.
        self.pin_memory = any(p.device.type == "cuda" for p in params) if pin_memory is None else bool(pin_memory)
        self._staging = {}
        self._gradient = torch.empty(self.chunk_elements, dtype=torch.float32, device="cpu")
        self.last_step = None
        self.transferred_bytes = {"initial_device_to_host": 0, "gradient_device_to_host": 0,
                                  "weight_host_to_device": 0, "checkpoint_weight_host_to_device": 0}
        for p in params:
            master = torch.empty(p.shape, device="cpu", dtype=torch.float32)
            for start, end in self._chunks(p):
                stage = self._buffer(p.dtype)[:end - start]
                stage.copy_(p.detach().view(-1)[start:end], non_blocking=False)
                master.view(-1)[start:end].copy_(stage)
                if p.device.type == "cuda":
                    self.transferred_bytes["initial_device_to_host"] += (end - start) * p.element_size()
            self.state[p] = {"master": master, "exp_avg": torch.zeros_like(master),
                             "exp_avg_sq": torch.zeros_like(master), "step": 0}

    @staticmethod
    def _validate_hyperparameters(group):
        if (not all(math.isfinite(float(group[k])) for k in ("lr", "eps", "weight_decay")) or
                group["lr"] < 0 or group["eps"] <= 0 or group["weight_decay"] < 0 or
                len(group["betas"]) != 2 or not all(0 <= b < 1 for b in group["betas"])):
            raise ValueError("Invalid AdamW hyperparameters")

    def _group(self):
        if len(self.param_groups) != 1 or tuple(map(id, self.param_groups[0]["params"])) != tuple(map(id, self._parameters)):
            raise ValueError("Parameter-group membership cannot change")
        group = self.param_groups[0]
        self._validate_hyperparameters(group)
        return group

    def _buffer(self, dtype):
        if dtype not in self._staging:
            self._staging[dtype] = torch.empty(self._dtype_elements[dtype], dtype=dtype, device="cpu", pin_memory=self.pin_memory)
        return self._staging[dtype]

    def _chunks(self, parameter):
        for start in range(0, parameter.numel(), self.chunk_elements):
            yield start, min(start + self.chunk_elements, parameter.numel())

    @torch.no_grad()
    def gradient_norm(self, grad_scale=1.):
        """FP32 chunk reductions, one final synchronization per gradient device."""
        if not math.isfinite(grad_scale) or grad_scale <= 0:
            raise ValueError("Positive finite grad_scale required")
        totals = {}
        for p in self._parameters:
            if p.grad is None:
                continue
            if p.grad.is_sparse or not p.grad.is_contiguous():
                raise ValueError("Dense contiguous gradients required")
            device = p.grad.device
            if device not in totals:
                totals[device] = torch.zeros((), device=device, dtype=torch.float32)
            flat = p.grad.detach().view(-1)
            for start, end in self._chunks(p):
                chunk_norm = torch.linalg.vector_norm(flat[start:end].float()) / grad_scale
                totals[device].add_(chunk_norm.square())
        squared = sum(float(value) for value in totals.values())
        if not math.isfinite(squared):
            raise FloatingPointError("Nonfinite gradient norm; no optimizer state was updated")
        return math.sqrt(squared)

    @torch.no_grad()
    def step(self, closure=None, *, max_grad_norm=None, grad_scale=1.):
        if closure is not None:
            raise ValueError("Closures are not supported")
        group = self._group()
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("Positive finite max_grad_norm required")
        norm = self.gradient_norm(grad_scale)
        clip = min(1., max_grad_norm / (norm + 1e-6)) if max_grad_norm is not None else 1.
        beta1, beta2 = group["betas"]
        updated = chunks = d2h = h2d = 0
        for p in self._parameters:
            if p.grad is None:
                continue  # Match AdamW: no step or weight decay for absent gradients.
            state = self.state[p]; state["step"] += 1
            step = state["step"]
            size = group["lr"] / (1 - beta1**step)
            correction = math.sqrt(1 - beta2**step)
            master, mean, square = (state[k].view(-1) for k in ("master", "exp_avg", "exp_avg_sq"))
            for start, end in self._chunks(p):
                n = end - start
                stage = self._buffer(p.dtype)[:n]
                stage.copy_(p.grad.detach().view(-1)[start:end], non_blocking=False)
                gradient = self._gradient[:n]
                gradient.copy_(stage).mul_(clip / grad_scale)
                weight, m, v = master[start:end], mean[start:end], square[start:end]
                weight.mul_(1 - group["lr"] * group["weight_decay"])
                m.lerp_(gradient, 1 - beta1)
                v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = v.sqrt().div_(correction).add_(group["eps"])
                weight.addcdiv_(m, denominator, value=-size)
                del denominator
                # CPU rounding before transfer avoids a full-size GPU FP32 temporary.
                stage.copy_(weight)
                p.view(-1)[start:end].copy_(stage, non_blocking=False)
                if p.device.type == "cuda":
                    d2h += n * p.element_size(); h2d += n * p.element_size()
                chunks += 1
            updated += p.numel()
        self.transferred_bytes["gradient_device_to_host"] += d2h
        self.transferred_bytes["weight_host_to_device"] += h2d
        self.last_step = {"unclipped_gradient_norm": norm, "clip_coefficient": clip, "grad_scale": grad_scale,
                          "updated_scalars": updated, "chunks": chunks, "device_to_host_bytes": d2h,
                          "host_to_device_bytes": h2d, **self.memory_stats()}
        return self.last_step

    def memory_stats(self):
        return {"cpu_state_bytes": sum(s[k].numel() * s[k].element_size() for s in self.state.values()
                                       for k in ("master", "exp_avg", "exp_avg_sq")),
                "retained_cpu_work_bytes": self._gradient.numel() * 4 + sum(t.numel() * t.element_size() for t in self._staging.values()),
                "maximum_denominator_bytes": self.chunk_bytes,
                "maximum_checkpoint_payload_bytes": 3 * self.chunk_bytes,
                "chunk_bytes": self.chunk_bytes}

    def state_dict(self):
        raise RuntimeError("Use save_checkpoint(directory): a monolithic state would defeat bounded resume")

    def load_state_dict(self, state_dict):
        raise RuntimeError("Use load_checkpoint(directory) to preserve FP32 CPU masters")

    @staticmethod
    def _sha(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @torch.no_grad()
    def save_checkpoint(self, directory):
        """Append-only chunk files; final manifest marks completion. No model copy."""
        group, directory = self._group(), Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        rows = []
        for index, p in enumerate(self._parameters):
            state = self.state[p]
            row = {"shape": list(p.shape), "dtype": str(p.dtype), "step": state["step"], "chunks": []}
            for number, (start, end) in enumerate(self._chunks(p)):
                path = directory / f"parameter_{index:04d}_chunk_{number:04d}.pt"
                # Clones matter: torch.save(view) otherwise serializes its whole storage.
                payload = {k: state[k].view(-1)[start:end].clone() for k in ("master", "exp_avg", "exp_avg_sq")}
                torch.save(payload, path)
                del payload
                row["chunks"].append({"file": path.name, "start": start, "end": end, "sha256": self._sha(path)})
            rows.append(row)
        value = {"schema": self.SCHEMA, "status": "completed", "chunk_elements": self.chunk_elements,
                 "parameters": rows, "hyperparameters": {k: group[k] for k in ("lr", "betas", "eps", "weight_decay")}}
        (directory / "manifest.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        return {"directory": str(directory), "manifest_sha256": self._sha(directory / "manifest.json"),
                "cpu_state_bytes": self.memory_stats()["cpu_state_bytes"]}

    @torch.no_grad()
    def load_checkpoint(self, directory):
        """Restore one bounded CPU chunk at a time; discard the instance on error."""
        group, directory = self._group(), Path(directory)
        value = json.loads((directory / "manifest.json").read_text())
        if value["schema"] != self.SCHEMA or value["status"] != "completed" or len(value["parameters"]) != len(self._parameters):
            raise ValueError("Invalid optimizer checkpoint")
        self._validate_hyperparameters(value["hyperparameters"])
        for p, row in zip(self._parameters, value["parameters"]):
            if list(p.shape) != row["shape"] or str(p.dtype) != row["dtype"] or not isinstance(row["step"], int) or row["step"] < 0:
                raise ValueError("Optimizer parameter identity/shape/dtype mismatch")
            cursor = 0
            for chunk in row["chunks"]:
                if chunk["start"] != cursor or not cursor < chunk["end"] <= p.numel() or chunk["end"] - cursor > self.chunk_elements:
                    raise ValueError("Checkpoint chunks exceed staging capacity or do not cover the parameter")
                cursor = chunk["end"]
            if cursor != p.numel():
                raise ValueError("Incomplete optimizer parameter")
        for p, row in zip(self._parameters, value["parameters"]):
            state = self.state[p]
            for chunk in row["chunks"]:
                path = directory / chunk["file"]
                if path.parent != directory or self._sha(path) != chunk["sha256"]:
                    raise ValueError("Changed optimizer checkpoint chunk")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                start, end = chunk["start"], chunk["end"]
                if set(payload) != {"master", "exp_avg", "exp_avg_sq"} or any(
                    t.dtype != torch.float32 or t.device.type != "cpu" or t.shape != (end - start,) for t in payload.values()):
                    raise ValueError("Optimizer masters and moments must remain FP32 CPU chunks")
                for key in payload:
                    state[key].view(-1)[start:end].copy_(payload[key])
                stage = self._buffer(p.dtype)[:end - start]
                stage.copy_(payload["master"])
                p.view(-1)[start:end].copy_(stage, non_blocking=False)
                if p.device.type == "cuda":
                    self.transferred_bytes["checkpoint_weight_host_to_device"] += (end - start) * p.element_size()
                del payload
            state["step"] = row["step"]
        group.update(value["hyperparameters"]); group["betas"] = tuple(group["betas"])
        return {"restored": True, "parameters": len(self._parameters), **self.memory_stats()}
