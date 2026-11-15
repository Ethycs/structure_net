"""Experimental resident FP32 Adam state with bounded CPU spill and host bounce.

Model gradients remain on their original device. Global clipping is the frozen
StreamingCPUAdamW reduction; only a deterministic subset of optimizer states
moves to a spare device. GPU arithmetic is not asserted bit-identical to CPU.
The caller owns reservation admission, allocator caps and experiment selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .jspace_streaming_adamw import StreamingCPUAdamW

STATE_KEYS = ("master", "exp_avg", "exp_avg_sq")


def placement_plan(parameters, chunk_elements, resident_budget_bytes):
    """First-fit whole parameters, reserving all persistent and peak work bytes.

    Parameter placement is fixed for the entire run. Whole-parameter placement
    keeps per-parameter Adam step counters and inspectable state tensors intact.
    Work reserves one FP32 gradient, one FP32 denominator and one transport
    buffer in each model dtype. No full parameter gradient copy is retained.
    """
    parameters = tuple(parameters)
    if (type(chunk_elements) is not int or chunk_elements < 1 or
            type(resident_budget_bytes) is not int or resident_budget_bytes < 0):
        raise ValueError("Nonnegative integer budget and positive chunk size required")
    work = chunk_elements * 8 + sum(
        min(chunk_elements, max(p.numel() for p in parameters if p.dtype == dtype)) *
        next(p.element_size() for p in parameters if p.dtype == dtype)
        for dtype in {p.dtype for p in parameters})
    remaining = max(0, resident_budget_bytes - work)
    resident = []
    for p in parameters:
        size = p.numel() * 12
        chosen = size <= remaining and resident_budget_bytes >= work
        resident.append(chosen)
        if chosen:
            remaining -= size
    state = sum(p.numel() * 12 for p, chosen in zip(parameters, resident) if chosen)
    return dict(resident=resident, resident_state_bytes=state,
                resident_work_peak_bytes=work if any(resident) else 0,
                resident_budget_bytes=resident_budget_bytes,
                resident_planned_peak_bytes=state + (work if any(resident) else 0),
                cpu_state_bytes=sum(p.numel()*12 for p, chosen in zip(parameters, resident) if not chosen),
                policy="deterministic_parameter_order_first_fit")


class ResidentSpillAdamW(StreamingCPUAdamW):
    """Adopt an existing CPU optimizer without changing its masters or steps.

    CPU ``resident_device`` is a test backend, not an acceleration claim. For
    CUDA, a live caller-supplied guard is mandatory and is called before the
    first allocation and every update/checkpoint. This helper never assumes P2P.
    ``from_cpu_optimizer`` consumes the source optimizer; do not reuse it.
    """
    SCHEMA = "jspace.resident-spill-adamw.v1"

    @classmethod
    @torch.no_grad()
    def from_cpu_optimizer(cls, source, *, resident_device, resident_budget_bytes,
                           guard=None, pin_memory=None):
        if type(source) is not StreamingCPUAdamW:
            raise ValueError("Adoption requires the frozen StreamingCPUAdamW implementation")
        group = source._group()
        parameters = tuple(source._parameters)
        if not parameters:
            raise ValueError("Source optimizer was already consumed")
        device = torch.device(resident_device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU test or CUDA resident state is supported")
        if device.type == "cuda" or any(p.device.type == "cuda" for p in parameters):
            if guard is None:
                raise ValueError("CUDA optimizer requires an active reservation guard")
            guard()  # Before CUDA allocation, including pinned staging allocation.
        for p in parameters:
            state = source.state[p]
            if (type(state["step"]) is not int or state["step"] < 0 or
                    any(state[k].device.type != "cpu" or state[k].dtype != torch.float32 or
                        state[k].shape != p.shape for k in STATE_KEYS)):
                raise ValueError("Source state must contain CPU FP32 masters and moments")
        plan = placement_plan(parameters, source.chunk_elements, resident_budget_bytes)
        self = cls.__new__(cls)
        defaults = {k: group[k] for k in ("lr", "betas", "eps", "weight_decay")}
        torch.optim.Optimizer.__init__(self, parameters, defaults)
        self._parameters = parameters
        self.chunk_elements, self.chunk_bytes = source.chunk_elements, source.chunk_bytes
        self._dtype_elements = dict(source._dtype_elements)
        self.resident_device, self.guard, self.placement = device, guard, plan
        self.pin_memory = source.pin_memory if pin_memory is None else bool(pin_memory)
        if device.type == "cuda" and not self.pin_memory:
            raise ValueError("CUDA resident transport requires pinned host bounce buffers")
        self._staging = source._staging if self.pin_memory == source.pin_memory else {}
        self._gradient = source._gradient
        self._resident_staging = {}
        self._resident_gradient = None
        if any(plan["resident"]):
            self._resident_gradient = torch.empty(self.chunk_elements, dtype=torch.float32, device=device)
            self._resident_staging = {
                dtype: torch.empty(count, dtype=dtype, device=device)
                for dtype, count in self._dtype_elements.items()}
        self.last_step = None
        self.transferred_bytes = dict(source.transferred_bytes)
        self.transferred_bytes.update(resident_initial_host_to_device=0,
            resident_gradient_host_to_device=0, resident_weight_device_to_host=0,
            explicit_state_device_to_host=0, resident_checkpoint_host_to_device=0)
        for p, chosen in zip(parameters, plan["resident"]):
            old = source.state.pop(p)
            if chosen and device.type == "cuda":
                self.state[p] = {k: old[k].to(device=device) for k in STATE_KEYS}
                self.state[p]["step"] = old["step"]
                self.transferred_bytes["resident_initial_host_to_device"] += p.numel()*12
            else:
                self.state[p] = old
        # Explicit transfer of ownership: retaining a second functional optimizer
        # would allow stale CPU masters to overwrite the model later.
        source._parameters = ()
        source.param_groups.clear()
        source._staging = {}
        source._gradient = torch.empty(0)
        return self

    def _guard(self):
        if self.guard is not None:
            self.guard()

    @torch.no_grad()
    def step(self, closure=None, *, max_grad_norm=None, grad_scale=1.):
        if closure is not None:
            raise ValueError("Closures are not supported")
        self._guard()
        group = self._group()
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("Positive finite max_grad_norm required")
        started = time.perf_counter()
        norm = self.gradient_norm(grad_scale)  # Unmodified model-device reduction/order.
        norm_seconds = time.perf_counter() - started
        clip = min(1., max_grad_norm/(norm+1e-6)) if max_grad_norm is not None else 1.
        beta1, beta2 = group["betas"]
        updated = chunks = d2h = h2d = spare_h2d = spare_d2h = 0
        resident_chunks = spill_chunks = 0
        for p, resident in zip(self._parameters, self.placement["resident"]):
            if p.grad is None:
                continue
            state = self.state[p]
            state["step"] += 1
            size = group["lr"]/(1-beta1**state["step"])
            correction = math.sqrt(1-beta2**state["step"])
            master, mean, square = (state[k].view(-1) for k in STATE_KEYS)
            for start, end in self._chunks(p):
                n = end-start
                stage = self._buffer(p.dtype)[:n]
                stage.copy_(p.grad.detach().view(-1)[start:end], non_blocking=False)
                if resident:
                    transport = self._resident_staging[p.dtype][:n]
                    transport.copy_(stage, non_blocking=False)
                    gradient = self._resident_gradient[:n]
                    gradient.copy_(transport).mul_(clip/grad_scale)
                    resident_chunks += 1
                else:
                    gradient = self._gradient[:n]
                    gradient.copy_(stage).mul_(clip/grad_scale)
                    spill_chunks += 1
                weight, m, v = master[start:end], mean[start:end], square[start:end]
                # Preserve the reference ATen sequence and Python scalar values.
                weight.mul_(1-group["lr"]*group["weight_decay"])
                m.lerp_(gradient, 1-beta1)
                v.mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
                denominator = v.sqrt().div_(correction).add_(group["eps"])
                weight.addcdiv_(m, denominator, value=-size)
                del denominator
                if resident:
                    transport.copy_(weight)
                    stage.copy_(transport, non_blocking=False)
                    if self.resident_device.type == "cuda":
                        spare_h2d += n*p.element_size()
                        spare_d2h += n*p.element_size()
                else:
                    stage.copy_(weight)
                p.view(-1)[start:end].copy_(stage, non_blocking=False)
                if p.device.type == "cuda":
                    d2h += n*p.element_size()
                    h2d += n*p.element_size()
                chunks += 1
            updated += p.numel()
        elapsed = time.perf_counter()-started
        for key, amount in (("gradient_device_to_host", d2h), ("weight_host_to_device", h2d),
                            ("resident_gradient_host_to_device", spare_h2d),
                            ("resident_weight_device_to_host", spare_d2h)):
            self.transferred_bytes[key] += amount
        self.last_step = dict(unclipped_gradient_norm=norm, clip_coefficient=clip, grad_scale=grad_scale,
            updated_scalars=updated, chunks=chunks, resident_chunks=resident_chunks, spill_chunks=spill_chunks,
            device_to_host_bytes=d2h, host_to_device_bytes=h2d,
            resident_gradient_host_to_device_bytes=spare_h2d, resident_weight_device_to_host_bytes=spare_d2h,
            optimizer_seconds=elapsed, gradient_norm_seconds=norm_seconds,
            timing_scope="Global clipping, bounded CPU/spare-device updates and synchronous model copies; excludes lease checks",
            **self.memory_stats())
        self._guard()
        return self.last_step

    def memory_stats(self):
        return {**self.placement,
            "retained_cpu_work_bytes": self._gradient.numel()*4 + sum(t.numel()*t.element_size() for t in self._staging.values()),
            "maximum_denominator_bytes": self.chunk_bytes,
            "maximum_checkpoint_payload_bytes": 3*self.chunk_bytes,
            "chunk_bytes": self.chunk_bytes,
            "state_arithmetic": "FP32; CPU and CUDA rounding may differ",
            "resident_backend": str(self.resident_device)}

    @torch.no_grad()
    def _cpu_chunk(self, state, start, end):
        payload = {k: state[k].view(-1)[start:end].to(device="cpu", copy=True) for k in STATE_KEYS}
        if state["master"].device.type == "cuda":
            self.transferred_bytes["explicit_state_device_to_host"] += 12*(end-start)
        return payload

    @torch.no_grad()
    def fingerprint(self):
        """Explicit bounded state download, intended outside measured steps."""
        self._guard()
        digest = hashlib.sha256()
        for index, p in enumerate(self._parameters):
            state = self.state[p]
            digest.update(json.dumps([index, list(p.shape), str(p.dtype), state["step"]]).encode())
            for start, end in self._chunks(p):
                payload = self._cpu_chunk(state, start, end)
                for k in STATE_KEYS:
                    digest.update(payload[k].view(torch.uint8).numpy().tobytes())
                del payload
        return dict(sha256=digest.hexdigest(), scope="masters,moments,steps; explicit untimed bounded CPU download")

    @torch.no_grad()
    def save_checkpoint(self, directory):
        self._guard()
        group, directory = self._group(), Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        rows = []
        for index, p in enumerate(self._parameters):
            state = self.state[p]
            row = dict(shape=list(p.shape), dtype=str(p.dtype), step=state["step"], chunks=[])
            for number, (start, end) in enumerate(self._chunks(p)):
                path = directory/f"parameter_{index:04d}_chunk_{number:04d}.pt"
                payload = self._cpu_chunk(state, start, end)
                torch.save(payload, path)
                del payload
                row["chunks"].append(dict(file=path.name, start=start, end=end, sha256=self._sha(path)))
            rows.append(row)
        value = dict(schema=self.SCHEMA, status="completed", chunk_elements=self.chunk_elements,
            parameters=rows, placement=self.placement,
            hyperparameters={k: group[k] for k in ("lr", "betas", "eps", "weight_decay")},
            scope="FP32 CPU checkpoint chunks; resident downloads outside optimizer steps")
        (directory/"manifest.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
        return dict(directory=str(directory), manifest_sha256=self._sha(directory/"manifest.json"))

    @torch.no_grad()
    def load_checkpoint(self, directory):
        self._guard()
        # The inherited streaming restore writes CPU payloads into whichever
        # device owns each state tensor, and returns model weights via CPU bounce.
        result = super().load_checkpoint(directory)
        if self.resident_device.type == "cuda":
            self.transferred_bytes["resident_checkpoint_host_to_device"] += self.placement["resident_state_bytes"]
        self._guard()
        return result
