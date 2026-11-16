"""Dense response distillation adapters for an already compiled, all-block Qwen.

This is separate from the frozen sparse-position experiments. It does not load
or recompile their artifacts, acquire leases, or infer a scientific success.
Construct the matching stack once, strictly load its authenticated checkpoint,
then pass these callbacks to NAL's DistillationTrainer. GPU ownership must be
established before constructing/moving a model; ``guard`` rechecks that lease.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from typing import Callable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_qwen_compact_stack import CompactQwenStack


def _tokens(batch: Mapping, device, *, max_length: int):
    ids = batch["input_ids"]
    mask = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    if ids.ndim != 2 or ids.dtype != torch.long or not ids.numel():
        raise ValueError("input_ids must be a nonempty [batch, length] int64 tensor")
    if mask.shape != ids.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must be binary and match input_ids")
    if ids.shape[1] > max_length:
        raise ValueError("Batch exceeds the validated positional envelope; no truncation")
    valid = mask.bool()
    if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("Only right-padded, nonempty independent sequences are supported")
    # No concatenated-document/packed support is implied by a 2-D mask.
    if "segment_ids" in batch:
        raise ValueError("Packed-document attention is not implemented by this adapter")
    return ids.to(device), valid.to(device)


class DenseCompactQwen(nn.Module):
    """Own a complete compact stack and expose bounded hidden/readout callbacks.

    Frozen embedding/readout is stored once. Norm and compiled block parameters
    remain trainable; geometry is inherited unchanged from the supplied stack.
    Normal ``nn.Module`` state_dict/strict reload retains all frozen buffers.
    No cache is used for the initial bounded student-rollout implementation.
    """

    def __init__(self, stack: CompactQwenStack, norm: nn.Module,
                 embedding: torch.Tensor, rotary: nn.Module, *,
                 max_length: int, expected_layers: int, guard: Callable | None = None):
        super().__init__()
        if stack.layer_indices != tuple(range(expected_layers)):
            raise ValueError("DenseCompactQwen requires every layer starting at zero")
        if stack.packed or stack.stripped:
            raise ValueError("Recovery requires the unpacked, unstripped training stack")
        if embedding.ndim != 2 or embedding.shape[1] != stack.width:
            raise ValueError("Tied readout width does not match the compact stack")
        if max_length <= 0:
            raise ValueError("max_length must be a positive validated limit")
        if any(b.position_count > 1 and max_length > b.position_count for b in stack.blocks):
            raise ValueError("max_length exceeds a fitted position-anchor table")
        if any(b.position_policy != "error" for b in stack.blocks):
            raise ValueError("Dense recovery forbids silently clamped position anchors")
        self.stack, self.norm, self.rotary = stack, norm, rotary
        self.register_buffer("embedding", embedding.detach())
        self.max_length, self.expected_layers, self.guard = max_length, expected_layers, guard
        self.rotary.requires_grad_(False)
        self._guard_device()

    def _guard_device(self):
        devices = {t.device for t in (*self.parameters(), *self.buffers())}
        if len(devices) != 1:
            raise ValueError("Compact model tensors must share one device")
        device = next(iter(devices))
        if device.type == "cuda":
            if self.guard is None:
                raise RuntimeError("CUDA dense recovery requires an active reservation guard")
            self.guard()
        return device

    def get_extra_state(self):
        return {"version": 1, "max_length": self.max_length,
                "expected_layers": self.expected_layers, "tied_readout": True}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Dense adapter reload requires the same model and context contract")

    def hidden_forward(self, batch: Mapping):
        device = self._guard_device()
        ids, valid = _tokens(batch, device, max_length=self.max_length)
        hidden = F.embedding(ids, self.embedding)
        positions = torch.arange(ids.shape[1], device=device)[None].expand_as(ids)
        # Explicit additive causal + right-padding mask, shared by every layer.
        length = ids.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        attention = torch.zeros(allowed.shape, device=device, dtype=hidden.dtype)
        attention.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        output = self.stack(hidden, attention_mask=attention,
            position_ids=positions, position_embeddings=self.rotary(hidden, positions),
            use_cache=False)[0]
        return self.norm(output)

    def readout(self, hidden_rows: torch.Tensor):
        self._guard_device()
        if hidden_rows.ndim != 2 or hidden_rows.shape[-1] != self.embedding.shape[1]:
            raise ValueError("readout expects selected [token, hidden] rows")
        return F.linear(hidden_rows.to(self.embedding.dtype), self.embedding)

    @torch.no_grad()
    def next_logits(self, input_ids: torch.Tensor):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Cache-free rollout accepts one prefix at a time")
        was_training = self.training
        try:
            self.eval()
            hidden = self.hidden_forward({"input_ids": input_ids})
            return self.readout(hidden[:, -1])[0]
        finally:
            self.train(was_training)


class HFTeacherTargets:
    """Frozen original HF Qwen backbone targets with one-batch CPU hidden cache.

    ``prepare`` MUST run before the student's activation graph is created. A
    caller can supply ``teacher_context()`` which activates the reserved teacher
    and offloads it on exit. Hidden states remain on CPU. The default readout
    is CPU; an authenticated, frozen shared readout can use the student's GPU
    without retaining a second embedding or the full teacher there. This is a correctness adapter; its
    disk-backed/offload performance must be separately piloted.
    """

    def __init__(self, teacher: nn.Module, *, max_length: int, teacher_id: str,
                 temperature: float = 1.,
                 teacher_context: Callable | None = None, guard: Callable | None = None,
                 shared_readout: torch.Tensor | None = None):
        if not teacher_id or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Authenticated teacher identity and positive temperature required")
        self.teacher = teacher.eval().requires_grad_(False)
        self.max_length = max_length
        self.teacher_id, self.temperature = teacher_id, temperature
        self.teacher_context = teacher_context or nullcontext
        self.guard = guard
        if shared_readout is not None:
            if shared_readout.device.type == 'cuda':
                if guard is None:
                    raise RuntimeError("CUDA shared readout requires a reservation guard")
                guard()
            reference = teacher.get_output_embeddings().weight
            if (shared_readout.requires_grad or shared_readout.dtype != reference.dtype
                    or not torch.equal(shared_readout.detach().cpu(), reference.detach().cpu())):
                raise ValueError("Shared readout must exactly match the frozen original teacher")
        self.shared_readout = shared_readout
        self._key = self._hidden = self._readout = None
        self.capture_count = 0

    def clear_cache(self):
        self._key = self._hidden = self._readout = None

    def _identity(self, batch):
        ids, valid = _tokens(batch, "cpu", max_length=self.max_length)
        digest = hashlib.sha256()
        digest.update(str(tuple(ids.shape)).encode())
        digest.update(ids.contiguous().numpy().tobytes())
        digest.update(valid.contiguous().numpy().tobytes())
        return digest.hexdigest(), ids, valid

    @torch.no_grad()
    def prepare(self, batch):
        key, ids, valid = self._identity(batch)
        if self._key == key:
            return
        # Invalidate before execution so a failed capture cannot reuse stale data.
        self._key = self._hidden = self._readout = None
        with self.teacher_context():
            device = self.teacher.get_input_embeddings().weight.device
            if device.type == "cuda":
                if self.guard is None:
                    raise RuntimeError("CUDA teacher capture requires a reservation guard")
                self.guard()
            self.teacher.eval()
            output = self.teacher.model(input_ids=ids.to(device),
                attention_mask=valid.to(device), use_cache=False, return_dict=True)
            self._hidden = output.last_hidden_state.detach().cpu()
            self._readout = (self.shared_readout if self.shared_readout is not None else
                             self.teacher.get_output_embeddings().weight.detach().cpu())
        self._key = key
        self.capture_count += 1

    @torch.no_grad()
    def __call__(self, batch, positions: torch.Tensor):
        key, ids, valid = self._identity(batch)
        if key != self._key or self._hidden is None:
            raise ValueError("Teacher targets must be prepared for these exact input prefixes")
        pos = positions.detach().cpu()
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.dtype != torch.long:
            raise ValueError("positions must contain int64 [row, predicting-position] pairs")
        if (pos < 0).any() or (pos[:, 0] >= ids.shape[0]).any() or (pos[:, 1] >= ids.shape[1]).any():
            raise ValueError("Teacher target position out of range")
        if not valid[pos[:, 0], pos[:, 1]].all():
            raise ValueError("Teacher targets cannot select padding")
        if self._readout.device.type == 'cuda':
            self.guard()
        hidden = self._hidden[pos[:, 0], pos[:, 1]].to(self._readout)
        return F.linear(hidden, self._readout)
