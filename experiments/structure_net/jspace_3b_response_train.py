"""Complete-response Qwen recovery with bounded CPU FP32 optimizer state.

This producer is separate from the immutable sparse-position studies. It reuses
NAL's exact token-chunked KL and complete-response validation, and the tested
CPU-gradient Adam implementation. The caller authenticates the initializer,
corpus and teacher/prefix caches and acquires a reservation before allocation.
No TEST data, checkpoint selection or implicit model loading occurs here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Callable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from experiments.neural_architecture_lab.distillation_recovery import module
from . import jspace_qwen3_train as offload
from .jspace_3b_dense_adapter import _tokens
from .jspace_qwen_compact_stack import CompactQwenStack

nal = module("distillation_training")


@dataclass(frozen=True)
class ResponseConfig:
    learning_rate: float = 5e-6
    accumulation_examples: int = 16
    token_chunk_size: int = 8
    max_supervised_tokens: int = 100_000
    gradient_clip: float = 1.
    optimizer_chunk_bytes: int = 4 * 1024**2
    max_cpu_state_bytes: int = 32 * 1024**3
    max_length: int = 416
    checkpoint_blocks: bool = True

    def __post_init__(self):
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0
                or any(type(v) is not int or v < 1 for v in (
                    self.accumulation_examples, self.token_chunk_size,
                    self.max_supervised_tokens, self.optimizer_chunk_bytes,
                    self.max_cpu_state_bytes, self.max_length))):
            raise ValueError("Positive finite training controls required")

    def loss_config(self):
        return nal.DistillationConfig(objective="forward_kl", temperature=1.,
            kd_weight=1., ce_weight=0., token_chunk_size=self.token_chunk_size,
            accumulation_batches=self.accumulation_examples,
            max_supervised_tokens=self.max_supervised_tokens,
            grad_clip_norm=self.gradient_clip)


class ResponseModel(nn.Module):
    """Train a compact suffix; execute the real native prefix for generation.

    ``boundary_hidden`` is allowed only for teacher-forced exact-prefix caches.
    ``next_logits`` never reads a cached training boundary: a suffix requires a
    callable native_prefix(ids, valid) returning the actual original-prefix
    residual for the current generated history. A full stack embeds directly.
    """
    def __init__(self, stack, norm, embedding, rotary, *, max_length=416,
                 expected_layers=36, guard=None, native_prefix: Callable | None = None):
        super().__init__()
        if (not isinstance(stack, CompactQwenStack) or stack.packed or stack.stripped
                or not stack.layer_indices
                or stack.layer_indices != tuple(range(stack.layer_indices[0], expected_layers))):
            raise ValueError("An unpacked contiguous suffix ending at the final native layer is required")
        if embedding.ndim != 2 or embedding.shape[1] != stack.width:
            raise ValueError("Original tied embedding/readout dimensions differ")
        if any(b.position_policy != "error" or b.position_count > 1 and max_length > b.position_count
               for b in stack.blocks):
            raise ValueError("The complete response must fit the fitted position anchors; no clamping")
        self.stack, self.norm, self.rotary = stack, norm, rotary
        self.register_buffer("embedding", embedding.detach())
        self.max_length, self.expected_layers = max_length, expected_layers
        self.guard, self.native_prefix = guard, native_prefix
        self.rotary.requires_grad_(False)
        self.cut = stack.layer_indices[0]
        self._guard()

    def _guard(self):
        device = self.embedding.device
        if any(t.device != device for t in (*self.parameters(), *self.buffers())):
            raise ValueError("Student parameters, geometry and readout must share a device")
        if device.type == "cuda":
            if self.guard is None:
                raise ValueError("CUDA response execution requires an active reservation guard")
            self.guard()
        return device

    def hidden_forward(self, batch: Mapping):
        device = self._guard()
        ids, valid = _tokens(batch, device, max_length=self.max_length)
        if "boundary_hidden" in batch:
            hidden = batch["boundary_hidden"].detach().to(self.embedding)
            if hidden.shape != (*ids.shape, self.stack.width):
                raise ValueError("Cached native-prefix boundary shape differs from the full token sequence")
        elif self.cut == 0:
            hidden = F.embedding(ids, self.embedding)
        elif self.native_prefix is not None:
            with torch.no_grad():
                hidden = self.native_prefix(ids, valid).detach().to(self.embedding)
            if hidden.shape != (*ids.shape, self.stack.width):
                raise ValueError("Native prefix returned an invalid boundary")
        else:
            raise ValueError("Progressive generation needs an actual native prefix; cached TRAIN boundaries cannot substitute")
        length = ids.shape[1]
        positions = torch.arange(length, device=device)[None].expand_as(ids)
        allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()[None, None]
        allowed = allowed & valid[:, None, None, :]
        attention = torch.zeros(allowed.shape, device=device, dtype=hidden.dtype)
        attention.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        output = self.stack(hidden, attention_mask=attention,
            position_ids=positions, position_embeddings=self.rotary(hidden, positions), use_cache=False)[0]
        return self.norm(output)

    def readout(self, hidden_rows):
        self._guard()
        if hidden_rows.ndim != 2 or hidden_rows.shape[-1] != self.stack.width:
            raise ValueError("Readout requires [selected tokens, native hidden width]")
        return F.linear(hidden_rows.to(self.embedding.dtype), self.embedding)

    @torch.no_grad()
    def next_logits(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Generation requires one unpadded prefix")
        previous = self.training
        try:
            self.eval()
            return self.readout(self.hidden_forward({"input_ids": input_ids})[:, -1])[0]
        finally:
            self.train(previous)


class CachedHiddenTeacher:
    """Read exact original-teacher final-normalized hidden rows from the batch.

    The dataset must bind ``teacher_hidden`` to the exact ``teacher_input_ids``
    and record identities below, after authenticating its cache manifest. The
    teacher shares the frozen original vocabulary matrix with the student.
    """
    temperature = 1.

    def __init__(self, readout, *, teacher_id, guard=None):
        if not teacher_id or readout.requires_grad:
            raise ValueError("Frozen original teacher readout and identity required")
        self.readout, self.teacher_id, self.guard = readout, teacher_id, guard
        self._batch = None

    def prepare(self, batch):
        ids, hidden = batch["input_ids"], batch["teacher_hidden"]
        if (not torch.equal(ids.cpu(), batch["teacher_input_ids"].cpu())
                or list(batch["teacher_record_ids"]) != [nal._record(r)["example_id"] for r in batch["records"]]
                or hidden.shape != (*ids.shape, self.readout.shape[1]) or hidden.requires_grad
                or not torch.isfinite(hidden).all()):
            raise ValueError("Teacher cache token identity, records, hidden shape or finiteness differs")
        self._batch = batch

    @torch.no_grad()
    def __call__(self, batch, positions):
        if batch is not self._batch:
            raise ValueError("Teacher.prepare must authenticate this exact batch before scoring")
        if self.guard is not None:
            self.guard()
        pos = positions.to(batch["teacher_hidden"].device)
        hidden = batch["teacher_hidden"][pos[:, 0], pos[:, 1]].to(self.readout)
        return F.linear(hidden, self.readout)

    def clear_cache(self):
        self._batch = None


class ResponseTrainer:
    """One-pass, token-weighted complete-response distillation on an admitted dataset."""
    def __init__(self, stack, norm, readout, rotary, dataset, *, teacher_id,
                 config=ResponseConfig(), guard=None, host_reserve_bytes=8*1024**3,
                 expected_layers=36, native_prefix=None):
        self.config, self.dataset, self.guard = config, dataset, guard
        self.model = ResponseModel(stack, norm, readout, rotary, max_length=config.max_length,
            expected_layers=expected_layers, guard=guard, native_prefix=native_prefix)
        self.stack, self.norm, self.readout, self.rotary = stack, norm, readout, rotary
        self.device, self.dtype = readout.device, readout.dtype
        stack.requires_grad_(True); norm.requires_grad_(True)
        stack.checkpoint_blocks = config.checkpoint_blocks
        self.parameters = [*stack.parameters(), *norm.parameters()]
        if self.dtype != torch.bfloat16 or any(p.dtype != self.dtype for p in self.parameters):
            raise ValueError("This qualified recovery profile requires BF16 learned parameters and readout")
        self.parameter_ids = tuple(map(id, self.parameters))
        count = sum(p.numel() for p in self.parameters)
        model_bytes = sum(t.numel()*t.element_size() for m in (stack, norm)
                          for t in (*m.parameters(), *m.buffers()))
        self.memory = dict(cpu_master_moments_bytes=12*count, cpu_gradient_bank_bytes=2*count,
            snapshot_bytes=model_bytes, cpu_additional_bound=14*count+model_bytes+6*config.optimizer_chunk_bytes)
        self.host_admission = offload.require_host_memory(self.memory["cpu_additional_bound"],
                                                         reserve_bytes=host_reserve_bytes)
        self.optimizer = offload.CPUGradientAdamW(self.parameters, lr=config.learning_rate,
            weight_decay=0., chunk_bytes=config.optimizer_chunk_bytes,
            max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.teacher = CachedHiddenTeacher(readout, teacher_id=teacher_id, guard=guard)
        self.loss_config = config.loss_config()
        self.updates = self.supervised_tokens = self.input_tokens = 0
        self.visited = set()
        self.fixed = offload.fixed_state(stack, readout, rotary)

    def train_window(self, indices):
        indices = list(indices)
        if (not indices or len(indices) > self.config.accumulation_examples
                or len(set(indices)) != len(indices) or any(i in self.visited for i in indices)
                or any(i not in self.dataset.training for i in indices)):
            raise ValueError("A one-pass TRAIN-only window of distinct unvisited examples is required")
        if tuple(map(id, (*self.stack.parameters(), *self.norm.parameters()))) != self.parameter_ids:
            raise RuntimeError("Live trainable parameter identities changed after optimizer registration")
        # Materialize only one small accumulation window of disk-backed hidden
        # rows, never the corpus. Inputs stay on CPU until each forward.
        batches = [self.dataset.batch([i]) for i in indices]
        counts = [nal.validate_batch(batch) for batch in batches]
        total = sum(counts)
        if self.supervised_tokens + total > self.config.max_supervised_tokens:
            raise ValueError("Complete response window exceeds the frozen supervised-token budget")
        if not all(counts):
            raise ValueError("Empty response masks cannot update weights")
        self.model._guard(); self.model.train(); self.optimizer.zero_grad()
        started = time.monotonic(); values = []
        transferred_before = self.optimizer.offload_bytes
        try:
            for batch in batches:
                self.teacher.prepare(batch)
                hidden = self.model.hidden_forward(batch)
                values.append(nal.chunked_backward(hidden, self.model.readout, self.teacher,
                              batch, self.loss_config, normalizer=total))
                del hidden
                self.teacher.clear_cache()
            received = self.optimizer.received
            by_block = [all(received.get(p, 0) == len(indices) for p in b.parameters()) for b in self.stack.blocks]
            if (not all(by_block) or any(received.get(p, 0) != len(indices) for p in self.norm.parameters())
                    or any(p.grad is not None for p in self.parameters)):
                raise RuntimeError("Every compressed block and final norm must contribute every microbatch gradient")
            self.model._guard()
            update = self.optimizer.step(max_grad_norm=self.config.gradient_clip)
            self.updates += 1; self.supervised_tokens += total
            self.input_tokens += sum(int(b["attention_mask"].sum()) for b in batches)
            self.visited.update(indices)
            return dict(update=self.updates, indices=indices, cases=len(indices),
                supervised_tokens=total, cumulative_supervised_tokens=self.supervised_tokens,
                cumulative_input_tokens=self.input_tokens,
                **{name: sum(v[name] for v in values)/total for name in
                   ("loss", "distillation", "forward_kl", "ce", "correct", "teacher_agreement")},
                gradients_present_by_block=by_block, optimizer=update,
                gradient_device_to_host_bytes=self.optimizer.offload_bytes-transferred_before,
                learning_rate=self.config.learning_rate, train_window_seconds=time.monotonic()-started)
        finally:
            self.teacher.clear_cache(); self.optimizer.zero_grad()

    @torch.inference_mode()
    def development(self, indices=None):
        indices = list(self.dataset.development if indices is None else indices)
        if not indices or any(i not in self.dataset.development for i in indices):
            raise ValueError("Development scoring requires nonempty admitted DEV indices")
        self.model.eval(); self.optimizer.zero_grad()
        rows = []; totals = dict(loss=0., forward_kl=0., ce=0., correct=0, teacher_agreement=0)
        tokens = 0
        for index in indices:
            batch = self.dataset.batch([index]); count = nal.validate_batch(batch, training=False)
            self.teacher.prepare(batch)
            hidden = self.model.hidden_forward(batch)
            positions = batch["loss_mask"].nonzero().cpu()
            values = dict.fromkeys(totals, 0.)
            for start in range(0, count, self.config.token_chunk_size):
                pos = positions[start:start+self.config.token_chunk_size]
                logits = self.model.readout(hidden[pos[:, 0].to(hidden.device), pos[:, 1].to(hidden.device)])
                teacher = self.teacher(batch, pos)
                targets = batch["targets"][pos[:, 0], pos[:, 1]]
                losses = nal.loss_vectors(logits, teacher, targets, self.loss_config)
                for key in ("loss", "forward_kl", "ce"):
                    values[key] += float(losses[key].sum())
                values["correct"] += int((logits.argmax(-1) == targets.to(logits.device)).sum())
                values["teacher_agreement"] += int((logits.argmax(-1) == teacher.argmax(-1)).sum())
            tokens += count
            for key in totals: totals[key] += values[key]
            rows.append(dict(index=index, example_id=nal._record(batch["records"][0])["example_id"],
                             supervised_tokens=count, **{k: v/count for k, v in values.items()}))
            del batch, hidden
            self.teacher.clear_cache()
        return dict(count=len(rows), supervised_tokens=tokens,
                    **{k: v/tokens for k, v in totals.items()},
                    scope="Teacher-forced complete-response DEV, not free-generation quality"), rows

    def parameter_fingerprints(self):
        return {name: offload.fingerprint([(name, p)])["sha256"]
                for name, p in self.model.named_parameters() if p.requires_grad}

    def change_report(self, before):
        after = self.parameter_fingerprints()
        if set(before) != set(after):
            raise ValueError("Pilot parameter registration changed")
        changed = [name for name in after if before[name] != after[name]]
        return dict(bf16_changed_parameters=changed,
                    bf16_changed_by_block=[any(n.startswith(f"stack.blocks.{i}.") for n in changed)
                                           for i in range(len(self.stack.blocks))],
                    norm_bf16_storage_changed=any(n.startswith("norm.") for n in changed))

    def snapshot(self):
        self.model._guard()
        offload.require_host_memory(self.memory["snapshot_bytes"], reserve_bytes=1024**3)
        return dict(stack=offload.recursive_cpu(self.stack.state_dict()),
                    norm=offload.recursive_cpu(self.norm.state_dict()))

    def save_checkpoint(self, path, *, identities, save_optimizer=False):
        path = Path(path)
        if offload.fixed_state(self.stack, self.readout, self.rotary) != self.fixed:
            raise ValueError("Frozen geometry, original readout or rotary state changed")
        with path.open("xb") as stream:
            torch.save(dict(schema="jspace.complete-response-checkpoint.v1", **self.snapshot(),
                identities=dict(identities), config=asdict(self.config), updates=self.updates,
                supervised_tokens=self.supervised_tokens, input_tokens=self.input_tokens,
                visited=sorted(self.visited), fixed_state=self.fixed), stream)
        optimizer = self.optimizer.save_checkpoint(path.with_suffix(".optimizer")) if save_optimizer else None
        return dict(path=str(path.resolve()), sha256=offload.sha(path), optimizer=optimizer,
                    resume_requires_optimizer=not save_optimizer)

    def close(self):
        self.teacher.clear_cache(); self.optimizer.close()
