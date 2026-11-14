"""Joint output-distillation recovery of a compact Qwen suffix from disk caches.

No prefix or teacher model is retained here. The caller owns frozen cache
provenance, reservations, native/accepted teacher choice, and the final held-out
evaluation. Development selects packed CE; original-teacher drift is diagnostic.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .jspace_qwen_compact_stack import CompactQwenStack
from .jspace_streaming_adamw import StreamingCPUAdamW


@dataclass(frozen=True)
class RecoveryConfig:
    epochs: int = 12
    microbatch: int = 4
    accumulation: int = 4
    learning_rate: float = 1.2e-4
    temperature: float = 2.
    kl_weight: float = .7
    ce_weight: float = .3
    gradient_clip: float = 1.
    checkpoint_blocks: bool = True
    dtype: str = "bfloat16"
    seed: int = 861
    optimizer_chunk_bytes: int = 32 * 1024**2
    max_cpu_state_bytes: int | None = None

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
               (self.epochs, self.microbatch, self.accumulation, self.optimizer_chunk_bytes)):
            raise ValueError("Positive epoch, batch, accumulation and chunk counts required")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Explicit FP32/BF16/FP16 execution dtype required")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.learning_rate, self.temperature, self.gradient_clip)):
            raise ValueError("Finite positive learning rate, temperature and clipping required")
        if (self.temperature, self.kl_weight, self.ce_weight) != (2., .7, .3):
            raise ValueError("This recipe fixes output loss to .7*T²*KL + .3*CE with T=2")


def recursive_cpu(value):
    """Own CPU tensor storage while retaining recursive non-tensor extra_state."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: recursive_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(recursive_cpu(item) for item in value)
    if isinstance(value, list):
        return [recursive_cpu(item) for item in value]
    return deepcopy(value)


def cpu_module_clone(module):
    """Deepcopy registered tensors directly to CPU, avoiding a GPU-side clone."""
    memo = {}
    for parameter in module.parameters():
        memo[id(parameter)] = nn.Parameter(parameter.detach().to(device="cpu", copy=True),
                                           requires_grad=parameter.requires_grad)
    for buffer in module.buffers():
        memo[id(buffer)] = buffer.detach().to(device="cpu", copy=True)
    return deepcopy(module, memo)


class BoundaryDataset:
    """Read-only, bounded batch copies of BF16-bit boundary and logit .npy files."""
    def __init__(self, items, boundary_path, accepted_logits_path, *, original_logits_path=None, original_indices=None):
        self.items = list(items)
        if not self.items or len({c["identity"] for c in self.items}) != len(self.items):
            raise ValueError("Distinct nonempty training/development cases required")
        if any(c["split"] not in ("train", "development") for c in self.items):
            raise ValueError("Held-out evaluation cases must never enter the recovery dataset")
        self.boundary = self._open(boundary_path, 2)
        self.accepted = self._open(accepted_logits_path, 3)
        self.original = self._open(original_logits_path, 3) if original_logits_path is not None else None
        self.width, self.vocabulary = self.boundary.shape[1], self.accepted.shape[-1]
        if self.accepted.shape[:2] != (len(self.items), 4):
            raise ValueError("Accepted teacher rows must align with exactly four positions per case")
        self.original_indices = None
        if self.original is not None:
            if original_indices is None:
                raise ValueError("Explicit original-teacher row mapping is required")
            self.original_indices = list(original_indices)
            if (len(self.original_indices) != len(self.items) or self.original.shape[1:] != (4, self.vocabulary) or
                    any(not isinstance(i, int) or i < 0 or i >= len(self.original) for i in self.original_indices)):
                raise ValueError("Invalid original-teacher row mapping")
        self.lengths = []
        for case in self.items:
            length = case["stop"] - case["offset"]
            if (not 0 <= case["offset"] < case["stop"] <= len(self.boundary) or
                    ("input_ids" in case and len(case["input_ids"]) != length) or
                    len(case["loss_positions"]) != 4 or len(case["targets"]) != 4 or
                    case["loss_positions"][-1] != length - 1 or
                    any(not isinstance(p, int) or not 0 <= p < length for p in case["loss_positions"]) or
                    any(not isinstance(t, int) or not 0 <= t < self.vocabulary for t in case["targets"])):
                raise ValueError("Invalid boundary offsets, loss positions or targets")
            self.lengths.append(length)
        self.training = [i for i, c in enumerate(self.items) if c["split"] == "train"]
        self.development = [i for i, c in enumerate(self.items) if c["split"] == "development"]
        if not self.training or not self.development:
            raise ValueError("Both training and development splits are required")
        self.read_bytes = {"boundary": 0, "accepted": 0, "original": 0}

    @staticmethod
    def _open(path, dimensions):
        value = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        if (not isinstance(value, np.memmap) or value.dtype != np.dtype("int16") or
                not value.dtype.isnative or value.ndim != dimensions or min(value.shape) <= 0):
            raise ValueError("Expected a nonempty read-only native-endian BF16-bit int16 .npy file")
        return value

    @staticmethod
    def _bits(array, device, dtype=torch.bfloat16):
        return torch.from_numpy(np.array(array, copy=True)).view(torch.bfloat16).to(device=device, dtype=dtype)

    def batch(self, indices, *, device="cpu", dtype=torch.bfloat16, include_original=False):
        indices = list(indices)
        if not indices or any(i < 0 or i >= len(self.items) for i in indices):
            raise ValueError("Nonempty valid batch indices required")
        lengths = [self.lengths[i] for i in indices]; length = max(lengths)
        storage = np.zeros((len(indices), length, self.width), dtype=np.int16)
        for row, index in enumerate(indices):
            case = self.items[index]
            storage[row, :lengths[row]] = self.boundary[case["offset"]:case["stop"]]
        hidden = torch.from_numpy(storage).view(torch.bfloat16).to(device=device, dtype=dtype)
        teacher = self._bits(self.accepted[indices], device)
        self.read_bytes["boundary"] += sum(lengths) * self.width * 2
        self.read_bytes["accepted"] += len(indices) * 4 * self.vocabulary * 2
        positions = torch.arange(length, device=device).expand(len(indices), -1)
        valid_keys = positions < torch.tensor(lengths, device=device)[:, None]
        causal = torch.arange(length, device=device)[None, :] <= torch.arange(length, device=device)[:, None]
        allowed = causal[None] & valid_keys[:, None, :]
        mask = torch.zeros((len(indices), 1, length, length), dtype=dtype, device=device)
        mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        result = {"indices": indices, "hidden": hidden, "teacher": teacher, "lengths": lengths,
                  "position_ids": positions, "attention_mask": mask,
                  "loss_positions": torch.tensor([self.items[i]["loss_positions"] for i in indices], device=device),
                  "targets": torch.tensor([self.items[i]["targets"] for i in indices], device=device)}
        if include_original and self.original is not None:
            result["original"] = self._bits(self.original[[self.original_indices[i] for i in indices]], device)
            self.read_bytes["original"] += len(indices) * 4 * self.vocabulary * 2
        if any(not torch.isfinite(result[k]).all() for k in ("hidden", "teacher", "original") if k in result):
            raise FloatingPointError("Nonfinite cached values or activation precision overflow")
        return result

    def close(self):
        for value in (self.boundary, self.accepted, self.original):
            if value is not None and not value._mmap.closed:
                value._mmap.close()


def output_objective(logits, teacher, targets, config):
    target_log = (teacher.float() / config.temperature).log_softmax(-1)
    student_log = (logits.float() / config.temperature).log_softmax(-1)
    kl = (target_log.exp() * (target_log - student_log)).sum(-1).mean()
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return config.kl_weight * config.temperature**2 * kl + config.ce_weight * ce, kl, ce


class SuffixTrainer:
    """Jointly train all suffix blocks; retain only E and final RMSNorm externally."""
    def __init__(self, stack, norm, readout_weight, rotary, dataset, label_ids, *, config=RecoveryConfig(), guard=None):
        if not isinstance(stack, CompactQwenStack) or stack.packed or stack.stripped:
            raise ValueError("Recovery requires an unpacked, unstripped CompactQwenStack")
        self.stack, self.norm, self.readout, self.rotary = stack, norm, readout_weight, rotary
        self.dataset, self.config, self.guard = dataset, config, guard
        self.dtype = getattr(torch, config.dtype); self.device = next(stack.parameters()).device
        if self.device.type == "cuda":
            if guard is None:
                raise ValueError("CUDA training requires the caller's experiment-reservation guard")
            guard()
        if (dataset.width != stack.width or readout_weight.shape != (dataset.vocabulary, stack.width) or
                readout_weight.device != self.device or readout_weight.dtype != self.dtype):
            raise ValueError("Boundary/readout width, vocabulary, device or execution dtype mismatch")
        if any(b.position_count > 1 and max(dataset.lengths) > b.position_count for b in stack.blocks):
            raise ValueError("Cached inputs exceed fixed position-table capacity")
        self.label_ids = torch.tensor(label_ids, device=self.device, dtype=torch.long)
        if not len(label_ids) or any(i < 0 or i >= dataset.vocabulary for i in label_ids):
            raise ValueError("Invalid choice-label token ids")
        if any(not 0 <= c["answer"] < len(c["choices"]) <= len(label_ids) or
               c["targets"][-1] != label_ids[c["answer"]] for c in dataset.items):
            raise ValueError("Choice labels and final supervised targets must agree")
        self.readout.requires_grad_(False)
        self.rotary.requires_grad_(False)
        self.stack.checkpoint_blocks = config.checkpoint_blocks
        self.stack.requires_grad_(True); self.norm.requires_grad_(True)
        for parameter in (*self.stack.parameters(), *self.norm.parameters()):
            parameter.data = parameter.data.to(device=self.device, dtype=self.dtype)
        self.parameters = [*self.stack.parameters(), *self.norm.parameters()]
        self.optimizer = StreamingCPUAdamW(self.parameters, lr=config.learning_rate, weight_decay=0.,
            chunk_bytes=config.optimizer_chunk_bytes, max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.updates = 0
        self.finished = False

    def _guard(self):
        if self.guard is not None:
            self.guard()

    def logits(self, batch, *, stack=None, norm=None):
        stack, norm = stack or self.stack, norm or self.norm
        hidden = batch["hidden"]
        rotary = self.rotary(hidden, batch["position_ids"])
        hidden = stack(hidden, attention_mask=batch["attention_mask"], position_ids=batch["position_ids"],
            cache_position=batch["position_ids"][0], position_embeddings=rotary, use_cache=False)[0]
        selected = hidden[torch.arange(len(batch["indices"]), device=self.device)[:, None], batch["loss_positions"]]
        return F.linear(norm(selected), self.readout).float()

    def snapshot(self):
        return {"stack": recursive_cpu(self.stack.state_dict()), "norm": recursive_cpu(self.norm.state_dict())}

    def train_window(self, indices, *, learning_rate=None):
        """One actual accumulation window, weighted by its true number of cases."""
        indices = list(indices)
        if self.finished:
            raise RuntimeError("Recovery finished; construct a new optimizer for any subsequent stage")
        if not indices or any(i not in self.dataset.training for i in indices):
            raise ValueError("Optimizer updates may use training cases only")
        self._guard(); self.stack.train(); self.norm.train()
        self.optimizer.zero_grad(set_to_none=True)
        if learning_rate is not None:
            self.optimizer.param_groups[0]["lr"] = learning_rate
        metrics = dict(loss=0., kl=0., ce=0.)
        for start in range(0, len(indices), self.config.microbatch):
            chosen = indices[start:start + self.config.microbatch]
            batch = self.dataset.batch(chosen, device=self.device, dtype=self.dtype)
            logits = self.logits(batch)
            loss, kl, ce = output_objective(logits, batch["teacher"], batch["targets"], self.config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite joint output-distillation loss")
            weight = len(chosen) / len(indices)
            (loss * weight).backward()
            for key, value in zip(metrics, (loss, kl, ce)):
                metrics[key] += float(value.detach()) * weight
            del batch, logits, loss, kl, ce
        present = [all(p.grad is not None for p in block.parameters()) for block in self.stack.blocks]
        if not all(present) or any(p.grad is None for p in self.norm.parameters()):
            raise RuntimeError("Every compact block and final norm must participate in joint recovery")
        update = self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        self.optimizer.zero_grad(set_to_none=True); self.updates += 1
        return {**metrics, "cases": len(indices), "update": self.updates, "learning_rate": self.optimizer.param_groups[0]["lr"],
                "gradients_present_by_block": present, "optimizer": update}

    @torch.no_grad()
    def packed_development(self):
        """CPU fold then temporary inference clone; never pack or replace the live stack."""
        self._guard(); self.optimizer.zero_grad(set_to_none=True)
        clone = cpu_module_clone(self.stack)
        clone.pack(dtype=self.dtype, buffer_dtype=self.dtype, strip_unused=True).eval().requires_grad_(False)
        clone.checkpoint_blocks = False
        packed_norm = cpu_module_clone(self.norm).to(dtype=self.dtype).eval().requires_grad_(False)
        clone.to(self.device); packed_norm.to(self.device)
        totals = dict(loss=0., kl=0., ce=0., original_kl=0., accepted_original_kl=0.)
        correct = accepted_agreement = original_agreement = count = 0
        rows = []
        try:
            with torch.inference_mode():
                for start in range(0, len(self.dataset.development), self.config.microbatch):
                    self._guard()
                    indices = self.dataset.development[start:start + self.config.microbatch]
                    batch = self.dataset.batch(indices, device=self.device, dtype=self.dtype, include_original=True)
                    logits = self.logits(batch, stack=clone, norm=packed_norm)
                    loss, kl, ce = output_objective(logits, batch["teacher"], batch["targets"], self.config)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite packed development loss")
                    for key, value in zip(("loss", "kl", "ce"), (loss, kl, ce)):
                        totals[key] += float(value) * len(indices)
                    if "original" in batch:
                        _, original_kl, _ = output_objective(logits, batch["original"], batch["targets"], self.config)
                        _, accepted_original_kl, _ = output_objective(batch["teacher"], batch["original"], batch["targets"], self.config)
                        totals["original_kl"] += float(original_kl) * len(indices)
                        totals["accepted_original_kl"] += float(accepted_original_kl) * len(indices)
                    for row, index in enumerate(indices):
                        case = self.dataset.items[index]; labels = self.label_ids[:len(case["choices"])]
                        values = logits[row, -1, labels]
                        prediction = int(values.argmax()); correct += prediction == case["answer"]
                        accepted_agreement += prediction == int(batch["teacher"][row, -1, labels].argmax())
                        if "original" in batch:
                            original_agreement += prediction == int(batch["original"][row, -1, labels].argmax())
                        rows.append({"identity": case["identity"], "family": case["family"], "answer": case["answer"],
                                     "prediction": prediction, "logits": values.cpu().tolist()})
                    count += len(indices)
        finally:
            del clone, packed_norm
            gc.collect()
        metrics = {k: v / count for k, v in totals.items()}
        if self.dataset.original is None:
            metrics["original_kl"] = metrics["accepted_original_kl"] = None
        metrics.update(correct=correct, count=count, accuracy=correct / count, accepted_choice_agreement=accepted_agreement / count,
            original_choice_agreement=original_agreement / count if self.dataset.original is not None else None,
            selection_metric="packed_development_ce", precision=self.config.dtype)
        return metrics, rows

    def fit(self, output_dir, *, event=None):
        """Twelve full epochs by default; no early stop, hidden loss, or test access."""
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
        if self.updates or self.finished:
            raise ValueError("fit requires a fresh initializer and optimizer; pilot steps must use a separate instance")
        started = time.perf_counter(); torch.manual_seed(self.config.seed)
        initial, initial_rows = self.packed_development()
        best, best_epoch, selected, selected_rows = initial["ce"], 0, initial, initial_rows
        best_state = self.snapshot(); history = [{"epoch": 0, "development": initial}]
        effective = self.config.microbatch * self.config.accumulation
        updates_per_epoch = math.ceil(len(self.dataset.training) / effective)
        total_updates = updates_per_epoch * self.config.epochs
        for epoch in range(1, self.config.epochs + 1):
            order = list(self.dataset.training); random.Random(self.config.seed + epoch).shuffle(order)
            training = []
            for start in range(0, len(order), effective):
                fraction = self.updates / total_updates
                lr = self.config.learning_rate * .5 * (1 + math.cos(math.pi * fraction))
                value = self.train_window(order[start:start + effective], learning_rate=lr)
                training.append(value)
                if event is not None:
                    event("update", epoch=epoch, **value)
            development, rows = self.packed_development()
            row = {"epoch": epoch, "training": training, "development": development}
            history.append(row)
            with (output_dir / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if development["ce"] < best:
                best, best_epoch, selected, selected_rows = development["ce"], epoch, development, rows
                best_state = self.snapshot()
            if event is not None:
                event("epoch", epoch=epoch, development=development, selected_epoch=best_epoch)
        if self.updates != total_updates:
            raise RuntimeError("Wrong optimizer budget, including partial accumulation windows")
        torch.save(best_state, output_dir / "recovered_unpacked.pt")
        self.stack.load_state_dict(best_state["stack"], strict=True); self.norm.load_state_dict(best_state["norm"], strict=True)
        self.finished = True
        final, final_rows = self.packed_development()
        if final_rows != selected_rows or final != selected:
            raise ValueError("Selected unpacked state failed its packed development reload")
        report = {"status": "completed", "config": asdict(self.config), "completed_epochs": self.config.epochs,
            "updates": self.updates, "effective_batch": effective, "selected_epoch": best_epoch,
            "selection_metric": "packed_development_ce", "initial": initial, "selected": selected,
            "history": history, "elapsed_seconds": time.perf_counter() - started,
            "optimizer_memory": self.optimizer.memory_stats(), "bulk_transferred_bytes": self.optimizer.transferred_bytes,
            "cache_read_bytes": self.dataset.read_bytes, "checkpoint": "recovered_unpacked.pt",
            "scope": "joint suffix output-only recovery; original teacher drift diagnostic; no held-out claims"}
        for name, value in (("training.json", report), ("selected_packed_development.json", selected_rows)):
            with (output_dir / name).open("x") as stream:
                json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        return report
