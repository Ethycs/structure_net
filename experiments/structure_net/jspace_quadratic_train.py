"""Frozen-prefix, one-layer-at-a-time Qwen coordinate distillation.

The caller pins input-bank provenance and advances the selected frozen student
prefix. This module never updates a previous block, changes the native teacher,
fits a basis, or consumes held-out cases. All trainable tensors are FP32.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_streaming_adamw import StreamingCPUAdamW


@dataclass(frozen=True)
class SequentialConfig:
    epochs: int = 12
    microbatch: int = 4
    accumulation: int = 4
    learning_rate: float = 1.2e-4
    temperature: float = 2.
    gradient_clip: float = 1.
    seed: int = 20861
    optimizer_chunk_bytes: int = 32 * 1024**2
    max_cpu_state_bytes: int | None = None

    def __post_init__(self):
        for name in ("epochs", "microbatch", "accumulation", "optimizer_chunk_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (4 if name == "optimizer_chunk_bytes" else 1):
                raise ValueError("Positive integer configuration required: " + name)
        if any(not math.isfinite(value) or value <= 0 for value in
               (self.learning_rate, self.temperature, self.gradient_clip)):
            raise ValueError("Finite positive learning rate, temperature and gradient clip required")


def _cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(v) for v in value)
    return deepcopy(value)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class CoordinateDataset:
    """Disk-backed FP32 student inputs and original BF16 teacher boundaries.

    ``basis`` is [native_width, rank]; ``anchor`` is one global row or a fixed
    absolute-position table. The target is U^T(teacher_boundary-anchor).
    Passing teacher_logits_path selects final-layer supervision at exactly four
    positions per case; otherwise all valid output positions are supervised.
    No projected target bank is materialized on disk or retained in host RAM.
    """
    def __init__(self, items, coordinates_path, teacher_boundary_path, *, basis, anchor,
                 teacher_logits_path=None):
        self.items = deepcopy(list(items))
        if not self.items or any(c.get("split") not in ("train", "development") for c in self.items):
            raise ValueError("Held-out cases are forbidden in the distillation dataset")
        if len({c["identity"] for c in self.items}) != len(self.items):
            raise ValueError("Duplicate training/development identity")
        self.training = [i for i, c in enumerate(self.items) if c["split"] == "train"]
        self.development = [i for i, c in enumerate(self.items) if c["split"] == "development"]
        if not self.training or not self.development:
            raise ValueError("Both training and development cases are required")
        self.lengths = []
        cursor = 0
        for case in self.items:
            length = len(case["input_ids"])
            if length < 1 or case["offset"] != cursor or case["stop"] != cursor + length:
                raise ValueError("Case offsets must partition the complete token banks")
            positions = case["loss_positions"]
            if len(positions) != 4 or any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < length for p in positions):
                raise ValueError("Four valid teacher-logit positions are required")
            self.lengths.append(length); cursor += length
        self.basis = basis.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
        self.anchor = anchor.detach().to(device="cpu", dtype=torch.float32).clone()
        if self.anchor.ndim == 1:
            self.anchor = self.anchor[None]
        if (self.basis.ndim != 2 or not 0 < self.basis.shape[1] <= self.basis.shape[0]
                or self.anchor.ndim != 2 or self.anchor.shape[1] != self.basis.shape[0]
                or self.anchor.shape[0] not in (1,) and self.anchor.shape[0] < max(self.lengths)
                or not torch.isfinite(self.basis).all() or not torch.isfinite(self.anchor).all()):
            raise ValueError("Invalid fixed projection geometry or position capacity")
        self.width, self.rank = self.basis.shape
        self.coordinates = self.teacher = self.teacher_logits = None
        self._device_geometry = {}
        self.final_layer = teacher_logits_path is not None
        self.normalizer = None
        self.read_bytes = dict(coordinates=0, teacher_boundary=0, teacher_logits=0)
        try:
            self.coordinates = np.load(coordinates_path, mmap_mode="r", allow_pickle=False)
            self.teacher = np.load(teacher_boundary_path, mmap_mode="r", allow_pickle=False)
            if self.coordinates.dtype != np.float32 or self.coordinates.shape != (cursor, self.rank):
                raise ValueError("Student coordinate bank must be FP32 [all tokens, rank]")
            if self.teacher.dtype != np.int16 or self.teacher.shape != (cursor, self.width):
                raise ValueError("Original teacher boundary must preserve BF16 bits as int16")
            if self.final_layer:
                self.teacher_logits = np.load(teacher_logits_path, mmap_mode="r", allow_pickle=False)
                if (self.teacher_logits.dtype != np.int16 or self.teacher_logits.ndim != 3
                        or self.teacher_logits.shape[:2] != (len(self.items), 4) or self.teacher_logits.shape[-1] < 1):
                    raise ValueError("Teacher logits must be aligned BF16 bits [cases,4,vocabulary]")
        except Exception:
            self.close()
            raise

    def close(self):
        for value in (self.coordinates, self.teacher, self.teacher_logits):
            if value is not None and getattr(value, "_mmap", None) is not None and not value._mmap.closed:
                value._mmap.close()
        self._device_geometry.clear()

    def position_count(self, indices):
        return 4 * len(indices) if self.final_layer else sum(self.lengths[i] for i in indices)

    def _layout(self, indices, device):
        indices = list(indices)
        if not indices or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self.items) for i in indices):
            raise ValueError("A nonempty list of valid case indices is required")
        lengths = [self.lengths[i] for i in indices]
        positions = torch.arange(max(lengths), device=device).expand(len(indices), -1)
        valid = positions < torch.tensor(lengths, device=device)[:, None]
        loss_positions = torch.tensor([self.items[i]["loss_positions"] for i in indices], device=device)
        return indices, lengths, positions, valid, loss_positions

    def _project_teacher(self, indices, lengths, positions, device):
        storage = np.zeros((len(indices), max(lengths), self.width), dtype=np.int16)
        for row, i in enumerate(indices):
            case = self.items[i]
            storage[row, :lengths[row]] = self.teacher[case["offset"]:case["stop"]]
        native = torch.from_numpy(storage).view(torch.bfloat16).to(device=device, dtype=torch.float32)
        self.read_bytes["teacher_boundary"] += sum(lengths) * self.width * 2
        key = str(torch.device(device))
        if key not in self._device_geometry:
            self._device_geometry[key] = (self.basis.to(device), self.anchor.to(device))
        basis, anchor = self._device_geometry[key]
        fixed = anchor[0] if len(anchor) == 1 else anchor[positions]
        with torch.no_grad(), torch.autocast(device_type=native.device.type, enabled=False):
            target = F.linear(native - fixed, basis.T)
        if not torch.isfinite(target).all():
            raise FloatingPointError("Nonfinite projected original-teacher target")
        return target

    def batch(self, indices, *, device="cpu"):
        indices, lengths, positions, valid, loss_positions = self._layout(indices, device)
        storage = np.zeros((len(indices), max(lengths), self.rank), dtype=np.float32)
        for row, i in enumerate(indices):
            case = self.items[i]
            storage[row, :lengths[row]] = self.coordinates[case["offset"]:case["stop"]]
        coordinates = torch.from_numpy(storage).to(device)
        self.read_bytes["coordinates"] += sum(lengths) * self.rank * 4
        target = self._project_teacher(indices, lengths, positions, device)
        causal = positions[:, :, None] >= positions[:, None, :]
        allowed = causal & valid[:, :, None] & valid[:, None, :]
        result = dict(indices=indices, coordinates=coordinates, target=target, position_ids=positions,
            valid=valid, attention_mask=allowed[:, None], loss_positions=loss_positions,
            lengths=lengths, feature_positions=self.position_count(indices))
        if self.final_layer:
            bits = np.array(self.teacher_logits[indices], copy=True)
            logits = torch.from_numpy(bits).view(torch.bfloat16).to(device=device, dtype=torch.float32)
            self.read_bytes["teacher_logits"] += logits.numel() * 2
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite frozen teacher logits")
            result["teacher_logits"] = logits
        if not torch.isfinite(coordinates).all():
            raise FloatingPointError("Nonfinite frozen student-prefix coordinates")
        return result

    @torch.no_grad()
    def fit_normalizer(self, *, device="cpu", microbatch=4, guard=None):
        """Mean target-coordinate energy over training supervision only."""
        if self.normalizer is not None:
            return self.normalizer
        total = 0.; count = 0
        for start in range(0, len(self.training), microbatch):
            if guard is not None:
                guard()
            indices, lengths, positions, valid, selected = self._layout(self.training[start:start + microbatch], device)
            target = self._project_teacher(indices, lengths, positions, device)
            if self.final_layer:
                target = target[torch.arange(len(indices), device=device)[:, None], selected]
            else:
                target = target[valid]
            total += float(target.double().square().sum()); count += target.numel()
        value = total / count
        if not math.isfinite(value) or value <= 1e-20:
            raise ValueError("Degenerate train-only target-coordinate energy")
        self.normalizer = value
        return value


def feature_loss(student, batch, normalizer, *, final_layer):
    if final_layer:
        rows = torch.arange(student.shape[0], device=student.device)[:, None]
        selected = student[rows, batch["loss_positions"]]
        target = batch["target"][rows, batch["loss_positions"]]
        errors = (selected - target).square().mean(-1)
        per_case = errors.mean(-1)
        mse = errors.mean()
    else:
        selected = None
        errors = (student - batch["target"]).square().mean(-1)
        masked = torch.where(batch["valid"], errors, 0.)
        per_case = masked.sum(-1) / batch["valid"].sum(-1)
        mse = masked.sum() / batch["valid"].sum()
    return mse / normalizer, mse, per_case, selected


def logit_kl(student, teacher, temperature=2.):
    log_p = (teacher.detach().float() / temperature).log_softmax(-1)
    log_q = (student.float() / temperature).log_softmax(-1)
    per_position = (log_p.exp() * (log_p - log_q)).sum(-1)
    return per_position.mean(), per_position.mean(-1)


class SequentialTrainer:
    """Train only the supplied current block and optional final compact readout."""
    def __init__(self, block, rotary, dataset, *, readout=None, label_ids=None,
                 config=SequentialConfig(), guard=None):
        from .jspace_quadratic_qwen import QuadraticQwenBlock
        if not isinstance(block, QuadraticQwenBlock) or block.original.packed:
            raise ValueError("An unpacked current QuadraticQwenBlock is required")
        if (block.layer_idx == 35) != dataset.final_layer or (readout is not None) != dataset.final_layer:
            raise ValueError("Only final layer35 trains a readout and uses cached teacher logits")
        if block.rank != dataset.rank or block.width != dataset.width:
            raise ValueError("Coordinate/teacher widths differ from the current block")
        if not torch.equal(block.basis.detach().float().cpu(), dataset.basis) or not torch.equal(
                block.anchor.detach().float().cpu().reshape(-1, block.width), dataset.anchor):
            raise ValueError("Teacher-coordinate targets must use the block's exact fixed geometry")
        self.block, self.readout, self.rotary, self.dataset = block, readout, rotary, dataset
        self.config, self.guard = config, guard
        self.device = next(block.parameters()).device
        if self.device.type == "cuda" and guard is None:
            raise ValueError("GPU training requires the caller's reservation guard")
        self._guard()
        self.block.requires_grad_(True)
        if readout is not None:
            readout.requires_grad_(True)
        self.rotary.eval().requires_grad_(False)
        self.named_parameters = [("block." + name, parameter) for name, parameter in block.named_parameters()]
        if readout is not None:
            self.named_parameters += [("readout." + name, parameter) for name, parameter in readout.named_parameters()]
        if any(p.dtype != torch.float32 or p.device != self.device for _, p in self.named_parameters):
            raise ValueError("Every current-block/readout parameter must be FP32 on the same device")
        self.parameters = [p for _, p in self.named_parameters]
        self.optimizer = StreamingCPUAdamW(self.parameters, lr=config.learning_rate, weight_decay=0.,
            chunk_bytes=config.optimizer_chunk_bytes, max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.gradient_updates = {name: 0 for name, _ in self.named_parameters}
        self.label_ids = None if label_ids is None else torch.tensor(label_ids, dtype=torch.long, device=self.device)
        if self.label_ids is not None and dataset.final_layer and (self.label_ids.min() < 0 or
                self.label_ids.max() >= dataset.teacher_logits.shape[-1]):
            raise ValueError("Choice-label ids exceed the cached vocabulary")
        self.updates = 0
        self.finished = False

    def _guard(self):
        if self.guard is not None:
            self.guard()

    def _normalizer(self):
        return self.dataset.fit_normalizer(device=self.device, microbatch=self.config.microbatch, guard=self.guard)

    def forward(self, batch):
        z = batch["coordinates"]
        rope = self.rotary(z, batch["position_ids"])
        return self.block.forward_coordinates(z, attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], cache_position=batch["position_ids"][0], position_embeddings=rope)

    def losses(self, batch):
        normalizer = self._normalizer()
        output = self.forward(batch)
        normalized, mse, per_case, selected = feature_loss(output, batch, normalizer, final_layer=self.dataset.final_layer)
        kl = torch.zeros((), device=self.device)
        logits = None
        per_case_kl = torch.zeros(len(batch["indices"]), device=self.device)
        if self.readout is not None:
            logits = self.readout(selected, position_ids=batch["loss_positions"])
            if logits.shape != batch["teacher_logits"].shape:
                raise ValueError("Compact readout and frozen teacher vocabulary differ")
            kl, per_case_kl = logit_kl(logits, batch["teacher_logits"], self.config.temperature)
        loss = normalized + self.config.temperature**2 * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite sequential feature/logit objective")
        return dict(loss=loss, normalized_feature=normalized, feature_mse=mse, kl=kl,
                    per_case_mse=per_case, per_case_kl=per_case_kl, logits=logits)

    def snapshot(self):
        return {"block": _cpu_state(self.block.state_dict()),
                "readout": None if self.readout is None else _cpu_state(self.readout.state_dict())}

    def train_window(self, indices, *, learning_rate=None):
        indices = list(indices)
        if self.finished:
            raise RuntimeError("Sequential fit finished; construct a fresh trainer to continue")
        if not indices or any(i not in self.dataset.training for i in indices):
            raise ValueError("Optimizer updates may consume training cases only")
        self._guard(); self._normalizer()
        self.block.train()
        if self.readout is not None:
            self.readout.train()
        self.optimizer.zero_grad(set_to_none=True)
        if learning_rate is not None:
            self.optimizer.param_groups[0]["lr"] = learning_rate
        total_positions = self.dataset.position_count(indices)
        metrics = dict(loss=0., normalized_feature=0., feature_mse=0., kl=0.)
        for start in range(0, len(indices), self.config.microbatch):
            selected = indices[start:start + self.config.microbatch]
            batch = self.dataset.batch(selected, device=self.device)
            values = self.losses(batch)
            weight = batch["feature_positions"] / total_positions
            (values["loss"] * weight).backward()
            for key in metrics:
                metrics[key] += float(values[key].detach()) * weight
            del values, batch
        missing = [name for name, p in self.named_parameters if p.grad is None]
        if missing:
            raise RuntimeError("Current parameters missing gradients: " + ", ".join(missing))
        step = self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        for name in self.gradient_updates:
            self.gradient_updates[name] += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        return {**metrics, "cases": len(indices), "feature_positions": total_positions, "update": self.updates,
            "learning_rate": self.optimizer.param_groups[0]["lr"], "gradient_parameter_count": len(self.named_parameters),
            "gradient_scalar_count": sum(p.numel() for p in self.parameters), "optimizer": step}

    @torch.no_grad()
    def development(self):
        self._guard(); self._normalizer(); self.optimizer.zero_grad(set_to_none=True)
        self.block.eval()
        if self.readout is not None:
            self.readout.eval()
        totals = dict(loss=0., normalized_feature=0., feature_mse=0., kl=0.)
        rows = []; positions = 0
        for start in range(0, len(self.dataset.development), self.config.microbatch):
            self._guard()
            indices = self.dataset.development[start:start + self.config.microbatch]
            batch = self.dataset.batch(indices, device=self.device)
            values = self.losses(batch)
            count = batch["feature_positions"]
            for key in totals:
                totals[key] += float(values[key]) * count
            for row, index in enumerate(indices):
                case = self.dataset.items[index]
                record = {"identity": case["identity"], "family": case["family"],
                    "feature_mse": float(values["per_case_mse"][row]), "kl": float(values["per_case_kl"][row])}
                if self.readout is not None and self.label_ids is not None:
                    ids = self.label_ids[:len(case["choices"])]
                    choice = values["logits"][row, -1, ids]
                    record.update(answer=case["answer"], prediction=int(choice.argmax()), logits=choice.cpu().tolist())
                rows.append(record)
            positions += count
        metrics = {k: v / positions for k, v in totals.items()}
        metrics.update(count=len(rows), feature_positions=positions, target_coordinate_energy=self.dataset.normalizer,
                       selection_metric="normalized_feature_plus_temperature_squared_kl" if self.dataset.final_layer else "normalized_feature")
        if rows and "prediction" in rows[0]:
            metrics["correct"] = sum(r["prediction"] == r["answer"] for r in rows)
            metrics["accuracy"] = metrics["correct"] / len(rows)
        return metrics, rows

    def fit(self, output_dir, *, event=None):
        output_dir = Path(output_dir)
        if self.updates or self.finished:
            raise ValueError("Fit requires a fresh current-block optimizer")
        output_dir.mkdir(parents=True, exist_ok=False)
        start_time = time.perf_counter(); torch.manual_seed(self.config.seed)
        initial, initial_rows = self.development()
        _write(output_dir / "epoch0_development.json", {"metrics": initial, "records": initial_rows})
        best, best_epoch, selected, selected_rows = initial["loss"], 0, initial, initial_rows
        best_state = self.snapshot()
        history = [{"epoch": 0, "development": initial}]
        effective = self.config.microbatch * self.config.accumulation
        total_updates = self.config.epochs * math.ceil(len(self.dataset.training) / effective)
        for epoch in range(1, self.config.epochs + 1):
            order = list(self.dataset.training); random.Random(self.config.seed + epoch).shuffle(order)
            training = []
            for start in range(0, len(order), effective):
                lr = self.config.learning_rate * .5 * (1 + math.cos(math.pi * self.updates / total_updates))
                update = self.train_window(order[start:start + effective], learning_rate=lr)
                training.append(update)
                if event is not None:
                    event("update", epoch=epoch, **update)
            development, records = self.development()
            row = {"epoch": epoch, "training": training, "development": development}
            history.append(row)
            with (output_dir / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if development["loss"] < best:
                best, best_epoch, selected, selected_rows = development["loss"], epoch, development, records
                best_state = self.snapshot()
            if event is not None:
                event("epoch", epoch=epoch, development=development, selected_epoch=best_epoch)
        if self.updates != total_updates:
            raise RuntimeError("Incomplete sequential training budget")
        with (output_dir / "selected.pt").open("xb") as stream:
            torch.save(best_state, stream)
        # Reload the actual on-disk export; no stale in-memory best checkpoint.
        restored = torch.load(output_dir / "selected.pt", map_location="cpu", weights_only=True)
        self.block.load_state_dict(restored["block"], strict=True)
        if self.readout is not None:
            self.readout.load_state_dict(restored["readout"], strict=True)
        self.finished = True
        verified, records = self.development()
        if verified != selected or records != selected_rows:
            raise ValueError("Selected disk checkpoint failed exact development reload")
        parameter_stats = {name: {"numel": p.numel(), "gradient_updates": self.gradient_updates[name],
            "optimizer_updates": self.optimizer.state[p]["step"]} for name, p in self.named_parameters}
        report = {"status": "completed", "layer": self.block.layer_idx, "mode": self.block.mode,
            "config": asdict(self.config), "completed_epochs": self.config.epochs, "updates": self.updates,
            "selected_epoch": best_epoch, "selection_metric": selected["selection_metric"],
            "initial": initial, "selected": selected, "history": history,
            "normalizer_scope": "fixed mean squared original-teacher target coordinates; training cases only",
            "target_coordinate_energy": self.dataset.normalizer, "final_layer": self.dataset.final_layer,
            "objective": "normalized_feature + T^2 KL" if self.dataset.final_layer else "normalized_feature",
            "trainable_parameters": parameter_stats, "trainable_scalar_count": sum(p.numel() for p in self.parameters),
            "optimizer_memory": self.optimizer.memory_stats(), "bulk_transferred_bytes": self.optimizer.transferred_bytes,
            "cache_read_bytes": self.dataset.read_bytes, "checkpoint": "selected.pt", "disk_reload_exact": True,
            "elapsed_seconds": time.perf_counter() - start_time,
            "scope": "one current block; frozen original teacher and cached frozen student prefix; no held-out evidence"}
        _write(output_dir / "selected_development.json", selected_rows)
        _write(output_dir / "training.json", report)
        self.block.eval().requires_grad_(False)
        if self.readout is not None:
            self.readout.eval().requires_grad_(False)
        return report
