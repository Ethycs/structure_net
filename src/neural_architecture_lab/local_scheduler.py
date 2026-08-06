"""Local device-slot planning and resumable experiment result storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from .core import Experiment, ExperimentResult, ExperimentStatus


RESULT_SCHEMA_VERSION = "nal.local-result.v1"


@dataclass(frozen=True)
class DeviceSlotPlan:
    """Logical CUDA devices expanded into schedulable experiment slots."""

    slots: tuple[int, ...]
    slots_by_device: Dict[int, int]
    free_memory_gb: Dict[int, float]
    calibration: str

    @property
    def concurrency(self) -> int:
        return len(self.slots)


def build_device_slot_plan(
    device_ids: Iterable[int],
    *,
    slots_per_device: int,
    memory_per_experiment_gb: Optional[float],
    max_slots_per_device: int,
    usable_memory_fraction: float,
    cuda_available: Optional[bool] = None,
    cuda_device_count: Optional[int] = None,
    free_memory_bytes: Optional[Mapping[int, int]] = None,
) -> DeviceSlotPlan:
    """Validate logical devices and create a static or memory-calibrated plan.

    ``device_ids`` are always logical PyTorch ordinals. Physical selection is
    performed by the parent process with ``CUDA_VISIBLE_DEVICES`` before NAL is
    imported. Injected CUDA facts keep this function deterministic in tests.
    """
    devices = tuple(dict.fromkeys(int(device) for device in device_ids))
    if not devices:
        raise ValueError("At least one logical GPU or the CPU sentinel -1 is required")
    if any(device < 0 for device in devices):
        if devices != (-1,):
            raise ValueError("The CPU sentinel -1 cannot be mixed with CUDA devices")
        return DeviceSlotPlan((-1,), {-1: 1}, {}, "cpu")
    if slots_per_device < 0:
        raise ValueError("gpu_slots_per_device cannot be negative")
    if max_slots_per_device < 1:
        raise ValueError("max_gpu_slots_per_device must be positive")
    if not 0.0 < usable_memory_fraction <= 1.0:
        raise ValueError("usable GPU memory fraction must be in (0, 1]")

    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    count = torch.cuda.device_count() if cuda_device_count is None else cuda_device_count
    if not available:
        raise RuntimeError("CUDA devices were requested, but PyTorch cannot access CUDA")
    invalid = [device for device in devices if device >= count]
    if invalid:
        raise ValueError(
            f"Logical CUDA device(s) {invalid} are unavailable; visible ordinals are "
            f"0..{max(-1, count - 1)}"
        )

    free_gb: Dict[int, float] = {}
    slots_by_device: Dict[int, int] = {}
    if slots_per_device:
        slots_by_device = {
            device: min(slots_per_device, max_slots_per_device) for device in devices
        }
        calibration = "fixed"
    else:
        if memory_per_experiment_gb is None or memory_per_experiment_gb <= 0:
            raise ValueError(
                "Memory-calibrated slots require gpu_memory_per_experiment_gb > 0"
            )
        supplied_memory = dict(free_memory_bytes or {})
        for device in devices:
            free_bytes = supplied_memory.get(device)
            if free_bytes is None:
                free_bytes, _ = torch.cuda.mem_get_info(device)
            free_gb[device] = free_bytes / (1024**3)
            fitted = math.floor(
                free_bytes * usable_memory_fraction
                / (memory_per_experiment_gb * 1024**3)
            )
            if fitted < 1:
                raise RuntimeError(
                    f"Logical CUDA device {device} has {free_gb[device]:.2f} GiB free, "
                    f"below the {memory_per_experiment_gb:.2f} GiB experiment estimate"
                )
            slots_by_device[device] = min(fitted, max_slots_per_device)
        calibration = "free-memory"

    slots = tuple(
        device
        for device in devices
        for _ in range(slots_by_device[device])
    )
    return DeviceSlotPlan(slots, slots_by_device, free_gb, calibration)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, ExperimentStatus):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return repr(value)


def experiment_fingerprint(experiment: Experiment) -> str:
    payload = {
        "experiment_id": experiment.id,
        "hypothesis_id": experiment.hypothesis_id,
        "parameters": _json_safe(experiment.parameters),
        "seed": experiment.seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ExperimentResultLedger:
    """Atomic, fingerprinted per-experiment completion records."""

    def __init__(self, results_dir: str | Path):
        self.root = Path(results_dir) / ".nal_runner" / "completed"

    def _path(self, experiment: Experiment) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment.id).strip("._")
        digest = hashlib.sha256(experiment.id.encode()).hexdigest()[:12]
        return self.root / f"{slug[:80] or 'experiment'}-{digest}.json"

    def load(self, experiment: Experiment) -> Optional[ExperimentResult]:
        path = self._path(experiment)
        if not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if envelope.get("schema_version") != RESULT_SCHEMA_VERSION:
            return None
        if envelope.get("fingerprint") != experiment_fingerprint(experiment):
            return None
        result = envelope.get("result")
        if not isinstance(result, Mapping):
            return None
        if result.get("status") != ExperimentStatus.COMPLETED.value or result.get("error"):
            return None
        try:
            return ExperimentResult(
                experiment_id=str(result["experiment_id"]),
                hypothesis_id=str(result["hypothesis_id"]),
                metrics={str(key): float(value) for key, value in result["metrics"].items()},
                primary_metric=float(result["primary_metric"]),
                model_architecture=[int(value) for value in result["model_architecture"]],
                model_parameters=int(result["model_parameters"]),
                training_time=float(result["training_time"]),
                training_history=list(result.get("training_history", [])),
                model_checkpoint=result.get("model_checkpoint"),
                observations=list(result.get("observations", [])),
                anomalies=list(result.get("anomalies", [])),
                timestamp=datetime.fromisoformat(result["timestamp"]),
                error=result.get("error"),
                status=ExperimentStatus(result["status"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def save(
        self,
        experiment: Experiment,
        result: ExperimentResult,
        *,
        attempts: int,
        device_id: int,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(experiment)
        envelope = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "fingerprint": experiment_fingerprint(experiment),
            "attempts": attempts,
            "logical_device_id": device_id,
            "result": _json_safe(asdict(result)),
        }
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


__all__ = [
    "DeviceSlotPlan",
    "ExperimentResultLedger",
    "RESULT_SCHEMA_VERSION",
    "build_device_slot_plan",
    "experiment_fingerprint",
]
