#!/usr/bin/env python3
"""Train a physically typed scalar interface around frozen TinyLLM backbones."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibrated_architecture_replication_preflight as preflight
import experiments.structure_net.tinyllm_frozen_interval_readout_decomposition as interval
import experiments.structure_net.tinyllm_frozen_scalar_interface_decomposition as source
from experiments.structure_net.tinyllm_conditional_branch_depth_scan import (
    _add_log_loss_gains,
    fit_conditional_probe,
    fit_cosine_only_null,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-joint-physical-scalar-interface.v1"
HYPOTHESIS_ID = "tinyllm-joint-physical-scalar-interface-v1"
EVIDENCE_ROLE = "prospective_frozen_backbone_joint_physical_interface"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-physical-scalar-interface-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "1f83fb2802340a51f4dc281f898e99122b99089c48e6c207bd46626c20aab838"
)
SOURCE_ROOT = source.SOURCE_ROOT
COMPARATOR_ROOT = Path(
    "data/experiments/tinyllm_frozen_interval_readout_decomposition/"
    "20260811_d6_d10_preregistered"
)
COMPARATOR_CAMPAIGN_SHA256 = (
    "3f15245386d1fb41e797f0688ff512aadc7c9690a552c3d58c8ba92754ee9208"
)
COMPARATOR_RESULT_MANIFEST_SHA256 = (
    "8a567df1952634a0be782b0dbd52e8dd92bd8d5d39f3f2d4f23c4f6df2a04638"
)
PRESETS = ("d6", "d10")
CONDITIONS = (
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
SEEDS = (7, 17, 29, 41, 53)
ARMS = ("physical_true", "pair_shuffled")
CUTS = ("frontend", "full")
REGIMES = ("composition", "extrapolation")
ANALYSIS_SPLITS = ("train", "validation", "in_distribution", *REGIMES)
ANALYSIS_SPECS = {
    "train": (184, "interpolation"),
    "validation": (294, "interpolation"),
    "in_distribution": (390, "interpolation"),
    "composition": source.DATASET_SPECS["composition"],
    "extrapolation": source.DATASET_SPECS["extrapolation"],
}


@dataclass(frozen=True)
class JointPhysicalInterfaceConfig:
    source_root: str = str(SOURCE_ROOT)
    comparator_root: str = str(COMPARATOR_ROOT)
    presets: tuple[str, ...] = PRESETS
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    sensor_loss_weight: float = 1.0
    final_loss_weight: float = 1.0
    task_loss_weight: float = 1.0
    analysis_seed: int = 83
    probe_train_samples: int = 2_048
    probe_validation_samples: int = 512
    probe_test_samples: int = 1_024
    activation_batch_size: int = 256
    probe_batch_size: int = 256
    probe_steps: int = 240
    probe_width: int = 128
    validation_interval: int = 20
    early_stopping_patience: int = 5
    cosine_minimum: float = 0.90
    branch_accuracy_maximum: float = 0.55
    conditional_log_loss_gain_maximum: float = 0.02
    required_seed_passes: int = 4
    shuffled_seed_pass_ceiling: int = 1
    shuffle_seed: int = 20_260_811
    replay_tolerance: float = 2e-6
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 1
    gpu_memory_per_experiment_gb: Optional[float] = 3.0
    max_gpu_slots_per_device: int = 1
    max_parallel_experiments: int = 1
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.presets or set(self.presets).difference(PRESETS):
            raise ValueError("unknown or empty preset selection")
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty condition selection")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.training_steps < 1 or self.train_samples < 4:
            raise ValueError("training steps and sample count must be positive")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch size must be positive and even")
        if min(
            self.probe_train_samples,
            self.probe_validation_samples,
            self.probe_test_samples,
            self.probe_batch_size,
            self.probe_steps,
        ) < 1:
            raise ValueError("probe configuration must be positive")
        if (
            self.sensor_loss_weight != 1.0
            or self.final_loss_weight != 1.0
            or self.task_loss_weight != 1.0
        ):
            raise ValueError("the three preregistered loss weights are fixed to one")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed passes are outside the population")
        if self.shuffled_seed_pass_ceiling < 0:
            raise ValueError("shuffled seed ceiling must be non-negative")
        if not self.allow_underpowered:
            expected = {
                "source_root": str(SOURCE_ROOT),
                "comparator_root": str(COMPARATOR_ROOT),
                "presets": PRESETS,
                "conditions": CONDITIONS,
                "seeds": SEEDS,
                "training_steps": 600,
                "train_samples": 4_096,
                "batch_size": 64,
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "probe_train_samples": 2_048,
                "probe_validation_samples": 512,
                "probe_test_samples": 1_024,
                "probe_steps": 240,
                "required_seed_passes": 4,
                "shuffled_seed_pass_ceiling": 1,
                "shuffle_seed": 20_260_811,
            }
            actual = {
                "source_root": self.source_root,
                "comparator_root": self.comparator_root,
                "presets": self.presets,
                "conditions": self.conditions,
                "seeds": self.seeds,
                "training_steps": self.training_steps,
                "train_samples": self.train_samples,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "gradient_clip": self.gradient_clip,
                "probe_train_samples": self.probe_train_samples,
                "probe_validation_samples": self.probe_validation_samples,
                "probe_test_samples": self.probe_test_samples,
                "probe_steps": self.probe_steps,
                "required_seed_passes": self.required_seed_passes,
                "shuffled_seed_pass_ceiling": self.shuffled_seed_pass_ceiling,
                "shuffle_seed": self.shuffle_seed,
            }
            if actual != expected:
                raise ValueError("primary joint-interface configuration changed")


class PhysicalScalarInterface(nn.Module):
    """A typed scalar before and after a completely frozen TinyLLM backbone."""

    def __init__(self, system: calibrated.CalibratedTinyLLM):
        super().__init__()
        self.system = system
        self.final_scalar = nn.Linear(system.model.config.n_embd, 1)
        nn.init.zeros_(self.final_scalar.weight)
        nn.init.zeros_(self.final_scalar.bias)

    def forward_scalars(
        self,
        input_ids: torch.Tensor,
        sensor: torch.Tensor,
        calibration_packet: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        frontend = self.system.feature(sensor, calibration_packet).reshape(-1, 1)
        residual = self.system.forward_cuts(
            input_ids, sensor, calibration_packet
        )["full"]
        normalized = self.system.model.transformer["ln_f"](residual)
        return {"frontend": frontend, "full": self.final_scalar(normalized)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _json_config(config: JointPhysicalInterfaceConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(values: Mapping[str, Any]) -> JointPhysicalInterfaceConfig:
    converted = dict(values)
    for field in ("presets", "conditions", "seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return JointPhysicalInterfaceConfig(**converted)


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "calibrated_frontend": _sha256(Path(calibrated.__file__)),
        "source_validator": _sha256(Path(source.__file__)),
        "interval_comparator": _sha256(Path(interval.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("joint-interface preregistration changed")
    return values


def _implementation_digest(values: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(values or _implementation_sources()))


def interval_posterior_unclipped(
    scalar: torch.Tensor, phase_bins: int = 16
) -> torch.Tensor:
    scalar = scalar.reshape(-1, 1)
    centers = torch.linspace(
        -1.0,
        1.0,
        phase_bins,
        dtype=scalar.dtype,
        device=scalar.device,
    )
    width = 2.0 / float(phase_bins - 1)
    logits = -0.5 * ((centers[None] - scalar) / width).square()
    return torch.softmax(logits, dim=-1)


def pair_preserving_target_permutation(
    sample_count: int,
    preset: str,
    condition: str,
    seed: int,
    base_seed: int,
) -> torch.Tensor:
    if sample_count < 4 or sample_count % 2:
        raise ValueError("paired target permutation requires an even sample count")
    material = f"{base_seed}:{preset}:{condition}:{seed}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(derived % (2**63 - 1))
    pairs = torch.randperm(sample_count // 2, generator=generator)
    if torch.equal(pairs, torch.arange(len(pairs))):
        pairs = torch.roll(pairs, 1)
    return torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)


def _interface_state_digest(module: PhysicalScalarInterface) -> str:
    state: dict[str, torch.Tensor] = {}
    if module.system.encoder is not None:
        state.update(
            {f"encoder.{key}": value for key, value in module.system.encoder.state_dict().items()}
        )
    if module.system.scalar_embedding is None:
        raise AssertionError("structured scalar embedding missing")
    state.update(
        {
            f"scalar_embedding.{key}": value
            for key, value in module.system.scalar_embedding.state_dict().items()
        }
    )
    state.update(
        {f"final_scalar.{key}": value for key, value in module.final_scalar.state_dict().items()}
    )
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(calibrated._tensor_digest(value.detach().cpu()).encode("ascii"))
    return digest.hexdigest()


def _cell_directory(root: Path, preset: str, condition: str, seed: int) -> Path:
    return root / "runs" / preset / condition / f"seed_{seed}"


def _arm_checkpoint_path(cell: Path, arm: str) -> Path:
    return cell / f"{arm}_interface.pt"


def _verify_comparator(root: Path) -> dict[str, Any]:
    path = root / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != COMPARATOR_CAMPAIGN_SHA256
        or campaign.get("status") != "completed"
        or campaign.get("aggregates", {}).get("classification")
        != "partial_frozen_interface_repair"
        or campaign.get("result_manifest_sha256")
        != COMPARATOR_RESULT_MANIFEST_SHA256
    ):
        raise ValueError(f"invalid frozen interval comparator {path}")
    return campaign


def _source_population(
    config: JointPhysicalInterfaceConfig,
) -> tuple[
    dict[str, Any],
    CircleTaskConfig,
    dict[tuple[str, str, int], dict[str, Any]],
]:
    source_config = source.ScalarInterfaceConfig(
        source_root=config.source_root,
        device="cpu",
    )
    return source._source_details(source_config)


def _analysis_datasets(
    task: CircleTaskConfig, config: JointPhysicalInterfaceConfig
) -> dict[str, calibrated.CalibratedDataset]:
    counts = {
        "train": config.probe_train_samples,
        "validation": config.probe_validation_samples,
        "in_distribution": config.probe_test_samples,
        "composition": config.probe_test_samples,
        "extrapolation": config.probe_test_samples,
    }
    return {
        split: calibrated.generate_calibrated_dataset(
            task,
            sample_count=counts[split],
            seed=ANALYSIS_SPECS[split][0],
            regime=ANALYSIS_SPECS[split][1],
            shuffle=True,
        )
        for split in ANALYSIS_SPLITS
    }


def _analysis_dataset_hashes(
    datasets: Mapping[str, calibrated.CalibratedDataset],
) -> dict[str, str]:
    return {name: source.closure._dataset_hash(value) for name, value in datasets.items()}


def _source_config(
    config: JointPhysicalInterfaceConfig, preset: str
) -> calibrated.CalibratedFrontendConfig:
    return calibrated.CalibratedFrontendConfig(
        preset=preset,
        seeds=config.seeds,
        training_steps=config.training_steps,
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        vector_channels=16,
        analysis_seed=config.analysis_seed,
        probe_train_samples=config.probe_train_samples,
        probe_validation_samples=config.probe_validation_samples,
        probe_test_samples=config.probe_test_samples,
        activation_batch_size=config.activation_batch_size,
        probe_batch_size=config.probe_batch_size,
        probe_steps=config.probe_steps,
        probe_width=config.probe_width,
        validation_interval=config.validation_interval,
        early_stopping_patience=config.early_stopping_patience,
        allow_underpowered=True,
    )


def _sealed_training_protocol_config(
    preset: str,
) -> calibrated.CalibratedFrontendConfig:
    """Reconstruct the complete source schedule before taking lifecycle slices."""
    return calibrated.CalibratedFrontendConfig(
        preset=preset,
        seeds=SEEDS,
        training_steps=600,
        train_samples=4_096,
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=0.01,
        gradient_clip=1.0,
        vector_channels=16,
        allow_underpowered=True,
    )


def _load_trainable_interface(
    source_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: JointPhysicalInterfaceConfig,
    preset: str,
    condition: str,
    device: torch.device,
) -> PhysicalScalarInterface:
    system = source._load_system(source_detail, task, preset, condition, device)
    if system.scalar_embedding is None:
        raise AssertionError("source scalar embedding missing")
    for parameter in system.model.parameters():
        parameter.requires_grad_(False)
    system.scalar_embedding.requires_grad_(True)
    if condition == "learned_calibrated_equivariant":
        if system.encoder is None:
            raise AssertionError("learned source encoder missing")
        system.encoder.requires_grad_(True)
    elif system.encoder is not None:
        raise AssertionError("analytic source unexpectedly has a learned encoder")
    interface = PhysicalScalarInterface(system).to(device)
    allowed_prefixes = ("system.scalar_embedding.", "final_scalar.")
    if condition == "learned_calibrated_equivariant":
        allowed_prefixes = (*allowed_prefixes, "system.encoder.")
    for name, parameter in interface.named_parameters():
        expected = name.startswith(allowed_prefixes)
        if parameter.requires_grad != expected:
            raise AssertionError(f"unexpected gradient route for {name}")
    return interface


def _trainable_parameters(
    interface: PhysicalScalarInterface,
) -> tuple[list[tuple[str, nn.Parameter]], int]:
    selected = [
        (name, parameter)
        for name, parameter in interface.named_parameters()
        if parameter.requires_grad
    ]
    if not selected:
        raise AssertionError("joint interface has no trainable parameters")
    return selected, sum(parameter.numel() for _, parameter in selected)


def _training_targets(
    cosine: torch.Tensor, permutation: torch.Tensor, arm: str
) -> torch.Tensor:
    if arm == "physical_true":
        return cosine.reshape(-1)
    if arm == "pair_shuffled":
        return cosine.reshape(-1)[permutation]
    raise ValueError(f"unknown arm: {arm}")


def train_arm(
    source_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: JointPhysicalInterfaceConfig,
    preset: str,
    condition: str,
    seed: int,
    arm: str,
    training_dataset: calibrated.CalibratedDataset,
    pair_batches: torch.Tensor,
    permutation: torch.Tensor,
    device: torch.device,
) -> tuple[PhysicalScalarInterface, dict[str, Any]]:
    torch.manual_seed(seed + 90_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 90_001)
    interface = _load_trainable_interface(
        source_detail, task, config, preset, condition, device
    )
    initial_model = calibrated._state_digest(interface.system.model)
    initial_interface = _interface_state_digest(interface)
    selected, parameter_count = _trainable_parameters(interface)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    circle = training_dataset.paired.circle
    sensor = calibrated.decode_sensor_tokens(circle.input_ids, task)
    target_scalar = _training_targets(
        training_dataset.paired.fiber.cosine, permutation, arm
    )
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    interface.train()
    for step in range(1, config.training_steps + 1):
        pairs = pair_batches[step - 1]
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        ids = circle.input_ids[indices].to(device, non_blocking=True)
        observed = sensor[indices].to(device, non_blocking=True)
        packet = training_dataset.calibration[indices].to(device, non_blocking=True)
        target = target_scalar[indices].to(device, non_blocking=True).reshape(-1, 1)
        target_posterior = interval_posterior_unclipped(target, task.phase_bins)
        optimizer.zero_grad(set_to_none=True)
        scalars = interface.forward_scalars(ids, observed, packet)
        predicted_posterior = interval_posterior_unclipped(
            scalars["full"], task.phase_bins
        )
        task_loss = -(
            target_posterior * predicted_posterior.clamp_min(1e-12).log()
        ).sum(-1).mean()
        sensor_loss = F.mse_loss(scalars["frontend"], target)
        final_loss = F.mse_loss(scalars["full"], target)
        loss = (
            config.task_loss_weight * task_loss
            + config.sensor_loss_weight * sensor_loss
            + config.final_loss_weight * final_loss
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in selected], config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            history.append(
                {
                    "step": float(step),
                    "total_loss": float(loss.detach()),
                    "task_cross_entropy": float(task_loss.detach()),
                    "sensor_mse": float(sensor_loss.detach()),
                    "final_mse": float(final_loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_model = calibrated._state_digest(interface.system.model)
    final_interface = _interface_state_digest(interface)
    if final_model != initial_model:
        raise RuntimeError("frozen TinyLLM backbone changed during interface training")
    return interface, {
        "arm": arm,
        "objective": (
            "fixed_interval_cross_entropy + sensor_physical_cosine_mse + "
            "final_physical_cosine_mse"
        ),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "trainable_parameter_names": [name for name, _ in selected],
        "trainable_parameter_count": parameter_count,
        "initial_model_state_sha256": initial_model,
        "final_model_state_sha256": final_model,
        "initial_interface_state_sha256": initial_interface,
        "final_interface_state_sha256": final_interface,
    }


def _checkpoint_payload(
    interface: PhysicalScalarInterface,
    preset: str,
    condition: str,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    if interface.system.scalar_embedding is None:
        raise AssertionError("scalar embedding missing at save")
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "arm": arm,
        "encoder": (
            interface.system.encoder.to("cpu").state_dict()
            if interface.system.encoder is not None
            else None
        ),
        "scalar_embedding": interface.system.scalar_embedding.to("cpu").state_dict(),
        "final_scalar": interface.final_scalar.to("cpu").state_dict(),
    }


def save_interface_checkpoint(
    interface: PhysicalScalarInterface,
    path: Path,
    preset: str,
    condition: str,
    seed: int,
    arm: str,
    device: torch.device,
) -> dict[str, Any]:
    expected_digest = _interface_state_digest(interface)
    payload = _checkpoint_payload(interface, preset, condition, seed, arm)
    torch.save(payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    if (
        reloaded.get("schema_version") != SCHEMA_VERSION
        or reloaded.get("hypothesis_id") != HYPOTHESIS_ID
        or reloaded.get("preset") != preset
        or reloaded.get("condition") != condition
        or int(reloaded.get("seed", -1)) != seed
        or reloaded.get("arm") != arm
    ):
        raise RuntimeError("interface checkpoint metadata failed reload")
    interface.to(device)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "state_sha256": expected_digest,
        "reload_pass": True,
    }


@torch.inference_mode()
def extract_scalars(
    interface: PhysicalScalarInterface,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: JointPhysicalInterfaceConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    values: dict[str, list[torch.Tensor]] = {cut: [] for cut in CUTS}
    interface.eval()
    for start in range(0, len(sensor), config.activation_batch_size):
        stop = min(len(sensor), start + config.activation_batch_size)
        batch = interface.forward_scalars(
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        for cut in CUTS:
            values[cut].append(batch[cut].reshape(-1).float().cpu())
    return {cut: torch.cat(parts) for cut, parts in values.items()}


@torch.inference_mode()
def source_task_replay(
    source_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: JointPhysicalInterfaceConfig,
    preset: str,
    condition: str,
    datasets: Mapping[str, calibrated.CalibratedDataset],
    device: torch.device,
) -> dict[str, Any]:
    system = source._load_system(source_detail, task, preset, condition, device)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    records: dict[str, Any] = {}
    maximum = 0.0
    same_cohort = config.probe_test_samples == 1_024
    for regime in REGIMES:
        dataset = datasets[regime]
        sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
        posteriors = []
        for start in range(0, len(sensor), config.activation_batch_size):
            stop = min(len(sensor), start + config.activation_batch_size)
            residual = system.forward_cuts(
                dataset.paired.circle.input_ids[start:stop].to(device),
                sensor[start:stop].to(device),
                dataset.calibration[start:stop].to(device),
            )["full"]
            logits = calibrated._task_logits(system.model, residual, answer_ids)
            posteriors.append(torch.softmax(logits, -1).float().cpu())
        measured = interval.task_metrics(
            torch.cat(posteriors), dataset.paired.circle.target_posteriors
        )
        stored = source_detail["representation"]["task_metrics"][regime]
        error = max(
            abs(measured["exact_bin_accuracy"] - float(stored["exact_bin_accuracy"])),
            abs(
                measured["mean_target_cross_entropy"]
                - float(stored["mean_target_cross_entropy"])
            ),
        )
        if same_cohort:
            maximum = max(maximum, error)
        records[regime] = {
            "measured": measured,
            "stored": {
                "exact_bin_accuracy": float(stored["exact_bin_accuracy"]),
                "mean_target_cross_entropy": float(
                    stored["mean_target_cross_entropy"]
                ),
            },
            "stored_metric_replay_error": error if same_cohort else None,
            "stored_metrics_same_cohort": same_cohort,
            "role": (
                "required_primary_same_cohort_replay"
                if same_cohort
                else "not_comparable_reduced_lifecycle_cohort"
            ),
        }
    del system
    return {"regimes": records, "maximum_absolute_error": maximum}


def _probe_config(config: JointPhysicalInterfaceConfig, preset: str):
    return calibrated._analysis_config(_source_config(config, preset))


def _affine_calibration(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    x = target.reshape(-1).double()
    y = predicted.reshape(-1).double()
    design = torch.stack((x, torch.ones_like(x)), dim=1)
    solution = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    return {"slope": float(solution[0]), "intercept": float(solution[1])}


def endpoint_pass(
    record: Mapping[str, Any], config: JointPhysicalInterfaceConfig
) -> bool:
    return bool(
        float(record["scalar_metrics"]["cosine_pearson"])
        >= config.cosine_minimum
        and float(record["conditional_branch"]["balanced_accuracy"])
        <= config.branch_accuracy_maximum
        and float(
            record["conditional_branch"][
                "conditional_log_loss_gain_over_cosine_only"
            ]
        )
        <= config.conditional_log_loss_gain_maximum
        and bool(record["task_gate"])
    )


def analyze_arm(
    interface: PhysicalScalarInterface,
    task: CircleTaskConfig,
    config: JointPhysicalInterfaceConfig,
    preset: str,
    seed: int,
    arm: str,
    datasets: Mapping[str, calibrated.CalibratedDataset],
    task_floors: Mapping[str, float],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scalars = {
        split: extract_scalars(interface, dataset, task, config, device)
        for split, dataset in datasets.items()
    }
    evaluation_fibers = {regime: datasets[regime].paired.fiber for regime in REGIMES}
    probe_config = _probe_config(config, preset)
    null = fit_cosine_only_null(
        datasets["train"].paired.fiber,
        datasets["validation"].paired.fiber,
        evaluation_fibers,
        probe_config,
        device,
        kind="nonlinear",
        seed=config.analysis_seed + seed * 101,
    )
    probes: dict[str, Any] = {}
    for cut_index, cut in enumerate(CUTS):
        probe = fit_conditional_probe(
            scalars["train"][cut][:, None].numpy(),
            scalars["validation"][cut][:, None].numpy(),
            {regime: scalars[regime][cut][:, None].numpy() for regime in REGIMES},
            datasets["train"].paired.fiber,
            datasets["validation"].paired.fiber,
            evaluation_fibers,
            probe_config,
            device,
            kind="nonlinear",
            seed=(
                config.analysis_seed
                + seed * 10_007
                + cut_index * 1_009
                + (0 if arm == "physical_true" else 500_009)
            ),
        )
        _add_log_loss_gains(probe, null)
        probes[cut] = probe

    cuts: dict[str, Any] = {cut: {} for cut in CUTS}
    arrays: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        dataset = datasets[regime]
        exact_cosine = dataset.paired.fiber.cosine.reshape(-1)
        target_posterior = dataset.paired.circle.target_posteriors
        arrays[f"{regime}__exact_cosine"] = exact_cosine.numpy()
        arrays[f"{regime}__branch"] = dataset.paired.fiber.branch.numpy()
        arrays[f"{regime}__fiber_id"] = dataset.paired.fiber.fiber_id.numpy()
        arrays[f"{regime}__target_posterior"] = target_posterior.numpy()
        for cut in CUTS:
            predicted = scalars[regime][cut]
            posterior = interval_posterior_unclipped(
                predicted.double(), task.phase_bins
            ).cpu()
            scalar_record = interval.scalar_metrics(
                predicted, exact_cosine, dataset.paired.fiber.fiber_id
            )
            scalar_record.update(_affine_calibration(predicted, exact_cosine))
            task_record = interval.task_metrics(posterior, target_posterior)
            branch_record = dict(probes[cut]["evaluations"][regime])
            branch_record["probe_cosine_pearson"] = branch_record.pop(
                "cosine_pearson"
            )
            branch_record["probe_cosine_rmse"] = branch_record.pop("cosine_rmse")
            record = {
                "scalar_metrics": scalar_record,
                "task_metrics": task_record,
                "task_accuracy_floor": float(task_floors[regime]),
                "task_gate": (
                    float(task_record["exact_bin_accuracy"])
                    >= float(task_floors[regime])
                ),
                "conditional_branch": branch_record,
            }
            record["endpoint_pass"] = endpoint_pass(record, config)
            cuts[cut][regime] = record
            arrays[f"{regime}__{cut}__scalar"] = predicted.numpy()
            arrays[f"{regime}__{cut}__posterior"] = posterior.numpy()
    joint = all(
        cuts[cut][regime]["endpoint_pass"] for cut in CUTS for regime in REGIMES
    )
    return {
        "cuts": cuts,
        "probe_training": {
            cut: {
                key: value
                for key, value in probes[cut].items()
                if key != "evaluations"
            }
            for cut in CUTS
        },
        "joint_seed_pass": bool(joint),
    }, arrays


def _fingerprint(experiment: Experiment) -> str:
    keys = (
        "configuration",
        "task_config",
        "preset",
        "condition",
        "seed",
        "implementation_sha256",
        "source_result_sha256",
        "comparator_campaign_sha256",
        "analysis_dataset_hashes",
        "task_accuracy_floors",
    )
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            **{key: experiment.parameters[key] for key in keys},
        }
    )


def joint_interface_worker(
    experiment: Experiment, device_id: int
) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    preset = str(experiment.parameters["preset"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    implementation_sources = _implementation_sources()
    implementation = _implementation_digest(implementation_sources)
    if implementation != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    _, source_task, source_details = _source_population(config)
    if asdict(source_task) != asdict(task):
        raise RuntimeError("source task changed inside worker")
    source_detail = source_details[(preset, condition, seed)]
    if source_detail["_result_sha256"] != experiment.parameters["source_result_sha256"]:
        raise RuntimeError("source detail changed inside worker")
    _verify_comparator(Path(config.comparator_root))

    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    output_root = Path(experiment.parameters["output_dir"])
    cell = _cell_directory(output_root, preset, condition, seed)
    cell.mkdir(parents=True, exist_ok=True)

    datasets = _analysis_datasets(task, config)
    dataset_hashes = _analysis_dataset_hashes(datasets)
    if dataset_hashes != experiment.parameters["analysis_dataset_hashes"]:
        raise RuntimeError("analysis cohorts changed inside worker")
    sealed_protocol = _sealed_training_protocol_config(preset)
    training_dataset, sealed_pair_batches, training_hash, batch_hash = (
        calibrated._protocol_material(task, sealed_protocol, seed)
    )
    if (
        training_hash != source_detail["training"]["training_data_sha256"]
        or batch_hash != source_detail["training"]["minibatch_schedule_sha256"]
    ):
        raise RuntimeError("source training tensors or minibatches changed")
    if (
        config.training_steps > len(sealed_pair_batches)
        or config.batch_size > 2 * sealed_pair_batches.shape[1]
    ):
        raise RuntimeError("requested lifecycle slice exceeds the sealed source schedule")
    pair_batches = sealed_pair_batches[
        : config.training_steps, : config.batch_size // 2
    ].clone()
    permutation = pair_preserving_target_permutation(
        config.train_samples, preset, condition, seed, config.shuffle_seed
    )
    target_permutation_hash = calibrated._tensor_digest(permutation)
    replay = source_task_replay(
        source_detail, task, config, preset, condition, datasets, device
    )
    task_floors = {
        key: float(value)
        for key, value in experiment.parameters["task_accuracy_floors"].items()
    }

    arm_records: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "training_pair_batches": pair_batches.numpy(),
        "pair_shuffled_target_permutation": permutation.numpy(),
    }
    checkpoint_records: dict[str, Any] = {}
    source_model_digest = source_detail["training"]["final_model_state_sha256"]
    for arm in ARMS:
        interface, training = train_arm(
            source_detail,
            task,
            config,
            preset,
            condition,
            seed,
            arm,
            training_dataset,
            pair_batches,
            permutation,
            device,
        )
        analysis, arm_arrays = analyze_arm(
            interface,
            task,
            config,
            preset,
            seed,
            arm,
            datasets,
            task_floors,
            device,
        )
        checkpoint = save_interface_checkpoint(
            interface,
            _arm_checkpoint_path(cell, arm),
            preset,
            condition,
            seed,
            arm,
            device,
        )
        model_unchanged = (
            training["initial_model_state_sha256"]
            == training["final_model_state_sha256"]
            == source_model_digest
        )
        arm_records[arm] = {
            "training": training,
            "analysis": analysis,
            "checkpoint": checkpoint,
            "model_unchanged": model_unchanged,
        }
        checkpoint_records[arm] = checkpoint
        arrays.update(
            {f"{arm}__{name}": value for name, value in arm_arrays.items()}
        )
        del interface
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    diagnostics_path = cell / "diagnostics.npz"
    _write_npz(diagnostics_path, arrays)
    with np.load(diagnostics_path, allow_pickle=False) as reloaded:
        diagnostics_reload = bool(
            set(reloaded.files) == set(arrays)
            and all(np.array_equal(reloaded[name], arrays[name]) for name in arrays)
        )
    finite = source._finite({"replay": replay, "arms": arm_records}) and all(
        np.isfinite(value).all() for value in arrays.values()
    )
    validity = bool(
        replay["maximum_absolute_error"] <= config.replay_tolerance
        and all(record["model_unchanged"] for record in arm_records.values())
        and all(record["checkpoint"]["reload_pass"] for record in arm_records.values())
        and diagnostics_reload
        and finite
    )
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result_path = cell / "result.json"
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "implementation_sha256": implementation,
        "implementation_sources": implementation_sources,
        "scientific_fingerprint": _fingerprint(experiment),
        "source": {
            "result": source_detail["_result_path"],
            "result_sha256": source_detail["_result_sha256"],
            "model_checkpoint": source_detail["artifacts"]["checkpoint"],
            "model_checkpoint_sha256": source_detail["artifacts"][
                "checkpoint_sha256"
            ],
            "frontend_checkpoint": source_detail["artifacts"][
                "frontend_checkpoint"
            ],
            "frontend_checkpoint_sha256": source_detail["artifacts"][
                "frontend_checkpoint_sha256"
            ],
            "training_data_sha256": training_hash,
            "minibatch_schedule_sha256": batch_hash,
            "task_replay": replay,
        },
        "target_permutation_sha256": target_permutation_hash,
        "analysis_dataset_hashes": dataset_hashes,
        "task_accuracy_floors": task_floors,
        "arms": arm_records,
        "gates": {
            "physical_true_joint_seed_pass": bool(
                arm_records["physical_true"]["analysis"]["joint_seed_pass"]
            ),
            "pair_shuffled_joint_seed_pass": bool(
                arm_records["pair_shuffled"]["analysis"]["joint_seed_pass"]
            ),
            "source_replay": replay["maximum_absolute_error"]
            <= config.replay_tolerance,
            "frozen_model_unchanged": all(
                record["model_unchanged"] for record in arm_records.values()
            ),
            "checkpoint_reload": all(
                record["checkpoint"]["reload_pass"]
                for record in arm_records.values()
            ),
            "diagnostics_reload": diagnostics_reload,
            "finite": finite,
            "validity": validity,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gb": peak,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
            "interfaces": checkpoint_records,
        },
    }
    _write_json(result_path, detail)
    width = preflight.TINYLLM_PRESETS[preset][2]
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics={
            "physical_true_joint_seed_pass": float(
                detail["gates"]["physical_true_joint_seed_pass"]
            ),
            "pair_shuffled_joint_seed_pass": float(
                detail["gates"]["pair_shuffled_joint_seed_pass"]
            ),
            "validity": float(validity),
            "peak_cuda_allocated_gb": peak,
        },
        primary_metric=float(detail["gates"]["physical_true_joint_seed_pass"]),
        model_architecture=[
            task.vocab_size,
            *([width] * preflight.TINYLLM_PRESETS[preset][0]),
            1,
        ],
        model_parameters=int(
            arm_records["physical_true"]["training"]["trainable_parameter_count"]
        ),
        training_time=float(detail["wall_seconds"]),
        training_history=arm_records["physical_true"]["training"]["history"],
        model_checkpoint=checkpoint_records["physical_true"]["path"],
        observations=[
            f"detail={result_path}",
            f"preset={preset}",
            f"condition={condition}",
        ],
    )


def classify_aggregates(
    strata: Mapping[str, Mapping[str, Any]], valid: bool
) -> tuple[str, bool, bool]:
    if not valid:
        return "invalid", False, False
    analytic = all(
        bool(strata[f"{preset}/analytic_calibrated"]["family_gate"])
        for preset in PRESETS
    )
    learned = all(
        bool(strata[f"{preset}/learned_calibrated_equivariant"]["family_gate"])
        for preset in PRESETS
    )
    specificity = all(
        bool(record["shuffled_specificity_gate"]) for record in strata.values()
    )
    if not specificity:
        return "specificity_control_failed", False, False
    if not analytic:
        return "analytic_positive_control_failed", False, False
    if learned:
        return (
            "frozen_backbone_joint_physical_interface_architecture_stable",
            True,
            False,
        )
    return "frozen_backbone_joint_interface_insufficient", False, True


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: JointPhysicalInterfaceConfig
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for preset in config.presets:
        for condition in config.conditions:
            selected = sorted(
                (
                    detail
                    for detail in details
                    if detail["preset"] == preset and detail["condition"] == condition
                ),
                key=lambda detail: detail["seed"],
            )
            if len(selected) != len(config.seeds):
                raise ValueError(f"incomplete stratum {preset}/{condition}")
            true_count = sum(
                bool(detail["gates"]["physical_true_joint_seed_pass"])
                for detail in selected
            )
            shuffled_count = sum(
                bool(detail["gates"]["pair_shuffled_joint_seed_pass"])
                for detail in selected
            )
            specificity = shuffled_count <= config.shuffled_seed_pass_ceiling
            strata[f"{preset}/{condition}"] = {
                "valid_count": sum(
                    bool(detail["gates"]["validity"]) for detail in selected
                ),
                "physical_true_pass_count": true_count,
                "physical_true_pass_by_seed": {
                    str(detail["seed"]): bool(
                        detail["gates"]["physical_true_joint_seed_pass"]
                    )
                    for detail in selected
                },
                "pair_shuffled_pass_count": shuffled_count,
                "pair_shuffled_pass_by_seed": {
                    str(detail["seed"]): bool(
                        detail["gates"]["pair_shuffled_joint_seed_pass"]
                    )
                    for detail in selected
                },
                "shuffled_specificity_gate": specificity,
                "family_gate": bool(
                    true_count >= config.required_seed_passes and specificity
                ),
            }
    complete = bool(
        not config.allow_underpowered
        and len(details)
        == len(config.presets) * len(config.conditions) * len(config.seeds)
    )
    valid = bool(
        all(detail["gates"]["validity"] for detail in details)
        and (complete or config.allow_underpowered)
    )
    if config.allow_underpowered:
        classification = "systems_lifecycle_only_not_scientific_evidence"
        primary = False
        extension = False
    else:
        classification, primary, extension = classify_aggregates(strata, valid)
    return {
        "valid": valid,
        "complete_primary_population": complete,
        "classification": classification,
        "primary_hypothesis_pass": primary,
        "full_interface_extension_licensed": extension,
        "stop_before_transformer_finetuning": primary,
        "strata": strata,
        "preregistered_gate": {
            "required_seed_passes": config.required_seed_passes,
            "shuffled_seed_pass_ceiling": config.shuffled_seed_pass_ceiling,
            "cosine_minimum": config.cosine_minimum,
            "branch_accuracy_maximum": config.branch_accuracy_maximum,
            "conditional_log_loss_gain_maximum": (
                config.conditional_log_loss_gain_maximum
            ),
            "cuts": list(CUTS),
            "regimes": list(REGIMES),
        },
    }


def _task_floors(source_detail: Mapping[str, Any]) -> dict[str, float]:
    return {
        regime: float(source_detail["gates"]["task_accuracy_floors"][regime])
        for regime in REGIMES
    }


def _experiments(
    config: JointPhysicalInterfaceConfig,
    task: CircleTaskConfig,
    source_details: Mapping[tuple[str, str, int], Mapping[str, Any]],
    output_dir: Path,
    implementation: str,
    analysis_dataset_hashes: Mapping[str, str],
) -> list[Experiment]:
    common = {
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "implementation_sha256": implementation,
        "comparator_campaign_sha256": COMPARATOR_CAMPAIGN_SHA256,
        "analysis_dataset_hashes": dict(analysis_dataset_hashes),
    }
    return [
        Experiment(
            id=f"tinyllm-joint-physical-interface-{preset}-{condition}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM joint physical interface {preset} {condition} seed {seed}",
            parameters={
                **common,
                "preset": preset,
                "condition": condition,
                "seed": seed,
                "source_result_sha256": source_details[(preset, condition, seed)][
                    "_result_sha256"
                ],
                "task_accuracy_floors": _task_floors(
                    source_details[(preset, condition, seed)]
                ),
            },
            seed=seed,
        )
        for preset in config.presets
        for condition in config.conditions
        for seed in config.seeds
    ]


def _existing_detail(
    experiment: Experiment, output_dir: Path
) -> Optional[dict[str, Any]]:
    path = _cell_directory(
        output_dir,
        str(experiment.parameters["preset"]),
        str(experiment.parameters["condition"]),
        int(experiment.seed),
    ) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint") != _fingerprint(experiment)
        or detail.get("gates", {}).get("validity") is not True
    ):
        return None
    diagnostics = Path(str(detail.get("artifacts", {}).get("diagnostics", "")))
    if (
        not diagnostics.is_file()
        or _sha256(diagnostics)
        != detail.get("artifacts", {}).get("diagnostics_sha256")
    ):
        return None
    for arm in ARMS:
        checkpoint = detail.get("artifacts", {}).get("interfaces", {}).get(arm, {})
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if (
            not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint.get("sha256")
        ):
            return None
    return detail


def _campaign_fingerprint(
    config: JointPhysicalInterfaceConfig,
    task: CircleTaskConfig,
    implementation: str,
    source_manifest: str,
    analysis_dataset_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "task_config": asdict(task),
            "implementation_sha256": implementation,
            "source_manifest_sha256": source_manifest,
            "comparator_campaign_sha256": COMPARATOR_CAMPAIGN_SHA256,
            "analysis_dataset_hashes": dict(analysis_dataset_hashes),
        }
    )


async def run_campaign(
    config: JointPhysicalInterfaceConfig, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_sources = _implementation_sources()
    implementation = _implementation_digest(implementation_sources)
    source_campaign, task, source_details = _source_population(config)
    comparator = _verify_comparator(Path(config.comparator_root))
    datasets = _analysis_datasets(task, config)
    dataset_hashes = _analysis_dataset_hashes(datasets)
    if not config.allow_underpowered:
        for regime in REGIMES:
            if dataset_hashes[regime] != source.EXPECTED_DATASET_HASHES[regime]:
                raise RuntimeError("primary held-out cohort hashes changed")
    selected_source_paths = [
        Path(source_details[(preset, condition, seed)]["_result_path"])
        for preset in config.presets
        for condition in config.conditions
        for seed in config.seeds
    ]
    source_manifest = _manifest_sha256(selected_source_paths)
    experiments = _experiments(
        config,
        task,
        source_details,
        output_dir,
        implementation,
        dataset_hashes,
    )
    fingerprint = _campaign_fingerprint(
        config, task, implementation, source_manifest, dataset_hashes
    )
    campaign_path = output_dir / "campaign_results.json"
    if campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing_campaign.get("status") == "completed"
            and existing_campaign.get("campaign_fingerprint") == fingerprint
            and all(_existing_detail(item, output_dir) is not None for item in experiments)
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing_campaign

    existing = {
        experiment.id: detail
        for experiment in experiments
        if (detail := _existing_detail(experiment, output_dir)) is not None
    }
    pending = [item for item in experiments if item.id not in existing]
    lab = LabConfig(
        project_name="tinyllm_joint_physical_scalar_interface",
        results_dir=str(output_dir),
        device_ids=list(config.device_ids),
        max_parallel_experiments=config.max_parallel_experiments,
        gpu_slots_per_device=config.gpu_slots_per_device,
        gpu_memory_per_experiment_gb=config.gpu_memory_per_experiment_gb,
        max_gpu_slots_per_device=config.max_gpu_slots_per_device,
        max_experiment_retries=config.max_retries,
        resume_completed_experiments=config.resume,
        auto_balance=False,
        enable_wandb=False,
        verbose=True,
    )
    runner = AsyncExperimentRunner(lab, joint_interface_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful = {result.experiment_id for result in results if result.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id in successful:
            detail = _existing_detail(experiment, output_dir)
            if detail is not None:
                details.append(detail)
    complete = len(details) == len(experiments)
    result_paths = [Path(detail["artifacts"]["result"]) for detail in details]
    diagnostics_paths = [Path(detail["artifacts"]["diagnostics"]) for detail in details]
    interface_paths = [
        Path(detail["artifacts"]["interfaces"][arm]["path"])
        for detail in details
        for arm in ARMS
    ]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if complete else "partial",
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "implementation_sha256": implementation,
        "implementation_sources": implementation_sources,
        "campaign_fingerprint": fingerprint,
        "source": {
            "campaign": str(Path(config.source_root) / "campaign_results.json"),
            "campaign_sha256": source.SOURCE_CAMPAIGN_SHA256,
            "selected_result_manifest_sha256": source_manifest,
            "source_campaign_fingerprint": source_campaign.get(
                "campaign_fingerprint"
            ),
        },
        "comparator": {
            "campaign": str(Path(config.comparator_root) / "campaign_results.json"),
            "campaign_sha256": COMPARATOR_CAMPAIGN_SHA256,
            "classification": comparator["aggregates"]["classification"],
            "strata": comparator["aggregates"]["strata"],
        },
        "analysis_dataset_hashes": dataset_hashes,
        "summary": {
            "requested_source_cells": len(experiments),
            "requested_interface_fits": 2 * len(experiments),
            "reused_source_cells": len(existing),
            "scheduled_source_cells": len(pending),
            "completed_source_cells": len(details),
            "failed_source_cells": len(experiments) - len(details),
            "completed_interface_fits": 2 * len(details),
            "frozen_backbones": len(details),
        },
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value
                for key, value in runner.slot_plan.slots_by_device.items()
            },
            "calibration": runner.slot_plan.calibration,
        },
        "aggregates": aggregate_details(details, config) if complete else {},
        "result_manifest_sha256": (
            _manifest_sha256(result_paths) if result_paths else None
        ),
        "diagnostics_manifest_sha256": (
            _manifest_sha256(diagnostics_paths) if diagnostics_paths else None
        ),
        "interface_manifest_sha256": (
            _manifest_sha256(interface_paths) if interface_paths else None
        ),
        "results": [
            {
                "experiment_id": result.experiment_id,
                "status": result.status.value,
                "metrics": result.metrics,
                "error": result.error,
                "model_checkpoint": result.model_checkpoint,
            }
            for result in results
        ],
        "method_boundaries": [
            "All transformer, token, position, layer-normalization, and LM-head parameters are frozen.",
            "The learned encoder and scalar embedding begin from outcome-known source checkpoints.",
            "The affine final scalar extractor is initialized to zero and supervised as physical cosine.",
            "The pair-shuffled control preserves the two-sheet quotient pairing and target marginal.",
            "The source frozen-readout campaign is a sealed external comparator, not a pooled arm.",
            "d6 and d10 jointly vary depth, width, and attention-head count.",
        ],
    }
    _write_json(campaign_path, bundle)
    return bundle


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_devices(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "cpu":
        return (-1,)
    if value.strip().lower() == "auto":
        return tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else (-1,)
    return _comma_ints(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--comparator-root", default=str(COMPARATOR_ROOT))
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--probe-train-samples", type=int, default=2_048)
    parser.add_argument("--probe-validation-samples", type=int, default=512)
    parser.add_argument("--probe-test-samples", type=int, default=1_024)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--concurrency-shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_joint_physical_scalar_interface/"
            "20260811_d6_d10_preregistered"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown and args.concurrency_shakedown:
        raise ValueError("choose only one shakedown mode")
    lifecycle = args.shakedown or args.concurrency_shakedown
    if lifecycle:
        args.presets = "d6"
        args.conditions = "analytic_calibrated"
        args.seeds = "7,17,29" if args.concurrency_shakedown else "7"
        args.steps = 2
        args.train_samples = 4_096
        args.batch_size = 8
        args.probe_steps = 4
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 64
        args.allow_underpowered = True
    config = JointPhysicalInterfaceConfig(
        source_root=args.source_root,
        comparator_root=args.comparator_root,
        presets=_comma_strings(args.presets),
        conditions=_comma_strings(args.conditions),
        seeds=_comma_ints(args.seeds),
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
        probe_steps=args.probe_steps,
        required_seed_passes=1 if lifecycle else 4,
        device_ids=_parse_devices(args.gpus),
        gpu_slots_per_device=args.slots_per_gpu,
        max_gpu_slots_per_device=args.slots_per_gpu,
        max_parallel_experiments=args.max_parallel,
        max_retries=args.retries,
        resume=args.resume,
        allow_underpowered=args.allow_underpowered,
    )
    result = asyncio.run(run_campaign(config, args.output))
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result.get("aggregates", {}).get(
                    "classification", "partial"
                ),
                "summary": result["summary"],
                "output": str(args.output / "campaign_results.json"),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
