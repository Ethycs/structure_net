#!/usr/bin/env python3
"""Fine-tune the complete TinyLLM physical-scalar interface under matched controls."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import experiments.structure_net.tinyllm_joint_interface_block_clipping as block
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-joint-full-interface.v1"
HYPOTHESIS_ID = "tinyllm-joint-full-interface-physical-typing-v1"
EVIDENCE_ROLE = "prospective_full_interface_physical_typing"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-full-interface-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "1921afdeb0fca28c80ff9fb151c767dd6b8baa8aac39967b3ae614bef5df9329"
)
STAGE_A_ROOT = block.STAGE_A_ROOT
STAGE_A_CAMPAIGN_SHA256 = block.STAGE_A_CAMPAIGN_SHA256
BLOCK_ROOT = Path(
    "data/experiments/tinyllm_joint_interface_block_clipping/"
    "20260811_d6_d10_preregistered"
)
BLOCK_CAMPAIGN_SHA256 = (
    "2f7c7cdd5494322ff89e20fb55407c6d4d8de66dde852ca9a8ec67fbc22a2349"
)
BLOCK_RESULT_MANIFEST_SHA256 = (
    "a78a57e3a2bf0f44946a8b3081a64a3bb915ef1e3426c068a765bba70b0e6d69"
)
JOINT_RUNNER_SHA256 = (
    "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
)
BLOCK_RUNNER_SHA256 = (
    "7dc67af033523e7819b0f7cadeac9848508ce2cbe9cce201bef67c6e26dcacf6"
)
PRESETS = block.PRESETS
CONDITION = block.CONDITION
CONDITIONS = block.CONDITIONS
SEEDS = block.SEEDS
ARMS = joint.ARMS
CUTS = joint.CUTS
REGIMES = joint.REGIMES


@dataclass(frozen=True)
class FullInterfaceConfig:
    source_root: str = str(joint.SOURCE_ROOT)
    stage_a_root: str = str(STAGE_A_ROOT)
    block_root: str = str(BLOCK_ROOT)
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
    gpu_memory_per_experiment_gb: Optional[float] = 4.0
    max_gpu_slots_per_device: int = 1
    max_parallel_experiments: int = 1
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.presets or set(self.presets).difference(PRESETS):
            raise ValueError("unknown or empty preset selection")
        if self.conditions != CONDITIONS:
            raise ValueError("only the learned calibrated condition is registered")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if min(
            self.training_steps,
            self.train_samples,
            self.batch_size,
            self.probe_train_samples,
            self.probe_validation_samples,
            self.probe_test_samples,
            self.probe_steps,
            self.gradient_clip,
        ) <= 0:
            raise ValueError("registered sizes and clip must be positive")
        if self.batch_size % 2:
            raise ValueError("paired batch size must be even")
        if self.train_samples != 4_096:
            raise ValueError("the sealed target permutation requires 4096 examples")
        if (
            self.sensor_loss_weight != 1.0
            or self.final_loss_weight != 1.0
            or self.task_loss_weight != 1.0
        ):
            raise ValueError("registered loss weights are fixed to one")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed passes are outside the population")
        if not self.allow_underpowered:
            expected = {
                "source_root": str(joint.SOURCE_ROOT),
                "stage_a_root": str(STAGE_A_ROOT),
                "block_root": str(BLOCK_ROOT),
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
            actual = {key: getattr(self, key) for key in expected}
            if actual != expected:
                raise ValueError("primary full-interface configuration changed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return block._sha256(path)


def _json_hash(value: Any) -> str:
    return block._json_hash(value)


def _manifest_sha256(paths: Iterable[Path]) -> str:
    return block._manifest_sha256(paths)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    block._write_json(path, value)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    block._write_npz(path, arrays)


def _json_config(config: FullInterfaceConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(values: Mapping[str, Any]) -> FullInterfaceConfig:
    converted = dict(values)
    for field in ("presets", "conditions", "seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return FullInterfaceConfig(**converted)


def _joint_config(config: FullInterfaceConfig) -> joint.JointPhysicalInterfaceConfig:
    return joint.JointPhysicalInterfaceConfig(
        source_root=config.source_root,
        presets=config.presets,
        conditions=config.conditions,
        seeds=config.seeds,
        training_steps=config.training_steps,
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        sensor_loss_weight=config.sensor_loss_weight,
        final_loss_weight=config.final_loss_weight,
        task_loss_weight=config.task_loss_weight,
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
        cosine_minimum=config.cosine_minimum,
        branch_accuracy_maximum=config.branch_accuracy_maximum,
        conditional_log_loss_gain_maximum=(
            config.conditional_log_loss_gain_maximum
        ),
        required_seed_passes=config.required_seed_passes,
        shuffled_seed_pass_ceiling=config.shuffled_seed_pass_ceiling,
        shuffle_seed=config.shuffle_seed,
        replay_tolerance=config.replay_tolerance,
        device_ids=config.device_ids,
        gpu_slots_per_device=config.gpu_slots_per_device,
        gpu_memory_per_experiment_gb=config.gpu_memory_per_experiment_gb,
        max_gpu_slots_per_device=config.max_gpu_slots_per_device,
        max_parallel_experiments=config.max_parallel_experiments,
        max_retries=config.max_retries,
        resume=config.resume,
        allow_underpowered=True,
    )


def _block_config(config: FullInterfaceConfig) -> block.BlockClippingConfig:
    return block.BlockClippingConfig(
        source_root=config.source_root,
        stage_a_root=config.stage_a_root,
        presets=config.presets,
        conditions=config.conditions,
        seeds=config.seeds,
        training_steps=config.training_steps,
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        probe_train_samples=config.probe_train_samples,
        probe_validation_samples=config.probe_validation_samples,
        probe_test_samples=config.probe_test_samples,
        probe_steps=config.probe_steps,
        required_seed_passes=config.required_seed_passes,
        shuffled_seed_pass_ceiling=config.shuffled_seed_pass_ceiling,
        device_ids=config.device_ids,
        allow_underpowered=True,
    )


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "stage_a_runner": _sha256(Path(joint.__file__)),
        "block_clipping_runner": _sha256(Path(block.__file__)),
        "calibrated_frontend": _sha256(Path(joint.calibrated.__file__)),
        "source_validator": _sha256(Path(joint.source.__file__)),
        "interval_metrics": _sha256(Path(joint.interval.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("full-interface preregistration changed")
    if values["stage_a_runner"] != JOINT_RUNNER_SHA256:
        raise RuntimeError("Stage A runner changed")
    if values["block_clipping_runner"] != BLOCK_RUNNER_SHA256:
        raise RuntimeError("block-clipping runner changed")
    return values


def _implementation_digest(values: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(values or _implementation_sources()))


def _stage_a_population(
    config: FullInterfaceConfig,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    return block._stage_a_population(_block_config(config))


def _block_comparator(config: FullInterfaceConfig) -> dict[str, Any]:
    path = Path(config.block_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != BLOCK_CAMPAIGN_SHA256
        or campaign.get("schema_version") != block.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != block.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("result_manifest_sha256")
        != BLOCK_RESULT_MANIFEST_SHA256
        or campaign.get("aggregates", {}).get("valid") is not True
        or campaign.get("aggregates", {}).get("classification")
        != "parameter_block_clipping_insufficient"
        or campaign.get("aggregates", {}).get("full_interface_extension_licensed")
        is not True
    ):
        raise ValueError(f"invalid block-clipping comparator {path}")
    return campaign


def _state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_state_digest(model: nn.Module) -> str:
    return _state_dict_digest(model.state_dict())


def model_parameter_signature(model: nn.Module) -> dict[str, Any]:
    """Describe names, shapes, buffers, and aliasing without tensor values."""

    aliases: dict[int, int] = {}
    parameters = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity not in aliases:
            aliases[identity] = len(aliases)
        parameters.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "alias_group": aliases[identity],
            }
        )
    buffers = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in model.named_buffers()
    ]
    config = getattr(model, "config", None)
    if is_dataclass(config):
        config_value: Any = asdict(config)
    elif config is not None:
        config_value = dict(vars(config))
    else:
        config_value = None
    feedback = (
        model.get_feedback_topology()
        if hasattr(model, "get_feedback_topology")
        else None
    )
    return {
        "parameters": parameters,
        "buffers": buffers,
        "configuration": config_value,
        "feedback_topology": feedback,
        "refinement_steps": getattr(model, "refinement_steps", None),
    }


def _signature_digest(signature: Mapping[str, Any]) -> str:
    return _json_hash(signature)


def _load_full_interface(
    source_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: FullInterfaceConfig,
    preset: str,
    condition: str,
    device: torch.device,
) -> joint.PhysicalScalarInterface:
    interface = joint._load_trainable_interface(
        source_detail, task, _joint_config(config), preset, condition, device
    )
    model = interface.system.model
    if len(model.feedback_connections) != 0:
        raise AssertionError("registered source unexpectedly has feedback")
    model.requires_grad_(True)
    selected = {
        name for name, parameter in interface.named_parameters() if parameter.requires_grad
    }
    declared = {name for name, _parameter in interface.named_parameters()}
    if selected != declared:
        raise AssertionError("full interface does not expose every declared parameter")
    if interface.system.encoder is None or interface.system.scalar_embedding is None:
        raise AssertionError("learned full interface is incomplete")
    return interface


def trainable_parameters(
    interface: joint.PhysicalScalarInterface,
) -> tuple[list[tuple[str, nn.Parameter]], dict[str, Any]]:
    selected = [
        (name, parameter)
        for name, parameter in interface.named_parameters()
        if parameter.requires_grad
    ]
    all_named = list(interface.named_parameters())
    if [name for name, _parameter in selected] != [
        name for name, _parameter in all_named
    ]:
        raise AssertionError("not every unique full-interface parameter is trainable")
    groups = {
        "encoder": sum(
            parameter.numel()
            for name, parameter in selected
            if name.startswith("system.encoder.")
        ),
        "scalar_embedding": sum(
            parameter.numel()
            for name, parameter in selected
            if name.startswith("system.scalar_embedding.")
        ),
        "transformer": sum(
            parameter.numel()
            for name, parameter in selected
            if name.startswith("system.model.")
        ),
        "final_scalar": sum(
            parameter.numel()
            for name, parameter in selected
            if name.startswith("final_scalar.")
        ),
    }
    if any(value <= 0 for value in groups.values()):
        raise AssertionError("a declared full-interface parameter group is empty")
    return selected, {
        "parameter_names": [name for name, _parameter in selected],
        "parameter_count": sum(parameter.numel() for _name, parameter in selected),
        "group_parameter_counts": groups,
    }


def train_arm(
    source_detail: Mapping[str, Any],
    stage_a_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: FullInterfaceConfig,
    preset: str,
    condition: str,
    seed: int,
    arm: str,
    training_dataset: joint.calibrated.CalibratedDataset,
    pair_batches: torch.Tensor,
    permutation: torch.Tensor,
    device: torch.device,
) -> tuple[joint.PhysicalScalarInterface, dict[str, Any]]:
    torch.manual_seed(seed + 90_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 90_001)
    interface = _load_full_interface(
        source_detail, task, config, preset, condition, device
    )
    initial_model = _model_state_digest(interface.system.model)
    initial_interface = joint._interface_state_digest(interface)
    expected_model = source_detail["training"]["final_model_state_sha256"]
    expected_interface = stage_a_detail["arms"][arm]["training"][
        "initial_interface_state_sha256"
    ]
    if initial_model != expected_model:
        raise RuntimeError("full-interface source model differs from source checkpoint")
    if initial_interface != expected_interface:
        raise RuntimeError("full-interface initialization differs from Stage A")
    initial_signature = model_parameter_signature(interface.system.model)
    signature_digest = _signature_digest(initial_signature)
    selected, selection = trainable_parameters(interface)
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in selected],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    circle = training_dataset.paired.circle
    sensor = joint.calibrated.decode_sensor_tokens(circle.input_ids, task)
    target_scalar = joint._training_targets(
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
        target_posterior = joint.interval_posterior_unclipped(
            target, task.phase_bins
        )
        optimizer.zero_grad(set_to_none=True)
        scalars = interface.forward_scalars(ids, observed, packet)
        predicted_posterior = joint.interval_posterior_unclipped(
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
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for _name, parameter in selected], config.gradient_clip
            )
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
                    "preclip_global_gradient_norm": gradient_norm,
                    "global_clip_coefficient": min(
                        1.0, config.gradient_clip / max(gradient_norm, 1e-30)
                    ),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_model = _model_state_digest(interface.system.model)
    final_interface = joint._interface_state_digest(interface)
    final_signature = model_parameter_signature(interface.system.model)
    final_signature_digest = _signature_digest(final_signature)
    model_changed = final_model != initial_model
    topology_unchanged = final_signature_digest == signature_digest
    if not model_changed:
        raise RuntimeError("full-interface continuation did not change")
    if not topology_unchanged:
        raise RuntimeError("full-interface model topology changed")
    return interface, {
        "arm": arm,
        "objective": (
            "fixed_interval_cross_entropy + sensor_physical_cosine_mse + "
            "final_physical_cosine_mse"
        ),
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "global_gradient_clip": config.gradient_clip,
        "parameter_block_clipping": False,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "trainable_parameter_names": selection["parameter_names"],
        "trainable_parameter_count": selection["parameter_count"],
        "group_parameter_counts": selection["group_parameter_counts"],
        "all_declared_parameters_trainable": True,
        "initial_model_state_sha256": initial_model,
        "source_model_state_sha256": expected_model,
        "final_model_state_sha256": final_model,
        "model_changed": model_changed,
        "initial_model_parameter_signature_sha256": signature_digest,
        "final_model_parameter_signature_sha256": final_signature_digest,
        "model_topology_unchanged": topology_unchanged,
        "initial_interface_state_sha256": initial_interface,
        "stage_a_initial_interface_state_sha256": expected_interface,
        "final_interface_state_sha256": final_interface,
    }


def save_full_checkpoint(
    interface: joint.PhysicalScalarInterface,
    source_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: FullInterfaceConfig,
    path: Path,
    preset: str,
    condition: str,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    if interface.system.encoder is None or interface.system.scalar_embedding is None:
        raise AssertionError("learned interface missing checkpoint modules")
    expected_model = _model_state_digest(interface.system.model)
    expected_interface = joint._interface_state_digest(interface)
    signature = model_parameter_signature(interface.system.model)
    signature_digest = _signature_digest(signature)
    interface.to("cpu")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "arm": arm,
        "model": interface.system.model.state_dict(),
        "encoder": interface.system.encoder.state_dict(),
        "scalar_embedding": interface.system.scalar_embedding.state_dict(),
        "final_scalar": interface.final_scalar.state_dict(),
        "model_state_sha256": expected_model,
        "interface_state_sha256": expected_interface,
        "model_parameter_signature_sha256": signature_digest,
    }
    torch.save(payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    if (
        reloaded.get("schema_version") != SCHEMA_VERSION
        or reloaded.get("hypothesis_id") != HYPOTHESIS_ID
        or reloaded.get("preset") != preset
        or reloaded.get("condition") != condition
        or int(reloaded.get("seed", -1)) != seed
        or reloaded.get("arm") != arm
        or reloaded.get("model_state_sha256") != expected_model
        or reloaded.get("interface_state_sha256") != expected_interface
        or reloaded.get("model_parameter_signature_sha256") != signature_digest
    ):
        raise RuntimeError("full-interface checkpoint metadata failed reload")
    probe = _load_full_interface(
        source_detail, task, config, preset, condition, torch.device("cpu")
    )
    probe.system.model.load_state_dict(reloaded["model"])
    if probe.system.encoder is None or probe.system.scalar_embedding is None:
        raise AssertionError("checkpoint reload probe is incomplete")
    probe.system.encoder.load_state_dict(reloaded["encoder"])
    probe.system.scalar_embedding.load_state_dict(reloaded["scalar_embedding"])
    probe.final_scalar.load_state_dict(reloaded["final_scalar"])
    measured_model = _model_state_digest(probe.system.model)
    measured_interface = joint._interface_state_digest(probe)
    measured_signature = _signature_digest(model_parameter_signature(probe.system.model))
    if (
        measured_model != expected_model
        or measured_interface != expected_interface
        or measured_signature != signature_digest
    ):
        raise RuntimeError("full-interface checkpoint state failed exact reload")
    del probe, reloaded, payload
    gc.collect()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "model_state_sha256": expected_model,
        "interface_state_sha256": expected_interface,
        "model_parameter_signature_sha256": signature_digest,
        "reload_pass": True,
    }


def _cell_directory(root: Path, preset: str, seed: int) -> Path:
    return root / "runs" / preset / CONDITION / f"seed_{seed}"


def _fingerprint(experiment: Experiment) -> str:
    keys = (
        "configuration",
        "task_config",
        "preset",
        "condition",
        "seed",
        "implementation_sha256",
        "source_result_sha256",
        "stage_a_result_sha256",
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


def full_interface_worker(
    experiment: Experiment, device_id: int
) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    base = _joint_config(config)
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    preset = str(experiment.parameters["preset"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    if implementation != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    _source_campaign, source_task, source_details = joint._source_population(base)
    if asdict(source_task) != asdict(task):
        raise RuntimeError("source task changed inside worker")
    source_detail = source_details[(preset, condition, seed)]
    if source_detail["_result_sha256"] != experiment.parameters["source_result_sha256"]:
        raise RuntimeError("source detail changed inside worker")
    _stage_a_campaign, stage_a_details = _stage_a_population(config)
    stage_a_detail = stage_a_details[(preset, condition, seed)]
    if stage_a_detail["_result_sha256"] != experiment.parameters["stage_a_result_sha256"]:
        raise RuntimeError("Stage A detail changed inside worker")
    _block_comparator(config)

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
    cell = _cell_directory(output_root, preset, seed)
    cell.mkdir(parents=True, exist_ok=True)

    datasets = joint._analysis_datasets(task, base)
    dataset_hashes = joint._analysis_dataset_hashes(datasets)
    if dataset_hashes != experiment.parameters["analysis_dataset_hashes"]:
        raise RuntimeError("analysis cohorts changed inside worker")
    sealed_protocol = joint._sealed_training_protocol_config(preset)
    training_dataset, sealed_pair_batches, training_hash, batch_hash = (
        joint.calibrated._protocol_material(task, sealed_protocol, seed)
    )
    if (
        training_hash != source_detail["training"]["training_data_sha256"]
        or batch_hash != source_detail["training"]["minibatch_schedule_sha256"]
    ):
        raise RuntimeError("source training tensors or minibatches changed")
    pair_batches = sealed_pair_batches[
        : config.training_steps, : config.batch_size // 2
    ].clone()
    permutation = joint.pair_preserving_target_permutation(
        config.train_samples, preset, condition, seed, config.shuffle_seed
    )
    permutation_hash = joint.calibrated._tensor_digest(permutation)
    if permutation_hash != stage_a_detail["target_permutation_sha256"]:
        raise RuntimeError("Stage A target permutation changed")
    replay = joint.source_task_replay(
        source_detail, task, base, preset, condition, datasets, device
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
    checkpoints: dict[str, Any] = {}
    source_model_digest = source_detail["training"]["final_model_state_sha256"]
    for arm in ARMS:
        interface, training = train_arm(
            source_detail,
            stage_a_detail,
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
        analysis, arm_arrays = joint.analyze_arm(
            interface,
            task,
            base,
            preset,
            seed,
            arm,
            datasets,
            task_floors,
            device,
        )
        checkpoint = save_full_checkpoint(
            interface,
            source_detail,
            task,
            config,
            cell / f"{arm}_full_interface.pt",
            preset,
            condition,
            seed,
            arm,
        )
        model_source_replay = training["initial_model_state_sha256"] == source_model_digest
        arm_records[arm] = {
            "training": training,
            "analysis": analysis,
            "checkpoint": checkpoint,
            "source_model_replay": model_source_replay,
        }
        checkpoints[arm] = checkpoint
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
    finite = joint.source._finite({"replay": replay, "arms": arm_records}) and all(
        np.isfinite(value).all() for value in arrays.values()
    )
    initial_replay = all(
        record["training"]["initial_interface_state_sha256"]
        == record["training"]["stage_a_initial_interface_state_sha256"]
        for record in arm_records.values()
    )
    continuation_changed = all(
        record["training"]["model_changed"] for record in arm_records.values()
    )
    topology_unchanged = all(
        record["training"]["model_topology_unchanged"]
        for record in arm_records.values()
    )
    full_parameter_route = all(
        record["training"]["all_declared_parameters_trainable"]
        for record in arm_records.values()
    )
    validity = bool(
        replay["maximum_absolute_error"] <= config.replay_tolerance
        and initial_replay
        and all(record["source_model_replay"] for record in arm_records.values())
        and continuation_changed
        and topology_unchanged
        and full_parameter_route
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
        "implementation_sources": sources,
        "scientific_fingerprint": _fingerprint(experiment),
        "source": {
            "result": source_detail["_result_path"],
            "result_sha256": source_detail["_result_sha256"],
            "model_checkpoint": source_detail["artifacts"]["checkpoint"],
            "model_checkpoint_sha256": source_detail["artifacts"][
                "checkpoint_sha256"
            ],
            "model_state_sha256": source_model_digest,
            "training_data_sha256": training_hash,
            "minibatch_schedule_sha256": batch_hash,
            "task_replay": replay,
        },
        "stage_a": {
            "result": stage_a_detail["_result_path"],
            "result_sha256": stage_a_detail["_result_sha256"],
            "target_permutation_sha256": stage_a_detail[
                "target_permutation_sha256"
            ],
            "physical_true_joint_seed_pass": False,
            "pair_shuffled_joint_seed_pass": False,
        },
        "block_clipping_comparator": {
            "campaign": str(Path(config.block_root) / "campaign_results.json"),
            "campaign_sha256": BLOCK_CAMPAIGN_SHA256,
            "physical_true_joint_seed_pass": False,
            "pair_shuffled_joint_seed_pass": False,
        },
        "target_permutation_sha256": permutation_hash,
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
            "stage_a_initial_state_replay": initial_replay,
            "source_model_state_replay": all(
                record["source_model_replay"] for record in arm_records.values()
            ),
            "continuation_changed": continuation_changed,
            "model_topology_unchanged": topology_unchanged,
            "full_parameter_route": full_parameter_route,
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
            "full_interfaces": checkpoints,
        },
    }
    _write_json(result_path, detail)
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
            *(
                [joint.preflight.TINYLLM_PRESETS[preset][2]]
                * joint.preflight.TINYLLM_PRESETS[preset][0]
            ),
            1,
        ],
        model_parameters=int(
            arm_records["physical_true"]["training"]["trainable_parameter_count"]
        ),
        training_time=float(detail["wall_seconds"]),
        training_history=arm_records["physical_true"]["training"]["history"],
        model_checkpoint=checkpoints["physical_true"]["path"],
        observations=[
            f"detail={result_path}",
            "Full TinyLLM continuation and physical interface trained jointly.",
        ],
    )


def classify_aggregates(
    strata: Mapping[str, Mapping[str, Any]], valid: bool
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    specificity = all(record["shuffled_specificity_gate"] for record in strata.values())
    if not specificity:
        return "specificity_control_failed", False
    passed = [preset for preset in PRESETS if strata[preset]["family_gate"]]
    if len(passed) == len(PRESETS):
        return "full_interface_physical_typing_architecture_stable", True
    if passed:
        return "architecture_conditional_full_interface_repair", False
    return "flexible_full_interface_physical_typing_insufficient", False


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: FullInterfaceConfig
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for preset in config.presets:
        selected = sorted(
            (detail for detail in details if detail["preset"] == preset),
            key=lambda detail: detail["seed"],
        )
        if len(selected) != len(config.seeds):
            raise ValueError(f"incomplete stratum {preset}")
        true_count = sum(
            bool(detail["gates"]["physical_true_joint_seed_pass"])
            for detail in selected
        )
        shuffled_count = sum(
            bool(detail["gates"]["pair_shuffled_joint_seed_pass"])
            for detail in selected
        )
        specificity = shuffled_count <= config.shuffled_seed_pass_ceiling
        strata[preset] = {
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
            "stage_a_physical_true_pass_count": 0,
            "block_clipping_physical_true_pass_count": 0,
        }
    complete = bool(
        not config.allow_underpowered
        and len(details) == len(config.presets) * len(config.seeds)
    )
    valid = bool(
        all(detail["gates"]["validity"] for detail in details)
        and (complete or config.allow_underpowered)
    )
    if config.allow_underpowered:
        classification = "systems_lifecycle_only_not_scientific_evidence"
        primary = False
    else:
        classification, primary = classify_aggregates(strata, valid)
    return {
        "valid": valid,
        "complete_primary_population": complete,
        "classification": classification,
        "primary_hypothesis_pass": primary,
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
    return block._task_floors(source_detail)


def _experiments(
    config: FullInterfaceConfig,
    task: CircleTaskConfig,
    source_details: Mapping[tuple[str, str, int], Mapping[str, Any]],
    stage_a_details: Mapping[tuple[str, str, int], Mapping[str, Any]],
    output_dir: Path,
    implementation: str,
    dataset_hashes: Mapping[str, str],
) -> list[Experiment]:
    common = {
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "implementation_sha256": implementation,
        "analysis_dataset_hashes": dict(dataset_hashes),
    }
    return [
        Experiment(
            id=f"tinyllm-joint-full-interface-{preset}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM joint full interface {preset} seed {seed}",
            parameters={
                **common,
                "preset": preset,
                "condition": CONDITION,
                "seed": seed,
                "source_result_sha256": source_details[(preset, CONDITION, seed)][
                    "_result_sha256"
                ],
                "stage_a_result_sha256": stage_a_details[
                    (preset, CONDITION, seed)
                ]["_result_sha256"],
                "task_accuracy_floors": _task_floors(
                    source_details[(preset, CONDITION, seed)]
                ),
            },
            seed=seed,
        )
        for preset in config.presets
        for seed in config.seeds
    ]


def _existing_detail(
    experiment: Experiment, output_dir: Path
) -> Optional[dict[str, Any]]:
    path = _cell_directory(
        output_dir, str(experiment.parameters["preset"]), int(experiment.seed)
    ) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    diagnostics = Path(str(detail.get("artifacts", {}).get("diagnostics", "")))
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint") != _fingerprint(experiment)
        or detail.get("gates", {}).get("validity") is not True
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != detail.get("artifacts", {}).get("diagnostics_sha256")
    ):
        return None
    for arm in ARMS:
        checkpoint = detail.get("artifacts", {}).get("full_interfaces", {}).get(
            arm, {}
        )
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if (
            not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint.get("sha256")
        ):
            return None
    return detail


def _campaign_fingerprint(
    config: FullInterfaceConfig,
    task: CircleTaskConfig,
    implementation: str,
    source_manifest: str,
    stage_a_manifest: str,
    dataset_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "task_config": asdict(task),
            "implementation_sha256": implementation,
            "source_manifest_sha256": source_manifest,
            "stage_a_selected_manifest_sha256": stage_a_manifest,
            "block_comparator_campaign_sha256": BLOCK_CAMPAIGN_SHA256,
            "analysis_dataset_hashes": dict(dataset_hashes),
        }
    )


async def run_campaign(
    config: FullInterfaceConfig, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _joint_config(config)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    source_campaign, task, source_details = joint._source_population(base)
    stage_a_campaign, stage_a_details = _stage_a_population(config)
    block_campaign = _block_comparator(config)
    datasets = joint._analysis_datasets(task, base)
    dataset_hashes = joint._analysis_dataset_hashes(datasets)
    if not config.allow_underpowered:
        for regime in REGIMES:
            if dataset_hashes[regime] != joint.source.EXPECTED_DATASET_HASHES[regime]:
                raise RuntimeError("primary held-out cohort hashes changed")
    source_paths = [
        Path(source_details[(preset, CONDITION, seed)]["_result_path"])
        for preset in config.presets
        for seed in config.seeds
    ]
    stage_a_paths = [
        Path(stage_a_details[(preset, CONDITION, seed)]["_result_path"])
        for preset in config.presets
        for seed in config.seeds
    ]
    source_manifest = _manifest_sha256(source_paths)
    stage_a_manifest = _manifest_sha256(stage_a_paths)
    experiments = _experiments(
        config,
        task,
        source_details,
        stage_a_details,
        output_dir,
        implementation,
        dataset_hashes,
    )
    fingerprint = _campaign_fingerprint(
        config,
        task,
        implementation,
        source_manifest,
        stage_a_manifest,
        dataset_hashes,
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
        project_name="tinyllm_joint_full_interface",
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
    runner = AsyncExperimentRunner(lab, full_interface_worker)
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
    diagnostics = [Path(detail["artifacts"]["diagnostics"]) for detail in details]
    interfaces = [
        Path(detail["artifacts"]["full_interfaces"][arm]["path"])
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
        "implementation_sources": sources,
        "campaign_fingerprint": fingerprint,
        "source": {
            "campaign": str(Path(config.source_root) / "campaign_results.json"),
            "campaign_sha256": joint.source.SOURCE_CAMPAIGN_SHA256,
            "selected_result_manifest_sha256": source_manifest,
            "source_campaign_fingerprint": source_campaign.get(
                "campaign_fingerprint"
            ),
        },
        "stage_a_comparator": {
            "campaign": str(Path(config.stage_a_root) / "campaign_results.json"),
            "campaign_sha256": STAGE_A_CAMPAIGN_SHA256,
            "selected_result_manifest_sha256": stage_a_manifest,
            "campaign_fingerprint": stage_a_campaign["campaign_fingerprint"],
            "learned_pass_counts": {"d6": 0, "d10": 0},
            "pair_shuffled_pass_counts": {"d6": 0, "d10": 0},
            "analytic_positive_control_counts": {"d6": 5, "d10": 4},
        },
        "block_clipping_comparator": {
            "campaign": str(Path(config.block_root) / "campaign_results.json"),
            "campaign_sha256": BLOCK_CAMPAIGN_SHA256,
            "campaign_fingerprint": block_campaign["campaign_fingerprint"],
            "classification": block_campaign["aggregates"]["classification"],
            "physical_true_pass_counts": {"d6": 0, "d10": 0},
            "pair_shuffled_pass_counts": {"d6": 0, "d10": 0},
        },
        "analysis_dataset_hashes": dataset_hashes,
        "summary": {
            "requested_source_cells": len(experiments),
            "requested_full_interface_fits": 2 * len(experiments),
            "reused_source_cells": len(existing),
            "scheduled_source_cells": len(pending),
            "completed_source_cells": len(details),
            "failed_source_cells": len(experiments) - len(details),
            "completed_full_interface_fits": 2 * len(details),
            "changed_continuations": sum(
                int(detail["gates"]["continuation_changed"]) for detail in details
            ),
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
            _manifest_sha256(diagnostics) if diagnostics else None
        ),
        "full_interface_manifest_sha256": (
            _manifest_sha256(interfaces) if interfaces else None
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
            "Stage A and block-clipping outcomes were known before this conditional extension executed.",
            "Every arm starts from the original source model and exact Stage A zero-head interface.",
            "The sole scientific change from Stage A is unfreezing the declared TinyLLM continuation.",
            "One AdamW optimizer, one global clip, and the three unit-weight Stage A losses are retained.",
            "No warm start, differential learning rate, layer schedule, loss sweep, or endpoint reinterpretation occurs.",
            "Stage A and block-clipping cells remain sealed external comparators.",
        ],
    }
    _write_json(campaign_path, bundle)
    return bundle


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_devices(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "cpu":
        return (-1,)
    if value.strip().lower() == "auto":
        return tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else (-1,)
    return _comma_ints(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default=",".join(PRESETS))
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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_joint_full_interface/"
            "20260811_d6_d10_preregistered"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.presets = "d6"
        args.seeds = "7"
        args.steps = 2
        args.batch_size = 16
        args.probe_steps = 2
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 64
        args.allow_underpowered = True
    config = FullInterfaceConfig(
        presets=_comma_strings(args.presets),
        seeds=_comma_ints(args.seeds),
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        probe_steps=args.probe_steps,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
        required_seed_passes=1 if args.shakedown else 4,
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
