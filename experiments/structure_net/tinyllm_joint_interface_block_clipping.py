#!/usr/bin/env python3
"""Test parameter-block clipping on the frozen TinyLLM physical interface."""

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

import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-joint-interface-block-clipping.v1"
HYPOTHESIS_ID = "tinyllm-joint-interface-block-clipping-v1"
EVIDENCE_ROLE = "prospective_frozen_backbone_parameter_block_clipping"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-interface-block-clipping-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "4264a56a0be6f70fc5c4e812f2fef8aadcdc541450d51993f5f1bbc52fc26f6f"
)
STAGE_A_ROOT = Path(
    "data/experiments/tinyllm_joint_physical_scalar_interface/"
    "20260811_d6_d10_preregistered"
)
STAGE_A_CAMPAIGN_SHA256 = (
    "65ab4b4e887212c4754cf918908cd5e3f04727af4d7876de1d3aa749bc50ac51"
)
STAGE_A_RESULT_MANIFEST_SHA256 = (
    "3299a6cd2edf8816b8bb65ef1ddfb7dfc18f0f6edd41ff09730af632580fc9f3"
)
STAGE_A_IMPLEMENTATION_SHA256 = (
    "ac1f5b42e2e8bcfc645de7dba048ae258e692db79f09b20f4e25f8d00940e1a1"
)
GRADIENT_ROOT = Path(
    "data/experiments/tinyllm_joint_interface_gradient_attribution/"
    "20260811_d6_d10_registered_v2"
)
GRADIENT_CAMPAIGN_SHA256 = (
    "a3540216800a0cccf0d3725cf349f8a5c91bf01b8680d44c814afd8f4fa6ba25"
)
JOINT_RUNNER_SHA256 = (
    "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
)
PRESETS = ("d6", "d10")
CONDITION = "learned_calibrated_equivariant"
CONDITIONS = (CONDITION,)
SEEDS = (7, 17, 29, 41, 53)
ARMS = joint.ARMS
BLOCKS = ("encoder", "scalar_embedding", "final_scalar")
CUTS = joint.CUTS
REGIMES = joint.REGIMES


@dataclass(frozen=True)
class BlockClippingConfig:
    source_root: str = str(joint.SOURCE_ROOT)
    stage_a_root: str = str(STAGE_A_ROOT)
    gradient_root: str = str(GRADIENT_ROOT)
    presets: tuple[str, ...] = PRESETS
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    block_gradient_clip: float = 1.0
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
            self.block_gradient_clip,
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
                "gradient_root": str(GRADIENT_ROOT),
                "presets": PRESETS,
                "conditions": CONDITIONS,
                "seeds": SEEDS,
                "training_steps": 600,
                "train_samples": 4_096,
                "batch_size": 64,
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
                "block_gradient_clip": 1.0,
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
                raise ValueError("primary block-clipping configuration changed")


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


def _json_config(config: BlockClippingConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(values: Mapping[str, Any]) -> BlockClippingConfig:
    converted = dict(values)
    for field in ("presets", "conditions", "seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return BlockClippingConfig(**converted)


def _joint_config(config: BlockClippingConfig) -> joint.JointPhysicalInterfaceConfig:
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
        gradient_clip=config.block_gradient_clip,
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


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "stage_a_runner": _sha256(Path(joint.__file__)),
        "calibrated_frontend": _sha256(Path(joint.calibrated.__file__)),
        "source_validator": _sha256(Path(joint.source.__file__)),
        "interval_metrics": _sha256(Path(joint.interval.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("block-clipping preregistration changed")
    if values["stage_a_runner"] != JOINT_RUNNER_SHA256:
        raise RuntimeError("Stage A runner changed")
    return values


def _implementation_digest(values: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(values or _implementation_sources()))


def _stage_a_paths(root: Path) -> list[Path]:
    return [
        root / "runs" / preset / condition / f"seed_{seed}" / "result.json"
        for preset in joint.PRESETS
        for condition in joint.CONDITIONS
        for seed in joint.SEEDS
    ]


def _stage_a_population(
    config: BlockClippingConfig,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    root = Path(config.stage_a_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != STAGE_A_CAMPAIGN_SHA256
        or campaign.get("schema_version") != joint.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != joint.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != STAGE_A_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != STAGE_A_RESULT_MANIFEST_SHA256
        or campaign.get("aggregates", {}).get("classification")
        != "frozen_backbone_joint_interface_insufficient"
        or campaign.get("aggregates", {}).get("valid") is not True
    ):
        raise ValueError(f"invalid Stage A campaign {campaign_path}")
    all_paths = _stage_a_paths(root)
    if _manifest_sha256(all_paths) != STAGE_A_RESULT_MANIFEST_SHA256:
        raise ValueError("Stage A result manifest changed")
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for preset in config.presets:
        for condition in config.conditions:
            for seed in config.seeds:
                path = root / "runs" / preset / condition / f"seed_{seed}" / "result.json"
                detail = json.loads(path.read_text(encoding="utf-8"))
                if (
                    detail.get("schema_version") != joint.SCHEMA_VERSION
                    or detail.get("status") != "completed"
                    or detail.get("implementation_sha256")
                    != STAGE_A_IMPLEMENTATION_SHA256
                    or detail.get("gates", {}).get("validity") is not True
                    or detail.get("gates", {}).get(
                        "physical_true_joint_seed_pass"
                    )
                    is not False
                    or detail.get("gates", {}).get(
                        "pair_shuffled_joint_seed_pass"
                    )
                    is not False
                ):
                    raise ValueError(f"invalid Stage A learned cell {path}")
                first = detail["arms"]["physical_true"]["training"][
                    "initial_interface_state_sha256"
                ]
                second = detail["arms"]["pair_shuffled"]["training"][
                    "initial_interface_state_sha256"
                ]
                if first != second:
                    raise ValueError("Stage A matched arms changed initial state")
                detail["_result_path"] = str(path)
                detail["_result_sha256"] = _sha256(path)
                selected[(preset, condition, seed)] = detail
    return campaign, selected


def _gradient_comparator(config: BlockClippingConfig) -> dict[str, Any]:
    path = Path(config.gradient_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != GRADIENT_CAMPAIGN_SHA256
        or campaign.get("status") != "completed"
        or campaign.get("aggregates", {}).get("valid") is not True
        or campaign.get("aggregates", {}).get("classification")
        != "no_registered_gradient_failure_mechanism"
    ):
        raise ValueError(f"invalid gradient comparator {path}")
    return campaign


def parameter_blocks(
    interface: joint.PhysicalScalarInterface,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    if interface.system.encoder is None or interface.system.scalar_embedding is None:
        raise AssertionError("learned structured interface is incomplete")
    blocks = {
        "encoder": [
            (f"system.encoder.{name}", parameter)
            for name, parameter in interface.system.encoder.named_parameters()
        ],
        "scalar_embedding": [
            (f"system.scalar_embedding.{name}", parameter)
            for name, parameter in interface.system.scalar_embedding.named_parameters()
        ],
        "final_scalar": [
            (f"final_scalar.{name}", parameter)
            for name, parameter in interface.final_scalar.named_parameters()
        ],
    }
    selected = {
        name
        for name, parameter in interface.named_parameters()
        if parameter.requires_grad
    }
    declared = {name for values in blocks.values() for name, _parameter in values}
    if selected != declared or any(not values for values in blocks.values()):
        raise AssertionError("trainable parameters do not match declared blocks")
    return blocks


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().square().sum())
    return math.sqrt(squared)


def clip_parameter_blocks(
    blocks: Mapping[str, Sequence[tuple[str, nn.Parameter]]], ceiling: float
) -> dict[str, Any]:
    preclip = {
        block: _gradient_norm(parameter for _name, parameter in values)
        for block, values in blocks.items()
    }
    global_preclip = math.sqrt(sum(value * value for value in preclip.values()))
    global_equivalent = min(1.0, ceiling / max(global_preclip, 1e-30))
    coefficients = {}
    for block, values in blocks.items():
        parameters = [parameter for _name, parameter in values]
        returned = float(torch.nn.utils.clip_grad_norm_(parameters, ceiling))
        if not math.isclose(returned, preclip[block], rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError(f"pre-clip norm mismatch in {block}")
        coefficients[block] = min(1.0, ceiling / max(preclip[block], 1e-30))
    postclip = {
        block: _gradient_norm(parameter for _name, parameter in values)
        for block, values in blocks.items()
    }
    return {
        "preclip_norms": preclip,
        "clip_coefficients": coefficients,
        "postclip_norms": postclip,
        "global_preclip_norm": global_preclip,
        "global_equivalent_clip_coefficient": global_equivalent,
        "post_block_clip_global_norm": math.sqrt(
            sum(value * value for value in postclip.values())
        ),
    }


def train_arm(
    source_detail: Mapping[str, Any],
    stage_a_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: BlockClippingConfig,
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
    interface = joint._load_trainable_interface(
        source_detail, task, _joint_config(config), preset, condition, device
    )
    initial_model = joint.calibrated._state_digest(interface.system.model)
    initial_interface = joint._interface_state_digest(interface)
    expected_initial = stage_a_detail["arms"][arm]["training"][
        "initial_interface_state_sha256"
    ]
    if initial_interface != expected_initial:
        raise RuntimeError("block-clipping initial interface differs from Stage A")
    blocks = parameter_blocks(interface)
    selected = [item for values in blocks.values() for item in values]
    parameter_count = sum(parameter.numel() for _name, parameter in selected)
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
    history: list[dict[str, Any]] = []
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
        loss = task_loss + sensor_loss + final_loss
        loss.backward()
        clipping = clip_parameter_blocks(blocks, config.block_gradient_clip)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            history.append(
                {
                    "step": float(step),
                    "total_loss": float(loss.detach()),
                    "task_cross_entropy": float(task_loss.detach()),
                    "sensor_mse": float(sensor_loss.detach()),
                    "final_mse": float(final_loss.detach()),
                    **clipping,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_model = joint.calibrated._state_digest(interface.system.model)
    final_interface = joint._interface_state_digest(interface)
    if final_model != initial_model:
        raise RuntimeError("frozen TinyLLM backbone changed")
    return interface, {
        "arm": arm,
        "objective": (
            "fixed_interval_cross_entropy + sensor_physical_cosine_mse + "
            "final_physical_cosine_mse"
        ),
        "optimizer_intervention": "independent_parameter_block_gradient_clipping",
        "blocks": list(BLOCKS),
        "block_gradient_clip": config.block_gradient_clip,
        "subsequent_global_clip": False,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "trainable_parameter_names": [name for name, _parameter in selected],
        "trainable_parameter_count": parameter_count,
        "initial_model_state_sha256": initial_model,
        "final_model_state_sha256": final_model,
        "initial_interface_state_sha256": initial_interface,
        "stage_a_initial_interface_state_sha256": expected_initial,
        "final_interface_state_sha256": final_interface,
    }


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def save_interface_checkpoint(
    interface: joint.PhysicalScalarInterface,
    source_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: BlockClippingConfig,
    path: Path,
    preset: str,
    condition: str,
    seed: int,
    arm: str,
    device: torch.device,
) -> dict[str, Any]:
    if interface.system.encoder is None or interface.system.scalar_embedding is None:
        raise AssertionError("learned interface missing checkpoint modules")
    expected = joint._interface_state_digest(interface)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "arm": arm,
        "encoder": _cpu_state(interface.system.encoder),
        "scalar_embedding": _cpu_state(interface.system.scalar_embedding),
        "final_scalar": _cpu_state(interface.final_scalar),
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
    ):
        raise RuntimeError("block-clipping checkpoint metadata failed reload")
    probe = joint._load_trainable_interface(
        source_detail, task, _joint_config(config), preset, condition, device
    )
    if probe.system.encoder is None or probe.system.scalar_embedding is None:
        raise AssertionError("checkpoint probe is incomplete")
    probe.system.encoder.load_state_dict(reloaded["encoder"])
    probe.system.scalar_embedding.load_state_dict(reloaded["scalar_embedding"])
    probe.final_scalar.load_state_dict(reloaded["final_scalar"])
    measured = joint._interface_state_digest(probe)
    del probe
    if measured != expected:
        raise RuntimeError("block-clipping checkpoint state failed reload")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "state_sha256": expected,
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


def block_clipping_worker(
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
    _gradient_comparator(config)

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
        checkpoint = save_interface_checkpoint(
            interface,
            source_detail,
            task,
            config,
            cell / f"{arm}_interface.pt",
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
    validity = bool(
        replay["maximum_absolute_error"] <= config.replay_tolerance
        and initial_replay
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
        "implementation_sources": sources,
        "scientific_fingerprint": _fingerprint(experiment),
        "source": {
            "result": source_detail["_result_path"],
            "result_sha256": source_detail["_result_sha256"],
            "model_checkpoint": source_detail["artifacts"]["checkpoint"],
            "model_checkpoint_sha256": source_detail["artifacts"][
                "checkpoint_sha256"
            ],
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
            "interfaces": checkpoints,
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
            *([joint.preflight.TINYLLM_PRESETS[preset][2]] * joint.preflight.TINYLLM_PRESETS[preset][0]),
            1,
        ],
        model_parameters=int(
            arm_records["physical_true"]["training"]["trainable_parameter_count"]
        ),
        training_time=float(detail["wall_seconds"]),
        training_history=arm_records["physical_true"]["training"]["history"],
        model_checkpoint=checkpoints["physical_true"]["path"],
        observations=[f"detail={result_path}", "Frozen backbone; block clipping."],
    )


def classify_aggregates(
    strata: Mapping[str, Mapping[str, Any]], valid: bool
) -> tuple[str, bool, bool]:
    if not valid:
        return "invalid", False, False
    specificity = all(record["shuffled_specificity_gate"] for record in strata.values())
    if not specificity:
        return "specificity_control_failed", False, False
    passed = [preset for preset in PRESETS if strata[preset]["family_gate"]]
    if len(passed) == len(PRESETS):
        return "parameter_block_clipping_repairs_physical_interface", True, False
    if passed:
        return "architecture_conditional_block_clipping_repair", False, True
    return "parameter_block_clipping_insufficient", False, True


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: BlockClippingConfig
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
            "stage_a_pair_shuffled_pass_count": 0,
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
        primary = extension = False
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
    config: BlockClippingConfig,
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
            id=f"tinyllm-joint-block-clip-{preset}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM joint block clipping {preset} seed {seed}",
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
        checkpoint = detail.get("artifacts", {}).get("interfaces", {}).get(arm, {})
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if (
            not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint.get("sha256")
        ):
            return None
    return detail


def _campaign_fingerprint(
    config: BlockClippingConfig,
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
            "analysis_dataset_hashes": dict(dataset_hashes),
        }
    )


async def run_campaign(
    config: BlockClippingConfig, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _joint_config(config)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    source_campaign, task, source_details = joint._source_population(base)
    stage_a_campaign, stage_a_details = _stage_a_population(config)
    gradient_campaign = _gradient_comparator(config)
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
        project_name="tinyllm_joint_interface_block_clipping",
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
    runner = AsyncExperimentRunner(lab, block_clipping_worker)
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
        "gradient_comparator": {
            "campaign": str(Path(config.gradient_root) / "campaign_results.json"),
            "campaign_sha256": GRADIENT_CAMPAIGN_SHA256,
            "campaign_fingerprint": gradient_campaign["campaign_fingerprint"],
            "classification": gradient_campaign["aggregates"]["classification"],
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
            _manifest_sha256(diagnostics) if diagnostics else None
        ),
        "interface_manifest_sha256": (
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
            "The Stage A and gradient-attribution outcomes were known before registration.",
            "Only the gradient clipping partition changes from Stage A.",
            "All TinyLLM backbone, embedding, layer-normalization, and LM-head parameters remain frozen.",
            "No loss reweighting, warm start, schedule change, or transformer update occurs.",
            "The Stage A global-clip cells are a sealed external comparator, not a pooled rerun.",
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
            "data/experiments/tinyllm_joint_interface_block_clipping/"
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
    config = BlockClippingConfig(
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
