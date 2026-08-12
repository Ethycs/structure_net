#!/usr/bin/env python3
"""Attribute the saved joint-interface objective gradients without retraining."""

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
import experiments.structure_net.tinyllm_frozen_scalar_interface_decomposition as source
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-joint-interface-gradient-attribution.v2"
HYPOTHESIS_ID = "tinyllm-joint-interface-gradient-attribution-v2"
EVIDENCE_ROLE = "registered_post_outcome_no_training_gradient_attribution"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-interface-gradient-attribution-v2-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "74aef578a00b4cd71cfdd7a94852cb65ba1bef76e3b13be6e1b12b9acab4cbdd"
)
SOURCE_ROOT = Path(
    "data/experiments/tinyllm_joint_physical_scalar_interface/"
    "20260811_d6_d10_preregistered"
)
SOURCE_CAMPAIGN_SHA256 = (
    "65ab4b4e887212c4754cf918908cd5e3f04727af4d7876de1d3aa749bc50ac51"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "3299a6cd2edf8816b8bb65ef1ddfb7dfc18f0f6edd41ff09730af632580fc9f3"
)
SOURCE_INTERFACE_MANIFEST_SHA256 = (
    "50dec4731b55d118561e36f0ff35e8bbcadb3fbca856f7cd234151332e09322a"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "ac1f5b42e2e8bcfc645de7dba048ae258e692db79f09b20f4e25f8d00940e1a1"
)
PRESETS = joint.PRESETS
CONDITIONS = joint.CONDITIONS
SEEDS = joint.SEEDS
STATES = ("initial", "final")
BATCHES = ("first", "last")
OBJECTIVES = ("sensor", "final", "task")
BLOCKS = ("encoder", "scalar_embedding", "final_scalar")


@dataclass(frozen=True)
class GradientAttributionConfig:
    source_root: str = str(SOURCE_ROOT)
    presets: tuple[str, ...] = PRESETS
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    schedule_steps: tuple[int, ...] = (0, 599)
    gradient_clip: float = 1.0
    global_clip_maximum: float = 0.10
    cross_block_suppression_maximum: float = 0.10
    additivity_tolerance: float = 2e-5
    relative_additivity_tolerance: float = 1e-6
    nonzero_tolerance: float = 1e-12
    required_seed_passes: int = 4
    batch_size: int = 64
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 1
    gpu_memory_per_experiment_gb: Optional[float] = 2.0
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
        if self.schedule_steps != (0, 599):
            raise ValueError("registered diagnostic fixes first and last schedule rows")
        if self.batch_size != 64 or self.gradient_clip != 1.0:
            raise ValueError("registered batch size and gradient clip changed")
        if min(
            self.global_clip_maximum,
            self.cross_block_suppression_maximum,
            self.additivity_tolerance,
            self.relative_additivity_tolerance,
            self.nonzero_tolerance,
        ) <= 0.0:
            raise ValueError("diagnostic tolerances must be positive")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed passes are outside the population")
        if not self.allow_underpowered:
            expected = {
                "source_root": str(SOURCE_ROOT),
                "presets": PRESETS,
                "conditions": CONDITIONS,
                "seeds": SEEDS,
                "schedule_steps": (0, 599),
                "gradient_clip": 1.0,
                "global_clip_maximum": 0.10,
                "cross_block_suppression_maximum": 0.10,
                "additivity_tolerance": 2e-5,
                "relative_additivity_tolerance": 1e-6,
                "required_seed_passes": 4,
                "batch_size": 64,
            }
            actual = {
                "source_root": self.source_root,
                "presets": self.presets,
                "conditions": self.conditions,
                "seeds": self.seeds,
                "schedule_steps": self.schedule_steps,
                "gradient_clip": self.gradient_clip,
                "global_clip_maximum": self.global_clip_maximum,
                "cross_block_suppression_maximum": (
                    self.cross_block_suppression_maximum
                ),
                "additivity_tolerance": self.additivity_tolerance,
                "relative_additivity_tolerance": (
                    self.relative_additivity_tolerance
                ),
                "required_seed_passes": self.required_seed_passes,
                "batch_size": self.batch_size,
            }
            if actual != expected:
                raise ValueError("primary gradient-attribution configuration changed")


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


def _json_config(config: GradientAttributionConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(values: Mapping[str, Any]) -> GradientAttributionConfig:
    converted = dict(values)
    for field in ("presets", "conditions", "seeds", "schedule_steps", "device_ids"):
        converted[field] = tuple(converted[field])
    return GradientAttributionConfig(**converted)


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "joint_source_runner": _sha256(Path(joint.__file__)),
        "calibrated_frontend": _sha256(Path(calibrated.__file__)),
        "source_validator": _sha256(Path(source.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("gradient-attribution preregistration changed")
    if values["joint_source_runner"] != joint._implementation_sources()["runner"]:
        raise RuntimeError("joint source runner identity changed")
    return values


def _implementation_digest(values: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(values or _implementation_sources()))


def _source_detail_paths(root: Path) -> list[Path]:
    return [
        root / "runs" / preset / condition / f"seed_{seed}" / "result.json"
        for preset in PRESETS
        for condition in CONDITIONS
        for seed in SEEDS
    ]


def _source_population(
    config: GradientAttributionConfig,
) -> tuple[dict[str, Any], CircleTaskConfig, dict[tuple[str, str, int], dict[str, Any]]]:
    root = Path(config.source_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != joint.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != joint.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or campaign.get("interface_manifest_sha256")
        != SOURCE_INTERFACE_MANIFEST_SHA256
        or campaign.get("aggregates", {}).get("classification")
        != "frozen_backbone_joint_interface_insufficient"
        or campaign.get("aggregates", {}).get("valid") is not True
    ):
        raise ValueError(f"invalid joint-interface source campaign {campaign_path}")
    paths = _source_detail_paths(root)
    if _manifest_sha256(paths) != SOURCE_RESULT_MANIFEST_SHA256:
        raise ValueError("joint-interface source result manifest changed")
    details: dict[tuple[str, str, int], dict[str, Any]] = {}
    interfaces = []
    for path in paths:
        detail = json.loads(path.read_text(encoding="utf-8"))
        key = (detail["preset"], detail["condition"], int(detail["seed"]))
        if (
            detail.get("schema_version") != joint.SCHEMA_VERSION
            or detail.get("status") != "completed"
            or detail.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
            or detail.get("gates", {}).get("validity") is not True
        ):
            raise ValueError(f"invalid joint-interface source detail {path}")
        detail["_result_path"] = str(path)
        detail["_result_sha256"] = _sha256(path)
        details[key] = detail
        for arm in joint.ARMS:
            interface = Path(detail["artifacts"]["interfaces"][arm]["path"])
            if (
                not interface.is_file()
                or _sha256(interface)
                != detail["artifacts"]["interfaces"][arm]["sha256"]
            ):
                raise ValueError(f"invalid joint-interface checkpoint {interface}")
            interfaces.append(interface)
    if _manifest_sha256(interfaces) != SOURCE_INTERFACE_MANIFEST_SHA256:
        raise ValueError("joint-interface source checkpoint manifest changed")
    return campaign, CircleTaskConfig(**campaign["task_config"]), details


def _cell_directory(root: Path, preset: str, condition: str, seed: int) -> Path:
    return root / "runs" / preset / condition / f"seed_{seed}"


def _load_source_record(detail: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(Path(detail["source"]["result"]).read_text(encoding="utf-8"))
    if _sha256(Path(detail["source"]["result"])) != detail["source"]["result_sha256"]:
        raise ValueError("architecture source result changed")
    return record


def _base_joint_config() -> joint.JointPhysicalInterfaceConfig:
    return joint.JointPhysicalInterfaceConfig()


def _load_state(
    detail: Mapping[str, Any],
    source_detail: Mapping[str, Any],
    task: CircleTaskConfig,
    preset: str,
    condition: str,
    state: str,
    device: torch.device,
) -> joint.PhysicalScalarInterface:
    interface = joint._load_trainable_interface(
        source_detail,
        task,
        _base_joint_config(),
        preset,
        condition,
        device,
    )
    if state == "final":
        path = Path(detail["artifacts"]["interfaces"]["physical_true"]["path"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("arm") != "physical_true":
            raise ValueError("physical interface checkpoint arm changed")
        if interface.system.scalar_embedding is None:
            raise AssertionError("scalar embedding missing")
        interface.system.scalar_embedding.load_state_dict(payload["scalar_embedding"])
        interface.final_scalar.load_state_dict(payload["final_scalar"])
        if interface.system.encoder is not None:
            if payload.get("encoder") is None:
                raise AssertionError("learned encoder checkpoint missing")
            interface.system.encoder.load_state_dict(payload["encoder"])
    elif state != "initial":
        raise ValueError(f"unknown state: {state}")
    interface.to(device).eval()
    expected = detail["arms"]["physical_true"]["training"][
        f"{state}_interface_state_sha256"
    ]
    if joint._interface_state_digest(interface) != expected:
        raise ValueError(f"{state} interface state changed")
    return interface


def parameter_blocks(
    interface: joint.PhysicalScalarInterface,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    if interface.system.scalar_embedding is None:
        raise AssertionError("scalar embedding missing")
    blocks = {
        "encoder": (
            list(interface.system.encoder.named_parameters())
            if interface.system.encoder is not None
            else []
        ),
        "scalar_embedding": list(
            interface.system.scalar_embedding.named_parameters()
        ),
        "final_scalar": list(interface.final_scalar.named_parameters()),
    }
    for values in blocks.values():
        for _name, parameter in values:
            parameter.requires_grad_(True)
    return blocks


def _flatten_gradients(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if not parameters:
        return torch.zeros(0, dtype=torch.float32, device=loss.device)
    if not loss.requires_grad:
        return torch.zeros(
            sum(parameter.numel() for parameter in parameters),
            dtype=torch.float32,
            device=loss.device,
        )
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat(
        [
            (
                gradient.reshape(-1)
                if gradient is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.double()))


def gradient_cosine(first: torch.Tensor, second: torch.Tensor) -> Optional[float]:
    denominator = _norm(first) * _norm(second)
    if denominator <= 0.0:
        return None
    return float(torch.dot(first.double(), second.double()) / denominator)


def block_gradient_record(
    objective_vectors: Mapping[str, torch.Tensor],
    clip_ceiling: float,
    global_clip_coefficient: float,
) -> dict[str, Any]:
    total = sum(objective_vectors.values(), torch.zeros_like(next(iter(objective_vectors.values()))))
    total_norm = _norm(total)
    block_clip = min(1.0, clip_ceiling / max(total_norm, 1e-30))
    sensor = objective_vectors["sensor"]
    downstream = objective_vectors["final"] + objective_vectors["task"]
    sensor_norm = _norm(sensor)
    sensor_descent_ratio = (
        float(torch.dot(sensor.double(), total.double()) / sensor_norm**2)
        if sensor_norm > 0.0
        else None
    )
    return {
        "parameter_count": int(total.numel()),
        "objective_norms": {
            name: _norm(vector) for name, vector in objective_vectors.items()
        },
        "total_norm": total_norm,
        "block_clip_coefficient": block_clip,
        "global_clip_coefficient": global_clip_coefficient,
        "cross_block_suppression": (
            global_clip_coefficient / block_clip if block_clip > 0.0 else None
        ),
        "objective_cosines": {
            "sensor_final": gradient_cosine(sensor, objective_vectors["final"]),
            "sensor_task": gradient_cosine(sensor, objective_vectors["task"]),
            "sensor_downstream": gradient_cosine(sensor, downstream),
            "final_task": gradient_cosine(
                objective_vectors["final"], objective_vectors["task"]
            ),
        },
        "downstream_norm": _norm(downstream),
        "sensor_descent_ratio": sensor_descent_ratio,
    }


def gradient_snapshot(
    interface: joint.PhysicalScalarInterface,
    input_ids: torch.Tensor,
    sensor: torch.Tensor,
    packet: torch.Tensor,
    target: torch.Tensor,
    task: CircleTaskConfig,
    clip_ceiling: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    interface.eval()
    blocks = parameter_blocks(interface)
    block_parameters = {
        block: [parameter for _name, parameter in values]
        for block, values in blocks.items()
    }
    values = interface.forward_scalars(input_ids, sensor, packet)
    target = target.reshape(-1, 1)
    target_posterior = joint.interval_posterior_unclipped(target, task.phase_bins)
    predicted_posterior = joint.interval_posterior_unclipped(
        values["full"], task.phase_bins
    )
    losses = {
        "sensor": F.mse_loss(values["frontend"], target),
        "final": F.mse_loss(values["full"], target),
        "task": -(
            target_posterior * predicted_posterior.clamp_min(1e-12).log()
        ).sum(-1).mean(),
    }
    objective_vectors: dict[str, dict[str, torch.Tensor]] = {
        block: {} for block in BLOCKS
    }
    for objective in OBJECTIVES:
        for block in BLOCKS:
            objective_vectors[block][objective] = _flatten_gradients(
                losses[objective], block_parameters[block], retain_graph=True
            )
    combined_loss = sum(losses.values())
    direct_total = {
        block: _flatten_gradients(
            combined_loss, block_parameters[block], retain_graph=True
        )
        for block in BLOCKS
    }
    summed_total = {
        block: sum(
            objective_vectors[block].values(),
            torch.zeros_like(direct_total[block]),
        )
        for block in BLOCKS
    }
    additivity_errors = {
        block: (
            float((direct_total[block] - summed_total[block]).abs().max())
            if direct_total[block].numel()
            else 0.0
        )
        for block in BLOCKS
    }
    global_vector = torch.cat([summed_total[block] for block in BLOCKS])
    global_norm = _norm(global_vector)
    maximum_additivity_error = max(additivity_errors.values())
    relative_additivity_error = maximum_additivity_error / max(1.0, global_norm)
    global_clip = min(1.0, clip_ceiling / max(global_norm, 1e-30))
    records = {
        block: block_gradient_record(
            objective_vectors[block], clip_ceiling, global_clip
        )
        for block in BLOCKS
    }
    arrays = {
        f"gradient__{block}__{objective}": vector.detach().float().cpu().numpy()
        for block in BLOCKS
        for objective, vector in objective_vectors[block].items()
    }
    arrays.update(
        {
            f"gradient__{block}__total": summed_total[block]
            .detach()
            .float()
            .cpu()
            .numpy()
            for block in BLOCKS
        }
    )
    return {
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "combined_loss": float(combined_loss.detach()),
        "global_gradient_norm": global_norm,
        "global_clip_coefficient": global_clip,
        "blocks": records,
        "gradient_additivity_maximum_absolute_errors": additivity_errors,
        "gradient_additivity_maximum_absolute_error": maximum_additivity_error,
        "gradient_additivity_maximum_relative_error": relative_additivity_error,
    }, arrays


def _seed_gates(
    records: Mapping[str, Mapping[str, Any]],
    condition: str,
    config: GradientAttributionConfig,
) -> dict[str, Any]:
    if condition != "learned_calibrated_equivariant":
        return {
            "initial_cross_block_starvation": None,
            "persistent_learned_state_conflict": None,
            "role": "fixed_sensor_control_not_applicable",
        }
    initial = all(
        records["initial"][batch]["blocks"]["encoder"][
            "block_clip_coefficient"
        ]
        == 1.0
        and records["initial"][batch]["global_clip_coefficient"]
        <= config.global_clip_maximum
        and records["initial"][batch]["blocks"]["encoder"][
            "cross_block_suppression"
        ]
        <= config.cross_block_suppression_maximum
        and records["initial"][batch]["blocks"]["encoder"]["objective_norms"][
            "sensor"
        ]
        > config.nonzero_tolerance
        for batch in BATCHES
    )
    persistent = all(
        records["final"][batch]["blocks"]["encoder"]["objective_norms"][
            "sensor"
        ]
        > config.nonzero_tolerance
        and records["final"][batch]["blocks"]["encoder"][
            "sensor_descent_ratio"
        ]
        <= 0.0
        for batch in BATCHES
    )
    return {
        "initial_cross_block_starvation": bool(initial),
        "persistent_learned_state_conflict": bool(persistent),
        "role": "primary_learned_gradient_gate",
    }


def _fingerprint(experiment: Experiment) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            **{
                key: experiment.parameters[key]
                for key in (
                    "configuration",
                    "task_config",
                    "preset",
                    "condition",
                    "seed",
                    "implementation_sha256",
                    "source_result_sha256",
                )
            },
        }
    )


def gradient_attribution_worker(
    experiment: Experiment, device_id: int
) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    preset = str(experiment.parameters["preset"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    if implementation != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    _campaign, source_task, details = _source_population(config)
    if asdict(source_task) != asdict(task):
        raise RuntimeError("source task changed inside worker")
    detail = details[(preset, condition, seed)]
    if detail["_result_sha256"] != experiment.parameters["source_result_sha256"]:
        raise RuntimeError("joint-interface source detail changed")
    architecture_source = _load_source_record(detail)
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed + 330_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 330_001)
    started = time.perf_counter()

    protocol_config = joint._sealed_training_protocol_config(preset)
    training_dataset, pair_schedule, training_hash, schedule_hash = (
        calibrated._protocol_material(task, protocol_config, seed)
    )
    if (
        training_hash != detail["source"]["training_data_sha256"]
        or schedule_hash != detail["source"]["minibatch_schedule_sha256"]
    ):
        raise RuntimeError("sealed source protocol changed")
    sensor_values = calibrated.decode_sensor_tokens(
        training_dataset.paired.circle.input_ids, task
    )
    batch_indices = {}
    for name, step in zip(BATCHES, config.schedule_steps):
        pairs = pair_schedule[step]
        batch_indices[name] = torch.stack(
            (2 * pairs, 2 * pairs + 1), dim=1
        ).reshape(-1)

    output_root = Path(experiment.parameters["output_dir"])
    cell = _cell_directory(output_root, preset, condition, seed)
    cell.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {state: {} for state in STATES}
    arrays: dict[str, np.ndarray] = {
        "first_batch_indices": batch_indices["first"].numpy(),
        "last_batch_indices": batch_indices["last"].numpy(),
    }
    model_state_expected = architecture_source["training"]["final_model_state_sha256"]
    state_checks = {}
    maximum_additivity_error = 0.0
    maximum_relative_additivity_error = 0.0
    for state in STATES:
        interface = _load_state(
            detail,
            architecture_source,
            task,
            preset,
            condition,
            state,
            device,
        )
        interface_before = joint._interface_state_digest(interface)
        model_before = calibrated._state_digest(interface.system.model)
        for batch in BATCHES:
            indices = batch_indices[batch]
            record, snapshot_arrays = gradient_snapshot(
                interface,
                training_dataset.paired.circle.input_ids[indices].to(device),
                sensor_values[indices].to(device),
                training_dataset.calibration[indices].to(device),
                training_dataset.paired.fiber.cosine[indices].to(device),
                task,
                config.gradient_clip,
            )
            records[state][batch] = record
            maximum_additivity_error = max(
                maximum_additivity_error,
                record["gradient_additivity_maximum_absolute_error"],
            )
            maximum_relative_additivity_error = max(
                maximum_relative_additivity_error,
                record["gradient_additivity_maximum_relative_error"],
            )
            arrays.update(
                {
                    f"{state}__{batch}__{name}": value
                    for name, value in snapshot_arrays.items()
                }
            )
        interface_after = joint._interface_state_digest(interface)
        model_after = calibrated._state_digest(interface.system.model)
        state_checks[state] = {
            "interface_before_sha256": interface_before,
            "interface_after_sha256": interface_after,
            "interface_unchanged": interface_before == interface_after,
            "model_before_sha256": model_before,
            "model_after_sha256": model_after,
            "model_unchanged": (
                model_before == model_after == model_state_expected
            ),
        }
        del interface
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_gates = _seed_gates(records, condition, config)
    diagnostics_path = cell / "gradient_diagnostics.npz"
    _write_npz(diagnostics_path, arrays)
    with np.load(diagnostics_path, allow_pickle=False) as reloaded:
        diagnostics_reload = bool(
            set(reloaded.files) == set(arrays)
            and all(np.array_equal(reloaded[name], arrays[name]) for name in arrays)
        )
    finite = source._finite({"records": records, "state_checks": state_checks}) and all(
        np.isfinite(value).all() for value in arrays.values()
    )
    validity = bool(
        maximum_additivity_error <= config.additivity_tolerance
        and maximum_relative_additivity_error
        <= config.relative_additivity_tolerance
        and all(value["interface_unchanged"] for value in state_checks.values())
        and all(value["model_unchanged"] for value in state_checks.values())
        and diagnostics_reload
        and finite
    )
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    result_path = cell / "result.json"
    result = {
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
            "joint_interface_result": detail["_result_path"],
            "joint_interface_result_sha256": detail["_result_sha256"],
            "architecture_source_result": detail["source"]["result"],
            "architecture_source_result_sha256": detail["source"]["result_sha256"],
            "physical_interface_checkpoint": detail["artifacts"]["interfaces"][
                "physical_true"
            ]["path"],
            "physical_interface_checkpoint_sha256": detail["artifacts"][
                "interfaces"
            ]["physical_true"]["sha256"],
            "training_data_sha256": training_hash,
            "minibatch_schedule_sha256": schedule_hash,
        },
        "schedule_rows": {
            "first": config.schedule_steps[0],
            "last": config.schedule_steps[1],
        },
        "records": records,
        "state_checks": state_checks,
        "gates": {
            **seed_gates,
            "gradient_additivity_absolute": (
                maximum_additivity_error <= config.additivity_tolerance
            ),
            "gradient_additivity_relative": (
                maximum_relative_additivity_error
                <= config.relative_additivity_tolerance
            ),
            "gradient_additivity": (
                maximum_additivity_error <= config.additivity_tolerance
                and maximum_relative_additivity_error
                <= config.relative_additivity_tolerance
            ),
            "maximum_gradient_additivity_error": maximum_additivity_error,
            "maximum_relative_gradient_additivity_error": (
                maximum_relative_additivity_error
            ),
            "state_unchanged": all(
                value["interface_unchanged"] and value["model_unchanged"]
                for value in state_checks.values()
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
        },
    }
    _write_json(result_path, result)
    learned = condition == "learned_calibrated_equivariant"
    primary = float(
        bool(seed_gates["initial_cross_block_starvation"]) if learned else 0.0
    )
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics={
            "validity": float(validity),
            "initial_cross_block_starvation": primary,
            "persistent_learned_state_conflict": float(
                bool(seed_gates["persistent_learned_state_conflict"])
                if learned
                else 0.0
            ),
            "maximum_gradient_additivity_error": maximum_additivity_error,
            "maximum_relative_gradient_additivity_error": (
                maximum_relative_additivity_error
            ),
            "peak_cuda_allocated_gb": peak,
        },
        primary_metric=primary,
        model_architecture=[
            joint.preflight.TINYLLM_PRESETS[preset][0],
            joint.preflight.TINYLLM_PRESETS[preset][2],
            1,
        ],
        model_parameters=0,
        training_time=float(result["wall_seconds"]),
        model_checkpoint=detail["artifacts"]["interfaces"]["physical_true"]["path"],
        observations=[f"detail={result_path}", "No optimizer step; gradients only."],
    )


def classify_aggregates(
    strata: Mapping[str, Mapping[str, Any]], valid: bool
) -> tuple[str, bool, bool]:
    if not valid:
        return "invalid", False, False
    initial = all(
        strata[f"{preset}/learned_calibrated_equivariant"][
            "initial_cross_block_starvation_gate"
        ]
        for preset in PRESETS
    )
    persistent = all(
        strata[f"{preset}/learned_calibrated_equivariant"][
            "persistent_learned_state_conflict_gate"
        ]
        for preset in PRESETS
    )
    if initial and persistent:
        return "global_clip_starvation_and_persistent_conflict", True, True
    if initial:
        return "initial_cross_block_clip_starvation_only", True, False
    if persistent:
        return "persistent_objective_conflict_without_initial_starvation", False, True
    return "no_registered_gradient_failure_mechanism", False, False


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: GradientAttributionConfig
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
            learned = condition == "learned_calibrated_equivariant"
            initial_count = (
                sum(
                    bool(detail["gates"]["initial_cross_block_starvation"])
                    for detail in selected
                )
                if learned
                else None
            )
            persistent_count = (
                sum(
                    bool(detail["gates"]["persistent_learned_state_conflict"])
                    for detail in selected
                )
                if learned
                else None
            )
            strata[f"{preset}/{condition}"] = {
                "valid_count": sum(
                    bool(detail["gates"]["validity"]) for detail in selected
                ),
                "initial_cross_block_starvation_count": initial_count,
                "persistent_learned_state_conflict_count": persistent_count,
                "initial_cross_block_starvation_gate": (
                    initial_count >= config.required_seed_passes if learned else None
                ),
                "persistent_learned_state_conflict_gate": (
                    persistent_count >= config.required_seed_passes
                    if learned
                    else None
                ),
                "initial_by_seed": (
                    {
                        str(detail["seed"]): bool(
                            detail["gates"]["initial_cross_block_starvation"]
                        )
                        for detail in selected
                    }
                    if learned
                    else None
                ),
                "persistent_by_seed": (
                    {
                        str(detail["seed"]): bool(
                            detail["gates"]["persistent_learned_state_conflict"]
                        )
                        for detail in selected
                    }
                    if learned
                    else None
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
        initial = persistent = False
    else:
        classification, initial, persistent = classify_aggregates(strata, valid)
    return {
        "valid": valid,
        "complete_primary_population": complete,
        "classification": classification,
        "initial_cross_block_starvation_population_gate": initial,
        "persistent_learned_state_conflict_population_gate": persistent,
        "strata": strata,
        "registered_gate": {
            "required_seed_passes": config.required_seed_passes,
            "global_clip_maximum": config.global_clip_maximum,
            "cross_block_suppression_maximum": (
                config.cross_block_suppression_maximum
            ),
            "absolute_additivity_tolerance": config.additivity_tolerance,
            "relative_additivity_tolerance": (
                config.relative_additivity_tolerance
            ),
            "sensor_descent_ratio_maximum": 0.0,
            "states": list(STATES),
            "batches": list(BATCHES),
        },
    }


def _experiments(
    config: GradientAttributionConfig,
    task: CircleTaskConfig,
    details: Mapping[tuple[str, str, int], Mapping[str, Any]],
    output_dir: Path,
    implementation: str,
) -> list[Experiment]:
    common = {
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "implementation_sha256": implementation,
    }
    return [
        Experiment(
            id=f"tinyllm-joint-gradient-{preset}-{condition}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM joint gradient attribution {preset} {condition} seed {seed}",
            parameters={
                **common,
                "preset": preset,
                "condition": condition,
                "seed": seed,
                "source_result_sha256": details[(preset, condition, seed)][
                    "_result_sha256"
                ],
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
    return detail


def _campaign_fingerprint(
    config: GradientAttributionConfig,
    task: CircleTaskConfig,
    implementation: str,
    source_manifest: str,
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "task_config": asdict(task),
            "implementation_sha256": implementation,
            "source_manifest_sha256": source_manifest,
        }
    )


async def run_campaign(
    config: GradientAttributionConfig, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    source_campaign, task, source_details = _source_population(config)
    source_paths = [
        Path(source_details[(preset, condition, seed)]["_result_path"])
        for preset in config.presets
        for condition in config.conditions
        for seed in config.seeds
    ]
    source_manifest = _manifest_sha256(source_paths)
    experiments = _experiments(
        config, task, source_details, output_dir, implementation
    )
    fingerprint = _campaign_fingerprint(
        config, task, implementation, source_manifest
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
        project_name="tinyllm_joint_interface_gradient_attribution",
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
    runner = AsyncExperimentRunner(lab, gradient_attribution_worker)
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
    diagnostic_paths = [Path(detail["artifacts"]["diagnostics"]) for detail in details]
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
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_campaign_fingerprint": source_campaign["campaign_fingerprint"],
            "selected_result_manifest_sha256": source_manifest,
        },
        "summary": {
            "requested": len(experiments),
            "reused": len(existing),
            "scheduled": len(pending),
            "completed": len(details),
            "failed": len(experiments) - len(details),
            "trained_parameters": 0,
            "optimizer_steps": 0,
            "gradient_snapshots": 4 * len(details),
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
            _manifest_sha256(diagnostic_paths) if diagnostic_paths else None
        ),
        "results": [
            {
                "experiment_id": result.experiment_id,
                "status": result.status.value,
                "metrics": result.metrics,
                "error": result.error,
            }
            for result in results
        ],
        "method_boundaries": [
            "The Stage A outcome and logged aggregate gradients were known before registration.",
            "No optimizer step, parameter fit, probe, or checkpoint modification occurs.",
            "Gradients are local to the saved initial/final states and first/last training minibatches.",
            "Evaluation mode removes dropout and does not replay every stochastic training gradient.",
            "Initial starvation is a causal fact about the registered update geometry, not by itself a complete 600-step explanation.",
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
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
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
            "data/experiments/tinyllm_joint_interface_gradient_attribution/"
            "20260811_d6_d10_registered_v2"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.presets = "d6"
        args.conditions = "learned_calibrated_equivariant"
        args.seeds = "7"
        args.allow_underpowered = True
    config = GradientAttributionConfig(
        source_root=args.source_root,
        presets=_comma_strings(args.presets),
        conditions=_comma_strings(args.conditions),
        seeds=_comma_ints(args.seeds),
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
