#!/usr/bin/env python3
"""Run the prospective d6/d10 calibrated TinyLLM family replication."""

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
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_architecture_replication_preflight as preflight
import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibrated_frontend_causal_closure as closure
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.tinyllm-calibrated-architecture-replication.v1"
HYPOTHESIS_ID = preflight.HYPOTHESIS_ID
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-calibrated-architecture-replication-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "36c1a8c35823fda3076b6a73648facd7fc18513c3c969bfa297ca9c0b34c4c77"
)

PRIMARY_PRESETS = preflight.PROSPECTIVE_PRESETS
CONDITIONS = preflight.CONDITIONS
PRIMARY_CONDITIONS = preflight.PRIMARY_CONDITIONS
SEEDS = preflight.SEEDS
REPRESENTATION_CUTS = calibrated.CUTS
PRIMARY_REGIMES = preflight.PRIMARY_REGIMES
CAUSAL_CUTS = closure.CUTS


@dataclass(frozen=True)
class ArchitectureReplicationConfig:
    presets: tuple[str, ...] = PRIMARY_PRESETS
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    vector_channels: int = 16
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
    closure_batch_size: int = 256
    cosine_minimum: float = 0.90
    branch_accuracy_maximum: float = 0.55
    conditional_log_loss_gain_maximum: float = 0.02
    task_accuracy_margin: float = 0.03
    patch_accuracy_loss_maximum: float = 0.03
    patch_circular_error_increase_maximum: float = math.pi / 16.0
    patch_cross_entropy_increase_maximum: float = 0.10
    replay_tolerance: float = 2e-6
    state_identity_tolerance: float = 1e-7
    group_contract_tolerance: float = 1e-5
    required_seed_passes: int = 4
    shuffle_seed: int = closure.SHUFFLE_SEED
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 0
    gpu_memory_per_experiment_gb: Optional[float] = 4.0
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.presets or any(item not in preflight.ALL_PRESETS for item in self.presets):
            raise ValueError("unknown or empty preset selection")
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty condition selection")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.training_steps < 1 or self.train_samples < 4:
            raise ValueError("training steps and samples must be positive")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("training batch size must be positive and even")
        if self.closure_batch_size < 1:
            raise ValueError("closure batch size must be positive")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside the selected population")
        if not self.allow_underpowered:
            expected = {
                "presets": PRIMARY_PRESETS,
                "conditions": CONDITIONS,
                "seeds": SEEDS,
                "training_steps": 600,
                "train_samples": 4_096,
                "batch_size": 64,
                "probe_steps": 240,
                "probe_train_samples": 2_048,
                "probe_validation_samples": 512,
                "probe_test_samples": 1_024,
                "required_seed_passes": 4,
                "shuffle_seed": closure.SHUFFLE_SEED,
            }
            actual = {
                "presets": self.presets,
                "conditions": self.conditions,
                "seeds": self.seeds,
                "training_steps": self.training_steps,
                "train_samples": self.train_samples,
                "batch_size": self.batch_size,
                "probe_steps": self.probe_steps,
                "probe_train_samples": self.probe_train_samples,
                "probe_validation_samples": self.probe_validation_samples,
                "probe_test_samples": self.probe_test_samples,
                "required_seed_passes": self.required_seed_passes,
                "shuffle_seed": self.shuffle_seed,
            }
            if actual != expected:
                raise ValueError("primary architecture-replication configuration changed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_sources() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "preflight": Path(preflight.__file__),
        "calibrated_source": Path(calibrated.__file__),
        "causal_source": Path(closure.__file__),
        "preregistration": PREREGISTRATION_PATH,
    }
    values = {name: _sha256(path) for name, path in paths.items()}
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("architecture-replication preregistration changed")
    if values["calibrated_source"] != preflight.SOURCE_IMPLEMENTATION_SHA256:
        raise RuntimeError("pinned calibrated source implementation changed")
    return values


def _implementation_digest(sources: Mapping[str, str] | None = None) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(sources or _implementation_sources()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _scientific_preflight() -> tuple[dict[str, Any], str]:
    record = preflight.build_preflight()
    scientific = dict(record)
    scientific.pop("generated_at", None)
    digest = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record, digest


def _json_config(config: ArchitectureReplicationConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(values: Mapping[str, Any]) -> ArchitectureReplicationConfig:
    converted = dict(values)
    for field in ("presets", "conditions", "seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return ArchitectureReplicationConfig(**converted)


def _source_config(
    config: ArchitectureReplicationConfig, preset: str
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
        vector_channels=config.vector_channels,
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
        device_ids=config.device_ids,
        allow_underpowered=True,
    )


def _closure_config(
    config: ArchitectureReplicationConfig,
    condition: str,
    seed: int,
    device: torch.device,
) -> closure.CausalClosureConfig:
    return closure.CausalClosureConfig(
        source_root="unused_in_combined_prospective_worker",
        conditions=(condition,),
        seeds=(seed,),
        accuracy_loss_ceiling=config.patch_accuracy_loss_maximum,
        circular_error_increase_ceiling=(
            config.patch_circular_error_increase_maximum
        ),
        cross_entropy_increase_ceiling=(
            config.patch_cross_entropy_increase_maximum
        ),
        required_seed_passes=1,
        replay_tolerance=config.replay_tolerance,
        state_identity_tolerance=config.state_identity_tolerance,
        shuffle_seed=config.shuffle_seed,
        batch_size=config.closure_batch_size,
        device=str(device),
        allow_underpowered=True,
    )


def representation_endpoint_pass(
    metrics: Mapping[str, Any], config: ArchitectureReplicationConfig
) -> bool:
    return bool(
        float(metrics["cosine_pearson"]) >= config.cosine_minimum
        and float(metrics["balanced_accuracy"]) <= config.branch_accuracy_maximum
        and float(metrics["conditional_log_loss_gain_over_cosine_only"])
        <= config.conditional_log_loss_gain_maximum
    )


def _cell_directory(root: Path, preset: str, condition: str, seed: int) -> Path:
    return root / "runs" / preset / condition / f"seed_{seed}"


def _task_floors(
    preflight_record: Mapping[str, Any], condition: str, seed: int
) -> dict[str, float]:
    if condition not in PRIMARY_CONDITIONS:
        return {}
    anchors = preflight_record["locked_primary_gates"]["task_adequacy"]["anchors"]
    return {
        regime: float(anchors[condition][str(seed)][regime]["prospective_floor"])
        for regime in PRIMARY_REGIMES
    }


def _fingerprint(
    experiment: Experiment,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": experiment.parameters["configuration"],
        "task_config": experiment.parameters["task_config"],
        "preset": experiment.parameters["preset"],
        "condition": experiment.parameters["condition"],
        "seed": experiment.seed,
        "implementation_sha256": experiment.parameters["implementation_sha256"],
        "preflight_sha256": experiment.parameters["preflight_sha256"],
        "identifiability_contract": experiment.parameters["identifiability_contract"],
        "dataset_hashes": experiment.parameters["dataset_hashes"],
        "task_accuracy_floors": experiment.parameters["task_accuracy_floors"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _save_owned_checkpoints(
    system: calibrated.CalibratedTinyLLM,
    output_dir: Path,
    preset: str,
    condition: str,
    seed: int,
    training: Mapping[str, Any],
) -> dict[str, str]:
    model_path = output_dir / "model.pt"
    frontend_path = output_dir / "frontend.pt"
    system.model.to("cpu").save_checkpoint(
        model_path,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "preset": preset,
            "condition": condition,
            "seed": seed,
            "training_data_sha256": training["training_data_sha256"],
            "minibatch_schedule_sha256": training["minibatch_schedule_sha256"],
        },
    )
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "preset": preset,
            "condition": condition,
            "seed": seed,
            "calibration_embedding": (
                system.calibration_embedding.to("cpu").state_dict()
                if system.calibration_embedding is not None
                else None
            ),
            "scalar_embedding": (
                system.scalar_embedding.to("cpu").state_dict()
                if system.scalar_embedding is not None
                else None
            ),
            "encoder": (
                system.encoder.to("cpu").state_dict()
                if system.encoder is not None
                else None
            ),
        },
        frontend_path,
    )
    return {
        "checkpoint": str(model_path),
        "checkpoint_sha256": _sha256(model_path),
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
    }


@torch.no_grad()
def _trained_frontend_contract(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: ArchitectureReplicationConfig,
    device: torch.device,
) -> dict[str, Any]:
    if system.condition == "raw_calibrated":
        return {"kind": "not_applicable_raw_comparator", "pass": True}
    if system.condition == "analytic_calibrated":
        return {
            "kind": "static_analytic_positive_control",
            "pass": True,
            "source": "sealed_preflight_sensor_positive_controls",
        }
    if system.encoder is None:
        raise AssertionError("learned encoder missing")
    sensor = calibrated.decode_sensor_tokens(
        dataset.paired.circle.input_ids[:128], task
    ).to(device)
    packet = dataset.calibration[:128].to(device)
    baseline = preflight._canonical_feature(system.encoder, sensor, packet)
    cells = []
    for action in preflight.GROUP_ACTIONS:
        transformed_sensor, transformed_packet = preflight._apply_acquisition_action(
            sensor, packet, system.encoder.normalized_history, action
        )
        transformed = preflight._canonical_feature(
            system.encoder, transformed_sensor, transformed_packet
        )
        cells.append(
            {
                "action": dict(action),
                "maximum_absolute_error": float(
                    (baseline - transformed).abs().max()
                ),
            }
        )
    maximum = max(item["maximum_absolute_error"] for item in cells)
    return {
        "kind": "trained_learned_encoder_group_contract",
        "actions": cells,
        "maximum_absolute_error": maximum,
        "tolerance": config.group_contract_tolerance,
        "pass": maximum <= config.group_contract_tolerance,
    }


def architecture_replication_worker(
    experiment: Experiment, device_id: int
) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    preset = str(experiment.parameters["preset"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    sources = _implementation_sources()
    if _implementation_digest(sources) != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    preflight.validate_anchors()
    if (
        calibrated.identifiability_contract()
        != experiment.parameters["identifiability_contract"]
    ):
        raise RuntimeError("identifiability contract changed after campaign construction")

    output_root = Path(experiment.parameters["output_dir"])
    output_dir = _cell_directory(output_root, preset, condition, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    source_config = _source_config(config, preset)
    started = time.perf_counter()
    system, training = calibrated.train_cell(
        task, source_config, condition, seed, device, output_dir
    )
    owned_artifacts = _save_owned_checkpoints(
        system, output_dir, preset, condition, seed, training
    )
    system.to(device).eval()
    representation = calibrated.analyze(system, task, source_config, seed, device)

    datasets = closure._datasets(task)
    dataset_hashes = {
        regime: closure._dataset_hash(dataset) for regime, dataset in datasets.items()
    }
    if dataset_hashes != experiment.parameters["dataset_hashes"]:
        raise RuntimeError("held-out causal cohorts changed inside worker")
    frontend_contract = _trained_frontend_contract(
        system, datasets["composition"], task, config, device
    )
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    state_before_causal = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    causal_config = _closure_config(config, condition, seed, device)
    causal_regimes = {}
    causal_arrays: dict[str, np.ndarray] = {}
    for regime_index, regime in enumerate(PRIMARY_REGIMES):
        fiber_count = len(torch.unique(datasets[regime].paired.fiber.fiber_id))
        generator = np.random.default_rng(
            config.shuffle_seed + regime_index * 1_000_003
        )
        shift = int(generator.integers(1, fiber_count))
        record, arrays = closure.analyze_regime(
            system,
            datasets[regime],
            task,
            causal_config,
            device,
            shift,
        )
        source_metrics = representation["task_metrics"][regime]
        same_analysis_cohort = config.probe_test_samples == closure.SPLIT_SPECS[
            regime
        ][2]
        analysis_replay_error = (
            max(
                abs(
                    float(record["baseline_task_metrics"][metric])
                    - float(source_metrics[metric])
                )
                for metric in (
                    "exact_bin_accuracy",
                    "mean_circular_error_radians",
                    "mean_target_cross_entropy",
                )
            )
            if same_analysis_cohort
            else None
        )
        record["analysis_task_replay"] = {
            "maximum_absolute_error": analysis_replay_error,
            "same_cohort": same_analysis_cohort,
            "pass": (
                analysis_replay_error <= config.replay_tolerance
                if analysis_replay_error is not None
                else True
            ),
            "role": (
                "required_primary_same_cohort_replay"
                if same_analysis_cohort
                else "not_comparable_reduced_lifecycle_cohort"
            ),
        }
        record["shuffle_cyclic_shift"] = shift
        causal_regimes[regime] = record
        causal_arrays.update(
            {f"{regime}__{name}": value for name, value in arrays.items()}
        )

    diagnostics_path = output_dir / "closure_diagnostics.npz"
    closure._write_npz(diagnostics_path, causal_arrays)
    diagnostics_sha256 = _sha256(diagnostics_path)
    state_after_causal = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    finite = closure._finite(
        {"representation": representation, "causal": causal_regimes}
    )
    replay_pass = all(
        causal_regimes[regime]["maximum_replay_error"] <= config.replay_tolerance
        and causal_regimes[regime]["analysis_task_replay"]["pass"]
        for regime in PRIMARY_REGIMES
    )
    state_identity_pass = all(
        causal_regimes[regime]["maximum_state_identity_error"]
        <= config.state_identity_tolerance
        for regime in PRIMARY_REGIMES
    )
    state_unchanged = state_before_causal == state_after_causal == {
        "model": training["final_model_state_sha256"],
        "system": training["final_system_state_sha256"],
    }
    representation_cells = {
        f"{cut}:{regime}": representation_endpoint_pass(
            representation["cuts"][cut]["probe"]["evaluations"][regime], config
        )
        for cut in REPRESENTATION_CUTS
        for regime in PRIMARY_REGIMES
    }
    representation_pass = all(representation_cells.values())
    task_floors = {
        str(regime): float(value)
        for regime, value in experiment.parameters["task_accuracy_floors"].items()
    }
    task_adequacy_cells = {
        regime: float(
            representation["task_metrics"][regime]["exact_bin_accuracy"]
        )
        >= floor
        for regime, floor in task_floors.items()
    }
    task_adequacy_pass = (
        all(task_adequacy_cells.values()) if task_adequacy_cells else True
    )
    cut_seed_gates = {
        cut: all(
            causal_regimes[regime]["cuts"][cut]["orbit_average"]["task_gate"]
            for regime in PRIMARY_REGIMES
        )
        for cut in CAUSAL_CUTS
    }
    shuffle_seed_gates = {
        cut: all(
            causal_regimes[regime]["cuts"][cut]["fiber_shuffled"]["task_gate"]
            for regime in PRIMARY_REGIMES
        )
        for cut in CAUSAL_CUTS
    }
    transition_seed_gates = {
        transition: all(
            causal_regimes[regime]["transitions"][transition]["causal_regime"]
            == "quotient_already_closed"
            for regime in PRIMARY_REGIMES
        )
        for transition in closure.TRANSITIONS
    }
    validity = bool(
        finite
        and replay_pass
        and state_identity_pass
        and state_unchanged
        and frontend_contract["pass"]
    )
    causal_pass = all(cut_seed_gates.values())
    joint_seed_pass = bool(
        validity and representation_pass and task_adequacy_pass and causal_pass
    )
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    system_layers, _system_heads, system_width = preflight.TINYLLM_PRESETS[preset]
    fingerprint = _fingerprint(experiment)
    result_path = output_dir / "result.json"
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else (
                "prospective_primary"
                if condition in PRIMARY_CONDITIONS
                else "prospective_matched_comparator"
            )
        ),
        "completed_at": _utc_now(),
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "implementation_sha256": experiment.parameters["implementation_sha256"],
        "implementation_sources": sources,
        "preflight_sha256": experiment.parameters["preflight_sha256"],
        "scientific_fingerprint": fingerprint,
        "model_parameters": sum(parameter.numel() for parameter in system.model.parameters()),
        "system_parameters": sum(parameter.numel() for parameter in system.parameters()),
        "training": {**training, **owned_artifacts},
        "representation": representation,
        "frontend_contract": frontend_contract,
        "causal": {
            "dataset_hashes": dataset_hashes,
            "regimes": causal_regimes,
            "cut_seed_gates": cut_seed_gates,
            "shuffle_seed_gates": shuffle_seed_gates,
            "transition_seed_gates": transition_seed_gates,
        },
        "gates": {
            "representation_cells": representation_cells,
            "representation_pass": representation_pass,
            "task_accuracy_floors": task_floors,
            "task_adequacy_cells": task_adequacy_cells,
            "task_adequacy_pass": task_adequacy_pass,
            "causal_all_cuts_pass": causal_pass,
            "source_and_cut_replay": replay_pass,
            "paired_state_identity": state_identity_pass,
            "state_unchanged": state_unchanged,
            "frontend_contract": frontend_contract["pass"],
            "finite": finite,
            "validity": validity,
            "joint_seed_pass": joint_seed_pass,
        },
        "wall_seconds": elapsed,
        "peak_cuda_allocated_gb": peak,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": diagnostics_sha256,
            **owned_artifacts,
        },
    }
    _write_json(result_path, detail)
    del system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics={
            "joint_seed_pass": float(joint_seed_pass),
            "representation_pass": float(representation_pass),
            "task_adequacy_pass": float(task_adequacy_pass),
            "causal_all_cuts_pass": float(causal_pass),
            "peak_cuda_allocated_gb": peak,
        },
        primary_metric=float(joint_seed_pass),
        model_architecture=[
            task.vocab_size,
            *([system_width] * system_layers),
            task.phase_bins,
        ],
        model_parameters=detail["system_parameters"],
        training_time=elapsed,
        training_history=training["training_history"],
        model_checkpoint=owned_artifacts["checkpoint"],
        observations=[
            f"detail={result_path}",
            f"preset={preset}",
            f"condition={condition}",
        ],
    )


def classify_aggregates(
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]], valid: bool
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    structured_pass = {
        preset: all(arms[preset][condition]["success"] for condition in PRIMARY_CONDITIONS)
        for preset in PRIMARY_PRESETS
    }
    if all(structured_pass.values()):
        raw_specific = all(
            not arms[preset]["raw_calibrated"]["success"]
            for preset in PRIMARY_PRESETS
        )
        if raw_specific:
            return "structured_family_replication_with_specificity", True
        return "structured_family_replication_without_raw_specificity", True
    analytic_all = all(
        arms[preset]["analytic_calibrated"]["success"]
        for preset in PRIMARY_PRESETS
    )
    learned_all = all(
        arms[preset]["learned_calibrated_equivariant"]["success"]
        for preset in PRIMARY_PRESETS
    )
    if analytic_all and not learned_all:
        return "analytic_closure_stable_learned_family_dependent", False
    if any(structured_pass.values()):
        return "preset_dependent_structured_closure", False
    if not analytic_all:
        return "structured_closure_not_architecture_stable", False
    return "mixed_architecture_family_result", False


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: ArchitectureReplicationConfig
) -> dict[str, Any]:
    required = config.required_seed_passes
    arms: dict[str, Any] = {}
    all_valid = all(bool(item["gates"]["validity"]) for item in details)
    controls_pass = True
    for preset in config.presets:
        arms[preset] = {}
        for condition in config.conditions:
            selected = sorted(
                (
                    item
                    for item in details
                    if item["preset"] == preset and item["condition"] == condition
                ),
                key=lambda item: item["seed"],
            )
            if len(selected) != len(config.seeds):
                raise ValueError(f"incomplete detail population {preset}/{condition}")
            joint_count = sum(bool(item["gates"]["joint_seed_pass"]) for item in selected)
            shuffle_count = sum(
                bool(item["causal"]["shuffle_seed_gates"]["pre_block"])
                for item in selected
            )
            population_control = (
                shuffle_count <= 1 if condition in PRIMARY_CONDITIONS else True
            )
            controls_pass = controls_pass and population_control
            arms[preset][condition] = {
                "valid_count": sum(bool(item["gates"]["validity"]) for item in selected),
                "representation_pass_count": sum(
                    bool(item["gates"]["representation_pass"]) for item in selected
                ),
                "task_adequacy_pass_count": sum(
                    bool(item["gates"]["task_adequacy_pass"]) for item in selected
                ),
                "causal_all_cuts_pass_count": sum(
                    bool(item["gates"]["causal_all_cuts_pass"]) for item in selected
                ),
                "joint_pass_count": joint_count,
                "joint_pass_by_seed": {
                    str(item["seed"]): bool(item["gates"]["joint_seed_pass"])
                    for item in selected
                },
                "shuffle_preblock_pass_count": shuffle_count,
                "population_control_pass": population_control,
                "success": joint_count >= required and population_control,
            }
    valid = bool(all_valid and controls_pass)
    if config.allow_underpowered:
        return {
            "valid": valid,
            "controls_pass": controls_pass,
            "required_seed_passes": required,
            "arms": arms,
            "classification": "lifecycle_completed_not_quality_evidence",
            "primary_hypothesis_pass": False,
        }
    classification, primary = classify_aggregates(arms, valid)
    return {
        "valid": valid,
        "controls_pass": controls_pass,
        "required_seed_passes": required,
        "arms": arms,
        "classification": classification,
        "primary_hypothesis_pass": primary,
    }


def _experiments(
    config: ArchitectureReplicationConfig,
    task: CircleTaskConfig,
    output_dir: Path,
    implementation: str,
    preflight_record: Mapping[str, Any],
    preflight_digest: str,
    dataset_hashes: Mapping[str, str],
) -> list[Experiment]:
    common = {
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "implementation_sha256": implementation,
        "preflight_sha256": preflight_digest,
        "identifiability_contract": preflight_record["identifiability_contract"],
        "dataset_hashes": dict(dataset_hashes),
    }
    return [
        Experiment(
            id=f"tinyllm-calibrated-architecture-{preset}-{condition}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM calibrated architecture {preset} {condition} seed {seed}",
            parameters={
                **common,
                "preset": preset,
                "condition": condition,
                "seed": seed,
                "task_accuracy_floors": _task_floors(
                    preflight_record, condition, seed
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
    preset = str(experiment.parameters["preset"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    path = _cell_directory(output_dir, preset, condition, seed) / "result.json"
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
        or detail.get("implementation_sha256")
        != experiment.parameters["implementation_sha256"]
    ):
        return None
    artifacts = detail.get("artifacts", {})
    checks = (
        ("checkpoint", "checkpoint_sha256"),
        ("frontend_checkpoint", "frontend_checkpoint_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    )
    for path_key, digest_key in checks:
        artifact = Path(str(artifacts.get(path_key, "")))
        if not artifact.is_file() or _sha256(artifact) != artifacts.get(digest_key):
            return None
    return detail


def _campaign_fingerprint(
    config: ArchitectureReplicationConfig,
    task: CircleTaskConfig,
    implementation: str,
    preflight_digest: str,
    dataset_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "implementation_sha256": implementation,
        "preflight_sha256": preflight_digest,
        "dataset_hashes": dict(dataset_hashes),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def run_campaign(
    config: ArchitectureReplicationConfig,
    task: CircleTaskConfig,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    preflight_record, preflight_digest = _scientific_preflight()
    datasets = closure._datasets(task)
    dataset_hashes = {
        regime: closure._dataset_hash(dataset) for regime, dataset in datasets.items()
    }
    if dataset_hashes != closure.EXPECTED_DATASET_HASHES:
        raise RuntimeError("held-out causal cohort hashes changed")
    campaign_fingerprint = _campaign_fingerprint(
        config, task, implementation, preflight_digest, dataset_hashes
    )
    experiments = _experiments(
        config,
        task,
        output_dir,
        implementation,
        preflight_record,
        preflight_digest,
        dataset_hashes,
    )
    campaign_path = output_dir / "campaign_results.json"
    if campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing_campaign.get("status") == "completed"
            and existing_campaign.get("campaign_fingerprint") == campaign_fingerprint
            and all(_existing_detail(item, output_dir) is not None for item in experiments)
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing_campaign

    preflight_path = output_dir / "preflight.json"
    if preflight_path.is_file():
        stored_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        stored_scientific = dict(stored_preflight)
        stored_scientific.pop("generated_at", None)
        stored_digest = hashlib.sha256(
            json.dumps(
                stored_scientific, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if stored_digest != preflight_digest:
            raise RuntimeError("stored preflight is incompatible with current contract")
    else:
        _write_json(preflight_path, preflight_record)

    existing = {
        experiment.id: detail
        for experiment in experiments
        if (detail := _existing_detail(experiment, output_dir)) is not None
    }
    pending = [experiment for experiment in experiments if experiment.id not in existing]
    lab = LabConfig(
        project_name="tinyllm_calibrated_architecture_replication",
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
    runner = AsyncExperimentRunner(lab, architecture_replication_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful = {item.experiment_id for item in results if item.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id in successful:
            detail = _existing_detail(experiment, output_dir)
            if detail is not None:
                details.append(detail)
    complete = len(details) == len(experiments)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if complete else "partial",
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "prospective_architecture_family_replication"
        ),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "implementation_sha256": implementation,
        "implementation_sources": sources,
        "preflight_sha256": preflight_digest,
        "preflight": str(preflight_path),
        "campaign_fingerprint": campaign_fingerprint,
        "dataset_hashes": dataset_hashes,
        "anchor_provenance": preflight_record["anchors"],
        "summary": {
            "requested": len(experiments),
            "reused": len(existing),
            "scheduled": len(pending),
            "completed": len(details),
            "failed": len(experiments) - len(details),
        },
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value for key, value in runner.slot_plan.slots_by_device.items()
            },
            "calibration": runner.slot_plan.calibration,
        },
        "aggregates": aggregate_details(details, config) if complete else {},
        "results": [
            {
                "experiment_id": item.experiment_id,
                "status": item.status.value,
                "metrics": item.metrics,
                "error": item.error,
                "model_checkpoint": item.model_checkpoint,
            }
            for item in results
        ],
        "method_boundaries": [
            "d6/d8/d10 presets co-vary depth, width, and head count.",
            "The retained d8 population is an outcome-known anchor, not fresh confirmation.",
            "Raw calibrated cells are specificity comparators because their natural task accuracy is low.",
            "Exact task-orbit averaging is an oracle causal intervention in the synthetic generator.",
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
    parser.add_argument("--presets", default="d6,d10")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--probe-train-samples", type=int, default=2_048)
    parser.add_argument("--probe-validation-samples", type=int, default=512)
    parser.add_argument("--probe-test-samples", type=int, default=1_024)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--slots-per-gpu", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calibrated_architecture_replication/"
            "20260810_d6_d10_preregistered"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.presets = "d10"
        args.conditions = "raw_calibrated"
        args.seeds = "7"
        args.steps = 2
        args.train_samples = 32
        args.batch_size = 8
        args.probe_steps = 20
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 32
        args.allow_underpowered = True
    seeds = _comma_ints(args.seeds)
    config = ArchitectureReplicationConfig(
        presets=_comma_strings(args.presets),
        conditions=_comma_strings(args.conditions),
        seeds=seeds,
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        probe_steps=args.probe_steps,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
        required_seed_passes=4 if len(seeds) >= 5 else len(seeds),
        device_ids=_parse_devices(args.gpus),
        gpu_slots_per_device=args.slots_per_gpu,
        max_parallel_experiments=args.max_parallel,
        max_retries=args.retries,
        resume=args.resume,
        allow_underpowered=args.allow_underpowered,
    )
    task = CircleTaskConfig(train_samples=config.train_samples)
    bundle = asyncio.run(run_campaign(config, task, args.output))
    print(
        json.dumps(
            {
                "summary": bundle["summary"],
                "classification": bundle.get("aggregates", {}).get("classification"),
                "primary_hypothesis_pass": bundle.get("aggregates", {}).get(
                    "primary_hypothesis_pass"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
