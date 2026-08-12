#!/usr/bin/env python3
"""Run the preregistered d6 observable-C3 temporal quotient campaign."""

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

import torch

import experiments.structure_net.tinyllm_c3_temporal_quotient_analysis as analysis
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as preflight
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-quotient-d6.v1"
HYPOTHESIS_ID = stage0.HYPOTHESIS_ID
EVIDENCE_ROLE = "prospective_c3_temporal_quotient_d6"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-quotient-d6-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "33537d32c4e8361fb325a1792a3779a2cb76a0248ef2945d727ac78dcb17d71a"
)
STAGE0_RUNNER_SHA256 = (
    "dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6"
)
ANALYSIS_SHA256 = (
    "89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3"
)
ARMS = stage0.ARMS
STRUCTURED_ARMS = stage0.STRUCTURED_ARMS
SEEDS = (7, 17, 29, 41, 53)
REGIMES = analysis.REGIMES
PRIMARY_CUTS = ("frontend", "full")


@dataclass(frozen=True)
class C3CampaignConfig:
    arms: tuple[str, ...] = ARMS
    seeds: tuple[int, ...] = SEEDS
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    probe_train_latents: int = 2_048
    probe_validation_latents: int = 512
    probe_test_latents: int = 1_024
    probe_steps: int = 240
    probe_width: int = 128
    probe_batch_size: int = 256
    probe_validation_interval: int = 20
    probe_patience: int = 5
    extraction_batch_size: int = 256
    causal_batch_size: int = 128
    mechanism_latents: int = 256
    required_seed_passes: int = 4
    target_correlation_minimum: float = 0.90
    composition_accuracy_minimum: float = 0.50
    extrapolation_accuracy_minimum: float = 0.35
    composition_cross_entropy_maximum: float = 1.80
    extrapolation_cross_entropy_maximum: float = 2.20
    composition_coverage_minimum: int = 14
    extrapolation_coverage_minimum: int = 12
    relative_accuracy_margin: float = 0.03
    deck_accuracy_maximum: float = 0.3834
    conditional_log_loss_gain_maximum: float = 0.02
    action_error_maximum: float = 1e-5
    replay_error_maximum: float = 2e-6
    derangement_population_maximum: int = 1
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 0
    gpu_memory_per_experiment_gb: Optional[float] = 2.0
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.arms or set(self.arms).difference(ARMS):
            raise ValueError("unknown or empty arm selection")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.training_steps < 1 or self.train_samples < 4:
            raise ValueError("training sizes must be positive")
        if self.train_samples % 2 or self.batch_size % 2:
            raise ValueError("training and minibatches must contain complete pairs")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside the population")
        if not self.allow_underpowered:
            expected = {
                "arms": ARMS,
                "seeds": SEEDS,
                "training_steps": 600,
                "train_samples": 4_096,
                "batch_size": 64,
                "probe_train_latents": 2_048,
                "probe_validation_latents": 512,
                "probe_test_latents": 1_024,
                "probe_steps": 240,
                "required_seed_passes": 4,
            }
            actual = {name: getattr(self, name) for name in expected}
            if actual != expected:
                raise ValueError("primary C3 d6 configuration changed")


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


def _finite(value: Any) -> bool:
    return stage0._finite(value)


def _json_config(config: C3CampaignConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _config_from_mapping(values: Mapping[str, Any]) -> C3CampaignConfig:
    converted = dict(values)
    for field in ("arms", "seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return C3CampaignConfig(**converted)


def _source_hashes() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "stage0_runner": _sha256(Path(stage0.__file__)),
        "analysis": _sha256(Path(analysis.__file__)),
        "generator_preflight": _sha256(Path(preflight.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("C3 d6 preregistration changed")
    if values["stage0_runner"] != STAGE0_RUNNER_SHA256:
        raise RuntimeError("sealed C3 Stage-0 runner changed")
    if values["analysis"] != ANALYSIS_SHA256:
        raise RuntimeError("sealed C3 analysis implementation changed")
    return values


def _implementation_digest(sources: Mapping[str, str] | None = None) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(sources or _source_hashes()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _probe_config(config: C3CampaignConfig) -> analysis.ProbeConfig:
    return analysis.ProbeConfig(
        train_latents=config.probe_train_latents,
        validation_latents=config.probe_validation_latents,
        test_latents=config.probe_test_latents,
        steps=config.probe_steps,
        width=config.probe_width,
        batch_size=config.probe_batch_size,
        validation_interval=config.probe_validation_interval,
        patience=config.probe_patience,
        extraction_batch_size=config.extraction_batch_size,
    )


def _lifecycle_config(config: C3CampaignConfig, seed: int) -> stage0.LifecycleConfig:
    return stage0.LifecycleConfig(
        preset="d6",
        steps=config.training_steps,
        split_step=max(1, config.training_steps // 2),
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        evaluation_samples=config.probe_test_latents,
        seed=seed,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
    )


def _dataset_contract(
    task: stage0.C3TaskConfig, config: C3CampaignConfig
) -> dict[str, Any]:
    probe = analysis.build_probe_datasets(task, _probe_config(config))
    task_sets = analysis.build_task_datasets(task, config.probe_test_latents)
    split = analysis.validate_split_contract(probe, task_sets)
    if not split["pass"]:
        raise RuntimeError("C3 final split contract failed")
    return {
        "split_contract": split,
        "probe_hashes": {
            name: analysis.dataset_hash(dataset) for name, dataset in probe.items()
        },
        "task_hashes": {
            name: analysis.dataset_hash(dataset)
            for name, dataset in task_sets.items()
        },
    }


def _cell_directory(root: Path, arm: str, seed: int) -> Path:
    return root / "runs" / arm / f"seed_{seed}"


def _fingerprint(experiment: Experiment) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": experiment.parameters["configuration"],
        "task": experiment.parameters["task"],
        "arm": experiment.parameters["arm"],
        "seed": experiment.seed,
        "implementation_sha256": experiment.parameters["implementation_sha256"],
        "dataset_contract": experiment.parameters["dataset_contract"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def natural_task_pass(
    metrics: Mapping[str, Mapping[str, Any]], config: C3CampaignConfig
) -> bool:
    for regime in REGIMES:
        item = metrics[regime]
        accuracy_minimum = (
            config.composition_accuracy_minimum
            if regime == "composition"
            else config.extrapolation_accuracy_minimum
        )
        cross_entropy_maximum = (
            config.composition_cross_entropy_maximum
            if regime == "composition"
            else config.extrapolation_cross_entropy_maximum
        )
        coverage_minimum = (
            config.composition_coverage_minimum
            if regime == "composition"
            else config.extrapolation_coverage_minimum
        )
        if not (
            float(item["posterior_mean_correlation"])
            >= config.target_correlation_minimum
            and float(item["exact_bin_accuracy"]) >= accuracy_minimum
            and float(item["target_cross_entropy"]) <= cross_entropy_maximum
            and int(item["predicted_bin_coverage"]) >= coverage_minimum
        ):
            return False
    return True


def representation_pass(
    record: Mapping[str, Any], config: C3CampaignConfig
) -> bool:
    for cut in PRIMARY_CUTS:
        cell = record["cuts"][cut]
        for regime in REGIMES:
            semantic = cell["semantic"]["evaluations"][regime]
            deck = cell["conditional_deck"]["evaluations"][regime]
            if not (
                float(semantic["target_correlation"])
                >= config.target_correlation_minimum
                and float(deck["balanced_accuracy"])
                <= config.deck_accuracy_maximum
                and float(deck["conditional_log_loss_gain"])
                <= config.conditional_log_loss_gain_maximum
            ):
                return False
    return True


def causal_pass(record: Mapping[str, Any], config: C3CampaignConfig) -> bool:
    return all(
        cell["orbit_barycenter_preservation"]["pass"]
        and float(cell["maximum_identity_replay_logit_error"])
        <= config.replay_error_maximum
        for cell in record["cuts"].values()
    )


def action_pass(record: Mapping[str, Any], config: C3CampaignConfig) -> bool:
    return all(
        float(cell["maximum_orbit_state_error"])
        <= config.action_error_maximum
        for cell in record["cuts"].values()
    )


def derangement_pass(record: Mapping[str, Any]) -> bool:
    return all(
        cell["target_derangement_preservation"]["pass"]
        for cell in record["cuts"].values()
    )


def _save_artifacts(
    system: stage0.C3TemporalTinyLLM,
    root: Path,
    *,
    arm: str,
    seed: int,
    training: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.pt"
    frontend_path = root / "frontend.pt"
    diagnostics_path = root / "diagnostics.pt"
    final_digest = stage0._state_digest(system)
    system.model.to("cpu").save_checkpoint(
        model_path,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "arm": arm,
            "seed": seed,
            "training_data_sha256": training["training_data_sha256"],
            "minibatch_sha256": training["minibatch_sha256"],
        },
    )
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "arm": arm,
            "seed": seed,
            "sequence_embedding": system.sequence_embedding.to("cpu").state_dict(),
            "encoder": (
                system.encoder.to("cpu").state_dict()
                if system.encoder is not None
                else None
            ),
        },
        frontend_path,
    )
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "arm": arm,
            "seed": seed,
            "diagnostics": dict(diagnostics),
        },
        diagnostics_path,
    )
    model = TinyLLMModel.from_checkpoint(model_path, map_location="cpu")
    loaded = stage0.C3TemporalTinyLLM(model, arm)
    frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
    loaded.sequence_embedding.load_state_dict(frontend["sequence_embedding"])
    if loaded.encoder is not None:
        if frontend["encoder"] is None:
            raise RuntimeError("learned frontend checkpoint is missing encoder")
        loaded.encoder.load_state_dict(frontend["encoder"])
    reload_pass = stage0._state_digest(loaded) == final_digest
    reloaded_diagnostics = torch.load(
        diagnostics_path, map_location="cpu", weights_only=True
    )
    diagnostics_reload_pass = (
        reloaded_diagnostics.get("schema_version") == SCHEMA_VERSION
        and reloaded_diagnostics.get("hypothesis_id") == HYPOTHESIS_ID
        and reloaded_diagnostics.get("arm") == arm
        and int(reloaded_diagnostics.get("seed", -1)) == seed
    )
    return {
        "checkpoint": str(model_path),
        "checkpoint_sha256": _sha256(model_path),
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "checkpoint_reload_pass": reload_pass,
        "diagnostics_reload_pass": diagnostics_reload_pass,
    }


def campaign_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    started = time.perf_counter()
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = stage0.C3TaskConfig(**experiment.parameters["task"])
    arm = str(experiment.parameters["arm"])
    seed = int(experiment.seed)
    sources = _source_hashes()
    implementation = _implementation_digest(sources)
    if implementation != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("campaign implementation changed after construction")
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    torch.use_deterministic_algorithms(True)
    preflight_record = preflight.build_preflight()
    if preflight_record["classification"] != "c3_temporal_quotient_preflight_passed":
        raise RuntimeError("pinned no-training C3 preflight no longer passes")
    dataset_contract = _dataset_contract(task, config)
    if dataset_contract != experiment.parameters["dataset_contract"]:
        raise RuntimeError("fresh C3 dataset contract changed")

    lifecycle = _lifecycle_config(config, seed)
    training_dataset, batches, data_hash, batch_hash = stage0.protocol_material(
        task, lifecycle
    )
    protocol = stage0.validate_training_protocol(training_dataset)
    if not protocol["pass"]:
        raise RuntimeError("paired training protocol failed")
    system = stage0.build_system(task, lifecycle, arm, device)
    initial_model = stage0._state_digest(system.model)
    initial_system = stage0._state_digest(system)
    counts = stage0.parameter_counts(system)
    probe_datasets = analysis.build_probe_datasets(task, _probe_config(config))
    feature_before = stage0._feature_contract(system, probe_datasets["composition"])
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    training_started = time.perf_counter()
    history = stage0.train_steps(
        system,
        optimizer,
        training_dataset,
        batches,
        task,
        lifecycle,
        device,
        0,
        config.training_steps,
    )
    training_seconds = time.perf_counter() - training_started
    optimizer_digest = stage0._optimizer_digest(optimizer)
    del optimizer
    for parameter in system.parameters():
        parameter.grad = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_system = stage0._state_digest(system)
    feature_after = stage0._feature_contract(system, probe_datasets["composition"])

    analysis_started = time.perf_counter()
    task_datasets = analysis.build_task_datasets(task, config.probe_test_latents)
    task_metrics = {
        regime: stage0.evaluate(
            system,
            dataset,
            task,
            device,
            batch_size=config.extraction_batch_size,
        )[0]
        for regime, dataset in task_datasets.items()
    }
    representations = analysis.representation_analysis(
        system,
        probe_datasets,
        _probe_config(config),
        seed=seed,
        device=device,
    )
    causal = {
        regime: analysis.causal_orbit_analysis(
            system,
            probe_datasets[regime],
            task,
            device=device,
            batch_size=config.causal_batch_size,
        )
        for regime in REGIMES
    }
    mechanism = {
        regime: analysis.raw_reynolds_analysis(
            system,
            probe_datasets[regime],
            task,
            device=device,
            latent_limit=config.mechanism_latents,
        )
        for regime in REGIMES
    }
    analysis_seconds = time.perf_counter() - analysis_started

    validity = (
        protocol["pass"]
        and feature_before["pass"]
        and feature_after["pass"]
        and final_system != initial_system
        and all(
            math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"])
            for item in history
        )
    )
    task_gate = natural_task_pass(task_metrics, config)
    representation_gate = (
        representation_pass(representations, config)
        if arm in STRUCTURED_ARMS
        else False
    )
    causal_gate = (
        all(causal_pass(causal[regime], config) for regime in REGIMES)
        if arm in STRUCTURED_ARMS
        else False
    )
    action_gate = (
        feature_before["pass"]
        and feature_after["pass"]
        and all(action_pass(causal[regime], config) for regime in REGIMES)
        if arm in STRUCTURED_ARMS
        else True
    )
    replay_gate = all(
        all(
            float(cell["maximum_identity_replay_logit_error"])
            <= config.replay_error_maximum
            for cell in causal[regime]["cuts"].values()
        )
        for regime in REGIMES
    )
    derangement = {
        regime: derangement_pass(causal[regime]) for regime in REGIMES
    }
    provisional_joint = (
        validity
        and task_gate
        and representation_gate
        and causal_gate
        and action_gate
        and replay_gate
    ) if arm in STRUCTURED_ARMS else validity
    diagnostics = {
        "task_metrics": task_metrics,
        "representation": representations,
        "causal": causal,
        "raw_reynolds": mechanism,
    }
    root = Path(experiment.parameters["output_dir"])
    artifacts = _save_artifacts(
        system,
        root,
        arm=arm,
        seed=seed,
        training={
            "training_data_sha256": data_hash,
            "minibatch_sha256": batch_hash,
        },
        diagnostics=diagnostics,
    )
    validity = validity and artifacts["checkpoint_reload_pass"] and artifacts[
        "diagnostics_reload_pass"
    ]
    provisional_joint = provisional_joint and validity
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed" if validity else "invalid",
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "arm": arm,
        "seed": seed,
        "scientific_fingerprint": _fingerprint(experiment),
        "implementation_sha256": implementation,
        "source_hashes": sources,
        "configuration": _json_config(config),
        "task": asdict(task),
        "protocol": {
            "training": protocol,
            "training_data_sha256": data_hash,
            "minibatch_sha256": batch_hash,
            "dataset_contract": dataset_contract,
        },
        "parameter_counts": counts,
        "initialization": {
            "tinyllm_sha256": initial_model,
            "system_sha256": initial_system,
            "final_system_sha256": final_system,
            "final_optimizer_sha256": optimizer_digest,
        },
        "training": {
            "seconds": training_seconds,
            "history": history,
        },
        "analysis_seconds": analysis_seconds,
        "feature_contract_before": feature_before,
        "feature_contract_after": feature_after,
        **diagnostics,
        "gates": {
            "validity": validity,
            "natural_task": task_gate,
            "representation": representation_gate,
            "causal_all_cuts": causal_gate,
            "exact_action": action_gate,
            "identity_replay": replay_gate,
            "target_derangement_pass_by_regime": derangement,
            "joint_without_relative_utility": provisional_joint,
        },
        "artifacts": artifacts,
        "method_boundaries": [
            "Conditional probes establish only registered estimator decodability.",
            "The exact Reynolds defect defines raw synthesis; Taylor terms are secondary approximations.",
            "D10 is not authorized by this d6 campaign.",
        ],
    }
    result_path = root / "result.json"
    detail["artifacts"]["result"] = str(result_path)
    _write_json(result_path, detail)
    detail["artifacts"]["result_sha256"] = _sha256(result_path)
    # Record the result digest in the returned object; the JSON cannot contain
    # its own digest without a recursive fixed-point convention.
    total_seconds = time.perf_counter() - started
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics={
            "accuracy": task_metrics["composition"]["exact_bin_accuracy"],
            "validity": float(validity),
            "natural_task_pass": float(task_gate),
            "representation_pass": float(representation_gate),
            "causal_pass": float(causal_gate),
            "joint_without_relative_utility": float(provisional_joint),
        },
        primary_metric=float(provisional_joint),
        model_architecture=[6, 6, 384],
        model_parameters=counts["total"],
        training_time=total_seconds,
        training_history=history,
        model_checkpoint=artifacts["checkpoint"],
        observations=[str(result_path), artifacts["diagnostics"]],
    )


def relative_utility(
    structured: Mapping[str, Any],
    raw: Mapping[str, Any] | None,
    config: C3CampaignConfig,
) -> dict[str, Any]:
    if raw is None or not raw["gates"]["natural_task"]:
        return {
            "applicable": False,
            "reason": "matched_raw_arm_misses_absolute_task_gate",
            "pass": True,
        }
    cells = {}
    for regime in REGIMES:
        structured_accuracy = float(
            structured["task_metrics"][regime]["exact_bin_accuracy"]
        )
        raw_accuracy = float(raw["task_metrics"][regime]["exact_bin_accuracy"])
        cells[regime] = {
            "structured_accuracy": structured_accuracy,
            "raw_accuracy": raw_accuracy,
            "difference": structured_accuracy - raw_accuracy,
            "pass": structured_accuracy >= raw_accuracy - config.relative_accuracy_margin,
        }
    return {
        "applicable": True,
        "cells": cells,
        "pass": all(cell["pass"] for cell in cells.values()),
    }


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: C3CampaignConfig
) -> dict[str, Any]:
    by_key = {(str(item["arm"]), int(item["seed"])): item for item in details}
    requested = {(arm, seed) for arm in config.arms for seed in config.seeds}
    complete = requested.issubset(by_key)
    validity = all(
        item.get("status") == "completed"
        and item.get("gates", {}).get("validity") is True
        and _finite(item)
        for item in details
    )
    arms: dict[str, Any] = {}
    for arm in config.arms:
        selected = [by_key[(arm, seed)] for seed in config.seeds if (arm, seed) in by_key]
        joint_by_seed = {}
        relative_by_seed = {}
        for item in selected:
            seed = int(item["seed"])
            relative = (
                relative_utility(item, by_key.get(("raw", seed)), config)
                if arm in STRUCTURED_ARMS
                else {"applicable": False, "pass": True, "reason": "raw_arm"}
            )
            relative_by_seed[str(seed)] = relative
            joint_by_seed[str(seed)] = bool(
                item["gates"]["joint_without_relative_utility"] and relative["pass"]
            )
        derangement_counts = {
            regime: sum(
                bool(item["gates"]["target_derangement_pass_by_regime"][regime])
                for item in selected
            )
            for regime in REGIMES
        }
        control_pass = all(
            count <= config.derangement_population_maximum
            for count in derangement_counts.values()
        ) if arm in STRUCTURED_ARMS else True
        joint_count = sum(joint_by_seed.values())
        arms[arm] = {
            "valid_count": sum(item["gates"]["validity"] for item in selected),
            "natural_task_pass_count": sum(
                item["gates"]["natural_task"] for item in selected
            ),
            "representation_pass_count": sum(
                item["gates"]["representation"] for item in selected
            ),
            "causal_pass_count": sum(
                item["gates"]["causal_all_cuts"] for item in selected
            ),
            "exact_action_pass_count": sum(
                item["gates"]["exact_action"] for item in selected
            ),
            "joint_pass_by_seed": joint_by_seed,
            "joint_pass_count": joint_count,
            "relative_utility_by_seed": relative_by_seed,
            "derangement_pass_count_by_regime": derangement_counts,
            "population_control_pass": control_pass,
            "success": (
                joint_count >= config.required_seed_passes and control_pass
                if arm in STRUCTURED_ARMS
                else None
            ),
        }
    analytic = arms.get("analytic", {})
    learned = arms.get("learned_c3", {})
    analytic_success = analytic.get("success") is True
    learned_success = learned.get("success") is True
    controls_pass = all(
        arms.get(arm, {}).get("population_control_pass") is True
        for arm in STRUCTURED_ARMS
        if arm in arms
    )
    if not validity or not controls_pass:
        classification = "invalid"
    elif "analytic" in arms and not analytic_success:
        classification = "c3_positive_control_task_failure"
    elif analytic_success and learned_success:
        classification = "c3_d6_structured_quotient_supported"
    elif analytic_success and learned.get("representation_pass_count", 0) >= config.required_seed_passes:
        classification = "c3_representation_without_causal_utility"
    elif analytic_success:
        classification = "c3_architectural_invariance_not_learned_useful"
    else:
        classification = "incomplete"
    return {
        "valid": validity,
        "complete_requested_grid": complete,
        "required_seed_passes": config.required_seed_passes,
        "arms": arms,
        "controls_pass": controls_pass,
        "classification": classification,
        "primary_hypothesis_pass": analytic_success and learned_success and controls_pass,
    }


def _experiments(
    config: C3CampaignConfig,
    task: stage0.C3TaskConfig,
    output: Path,
    implementation: str,
    dataset_contract: Mapping[str, Any],
    arms: Sequence[str],
) -> list[Experiment]:
    result = []
    for arm in arms:
        if arm not in config.arms:
            continue
        for seed in config.seeds:
            experiment = Experiment(
                id=f"tinyllm-c3-temporal-d6-{arm}-seed{seed}",
                hypothesis_id=HYPOTHESIS_ID,
                name=f"TinyLLM C3 temporal d6 {arm} seed {seed}",
                parameters={
                    "configuration": _json_config(config),
                    "task": asdict(task),
                    "arm": arm,
                    "implementation_sha256": implementation,
                    "dataset_contract": dict(dataset_contract),
                    "output_dir": str(_cell_directory(output, arm, seed)),
                    "architecture": [6, 6, 384],
                    "epochs": config.training_steps,
                },
                seed=seed,
            )
            result.append(experiment)
    return result


def _existing_detail(experiment: Experiment, output: Path) -> dict[str, Any] | None:
    arm = str(experiment.parameters["arm"])
    path = _cell_directory(output, arm, int(experiment.seed)) / "result.json"
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
    for path_key, digest_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("frontend_checkpoint", "frontend_checkpoint_sha256"),
        ("diagnostics", "diagnostics_sha256"),
    ):
        artifact = Path(str(artifacts.get(path_key, "")))
        if not artifact.is_file() or _sha256(artifact) != artifacts.get(digest_key):
            return None
    return detail


async def _run_stage(
    runner: AsyncExperimentRunner,
    experiments: Sequence[Experiment],
    output: Path,
) -> tuple[list[dict[str, Any]], list[ExperimentResult], int]:
    existing = {
        item.id: detail
        for item in experiments
        if (detail := _existing_detail(item, output)) is not None
    }
    pending = [item for item in experiments if item.id not in existing]
    results = await runner.run_experiments(list(pending)) if pending else []
    successful = {item.experiment_id for item in results if item.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id in successful:
            detail = _existing_detail(experiment, output)
            if detail is not None:
                details.append(detail)
    return details, results, len(existing)


async def run_campaign(
    config: C3CampaignConfig,
    task: stage0.C3TaskConfig,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sources = _source_hashes()
    implementation = _implementation_digest(sources)
    dataset_contract = _dataset_contract(task, config)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "configuration": _json_config(config),
                "task": asdict(task),
                "implementation": implementation,
                "dataset_contract": dataset_contract,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    all_experiments = _experiments(
        config, task, output, implementation, dataset_contract, config.arms
    )
    lab = LabConfig(
        project_name="tinyllm_c3_temporal_quotient_d6",
        results_dir=str(output),
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
    runner = AsyncExperimentRunner(lab, campaign_worker)
    details: list[dict[str, Any]] = []
    results: list[ExperimentResult] = []
    reused = 0
    stopped = 0
    stage_order = (("analytic",), ("raw", "learned_c3"))
    for stage_index, arms in enumerate(stage_order):
        stage_experiments = [
            item for item in all_experiments if item.parameters["arm"] in arms
        ]
        if not stage_experiments:
            continue
        stage_details, stage_results, stage_reused = await _run_stage(
            runner, stage_experiments, output
        )
        details.extend(stage_details)
        results.extend(stage_results)
        reused += stage_reused
        if stage_index == 0 and not config.allow_underpowered:
            analytic_aggregate = aggregate_details(stage_details, config)
            analytic = analytic_aggregate["arms"].get("analytic", {})
            if analytic.get("success") is not True:
                stopped = len(all_experiments) - len(details)
                break
    complete_or_stopped = len(details) + stopped == len(all_experiments)
    aggregate = aggregate_details(details, config)
    if stopped and aggregate["classification"] == "invalid":
        # A valid preregistered positive-control stop has no learned control
        # population, so absence of that later population is not invalidity.
        analytic = aggregate["arms"].get("analytic", {})
        if analytic.get("valid_count") == len(config.seeds):
            aggregate["valid"] = True
            aggregate["controls_pass"] = analytic.get("population_control_pass", False)
            aggregate["classification"] = "c3_positive_control_task_failure"
    status = "completed" if complete_or_stopped and aggregate["valid"] else "partial"
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": status,
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "configuration": _json_config(config),
        "task": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "implementation_sha256": implementation,
        "source_hashes": sources,
        "campaign_fingerprint": fingerprint,
        "dataset_contract": dataset_contract,
        "summary": {
            "requested": len(all_experiments),
            "completed": len(details),
            "failed": len(all_experiments) - len(details) - stopped,
            "stopped_by_positive_control": stopped,
            "reused": reused,
            "scheduled": len(results),
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
        "aggregates": aggregate,
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
            "The analytic population is executed first under a preregistered stop rule.",
            "Probe conclusions are estimator- and split-relative.",
            "D10 is neither scheduled nor licensed by this campaign.",
        ],
    }
    _write_json(output / "campaign_results.json", bundle)
    return bundle


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shakedown", action="store_true")
    mode.add_argument("--execute-primary", action="store_true")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--slots-per-gpu", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_c3_temporal_quotient/"
            "20260811_d6_preregistered"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        config = C3CampaignConfig(
            arms=("analytic",),
            seeds=(7,),
            training_steps=2,
            train_samples=32,
            batch_size=8,
            probe_train_latents=32,
            probe_validation_latents=16,
            probe_test_latents=24,
            probe_steps=4,
            probe_width=8,
            probe_batch_size=16,
            probe_validation_interval=2,
            probe_patience=2,
            extraction_batch_size=32,
            causal_batch_size=24,
            mechanism_latents=8,
            required_seed_passes=1,
            device_ids=_parse_devices(args.gpus),
            gpu_slots_per_device=args.slots_per_gpu,
            max_parallel_experiments=1,
            max_retries=args.retries,
            resume=args.resume,
            allow_underpowered=True,
        )
    else:
        config = C3CampaignConfig(
            device_ids=_parse_devices(args.gpus),
            gpu_slots_per_device=args.slots_per_gpu,
            max_parallel_experiments=args.max_parallel,
            max_retries=args.retries,
            resume=args.resume,
        )
    bundle = asyncio.run(run_campaign(config, stage0.C3TaskConfig(), args.output))
    print(
        json.dumps(
            {
                "status": bundle["status"],
                "summary": bundle["summary"],
                "classification": bundle["aggregates"]["classification"],
                "primary_hypothesis_pass": bundle["aggregates"][
                    "primary_hypothesis_pass"
                ],
                "output": str(args.output / "campaign_results.json"),
            },
            indent=2,
        )
    )
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
