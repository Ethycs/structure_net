#!/usr/bin/env python3
"""Test whether TinyLLM topology follows an S1 or interval task quotient.

The periodic sensor inputs, model initialization, and minibatch schedule are
matched.  Only the supervised target changes: future phase is either predicted
on S1 or passed through cosine, whose quotient image is an interval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from persim import bottleneck
import ripser as ripser_package
from scipy.stats import pearsonr, spearmanr
import torch

try:
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleDataset,
        CircleTaskConfig,
        _checkpoint_bottleneck_tracking,
        _conditioned_training_data,
        _phase_grid_outputs,
        _persistence_record,
        _pullback_fisher_edge_lengths,
        _resolve_device,
        _state_digest,
        _task_logits,
        _tinyllm_config,
        evaluate_task,
        generate_circle_dataset,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleDataset,
        CircleTaskConfig,
        _checkpoint_bottleneck_tracking,
        _conditioned_training_data,
        _phase_grid_outputs,
        _persistence_record,
        _pullback_fisher_edge_lengths,
        _resolve_device,
        _state_digest,
        _task_logits,
        _tinyllm_config,
        evaluate_task,
        generate_circle_dataset,
    )
from structure_net.components.analyzers import (
    bootstrap_max_h1,
    circular_phase_alignment,
    circular_winding_degree,
    fisher_rao_distance_matrix,
    knn_geodesic_distance_matrix,
    nuisance_collapse_ratio,
    persistence_diagrams,
    representation_distance_matrix,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-task-quotient-contrast.v1"
HYPOTHESIS_ID = "tinyllm-task-topology-follows-quotient-v1"
QUOTIENTS = ("phase_circle", "cosine_interval")
CONDITIONS = ("trained", "label_shuffled")


@dataclass(frozen=True)
class QuotientContrastConfig:
    presets: tuple[str, ...] = ("d6", "d8")
    seeds: tuple[int, ...] = (7, 17, 29)
    quotients: tuple[str, ...] = QUOTIENTS
    conditions: tuple[str, ...] = CONDITIONS
    training_steps: int = 600
    checkpoints: tuple[int, ...] = (0, 100, 300, 600)
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    bootstrap_repeats: int = 20
    bootstrap_fraction: float = 0.8
    fisher_neighbors: int = 6
    device: str = "cuda:auto"
    save_primary_checkpoints: bool = True

    def __post_init__(self) -> None:
        if set(self.quotients).difference(QUOTIENTS):
            raise ValueError(f"quotients must be drawn from {QUOTIENTS}")
        if set(self.conditions).difference(CONDITIONS):
            raise ValueError(f"conditions must be drawn from {CONDITIONS}")
        if not self.presets or not self.seeds or not self.quotients or not self.conditions:
            raise ValueError("campaign axes cannot be empty")
        if self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("training_steps and batch_size must be positive")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted and unique")
        if self.checkpoints[0] != 0 or self.checkpoints[-1] != self.training_steps:
            raise ValueError("checkpoints must include zero and the final step")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_coordinate(dataset: CircleDataset, task: CircleTaskConfig) -> np.ndarray:
    phases = dataset.phases.double().numpy()
    directions = dataset.directions.double().numpy()
    return (phases + directions * task.future_delta) % (2.0 * math.pi)


def with_quotient_targets(
    dataset: CircleDataset,
    task: CircleTaskConfig,
    quotient: str,
) -> CircleDataset:
    """Replace only the target posterior, leaving serialized inputs unchanged."""
    if quotient == "phase_circle":
        return dataset
    if quotient != "cosine_interval":
        raise ValueError(f"unknown quotient: {quotient}")
    values = np.cos(semantic_coordinate(dataset, task))
    centers = np.linspace(-1.0, 1.0, task.phase_bins)
    bin_width = 2.0 / (task.phase_bins - 1)
    probabilities = np.exp(-0.5 * ((centers[None, :] - values[:, None]) / bin_width) ** 2)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return CircleDataset(
        input_ids=dataset.input_ids,
        target_posteriors=torch.from_numpy(probabilities.astype(np.float32)),
        target_bins=torch.from_numpy(probabilities.argmax(axis=1).astype(np.int64)),
        phases=dataset.phases,
        directions=dataset.directions,
    )


def _map_diagnostics(
    posteriors: np.ndarray,
    dataset: CircleDataset,
    task: CircleTaskConfig,
    quotient: str,
) -> Dict[str, Any]:
    target_angles = semantic_coordinate(dataset, task)
    if quotient == "phase_circle":
        bin_angles = 2.0 * math.pi * np.arange(task.phase_bins) / task.phase_bins
        resultant = posteriors @ np.exp(1j * bin_angles)
        predicted = np.angle(resultant) % (2.0 * math.pi)
        alignment = circular_phase_alignment(predicted, target_angles)
        error = np.abs(np.angle(np.exp(1j * (predicted - target_angles))))
        return {
            "kind": "circle_map",
            **alignment,
            "winding_degree": circular_winding_degree(predicted),
            "mean_absolute_circular_error": float(error.mean()),
            "mean_resultant_length": float(np.abs(resultant).mean()),
        }
    centers = np.linspace(-1.0, 1.0, task.phase_bins)
    predicted = posteriors @ centers
    target = np.cos(target_angles)
    return {
        "kind": "interval_map",
        "pearson_correlation": float(pearsonr(predicted, target).statistic),
        "spearman_correlation": float(spearmanr(predicted, target).statistic),
        "root_mean_squared_error": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "predicted_range": float(predicted.max() - predicted.min()),
    }


def analyze_quotient_grid(
    model: TinyLLMModel,
    dataset: CircleDataset,
    answer_ids: torch.Tensor,
    device: torch.device,
    task: CircleTaskConfig,
    campaign: QuotientContrastConfig,
    quotient: str,
    *,
    bootstrap_seed: int,
    full_analysis: bool,
) -> Dict[str, Any]:
    posteriors, layer_residuals, final_residuals, posterior_tensor = _phase_grid_outputs(
        model, dataset, answer_ids, device
    )
    predicted_distances = fisher_rao_distance_matrix(posteriors)
    target_distances = fisher_rao_distance_matrix(dataset.target_posteriors.double().numpy())
    predicted_diagram = persistence_diagrams(predicted_distances)[1]
    target_diagram = persistence_diagrams(target_distances)[1]
    result: Dict[str, Any] = {
        "posterior_fisher_rao": _persistence_record(predicted_distances),
        "target_fisher_rao": _persistence_record(target_distances),
        "posterior_h1_bottleneck_to_target": float(
            bottleneck(target_diagram, predicted_diagram)
        ),
        "semantic_map": _map_diagnostics(posteriors, dataset, task, quotient),
        "final_residual_euclidean": _persistence_record(
            representation_distance_matrix(layer_residuals[-1], metric="euclidean")
        ),
        "final_residual_cosine": _persistence_record(
            representation_distance_matrix(layer_residuals[-1], metric="cosine")
        ),
    }
    if full_analysis:
        result["posterior_bootstrap"] = bootstrap_max_h1(
            predicted_distances,
            repeats=campaign.bootstrap_repeats,
            sample_fraction=campaign.bootstrap_fraction,
            seed=bootstrap_seed,
        )
        edge_lengths = _pullback_fisher_edge_lengths(
            model, final_residuals, posterior_tensor, answer_ids
        )
        neighborhood = representation_distance_matrix(
            layer_residuals[-1], metric="euclidean"
        )
        geodesic = knn_geodesic_distance_matrix(
            neighborhood,
            edge_lengths,
            neighbors=min(campaign.fisher_neighbors, len(edge_lengths) - 1),
        )
        result["final_residual_pullback_fisher"] = _persistence_record(geodesic)
    return result


def _nuisance_dataset(
    task: CircleTaskConfig,
    quotient: str,
    *,
    seed: int,
    groups: int = 8,
    replicates: int = 8,
) -> tuple[CircleDataset, np.ndarray]:
    if quotient == "phase_circle":
        target_angles = np.linspace(0.0, 2.0 * math.pi, groups, endpoint=False)
        current_phases = target_angles - task.future_delta
        phases = np.repeat(current_phases, replicates)
        semantic_groups = np.repeat(np.arange(groups), replicates)
    else:
        values = np.linspace(-0.85, 0.85, groups)
        positive = np.arccos(values)
        negative = (2.0 * math.pi - positive) % (2.0 * math.pi)
        target_angles = np.stack((positive, negative), axis=1)
        phases = np.repeat((target_angles - task.future_delta).reshape(-1), replicates)
        semantic_groups = np.repeat(np.repeat(np.arange(groups), 2), replicates)
    base = generate_circle_dataset(
        task,
        sample_count=len(phases),
        seed=seed,
        phases=phases,
        directions=np.ones(len(phases)),
    )
    return with_quotient_targets(base, task, quotient), semantic_groups


@torch.no_grad()
def analyze_nuisance_collapse(
    model: TinyLLMModel,
    dataset: CircleDataset,
    semantic_groups: np.ndarray,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    posteriors, residuals, _, _ = _phase_grid_outputs(model, dataset, answer_ids, device)
    return {
        "posterior_fisher_rao": nuisance_collapse_ratio(
            fisher_rao_distance_matrix(posteriors), semantic_groups
        ),
        "final_residual_euclidean": nuisance_collapse_ratio(
            representation_distance_matrix(residuals[-1], metric="euclidean"),
            semantic_groups,
        ),
        "final_residual_cosine": nuisance_collapse_ratio(
            representation_distance_matrix(residuals[-1], metric="cosine"),
            semantic_groups,
        ),
    }


def run_contrast_arm(
    *,
    preset: str,
    seed: int,
    quotient: str,
    condition: str,
    task: CircleTaskConfig,
    campaign: QuotientContrastConfig,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_config = _tinyllm_config(preset, task, seed)
    model = TinyLLMModel(
        model_config, name=f"QuotientContrast_{preset}_{quotient}_{condition}_{seed}"
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=campaign.learning_rate, weight_decay=campaign.weight_decay
    )
    base_training = generate_circle_dataset(
        task, sample_count=task.train_samples, seed=seed + 1_001
    )
    training = with_quotient_targets(base_training, task, quotient)
    training_inputs, training_targets = _conditioned_training_data(
        training, task, condition, seed=seed
    )
    base_evaluation = generate_circle_dataset(
        task, sample_count=task.evaluation_samples, seed=seed + 2_003
    )
    evaluation = with_quotient_targets(base_evaluation, task, quotient)
    base_shift = generate_circle_dataset(
        task,
        sample_count=task.evaluation_samples,
        seed=seed + 3_007,
        nuisance_shift=True,
    )
    shifted = with_quotient_targets(base_shift, task, quotient)
    phases = np.linspace(0.0, 2.0 * math.pi, task.phase_grid_points, endpoint=False)
    base_grid = generate_circle_dataset(
        task,
        sample_count=task.phase_grid_points,
        seed=seed + 4_009,
        phases=phases,
        directions=np.ones(task.phase_grid_points),
    )
    grid = with_quotient_targets(base_grid, task, quotient)
    nuisance_data, semantic_groups = _nuisance_dataset(
        task, quotient, seed=seed + 5_011
    )
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    batches = torch.randint(
        0,
        len(training_inputs),
        (campaign.training_steps, campaign.batch_size),
        generator=batch_generator,
    )
    checkpoints: List[Dict[str, Any]] = []
    history: List[Dict[str, float]] = []
    checkpoint_steps = set(campaign.checkpoints)
    started = time.perf_counter()

    def record(step: int) -> None:
        full = step == campaign.training_steps
        checkpoints.append(
            {
                "step": step,
                "in_domain": evaluate_task(
                    model, evaluation, answer_ids, device, batch_size=campaign.batch_size
                ),
                "nuisance_shift": evaluate_task(
                    model, shifted, answer_ids, device, batch_size=campaign.batch_size
                ),
                "quotient_analysis": analyze_quotient_grid(
                    model,
                    grid,
                    answer_ids,
                    device,
                    task,
                    campaign,
                    quotient,
                    bootstrap_seed=seed + step,
                    full_analysis=full,
                ),
                **(
                    {
                        "nuisance_collapse": analyze_nuisance_collapse(
                            model,
                            nuisance_data,
                            semantic_groups,
                            answer_ids,
                            device,
                        )
                    }
                    if full
                    else {}
                ),
            }
        )

    record(0)
    model.train()
    for step in range(1, campaign.training_steps + 1):
        indices = batches[step - 1]
        inputs = training_inputs[indices].to(device, non_blocking=True)
        targets = training_targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = _task_logits(model, inputs, answer_ids)
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), campaign.gradient_clip)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step in checkpoint_steps:
            history.append(
                {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm)}
            )
        if step in checkpoint_steps:
            record(step)
            model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started

    checkpoint_path: Optional[Path] = None
    if campaign.save_primary_checkpoints and condition == "trained" and seed == campaign.seeds[0]:
        checkpoint_path = (
            output_dir / "checkpoints" / f"{preset}_{quotient}_trained_seed{seed}.pt"
        )
        model.to("cpu").save_checkpoint(
            checkpoint_path,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "quotient": quotient,
                "condition": condition,
                "seed": seed,
            },
        )
        model.to(device)

    final = checkpoints[-1]
    topology = final["quotient_analysis"]
    semantic_map = topology["semantic_map"]
    primary = {
        "in_domain_exact_bin_accuracy": final["in_domain"]["exact_bin_accuracy"],
        "nuisance_shift_exact_bin_accuracy": final["nuisance_shift"]["exact_bin_accuracy"],
        "posterior_normalized_h1_lifetime": topology["posterior_fisher_rao"]["summary"]
        ["normalized_h1_lifetime"],
        "posterior_h1_bottleneck_to_target": topology[
            "posterior_h1_bottleneck_to_target"
        ],
        "posterior_nuisance_collapse_ratio": final["nuisance_collapse"]
        ["posterior_fisher_rao"]["nuisance_collapse_ratio"],
        "residual_nuisance_collapse_ratio": final["nuisance_collapse"]
        ["final_residual_euclidean"]["nuisance_collapse_ratio"],
        "pullback_fisher_normalized_h1_lifetime": topology[
            "final_residual_pullback_fisher"
        ]["summary"]["normalized_h1_lifetime"],
    }
    if quotient == "phase_circle":
        primary.update(
            {
                "map_alignment": semantic_map["alignment"],
                "map_winding_degree": semantic_map["winding_degree"],
            }
        )
    else:
        primary.update(
            {
                "map_pearson_correlation": semantic_map["pearson_correlation"],
                "map_spearman_correlation": semantic_map["spearman_correlation"],
            }
        )
    return {
        "experiment_id": f"tinyllm-quotient-{preset}-{quotient}-{condition}-seed{seed}",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "preset": preset,
        "seed": seed,
        "quotient": quotient,
        "condition": condition,
        "model_config": asdict(model_config),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "training_history": history,
        "checkpoints": checkpoints,
        "checkpoint_diagram_tracking": _checkpoint_bottleneck_tracking(
            [
                {
                    "step": item["step"],
                    "phase_topology": {
                        "posterior_fisher_rao": item["quotient_analysis"][
                            "posterior_fisher_rao"
                        ]
                    },
                }
                for item in checkpoints
            ]
        ),
        "primary_metrics": primary,
        "final_state_sha256": _state_digest(model),
        "model_checkpoint": (
            str(checkpoint_path.relative_to(output_dir)) if checkpoint_path else None
        ),
    }


def _metric_summary(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set.intersection(*(set(run["primary_metrics"]) for run in runs)))
    return {
        key: {
            "mean": statistics.fmean(float(run["primary_metrics"][key]) for run in runs),
            "values": [float(run["primary_metrics"][key]) for run in runs],
        }
        for key in keys
    }


def aggregate_contrast(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregates: Dict[str, Any] = {}
    for preset in sorted({run["preset"] for run in runs}):
        cells = {}
        for quotient in QUOTIENTS:
            for condition in CONDITIONS:
                selected = [
                    run
                    for run in runs
                    if run["preset"] == preset
                    and run["quotient"] == quotient
                    and run["condition"] == condition
                ]
                if selected:
                    cells[f"{quotient}:{condition}"] = _metric_summary(selected)
        circle = cells.get("phase_circle:trained")
        interval = cells.get("cosine_interval:trained")
        circle_null = cells.get("phase_circle:label_shuffled")
        interval_null = cells.get("cosine_interval:label_shuffled")
        criteria = {
            "circle_alignment_at_least_0_9": (
                circle["map_alignment"]["mean"] >= 0.9 if circle else None
            ),
            "circle_absolute_degree_near_one": (
                abs(abs(circle["map_winding_degree"]["mean"]) - 1.0) <= 0.1
                if circle
                else None
            ),
            "interval_correlation_at_least_0_9": (
                interval["map_pearson_correlation"]["mean"] >= 0.9 if interval else None
            ),
            "interval_normalized_h1_at_most_0_2": (
                interval["posterior_normalized_h1_lifetime"]["mean"] <= 0.2
                if interval
                else None
            ),
            "circle_h1_exceeds_interval_by_0_4": (
                circle["posterior_normalized_h1_lifetime"]["mean"]
                >= interval["posterior_normalized_h1_lifetime"]["mean"] + 0.4
                if circle and interval
                else None
            ),
            "both_trained_tasks_reach_60_percent_accuracy": (
                min(
                    circle["in_domain_exact_bin_accuracy"]["mean"],
                    interval["in_domain_exact_bin_accuracy"]["mean"],
                )
                >= 0.6
                if circle and interval
                else None
            ),
            "trained_tasks_beat_label_shuffle": (
                circle["in_domain_exact_bin_accuracy"]["mean"]
                > circle_null["in_domain_exact_bin_accuracy"]["mean"] + 0.3
                and interval["in_domain_exact_bin_accuracy"]["mean"]
                > interval_null["in_domain_exact_bin_accuracy"]["mean"] + 0.3
                if circle and interval and circle_null and interval_null
                else None
            ),
        }
        aggregates[preset] = {
            "cells": cells,
            "pre_registered_criteria": criteria,
            "criteria_passed": all(value is True for value in criteria.values()),
        }
    return aggregates


def target_class_baselines(
    task: CircleTaskConfig,
    campaign: QuotientContrastConfig,
) -> Dict[str, Any]:
    """Measure exact-bin null baselines on each deterministic evaluation set."""
    result: Dict[str, Any] = {}
    for quotient in campaign.quotients:
        rows = []
        for seed in campaign.seeds:
            base = generate_circle_dataset(
                task,
                sample_count=task.evaluation_samples,
                seed=seed + 2_003,
            )
            dataset = with_quotient_targets(base, task, quotient)
            counts = np.bincount(
                dataset.target_bins.numpy(), minlength=task.phase_bins
            )
            probabilities = counts / counts.sum()
            rows.append(
                {
                    "seed": seed,
                    "counts": counts.tolist(),
                    "uniform_random_accuracy": 1.0 / task.phase_bins,
                    "empirical_majority_class_accuracy": float(probabilities.max()),
                    "empirical_prior_random_accuracy": float(
                        np.square(probabilities).sum()
                    ),
                }
            )
        result[quotient] = {
            "per_seed": rows,
            "mean_uniform_random_accuracy": statistics.fmean(
                row["uniform_random_accuracy"] for row in rows
            ),
            "mean_empirical_majority_class_accuracy": statistics.fmean(
                row["empirical_majority_class_accuracy"] for row in rows
            ),
            "mean_empirical_prior_random_accuracy": statistics.fmean(
                row["empirical_prior_random_accuracy"] for row in rows
            ),
        }
    return result


def run_quotient_contrast(
    task: CircleTaskConfig,
    campaign: QuotientContrastConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(campaign.device)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": "tinyllm_task_quotient_contrast",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "running",
        "claim_scope": "synthetic_matched_s1_vs_interval_task_quotient",
        "started_at": _utc_now(),
        "task_config": asdict(task),
        "campaign_config": asdict(campaign),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ripser": ripser_package.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "pre_registered_interpretation": (
            "Topology follows the task quotient only if matched phase models induce degree "
            "plus-or-minus one with high phase alignment while matched cosine models form "
            "an interval with high scalar correlation and no material H1."
        ),
        "runs": [],
        "target_class_baselines": target_class_baselines(task, campaign),
        "deferred": [
            "persistent-cohomology-derived circular coordinates (the task decoder supplies the tested map)",
            "true zigzag persistence",
            "Whitney rank/jet localization",
            "real sensor and tokenizer-derived prompt transfer",
        ],
    }
    partial_path = output_dir / "results.partial.json"
    if partial_path.exists():
        previous = json.loads(partial_path.read_text(encoding="utf-8"))
        if (
            previous.get("schema_version") != SCHEMA_VERSION
            or previous.get("task_config") != json.loads(json.dumps(asdict(task)))
            or previous.get("campaign_config") != json.loads(json.dumps(asdict(campaign)))
        ):
            raise ValueError("partial result configuration differs; choose another output")
        bundle["started_at"] = previous["started_at"]
        bundle["runs"] = previous.get("runs", [])
        if previous.get("completed_at"):
            bundle["completed_at"] = previous["completed_at"]
    _write_json(partial_path, bundle)
    try:
        completed = {run["experiment_id"] for run in bundle["runs"]}
        for preset in campaign.presets:
            for seed in campaign.seeds:
                for quotient in campaign.quotients:
                    for condition in campaign.conditions:
                        experiment_id = (
                            f"tinyllm-quotient-{preset}-{quotient}-{condition}-seed{seed}"
                        )
                        if experiment_id in completed:
                            print(f"resuming: skipped {experiment_id}", flush=True)
                            continue
                        run = run_contrast_arm(
                            preset=preset,
                            seed=seed,
                            quotient=quotient,
                            condition=condition,
                            task=task,
                            campaign=campaign,
                            device=device,
                            output_dir=output_dir,
                        )
                        bundle["runs"].append(run)
                        completed.add(run["experiment_id"])
                        _write_json(partial_path, bundle)
                        print(experiment_id, json.dumps(run["primary_metrics"], sort_keys=True), flush=True)
        bundle["aggregates"] = aggregate_contrast(bundle["runs"])
        bundle["status"] = "completed"
        bundle.setdefault("completed_at", _utc_now())
        confirmed = all(value["criteria_passed"] for value in bundle["aggregates"].values())
        bundle["claim_status"] = {
            "confirmed": confirmed,
            "interpretation": (
                "The learned output topology followed the supervised task quotient."
                if confirmed
                else "At least one map-aware quotient criterion failed; inspect the matched cells."
            ),
            "does_not_establish": bundle["deferred"],
        }
        _write_json(output_dir / "results.json", bundle)
        _write_json(partial_path, bundle)
        return bundle
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.set_num_threads(previous_threads)


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _strings(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="d6,d8")
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--quotients", default=",".join(QUOTIENTS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--checkpoints", default="0,100,300,600")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8"
        ),
    )
    parser.add_argument("--no-save-checkpoints", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = QuotientContrastConfig(
        presets=_strings(args.presets),
        seeds=_ints(args.seeds),
        quotients=_strings(args.quotients),
        conditions=_strings(args.conditions),
        training_steps=args.steps,
        checkpoints=_ints(args.checkpoints),
        batch_size=args.batch_size,
        bootstrap_repeats=args.bootstrap_repeats,
        device=args.device,
        save_primary_checkpoints=not args.no_save_checkpoints,
    )
    bundle = run_quotient_contrast(CircleTaskConfig(), campaign, args.output)
    print(json.dumps(bundle["claim_status"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
