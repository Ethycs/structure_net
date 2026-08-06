#!/usr/bin/env python3
"""Train and measure a depth-graded TinyLLM task-quotient family.

The campaign compares ordinary final-depth training, integer multi-exit
training, and continuous residual-gate training. One shared LM head reads every
depth. The phase arm additionally computes indexed posterior-moment defect
charge along the continuous depth axis at each retained training checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

try:
    from experiments.structure_net.tinyllm_degree_defect_cobordism import (
        fixed_nuisance_sensor_values,
        moment_record,
        soft_sensor_token_embeddings,
    )
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleDataset,
        CircleTaskConfig,
        _conditioned_training_data,
        _persistence_record,
        _resolve_device,
        _state_digest,
        _tinyllm_config,
        generate_circle_dataset,
    )
    from experiments.structure_net.tinyllm_task_quotient_contrast import (
        QUOTIENTS,
        with_quotient_targets,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_degree_defect_cobordism import (  # type: ignore[no-redef]
        fixed_nuisance_sensor_values,
        moment_record,
        soft_sensor_token_embeddings,
    )
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleDataset,
        CircleTaskConfig,
        _conditioned_training_data,
        _persistence_record,
        _resolve_device,
        _state_digest,
        _tinyllm_config,
        generate_circle_dataset,
    )
    from tinyllm_task_quotient_contrast import (  # type: ignore[no-redef]
        QUOTIENTS,
        with_quotient_targets,
    )
from structure_net.components.analyzers import (
    complex_defect_charge,
    fisher_rao_distance_matrix,
    paired_geometry_alignment,
    representation_distance_matrix,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-depth-graded-quotient.v1"
HYPOTHESIS_ID = "tinyllm-depth-graded-task-fronts-v1"
TRAINING_ARMS = ("standard_final", "discrete_multi_exit", "continuous_gate")


@dataclass(frozen=True)
class DepthGradedCampaignConfig:
    preset: str = "d8"
    seed: int = 7
    quotients: tuple[str, ...] = QUOTIENTS
    training_arms: tuple[str, ...] = TRAINING_ARMS
    training_steps: int = 600
    checkpoints: tuple[int, ...] = (0, 25, 50, 100, 200, 400, 600)
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    sampled_depths_per_batch: int = 4
    depth_step: float = 0.25
    phase_points: int = 128
    max_phase_points: int = 1_024
    max_row_phase_points: int = 8_192
    device: str = "cuda:auto"
    save_checkpoints: bool = True

    def __post_init__(self) -> None:
        if set(self.quotients).difference(QUOTIENTS):
            raise ValueError(f"quotients must be drawn from {QUOTIENTS}")
        if set(self.training_arms).difference(TRAINING_ARMS):
            raise ValueError(f"training arms must be drawn from {TRAINING_ARMS}")
        if not self.quotients or not self.training_arms:
            raise ValueError("campaign axes cannot be empty")
        if self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("training sizes must be positive")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted and unique")
        if self.checkpoints[0] != 0 or self.checkpoints[-1] != self.training_steps:
            raise ValueError("checkpoints must contain zero and the final step")
        if self.sampled_depths_per_batch < 3:
            raise ValueError("include full, shallow, and at least one sampled depth")
        if not 0.0 < self.depth_step <= 1.0:
            raise ValueError("depth_step must be in (0, 1]")
        if self.phase_points < 32 or self.max_phase_points < self.phase_points:
            raise ValueError("invalid phase resolution")
        if self.max_row_phase_points < self.max_phase_points:
            raise ValueError("row resolution cannot be lower than the common grid")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def depth_grid(layer_count: int, step: float) -> np.ndarray:
    intervals = int(round(layer_count / step))
    values = np.linspace(0.0, float(layer_count), intervals + 1)
    if not np.allclose(np.diff(values), step):
        raise ValueError("depth_step must divide the model layer count")
    return values


def training_depth_schedule(
    arm: str,
    *,
    layer_count: int,
    steps: int,
    depths_per_batch: int,
    seed: int,
) -> np.ndarray:
    """Precompute a deterministic depth schedule shared across task quotients."""
    if arm not in TRAINING_ARMS:
        raise ValueError(f"unknown training arm: {arm}")
    if arm == "standard_final":
        return np.full((steps, 1), float(layer_count), dtype=np.float64)
    schedule = np.empty((steps, depths_per_batch), dtype=np.float64)
    schedule[:, 0] = float(layer_count)
    schedule[:, 1] = 1.0
    generator = np.random.default_rng(seed)
    if arm == "discrete_multi_exit":
        schedule[:, 2:] = generator.integers(
            2, layer_count, size=(steps, depths_per_batch - 2)
        )
    else:
        schedule[:, 2:] = generator.uniform(
            1.0, float(layer_count), size=(steps, depths_per_batch - 2)
        )
    return schedule


def _answer_logits_at_depth(
    model: TinyLLMModel,
    input_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    depth: float,
) -> torch.Tensor:
    logits = model.forward_at_depth(input_ids, depth)
    return logits[:, -1, :].index_select(-1, answer_ids)


@torch.no_grad()
def soft_depth_outputs(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    future_phases: np.ndarray,
    depth: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate posterior and query residual on the continuous input/depth lift."""
    if len(future_phases) > 512:
        chunks = [
            soft_depth_outputs(
                model, task, future_phases[start : start + 512], depth, device
            )
            for start in range(0, len(future_phases), 512)
        ]
        return (
            np.concatenate([item[0] for item in chunks]),
            np.concatenate([item[1] for item in chunks]),
        )
    model.eval()
    sensor_values = fixed_nuisance_sensor_values(task, future_phases)
    sensor_embeddings = soft_sensor_token_embeddings(
        model, task, sensor_values, device
    )
    sample_count = len(future_phases)
    embedded = torch.empty(
        (sample_count, task.sequence_length, model.config.n_embd),
        device=device,
        dtype=sensor_embeddings.dtype,
    )
    embedded[:, 0, :] = model.transformer["wte"](
        torch.tensor(1, device=device)
    )
    embedded[:, -1, :] = model.transformer["wte"](
        torch.tensor(2, device=device)
    )
    embedded[:, 1:-1, :] = sensor_embeddings.reshape(
        sample_count, -1, model.config.n_embd
    )
    positions = torch.arange(task.sequence_length, device=device)
    embedded = embedded + model.transformer["wpe"](positions)
    residual = model._run_blocks_to_depth(embedded, depth)
    normalized = model.transformer["ln_f"](residual)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    logits = model.lm_head(normalized[:, -1, :]).index_select(-1, answer_ids)
    probabilities = torch.softmax(logits, dim=-1)
    return (
        probabilities.cpu().double().numpy(),
        residual[:, -1, :].cpu().double().numpy(),
    )


def _phase_posterior_moment(
    posteriors: np.ndarray, task: CircleTaskConfig
) -> np.ndarray:
    angles = 2.0 * math.pi * np.arange(task.phase_bins) / task.phase_bins
    return posteriors @ np.exp(1j * angles)


def resolve_circle_depth_record(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    depth: float,
    device: torch.device,
    *,
    initial_phase_points: int,
    maximum_phase_points: int,
) -> Dict[str, Any]:
    """Adaptively resolve one circular posterior map at a fixed real depth."""
    phase_points = initial_phase_points
    refinements = 0
    while True:
        phases = (
            np.arange(phase_points, dtype=np.float64) + 0.5
        ) * (2.0 * math.pi / phase_points)
        posterior, _ = soft_depth_outputs(model, task, phases, depth, device)
        record = moment_record(_phase_posterior_moment(posterior, task), phases)
        record["depth"] = float(depth)
        record["phase_grid_points"] = phase_points
        record["refinement_levels"] = refinements
        if record["sampling_resolved"] or phase_points >= maximum_phase_points:
            return record
        phase_points *= 2
        refinements += 1


def _paired_branch_geometry(
    residuals: np.ndarray, future_phases: np.ndarray
) -> Dict[str, Any]:
    distances = representation_distance_matrix(residuals, metric="euclidean")
    point_count = len(future_phases)
    partners = np.arange(point_count - 1, -1, -1)
    within = distances[np.arange(point_count), partners]
    diagonal = np.eye(point_count, dtype=bool)
    paired = np.zeros_like(diagonal)
    paired[np.arange(point_count), partners] = True
    between = distances[~diagonal & ~paired]
    within_mean = float(within.mean())
    between_mean = float(between.mean())
    spread = float(distances[np.triu_indices(point_count, k=1)].mean())
    cosine_reference = np.abs(
        np.cos(future_phases)[:, None] - np.cos(future_phases)[None, :]
    )
    paired_geometry = paired_geometry_alignment(
        cosine_reference, distances, neighbors=min(8, point_count - 1)
    )
    return {
        "paired_branch_mean_distance": within_mean,
        "between_cosine_mean_distance": between_mean,
        "paired_branch_distance_ratio": (
            within_mean / between_mean if between_mean > 1e-12 else 0.0
        ),
        "mean_pairwise_spread": spread,
        "degenerate_representation": spread <= 1e-12,
        "cosine_reference_distance_spearman": paired_geometry[
            "distance_spearman"
        ],
        "cosine_reference_normalized_stress": paired_geometry[
            "normalized_stress"
        ],
    }


def _cosine_map_record(
    posteriors: np.ndarray,
    phases: np.ndarray,
    task: CircleTaskConfig,
) -> Dict[str, Any]:
    centers = np.linspace(-1.0, 1.0, task.phase_bins)
    predicted = posteriors @ centers
    target = np.cos(phases)
    pearson = (
        float(pearsonr(predicted, target).statistic)
        if np.std(predicted) > 1e-12
        else 0.0
    )
    spearman = (
        float(spearmanr(predicted, target).statistic)
        if np.std(predicted) > 1e-12
        else 0.0
    )
    return {
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "root_mean_squared_error": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "predicted_range": float(predicted.max() - predicted.min()),
    }


def _circle_front(cells: Sequence[Mapping[str, Any]]) -> Optional[float]:
    for cell in cells:
        circle = cell["circle_map"]
        if (
            circle["phase_alignment"] >= 0.9
            and circle["rounded_degree"] == 1
            and circle["minimum_moment_magnitude"] >= 0.05
            and circle["sampling_resolved"]
        ):
            return float(cell["depth"])
    return None


def _cosine_front(cells: Sequence[Mapping[str, Any]]) -> Optional[float]:
    for cell in cells:
        map_record = cell["cosine_map"]
        branch = cell["branch_geometry"]
        if (
            map_record["pearson_correlation"] >= 0.9
            and branch["paired_branch_distance_ratio"] <= 0.5
            and branch["cosine_reference_distance_spearman"] >= 0.6
            and not branch["degenerate_representation"]
        ):
            return float(cell["depth"])
    return None


def analyze_depth_diagram(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    depths: np.ndarray,
    quotient: str,
    campaign: DepthGradedCampaignConfig,
    device: torch.device,
    *,
    final_checkpoint: bool,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    phase_points = campaign.phase_points
    while True:
        phases = (
            np.arange(phase_points, dtype=np.float64) + 0.5
        ) * (2.0 * math.pi / phase_points)
        posteriors = []
        residuals = []
        for depth in depths:
            posterior, residual = soft_depth_outputs(
                model, task, phases, float(depth), device
            )
            posteriors.append(posterior)
            residuals.append(residual)
        posterior_field = np.stack(posteriors)
        residual_field = np.stack(residuals)
        if quotient != "phase_circle":
            break
        moment_field = np.stack(
            [_phase_posterior_moment(item, task) for item in posterior_field]
        )
        resolved = all(
            moment_record(row, phases)["sampling_resolved"]
            for row in moment_field
        )
        if resolved or phase_points >= campaign.max_phase_points:
            break
        phase_points *= 2

    cells: List[Dict[str, Any]] = []
    integer_depths = {
        float(value) for value in range(model.config.n_layer + 1)
    }
    for index, depth in enumerate(depths):
        cell: Dict[str, Any] = {"depth": float(depth)}
        if quotient == "phase_circle":
            circle = moment_record(moment_field[index], phases)
            circle["phase_grid_points"] = phase_points
            circle["refinement_levels"] = 0
            if not circle["sampling_resolved"]:
                circle = resolve_circle_depth_record(
                    model,
                    task,
                    float(depth),
                    device,
                    initial_phase_points=phase_points * 2,
                    maximum_phase_points=campaign.max_row_phase_points,
                )
            cell["circle_map"] = circle
        else:
            cell["cosine_map"] = _cosine_map_record(
                posterior_field[index], phases, task
            )
            cell["branch_geometry"] = _paired_branch_geometry(
                residual_field[index], phases
            )
        if final_checkpoint and float(depth) in integer_depths:
            persistence = _persistence_record(
                fisher_rao_distance_matrix(posterior_field[index])
            )
            cell["posterior_normalized_h1_lifetime"] = persistence["summary"][
                "normalized_h1_lifetime"
            ]
        cells.append(cell)

    result: Dict[str, Any] = {
        "phase_grid_points": phase_points,
        "depth_cells": cells,
        "front_depth": (
            _circle_front(cells)
            if quotient == "phase_circle"
            else _cosine_front(cells)
        ),
    }
    arrays: Dict[str, np.ndarray] = {
        "phases": phases,
        "depths": depths,
        "posteriors": posterior_field,
    }
    if quotient == "phase_circle":
        charge = complex_defect_charge(moment_field, phases, depths)
        charge["common_grid_phase_points"] = phase_points
        charge["common_grid_all_rows_resolved"] = resolved
        result["depth_defect_charge"] = charge
        result["all_depth_rows_resolved"] = all(
            cell["circle_map"]["sampling_resolved"] for cell in cells
        )
        arrays["moment_real"] = moment_field.real
        arrays["moment_imag"] = moment_field.imag
    return result, arrays


@torch.no_grad()
def evaluate_hard_depths(
    model: TinyLLMModel,
    dataset: CircleDataset,
    answer_ids: torch.Tensor,
    depths: Sequence[float],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    result = {}
    for depth in depths:
        predictions = []
        losses = []
        for start in range(0, len(dataset.input_ids), batch_size):
            inputs = dataset.input_ids[start : start + batch_size].to(device)
            targets = dataset.target_posteriors[start : start + batch_size].to(device)
            logits = _answer_logits_at_depth(model, inputs, answer_ids, float(depth))
            predictions.append(logits.argmax(-1).cpu())
            losses.append(
                (-(targets * torch.log_softmax(logits, dim=-1)).sum(-1)).cpu()
            )
        predicted = torch.cat(predictions)
        result[f"{float(depth):g}"] = {
            "exact_bin_accuracy": float(
                (predicted == dataset.target_bins).float().mean()
            ),
            "negative_log_likelihood": float(torch.cat(losses).mean()),
        }
    return result


def _source_hashes(
    source_results: Path, preset: str, seed: int
) -> Dict[str, str]:
    source = json.loads(source_results.read_text(encoding="utf-8"))
    return {
        run["quotient"]: run["final_state_sha256"]
        for run in source["runs"]
        if run["preset"] == preset
        and run["condition"] == "trained"
        and int(run["seed"]) == seed
    }


def run_arm(
    *,
    arm: str,
    quotient: str,
    task: CircleTaskConfig,
    campaign: DepthGradedCampaignConfig,
    device: torch.device,
    output_dir: Path,
    expected_standard_hash: str,
) -> Dict[str, Any]:
    seed = campaign.seed
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_config = _tinyllm_config(campaign.preset, task, seed)
    model = TinyLLMModel(
        model_config, name=f"DepthGraded_{arm}_{quotient}_{seed}"
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
    )
    base_training = generate_circle_dataset(
        task, sample_count=task.train_samples, seed=seed + 1_001
    )
    training = with_quotient_targets(base_training, task, quotient)
    training_inputs, training_targets = _conditioned_training_data(
        training, task, "trained", seed=seed
    )
    base_evaluation = generate_circle_dataset(
        task, sample_count=task.evaluation_samples, seed=seed + 2_003
    )
    evaluation = with_quotient_targets(base_evaluation, task, quotient)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    batches = torch.randint(
        0,
        len(training_inputs),
        (campaign.training_steps, campaign.batch_size),
        generator=batch_generator,
    )
    schedule = training_depth_schedule(
        arm,
        layer_count=model.config.n_layer,
        steps=campaign.training_steps,
        depths_per_batch=campaign.sampled_depths_per_batch,
        seed=seed + 8_019,
    )
    depths = depth_grid(model.config.n_layer, campaign.depth_step)
    checkpoint_steps = set(campaign.checkpoints)
    checkpoints: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    field_arrays: Dict[str, np.ndarray] = {}
    started = time.perf_counter()

    def record(step: int) -> None:
        diagram, arrays = analyze_depth_diagram(
            model,
            task,
            depths,
            quotient,
            campaign,
            device,
            final_checkpoint=step == campaign.training_steps,
        )
        hard_depths = (
            list(range(model.config.n_layer + 1))
            if step == campaign.training_steps
            else [1.0, float(model.config.n_layer)]
        )
        diagram["step"] = step
        diagram["hard_token_metrics"] = evaluate_hard_depths(
            model,
            evaluation,
            answer_ids,
            hard_depths,
            device,
            campaign.batch_size,
        )
        checkpoints.append(diagram)
        for name, value in arrays.items():
            field_arrays[f"step_{step}_{name}"] = value

    record(0)
    for step in range(1, campaign.training_steps + 1):
        model.train()
        indices = batches[step - 1]
        inputs = training_inputs[indices].to(device, non_blocking=True)
        targets = training_targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        depth_losses = []
        sampled_depths = schedule[step - 1]
        scale = 1.0 / len(sampled_depths)
        for depth in sampled_depths:
            logits = _answer_logits_at_depth(
                model, inputs, answer_ids, float(depth)
            )
            loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            (scale * loss).backward()
            depth_losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), campaign.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step in checkpoint_steps:
            history.append(
                {
                    "step": step,
                    "mean_loss": float(np.mean(depth_losses)),
                    "depth_losses": depth_losses,
                    "sampled_depths": sampled_depths.tolist(),
                    "gradient_norm": float(grad_norm),
                }
            )
        if step in checkpoint_steps:
            record(step)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    digest = _state_digest(model)

    field_path = output_dir / "fields" / f"{arm}_{quotient}_seed{seed}.npz"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(field_path, **field_arrays)
    checkpoint_path: Optional[Path] = None
    if campaign.save_checkpoints:
        checkpoint_path = (
            output_dir / "checkpoints" / f"{arm}_{quotient}_seed{seed}.pt"
        )
        model.to("cpu").save_checkpoint(
            checkpoint_path,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "training_arm": arm,
                "quotient": quotient,
                "seed": seed,
                "depth_family": "partial residual attention/MLP gate",
            },
        )
        model.to(device)
    final = checkpoints[-1]
    full_key = f"{float(model.config.n_layer):g}"
    shallow_key = "1"
    result = {
        "experiment_id": (
            f"tinyllm-depth-graded-{campaign.preset}-{arm}-{quotient}-seed{seed}"
        ),
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "preset": campaign.preset,
        "seed": seed,
        "training_arm": arm,
        "quotient": quotient,
        "model_config": asdict(model_config),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": elapsed,
        "training_depth_schedule_sha256": __import__("hashlib").sha256(
            schedule.tobytes()
        ).hexdigest(),
        "training_history": history,
        "checkpoints": checkpoints,
        "front_trajectory": [
            {"step": item["step"], "front_depth": item["front_depth"]}
            for item in checkpoints
        ],
        "final_full_depth_accuracy": final["hard_token_metrics"][full_key][
            "exact_bin_accuracy"
        ],
        "final_shallow_depth_accuracy": final["hard_token_metrics"][shallow_key][
            "exact_bin_accuracy"
        ],
        "final_state_sha256": digest,
        "expected_standard_state_sha256": (
            expected_standard_hash if arm == "standard_final" else None
        ),
        "standard_state_matches_source_campaign": (
            digest == expected_standard_hash if arm == "standard_final" else None
        ),
        "field_artifact": str(field_path.relative_to(output_dir)),
        "model_checkpoint": (
            str(checkpoint_path.relative_to(output_dir)) if checkpoint_path else None
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_campaign(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_cell = {
        (run["training_arm"], run["quotient"]): run for run in runs
    }
    comparisons: Dict[str, Any] = {}
    for quotient in QUOTIENTS:
        if all((arm, quotient) in by_cell for arm in TRAINING_ARMS):
            standard = by_cell[("standard_final", quotient)]
            comparisons[quotient] = {
                arm: {
                    "final_front_depth": by_cell[(arm, quotient)][
                        "front_trajectory"
                    ][-1]["front_depth"],
                    "final_shallow_depth_accuracy": by_cell[(arm, quotient)][
                        "final_shallow_depth_accuracy"
                    ],
                    "final_full_depth_accuracy": by_cell[(arm, quotient)][
                        "final_full_depth_accuracy"
                    ],
                    "shallow_accuracy_gain_over_standard": (
                        by_cell[(arm, quotient)]["final_shallow_depth_accuracy"]
                        - standard["final_shallow_depth_accuracy"]
                    ),
                    "full_accuracy_change_from_standard": (
                        by_cell[(arm, quotient)]["final_full_depth_accuracy"]
                        - standard["final_full_depth_accuracy"]
                    ),
                }
                for arm in TRAINING_ARMS
            }
    phase_charge_checks = {
        run["training_arm"]: [
            {
                "step": checkpoint["step"],
                "identity_holds": checkpoint["depth_defect_charge"][
                    "charge_identity_holds"
                ],
                "endpoint_degree_change": checkpoint["depth_defect_charge"][
                    "endpoint_degree_change"
                ],
                "defect_charge": checkpoint["depth_defect_charge"]["defect_charge"],
                "defect_cell_count": checkpoint["depth_defect_charge"][
                    "defect_cell_count"
                ],
            }
            for checkpoint in run["checkpoints"]
        ]
        for run in runs
        if run["quotient"] == "phase_circle"
    }
    return {
        "matched_arm_comparisons": comparisons,
        "phase_depth_charge_checks": phase_charge_checks,
    }


def run_campaign(
    task: CircleTaskConfig,
    campaign: DepthGradedCampaignConfig,
    source_results: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(campaign.device)
    torch.use_deterministic_algorithms(True)
    source_hashes = _source_hashes(
        source_results, campaign.preset, campaign.seed
    )
    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "running",
        "started_at": _utc_now(),
        "task_config": asdict(task),
        "campaign_config": asdict(campaign),
        "source_results": str(source_results),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
        "pre_registered_prediction": (
            "Integer multi-exit and continuous-gate supervision move the learned "
            "task front toward shallower depth relative to ordinary final-depth "
            "training, while the continuous depth family obeys the numerical "
            "degree-change/indexed-defect-charge identity."
        ),
        "method_boundaries": [
            "one d8 seed and one fixed nuisance slice are tested",
            "continuous depth is a residual gate path, not a neural ODE limit",
            "branch distance and paired geometry are proxies for conditional mutual information and a Reeb cosheaf",
            "posterior H1 is computed only at final integer depths",
            "independent persistent cohomology coordinates are not computed in this campaign",
            "training checkpoints do not define continuous optimization time without weight interpolation",
            "depth defect cells are finite-grid indices, not interval-isolated roots",
            "linked defect curves and Whitney strata are not constructed",
        ],
        "runs": [],
    }
    partial = output_dir / "results.partial.json"
    _write_json(partial, bundle)
    for arm in campaign.training_arms:
        for quotient in campaign.quotients:
            run = run_arm(
                arm=arm,
                quotient=quotient,
                task=task,
                campaign=campaign,
                device=device,
                output_dir=output_dir,
                expected_standard_hash=source_hashes[quotient],
            )
            bundle["runs"].append(run)
            _write_json(partial, bundle)
            print(
                run["experiment_id"],
                json.dumps(
                    {
                        "front": run["front_trajectory"][-1]["front_depth"],
                        "shallow_accuracy": run["final_shallow_depth_accuracy"],
                        "full_accuracy": run["final_full_depth_accuracy"],
                        "source_hash_match": run[
                            "standard_state_matches_source_campaign"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    bundle["aggregates"] = aggregate_campaign(bundle["runs"])
    criteria = {
        "all_standard_hashes_match": all(
            run["standard_state_matches_source_campaign"] is True
            for run in bundle["runs"]
            if run["training_arm"] == "standard_final"
        ),
        "all_phase_depth_charge_identities_hold": all(
            checkpoint["depth_defect_charge"]["charge_identity_holds"]
            for run in bundle["runs"]
            if run["quotient"] == "phase_circle"
            for checkpoint in run["checkpoints"]
        ),
        "all_phase_depth_rows_resolved": all(
            checkpoint["all_depth_rows_resolved"]
            for run in bundle["runs"]
            if run["quotient"] == "phase_circle"
            for checkpoint in run["checkpoints"]
        ),
        "all_final_phase_maps_learned": all(
            run["checkpoints"][-1]["depth_cells"][-1]["circle_map"][
                "phase_alignment"
            ]
            >= 0.95
            and run["checkpoints"][-1]["depth_cells"][-1]["circle_map"][
                "rounded_degree"
            ]
            == 1
            for run in bundle["runs"]
            if run["quotient"] == "phase_circle"
        ),
        "all_final_cosine_maps_learned": all(
            run["checkpoints"][-1]["depth_cells"][-1]["cosine_map"][
                "pearson_correlation"
            ]
            >= 0.95
            for run in bundle["runs"]
            if run["quotient"] == "cosine_interval"
        ),
    }
    bundle["pre_registered_criteria"] = criteria
    bundle["status"] = "completed"
    bundle["completed_at"] = _utc_now()
    bundle["claim_status"] = {
        "confirmed": False,
        "numerical_depth_family_supported": all(criteria.values()),
        "interpretation": (
            "Every numerical depth-family and provenance gate passed."
            if all(criteria.values())
            else "At least one numerical depth-family or provenance gate failed."
        ),
        "confirmation_limit": (
            "A single-seed gated mapping-telescope proxy is not a neural-ODE "
            "continuum, Whitney stratification, Reeb cosheaf, or multi-seed result."
        ),
    }
    _write_json(output_dir / "results.json", bundle)
    _write_json(partial, bundle)
    return bundle


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="d8")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quotients", default=",".join(QUOTIENTS))
    parser.add_argument("--training-arms", default=",".join(TRAINING_ARMS))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--checkpoints", default="0,25,50,100,200,400,600")
    parser.add_argument("--depth-step", type=float, default=0.25)
    parser.add_argument("--phase-points", type=int, default=128)
    parser.add_argument("--max-phase-points", type=int, default=1024)
    parser.add_argument("--max-row-phase-points", type=int, default=8192)
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_task_quotient_contrast/"
            "20260805_d6_d8/results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_depth_graded_quotient/"
            "20260805_d8_seed7"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = DepthGradedCampaignConfig(
        preset=args.preset,
        seed=args.seed,
        quotients=_comma_strings(args.quotients),
        training_arms=_comma_strings(args.training_arms),
        training_steps=args.steps,
        checkpoints=_comma_ints(args.checkpoints),
        depth_step=args.depth_step,
        phase_points=args.phase_points,
        max_phase_points=args.max_phase_points,
        max_row_phase_points=args.max_row_phase_points,
        device=args.device,
        save_checkpoints=not args.no_save_checkpoints,
    )
    result = run_campaign(
        CircleTaskConfig(), campaign, args.source_results, args.output
    )
    print(json.dumps(result["claim_status"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
