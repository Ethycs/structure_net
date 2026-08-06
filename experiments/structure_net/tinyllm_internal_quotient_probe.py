#!/usr/bin/env python3
"""Probe whether TinyLLM's hidden stream forms the supervised cosine quotient.

The source TinyLLM checkpoints are frozen. Nonlinear held-out probes are fit at
the embedding output and every transformer block on exact cosine-matched phase
pairs. A separate persistent-cohomology analysis derives circular coordinates
from hidden distances without consulting phase labels during coordinate fitting.
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
from scipy.stats import pearsonr
import torch
from torch import nn
import torch.nn.functional as F

try:
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleDataset,
        CircleTaskConfig,
        _resolve_device,
        _serialize_sensor_values,
        generate_circle_dataset,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleDataset,
        CircleTaskConfig,
        _resolve_device,
        _serialize_sensor_values,
        generate_circle_dataset,
    )
from structure_net.components.analyzers import (
    circular_phase_alignment,
    persistent_cohomology_circle_coordinate,
    representation_distance_matrix,
    summarize_persistence,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-internal-quotient-probe.v1"
HYPOTHESIS_ID = "tinyllm-internal-cosine-quotient-erases-branch-v1"


@dataclass(frozen=True)
class ProbeCampaignConfig:
    train_samples: int = 4_096
    validation_samples: int = 1_024
    test_samples: int = 2_048
    activation_batch_size: int = 256
    probe_batch_size: int = 256
    probe_steps: int = 300
    probe_width: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_interval: int = 20
    early_stopping_patience: int = 5
    phase_grid_points: int = 64
    device: str = "cuda:auto"
    seed: int = 41

    def __post_init__(self) -> None:
        sample_fields = (
            self.train_samples,
            self.validation_samples,
            self.test_samples,
        )
        if any(value < 4 or value % 2 for value in sample_fields):
            raise ValueError("probe sample counts must be even and at least four")
        if self.probe_steps < 1 or self.probe_width < 2:
            raise ValueError("probe steps and width must be positive")
        if self.validation_interval < 1 or self.early_stopping_patience < 1:
            raise ValueError("validation cadence and patience must be positive")


@dataclass(frozen=True)
class FiberDataset:
    input_ids: torch.Tensor
    cosine: torch.Tensor
    branch: torch.Tensor
    phase: torch.Tensor
    fiber_id: torch.Tensor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _disjoint_nuisance_inputs(
    task: CircleTaskConfig,
    phases: np.ndarray,
    directions: np.ndarray,
    *,
    seed: int,
) -> torch.Tensor:
    """Serialize a nuisance family absent from the source training generator."""
    generator = np.random.default_rng(seed)
    count = len(phases)
    history = np.arange(task.sensor_steps, dtype=np.float64) - (task.sensor_steps - 1)
    speed = generator.uniform(0.25, 0.45, (count, 1))
    angles = phases[:, None] + directions[:, None] * history[None, :] * speed
    orientation = generator.uniform(-0.3, 0.3, (count, 1))
    cosine = np.cos(orientation)
    sine = np.sin(orientation)
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    rotated_x = cosine * base_x - sine * base_y
    rotated_y = sine * base_x + cosine * base_y

    third_strength = generator.uniform(0.08, 0.22, (count, 1))
    third_phase = generator.uniform(-math.pi, math.pi, (count, 1))
    channel_x = rotated_x + third_strength * np.sin(3.0 * angles + third_phase)
    channel_y = rotated_y + third_strength * np.cos(3.0 * angles - third_phase)
    channel_z = generator.uniform(0.18, 0.38, (count, 1)) * np.sin(
        3.0 * angles + third_phase
    )
    values = np.stack((channel_x, channel_y, channel_z), axis=-1)

    amplitude = generator.uniform(0.8, 1.2, (count, 1, 1))
    normalized_history = history / max(1.0, float(task.sensor_steps - 1))
    drift = generator.uniform(-0.18, 0.18, (count, 1, 3))
    offsets = generator.uniform(-0.08, 0.08, (count, 1, 3))
    values = amplitude * values
    values += offsets + drift * normalized_history[None, :, None]
    values += generator.laplace(0.0, 0.018, values.shape)
    return torch.from_numpy(_serialize_sensor_values(values, task))


def _inputs_for_family(
    task: CircleTaskConfig,
    phases: np.ndarray,
    directions: np.ndarray,
    *,
    family: str,
    seed: int,
) -> torch.Tensor:
    if family == "training_support":
        return generate_circle_dataset(
            task,
            sample_count=len(phases),
            seed=seed,
            phases=phases,
            directions=directions,
        ).input_ids
    if family == "disjoint_third_harmonic_drift":
        return _disjoint_nuisance_inputs(task, phases, directions, seed=seed)
    raise ValueError(f"unknown nuisance family: {family}")


def generate_matched_fiber_dataset(
    task: CircleTaskConfig,
    *,
    sample_count: int,
    seed: int,
    family: str,
) -> FiberDataset:
    """Generate balanced branch pairs with exactly matched cosine values."""
    if sample_count < 4 or sample_count % 2:
        raise ValueError("sample_count must be even and at least four")
    generator = np.random.default_rng(seed)
    pair_count = sample_count // 2
    cosine = generator.uniform(-0.95, 0.95, pair_count)
    positive_phase = np.arccos(cosine)
    negative_phase = (2.0 * math.pi - positive_phase) % (2.0 * math.pi)
    future_phase = np.column_stack((positive_phase, negative_phase)).reshape(-1)
    branch = np.tile(np.array([0.0, 1.0]), pair_count)
    fiber_id = np.repeat(np.arange(pair_count), 2)
    directions = np.repeat(generator.choice(np.array([-1.0, 1.0]), pair_count), 2)
    current_phase = (future_phase - directions * task.future_delta) % (2.0 * math.pi)
    inputs = _inputs_for_family(
        task,
        current_phase,
        directions,
        family=family,
        seed=seed + 10_003,
    )
    permutation = generator.permutation(sample_count)
    return FiberDataset(
        input_ids=inputs[permutation],
        cosine=torch.from_numpy(np.repeat(cosine, 2)[permutation].astype(np.float32)),
        branch=torch.from_numpy(branch[permutation].astype(np.float32)),
        phase=torch.from_numpy(future_phase[permutation].astype(np.float32)),
        fiber_id=torch.from_numpy(fiber_id[permutation].astype(np.int64)),
    )


@torch.no_grad()
def extract_layer_representations(
    model: TinyLLMModel,
    inputs: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> List[np.ndarray]:
    """Return the fixed-query embedding and every block residual."""
    model.eval()
    batches: Optional[List[List[torch.Tensor]]] = None
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        positions = torch.arange(batch.shape[1], device=device, dtype=torch.long)
        embedded = model.transformer["wte"](batch) + model.transformer["wpe"](positions)
        output = model(batch, output_hidden_states=True, return_dict=True)
        assert output.hidden_states is not None
        states = (embedded,) + output.hidden_states
        if batches is None:
            batches = [[] for _ in states]
        for index, state in enumerate(states):
            batches[index].append(state[:, -1, :].detach().cpu())
    assert batches is not None
    return [torch.cat(layer).float().numpy() for layer in batches]


class _ProbeNetwork(nn.Module):
    def __init__(self, inputs: int, width: int, outputs: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, outputs),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _ProbeSuite(nn.Module):
    def __init__(self, features: int, width: int):
        super().__init__()
        self.branch = _ProbeNetwork(features + 1, width, 1)
        self.cosine = _ProbeNetwork(features, width, 1)
        self.phase = _ProbeNetwork(features, width, 2)


def _probe_targets(dataset: FiberDataset, device: torch.device) -> Dict[str, torch.Tensor]:
    phase = dataset.phase.to(device)
    return {
        "cosine": dataset.cosine.to(device),
        "branch": dataset.branch.to(device),
        "phase_vector": torch.stack((torch.cos(phase), torch.sin(phase)), dim=1),
        "phase": phase,
    }


def _suite_loss(
    suite: _ProbeSuite,
    features: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    branch_inputs = torch.cat((features, targets["cosine"][:, None]), dim=1)
    return (
        F.binary_cross_entropy_with_logits(
            suite.branch(branch_inputs).squeeze(1), targets["branch"]
        )
        + F.mse_loss(suite.cosine(features).squeeze(1), targets["cosine"])
        + F.mse_loss(suite.phase(features), targets["phase_vector"])
    )


@torch.no_grad()
def _evaluate_probe_suite(
    suite: _ProbeSuite,
    features: torch.Tensor,
    targets: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    suite.eval()
    branch_inputs = torch.cat((features, targets["cosine"][:, None]), dim=1)
    branch_probability = torch.sigmoid(suite.branch(branch_inputs).squeeze(1))
    predicted_cosine = suite.cosine(features).squeeze(1)
    predicted_phase_vector = suite.phase(features)
    predicted_phase = torch.atan2(
        predicted_phase_vector[:, 1], predicted_phase_vector[:, 0]
    ) % (2.0 * math.pi)
    phase_error = torch.abs(
        torch.atan2(
            torch.sin(predicted_phase - targets["phase"]),
            torch.cos(predicted_phase - targets["phase"]),
        )
    )
    alignment = circular_phase_alignment(
        predicted_phase.cpu().numpy(), targets["phase"].cpu().numpy()
    )
    cosine_np = predicted_cosine.cpu().numpy()
    target_cosine_np = targets["cosine"].cpu().numpy()
    if np.ptp(cosine_np) <= 1e-6 or np.ptp(target_cosine_np) <= 1e-6:
        correlation = 0.0
    else:
        correlation = float(pearsonr(cosine_np, target_cosine_np).statistic)
        if not np.isfinite(correlation):
            correlation = 0.0
    return {
        "conditional_branch_accuracy": float(
            ((branch_probability >= 0.5) == targets["branch"].bool()).float().mean()
        ),
        "cosine_pearson": correlation,
        "cosine_rmse": float(
            torch.sqrt(F.mse_loss(predicted_cosine, targets["cosine"]))
        ),
        "phase_alignment": alignment["alignment"],
        "phase_mean_absolute_circular_error": float(phase_error.mean()),
        "phase_branch_accuracy": float(
            ((torch.sin(predicted_phase) < 0.0) == targets["branch"].bool())
            .float()
            .mean()
        ),
    }


def fit_probe_suite(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    evaluation_features: Mapping[str, np.ndarray],
    train_dataset: FiberDataset,
    validation_dataset: FiberDataset,
    evaluation_datasets: Mapping[str, FiberDataset],
    campaign: ProbeCampaignConfig,
    device: torch.device,
    *,
    seed: int,
) -> Dict[str, Any]:
    """Fit nonlinear probes without updating the source representation model."""
    torch.manual_seed(seed)
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = np.maximum(standard_deviation, 1e-5)

    def normalized(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((values - mean) / standard_deviation).float().to(device)

    train_x = normalized(train_features)
    validation_x = normalized(validation_features)
    evaluation_x = {key: normalized(value) for key, value in evaluation_features.items()}
    train_targets = _probe_targets(train_dataset, device)
    validation_targets = _probe_targets(validation_dataset, device)
    evaluation_targets = {
        key: _probe_targets(value, device) for key, value in evaluation_datasets.items()
    }
    width = min(campaign.probe_width, max(16, train_features.shape[1]))
    suite = _ProbeSuite(train_features.shape[1], width,).to(device)
    optimizer = torch.optim.AdamW(
        suite.parameters(),
        lr=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_step = 0
    stale = 0
    steps_run = 0
    for step in range(1, campaign.probe_steps + 1):
        indices = torch.randint(
            0,
            len(train_x),
            (campaign.probe_batch_size,),
            generator=generator,
        ).to(device)
        batch_targets = {key: value[indices] for key, value in train_targets.items()}
        suite.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _suite_loss(suite, train_x[indices], batch_targets)
        loss.backward()
        optimizer.step()
        steps_run = step
        if step % campaign.validation_interval == 0 or step == campaign.probe_steps:
            suite.eval()
            with torch.no_grad():
                validation_loss = float(
                    _suite_loss(suite, validation_x, validation_targets)
                )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in suite.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= campaign.early_stopping_patience:
                    break
    if best_state is None:
        raise RuntimeError("probe training did not produce a validation checkpoint")
    suite.load_state_dict(best_state)
    evaluations = {
        key: _evaluate_probe_suite(suite, evaluation_x[key], evaluation_targets[key])
        for key in evaluation_x
    }
    return {
        "best_step": best_step,
        "steps_run": steps_run,
        "best_validation_loss": best_loss,
        "evaluations": evaluations,
    }


def _phase_grid(
    task: CircleTaskConfig,
    campaign: ProbeCampaignConfig,
    *,
    family: str,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    future = np.linspace(
        0.0, 2.0 * math.pi, campaign.phase_grid_points, endpoint=False
    )
    directions = np.ones(campaign.phase_grid_points)
    current = (future - task.future_delta) % (2.0 * math.pi)
    return (
        _inputs_for_family(
            task, current, directions, family=family, seed=seed
        ),
        future,
    )


def independent_cohomology_analysis(
    representations: Sequence[np.ndarray],
    target_phase: np.ndarray,
) -> List[Dict[str, Any]]:
    """Discover and then label-evaluate the longest H1 coordinate per layer."""
    records = []
    for layer, values in enumerate(representations):
        distances = representation_distance_matrix(values, metric="euclidean")
        summary = summarize_persistence(distances)
        coordinate = persistent_cohomology_circle_coordinate(distances)
        alignment = circular_phase_alignment(coordinate["angles"], target_phase)
        records.append(
            {
                "layer": layer,
                "layer_kind": "query_embedding" if layer == 0 else "block_residual",
                "normalized_h1_lifetime": summary.normalized_h1_lifetime,
                "coordinate_found": coordinate["found"],
                "coordinate_lifetime": coordinate["lifetime"],
                "phase_alignment": alignment["alignment"],
                "orientation": alignment["orientation"],
                "rotation_radians": alignment["rotation_radians"],
            }
        )
    return records


def analyze_checkpoint(
    checkpoint: Path,
    task: CircleTaskConfig,
    campaign: ProbeCampaignConfig,
    device: torch.device,
    datasets: Mapping[str, FiberDataset],
) -> Dict[str, Any]:
    started = time.perf_counter()
    model = TinyLLMModel.from_checkpoint(checkpoint, map_location="cpu").to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    quotient = str(metadata["quotient"])
    preset = model.config.preset or f"d{model.config.n_layer}"

    activations = {
        key: extract_layer_representations(
            model,
            dataset.input_ids,
            device,
            batch_size=campaign.activation_batch_size,
        )
        for key, dataset in datasets.items()
    }
    layers = []
    for layer in range(len(activations["train"])):
        probe = fit_probe_suite(
            activations["train"][layer],
            activations["validation"][layer],
            {
                "in_distribution": activations["in_distribution"][layer],
                "disjoint_nuisance": activations["disjoint_nuisance"][layer],
            },
            datasets["train"],
            datasets["validation"],
            {
                "in_distribution": datasets["in_distribution"],
                "disjoint_nuisance": datasets["disjoint_nuisance"],
            },
            campaign,
            device,
            seed=campaign.seed + 1_003 * layer + 17 * model.config.n_layer,
        )
        layers.append(
            {
                "layer": layer,
                "layer_kind": "query_embedding" if layer == 0 else "block_residual",
                **probe,
            }
        )

    cohomology: Dict[str, Any] = {}
    for family_index, family in enumerate(
        ("training_support", "disjoint_third_harmonic_drift")
    ):
        inputs, target_phase = _phase_grid(
            task,
            campaign,
            family=family,
            seed=campaign.seed + 70_001 + family_index * 1_009,
        )
        representations = extract_layer_representations(
            model,
            inputs,
            device,
            batch_size=campaign.activation_batch_size,
        )
        cohomology[family] = independent_cohomology_analysis(
            representations, target_phase
        )

    final_id = layers[-1]["evaluations"]["in_distribution"]
    final_ood = layers[-1]["evaluations"]["disjoint_nuisance"]
    final_ph = cohomology["training_support"][-1]
    result = {
        "experiment_id": f"tinyllm-internal-quotient-{preset}-{quotient}-seed{metadata['seed']}",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "preset": preset,
        "quotient": quotient,
        "seed": int(metadata["seed"]),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "layerwise_probes": layers,
        "independent_persistent_cohomology": cohomology,
        "primary_metrics": {
            "final_id_conditional_branch_accuracy": final_id[
                "conditional_branch_accuracy"
            ],
            "final_ood_conditional_branch_accuracy": final_ood[
                "conditional_branch_accuracy"
            ],
            "final_id_cosine_pearson": final_id["cosine_pearson"],
            "final_ood_cosine_pearson": final_ood["cosine_pearson"],
            "final_id_phase_alignment": final_id["phase_alignment"],
            "final_ood_phase_alignment": final_ood["phase_alignment"],
            "final_ph_normalized_h1_lifetime": final_ph[
                "normalized_h1_lifetime"
            ],
            "final_ph_phase_alignment": final_ph["phase_alignment"],
        },
        "analysis_seconds": time.perf_counter() - started,
    }
    del model, activations
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_results(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_preset: Dict[str, Any] = {}
    for preset in sorted({str(run["preset"]) for run in runs}):
        cells = {
            str(run["quotient"]): run
            for run in runs
            if run["preset"] == preset
        }
        phase = cells.get("phase_circle")
        cosine = cells.get("cosine_interval")
        if not phase or not cosine:
            continue
        phase_metrics = phase["primary_metrics"]
        cosine_metrics = cosine["primary_metrics"]
        criteria = {
            "cosine_recoverable_from_both_final_representations_id": min(
                phase_metrics["final_id_cosine_pearson"],
                cosine_metrics["final_id_cosine_pearson"],
            )
            >= 0.9,
            "phase_final_branch_recoverable_id": phase_metrics[
                "final_id_conditional_branch_accuracy"
            ]
            >= 0.8,
            "phase_final_branch_recoverable_ood": phase_metrics[
                "final_ood_conditional_branch_accuracy"
            ]
            >= 0.7,
            "cosine_final_branch_at_chance_id": cosine_metrics[
                "final_id_conditional_branch_accuracy"
            ]
            <= 0.6,
            "cosine_final_branch_at_chance_ood": cosine_metrics[
                "final_ood_conditional_branch_accuracy"
            ]
            <= 0.6,
            "phase_to_phase_easy_id": phase_metrics["final_id_phase_alignment"]
            >= 0.9,
            "cosine_to_phase_ambiguous_id": cosine_metrics[
                "final_id_phase_alignment"
            ]
            <= 0.6,
            "independent_phase_coordinate_recovered": phase_metrics[
                "final_ph_phase_alignment"
            ]
            >= 0.8,
            "cosine_hidden_circle_not_phase_aligned": cosine_metrics[
                "final_ph_phase_alignment"
            ]
            <= 0.6,
        }
        by_preset[preset] = {
            "cells": {
                quotient: run["primary_metrics"] for quotient, run in cells.items()
            },
            "pre_registered_criteria": criteria,
            "criteria_passed": all(criteria.values()),
        }
    return by_preset


def run_campaign(
    checkpoints: Sequence[Path],
    task: CircleTaskConfig,
    campaign: ProbeCampaignConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(campaign.device)
    torch.use_deterministic_algorithms(True)
    datasets = {
        "train": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.train_samples,
            seed=campaign.seed + 101,
            family="training_support",
        ),
        "validation": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.validation_samples,
            seed=campaign.seed + 211,
            family="training_support",
        ),
        "in_distribution": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.test_samples,
            seed=campaign.seed + 307,
            family="training_support",
        ),
        "disjoint_nuisance": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.test_samples,
            seed=campaign.seed + 401,
            family="disjoint_third_harmonic_drift",
        ),
    }
    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "running",
        "started_at": _utc_now(),
        "task_config": asdict(task),
        "campaign_config": asdict(campaign),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
        "pre_registered_prediction": (
            "Phase-trained residuals retain cosine and the phase branch; cosine-trained "
            "residuals retain cosine but erase the branch, making phase cross-decoding "
            "ambiguous. Independent cohomology recovers an aligned circle only from "
            "the phase-trained residual stream."
        ),
        "probe_protocol": {
            "source_models_frozen": True,
            "conditional_branch_definition": "sign(sin(future_phase)) given exact matched cosine",
            "branch_uniform_baseline": 0.5,
            "nuisance_training_family": "original periodic generator support",
            "nuisance_test_family": "third harmonic plus temporal drift and Laplace noise",
            "coordinate_discovery_uses_labels": False,
            "coordinate_alignment_uses_labels_after_discovery": True,
        },
        "runs": [],
    }
    partial = output_dir / "results.partial.json"
    _write_json(partial, bundle)
    for checkpoint in checkpoints:
        run = analyze_checkpoint(checkpoint, task, campaign, device, datasets)
        bundle["runs"].append(run)
        _write_json(partial, bundle)
        print(run["experiment_id"], json.dumps(run["primary_metrics"], sort_keys=True), flush=True)
    bundle["aggregates"] = aggregate_results(bundle["runs"])
    bundle["status"] = "completed"
    bundle["completed_at"] = _utc_now()
    criteria_passed = bool(bundle["aggregates"]) and all(
        value["criteria_passed"] for value in bundle["aggregates"].values()
    )
    bundle["claim_status"] = {
        "confirmed": False,
        "predicted_asymmetry_observed": criteria_passed,
        "interpretation": (
            "The one-seed saved-checkpoint probe supports the predicted internal quotient asymmetry."
            if criteria_passed
            else "At least one saved-checkpoint probe criterion contradicted the internal quotient prediction."
        ),
        "confirmation_limit": "Only seed 7 checkpoints were retained; this campaign cannot confirm a multi-seed claim.",
    }
    _write_json(output_dir / "results.json", bundle)
    _write_json(partial, bundle)
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_internal_quotient_probe/20260805_d6_d8_seed7"
        ),
    )
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--validation-samples", type=int, default=1_024)
    parser.add_argument("--test-samples", type=int, default=2_048)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = sorted(args.checkpoint_dir.glob("*_trained_seed7.pt"))
    if len(checkpoints) != 4:
        raise FileNotFoundError(
            f"expected four matched seed-7 checkpoints in {args.checkpoint_dir}, found {len(checkpoints)}"
        )
    campaign = ProbeCampaignConfig(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        probe_steps=args.probe_steps,
        device=args.device,
    )
    bundle = run_campaign(checkpoints, CircleTaskConfig(), campaign, args.output)
    print(json.dumps(bundle["claim_status"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
