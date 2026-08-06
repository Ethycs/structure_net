#!/usr/bin/env python3
"""Measure whether TinyLLM's internal cosine quotient survives nuisance shift.

The preregistered campaign crosses nested nuisance supports with final-depth,
multi-exit, continuous-gate, and explicit quotient-contrastive objectives. One
NAL experiment owns one support/objective/seed cell, including training,
frozen probes, block-1 localization, and branch-conditioned fiber geometry.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from experiments.structure_net.tinyllm_conditional_branch_depth_scan import (
        _add_log_loss_gains,
        extract_depth_representations,
        fit_conditional_probe,
        fit_cosine_only_null,
    )
    from experiments.structure_net.tinyllm_internal_quotient_probe import FiberDataset
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleDataset,
        CircleTaskConfig,
        _resolve_device,
        _serialize_sensor_values,
        _state_digest,
        _tinyllm_config,
        evaluate_task,
    )
    from experiments.structure_net.tinyllm_depth_graded_quotient import (
        training_depth_schedule,
    )
    from experiments.structure_net.tinyllm_task_quotient_contrast import (
        with_quotient_targets,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_conditional_branch_depth_scan import (  # type: ignore[no-redef]
        _add_log_loss_gains,
        extract_depth_representations,
        fit_conditional_probe,
        fit_cosine_only_null,
    )
    from tinyllm_internal_quotient_probe import FiberDataset  # type: ignore[no-redef]
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleDataset,
        CircleTaskConfig,
        _resolve_device,
        _serialize_sensor_values,
        _state_digest,
        _tinyllm_config,
        evaluate_task,
    )
    from tinyllm_depth_graded_quotient import (  # type: ignore[no-redef]
        training_depth_schedule,
    )
    from tinyllm_task_quotient_contrast import (  # type: ignore[no-redef]
        with_quotient_targets,
    )

from neural_architecture_lab.core import (
    Experiment,
    ExperimentResult,
    LabConfig,
)
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.analyzers import (
    fisher_rao_distance_matrix,
    representation_distance_matrix,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-nuisance-support-scaling.v1"
HYPOTHESIS_ID = "tinyllm-nuisance-invariant-internal-cosine-quotient-v1"
SUPPORTS = ("N0", "N1", "N2", "N3")
TRAINING_ARMS = (
    "standard_final",
    "discrete_multi_exit",
    "continuous_gate",
    "quotient_contrastive",
)
PRIMARY_CUTS = ("early", "full")
SHIFTED_REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class NuisanceScalingConfig:
    preset: str = "d8"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    supports: tuple[str, ...] = SUPPORTS
    training_arms: tuple[str, ...] = TRAINING_ARMS
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    sampled_depths_per_batch: int = 4
    quotient_loss_weight: float = 0.2
    quotient_margin: float = 0.5
    early_depth: float = 0.005
    probe_train_samples: int = 2_048
    probe_validation_samples: int = 512
    probe_test_samples: int = 1_024
    activation_batch_size: int = 256
    probe_batch_size: int = 256
    probe_steps: int = 240
    probe_width: int = 128
    validation_interval: int = 20
    early_stopping_patience: int = 5
    fiber_cosines: tuple[float, ...] = (-0.8, -0.4, 0.0, 0.4, 0.8)
    fiber_replicates: int = 12
    analysis_seed: int = 83
    device_ids: tuple[int, ...] = (-1,)
    gpu_slots_per_device: int = 1
    gpu_memory_per_experiment_gb: Optional[float] = 2.75
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 6
    max_retries: int = 1
    resume: bool = True
    design: str = "full_factorial"
    reuse_existing_cells: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("preregistered campaign requires at least five seeds")
        if not self.supports or set(self.supports).difference(SUPPORTS):
            raise ValueError(f"supports must be drawn from {SUPPORTS}")
        if not self.training_arms or set(self.training_arms).difference(TRAINING_ARMS):
            raise ValueError(f"training arms must be drawn from {TRAINING_ARMS}")
        if self.design not in {"full_factorial", "primary35"}:
            raise ValueError("design must be 'full_factorial' or 'primary35'")
        if self.design == "primary35" and (
            set(self.supports) != set(SUPPORTS)
            or set(self.training_arms) != set(TRAINING_ARMS)
        ):
            raise ValueError("primary35 requires all declared supports and training arms")
        if self.training_steps < 1 or self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("training steps must be positive and batch size even")
        sample_counts = (
            self.train_samples,
            self.probe_train_samples,
            self.probe_validation_samples,
            self.probe_test_samples,
        )
        if any(value < 4 or value % 2 for value in sample_counts):
            raise ValueError("all sample counts must be even and at least four")
        if self.sampled_depths_per_batch < 3:
            raise ValueError("sampled_depths_per_batch must be at least three")
        if not 0.0 < self.early_depth < 1.0:
            raise ValueError("early_depth must lie inside block 1")
        if self.probe_steps < 1 or self.probe_width < 2:
            raise ValueError("probe steps and width must be positive")
        if self.fiber_replicates < 2:
            raise ValueError("fiber_replicates must be at least two")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def nuisance_axes(support: str) -> tuple[str, ...]:
    """Return the cumulative nuisance axes enabled by one nested support."""
    index = SUPPORTS.index(support)
    axes = [
        (),
        ("amplitude", "offset"),
        ("amplitude", "offset", "orientation", "angular_speed"),
        (
            "amplitude",
            "offset",
            "orientation",
            "angular_speed",
            "harmonic",
            "direction",
            "noise",
        ),
    ]
    return axes[index]


def _sample_directions(
    support: str, regime: str, sample_count: int, generator: np.random.Generator
) -> np.ndarray:
    if support == "N3" or regime in {"composition", "extrapolation"}:
        return generator.choice(np.array([-1.0, 1.0]), sample_count)
    return np.ones(sample_count, dtype=np.float64)


def _nuisance_values(
    support: str,
    regime: str,
    sample_count: int,
    generator: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Draw a nested support, a held-out composition, or extrapolation family."""
    if regime == "extrapolation":
        sign = generator.choice(np.array([-1.0, 1.0]), (sample_count, 1))
        return {
            "amplitude": 1.0 + sign[:, :, None] * generator.uniform(
                0.30, 0.50, (sample_count, 1, 1)
            ),
            "orientation": sign * generator.uniform(0.22, 0.40, (sample_count, 1)),
            "offset": sign[:, :, None]
            * generator.uniform(0.15, 0.30, (sample_count, 1, 3)),
            "speed": generator.choice(np.array([0.20, 0.50]), (sample_count, 1))
            + generator.uniform(-0.03, 0.03, (sample_count, 1)),
            "harmonic_strength": generator.uniform(0.40, 0.58, (sample_count, 1)),
            "harmonic_order": np.full((sample_count, 1), 4.0),
            "harmonic_phase": generator.uniform(-0.6, 0.6, (sample_count, 1)),
            "noise_scale": generator.uniform(0.04, 0.07, (sample_count, 1, 1)),
            "drift": generator.uniform(-0.16, 0.16, (sample_count, 1, 3)),
        }

    level = SUPPORTS.index(support)
    amplitude = np.ones((sample_count, 1, 1))
    offset = np.zeros((sample_count, 1, 3))
    orientation = np.zeros((sample_count, 1))
    speed = np.full((sample_count, 1), 0.35)
    harmonic_strength = np.full((sample_count, 1), 0.25)
    harmonic_order = np.full((sample_count, 1), 2.0)
    harmonic_phase = np.zeros((sample_count, 1))
    noise_scale = np.zeros((sample_count, 1, 1))
    drift = np.zeros((sample_count, 1, 3))

    if level >= 1:
        offset = generator.uniform(-0.10, 0.10, (sample_count, 1, 3))
        amplitude = generator.uniform(0.75, 1.25, (sample_count, 1, 1))
    if level >= 2:
        magnitude_a = generator.uniform(0.01, 0.25, (sample_count, 1, 1))
        magnitude_o = generator.uniform(0.01, 0.15, (sample_count, 1))
        sign = generator.choice(np.array([-1.0, 1.0]), (sample_count, 1))
        orientation_sign = sign if regime == "composition" else -sign
        amplitude = 1.0 + sign[:, :, None] * magnitude_a
        orientation = orientation_sign * magnitude_o
        speed = generator.uniform(0.28, 0.42, (sample_count, 1))
    if level >= 3:
        harmonic_strength = generator.uniform(0.15, 0.35, (sample_count, 1))
        harmonic_order = generator.choice(np.array([2.0, 3.0]), (sample_count, 1))
        harmonic_phase = generator.uniform(-0.25, 0.25, (sample_count, 1))
        noise_scale = generator.uniform(0.0, 0.025, (sample_count, 1, 1))

    return {
        "amplitude": amplitude,
        "orientation": orientation,
        "offset": offset,
        "speed": speed,
        "harmonic_strength": harmonic_strength,
        "harmonic_order": harmonic_order,
        "harmonic_phase": harmonic_phase,
        "noise_scale": noise_scale,
        "drift": drift,
    }


def serialize_nuisance_inputs(
    task: CircleTaskConfig,
    current_phases: np.ndarray,
    directions: np.ndarray,
    *,
    support: str,
    regime: str,
    seed: int,
) -> torch.Tensor:
    """Serialize sensor histories for a declared operational nuisance family."""
    generator = np.random.default_rng(seed)
    count = len(current_phases)
    values = _nuisance_values(support, regime, count, generator)
    history = np.arange(task.sensor_steps, dtype=np.float64) - (task.sensor_steps - 1)
    angles = (
        current_phases[:, None]
        + directions[:, None] * history[None, :] * values["speed"]
    )
    base_x = np.cos(angles)
    base_y = np.sin(angles)
    cosine = np.cos(values["orientation"])
    sine = np.sin(values["orientation"])
    rotated_x = cosine * base_x - sine * base_y
    rotated_y = sine * base_x + cosine * base_y
    harmonic = values["harmonic_strength"] * np.cos(
        values["harmonic_order"] * angles + values["harmonic_phase"]
    )
    sensor = values["amplitude"] * np.stack((rotated_x, rotated_y, harmonic), axis=-1)
    sensor += values["offset"]
    normalized_history = history / max(1.0, float(task.sensor_steps - 1))
    sensor += values["drift"] * normalized_history[None, :, None]
    sensor += generator.normal(0.0, 1.0, sensor.shape) * values["noise_scale"]
    return torch.from_numpy(_serialize_sensor_values(sensor, task))


@dataclass(frozen=True)
class PairedCircleDataset:
    circle: CircleDataset
    fiber: FiberDataset


def generate_paired_dataset(
    task: CircleTaskConfig,
    *,
    sample_count: int,
    seed: int,
    support: str,
    regime: str = "interpolation",
    shuffle: bool = True,
) -> PairedCircleDataset:
    """Generate exact cosine-matched branch pairs with independent nuisances."""
    if sample_count < 4 or sample_count % 2:
        raise ValueError("sample_count must be even and at least four")
    generator = np.random.default_rng(seed)
    pair_count = sample_count // 2
    cosine = generator.uniform(-0.95, 0.95, pair_count)
    positive = np.arccos(cosine)
    negative = (2.0 * math.pi - positive) % (2.0 * math.pi)
    future = np.column_stack((positive, negative)).reshape(-1)
    branch = np.tile(np.array([0.0, 1.0]), pair_count)
    fiber_id = np.repeat(np.arange(pair_count), 2)
    directions = _sample_directions(support, regime, sample_count, generator)
    current = (future - directions * task.future_delta) % (2.0 * math.pi)
    inputs = serialize_nuisance_inputs(
        task,
        current,
        directions,
        support=support,
        regime=regime,
        seed=seed + 10_003,
    )
    circle = CircleDataset(
        input_ids=inputs,
        target_posteriors=torch.zeros(sample_count, task.phase_bins),
        target_bins=torch.zeros(sample_count, dtype=torch.int64),
        phases=torch.from_numpy(current.astype(np.float32)),
        directions=torch.from_numpy(directions.astype(np.float32)),
    )
    circle = with_quotient_targets(circle, task, "cosine_interval")
    permutation = generator.permutation(sample_count) if shuffle else np.arange(sample_count)
    perm = torch.from_numpy(permutation)
    return PairedCircleDataset(
        circle=CircleDataset(
            input_ids=circle.input_ids[perm],
            target_posteriors=circle.target_posteriors[perm],
            target_bins=circle.target_bins[perm],
            phases=circle.phases[perm],
            directions=circle.directions[perm],
        ),
        fiber=FiberDataset(
            input_ids=inputs[perm],
            cosine=torch.from_numpy(np.repeat(cosine, 2).astype(np.float32))[perm],
            branch=torch.from_numpy(branch.astype(np.float32))[perm],
            phase=torch.from_numpy(future.astype(np.float32))[perm],
            fiber_id=torch.from_numpy(fiber_id.astype(np.int64))[perm],
        ),
    )


def generate_geometry_dataset(
    task: CircleTaskConfig,
    *,
    cosine_values: Sequence[float],
    replicates: int,
    seed: int,
    support: str,
    regime: str,
) -> FiberDataset:
    """Sample nuisance clouds independently on both branches of exact fibers."""
    future: List[float] = []
    branch: List[float] = []
    cosine: List[float] = []
    fiber_id: List[int] = []
    for index, value in enumerate(cosine_values):
        phases = (math.acos(float(value)), 2.0 * math.pi - math.acos(float(value)))
        for label, phase in enumerate(phases):
            future.extend([phase] * replicates)
            branch.extend([float(label)] * replicates)
            cosine.extend([float(value)] * replicates)
            fiber_id.extend([index] * replicates)
    generator = np.random.default_rng(seed)
    future_values = np.asarray(future)
    directions = _sample_directions(support, regime, len(future_values), generator)
    current = (future_values - directions * task.future_delta) % (2.0 * math.pi)
    inputs = serialize_nuisance_inputs(
        task,
        current,
        directions,
        support=support,
        regime=regime,
        seed=seed + 10_003,
    )
    permutation = generator.permutation(len(future_values))
    return FiberDataset(
        input_ids=inputs[permutation],
        cosine=torch.tensor(np.asarray(cosine)[permutation], dtype=torch.float32),
        branch=torch.tensor(np.asarray(branch)[permutation], dtype=torch.float32),
        phase=torch.tensor(future_values[permutation], dtype=torch.float32),
        fiber_id=torch.tensor(np.asarray(fiber_id)[permutation], dtype=torch.int64),
    )


@torch.no_grad()
def extract_block1_representations(
    model: TinyLLMModel,
    inputs: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Capture query, post-attention, and post-MLP residuals in block 1."""
    model.eval()
    captured: Dict[str, List[torch.Tensor]] = {
        "query": [],
        "post_attention": [],
        "post_mlp": [],
    }
    block = model.transformer["h"][0]
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        positions = torch.arange(batch.shape[1], device=device)
        query = model.transformer["wte"](batch) + model.transformer["wpe"](positions)
        attended = query + block.attn(block.ln_1(query))
        complete = attended + block.mlp(block.ln_2(attended))
        captured["query"].append(query[:, -1, :].cpu())
        captured["post_attention"].append(attended[:, -1, :].cpu())
        captured["post_mlp"].append(complete[:, -1, :].cpu())
    return {key: torch.cat(parts).float().numpy() for key, parts in captured.items()}


def _distance_gamma(
    distances: np.ndarray, dataset: FiberDataset
) -> Dict[str, Any]:
    fibers = dataset.fiber_id.numpy()
    branches = dataset.branch.numpy().astype(int)
    records = []
    for fiber in np.unique(fibers):
        indices = np.flatnonzero(fibers == fiber)
        local = distances[np.ix_(indices, indices)]
        labels = branches[indices]
        within_values = []
        for label in (0, 1):
            subset = local[np.ix_(labels == label, labels == label)]
            triangle = subset[np.triu_indices(len(subset), k=1)]
            within_values.append(float(triangle.mean()))
        within = 0.5 * sum(within_values)
        cross = float(local[np.ix_(labels == 0, labels == 1)].mean())
        gamma = 1.0 if within <= 1e-12 and cross <= 1e-12 else cross / max(within, 1e-12)
        records.append(
            {
                "fiber_id": int(fiber),
                "cosine": float(dataset.cosine[indices[0]]),
                "cross_distance": cross,
                "within_distance": within,
                "gamma": gamma,
            }
        )
    values = np.asarray([record["gamma"] for record in records])
    return {
        "fibers": records,
        "mean_gamma": float(values.mean()),
        "median_gamma": float(np.median(values)),
        "merged_fraction": float(np.mean(values <= 1.25)),
    }


def fiber_geometry_record(
    features: np.ndarray,
    posteriors: np.ndarray,
    dataset: FiberDataset,
) -> Dict[str, Any]:
    """Compute branch-conditioned Fisher, Euclidean, and cosine fiber ratios."""
    metrics = {
        "fisher_rao": fisher_rao_distance_matrix(posteriors),
        "euclidean": representation_distance_matrix(features, metric="euclidean"),
        "cosine": representation_distance_matrix(features, metric="cosine"),
    }
    return {
        name: _distance_gamma(distances, dataset)
        for name, distances in metrics.items()
    }


@torch.no_grad()
def decoded_posteriors(
    model: TinyLLMModel,
    features: np.ndarray,
    answer_ids: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(features), batch_size):
        residual = torch.from_numpy(features[start : start + batch_size]).to(device)
        logits = model.lm_head(model.transformer["ln_f"](residual)).index_select(
            -1, answer_ids
        )
        batches.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(batches).double().numpy()


def _training_schedule(
    arm: str, config: NuisanceScalingConfig, layer_count: int, seed: int
) -> np.ndarray:
    if arm == "quotient_contrastive":
        return np.full((config.training_steps, 1), float(layer_count))
    return training_depth_schedule(
        arm,
        layer_count=layer_count,
        steps=config.training_steps,
        depths_per_batch=config.sampled_depths_per_batch,
        seed=seed + 8_019,
    )


def _task_logits_from_residual(
    model: TinyLLMModel, residual: torch.Tensor, answer_ids: torch.Tensor
) -> torch.Tensor:
    return model.lm_head(model.transformer["ln_f"](residual))[:, -1, :].index_select(
        -1, answer_ids
    )


def _quotient_contrastive_loss(
    residual: torch.Tensor, margin: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = F.normalize(residual[:, -1, :], dim=-1)
    positive = 1.0 - F.cosine_similarity(query[0::2], query[1::2], dim=-1)
    negative = 1.0 - F.cosine_similarity(query[0::2], query[1::2].roll(1, 0), dim=-1)
    loss = positive.mean() + F.relu(margin - negative).mean()
    return loss, positive.mean(), negative.mean()


def train_cell(
    task: CircleTaskConfig,
    config: NuisanceScalingConfig,
    support: str,
    arm: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[TinyLLMModel, Dict[str, Any]]:
    """Train one matched support/objective/seed cell and retain its weights."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_config = _tinyllm_config(config.preset, task, seed)
    model = TinyLLMModel(
        model_config, name=f"NuisanceScaling_{support}_{arm}_{seed}"
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training = generate_paired_dataset(
        task,
        sample_count=config.train_samples,
        seed=seed + 1_001,
        support=support,
        shuffle=False,
    ).circle
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    pair_batches = torch.randint(
        0,
        config.train_samples // 2,
        (config.training_steps, config.batch_size // 2),
        generator=generator,
    )
    schedule = _training_schedule(arm, config, model.config.n_layer, seed)
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    for step in range(1, config.training_steps + 1):
        pairs = pair_batches[step - 1]
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        inputs = training.input_ids[indices].to(device, non_blocking=True)
        targets = training.target_posteriors[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        task_loss_values: List[float] = []
        quotient_loss_value = torch.tensor(0.0, device=device)
        positive_distance = torch.tensor(0.0, device=device)
        negative_distance = torch.tensor(0.0, device=device)
        sampled_depths = schedule[step - 1]
        for depth in sampled_depths:
            residual = model.residual_at_depth(inputs, float(depth))
            logits = _task_logits_from_residual(model, residual, answer_ids)
            task_loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            task_loss_values.append(float(task_loss.detach()))
            if arm == "quotient_contrastive":
                quotient_loss_value, positive_distance, negative_distance = (
                    _quotient_contrastive_loss(residual, config.quotient_margin)
                )
                (
                    task_loss
                    + config.quotient_loss_weight * quotient_loss_value
                ).backward()
            else:
                # The scalar objective is the mean over sampled depths. Backward
                # immediately so each full transformer graph can be released before
                # the next depth is constructed.
                (task_loss / len(sampled_depths)).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            history.append(
                {
                    "step": float(step),
                    "task_loss": float(np.mean(task_loss_values)),
                    "quotient_loss": float(quotient_loss_value.detach()),
                    "positive_pair_distance": float(positive_distance.detach()),
                    "negative_pair_distance": float(negative_distance.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    digest = _state_digest(model)
    checkpoint = output_dir / "model.pt"
    model.to("cpu").save_checkpoint(
        checkpoint,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "support": support,
            "training_arm": arm,
            "seed": seed,
            "training_sample_count": config.train_samples,
            "training_steps": config.training_steps,
        },
    )
    model.to(device)
    evaluation = generate_paired_dataset(
        task,
        sample_count=min(config.probe_test_samples, 512),
        seed=seed + 2_003,
        support=support,
    ).circle
    metrics = evaluate_task(
        model,
        evaluation,
        answer_ids,
        device,
        batch_size=config.activation_batch_size,
    )
    return model, {
        "training_seconds": training_seconds,
        "training_history": history,
        "final_state_sha256": digest,
        "checkpoint": str(checkpoint),
        "task_metrics_on_training_support": metrics,
        "training_depth_schedule_sha256": __import__("hashlib").sha256(
            schedule.tobytes()
        ).hexdigest(),
    }


def _evaluation_specs(training_support: str) -> List[tuple[str, str, str]]:
    start = SUPPORTS.index(training_support)
    specs = [("interpolation", training_support, "interpolation")]
    specs.extend(
        (f"support_{support}", support, "interpolation")
        for support in SUPPORTS[start + 1 :]
    )
    specs.extend(
        (
            ("composition", "N3", "composition"),
            ("extrapolation", "N3", "extrapolation"),
        )
    )
    return specs


def analyze_cell(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    config: NuisanceScalingConfig,
    support: str,
    arm: str,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Fit frozen probes and compute shifted block-1/fiber measurements."""
    probe_train = generate_paired_dataset(
        task,
        sample_count=config.probe_train_samples,
        seed=config.analysis_seed + 101,
        support=support,
    ).fiber
    probe_validation = generate_paired_dataset(
        task,
        sample_count=config.probe_validation_samples,
        seed=config.analysis_seed + 211,
        support=support,
    ).fiber
    specs = _evaluation_specs(support)
    evaluations = {
        name: generate_paired_dataset(
            task,
            sample_count=config.probe_test_samples,
            seed=config.analysis_seed + 307 + index * 1_009,
            support=test_support,
            regime=regime,
        ).fiber
        for index, (name, test_support, regime) in enumerate(specs)
    }
    geometry_datasets = {
        name: generate_geometry_dataset(
            task,
            cosine_values=config.fiber_cosines,
            replicates=config.fiber_replicates,
            seed=config.analysis_seed + 20_003 + index * 1_009,
            support=test_support,
            regime=regime,
        )
        for index, (name, test_support, regime) in enumerate(specs)
    }
    all_probe = {"train": probe_train, "validation": probe_validation, **evaluations}
    all_geometry = {f"geometry_{key}": value for key, value in geometry_datasets.items()}
    all_datasets = {**all_probe, **all_geometry}
    depths = (config.early_depth, float(model.config.n_layer))
    depth_features = {
        name: extract_depth_representations(
            model,
            dataset.input_ids,
            depths,
            device,
            batch_size=config.activation_batch_size,
        )
        for name, dataset in all_datasets.items()
    }
    block1_features = {
        name: extract_block1_representations(
            model,
            dataset.input_ids,
            device,
            batch_size=config.activation_batch_size,
        )
        for name, dataset in all_datasets.items()
    }
    features: Dict[str, Dict[str, np.ndarray]] = {
        "early": {name: value[config.early_depth] for name, value in depth_features.items()},
        "full": {name: value[float(model.config.n_layer)] for name, value in depth_features.items()},
        "query": {name: value["query"] for name, value in block1_features.items()},
        "post_attention": {
            name: value["post_attention"] for name, value in block1_features.items()
        },
        "post_mlp": {name: value["post_mlp"] for name, value in block1_features.items()},
    }
    null = fit_cosine_only_null(
        probe_train,
        probe_validation,
        evaluations,
        config,
        device,
        kind="nonlinear",
        seed=config.analysis_seed + seed * 101,
    )
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    cuts: Dict[str, Any] = {}
    for cut_index, (cut, cut_features) in enumerate(features.items()):
        probe = fit_conditional_probe(
            cut_features["train"],
            cut_features["validation"],
            {name: cut_features[name] for name in evaluations},
            probe_train,
            probe_validation,
            evaluations,
            config,
            device,
            kind="nonlinear",
            seed=config.analysis_seed + seed * 10_007 + cut_index * 1_009,
        )
        _add_log_loss_gains(probe, null)
        geometry: Dict[str, Any] = {}
        for name, dataset in geometry_datasets.items():
            hidden = cut_features[f"geometry_{name}"]
            posteriors = decoded_posteriors(
                model,
                hidden,
                answer_ids,
                device,
                batch_size=config.activation_batch_size,
            )
            geometry[name] = fiber_geometry_record(hidden, posteriors, dataset)
        endpoint = {}
        for name, metrics in probe["evaluations"].items():
            fisher_gamma = geometry[name]["fisher_rao"]["median_gamma"]
            endpoint[name] = {
                "base_retained": metrics["cosine_pearson"] >= 0.9,
                "fiber_probe_erased": metrics["balanced_accuracy"] <= 0.55,
                "fiber_geometry_merged": fisher_gamma <= 1.25,
                "invariant_quotient": (
                    metrics["cosine_pearson"] >= 0.9
                    and metrics["balanced_accuracy"] <= 0.55
                    and fisher_gamma <= 1.25
                ),
            }
        cuts[cut] = {
            "probe": probe,
            "fiber_geometry": geometry,
            "endpoint": endpoint,
        }
        print(
            f"{support}/{arm}/seed{seed}",
            cut,
            " ".join(
                f"{name}:cos={probe['evaluations'][name]['cosine_pearson']:.3f},"
                f"branch={probe['evaluations'][name]['balanced_accuracy']:.3f},"
                f"gamma={geometry[name]['fisher_rao']['median_gamma']:.3f}"
                for name in SHIFTED_REGIMES
            ),
            flush=True,
        )
    return {
        "probe_training_support": support,
        "evaluation_specs": [
            {"name": name, "support": test_support, "regime": regime}
            for name, test_support, regime in specs
        ],
        "conditional_cosine_only_null": null,
        "cuts": cuts,
    }


def _config_from_mapping(values: Mapping[str, Any]) -> NuisanceScalingConfig:
    converted = dict(values)
    for field in ("seeds", "supports", "training_arms", "fiber_cosines", "device_ids"):
        converted[field] = tuple(converted[field])
    return NuisanceScalingConfig(**converted)


def _cell_directory(root: Path, support: str, arm: str, seed: int) -> Path:
    return root / "runs" / support / arm / f"seed_{seed}"


def nuisance_scaling_worker(
    experiment: Experiment, device_id: int
) -> ExperimentResult:
    """Picklable NAL worker for one complete nuisance-scaling cell."""
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    support = str(experiment.parameters["support"])
    arm = str(experiment.parameters["training_arm"])
    seed = int(experiment.seed)
    output_dir = _cell_directory(
        Path(experiment.parameters["output_dir"]), support, arm, seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model, training = train_cell(
        task, config, support, arm, seed, device, output_dir
    )
    analysis = analyze_cell(model, task, config, support, arm, seed, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = (
        torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() / (1024**3) if device.type == "cuda" else 0.0
    )
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "completed_at": _utc_now(),
        "support": support,
        "training_arm": arm,
        "seed": seed,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "training": training,
        "analysis": analysis,
        "peak_cuda_allocated_gb": peak_allocated,
        "peak_cuda_reserved_gb": peak_reserved,
        "wall_seconds": elapsed,
    }
    detail_path = output_dir / "result.json"
    _write_json(detail_path, detail)
    full_composition = analysis["cuts"]["full"]["probe"]["evaluations"][
        "composition"
    ]
    full_extrapolation = analysis["cuts"]["full"]["probe"]["evaluations"][
        "extrapolation"
    ]
    metrics = {
        "support_index": float(SUPPORTS.index(support)),
        "logical_device_id": float(device_id),
        "full_composition_cosine": float(full_composition["cosine_pearson"]),
        "full_composition_branch": float(full_composition["balanced_accuracy"]),
        "full_extrapolation_cosine": float(full_extrapolation["cosine_pearson"]),
        "full_extrapolation_branch": float(full_extrapolation["balanced_accuracy"]),
        "peak_cuda_allocated_gb": peak_allocated,
        "peak_cuda_reserved_gb": peak_reserved,
    }
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics=metrics,
        primary_metric=min(
            metrics["full_composition_cosine"], metrics["full_extrapolation_cosine"]
        ),
        model_architecture=[
            task.vocab_size,
            *([model.config.n_embd] * model.config.n_layer),
            task.phase_bins,
        ],
        model_parameters=detail["model_parameters"],
        training_time=elapsed,
        training_history=training["training_history"],
        model_checkpoint=training["checkpoint"],
        observations=[f"detail={detail_path}", f"support={support}", f"arm={arm}"],
    )


def _experiments(
    config: NuisanceScalingConfig,
    task: CircleTaskConfig,
    output_dir: Path,
) -> List[Experiment]:
    parameters = {
        "configuration": asdict(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "worker_schema_version": SCHEMA_VERSION,
    }
    if config.design == "primary35":
        cells = [
            (support, "standard_final") for support in SUPPORTS
        ] + [
            ("N3", arm) for arm in TRAINING_ARMS if arm != "standard_final"
        ]
    else:
        cells = [
            (support, arm)
            for support in config.supports
            for arm in config.training_arms
        ]
    return [
        Experiment(
            id=f"tinyllm-nuisance-{support}-{arm}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM nuisance quotient {support} {arm} seed {seed}",
            parameters={
                **parameters,
                "support": support,
                "training_arm": arm,
                "seed": seed,
            },
            seed=seed,
        )
        for support, arm in cells
        for seed in config.seeds
    ]


def _existing_detail(
    experiment: Experiment, output_dir: Path
) -> Optional[Dict[str, Any]]:
    """Load a complete compatible cell artifact without retraining its model."""
    support = str(experiment.parameters["support"])
    arm = str(experiment.parameters["training_arm"])
    seed = int(experiment.seed)
    path = _cell_directory(output_dir, support, arm, seed) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    checkpoint = Path(detail.get("training", {}).get("checkpoint", ""))
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("support") != support
        or detail.get("training_arm") != arm
        or int(detail.get("seed", -1)) != seed
        or not checkpoint.is_file()
    ):
        return None
    return detail


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: NuisanceScalingConfig
) -> Dict[str, Any]:
    """Aggregate the preregistered scaling curve and conclusion gates."""
    cells: Dict[str, Any] = {}
    for support in config.supports:
        for arm in config.training_arms:
            selected = [
                item
                for item in details
                if item["support"] == support and item["training_arm"] == arm
            ]
            if not selected:
                continue
            cut_records: Dict[str, Any] = {}
            for cut in ("early", "full", "query", "post_attention", "post_mlp"):
                regimes: Dict[str, Any] = {}
                names = selected[0]["analysis"]["cuts"][cut]["probe"]["evaluations"]
                for regime in names:
                    probes = [
                        item["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]
                        for item in selected
                    ]
                    geometry = [
                        item["analysis"]["cuts"][cut]["fiber_geometry"][regime]
                        ["fisher_rao"]["median_gamma"]
                        for item in selected
                    ]
                    invariant = [
                        item["analysis"]["cuts"][cut]["endpoint"][regime]
                        ["invariant_quotient"]
                        for item in selected
                    ]
                    regimes[regime] = {
                        "seed_count": len(selected),
                        "cosine_pearson_mean": float(
                            np.mean([probe["cosine_pearson"] for probe in probes])
                        ),
                        "cosine_pearson_by_seed": {
                            str(item["seed"]): probe["cosine_pearson"]
                            for item, probe in zip(selected, probes)
                        },
                        "branch_accuracy_mean": float(
                            np.mean([probe["balanced_accuracy"] for probe in probes])
                        ),
                        "branch_accuracy_by_seed": {
                            str(item["seed"]): probe["balanced_accuracy"]
                            for item, probe in zip(selected, probes)
                        },
                        "fisher_gamma_median_mean": float(np.mean(geometry)),
                        "invariant_seed_fraction": float(np.mean(invariant)),
                        "all_seeds_invariant": bool(all(invariant)),
                    }
                cut_records[cut] = regimes
            cells[f"{support}:{arm}"] = {"cuts": cut_records}

    strong_by_arm: Dict[str, bool] = {}
    if "N3" in config.supports:
        for arm in config.training_arms:
            cell = cells.get(f"N3:{arm}")
            strong_by_arm[arm] = bool(
                cell
                and all(
                    cell["cuts"][cut][regime]["all_seeds_invariant"]
                    for cut in PRIMARY_CUTS
                    for regime in SHIFTED_REGIMES
                )
            )

    scaling: Dict[str, Any] = {}
    for arm in config.training_arms:
        if not all(f"{support}:{arm}" in cells for support in SUPPORTS):
            continue
        for cut in PRIMARY_CUTS:
            for regime in SHIFTED_REGIMES:
                slopes = []
                for seed in config.seeds:
                    values = [
                        cells[f"{support}:{arm}"]["cuts"][cut][regime]
                        ["cosine_pearson_by_seed"][str(seed)]
                        for support in SUPPORTS
                    ]
                    slopes.append(float(np.polyfit(np.arange(4), values, 1)[0]))
                n0 = cells[f"N0:{arm}"]["cuts"][cut][regime]
                n3 = cells[f"N3:{arm}"]["cuts"][cut][regime]
                scaling[f"{arm}:{cut}:{regime}"] = {
                    "paired_seed_slopes": slopes,
                    "mean_slope": float(np.mean(slopes)),
                    "positive_slope_fraction": float(np.mean(np.asarray(slopes) > 0.0)),
                    "N3_minus_N0_cosine": (
                        n3["cosine_pearson_mean"] - n0["cosine_pearson_mean"]
                    ),
                    "N3_branch_accuracy_mean": n3["branch_accuracy_mean"],
                }

    coverage_by_arm = {
        arm: bool(
            all(
                scaling[f"{arm}:{cut}:{regime}"]["positive_slope_fraction"] == 1.0
                and scaling[f"{arm}:{cut}:{regime}"]["N3_minus_N0_cosine"] >= 0.05
                and scaling[f"{arm}:{cut}:{regime}"]["N3_branch_accuracy_mean"] <= 0.55
                for cut in PRIMARY_CUTS
                for regime in SHIFTED_REGIMES
            )
        )
        for arm in config.training_arms
        if all(f"{arm}:{cut}:{regime}" in scaling for cut in PRIMARY_CUTS for regime in SHIFTED_REGIMES)
    }
    standard_strong = strong_by_arm.get("standard_final", False)
    objective_strong = any(
        value for arm, value in strong_by_arm.items() if arm != "standard_final"
    )
    objective_dependent = objective_strong and not standard_strong
    broad_coverage_failure = bool(
        "N3" in config.supports
        and not any(
            all(
                cells[f"N3:{arm}"]["cuts"][cut][regime]["cosine_pearson_mean"]
                >= 0.9
                for cut in PRIMARY_CUTS
                for regime in SHIFTED_REGIMES
            )
            for arm in config.training_arms
            if f"N3:{arm}" in cells
        )
    )
    strong_success = any(strong_by_arm.values())
    coverage_success = any(coverage_by_arm.values()) and not strong_success
    if objective_dependent:
        conclusion = "objective_dependent_success"
    elif strong_success:
        conclusion = "strong_success"
    elif coverage_success:
        conclusion = "coverage_dependent_success"
    elif broad_coverage_failure:
        conclusion = "failure_despite_broad_coverage"
    else:
        conclusion = "inconclusive_or_mixed"
    return {
        "cells": cells,
        "scaling_curves": scaling,
        "preregistered_gates": {
            "strong_success_by_arm": strong_by_arm,
            "coverage_dependent_success_by_arm": coverage_by_arm,
            "objective_dependent_success": objective_dependent,
            "failure_despite_broad_coverage": broad_coverage_failure,
        },
        "conclusion": conclusion,
    }


async def run_campaign(
    config: NuisanceScalingConfig,
    task: CircleTaskConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = _experiments(config, task, output_dir)
    existing = {
        experiment.id: detail
        for experiment in experiments
        if config.reuse_existing_cells
        and (detail := _existing_detail(experiment, output_dir)) is not None
    }
    pending = [
        experiment for experiment in experiments if experiment.id not in existing
    ]
    lab = LabConfig(
        project_name="tinyllm_nuisance_support_scaling",
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
    runner = AsyncExperimentRunner(lab, nuisance_scaling_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful_ids = {result.experiment_id for result in results if result.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id not in successful_ids:
            continue
        path = _cell_directory(
            output_dir,
            str(experiment.parameters["support"]),
            str(experiment.parameters["training_arm"]),
            int(experiment.seed),
        ) / "result.json"
        details.append(json.loads(path.read_text(encoding="utf-8")))
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if len(details) == len(experiments) else "partial",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value for key, value in runner.slot_plan.slots_by_device.items()
            },
            "free_memory_gb": {
                str(key): value for key, value in runner.slot_plan.free_memory_gb.items()
            },
            "calibration": runner.slot_plan.calibration,
        },
        "summary": {
            "requested": len(experiments),
            "reused": len(existing),
            "scheduled": len(pending),
            "completed": len(details),
            "failed": len(experiments) - len(details),
        },
        "aggregates": aggregate_details(details, config) if details else {},
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
            "frozen nonlinear probes bound tested decodability, not conditional mutual information",
            "Fisher gamma is computed from task-posterior distances at sampled fibers",
            "Euclidean and cosine gamma are hidden-space controls",
            "the composition claim applies strictly to N2 and N3, where both marginals were trained",
            "shared-run wall time is not a benchmark",
        ],
    }
    _write_json(output_dir / "campaign_results.json", bundle)
    return bundle


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_devices(value: str) -> tuple[int, ...]:
    normalized = value.strip().lower()
    if normalized == "cpu":
        return (-1,)
    if normalized == "auto":
        return tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else (-1,)
    return _comma_ints(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="d8")
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--supports", default=",".join(SUPPORTS))
    parser.add_argument("--training-arms", default=",".join(TRAINING_ARMS))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--probe-train-samples", type=int, default=2_048)
    parser.add_argument("--probe-validation-samples", type=int, default=512)
    parser.add_argument("--probe-test-samples", type=int, default=1_024)
    parser.add_argument("--fiber-replicates", type=int, default=12)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--slots-per-gpu", type=int, default=0)
    parser.add_argument("--memory-per-cell-gb", type=float, default=2.75)
    parser.add_argument("--max-slots-per-gpu", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--design",
        choices=("full_factorial", "primary35"),
        default="full_factorial",
    )
    parser.add_argument(
        "--reuse-existing-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_nuisance_support_scaling/20260806_d8_full"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.preset = "tiny"
        args.seeds = "7"
        args.supports = "N0,N3"
        args.training_arms = "standard_final,quotient_contrastive"
        args.steps = 2
        args.train_samples = 32
        args.batch_size = 8
        args.probe_steps = 20
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 32
        args.fiber_replicates = 3
        args.allow_underpowered = True
    config = NuisanceScalingConfig(
        preset=args.preset,
        seeds=_comma_ints(args.seeds),
        supports=_comma_strings(args.supports),
        training_arms=_comma_strings(args.training_arms),
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        probe_steps=args.probe_steps,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
        fiber_replicates=args.fiber_replicates,
        device_ids=_parse_devices(args.gpus),
        gpu_slots_per_device=args.slots_per_gpu,
        gpu_memory_per_experiment_gb=args.memory_per_cell_gb,
        max_gpu_slots_per_device=args.max_slots_per_gpu,
        max_parallel_experiments=args.max_parallel,
        max_retries=args.retries,
        resume=args.resume,
        design=args.design,
        reuse_existing_cells=args.reuse_existing_cells,
        allow_underpowered=args.allow_underpowered,
    )
    task = CircleTaskConfig(train_samples=config.train_samples)
    bundle = asyncio.run(run_campaign(config, task, args.output))
    print(
        json.dumps(
            {
                "summary": bundle["summary"],
                "conclusion": bundle.get("aggregates", {}).get("conclusion"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
