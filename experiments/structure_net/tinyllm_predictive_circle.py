#!/usr/bin/env python3
"""Run the predictive-circle semantic-quotient experiment on TinyLLM.

The task is deliberately synthetic: the semantic variable is a future phase on
S^1, while amplitude, offset, orientation, harmonic content, and observation
noise are nuisances.  The task posterior is normalized over a fixed set of
allowed answer tokens, so its Fisher--Rao topology can be compared directly to
the known semantic quotient.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from persim import bottleneck
import ripser as ripser_package
import torch
import torch.nn.functional as F

from structure_net.components.analyzers import (
    bootstrap_max_h1,
    fisher_rao_distance_matrix,
    knn_geodesic_distance_matrix,
    persistence_diagrams,
    representation_distance_matrix,
    summarize_persistence,
)
from structure_net.components.models import TinyLLMConfig, TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-predictive-circle.v1"
HYPOTHESIS_ID = "tinyllm-semantic-quotient-circle-v1"
CONDITIONS = ("trained", "label_shuffled", "time_shuffled")


@dataclass(frozen=True)
class CircleTaskConfig:
    phase_bins: int = 16
    sensor_steps: int = 8
    value_bins: int = 32
    vocab_size: int = 50_257
    answer_token_start: int = 32_000
    train_samples: int = 4_096
    evaluation_samples: int = 512
    phase_grid_points: int = 64
    nuisance_phase_count: int = 8
    nuisance_replicates: int = 12
    future_delta: float = 0.55
    target_noise_radians: float = 0.42
    quantization_limit: float = 2.0

    def __post_init__(self) -> None:
        if self.phase_bins < 8 or self.sensor_steps < 3 or self.value_bins < 4:
            raise ValueError("circle task resolution is too small")
        if self.answer_token_start + self.phase_bins > self.vocab_size:
            raise ValueError("answer tokens exceed the vocabulary")
        if self.sequence_length > 1_024:
            raise ValueError("serialized sensor sequence exceeds TinyLLM's block limit")

    @property
    def sequence_length(self) -> int:
        return 2 + 3 * self.sensor_steps

    @property
    def answer_token_ids(self) -> List[int]:
        return list(range(self.answer_token_start, self.answer_token_start + self.phase_bins))


@dataclass(frozen=True)
class CircleCampaignConfig:
    presets: tuple[str, ...] = ("d6", "d8")
    seeds: tuple[int, ...] = (7, 17, 29)
    conditions: tuple[str, ...] = CONDITIONS
    training_steps: int = 300
    checkpoints: tuple[int, ...] = (0, 50, 100, 300)
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
        if not self.presets or not self.seeds or not self.conditions:
            raise ValueError("presets, seeds, and conditions cannot be empty")
        unknown = set(self.conditions).difference(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        if self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("training_steps and batch_size must be positive")
        if 0 not in self.checkpoints or self.training_steps not in self.checkpoints:
            raise ValueError("checkpoints must include zero and the final training step")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be unique and sorted")
        if self.checkpoints[-1] != self.training_steps:
            raise ValueError("no checkpoint can follow the final training step")
        if not 0.0 < self.bootstrap_fraction <= 1.0:
            raise ValueError("bootstrap_fraction must be in (0, 1]")


@dataclass(frozen=True)
class CircleDataset:
    input_ids: torch.Tensor
    target_posteriors: torch.Tensor
    target_bins: torch.Tensor
    phases: torch.Tensor
    directions: torch.Tensor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrapped_angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (first - second + math.pi) % (2.0 * math.pi) - math.pi


def _wrapped_targets(centers: np.ndarray, config: CircleTaskConfig) -> np.ndarray:
    bin_angles = 2.0 * math.pi * np.arange(config.phase_bins) / config.phase_bins
    differences = _wrapped_angle_difference(bin_angles[None, :], centers[:, None])
    probabilities = np.exp(-0.5 * (differences / config.target_noise_radians) ** 2)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _serialize_sensor_values(values: np.ndarray, config: CircleTaskConfig) -> np.ndarray:
    scaled = (values + config.quantization_limit) / (2.0 * config.quantization_limit)
    quantized = np.floor(scaled * config.value_bins).astype(np.int64)
    quantized = np.clip(quantized, 0, config.value_bins - 1)
    channel_offsets = 100 + config.value_bins * np.arange(3, dtype=np.int64)
    tokens = quantized + channel_offsets[None, None, :]
    batch = len(values)
    serialized = np.empty((batch, config.sequence_length), dtype=np.int64)
    serialized[:, 0] = 1
    serialized[:, 1:-1] = tokens.reshape(batch, -1)
    serialized[:, -1] = 2
    return serialized


def generate_circle_dataset(
    config: CircleTaskConfig,
    *,
    sample_count: int,
    seed: int,
    nuisance_shift: bool = False,
    phases: Optional[np.ndarray] = None,
    directions: Optional[np.ndarray] = None,
) -> CircleDataset:
    """Generate serialized periodic sensors with a known S^1 semantic quotient."""
    generator = np.random.default_rng(seed)
    if phases is None:
        phase_values = generator.uniform(0.0, 2.0 * math.pi, sample_count)
    else:
        phase_values = np.asarray(phases, dtype=np.float64)
        if len(phase_values) != sample_count:
            raise ValueError("phase count does not match sample_count")
    if directions is None:
        direction_values = generator.choice(np.array([-1.0, 1.0]), sample_count)
    else:
        direction_values = np.asarray(directions, dtype=np.float64)
        if len(direction_values) != sample_count:
            raise ValueError("direction count does not match sample_count")

    if nuisance_shift:
        amplitude = generator.uniform(1.3, 1.6, (sample_count, 1, 1))
        orientation = generator.uniform(0.25, 0.45, (sample_count, 1))
        offsets = generator.uniform(0.18, 0.32, (sample_count, 1, 3))
        harmonic = generator.uniform(0.4, 0.55, (sample_count, 1))
        noise_scale = 0.08
        angular_speed = generator.uniform(0.24, 0.46, (sample_count, 1))
    else:
        amplitude = generator.uniform(0.75, 1.25, (sample_count, 1, 1))
        orientation = generator.uniform(-0.15, 0.15, (sample_count, 1))
        offsets = generator.uniform(-0.1, 0.1, (sample_count, 1, 3))
        harmonic = generator.uniform(0.15, 0.35, (sample_count, 1))
        noise_scale = 0.025
        angular_speed = generator.uniform(0.28, 0.42, (sample_count, 1))

    history = np.arange(config.sensor_steps, dtype=np.float64) - (config.sensor_steps - 1)
    angles = (
        phase_values[:, None]
        + direction_values[:, None] * history[None, :] * angular_speed
    )
    x_values = np.cos(angles)
    y_values = np.sin(angles)
    cosine = np.cos(orientation)
    sine = np.sin(orientation)
    rotated_x = cosine * x_values - sine * y_values
    rotated_y = sine * x_values + cosine * y_values
    z_values = harmonic * np.cos(2.0 * angles)
    values = amplitude * np.stack((rotated_x, rotated_y, z_values), axis=-1)
    values += offsets
    values += generator.normal(0.0, noise_scale, values.shape)

    future = (phase_values + direction_values * config.future_delta) % (2.0 * math.pi)
    target_posteriors = _wrapped_targets(future, config)
    target_bins = target_posteriors.argmax(axis=1)
    return CircleDataset(
        input_ids=torch.from_numpy(_serialize_sensor_values(values, config)),
        target_posteriors=torch.from_numpy(target_posteriors.astype(np.float32)),
        target_bins=torch.from_numpy(target_bins.astype(np.int64)),
        phases=torch.from_numpy(phase_values.astype(np.float32)),
        directions=torch.from_numpy(direction_values.astype(np.float32)),
    )


def shuffle_time_steps(input_ids: torch.Tensor, config: CircleTaskConfig, *, seed: int) -> torch.Tensor:
    """Shuffle sensor time blocks while keeping channels within each block together."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = input_ids[:, 1:-1].reshape(len(input_ids), config.sensor_steps, 3).clone()
    for row in range(len(values)):
        values[row] = values[row, torch.randperm(config.sensor_steps, generator=generator)]
    result = input_ids.clone()
    result[:, 1:-1] = values.reshape(len(values), -1)
    return result


def _conditioned_training_data(
    dataset: CircleDataset,
    config: CircleTaskConfig,
    condition: str,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = dataset.input_ids
    targets = dataset.target_posteriors
    if condition == "label_shuffled":
        generator = torch.Generator(device="cpu").manual_seed(seed + 20_003)
        targets = targets[torch.randperm(len(targets), generator=generator)]
    elif condition == "time_shuffled":
        inputs = shuffle_time_steps(inputs, config, seed=seed + 30_007)
    elif condition != "trained":
        raise ValueError(f"unknown condition: {condition}")
    return inputs, targets


def _resolve_device(requested: str) -> torch.device:
    if requested != "cuda:auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access it")
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:auto requested but PyTorch cannot access CUDA")
    available: List[tuple[int, int]] = []
    failures: List[str] = []
    for index in range(torch.cuda.device_count()):
        try:
            free_memory, _ = torch.cuda.mem_get_info(index)
            available.append((free_memory, index))
        except RuntimeError as error:
            failures.append(f"cuda:{index}: {error}")
    if not available:
        detail = "; ".join(failures) or "no CUDA devices were enumerated"
        raise RuntimeError(f"no CUDA device could be initialized: {detail}")
    _, device_index = max(available)
    return torch.device(f"cuda:{device_index}")


def _tinyllm_config(preset: str, task: CircleTaskConfig, seed: int) -> TinyLLMConfig:
    if preset == "tiny":
        return TinyLLMConfig(
            block_size=task.sequence_length,
            vocab_size=task.vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=32,
            initialization_seed=seed,
        )
    return TinyLLMConfig.from_preset(
        preset,
        block_size=task.sequence_length,
        vocab_size=task.vocab_size,
        initialization_seed=seed,
    )


def _task_logits(
    model: TinyLLMModel,
    input_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    *,
    hidden_states: bool = False,
) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
    output = model(
        input_ids,
        output_hidden_states=hidden_states,
        return_dict=hidden_states,
    )
    if hidden_states:
        logits = output.logits[:, -1, :]
        return logits.index_select(-1, answer_ids), output.hidden_states
    logits, _ = output
    return logits[:, -1, :].index_select(-1, answer_ids), None


@torch.no_grad()
def evaluate_task(
    model: TinyLLMModel,
    dataset: CircleDataset,
    answer_ids: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    probabilities = []
    for start in range(0, len(dataset.input_ids), batch_size):
        inputs = dataset.input_ids[start : start + batch_size].to(device)
        logits, _ = _task_logits(model, inputs, answer_ids)
        probabilities.append(torch.softmax(logits, dim=-1).cpu())
    predicted = torch.cat(probabilities)
    targets = dataset.target_posteriors
    kl = (targets * (targets.clamp_min(1e-12).log() - predicted.clamp_min(1e-12).log())).sum(1)
    predicted_bins = predicted.argmax(1)
    target_bins = targets.argmax(1)
    bin_delta = (predicted_bins - target_bins).abs()
    circular_bin_delta = torch.minimum(bin_delta, targets.shape[1] - bin_delta)
    return {
        "mean_kl": float(kl.mean()),
        "exact_bin_accuracy": float((predicted_bins == target_bins).float().mean()),
        "mean_circular_error_radians": float(
            circular_bin_delta.float().mean() * (2.0 * math.pi / targets.shape[1])
        ),
        "mean_target_cross_entropy": float(
            -(targets * predicted.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def _diagram_as_list(diagram: np.ndarray) -> List[List[float]]:
    return [[float(birth), float(death)] for birth, death in diagram if np.isfinite(death)]


def _persistence_record(distances: np.ndarray) -> Dict[str, Any]:
    diagrams = persistence_diagrams(distances)
    return {
        "summary": summarize_persistence(distances).as_dict(),
        "h1_diagram": _diagram_as_list(diagrams[1]),
    }


def _pullback_fisher_edge_lengths(
    model: TinyLLMModel,
    final_residuals: torch.Tensor,
    posteriors: torch.Tensor,
    answer_ids: torch.Tensor,
) -> np.ndarray:
    """Endpoint-averaged pullback-Fisher lengths at the final residual stream."""
    residuals = final_residuals.detach()
    answer_weights = model.lm_head.weight.index_select(0, answer_ids).detach()
    layer_norm = model.transformer["ln_f"]

    def selected_logits(hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            layer_norm.weight,
            layer_norm.bias,
            layer_norm.eps,
        )
        return F.linear(normalized, answer_weights)

    jacobians = torch.vmap(torch.func.jacrev(selected_logits))(residuals)
    deltas = residuals[None, :, :] - residuals[:, None, :]
    left_projection = torch.einsum("ikd,ijd->ijk", jacobians, deltas)
    right_projection = torch.einsum("jkd,ijd->ijk", jacobians, deltas)
    left_mean = (posteriors[:, None, :] * left_projection).sum(-1)
    right_mean = (posteriors[None, :, :] * right_projection).sum(-1)
    left_quadratic = (
        (posteriors[:, None, :] * left_projection.square()).sum(-1) - left_mean.square()
    )
    right_quadratic = (
        (posteriors[None, :, :] * right_projection.square()).sum(-1)
        - right_mean.square()
    )
    lengths = torch.sqrt((0.5 * (left_quadratic + right_quadratic)).clamp_min(0.0))
    lengths = 0.5 * (lengths + lengths.T)
    lengths.fill_diagonal_(0.0)
    return lengths.detach().cpu().double().numpy()


def _cyclic_sublevel_h0(values: np.ndarray) -> Dict[str, Any]:
    """Compute lower-star H0 lifetimes on a one-dimensional cyclic grid."""
    scalar = np.asarray(values, dtype=np.float64)
    count = len(scalar)
    order = np.argsort(scalar, kind="stable")
    parent = np.arange(count)
    births = scalar.copy()
    active = np.zeros(count, dtype=bool)
    lifetimes: List[float] = []

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index in order:
        active[index] = True
        for neighbor in ((index - 1) % count, (index + 1) % count):
            if not active[neighbor]:
                continue
            first = root(index)
            second = root(neighbor)
            if first == second:
                continue
            if births[first] <= births[second]:
                older, younger = first, second
            else:
                older, younger = second, first
            lifetimes.append(max(0.0, float(scalar[index] - births[younger])))
            parent[younger] = older
    positive = [value for value in lifetimes if value > 0.0]
    return {
        "finite_component_count": len(lifetimes),
        "positive_component_count": len(positive),
        "max_finite_lifetime": max(positive, default=0.0),
        "total_finite_lifetime": sum(positive),
    }


@torch.no_grad()
def _phase_grid_outputs(
    model: TinyLLMModel,
    dataset: CircleDataset,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, List[np.ndarray], torch.Tensor, torch.Tensor]:
    model.eval()
    inputs = dataset.input_ids.to(device)
    logits, hidden_states = _task_logits(model, inputs, answer_ids, hidden_states=True)
    assert hidden_states is not None
    posteriors = torch.softmax(logits, dim=-1)
    residuals = [state[:, -1, :].detach().cpu().double().numpy() for state in hidden_states]
    return (
        posteriors.detach().cpu().double().numpy(),
        residuals,
        hidden_states[-1][:, -1, :].detach(),
        posteriors.detach(),
    )


def _ground_truth_record(dataset: CircleDataset) -> Dict[str, Any]:
    distances = fisher_rao_distance_matrix(dataset.target_posteriors.double().numpy())
    return _persistence_record(distances)


def analyze_phase_grid(
    model: TinyLLMModel,
    dataset: CircleDataset,
    answer_ids: torch.Tensor,
    device: torch.device,
    campaign: CircleCampaignConfig,
    *,
    bootstrap_seed: int,
    full_analysis: bool,
) -> Dict[str, Any]:
    posteriors, layer_residuals, final_residuals, posterior_tensor = _phase_grid_outputs(
        model, dataset, answer_ids, device
    )
    posterior_distances = fisher_rao_distance_matrix(posteriors)
    posterior_record = _persistence_record(posterior_distances)
    target_distances = fisher_rao_distance_matrix(dataset.target_posteriors.double().numpy())
    target_diagram = persistence_diagrams(target_distances)[1]
    posterior_diagram = persistence_diagrams(posterior_distances)[1]
    layer_records = []
    for index, residuals in enumerate(layer_residuals):
        layer_records.append(
            {
                "block": index,
                "euclidean": _persistence_record(
                    representation_distance_matrix(residuals, metric="euclidean")
                ),
                "cosine": _persistence_record(
                    representation_distance_matrix(residuals, metric="cosine")
                ),
            }
        )

    confidence = np.partition(posteriors, -2, axis=1)
    top_two_margin = confidence[:, -1] - confidence[:, -2]
    result: Dict[str, Any] = {
        "posterior_fisher_rao": posterior_record,
        "posterior_h1_bottleneck_to_target": float(bottleneck(target_diagram, posterior_diagram)),
        "layer_residual_persistence": layer_records,
        "decision_margin_sublevel_h0": _cyclic_sublevel_h0(-top_two_margin),
    }
    if full_analysis:
        result["posterior_bootstrap"] = bootstrap_max_h1(
            posterior_distances,
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
        adjacent = np.array(
            [edge_lengths[index, (index + 1) % len(edge_lengths)] for index in range(len(edge_lengths))]
        )
        result["final_residual_pullback_fisher"] = _persistence_record(geodesic)
        result["semantic_tangent_rank_proxy"] = {
            "minimum_fisher_speed": float(adjacent.min()),
            "median_fisher_speed": float(np.median(adjacent)),
            "positive_speed_fraction": float(np.mean(adjacent > 1e-8)),
        }
    return result


@torch.no_grad()
def analyze_nuisance_fibers(
    model: TinyLLMModel,
    dataset: CircleDataset,
    answer_ids: torch.Tensor,
    device: torch.device,
    task: CircleTaskConfig,
) -> Dict[str, Any]:
    posteriors, residuals, _, _ = _phase_grid_outputs(model, dataset, answer_ids, device)
    final_residuals = residuals[-1]
    posterior_diameters = []
    posterior_h1 = []
    residual_diameters = []
    residual_h1 = []
    for phase_index in range(task.nuisance_phase_count):
        start = phase_index * task.nuisance_replicates
        stop = start + task.nuisance_replicates
        posterior_distance = fisher_rao_distance_matrix(posteriors[start:stop])
        residual_distance = representation_distance_matrix(
            final_residuals[start:stop], metric="euclidean"
        )
        posterior_diameters.append(float(posterior_distance.max()))
        residual_diameters.append(float(residual_distance.max()))
        posterior_h1.append(summarize_persistence(posterior_distance).total_h1_lifetime)
        residual_h1.append(summarize_persistence(residual_distance).total_h1_lifetime)
    return {
        "posterior_fisher_rao": {
            "mean_diameter": statistics.fmean(posterior_diameters),
            "max_diameter": max(posterior_diameters),
            "mean_total_h1_lifetime": statistics.fmean(posterior_h1),
        },
        "final_residual_euclidean": {
            "mean_diameter": statistics.fmean(residual_diameters),
            "max_diameter": max(residual_diameters),
            "mean_total_h1_lifetime": statistics.fmean(residual_h1),
        },
    }


def _state_digest(model: TinyLLMModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_bottleneck_tracking(checkpoints: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    tracking = []
    for previous, current in zip(checkpoints, checkpoints[1:]):
        first = np.asarray(previous["phase_topology"]["posterior_fisher_rao"]["h1_diagram"])
        second = np.asarray(current["phase_topology"]["posterior_fisher_rao"]["h1_diagram"])
        first = first.reshape(-1, 2) if first.size else np.empty((0, 2))
        second = second.reshape(-1, 2) if second.size else np.empty((0, 2))
        tracking.append(
            {
                "from_step": previous["step"],
                "to_step": current["step"],
                "posterior_h1_bottleneck_distance": float(bottleneck(first, second)),
            }
        )
    return tracking


def run_arm(
    *,
    preset: str,
    seed: int,
    condition: str,
    task: CircleTaskConfig,
    campaign: CircleCampaignConfig,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_config = _tinyllm_config(preset, task, seed)
    model = TinyLLMModel(model_config, name=f"PredictiveCircle_{preset}_{condition}_{seed}").to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
    )
    training = generate_circle_dataset(
        task, sample_count=task.train_samples, seed=seed + 1_001
    )
    training_inputs, training_targets = _conditioned_training_data(
        training, task, condition, seed=seed
    )
    in_domain = generate_circle_dataset(
        task, sample_count=task.evaluation_samples, seed=seed + 2_003
    )
    nuisance_shift = generate_circle_dataset(
        task,
        sample_count=task.evaluation_samples,
        seed=seed + 3_007,
        nuisance_shift=True,
    )
    phase_values = np.linspace(0.0, 2.0 * math.pi, task.phase_grid_points, endpoint=False)
    phase_grid = generate_circle_dataset(
        task,
        sample_count=task.phase_grid_points,
        seed=seed + 4_009,
        phases=phase_values,
        directions=np.ones(task.phase_grid_points),
    )
    fiber_phases = np.repeat(
        np.linspace(0.0, 2.0 * math.pi, task.nuisance_phase_count, endpoint=False),
        task.nuisance_replicates,
    )
    nuisance_fibers = generate_circle_dataset(
        task,
        sample_count=len(fiber_phases),
        seed=seed + 5_011,
        phases=fiber_phases,
        directions=np.ones(len(fiber_phases)),
    )
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    batch_indices = torch.randint(
        0,
        len(training_inputs),
        (campaign.training_steps, campaign.batch_size),
        generator=batch_generator,
    )
    checkpoint_steps = set(campaign.checkpoints)
    checkpoints: List[Dict[str, Any]] = []
    loss_history: List[Dict[str, float]] = []
    started = time.perf_counter()

    def record_checkpoint(step: int) -> None:
        full_analysis = step == campaign.training_steps
        checkpoints.append(
            {
                "step": step,
                "in_domain": evaluate_task(
                    model, in_domain, answer_ids, device, batch_size=campaign.batch_size
                ),
                "nuisance_shift": evaluate_task(
                    model, nuisance_shift, answer_ids, device, batch_size=campaign.batch_size
                ),
                "phase_topology": analyze_phase_grid(
                    model,
                    phase_grid,
                    answer_ids,
                    device,
                    campaign,
                    bootstrap_seed=seed + step,
                    full_analysis=full_analysis,
                ),
                **(
                    {
                        "nuisance_fibers": analyze_nuisance_fibers(
                            model, nuisance_fibers, answer_ids, device, task
                        )
                    }
                    if full_analysis
                    else {}
                ),
            }
        )

    record_checkpoint(0)
    model.train()
    for step in range(1, campaign.training_steps + 1):
        indices = batch_indices[step - 1]
        inputs = training_inputs[indices].to(device, non_blocking=True)
        targets = training_targets[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = _task_logits(model, inputs, answer_ids)
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), campaign.gradient_clip)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step in checkpoint_steps:
            loss_history.append(
                {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm)}
            )
        if step in checkpoint_steps:
            record_checkpoint(step)
            model.train()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    checkpoint_path: Optional[Path] = None
    if (
        campaign.save_primary_checkpoints
        and condition == "trained"
        and seed == campaign.seeds[0]
    ):
        checkpoint_path = output_dir / "checkpoints" / f"{preset}_trained_seed{seed}.pt"
        model.to("cpu").save_checkpoint(
            checkpoint_path,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "condition": condition,
                "seed": seed,
            },
        )
        model.to(device)

    final = checkpoints[-1]
    return {
        "experiment_id": f"tinyllm-circle-{preset}-{condition}-seed{seed}",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "preset": preset,
        "seed": seed,
        "condition": condition,
        "model_config": asdict(model_config),
        "model_parameters": parameter_count,
        "training_seconds": elapsed,
        "training_history": loss_history,
        "checkpoints": checkpoints,
        "checkpoint_diagram_tracking": _checkpoint_bottleneck_tracking(checkpoints),
        "final_state_sha256": _state_digest(model),
        "model_checkpoint": str(checkpoint_path.relative_to(output_dir)) if checkpoint_path else None,
        "primary_metrics": {
            "in_domain_exact_bin_accuracy": final["in_domain"]["exact_bin_accuracy"],
            "nuisance_shift_exact_bin_accuracy": final["nuisance_shift"]["exact_bin_accuracy"],
            "posterior_normalized_h1_lifetime": final["phase_topology"]
            ["posterior_fisher_rao"]["summary"]["normalized_h1_lifetime"],
            "posterior_h1_bottleneck_to_target": final["phase_topology"]
            ["posterior_h1_bottleneck_to_target"],
            "pullback_fisher_normalized_h1_lifetime": final["phase_topology"]
            ["final_residual_pullback_fisher"]["summary"]["normalized_h1_lifetime"],
            "nuisance_posterior_mean_diameter": final["nuisance_fibers"]
            ["posterior_fisher_rao"]["mean_diameter"],
        },
    }


def aggregate_results(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregates: Dict[str, Any] = {}
    for preset in sorted({run["preset"] for run in runs}):
        preset_runs = [run for run in runs if run["preset"] == preset]
        condition_records = {}
        for condition in CONDITIONS:
            selected = [run for run in preset_runs if run["condition"] == condition]
            if not selected:
                continue
            keys = selected[0]["primary_metrics"].keys()
            condition_records[condition] = {
                key: {
                    "mean": statistics.fmean(run["primary_metrics"][key] for run in selected),
                    "values": [run["primary_metrics"][key] for run in selected],
                }
                for key in keys
            }
        trained = condition_records.get("trained")
        shuffled = condition_records.get("label_shuffled")
        criteria: Dict[str, Optional[bool]] = {
            "trained_exact_accuracy_at_least_80_percent": (
                trained["in_domain_exact_bin_accuracy"]["mean"] >= 0.8 if trained else None
            ),
            "trained_posterior_h1_closer_than_label_shuffle": (
                trained["posterior_h1_bottleneck_to_target"]["mean"]
                < shuffled["posterior_h1_bottleneck_to_target"]["mean"]
                if trained and shuffled
                else None
            ),
            "trained_ood_accuracy_beats_label_shuffle": (
                trained["nuisance_shift_exact_bin_accuracy"]["mean"]
                > shuffled["nuisance_shift_exact_bin_accuracy"]["mean"]
                if trained and shuffled
                else None
            ),
        }
        aggregates[preset] = {
            "conditions": condition_records,
            "pre_registered_criteria": criteria,
            "criteria_passed": all(value is True for value in criteria.values()),
        }
    return aggregates


def run_campaign(
    task: CircleTaskConfig,
    campaign: CircleCampaignConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(campaign.device)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    started_at = _utc_now()
    ground_truth_phases = np.linspace(0.0, 2.0 * math.pi, task.phase_grid_points, endpoint=False)
    ground_truth = generate_circle_dataset(
        task,
        sample_count=task.phase_grid_points,
        seed=0,
        phases=ground_truth_phases,
        directions=np.ones(task.phase_grid_points),
    )
    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": "tinyllm_predictive_circle",
        "hypothesis_id": HYPOTHESIS_ID,
        "claim_scope": "multi_seed_d6_d8_synthetic_predictive_circle",
        "status": "running",
        "started_at": started_at,
        "task_config": asdict(task),
        "campaign_config": asdict(campaign),
        "ground_truth_semantic_quotient": {
            "space": "S1",
            "expected_betti_numbers": {"beta_0": 1, "beta_1": 1},
            "posterior_fisher_rao": _ground_truth_record(ground_truth),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ripser": ripser_package.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "deterministic_algorithms": True,
        },
        "runs": [],
        "unsupported_or_deferred": [
            "true zigzag persistence across checkpoints (diagram bottleneck tracking is reported instead)",
            "attention score-gap routing complexes",
            "pullback-Fisher metrics before the final residual stream",
            "Whitney jet-kernel and reach estimation beyond the reported tangent-speed proxy",
            "pretrained TinyLLM checkpoint comparison",
            "tokenizer-derived textual prompts (this run uses deterministic direct token serialization)",
        ],
    }
    partial_path = output_dir / "results.partial.json"
    if partial_path.exists():
        previous = json.loads(partial_path.read_text(encoding="utf-8"))
        expected_task = json.loads(json.dumps(asdict(task)))
        expected_campaign = json.loads(json.dumps(asdict(campaign)))
        if (
            previous.get("schema_version") != SCHEMA_VERSION
            or previous.get("task_config") != expected_task
            or previous.get("campaign_config") != expected_campaign
        ):
            raise ValueError(
                "existing partial result uses a different schema or configuration; "
                "choose a new output directory"
            )
        bundle["started_at"] = previous["started_at"]
        bundle["runs"] = previous.get("runs", [])
    _json_write(partial_path, bundle)
    try:
        completed_ids = {run["experiment_id"] for run in bundle["runs"]}
        for preset in campaign.presets:
            for seed in campaign.seeds:
                for condition in campaign.conditions:
                    experiment_id = f"tinyllm-circle-{preset}-{condition}-seed{seed}"
                    if experiment_id in completed_ids:
                        print(f"resuming: skipped completed {experiment_id}", flush=True)
                        continue
                    run = run_arm(
                        preset=preset,
                        seed=seed,
                        condition=condition,
                        task=task,
                        campaign=campaign,
                        device=device,
                        output_dir=output_dir,
                    )
                    bundle["runs"].append(run)
                    completed_ids.add(run["experiment_id"])
                    _json_write(partial_path, bundle)
                    print(
                        preset,
                        seed,
                        condition,
                        json.dumps(run["primary_metrics"], sort_keys=True),
                        flush=True,
                    )
        bundle["aggregates"] = aggregate_results(bundle["runs"])
        bundle["status"] = "completed"
        bundle["completed_at"] = _utc_now()
        bundle["claim_status"] = {
            "confirmed": all(
                value["criteria_passed"] for value in bundle["aggregates"].values()
            ),
            "interpretation": (
                "The pre-registered synthetic-circle criteria passed for every model class."
                if all(value["criteria_passed"] for value in bundle["aggregates"].values())
                else "At least one pre-registered synthetic-circle criterion failed; inspect controls."
            ),
            "does_not_establish": [
                "the same topology in natural-language or real sensor tasks",
                "a causal advantage from Structure Net feedback connections",
                "edge inference or parameter-efficiency gains",
            ],
        }
        _json_write(partial_path, bundle)
        _json_write(output_dir / "results.json", bundle)
        return bundle
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.set_num_threads(previous_threads)


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _comma_strings(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="d6,d8")
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--checkpoints", default="0,50,100,300")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_predictive_circle/20260805_d6_d8"),
    )
    parser.add_argument("--no-save-checkpoints", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = CircleCampaignConfig(
        presets=_comma_strings(args.presets),
        seeds=_comma_ints(args.seeds),
        conditions=_comma_strings(args.conditions),
        training_steps=args.steps,
        checkpoints=_comma_ints(args.checkpoints),
        batch_size=args.batch_size,
        bootstrap_repeats=args.bootstrap_repeats,
        device=args.device,
        save_primary_checkpoints=not args.no_save_checkpoints,
    )
    bundle = run_campaign(CircleTaskConfig(), campaign, args.output)
    print(json.dumps(bundle["claim_status"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
