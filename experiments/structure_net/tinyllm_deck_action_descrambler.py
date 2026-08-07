#!/usr/bin/env python3
"""Descramble and causally test deck-group actions in retained degree-k TinyLLMs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import schur
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_degree_k_ladder as ladder
import experiments.structure_net.tinyllm_io_correspondence as io_source
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    _nuisance_values,
    _sample_directions,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
)
from structure_net.components.analyzers import (
    circular_phase_alignment,
    circular_winding_degree,
    fisher_rao_distance_matrix,
    persistence_diagrams,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-deck-action-descrambler.v1"
HYPOTHESIS_ID = "tinyllm-deck-action-carrier-cover-v1"
SOURCE_SCHEMA = "nal.tinyllm-degree-k-ladder.v1"
REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class DeckDescramblerConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    degrees: tuple[int, ...] = (1, 2, 3)
    fit_orbits: int = 512
    evaluation_orbits: int = 256
    map_points: int = 192
    first_blocks: int = 3
    ridge: float = 1e-3
    activation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if set(self.degrees).difference((1, 2, 3)):
            raise ValueError("degrees must be drawn from 1,2,3")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory descrambler requires five seeds")
        if self.fit_orbits < 4 or self.evaluation_orbits < 4:
            raise ValueError("orbit cohorts are too small")
        if self.map_points < 24:
            raise ValueError("map grid is too small")
        if self.first_blocks < 1:
            raise ValueError("at least one block must be inspected")


@dataclass
class OrbitDataset:
    input_ids: torch.Tensor
    sensor: torch.Tensor
    calibration: torch.Tensor
    phase: torch.Tensor
    quotient_phase: torch.Tensor
    branch: torch.Tensor
    target_posteriors: torch.Tensor
    target_bins: torch.Tensor
    orbit_count: int
    k: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_digest() -> str:
    result = hashlib.sha256()
    for path in (Path(__file__), Path(ladder.__file__), Path(io_source.__file__)):
        result.update(str(path).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def cut_names(first_blocks: int) -> tuple[str, ...]:
    names = ["frontend", "block_0_pre_attention"]
    for index in range(first_blocks):
        names.extend((f"block_{index}_post_attention", f"block_{index}_post_mlp"))
    names.append("full")
    return tuple(dict.fromkeys(names))


def _repeat_orbits(values: Mapping[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
    return {name: np.repeat(value, k, axis=0) for name, value in values.items()}


def generate_exact_orbits(
    task: CircleTaskConfig,
    *,
    k: int,
    orbit_count: int,
    seed: int,
    regime: str,
    quotient_phases: Optional[np.ndarray] = None,
    fixed_nuisance: bool = False,
) -> OrbitDataset:
    """Generate complete deck fibers with identical nuisance/noise within an orbit."""
    generator = np.random.default_rng(seed)
    theta = (
        generator.uniform(0.0, 2.0 * math.pi, orbit_count)
        if quotient_phases is None
        else np.asarray(quotient_phases, dtype=np.float64)
    )
    orbit_count = len(theta)
    branch = np.tile(np.arange(k, dtype=np.int64), orbit_count)
    future = np.stack(
        [(theta + 2.0 * math.pi * item) / k for item in range(k)], axis=1
    ).reshape(-1)
    theta_flat = np.repeat(theta, k)
    if fixed_nuisance:
        one = _nuisance_values("N3", regime, 1, generator)
        base_values = {name: np.repeat(value, orbit_count, axis=0) for name, value in one.items()}
        base_directions = np.ones(orbit_count, dtype=np.float64)
    else:
        base_values = _nuisance_values("N3", regime, orbit_count, generator)
        base_directions = _sample_directions("N3", regime, orbit_count, generator)
    values = _repeat_orbits(base_values, k)
    directions = np.repeat(base_directions, k)
    current = np.remainder(future - directions * task.future_delta, 2.0 * math.pi)

    history = np.arange(task.sensor_steps, dtype=np.float64) - (task.sensor_steps - 1)
    angles = current[:, None] + directions[:, None] * history[None, :] * values["speed"]
    base_x, base_y = np.cos(angles), np.sin(angles)
    cosine, sine = np.cos(values["orientation"]), np.sin(values["orientation"])
    rotated_x = cosine * base_x - sine * base_y
    rotated_y = sine * base_x + cosine * base_y
    harmonic = values["harmonic_strength"] * np.cos(
        values["harmonic_order"] * angles + values["harmonic_phase"]
    )
    sensor_values = values["amplitude"] * np.stack((rotated_x, rotated_y, harmonic), axis=-1)
    sensor_values += values["offset"]
    normalized_history = history / max(1.0, float(task.sensor_steps - 1))
    sensor_values += values["drift"] * normalized_history[None, :, None]
    orbit_noise = generator.normal(0.0, 1.0, (orbit_count, task.sensor_steps, 3))
    sensor_values += np.repeat(orbit_noise, k, axis=0) * values["noise_scale"]
    input_ids = torch.from_numpy(io_source._serialize_sensor_values(sensor_values, task))
    sensor = io_source.decode_sensor_tokens(input_ids, task)
    calibration_packet = calibrated.calibration_packet(
        values, torch.tensor(directions, dtype=torch.float32)
    )
    posterior, bins = ladder._targets(future, k, task.phase_bins)
    return OrbitDataset(
        input_ids=input_ids,
        sensor=sensor,
        calibration=calibration_packet,
        phase=torch.tensor(future, dtype=torch.float32),
        quotient_phase=torch.tensor(theta_flat, dtype=torch.float32),
        branch=torch.tensor(branch, dtype=torch.long),
        target_posteriors=posterior,
        target_bins=bins,
        orbit_count=orbit_count,
        k=k,
    )


def _source_paths(source_root: Path, k: int, seed: int) -> tuple[Path, Path, Path]:
    root = source_root / "runs" / f"k{k}" / f"seed_{seed}"
    return root / "result.json", root / "model.pt", root / "frontend.pt"


def load_source(
    task: CircleTaskConfig,
    config: DeckDescramblerConfig,
    k: int,
    seed: int,
    device: torch.device,
) -> tuple[ladder.DegreeKSystem, Dict[str, Any]]:
    detail_path, checkpoint, frontend_path = _source_paths(Path(config.source_root), k, seed)
    detail = json.loads(detail_path.read_text())
    if (
        detail.get("schema_version") != SOURCE_SCHEMA
        or detail.get("status") != "completed"
        or int(detail.get("k", -1)) != k
        or int(detail.get("seed", -1)) != seed
        or not checkpoint.is_file()
        or not frontend_path.is_file()
    ):
        raise ValueError(f"invalid degree-ladder source {detail_path}")
    model = TinyLLMModel.from_checkpoint(checkpoint, map_location="cpu")
    system = ladder.DegreeKSystem(model, task)
    frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
    system.vector_embedding.load_state_dict(frontend["vector_embedding"])
    if _state_digest(model) != detail["training"]["final_model_state_sha256"]:
        raise ValueError(f"source model digest mismatch {detail_path}")
    if len(model.transformer["h"]) < config.first_blocks:
        raise ValueError("source model has fewer blocks than requested cuts")
    system.to(device).eval()
    return system, {
        "source_result": str(detail_path),
        "source_result_sha256": _sha256(detail_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "frontend_checkpoint": str(frontend_path),
        "frontend_checkpoint_sha256": _sha256(frontend_path),
        "source_model_state_sha256": detail["training"]["final_model_state_sha256"],
    }


def _initial_sequence(
    system: ladder.DegreeKSystem,
    input_ids: torch.Tensor,
    carrier: torch.Tensor,
) -> torch.Tensor:
    model = system.model
    bos = model.transformer["wte"](input_ids[:, :1])
    query = model.transformer["wte"](input_ids[:, -1:])
    value = torch.cat((bos, system.vector_embedding(carrier)[:, None, :], query), dim=1)
    positions = torch.arange(value.shape[1], device=value.device)
    return value + model.transformer["wpe"](positions)


@torch.no_grad()
def capture_sequences(
    system: ladder.DegreeKSystem,
    dataset: OrbitDataset,
    config: DeckDescramblerConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    names = cut_names(config.first_blocks)
    captured: Dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for start in range(0, len(dataset.phase), config.activation_batch_size):
        stop = start + config.activation_batch_size
        inputs = dataset.input_ids[start:stop].to(device)
        sensor = dataset.sensor[start:stop].to(device)
        calibration_packet = dataset.calibration[start:stop].to(device)
        carrier = system.carrier(sensor, calibration_packet)
        captured["frontend"].append(carrier.cpu())
        value = _initial_sequence(system, inputs, carrier)
        captured["block_0_pre_attention"].append(value.cpu())
        for index, block in enumerate(system.model.transformer["h"]):
            attended = value + block.attn(block.ln_1(value))
            if index < config.first_blocks:
                captured[f"block_{index}_post_attention"].append(attended.cpu())
            value = attended + block.mlp(block.ln_2(attended))
            if index < config.first_blocks:
                captured[f"block_{index}_post_mlp"].append(value.cpu())
        captured["full"].append(value.cpu())
    return {name: torch.cat(parts).float().numpy() for name, parts in captured.items()}


def query_features(captured: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        name: value if value.ndim == 2 else value[:, -1, :]
        for name, value in captured.items()
    }


def deck_shift(values: np.ndarray, orbit_count: int, k: int) -> np.ndarray:
    shaped = values.reshape((orbit_count, k) + values.shape[1:])
    return np.roll(shaped, -1, axis=1).reshape(values.shape)


def orthogonal_procrustes_action(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cross = source.astype(np.float64).T @ target.astype(np.float64)
    left, _, right = np.linalg.svd(cross, full_matrices=False)
    return left @ right


def action_metrics(source: np.ndarray, target: np.ndarray, action: np.ndarray, k: int) -> Dict[str, float]:
    predicted = source @ action
    error = predicted - target
    target_norm = max(1e-12, float(np.linalg.norm(target)))
    identity_sse = float(np.square(source - target).sum())
    action_sse = float(np.square(error).sum())
    paired_cosine = np.sum(predicted * target, axis=1) / np.maximum(
        1e-12, np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)
    )
    closure = np.linalg.matrix_power(action, k) - np.eye(action.shape[0])
    return {
        "target_normalized_transport_error": float(np.linalg.norm(error) / target_norm),
        "mean_paired_cosine": float(np.mean(paired_cosine)),
        "identity_error_removed_fraction": float(
            1.0 - action_sse / identity_sse if identity_sse > 1e-12 else 1.0
        ),
        "activation_weighted_closure_error": float(
            np.linalg.norm(source @ closure) / max(1e-12, np.linalg.norm(source))
        ),
        "full_matrix_closure_error": float(
            np.linalg.norm(closure) / math.sqrt(action.shape[0])
        ),
    }


def canonical_decomposition(action: np.ndarray, k: int) -> tuple[np.ndarray, Dict[str, Any]]:
    schur_form, basis = schur(action, output="real")
    eigenvalues = np.linalg.eigvals(schur_form)
    roots = np.exp(2j * math.pi * np.arange(k) / k)
    assignment = np.argmin(np.abs(eigenvalues[:, None] - roots[None, :]), axis=1)
    projector = sum(np.linalg.matrix_power(action, power) for power in range(k)) / k
    counts = {str(index): int(np.sum(assignment == index)) for index in range(k)}
    return projector, {
        "root_multiplicities": counts,
        "invariant_dimension": counts["0"],
        "nontrivial_dimension": int(action.shape[0] - counts["0"]),
        "maximum_root_distance": float(np.max(np.min(np.abs(eigenvalues[:, None] - roots[None, :]), axis=1))),
        "projector_idempotence_error": float(
            np.linalg.norm(projector @ projector - projector) / math.sqrt(action.shape[0])
        ),
        "schur_orthogonality_error": float(
            np.linalg.norm(basis.T @ basis - np.eye(action.shape[0])) / math.sqrt(action.shape[0])
        ),
    }


def component_energy(values: np.ndarray, projector: np.ndarray) -> Dict[str, float]:
    invariant = values @ projector
    branch = values - invariant
    total = max(1e-12, float(np.square(values).sum()))
    return {
        "invariant_fraction": float(np.square(invariant).sum() / total),
        "branch_fraction": float(np.square(branch).sum() / total),
        "cross_fraction": float(2.0 * np.sum(invariant * branch) / total),
    }


def _ridge_fit(x: np.ndarray, targets: np.ndarray, ridge_value: float) -> Dict[str, np.ndarray]:
    mean = x.mean(0, dtype=np.float64)
    scale = np.maximum(x.std(0, dtype=np.float64), 1e-5)
    normalized = (x - mean) / scale
    design = np.concatenate((normalized, np.ones((len(x), 1))), axis=1)
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0]) * ridge_value
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(gram + regularizer, design.T @ targets)
    return {"mean": mean, "scale": scale, "weights": weights}


def _ridge_predict(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    normalized = (x - model["mean"]) / model["scale"]
    design = np.concatenate((normalized, np.ones((len(x), 1))), axis=1)
    return design @ model["weights"]


def _balanced_accuracy(predicted: np.ndarray, labels: np.ndarray, k: int) -> float:
    return statistics.fmean(float(np.mean(predicted[labels == item] == item)) for item in range(k))


def fit_component_probes(
    train: np.ndarray,
    evaluations: Mapping[str, np.ndarray],
    projector: np.ndarray,
    train_dataset: OrbitDataset,
    evaluation_datasets: Mapping[str, OrbitDataset],
    config: DeckDescramblerConfig,
) -> tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    components = {
        "full": (train, evaluations),
        "invariant": (train @ projector, {name: value @ projector for name, value in evaluations.items()}),
        "branch": (train - train @ projector, {name: value - value @ projector for name, value in evaluations.items()}),
    }
    result: Dict[str, Any] = {}
    frozen_full_branch: Optional[Dict[str, np.ndarray]] = None
    theta_train = train_dataset.quotient_phase.numpy()
    task_target = np.column_stack((np.cos(theta_train), np.sin(theta_train)))
    condition_train = task_target
    labels_train = train_dataset.branch.numpy()
    for name, (train_x, eval_x) in components.items():
        task_model = _ridge_fit(train_x, task_target, config.ridge)
        task_metrics = {}
        for regime, values in eval_x.items():
            predicted = _ridge_predict(task_model, values)
            theta = evaluation_datasets[regime].quotient_phase.numpy()
            target_angles = np.remainder(theta, 2.0 * math.pi)
            predicted_angles = np.angle(predicted[:, 0] + 1j * predicted[:, 1]) % (2.0 * math.pi)
            task_metrics[regime] = {
                "circular_alignment": float(circular_phase_alignment(predicted_angles, target_angles)["alignment"]),
                "vector_mse": float(np.mean(np.square(predicted - np.column_stack((np.cos(theta), np.sin(theta)))))),
            }
        item: Dict[str, Any] = {"task": task_metrics}
        if train_dataset.k > 1:
            branch_train = np.concatenate((train_x, condition_train), axis=1)
            one_hot = np.eye(train_dataset.k)[labels_train]
            branch_model = _ridge_fit(branch_train, one_hot, config.ridge)
            branch_metrics = {}
            for regime, values in eval_x.items():
                theta = evaluation_datasets[regime].quotient_phase.numpy()
                conditioned = np.concatenate((values, np.column_stack((np.cos(theta), np.sin(theta)))), axis=1)
                predicted = _ridge_predict(branch_model, conditioned).argmax(1)
                labels = evaluation_datasets[regime].branch.numpy()
                branch_metrics[regime] = {
                    "balanced_accuracy": _balanced_accuracy(predicted, labels, train_dataset.k),
                    "chance_accuracy": 1.0 / train_dataset.k,
                }
            item["branch"] = branch_metrics
            if name == "full":
                frozen_full_branch = branch_model
        result[name] = item
    return result, frozen_full_branch


@torch.no_grad()
def continue_from_cut(
    system: ladder.DegreeKSystem,
    dataset: OrbitDataset,
    cut: str,
    patched: np.ndarray,
    config: DeckDescramblerConfig,
    device: torch.device,
) -> np.ndarray:
    posteriors = []
    answer = torch.tensor(CircleTaskConfig().answer_token_ids, device=device)
    for start in range(0, len(dataset.phase), config.activation_batch_size):
        stop = start + config.activation_batch_size
        patch = torch.from_numpy(patched[start:stop]).to(device=device, dtype=torch.float32)
        if cut == "frontend":
            value = _initial_sequence(system, dataset.input_ids[start:stop].to(device), patch)
            next_block = 0
        elif cut == "block_0_pre_attention":
            value, next_block = patch, 0
        elif cut == "full":
            value, next_block = patch, len(system.model.transformer["h"])
        else:
            pieces = cut.split("_")
            block_index = int(pieces[1])
            block = system.model.transformer["h"][block_index]
            if cut.endswith("post_attention"):
                value = patch + block.mlp(block.ln_2(patch))
            else:
                value = patch
            next_block = block_index + 1
        for block in system.model.transformer["h"][next_block:]:
            value = block(value)
        logits = io_source._task_logits(system.model, value[:, -1, :], answer)
        posteriors.append(torch.softmax(logits, -1).cpu())
    return torch.cat(posteriors).double().numpy()


def orbit_average(values: np.ndarray, orbit_count: int, k: int) -> np.ndarray:
    shaped = values.reshape((orbit_count, k) + values.shape[1:])
    averaged = shaped.mean(axis=1, keepdims=True)
    return np.repeat(averaged, k, axis=1).reshape(values.shape)


def intervention_arrays(
    values: np.ndarray,
    projector_rank: int,
    dataset: OrbitDataset,
    seed: int,
) -> Dict[str, np.ndarray]:
    exact = orbit_average(values, dataset.orbit_count, dataset.k)
    if dataset.k == 1:
        return {"original": values, "orbit_average": exact}
    generator = np.random.default_rng(seed)
    dimension = values.shape[-1]
    rank = min(max(1, projector_rank), dimension)
    random_basis, _ = np.linalg.qr(generator.normal(size=(dimension, rank)))
    random_projector = random_basis @ random_basis.T
    projected = values @ random_projector

    flat_count = len(values)
    random_parts = [values]
    for _ in range(1, dataset.k):
        random_parts.append(values[generator.permutation(flat_count)])
    random_pairing = np.mean(random_parts, axis=0)

    shaped = values.reshape((dataset.orbit_count, dataset.k) + values.shape[1:])
    phase_parts = []
    for shift in range(dataset.k):
        phase_parts.append(np.roll(np.roll(shaped, shift, axis=0), -shift, axis=1))
    phase_shuffled = np.mean(phase_parts, axis=0).reshape(values.shape)

    difference = exact - values
    noise = generator.normal(size=values.shape)
    axes = tuple(range(1, values.ndim))
    noise_norm = np.sqrt(np.sum(np.square(noise), axis=axes, keepdims=True)).clip(1e-12)
    difference_norm = np.sqrt(np.sum(np.square(difference), axis=axes, keepdims=True))
    equal_norm = values + noise / noise_norm * difference_norm
    return {
        "original": values,
        "orbit_average": exact,
        "random_same_rank_projection": projected,
        "random_orbit_pairing": random_pairing,
        "phase_shuffled_average": phase_shuffled,
        "equal_norm_ablation": equal_norm,
    }


def output_diagnostics(
    posterior: np.ndarray,
    dataset: OrbitDataset,
    *,
    include_topology: bool,
) -> Dict[str, Any]:
    angles = 2.0 * math.pi * np.arange(posterior.shape[1]) / posterior.shape[1]
    moment = posterior @ np.exp(1j * angles)
    order = np.argsort(dataset.phase.numpy())
    predicted = np.angle(moment[order]) % (2.0 * math.pi)
    target = np.remainder(dataset.k * dataset.phase.numpy()[order], 2.0 * math.pi)
    closed = np.concatenate((moment[order], moment[order][:1]))
    increments = np.angle(closed[1:] * np.conj(closed[:-1]))
    result: Dict[str, Any] = {
        "circular_alignment": float(circular_phase_alignment(predicted, target)["alignment"]),
        "winding_degree": float(circular_winding_degree(predicted)),
        "minimum_moment_magnitude": float(np.abs(moment).min()),
        "maximum_angular_increment": float(np.abs(increments).max()),
        "sampling_resolved": bool(np.abs(increments).max() < math.pi / 2.0),
        "exact_bin_accuracy": float(np.mean(posterior.argmax(1) == dataset.target_bins.numpy())),
    }
    if include_topology:
        distances = fisher_rao_distance_matrix(posterior[order])
        diagram = persistence_diagrams(distances)[1]
        lifetime = float(np.max(diagram[:, 1] - diagram[:, 0])) if len(diagram) else 0.0
        result["posterior_normalized_h1_lifetime"] = lifetime / max(1e-12, float(np.max(distances)))
    return result


def _classify_causal(patched: Mapping[str, Any], baseline: Mapping[str, Any], k: int) -> str:
    loss = float(baseline["exact_bin_accuracy"] - patched["exact_bin_accuracy"])
    preserves = (
        patched["circular_alignment"] >= 0.90
        and patched["sampling_resolved"]
        and abs(patched["winding_degree"] - k) <= 0.10
        and loss <= 0.03
    )
    destroys = (
        patched["circular_alignment"] < 0.50
        or (patched["sampling_resolved"] and abs(patched["winding_degree"] - k) > 0.10)
        or loss > 0.20
    )
    return "preserved" if preserves else "destroyed" if destroys else "partial_or_unresolved"


def analyze_cell(
    task: CircleTaskConfig,
    config: DeckDescramblerConfig,
    k: int,
    seed: int,
    output: Path,
    device: torch.device,
) -> Dict[str, Any]:
    started = time.perf_counter()
    system, provenance = load_source(task, config, k, seed, device)
    cohorts = {
        "train": generate_exact_orbits(task, k=k, orbit_count=config.fit_orbits, seed=seed + 101, regime="interpolation"),
        "composition": generate_exact_orbits(task, k=k, orbit_count=config.evaluation_orbits, seed=seed + 211, regime="composition"),
        "extrapolation": generate_exact_orbits(task, k=k, orbit_count=config.evaluation_orbits, seed=seed + 307, regime="extrapolation"),
    }
    captures = {name: capture_sequences(system, data, config, device) for name, data in cohorts.items()}
    features = {name: query_features(value) for name, value in captures.items()}
    map_datasets = {}
    map_captures = {}
    map_orbits = config.map_points // k
    theta_grid = np.linspace(0.0, 2.0 * math.pi, map_orbits, endpoint=False)
    for regime in REGIMES:
        map_datasets[regime] = generate_exact_orbits(
            task, k=k, orbit_count=map_orbits, seed=seed + 10_003,
            regime=regime, quotient_phases=theta_grid, fixed_nuisance=True,
        )
        map_captures[regime] = capture_sequences(
            system, map_datasets[regime], config, device
        )
    cuts: Dict[str, Any] = {}
    action_arrays: Dict[str, np.ndarray] = {}
    for cut_index, cut in enumerate(cut_names(config.first_blocks)):
        train_x = features["train"][cut]
        train_y = deck_shift(train_x, cohorts["train"].orbit_count, k)
        action = orthogonal_procrustes_action(train_x, train_y)
        projector, canonical = canonical_decomposition(action, k)
        action_arrays[f"{cut}__action"] = action.astype(np.float32)
        action_arrays[f"{cut}__projector"] = projector.astype(np.float32)
        evaluations = {name: features[name][cut] for name in REGIMES}
        probes, branch_probe = fit_component_probes(
            train_x, evaluations, projector, cohorts["train"],
            {name: cohorts[name] for name in REGIMES}, config,
        )
        transport = {}
        energies = {}
        causal = {}
        numerical_rank = int(np.sum(np.linalg.svd(projector, compute_uv=False) > 0.5))
        for regime in REGIMES:
            x = features[regime][cut]
            y = deck_shift(x, cohorts[regime].orbit_count, k)
            transport[regime] = action_metrics(x, y, action, k)
            energies[regime] = component_energy(x, projector)
            map_data = map_datasets[regime]
            map_capture = map_captures[regime][cut]
            task_capture = captures[regime][cut]
            map_interventions = intervention_arrays(map_capture, numerical_rank, map_data, seed + cut_index * 1009)
            task_interventions = intervention_arrays(task_capture, numerical_rank, cohorts[regime], seed + cut_index * 1013)
            condition_result = {}
            for condition in map_interventions:
                map_posterior = continue_from_cut(system, map_data, cut, map_interventions[condition], config, device)
                task_posterior = continue_from_cut(system, cohorts[regime], cut, task_interventions[condition], config, device)
                map_metrics = output_diagnostics(map_posterior, map_data, include_topology=True)
                task_accuracy = float(np.mean(task_posterior.argmax(1) == cohorts[regime].target_bins.numpy()))
                branch_accuracy = None
                if branch_probe is not None:
                    patch_query = (
                        task_interventions[condition]
                        if task_interventions[condition].ndim == 2
                        else task_interventions[condition][:, -1, :]
                    )
                    theta_values = cohorts[regime].quotient_phase.numpy()
                    conditioned = np.concatenate((patch_query, np.column_stack((np.cos(theta_values), np.sin(theta_values)))), axis=1)
                    labels = cohorts[regime].branch.numpy()
                    branch_accuracy = _balanced_accuracy(
                        _ridge_predict(branch_probe, conditioned).argmax(1), labels, k
                    )
                condition_result[condition] = {
                    **map_metrics,
                    "exact_bin_accuracy": task_accuracy,
                    "patched_branch_balanced_accuracy": branch_accuracy,
                }
            baseline = condition_result["original"]
            condition_result["orbit_average"]["causal_classification"] = _classify_causal(
                condition_result["orbit_average"], baseline, k
            )
            causal[regime] = condition_result
        gates = {}
        if k > 1:
            chance = 1.0 / k
            for regime in REGIMES:
                item = transport[regime]
                full_accuracy = probes["full"]["branch"][regime]["balanced_accuracy"]
                inv_accuracy = probes["invariant"]["branch"][regime]["balanced_accuracy"]
                branch_accuracy = probes["branch"]["branch"][regime]["balanced_accuracy"]
                gates[regime] = {
                    "heldout_deck_transport": bool(
                        item["target_normalized_transport_error"] <= 0.15
                        and item["mean_paired_cosine"] >= 0.95
                        and item["identity_error_removed_fraction"] >= 0.50
                    ),
                    "group_closure": bool(item["activation_weighted_closure_error"] <= 0.10),
                    "canonical_branch_localization": bool(
                        branch_accuracy >= chance + 0.20 and branch_accuracy >= full_accuracy - 0.05
                    ),
                    "invariant_branch_control": bool(inv_accuracy <= chance + 0.05),
                    "causal_classified": causal[regime]["orbit_average"]["causal_classification"] in {"preserved", "destroyed"},
                }
                gates[regime]["deck_geometry_pass"] = all(
                    gates[regime][name]
                    for name in (
                        "heldout_deck_transport", "group_closure",
                        "canonical_branch_localization", "invariant_branch_control",
                    )
                )
        cuts[cut] = {
            "canonical": canonical,
            "projector_numerical_rank": numerical_rank,
            "transport": transport,
            "energy": energies,
            "component_probes": probes,
            "causal": causal,
            "gates": gates,
        }
    baseline_replay = {}
    for regime in REGIMES:
        baselines = [cuts[cut]["causal"][regime]["original"] for cut in cut_names(config.first_blocks)]
        numeric_keys = (
            "circular_alignment", "winding_degree", "minimum_moment_magnitude",
            "maximum_angular_increment", "exact_bin_accuracy",
            "posterior_normalized_h1_lifetime",
        )
        spreads = {
            key: float(max(item[key] for item in baselines) - min(item[key] for item in baselines))
            for key in numeric_keys
        }
        baseline_replay[regime] = {
            "metric_spreads_across_cuts": spreads,
            "passed": max(spreads.values()) <= 1e-5,
        }
    root = output / "runs" / f"k{k}" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "deck_actions.npz", **action_arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-deck-action-k{k}-seed{seed}",
        "status": "completed",
        "completed_at": _utc_now(),
        "k": k,
        "seed": seed,
        "configuration": asdict(config),
        "provenance": provenance,
        "cuts": cuts,
        "baseline_replay_integrity": baseline_replay,
        "actions_path": str(root / "deck_actions.npz"),
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(root / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: DeckDescramblerConfig) -> Dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    degrees: Dict[str, Any] = {}
    for k in config.degrees:
        selected = [run for run in runs if int(run["k"]) == k]
        cut_records = {}
        for cut in cut_names(config.first_blocks):
            if k == 1:
                unchanged = {
                    regime: sum(
                        run["cuts"][cut]["causal"][regime]["orbit_average"]["causal_classification"] == "preserved"
                        for run in selected
                    )
                    for regime in REGIMES
                }
                cut_records[cut] = {"trivial_orbit_preserved_counts": unchanged}
                continue
            per_seed = {}
            for run in selected:
                classifications = {
                    regime: run["cuts"][cut]["causal"][regime]["orbit_average"]["causal_classification"]
                    for regime in REGIMES
                }
                deck_pass = all(run["cuts"][cut]["gates"][regime]["deck_geometry_pass"] for regime in REGIMES)
                causal_consistent = (
                    classifications["composition"] == classifications["extrapolation"]
                    and classifications["composition"] in {"preserved", "destroyed"}
                )
                per_seed[str(run["seed"])] = {
                    "deck_geometry_pass": deck_pass,
                    "causal_classifications": classifications,
                    "causal_consistent": causal_consistent,
                }
            deck_count = sum(item["deck_geometry_pass"] for item in per_seed.values())
            preserved = sum(
                item["causal_consistent"] and item["causal_classifications"]["composition"] == "preserved"
                for item in per_seed.values()
            )
            destroyed = sum(
                item["causal_consistent"] and item["causal_classifications"]["composition"] == "destroyed"
                for item in per_seed.values()
            )
            causal_kind = "redundant_co_representation" if preserved >= required else "computational_cover" if destroyed >= required else "heterogeneous_or_unresolved"
            cut_records[cut] = {
                "per_seed": per_seed,
                "deck_geometry_pass_count": deck_count,
                "deck_geometry_stable": deck_count >= required,
                "causal_preserved_count": preserved,
                "causal_destroyed_count": destroyed,
                "causal_mechanism": causal_kind,
                "canonical_dimension_mean": {
                    "invariant": statistics.fmean(run["cuts"][cut]["canonical"]["invariant_dimension"] for run in selected),
                    "nontrivial": statistics.fmean(run["cuts"][cut]["canonical"]["nontrivial_dimension"] for run in selected),
                },
                "energy_means": {
                    regime: {
                        component: statistics.fmean(run["cuts"][cut]["energy"][regime][component] for run in selected)
                        for component in ("invariant_fraction", "branch_fraction")
                    }
                    for regime in REGIMES
                },
            }
        stable = [cut for cut, value in cut_records.items() if value.get("deck_geometry_stable")]
        classified = [
            cut for cut, value in cut_records.items()
            if value.get("causal_mechanism") in {"redundant_co_representation", "computational_cover"}
        ]
        degrees[str(k)] = {
            "cuts": cut_records,
            "stable_deck_cuts": stable,
            "causally_classified_cuts": classified,
            "success": bool(stable and classified) if k > 1 else True,
        }
    return {
        "degrees": degrees,
        "required_seed_count": required,
        "confirmed": all(degrees[str(k)]["success"] for k in config.degrees if k > 1),
    }


def run_campaign(config: DeckDescramblerConfig, output: Path) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    runs = []
    for k in config.degrees:
        for seed in config.seeds:
            path = output / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            if path.is_file():
                existing = json.loads(path.read_text())
                if existing.get("implementation_sha256") == _implementation_digest():
                    runs.append(existing)
                    print(f"resuming {existing['experiment_id']}", flush=True)
                    continue
            result = analyze_cell(task, config, k, seed, output, device)
            runs.append(result)
            print(result["experiment_id"], f"{result['analysis_seconds']:.1f}s", flush=True)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": _implementation_digest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "summary": {
            "requested": len(config.degrees) * len(config.seeds),
            "completed": len(runs),
            "failed": len(config.degrees) * len(config.seeds) - len(runs),
        },
        "aggregates": aggregate(runs, config),
        "method_boundaries": [
            "Orthogonal descrambling preserves information and cannot rescue the failed quotient claim.",
            "The projector is the finite group average of an approximate Procrustes action and need not be exactly idempotent.",
            "Component decodability uses one fixed conditioned linear ridge probe family.",
            "Schur coordinates inside repeated isotypic blocks are non-identifiable across seeds.",
        ],
    }
    _write_json(output / "campaign_results.json", bundle)
    return bundle


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DeckDescramblerConfig.source_root)
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--degrees", default="1,2,3")
    parser.add_argument("--fit-orbits", type=int, default=512)
    parser.add_argument("--evaluation-orbits", type=int, default=256)
    parser.add_argument("--map-points", type=int, default=192)
    parser.add_argument("--first-blocks", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered"))
    args = parser.parse_args(argv)
    if args.shakedown:
        args.seeds, args.degrees = "7", "1,2"
        args.fit_orbits, args.evaluation_orbits, args.map_points = 12, 8, 24
        args.first_blocks, args.allow_underpowered = 1, True
    config = DeckDescramblerConfig(
        source_root=args.source_root,
        seeds=_ints(args.seeds),
        degrees=_ints(args.degrees),
        fit_orbits=args.fit_orbits,
        evaluation_orbits=args.evaluation_orbits,
        map_points=args.map_points,
        first_blocks=args.first_blocks,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    bundle = run_campaign(config, args.output)
    print(json.dumps(bundle["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
