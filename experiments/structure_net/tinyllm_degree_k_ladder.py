#!/usr/bin/env python3
"""Train matched TinyLLM maps of degree k=1,2,3 on an analytic phase carrier."""

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
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
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
    complex_defect_charge,
    fisher_rao_distance_matrix,
    persistence_diagrams,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-degree-k-ladder.v1"
HYPOTHESIS_ID = "tinyllm-degree-k-finite-quotient-ladder-v1"
KS = (1, 2, 3)
REGIMES = ("composition", "extrapolation")
CUTS = ("frontend", "post_mlp", "full")


@dataclass(frozen=True)
class DegreeKLadderConfig:
    preset: str = "d6"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    degrees: tuple[int, ...] = KS
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    phase_grid_points: int = 192
    trace_phase_points: int = 96
    transition_phase_points: int = 256
    transition_path_points: int = 66
    probe_train_samples: int = 2_048
    probe_validation_samples: int = 512
    probe_test_samples: int = 1_024
    probe_steps: int = 240
    probe_width: int = 128
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if set(self.degrees).difference(KS):
            raise ValueError("degrees must be drawn from 1,2,3")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory ladder requires five seeds")
        if self.preset != "d6" and not self.allow_underpowered:
            raise ValueError("confirmatory ladder fixes d6")
        if min(self.training_steps, self.train_samples, self.batch_size) < 1:
            raise ValueError("training sizes must be positive")


@dataclass
class LadderDataset:
    input_ids: torch.Tensor
    sensor: torch.Tensor
    calibration: torch.Tensor
    phase: torch.Tensor
    target_posteriors: torch.Tensor
    target_bins: torch.Tensor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(*values: torch.Tensor) -> str:
    result = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _implementation_digest() -> str:
    result = hashlib.sha256()
    for path in (Path(__file__), Path(calibrated.__file__), Path(io_source.__file__)):
        result.update(str(path).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


class AnalyticPhaseCarrier:
    """Recover the absolute future phase vector from observation plus calibration."""

    def __init__(self, task: CircleTaskConfig):
        history = torch.arange(task.sensor_steps, dtype=torch.float32)
        self.history = history / max(1.0, float(task.sensor_steps - 1)) - 1.0
        self.future_delta = task.future_delta

    def __call__(self, sensor: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
        history = self.history.to(sensor.device)
        orientation = calibration[:, :2]
        signed_speed = calibration[:, 2]
        amplitude = calibration[:, 3:4]
        offset = calibration[:, 4:6]
        drift = calibration[:, 6:8]
        corrected = (
            sensor[..., :2] - offset[:, None, :] - drift[:, None, :] * history[None, :, None]
        ) / amplitude[:, None, :]
        endpoint = corrected[:, -1, :]
        lab_x = (endpoint * orientation).sum(-1)
        lab_y = orientation[:, 0] * endpoint[:, 1] - orientation[:, 1] * endpoint[:, 0]
        norm = torch.sqrt(lab_x.square() + lab_y.square()).clamp_min(1e-6)
        lab_x, lab_y = lab_x / norm, lab_y / norm
        delta = signed_speed.sign() * self.future_delta
        cosine, sine = torch.cos(delta), torch.sin(delta)
        return torch.stack((lab_x * cosine - lab_y * sine, lab_x * sine + lab_y * cosine), -1)


class DegreeKSystem(nn.Module):
    def __init__(self, model: TinyLLMModel, task: CircleTaskConfig):
        super().__init__()
        self.model = model
        self.carrier = AnalyticPhaseCarrier(task)
        self.vector_embedding = nn.Linear(2, model.config.n_embd)

    def forward_cuts(self, input_ids: torch.Tensor, sensor: torch.Tensor, calibration_packet: torch.Tensor) -> Dict[str, torch.Tensor]:
        carrier = self.carrier(sensor, calibration_packet)
        bos = self.model.transformer["wte"](input_ids[:, :1])
        query = self.model.transformer["wte"](input_ids[:, -1:])
        value = torch.cat((bos, self.vector_embedding(carrier)[:, None, :], query), dim=1)
        cuts = io_source._transformer_cuts(self.model, value, carrier)
        return {"frontend": cuts["frontend"], "post_mlp": cuts["post_mlp"], "full": cuts["full"]}


def _targets(phase: np.ndarray, k: int, bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    mapped = np.remainder(k * phase, 2.0 * math.pi)
    centers = 2.0 * math.pi * np.arange(bins) / bins
    delta = np.angle(np.exp(1j * (mapped[:, None] - centers[None, :])))
    width = 2.0 * math.pi / bins
    probabilities = np.exp(-0.5 * np.square(delta / width))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    posterior = torch.tensor(probabilities, dtype=torch.float32)
    return posterior, posterior.argmax(1)


def generate_dataset(
    task: CircleTaskConfig,
    *,
    k: int,
    count: int,
    seed: int,
    regime: str = "interpolation",
    phases: Optional[np.ndarray] = None,
    fixed_nuisance: bool = False,
) -> LadderDataset:
    generator = np.random.default_rng(seed)
    future = generator.uniform(0.0, 2.0 * math.pi, count) if phases is None else np.asarray(phases)
    count = len(future)
    directions = _sample_directions("N3", regime, count, generator)
    nuisance_generator = np.random.default_rng(seed + 10_003)
    if fixed_nuisance:
        one = _nuisance_values("N3", regime, 1, nuisance_generator)
        values = {name: np.repeat(value, count, axis=0) for name, value in one.items()}
        directions[:] = 1.0
    else:
        values = _nuisance_values("N3", regime, count, nuisance_generator)
    current = np.remainder(future - directions * task.future_delta, 2.0 * math.pi)
    inputs = io_source._serialize_with_nuisance(task, current, directions, values, nuisance_generator)
    sensor = io_source.decode_sensor_tokens(inputs, task)
    calibration_packet = calibrated.calibration_packet(values, torch.tensor(directions, dtype=torch.float32))
    posterior, bins = _targets(future, k, task.phase_bins)
    return LadderDataset(inputs, sensor, calibration_packet, torch.tensor(future, dtype=torch.float32), posterior, bins)


@torch.no_grad()
def _features(system: DegreeKSystem, dataset: LadderDataset, device: torch.device, batch: int = 256) -> Dict[str, np.ndarray]:
    system.eval()
    values: Dict[str, List[torch.Tensor]] = {cut: [] for cut in CUTS}
    for start in range(0, len(dataset.phase), batch):
        stop = start + batch
        cuts = system.forward_cuts(
            dataset.input_ids[start:stop].to(device), dataset.sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        for cut in CUTS:
            values[cut].append(cuts[cut].detach().cpu())
    return {cut: torch.cat(parts).float().numpy() for cut, parts in values.items()}


@torch.no_grad()
def posterior_map(system: DegreeKSystem, dataset: LadderDataset, task: CircleTaskConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    features = _features(system, dataset, device)
    answer = torch.tensor(task.answer_token_ids, device=device)
    posteriors = []
    final = torch.from_numpy(features["full"])
    for start in range(0, len(final), 256):
        logits = io_source._task_logits(system.model, final[start:start + 256].to(device), answer)
        posteriors.append(torch.softmax(logits, -1).cpu())
    posterior = torch.cat(posteriors).double().numpy()
    angles = 2.0 * math.pi * np.arange(task.phase_bins) / task.phase_bins
    moment = posterior @ np.exp(1j * angles)
    return posterior, moment


def map_diagnostics(system: DegreeKSystem, task: CircleTaskConfig, config: DegreeKLadderConfig, k: int, seed: int, regime: str, device: torch.device, points: Optional[int] = None, *, include_topology: bool = True) -> Dict[str, Any]:
    count = points or config.phase_grid_points
    phases = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    dataset = generate_dataset(task, k=k, count=count, seed=seed, regime=regime, phases=phases, fixed_nuisance=True)
    posterior, moment = posterior_map(system, dataset, task, device)
    predicted = np.angle(moment) % (2.0 * math.pi)
    target = np.remainder(k * phases, 2.0 * math.pi)
    alignment = circular_phase_alignment(predicted, target)
    closed = np.concatenate((moment, moment[:1]))
    increments = np.angle(closed[1:] * np.conj(closed[:-1]))
    result = {
        "alignment": float(alignment["alignment"]),
        "orientation": alignment["orientation"],
        "winding_degree": float(circular_winding_degree(predicted)),
        "minimum_moment_magnitude": float(np.abs(moment).min()),
        "maximum_angular_increment": float(np.abs(increments).max()),
        "sampling_resolved": bool(np.abs(increments).max() < math.pi / 2.0),
        "phase_points": count,
    }
    if include_topology:
        distances = fisher_rao_distance_matrix(posterior)
        diagram = persistence_diagrams(distances)[1]
        lifetime = float(np.max(diagram[:, 1] - diagram[:, 0])) if len(diagram) else 0.0
        result["posterior_normalized_h1_lifetime"] = lifetime / max(
            1e-12, float(np.max(distances))
        )
    return result


def _branch_dataset(task: CircleTaskConfig, *, k: int, count: int, seed: int, regime: str) -> tuple[LadderDataset, np.ndarray, np.ndarray]:
    if count % k:
        count -= count % k
    generator = np.random.default_rng(seed)
    outputs = generator.uniform(0.0, 2.0 * math.pi, count // k)
    phase = np.stack([(outputs + 2.0 * math.pi * branch) / k for branch in range(k)], axis=1).reshape(-1)
    branch = np.tile(np.arange(k), len(outputs))
    condition = np.repeat(np.column_stack((np.cos(outputs), np.sin(outputs))), k, axis=0)
    dataset = generate_dataset(task, k=k, count=len(phase), seed=seed + 17, regime=regime, phases=phase)
    permutation = generator.permutation(len(phase))
    index = torch.tensor(permutation)
    dataset = LadderDataset(
        dataset.input_ids[index], dataset.sensor[index], dataset.calibration[index],
        dataset.phase[index], dataset.target_posteriors[index], dataset.target_bins[index],
    )
    return dataset, branch[permutation], condition[permutation]


def _balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor, classes: int) -> float:
    predicted = logits.argmax(-1)
    recalls = []
    for label in range(classes):
        selected = labels == label
        recalls.append(float((predicted[selected] == label).float().mean()))
    return statistics.fmean(recalls)


def fit_branch_probe(train_x: np.ndarray, validation_x: np.ndarray, evaluations: Mapping[str, np.ndarray], train_y: np.ndarray, validation_y: np.ndarray, evaluation_y: Mapping[str, np.ndarray], train_c: np.ndarray, validation_c: np.ndarray, evaluation_c: Mapping[str, np.ndarray], *, k: int, config: DegreeKLadderConfig, seed: int, device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(seed)
    mean = train_x.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train_x.std(0, dtype=np.float64).astype(np.float32), 1e-5)
    def matrix(x: np.ndarray, c: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.concatenate(((x - mean) / scale, c), axis=1), dtype=torch.float32, device=device)
    tx, vx = matrix(train_x, train_c), matrix(validation_x, validation_c)
    ty = torch.tensor(train_y, dtype=torch.long, device=device)
    vy = torch.tensor(validation_y, dtype=torch.long, device=device)
    model = nn.Sequential(nn.Linear(tx.shape[1], config.probe_width), nn.GELU(), nn.Linear(config.probe_width, k)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0
    for step in range(1, config.probe_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(tx), ty)
        loss.backward(); optimizer.step()
        if step % 20 == 0 or step == config.probe_steps:
            with torch.no_grad(): value = float(nn.functional.cross_entropy(model(vx), vy))
            if value < best - 1e-5:
                best, patience = value, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 5: break
    if best_state is not None: model.load_state_dict(best_state)
    result = {}
    model.eval()
    with torch.no_grad():
        for regime, values in evaluations.items():
            labels = torch.tensor(evaluation_y[regime], dtype=torch.long, device=device)
            logits = model(matrix(values, evaluation_c[regime]))
            result[regime] = {
                "balanced_accuracy": _balanced_accuracy(logits, labels, k),
                "cross_entropy": float(nn.functional.cross_entropy(logits, labels)),
                "chance_accuracy": 1.0 / k,
            }
    return {"evaluations": result, "validation_cross_entropy": best, "steps_run": step}


def analyze_branches(system: DegreeKSystem, task: CircleTaskConfig, config: DegreeKLadderConfig, k: int, seed: int, device: torch.device) -> Dict[str, Any]:
    datasets = {}
    labels = {}; conditions = {}; features = {}
    specs = {
        "train": (config.probe_train_samples, "interpolation", config.analysis_seed if hasattr(config, 'analysis_seed') else 83),
        "validation": (config.probe_validation_samples, "interpolation", 294),
        "composition": (config.probe_test_samples, "composition", 1399),
        "extrapolation": (config.probe_test_samples, "extrapolation", 2408),
    }
    for name, (count, regime, source_seed) in specs.items():
        datasets[name], labels[name], conditions[name] = _branch_dataset(task, k=k, count=count, seed=source_seed, regime=regime)
        features[name] = _features(system, datasets[name], device)
    result = {}
    for index, cut in enumerate(CUTS):
        result[cut] = fit_branch_probe(
            features["train"][cut], features["validation"][cut],
            {name: features[name][cut] for name in REGIMES}, labels["train"], labels["validation"],
            {name: labels[name] for name in REGIMES}, conditions["train"], conditions["validation"],
            {name: conditions[name] for name in REGIMES}, k=k, config=config,
            seed=seed * 10_007 + index * 1_009, device=device,
        )
    return result


@torch.no_grad()
def _interpolate_system(system: DegreeKSystem, left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], amount: float, device: torch.device) -> None:
    for name, destination in system.state_dict().items():
        a, b = left[name].to(device), right[name].to(device)
        destination.copy_(torch.lerp(a, b, amount) if torch.is_floating_point(destination) else a)


def transition_charge(system: DegreeKSystem, left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], task: CircleTaskConfig, config: DegreeKLadderConfig, k: int, seed: int, device: torch.device) -> Dict[str, Any]:
    phases = np.linspace(0.0, 2.0 * math.pi, config.transition_phase_points, endpoint=False)
    path = np.linspace(0.0, 1.0, config.transition_path_points)
    dataset = generate_dataset(task, k=k, count=len(phases), seed=seed, phases=phases, fixed_nuisance=True)
    field = np.empty((len(path), len(phases)), dtype=np.complex128)
    for index, amount in enumerate(path):
        _interpolate_system(system, left, right, float(amount), device)
        _, field[index] = posterior_map(system, dataset, task, device)
    return complex_defect_charge(field, phases, path)


def run_arm(k: int, seed: int, task: CircleTaskConfig, config: DegreeKLadderConfig, output: Path, device: torch.device) -> Dict[str, Any]:
    root = output / "runs" / f"k{k}" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    model = TinyLLMModel(calibrated._model_config(config.preset, task, seed)).to(device)
    system = DegreeKSystem(model, task).to(device)
    train = generate_dataset(task, k=k, count=config.train_samples, seed=seed + 1_001)
    generator = torch.Generator().manual_seed(seed + 6_013)
    batches = torch.randint(0, config.train_samples, (config.training_steps, config.batch_size), generator=generator)
    optimizer = torch.optim.AdamW(system.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    answer = torch.tensor(task.answer_token_ids, device=device)
    history = []; trace = []; transitions = []
    previous_state = {name: value.detach().cpu().clone() for name, value in system.state_dict().items()} if seed == 7 else None
    previous_degree = None
    if seed == 7:
        initial = map_diagnostics(system, task, config, k, seed + 44_001, "interpolation", device, config.trace_phase_points, include_topology=False)
        previous_degree = round(initial["winding_degree"]); trace.append({"step": 0, **initial})
    started = time.perf_counter()
    system.train()
    for step in range(1, config.training_steps + 1):
        index = batches[step - 1]
        optimizer.zero_grad(set_to_none=True)
        residual = system.forward_cuts(train.input_ids[index].to(device), train.sensor[index].to(device), train.calibration[index].to(device))["full"]
        logits = io_source._task_logits(model, residual, answer)
        targets = train.target_posteriors[index].to(device)
        loss = -(targets * torch.log_softmax(logits, -1)).sum(-1).mean()
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(system.parameters(), config.gradient_clip); optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            history.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        if seed == 7:
            diag = map_diagnostics(system, task, config, k, seed + 44_001, "interpolation", device, config.trace_phase_points, include_topology=False)
            degree = round(diag["winding_degree"]); trace.append({"step": step, **diag})
            current_state = {name: value.detach().cpu().clone() for name, value in system.state_dict().items()}
            if degree != previous_degree:
                charge = transition_charge(system, previous_state, current_state, task, config, k, seed + step * 101, device)  # type: ignore[arg-type]
                transitions.append({"step": step, "degree_before": previous_degree, "degree_after": degree, "charge": charge})
            previous_state, previous_degree = current_state, degree
            system.train()
    if device.type == "cuda": torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    maps = {regime: map_diagnostics(system, task, config, k, seed + 55_003, regime, device) for regime in REGIMES}
    probes = analyze_branches(system, task, config, k, seed, device) if k > 1 else None
    evaluations = {}
    for regime in REGIMES:
        data = generate_dataset(task, k=k, count=config.probe_test_samples, seed=seed + 66_007, regime=regime)
        posterior, _ = posterior_map(system, data, task, device)
        evaluations[regime] = {"exact_bin_accuracy": float((posterior.argmax(1) == data.target_bins.numpy()).mean())}
    checkpoint = root / "model.pt"
    model.to("cpu").save_checkpoint(checkpoint, metadata={"schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID, "k": k, "seed": seed})
    frontend = root / "frontend.pt"; torch.save({"vector_embedding": system.vector_embedding.to("cpu").state_dict()}, frontend)
    system.to(device)
    result = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-degree-k{k}-seed{seed}", "status": "completed",
        "completed_at": _utc_now(), "k": k, "seed": seed, "maps": maps,
        "branch_probes": probes, "task_metrics": evaluations, "degree_trace": trace,
        "degree_transitions": transitions,
        "net_defect_charge": sum(item["charge"]["defect_charge"] for item in transitions),
        "training": {"seconds": training_seconds, "history": history, "checkpoint": str(checkpoint), "frontend_checkpoint": str(frontend), "final_model_state_sha256": _state_digest(model), "training_data_sha256": _digest(train.input_ids, train.target_posteriors, train.calibration), "minibatch_schedule_sha256": _digest(batches)},
        "implementation_sha256": _implementation_digest(),
    }
    _write_json(root / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: DegreeKLadderConfig) -> Dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    result = {}
    for k in config.degrees:
        selected = sorted((run for run in runs if run["k"] == k), key=lambda run: run["seed"])
        per_seed = {}
        ceiling = None if k == 1 else 0.55 if k == 2 else 0.3834
        for run in selected:
            cells = []
            for regime in REGIMES:
                item = run["maps"][regime]
                cells.append(item["alignment"] >= 0.90 and abs(item["winding_degree"] - k) <= 0.10 and item["sampling_resolved"] and item["posterior_normalized_h1_lifetime"] >= 0.40)
                if k > 1:
                    cells.extend(run["branch_probes"][cut]["evaluations"][regime]["balanced_accuracy"] <= ceiling for cut in ("post_mlp", "full"))
                    cells.append(run["branch_probes"]["frontend"]["evaluations"][regime]["balanced_accuracy"] >= 0.90)
            per_seed[str(run["seed"])] = {"cells": cells, "joint_pass": all(cells)}
        seed7 = next((run for run in selected if run["seed"] == 7), None)
        result[str(k)] = {
            "joint_pass_by_seed": per_seed, "joint_pass_count": sum(item["joint_pass"] for item in per_seed.values()),
            "success": sum(item["joint_pass"] for item in per_seed.values()) >= required,
            "map_means": {regime: {key: statistics.fmean(run["maps"][regime][key] for run in selected) for key in ("alignment", "winding_degree", "posterior_normalized_h1_lifetime")} for regime in REGIMES},
            "seed7_net_defect_charge": seed7["net_defect_charge"] if seed7 else None,
            "seed7_charge_matches_k": seed7["net_defect_charge"] == k if seed7 else None,
        }
    return {"degrees": result, "required_joint_seed_passes": required, "confirmed": all(item["success"] for item in result.values())}


def run_campaign(config: DegreeKLadderConfig, task: CircleTaskConfig, output: Path) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    runs = []
    for k in config.degrees:
        for seed in config.seeds:
            path = output / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            if path.is_file():
                existing = json.loads(path.read_text())
                if existing.get("implementation_sha256") == _implementation_digest():
                    runs.append(existing); print(f"resuming {existing['experiment_id']}", flush=True); continue
            run = run_arm(k, seed, task, config, output, device); runs.append(run)
            print(run["experiment_id"], {name: run["maps"][name]["winding_degree"] for name in REGIMES}, flush=True)
    bundle = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed", "completed_at": _utc_now(), "configuration": asdict(config),
        "task_config": asdict(task), "implementation_sha256": _implementation_digest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "summary": {"requested": len(config.degrees) * len(config.seeds), "completed": len(runs), "failed": len(config.degrees) * len(config.seeds) - len(runs)},
        "aggregates": aggregate(runs, config),
        "method_boundaries": ["The carrier is fixed and analytic; this isolates downstream finite-group quotient formation.", "Conditional probes test one declared nonlinear estimator.", "Defect cells are grid-indexed and are not interval-certified roots."],
    }
    _write_json(output / "campaign_results.json", bundle)
    return bundle


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,17,29,41,53"); parser.add_argument("--degrees", default="1,2,3")
    parser.add_argument("--steps", type=int, default=600); parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--phase-grid-points", type=int, default=192); parser.add_argument("--trace-phase-points", type=int, default=96)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--allow-underpowered", action="store_true"); parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"))
    args = parser.parse_args(argv)
    if args.shakedown:
        args.seeds, args.degrees, args.steps, args.train_samples, args.batch_size = "7", "1,2,3", 2, 48, 12
        args.probe_steps, args.phase_grid_points, args.trace_phase_points, args.allow_underpowered = 20, 48, 32, True
    config = DegreeKLadderConfig(
        seeds=_ints(args.seeds), degrees=_ints(args.degrees), training_steps=args.steps,
        train_samples=args.train_samples, batch_size=args.batch_size, probe_steps=args.probe_steps,
        phase_grid_points=args.phase_grid_points, trace_phase_points=args.trace_phase_points,
        device=args.device, allow_underpowered=args.allow_underpowered,
        probe_train_samples=96 if args.shakedown else 2_048,
        probe_validation_samples=48 if args.shakedown else 512,
        probe_test_samples=96 if args.shakedown else 1_024,
        transition_phase_points=64 if args.shakedown else 256,
        transition_path_points=8 if args.shakedown else 66,
    )
    bundle = run_campaign(config, CircleTaskConfig(train_samples=config.train_samples), args.output)
    print(json.dumps(bundle["aggregates"], indent=2, sort_keys=True)); print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
