#!/usr/bin/env python3
"""Morse--Cerf scan of exact deck-orbit mixture geometry in frozen TinyLLMs."""

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
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_io_correspondence as io_source
import experiments.structure_net.tinyllm_reynolds_koopman as koopman
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-morse-cerf.v1"
HYPOTHESIS_ID = "tinyllm-equivariant-morse-cerf-quotient-front-v1"
REGIMES = ("composition", "extrapolation")
CONDITIONS = ("exact", "random_pairing", "phase_shuffled")


@dataclass(frozen=True)
class MorseCerfConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    causal_root: str = "data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered"
    koopman_root: str = "data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    degrees: tuple[int, ...] = (2, 3)
    orbit_count: int = 32
    interval_points: int = 49
    triangle_subdivisions: int = 12
    alpha_points: int = 9
    control_alpha_points: int = 5
    first_blocks: int = 3
    activation_batch_size: int = 256
    continuation_batch_size: int = 512
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if set(self.degrees).difference((2, 3)):
            raise ValueError("degrees must be drawn from 2,3")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.interval_points < 5 or self.interval_points % 2 == 0:
            raise ValueError("interval grid must have an odd number of at least five points")
        if self.triangle_subdivisions < 3 or self.triangle_subdivisions % 3:
            raise ValueError("triangle subdivisions must be a positive multiple of three")
        if min(self.alpha_points, self.control_alpha_points) < 3:
            raise ValueError("each Cerf family needs at least three alpha points")


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
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(deck.__file__), Path(koopman.__file__)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def simplex_grid(k: int, interval_points: int = 49, subdivisions: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return simplex weights and integer lattice coordinates."""
    if k == 2:
        integer = np.column_stack(
            (np.arange(interval_points), np.arange(interval_points - 1, -1, -1))
        )
        return integer / float(interval_points - 1), integer
    if k != 3:
        raise ValueError("only k=2 and k=3 are supported")
    integer = np.asarray(
        [(a, b, subdivisions - a - b) for a in range(subdivisions + 1) for b in range(subdivisions - a + 1)],
        dtype=np.int64,
    )
    return integer / float(subdivisions), integer


def lattice_adjacency(integer: np.ndarray) -> tuple[tuple[int, ...], ...]:
    adjacency = []
    for index, value in enumerate(integer):
        distance = np.abs(integer - value).sum(axis=1)
        adjacency.append(tuple(int(item) for item in np.flatnonzero(distance == 2)))
    return tuple(adjacency)


def cyclic_symmetrize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    lookup = {tuple(np.round(item, 12)): index for index, item in enumerate(weights)}
    result = np.zeros_like(values, dtype=np.float64)
    for index, weight in enumerate(weights):
        orbit = []
        for shift in range(weights.shape[1]):
            orbit.append(values[lookup[tuple(np.round(np.roll(weight, shift), 12))]])
        result[index] = float(np.mean(orbit))
    return result


def _vertices_center(weights: np.ndarray) -> tuple[list[int], int]:
    vertices = [int(np.argmax(np.all(np.isclose(weights, np.eye(weights.shape[1])[item]), axis=1))) for item in range(weights.shape[1])]
    center = int(np.argmin(np.square(weights - 1.0 / weights.shape[1]).sum(axis=1)))
    return vertices, center


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _components_at(values: np.ndarray, adjacency: Sequence[Sequence[int]], threshold: float) -> tuple[_UnionFind, np.ndarray]:
    active = values <= threshold
    union = _UnionFind(len(values))
    for index in np.flatnonzero(active):
        for neighbor in adjacency[index]:
            if active[neighbor]:
                union.union(int(index), int(neighbor))
    return union, active


def _merge_threshold(values: np.ndarray, adjacency: Sequence[Sequence[int]], required: Sequence[int]) -> float:
    active = np.zeros(len(values), dtype=bool)
    union = _UnionFind(len(values))
    for index in np.argsort(values, kind="stable"):
        active[index] = True
        for neighbor in adjacency[index]:
            if active[neighbor]:
                union.union(int(index), int(neighbor))
        if all(active[item] for item in required):
            root = union.find(required[0])
            if all(union.find(item) == root for item in required[1:]):
                return float(values[index])
    return float("inf")


def _link_components(nodes: Sequence[int], adjacency: Sequence[Sequence[int]]) -> int:
    remaining = set(nodes)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            item = stack.pop()
            connected = remaining.intersection(adjacency[item])
            remaining.difference_update(connected)
            stack.extend(connected)
    return components


def critical_census(values: np.ndarray, adjacency: Sequence[Sequence[int]], weights: np.ndarray) -> dict[str, int]:
    result = {"minima": 0, "maxima": 0, "saddles": 0, "interior_saddles": 0}
    for index, neighbors in enumerate(adjacency):
        lower = [item for item in neighbors if values[item] < values[index] - 1e-12]
        upper = [item for item in neighbors if values[item] > values[index] + 1e-12]
        if not lower:
            result["minima"] += 1
        if not upper:
            result["maxima"] += 1
        if len(lower) >= 2 and _link_components(lower, adjacency) >= 2:
            result["saddles"] += 1
            if np.all(weights[index] > 0):
                result["interior_saddles"] += 1
    return result


def _center_index(values: np.ndarray, weights: np.ndarray, integer: np.ndarray, center: int, tolerance: float = 1e-4) -> dict[str, Any]:
    k = weights.shape[1]
    if k == 2:
        step = 1.0 / (len(weights) - 1)
        ordered = np.argsort(weights[:, 0])
        position = int(np.flatnonzero(ordered == center)[0])
        curvature = float((values[ordered[position - 1]] - 2 * values[center] + values[ordered[position + 1]]) / step**2)
        index = 0 if curvature > tolerance else 1 if curvature < -tolerance else None
        return {"index": index, "degenerate": index is None, "curvature": curvature}
    delta = integer - integer[center]
    local = np.max(np.abs(delta), axis=1) <= 2
    x = weights[local, 0] - weights[center, 0]
    y = weights[local, 1] - weights[center, 1]
    design = np.column_stack((np.ones(len(x)), x, y, 0.5 * x * x, x * y, 0.5 * y * y))
    coefficient = np.linalg.lstsq(design, values[local], rcond=None)[0]
    hessian = np.asarray([[coefficient[3], coefficient[4]], [coefficient[4], coefficient[5]]])
    eigenvalues = np.linalg.eigvalsh(hessian)
    degenerate = bool(np.any(np.abs(eigenvalues) <= tolerance))
    return {
        "index": None if degenerate else int(np.sum(eigenvalues < 0)),
        "degenerate": degenerate,
        "hessian": hessian.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "local_fit_points": int(local.sum()),
    }


def landscape_metrics(values: np.ndarray, weights: np.ndarray, integer: np.ndarray) -> dict[str, Any]:
    adjacency = lattice_adjacency(integer)
    vertices, center = _vertices_center(weights)
    vertex_values = values[vertices]
    threshold = float(np.max(vertex_values) + 0.05)
    union, active = _components_at(values, adjacency, threshold)
    vertex_roots = {union.find(item) for item in vertices if active[item]}
    center_connected = bool(
        active[center]
        and all(active[item] for item in vertices)
        and len({union.find(item) for item in [*vertices, center]}) == 1
    )
    merge = _merge_threshold(values, adjacency, [*vertices, center])
    metrics = {
        "vertex_values": vertex_values.tolist(),
        "center_value": float(values[center]),
        "tau_valid": threshold,
        "excess": float(values[center] - np.mean(vertex_values)),
        "merge_threshold": merge,
        "merge_barrier": float(merge - np.max(vertex_values)),
        "sheet_component_count": len(vertex_roots),
        "center_in_vertex_component": center_connected,
        "center": _center_index(values, weights, integer, center),
        "critical_census": critical_census(values, adjacency, weights),
    }
    metrics["mature"] = bool(
        metrics["center"]["index"] == 0
        and metrics["excess"] <= 0.05
        and metrics["merge_barrier"] <= 0.05
        and center_connected
    )
    return metrics


def vertex_exact_homotopy(
    sheets: torch.Tensor,
    deltas: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
    residual_update: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return corrected task states and the uncorrected commutator discrepancy."""
    moving = sheets + alpha * deltas
    flat = moving.flatten(0, 1)
    sheet_updates = residual_update(flat).reshape_as(moving)
    mixture = torch.einsum("wk,nk...->wn...", weights, moving)
    flat_mixture = mixture.flatten(0, 1)
    mixture_updates = residual_update(flat_mixture).reshape_as(mixture)
    correction = torch.einsum("wk,nk...->wn...", weights, deltas - sheet_updates)
    completed = mixture + (1.0 - alpha) * (mixture_updates + correction)
    uncorrected_mixture = mixture + (1.0 - alpha) * mixture_updates
    uncorrected_sheets = moving + (1.0 - alpha) * sheet_updates
    sheet_average = torch.einsum("wk,nk...->wn...", weights, uncorrected_sheets)
    axes = tuple(range(2, uncorrected_mixture.ndim))
    numerator = torch.square(uncorrected_mixture - sheet_average).sum(dim=axes)
    denominator = torch.square(sheet_average).sum(dim=axes).clamp_min(1e-12)
    return completed, numerator / denominator


def _bridge_config(config: MorseCerfConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_root,
        seeds=config.seeds,
        degrees=config.degrees,
        fit_orbits=config.orbit_count,
        evaluation_orbits=config.orbit_count,
        map_points=24,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _residual_update(system: Any, transition_index: int, values: torch.Tensor) -> torch.Tensor:
    block = system.model.transformer["h"][transition_index // 2]
    if transition_index % 2 == 0:
        return block.attn(block.ln_1(values))
    return block.mlp(block.ln_2(values))


@torch.no_grad()
def _continue_tensor(
    system: Any,
    cut: str,
    values: torch.Tensor,
    task: CircleTaskConfig,
    batch_size: int,
) -> torch.Tensor:
    device = values.device
    answer = torch.tensor(task.answer_token_ids, device=device)
    posteriors = []
    for start in range(0, len(values), batch_size):
        value = values[start : start + batch_size]
        pieces = cut.split("_")
        block_index = int(pieces[1])
        if cut.endswith("post_attention"):
            block = system.model.transformer["h"][block_index]
            value = value + block.mlp(block.ln_2(value))
        next_block = block_index + 1
        for block in system.model.transformer["h"][next_block:]:
            value = block(value)
        logits = io_source._task_logits(system.model, value[:, -1, :], answer)
        posteriors.append(torch.softmax(logits, -1))
    return torch.cat(posteriors)


def _paired_sheets(values: torch.Tensor, condition: str, seed: int) -> torch.Tensor:
    if condition == "exact":
        return values
    result = values.clone()
    orbit_count, k = values.shape[:2]
    if condition == "random_pairing":
        generator = np.random.default_rng(seed)
        for sheet in range(1, k):
            order = torch.tensor(generator.permutation(orbit_count), device=values.device)
            result[:, sheet] = values[order, sheet]
    elif condition == "phase_shuffled":
        for sheet in range(1, k):
            shift = sheet * max(1, orbit_count // (k + 1))
            result[:, sheet] = torch.roll(values[:, sheet], shifts=shift, dims=0)
    else:
        raise ValueError(condition)
    return result


@torch.no_grad()
def scan_landscape(
    system: Any,
    dataset: deck.OrbitDataset,
    source: np.ndarray,
    target: np.ndarray,
    target_cut: str,
    transition_index: int,
    weights: np.ndarray,
    alpha: float,
    condition: str,
    pairing_seed: int,
    task: CircleTaskConfig,
    config: MorseCerfConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k, n = dataset.k, dataset.orbit_count
    source_t = torch.from_numpy(source).to(device).reshape((n, k) + source.shape[1:])
    target_t = torch.from_numpy(target).to(device).reshape((n, k) + target.shape[1:])
    source_t = _paired_sheets(source_t, condition, pairing_seed)
    target_t = _paired_sheets(target_t, condition, pairing_seed)
    deltas = target_t - source_t
    weight_t = torch.from_numpy(weights).to(device=device, dtype=source_t.dtype)
    states, commutator = vertex_exact_homotopy(
        source_t,
        deltas,
        weight_t,
        alpha,
        lambda value: _residual_update(system, transition_index, value),
    )
    posteriors = _continue_tensor(
        system, target_cut, states.flatten(0, 1), task, config.continuation_batch_size
    ).reshape(len(weights), n, -1)
    target_p = dataset.target_posteriors.reshape(n, k, -1)[:, 0].to(device=device, dtype=posteriors.dtype)
    kl = (target_p[None] * (target_p.clamp_min(1e-12).log()[None] - posteriors.clamp_min(1e-12).log())).sum(-1).mean(-1)
    target_bins = dataset.target_bins.reshape(n, k)[:, 0].to(device)
    accuracy = (posteriors.argmax(-1) == target_bins[None]).float().mean(-1)
    return kl.double().cpu().numpy(), commutator.mean(-1).double().cpu().numpy(), accuracy.double().cpu().numpy()


def _load_comparators(config: MorseCerfConfig, k: int, seed: int, checkpoint_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    causal_path = Path(config.causal_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
    koopman_path = Path(config.koopman_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
    causal = json.loads(causal_path.read_text())
    reynolds = json.loads(koopman_path.read_text())
    for path, item in ((causal_path, causal), (koopman_path, reynolds)):
        if item.get("status") != "completed":
            raise ValueError(f"incomplete comparator {path}")
        source_sha = item["provenance"]["checkpoint_sha256"]
        if source_sha != checkpoint_sha:
            raise ValueError(f"checkpoint mismatch in {path}")
    ordered = deck.cut_names(config.first_blocks)
    fronts = {}
    validations = {}
    for regime in REGIMES:
        preserved = [
            cut for cut in ordered
            if causal["cuts"][cut]["causal"][regime]["orbit_average"].get("causal_classification") == "preserved"
        ]
        front = preserved[0] if preserved else None
        fronts[regime] = front
        validations[regime] = None if front is None else causal["cuts"][front]["causal"][regime]["orbit_average"]
    return {
        "path": str(causal_path), "sha256": _sha256(causal_path), "fronts": fronts, "validations": validations
    }, {
        "path": str(koopman_path),
        "sha256": _sha256(koopman_path),
        "autonomous_one_step_near_front": reynolds["autonomous_one_step_near_front"],
    }


def _cut_depth(cut: Optional[str], transitions: Sequence[tuple[str, str]]) -> Optional[float]:
    if cut is None:
        return None
    if cut == "block_0_pre_attention":
        return 0.0
    for index, (_, target) in enumerate(transitions):
        if cut == target:
            return float(index + 1)
    return None


def _persistent_front(records: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    endpoints = [item for item in records if math.isclose(float(item["alpha"]), 1.0)]
    for item in records:
        depth = float(item["depth"])
        condition = bool(item[field])
        later = all(bool(endpoint[field]) for endpoint in endpoints if float(endpoint["depth"]) >= depth)
        if condition and later:
            return depth
    return None


def _events(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for left, right in zip(records, records[1:]):
        if int(left["transition_index"]) != int(right["transition_index"]):
            continue
        kinds = []
        if left["task"]["center"]["index"] != right["task"]["center"]["index"]:
            kinds.append("center_index_change")
        if (left["task"]["merge_barrier"] <= 0.05) != (right["task"]["merge_barrier"] <= 0.05):
            kinds.append("merge_barrier_crossing")
        if left["task"]["sheet_component_count"] != right["task"]["sheet_component_count"]:
            kinds.append("sheet_component_change")
        if left["task"]["critical_census"]["saddles"] != right["task"]["critical_census"]["saddles"]:
            kinds.append("saddle_census_change")
        if kinds:
            events.append({"depth": (float(left["depth"]) + float(right["depth"])) / 2.0, "kinds": kinds})
    return events


def _regime_summary(records: Mapping[str, Sequence[Mapping[str, Any]]], causal_front: Optional[str], causal_validation: Any, autonomous: bool, transitions: Sequence[tuple[str, str]], k: int) -> dict[str, Any]:
    exact = records["exact"]
    causal_depth = _cut_depth(causal_front, transitions)
    task_front = _persistent_front(exact, "task_mature")
    comm_front = _persistent_front(exact, "commutator_mature")
    events = _events(exact)
    early = [item for item in exact if causal_depth is not None and float(item["depth"]) < causal_depth]
    early_gate = any(item["task"]["sheet_component_count"] >= k or item["task"]["center"]["index"] not in (0, None) for item in early)
    near_gate = bool(causal_depth is not None and any(abs(event["depth"] - causal_depth) <= 1.0 for event in events))
    later_endpoints = [item for item in exact if causal_depth is not None and float(item["depth"]) >= causal_depth and math.isclose(float(item["alpha"]), 1.0)]
    causal_ok = bool(causal_validation and causal_validation.get("causal_classification") == "preserved")
    mature_gate = bool(later_endpoints and all(item["task_mature"] for item in later_endpoints) and causal_ok)
    control_fronts = {condition: _persistent_front(records[condition], "task_mature") for condition in CONDITIONS[1:]}
    control_gate = bool(
        causal_depth is not None
        and all(front is None or abs(front - causal_depth) > 1.0 for front in control_fronts.values())
    )
    if task_front is None or comm_front is None:
        mismatch_gate = False
        front_distance = None
    else:
        front_distance = abs(task_front - comm_front)
        mismatch_gate = front_distance <= 1.0 if autonomous else front_distance > 0.5
    gates = {
        "early_cover_structure": early_gate,
        "near_front_morse_event": near_gate,
        "mature_barycenter_basin": mature_gate,
        "control_specificity": control_gate,
        "task_closure_distinction": mismatch_gate,
    }
    return {
        "causal_front": causal_front,
        "causal_depth": causal_depth,
        "task_morse_front": task_front,
        "commutator_front": comm_front,
        "front_distance": front_distance,
        "control_fronts": control_fronts,
        "events": events,
        "causal_validation": causal_validation,
        "reynolds_autonomous_near_front": autonomous,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def analyze_cell(task: CircleTaskConfig, config: MorseCerfConfig, k: int, seed: int, output: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, provenance = deck.load_source(task, bridge, k, seed, device)
    causal, reynolds = _load_comparators(config, k, seed, provenance["checkpoint_sha256"])
    weights, integer = simplex_grid(k, config.interval_points, config.triangle_subdivisions)
    transitions = koopman.one_step_transitions(config.first_blocks)
    arrays: dict[str, np.ndarray] = {"weights": weights, "integer_lattice": integer}
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        dataset = deck.generate_exact_orbits(
            task, k=k, orbit_count=config.orbit_count, seed=seed + 1701 + regime_index * 101, regime=regime
        )
        captured = deck.capture_sequences(system, dataset, bridge, device)
        condition_records: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}
        for condition_index, condition in enumerate(CONDITIONS):
            alpha_count = config.alpha_points if condition == "exact" else config.control_alpha_points
            for transition_index, (source_cut, target_cut) in enumerate(transitions):
                for alpha in np.linspace(0.0, 1.0, alpha_count):
                    potential, commutator, accuracy = scan_landscape(
                        system, dataset, captured[source_cut], captured[target_cut], target_cut,
                        transition_index, weights, float(alpha), condition,
                        seed + 100_003 * regime_index + 10_007 * condition_index + transition_index,
                        task, config, device,
                    )
                    potential = cyclic_symmetrize(potential, weights)
                    commutator = cyclic_symmetrize(commutator, weights)
                    accuracy = cyclic_symmetrize(accuracy, weights)
                    key = f"{regime}__{condition}__t{transition_index}__a{len(condition_records[condition])}"
                    arrays[f"{key}__potential"] = potential.astype(np.float32)
                    arrays[f"{key}__commutator"] = commutator.astype(np.float32)
                    arrays[f"{key}__accuracy"] = accuracy.astype(np.float32)
                    task_metrics = landscape_metrics(potential, weights, integer)
                    comm_metrics = landscape_metrics(commutator, weights, integer)
                    center = _vertices_center(weights)[1]
                    record = {
                        "transition_index": transition_index,
                        "source_cut": source_cut,
                        "target_cut": target_cut,
                        "alpha": float(alpha),
                        "depth": float(transition_index + alpha),
                        "task": task_metrics,
                        "commutator": comm_metrics,
                        "center_accuracy": float(accuracy[center]),
                        "task_mature": bool(task_metrics["mature"]),
                        "commutator_mature": bool(
                            comm_metrics["center"]["index"] == 0 and comm_metrics["center_value"] <= 0.01
                        ),
                    }
                    condition_records[condition].append(record)
            print(f"k{k} seed {seed} {regime} {condition} complete", flush=True)
        regimes[regime] = {
            "conditions": condition_records,
            "summary": _regime_summary(
                condition_records, causal["fronts"][regime], causal["validations"][regime],
                bool(reynolds["autonomous_one_step_near_front"][regime]), transitions, k,
            ),
        }
    root = output / "runs" / f"k{k}" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    arrays_path = root / "landscapes.npz"
    np.savez_compressed(arrays_path, **arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-morse-cerf-k{k}-seed{seed}",
        "status": "completed",
        "completed_at": _utc_now(),
        "k": k,
        "seed": seed,
        "configuration": asdict(config),
        "provenance": {**provenance, "causal_comparator": causal, "reynolds_comparator": reynolds},
        "regimes": regimes,
        "landscapes_path": str(arrays_path),
        "landscapes_sha256": _sha256(arrays_path),
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(root / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: MorseCerfConfig) -> dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    gate_names = (
        "early_cover_structure", "near_front_morse_event", "mature_barycenter_basin",
        "control_specificity", "task_closure_distinction",
    )
    degrees = {}
    for k in config.degrees:
        selected = [item for item in runs if int(item["k"]) == k]
        counts = {
            regime: {
                gate: sum(bool(item["regimes"][regime]["summary"]["gates"][gate]) for item in selected)
                for gate in gate_names
            }
            for regime in REGIMES
        }
        passes = {regime: {gate: count >= required for gate, count in counts[regime].items()} for regime in REGIMES}
        per_seed = {
            str(item["seed"]): {
                regime: item["regimes"][regime]["summary"] for regime in REGIMES
            }
            for item in selected
        }
        degrees[str(k)] = {
            "gate_counts": counts,
            "gate_passes": passes,
            "per_seed": per_seed,
            "confirmed": all(all(value.values()) for value in passes.values()),
        }
    return {
        "required_seed_count": required,
        "degrees": degrees,
        "confirmed": all(degrees[str(k)]["confirmed"] for k in config.degrees),
    }


def run_campaign(config: MorseCerfConfig, output: Path) -> dict[str, Any]:
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
            "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "summary": {"requested": len(config.degrees) * len(config.seeds), "completed": len(runs)},
        "aggregates": aggregate(runs, config),
        "certification": {
            "candidate": "k3 seed 7 only",
            "status": "not_attempted_pending_numerical_candidate",
            "scope": "stored polynomial only unless a rigorous transformer remainder is supplied",
        },
        "method_boundaries": [
            "The result is a finite-grid statement for the declared vertex-exact intervention homotopy.",
            "Discrete critical counts do not prove continuum Morse nondegeneracy.",
            "The commutator potential measures autonomous Reynolds closure and is not a task loss.",
            "Controls deliberately break exact orbit pairing and need not preserve vertex task loss.",
        ],
    }
    _write_json(output / "campaign_results.json", bundle)
    return bundle


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=MorseCerfConfig.source_root)
    parser.add_argument("--causal-root", default=MorseCerfConfig.causal_root)
    parser.add_argument("--koopman-root", default=MorseCerfConfig.koopman_root)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_morse_cerf/20260806_d6_preregistered"))
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--degrees", default="2,3")
    parser.add_argument("--orbit-count", type=int, default=32)
    parser.add_argument("--interval-points", type=int, default=49)
    parser.add_argument("--triangle-subdivisions", type=int, default=12)
    parser.add_argument("--alpha-points", type=int, default=9)
    parser.add_argument("--control-alpha-points", type=int, default=5)
    parser.add_argument("--first-blocks", type=int, default=3)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--continuation-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = MorseCerfConfig(
        source_root=args.source_root, causal_root=args.causal_root, koopman_root=args.koopman_root,
        seeds=_ints(args.seeds), degrees=_ints(args.degrees), orbit_count=args.orbit_count,
        interval_points=args.interval_points, triangle_subdivisions=args.triangle_subdivisions,
        alpha_points=args.alpha_points, control_alpha_points=args.control_alpha_points,
        first_blocks=args.first_blocks, activation_batch_size=args.activation_batch_size,
        continuation_batch_size=args.continuation_batch_size, device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    bundle = run_campaign(config, args.output)
    print(json.dumps(bundle["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
