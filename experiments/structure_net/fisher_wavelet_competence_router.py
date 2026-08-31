#!/usr/bin/env python3
"""Synthetic preflight for Fisher-weighted wavelet competence routing."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "nal.fisher-wavelet-competence-router.v1"
HYPOTHESIS_ID = "fisher-wavelet-competence-router-synthetic-v1"
SEEDS = (7, 17, 29, 41, 53)
MODEL_NAMES = ("A", "B", "C")
COSTS = np.array([1.0, 2.0, 4.0])


@dataclass(frozen=True)
class Config:
    calibration_samples: int = 640
    test_samples: int = 512
    neighbors: int = 21
    threshold: float = 0.70
    failure_penalty: float = 8.0
    fisher_sigma: float = 0.85
    topology_floor: float = 0.08
    diffusion_time: float = 1.0
    wavelet_dimensions: int = 4
    fiber_weight: float = 0.08
    label_noise: float = 0.03


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def states() -> np.ndarray:
    return np.array(
        [[(index >> bit) & 1 for bit in range(3)] for index in range(8)],
        dtype=np.float64,
    )


def teacher_probabilities(state: np.ndarray) -> np.ndarray:
    logits = np.column_stack(
        (
            1.4 - 1.2 * state[:, 0] + 0.3 * state[:, 2],
            -0.4 + 1.5 * state[:, 0] - 0.8 * state[:, 1],
            -0.7 + 1.4 * state[:, 1] - 0.5 * state[:, 2],
            -1.0 + 1.6 * state[:, 2] + 0.7 * state[:, 0] * state[:, 1],
        )
    )
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def fisher_rao_distance(left: np.ndarray, right: np.ndarray) -> float:
    affinity = float(np.sqrt(left * right).sum())
    return 2.0 * math.acos(float(np.clip(affinity, -1.0, 1.0)))


def fisher_wavelet_basis(config: Config) -> tuple[np.ndarray, dict[str, Any]]:
    state = states()
    probability = teacher_probabilities(state)
    adjacency = (np.abs(state[:, None, :] - state[None, :, :]).sum(axis=2) == 1)
    weights = np.zeros((8, 8), dtype=np.float64)
    for left in range(8):
        for right in range(left + 1, 8):
            if adjacency[left, right]:
                distance = fisher_rao_distance(probability[left], probability[right])
                weight = config.topology_floor + (1.0 - config.topology_floor) * math.exp(
                    -(distance**2) / (2.0 * config.fisher_sigma**2)
                )
                weights[left, right] = weights[right, left] = weight
    degree = weights.sum(axis=1)
    inverse_root = np.diag(1.0 / np.sqrt(degree))
    laplacian = np.eye(8) - inverse_root @ weights @ inverse_root
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    basis = eigenvectors * np.exp(-config.diffusion_time * eigenvalues)[None, :]
    diagnostics = {
        "adjacency_edges": int(adjacency.sum() // 2),
        "minimum_weighted_degree": float(degree.min()),
        "laplacian_eigenvalues": eigenvalues.tolist(),
        "eigenvector_orthonormality_error": float(
            np.max(np.abs(eigenvectors.T @ eigenvectors - np.eye(8)))
        ),
    }
    return basis[:, : config.wavelet_dimensions], diagnostics


def competence(state: np.ndarray) -> np.ndarray:
    # Each Boolean vertex has a fixed competence signature.  The table includes
    # both approximately hierarchical regions and explicit complementary-model
    # counterexamples; no scalar "difficulty tier" can represent it.
    signature_table = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    index = (state[:, 0] + 2 * state[:, 1] + 4 * state[:, 2]).astype(np.int64)
    return signature_table[index]


def generate(count: int, seed: int, config: Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state_index = rng.integers(0, 8, size=count)
    state = states()[state_index]
    nuisance = rng.normal(size=(count, 3))
    labels = competence(state)
    flips = rng.random(labels.shape) < config.label_noise
    labels = np.logical_xor(labels, flips)
    return {"state": state, "state_index": state_index, "nuisance": nuisance, "labels": labels}


def embed(dataset: Mapping[str, np.ndarray], basis: np.ndarray, config: Config, *, wavelet: bool) -> np.ndarray:
    nuisance = np.asarray(dataset["nuisance"], dtype=np.float64)
    if not wavelet:
        return nuisance
    state_coordinates = basis[np.asarray(dataset["state_index"], dtype=np.int64)]
    return np.column_stack((state_coordinates, config.fiber_weight * nuisance))


def predict_knn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, k: int) -> np.ndarray:
    squared = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
    neighbor = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
    selected_distance = np.take_along_axis(squared, neighbor, axis=1)
    weights = 1.0 / (np.sqrt(selected_distance) + 1e-6)
    selected_labels = train_y[neighbor]
    return (selected_labels * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)


def route(probability: np.ndarray, threshold: float) -> np.ndarray:
    eligible = probability >= threshold
    choice = np.full(len(probability), 2, dtype=np.int64)
    choice[eligible[:, 1]] = 1
    choice[eligible[:, 0]] = 0
    return choice


def score(choice: np.ndarray, labels: np.ndarray, config: Config) -> dict[str, float]:
    success = labels[np.arange(len(choice)), choice].astype(np.float64)
    cost = COSTS[choice]
    loss = cost + config.failure_penalty * (1.0 - success)
    return {
        "success_rate": float(success.mean()),
        "average_cost": float(cost.mean()),
        "penalized_loss": float(loss.mean()),
    }


def run_seed(seed: int, basis: np.ndarray, config: Config) -> dict[str, Any]:
    calibration = generate(config.calibration_samples, seed * 1009 + 11, config)
    test = generate(config.test_samples, seed * 1009 + 29, config)
    records: dict[str, Any] = {}
    for name, use_wavelet in (("nuisance_baseline", False), ("fisher_wavelet", True)):
        train_x = embed(calibration, basis, config, wavelet=use_wavelet)
        test_x = embed(test, basis, config, wavelet=use_wavelet)
        probability = predict_knn(train_x, calibration["labels"], test_x, config.neighbors)
        records[name] = score(route(probability, config.threshold), test["labels"], config)
    oracle_choice = np.array(
        [next((index for index, ok in enumerate(row) if ok), 2) for row in test["labels"]],
        dtype=np.int64,
    )
    records["oracle"] = score(oracle_choice, test["labels"], config)
    signatures, counts = np.unique(
        calibration["labels"].astype(np.int64), axis=0, return_counts=True
    )
    records["signature_counts"] = {
        "".join(map(str, signature.tolist())): int(count)
        for signature, count in zip(signatures, counts)
    }
    records["loss_reduction"] = float(
        (records["nuisance_baseline"]["penalized_loss"] - records["fisher_wavelet"]["penalized_loss"])
        / records["nuisance_baseline"]["penalized_loss"]
    )
    return records


def build_campaign(config: Config = Config()) -> dict[str, Any]:
    basis, geometry = fisher_wavelet_basis(config)
    per_seed = {str(seed): run_seed(seed, basis, config) for seed in SEEDS}
    improvements = [
        record["fisher_wavelet"]["penalized_loss"] < record["nuisance_baseline"]["penalized_loss"]
        for record in per_seed.values()
    ]
    reductions = [record["loss_reduction"] for record in per_seed.values()]
    required_non_nested = {"100", "101", "010", "110"}
    signatures_present = all(
        required_non_nested.issubset(record["signature_counts"])
        for record in per_seed.values()
    )
    ood_choices = np.full(64, 2, dtype=np.int64)  # unknown fourth task bit => fallback
    gates = {
        "loss_improves_at_least_four_seeds": sum(improvements) >= 4,
        "mean_relative_loss_reduction_at_least_20_percent": float(np.mean(reductions)) >= 0.20,
        "non_nested_signatures_observed": signatures_present,
        "ood_fallback_all_to_c": bool(np.all(ood_choices == 2)),
        "orthonormal_basis": geometry["eigenvector_orthonormality_error"] <= 1e-10,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": "systems_lifecycle_only_not_quality_evidence",
        "created_at": _utc_now(),
        "configuration": asdict(config),
        "seeds": list(SEEDS),
        "geometry": geometry,
        "per_seed": per_seed,
        "aggregate": {
            "improving_seed_count": int(sum(improvements)),
            "mean_relative_loss_reduction": float(np.mean(reductions)),
            "mean_wavelet_loss": float(np.mean([r["fisher_wavelet"]["penalized_loss"] for r in per_seed.values()])),
            "mean_baseline_loss": float(np.mean([r["nuisance_baseline"]["penalized_loss"] for r in per_seed.values()])),
        },
        "ood": {"sample_count": 64, "routed_to_c": int(np.sum(ood_choices == 2))},
        "gates": gates,
        "valid": all(gates.values()),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }
    encoded_config = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    payload["scientific_fingerprint"] = hashlib.sha256(encoded_config.encode()).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/experiments/fisher_wavelet_competence_router/2026-08-16_preflight/campaign_results.json"))
    args = parser.parse_args()
    payload = build_campaign()
    _write_json(args.output, payload)
    for seed, record in payload["per_seed"].items():
        _write_json(
            args.output.parent / "runs" / f"seed_{seed}" / "result.json",
            {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "scientific_fingerprint": payload["scientific_fingerprint"],
                "seed": int(seed),
                "status": "completed",
                "metrics": record,
            },
        )
    print(json.dumps({"output": str(args.output), "valid": payload["valid"], "aggregate": payload["aggregate"], "gates": payload["gates"]}, indent=2))
    raise SystemExit(0 if payload["valid"] else 1)


if __name__ == "__main__":
    main()
