#!/usr/bin/env python3
"""Experiment 11: leakage-controlled out-of-sample competence-wavelet routing."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner

try:
    from experiments.structure_net.paws_embedding_router import COSTS, lexical_features, lower_bound, metrics, route
    from experiments.structure_net.paws_strata_wavelets import deterministic_landmarks, stratum_keys, wavelet_basis
    from experiments.structure_net.paws_teacher_annotation import eligible_rows, sha256
except ModuleNotFoundError:
    from paws_embedding_router import COSTS, lexical_features, lower_bound, metrics, route
    from paws_strata_wavelets import deterministic_landmarks, stratum_keys, wavelet_basis
    from paws_teacher_annotation import eligible_rows, sha256

SCHEMA_VERSION = "nal.paws-competence-wavelet-router.v1"
HYPOTHESIS_ID = "paws-competence-wavelet-router-followup-v1"


def build_carrier(atlas: Any, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], bool]:
    partition = atlas["partition"]
    construction = np.isin(partition, [1, 2])
    semantic_pca = PCA(n_components=16, random_state=17).fit(atlas["embedding"][construction].astype(np.float32))
    semantic = semantic_pca.transform(atlas["embedding"].astype(np.float32))
    lexical = np.stack([lexical_features(row["sentence1"], row["sentence2"]) for row in rows])
    base_scaler = StandardScaler().fit(np.concatenate((semantic, lexical), axis=1)[construction])
    base = base_scaler.transform(np.concatenate((semantic, lexical), axis=1))
    success = np.stack((atlas["a_prediction"] == atlas["labels"], atlas["b_prediction"] == atlas["labels"], atlas["c_prediction"] == atlas["labels"]), axis=1).astype(float)
    atlas_mask = partition == 1
    atlas_indices = np.flatnonzero(atlas_mask)
    neighbors = NearestNeighbors(n_neighbors=32).fit(base[atlas_mask])
    distance, index = neighbors.kneighbors(base)
    probability = np.empty((len(base), 3), dtype=np.float64)
    no_self_leakage = True
    global_to_local = {global_index: local_index for local_index, global_index in enumerate(atlas_indices)}
    for global_index in range(len(base)):
        local_indices = index[global_index]
        local_distances = distance[global_index]
        if global_index in global_to_local:
            own = global_to_local[global_index]
            no_self_leakage = no_self_leakage and bool(np.any(local_indices == own))
            keep = local_indices != own
            no_self_leakage = no_self_leakage and not np.any((local_indices[keep] == own) & (local_distances[keep] == 0))
            local_indices = local_indices[keep][:31]
            local_distances = local_distances[keep][:31]
        else:
            local_indices = local_indices[:31]
            local_distances = local_distances[:31]
        weights = 1 / (local_distances + 1e-6)
        probability[global_index] = (weights[:, None] * success[atlas_mask][local_indices]).sum(0) / weights.sum()
    probability = np.clip(probability, 1e-4, 1 - 1e-4)
    logits = np.log(probability / (1 - probability))
    transform = {
        "semantic_components": semantic_pca.components_, "semantic_mean": semantic_pca.mean_,
        "base_mean": base_scaler.mean_, "base_scale": base_scaler.scale_,
    }
    return np.concatenate((base, logits), axis=1), probability, success, transform, no_self_leakage


def extend_from_landmarks(query_x: np.ndarray, query_probability: np.ndarray, landmark_x: np.ndarray, landmark_probability: np.ndarray, basis: np.ndarray, dimensions: int = 128) -> np.ndarray:
    distance, neighbor = NearestNeighbors(n_neighbors=min(13,len(landmark_x))).fit(landmark_x).kneighbors(landmark_x)
    sigma = float(np.median(distance[:, 1:]))
    query_distance, query_neighbor = NearestNeighbors(n_neighbors=min(12,len(landmark_x))).fit(landmark_x).kneighbors(query_x)
    coordinates = np.empty((len(query_x), dimensions), dtype=np.float64)
    for query_index in range(len(query_x)):
        local = query_neighbor[query_index]
        midpoint = np.clip((query_probability[query_index] + landmark_probability[local]) / 2, 1e-4, 1 - 1e-4)
        fisher = np.sum((query_probability[query_index] - landmark_probability[local]) ** 2 / (midpoint * (1 - midpoint)), axis=1)
        affinity = np.exp(-(query_distance[query_index] / max(sigma, 1e-8)) ** 2) / (1 + fisher)
        affinity /= affinity.sum()
        coordinates[query_index] = affinity @ basis[local, :dimensions]
    return coordinates


def extend_wavelet_coordinates(x: np.ndarray, probability: np.ndarray, landmark_indices: np.ndarray, basis: np.ndarray, dimensions: int = 128) -> np.ndarray:
    landmark_x = x[landmark_indices]
    landmark_probability = probability[landmark_indices]
    coordinates = extend_from_landmarks(x, probability, landmark_x, landmark_probability, basis, dimensions)
    landmark_lookup = {int(global_index): local_index for local_index, global_index in enumerate(landmark_indices)}
    for query_index in range(len(x)):
        if query_index in landmark_lookup:
            coordinates[query_index] = basis[landmark_lookup[query_index], :dimensions]
    return coordinates


def out_of_sample_coordinates(embedding: np.ndarray, rows: list[dict[str, Any]], bundle: Any) -> np.ndarray:
    semantic = (embedding.astype(float) - bundle["semantic_mean"]) @ bundle["semantic_components"].T
    lexical = np.stack([lexical_features(row["sentence1"], row["sentence2"]) for row in rows])
    base = (np.concatenate((semantic, lexical), axis=1) - bundle["base_mean"]) / bundle["base_scale"]
    distance, index = NearestNeighbors(n_neighbors=31).fit(bundle["competence_atlas_base"]).kneighbors(base)
    weights = 1 / (distance + 1e-6)
    probability = (weights[:, :, None] * bundle["competence_atlas_success"][index]).sum(1) / weights.sum(1)[:, None]
    probability = np.clip(probability, 1e-4, 1 - 1e-4)
    carrier = np.concatenate((base, np.log(probability / (1 - probability))), axis=1)
    return extend_from_landmarks(carrier, probability, bundle["landmark_carrier"], bundle["landmark_competence_probability"], bundle["basis_128"], bundle["basis_128"].shape[1])


def estimate(coords_train: np.ndarray, success_train: np.ndarray, coords_query: np.ndarray, k: int, weighting: str) -> tuple[np.ndarray, np.ndarray]:
    distance, index = NearestNeighbors(n_neighbors=k).fit(coords_train).kneighbors(coords_query)
    weights = np.ones_like(distance) if weighting == "unweighted" else 1 / (distance + 1e-6)
    neighbor_success = success_train[index]
    probability = (weights[:, :, None] * neighbor_success).sum(1) / weights.sum(1)[:, None]
    return probability, lower_bound(neighbor_success, weights)


def select_router(coordinates: np.ndarray, success: np.ndarray, partition: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train = partition == 1
    validation = partition == 2
    candidates = []
    always_c = float(success[validation, 2].mean())
    for k in (5, 15, 31, 63):
        for weighting in ("unweighted", "inverse_distance"):
            probability, lcb = estimate(coordinates[train], success[train], coordinates[validation], k, weighting)
            for tau in (.80, .85, .90, .95):
                candidate = {"family": "competence_wavelet_knn", "k": k, "weighting": weighting, "tau": tau, **metrics(route(lcb, tau), success[validation], probability)}
                candidates.append(candidate)
    feasible = [item for item in candidates if item["accuracy"] >= always_c - .01]
    selected = sorted(feasible, key=lambda item: (item["mean_cost"], -item["accuracy"], item["ece"], json.dumps(item, sort_keys=True)))[0] if feasible else sorted(candidates, key=lambda item: (-item["accuracy"], item["mean_cost"], item["ece"]))[0]
    return selected, candidates


def evaluate(selected: dict[str, Any], coordinates: np.ndarray, success: np.ndarray, partition: np.ndarray) -> dict[str, float]:
    train = partition == 1
    test = partition == 3
    probability, lcb = estimate(coordinates[train], success[train], coordinates[test], selected["k"], selected["weighting"])
    return metrics(route(lcb, selected["tau"]), success[test], probability)


def generic_comparator(atlas: Any, router_path: Path) -> dict[str, float]:
    artifact = json.loads(router_path.read_text(encoding="utf-8"))
    selected = artifact["selected"]
    if selected["family"] != "embedding_knn":
        raise RuntimeError("Experiment 07 selected a non-embedding comparator")
    components = np.asarray(artifact["pca_components"]); pca_mean = np.asarray(artifact["pca_mean"])
    scaler_mean = np.asarray(artifact["scaler_mean"]); scaler_scale = np.asarray(artifact["scaler_scale"])
    coordinates = ((atlas["embedding"].astype(float) - pca_mean) @ components.T - scaler_mean) / scaler_scale
    success = np.stack((atlas["a_prediction"] == atlas["labels"], atlas["b_prediction"] == atlas["labels"], atlas["c_prediction"] == atlas["labels"]), axis=1).astype(float)
    return evaluate(selected, coordinates, success, atlas["partition"])


def worker(experiment: Experiment, _device_id: int) -> ExperimentResult:
    started = time.perf_counter(); p = experiment.parameters; output = Path(p["output"]); output.mkdir(parents=True, exist_ok=True)
    atlas = np.load(p["atlas"]); rows = eligible_rows(Path("data/datasets/paws-wiki/labeled/dev.csv"))
    x, competence_probability, success, transform, no_self_leakage = build_carrier(atlas, rows)
    if not no_self_leakage: raise RuntimeError("partition-1 self-neighbor leakage detected")
    keys = stratum_keys(atlas["labels"], success); construction = np.isin(atlas["partition"], [1, 2])
    landmark_indices = deterministic_landmarks(keys, construction, 512)
    basis_metrics, arrays = wavelet_basis(x, competence_probability, keys, landmark_indices)
    if basis_metrics["connected_components"] != 1 or basis_metrics["orthogonality_error"] > 1e-5: raise RuntimeError("wavelet construction gate failed")
    coordinates = extend_wavelet_coordinates(x, competence_probability, landmark_indices, arrays["basis"], 128)
    if coordinates.shape != (len(rows), 128) or not np.isfinite(coordinates).all(): raise RuntimeError("out-of-sample coordinate gate failed")
    selected, candidates = select_router(coordinates, success, atlas["partition"])
    held_out = evaluate(selected, coordinates, success, atlas["partition"])
    generic = generic_comparator(atlas, Path(p["generic_router"]))
    transform_path = output / "wavelet_router_transform.npz"
    atlas_mask=atlas["partition"]==1
    np.savez_compressed(transform_path, **{key: value.astype(np.float32) for key, value in transform.items()}, competence_atlas_base=x[atlas_mask,:21].astype(np.float32), competence_atlas_success=success[atlas_mask].astype(np.float32), landmark_indices=landmark_indices, landmark_carrier=x[landmark_indices].astype(np.float32), landmark_competence_probability=competence_probability[landmark_indices].astype(np.float32), basis_128=arrays["basis"][:, :128].astype(np.float32))
    payload = {
        "schema_version": SCHEMA_VERSION, "complete": True, "evidence_class": "held-out development follow-up",
        "no_self_leakage": no_self_leakage, "coordinate_shape": list(coordinates.shape), "basis_metrics": basis_metrics,
        "selected": selected, "partition_3_wavelet": held_out, "partition_3_generic": generic,
        "accuracy_delta": held_out["accuracy"] - generic["accuracy"], "cost_delta": held_out["mean_cost"] - generic["mean_cost"],
        "candidates": candidates, "atlas_sha256": sha256(Path(p["atlas"])), "generic_router_sha256": sha256(Path(p["generic_router"])),
        "transform_sha256": sha256(transform_path),
    }
    result_path = output / "result.json"; result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExperimentResult(experiment_id=experiment.id, hypothesis_id=HYPOTHESIS_ID, metrics={"accuracy": held_out["accuracy"], "mean_cost": held_out["mean_cost"], "accuracy_delta": payload["accuracy_delta"], "cost_delta": payload["cost_delta"], "ece": held_out["ece"]}, primary_metric=held_out["accuracy"], model_architecture=[24, 128, 3], model_parameters=0, training_time=time.perf_counter() - started, observations=[f"result={result_path}", f"transform={transform_path}"])


async def run(output: Path) -> dict[str, Any]:
    parameters = {"output": str(output), "atlas": "data/experiments/paws_abc_routing/2026-08-16_experiment_06/competence_atlas.npz", "generic_router": "data/experiments/paws_abc_routing/2026-08-16_experiment_07/router.json"}
    experiment = Experiment(id="paws-competence-wavelet-router", hypothesis_id=HYPOTHESIS_ID, name="PAWS competence-wavelet router", parameters=parameters)
    runner = AsyncExperimentRunner(LabConfig(project_name="paws_competence_wavelet_router", results_dir=str(output / "nal"), device_ids=[-1], max_parallel_experiments=1, min_experiments_per_hypothesis=1, require_statistical_significance=False, enable_wandb=False), worker)
    result = (await runner.run_experiments([experiment]))[0]
    return {"complete": result.error is None, "metrics": result.metrics, "error": result.error}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("data/experiments/paws_abc_routing/2026-08-16_experiment_11")); args = parser.parse_args()
    result = asyncio.run(run(args.output)); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__": main()
