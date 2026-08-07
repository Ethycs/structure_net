#!/usr/bin/env python3
"""Scan Reynolds-barycenter Koopman closure in frozen degree-k TinyLLMs."""

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
from sklearn.neural_network import MLPRegressor
import torch

import experiments.structure_net.tinyllm_deck_action_descrambler as deck
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-reynolds-koopman.v1"
HYPOTHESIS_ID = "tinyllm-reynolds-koopman-quotient-closure-v1"
SOURCE_DECK_SCHEMA = "nal.tinyllm-deck-action-descrambler.v1"
REGIMES = ("composition", "extrapolation")
TRAIN_LAMBDAS = (0.0, 0.5, 1.0)
TEST_LAMBDAS = (0.25, 0.75, 1.25)


@dataclass(frozen=True)
class KoopmanConfig:
    source_root: str = "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    causal_root: str = "data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    degrees: tuple[int, ...] = (2, 3)
    fit_orbits: int = 384
    evaluation_orbits: int = 192
    response_orbits: int = 96
    map_points: int = 192
    first_blocks: int = 3
    barycenter_rank: int = 48
    sketch_rank: int = 24
    ridge: float = 1e-3
    activation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if set(self.degrees).difference((2, 3)):
            raise ValueError("degrees must be drawn from 2,3")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("confirmatory campaign requires five seeds")
        if min(self.fit_orbits, self.evaluation_orbits, self.response_orbits) < 8:
            raise ValueError("orbit cohorts are underpowered")
        if self.response_orbits > min(self.fit_orbits, self.evaluation_orbits):
            raise ValueError("response cohort must be a subset of fit and evaluation cohorts")
        if self.map_points < 24:
            raise ValueError("map grid is too small")


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
    value = hashlib.sha256()
    for path in (Path(__file__), Path(deck.__file__)):
        value.update(str(path).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def state_cuts(first_blocks: int) -> tuple[str, ...]:
    return deck.cut_names(first_blocks)


def one_step_transitions(first_blocks: int) -> tuple[tuple[str, str], ...]:
    transitions: list[tuple[str, str]] = []
    for index in range(first_blocks):
        pre = "block_0_pre_attention" if index == 0 else f"block_{index - 1}_post_mlp"
        attention = f"block_{index}_post_attention"
        post_mlp = f"block_{index}_post_mlp"
        transitions.extend(((pre, attention), (attention, post_mlp)))
    return tuple(transitions)


def orbit_tensor(values: np.ndarray, orbit_count: int, k: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return flat.reshape(orbit_count, k, flat.shape[-1])


def reynolds_parts(
    values: np.ndarray,
    orbit_count: int,
    k: int,
    *,
    member_permutation: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    orbit = orbit_tensor(values, orbit_count, k)
    if member_permutation is not None:
        orbit = orbit[:, member_permutation, :]
    barycenter = orbit.mean(axis=1)
    characters = []
    members = np.arange(k)
    for character in range(1, k):
        weights = np.exp(-2j * math.pi * character * members / k)
        characters.append(np.einsum("j,njd->nd", weights, orbit) / k)
    return barycenter, tuple(characters)


def scale_cover(values: np.ndarray, orbit_count: int, k: int, scale: float) -> np.ndarray:
    original_shape = values.shape
    orbit = orbit_tensor(values, orbit_count, k)
    barycenter = orbit.mean(axis=1, keepdims=True)
    scaled = barycenter + scale * (orbit - barycenter)
    return scaled.reshape(original_shape).astype(np.float32)


def _fit_pca(values: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    maximum = min(rank, centered.shape[0] - 1, centered.shape[1])
    if maximum < 1:
        raise ValueError("PCA input has no nonconstant sample direction")
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    return {"mean": mean, "basis": right[:maximum].T, "singular_values": singular[:maximum]}


def _pca_transform(model: Mapping[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    return (values - model["mean"]) @ model["basis"]


def _standardize_fit(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": values.mean(axis=0, dtype=np.float64),
        "scale": np.maximum(values.std(axis=0, dtype=np.float64), 1e-8),
    }


def _standardize(model: Mapping[str, np.ndarray], values: np.ndarray) -> np.ndarray:
    return (values - model["mean"]) / model["scale"]


def invariant_character_features(
    characters: Sequence[np.ndarray],
    sketch: np.ndarray,
    k: int,
    barycenter_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low-rank quadratic and cubic deck-invariant character features."""
    primary = characters[0] @ sketch
    if k == 2:
        real = np.real(primary)
        quadratic = np.concatenate((np.square(real), real * np.roll(real, -1, axis=1)), axis=1)
        bary = barycenter_coordinates[:, : real.shape[1]]
        if bary.shape[1] < real.shape[1]:
            bary = np.pad(bary, ((0, 0), (0, real.shape[1] - bary.shape[1])))
        cubic = quadratic[:, : real.shape[1]] * bary
    elif k == 3:
        quadratic = np.concatenate(
            (
                np.square(np.abs(primary)),
                np.real(primary * np.conj(np.roll(primary, -1, axis=1))),
            ),
            axis=1,
        )
        cubic = np.concatenate((np.real(primary**3), np.imag(primary**3)), axis=1)
    else:
        raise ValueError("only k=2,3 are supported")
    return quadratic.astype(np.float64), cubic.astype(np.float64)


class ReynoldsDictionary:
    """Frozen barycenter PCA plus low-rank invariant character products."""

    def __init__(self, k: int, barycenter_rank: int, sketch_rank: int, seed: int):
        self.k = k
        self.barycenter_rank = barycenter_rank
        self.sketch_rank = sketch_rank
        self.seed = seed
        self.barycenter_pca: dict[str, np.ndarray] | None = None
        self.sketch: np.ndarray | None = None
        self.quadratic_scaler: dict[str, np.ndarray] | None = None
        self.cubic_scaler: dict[str, np.ndarray] | None = None
        self.random_weights: np.ndarray | None = None
        self.random_bias: np.ndarray | None = None
        self.kernel_weights: np.ndarray | None = None
        self.kernel_bias: np.ndarray | None = None

    def fit(self, values: np.ndarray, orbit_count: int) -> "ReynoldsDictionary":
        barycenter, characters = reynolds_parts(values, orbit_count, self.k)
        self.barycenter_pca = _fit_pca(barycenter, self.barycenter_rank)
        bary = _pca_transform(self.barycenter_pca, barycenter)
        generator = np.random.default_rng(self.seed)
        self.sketch = generator.normal(
            0.0, 1.0 / math.sqrt(barycenter.shape[1]),
            (barycenter.shape[1], self.sketch_rank),
        )
        quadratic, cubic = invariant_character_features(characters, self.sketch, self.k, bary)
        self.quadratic_scaler = _standardize_fit(quadratic)
        self.cubic_scaler = _standardize_fit(cubic)
        added = quadratic.shape[1] + cubic.shape[1]
        raw_cover = np.concatenate(
            tuple(np.concatenate((np.real(item), np.imag(item)), axis=1) for item in characters),
            axis=1,
        )
        self.random_weights = generator.normal(
            0.0, 1.0 / math.sqrt(raw_cover.shape[1]), (raw_cover.shape[1], added)
        )
        self.random_bias = generator.uniform(-math.pi, math.pi, added)
        self.kernel_weights = generator.normal(0.0, 1.0, (bary.shape[1], added))
        self.kernel_bias = generator.uniform(0.0, 2.0 * math.pi, added)
        return self

    def transform(
        self,
        values: np.ndarray,
        orbit_count: int,
        *,
        phase_shuffle: bool = False,
        shuffled_membership: bool = False,
        shuffle_seed: int = 0,
    ) -> dict[str, np.ndarray]:
        if self.barycenter_pca is None or self.sketch is None:
            raise RuntimeError("dictionary is not fitted")
        working = values
        if shuffled_membership:
            generator = np.random.default_rng(shuffle_seed)
            flat = np.asarray(values).copy()
            working = flat[generator.permutation(len(flat))]
        barycenter, characters = reynolds_parts(working, orbit_count, self.k)
        bary = _pca_transform(self.barycenter_pca, barycenter)
        quadratic, cubic = invariant_character_features(characters, self.sketch, self.k, bary)
        quadratic = _standardize(self.quadratic_scaler, quadratic)
        cubic = _standardize(self.cubic_scaler, cubic)
        if phase_shuffle:
            generator = np.random.default_rng(shuffle_seed)
            quadratic = quadratic[generator.permutation(len(quadratic))]
            cubic = cubic[generator.permutation(len(cubic))]
        raw_cover = np.concatenate(
            tuple(np.concatenate((np.real(item), np.imag(item)), axis=1) for item in characters),
            axis=1,
        )
        random_features = np.sqrt(2.0 / self.random_weights.shape[1]) * np.cos(
            raw_cover @ self.random_weights + self.random_bias
        )
        kernel_features = np.sqrt(2.0 / self.kernel_weights.shape[1]) * np.cos(
            bary @ self.kernel_weights + self.kernel_bias
        )
        result = {
            "B": bary,
            "B+Q": np.concatenate((bary, quadratic), axis=1),
            "B+Q+C": np.concatenate((bary, quadratic, cubic), axis=1),
            "random_same_size": np.concatenate((bary, random_features), axis=1),
            "kernel_rff": np.concatenate((bary, kernel_features), axis=1),
        }
        if phase_shuffle:
            result["phase_shuffled"] = result["B+Q+C"]
        if shuffled_membership:
            result["shuffled_orbits"] = result["B+Q+C"]
        return result

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        result = {
            f"{prefix}__barycenter_mean": self.barycenter_pca["mean"],
            f"{prefix}__barycenter_basis": self.barycenter_pca["basis"],
            f"{prefix}__sketch": self.sketch,
            f"{prefix}__quadratic_mean": self.quadratic_scaler["mean"],
            f"{prefix}__quadratic_scale": self.quadratic_scaler["scale"],
            f"{prefix}__cubic_mean": self.cubic_scaler["mean"],
            f"{prefix}__cubic_scale": self.cubic_scaler["scale"],
            f"{prefix}__random_weights": self.random_weights,
            f"{prefix}__random_bias": self.random_bias,
            f"{prefix}__kernel_weights": self.kernel_weights,
            f"{prefix}__kernel_bias": self.kernel_bias,
        }
        return {name: np.asarray(value, dtype=np.float32) for name, value in result.items()}


def _ridge_fit(x: np.ndarray, targets: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    scaler = _standardize_fit(x)
    normalized = _standardize(scaler, x)
    design = np.concatenate((normalized, np.ones((len(x), 1))), axis=1)
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0]) * ridge
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(gram + regularizer, design.T @ targets)
    return {**scaler, "weights": weights}


def _ridge_predict(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    normalized = _standardize(model, x)
    design = np.concatenate((normalized, np.ones((len(x), 1))), axis=1)
    return design @ model["weights"]


def variance_r2(target: np.ndarray, predicted: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = float(np.square(target - target.mean(axis=0)).sum())
    return float(1.0 - np.square(target - predicted).sum() / max(denominator, 1e-12))


def centered_log_posterior(posteriors: np.ndarray) -> np.ndarray:
    values = np.log(np.clip(posteriors, 1e-8, 1.0))
    return values - values.mean(axis=1, keepdims=True)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def orbit_mean(values: np.ndarray, orbit_count: int, k: int) -> np.ndarray:
    return values.reshape((orbit_count, k) + values.shape[1:]).mean(axis=1)


def repeat_orbit(values: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(values[:, None, ...], k, axis=1).reshape((-1,) + values.shape[1:])


def _moment(posteriors: np.ndarray) -> np.ndarray:
    angles = 2.0 * math.pi * np.arange(posteriors.shape[1]) / posteriors.shape[1]
    complex_value = posteriors @ np.exp(1j * angles)
    return np.column_stack((np.real(complex_value), np.imag(complex_value)))


def task_prediction_metrics(
    predicted_orbit: np.ndarray,
    actual_member: np.ndarray,
    dataset: deck.OrbitDataset,
    map_predicted_orbit: np.ndarray,
    map_actual_member: np.ndarray,
    map_dataset: deck.OrbitDataset,
) -> dict[str, Any]:
    actual_orbit = orbit_mean(actual_member, dataset.orbit_count, dataset.k)
    predicted_member = repeat_orbit(predicted_orbit, dataset.k)
    map_predicted_member = repeat_orbit(map_predicted_orbit, map_dataset.k)
    predicted_map = deck.output_diagnostics(map_predicted_member, map_dataset, include_topology=False)
    actual_map = deck.output_diagnostics(map_actual_member, map_dataset, include_topology=False)
    predicted_accuracy = float(np.mean(predicted_member.argmax(axis=1) == dataset.target_bins.numpy()))
    actual_accuracy = float(np.mean(actual_member.argmax(axis=1) == dataset.target_bins.numpy()))
    return {
        "moment_r2": variance_r2(_moment(actual_orbit), _moment(predicted_orbit)),
        "exact_bin_accuracy": predicted_accuracy,
        "actual_exact_bin_accuracy": actual_accuracy,
        "accuracy_loss": actual_accuracy - predicted_accuracy,
        "map_circular_alignment": predicted_map["circular_alignment"],
        "map_winding_degree": predicted_map["winding_degree"],
        "map_sampling_resolved": predicted_map["sampling_resolved"],
        "actual_map_circular_alignment": actual_map["circular_alignment"],
        "actual_map_winding_degree": actual_map["winding_degree"],
    }


def subset_dataset(dataset: deck.OrbitDataset, orbit_count: int) -> deck.OrbitDataset:
    count = orbit_count * dataset.k
    return deck.OrbitDataset(
        input_ids=dataset.input_ids[:count],
        sensor=dataset.sensor[:count],
        calibration=dataset.calibration[:count],
        phase=dataset.phase[:count],
        quotient_phase=dataset.quotient_phase[:count],
        branch=dataset.branch[:count],
        target_posteriors=dataset.target_posteriors[:count],
        target_bins=dataset.target_bins[:count],
        orbit_count=orbit_count,
        k=dataset.k,
    )


def subset_capture(values: np.ndarray, dataset: deck.OrbitDataset, orbit_count: int) -> np.ndarray:
    return values[: orbit_count * dataset.k]


def _actual_posteriors(
    system: Any,
    dataset: deck.OrbitDataset,
    captured: Mapping[str, np.ndarray],
    config: KoopmanConfig,
    device: torch.device,
) -> np.ndarray:
    bridge = deck.DeckDescramblerConfig(
        source_root=config.source_root,
        seeds=config.seeds,
        degrees=config.degrees,
        fit_orbits=config.fit_orbits,
        evaluation_orbits=config.evaluation_orbits,
        map_points=config.map_points,
        first_blocks=config.first_blocks,
        ridge=config.ridge,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    return deck.continue_from_cut(system, dataset, "full", captured["full"], bridge, device)


def _bridge_config(config: KoopmanConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_root,
        seeds=config.seeds,
        degrees=config.degrees,
        fit_orbits=config.fit_orbits,
        evaluation_orbits=config.evaluation_orbits,
        map_points=config.map_points,
        first_blocks=config.first_blocks,
        ridge=config.ridge,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _fit_task_family(
    dictionary: ReynoldsDictionary,
    train_values: np.ndarray,
    evaluation_values: Mapping[str, np.ndarray],
    map_values: Mapping[str, np.ndarray],
    cohorts: Mapping[str, deck.OrbitDataset],
    map_cohorts: Mapping[str, deck.OrbitDataset],
    actual: Mapping[str, np.ndarray],
    map_actual: Mapping[str, np.ndarray],
    config: KoopmanConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_features = dictionary.transform(train_values, cohorts["train"].orbit_count)
    evaluation_features = {
        regime: dictionary.transform(evaluation_values[regime], cohorts[regime].orbit_count)
        for regime in REGIMES
    }
    map_features = {
        regime: dictionary.transform(map_values[regime], map_cohorts[regime].orbit_count)
        for regime in REGIMES
    }
    train_target = centered_log_posterior(
        orbit_mean(actual["train"], cohorts["train"].orbit_count, cohorts["train"].k)
    )
    names = ("B", "B+Q", "B+Q+C", "random_same_size", "kernel_rff")
    result: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        model = _ridge_fit(train_features[name], train_target, config.ridge)
        arrays.update({f"task__{name}__{key}": value for key, value in model.items()})
        evaluations = {}
        for regime in REGIMES:
            predicted = softmax(_ridge_predict(model, evaluation_features[regime][name]))
            predicted_map = softmax(_ridge_predict(model, map_features[regime][name]))
            evaluations[regime] = task_prediction_metrics(
                predicted,
                actual[regime],
                cohorts[regime],
                predicted_map,
                map_actual[regime],
                map_cohorts[regime],
            )
        result[name] = evaluations

    for control_name, control_kwargs in (
        ("phase_shuffled", {"phase_shuffle": True}),
        ("shuffled_orbits", {"shuffled_membership": True}),
    ):
        train_control = dictionary.transform(
            train_values,
            cohorts["train"].orbit_count,
            shuffle_seed=seed + 11,
            **control_kwargs,
        )[control_name]
        model = _ridge_fit(train_control, train_target, config.ridge)
        arrays.update({f"task__{control_name}__{key}": value for key, value in model.items()})
        evaluations = {}
        for regime_index, regime in enumerate(REGIMES):
            evaluated = dictionary.transform(
                evaluation_values[regime],
                cohorts[regime].orbit_count,
                shuffle_seed=seed + 101 + regime_index,
                **control_kwargs,
            )[control_name]
            mapped = dictionary.transform(
                map_values[regime],
                map_cohorts[regime].orbit_count,
                shuffle_seed=seed + 211 + regime_index,
                **control_kwargs,
            )[control_name]
            evaluations[regime] = task_prediction_metrics(
                softmax(_ridge_predict(model, evaluated)),
                actual[regime],
                cohorts[regime],
                softmax(_ridge_predict(model, mapped)),
                map_actual[regime],
                map_cohorts[regime],
            )
        result[control_name] = evaluations

    bary_scaler = _standardize_fit(train_features["B"])
    mlp = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        batch_size=min(64, len(train_target)),
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    mlp.fit(_standardize(bary_scaler, train_features["B"]), train_target)
    arrays["task__mlp__input_mean"] = bary_scaler["mean"]
    arrays["task__mlp__input_scale"] = bary_scaler["scale"]
    for index, value in enumerate(mlp.coefs_):
        arrays[f"task__mlp__coef_{index}"] = value
    for index, value in enumerate(mlp.intercepts_):
        arrays[f"task__mlp__intercept_{index}"] = value
    mlp_metrics = {}
    for regime in REGIMES:
        predicted = softmax(mlp.predict(_standardize(bary_scaler, evaluation_features[regime]["B"])))
        predicted_map = softmax(mlp.predict(_standardize(bary_scaler, map_features[regime]["B"])))
        mlp_metrics[regime] = task_prediction_metrics(
            predicted,
            actual[regime],
            cohorts[regime],
            predicted_map,
            map_actual[regime],
            map_cohorts[regime],
        )
    result["barycenter_mlp"] = mlp_metrics

    for regime in REGIMES:
        base = result["B"][regime]["moment_r2"]
        result["B+Q"][regime]["positive_cover_gain"] = max(
            0.0, result["B+Q"][regime]["moment_r2"] - base
        )
        result["B+Q+C"][regime]["positive_cover_gain"] = max(
            0.0, result["B+Q+C"][regime]["moment_r2"] - base
        )
    return result, {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}


def _fit_one_step_family(
    dictionary: ReynoldsDictionary,
    source: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    cohorts: Mapping[str, deck.OrbitDataset],
    config: KoopmanConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_features = dictionary.transform(source["train"], cohorts["train"].orbit_count)
    target_barycenter, _ = reynolds_parts(
        target["train"], cohorts["train"].orbit_count, cohorts["train"].k
    )
    target_pca = _fit_pca(target_barycenter, config.barycenter_rank)
    train_target = _pca_transform(target_pca, target_barycenter)
    arrays = {
        "target_mean": target_pca["mean"],
        "target_basis": target_pca["basis"],
    }
    result: dict[str, Any] = {}
    for name in ("B", "B+Q", "B+Q+C"):
        model = _ridge_fit(train_features[name], train_target, config.ridge)
        arrays.update({f"{name}__{key}": value for key, value in model.items()})
        metrics = {}
        for regime in REGIMES:
            features = dictionary.transform(source[regime], cohorts[regime].orbit_count)[name]
            target_bary, _ = reynolds_parts(
                target[regime], cohorts[regime].orbit_count, cohorts[regime].k
            )
            target_coordinates = _pca_transform(target_pca, target_bary)
            metrics[regime] = {
                "next_barycenter_r2": variance_r2(
                    target_coordinates, _ridge_predict(model, features)
                )
            }
        result[name] = metrics
    for regime in REGIMES:
        base = result["B"][regime]["next_barycenter_r2"]
        result["B+Q"][regime]["positive_cover_gain"] = max(
            0.0, result["B+Q"][regime]["next_barycenter_r2"] - base
        )
        result["B+Q+C"][regime]["positive_cover_gain"] = max(
            0.0, result["B+Q+C"][regime]["next_barycenter_r2"] - base
        )
    return result, {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}


def _continue_scaled(
    system: Any,
    dataset: deck.OrbitDataset,
    cut: str,
    values: np.ndarray,
    scale: float,
    config: KoopmanConfig,
    device: torch.device,
) -> np.ndarray:
    patched = scale_cover(values, dataset.orbit_count, dataset.k, scale)
    return deck.continue_from_cut(
        system, dataset, cut, patched, _bridge_config(config), device
    )


def _response_scan(
    system: Any,
    dictionary: ReynoldsDictionary,
    cut: str,
    captures: Mapping[str, np.ndarray],
    cohorts: Mapping[str, deck.OrbitDataset],
    config: KoopmanConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = config.response_orbits
    train_dataset = subset_dataset(cohorts["train"], count)
    train_values = subset_capture(captures["train"], cohorts["train"], count)
    train_x = []
    train_y = []
    for scale in TRAIN_LAMBDAS:
        scaled = scale_cover(train_values, count, train_dataset.k, scale)
        train_x.append(dictionary.transform(scaled, count)["B+Q+C"])
        posterior = _continue_scaled(
            system, train_dataset, cut, train_values, scale, config, device
        )
        train_y.append(centered_log_posterior(orbit_mean(posterior, count, train_dataset.k)))
    model = _ridge_fit(np.concatenate(train_x), np.concatenate(train_y), config.ridge)
    result = {}
    for regime in REGIMES:
        dataset = subset_dataset(cohorts[regime], count)
        values = subset_capture(captures[regime], cohorts[regime], count)
        evaluations = {}
        for scale in TEST_LAMBDAS:
            scaled = scale_cover(values, count, dataset.k, scale)
            features = dictionary.transform(scaled, count)["B+Q+C"]
            predicted = softmax(_ridge_predict(model, features))
            actual_member = _continue_scaled(
                system, dataset, cut, values, scale, config, device
            )
            actual = orbit_mean(actual_member, count, dataset.k)
            predicted_moment = _moment(predicted)
            actual_moment = _moment(actual)
            cosine = np.sum(predicted_moment * actual_moment, axis=1) / np.maximum(
                1e-12,
                np.linalg.norm(predicted_moment, axis=1) * np.linalg.norm(actual_moment, axis=1),
            )
            evaluations[str(scale)] = {
                "moment_r2": variance_r2(actual_moment, predicted_moment),
                "mean_circular_cosine": float(np.mean(cosine)),
                "actual_exact_bin_accuracy": float(
                    np.mean(actual_member.argmax(axis=1) == dataset.target_bins.numpy())
                ),
                "predicted_exact_bin_accuracy": float(
                    np.mean(repeat_orbit(predicted, dataset.k).argmax(axis=1) == dataset.target_bins.numpy())
                ),
            }
        result[regime] = evaluations
    return result, {f"response__{key}": np.asarray(value, dtype=np.float32) for key, value in model.items()}


def _harmonic_scan(
    system: Any,
    cut: str,
    values: np.ndarray,
    dataset: deck.OrbitDataset,
    config: KoopmanConfig,
    device: torch.device,
) -> dict[str, Any]:
    scales = np.asarray(TRAIN_LAMBDAS + TEST_LAMBDAS, dtype=np.float64)
    coefficients = []
    for scale in scales:
        posterior = _continue_scaled(system, dataset, cut, values, float(scale), config, device)
        angles = 2.0 * math.pi * np.arange(posterior.shape[1]) / posterior.shape[1]
        moment = posterior @ np.exp(1j * angles)
        phase = dataset.phase.numpy()
        coefficients.append(np.mean(moment * np.exp(-1j * dataset.k * phase)))
    design = np.column_stack(tuple(scales**order for order in range(4)))
    real = np.linalg.lstsq(design, np.real(coefficients), rcond=None)[0]
    imaginary = np.linalg.lstsq(design, np.imag(coefficients), rcond=None)[0]
    polynomial = real + 1j * imaginary
    magnitudes = np.abs(polynomial)
    return {
        "lambdas": scales.tolist(),
        "harmonic_coefficients_real": np.real(coefficients).tolist(),
        "harmonic_coefficients_imag": np.imag(coefficients).tolist(),
        "polynomial_coefficient_magnitudes": magnitudes.tolist(),
        "dominant_nonconstant_order": int(1 + np.argmax(magnitudes[1:])),
        "predicted_order_supported": bool(
            magnitudes[2] >= max(magnitudes[1], magnitudes[3])
            if dataset.k == 2
            else magnitudes[3] >= max(magnitudes[1], magnitudes[2])
        ),
    }


def _load_causal_source(
    config: KoopmanConfig,
    k: int,
    seed: int,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Optional[str]]]:
    path = Path(config.causal_root) / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != SOURCE_DECK_SCHEMA
        or value.get("status") != "completed"
        or int(value.get("k", -1)) != k
        or int(value.get("seed", -1)) != seed
        or value["provenance"]["checkpoint_sha256"] != checkpoint_sha256
        or not all(item["passed"] for item in value["baseline_replay_integrity"].values())
    ):
        raise ValueError(f"invalid causal comparator {path}")
    fronts: dict[str, Optional[str]] = {}
    for regime in REGIMES:
        fronts[regime] = next(
            (
                cut
                for cut in state_cuts(config.first_blocks)
                if value["cuts"][cut]["causal"][regime]["orbit_average"]["causal_classification"]
                == "preserved"
            ),
            None,
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "implementation_sha256": value["implementation_sha256"],
    }, fronts


def _task_gate(metrics: Mapping[str, Any], gain: float, k: int) -> bool:
    return bool(
        metrics["moment_r2"] >= 0.90
        and metrics["map_circular_alignment"] >= 0.90
        and metrics["map_sampling_resolved"]
        and abs(metrics["map_winding_degree"] - k) <= 0.10
        and metrics["accuracy_loss"] <= 0.05
        and gain <= 0.02
    )


def _front(
    cuts: Mapping[str, Any],
    regime: str,
    k: int,
    ordered_cuts: Sequence[str],
) -> Optional[str]:
    for cut in ordered_cuts:
        task = cuts[cut]["task_closure"]
        gain = task["B+Q+C"][regime]["positive_cover_gain"]
        if _task_gate(task["B"][regime], gain, k):
            return cut
    return None


def _front_distance(left: Optional[str], right: Optional[str], cuts: Sequence[str]) -> Optional[int]:
    if left is None or right is None:
        return None
    return abs(cuts.index(left) - cuts.index(right))


def _one_step_near_front(
    front: Optional[str],
    regime: str,
    cuts: Sequence[str],
    transitions: Mapping[str, Any],
) -> bool:
    if front is None:
        return False
    front_index = cuts.index(front)
    candidates = []
    for source, item in transitions.items():
        source_index = cuts.index(source)
        target_index = cuts.index(item["target_cut"])
        if source_index in {front_index, front_index - 1} or target_index in {front_index, front_index - 1}:
            candidates.append(item)
    return any(
        item["models"]["B"][regime]["next_barycenter_r2"] >= 0.90
        and item["models"]["B+Q+C"][regime]["positive_cover_gain"] <= 0.02
        for item in candidates
    )


def _intervention_gate(response: Mapping[str, Any], regime: str) -> bool:
    return all(
        item["moment_r2"] >= 0.80 and item["mean_circular_cosine"] >= 0.90
        for item in response[regime].values()
    )


def analyze_cell(
    task: CircleTaskConfig,
    config: KoopmanConfig,
    k: int,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    bridge = _bridge_config(config)
    system, provenance = deck.load_source(task, bridge, k, seed, device)
    causal_provenance, causal_fronts = _load_causal_source(
        config, k, seed, provenance["checkpoint_sha256"]
    )
    cohorts = {
        "train": deck.generate_exact_orbits(
            task, k=k, orbit_count=config.fit_orbits, seed=seed + 401, regime="interpolation"
        ),
        "composition": deck.generate_exact_orbits(
            task, k=k, orbit_count=config.evaluation_orbits, seed=seed + 503, regime="composition"
        ),
        "extrapolation": deck.generate_exact_orbits(
            task, k=k, orbit_count=config.evaluation_orbits, seed=seed + 607, regime="extrapolation"
        ),
    }
    captures = {
        name: deck.capture_sequences(system, dataset, bridge, device)
        for name, dataset in cohorts.items()
    }
    map_orbits = config.map_points // k
    theta_grid = np.linspace(0.0, 2.0 * math.pi, map_orbits, endpoint=False)
    map_cohorts = {
        regime: deck.generate_exact_orbits(
            task,
            k=k,
            orbit_count=map_orbits,
            seed=seed + 701 + index,
            regime=regime,
            quotient_phases=theta_grid,
            fixed_nuisance=True,
        )
        for index, regime in enumerate(REGIMES)
    }
    map_captures = {
        regime: deck.capture_sequences(system, dataset, bridge, device)
        for regime, dataset in map_cohorts.items()
    }
    actual = {
        name: _actual_posteriors(system, cohorts[name], captures[name], config, device)
        for name in cohorts
    }
    map_actual = {
        regime: _actual_posteriors(system, map_cohorts[regime], map_captures[regime], config, device)
        for regime in REGIMES
    }

    cuts: dict[str, Any] = {}
    frozen_arrays: dict[str, np.ndarray] = {}
    dictionaries: dict[str, ReynoldsDictionary] = {}
    for cut_index, cut in enumerate(state_cuts(config.first_blocks)):
        dictionary = ReynoldsDictionary(
            k, config.barycenter_rank, config.sketch_rank, seed + 10_007 * (cut_index + 1)
        ).fit(captures["train"][cut], cohorts["train"].orbit_count)
        dictionaries[cut] = dictionary
        frozen_arrays.update(dictionary.arrays(cut))
        task_models, task_arrays = _fit_task_family(
            dictionary,
            captures["train"][cut],
            {regime: captures[regime][cut] for regime in REGIMES},
            {regime: map_captures[regime][cut] for regime in REGIMES},
            cohorts,
            map_cohorts,
            actual,
            map_actual,
            config,
            seed + cut_index,
        )
        frozen_arrays.update({f"{cut}__{name}": value for name, value in task_arrays.items()})
        response, response_arrays = _response_scan(
            system,
            dictionary,
            cut,
            {name: captures[name][cut] for name in captures},
            cohorts,
            config,
            device,
        )
        frozen_arrays.update({f"{cut}__{name}": value for name, value in response_arrays.items()})
        harmonic = {
            regime: _harmonic_scan(
                system,
                cut,
                map_captures[regime][cut],
                map_cohorts[regime],
                config,
                device,
            )
            for regime in REGIMES
        }
        cuts[cut] = {
            "dictionary_dimensions": {
                name: int(value.shape[1])
                for name, value in dictionary.transform(
                    captures["train"][cut], cohorts["train"].orbit_count
                ).items()
            },
            "task_closure": task_models,
            "scaled_cover_response": response,
            "harmonic_synthesis": harmonic,
        }

    transitions: dict[str, Any] = {}
    for source, target in one_step_transitions(config.first_blocks):
        models, arrays = _fit_one_step_family(
            dictionaries[source],
            {name: captures[name][source] for name in cohorts},
            {name: captures[name][target] for name in cohorts},
            cohorts,
            config,
        )
        transitions[source] = {"target_cut": target, "models": models}
        frozen_arrays.update(
            {f"transition__{source}__to__{target}__{name}": value for name, value in arrays.items()}
        )

    ordered = state_cuts(config.first_blocks)
    koopman_fronts = {regime: _front(cuts, regime, k, ordered) for regime in REGIMES}
    front_agreement = {
        regime: {
            "causal_front": causal_fronts[regime],
            "task_closure_front": koopman_fronts[regime],
            "cut_distance": _front_distance(causal_fronts[regime], koopman_fronts[regime], ordered),
        }
        for regime in REGIMES
    }
    cover_transition = {}
    autonomous = {}
    intervention = {}
    for regime in REGIMES:
        causal = causal_fronts[regime]
        front = koopman_fronts[regime]
        causal_index = len(ordered) if causal is None else ordered.index(causal)
        front_index = len(ordered) if front is None else ordered.index(front)
        gains = {
            cut: cuts[cut]["task_closure"]["B+Q+C"][regime]["positive_cover_gain"]
            for cut in ordered
        }
        cover_transition[regime] = {
            "substantial_before_causal_front": any(
                gains[cut] >= 0.05 for cut in ordered[:causal_index]
            ),
            "negligible_from_task_front": bool(
                front is not None and all(gains[cut] <= 0.02 for cut in ordered[front_index:])
            ),
            "passed": False,
        }
        cover_transition[regime]["passed"] = bool(
            cover_transition[regime]["substantial_before_causal_front"]
            and cover_transition[regime]["negligible_from_task_front"]
        )
        autonomous[regime] = _one_step_near_front(front, regime, ordered, transitions)
        intervention[regime] = bool(
            front is not None and _intervention_gate(cuts[front]["scaled_cover_response"], regime)
        )
    gates = {
        "front_agreement_both_shifts": all(
            item["cut_distance"] is not None and item["cut_distance"] <= 1
            for item in front_agreement.values()
        ),
        "cover_transition_both_shifts": all(item["passed"] for item in cover_transition.values()),
        "autonomous_one_step_near_front_both_shifts": all(autonomous.values()),
        "unseen_lambda_intervention_both_shifts": all(intervention.values()),
    }

    root = output / "runs" / f"k{k}" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    arrays_path = root / "koopman_models.npz"
    np.savez_compressed(arrays_path, **frozen_arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-reynolds-koopman-k{k}-seed{seed}",
        "status": "completed",
        "completed_at": _utc_now(),
        "k": k,
        "seed": seed,
        "configuration": asdict(config),
        "provenance": {**provenance, "causal_comparator": causal_provenance},
        "causal_fronts": causal_fronts,
        "task_closure_fronts": koopman_fronts,
        "front_agreement": front_agreement,
        "cover_transition": cover_transition,
        "autonomous_one_step_near_front": autonomous,
        "unseen_lambda_intervention_at_front": intervention,
        "gates": gates,
        "cuts": cuts,
        "one_step_transitions": transitions,
        "models_path": str(arrays_path),
        "models_sha256": _sha256(arrays_path),
        "implementation_sha256": _implementation_digest(),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(root / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]], config: KoopmanConfig) -> dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    degrees = {}
    joint_front_positions: dict[int, list[int]] = {}
    ordered = state_cuts(config.first_blocks)
    for k in config.degrees:
        selected = [run for run in runs if int(run["k"]) == k]
        counts = {
            gate: sum(bool(run["gates"][gate]) for run in selected)
            for gate in (
                "front_agreement_both_shifts",
                "cover_transition_both_shifts",
                "autonomous_one_step_near_front_both_shifts",
                "unseen_lambda_intervention_both_shifts",
            )
        }
        positions = []
        per_seed = {}
        for run in selected:
            fronts = run["task_closure_fronts"]
            joint = None
            if all(front is not None for front in fronts.values()):
                joint = max(ordered.index(front) for front in fronts.values())
                positions.append(joint)
            per_seed[str(run["seed"])] = {
                "causal_fronts": run["causal_fronts"],
                "task_closure_fronts": fronts,
                "joint_front_index": joint,
                "gates": run["gates"],
            }
        joint_front_positions[k] = positions
        degrees[str(k)] = {
            "gate_pass_counts": counts,
            "per_seed": per_seed,
            "median_joint_front_index": None if not positions else float(np.median(positions)),
            "success": all(value >= required for value in counts.values()),
        }
    degree_ordering = bool(
        joint_front_positions.get(2)
        and joint_front_positions.get(3)
        and np.median(joint_front_positions[3]) >= np.median(joint_front_positions[2])
    )
    return {
        "degrees": degrees,
        "required_seed_count": required,
        "degree_ordering_k3_no_earlier": degree_ordering,
        "confirmed": bool(
            degree_ordering and all(degrees[str(k)]["success"] for k in config.degrees)
        ),
    }


def run_campaign(config: KoopmanConfig, output: Path) -> dict[str, Any]:
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
    campaign = {
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
            "Approximate predictive closure is tested only for supplied observables and cohorts.",
            "Depth is a nonstationary Koopman cocycle; fitted spectra are not stationary dynamics.",
            "Frontend state is not used for autonomous closure because it is not Markov-complete.",
            "Cross-seed evidence is threshold reproducibility, not hidden-coordinate weight transfer.",
            "Random sketches test a declared finite dictionary, not the maximal Koopman-invariant subspace.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=KoopmanConfig.source_root)
    parser.add_argument("--causal-root", default=KoopmanConfig.causal_root)
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--degrees", default="2,3")
    parser.add_argument("--fit-orbits", type=int, default=384)
    parser.add_argument("--evaluation-orbits", type=int, default=192)
    parser.add_argument("--response-orbits", type=int, default=96)
    parser.add_argument("--map-points", type=int, default=192)
    parser.add_argument("--first-blocks", type=int, default=3)
    parser.add_argument("--barycenter-rank", type=int, default=48)
    parser.add_argument("--sketch-rank", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_reynolds_koopman/20260806_d6_preregistered"),
    )
    args = parser.parse_args(argv)
    if args.shakedown:
        args.seeds, args.degrees = "7", "2"
        args.fit_orbits, args.evaluation_orbits, args.response_orbits = 24, 16, 8
        args.map_points, args.barycenter_rank, args.sketch_rank = 48, 12, 6
        args.allow_underpowered = True
    config = KoopmanConfig(
        source_root=args.source_root,
        causal_root=args.causal_root,
        seeds=_ints(args.seeds),
        degrees=_ints(args.degrees),
        fit_orbits=args.fit_orbits,
        evaluation_orbits=args.evaluation_orbits,
        response_orbits=args.response_orbits,
        map_points=args.map_points,
        first_blocks=args.first_blocks,
        barycenter_rank=args.barycenter_rank,
        sketch_rank=args.sketch_rank,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0 if campaign["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
