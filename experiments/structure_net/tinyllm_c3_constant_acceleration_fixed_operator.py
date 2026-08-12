#!/usr/bin/env python3
"""Preflight fixed C3 group operators under constant angular acceleration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_temporal_fixed_operator_ceiling as fixed
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as source
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint


SCHEMA_VERSION = "nal.tinyllm-c3-constant-acceleration-fixed-operator.v1"
HYPOTHESIS_ID = "tinyllm-c3-constant-acceleration-fixed-operator-v1"
EVIDENCE_ROLE = "prospective_no_training_dynamics_scope_preflight"
SEEDS = (107, 127, 149, 173, 197)
REGIMES = ("composition", "extrapolation")
OPERATORS = ("constant_speed_mean", "recent_degree2", "all_frame_degree2")
SAMPLE_COUNT = 4_096
DATASET_SEED_BASES = {"composition": 811_107, "extrapolation": 813_107}
SHUFFLE_SEED_BASES = {"composition": 821_107, "extrapolation": 823_107}
REGIME_RANGES = {
    "composition": {
        "velocity": (0.04, 0.12),
        "acceleration": (0.010, 0.025),
        "amplitude": (0.7, 1.8),
        "offset": (-0.4, 0.4),
        "drift": (-0.06, 0.06),
    },
    "extrapolation": {
        "velocity": (0.13, 0.20),
        "acceleration": (0.026, 0.050),
        "amplitude": (0.5, 2.2),
        "offset": (-0.7, 0.7),
        "drift": (-0.10, 0.10),
    },
}
ACTION_ERROR_MAXIMUM = 2e-12
CONTINUOUS_ERROR_MAXIMUM = 1e-12
SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM = 0.10
SHUFFLED_RMSE_MINIMUM = 0.80
FIXED_CEILING_RMSE_MAXIMUM = 0.020
FIXED_CEILING_ACCURACY_MINIMUM = 0.90
MATERIAL_RMSE_RATIO_MAXIMUM = 0.75
MATERIAL_ACCURACY_DELTA_MINIMUM = -0.005
MATERIAL_CROSS_ENTROPY_DELTA_MAXIMUM = 0.005
REQUIRED_SEED_PASSES = 4
TASK_GATES = {
    "posterior_mean_correlation_minimum": 0.90,
    "composition_accuracy_minimum": 0.50,
    "extrapolation_accuracy_minimum": 0.35,
    "composition_cross_entropy_maximum": 1.80,
    "extrapolation_cross_entropy_maximum": 2.20,
    "composition_coverage_minimum": 14,
    "extrapolation_coverage_minimum": 12,
}
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-constant-acceleration-fixed-operator-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "ae4e15e88fc16ec3cd1cc3a52724e042402e80348af6057df5b65b4719c4ee7b"
)
GENERATOR_SHA256 = "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
INTERVAL_SHA256 = "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
FIXED_OPERATOR_RUNNER_SHA256 = (
    "9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37"
)
FIXED_OPERATOR_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)
FIXED_OPERATOR_RESULT_SHA256 = (
    "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
)
FEATURE_TRAJECTORY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_feature_trajectory_causal/"
    "20260811_d6_preregistered/result.json"
)
FEATURE_TRAJECTORY_RESULT_SHA256 = (
    "a0ac3315b03aa65df273539a24d8c08f51f12e8ceb702859cc16a282886ddf27"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_constant_acceleration_fixed_operator/"
    "20260811_preregistered/result.json"
)


@dataclass(frozen=True)
class AccelerationDataset:
    canonical_tokens: torch.Tensor
    tokens: torch.Tensor
    calibration: torch.Tensor
    target: torch.Tensor
    phase: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    deck: torch.Tensor
    saturation_count: int
    dataset_seed: int

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def validate_sources() -> dict[str, str]:
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "generator": _sha256(Path(source.__file__)),
        "interval_likelihood": _sha256(Path(joint.__file__)),
        "fixed_operator_runner": _sha256(Path(fixed.__file__)),
        "fixed_operator_result": _sha256(FIXED_OPERATOR_RESULT_PATH),
        "feature_trajectory_result": _sha256(FEATURE_TRAJECTORY_RESULT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "generator": GENERATOR_SHA256,
        "interval_likelihood": INTERVAL_SHA256,
        "fixed_operator_runner": FIXED_OPERATOR_RUNNER_SHA256,
        "fixed_operator_result": FIXED_OPERATOR_RESULT_SHA256,
        "feature_trajectory_result": FEATURE_TRAJECTORY_RESULT_SHA256,
    }
    fixed_result = json.loads(
        FIXED_OPERATOR_RESULT_PATH.read_text(encoding="utf-8")
    )
    trajectory_result = json.loads(
        FEATURE_TRAJECTORY_RESULT_PATH.read_text(encoding="utf-8")
    )
    if (
        hashes != expected
        or fixed_result.get("status") != "completed"
        or fixed_result.get("aggregates", {}).get("classification")
        != "fixed_multistep_group_operator_dominates_last_step"
        or trajectory_result.get("status") != "completed"
        or trajectory_result.get("aggregates", {}).get("classification")
        != (
            "fixed_operator_available_but_frozen_continuation_cannot_use_"
            "projected_trajectory"
        )
    ):
        raise RuntimeError("constant-acceleration source contract changed")
    return hashes


def _uniform(
    count: int,
    bounds: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.empty(count, dtype=torch.float64).uniform_(
        *bounds, generator=generator
    )


def _signed_uniform(
    count: int,
    bounds: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    magnitude = _uniform(count, bounds, generator)
    sign = (
        2.0
        * torch.randint(
            0, 2, (count,), dtype=torch.int64, generator=generator
        ).double()
        - 1.0
    )
    return magnitude * sign


def _continuous_observation(
    phase: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    amplitude: torch.Tensor,
    offset: torch.Tensor,
    drift: torch.Tensor,
    *,
    time_steps: int = source.TIME_STEPS,
) -> torch.Tensor:
    time = torch.arange(time_steps, dtype=torch.float64)[None, :, None]
    channel = (
        2.0
        * math.pi
        * torch.arange(source.CHANNELS, dtype=torch.float64)[None, None, :]
        / source.CHANNELS
    )
    angle = (
        phase[:, None, None]
        + velocity[:, None, None] * time
        + 0.5 * acceleration[:, None, None] * time * (time - 1.0)
        + channel
    )
    return (
        amplitude[:, None, None] * torch.cos(angle)
        + offset[:, None, None]
        + drift[:, None, None] * time
    )


def generate_dataset(
    regime: str,
    replicate_seed: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> AccelerationDataset:
    registered = replicate_seed in SEEDS and sample_count == SAMPLE_COUNT
    pilot = allow_pilot and replicate_seed == 991 and sample_count == 64
    if regime not in REGIMES or not (registered or pilot):
        raise ValueError(f"unknown acceleration cohort {regime}/{replicate_seed}")
    if sample_count < 2:
        raise ValueError("sample count must permit a derangement")
    ranges = REGIME_RANGES[regime]
    dataset_seed = DATASET_SEED_BASES[regime] + replicate_seed
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    phase = 2.0 * math.pi * torch.rand(
        sample_count, dtype=torch.float64, generator=generator
    )
    velocity = _signed_uniform(
        sample_count, ranges["velocity"], generator
    )
    acceleration = _signed_uniform(
        sample_count, ranges["acceleration"], generator
    )
    amplitude = _uniform(sample_count, ranges["amplitude"], generator)
    offset = _uniform(sample_count, ranges["offset"], generator)
    drift = _uniform(sample_count, ranges["drift"], generator)
    deck = torch.randint(
        0, source.CHANNELS, (sample_count,), generator=generator
    )
    continuous = _continuous_observation(
        phase, velocity, acceleration, amplitude, offset, drift
    )
    saturation_count = int(
        (continuous.abs() >= source.QUANTIZATION_LIMIT).sum()
    )
    canonical = source.quantize(continuous)
    tokens = torch.stack(
        [
            source.apply_deck_action(row, int(element))
            for row, element in zip(canonical, deck)
        ]
    )
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    future_time = float(source.TIME_STEPS)
    future_phase = (
        phase
        + velocity * future_time
        + 0.5 * acceleration * future_time * (future_time - 1.0)
    )
    target = torch.cos(3.0 * future_phase)
    return AccelerationDataset(
        canonical_tokens=canonical,
        tokens=tokens,
        calibration=calibration,
        target=target,
        phase=phase,
        velocity=velocity,
        acceleration=acceleration,
        deck=deck,
        saturation_count=saturation_count,
        dataset_seed=dataset_seed,
    )


def dataset_hash(dataset: AccelerationDataset) -> str:
    return _tensor_hash(
        dataset.canonical_tokens,
        dataset.tokens,
        dataset.calibration,
        dataset.target,
        dataset.phase,
        dataset.velocity,
        dataset.acceleration,
        dataset.deck,
    )


def continuous_carrier(
    dataset: AccelerationDataset,
    *,
    include_future: bool = False,
) -> torch.Tensor:
    steps = source.TIME_STEPS + int(include_future)
    time = torch.arange(steps, dtype=torch.float64)[None]
    phase = (
        dataset.phase[:, None]
        + dataset.velocity[:, None] * time
        + 0.5 * dataset.acceleration[:, None] * time * (time - 1.0)
    )
    return torch.polar(torch.ones_like(phase), 3.0 * phase).to(torch.complex128)


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.abs().clamp_min(1e-12)


def operator_complex_predictions(
    carrier: torch.Tensor,
) -> dict[str, torch.Tensor]:
    increments = carrier[:, 1:] * carrier[:, :-1].conj()
    accelerations = increments[:, 1:] * increments[:, :-1].conj()
    constant_step = _normalize(increments.sum(dim=1))
    recent_acceleration = accelerations[:, -1]
    mean_acceleration = _normalize(accelerations.sum(dim=1))
    increment_indices = torch.arange(
        1, carrier.shape[1], dtype=torch.int64, device=carrier.device
    )
    distance_to_final = (carrier.shape[1] - 1) - increment_indices
    final_increment_candidates = increments * mean_acceleration[:, None].pow(
        distance_to_final[None]
    )
    mean_final_increment = _normalize(final_increment_candidates.sum(dim=1))
    return {
        "constant_speed_mean": carrier[:, -1] * constant_step,
        "recent_degree2": (
            carrier[:, -1] * increments[:, -1] * recent_acceleration
        ),
        "all_frame_degree2": (
            carrier[:, -1] * mean_final_increment * mean_acceleration
        ),
    }


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.reshape(-1).double()
    y = right.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x * y).sum() / denominator.clamp_min(1e-12))


def _scalar_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = prediction.double() - target.double()
    return {
        "correlation": _correlation(prediction, target),
        "rmse": float(torch.sqrt(difference.square().mean())),
        "maximum_absolute_error": float(difference.abs().max()),
    }


def _task_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    posterior = joint.interval_posterior_unclipped(prediction.double(), 16)
    target_posterior = joint.interval_posterior_unclipped(target.double(), 16)
    centers = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    predicted_scalar = posterior @ centers
    predicted_bins = posterior.argmax(-1)
    target_bins = target_posterior.argmax(-1)
    return {
        "posterior_mean_correlation": _correlation(predicted_scalar, target),
        "posterior_mean_rmse": float(
            torch.sqrt((predicted_scalar - target.double()).square().mean())
        ),
        "exact_bin_accuracy": float(
            (predicted_bins == target_bins).double().mean()
        ),
        "target_cross_entropy": float(
            -(target_posterior * posterior.clamp_min(1e-12).log()).sum(-1).mean()
        ),
        "predicted_bin_coverage": int(torch.unique(predicted_bins).numel()),
    }


def _task_gate(metrics: Mapping[str, Any], regime: str) -> bool:
    return bool(
        float(metrics["posterior_mean_correlation"])
        >= TASK_GATES["posterior_mean_correlation_minimum"]
        and float(metrics["exact_bin_accuracy"])
        >= TASK_GATES[f"{regime}_accuracy_minimum"]
        and float(metrics["target_cross_entropy"])
        <= TASK_GATES[f"{regime}_cross_entropy_maximum"]
        and int(metrics["predicted_bin_coverage"])
        >= TASK_GATES[f"{regime}_coverage_minimum"]
    )


def _fixed_ceiling_gate(
    scalar: Mapping[str, Any],
    task: Mapping[str, Any],
    regime: str,
) -> bool:
    return bool(
        float(scalar["rmse"]) <= FIXED_CEILING_RMSE_MAXIMUM
        and float(task["exact_bin_accuracy"])
        >= FIXED_CEILING_ACCURACY_MINIMUM
        and _task_gate(task, regime)
    )


def _group_contracts(dataset: AccelerationDataset) -> dict[str, Any]:
    identity_errors = int(
        (source.apply_deck_action(dataset.canonical_tokens, 0)
         != dataset.canonical_tokens).sum()
    )
    composition_errors = 0
    for left in range(source.CHANNELS):
        for right in range(source.CHANNELS):
            composed = source.apply_deck_action(
                source.apply_deck_action(dataset.canonical_tokens, right), left
            )
            direct = source.apply_deck_action(
                dataset.canonical_tokens, (left + right) % source.CHANNELS
            )
            composition_errors += int((composed != direct).sum())
    order_three = dataset.canonical_tokens
    for _ in range(3):
        order_three = source.apply_deck_action(order_three, 1)
    stored = torch.stack(
        [
            source.apply_deck_action(row, int(element))
            for row, element in zip(dataset.canonical_tokens, dataset.deck)
        ]
    )
    amplitude, offset, drift = (
        dataset.calibration[:, index] for index in range(3)
    )
    shifted_phase = dataset.phase - (
        2.0 * math.pi * dataset.deck.double() / source.CHANNELS
    )
    regenerated = source.quantize(
        _continuous_observation(
            shifted_phase,
            dataset.velocity,
            dataset.acceleration,
            amplitude,
            offset,
            drift,
        )
    )
    maximum_target_error = 0.0
    time = float(source.TIME_STEPS)
    for element in range(source.CHANNELS):
        shifted_future = (
            dataset.phase
            - 2.0 * math.pi * element / source.CHANNELS
            + dataset.velocity * time
            + 0.5 * dataset.acceleration * time * (time - 1.0)
        )
        maximum_target_error = max(
            maximum_target_error,
            float((torch.cos(3.0 * shifted_future) - dataset.target).abs().max()),
        )
    values = {
        "identity_token_errors": identity_errors,
        "composition_token_errors": composition_errors,
        "order_three_token_errors": int((order_three != dataset.canonical_tokens).sum()),
        "stored_action_token_errors": int((stored != dataset.tokens).sum()),
        "latent_regeneration_token_errors": int((regenerated != dataset.tokens).sum()),
        "maximum_target_action_error": maximum_target_error,
    }
    values["pass"] = bool(
        sum(
            values[name]
            for name in (
                "identity_token_errors",
                "composition_token_errors",
                "order_three_token_errors",
                "stored_action_token_errors",
                "latent_regeneration_token_errors",
            )
        )
        == 0
        and maximum_target_error <= ACTION_ERROR_MAXIMUM
    )
    return values


def analyze_cell(
    regime: str,
    replicate_seed: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    dataset = generate_dataset(
        regime,
        replicate_seed,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    first_hash = dataset_hash(dataset)
    repeated_hash = dataset_hash(
        generate_dataset(
            regime,
            replicate_seed,
            sample_count=sample_count,
            allow_pilot=allow_pilot,
        )
    )
    group_contracts = _group_contracts(dataset)
    carrier, magnitude = source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    complex_predictions = operator_complex_predictions(carrier)
    predictions = {
        name: value.real for name, value in complex_predictions.items()
    }
    exact_carrier = continuous_carrier(dataset)
    exact_future = continuous_carrier(dataset, include_future=True)[:, -1]
    exact_predictions = operator_complex_predictions(exact_carrier)
    continuous_errors = {
        name: float((prediction - exact_future).abs().max())
        for name, prediction in exact_predictions.items()
    }
    action_errors: dict[str, dict[str, float]] = {
        name: {} for name in OPERATORS
    }
    for element in (1, 2):
        transformed_carrier, _ = source.analytic_carrier(
            source.apply_deck_action(dataset.tokens, element),
            dataset.calibration,
        )
        transformed = operator_complex_predictions(transformed_carrier)
        for name in OPERATORS:
            action_errors[name][str(element)] = float(
                (transformed[name] - complex_predictions[name]).abs().max()
            )
    permutation = fixed.sattolo_derangement(
        len(dataset), SHUFFLE_SEED_BASES[regime] + replicate_seed
    )
    shuffled_target = dataset.target[permutation]
    operators = {}
    for name in OPERATORS:
        prediction = predictions[name]
        scalar = _scalar_metrics(prediction, dataset.target)
        task = _task_metrics(prediction, dataset.target)
        shuffled_scalar = _scalar_metrics(prediction, shuffled_target)
        shuffled_task = _task_metrics(prediction, shuffled_target)
        maximum_action_error = max(action_errors[name].values())
        operators[name] = {
            "scalar": scalar,
            "task": task,
            "task_pass": _task_gate(task, regime),
            "fixed_ceiling_pass": _fixed_ceiling_gate(scalar, task, regime),
            "deck_action_maximum_errors": action_errors[name],
            "maximum_deck_action_error": maximum_action_error,
            "action_pass": maximum_action_error <= ACTION_ERROR_MAXIMUM,
            "shuffled_target_scalar": shuffled_scalar,
            "shuffled_target_task": shuffled_task,
            "shuffled_task_pass": _task_gate(shuffled_task, regime),
            "shuffle_pass": bool(
                abs(shuffled_scalar["correlation"])
                <= SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM
                and shuffled_scalar["rmse"] >= SHUFFLED_RMSE_MINIMUM
                and not _task_gate(shuffled_task, regime)
            ),
        }
    recent = operators["recent_degree2"]
    candidate = operators["all_frame_degree2"]
    comparison = {
        "rmse_ratio": (
            candidate["scalar"]["rmse"]
            / max(recent["scalar"]["rmse"], 1e-18)
        ),
        "accuracy_delta": (
            candidate["task"]["exact_bin_accuracy"]
            - recent["task"]["exact_bin_accuracy"]
        ),
        "cross_entropy_delta": (
            candidate["task"]["target_cross_entropy"]
            - recent["task"]["target_cross_entropy"]
        ),
    }
    comparison["material_dominance_pass"] = bool(
        candidate["fixed_ceiling_pass"]
        and comparison["rmse_ratio"] <= MATERIAL_RMSE_RATIO_MAXIMUM
        and comparison["accuracy_delta"] >= MATERIAL_ACCURACY_DELTA_MINIMUM
        and comparison["cross_entropy_delta"]
        <= MATERIAL_CROSS_ENTROPY_DELTA_MAXIMUM
    )
    validity = bool(
        dataset.saturation_count == 0
        and first_hash == repeated_hash
        and group_contracts["pass"]
        and int((permutation == torch.arange(len(dataset))).sum()) == 0
        and continuous_errors["recent_degree2"] <= CONTINUOUS_ERROR_MAXIMUM
        and continuous_errors["all_frame_degree2"] <= CONTINUOUS_ERROR_MAXIMUM
        and all(operator["action_pass"] for operator in operators.values())
        and all(operator["shuffle_pass"] for operator in operators.values())
        and float(magnitude.min()) > 1e-12
        and _finite(operators)
    )
    return {
        "seed": replicate_seed,
        "regime": regime,
        "dataset_seed": dataset.dataset_seed,
        "dataset_sha256": first_hash,
        "dataset_deterministic": first_hash == repeated_hash,
        "sample_count": len(dataset),
        "saturation_count": dataset.saturation_count,
        "minimum_character_magnitude": float(magnitude.min()),
        "group_contracts": group_contracts,
        "continuous_operator_maximum_errors": continuous_errors,
        "shuffle_fixed_points": int(
            (permutation == torch.arange(len(dataset))).sum()
        ),
        "operators": operators,
        "all_frame_vs_recent": comparison,
        "valid": validity,
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    def seed_count(arm: str, field: str) -> int:
        return sum(
            all(
                bool(
                    next(
                        cell
                        for cell in cells
                        if cell["seed"] == seed and cell["regime"] == regime
                    )["operators"][arm][field]
                )
                for regime in REGIMES
            )
            for seed in SEEDS
        )

    constant_count = seed_count("constant_speed_mean", "fixed_ceiling_pass")
    recent_count = seed_count("recent_degree2", "fixed_ceiling_pass")
    all_frame_count = seed_count("all_frame_degree2", "fixed_ceiling_pass")
    material_count = sum(
        all(
            bool(
                next(
                    cell
                    for cell in cells
                    if cell["seed"] == seed and cell["regime"] == regime
                )["all_frame_vs_recent"]["material_dominance_pass"]
            )
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    if not valid:
        classification = "invalid_constant_acceleration_preflight"
    elif (
        all_frame_count >= REQUIRED_SEED_PASSES
        and material_count >= REQUIRED_SEED_PASSES
    ):
        classification = "fixed_all_frame_degree2_closes_constant_acceleration"
    elif max(recent_count, all_frame_count) >= REQUIRED_SEED_PASSES:
        classification = "fixed_degree2_closes_without_multistep_dominance"
    else:
        classification = "registered_fixed_degree2_family_insufficient"
    return {
        "classification": classification,
        "valid": valid,
        "constant_speed_fixed_ceiling_seed_pass_count": constant_count,
        "recent_degree2_fixed_ceiling_seed_pass_count": recent_count,
        "all_frame_degree2_fixed_ceiling_seed_pass_count": all_frame_count,
        "all_frame_material_dominance_seed_pass_count": material_count,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "constant_acceleration_tinyllm_training_licensed": False,
        "robust_typed_estimator_preflight_licensed": (
            classification == "registered_fixed_degree2_family_insufficient"
        ),
    }


def build_result() -> dict[str, Any]:
    source_hashes = validate_sources()
    cells = [
        analyze_cell(regime, seed)
        for seed in SEEDS
        for regime in REGIMES
    ]
    valid = len(cells) == 10 and all(cell["valid"] for cell in cells)
    aggregates = classify(cells, valid)
    source_hashes["runner"] = _sha256(Path(__file__))
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if valid else "invalid",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "source_hashes": source_hashes,
        "configuration": {
            "seeds": list(SEEDS),
            "regimes": list(REGIMES),
            "operators": list(OPERATORS),
            "sample_count_per_cell": SAMPLE_COUNT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "regime_ranges": REGIME_RANGES,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "fixed_ceiling_rmse_maximum": FIXED_CEILING_RMSE_MAXIMUM,
            "fixed_ceiling_accuracy_minimum": FIXED_CEILING_ACCURACY_MINIMUM,
            "material_rmse_ratio_maximum": MATERIAL_RMSE_RATIO_MAXIMUM,
            "material_accuracy_delta_minimum": MATERIAL_ACCURACY_DELTA_MINIMUM,
            "material_cross_entropy_delta_maximum": (
                MATERIAL_CROSS_ENTROPY_DELTA_MAXIMUM
            ),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": TASK_GATES,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "models_instantiated": 0,
            "checkpoints_loaded": 0,
            "target_using_fits": 0,
            "new_evaluation_examples": len(cells) * SAMPLE_COUNT,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "The temporal-law change and all three fixed operators were registered before primary data generation.",
            "The study tests calibrated noiseless constant acceleration only.",
            "No operator coefficient, frame subset, phase unwrap, or threshold was fitted.",
            "Failure of the registered fixed family would license an estimator preflight, not unrestricted TinyLLM.",
        ],
    }
    if not _finite(result):
        raise RuntimeError("non-finite constant-acceleration result")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY_RESULT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_result()
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "recent_degree2_ceiling_seeds": result["aggregates"][
                    "recent_degree2_fixed_ceiling_seed_pass_count"
                ],
                "all_frame_degree2_ceiling_seeds": result["aggregates"][
                    "all_frame_degree2_fixed_ceiling_seed_pass_count"
                ],
                "all_frame_material_seeds": result["aggregates"][
                    "all_frame_material_dominance_seed_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
