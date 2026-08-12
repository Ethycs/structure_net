#!/usr/bin/env python3
"""Test whether a fixed all-frame C3 operator exhausts continuation utility."""

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

import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as source
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-fixed-operator-ceiling.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-fixed-operator-ceiling-v1"
EVIDENCE_ROLE = "prospective_no_training_group_operator_comparison"
SEEDS = (7, 17, 29, 41, 53)
REGIMES = source.REGIMES
SAMPLE_COUNT = 4_096
DATASET_SEED_BASES = {"composition": 531_000, "extrapolation": 533_000}
SHUFFLE_SEED_BASE = 631_000
ACTION_ERROR_MAXIMUM = 2e-12
CONTINUOUS_ERROR_MAXIMUM = 1e-12
SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM = 0.10
SHUFFLED_RMSE_MINIMUM = 0.80
MATERIAL_RMSE_RATIO_MAXIMUM = 0.75
MATERIAL_ACCURACY_DELTA_MINIMUM = -0.005
MATERIAL_CROSS_ENTROPY_DELTA_MAXIMUM = 0.005
BASELINE_CEILING_RMSE_MAXIMUM = 0.01
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
    "2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "07d54cb5b4d65b080fe59a5e06d0853ee59a7d0132fc04faa68f303775a2f8bf"
)
GENERATOR_SHA256 = "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
INTERVAL_SHA256 = "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
SENSOR_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_only/"
    "20260811_preregistered/campaign_results.json"
)
SENSOR_CAMPAIGN_SHA256 = (
    "4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012"
)
MECHANISM_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_mechanism/"
    "20260811_preregistered/campaign_results.json"
)
MECHANISM_RESULT_SHA256 = (
    "c3dbfecd7a6381c2129e4d99f135557f003ad8a225fa2b4d3f4fa0cb429f669b"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)


@dataclass(frozen=True)
class CeilingDataset:
    canonical_tokens: torch.Tensor
    tokens: torch.Tensor
    calibration: torch.Tensor
    target: torch.Tensor
    phase: torch.Tensor
    speed: torch.Tensor
    deck: torch.Tensor
    saturation_count: int
    dataset_seed: int


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
        "sensor_campaign": _sha256(SENSOR_CAMPAIGN_PATH),
        "sensor_mechanism_result": _sha256(MECHANISM_RESULT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "generator": GENERATOR_SHA256,
        "interval_likelihood": INTERVAL_SHA256,
        "sensor_campaign": SENSOR_CAMPAIGN_SHA256,
        "sensor_mechanism_result": MECHANISM_RESULT_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"fixed-operator source contract changed: {hashes}")
    campaign = json.loads(SENSOR_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    mechanism = json.loads(MECHANISM_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        campaign.get("status") != "completed"
        or campaign.get("aggregates", {}).get("classification")
        != "task_only_sensor_acquisition_supported"
        or mechanism.get("status") != "completed"
        or mechanism.get("aggregates", {}).get("classification")
        != "affine_identity_character_carries_learned_solution"
    ):
        raise RuntimeError("fixed-operator predecessor result changed")
    return hashes


def _uniform(
    count: int,
    bounds: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.empty(count, dtype=torch.float64).uniform_(
        *bounds, generator=generator
    )


def generate_dataset(regime: str, replicate_seed: int) -> CeilingDataset:
    if regime not in REGIMES or replicate_seed not in SEEDS:
        raise ValueError(f"unknown ceiling cohort {regime}/{replicate_seed}")
    ranges = source.REGIME_RANGES[regime]
    dataset_seed = DATASET_SEED_BASES[regime] + replicate_seed
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    phase = 2.0 * math.pi * torch.rand(
        SAMPLE_COUNT, dtype=torch.float64, generator=generator
    )
    speed_magnitude = _uniform(SAMPLE_COUNT, ranges["speed"], generator)
    speed_sign = (
        2.0
        * torch.randint(
            0, 2, (SAMPLE_COUNT,), generator=generator, dtype=torch.int64
        ).double()
        - 1.0
    )
    speed = speed_magnitude * speed_sign
    amplitude = _uniform(SAMPLE_COUNT, ranges["amplitude"], generator)
    offset = _uniform(SAMPLE_COUNT, ranges["offset"], generator)
    drift = _uniform(SAMPLE_COUNT, ranges["drift"], generator)
    deck = torch.randint(
        0, source.CHANNELS, (SAMPLE_COUNT,), generator=generator
    )
    continuous = source._continuous_observation(
        phase, speed, amplitude, offset, drift
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
    target = torch.cos(3.0 * (phase + source.TIME_STEPS * speed))
    return CeilingDataset(
        canonical_tokens=canonical,
        tokens=tokens,
        calibration=calibration,
        target=target,
        phase=phase,
        speed=speed,
        deck=deck,
        saturation_count=saturation_count,
        dataset_seed=dataset_seed,
    )


def dataset_hash(dataset: CeilingDataset) -> str:
    return _tensor_hash(
        dataset.canonical_tokens,
        dataset.tokens,
        dataset.calibration,
        dataset.target,
        dataset.phase,
        dataset.speed,
        dataset.deck,
    )


def mean_increment_prediction(carrier: torch.Tensor) -> torch.Tensor:
    increments = carrier[:, 1:] * carrier[:, :-1].conj()
    total = increments.sum(dim=1)
    mean = total / total.abs().clamp_min(1e-12)
    return (carrier[:, -1] * mean).real


def continuous_carrier(dataset: CeilingDataset) -> torch.Tensor:
    time = torch.arange(source.TIME_STEPS, dtype=torch.float64)[None]
    phase = dataset.phase[:, None] + dataset.speed[:, None] * time
    return torch.polar(torch.ones_like(phase), 3.0 * phase).to(torch.complex128)


def sattolo_derangement(count: int, seed: int) -> torch.Tensor:
    if count < 2:
        raise ValueError("derangement requires at least two elements")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.arange(count)
    for index in range(count - 1, 0, -1):
        other = int(torch.randint(0, index, (), generator=generator))
        saved = int(permutation[index])
        permutation[index] = permutation[other]
        permutation[other] = saved
    return permutation


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.reshape(-1).double()
    y = right.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x * y).sum() / denominator.clamp_min(1e-12))


def _scalar_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "correlation": _correlation(prediction, target),
        "rmse": float(
            torch.sqrt((prediction.double() - target.double()).square().mean())
        ),
        "maximum_absolute_error": float(
            (prediction.double() - target.double()).abs().max()
        ),
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
            -(
                target_posterior
                * posterior.clamp_min(1e-12).log()
            ).sum(-1).mean()
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


def _operator_predictions(carrier: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "last_increment": source.temporal_prediction(carrier),
        "mean_increment": mean_increment_prediction(carrier),
    }


def analyze_cell(regime: str, replicate_seed: int) -> dict[str, Any]:
    dataset = generate_dataset(regime, replicate_seed)
    repeated_hash = dataset_hash(generate_dataset(regime, replicate_seed))
    first_hash = dataset_hash(dataset)
    carrier, magnitude = source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    predictions = _operator_predictions(carrier)
    exact_predictions = _operator_predictions(continuous_carrier(dataset))
    exact_errors = {
        name: float((value - dataset.target).abs().max())
        for name, value in exact_predictions.items()
    }
    action_errors: dict[str, dict[str, float]] = {
        name: {} for name in predictions
    }
    for element in (1, 2):
        transformed_carrier, _ = source.analytic_carrier(
            source.apply_deck_action(dataset.tokens, element),
            dataset.calibration,
        )
        transformed = _operator_predictions(transformed_carrier)
        for name in predictions:
            action_errors[name][str(element)] = float(
                (transformed[name] - predictions[name]).abs().max()
            )
    permutation = sattolo_derangement(
        SAMPLE_COUNT, SHUFFLE_SEED_BASE + dataset.dataset_seed
    )
    shuffled_target = dataset.target[permutation]
    operators = {}
    for name, prediction in predictions.items():
        scalar = _scalar_metrics(prediction, dataset.target)
        task = _task_metrics(prediction, dataset.target)
        shuffled = _scalar_metrics(prediction, shuffled_target)
        shuffle_pass = bool(
            abs(shuffled["correlation"])
            <= SHUFFLED_ABSOLUTE_CORRELATION_MAXIMUM
            and shuffled["rmse"] >= SHUFFLED_RMSE_MINIMUM
        )
        maximum_action_error = max(action_errors[name].values())
        operators[name] = {
            "scalar": scalar,
            "task": task,
            "task_pass": _task_gate(task, regime),
            "deck_action_maximum_errors": action_errors[name],
            "maximum_deck_action_error": maximum_action_error,
            "action_pass": maximum_action_error <= ACTION_ERROR_MAXIMUM,
            "shuffled_target": shuffled,
            "shuffle_pass": shuffle_pass,
        }
    baseline = operators["last_increment"]
    candidate = operators["mean_increment"]
    comparison = {
        "rmse_ratio": (
            candidate["scalar"]["rmse"]
            / max(baseline["scalar"]["rmse"], 1e-18)
        ),
        "accuracy_delta": (
            candidate["task"]["exact_bin_accuracy"]
            - baseline["task"]["exact_bin_accuracy"]
        ),
        "cross_entropy_delta": (
            candidate["task"]["target_cross_entropy"]
            - baseline["task"]["target_cross_entropy"]
        ),
    }
    comparison["material_dominance_pass"] = bool(
        candidate["task_pass"]
        and comparison["rmse_ratio"] <= MATERIAL_RMSE_RATIO_MAXIMUM
        and comparison["accuracy_delta"] >= MATERIAL_ACCURACY_DELTA_MINIMUM
        and comparison["cross_entropy_delta"]
        <= MATERIAL_CROSS_ENTROPY_DELTA_MAXIMUM
    )
    baseline_ceiling_pass = bool(
        baseline["scalar"]["rmse"] <= BASELINE_CEILING_RMSE_MAXIMUM
    )
    validity = bool(
        dataset.saturation_count == 0
        and first_hash == repeated_hash
        and int((dataset.tokens != torch.stack([
            source.apply_deck_action(row, int(element))
            for row, element in zip(dataset.canonical_tokens, dataset.deck)
        ])).sum()) == 0
        and int((permutation == torch.arange(SAMPLE_COUNT)).sum()) == 0
        and all(error <= CONTINUOUS_ERROR_MAXIMUM for error in exact_errors.values())
        and all(value["action_pass"] for value in operators.values())
        and all(value["shuffle_pass"] for value in operators.values())
        and all(value["task_pass"] for value in operators.values())
        and _finite(operators)
        and float(magnitude.min()) > 1e-12
    )
    return {
        "seed": replicate_seed,
        "regime": regime,
        "dataset_seed": dataset.dataset_seed,
        "dataset_sha256": first_hash,
        "dataset_deterministic": first_hash == repeated_hash,
        "sample_count": SAMPLE_COUNT,
        "saturation_count": dataset.saturation_count,
        "minimum_character_magnitude": float(magnitude.min()),
        "continuous_positive_control_maximum_errors": exact_errors,
        "shuffle_fixed_points": int(
            (permutation == torch.arange(SAMPLE_COUNT)).sum()
        ),
        "operators": operators,
        "comparison": comparison,
        "baseline_ceiling_pass": baseline_ceiling_pass,
        "valid": validity,
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    def seed_pass(field: str) -> int:
        return sum(
            all(
                bool(
                    next(
                        cell
                        for cell in cells
                        if cell["seed"] == seed and cell["regime"] == regime
                    )[field]
                )
                for regime in REGIMES
            )
            for seed in SEEDS
        )

    material_cells = [
        {**cell, "material_dominance_pass": cell["comparison"]["material_dominance_pass"]}
        for cell in cells
    ]
    material_count = sum(
        all(
            bool(
                next(
                    cell
                    for cell in material_cells
                    if cell["seed"] == seed and cell["regime"] == regime
                )["material_dominance_pass"]
            )
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    baseline_count = seed_pass("baseline_ceiling_pass")
    if not valid:
        classification = "invalid_fixed_operator_ceiling_contract"
    elif material_count >= REQUIRED_SEED_PASSES:
        classification = "fixed_multistep_group_operator_dominates_last_step"
    elif baseline_count >= REQUIRED_SEED_PASSES:
        classification = "last_step_already_at_declared_fixed_operator_ceiling"
    else:
        classification = "fixed_operator_family_leaves_typed_residual_scope"
    return {
        "classification": classification,
        "valid": valid,
        "material_dominance_seed_pass_count": material_count,
        "baseline_ceiling_seed_pass_count": baseline_count,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "same_task_unrestricted_tinyllm_training_licensed": False,
        "typed_residual_function_class_preflight_licensed": (
            classification == "fixed_operator_family_leaves_typed_residual_scope"
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
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if valid else "invalid",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "source_hashes": source_hashes,
        "configuration": {
            "seeds": list(SEEDS),
            "regimes": list(REGIMES),
            "sample_count_per_cell": SAMPLE_COUNT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "shuffle_seed_base": SHUFFLE_SEED_BASE,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "material_rmse_ratio_maximum": MATERIAL_RMSE_RATIO_MAXIMUM,
            "material_accuracy_delta_minimum": MATERIAL_ACCURACY_DELTA_MINIMUM,
            "material_cross_entropy_delta_maximum": (
                MATERIAL_CROSS_ENTROPY_DELTA_MAXIMUM
            ),
            "baseline_ceiling_rmse_maximum": BASELINE_CEILING_RMSE_MAXIMUM,
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": TASK_GATES,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "checkpoints_loaded": 0,
            "tinyllm_models_instantiated": 0,
            "target_using_fits": 0,
            "new_evaluation_examples": len(cells) * SAMPLE_COUNT,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "The all-increment operator was frozen before generating the ten new cohorts.",
            "No temporal estimator, coefficient, weight, or subset was fitted.",
            "The continuous phase/speed carrier is an algebraic positive control only.",
            "No outcome licenses unrestricted TinyLLM training on the same task.",
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite fixed-operator ceiling result")
    return record


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
                "material_dominance_seeds": result["aggregates"][
                    "material_dominance_seed_pass_count"
                ],
                "baseline_ceiling_seeds": result["aggregates"][
                    "baseline_ceiling_seed_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
