#!/usr/bin/env python3
"""Test one robust C3 estimator across hidden degree-1/degree-2 dynamics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_constant_acceleration_fixed_operator as acceleration
import experiments.structure_net.tinyllm_c3_single_frame_corruption_fixed_estimator as corruption
import experiments.structure_net.tinyllm_c3_temporal_fixed_operator_ceiling as speed
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as source


SCHEMA_VERSION = "nal.tinyllm-c3-hidden-order-corruption-fixed-estimator.v1"
HYPOTHESIS_ID = "tinyllm-c3-hidden-order-corruption-fixed-estimator-v1"
EVIDENCE_ROLE = "prospective_no_training_joint_scope_preflight"
PAIR_SEEDS = ((7, 107), (17, 127), (29, 149), (41, 173), (53, 197))
REGIMES = ("composition", "extrapolation")
FAMILIES = ("constant_speed", "constant_acceleration")
POPULATIONS = (*FAMILIES, "mixture")
ARMS = (
    "clean_law_specific",
    "corrupted_law_specific",
    "oracle_drop_one_quadratic",
    "robust_drop_one_quadratic",
)
SAMPLE_COUNT = 4_096
SHUFFLE_SEED_BASES = {
    "constant_speed": {"composition": 861_107, "extrapolation": 863_107},
    "constant_acceleration": {
        "composition": 865_107,
        "extrapolation": 867_107,
    },
}
MINIMUM_FRAME_COUNT = 400
ACTION_ERROR_MAXIMUM = corruption.ACTION_ERROR_MAXIMUM
CONTINUOUS_ERROR_MAXIMUM = corruption.CONTINUOUS_ERROR_MAXIMUM
INCLUSION_ERROR_MAXIMUM = 1e-10
CHART_MARGIN_MINIMUM = corruption.CHART_MARGIN_MINIMUM
REQUIRED_SEED_PASSES = 4
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-hidden-order-corruption-fixed-estimator-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "5530806f4f71ff72eba9abdcc3001e5ebe072bfb0217891662fc4e748fe95da5"
)
SPEED_RUNNER_SHA256 = (
    "9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37"
)
SPEED_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)
SPEED_RESULT_SHA256 = (
    "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
)
CORRUPTION_RUNNER_SHA256 = (
    "8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8"
)
CORRUPTION_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_single_frame_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)
CORRUPTION_RESULT_SHA256 = (
    "59681f2764b988f05b0916965898b87d5b233b2165151fe19e0c97391fe467b9"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)


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
    return corruption._finite(value)


def validate_sources() -> tuple[
    dict[str, str],
    dict[tuple[int, str], str],
    dict[tuple[int, str], str],
]:
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "constant_speed_runner": _sha256(Path(speed.__file__)),
        "constant_speed_result": _sha256(SPEED_RESULT_PATH),
        "single_frame_corruption_runner": _sha256(Path(corruption.__file__)),
        "single_frame_corruption_result": _sha256(CORRUPTION_RESULT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "constant_speed_runner": SPEED_RUNNER_SHA256,
        "constant_speed_result": SPEED_RESULT_SHA256,
        "single_frame_corruption_runner": CORRUPTION_RUNNER_SHA256,
        "single_frame_corruption_result": CORRUPTION_RESULT_SHA256,
    }
    speed_result = json.loads(SPEED_RESULT_PATH.read_text(encoding="utf-8"))
    corruption_result = json.loads(
        CORRUPTION_RESULT_PATH.read_text(encoding="utf-8")
    )
    if (
        hashes != expected
        or speed_result.get("status") != "completed"
        or speed_result.get("aggregates", {}).get("classification")
        != "fixed_multistep_group_operator_dominates_last_step"
        or corruption_result.get("status") != "completed"
        or corruption_result.get("aggregates", {}).get("classification")
        != "fixed_robust_estimator_closes_single_frame_corruption"
    ):
        raise RuntimeError("hidden-order source contract changed")
    speed_hashes = {
        (int(cell["seed"]), str(cell["regime"])): str(cell["dataset_sha256"])
        for cell in speed_result["cells"]
    }
    acceleration_hashes = {
        (int(cell["seed"]), str(cell["regime"])): str(
            cell["base_dataset_sha256"]
        )
        for cell in corruption_result["cells"]
    }
    if set(speed_hashes) != {
        (pair[0], regime) for pair in PAIR_SEEDS for regime in REGIMES
    } or set(acceleration_hashes) != {
        (pair[1], regime) for pair in PAIR_SEEDS for regime in REGIMES
    }:
        raise RuntimeError("hidden-order predecessor population changed")
    return hashes, speed_hashes, acceleration_hashes


def _pilot_speed_dataset(regime: str, sample_count: int) -> speed.CeilingDataset:
    ranges = source.REGIME_RANGES[regime]
    dataset_seed = (881_107 if regime == "composition" else 883_107) + 991
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    phase = 2.0 * math.pi * torch.rand(
        sample_count, dtype=torch.float64, generator=generator
    )
    magnitude = speed._uniform(sample_count, ranges["speed"], generator)
    sign = (
        2.0
        * torch.randint(
            0, 2, (sample_count,), dtype=torch.int64, generator=generator
        ).double()
        - 1.0
    )
    velocity = magnitude * sign
    amplitude = speed._uniform(sample_count, ranges["amplitude"], generator)
    offset = speed._uniform(sample_count, ranges["offset"], generator)
    drift = speed._uniform(sample_count, ranges["drift"], generator)
    deck = torch.randint(
        0, source.CHANNELS, (sample_count,), generator=generator
    )
    continuous = source._continuous_observation(
        phase, velocity, amplitude, offset, drift
    )
    canonical = source.quantize(continuous)
    tokens = torch.stack(
        [
            source.apply_deck_action(row, int(element))
            for row, element in zip(canonical, deck)
        ]
    )
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    return speed.CeilingDataset(
        canonical_tokens=canonical,
        tokens=tokens,
        calibration=calibration,
        target=torch.cos(3.0 * (phase + source.TIME_STEPS * velocity)),
        phase=phase,
        speed=velocity,
        deck=deck,
        saturation_count=int(
            (continuous.abs() >= source.QUANTIZATION_LIMIT).sum()
        ),
        dataset_seed=dataset_seed,
    )


def generate_family_dataset(
    family: str,
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> Any:
    primary = sample_count == SAMPLE_COUNT and (
        (family == "constant_speed" and seed_value in {pair[0] for pair in PAIR_SEEDS})
        or (
            family == "constant_acceleration"
            and seed_value in {pair[1] for pair in PAIR_SEEDS}
        )
    )
    pilot = allow_pilot and sample_count == 64 and seed_value == 991
    if family not in FAMILIES or regime not in REGIMES or not (primary or pilot):
        raise ValueError(f"unknown hidden-order cohort {family}/{regime}/{seed_value}")
    if family == "constant_speed":
        return (
            speed.generate_dataset(regime, seed_value)
            if primary
            else _pilot_speed_dataset(regime, sample_count)
        )
    return acceleration.generate_dataset(
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )


def _dataset_hash(family: str, dataset: Any) -> str:
    return (
        speed.dataset_hash(dataset)
        if family == "constant_speed"
        else acceleration.dataset_hash(dataset)
    )


def _mean_increment_complex(carrier: torch.Tensor) -> torch.Tensor:
    increments = carrier[:, 1:] * carrier[:, :-1].conj()
    mean = acceleration._normalize(increments.sum(dim=1))
    return carrier[:, -1] * mean


def _law_specific_prediction(family: str, carrier: torch.Tensor) -> torch.Tensor:
    if family == "constant_speed":
        return _mean_increment_complex(carrier)
    return acceleration.operator_complex_predictions(carrier)[
        "all_frame_degree2"
    ]


def common_predictions(
    family: str,
    clean_carrier: torch.Tensor,
    corrupted_carrier: torch.Tensor,
    true_frame: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    oracle, robust, selected, _ = corruption.deletion_predictions(
        corrupted_carrier, true_frame
    )
    return (
        {
            "clean_law_specific": _law_specific_prediction(
                family, clean_carrier
            ),
            "corrupted_law_specific": _law_specific_prediction(
                family, corrupted_carrier
            ),
            "oracle_drop_one_quadratic": oracle,
            "robust_drop_one_quadratic": robust,
        },
        selected,
    )


def _continuous_carriers(
    family: str, dataset: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if family == "constant_speed":
        clean = speed.continuous_carrier(dataset)
        future_phase = dataset.phase + source.TIME_STEPS * dataset.speed
        future = torch.polar(
            torch.ones_like(future_phase), 3.0 * future_phase
        ).to(torch.complex128)
        expected_beta2 = torch.zeros_like(dataset.speed)
        return clean, future, expected_beta2
    clean = acceleration.continuous_carrier(dataset)
    future = acceleration.continuous_carrier(
        dataset, include_future=True
    )[:, -1]
    return clean, future, 3.0 * dataset.acceleration


def _true_deletion_coefficients(
    carrier: torch.Tensor, true_frame: torch.Tensor
) -> torch.Tensor:
    coefficients = torch.empty(
        (carrier.shape[0], 3), dtype=torch.float64, device=carrier.device
    )
    all_times = torch.arange(source.TIME_STEPS, dtype=torch.float64)
    for omitted in range(source.TIME_STEPS):
        rows = true_frame == omitted
        if not bool(rows.any()):
            continue
        keep = torch.arange(source.TIME_STEPS) != omitted
        times = all_times[keep]
        phase = torch.angle(carrier[rows][:, keep]).double()
        increments = corruption._principal_increment(
            phase[:, 1:] - phase[:, :-1]
        )
        unwrapped = torch.cat(
            (phase[:, :1], phase[:, :1] + torch.cumsum(increments, dim=1)),
            dim=1,
        )
        design = torch.stack(
            (torch.ones_like(times), times, 0.5 * times * (times - 1.0)),
            dim=1,
        )
        coefficients[rows] = unwrapped @ torch.linalg.pinv(design).T
    return coefficients


def _speed_group_contracts(dataset: speed.CeilingDataset) -> dict[str, Any]:
    identity_errors = int(
        (
            source.apply_deck_action(dataset.canonical_tokens, 0)
            != dataset.canonical_tokens
        ).sum()
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
    for _ in range(source.CHANNELS):
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
        source._continuous_observation(
            shifted_phase, dataset.speed, amplitude, offset, drift
        )
    )
    maximum_target_error = 0.0
    for element in range(source.CHANNELS):
        shifted_future = (
            dataset.phase
            - 2.0 * math.pi * element / source.CHANNELS
            + source.TIME_STEPS * dataset.speed
        )
        maximum_target_error = max(
            maximum_target_error,
            float((torch.cos(3.0 * shifted_future) - dataset.target).abs().max()),
        )
    values = {
        "identity_token_errors": identity_errors,
        "composition_token_errors": composition_errors,
        "order_three_token_errors": int(
            (order_three != dataset.canonical_tokens).sum()
        ),
        "stored_action_token_errors": int((stored != dataset.tokens).sum()),
        "latent_regeneration_token_errors": int(
            (regenerated != dataset.tokens).sum()
        ),
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


def _comparisons(operators: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    clean = operators["clean_law_specific"]
    naive = operators["corrupted_law_specific"]
    oracle = operators["oracle_drop_one_quadratic"]
    robust = operators["robust_drop_one_quadratic"]
    repair = {
        "rmse_ratio": robust["scalar"]["rmse"]
        / max(naive["scalar"]["rmse"], 1e-18),
        "accuracy_delta": robust["task"]["exact_bin_accuracy"]
        - naive["task"]["exact_bin_accuracy"],
        "cross_entropy_delta": robust["task"]["target_cross_entropy"]
        - naive["task"]["target_cross_entropy"],
    }
    repair["pass"] = bool(
        robust["fixed_ceiling_pass"]
        and repair["rmse_ratio"] <= corruption.REPAIR_RMSE_RATIO_MAXIMUM
        and repair["accuracy_delta"]
        >= corruption.REPAIR_ACCURACY_DELTA_MINIMUM
        and repair["cross_entropy_delta"]
        <= corruption.REPAIR_CROSS_ENTROPY_DELTA_MAXIMUM
    )
    fidelity = {
        "rmse_excess": robust["scalar"]["rmse"] - oracle["scalar"]["rmse"],
        "accuracy_delta": robust["task"]["exact_bin_accuracy"]
        - oracle["task"]["exact_bin_accuracy"],
        "cross_entropy_excess": robust["task"]["target_cross_entropy"]
        - oracle["task"]["target_cross_entropy"],
    }
    fidelity["pass"] = bool(
        fidelity["rmse_excess"] <= corruption.ORACLE_RMSE_SLACK_MAXIMUM
        and fidelity["accuracy_delta"]
        >= -corruption.ORACLE_ACCURACY_SLACK_MAXIMUM
        and fidelity["cross_entropy_excess"]
        <= corruption.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
    )
    return {
        "corruption_material": bool(
            clean["fixed_ceiling_pass"] and not naive["fixed_ceiling_pass"]
        ),
        "robust_repair": repair,
        "oracle_fidelity": fidelity,
    }


def analyze_family(
    family: str,
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
    expected_base_hash: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registered = sample_count == SAMPLE_COUNT and seed_value != 991
    dataset = generate_family_dataset(
        family,
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_hash = _dataset_hash(family, dataset)
    base_hash_replayed = expected_base_hash is None or base_hash == expected_base_hash
    donor, frame = corruption.corruption_plan(regime, seed_value, sample_count)
    corrupted_tokens = corruption.corrupt_frames(dataset.tokens, donor, frame)
    replay_donor, replay_frame = corruption.corruption_plan(
        regime, seed_value, sample_count
    )
    replay_corrupted = corruption.corrupt_frames(
        dataset.tokens, replay_donor, replay_frame
    )
    corruption_deterministic = bool(
        torch.equal(donor, replay_donor)
        and torch.equal(frame, replay_frame)
        and torch.equal(corrupted_tokens, replay_corrupted)
    )
    clean_carrier, clean_magnitude = source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    corrupted_carrier, corrupted_magnitude = source.analytic_carrier(
        corrupted_tokens, dataset.calibration
    )
    predictions, selected = common_predictions(
        family, clean_carrier, corrupted_carrier, frame
    )
    action_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    corruption_equivariance_errors: dict[str, int] = {}
    for element in (1, 2):
        transformed_clean_tokens = source.apply_deck_action(dataset.tokens, element)
        transformed_corrupted_tokens = corruption.corrupt_frames(
            transformed_clean_tokens, donor, frame
        )
        expected_corruption = source.apply_deck_action(corrupted_tokens, element)
        corruption_equivariance_errors[str(element)] = int(
            (transformed_corrupted_tokens != expected_corruption).sum()
        )
        transformed_clean_carrier, _ = source.analytic_carrier(
            transformed_clean_tokens, dataset.calibration
        )
        transformed_corrupted_carrier, _ = source.analytic_carrier(
            transformed_corrupted_tokens, dataset.calibration
        )
        transformed_predictions, _ = common_predictions(
            family,
            transformed_clean_carrier,
            transformed_corrupted_carrier,
            frame,
        )
        for arm in ARMS:
            action_errors[arm][str(element)] = float(
                (transformed_predictions[arm] - predictions[arm]).abs().max()
            )

    exact_clean, exact_future, expected_beta2 = _continuous_carriers(
        family, dataset
    )
    exact_corrupted = corruption.corrupt_frames(exact_clean, donor, frame)
    exact_oracle, exact_robust, exact_selected, _ = (
        corruption.deletion_predictions(exact_corrupted, frame)
    )
    continuous_errors = {
        "oracle_drop_one_quadratic": float(
            (exact_oracle - exact_future).abs().max()
        ),
        "robust_drop_one_quadratic": float(
            (exact_robust - exact_future).abs().max()
        ),
    }
    coefficients = _true_deletion_coefficients(exact_clean, frame)
    inclusion_error = float((coefficients[:, 2] - expected_beta2).abs().max())
    chart_margin = corruption._chart_margin(exact_clean, frame)
    shuffle = speed.sattolo_derangement(
        sample_count,
        SHUFFLE_SEED_BASES[family][regime] + seed_value,
    )
    shuffled_target = dataset.target[shuffle]
    operators = {
        arm: corruption._arm_metrics(
            predictions[arm],
            dataset.target,
            shuffled_target,
            regime,
            action_errors[arm],
        )
        for arm in ARMS
    }
    comparisons = _comparisons(operators)
    frame_counts = torch.bincount(frame, minlength=source.TIME_STEPS)
    donor_fixed_points = int(
        (donor == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    shuffle_fixed_points = int(
        (shuffle == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    base_contracts = (
        _speed_group_contracts(dataset)
        if family == "constant_speed"
        else acceleration._group_contracts(dataset)
    )
    validity = bool(
        dataset.saturation_count == 0
        and base_hash_replayed
        and base_contracts["pass"]
        and corruption_deterministic
        and donor_fixed_points == 0
        and int(frame_counts.min()) >= (MINIMUM_FRAME_COUNT if registered else 0)
        and sum(corruption_equivariance_errors.values()) == 0
        and all(operator["action_pass"] for operator in operators.values())
        and (
            all(operator["shuffle_pass"] for operator in operators.values())
            if registered
            else True
        )
        and max(continuous_errors.values()) <= CONTINUOUS_ERROR_MAXIMUM
        and int((exact_selected == frame).sum()) == sample_count
        and inclusion_error <= INCLUSION_ERROR_MAXIMUM
        and chart_margin >= CHART_MARGIN_MINIMUM
        and shuffle_fixed_points == 0
        and float(clean_magnitude.min()) > 1e-12
        and float(corrupted_magnitude.min()) > 1e-12
        and _finite(operators)
        and _finite(comparisons)
    )
    record = {
        "family": family,
        "seed": seed_value,
        "dataset_seed": dataset.dataset_seed,
        "base_dataset_sha256": base_hash,
        "base_dataset_hash_replayed": base_hash_replayed,
        "corruption_sha256": _tensor_hash(corrupted_tokens, donor, frame),
        "corruption_deterministic": corruption_deterministic,
        "sample_count": sample_count,
        "saturation_count": dataset.saturation_count,
        "donor_fixed_points": donor_fixed_points,
        "frame_counts": [int(value) for value in frame_counts],
        "minimum_frame_count": int(frame_counts.min()),
        "minimum_clean_chart_margin": chart_margin,
        "continuous_law_inclusion_maximum_error": inclusion_error,
        "continuous_prediction_maximum_errors": continuous_errors,
        "continuous_corruption_index_recovery_count": int(
            (exact_selected == frame).sum()
        ),
        "quantized_corruption_index_recovery_count": int((selected == frame).sum()),
        "quantized_corruption_index_recovery_rate": float(
            (selected == frame).double().mean()
        ),
        "minimum_clean_character_magnitude": float(clean_magnitude.min()),
        "minimum_corrupted_character_magnitude": float(
            corrupted_magnitude.min()
        ),
        "base_group_contracts": base_contracts,
        "corruption_equivariance_token_errors": corruption_equivariance_errors,
        "shuffle_fixed_points": shuffle_fixed_points,
        "operators": operators,
        **comparisons,
        "valid": validity,
    }
    payload = {
        "target": dataset.target,
        "shuffled_target": shuffled_target,
        "predictions": predictions,
        "action_errors": action_errors,
    }
    return record, payload


def analyze_cell(
    replicate: int,
    regime: str,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
    speed_hash: str | None = None,
    acceleration_hash: str | None = None,
) -> dict[str, Any]:
    if replicate not in range(1, 6) and not allow_pilot:
        raise ValueError(f"unknown hidden-order replicate {replicate}")
    speed_seed, acceleration_seed = (
        PAIR_SEEDS[replicate - 1] if not allow_pilot else (991, 991)
    )
    speed_record, speed_payload = analyze_family(
        "constant_speed",
        regime,
        speed_seed,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
        expected_base_hash=speed_hash,
    )
    acceleration_record, acceleration_payload = analyze_family(
        "constant_acceleration",
        regime,
        acceleration_seed,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
        expected_base_hash=acceleration_hash,
    )
    mixed_target = torch.cat(
        (speed_payload["target"], acceleration_payload["target"])
    )
    mixed_shuffled = torch.cat(
        (
            speed_payload["shuffled_target"],
            acceleration_payload["shuffled_target"],
        )
    )
    mixture_operators = {}
    for arm in ARMS:
        prediction = torch.cat(
            (
                speed_payload["predictions"][arm],
                acceleration_payload["predictions"][arm],
            )
        )
        action_errors = {
            element: max(
                speed_payload["action_errors"][arm][element],
                acceleration_payload["action_errors"][arm][element],
            )
            for element in ("1", "2")
        }
        mixture_operators[arm] = corruption._arm_metrics(
            prediction,
            mixed_target,
            mixed_shuffled,
            regime,
            action_errors,
        )
    mixture_comparisons = _comparisons(mixture_operators)
    mixture = {
        "law_proportions": {
            "constant_speed": 0.5,
            "constant_acceleration": 0.5,
        },
        "sample_count": int(mixed_target.numel()),
        "operators": mixture_operators,
        **mixture_comparisons,
        "valid": bool(
            all(operator["action_pass"] for operator in mixture_operators.values())
            and (
                all(
                    operator["shuffle_pass"]
                    for operator in mixture_operators.values()
                )
                if not allow_pilot
                else True
            )
            and _finite(mixture_operators)
            and _finite(mixture_comparisons)
        ),
    }
    populations = {
        "constant_speed": speed_record,
        "constant_acceleration": acceleration_record,
        "mixture": mixture,
    }
    return {
        "replicate": replicate,
        "regime": regime,
        "speed_seed": speed_seed,
        "acceleration_seed": acceleration_seed,
        "populations": populations,
        "common_robust_endpoint_pass": all(
            populations[name]["operators"]["robust_drop_one_quadratic"][
                "fixed_ceiling_pass"
            ]
            for name in POPULATIONS
        ),
        "oracle_endpoint_pass": all(
            populations[name]["operators"]["oracle_drop_one_quadratic"][
                "fixed_ceiling_pass"
            ]
            for name in POPULATIONS
        ),
        "materiality_pass": all(
            populations[name]["corruption_material"] for name in POPULATIONS
        ),
        "repair_pass": all(
            populations[name]["robust_repair"]["pass"] for name in POPULATIONS
        ),
        "oracle_fidelity_pass": all(
            populations[name]["oracle_fidelity"]["pass"] for name in POPULATIONS
        ),
        "valid": all(populations[name]["valid"] for name in POPULATIONS),
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    def seed_count(field: str) -> int:
        return sum(
            all(
                bool(
                    next(
                        cell
                        for cell in cells
                        if cell["replicate"] == replicate
                        and cell["regime"] == regime
                    )[field]
                )
                for regime in REGIMES
            )
            for replicate in range(1, 6)
        )

    common_count = seed_count("common_robust_endpoint_pass")
    oracle_count = seed_count("oracle_endpoint_pass")
    material_count = seed_count("materiality_pass")
    repair_count = seed_count("repair_pass")
    fidelity_count = seed_count("oracle_fidelity_pass")
    naive_mixture_count = sum(
        all(
            bool(
                next(
                    cell
                    for cell in cells
                    if cell["replicate"] == replicate
                    and cell["regime"] == regime
                )["populations"]["mixture"]["operators"][
                    "corrupted_law_specific"
                ]["fixed_ceiling_pass"]
            )
            for regime in REGIMES
        )
        for replicate in range(1, 6)
    )
    if not valid:
        classification = "invalid_hidden_order_corruption_preflight"
    elif all(
        count >= REQUIRED_SEED_PASSES
        for count in (
            common_count,
            material_count,
            repair_count,
            fidelity_count,
        )
    ):
        classification = "single_robust_quadratic_closes_hidden_order_and_corruption"
    elif naive_mixture_count >= REQUIRED_SEED_PASSES:
        classification = "hidden_order_corruption_not_material"
    elif oracle_count >= REQUIRED_SEED_PASSES and common_count < REQUIRED_SEED_PASSES:
        classification = "recoverable_hidden_order_corruption_exceeds_common_fixed_estimator"
    elif oracle_count < REQUIRED_SEED_PASSES:
        classification = "hidden_order_corruption_not_recoverable_at_required_ceiling"
    else:
        classification = "inconclusive_hidden_order_corruption_preflight"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "common_robust_endpoint_seed_pass_count": common_count,
        "oracle_endpoint_seed_pass_count": oracle_count,
        "materiality_seed_pass_count": material_count,
        "repair_seed_pass_count": repair_count,
        "oracle_fidelity_seed_pass_count": fidelity_count,
        "corrupted_naive_mixture_ceiling_seed_pass_count": naive_mixture_count,
        "tinyllm_training_licensed": False,
        "compact_typed_selector_comparison_licensed": classification
        == "recoverable_hidden_order_corruption_exceeds_common_fixed_estimator",
    }


def build_result() -> dict[str, Any]:
    source_hashes, speed_hashes, acceleration_hashes = validate_sources()
    cells = [
        analyze_cell(
            replicate,
            regime,
            speed_hash=speed_hashes[(PAIR_SEEDS[replicate - 1][0], regime)],
            acceleration_hash=acceleration_hashes[
                (PAIR_SEEDS[replicate - 1][1], regime)
            ],
        )
        for replicate in range(1, 6)
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
            "pair_seeds": [list(pair) for pair in PAIR_SEEDS],
            "regimes": list(REGIMES),
            "families": list(FAMILIES),
            "populations": list(POPULATIONS),
            "arms": list(ARMS),
            "sample_count_per_family_cell": SAMPLE_COUNT,
            "mixture_law_proportions": {
                "constant_speed": 0.5,
                "constant_acceleration": 0.5,
            },
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "minimum_frame_count": MINIMUM_FRAME_COUNT,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "inclusion_error_maximum": INCLUSION_ERROR_MAXIMUM,
            "chart_margin_minimum": CHART_MARGIN_MINIMUM,
            "fixed_ceiling_rmse_maximum": (
                corruption.FIXED_CEILING_RMSE_MAXIMUM
            ),
            "fixed_ceiling_accuracy_minimum": (
                corruption.FIXED_CEILING_ACCURACY_MINIMUM
            ),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": acceleration.TASK_GATES,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "matched_base_examples_replayed": len(cells) * SAMPLE_COUNT * 2,
            "new_corrupted_evaluations": len(cells) * SAMPLE_COUNT * 2,
            "closed_form_observation_only_fits": (
                len(cells) * SAMPLE_COUNT * 2 * (source.TIME_STEPS + 1)
            ),
            "law_label_reads_by_primary_estimator": 0,
            "adaptive_selector_decisions": 0,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "models_instantiated": 0,
            "checkpoints_loaded": 0,
            "target_using_fits": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "The paired cohorts, actual mixture, corruption, common estimator, gates, and outcomes were registered before primary corruption generation.",
            "The primary robust prediction uses one frozen quadratic estimator and never reads or selects a law label.",
            "Law-specific branches occur only in registered positive and naive controls.",
            "The study covers a nested degree-one/degree-two family, not unknown nonlinear, switching, or stochastic dynamics.",
        ],
    }
    if not _finite(result):
        raise RuntimeError("non-finite hidden-order result")
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
                "common_robust_seeds": result["aggregates"][
                    "common_robust_endpoint_seed_pass_count"
                ],
                "materiality_seeds": result["aggregates"][
                    "materiality_seed_pass_count"
                ],
                "repair_seeds": result["aggregates"]["repair_seed_pass_count"],
                "oracle_fidelity_seeds": result["aggregates"][
                    "oracle_fidelity_seed_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
