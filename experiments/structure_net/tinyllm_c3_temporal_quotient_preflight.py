#!/usr/bin/env python3
"""Preflight an exact observable C3 temporal quotient before TinyLLM training."""

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

import numpy as np
import torch


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-quotient-preflight.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-quotient-preflight-v1"
EVIDENCE_ROLE = "no_training_observable_group_generator_preflight"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-quotient-preflight.md"
)
PREREGISTRATION_SHA256 = (
    "1da4d34892df7a988cee3e42e44a90b94e3e2f74af7f982ec4b515c0d6fbe8e0"
)
REGIMES = ("composition", "extrapolation")
SAMPLE_COUNT = 4_096
TIME_STEPS = 8
CHANNELS = 3
QUANTIZATION_BINS = 1_024
QUANTIZATION_LIMIT = 4.0
DATASET_SEEDS = {"composition": 131_003, "extrapolation": 131_021}
REGIME_RANGES = {
    "composition": {
        "speed": (0.04, 0.12),
        "amplitude": (0.7, 1.8),
        "offset": (-0.4, 0.4),
        "drift": (-0.06, 0.06),
    },
    "extrapolation": {
        "speed": (0.13, 0.20),
        "amplitude": (0.5, 2.2),
        "offset": (-0.7, 0.7),
        "drift": (-0.10, 0.10),
    },
}
TEMPORAL_GATES = {"correlation_minimum": 0.99, "rmse_maximum": 0.08}
SHUFFLED_GATES = {"absolute_correlation_maximum": 0.10, "rmse_minimum": 0.80}


@dataclass(frozen=True)
class C3TemporalDataset:
    base_tokens: torch.Tensor
    tokens: torch.Tensor
    calibration: torch.Tensor
    target: torch.Tensor
    phase: torch.Tensor
    speed: torch.Tensor
    deck: torch.Tensor
    saturation_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
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


def quantize(value: torch.Tensor) -> torch.Tensor:
    scaled = (value + QUANTIZATION_LIMIT) / (2.0 * QUANTIZATION_LIMIT)
    return torch.round(scaled * (QUANTIZATION_BINS - 1)).clamp(
        0, QUANTIZATION_BINS - 1
    ).to(torch.int64)


def decode(tokens: torch.Tensor) -> torch.Tensor:
    return (
        tokens.to(torch.float64)
        / float(QUANTIZATION_BINS - 1)
        * (2.0 * QUANTIZATION_LIMIT)
        - QUANTIZATION_LIMIT
    )


def apply_deck_action(tokens: torch.Tensor, element: int) -> torch.Tensor:
    if element not in (0, 1, 2):
        raise ValueError("C3 element must be 0, 1, or 2")
    return torch.roll(tokens, shifts=element, dims=-1)


def _roll_by_example(tokens: torch.Tensor, elements: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [apply_deck_action(row, int(element)) for row, element in zip(tokens, elements)]
    )


def _continuous_observation(
    phase: torch.Tensor,
    speed: torch.Tensor,
    amplitude: torch.Tensor,
    offset: torch.Tensor,
    drift: torch.Tensor,
) -> torch.Tensor:
    time = torch.arange(TIME_STEPS, dtype=torch.float64)[None, :, None]
    channel = (
        2.0
        * math.pi
        * torch.arange(CHANNELS, dtype=torch.float64)[None, None, :]
        / CHANNELS
    )
    angle = phase[:, None, None] + speed[:, None, None] * time + channel
    return (
        amplitude[:, None, None] * torch.cos(angle)
        + offset[:, None, None]
        + drift[:, None, None] * time
    )


def generate_dataset(regime: str) -> C3TemporalDataset:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    ranges = REGIME_RANGES[regime]
    generator = torch.Generator(device="cpu").manual_seed(DATASET_SEEDS[regime])
    phase = 2.0 * math.pi * torch.rand(
        SAMPLE_COUNT, dtype=torch.float64, generator=generator
    )
    speed_magnitude = torch.empty(SAMPLE_COUNT, dtype=torch.float64).uniform_(
        *ranges["speed"], generator=generator
    )
    speed_sign = 2.0 * torch.randint(
        0, 2, (SAMPLE_COUNT,), generator=generator, dtype=torch.int64
    ).to(torch.float64) - 1.0
    speed = speed_magnitude * speed_sign
    amplitude = torch.empty(SAMPLE_COUNT, dtype=torch.float64).uniform_(
        *ranges["amplitude"], generator=generator
    )
    offset = torch.empty(SAMPLE_COUNT, dtype=torch.float64).uniform_(
        *ranges["offset"], generator=generator
    )
    drift = torch.empty(SAMPLE_COUNT, dtype=torch.float64).uniform_(
        *ranges["drift"], generator=generator
    )
    deck = torch.randint(0, CHANNELS, (SAMPLE_COUNT,), generator=generator)
    base_continuous = _continuous_observation(
        phase, speed, amplitude, offset, drift
    )
    saturation_count = int((base_continuous.abs() >= QUANTIZATION_LIMIT).sum())
    base_tokens = quantize(base_continuous)
    tokens = _roll_by_example(base_tokens, deck)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    target = torch.cos(3.0 * (phase + TIME_STEPS * speed))
    return C3TemporalDataset(
        base_tokens=base_tokens,
        tokens=tokens,
        calibration=calibration,
        target=target,
        phase=phase,
        speed=speed,
        deck=deck,
        saturation_count=saturation_count,
    )


def analytic_carrier(
    tokens: torch.Tensor, calibration: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = decode(tokens)
    amplitude = calibration[:, 0]
    offset = calibration[:, 1]
    drift = calibration[:, 2]
    time = torch.arange(TIME_STEPS, dtype=torch.float64)[None, :, None]
    corrected = (
        observed - offset[:, None, None] - drift[:, None, None] * time
    ) / amplitude[:, None, None]
    angles = 2.0 * math.pi * torch.arange(CHANNELS, dtype=torch.float64) / CHANNELS
    weights = torch.polar(torch.ones_like(angles), -angles).to(torch.complex128)
    coefficient = torch.einsum("btc,c->bt", corrected.to(torch.complex128), weights)
    magnitude = coefficient.abs()
    normalized = coefficient / magnitude.clamp_min(1e-12)
    return normalized.pow(3), magnitude


def temporal_prediction(carrier: torch.Tensor) -> torch.Tensor:
    step = carrier[:, -1] * carrier[:, -2].conj()
    return (carrier[:, -1] * step).real


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.reshape(-1).double()
    y = right.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x * y).sum() / denominator.clamp_min(1e-12))


def _rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((left.double() - right.double()).square())))


def validate_group_action(dataset: C3TemporalDataset) -> dict[str, Any]:
    identity_error = int(
        (apply_deck_action(dataset.base_tokens, 0) != dataset.base_tokens).sum()
    )
    composition_errors = 0
    for left in range(CHANNELS):
        for right in range(CHANNELS):
            composed = apply_deck_action(
                apply_deck_action(dataset.base_tokens, right), left
            )
            direct = apply_deck_action(dataset.base_tokens, (left + right) % CHANNELS)
            composition_errors += int((composed != direct).sum())
    order_three_error = int(
        (
            apply_deck_action(
                apply_deck_action(
                    apply_deck_action(dataset.base_tokens, 1), 1
                ),
                1,
            )
            != dataset.base_tokens
        ).sum()
    )
    stored_action_error = int(
        (_roll_by_example(dataset.base_tokens, dataset.deck) != dataset.tokens).sum()
    )

    independently_generated = []
    amplitude, offset, drift = (dataset.calibration[:, index] for index in range(3))
    for index in range(SAMPLE_COUNT):
        shifted_phase = dataset.phase[index : index + 1] - (
            2.0 * math.pi * float(dataset.deck[index]) / CHANNELS
        )
        continuous = _continuous_observation(
            shifted_phase,
            dataset.speed[index : index + 1],
            amplitude[index : index + 1],
            offset[index : index + 1],
            drift[index : index + 1],
        )[0]
        independently_generated.append(quantize(continuous))
    latent_generation_error = int(
        (torch.stack(independently_generated) != dataset.tokens).sum()
    )

    maximum_target_error = 0.0
    for element in range(CHANNELS):
        shifted_target = torch.cos(
            3.0
            * (
                dataset.phase
                - 2.0 * math.pi * element / CHANNELS
                + TIME_STEPS * dataset.speed
            )
        )
        maximum_target_error = max(
            maximum_target_error,
            float(torch.max(torch.abs(shifted_target - dataset.target))),
        )
    passed = (
        identity_error == 0
        and composition_errors == 0
        and order_three_error == 0
        and stored_action_error == 0
        and latent_generation_error == 0
        and maximum_target_error <= 1e-12
        and dataset.saturation_count == 0
    )
    return {
        "identity_token_error_count": identity_error,
        "composition_token_error_count": composition_errors,
        "order_three_token_error_count": order_three_error,
        "stored_action_token_error_count": stored_action_error,
        "latent_generation_token_error_count": latent_generation_error,
        "maximum_target_invariance_error": maximum_target_error,
        "calibration_action": "identity_exact_by_declared_type",
        "quantizer_saturation_count": dataset.saturation_count,
        "pass": passed,
    }


def analyze_carrier(dataset: C3TemporalDataset) -> dict[str, Any]:
    carrier, magnitude = analytic_carrier(dataset.base_tokens, dataset.calibration)
    maximum_invariance_error = 0.0
    for element in range(CHANNELS):
        transformed, _ = analytic_carrier(
            apply_deck_action(dataset.base_tokens, element), dataset.calibration
        )
        maximum_invariance_error = max(
            maximum_invariance_error,
            float(torch.max(torch.abs(transformed - carrier))),
        )

    observed = decode(dataset.base_tokens)
    amplitude = dataset.calibration[:, 0]
    offset = dataset.calibration[:, 1]
    drift = dataset.calibration[:, 2]
    time = torch.arange(TIME_STEPS, dtype=torch.float64)[None, :, None]
    corrected = (
        observed - offset[:, None, None] - drift[:, None, None] * time
    ) / amplitude[:, None, None]
    centered = corrected - corrected.mean(dim=-1, keepdim=True)
    orbit_mean = torch.stack(
        [torch.roll(centered, shifts=element, dims=-1) for element in range(CHANNELS)]
    ).mean(dim=0)
    maximum_raw_reynolds_norm = float(
        torch.linalg.vector_norm(orbit_mean, dim=-1).max()
    )

    predicted = temporal_prediction(carrier)
    correlation = _pearson(predicted, dataset.target)
    rmse = _rmse(predicted, dataset.target)
    permutation = torch.roll(torch.arange(SAMPLE_COUNT), shifts=1)
    shuffled_target = dataset.target[permutation]
    shuffled_correlation = _pearson(predicted, shuffled_target)
    shuffled_rmse = _rmse(predicted, shuffled_target)
    carrier_contract = (
        maximum_invariance_error <= 1e-6
        and float(magnitude.min()) >= 0.25
        and float(carrier.real.var(unbiased=False)) >= 0.20
        and float(carrier.imag.var(unbiased=False)) >= 0.20
        and maximum_raw_reynolds_norm <= 1e-6
    )
    temporal_pass = (
        correlation >= TEMPORAL_GATES["correlation_minimum"]
        and rmse <= TEMPORAL_GATES["rmse_maximum"]
    )
    specificity_pass = (
        abs(shuffled_correlation)
        <= SHUFFLED_GATES["absolute_correlation_maximum"]
        and shuffled_rmse >= SHUFFLED_GATES["rmse_minimum"]
    )
    return {
        "maximum_deck_invariance_error": maximum_invariance_error,
        "minimum_pre_normalization_magnitude": float(magnitude.min()),
        "carrier_real_variance": float(carrier.real.var(unbiased=False)),
        "carrier_imaginary_variance": float(carrier.imag.var(unbiased=False)),
        "maximum_centered_raw_reynolds_norm": maximum_raw_reynolds_norm,
        "temporal_prediction": {
            "cosine_correlation": correlation,
            "cosine_rmse": rmse,
            "pass": temporal_pass,
        },
        "shuffled_specificity": {
            "cosine_correlation": shuffled_correlation,
            "cosine_rmse": shuffled_rmse,
            "pass": specificity_pass,
        },
        "carrier_contract_pass": carrier_contract,
        "pass": carrier_contract and temporal_pass and specificity_pass,
    }


def instantaneous_insufficiency_witness() -> dict[str, Any]:
    final_phase = 0.30
    speeds = torch.tensor((0.15, -0.15), dtype=torch.float64)
    initial_phase = final_phase - (TIME_STEPS - 1) * speeds
    time = torch.arange(TIME_STEPS, dtype=torch.float64)[None, :]
    phase = initial_phase[:, None] + speeds[:, None] * time
    histories = torch.polar(torch.ones_like(phase), 3.0 * phase).to(torch.complex128)
    targets = torch.cos(3.0 * (initial_phase + TIME_STEPS * speeds))
    final_error = float(torch.abs(histories[0, -1] - histories[1, -1]))
    history_difference = float(
        torch.sqrt(torch.mean(torch.abs(histories[0] - histories[1]).square()))
    )
    target_separation = float(torch.abs(targets[0] - targets[1]))
    passed = (
        final_error <= 1e-12
        and history_difference > 0.10
        and target_separation >= 0.25
    )
    return {
        "speeds": speeds.tolist(),
        "final_carrier_absolute_difference": final_error,
        "full_history_rms_difference": history_difference,
        "future_target_absolute_difference": target_separation,
        "pass": passed,
    }


def build_preflight() -> dict[str, Any]:
    preregistration = _sha256(PREREGISTRATION_PATH)
    if preregistration != PREREGISTRATION_SHA256:
        raise RuntimeError("C3 temporal preflight registration changed")
    datasets = {regime: generate_dataset(regime) for regime in REGIMES}
    group = {regime: validate_group_action(datasets[regime]) for regime in REGIMES}
    carriers = {regime: analyze_carrier(datasets[regime]) for regime in REGIMES}
    witness = instantaneous_insufficiency_witness()
    group_pass = all(record["pass"] for record in group.values())
    carrier_pass = all(record["carrier_contract_pass"] for record in carriers.values())
    temporal_pass = all(
        record["temporal_prediction"]["pass"] for record in carriers.values()
    )
    specificity_pass = all(
        record["shuffled_specificity"]["pass"] for record in carriers.values()
    )
    if not group_pass:
        classification = "invalid_observable_group_action"
        valid = False
    elif not carrier_pass:
        classification = "c3_invariant_carrier_contract_failed"
        valid = True
    elif not temporal_pass:
        classification = "c3_invariant_carrier_not_sufficient"
        valid = True
    elif not witness["pass"]:
        classification = "c3_frontend_collapses_to_answer"
        valid = True
    elif not specificity_pass:
        classification = "c3_temporal_specificity_failed"
        valid = True
    else:
        classification = "c3_temporal_quotient_preflight_passed"
        valid = True
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "status": "completed" if valid else "invalid",
        "completed_at": _utc_now(),
        "classification": classification,
        "valid": valid,
        "implementation_sha256": _sha256(Path(__file__)),
        "preregistration_sha256": preregistration,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "regimes": REGIMES,
            "sample_count_per_regime": SAMPLE_COUNT,
            "time_steps": TIME_STEPS,
            "channels": CHANNELS,
            "quantization_bins": QUANTIZATION_BINS,
            "quantization_limit": QUANTIZATION_LIMIT,
            "dataset_seeds": DATASET_SEEDS,
            "regime_ranges": REGIME_RANGES,
            "temporal_gates": TEMPORAL_GATES,
            "shuffled_gates": SHUFFLED_GATES,
        },
        "group_action": group,
        "invariant_carrier": carriers,
        "instantaneous_insufficiency": witness,
        "gates": {
            "observable_group_action": group_pass,
            "invariant_carrier": carrier_pass,
            "temporal_sufficiency": temporal_pass,
            "instantaneous_insufficiency": witness["pass"],
            "shuffled_specificity": specificity_pass,
            "zero_optimizer_steps": True,
            "zero_checkpoint_loads": True,
        },
        "optimizer_steps_executed": 0,
        "checkpoints_loaded": 0,
        "method_boundaries": [
            (
                "This is a generator and analytic-carrier preflight, not a "
                "trained TinyLLM result."
            ),
            (
                "The calibration packet contains amplitude, common offset, and "
                "common drift but no phase, speed, branch, or target."
            ),
            (
                "Passing licenses a prospective design only; it does not "
                "establish learned equivariance, natural utility, or causal "
                "closure."
            ),
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite C3 temporal preflight record")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/tinyllm-c3-temporal-quotient-preflight.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = build_preflight()
    _write_json(args.output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "classification": record["classification"],
                "valid": record["valid"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if record["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
