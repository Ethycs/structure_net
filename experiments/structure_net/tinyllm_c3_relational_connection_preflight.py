#!/usr/bin/env python3
"""Preflight an identifiable C3 relational target with an observed connection."""

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


SCHEMA_VERSION = "nal.tinyllm-c3-relational-connection-preflight.v1"
HYPOTHESIS_ID = "tinyllm-c3-relational-connection-preflight-v1"
EVIDENCE_ROLE = "prospective_no_training_identifiability_and_fixed_ceiling"
SEEDS = (1213, 1231, 1277, 1301, 1321)
REGIMES = source.REGIMES
SAMPLE_COUNT = 4_096
TIME_STEPS = source.TIME_STEPS
CHANNELS = source.CHANNELS
REQUIRED_SEED_PASSES = 4
DATASET_SEED_BASES = {
    "composition": 1_121_107,
    "extrapolation": 1_123_107,
}
CONNECTION_SHUFFLE_SEED_BASES = {
    "composition": 1_125_107,
    "extrapolation": 1_127_107,
}
ACTION_SEED_BASES = {
    "composition": 1_129_107,
    "extrapolation": 1_131_107,
}
SECOND_ACTION_SEED_BASES = {
    "composition": 1_133_107,
    "extrapolation": 1_135_107,
}
TARGET_SHUFFLE_SEED_BASES = {
    "composition": 1_137_107,
    "extrapolation": 1_139_107,
}
POSITIVE_GATES = {
    "scalar_correlation_minimum": 0.999,
    "scalar_rmse_maximum": 0.01,
    "exact_bin_accuracy_minimum": 0.98,
    "target_cross_entropy_maximum": 1.35,
    "predicted_bin_coverage": 16,
    "local_action_prediction_error_maximum": 2e-12,
}
FAILURE_CONTROL_GATES = {
    "absolute_scalar_correlation_maximum": 0.10,
    "scalar_rmse_minimum": 0.80,
}
BAYES_ZERO_GATES = {
    "scalar_rmse_minimum": 0.65,
    "scalar_rmse_maximum": 0.76,
    "exact_bin_accuracy_maximum": 0.10,
    "predicted_bin_coverage": 1,
}
CONTINUOUS_ORACLE_ERROR_MAXIMUM = 1e-12
CHARACTER_MAGNITUDE_MINIMUM = 0.25
ACTION_ERROR_MAXIMUM = 2e-12
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-preflight-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "702196e0eaaaacda90293d202859e258889e1bed553dea4e2f7986f1ac8a57cc"
)
GENERATOR_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_preflight.py"
)
GENERATOR_SHA256 = (
    "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
)
INTERVAL_PATH = Path(
    "experiments/structure_net/tinyllm_joint_physical_scalar_interface.py"
)
INTERVAL_SHA256 = (
    "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
)
CLOSURE_PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-full-local-gauge-closure-preregistration.md"
)
CLOSURE_PREREGISTRATION_SHA256 = (
    "0f620df51c2f1d4278a8bbd82f8a6071c3159bdeb5e9420f4e08722c41115f90"
)
CLOSURE_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_full_local_gauge_closure.py"
)
CLOSURE_RUNNER_SHA256 = (
    "dae960759a2412451f8b15e15b0b6fb479603938a323ea54c86c77c68839005d"
)
CLOSURE_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_full_local_gauge_closure/"
    "20260811_artifact_audit/result.json"
)
CLOSURE_RESULT_SHA256 = (
    "31f7c7301c889db67e436fba4b8de1909dfbc372573681a9950b5b330f22db35"
)
CLOSURE_REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-c3-full-local-gauge-closure.md"
)
CLOSURE_REPORT_SHA256 = (
    "e1a237410bd3dfb050d22aadbfdf5b68f8b19a83c2977ae483fdebdbe468718e"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_preflight/"
    "20260811_preregistered/result.json"
)


@dataclass(frozen=True)
class RelationalDataset:
    canonical_tokens: torch.Tensor
    tokens: torch.Tensor
    calibration: torch.Tensor
    target: torch.Tensor
    phase: torch.Tensor
    delta: torch.Tensor
    gauge: torch.Tensor
    connection: torch.Tensor
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


def validate_sources() -> dict[str, str]:
    paths = {
        "preregistration": PREREGISTRATION_PATH,
        "generator": GENERATOR_PATH,
        "interval_likelihood": INTERVAL_PATH,
        "full_local_gauge_preregistration": CLOSURE_PREREGISTRATION_PATH,
        "full_local_gauge_runner": CLOSURE_RUNNER_PATH,
        "full_local_gauge_result": CLOSURE_RESULT_PATH,
        "full_local_gauge_report": CLOSURE_REPORT_PATH,
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "generator": GENERATOR_SHA256,
        "interval_likelihood": INTERVAL_SHA256,
        "full_local_gauge_preregistration": CLOSURE_PREREGISTRATION_SHA256,
        "full_local_gauge_runner": CLOSURE_RUNNER_SHA256,
        "full_local_gauge_result": CLOSURE_RESULT_SHA256,
        "full_local_gauge_report": CLOSURE_REPORT_SHA256,
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    if observed != expected:
        raise RuntimeError(f"relational preflight source contract changed: {observed}")
    predecessor = json.loads(CLOSURE_RESULT_PATH.read_text(encoding="utf-8"))
    aggregates = predecessor.get("aggregates", {})
    if (
        predecessor.get("status") != "completed"
        or aggregates.get("classification")
        != "pointwise_cubic_quotient_closes_full_local_c3_gauge"
        or aggregates.get("tinyllm_training_licensed") is not False
    ):
        raise RuntimeError("full local-gauge predecessor changed")
    return observed


def _uniform(
    count: int,
    bounds: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.empty(count, dtype=torch.float64).uniform_(
        *bounds, generator=generator
    )


def _continuous_observation(
    phase: torch.Tensor,
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
    return (
        amplitude[:, None, None] * torch.cos(phase[:, :, None] + channel)
        + offset[:, None, None]
        + drift[:, None, None] * time
    )


def apply_local_action(tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if tokens.shape[:-1] != action.shape or tokens.shape[-1] != CHANNELS:
        raise ValueError("local action shape does not match three-channel tokens")
    channel = torch.arange(CHANNELS, device=tokens.device)
    index = (channel - action[..., None]) % CHANNELS
    return torch.gather(tokens, -1, index)


def edge_connection(gauge: torch.Tensor) -> torch.Tensor:
    if gauge.ndim != 2 or gauge.shape[1] != TIME_STEPS:
        raise ValueError("gauge must have one C3 element per frame")
    return (gauge[:, 1:] - gauge[:, :-1]) % CHANNELS


def transform_connection(
    connection: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    if connection.shape != (action.shape[0], TIME_STEPS - 1):
        raise ValueError("connection/action shape mismatch")
    return (
        connection + action[:, 1:] - action[:, :-1]
    ) % CHANNELS


def generate_dataset(regime: str, replicate_seed: int) -> RelationalDataset:
    if regime not in REGIMES or replicate_seed not in SEEDS:
        raise ValueError(f"unknown relational cohort {regime}/{replicate_seed}")
    dataset_seed = DATASET_SEED_BASES[regime] + replicate_seed
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    theta_0 = 2.0 * math.pi * torch.rand(
        SAMPLE_COUNT, dtype=torch.float64, generator=generator
    )
    delta = math.pi * (
        2.0
        * torch.rand(SAMPLE_COUNT, dtype=torch.float64, generator=generator)
        - 1.0
    )
    interior = 2.0 * math.pi * torch.rand(
        SAMPLE_COUNT,
        TIME_STEPS - 2,
        dtype=torch.float64,
        generator=generator,
    )
    phase = torch.cat(
        (theta_0[:, None], interior, (theta_0 + delta)[:, None]), dim=1
    )
    ranges = source.REGIME_RANGES[regime]
    amplitude = _uniform(SAMPLE_COUNT, ranges["amplitude"], generator)
    offset = _uniform(SAMPLE_COUNT, ranges["offset"], generator)
    drift = _uniform(SAMPLE_COUNT, ranges["drift"], generator)
    gauge = torch.randint(
        0,
        CHANNELS,
        (SAMPLE_COUNT, TIME_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    continuous = _continuous_observation(phase, amplitude, offset, drift)
    saturation_count = int(
        (continuous.abs() >= source.QUANTIZATION_LIMIT).sum()
    )
    canonical_tokens = source.quantize(continuous)
    tokens = apply_local_action(canonical_tokens, gauge)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    return RelationalDataset(
        canonical_tokens=canonical_tokens,
        tokens=tokens,
        calibration=calibration,
        target=torch.cos(delta),
        phase=phase,
        delta=delta,
        gauge=gauge,
        connection=edge_connection(gauge),
        saturation_count=saturation_count,
        dataset_seed=dataset_seed,
    )


def dataset_hash(dataset: RelationalDataset) -> str:
    return _tensor_hash(
        dataset.canonical_tokens,
        dataset.tokens,
        dataset.calibration,
        dataset.target,
        dataset.phase,
        dataset.delta,
        dataset.gauge,
        dataset.connection,
    )


def charged_character(
    tokens: torch.Tensor, calibration: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = source.decode(tokens)
    amplitude = calibration[:, 0]
    offset = calibration[:, 1]
    drift = calibration[:, 2]
    time = torch.arange(TIME_STEPS, dtype=torch.float64)[None, :, None]
    corrected = (
        observed - offset[:, None, None] - drift[:, None, None] * time
    ) / amplitude[:, None, None]
    angles = (
        2.0 * math.pi * torch.arange(CHANNELS, dtype=torch.float64) / CHANNELS
    )
    weights = torch.polar(torch.ones_like(angles), -angles).to(torch.complex128)
    coefficient = torch.einsum(
        "btc,c->bt", corrected.to(torch.complex128), weights
    )
    magnitude = coefficient.abs()
    return coefficient / magnitude.clamp_min(1e-12), magnitude


def connection_prediction(
    character: torch.Tensor,
    connection: torch.Tensor,
    *,
    sign: int = 1,
) -> torch.Tensor:
    total = connection.sum(dim=1) % CHANNELS
    angle = sign * 2.0 * math.pi * total.to(torch.float64) / CHANNELS
    transport = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    return (character[:, -1] * transport * character[:, 0].conj()).real


def principal_invariant_prediction(character: torch.Tensor) -> torch.Tensor:
    relative_cube = character[:, -1].pow(3) * character[:, 0].pow(3).conj()
    return torch.cos(torch.angle(relative_cube) / 3.0)


def exact_continuous_prediction(dataset: RelationalDataset) -> torch.Tensor:
    angle = dataset.phase - (
        2.0 * math.pi * dataset.gauge.to(torch.float64) / CHANNELS
    )
    character = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    return connection_prediction(character, dataset.connection)


def sattolo_derangement(count: int, seed: int) -> torch.Tensor:
    if count < 2:
        raise ValueError("derangement requires at least two values")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.arange(count)
    for index in range(count - 1, 0, -1):
        other = int(torch.randint(0, index, (), generator=generator))
        saved = int(permutation[index])
        permutation[index] = permutation[other]
        permutation[other] = saved
    return permutation


def action_stream(
    regime: str,
    replicate_seed: int,
    count: int,
    *,
    second: bool = False,
) -> torch.Tensor:
    bases = SECOND_ACTION_SEED_BASES if second else ACTION_SEED_BASES
    generator = torch.Generator(device="cpu").manual_seed(
        bases[regime] + replicate_seed
    )
    return torch.randint(
        0,
        CHANNELS,
        (count, TIME_STEPS),
        generator=generator,
        dtype=torch.int64,
    )


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
    posterior_mean = posterior @ centers
    predicted_bin = posterior.argmax(dim=-1)
    target_bin = target_posterior.argmax(dim=-1)
    return {
        "posterior_mean_correlation": _correlation(posterior_mean, target),
        "posterior_mean_rmse": float(
            torch.sqrt((posterior_mean - target.double()).square().mean())
        ),
        "exact_bin_accuracy": float(
            (predicted_bin == target_bin).double().mean()
        ),
        "target_cross_entropy": float(
            -(
                target_posterior * posterior.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()
        ),
        "predicted_bin_coverage": int(torch.unique(predicted_bin).numel()),
        "target_bin_coverage": int(torch.unique(target_bin).numel()),
    }


def _positive_arm_pass(
    scalar: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    action_error: float = 0.0,
) -> bool:
    return bool(
        float(scalar["correlation"])
        >= POSITIVE_GATES["scalar_correlation_minimum"]
        and float(scalar["rmse"]) <= POSITIVE_GATES["scalar_rmse_maximum"]
        and float(task["exact_bin_accuracy"])
        >= POSITIVE_GATES["exact_bin_accuracy_minimum"]
        and float(task["target_cross_entropy"])
        <= POSITIVE_GATES["target_cross_entropy_maximum"]
        and int(task["predicted_bin_coverage"])
        == POSITIVE_GATES["predicted_bin_coverage"]
        and action_error
        <= POSITIVE_GATES["local_action_prediction_error_maximum"]
    )


def _failure_control_pass(
    scalar: Mapping[str, Any], task: Mapping[str, Any]
) -> bool:
    return bool(
        abs(float(scalar["correlation"]))
        <= FAILURE_CONTROL_GATES["absolute_scalar_correlation_maximum"]
        and float(scalar["rmse"])
        >= FAILURE_CONTROL_GATES["scalar_rmse_minimum"]
        and not _positive_arm_pass(scalar, task)
    )


def _arm(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    action_error: float = 0.0,
) -> dict[str, Any]:
    scalar = _scalar_metrics(prediction, target)
    task = _task_metrics(prediction, target)
    return {
        "scalar": scalar,
        "task": task,
        "positive_task_pass": _positive_arm_pass(
            scalar, task, action_error=action_error
        ),
        "failure_control_pass": _failure_control_pass(scalar, task),
    }


def observed_pair_contracts(
    dataset: RelationalDataset,
    first: torch.Tensor,
    second: torch.Tensor,
) -> dict[str, Any]:
    zero = torch.zeros_like(first)
    inverse = (-first) % CHANNELS
    direct_sum = (first + second) % CHANNELS

    identity_tokens = apply_local_action(dataset.tokens, zero)
    identity_connection = transform_connection(dataset.connection, zero)
    first_tokens = apply_local_action(dataset.tokens, first)
    first_connection = transform_connection(dataset.connection, first)
    inverse_tokens = apply_local_action(first_tokens, inverse)
    inverse_connection = transform_connection(first_connection, inverse)
    sequential_tokens = apply_local_action(first_tokens, second)
    sequential_connection = transform_connection(first_connection, second)
    direct_tokens = apply_local_action(dataset.tokens, direct_sum)
    direct_connection = transform_connection(dataset.connection, direct_sum)
    order_tokens = dataset.tokens
    order_connection = dataset.connection
    for _ in range(CHANNELS):
        order_tokens = apply_local_action(order_tokens, first)
        order_connection = transform_connection(order_connection, first)

    transformed_gauge = (dataset.gauge + first) % CHANNELS
    covariance_connection = edge_connection(transformed_gauge)
    covariance_tokens = apply_local_action(
        dataset.canonical_tokens, transformed_gauge
    )
    counts = {
        "identity_token_error_count": int(
            (identity_tokens != dataset.tokens).sum()
        ),
        "identity_connection_error_count": int(
            (identity_connection != dataset.connection).sum()
        ),
        "inverse_token_error_count": int((inverse_tokens != dataset.tokens).sum()),
        "inverse_connection_error_count": int(
            (inverse_connection != dataset.connection).sum()
        ),
        "composition_token_error_count": int(
            (sequential_tokens != direct_tokens).sum()
        ),
        "composition_connection_error_count": int(
            (sequential_connection != direct_connection).sum()
        ),
        "order_three_token_error_count": int(
            (order_tokens != dataset.tokens).sum()
        ),
        "order_three_connection_error_count": int(
            (order_connection != dataset.connection).sum()
        ),
        "covariant_token_error_count": int(
            (first_tokens != covariance_tokens).sum()
        ),
        "covariant_connection_error_count": int(
            (first_connection != covariance_connection).sum()
        ),
    }
    counts["pass"] = all(value == 0 for value in counts.values())
    return counts


def collision_witness() -> dict[str, Any]:
    phase_a = torch.tensor(
        [[0.0, 0.2, -0.4, 0.7, -1.0, 0.4, -0.2, 0.0]],
        dtype=torch.float64,
    )
    phase_b = phase_a.clone()
    phase_b[:, -1] += 2.0 * math.pi / CHANNELS
    amplitude = torch.ones(1, dtype=torch.float64)
    offset = torch.zeros(1, dtype=torch.float64)
    drift = torch.zeros(1, dtype=torch.float64)
    gauge_a = torch.zeros((1, TIME_STEPS), dtype=torch.int64)
    gauge_b = gauge_a.clone()
    gauge_b[:, -1] = 1
    canonical_a = _continuous_observation(phase_a, amplitude, offset, drift)
    canonical_b = _continuous_observation(phase_b, amplitude, offset, drift)
    observed_a = apply_local_action(canonical_a, gauge_a)
    observed_b = apply_local_action(canonical_b, gauge_b)
    tokens_a = source.quantize(canonical_a)
    tokens_b = apply_local_action(source.quantize(canonical_b), gauge_b)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    character_a, _ = charged_character(tokens_a, calibration)
    character_b, _ = charged_character(tokens_b, calibration)
    cube_error = float(
        (character_a.pow(3) - character_b.pow(3)).abs().max()
    )
    target_a = float(torch.cos(phase_a[:, -1] - phase_a[:, 0]))
    target_b = float(torch.cos(phase_b[:, -1] - phase_b[:, 0]))
    connection_a = edge_connection(gauge_a)
    connection_b = edge_connection(gauge_b)
    record = {
        "continuous_observation_maximum_error": float(
            (observed_a - observed_b).abs().max()
        ),
        "token_error_count_without_connection": int(
            (tokens_a != tokens_b).sum()
        ),
        "pointwise_cube_maximum_error": cube_error,
        "connection_difference_count": int(
            (connection_a != connection_b).sum()
        ),
        "target_a": target_a,
        "target_b": target_b,
        "target_absolute_separation": abs(target_a - target_b),
    }
    record["pass"] = bool(
        record["continuous_observation_maximum_error"] <= 1e-12
        and record["token_error_count_without_connection"] == 0
        and record["pointwise_cube_maximum_error"] <= 2e-12
        and record["connection_difference_count"] >= 1
        and record["target_absolute_separation"] >= 1.0
    )
    return record


def analyze_cell(regime: str, replicate_seed: int) -> dict[str, Any]:
    dataset = generate_dataset(regime, replicate_seed)
    repeated_hash = dataset_hash(generate_dataset(regime, replicate_seed))
    first_hash = dataset_hash(dataset)
    character, magnitude = charged_character(dataset.tokens, dataset.calibration)
    action = action_stream(regime, replicate_seed, SAMPLE_COUNT)
    second_action = action_stream(
        regime, replicate_seed, SAMPLE_COUNT, second=True
    )
    transformed_tokens = apply_local_action(dataset.tokens, action)
    transformed_connection = transform_connection(dataset.connection, action)
    transformed_character, transformed_magnitude = charged_character(
        transformed_tokens, dataset.calibration
    )
    expected_character = character * torch.polar(
        torch.ones_like(action, dtype=torch.float64),
        -2.0 * math.pi * action.to(torch.float64) / CHANNELS,
    ).to(torch.complex128)
    character_covariance_error = float(
        (transformed_character - expected_character).abs().max()
    )

    analytic_prediction = connection_prediction(character, dataset.connection)
    transformed_prediction = connection_prediction(
        transformed_character, transformed_connection
    )
    local_action_prediction_error = float(
        (transformed_prediction - analytic_prediction).abs().max()
    )
    continuous_prediction = exact_continuous_prediction(dataset)
    continuous_oracle_error = float(
        (continuous_prediction - dataset.target).abs().max()
    )

    connection_permutation = sattolo_derangement(
        SAMPLE_COUNT,
        CONNECTION_SHUFFLE_SEED_BASES[regime] + replicate_seed,
    )
    target_permutation = sattolo_derangement(
        SAMPLE_COUNT,
        TARGET_SHUFFLE_SEED_BASES[regime] + replicate_seed,
    )
    shuffled_connection = dataset.connection[connection_permutation]
    predictions = {
        "observed_connection": analytic_prediction,
        "no_connection": connection_prediction(
            character, torch.zeros_like(dataset.connection)
        ),
        "wrong_sign_connection": connection_prediction(
            character, dataset.connection, sign=-1
        ),
        "shuffled_connection": connection_prediction(
            character, shuffled_connection
        ),
        "principal_pointwise_invariant": principal_invariant_prediction(character),
        "pointwise_invariant_bayes_zero": torch.zeros_like(dataset.target),
    }
    arms = {
        name: _arm(
            prediction,
            dataset.target,
            action_error=(
                local_action_prediction_error
                if name == "observed_connection"
                else 0.0
            ),
        )
        for name, prediction in predictions.items()
    }
    bayes = arms["pointwise_invariant_bayes_zero"]
    bayes["bayes_zero_pass"] = bool(
        BAYES_ZERO_GATES["scalar_rmse_minimum"]
        <= bayes["scalar"]["rmse"]
        <= BAYES_ZERO_GATES["scalar_rmse_maximum"]
        and bayes["task"]["exact_bin_accuracy"]
        <= BAYES_ZERO_GATES["exact_bin_accuracy_maximum"]
        and bayes["task"]["predicted_bin_coverage"]
        == BAYES_ZERO_GATES["predicted_bin_coverage"]
        and not bayes["positive_task_pass"]
    )
    shuffled_target_metrics = _scalar_metrics(
        analytic_prediction, dataset.target[target_permutation]
    )
    shuffled_target_pass = bool(
        abs(shuffled_target_metrics["correlation"])
        <= FAILURE_CONTROL_GATES["absolute_scalar_correlation_maximum"]
        and shuffled_target_metrics["rmse"]
        >= FAILURE_CONTROL_GATES["scalar_rmse_minimum"]
    )
    pair_contracts = observed_pair_contracts(dataset, action, second_action)
    stored_token_errors = int(
        (
            dataset.tokens
            != apply_local_action(dataset.canonical_tokens, dataset.gauge)
        ).sum()
    )
    stored_connection_errors = int(
        (dataset.connection != edge_connection(dataset.gauge)).sum()
    )
    target_coverage = arms["observed_connection"]["task"]["target_bin_coverage"]
    connection_specificity_pass = bool(
        all(
            arms[name]["failure_control_pass"]
            for name in (
                "no_connection",
                "wrong_sign_connection",
                "shuffled_connection",
            )
        )
        and shuffled_target_pass
    )
    pointwise_insufficiency_pass = bool(
        arms["principal_pointwise_invariant"]["failure_control_pass"]
        and bayes["bayes_zero_pass"]
    )
    connection_free_shortcut_pass = bool(
        arms["no_connection"]["positive_task_pass"]
        or arms["principal_pointwise_invariant"]["positive_task_pass"]
        or arms["pointwise_invariant_bayes_zero"]["positive_task_pass"]
    )
    validity = bool(
        first_hash == repeated_hash
        and dataset.saturation_count == 0
        and stored_token_errors == 0
        and stored_connection_errors == 0
        and target_coverage == 16
        and float(magnitude.min()) >= CHARACTER_MAGNITUDE_MINIMUM
        and float(transformed_magnitude.min()) >= CHARACTER_MAGNITUDE_MINIMUM
        and continuous_oracle_error <= CONTINUOUS_ORACLE_ERROR_MAXIMUM
        and character_covariance_error <= ACTION_ERROR_MAXIMUM
        and local_action_prediction_error <= ACTION_ERROR_MAXIMUM
        and pair_contracts["pass"]
        and int(
            (connection_permutation == torch.arange(SAMPLE_COUNT)).sum()
        )
        == 0
        and int((target_permutation == torch.arange(SAMPLE_COUNT)).sum()) == 0
        and _finite(arms)
    )
    return {
        "seed": replicate_seed,
        "regime": regime,
        "dataset_seed": dataset.dataset_seed,
        "dataset_sha256": first_hash,
        "dataset_deterministic": first_hash == repeated_hash,
        "sample_count": SAMPLE_COUNT,
        "saturation_count": dataset.saturation_count,
        "stored_token_error_count": stored_token_errors,
        "stored_connection_error_count": stored_connection_errors,
        "target_bin_coverage": target_coverage,
        "minimum_character_magnitude": float(magnitude.min()),
        "minimum_transformed_character_magnitude": float(
            transformed_magnitude.min()
        ),
        "continuous_oracle_maximum_error": continuous_oracle_error,
        "character_covariance_maximum_error": character_covariance_error,
        "local_action_prediction_maximum_error": local_action_prediction_error,
        "connection_shuffle_fixed_points": int(
            (connection_permutation == torch.arange(SAMPLE_COUNT)).sum()
        ),
        "target_shuffle_fixed_points": int(
            (target_permutation == torch.arange(SAMPLE_COUNT)).sum()
        ),
        "observed_pair_contracts": pair_contracts,
        "arms": arms,
        "shuffled_target": {
            **shuffled_target_metrics,
            "pass": shuffled_target_pass,
        },
        "observed_connection_pass": arms["observed_connection"][
            "positive_task_pass"
        ],
        "connection_specificity_pass": connection_specificity_pass,
        "pointwise_insufficiency_pass": pointwise_insufficiency_pass,
        "connection_free_shortcut_pass": connection_free_shortcut_pass,
        "valid": validity,
    }


def _joint_seed_count(cells: Sequence[Mapping[str, Any]], field: str) -> int:
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


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    observed_count = _joint_seed_count(cells, "observed_connection_pass")
    specificity_count = _joint_seed_count(cells, "connection_specificity_pass")
    pointwise_count = _joint_seed_count(cells, "pointwise_insufficiency_pass")
    shortcut_count = _joint_seed_count(cells, "connection_free_shortcut_pass")
    if not valid:
        classification = "invalid_relational_connection_preflight"
    elif observed_count < REQUIRED_SEED_PASSES:
        classification = "relational_connection_quantization_ceiling_failed"
    elif shortcut_count >= REQUIRED_SEED_PASSES:
        classification = "relational_scope_has_connection_free_shortcut"
    elif specificity_count < REQUIRED_SEED_PASSES:
        classification = "relational_connection_not_causally_specific"
    elif pointwise_count >= REQUIRED_SEED_PASSES:
        classification = (
            "observed_edge_connection_identifies_nonpointwise_c3_relation"
        )
    else:
        classification = "relational_connection_preflight_inconclusive"
    return {
        "classification": classification,
        "valid": valid,
        "observed_connection_seed_pass_count": observed_count,
        "connection_specificity_seed_pass_count": specificity_count,
        "pointwise_insufficiency_seed_pass_count": pointwise_count,
        "connection_free_shortcut_seed_pass_count": shortcut_count,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "connection_conditioned_function_class_preflight_licensed": (
            classification
            == "observed_edge_connection_identifies_nonpointwise_c3_relation"
        ),
        "matched_training_directly_licensed": False,
        "unrestricted_tinyllm_training_licensed": False,
    }


def build_result() -> dict[str, Any]:
    source_hashes = validate_sources()
    witness = collision_witness()
    cells = [
        analyze_cell(regime, seed)
        for seed in SEEDS
        for regime in REGIMES
    ]
    valid = bool(
        len(cells) == len(SEEDS) * len(REGIMES)
        and witness["pass"]
        and all(cell["valid"] for cell in cells)
    )
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
            "time_steps": TIME_STEPS,
            "channels": CHANNELS,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "connection_shuffle_seed_bases": CONNECTION_SHUFFLE_SEED_BASES,
            "action_seed_bases": ACTION_SEED_BASES,
            "second_action_seed_bases": SECOND_ACTION_SEED_BASES,
            "target_shuffle_seed_bases": TARGET_SHUFFLE_SEED_BASES,
            "positive_gates": POSITIVE_GATES,
            "failure_control_gates": FAILURE_CONTROL_GATES,
            "bayes_zero_gates": BAYES_ZERO_GATES,
            "continuous_oracle_error_maximum": (
                CONTINUOUS_ORACLE_ERROR_MAXIMUM
            ),
            "character_magnitude_minimum": CHARACTER_MAGNITUDE_MINIMUM,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "nuisance_ranges": source.REGIME_RANGES,
        },
        "identifiability": {
            "continuous_formula": (
                "Re[c7*exp(+2pi*i*sum(edge)/3)*conj(c0)] = cos(theta7-theta0)"
            ),
            "pointwise_bayes_formula": (
                "E[cos(delta)|exp(i*3*delta)] = 0 for uniform delta"
            ),
            "collision_witness": witness,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "new_evaluation_examples": len(cells) * SAMPLE_COUNT,
            "local_action_counterfactual_evaluations": len(cells) * SAMPLE_COUNT,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "checkpoints_loaded": 0,
            "tinyllm_models_instantiated": 0,
            "reusable_parameter_fits": 0,
            "target_using_fits": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            (
                "Exact identifiability is a continuous calibrated-observation "
                "statement; token quantization is tested by a separate fixed ceiling."
            ),
            (
                "Interior phases are independent of the endpoint relation, so a "
                "smooth pointwise-cubic trajectory cannot unwrap the target."
            ),
            (
                "The observed edge connection contains local sensor-frame "
                "differences only and receives no phase or target data."
            ),
            (
                "A positive outcome licenses only a no-training function-class "
                "and lifecycle prerequisite, never direct or unrestricted training."
            ),
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite relational connection preflight result")
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
                "observed_connection_seeds": result["aggregates"][
                    "observed_connection_seed_pass_count"
                ],
                "connection_specificity_seeds": result["aggregates"][
                    "connection_specificity_seed_pass_count"
                ],
                "pointwise_insufficiency_seeds": result["aggregates"][
                    "pointwise_insufficiency_seed_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
