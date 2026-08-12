#!/usr/bin/env python3
"""Audit C3 switching-law identifiability and a fixed robust decoder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import sympy
import torch

from experiments.structure_net import (
    tinyllm_c3_constant_acceleration_fixed_operator as acceleration,
)
from experiments.structure_net import (
    tinyllm_c3_hidden_order_corruption_corrective as corrective,
)
from experiments.structure_net import (
    tinyllm_c3_single_frame_corruption_fixed_estimator as corruption,
)
from experiments.structure_net import tinyllm_c3_temporal_quotient_preflight as source


SCHEMA_VERSION = "nal.tinyllm-c3-switching-corruption-fixed-decoder.v1"
HYPOTHESIS_ID = "tinyllm-c3-switching-corruption-fixed-decoder-v1"
EVIDENCE_ROLE = "prospective_no_training_identifiability_fixed_ceiling_preflight"
SEEDS = (337, 347, 359, 373, 389)
REGIMES = ("composition", "extrapolation")
SWITCH_POINTS = (2, 3, 4)
LATE_SWITCH_POINTS = (2, 3, 4, 5)
ARMS = (
    "clean_known_switch",
    "clean_global_quadratic",
    "corrupted_global_quadratic",
    "corrupted_known_switch_no_drop",
    "oracle_switch_drop",
    "fixed_switch_drop",
)
SAMPLE_COUNT = 4_096
DATASET_SEED_BASES = {"composition": 941_107, "extrapolation": 943_107}
PILOT_DATASET_SEED_BASES = {
    "composition": 971_107,
    "extrapolation": 973_107,
}
DONOR_SEED_BASES = {"composition": 951_107, "extrapolation": 953_107}
FRAME_SEED_BASES = {"composition": 955_107, "extrapolation": 957_107}
SHUFFLE_SEED_BASES = {"composition": 961_107, "extrapolation": 963_107}
DELTA_VELOCITY_RANGES = {
    "composition": (0.05, 0.12),
    "extrapolation": (0.10, 0.18),
}
MINIMUM_SWITCH_COUNT = 1_200
MINIMUM_FRAME_COUNT = 400
ACTION_ERROR_MAXIMUM = corruption.ACTION_ERROR_MAXIMUM
CONTINUOUS_ERROR_MAXIMUM = corruption.CONTINUOUS_ERROR_MAXIMUM
CHART_MARGIN_MINIMUM = corruption.CHART_MARGIN_MINIMUM
PARETO_RMSE_RATIO_MAXIMUM = corrective.PARETO_RMSE_RATIO_MAXIMUM
PARETO_ACCURACY_DELTA_MINIMUM = corrective.PARETO_ACCURACY_DELTA_MINIMUM
PARETO_CROSS_ENTROPY_DELTA_MAXIMUM = (
    corrective.PARETO_CROSS_ENTROPY_DELTA_MAXIMUM
)
ORACLE_RMSE_SLACK_MAXIMUM = corruption.ORACLE_RMSE_SLACK_MAXIMUM
ORACLE_ACCURACY_SLACK_MAXIMUM = corruption.ORACLE_ACCURACY_SLACK_MAXIMUM
ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM = (
    corruption.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
)
REQUIRED_SEED_PASSES = 4
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "d7e7b0cd3774a0b661e331c0c6bf56631734152e05fdc156584952f56d6b6ee2"
)
CORRECTIVE_RUNNER_SHA256 = (
    "6d8e408f2347b86440ba6307345b86cad9f2740ee0945c358dea122d1c8d2789"
)
CORRECTIVE_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_corrective/"
    "20260811_preregistered/result.json"
)
CORRECTIVE_RESULT_SHA256 = (
    "bd5af6550e65c0b9048030a8f6d01cf80e06e9f027edb89066d8338bb8b47c2a"
)
CORRUPTION_RUNNER_SHA256 = (
    "8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8"
)
GENERATOR_SHA256 = "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
INTERVAL_SHA256 = "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_switching_corruption_fixed_decoder/"
    "20260811_preregistered/result.json"
)


@dataclass(frozen=True)
class SwitchingDataset:
    canonical_tokens: torch.Tensor
    tokens: torch.Tensor
    calibration: torch.Tensor
    target: torch.Tensor
    phase: torch.Tensor
    velocity: torch.Tensor
    delta_velocity: torch.Tensor
    switch: torch.Tensor
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
    return corruption._finite(value)


def validate_sources() -> dict[str, str]:
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "corrective_runner": _sha256(Path(corrective.__file__)),
        "corrective_result": _sha256(CORRECTIVE_RESULT_PATH),
        "single_frame_corruption_runner": _sha256(Path(corruption.__file__)),
        "generator": _sha256(Path(source.__file__)),
        "interval_likelihood": _sha256(Path(acceleration.joint.__file__)),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "corrective_runner": CORRECTIVE_RUNNER_SHA256,
        "corrective_result": CORRECTIVE_RESULT_SHA256,
        "single_frame_corruption_runner": CORRUPTION_RUNNER_SHA256,
        "generator": GENERATOR_SHA256,
        "interval_likelihood": INTERVAL_SHA256,
    }
    predecessor = json.loads(CORRECTIVE_RESULT_PATH.read_text(encoding="utf-8"))
    corrective.validate_sources()
    corruption.validate_sources()
    if (
        hashes != expected
        or predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != "fresh_corrective_confirms_common_robust_nested_law_closure"
    ):
        raise RuntimeError("switching-corruption source contract changed")
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


def phase_path(
    phase: torch.Tensor,
    velocity: torch.Tensor,
    delta_velocity: torch.Tensor,
    switch: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    time = torch.arange(steps, dtype=torch.float64)[None, :]
    hinge = (time - switch[:, None].double()).clamp_min(0.0)
    return (
        phase[:, None]
        + velocity[:, None] * time
        + delta_velocity[:, None] * hinge
    )


def _continuous_observation(
    phase: torch.Tensor,
    velocity: torch.Tensor,
    delta_velocity: torch.Tensor,
    switch: torch.Tensor,
    amplitude: torch.Tensor,
    offset: torch.Tensor,
    drift: torch.Tensor,
) -> torch.Tensor:
    theta = phase_path(
        phase,
        velocity,
        delta_velocity,
        switch,
        steps=source.TIME_STEPS,
    )
    time = torch.arange(source.TIME_STEPS, dtype=torch.float64)[None, :, None]
    channel = (
        2.0
        * math.pi
        * torch.arange(source.CHANNELS, dtype=torch.float64)[None, None, :]
        / source.CHANNELS
    )
    return (
        amplitude[:, None, None] * torch.cos(theta[:, :, None] + channel)
        + offset[:, None, None]
        + drift[:, None, None] * time
    )


def generate_dataset(
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> SwitchingDataset:
    primary = seed_value in SEEDS and sample_count == SAMPLE_COUNT
    pilot = allow_pilot and seed_value == 997 and sample_count == 64
    if regime not in REGIMES or not (primary or pilot):
        raise ValueError(f"unknown switching cohort {regime}/{seed_value}")
    bases = DATASET_SEED_BASES if primary else PILOT_DATASET_SEED_BASES
    dataset_seed = bases[regime] + seed_value
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    ranges = source.REGIME_RANGES[regime]
    phase = 2.0 * math.pi * torch.rand(
        sample_count, dtype=torch.float64, generator=generator
    )
    velocity = _signed_uniform(sample_count, ranges["speed"], generator)
    delta_velocity = _signed_uniform(
        sample_count, DELTA_VELOCITY_RANGES[regime], generator
    )
    switch = torch.randint(
        min(SWITCH_POINTS),
        max(SWITCH_POINTS) + 1,
        (sample_count,),
        dtype=torch.int64,
        generator=generator,
    )
    amplitude = _uniform(sample_count, ranges["amplitude"], generator)
    offset = _uniform(sample_count, ranges["offset"], generator)
    drift = _uniform(sample_count, ranges["drift"], generator)
    deck = torch.randint(
        0, source.CHANNELS, (sample_count,), generator=generator
    )
    continuous = _continuous_observation(
        phase,
        velocity,
        delta_velocity,
        switch,
        amplitude,
        offset,
        drift,
    )
    canonical = source.quantize(continuous)
    tokens = source._roll_by_example(canonical, deck)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    future = phase_path(
        phase,
        velocity,
        delta_velocity,
        switch,
        steps=source.TIME_STEPS + 1,
    )[:, -1]
    return SwitchingDataset(
        canonical_tokens=canonical,
        tokens=tokens,
        calibration=calibration,
        target=torch.cos(3.0 * future),
        phase=phase,
        velocity=velocity,
        delta_velocity=delta_velocity,
        switch=switch,
        deck=deck,
        saturation_count=int((continuous.abs() >= source.QUANTIZATION_LIMIT).sum()),
        dataset_seed=dataset_seed,
    )


def dataset_hash(dataset: SwitchingDataset) -> str:
    return _tensor_hash(
        dataset.canonical_tokens,
        dataset.tokens,
        dataset.calibration,
        dataset.target,
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        dataset.deck,
    )


def group_contracts(dataset: SwitchingDataset) -> dict[str, Any]:
    identity_error = int(
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
    order_error = int(
        (
            source.apply_deck_action(
                source.apply_deck_action(
                    source.apply_deck_action(dataset.canonical_tokens, 1), 1
                ),
                1,
            )
            != dataset.canonical_tokens
        ).sum()
    )
    stored_error = int(
        (
            source._roll_by_example(dataset.canonical_tokens, dataset.deck)
            != dataset.tokens
        ).sum()
    )
    shifted_phase = dataset.phase - (
        2.0 * math.pi * dataset.deck.double() / source.CHANNELS
    )
    independently_generated = source.quantize(
        _continuous_observation(
            shifted_phase,
            dataset.velocity,
            dataset.delta_velocity,
            dataset.switch,
            dataset.calibration[:, 0],
            dataset.calibration[:, 1],
            dataset.calibration[:, 2],
        )
    )
    latent_error = int((independently_generated != dataset.tokens).sum())
    future = phase_path(
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        steps=source.TIME_STEPS + 1,
    )[:, -1]
    target_error = 0.0
    for element in range(source.CHANNELS):
        transformed = torch.cos(
            3.0 * (future - 2.0 * math.pi * element / source.CHANNELS)
        )
        target_error = max(
            target_error, float((transformed - dataset.target).abs().max())
        )
    passed = bool(
        identity_error == 0
        and composition_errors == 0
        and order_error == 0
        and stored_error == 0
        and latent_error == 0
        and target_error <= 1e-12
        and dataset.saturation_count == 0
    )
    return {
        "identity_token_error_count": identity_error,
        "composition_token_error_count": composition_errors,
        "order_three_token_error_count": order_error,
        "stored_action_token_error_count": stored_error,
        "latent_generation_token_error_count": latent_error,
        "maximum_target_invariance_error": target_error,
        "pass": passed,
    }


def corruption_plan(
    regime: str,
    seed_value: int,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    donor = corrective.hidden.speed.sattolo_derangement(
        sample_count, DONOR_SEED_BASES[regime] + seed_value
    )
    generator = torch.Generator(device="cpu").manual_seed(
        FRAME_SEED_BASES[regime] + seed_value
    )
    frame = torch.randint(
        0,
        source.TIME_STEPS,
        (sample_count,),
        dtype=torch.int64,
        generator=generator,
    )
    return donor, frame


def _design(times: torch.Tensor, switch: int) -> torch.Tensor:
    return torch.stack(
        (
            torch.ones_like(times),
            times,
            (times - float(switch)).clamp_min(0.0),
        ),
        dim=1,
    )


def _fit_one_design(
    carrier: torch.Tensor,
    times: torch.Tensor,
    switch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = torch.angle(carrier).double()
    increments = corruption._principal_increment(phase[:, 1:] - phase[:, :-1])
    unwrapped = torch.cat(
        (phase[:, :1], phase[:, :1] + torch.cumsum(increments, dim=1)),
        dim=1,
    )
    design = _design(times, switch)
    coefficients = unwrapped @ torch.linalg.pinv(design).T
    fitted = coefficients @ design.T
    residual = (unwrapped - fitted).square().mean(dim=1)
    evaluation = torch.tensor(
        [1.0, float(source.TIME_STEPS), float(source.TIME_STEPS - switch)],
        dtype=torch.float64,
    )
    predicted_phase = coefficients @ evaluation
    prediction = torch.polar(
        torch.ones_like(predicted_phase), predicted_phase
    ).to(torch.complex128)
    return prediction, residual


def known_switch_prediction(
    carrier: torch.Tensor,
    switches: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.empty(
        carrier.shape[0], dtype=torch.complex128, device=carrier.device
    )
    times = torch.arange(source.TIME_STEPS, dtype=torch.float64)
    for switch in SWITCH_POINTS:
        rows = switches == switch
        if bool(rows.any()):
            prediction[rows], _ = _fit_one_design(
                carrier[rows], times, switch
            )
    return prediction


def switch_deletion_candidates(
    carrier: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    predictions = []
    residuals = []
    identities = []
    all_times = torch.arange(source.TIME_STEPS, dtype=torch.float64)
    for omitted in range(source.TIME_STEPS):
        keep = torch.arange(source.TIME_STEPS) != omitted
        times = all_times[keep]
        for switch in SWITCH_POINTS:
            prediction, residual = _fit_one_design(
                carrier[:, keep], times, switch
            )
            predictions.append(prediction)
            residuals.append(residual)
            identities.append((omitted, switch))
    return (
        torch.stack(predictions, dim=1),
        torch.stack(residuals, dim=1),
        identities,
    )


def switching_predictions(
    clean_carrier: torch.Tensor,
    corrupted_carrier: torch.Tensor,
    true_switch: torch.Tensor,
    true_frame: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_global = corruption.deletion_predictions(
        clean_carrier, torch.zeros_like(true_frame)
    )[1]
    corrupted_global = corruption.deletion_predictions(
        corrupted_carrier, true_frame
    )[1]
    candidates, residuals, identities = switch_deletion_candidates(
        corrupted_carrier
    )
    selected = residuals.argmin(dim=1)
    index = {identity: offset for offset, identity in enumerate(identities)}
    oracle_index = torch.tensor(
        [
            index[(int(frame), int(switch))]
            for frame, switch in zip(true_frame, true_switch)
        ],
        dtype=torch.int64,
    )
    row = torch.arange(clean_carrier.shape[0], dtype=torch.int64)
    return (
        {
            "clean_known_switch": known_switch_prediction(
                clean_carrier, true_switch
            ),
            "clean_global_quadratic": clean_global,
            "corrupted_global_quadratic": corrupted_global,
            "corrupted_known_switch_no_drop": known_switch_prediction(
                corrupted_carrier, true_switch
            ),
            "oracle_switch_drop": candidates[row, oracle_index],
            "fixed_switch_drop": candidates[row, selected],
        },
        selected,
        oracle_index,
        residuals,
    )


def _sympy_design(switch: int, time: int) -> sympy.Matrix:
    return sympy.Matrix([[1, time, max(time - switch, 0)]])


def _target_code_distance(switches: Sequence[int]) -> dict[str, Any]:
    times = tuple(range(source.TIME_STEPS))
    maximum_equal = -1
    witness: dict[str, Any] | None = None
    for left in switches:
        for right in switches:
            future = _sympy_design(left, source.TIME_STEPS).row_join(
                -_sympy_design(right, source.TIME_STEPS)
            )
            for equal_count in range(source.TIME_STEPS, -1, -1):
                found = False
                for subset in itertools.combinations(times, equal_count):
                    rows = [
                        _sympy_design(left, time).row_join(
                            -_sympy_design(right, time)
                        )
                        for time in subset
                    ]
                    matrix = sympy.Matrix.vstack(*rows) if rows else sympy.zeros(0, 6)
                    nullspace = matrix.nullspace()
                    vector = next(
                        (
                            item
                            for item in nullspace
                            if (future * item)[0] != 0
                        ),
                        None,
                    )
                    if vector is None:
                        continue
                    if equal_count > maximum_equal:
                        maximum_equal = equal_count
                        witness = {
                            "left_switch": left,
                            "right_switch": right,
                            "equal_times": list(subset),
                            "coefficient_vector": [str(item) for item in vector],
                            "future_phase_difference": str((future * vector)[0]),
                        }
                    found = True
                    break
                if found:
                    break
            if witness is not None and maximum_equal == source.TIME_STEPS:
                break
        if witness is not None and maximum_equal == source.TIME_STEPS:
            break
    if witness is None:
        raise RuntimeError("exact switching distance audit found no witness")
    return {
        "switch_points": list(switches),
        "maximum_equal_observed_coordinates": maximum_equal,
        "minimum_future_phase_code_distance": source.TIME_STEPS - maximum_equal,
        "witness": witness,
    }


def explicit_late_collision() -> dict[str, Any]:
    velocity = 0.08
    delta_a = 0.05
    delta_b = 0.10
    switch_a = 4
    switch_b = 5
    desired_a_future = math.pi / 6.0
    phase_value = (
        desired_a_future
        - velocity * source.TIME_STEPS
        - delta_a * (source.TIME_STEPS - switch_a)
    ) % (2.0 * math.pi)
    phase = torch.tensor([phase_value], dtype=torch.float64)
    velocity_tensor = torch.tensor([velocity], dtype=torch.float64)
    amplitude = torch.tensor([1.2], dtype=torch.float64)
    offset = torch.tensor([0.1], dtype=torch.float64)
    drift = torch.tensor([0.01], dtype=torch.float64)
    deck = 1

    def trajectory(delta: float, switch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_tensor = torch.tensor([delta], dtype=torch.float64)
        switch_tensor = torch.tensor([switch], dtype=torch.int64)
        continuous = _continuous_observation(
            phase,
            velocity_tensor,
            delta_tensor,
            switch_tensor,
            amplitude,
            offset,
            drift,
        )[0]
        tokens = source.apply_deck_action(source.quantize(continuous), deck)
        theta = phase_path(
            phase,
            velocity_tensor,
            delta_tensor,
            switch_tensor,
            steps=source.TIME_STEPS + 1,
        )[0]
        return continuous, tokens, theta

    continuous_a, tokens_a, theta_a = trajectory(delta_a, switch_a)
    continuous_b, tokens_b, theta_b = trajectory(delta_b, switch_b)
    observed_a = tokens_a.clone()
    observed_a[5] = tokens_b[5]
    observed_b = tokens_b.clone()
    observed_b[7] = tokens_a[7]
    continuous_observed_a = continuous_a.clone()
    continuous_observed_a[5] = continuous_b[5]
    continuous_observed_b = continuous_b.clone()
    continuous_observed_b[7] = continuous_a[7]
    target_a = math.cos(3.0 * float(theta_a[-1]))
    target_b = math.cos(3.0 * float(theta_b[-1]))
    equal_clean_frames = [
        time
        for time in range(source.TIME_STEPS)
        if torch.equal(tokens_a[time], tokens_b[time])
    ]
    return {
        "switch_a": switch_a,
        "switch_b": switch_b,
        "delta_velocity_a": delta_a,
        "delta_velocity_b": delta_b,
        "velocity": velocity,
        "phase": phase_value,
        "equal_clean_token_frames": equal_clean_frames,
        "continuous_collision_maximum_error": float(
            (continuous_observed_a - continuous_observed_b).abs().max()
        ),
        "token_collision_error_count": int((observed_a != observed_b).sum()),
        "target_a": target_a,
        "target_b": target_b,
        "absolute_target_separation": abs(target_a - target_b),
        "parameters_inside_composition_support": bool(
            source.REGIME_RANGES["composition"]["speed"][0]
            <= velocity
            <= source.REGIME_RANGES["composition"]["speed"][1]
            and DELTA_VELOCITY_RANGES["composition"][0]
            <= delta_a
            <= DELTA_VELOCITY_RANGES["composition"][1]
            and DELTA_VELOCITY_RANGES["composition"][0]
            <= delta_b
            <= DELTA_VELOCITY_RANGES["composition"][1]
        ),
    }


def identifiability_audit() -> dict[str, Any]:
    late = _target_code_distance(LATE_SWITCH_POINTS)
    primary = _target_code_distance(SWITCH_POINTS)
    collision = explicit_late_collision()
    passed = bool(
        late["minimum_future_phase_code_distance"] == 2
        and primary["minimum_future_phase_code_distance"] == 3
        and collision["continuous_collision_maximum_error"] <= 1e-12
        and collision["token_collision_error_count"] == 0
        and collision["absolute_target_separation"] >= 0.25
        and collision["parameters_inside_composition_support"]
    )
    return {
        "late_inclusive_support": late,
        "primary_support": primary,
        "explicit_late_collision": collision,
        "late_support_target_identifiable_under_one_corruption": False,
        "primary_support_one_corruption_correctable": bool(
            primary["minimum_future_phase_code_distance"] >= 3
        ),
        "pass": passed,
    }


def _chart_margin(carrier: torch.Tensor, true_frame: torch.Tensor) -> float:
    minimum = math.inf
    for omitted in range(source.TIME_STEPS):
        rows = true_frame == omitted
        if not bool(rows.any()):
            continue
        keep = torch.arange(source.TIME_STEPS) != omitted
        phase = torch.angle(carrier[rows][:, keep]).double()
        increments = corruption._principal_increment(
            phase[:, 1:] - phase[:, :-1]
        )
        minimum = min(minimum, float(math.pi - increments.abs().max()))
    return minimum


def _comparison_metrics(
    operators: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    global_corrupted = operators["corrupted_global_quadratic"]
    oracle = operators["oracle_switch_drop"]
    fixed = operators["fixed_switch_drop"]
    pareto = {
        "rmse_ratio": fixed["scalar"]["rmse"]
        / max(global_corrupted["scalar"]["rmse"], 1e-18),
        "accuracy_delta": fixed["task"]["exact_bin_accuracy"]
        - global_corrupted["task"]["exact_bin_accuracy"],
        "cross_entropy_delta": fixed["task"]["target_cross_entropy"]
        - global_corrupted["task"]["target_cross_entropy"],
    }
    pareto["pass"] = bool(
        fixed["fixed_ceiling_pass"]
        and pareto["rmse_ratio"] <= PARETO_RMSE_RATIO_MAXIMUM
        and pareto["accuracy_delta"] + 1e-12
        >= PARETO_ACCURACY_DELTA_MINIMUM
        and pareto["cross_entropy_delta"]
        <= PARETO_CROSS_ENTROPY_DELTA_MAXIMUM
    )
    fidelity = {
        "rmse_excess": fixed["scalar"]["rmse"] - oracle["scalar"]["rmse"],
        "accuracy_delta": fixed["task"]["exact_bin_accuracy"]
        - oracle["task"]["exact_bin_accuracy"],
        "cross_entropy_excess": fixed["task"]["target_cross_entropy"]
        - oracle["task"]["target_cross_entropy"],
    }
    fidelity["pass"] = bool(
        fidelity["rmse_excess"] <= ORACLE_RMSE_SLACK_MAXIMUM
        and fidelity["accuracy_delta"] >= -ORACLE_ACCURACY_SLACK_MAXIMUM
        and fidelity["cross_entropy_excess"]
        <= ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
    )
    return {"pareto_repair": pareto, "oracle_fidelity": fidelity}


def analyze_cell(
    regime: str,
    seed_value: int,
    audit: Mapping[str, Any],
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    registered = seed_value in SEEDS and sample_count == SAMPLE_COUNT
    dataset = generate_dataset(
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_hash = dataset_hash(dataset)
    replay = generate_dataset(
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    base_deterministic = base_hash == dataset_hash(replay)
    donor, frame = corruption_plan(regime, seed_value, sample_count)
    corrupted_tokens = corruption.corrupt_frames(dataset.tokens, donor, frame)
    replay_donor, replay_frame = corruption_plan(regime, seed_value, sample_count)
    corruption_deterministic = bool(
        torch.equal(donor, replay_donor)
        and torch.equal(frame, replay_frame)
        and torch.equal(
            corrupted_tokens,
            corruption.corrupt_frames(dataset.tokens, replay_donor, replay_frame),
        )
    )
    clean_carrier, clean_magnitude = source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    corrupted_carrier, corrupted_magnitude = source.analytic_carrier(
        corrupted_tokens, dataset.calibration
    )
    predictions, selected, oracle_index, residuals = switching_predictions(
        clean_carrier,
        corrupted_carrier,
        dataset.switch,
        frame,
    )
    action_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    equivariance_errors: dict[str, int] = {}
    for element in (1, 2):
        transformed_clean_tokens = source.apply_deck_action(dataset.tokens, element)
        transformed_corrupted_tokens = corruption.corrupt_frames(
            transformed_clean_tokens, donor, frame
        )
        equivariance_errors[str(element)] = int(
            (
                transformed_corrupted_tokens
                != source.apply_deck_action(corrupted_tokens, element)
            ).sum()
        )
        transformed_clean_carrier, _ = source.analytic_carrier(
            transformed_clean_tokens, dataset.calibration
        )
        transformed_corrupted_carrier, _ = source.analytic_carrier(
            transformed_corrupted_tokens, dataset.calibration
        )
        transformed_predictions, _, _, _ = switching_predictions(
            transformed_clean_carrier,
            transformed_corrupted_carrier,
            dataset.switch,
            frame,
        )
        for arm in ARMS:
            action_errors[arm][str(element)] = float(
                (transformed_predictions[arm] - predictions[arm]).abs().max()
            )

    exact_theta = phase_path(
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        steps=source.TIME_STEPS + 1,
    )
    exact_carrier = torch.polar(
        torch.ones_like(exact_theta), 3.0 * exact_theta
    ).to(torch.complex128)
    exact_corrupted = corruption.corrupt_frames(
        exact_carrier[:, :-1], donor, frame
    )
    exact_candidates, exact_residuals, exact_identities = (
        switch_deletion_candidates(exact_corrupted)
    )
    exact_selected = exact_residuals.argmin(dim=1)
    exact_index = {
        identity: offset for offset, identity in enumerate(exact_identities)
    }
    exact_oracle_index = torch.tensor(
        [
            exact_index[(int(item_frame), int(item_switch))]
            for item_frame, item_switch in zip(frame, dataset.switch)
        ],
        dtype=torch.int64,
    )
    row = torch.arange(sample_count, dtype=torch.int64)
    exact_oracle = exact_candidates[row, exact_oracle_index]
    exact_fixed = exact_candidates[row, exact_selected]
    exact_future = exact_carrier[:, -1]
    continuous_errors = {
        "oracle_switch_drop": float((exact_oracle - exact_future).abs().max()),
        "fixed_switch_drop": float((exact_fixed - exact_future).abs().max()),
    }
    exact_future_equivalent_count = int(
        ((exact_fixed - exact_future).abs() <= CONTINUOUS_ERROR_MAXIMUM).sum()
    )

    shuffle = corrective.hidden.speed.sattolo_derangement(
        sample_count, SHUFFLE_SEED_BASES[regime] + seed_value
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
    comparisons = _comparison_metrics(operators)
    switch_counts = torch.bincount(dataset.switch, minlength=6)
    frame_counts = torch.bincount(frame, minlength=source.TIME_STEPS)
    selected_frame = torch.tensor(
        [int(index) // len(SWITCH_POINTS) for index in selected],
        dtype=torch.int64,
    )
    selected_switch = torch.tensor(
        [SWITCH_POINTS[int(index) % len(SWITCH_POINTS)] for index in selected],
        dtype=torch.int64,
    )
    group = group_contracts(dataset)
    chart_margin = _chart_margin(exact_carrier[:, :-1], frame)
    donor_fixed_points = int(
        (donor == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    shuffle_fixed_points = int(
        (shuffle == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    required_switch_count = MINIMUM_SWITCH_COUNT if registered else 0
    required_frame_count = MINIMUM_FRAME_COUNT if registered else 0
    validity = bool(
        audit.get("pass") is True
        and dataset.saturation_count == 0
        and base_deterministic
        and group["pass"]
        and corruption_deterministic
        and donor_fixed_points == 0
        and min(int(switch_counts[item]) for item in SWITCH_POINTS)
        >= required_switch_count
        and int(frame_counts.min()) >= required_frame_count
        and sum(equivariance_errors.values()) == 0
        and all(operator["action_pass"] for operator in operators.values())
        and (
            all(operator["shuffle_pass"] for operator in operators.values())
            if registered
            else True
        )
        and max(continuous_errors.values()) <= CONTINUOUS_ERROR_MAXIMUM
        and exact_future_equivalent_count == sample_count
        and chart_margin >= CHART_MARGIN_MINIMUM
        and shuffle_fixed_points == 0
        and float(clean_magnitude.min()) > 1e-12
        and float(corrupted_magnitude.min()) > 1e-12
        and _finite(operators)
        and _finite(comparisons)
        and _finite(residuals)
    )
    return {
        "seed": seed_value,
        "regime": regime,
        "dataset_seed": dataset.dataset_seed,
        "base_dataset_sha256": base_hash,
        "base_dataset_deterministic": base_deterministic,
        "corruption_sha256": _tensor_hash(corrupted_tokens, donor, frame),
        "corruption_deterministic": corruption_deterministic,
        "sample_count": sample_count,
        "saturation_count": dataset.saturation_count,
        "switch_counts": {
            str(item): int(switch_counts[item]) for item in SWITCH_POINTS
        },
        "minimum_switch_count": min(
            int(switch_counts[item]) for item in SWITCH_POINTS
        ),
        "frame_counts": [int(item) for item in frame_counts],
        "minimum_frame_count": int(frame_counts.min()),
        "donor_fixed_points": donor_fixed_points,
        "shuffle_fixed_points": shuffle_fixed_points,
        "minimum_clean_chart_margin": chart_margin,
        "minimum_clean_character_magnitude": float(clean_magnitude.min()),
        "minimum_corrupted_character_magnitude": float(
            corrupted_magnitude.min()
        ),
        "base_group_contracts": group,
        "corruption_equivariance_token_errors": equivariance_errors,
        "continuous_prediction_maximum_errors": continuous_errors,
        "continuous_future_equivalent_count": exact_future_equivalent_count,
        "quantized_corruption_index_recovery_rate": float(
            (selected_frame == frame).double().mean()
        ),
        "quantized_switch_index_recovery_rate": float(
            (selected_switch == dataset.switch).double().mean()
        ),
        "quantized_joint_hidden_index_recovery_rate": float(
            (
                (selected_frame == frame)
                & (selected_switch == dataset.switch)
            ).double().mean()
        ),
        "quantized_oracle_candidate_selected_rate": float(
            (selected == oracle_index).double().mean()
        ),
        "operators": operators,
        "dynamics_material": bool(
            operators["clean_known_switch"]["fixed_ceiling_pass"]
            and not operators["clean_global_quadratic"]["fixed_ceiling_pass"]
        ),
        "corruption_material": bool(
            operators["clean_known_switch"]["fixed_ceiling_pass"]
            and not operators["corrupted_known_switch_no_drop"][
                "fixed_ceiling_pass"
            ]
        ),
        **comparisons,
        "valid": validity,
    }


def classify(
    cells: Sequence[Mapping[str, Any]],
    valid: bool,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    def cell(seed: int, regime: str) -> Mapping[str, Any]:
        return next(
            item
            for item in cells
            if item["seed"] == seed and item["regime"] == regime
        )

    def arm_count(arm: str) -> int:
        return sum(
            all(
                bool(cell(seed, regime)["operators"][arm]["fixed_ceiling_pass"])
                for regime in REGIMES
            )
            for seed in SEEDS
        )

    def field_count(field: str) -> int:
        return sum(
            all(bool(cell(seed, regime)[field]) for regime in REGIMES)
            for seed in SEEDS
        )

    def nested_count(field: str) -> int:
        return sum(
            all(bool(cell(seed, regime)[field]["pass"]) for regime in REGIMES)
            for seed in SEEDS
        )

    clean_count = arm_count("clean_known_switch")
    clean_global_count = arm_count("clean_global_quadratic")
    corrupted_global_count = arm_count("corrupted_global_quadratic")
    corrupted_known_count = arm_count("corrupted_known_switch_no_drop")
    oracle_count = arm_count("oracle_switch_drop")
    fixed_count = arm_count("fixed_switch_drop")
    dynamics_count = field_count("dynamics_material")
    corruption_count = field_count("corruption_material")
    pareto_count = nested_count("pareto_repair")
    fidelity_count = nested_count("oracle_fidelity")
    exact_contract = bool(audit.get("pass"))
    if not valid:
        classification = "invalid_switching_corruption_preflight"
    elif (
        exact_contract
        and dynamics_count >= REQUIRED_SEED_PASSES
        and corruption_count >= REQUIRED_SEED_PASSES
        and oracle_count >= REQUIRED_SEED_PASSES
        and fixed_count >= REQUIRED_SEED_PASSES
        and pareto_count >= REQUIRED_SEED_PASSES
        and fidelity_count >= REQUIRED_SEED_PASSES
    ):
        classification = (
            "fixed_change_point_decoder_closes_identifiable_switching_corruption"
        )
    elif (
        clean_global_count >= REQUIRED_SEED_PASSES
        or corrupted_global_count >= REQUIRED_SEED_PASSES
    ):
        classification = "switching_corruption_scope_not_material"
    elif oracle_count < REQUIRED_SEED_PASSES:
        classification = (
            "identifiable_switching_not_recoverable_at_required_ceiling"
        )
    elif fixed_count < REQUIRED_SEED_PASSES:
        classification = "recoverable_switching_exceeds_fixed_change_point_decoder"
    else:
        classification = "inconclusive_switching_corruption_preflight"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "identifiability_contract_pass": exact_contract,
        "late_support_target_identifiable_under_one_corruption": False,
        "primary_support_one_corruption_correctable": bool(
            audit.get("primary_support_one_corruption_correctable")
        ),
        "clean_known_switch_seed_pass_count": clean_count,
        "clean_global_quadratic_seed_pass_count": clean_global_count,
        "corrupted_global_quadratic_seed_pass_count": corrupted_global_count,
        "corrupted_known_switch_no_drop_seed_pass_count": corrupted_known_count,
        "oracle_switch_drop_seed_pass_count": oracle_count,
        "fixed_switch_drop_seed_pass_count": fixed_count,
        "dynamics_materiality_seed_pass_count": dynamics_count,
        "corruption_materiality_seed_pass_count": corruption_count,
        "pareto_repair_seed_pass_count": pareto_count,
        "oracle_fidelity_seed_pass_count": fidelity_count,
        "tinyllm_training_licensed": False,
        "compact_typed_continuation_comparison_licensed": classification
        == "recoverable_switching_exceeds_fixed_change_point_decoder",
    }


def build_result() -> dict[str, Any]:
    source_hashes = validate_sources()
    audit = identifiability_audit()
    cells = [
        analyze_cell(regime, seed, audit)
        for seed in SEEDS
        for regime in REGIMES
    ]
    valid = bool(
        audit["pass"]
        and len(cells) == len(SEEDS) * len(REGIMES)
        and all(cell["valid"] for cell in cells)
    )
    aggregates = classify(cells, valid, audit)
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
            "primary_switch_points": list(SWITCH_POINTS),
            "late_inclusive_switch_points": list(LATE_SWITCH_POINTS),
            "arms": list(ARMS),
            "sample_count_per_cell": SAMPLE_COUNT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "donor_seed_bases": DONOR_SEED_BASES,
            "frame_seed_bases": FRAME_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "delta_velocity_ranges": DELTA_VELOCITY_RANGES,
            "minimum_switch_count": MINIMUM_SWITCH_COUNT,
            "minimum_frame_count": MINIMUM_FRAME_COUNT,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "chart_margin_minimum": CHART_MARGIN_MINIMUM,
            "fixed_ceiling_rmse_maximum": corruption.FIXED_CEILING_RMSE_MAXIMUM,
            "fixed_ceiling_accuracy_minimum": (
                corruption.FIXED_CEILING_ACCURACY_MINIMUM
            ),
            "pareto_rmse_ratio_maximum": PARETO_RMSE_RATIO_MAXIMUM,
            "pareto_accuracy_delta_minimum": PARETO_ACCURACY_DELTA_MINIMUM,
            "pareto_cross_entropy_delta_maximum": (
                PARETO_CROSS_ENTROPY_DELTA_MAXIMUM
            ),
            "oracle_rmse_slack_maximum": ORACLE_RMSE_SLACK_MAXIMUM,
            "oracle_accuracy_slack_maximum": ORACLE_ACCURACY_SLACK_MAXIMUM,
            "oracle_cross_entropy_slack_maximum": (
                ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
            ),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": acceleration.TASK_GATES,
        },
        "identifiability_audit": audit,
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "fresh_base_examples": len(cells) * SAMPLE_COUNT,
            "fresh_corrupted_evaluations": len(cells) * SAMPLE_COUNT,
            "observation_only_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 42
            ),
            "continuous_validation_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 24
            ),
            "preregistration_pilot_examples_pooled": 0,
            "law_label_reads_by_fixed_decoder": 0,
            "reusable_parameter_fits": 0,
            "target_using_fits": 0,
            "models_instantiated": 0,
            "checkpoints_loaded": 0,
            "optimizer_steps": 0,
            "parameters_changed": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sympy": sympy.__version__,
            "device": "cpu",
        },
        "method_boundaries": [
            "The late-inclusive impossibility is an exact observation-level collision, not a sampled failure rate.",
            "The primary support requires three observed post-change frames and does not cover switch point five.",
            "The fixed decoder enumerates switch/deletion pairs per example and retains no learned parameter.",
            "Latent switch and corruption-index recovery are descriptive; future-target equivalence is the validity endpoint.",
            "The disclosed 1024-example pilots are excluded and no primary threshold was selected from them.",
            "No result licenses unrestricted TinyLLM training.",
        ],
    }
    if not _finite(result):
        raise RuntimeError("non-finite switching-corruption result")
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
                "late_code_distance": result["identifiability_audit"][
                    "late_inclusive_support"
                ]["minimum_future_phase_code_distance"],
                "primary_code_distance": result["identifiability_audit"][
                    "primary_support"
                ]["minimum_future_phase_code_distance"],
                "oracle_seed_passes": result["aggregates"][
                    "oracle_switch_drop_seed_pass_count"
                ],
                "fixed_seed_passes": result["aggregates"][
                    "fixed_switch_drop_seed_pass_count"
                ],
                "tinyllm_training_licensed": result["aggregates"][
                    "tinyllm_training_licensed"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
