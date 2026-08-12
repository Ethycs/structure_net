#!/usr/bin/env python3
"""Test a fixed discrete C3 connection under one hidden gauge jump."""

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

from experiments.structure_net import (
    tinyllm_c3_charged_character_fixed_decoder_corrective as charged,
)


base = charged.base
source = base.source
SCHEMA_VERSION = "nal.tinyllm-c3-gauge-jump-corruption-fixed-decoder.v1"
HYPOTHESIS_ID = "tinyllm-c3-gauge-jump-corruption-fixed-decoder-v1"
EVIDENCE_ROLE = "prospective_no_training_time_varying_gauge_fixed_decoder"
SEEDS = (673, 691, 709, 727, 751)
REGIMES = charged.REGIMES
SWITCH_POINTS = charged.SWITCH_POINTS
JUMP_TIMES = tuple(range(1, 7))
JUMP_ELEMENTS = (1, 2)
ARMS = (
    "oracle_invariant_switch_drop",
    "fixed_invariant_switch_drop",
    "fixed_charged_no_connection",
    "oracle_charged_connection",
    "fixed_charged_connection",
)
SAMPLE_COUNT = charged.SAMPLE_COUNT
DATASET_SEED_BASES = {"composition": 1_047_107, "extrapolation": 1_049_107}
PILOT_DATASET_SEED_BASES = {
    "composition": 1_071_107,
    "extrapolation": 1_073_107,
}
DONOR_SEED_BASES = {"composition": 1_051_107, "extrapolation": 1_053_107}
FRAME_SEED_BASES = {"composition": 1_055_107, "extrapolation": 1_057_107}
JUMP_TIME_SEED_BASES = {"composition": 1_059_107, "extrapolation": 1_061_107}
JUMP_ELEMENT_SEED_BASES = {
    "composition": 1_063_107,
    "extrapolation": 1_065_107,
}
SHUFFLE_SEED_BASES = {"composition": 1_067_107, "extrapolation": 1_069_107}
PILOT_SEED = 1019
MINIMUM_SWITCH_COUNT = 1_200
MINIMUM_FRAME_COUNT = 400
MINIMUM_JUMP_TIME_COUNT = 580
MINIMUM_JUMP_ELEMENT_COUNT = 1_800
MINIMUM_JOINT_JUMP_COUNT = 280
ACTION_ERROR_MAXIMUM = 2e-12
CHARGED_EQUIVARIANCE_ERROR_MAXIMUM = 2e-12
CONTINUOUS_ERROR_MAXIMUM = 1e-10
CHART_MARGIN_MINIMUM = 0.20
STABILIZATION_DISPLACEMENT_MAXIMUM = 1e-12
REQUIRED_SEED_PASSES = 4
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-gauge-jump-corruption-fixed-decoder-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "caab4704bb5c11a3f7353c31e20a80da33f14e1a925402df055b652916de10d9"
)
PARENT_RUNNER_SHA256 = (
    "5fcd691d3fd910a619bd61aa8dd0432e0050efc44699657ec98f5d7e2e01de97"
)
PARENT_RESULT_PATH = charged.PRIMARY_RESULT_PATH
PARENT_RESULT_SHA256 = (
    "d8076bd7bfc42819a17438e188300e8ea0131dcc72d9ceac0bee39ede72fcaaf"
)
PARENT_REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-charged-character-fixed-decoder.md"
)
PARENT_REPORT_SHA256 = (
    "9acd88cf100b67f04a27c5415dcbce755b97c3545d6ba78ce114e97885c5ba10"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_gauge_jump_corruption_fixed_decoder/"
    "20260811_preregistered/result.json"
)


@dataclass(frozen=True)
class GaugeJumpDataset:
    canonical_tokens: torch.Tensor
    global_tokens: torch.Tensor
    tokens: torch.Tensor
    calibration: torch.Tensor
    target: torch.Tensor
    phase: torch.Tensor
    velocity: torch.Tensor
    delta_velocity: torch.Tensor
    switch: torch.Tensor
    deck: torch.Tensor
    jump_time: torch.Tensor
    jump_element: torch.Tensor
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_sources() -> dict[str, str]:
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "charged_corrective_runner": _sha256(Path(charged.__file__)),
        "charged_corrective_result": _sha256(PARENT_RESULT_PATH),
        "charged_corrective_report": _sha256(PARENT_REPORT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "charged_corrective_runner": PARENT_RUNNER_SHA256,
        "charged_corrective_result": PARENT_RESULT_SHA256,
        "charged_corrective_report": PARENT_REPORT_SHA256,
    }
    charged.validate_sources()
    predecessor = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        hashes != expected
        or predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != "exact_charged_character_corrective_closes_invariant_quantization_gap"
        or predecessor.get("aggregates", {}).get(
            "compact_typed_continuation_comparison_licensed"
        )
        is not False
    ):
        raise RuntimeError("gauge-jump source contract changed")
    return hashes


def apply_suffix_jump(
    tokens: torch.Tensor,
    jump_time: torch.Tensor,
    jump_element: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != source.TIME_STEPS:
        raise ValueError("expected [batch,time,channel] token tensor")
    if jump_time.shape != (tokens.shape[0],) or jump_element.shape != (
        tokens.shape[0],
    ):
        raise ValueError("jump arrays must match token batch")
    output = tokens.clone()
    for time in range(source.TIME_STEPS):
        elements = torch.where(
            torch.tensor(time, dtype=torch.int64) >= jump_time,
            jump_element,
            torch.zeros_like(jump_element),
        )
        if inverse:
            elements = (-elements) % source.CHANNELS
        output[:, time] = source._roll_by_example(output[:, time], elements)
    return output


def apply_constant_jump_correction(
    tokens: torch.Tensor,
    jump_time: int,
    jump_element: int,
) -> torch.Tensor:
    if jump_time not in JUMP_TIMES or jump_element not in JUMP_ELEMENTS:
        raise ValueError("unknown fixed connection candidate")
    corrected = tokens.clone()
    corrected[:, jump_time:] = torch.roll(
        corrected[:, jump_time:], shifts=-jump_element, dims=-1
    )
    return corrected


def generate_dataset(
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> GaugeJumpDataset:
    primary = seed_value in SEEDS and sample_count == SAMPLE_COUNT
    pilot = (
        allow_pilot and seed_value == PILOT_SEED and sample_count == 64
    )
    if regime not in REGIMES or not (primary or pilot):
        raise ValueError(f"unknown gauge-jump cohort {regime}/{seed_value}")
    bases = DATASET_SEED_BASES if primary else PILOT_DATASET_SEED_BASES
    dataset_seed = bases[regime] + seed_value
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    ranges = source.REGIME_RANGES[regime]
    phase = 2.0 * math.pi * torch.rand(
        sample_count, dtype=torch.float64, generator=generator
    )
    velocity = base._signed_uniform(sample_count, ranges["speed"], generator)
    delta_velocity = base._signed_uniform(
        sample_count, base.DELTA_VELOCITY_RANGES[regime], generator
    )
    switch = torch.randint(
        min(SWITCH_POINTS),
        max(SWITCH_POINTS) + 1,
        (sample_count,),
        dtype=torch.int64,
        generator=generator,
    )
    amplitude = base._uniform(sample_count, ranges["amplitude"], generator)
    offset = base._uniform(sample_count, ranges["offset"], generator)
    drift = base._uniform(sample_count, ranges["drift"], generator)
    deck = torch.randint(
        0, source.CHANNELS, (sample_count,), generator=generator
    )
    continuous = base._continuous_observation(
        phase,
        velocity,
        delta_velocity,
        switch,
        amplitude,
        offset,
        drift,
    )
    canonical = source.quantize(continuous)
    global_tokens = source._roll_by_example(canonical, deck)
    jump_time_generator = torch.Generator(device="cpu").manual_seed(
        JUMP_TIME_SEED_BASES[regime] + seed_value
    )
    jump_time = torch.randint(
        min(JUMP_TIMES),
        max(JUMP_TIMES) + 1,
        (sample_count,),
        dtype=torch.int64,
        generator=jump_time_generator,
    )
    jump_element_generator = torch.Generator(device="cpu").manual_seed(
        JUMP_ELEMENT_SEED_BASES[regime] + seed_value
    )
    jump_element = torch.randint(
        min(JUMP_ELEMENTS),
        max(JUMP_ELEMENTS) + 1,
        (sample_count,),
        dtype=torch.int64,
        generator=jump_element_generator,
    )
    tokens = apply_suffix_jump(global_tokens, jump_time, jump_element)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    future = base.phase_path(
        phase,
        velocity,
        delta_velocity,
        switch,
        steps=source.TIME_STEPS + 1,
    )[:, -1]
    return GaugeJumpDataset(
        canonical_tokens=canonical,
        global_tokens=global_tokens,
        tokens=tokens,
        calibration=calibration,
        target=torch.cos(3.0 * future),
        phase=phase,
        velocity=velocity,
        delta_velocity=delta_velocity,
        switch=switch,
        deck=deck,
        jump_time=jump_time,
        jump_element=jump_element,
        saturation_count=int(
            (continuous.abs() >= source.QUANTIZATION_LIMIT).sum()
        ),
        dataset_seed=dataset_seed,
    )


def dataset_hash(dataset: GaugeJumpDataset) -> str:
    return base._tensor_hash(
        dataset.canonical_tokens,
        dataset.global_tokens,
        dataset.tokens,
        dataset.calibration,
        dataset.target,
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        dataset.deck,
        dataset.jump_time,
        dataset.jump_element,
    )


def corruption_plan(
    regime: str,
    seed_value: int,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    donor = base.corrective.hidden.speed.sattolo_derangement(
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


def group_and_jump_contracts(dataset: GaugeJumpDataset) -> dict[str, Any]:
    physical = base.SwitchingDataset(
        canonical_tokens=dataset.canonical_tokens,
        tokens=dataset.global_tokens,
        calibration=dataset.calibration,
        target=dataset.target,
        phase=dataset.phase,
        velocity=dataset.velocity,
        delta_velocity=dataset.delta_velocity,
        switch=dataset.switch,
        deck=dataset.deck,
        saturation_count=dataset.saturation_count,
        dataset_seed=dataset.dataset_seed,
    )
    inherited = base.group_contracts(physical)
    constructed = apply_suffix_jump(
        dataset.global_tokens, dataset.jump_time, dataset.jump_element
    )
    restored = apply_suffix_jump(
        dataset.tokens,
        dataset.jump_time,
        dataset.jump_element,
        inverse=True,
    )
    constructed_error = int((constructed != dataset.tokens).sum())
    inverse_error = int((restored != dataset.global_tokens).sum())
    commute_errors: dict[str, int] = {}
    for element in (1, 2):
        left = apply_suffix_jump(
            source.apply_deck_action(dataset.global_tokens, element),
            dataset.jump_time,
            dataset.jump_element,
        )
        right = source.apply_deck_action(dataset.tokens, element)
        commute_errors[str(element)] = int((left != right).sum())
    global_carrier, _ = source.analytic_carrier(
        dataset.global_tokens, dataset.calibration
    )
    jumped_carrier, _ = source.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    invariant_error = float((global_carrier - jumped_carrier).abs().max())
    passed = bool(
        inherited["pass"]
        and constructed_error == 0
        and inverse_error == 0
        and sum(commute_errors.values()) == 0
        and invariant_error <= ACTION_ERROR_MAXIMUM
    )
    return {
        "inherited_global_group_contracts": inherited,
        "jump_construction_token_error_count": constructed_error,
        "jump_inverse_token_error_count": inverse_error,
        "global_jump_commutation_token_errors": commute_errors,
        "maximum_cubic_carrier_jump_invariance_error": invariant_error,
        "pass": passed,
    }


def connection_candidates(
    tokens: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[tuple[int, int, int, int]],
    float,
]:
    predictions = []
    residuals = []
    identities = []
    maximum_displacement = 0.0
    for jump_time in JUMP_TIMES:
        for jump_element in JUMP_ELEMENTS:
            corrected = apply_constant_jump_correction(
                tokens, jump_time, jump_element
            )
            candidate, residual, subidentities, displacement = (
                charged.exact_charged_deletion_candidates(corrected)
            )
            predictions.append(candidate)
            residuals.append(residual)
            expected_subidentities = [
                (omitted, switch)
                for omitted in range(source.TIME_STEPS)
                for switch in SWITCH_POINTS
            ]
            if subidentities != expected_subidentities:
                raise RuntimeError("charged subcandidate order changed")
            identities.extend(
                (omitted, switch, jump_time, jump_element)
                for omitted, switch in expected_subidentities
            )
            maximum_displacement = max(maximum_displacement, displacement)
    return (
        torch.cat(predictions, dim=1),
        torch.cat(residuals, dim=1),
        identities,
        maximum_displacement,
    )


def connection_identities() -> list[tuple[int, int, int, int]]:
    return [
        (omitted, switch, jump_time, jump_element)
        for jump_time in JUMP_TIMES
        for jump_element in JUMP_ELEMENTS
        for omitted in range(source.TIME_STEPS)
        for switch in SWITCH_POINTS
    ]


def decoder_predictions(
    invariant: torch.Tensor,
    tokens: torch.Tensor,
    true_switch: torch.Tensor,
    true_frame: torch.Tensor,
    true_jump_time: torch.Tensor,
    true_jump_element: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    stable_invariant, invariant_displacement = charged.parent.stabilize_carrier(
        invariant
    )
    invariant_candidates, invariant_residuals, invariant_identities = (
        base.switch_deletion_candidates(stable_invariant)
    )
    no_connection_candidates, no_connection_residuals, _, no_connection_displacement = (
        charged.exact_charged_deletion_candidates(tokens)
    )
    (
        connected_candidates,
        connected_residuals,
        connection_identities,
        connection_displacement,
    ) = connection_candidates(tokens)
    invariant_index = {
        identity: offset
        for offset, identity in enumerate(invariant_identities)
    }
    connection_index = {
        identity: offset
        for offset, identity in enumerate(connection_identities)
    }
    oracle_invariant = torch.tensor(
        [
            invariant_index[(int(frame), int(switch))]
            for frame, switch in zip(true_frame, true_switch)
        ],
        dtype=torch.int64,
    )
    oracle_connection = torch.tensor(
        [
            connection_index[(
                int(frame),
                int(switch),
                int(jump_time),
                int(jump_element),
            )]
            for frame, switch, jump_time, jump_element in zip(
                true_frame,
                true_switch,
                true_jump_time,
                true_jump_element,
            )
        ],
        dtype=torch.int64,
    )
    fixed_invariant = invariant_residuals.argmin(dim=1)
    fixed_no_connection = no_connection_residuals.argmin(dim=1)
    fixed_connection = connected_residuals.argmin(dim=1)
    row = torch.arange(invariant.shape[0], dtype=torch.int64)
    predictions = {
        "oracle_invariant_switch_drop": invariant_candidates[
            row, oracle_invariant
        ],
        "fixed_invariant_switch_drop": invariant_candidates[
            row, fixed_invariant
        ],
        "fixed_charged_no_connection": no_connection_candidates[
            row, fixed_no_connection
        ],
        "oracle_charged_connection": connected_candidates[
            row, oracle_connection
        ],
        "fixed_charged_connection": connected_candidates[
            row, fixed_connection
        ],
    }
    selections = {
        "oracle_invariant": oracle_invariant,
        "fixed_invariant": fixed_invariant,
        "fixed_no_connection": fixed_no_connection,
        "oracle_connection": oracle_connection,
        "fixed_connection": fixed_connection,
    }
    return predictions, selections, {
        "invariant": invariant_displacement,
        "no_connection": no_connection_displacement,
        "connection": connection_displacement,
    }


def connected_integer_contract(
    tokens: torch.Tensor,
    transformed_tokens: torch.Tensor,
) -> dict[str, int | bool]:
    relative_errors = 0
    cube_errors = 0
    for jump_time in JUMP_TIMES:
        for jump_element in JUMP_ELEMENTS:
            natural = apply_constant_jump_correction(
                tokens, jump_time, jump_element
            )
            transformed = apply_constant_jump_correction(
                transformed_tokens, jump_time, jump_element
            )
            contract = charged.exact_pair_contract(natural, transformed)
            relative_errors += int(contract["relative_integer_error_count"])
            cube_errors += int(contract["anchor_cube_integer_error_count"])
    return {
        "relative_integer_error_count": relative_errors,
        "anchor_cube_integer_error_count": cube_errors,
        "pass": relative_errors == 0 and cube_errors == 0,
    }


def apply_complex_suffix_jump(
    character: torch.Tensor,
    jump_time: torch.Tensor,
    jump_element: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    output = character.clone()
    for time in range(source.TIME_STEPS):
        active = time >= jump_time
        signed = -jump_element.double()
        if inverse:
            signed = -signed
        angle = 2.0 * math.pi * signed / source.CHANNELS
        factor = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
        output[:, time] = torch.where(
            active,
            output[:, time] * factor,
            output[:, time],
        )
    return output


def floating_connection_candidates(
    character: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[tuple[int, int, int, int]],
    float,
]:
    predictions = []
    residuals = []
    identities = []
    maximum_displacement = 0.0
    for jump_time in JUMP_TIMES:
        for jump_element in JUMP_ELEMENTS:
            corrected = character.clone()
            root = torch.polar(
                torch.tensor(1.0, dtype=torch.float64),
                torch.tensor(
                    2.0 * math.pi * jump_element / source.CHANNELS,
                    dtype=torch.float64,
                ),
            ).to(torch.complex128)
            corrected[:, jump_time:] *= root
            candidate, residual, subidentities, displacement = (
                charged.invalid.charged_deletion_candidates(corrected)
            )
            predictions.append(candidate)
            residuals.append(residual)
            identities.extend(
                (omitted, switch, jump_time, jump_element)
                for omitted, switch in subidentities
            )
            maximum_displacement = max(maximum_displacement, displacement)
    return (
        torch.cat(predictions, dim=1),
        torch.cat(residuals, dim=1),
        identities,
        maximum_displacement,
    )


def continuous_validation(
    dataset: GaugeJumpDataset,
    donor: torch.Tensor,
    frame: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    exact_theta = base.phase_path(
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        steps=source.TIME_STEPS + 1,
    )
    deck_phase = 2.0 * math.pi * dataset.deck.double() / source.CHANNELS
    global_charged = torch.polar(
        torch.ones_like(exact_theta[:, :-1]),
        exact_theta[:, :-1] - deck_phase[:, None],
    ).to(torch.complex128)
    jumped_charged = apply_complex_suffix_jump(
        global_charged, dataset.jump_time, dataset.jump_element
    )
    corrupted_charged = base.corruption.corrupt_frames(
        jumped_charged, donor, frame
    )
    corrupted_invariant = corrupted_charged.pow(3)
    stable_invariant, invariant_displacement = charged.parent.stabilize_carrier(
        corrupted_invariant
    )
    invariant_candidates, invariant_residuals, invariant_identities = (
        base.switch_deletion_candidates(stable_invariant)
    )
    connected_candidates, connected_residuals, identities, displacement = (
        floating_connection_candidates(corrupted_charged)
    )
    invariant_index = {
        identity: offset for offset, identity in enumerate(invariant_identities)
    }
    connection_index = {
        identity: offset for offset, identity in enumerate(identities)
    }
    oracle_invariant = torch.tensor(
        [
            invariant_index[(int(item_frame), int(item_switch))]
            for item_frame, item_switch in zip(frame, dataset.switch)
        ],
        dtype=torch.int64,
    )
    oracle_connection = torch.tensor(
        [
            connection_index[(
                int(item_frame),
                int(item_switch),
                int(item_jump_time),
                int(item_jump_element),
            )]
            for item_frame, item_switch, item_jump_time, item_jump_element in zip(
                frame,
                dataset.switch,
                dataset.jump_time,
                dataset.jump_element,
            )
        ],
        dtype=torch.int64,
    )
    row = torch.arange(dataset.target.shape[0], dtype=torch.int64)
    predictions = {
        "oracle_invariant_switch_drop": invariant_candidates[
            row, oracle_invariant
        ],
        "fixed_invariant_switch_drop": invariant_candidates[
            row, invariant_residuals.argmin(dim=1)
        ],
        "oracle_charged_connection": connected_candidates[
            row, oracle_connection
        ],
        "fixed_charged_connection": connected_candidates[
            row, connected_residuals.argmin(dim=1)
        ],
    }
    exact_future = torch.polar(
        torch.ones_like(exact_theta[:, -1]), 3.0 * exact_theta[:, -1]
    ).to(torch.complex128)
    errors = {
        arm: float((prediction - exact_future).abs().max())
        for arm, prediction in predictions.items()
    }
    return errors, {
        "invariant": invariant_displacement,
        "connection": displacement,
    }


def _fidelity(
    fixed: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "rmse_excess": fixed["scalar"]["rmse"] - oracle["scalar"]["rmse"],
        "accuracy_delta": fixed["task"]["exact_bin_accuracy"]
        - oracle["task"]["exact_bin_accuracy"],
        "cross_entropy_excess": fixed["task"]["target_cross_entropy"]
        - oracle["task"]["target_cross_entropy"],
    }
    value["pass"] = bool(
        value["rmse_excess"] <= charged.invalid.ORACLE_RMSE_SLACK_MAXIMUM
        and value["accuracy_delta"]
        >= -charged.invalid.ORACLE_ACCURACY_SLACK_MAXIMUM
        and value["cross_entropy_excess"]
        <= charged.invalid.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
    )
    return value


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
    dataset_digest = dataset_hash(dataset)
    replay = generate_dataset(
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    dataset_deterministic = dataset_digest == dataset_hash(replay)
    jump_contracts = group_and_jump_contracts(dataset)
    donor, frame = corruption_plan(regime, seed_value, sample_count)
    corrupted_tokens = base.corruption.corrupt_frames(
        dataset.tokens, donor, frame
    )
    replay_donor, replay_frame = corruption_plan(regime, seed_value, sample_count)
    corruption_deterministic = bool(
        torch.equal(donor, replay_donor)
        and torch.equal(frame, replay_frame)
        and torch.equal(
            corrupted_tokens,
            base.corruption.corrupt_frames(
                dataset.tokens, replay_donor, replay_frame
            ),
        )
    )
    invariant, invariant_magnitude = source.analytic_carrier(
        corrupted_tokens, dataset.calibration
    )
    charged_character, charged_magnitude = charged.invalid.charged_character(
        corrupted_tokens, dataset.calibration
    )
    predictions, selections, displacements = decoder_predictions(
        invariant,
        corrupted_tokens,
        dataset.switch,
        frame,
        dataset.jump_time,
        dataset.jump_element,
    )

    action_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    equivariance_errors: dict[str, float] = {}
    integer_contracts: dict[str, dict[str, int | bool]] = {}
    corruption_equivariance_errors: dict[str, int] = {}
    transformed_displacements: dict[str, dict[str, float]] = {}
    for element in (1, 2):
        transformed_tokens = source.apply_deck_action(dataset.tokens, element)
        transformed_corrupted = base.corruption.corrupt_frames(
            transformed_tokens, donor, frame
        )
        corruption_equivariance_errors[str(element)] = int(
            (
                transformed_corrupted
                != source.apply_deck_action(corrupted_tokens, element)
            ).sum()
        )
        integer_contracts[str(element)] = connected_integer_contract(
            corrupted_tokens, transformed_corrupted
        )
        transformed_invariant, _ = source.analytic_carrier(
            transformed_corrupted, dataset.calibration
        )
        transformed_charged, _ = charged.invalid.charged_character(
            transformed_corrupted, dataset.calibration
        )
        root = torch.polar(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(
                -2.0 * math.pi * element / source.CHANNELS,
                dtype=torch.float64,
            ),
        ).to(torch.complex128)
        equivariance_errors[str(element)] = float(
            (transformed_charged - root * charged_character).abs().max()
        )
        transformed_predictions, _, element_displacements = decoder_predictions(
            transformed_invariant,
            transformed_corrupted,
            dataset.switch,
            frame,
            dataset.jump_time,
            dataset.jump_element,
        )
        transformed_displacements[str(element)] = element_displacements
        for arm in ARMS:
            action_errors[arm][str(element)] = float(
                (transformed_predictions[arm] - predictions[arm]).abs().max()
            )

    continuous_errors, continuous_displacements = continuous_validation(
        dataset, donor, frame
    )
    shuffle = base.corrective.hidden.speed.sattolo_derangement(
        sample_count, SHUFFLE_SEED_BASES[regime] + seed_value
    )
    shuffled_target = dataset.target[shuffle]
    operators = {
        arm: base.corruption._arm_metrics(
            prediction,
            dataset.target,
            shuffled_target,
            regime,
            action_errors[arm],
        )
        for arm, prediction in predictions.items()
    }
    connection_fidelity = _fidelity(
        operators["fixed_charged_connection"],
        operators["oracle_charged_connection"],
    )
    gauge_material_repair = bool(
        not operators["fixed_charged_no_connection"]["fixed_ceiling_pass"]
        and operators["fixed_charged_connection"]["fixed_ceiling_pass"]
    )
    switch_counts = torch.bincount(dataset.switch, minlength=6)
    frame_counts = torch.bincount(frame, minlength=source.TIME_STEPS)
    jump_time_counts = torch.bincount(dataset.jump_time, minlength=7)
    jump_element_counts = torch.bincount(dataset.jump_element, minlength=3)
    joint_jump_counts = torch.zeros((7, 3), dtype=torch.int64)
    for jump_time, jump_element in zip(
        dataset.jump_time, dataset.jump_element
    ):
        joint_jump_counts[int(jump_time), int(jump_element)] += 1
    donor_fixed_points = int(
        (donor == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    shuffle_fixed_points = int(
        (shuffle == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    exact_theta = base.phase_path(
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        steps=source.TIME_STEPS + 1,
    )
    deck_phase = 2.0 * math.pi * dataset.deck.double() / source.CHANNELS
    global_charged = torch.polar(
        torch.ones_like(exact_theta[:, :-1]),
        exact_theta[:, :-1] - deck_phase[:, None],
    ).to(torch.complex128)
    invariant_margin = base._chart_margin(global_charged.pow(3), frame)
    charged_margin = base._chart_margin(global_charged, frame)
    all_displacements = [
        *displacements.values(),
        *(
            value
            for item in transformed_displacements.values()
            for value in item.values()
        ),
        *continuous_displacements.values(),
    ]
    maximum_displacement = max(all_displacements)
    required_switch = MINIMUM_SWITCH_COUNT if registered else 0
    required_frame = MINIMUM_FRAME_COUNT if registered else 0
    required_jump_time = MINIMUM_JUMP_TIME_COUNT if registered else 0
    required_jump_element = MINIMUM_JUMP_ELEMENT_COUNT if registered else 0
    required_joint_jump = MINIMUM_JOINT_JUMP_COUNT if registered else 0
    charged_arms = (
        "fixed_charged_no_connection",
        "oracle_charged_connection",
        "fixed_charged_connection",
    )
    validity = bool(
        audit.get("pass") is True
        and dataset.saturation_count == 0
        and dataset_deterministic
        and jump_contracts["pass"]
        and corruption_deterministic
        and donor_fixed_points == 0
        and min(int(switch_counts[item]) for item in SWITCH_POINTS)
        >= required_switch
        and int(frame_counts.min()) >= required_frame
        and min(int(jump_time_counts[item]) for item in JUMP_TIMES)
        >= required_jump_time
        and min(int(jump_element_counts[item]) for item in JUMP_ELEMENTS)
        >= required_jump_element
        and min(
            int(joint_jump_counts[item_time, item_element])
            for item_time in JUMP_TIMES
            for item_element in JUMP_ELEMENTS
        )
        >= required_joint_jump
        and sum(corruption_equivariance_errors.values()) == 0
        and all(contract["pass"] for contract in integer_contracts.values())
        and max(equivariance_errors.values())
        <= CHARGED_EQUIVARIANCE_ERROR_MAXIMUM
        and maximum_displacement <= STABILIZATION_DISPLACEMENT_MAXIMUM
        and all(operator["action_pass"] for operator in operators.values())
        and all(
            operators[arm]["maximum_deck_action_error"] == 0.0
            for arm in charged_arms
        )
        and (
            all(operator["shuffle_pass"] for operator in operators.values())
            if registered
            else True
        )
        and max(continuous_errors.values()) <= CONTINUOUS_ERROR_MAXIMUM
        and invariant_margin >= CHART_MARGIN_MINIMUM
        and charged_margin >= CHART_MARGIN_MINIMUM
        and shuffle_fixed_points == 0
        and float(invariant_magnitude.min()) > 1e-12
        and float(charged_magnitude.min()) > 1e-12
        and base._finite(operators)
        and base._finite(connection_fidelity)
    )
    candidate_identities = connection_identities()
    selected_identity = [
        candidate_identities[int(item)]
        for item in selections["fixed_connection"]
    ]
    exact_tuple_rate = sum(
        identity
        == (
            int(item_frame),
            int(item_switch),
            int(item_jump_time),
            int(item_jump_element),
        )
        for identity, item_frame, item_switch, item_jump_time, item_jump_element in zip(
            selected_identity,
            frame,
            dataset.switch,
            dataset.jump_time,
            dataset.jump_element,
        )
    ) / sample_count
    return {
        "seed": seed_value,
        "regime": regime,
        "dataset_seed": dataset.dataset_seed,
        "base_dataset_sha256": dataset_digest,
        "base_dataset_deterministic": dataset_deterministic,
        "corruption_sha256": base._tensor_hash(corrupted_tokens, donor, frame),
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
        "jump_time_counts": {
            str(item): int(jump_time_counts[item]) for item in JUMP_TIMES
        },
        "minimum_jump_time_count": min(
            int(jump_time_counts[item]) for item in JUMP_TIMES
        ),
        "jump_element_counts": {
            str(item): int(jump_element_counts[item])
            for item in JUMP_ELEMENTS
        },
        "minimum_jump_element_count": min(
            int(jump_element_counts[item]) for item in JUMP_ELEMENTS
        ),
        "minimum_joint_jump_count": min(
            int(joint_jump_counts[item_time, item_element])
            for item_time in JUMP_TIMES
            for item_element in JUMP_ELEMENTS
        ),
        "donor_fixed_points": donor_fixed_points,
        "shuffle_fixed_points": shuffle_fixed_points,
        "group_and_jump_contracts": jump_contracts,
        "corruption_equivariance_token_errors": corruption_equivariance_errors,
        "connected_integer_action_contracts": integer_contracts,
        "charged_equivariance_errors": equivariance_errors,
        "stabilization_displacements": {
            "natural": displacements,
            "transformed": transformed_displacements,
            "continuous": continuous_displacements,
            "maximum": maximum_displacement,
        },
        "minimum_invariant_chart_margin": invariant_margin,
        "minimum_connected_charged_chart_margin": charged_margin,
        "minimum_invariant_character_magnitude": float(
            invariant_magnitude.min()
        ),
        "minimum_charged_character_magnitude": float(charged_magnitude.min()),
        "continuous_prediction_maximum_errors": continuous_errors,
        "operators": operators,
        "connection_oracle_fidelity": connection_fidelity,
        "gauge_material_repair": gauge_material_repair,
        "invariant_selected_oracle_rate": float(
            (
                selections["fixed_invariant"]
                == selections["oracle_invariant"]
            ).double().mean()
        ),
        "connection_selected_oracle_tuple_rate": float(exact_tuple_rate),
        "valid": validity,
    }


def classify(
    cells: Sequence[Mapping[str, Any]],
    valid: bool,
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

    invariant_oracle_count = arm_count("oracle_invariant_switch_drop")
    invariant_fixed_count = arm_count("fixed_invariant_switch_drop")
    no_connection_count = arm_count("fixed_charged_no_connection")
    connection_oracle_count = arm_count("oracle_charged_connection")
    connection_fixed_count = arm_count("fixed_charged_connection")
    fidelity_count = sum(
        all(
            bool(cell(seed, regime)["connection_oracle_fidelity"]["pass"])
            for regime in REGIMES
        )
        for seed in SEEDS
    )
    materiality_count = sum(
        all(bool(cell(seed, regime)["gauge_material_repair"]) for regime in REGIMES)
        for seed in SEEDS
    )
    charged_pass = bool(
        invariant_oracle_count >= REQUIRED_SEED_PASSES
        and connection_oracle_count >= REQUIRED_SEED_PASSES
        and connection_fixed_count >= REQUIRED_SEED_PASSES
        and fidelity_count >= REQUIRED_SEED_PASSES
        and materiality_count >= REQUIRED_SEED_PASSES
    )
    if not valid:
        classification = "invalid_time_varying_gauge_fixed_decoder"
    elif charged_pass and invariant_fixed_count < REQUIRED_SEED_PASSES:
        classification = (
            "discrete_c3_connection_closes_time_varying_gauge_and_invariant_gap"
        )
    elif charged_pass and invariant_fixed_count >= REQUIRED_SEED_PASSES:
        classification = (
            "discrete_c3_connection_closes_gauge_scope_without_"
            "unique_invariant_advantage"
        )
    elif (
        connection_oracle_count >= REQUIRED_SEED_PASSES
        and connection_fixed_count < REQUIRED_SEED_PASSES
    ):
        classification = (
            "recoverable_time_varying_gauge_exceeds_fixed_connection_decoder"
        )
    elif (
        invariant_oracle_count < REQUIRED_SEED_PASSES
        or connection_oracle_count < REQUIRED_SEED_PASSES
    ):
        classification = "time_varying_gauge_scope_not_oracle_recoverable"
    else:
        classification = "inconclusive_time_varying_gauge_fixed_decoder"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "invariant_oracle_seed_pass_count": invariant_oracle_count,
        "invariant_fixed_seed_pass_count": invariant_fixed_count,
        "no_connection_seed_pass_count": no_connection_count,
        "connection_oracle_seed_pass_count": connection_oracle_count,
        "connection_fixed_seed_pass_count": connection_fixed_count,
        "connection_oracle_fidelity_seed_pass_count": fidelity_count,
        "gauge_material_repair_seed_pass_count": materiality_count,
        "compact_typed_connection_comparison_licensed": classification
        == "recoverable_time_varying_gauge_exceeds_fixed_connection_decoder",
        "tinyllm_training_licensed": False,
    }


def build_result() -> dict[str, Any]:
    source_hashes = validate_sources()
    audit = base.identifiability_audit()
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
            "switch_points": list(SWITCH_POINTS),
            "jump_times": list(JUMP_TIMES),
            "jump_elements": list(JUMP_ELEMENTS),
            "arms": list(ARMS),
            "sample_count_per_cell": SAMPLE_COUNT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "donor_seed_bases": DONOR_SEED_BASES,
            "frame_seed_bases": FRAME_SEED_BASES,
            "jump_time_seed_bases": JUMP_TIME_SEED_BASES,
            "jump_element_seed_bases": JUMP_ELEMENT_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "minimum_switch_count": MINIMUM_SWITCH_COUNT,
            "minimum_frame_count": MINIMUM_FRAME_COUNT,
            "minimum_jump_time_count": MINIMUM_JUMP_TIME_COUNT,
            "minimum_jump_element_count": MINIMUM_JUMP_ELEMENT_COUNT,
            "minimum_joint_jump_count": MINIMUM_JOINT_JUMP_COUNT,
            "action_error_maximum": ACTION_ERROR_MAXIMUM,
            "charged_equivariance_error_maximum": (
                CHARGED_EQUIVARIANCE_ERROR_MAXIMUM
            ),
            "continuous_error_maximum": CONTINUOUS_ERROR_MAXIMUM,
            "chart_margin_minimum": CHART_MARGIN_MINIMUM,
            "stabilization_displacement_maximum": (
                STABILIZATION_DISPLACEMENT_MAXIMUM
            ),
            "fixed_ceiling_rmse_maximum": (
                base.corruption.FIXED_CEILING_RMSE_MAXIMUM
            ),
            "fixed_ceiling_accuracy_minimum": (
                base.corruption.FIXED_CEILING_ACCURACY_MINIMUM
            ),
            "oracle_rmse_slack_maximum": (
                charged.invalid.ORACLE_RMSE_SLACK_MAXIMUM
            ),
            "oracle_accuracy_slack_maximum": (
                charged.invalid.ORACLE_ACCURACY_SLACK_MAXIMUM
            ),
            "oracle_cross_entropy_slack_maximum": (
                charged.invalid.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
            ),
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "task_gates": base.acceleration.TASK_GATES,
        },
        "identifiability_audit": audit,
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "fresh_base_examples": len(cells) * SAMPLE_COUNT,
            "fresh_corrupted_evaluations": len(cells) * SAMPLE_COUNT,
            "primary_observation_only_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 336
            ),
            "continuous_validation_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 312
            ),
            "outcome_known_audit_examples_pooled": 0,
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
            "device": "cpu",
        },
        "method_boundaries": [
            "The study changes only the observation gauge from global to one hidden suffix jump.",
            "The fixed connection exhausts the declared one-jump C3 law and uses no latent labels for selection.",
            "The physical switch support remains the exactly identifiable {2,3,4} scope.",
            "Outcome-known pilot and stress-test examples contribute zero primary evidence.",
            "No result licenses unrestricted TinyLLM training.",
        ],
    }
    if not base._finite(result):
        raise RuntimeError("non-finite gauge-jump result")
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
                "invariant_fixed_seed_passes": result["aggregates"][
                    "invariant_fixed_seed_pass_count"
                ],
                "connection_fixed_seed_passes": result["aggregates"][
                    "connection_fixed_seed_pass_count"
                ],
                "connection_fidelity_seed_passes": result["aggregates"][
                    "connection_oracle_fidelity_seed_pass_count"
                ],
                "compact_typed_comparison_licensed": result["aggregates"][
                    "compact_typed_connection_comparison_licensed"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
