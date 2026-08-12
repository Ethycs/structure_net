#!/usr/bin/env python3
"""Compare invariant and charged fixed decoders on C3 switching data."""

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

from experiments.structure_net import (
    tinyllm_c3_switching_corruption_fixed_decoder as base,
)
from experiments.structure_net import (
    tinyllm_c3_switching_corruption_fixed_decoder_corrective as parent,
)


SCHEMA_VERSION = "nal.tinyllm-c3-charged-character-fixed-decoder.v1"
HYPOTHESIS_ID = "tinyllm-c3-charged-character-fixed-decoder-v1"
EVIDENCE_ROLE = "prospective_no_training_mechanistic_fixed_decoder_replication"
SEEDS = (487, 503, 521, 541, 563)
REGIMES = base.REGIMES
SWITCH_POINTS = base.SWITCH_POINTS
ARMS = (
    "oracle_invariant_switch_drop",
    "fixed_invariant_switch_drop",
    "oracle_charged_switch_drop",
    "fixed_charged_switch_drop",
)
SAMPLE_COUNT = base.SAMPLE_COUNT
DATASET_SEED_BASES = {"composition": 1_001_107, "extrapolation": 1_003_107}
PILOT_DATASET_SEED_BASES = {
    "composition": 1_017_107,
    "extrapolation": 1_019_107,
}
DONOR_SEED_BASES = {"composition": 1_005_107, "extrapolation": 1_007_107}
FRAME_SEED_BASES = {"composition": 1_009_107, "extrapolation": 1_011_107}
SHUFFLE_SEED_BASES = {"composition": 1_013_107, "extrapolation": 1_015_107}
MINIMUM_SWITCH_COUNT = base.MINIMUM_SWITCH_COUNT
MINIMUM_FRAME_COUNT = base.MINIMUM_FRAME_COUNT
ACTION_ERROR_MAXIMUM = base.ACTION_ERROR_MAXIMUM
CHARGED_EQUIVARIANCE_ERROR_MAXIMUM = 2e-12
CONTINUOUS_ERROR_MAXIMUM = base.CONTINUOUS_ERROR_MAXIMUM
CHART_MARGIN_MINIMUM = base.CHART_MARGIN_MINIMUM
STABILIZATION_DISPLACEMENT_MAXIMUM = (
    parent.STABILIZATION_DISPLACEMENT_MAXIMUM
)
ORACLE_RMSE_SLACK_MAXIMUM = base.ORACLE_RMSE_SLACK_MAXIMUM
ORACLE_ACCURACY_SLACK_MAXIMUM = base.ORACLE_ACCURACY_SLACK_MAXIMUM
ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM = base.ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
REQUIRED_SEED_PASSES = base.REQUIRED_SEED_PASSES
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-charged-character-fixed-decoder-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "9f1a6b280498f254e51324c8d87d385fa5ab72090cc2487db57d5f84b8925633"
)
PARENT_RUNNER_SHA256 = (
    "452b0fa8f8ff54ddb0afa98e32eef9dc38da9f240eccf6a867386d9b65197939"
)
PARENT_RESULT_PATH = parent.PRIMARY_RESULT_PATH
PARENT_RESULT_SHA256 = (
    "20d40a54ce42a904766cab9eb533f6e2baccff79f75a2703e6bd57ff9638675b"
)
PARENT_REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder.md"
)
PARENT_REPORT_SHA256 = (
    "bb082ec3965d5e8badb3e08d645a0fb958990b8c65667f2da8440cbe6f02295a"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_charged_character_fixed_decoder/"
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
        "switching_corrective_runner": _sha256(Path(parent.__file__)),
        "switching_corrective_result": _sha256(PARENT_RESULT_PATH),
        "switching_report": _sha256(PARENT_REPORT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "switching_corrective_runner": PARENT_RUNNER_SHA256,
        "switching_corrective_result": PARENT_RESULT_SHA256,
        "switching_report": PARENT_REPORT_SHA256,
    }
    parent.validate_sources()
    predecessor = json.loads(PARENT_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        hashes != expected
        or predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != "recoverable_switching_exceeds_fixed_change_point_decoder"
        or predecessor.get("aggregates", {}).get(
            "compact_typed_continuation_comparison_licensed"
        )
        is not True
    ):
        raise RuntimeError("charged-character source contract changed")
    return hashes


def generate_dataset(
    regime: str,
    seed_value: int,
    *,
    sample_count: int = SAMPLE_COUNT,
    allow_pilot: bool = False,
) -> base.SwitchingDataset:
    primary = seed_value in SEEDS and sample_count == SAMPLE_COUNT
    pilot = allow_pilot and seed_value == 997 and sample_count == 64
    if regime not in REGIMES or not (primary or pilot):
        raise ValueError(f"unknown charged-character cohort {regime}/{seed_value}")
    bases = DATASET_SEED_BASES if primary else PILOT_DATASET_SEED_BASES
    dataset_seed = bases[regime] + seed_value
    generator = torch.Generator(device="cpu").manual_seed(dataset_seed)
    ranges = base.source.REGIME_RANGES[regime]
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
        0, base.source.CHANNELS, (sample_count,), generator=generator
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
    canonical = base.source.quantize(continuous)
    tokens = base.source._roll_by_example(canonical, deck)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    future = base.phase_path(
        phase,
        velocity,
        delta_velocity,
        switch,
        steps=base.source.TIME_STEPS + 1,
    )[:, -1]
    return base.SwitchingDataset(
        canonical_tokens=canonical,
        tokens=tokens,
        calibration=calibration,
        target=torch.cos(3.0 * future),
        phase=phase,
        velocity=velocity,
        delta_velocity=delta_velocity,
        switch=switch,
        deck=deck,
        saturation_count=int(
            (continuous.abs() >= base.source.QUANTIZATION_LIMIT).sum()
        ),
        dataset_seed=dataset_seed,
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
        base.source.TIME_STEPS,
        (sample_count,),
        dtype=torch.int64,
        generator=generator,
    )
    return donor, frame


def charged_character(
    tokens: torch.Tensor,
    calibration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = base.source.decode(tokens)
    amplitude = calibration[:, 0]
    offset = calibration[:, 1]
    drift = calibration[:, 2]
    time = torch.arange(base.source.TIME_STEPS, dtype=torch.float64)[
        None, :, None
    ]
    corrected = (
        observed - offset[:, None, None] - drift[:, None, None] * time
    ) / amplitude[:, None, None]
    angles = (
        2.0
        * math.pi
        * torch.arange(base.source.CHANNELS, dtype=torch.float64)
        / base.source.CHANNELS
    )
    weights = torch.polar(torch.ones_like(angles), -angles).to(
        torch.complex128
    )
    coefficient = torch.einsum(
        "btc,c->bt", corrected.to(torch.complex128), weights
    )
    magnitude = coefficient.abs()
    return coefficient / magnitude.clamp_min(1e-12), magnitude


def _fit_charged_design(
    character: torch.Tensor,
    times: torch.Tensor,
    switch: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    anchor = character[:, :1]
    relative, relative_displacement = parent.stabilize_carrier(
        character * anchor.conj()
    )
    anchor_invariant, anchor_displacement = parent.stabilize_carrier(
        anchor[:, 0].pow(3)
    )
    phase = torch.angle(relative).double()
    increments = base.corruption._principal_increment(
        phase[:, 1:] - phase[:, :-1]
    )
    unwrapped = torch.cat(
        (phase[:, :1], phase[:, :1] + torch.cumsum(increments, dim=1)),
        dim=1,
    )
    design = base._design(times, switch)
    coefficients = unwrapped @ torch.linalg.pinv(design).T
    fitted = coefficients @ design.T
    residual = (unwrapped - fitted).square().mean(dim=1)
    evaluation = torch.tensor(
        [1.0, float(base.source.TIME_STEPS), float(base.source.TIME_STEPS - switch)],
        dtype=torch.float64,
    )
    predicted_relative_phase = coefficients @ evaluation
    prediction = anchor_invariant * torch.polar(
        torch.ones_like(predicted_relative_phase),
        3.0 * predicted_relative_phase,
    ).to(torch.complex128)
    return prediction, residual, max(relative_displacement, anchor_displacement)


def charged_deletion_candidates(
    character: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[tuple[int, int]],
    float,
]:
    predictions = []
    residuals = []
    identities = []
    maximum_displacement = 0.0
    all_times = torch.arange(base.source.TIME_STEPS, dtype=torch.float64)
    for omitted in range(base.source.TIME_STEPS):
        keep = torch.arange(base.source.TIME_STEPS) != omitted
        for switch in SWITCH_POINTS:
            prediction, residual, displacement = _fit_charged_design(
                character[:, keep], all_times[keep], switch
            )
            predictions.append(prediction)
            residuals.append(residual)
            identities.append((omitted, switch))
            maximum_displacement = max(maximum_displacement, displacement)
    return (
        torch.stack(predictions, dim=1),
        torch.stack(residuals, dim=1),
        identities,
        maximum_displacement,
    )


def decoder_predictions(
    invariant: torch.Tensor,
    charged: torch.Tensor,
    true_switch: torch.Tensor,
    true_frame: torch.Tensor,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, float],
]:
    stable_invariant, invariant_displacement = parent.stabilize_carrier(
        invariant
    )
    invariant_candidates, invariant_residuals, identities = (
        base.switch_deletion_candidates(stable_invariant)
    )
    (
        charged_candidates,
        charged_residuals,
        charged_identities,
        charged_displacement,
    ) = charged_deletion_candidates(charged)
    if charged_identities != identities:
        raise RuntimeError("charged and invariant candidate identities diverged")
    identity_index = {
        identity: offset for offset, identity in enumerate(identities)
    }
    oracle_index = torch.tensor(
        [
            identity_index[(int(frame), int(switch))]
            for frame, switch in zip(true_frame, true_switch)
        ],
        dtype=torch.int64,
    )
    invariant_selected = invariant_residuals.argmin(dim=1)
    charged_selected = charged_residuals.argmin(dim=1)
    row = torch.arange(invariant.shape[0], dtype=torch.int64)
    predictions = {
        "oracle_invariant_switch_drop": invariant_candidates[row, oracle_index],
        "fixed_invariant_switch_drop": invariant_candidates[
            row, invariant_selected
        ],
        "oracle_charged_switch_drop": charged_candidates[row, oracle_index],
        "fixed_charged_switch_drop": charged_candidates[row, charged_selected],
    }
    selections = {
        "oracle": oracle_index,
        "invariant": invariant_selected,
        "charged": charged_selected,
    }
    return predictions, selections, {
        "invariant": invariant_displacement,
        "charged": charged_displacement,
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
        value["rmse_excess"] <= ORACLE_RMSE_SLACK_MAXIMUM
        and value["accuracy_delta"] >= -ORACLE_ACCURACY_SLACK_MAXIMUM
        and value["cross_entropy_excess"]
        <= ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
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
    dataset_digest = base.dataset_hash(dataset)
    replay = generate_dataset(
        regime,
        seed_value,
        sample_count=sample_count,
        allow_pilot=allow_pilot,
    )
    dataset_deterministic = dataset_digest == base.dataset_hash(replay)
    donor, frame = corruption_plan(regime, seed_value, sample_count)
    corrupted_tokens = base.corruption.corrupt_frames(
        dataset.tokens, donor, frame
    )
    replay_donor, replay_frame = corruption_plan(
        regime, seed_value, sample_count
    )
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
    invariant, invariant_magnitude = base.source.analytic_carrier(
        corrupted_tokens, dataset.calibration
    )
    charged, charged_magnitude = charged_character(
        corrupted_tokens, dataset.calibration
    )
    predictions, selections, displacements = decoder_predictions(
        invariant, charged, dataset.switch, frame
    )

    action_errors: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    equivariance_errors: dict[str, dict[str, float]] = {}
    corruption_equivariance_errors: dict[str, int] = {}
    transformed_displacements: dict[str, dict[str, float]] = {}
    for element in (1, 2):
        transformed_tokens = base.source.apply_deck_action(
            dataset.tokens, element
        )
        transformed_corrupted_tokens = base.corruption.corrupt_frames(
            transformed_tokens, donor, frame
        )
        corruption_equivariance_errors[str(element)] = int(
            (
                transformed_corrupted_tokens
                != base.source.apply_deck_action(corrupted_tokens, element)
            ).sum()
        )
        transformed_invariant, _ = base.source.analytic_carrier(
            transformed_corrupted_tokens, dataset.calibration
        )
        transformed_charged, _ = charged_character(
            transformed_corrupted_tokens, dataset.calibration
        )
        root = torch.polar(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(
                -2.0 * math.pi * element / base.source.CHANNELS,
                dtype=torch.float64,
            ),
        ).to(torch.complex128)
        equivariance_errors[str(element)] = {
            "charged": float((transformed_charged - root * charged).abs().max()),
            "invariant": float(
                (transformed_invariant - invariant).abs().max()
            ),
        }
        transformed_predictions, _, element_displacements = decoder_predictions(
            transformed_invariant,
            transformed_charged,
            dataset.switch,
            frame,
        )
        transformed_displacements[str(element)] = element_displacements
        for arm in ARMS:
            action_errors[arm][str(element)] = float(
                (transformed_predictions[arm] - predictions[arm]).abs().max()
            )

    exact_theta = base.phase_path(
        dataset.phase,
        dataset.velocity,
        dataset.delta_velocity,
        dataset.switch,
        steps=base.source.TIME_STEPS + 1,
    )
    deck_phase = 2.0 * math.pi * dataset.deck.double() / base.source.CHANNELS
    exact_charged = torch.polar(
        torch.ones_like(exact_theta[:, :-1]),
        exact_theta[:, :-1] - deck_phase[:, None],
    ).to(torch.complex128)
    exact_invariant = exact_charged.pow(3)
    exact_corrupted_charged = base.corruption.corrupt_frames(
        exact_charged, donor, frame
    )
    exact_corrupted_invariant = exact_corrupted_charged.pow(3)
    exact_predictions, _, exact_displacements = decoder_predictions(
        exact_corrupted_invariant,
        exact_corrupted_charged,
        dataset.switch,
        frame,
    )
    exact_future = torch.polar(
        torch.ones_like(exact_theta[:, -1]), 3.0 * exact_theta[:, -1]
    ).to(torch.complex128)
    continuous_errors = {
        arm: float((prediction - exact_future).abs().max())
        for arm, prediction in exact_predictions.items()
    }

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
    charged_fidelity = _fidelity(
        operators["fixed_charged_switch_drop"],
        operators["oracle_charged_switch_drop"],
    )
    invariant_fidelity = _fidelity(
        operators["fixed_invariant_switch_drop"],
        operators["oracle_invariant_switch_drop"],
    )
    switch_counts = torch.bincount(dataset.switch, minlength=6)
    frame_counts = torch.bincount(frame, minlength=base.source.TIME_STEPS)
    group = base.group_contracts(dataset)
    invariant_margin = base._chart_margin(exact_invariant, frame)
    charged_margin = base._chart_margin(exact_charged, frame)
    donor_fixed_points = int(
        (donor == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    shuffle_fixed_points = int(
        (shuffle == torch.arange(sample_count, dtype=torch.int64)).sum()
    )
    all_displacements = [
        *displacements.values(),
        *(
            value
            for item in transformed_displacements.values()
            for value in item.values()
        ),
        *exact_displacements.values(),
    ]
    maximum_displacement = max(all_displacements)
    maximum_charged_equivariance_error = max(
        item["charged"] for item in equivariance_errors.values()
    )
    required_switch_count = MINIMUM_SWITCH_COUNT if registered else 0
    required_frame_count = MINIMUM_FRAME_COUNT if registered else 0
    validity = bool(
        audit.get("pass") is True
        and dataset.saturation_count == 0
        and dataset_deterministic
        and group["pass"]
        and corruption_deterministic
        and donor_fixed_points == 0
        and min(int(switch_counts[item]) for item in SWITCH_POINTS)
        >= required_switch_count
        and int(frame_counts.min()) >= required_frame_count
        and sum(corruption_equivariance_errors.values()) == 0
        and maximum_charged_equivariance_error
        <= CHARGED_EQUIVARIANCE_ERROR_MAXIMUM
        and maximum_displacement <= STABILIZATION_DISPLACEMENT_MAXIMUM
        and all(operator["action_pass"] for operator in operators.values())
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
        and base._finite(charged_fidelity)
        and base._finite(invariant_fidelity)
    )
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
        "donor_fixed_points": donor_fixed_points,
        "shuffle_fixed_points": shuffle_fixed_points,
        "minimum_invariant_chart_margin": invariant_margin,
        "minimum_charged_chart_margin": charged_margin,
        "minimum_invariant_character_magnitude": float(
            invariant_magnitude.min()
        ),
        "minimum_charged_character_magnitude": float(charged_magnitude.min()),
        "base_group_contracts": group,
        "corruption_equivariance_token_errors": corruption_equivariance_errors,
        "charged_equivariance_errors": equivariance_errors,
        "maximum_charged_equivariance_error": (
            maximum_charged_equivariance_error
        ),
        "stabilization_displacements": {
            "natural": displacements,
            "transformed": transformed_displacements,
            "exact": exact_displacements,
            "maximum": maximum_displacement,
        },
        "continuous_prediction_maximum_errors": continuous_errors,
        "operators": operators,
        "charged_oracle_fidelity": charged_fidelity,
        "invariant_oracle_fidelity": invariant_fidelity,
        "invariant_selected_oracle_rate": float(
            (selections["invariant"] == selections["oracle"]).double().mean()
        ),
        "charged_selected_oracle_rate": float(
            (selections["charged"] == selections["oracle"]).double().mean()
        ),
        "charged_invariant_selection_agreement_rate": float(
            (selections["charged"] == selections["invariant"]).double().mean()
        ),
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

    def fidelity_count(field: str) -> int:
        return sum(
            all(bool(cell(seed, regime)[field]["pass"]) for regime in REGIMES)
            for seed in SEEDS
        )

    invariant_oracle_count = arm_count("oracle_invariant_switch_drop")
    invariant_fixed_count = arm_count("fixed_invariant_switch_drop")
    charged_oracle_count = arm_count("oracle_charged_switch_drop")
    charged_fixed_count = arm_count("fixed_charged_switch_drop")
    charged_fidelity_count = fidelity_count("charged_oracle_fidelity")
    invariant_fidelity_count = fidelity_count("invariant_oracle_fidelity")
    if not valid:
        classification = "invalid_charged_character_fixed_decoder"
    elif (
        invariant_oracle_count >= REQUIRED_SEED_PASSES
        and charged_oracle_count >= REQUIRED_SEED_PASSES
        and charged_fixed_count >= REQUIRED_SEED_PASSES
        and charged_fidelity_count >= REQUIRED_SEED_PASSES
        and invariant_fixed_count < REQUIRED_SEED_PASSES
    ):
        classification = (
            "charged_character_fixed_decoder_closes_invariant_quantization_gap"
        )
    elif (
        invariant_oracle_count >= REQUIRED_SEED_PASSES
        and charged_oracle_count >= REQUIRED_SEED_PASSES
        and charged_fixed_count >= REQUIRED_SEED_PASSES
        and charged_fidelity_count >= REQUIRED_SEED_PASSES
        and invariant_fixed_count >= REQUIRED_SEED_PASSES
    ):
        classification = "both_fixed_decoders_close_fresh_switching_scope"
    elif (
        charged_oracle_count >= REQUIRED_SEED_PASSES
        and charged_fixed_count < REQUIRED_SEED_PASSES
    ):
        classification = "recoverable_switching_exceeds_charged_fixed_decoder"
    elif (
        invariant_oracle_count < REQUIRED_SEED_PASSES
        or charged_oracle_count < REQUIRED_SEED_PASSES
    ):
        classification = "fresh_switching_scope_not_oracle_recoverable"
    else:
        classification = "inconclusive_charged_character_fixed_decoder"
    return {
        "classification": classification,
        "valid": valid,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "invariant_oracle_seed_pass_count": invariant_oracle_count,
        "invariant_fixed_seed_pass_count": invariant_fixed_count,
        "charged_oracle_seed_pass_count": charged_oracle_count,
        "charged_fixed_seed_pass_count": charged_fixed_count,
        "charged_oracle_fidelity_seed_pass_count": charged_fidelity_count,
        "invariant_oracle_fidelity_seed_pass_count": invariant_fidelity_count,
        "compact_typed_continuation_comparison_licensed": classification
        == "recoverable_switching_exceeds_charged_fixed_decoder",
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
            "arms": list(ARMS),
            "sample_count_per_cell": SAMPLE_COUNT,
            "dataset_seed_bases": DATASET_SEED_BASES,
            "donor_seed_bases": DONOR_SEED_BASES,
            "frame_seed_bases": FRAME_SEED_BASES,
            "shuffle_seed_bases": SHUFFLE_SEED_BASES,
            "delta_velocity_ranges": base.DELTA_VELOCITY_RANGES,
            "minimum_switch_count": MINIMUM_SWITCH_COUNT,
            "minimum_frame_count": MINIMUM_FRAME_COUNT,
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
            "oracle_rmse_slack_maximum": ORACLE_RMSE_SLACK_MAXIMUM,
            "oracle_accuracy_slack_maximum": ORACLE_ACCURACY_SLACK_MAXIMUM,
            "oracle_cross_entropy_slack_maximum": (
                ORACLE_CROSS_ENTROPY_SLACK_MAXIMUM
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
            "observation_only_candidate_fits": len(cells) * SAMPLE_COUNT * 48,
            "continuous_validation_candidate_fits": (
                len(cells) * SAMPLE_COUNT * 48
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
            "The charged algorithm was selected after an outcome-known audit, but every primary cohort is fresh and no old example is pooled.",
            "Both decoders are fixed per-example estimators with no reusable or target-using fit.",
            "The study covers the identifiable three-post-change-frame support, not the provably ambiguous late-switch support.",
            "No result licenses unrestricted TinyLLM training.",
        ],
    }
    if not base._finite(result):
        raise RuntimeError("non-finite charged-character decoder result")
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
                "charged_fixed_seed_passes": result["aggregates"][
                    "charged_fixed_seed_pass_count"
                ],
                "charged_fidelity_seed_passes": result["aggregates"][
                    "charged_oracle_fidelity_seed_pass_count"
                ],
                "compact_typed_comparison_licensed": result["aggregates"][
                    "compact_typed_continuation_comparison_licensed"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
