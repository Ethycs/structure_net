#!/usr/bin/env python3
"""Replicate d6 C2/C3 causal quotient fronts with one-observation deck actions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_deck_action_descrambler as deck
import experiments.structure_net.tinyllm_degree_k_ladder as ladder
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
)


SCHEMA_VERSION = "nal.tinyllm-observed-cyclic-deck-twirl-front.v1"
HYPOTHESIS_ID = "tinyllm-observed-cyclic-deck-twirl-front-v1"
EVIDENCE_ROLE = "preregistered_frozen_observed_action_replication"
SOURCE_LADDER_SCHEMA = "nal.tinyllm-degree-k-ladder.v1"
SOURCE_LADDER_CAMPAIGN_SHA256 = (
    "cf12b76691da41b7bc15e47570bce324f6aaefc7c9f670ef68db1fa4d9421046"
)
SOURCE_LADDER_IMPLEMENTATION_SHA256 = (
    "5d8f861221cbaf94cdbb662ab50988bb796c07712983736f31db8965ea9e2455"
)
SOURCE_DECK_SCHEMA = "nal.tinyllm-deck-action-descrambler.v1"
SOURCE_DECK_CAMPAIGN_SHA256 = (
    "a3c14ce7022b7301344beaca876e0d454445c972a57de69c9cd4cd89098036b3"
)
SOURCE_DECK_IMPLEMENTATION_SHA256 = (
    "9a15ceaf735555cdd0aa843247e7f142e7585632ecd98f2a2c73b60a341c3a48"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-observed-cyclic-deck-twirl-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "4ea38b10c77eede7a67d07c9d253a80ad058cd879ac49995cf762ad680f1dd59"
)
DEGREES = (2, 3)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = deck.REGIMES
CUTS = deck.cut_names(3)


@dataclass(frozen=True)
class ObservedCyclicDeckConfig:
    source_ladder_root: str = (
        "data/experiments/tinyllm_degree_k_ladder/20260806_d6_preregistered"
    )
    source_deck_root: str = (
        "data/experiments/tinyllm_deck_action_descrambler/"
        "20260806_d6_preregistered"
    )
    degrees: tuple[int, ...] = DEGREES
    seeds: tuple[int, ...] = SEEDS
    evaluation_orbits: int = 256
    map_points: int = 192
    first_blocks: int = 3
    activation_batch_size: int = 256
    required_seed_passes: int = 4
    maximum_control_seed_passes: int = 1
    front_cut_tolerance: int = 1
    replay_tolerance: float = 2e-6
    action_closure_tolerance: float = 2e-6
    calibration_tolerance: float = 1e-7
    corrected_norm_tolerance: float = 1e-6
    carrier_equivariance_tolerance: float = 1e-6
    task_character_tolerance: float = 1e-6
    transformed_planar_limit: float = 2.0
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.degrees or set(self.degrees).difference(DEGREES):
            raise ValueError("degrees must be drawn from 2,3")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.evaluation_orbits < 4 or self.map_points < 24:
            raise ValueError("cohorts are too small")
        if any(self.map_points % k for k in self.degrees):
            raise ValueError("map_points must be divisible by every degree")
        if self.first_blocks != 3:
            raise ValueError("the locked source front uses three blocks")
        if self.activation_batch_size < 1:
            raise ValueError("activation batch size must be positive")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside the population")
        if not 0 <= self.maximum_control_seed_passes <= len(self.seeds):
            raise ValueError("control ceiling is outside the population")
        if self.front_cut_tolerance < 0:
            raise ValueError("front cut tolerance must be nonnegative")
        if not self.allow_underpowered:
            if self.degrees != DEGREES or self.seeds != SEEDS:
                raise ValueError("primary degrees and five checkpoints are fixed")
            if self.evaluation_orbits != 256 or self.map_points != 192:
                raise ValueError("primary cohort sizes are fixed")
            if self.required_seed_passes != 4:
                raise ValueError("primary population gate is four of five")
            if self.maximum_control_seed_passes != 1:
                raise ValueError("primary control ceiling is one of five")
            if self.front_cut_tolerance != 1:
                raise ValueError("primary front tolerance is one cut")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _json_config(config: ObservedCyclicDeckConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _source_digests() -> dict[str, str]:
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError("observed cyclic-deck preregistration changed")
    paths = {
        "runner": Path(__file__),
        "deck_source": Path(deck.__file__),
        "ladder_source": Path(ladder.__file__),
        "calibrated_source": Path(calibrated.__file__),
        "preregistration": PREREGISTRATION_PATH,
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(digests or _source_digests()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_source_campaigns(
    config: ObservedCyclicDeckConfig,
) -> tuple[CircleTaskConfig, dict[str, Any], dict[str, Any]]:
    ladder_path = Path(config.source_ladder_root) / "campaign_results.json"
    ladder_campaign = json.loads(ladder_path.read_text(encoding="utf-8"))
    if (
        _sha256(ladder_path) != SOURCE_LADDER_CAMPAIGN_SHA256
        or ladder_campaign.get("schema_version") != SOURCE_LADDER_SCHEMA
        or ladder_campaign.get("hypothesis_id") != ladder.HYPOTHESIS_ID
        or ladder_campaign.get("status") != "completed"
        or ladder_campaign.get("implementation_sha256")
        != SOURCE_LADDER_IMPLEMENTATION_SHA256
        or ladder_campaign.get("summary", {}).get("completed") != 15
        or ladder_campaign.get("summary", {}).get("failed") != 0
    ):
        raise ValueError(f"invalid degree-ladder source {ladder_path}")
    deck_path = Path(config.source_deck_root) / "campaign_results.json"
    deck_campaign = json.loads(deck_path.read_text(encoding="utf-8"))
    if (
        _sha256(deck_path) != SOURCE_DECK_CAMPAIGN_SHA256
        or deck_campaign.get("schema_version") != SOURCE_DECK_SCHEMA
        or deck_campaign.get("hypothesis_id") != deck.HYPOTHESIS_ID
        or deck_campaign.get("status") != "completed"
        or deck_campaign.get("implementation_sha256")
        != SOURCE_DECK_IMPLEMENTATION_SHA256
        or deck_campaign.get("summary", {}).get("completed") != 15
        or deck_campaign.get("summary", {}).get("failed") != 0
    ):
        raise ValueError(f"invalid deck-action source {deck_path}")
    task = CircleTaskConfig()
    if ladder_campaign.get("task_config") != asdict(task):
        raise ValueError("degree-ladder task configuration changed")
    return task, ladder_campaign, deck_campaign


def _reference_result(
    config: ObservedCyclicDeckConfig, k: int, seed: int
) -> tuple[dict[str, Any], Path, str]:
    path = (
        Path(config.source_deck_root)
        / "runs"
        / f"k{k}"
        / f"seed_{seed}"
        / "result.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    if (
        value.get("schema_version") != SOURCE_DECK_SCHEMA
        or value.get("hypothesis_id") != deck.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("k", -1)) != k
        or int(value.get("seed", -1)) != seed
        or value.get("baseline_replay_integrity", {}).get("composition", {}).get("passed")
        is not True
        or value.get("baseline_replay_integrity", {}).get("extrapolation", {}).get("passed")
        is not True
    ):
        raise ValueError(f"invalid deck reference result {path}")
    return value, path, digest


def _deck_config(config: ObservedCyclicDeckConfig) -> deck.DeckDescramblerConfig:
    return deck.DeckDescramblerConfig(
        source_root=config.source_ladder_root,
        seeds=config.seeds,
        degrees=config.degrees,
        fit_orbits=4,
        evaluation_orbits=config.evaluation_orbits,
        map_points=config.map_points,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        device=config.device,
        allow_underpowered=True,
    )


def _rotate(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angles).to(dtype=values.dtype, device=values.device)
    sine = torch.sin(angles).to(dtype=values.dtype, device=values.device)
    return torch.stack(
        (
            cosine * values[..., 0] - sine * values[..., 1],
            sine * values[..., 0] + cosine * values[..., 1],
        ),
        dim=-1,
    )


def observed_cyclic_action(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    task: CircleTaskConfig,
    angles: torch.Tensor | float,
) -> torch.Tensor:
    """Rotate detrended planar observations using no latent or target fields."""
    if not torch.is_tensor(angles):
        angles = torch.full(
            (len(sensor),), float(angles), dtype=sensor.dtype, device=sensor.device
        )
    else:
        angles = angles.to(dtype=sensor.dtype, device=sensor.device)
    if angles.ndim != 1 or len(angles) != len(sensor):
        raise ValueError("one declared action angle is required per observation")
    history = (
        torch.arange(task.sensor_steps, dtype=sensor.dtype, device=sensor.device)
        / max(1.0, float(task.sensor_steps - 1))
        - 1.0
    )
    offset = calibration[:, 4:6]
    drift = calibration[:, 6:8]
    baseline = offset[:, None, :] + drift[:, None, :] * history[None, :, None]
    corrected = sensor[..., :2] - baseline
    transformed = sensor.clone()
    transformed[..., :2] = _rotate(corrected, angles[:, None]) + baseline
    return transformed


def _corrected_planar(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    task: CircleTaskConfig,
) -> torch.Tensor:
    history = (
        torch.arange(task.sensor_steps, dtype=sensor.dtype, device=sensor.device)
        / max(1.0, float(task.sensor_steps - 1))
        - 1.0
    )
    return (
        sensor[..., :2]
        - calibration[:, None, 4:6]
        - calibration[:, None, 6:8] * history[None, :, None]
    ) / calibration[:, None, 3:4]


def _anchor_rows(values: torch.Tensor, orbit_count: int, k: int) -> torch.Tensor:
    return values.reshape((orbit_count, k) + values.shape[1:])[:, 0]


def construct_observed_orbit(
    exact: deck.OrbitDataset,
    task: CircleTaskConfig,
    *,
    angle_offset: float,
) -> deck.OrbitDataset:
    """Construct every orbit row from branch zero and declared group elements."""
    k = exact.k
    orbit_count = exact.orbit_count
    anchor_sensor = _anchor_rows(exact.sensor, orbit_count, k)
    anchor_calibration = _anchor_rows(exact.calibration, orbit_count, k)
    anchor_inputs = _anchor_rows(exact.input_ids, orbit_count, k)
    anchor_phase = _anchor_rows(exact.phase, orbit_count, k)
    group_angles = angle_offset + 2.0 * math.pi * torch.arange(k) / k
    angles = group_angles.repeat(orbit_count).to(dtype=anchor_sensor.dtype)
    sensor = observed_cyclic_action(
        anchor_sensor.repeat_interleave(k, dim=0),
        anchor_calibration.repeat_interleave(k, dim=0),
        task,
        angles,
    )
    phase = torch.remainder(
        anchor_phase.repeat_interleave(k) + angles,
        2.0 * math.pi,
    )
    posterior, bins = ladder._targets(phase.double().numpy(), k, task.phase_bins)
    return deck.OrbitDataset(
        input_ids=anchor_inputs.repeat_interleave(k, dim=0),
        sensor=sensor,
        calibration=anchor_calibration.repeat_interleave(k, dim=0),
        phase=phase.float(),
        quotient_phase=torch.remainder(k * phase, 2.0 * math.pi).float(),
        branch=torch.arange(k, dtype=torch.long).repeat(orbit_count),
        target_posteriors=posterior,
        target_bins=bins,
        orbit_count=orbit_count,
        k=k,
    )


def _dataset_hash(dataset: deck.OrbitDataset) -> str:
    digest = hashlib.sha256()
    for name in (
        "input_ids",
        "sensor",
        "calibration",
        "phase",
        "quotient_phase",
        "branch",
        "target_posteriors",
        "target_bins",
    ):
        tensor = getattr(dataset, name).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def action_contract(
    exact: deck.OrbitDataset,
    correct: deck.OrbitDataset,
    control: deck.OrbitDataset,
    task: CircleTaskConfig,
    config: ObservedCyclicDeckConfig,
) -> dict[str, Any]:
    k = exact.k
    anchor_sensor = _anchor_rows(exact.sensor, exact.orbit_count, k)
    anchor_calibration = _anchor_rows(exact.calibration, exact.orbit_count, k)
    repeated_sensor = anchor_sensor.repeat_interleave(k, dim=0)
    repeated_calibration = anchor_calibration.repeat_interleave(k, dim=0)
    correct_angles = (2.0 * math.pi * torch.arange(k) / k).repeat(exact.orbit_count)
    control_angles = correct_angles + math.pi / k
    restored = anchor_sensor.clone()
    for _ in range(k):
        restored = observed_cyclic_action(
            restored,
            anchor_calibration,
            task,
            2.0 * math.pi / k,
        )
    carrier = ladder.AnalyticPhaseCarrier(task)
    anchor_carrier = carrier(anchor_sensor, anchor_calibration)
    correct_carrier = carrier(correct.sensor, correct.calibration)
    control_carrier = carrier(control.sensor, control.calibration)
    expected_correct = _rotate(
        anchor_carrier.repeat_interleave(k, dim=0), correct_angles
    )
    expected_control = _rotate(
        anchor_carrier.repeat_interleave(k, dim=0), control_angles
    )
    anchor_complex = torch.view_as_complex(
        anchor_carrier.repeat_interleave(k, dim=0).double().contiguous()
    )
    correct_complex = torch.view_as_complex(correct_carrier.double().contiguous())
    control_complex = torch.view_as_complex(control_carrier.double().contiguous())
    corrected_anchor = _corrected_planar(repeated_sensor, repeated_calibration, task)
    corrected_correct = _corrected_planar(correct.sensor, correct.calibration, task)
    corrected_control = _corrected_planar(control.sensor, control.calibration, task)
    exact_rms = torch.sqrt(
        torch.mean((correct.sensor.double() - exact.sensor.double()).square())
    )
    exact_scale = torch.sqrt(torch.mean(exact.sensor.double().square())).clamp_min(1e-12)
    values = {
        "group_closure_maximum_error": float((restored - anchor_sensor).abs().max()),
        "calibration_maximum_error": float(
            torch.maximum(
                (correct.calibration - repeated_calibration).abs().max(),
                (control.calibration - repeated_calibration).abs().max(),
            )
        ),
        "corrected_norm_maximum_error": float(
            torch.maximum(
                (corrected_correct.norm(dim=-1) - corrected_anchor.norm(dim=-1)).abs().max(),
                (corrected_control.norm(dim=-1) - corrected_anchor.norm(dim=-1)).abs().max(),
            )
        ),
        "carrier_equivariance_maximum_error": float(
            torch.maximum(
                (correct_carrier - expected_correct).abs().max(),
                (control_carrier - expected_control).abs().max(),
            )
        ),
        "correct_task_character_maximum_error": float(
            (correct_complex.pow(k) - anchor_complex.pow(k)).abs().max()
        ),
        "control_antipodal_character_maximum_error": float(
            (control_complex.pow(k) + anchor_complex.pow(k)).abs().max()
        ),
        "transformed_planar_maximum_absolute_value": float(
            torch.maximum(
                correct.sensor[..., :2].abs().max(),
                control.sensor[..., :2].abs().max(),
            )
        ),
        "observed_vs_separately_quantized_orbit_relative_rms": float(
            exact_rms / exact_scale
        ),
        "source_boundary_token_maximum_error": int(
            torch.maximum(
                (correct.input_ids[:, 0] - exact.input_ids[:, 0]).abs().max(),
                (correct.input_ids[:, -1] - exact.input_ids[:, -1]).abs().max(),
            )
        ),
    }
    values["pass"] = bool(
        values["group_closure_maximum_error"] <= config.action_closure_tolerance
        and values["calibration_maximum_error"] <= config.calibration_tolerance
        and values["corrected_norm_maximum_error"] <= config.corrected_norm_tolerance
        and values["carrier_equivariance_maximum_error"]
        <= config.carrier_equivariance_tolerance
        and values["correct_task_character_maximum_error"]
        <= config.task_character_tolerance
        and values["control_antipodal_character_maximum_error"]
        <= config.task_character_tolerance
        and values["transformed_planar_maximum_absolute_value"]
        <= config.transformed_planar_limit
        and values["source_boundary_token_maximum_error"] == 0
    )
    return values


def _generate_cohorts(
    task: CircleTaskConfig,
    config: ObservedCyclicDeckConfig,
    k: int,
    seed: int,
) -> tuple[
    dict[str, dict[str, deck.OrbitDataset]],
    dict[str, str],
    dict[str, Any],
]:
    cohorts: dict[str, dict[str, deck.OrbitDataset]] = {}
    hashes: dict[str, str] = {}
    contracts: dict[str, Any] = {}
    for regime, seed_offset in (("composition", 211), ("extrapolation", 307)):
        exact_task = deck.generate_exact_orbits(
            task,
            k=k,
            orbit_count=config.evaluation_orbits,
            seed=seed + seed_offset,
            regime=regime,
        )
        theta = np.linspace(
            0.0, 2.0 * math.pi, config.map_points // k, endpoint=False
        )
        exact_map = deck.generate_exact_orbits(
            task,
            k=k,
            orbit_count=config.map_points // k,
            seed=seed + 10_003,
            regime=regime,
            quotient_phases=theta,
            fixed_nuisance=True,
        )
        cohort = {
            "correct_task": construct_observed_orbit(
                exact_task, task, angle_offset=0.0
            ),
            "control_task": construct_observed_orbit(
                exact_task, task, angle_offset=math.pi / k
            ),
            "correct_map": construct_observed_orbit(
                exact_map, task, angle_offset=0.0
            ),
            "control_map": construct_observed_orbit(
                exact_map, task, angle_offset=math.pi / k
            ),
        }
        task_contract = action_contract(
            exact_task,
            cohort["correct_task"],
            cohort["control_task"],
            task,
            config,
        )
        map_contract = action_contract(
            exact_map,
            cohort["correct_map"],
            cohort["control_map"],
            task,
            config,
        )
        contracts[regime] = {
            "task": task_contract,
            "map": map_contract,
            "pass": bool(task_contract["pass"] and map_contract["pass"]),
        }
        cohorts[regime] = cohort
        for name, dataset in cohort.items():
            hashes[f"{regime}__{name}"] = _dataset_hash(dataset)
    contracts["construction_inputs"] = [
        "decoded_sensor",
        "calibration_offset",
        "calibration_drift",
        "fixed_time_grid",
        "declared_group_element",
    ]
    contracts["forbidden_inputs"] = [
        "latent_phase",
        "target_posterior",
        "target_bin",
        "branch",
        "quotient_phase",
        "fiber_id",
        "independently_generated_orbit_row",
    ]
    contracts["pass"] = all(contracts[regime]["pass"] for regime in REGIMES)
    return cohorts, hashes, contracts


def _posterior_js(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(left.astype(np.float64), 1e-12, 1.0)
    right = np.clip(right.astype(np.float64), 1e-12, 1.0)
    middle = 0.5 * (left + right)
    value = 0.5 * np.sum(left * np.log(left / middle), axis=-1)
    value += 0.5 * np.sum(right * np.log(right / middle), axis=-1)
    return float(np.mean(value))


def _posterior_action_dispersion(
    posterior: np.ndarray, orbit_count: int, k: int
) -> dict[str, float]:
    shaped = posterior.reshape(orbit_count, k, -1)
    anchor = np.repeat(shaped[:, :1], k, axis=1).reshape(posterior.shape)
    mean = np.repeat(shaped.mean(axis=1, keepdims=True), k, axis=1).reshape(
        posterior.shape
    )
    return {
        "mean_js_from_anchor": _posterior_js(posterior, anchor),
        "mean_js_from_orbit_mean": _posterior_js(posterior, mean),
        "maximum_probability_difference_from_anchor": float(
            np.max(np.abs(posterior - anchor))
        ),
    }


def _state_geometry(
    values: np.ndarray, orbit_count: int, k: int
) -> dict[str, float]:
    shaped = values.astype(np.float64).reshape(
        (orbit_count, k) + values.shape[1:]
    )
    anchor = np.repeat(shaped[:, :1], k, axis=1)
    mean = np.repeat(shaped.mean(axis=1, keepdims=True), k, axis=1)
    scale = max(1e-12, float(np.sqrt(np.mean(np.square(anchor)))))
    return {
        "action_difference_relative_rms": float(
            np.sqrt(np.mean(np.square(shaped - anchor))) / scale
        ),
        "twirl_displacement_relative_rms": float(
            np.sqrt(np.mean(np.square(mean - shaped))) / scale
        ),
        "action_maximum_absolute_difference": float(np.max(np.abs(shaped - anchor))),
    }


def _metrics(
    map_posterior: np.ndarray,
    task_posterior: np.ndarray,
    map_dataset: deck.OrbitDataset,
    task_dataset: deck.OrbitDataset,
) -> dict[str, Any]:
    result = deck.output_diagnostics(
        map_posterior, map_dataset, include_topology=False
    )
    result["exact_bin_accuracy"] = float(
        np.mean(task_posterior.argmax(1) == task_dataset.target_bins.numpy())
    )
    angles = 2.0 * math.pi * np.arange(task_posterior.shape[1]) / task_posterior.shape[1]
    moment = task_posterior @ np.exp(1j * angles)
    target = np.remainder(task_dataset.k * task_dataset.phase.numpy(), 2.0 * math.pi)
    result["mean_circular_error_radians"] = float(
        np.mean(np.abs(np.angle(np.exp(1j * (np.angle(moment) - target)))))
    )
    return result


def _first_preserved(records: Mapping[str, Any], key: str) -> tuple[int | None, str | None]:
    for index, cut in enumerate(CUTS):
        if records[cut][key]["causal_classification"] == "preserved":
            return index, cut
    return None, None


@torch.no_grad()
def analyze_regime(
    system: ladder.DegreeKSystem,
    cohort: Mapping[str, deck.OrbitDataset],
    reference: Mapping[str, Any],
    config: ObservedCyclicDeckConfig,
    load_config: deck.DeckDescramblerConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    captures = {
        name: deck.capture_sequences(system, dataset, load_config, device)
        for name, dataset in cohort.items()
    }
    full_posteriors = {
        name: deck.continue_from_cut(
            system, cohort[name], "full", captures[name]["full"], load_config, device
        )
        for name in cohort
    }
    cut_records: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    maximum_replay_error = 0.0
    k = cohort["correct_task"].k
    for cut in CUTS:
        replays = {
            name: deck.continue_from_cut(
                system, cohort[name], cut, captures[name][cut], load_config, device
            )
            for name in cohort
        }
        replay_error = max(
            float(np.max(np.abs(replays[name] - full_posteriors[name])))
            for name in cohort
        )
        maximum_replay_error = max(maximum_replay_error, replay_error)
        correct_task_twirl = deck.orbit_average(
            captures["correct_task"][cut], cohort["correct_task"].orbit_count, k
        )
        correct_map_twirl = deck.orbit_average(
            captures["correct_map"][cut], cohort["correct_map"].orbit_count, k
        )
        control_task_twirl = deck.orbit_average(
            captures["control_task"][cut], cohort["control_task"].orbit_count, k
        )
        control_map_twirl = deck.orbit_average(
            captures["control_map"][cut], cohort["control_map"].orbit_count, k
        )
        correct_twirl_task_posterior = deck.continue_from_cut(
            system,
            cohort["correct_task"],
            cut,
            correct_task_twirl,
            load_config,
            device,
        )
        correct_twirl_map_posterior = deck.continue_from_cut(
            system,
            cohort["correct_map"],
            cut,
            correct_map_twirl,
            load_config,
            device,
        )
        control_twirl_task_posterior = deck.continue_from_cut(
            system,
            cohort["control_task"],
            cut,
            control_task_twirl,
            load_config,
            device,
        )
        control_twirl_map_posterior = deck.continue_from_cut(
            system,
            cohort["control_map"],
            cut,
            control_map_twirl,
            load_config,
            device,
        )
        baseline_correct = _metrics(
            replays["correct_map"],
            replays["correct_task"],
            cohort["correct_map"],
            cohort["correct_task"],
        )
        baseline_control = _metrics(
            replays["control_map"],
            replays["control_task"],
            cohort["control_map"],
            cohort["control_task"],
        )
        correct_twirl = _metrics(
            correct_twirl_map_posterior,
            correct_twirl_task_posterior,
            cohort["correct_map"],
            cohort["correct_task"],
        )
        control_twirl_own = _metrics(
            control_twirl_map_posterior,
            control_twirl_task_posterior,
            cohort["control_map"],
            cohort["control_task"],
        )
        control_twirl_source = _metrics(
            control_twirl_map_posterior,
            control_twirl_task_posterior,
            cohort["correct_map"],
            cohort["correct_task"],
        )
        correct_twirl["causal_classification"] = deck._classify_causal(
            correct_twirl, baseline_correct, k
        )
        control_twirl_own["causal_classification"] = deck._classify_causal(
            control_twirl_own, baseline_control, k
        )
        control_twirl_source["causal_classification"] = deck._classify_causal(
            control_twirl_source, baseline_correct, k
        )
        reference_classification = reference["cuts"][cut]["causal"][
            "orbit_average"
        ]["causal_classification"]
        cut_records[cut] = {
            "baseline_correct": baseline_correct,
            "baseline_control": baseline_control,
            "correct_twirl": correct_twirl,
            "control_twirl_own_target": control_twirl_own,
            "control_twirl_source_target": control_twirl_source,
            "locked_reference_classification": reference_classification,
            "replay_maximum_absolute_posterior_error": replay_error,
            "correct_action_posterior_dispersion": _posterior_action_dispersion(
                replays["correct_task"], cohort["correct_task"].orbit_count, k
            ),
            "control_action_posterior_dispersion": _posterior_action_dispersion(
                replays["control_task"], cohort["control_task"].orbit_count, k
            ),
            "correct_action_state_geometry": _state_geometry(
                captures["correct_task"][cut], cohort["correct_task"].orbit_count, k
            ),
            "control_action_state_geometry": _state_geometry(
                captures["control_task"][cut], cohort["control_task"].orbit_count, k
            ),
        }
        arrays[f"{cut}__correct_twirl_task_posterior"] = (
            correct_twirl_task_posterior.astype(np.float32)
        )
        arrays[f"{cut}__control_twirl_task_posterior"] = (
            control_twirl_task_posterior.astype(np.float32)
        )
    observed_index, observed_cut = _first_preserved(cut_records, "correct_twirl")
    reference_index = next(
        (
            index
            for index, cut in enumerate(CUTS)
            if cut_records[cut]["locked_reference_classification"] == "preserved"
        ),
        None,
    )
    reference_cut = CUTS[reference_index] if reference_index is not None else None
    front_distance = (
        abs(observed_index - reference_index)
        if observed_index is not None and reference_index is not None
        else None
    )
    return {
        "cuts": cut_records,
        "maximum_replay_error": maximum_replay_error,
        "observed_first_preserved_index": observed_index,
        "observed_first_preserved_cut": observed_cut,
        "reference_first_preserved_index": reference_index,
        "reference_first_preserved_cut": reference_cut,
        "front_cut_distance": front_distance,
        "front_match": bool(
            front_distance is not None
            and front_distance <= config.front_cut_tolerance
        ),
    }, arrays


def _fingerprint(
    config: ObservedCyclicDeckConfig,
    implementation: str,
    k: int,
    seed: int,
    provenance: Mapping[str, Any],
    reference_sha256: str,
    dataset_hashes: Mapping[str, str],
    action_contract_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "k": k,
        "seed": seed,
        "provenance": dict(provenance),
        "reference_sha256": reference_sha256,
        "dataset_hashes": dict(dataset_hashes),
        "action_contract_sha256": action_contract_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    path: Path, fingerprint: str, implementation: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = Path(value.get("artifacts", {}).get("diagnostics", ""))
    if (
        value.get("status") != "completed"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("scientific_fingerprint") != fingerprint
        or value.get("implementation_sha256") != implementation
        or value.get("artifacts", {}).get("result") != str(path)
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != value.get("artifacts", {}).get("diagnostics_sha256")
    ):
        raise ValueError(f"incompatible completed cyclic-deck result {path}")
    return value


def _campaign_reusable(
    campaign: Mapping[str, Any],
    config: ObservedCyclicDeckConfig,
    implementation: str,
) -> bool:
    expected = {(k, seed) for k in config.degrees for seed in config.seeds}
    entries = campaign.get("results", [])
    observed = {(entry.get("k"), entry.get("seed")) for entry in entries}
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and observed == expected
        and len(entries) == len(expected)
        and all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry["result_sha256"]
            and Path(entry["diagnostics_path"]).is_file()
            and _sha256(Path(entry["diagnostics_path"]))
            == entry["diagnostics_sha256"]
            for entry in entries
        )
    )


def _classify_campaign(
    *,
    valid: bool,
    degrees: Mapping[str, Any],
    config: ObservedCyclicDeckConfig,
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    degree_pass = {
        int(k): bool(value["all_primary_gates_pass"])
        for k, value in degrees.items()
    }
    if all(degree_pass.values()):
        return "observed_cyclic_deck_front_reproduced", True
    if degree_pass.get(2) and not degree_pass.get(3):
        return "c2_only_observed_front", False
    mature = all(
        value["mature_observed_twirl_pass_count"] >= config.required_seed_passes
        for value in degrees.values()
    )
    if mature:
        return "mature_twirl_closed_front_not_reproduced", False
    return "observed_cyclic_action_not_closed", False


def run_campaign(
    config: ObservedCyclicDeckConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_reusable(existing, config, implementation):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    task, ladder_campaign, deck_campaign = _validate_source_campaigns(config)
    load_config = _deck_config(config)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    results: list[dict[str, Any]] = []
    reused = 0
    global_contracts: dict[str, Any] = {}
    for k in config.degrees:
        for seed in config.seeds:
            cell_started = time.perf_counter()
            reference, reference_path, reference_sha256 = _reference_result(
                config, k, seed
            )
            cohorts, dataset_hashes, contract = _generate_cohorts(
                task, config, k, seed
            )
            contract_sha256 = hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            global_contracts[f"k{k}_seed{seed}"] = {
                "sha256": contract_sha256,
                "pass": contract["pass"],
            }
            if not contract["pass"]:
                raise ValueError(f"observed cyclic action contract failed k{k} seed {seed}")
            system, provenance = deck.load_source(task, load_config, k, seed, device)
            for parameter in system.parameters():
                parameter.requires_grad_(False)
            initial_model_state = _state_digest(system.model)
            initial_system_state = calibrated._module_digest(system)
            fingerprint = _fingerprint(
                config,
                implementation,
                k,
                seed,
                provenance,
                reference_sha256,
                dataset_hashes,
                contract_sha256,
            )
            result_dir = output / "runs" / f"k{k}" / f"seed_{seed}"
            result_path = result_dir / "result.json"
            existing = _reusable_result(result_path, fingerprint, implementation)
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"resuming k{k} seed {seed}", flush=True)
                del system
                continue
            regime_results: dict[str, Any] = {}
            diagnostic_arrays: dict[str, np.ndarray] = {}
            for regime in REGIMES:
                result, arrays = analyze_regime(
                    system,
                    cohorts[regime],
                    {"cuts": {
                        cut: {"causal": {"orbit_average": {
                            "causal_classification": reference["cuts"][cut]["causal"][regime]["orbit_average"]["causal_classification"]
                        }}}
                        for cut in CUTS
                    }},
                    config,
                    load_config,
                    device,
                )
                regime_results[regime] = result
                diagnostic_arrays.update(
                    {f"{regime}__{name}": value for name, value in arrays.items()}
                )
            diagnostics_path = result_dir / "observed_cyclic_deck_diagnostics.npz"
            _write_npz(diagnostics_path, diagnostic_arrays)
            diagnostics_sha256 = _sha256(diagnostics_path)
            state_unchanged = bool(
                _state_digest(system.model) == initial_model_state
                and calibrated._module_digest(system) == initial_system_state
            )
            replay_pass = all(
                regime_results[regime]["maximum_replay_error"]
                <= config.replay_tolerance
                for regime in REGIMES
            )
            finite = _finite(regime_results)
            validity = bool(contract["pass"] and state_unchanged and replay_pass and finite)
            mature = all(
                regime_results[regime]["cuts"]["full"]["correct_twirl"][
                    "causal_classification"
                ]
                == "preserved"
                for regime in REGIMES
            )
            cover = all(
                regime_results[regime]["cuts"][cut]["correct_twirl"][
                    "causal_classification"
                ]
                == "destroyed"
                for regime in REGIMES
                for cut in ("frontend", "block_0_pre_attention")
            )
            front_match = all(
                regime_results[regime]["front_match"] for regime in REGIMES
            )
            control_valid = all(
                regime_results[regime]["cuts"]["full"][
                    "control_twirl_own_target"
                ]["causal_classification"]
                == "preserved"
                for regime in REGIMES
            )
            semantic_control_preserved = all(
                regime_results[regime]["cuts"]["full"][
                    "control_twirl_source_target"
                ]["causal_classification"]
                == "preserved"
                for regime in REGIMES
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-observed-cyclic-deck-k{k}-seed{seed}",
                "status": "completed",
                "evidence_role": EVIDENCE_ROLE,
                "completed_at": _utc_now(),
                "k": k,
                "seed": seed,
                "configuration": _json_config(config),
                "implementation_sha256": implementation,
                "scientific_fingerprint": fingerprint,
                "provenance": {
                    **provenance,
                    "reference_result": str(reference_path),
                    "reference_result_sha256": reference_sha256,
                },
                "dataset_hashes": dataset_hashes,
                "action_contract": contract,
                "action_contract_sha256": contract_sha256,
                "regimes": regime_results,
                "seed_gates": {
                    "mature_observed_twirl": mature,
                    "computational_cover": cover,
                    "front_replication": front_match,
                    "control_validity": control_valid,
                    "semantic_control_preserved": semantic_control_preserved,
                },
                "gates": {
                    "action_contract": contract["pass"],
                    "source_and_cut_replay": replay_pass,
                    "state_unchanged": state_unchanged,
                    "finite": finite,
                    "validity": validity,
                },
                "analysis_seconds": time.perf_counter() - cell_started,
                "artifacts": {
                    "result": str(result_path),
                    "diagnostics": str(diagnostics_path),
                    "diagnostics_sha256": diagnostics_sha256,
                },
            }
            _write_json(result_path, result)
            results.append(result)
            print(
                f"k{k} seed {seed}: mature={mature} cover={cover} "
                f"front={front_match} control_valid={control_valid} "
                f"control_source={semantic_control_preserved} valid={validity}",
                flush=True,
            )
            del system
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("observed cyclic-deck implementation changed during run")

    degree_aggregates: dict[str, Any] = {}
    for k in config.degrees:
        selected = [item for item in results if item["k"] == k]
        mature_count = sum(
            int(item["seed_gates"]["mature_observed_twirl"]) for item in selected
        )
        cover_count = sum(
            int(item["seed_gates"]["computational_cover"]) for item in selected
        )
        front_count = sum(
            int(item["seed_gates"]["front_replication"]) for item in selected
        )
        control_valid_count = sum(
            int(item["seed_gates"]["control_validity"]) for item in selected
        )
        semantic_control_count = sum(
            int(item["seed_gates"]["semantic_control_preserved"])
            for item in selected
        )
        degree_aggregates[str(k)] = {
            "mature_observed_twirl_pass_count": mature_count,
            "computational_cover_pass_count": cover_count,
            "front_replication_pass_count": front_count,
            "control_validity_pass_count": control_valid_count,
            "semantic_control_preserved_count": semantic_control_count,
            "all_primary_gates_pass": bool(
                mature_count >= config.required_seed_passes
                and cover_count >= config.required_seed_passes
                and front_count >= config.required_seed_passes
                and control_valid_count >= config.required_seed_passes
                and semantic_control_count <= config.maximum_control_seed_passes
            ),
            "per_seed_fronts": {
                str(item["seed"]): {
                    regime: {
                        "observed": item["regimes"][regime]["observed_first_preserved_cut"],
                        "reference": item["regimes"][regime]["reference_first_preserved_cut"],
                        "distance": item["regimes"][regime]["front_cut_distance"],
                    }
                    for regime in REGIMES
                }
                for item in selected
            },
        }
    valid = all(item["gates"]["validity"] for item in results)
    if config.allow_underpowered:
        classification, primary_pass = (
            "systems_lifecycle_only_not_quality_evidence",
            False,
        )
    else:
        classification, primary_pass = _classify_campaign(
            valid=valid, degrees=degree_aggregates, config=config
        )
    result_entries = [
        {
            "experiment_id": item["experiment_id"],
            "k": item["k"],
            "seed": item["seed"],
            "scientific_fingerprint": item["scientific_fingerprint"],
            "path": item["artifacts"]["result"],
            "result_sha256": _sha256(Path(item["artifacts"]["result"])),
            "diagnostics_path": item["artifacts"]["diagnostics"],
            "diagnostics_sha256": item["artifacts"]["diagnostics_sha256"],
        }
        for item in results
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(result_entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "provenance": {
            "source_ladder_campaign": str(
                Path(config.source_ladder_root) / "campaign_results.json"
            ),
            "source_ladder_campaign_sha256": SOURCE_LADDER_CAMPAIGN_SHA256,
            "source_ladder_implementation_sha256": SOURCE_LADDER_IMPLEMENTATION_SHA256,
            "source_deck_campaign": str(
                Path(config.source_deck_root) / "campaign_results.json"
            ),
            "source_deck_campaign_sha256": SOURCE_DECK_CAMPAIGN_SHA256,
            "source_deck_implementation_sha256": SOURCE_DECK_IMPLEMENTATION_SHA256,
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
        "task_config": asdict(task),
        "action_contracts": global_contracts,
        "summary": {
            "requested": len(config.degrees) * len(config.seeds),
            "scheduled": len(config.degrees) * len(config.seeds) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "fitted_probes": 0,
            "fitted_observers": 0,
            "fitted_action_parameters": 0,
        },
        "aggregates": {
            "classification": classification,
            "primary_hypothesis_pass": primary_pass,
            "valid": valid,
            "required_seed_passes": config.required_seed_passes,
            "maximum_control_seed_passes": config.maximum_control_seed_passes,
            "degrees": degree_aggregates,
        },
        "results": result_entries,
        "result_manifest_sha256": manifest_sha256,
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The action reads only one decoded sensor, observed offset/drift, the fixed time grid, and a declared cyclic group element.",
            "Latent phase, targets, branch labels, quotient phase, fiber IDs, and independently generated orbit rows are forbidden action inputs.",
            "The transform is continuous after token decoding and is not re-quantized.",
            "The analytic carrier and ten d6 TinyLLMs remain frozen; no parameter is fit.",
            "C2/C3 synthetic calibrated evidence does not establish other groups, sensors, or architecture-population prevalence.",
        ],
        "artifacts": {"campaign": str(campaign_path)},
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_observed_cyclic_deck_twirl/"
            "20260810_d6_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--degrees", type=_ints, default=DEGREES)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--evaluation-orbits", type=int, default=256)
    parser.add_argument("--map-points", type=int, default=192)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--maximum-control-seed-passes", type=int, default=1)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = ObservedCyclicDeckConfig(
        degrees=args.degrees,
        seeds=args.seeds,
        evaluation_orbits=args.evaluation_orbits,
        map_points=args.map_points,
        activation_batch_size=args.activation_batch_size,
        required_seed_passes=args.required_seed_passes,
        maximum_control_seed_passes=args.maximum_control_seed_passes,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
