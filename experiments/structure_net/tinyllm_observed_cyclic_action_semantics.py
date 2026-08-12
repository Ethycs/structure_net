#!/usr/bin/env python3
"""Decompose quantization and noise transport in observed cyclic actions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_io_correspondence as io_source
import experiments.structure_net.tinyllm_observed_cyclic_deck_twirl as base
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
)


SCHEMA_VERSION = "nal.tinyllm-observed-cyclic-action-semantics-front.v1"
HYPOTHESIS_ID = "tinyllm-observed-cyclic-action-semantics-front-v1"
EVIDENCE_ROLE = "preregistered_staged_frozen_action_semantics_intervention"
SOURCE_OBSERVED_SCHEMA = base.SCHEMA_VERSION
SOURCE_OBSERVED_CAMPAIGN_SHA256 = (
    "7a1b099495f7ecb6c3eeea7c9b836411a5baee709eb6b51ab8103f88927e8a86"
)
SOURCE_OBSERVED_IMPLEMENTATION_SHA256 = (
    "0c377e7a928e1926e916a981fec7464f0d4575e7f0e229bbbf1b7fc5298a0e56"
)
SOURCE_OBSERVED_RESULT_MANIFEST_SHA256 = (
    "52648e21fc8c3c47d62d588449fcc9cd1d36ac63302d836c5785eeb130f20184"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-observed-cyclic-action-semantics-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "8c649c67500fecd3bf672e7a879f83c8af316f838237ca051db7539e2d25977d"
)
DEGREES = base.DEGREES
SEEDS = base.SEEDS
REGIMES = base.REGIMES
CUTS = base.CUTS
OBSERVABLE_VARIANTS = (
    "rotate_all_continuous",
    "rotate_all_requantized",
    "residual_fixed_continuous",
    "residual_fixed_requantized",
)
ORACLE_VARIANTS = (
    "oracle_residual_fixed_continuous",
    "oracle_residual_fixed_requantized",
)
ALL_VARIANTS = OBSERVABLE_VARIANTS + ORACLE_VARIANTS
SELECTABLE_VARIANTS = (
    "rotate_all_requantized",
    "residual_fixed_continuous",
    "residual_fixed_requantized",
)


@dataclass(frozen=True)
class ActionSemanticsConfig:
    source_ladder_root: str = base.ObservedCyclicDeckConfig.source_ladder_root
    source_deck_root: str = base.ObservedCyclicDeckConfig.source_deck_root
    source_observed_root: str = (
        "data/experiments/tinyllm_observed_cyclic_deck_twirl/"
        "20260810_d6_preregistered"
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
    minimum_distance_reduction: float = 0.50
    character_median_ceiling: float = 0.05
    character_p95_ceiling: float = 0.20
    continuous_closure_ceiling: float = 2e-6
    requantized_closure_ceiling: float = 0.13
    continuous_composition_ceiling: float = 0.02
    requantized_composition_ceiling: float = 0.05
    norm_p95_ceiling: float = 0.05
    transformed_planar_limit: float = 2.0
    replay_tolerance: float = 2e-6
    forced_variant: str | None = None
    stage_a_only: bool = False
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
        if self.first_blocks != 3 or self.activation_batch_size < 1:
            raise ValueError("the three-block cut audit and positive batch are fixed")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed count is outside the population")
        if not 0 <= self.maximum_control_seed_passes <= len(self.seeds):
            raise ValueError("control ceiling is outside the population")
        if self.forced_variant is not None:
            if self.forced_variant not in SELECTABLE_VARIANTS:
                raise ValueError("forced variant is not selectable")
            if not self.allow_underpowered:
                raise ValueError("variant forcing is systems-only")
        if not self.allow_underpowered:
            if self.degrees != DEGREES or self.seeds != SEEDS:
                raise ValueError("primary degrees and checkpoint population are fixed")
            if self.evaluation_orbits != 256 or self.map_points != 192:
                raise ValueError("primary cohort sizes are fixed")
            if self.required_seed_passes != 4 or self.maximum_control_seed_passes != 1:
                raise ValueError("primary population gates are fixed")
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


def _json_config(config: ActionSemanticsConfig) -> dict[str, Any]:
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
        raise ValueError("action-semantics preregistration changed")
    paths = {
        "runner": Path(__file__),
        "observed_cyclic_source": Path(base.__file__),
        "deck_source": Path(base.deck.__file__),
        "ladder_source": Path(base.ladder.__file__),
        "io_source": Path(io_source.__file__),
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


def _base_config(config: ActionSemanticsConfig) -> base.ObservedCyclicDeckConfig:
    return base.ObservedCyclicDeckConfig(
        source_ladder_root=config.source_ladder_root,
        source_deck_root=config.source_deck_root,
        degrees=config.degrees,
        seeds=config.seeds,
        evaluation_orbits=config.evaluation_orbits,
        map_points=config.map_points,
        first_blocks=config.first_blocks,
        activation_batch_size=config.activation_batch_size,
        required_seed_passes=config.required_seed_passes,
        maximum_control_seed_passes=config.maximum_control_seed_passes,
        front_cut_tolerance=config.front_cut_tolerance,
        replay_tolerance=config.replay_tolerance,
        device=config.device,
        allow_underpowered=True,
    )


def _validate_sources(
    config: ActionSemanticsConfig,
) -> tuple[CircleTaskConfig, dict[str, Any], dict[str, Any], dict[str, Any]]:
    task, ladder_campaign, deck_campaign = base._validate_source_campaigns(
        _base_config(config)
    )
    observed_path = Path(config.source_observed_root) / "campaign_results.json"
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    if (
        _sha256(observed_path) != SOURCE_OBSERVED_CAMPAIGN_SHA256
        or observed.get("schema_version") != SOURCE_OBSERVED_SCHEMA
        or observed.get("hypothesis_id") != base.HYPOTHESIS_ID
        or observed.get("status") != "completed"
        or observed.get("implementation_sha256")
        != SOURCE_OBSERVED_IMPLEMENTATION_SHA256
        or observed.get("result_manifest_sha256")
        != SOURCE_OBSERVED_RESULT_MANIFEST_SHA256
        or observed.get("aggregates", {}).get("valid") is not True
        or observed.get("aggregates", {}).get("primary_hypothesis_pass") is not False
        or observed.get("aggregates", {}).get("classification")
        != "c2_only_observed_front"
        or observed.get("summary", {}).get("completed") != 10
        or observed.get("summary", {}).get("failed") != 0
    ):
        raise ValueError(f"invalid observed cyclic source {observed_path}")
    manifest = hashlib.sha256(
        json.dumps(
            observed.get("results", []), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if manifest != SOURCE_OBSERVED_RESULT_MANIFEST_SHA256:
        raise ValueError("observed cyclic source manifest changed")
    return task, ladder_campaign, deck_campaign, observed


def _history(task: CircleTaskConfig, *, normalized: bool, reference: torch.Tensor) -> torch.Tensor:
    value = torch.arange(
        task.sensor_steps, dtype=reference.dtype, device=reference.device
    ) - float(task.sensor_steps - 1)
    return value / max(1.0, float(task.sensor_steps - 1)) if normalized else value


def _calibrated_baseline(
    calibration: torch.Tensor, task: CircleTaskConfig, reference: torch.Tensor
) -> torch.Tensor:
    time = _history(task, normalized=True, reference=reference)
    return (
        calibration[:, None, 4:6]
        + calibration[:, None, 6:8] * time[None, :, None]
    )


def _observed_coherent_signal(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    task: CircleTaskConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline = _calibrated_baseline(calibration, task, sensor)
    physical = sensor[..., :2] - baseline
    orientation = calibration[:, :2]
    perpendicular = torch.stack((-orientation[:, 1], orientation[:, 0]), dim=-1)
    amplitude = calibration[:, 3:4].clamp_min(1e-6)
    normalized = physical / amplitude[:, None, :]
    lab_x = (normalized * orientation[:, None]).sum(-1)
    lab_y = (normalized * perpendicular[:, None]).sum(-1)
    step_time = _history(task, normalized=False, reference=sensor)
    omega = calibration[:, 2:3]
    angle = omega * step_time[None]
    demod_x = lab_x * torch.cos(angle) + lab_y * torch.sin(angle)
    demod_y = -lab_x * torch.sin(angle) + lab_y * torch.cos(angle)
    current = torch.stack((demod_x.mean(-1), demod_y.mean(-1)), dim=-1)
    current = current / current.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    current_angle = torch.atan2(current[:, 1], current[:, 0])
    trajectory = current_angle[:, None] + angle
    clean = amplitude[:, None, :] * (
        orientation[:, None] * torch.cos(trajectory)[..., None]
        + perpendicular[:, None] * torch.sin(trajectory)[..., None]
    )
    residual = physical - clean
    return clean, residual, current


def _oracle_coherent_signal(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    future_phase: torch.Tensor,
    task: CircleTaskConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = _calibrated_baseline(calibration, task, sensor)
    physical = sensor[..., :2] - baseline
    orientation = calibration[:, :2]
    perpendicular = torch.stack((-orientation[:, 1], orientation[:, 0]), dim=-1)
    amplitude = calibration[:, 3:4].clamp_min(1e-6)
    direction = torch.sign(calibration[:, 2]).clamp(min=-1.0, max=1.0)
    current = future_phase.to(sensor) - direction * task.future_delta
    step_time = _history(task, normalized=False, reference=sensor)
    trajectory = current[:, None] + calibration[:, 2:3] * step_time[None]
    clean = amplitude[:, None, :] * (
        orientation[:, None] * torch.cos(trajectory)[..., None]
        + perpendicular[:, None] * torch.sin(trajectory)[..., None]
    )
    residual = physical - clean
    return clean, residual


def residual_fixed_action(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    task: CircleTaskConfig,
    angles: torch.Tensor | float,
    *,
    oracle_future_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    if not torch.is_tensor(angles):
        angles = torch.full(
            (len(sensor),), float(angles), dtype=sensor.dtype, device=sensor.device
        )
    else:
        angles = angles.to(dtype=sensor.dtype, device=sensor.device)
    if angles.ndim != 1 or len(angles) != len(sensor):
        raise ValueError("one action angle is required per observation")
    baseline = _calibrated_baseline(calibration, task, sensor)
    if oracle_future_phase is None:
        clean, residual, _ = _observed_coherent_signal(sensor, calibration, task)
    else:
        clean, residual = _oracle_coherent_signal(
            sensor, calibration, oracle_future_phase, task
        )
    transformed = sensor.clone()
    transformed[..., :2] = (
        baseline
        + base._rotate(clean, angles[:, None])
        + residual
    )
    return transformed


def requantize_sensor(sensor: torch.Tensor, task: CircleTaskConfig) -> torch.Tensor:
    values = sensor.detach().cpu().double().numpy()
    tokens = torch.from_numpy(io_source._serialize_sensor_values(values, task))
    return io_source.decode_sensor_tokens(tokens, task)


def _anchor(values: torch.Tensor, exact: base.deck.OrbitDataset) -> torch.Tensor:
    return base._anchor_rows(values, exact.orbit_count, exact.k)


def construct_variant_orbit(
    exact: base.deck.OrbitDataset,
    task: CircleTaskConfig,
    *,
    variant: str,
    angle_offset: float,
) -> base.deck.OrbitDataset:
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown action semantics {variant}")
    metadata = base.construct_observed_orbit(
        exact, task, angle_offset=angle_offset
    )
    if variant == "rotate_all_continuous":
        return metadata
    k = exact.k
    anchor_sensor = _anchor(exact.sensor, exact).repeat_interleave(k, dim=0)
    anchor_calibration = _anchor(exact.calibration, exact).repeat_interleave(k, dim=0)
    angles = (
        angle_offset + 2.0 * math.pi * torch.arange(k) / k
    ).repeat(exact.orbit_count).to(dtype=anchor_sensor.dtype)
    if variant.startswith("rotate_all"):
        sensor = base.observed_cyclic_action(
            anchor_sensor, anchor_calibration, task, angles
        )
    else:
        oracle = variant.startswith("oracle_")
        future = (
            _anchor(exact.phase, exact).repeat_interleave(k) if oracle else None
        )
        sensor = residual_fixed_action(
            anchor_sensor,
            anchor_calibration,
            task,
            angles,
            oracle_future_phase=future,
        )
    if variant.endswith("requantized"):
        sensor = requantize_sensor(sensor, task)
    return replace(metadata, sensor=sensor)


def _per_row_relative_rms(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    axes = tuple(range(1, left.ndim))
    difference = torch.sqrt(torch.mean((left.double() - right.double()).square(), dim=axes))
    scale = torch.sqrt(torch.mean(right.double().square(), dim=axes)).clamp_min(1e-12)
    return (difference / scale).cpu().numpy()


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _apply_single_action(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    task: CircleTaskConfig,
    angle: float,
    variant: str,
) -> torch.Tensor:
    angles = torch.full((len(sensor),), angle, dtype=sensor.dtype)
    if variant.startswith("rotate_all"):
        value = base.observed_cyclic_action(sensor, calibration, task, angles)
    elif variant.startswith("oracle_"):
        raise ValueError("oracle composition is not an observable action contract")
    else:
        value = residual_fixed_action(sensor, calibration, task, angles)
    return requantize_sensor(value, task) if variant.endswith("requantized") else value


def variant_metrics(
    exact: base.deck.OrbitDataset,
    candidate: base.deck.OrbitDataset,
    task: CircleTaskConfig,
    variant: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    k = exact.k
    anchor_sensor = _anchor(exact.sensor, exact)
    anchor_calibration = _anchor(exact.calibration, exact)
    repeated_sensor = anchor_sensor.repeat_interleave(k, dim=0)
    repeated_calibration = anchor_calibration.repeat_interleave(k, dim=0)
    carrier = base.ladder.AnalyticPhaseCarrier(task)
    anchor_carrier = carrier(anchor_sensor, anchor_calibration).repeat_interleave(k, dim=0)
    candidate_carrier = carrier(candidate.sensor, candidate.calibration)
    anchor_complex = torch.view_as_complex(anchor_carrier.double().contiguous())
    candidate_complex = torch.view_as_complex(candidate_carrier.double().contiguous())
    character_angle = torch.abs(
        torch.angle(candidate_complex.pow(k) * torch.conj(anchor_complex.pow(k)))
    ).cpu().numpy()
    distance = _per_row_relative_rms(
        candidate.sensor[..., :2], exact.sensor[..., :2]
    )
    anchor_corrected = base._corrected_planar(
        repeated_sensor, repeated_calibration, task
    ).norm(dim=-1)
    candidate_corrected = base._corrected_planar(
        candidate.sensor, candidate.calibration, task
    ).norm(dim=-1)
    norm_relative = (
        torch.abs(candidate_corrected - anchor_corrected)
        / anchor_corrected.clamp_min(1e-8)
    ).reshape(len(candidate.sensor), -1).max(dim=1).values.cpu().numpy()
    if variant.startswith("oracle_"):
        composition = np.array([0.0], dtype=np.float64)
    else:
        generator_angle = 2.0 * math.pi / k
        once = _apply_single_action(
            anchor_sensor, anchor_calibration, task, generator_angle, variant
        )
        twice = _apply_single_action(
            once, anchor_calibration, task, generator_angle, variant
        )
        direct = _apply_single_action(
            anchor_sensor,
            anchor_calibration,
            task,
            2.0 * generator_angle,
            variant,
        )
        composition = _per_row_relative_rms(
            twice[..., :2], direct[..., :2]
        )
    if variant.startswith("rotate_all"):
        closed = base.observed_cyclic_action(
            anchor_sensor,
            anchor_calibration,
            task,
            torch.full((len(anchor_sensor),), 2.0 * math.pi),
        )
    elif variant.startswith("oracle_"):
        closed = residual_fixed_action(
            anchor_sensor,
            anchor_calibration,
            task,
            torch.full((len(anchor_sensor),), 2.0 * math.pi),
            oracle_future_phase=_anchor(exact.phase, exact),
        )
    else:
        closed = residual_fixed_action(
            anchor_sensor,
            anchor_calibration,
            task,
            torch.full((len(anchor_sensor),), 2.0 * math.pi),
        )
    if variant.endswith("requantized"):
        closed = requantize_sensor(closed, task)
    closure = _per_row_relative_rms(closed[..., :2], anchor_sensor[..., :2])
    arrays = {
        "generator_orbit_planar_relative_rms": distance.astype(np.float32),
        "task_character_angular_error": character_angle.astype(np.float32),
        "corrected_norm_relative_error": norm_relative.astype(np.float32),
        "composition_planar_relative_rms": composition.astype(np.float32),
        "closure_planar_relative_rms": closure.astype(np.float32),
    }
    metrics = {
        "generator_orbit_planar_relative_rms": _quantile_summary(distance),
        "task_character_angular_error": _quantile_summary(character_angle),
        "corrected_norm_relative_error": _quantile_summary(norm_relative),
        "composition_planar_relative_rms": _quantile_summary(composition),
        "closure_planar_relative_rms": _quantile_summary(closure),
        "transformed_planar_maximum_absolute_value": float(
            candidate.sensor[..., :2].abs().max()
        ),
        "finite": bool(
            torch.isfinite(candidate.sensor).all()
            and all(np.isfinite(value).all() for value in arrays.values())
        ),
    }
    return metrics, arrays


def _generate_exact_task(
    task: CircleTaskConfig,
    config: ActionSemanticsConfig,
    k: int,
    seed: int,
    regime: str,
) -> base.deck.OrbitDataset:
    offset = 211 if regime == "composition" else 307
    return base.deck.generate_exact_orbits(
        task,
        k=k,
        orbit_count=config.evaluation_orbits,
        seed=seed + offset,
        regime=regime,
    )


def run_stage_a(
    task: CircleTaskConfig,
    config: ActionSemanticsConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    pools: dict[tuple[str, int, str, str], list[np.ndarray]] = {}
    for k in config.degrees:
        for seed in config.seeds:
            for regime in REGIMES:
                exact = _generate_exact_task(task, config, k, seed, regime)
                cell_key = f"k{k}_seed{seed}_{regime}"
                cell: dict[str, Any] = {}
                for variant in ALL_VARIANTS:
                    candidate = construct_variant_orbit(
                        exact, task, variant=variant, angle_offset=0.0
                    )
                    metrics, raw = variant_metrics(
                        exact, candidate, task, variant
                    )
                    cell[variant] = metrics
                    for name, value in raw.items():
                        arrays[f"{cell_key}__{variant}__{name}"] = value
                        pools.setdefault((variant, k, regime, name), []).append(value)
                cells[cell_key] = cell

    variants: dict[str, Any] = {}
    baseline_name = "rotate_all_continuous"
    for variant in ALL_VARIANTS:
        by_degree_shift: dict[str, Any] = {}
        gates = []
        pooled_distance_all = []
        for k in config.degrees:
            for regime in REGIMES:
                prefix = f"k{k}_{regime}"
                values = {
                    name: np.concatenate(pools[(variant, k, regime, name)])
                    for name in (
                        "generator_orbit_planar_relative_rms",
                        "task_character_angular_error",
                        "corrected_norm_relative_error",
                        "composition_planar_relative_rms",
                        "closure_planar_relative_rms",
                    )
                }
                baseline_distance = np.concatenate(
                    pools[(baseline_name, k, regime, "generator_orbit_planar_relative_rms")]
                )
                distance = _quantile_summary(
                    values["generator_orbit_planar_relative_rms"]
                )
                character = _quantile_summary(values["task_character_angular_error"])
                norm = _quantile_summary(values["corrected_norm_relative_error"])
                composition = _quantile_summary(
                    values["composition_planar_relative_rms"]
                )
                closure = _quantile_summary(values["closure_planar_relative_rms"])
                reduction = float(
                    1.0
                    - distance["median"]
                    / max(1e-12, float(np.median(baseline_distance)))
                )
                closure_ceiling = (
                    config.requantized_closure_ceiling
                    if variant.endswith("requantized")
                    else config.continuous_closure_ceiling
                )
                composition_ceiling = (
                    config.requantized_composition_ceiling
                    if variant.endswith("requantized")
                    else config.continuous_composition_ceiling
                )
                support = max(
                    cells[f"k{k}_seed{seed}_{regime}"][variant][
                        "transformed_planar_maximum_absolute_value"
                    ]
                    for seed in config.seeds
                )
                eligible = bool(
                    variant in SELECTABLE_VARIANTS
                    and reduction >= config.minimum_distance_reduction
                    and character["median"] <= config.character_median_ceiling
                    and character["p95"] <= config.character_p95_ceiling
                    and closure["maximum"] <= closure_ceiling
                    and composition["maximum"] <= composition_ceiling
                    and norm["p95"] <= config.norm_p95_ceiling
                    and support <= config.transformed_planar_limit
                )
                by_degree_shift[prefix] = {
                    "distance": distance,
                    "distance_reduction_from_rotate_all": reduction,
                    "character_angular_error": character,
                    "norm_relative_error": norm,
                    "composition_relative_rms": composition,
                    "closure_relative_rms": closure,
                    "support_maximum": support,
                    "eligible": eligible,
                }
                gates.append(eligible)
                pooled_distance_all.append(values["generator_orbit_planar_relative_rms"])
        variants[variant] = {
            "by_degree_shift": by_degree_shift,
            "pooled_distance_median": float(
                np.median(np.concatenate(pooled_distance_all))
            ),
            "eligible": bool(variant in SELECTABLE_VARIANTS and all(gates)),
        }
    eligible = [
        variant for variant in SELECTABLE_VARIANTS if variants[variant]["eligible"]
    ]
    selected = (
        min(eligible, key=lambda item: (variants[item]["pooled_distance_median"], item))
        if eligible
        else None
    )
    selection_reason = "minimum_pooled_median_among_eligible_observable_variants"
    if config.forced_variant is not None:
        selected = config.forced_variant
        selection_reason = "underpowered_forced_systems_lifecycle_only"
    stage = {
        "schema_version": f"{SCHEMA_VERSION}.stage-a",
        "status": "completed",
        "construction_inputs": [
            "decoded_planar_history",
            "calibration_orientation",
            "calibration_signed_speed",
            "calibration_amplitude",
            "calibration_offset",
            "calibration_drift",
            "fixed_time_grid",
            "declared_group_element",
        ],
        "forbidden_inputs": [
            "latent_phase",
            "future_phase",
            "target_posterior",
            "target_bin",
            "branch",
            "quotient_phase",
            "fiber_id",
            "noise_realization",
            "nuisance_dictionary",
            "non_anchor_generator_row",
        ],
        "variants": variants,
        "cells": cells,
        "selection": {
            "eligible_variants": eligible,
            "selected_variant": selected,
            "selection_reason": selection_reason,
            "requantization_alone_eligible": variants[
                "rotate_all_requantized"
            ]["eligible"],
            "causal_stage_authorized": bool(
                selected is not None and not config.stage_a_only
            ),
        },
        "finite": _finite(cells),
    }
    stage["valid"] = bool(stage["finite"] and selected is not None)
    return stage, arrays


def _exact_cohorts_for_stage_b(
    task: CircleTaskConfig,
    config: ActionSemanticsConfig,
    k: int,
    seed: int,
    variant: str,
) -> tuple[dict[str, dict[str, base.deck.OrbitDataset]], dict[str, str]]:
    cohorts: dict[str, dict[str, base.deck.OrbitDataset]] = {}
    hashes: dict[str, str] = {}
    for regime, offset in (("composition", 211), ("extrapolation", 307)):
        exact_task = base.deck.generate_exact_orbits(
            task,
            k=k,
            orbit_count=config.evaluation_orbits,
            seed=seed + offset,
            regime=regime,
        )
        theta = np.linspace(
            0.0, 2.0 * math.pi, config.map_points // k, endpoint=False
        )
        exact_map = base.deck.generate_exact_orbits(
            task,
            k=k,
            orbit_count=config.map_points // k,
            seed=seed + 10_003,
            regime=regime,
            quotient_phases=theta,
            fixed_nuisance=True,
        )
        item = {
            "correct_task": construct_variant_orbit(
                exact_task, task, variant=variant, angle_offset=0.0
            ),
            "control_task": construct_variant_orbit(
                exact_task, task, variant=variant, angle_offset=math.pi / k
            ),
            "correct_map": construct_variant_orbit(
                exact_map, task, variant=variant, angle_offset=0.0
            ),
            "control_map": construct_variant_orbit(
                exact_map, task, variant=variant, angle_offset=math.pi / k
            ),
        }
        cohorts[regime] = item
        for name, dataset in item.items():
            hashes[f"{regime}__{name}"] = base._dataset_hash(dataset)
    return cohorts, hashes


def _prior_observed_detail(
    observed: Mapping[str, Any], k: int, seed: int
) -> tuple[dict[str, Any], str]:
    entry = next(
        item
        for item in observed["results"]
        if int(item["k"]) == k and int(item["seed"]) == seed
    )
    path = Path(entry["path"])
    detail = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != entry["result_sha256"]
        or detail.get("implementation_sha256")
        != SOURCE_OBSERVED_IMPLEMENTATION_SHA256
        or detail.get("gates", {}).get("validity") is not True
    ):
        raise ValueError(f"invalid prior observed detail {path}")
    return detail, entry["result_sha256"]


def _fingerprint(
    config: ActionSemanticsConfig,
    implementation: str,
    selected_variant: str,
    stage_a_sha256: str,
    k: int,
    seed: int,
    provenance: Mapping[str, Any],
    reference_sha256: str,
    prior_sha256: str,
    dataset_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "selected_variant": selected_variant,
        "stage_a_sha256": stage_a_sha256,
        "k": k,
        "seed": seed,
        "provenance": dict(provenance),
        "reference_sha256": reference_sha256,
        "prior_sha256": prior_sha256,
        "dataset_hashes": dict(dataset_hashes),
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
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != value.get("artifacts", {}).get("diagnostics_sha256")
    ):
        raise ValueError(f"incompatible action-semantics result {path}")
    return value


def _campaign_reusable(
    campaign: Mapping[str, Any], config: ActionSemanticsConfig, implementation: str
) -> bool:
    entries = campaign.get("results", [])
    selected = campaign.get("stage_a", {}).get("selection", {}).get("selected_variant")
    expected = (
        {(k, seed) for k in config.degrees for seed in config.seeds}
        if selected is not None
        and not config.stage_a_only
        and campaign.get("evidence_role")
        != "systems_lifecycle_only_not_quality_evidence_stage_a"
        else set()
    )
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and {(item.get("k"), item.get("seed")) for item in entries} == expected
        and all(
            Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["result_sha256"]
            and Path(item["diagnostics_path"]).is_file()
            and _sha256(Path(item["diagnostics_path"]))
            == item["diagnostics_sha256"]
            for item in entries
        )
    )


def run_campaign(
    config: ActionSemanticsConfig, output: Path
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

    task, ladder_campaign, deck_campaign, observed_campaign = _validate_sources(config)
    stage_a, stage_arrays = run_stage_a(task, config)
    stage_a_path = output / "stage_a_results.json"
    stage_a_arrays_path = output / "stage_a_diagnostics.npz"
    _write_json(stage_a_path, stage_a)
    _write_npz(stage_a_arrays_path, stage_arrays)
    stage_a_sha256 = _sha256(stage_a_path)
    stage_a_arrays_sha256 = _sha256(stage_a_arrays_path)
    selected_variant = stage_a["selection"]["selected_variant"]
    causal_authorized = bool(stage_a["selection"]["causal_stage_authorized"])

    device = torch.device(config.device)
    if causal_authorized and device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    load_config = base._deck_config(_base_config(config))
    results: list[dict[str, Any]] = []
    reused = 0
    if causal_authorized:
        assert selected_variant is not None
        for k in config.degrees:
            for seed in config.seeds:
                cell_started = time.perf_counter()
                reference, reference_path, reference_sha256 = base._reference_result(
                    _base_config(config), k, seed
                )
                prior, prior_sha256 = _prior_observed_detail(
                    observed_campaign, k, seed
                )
                cohorts, dataset_hashes = _exact_cohorts_for_stage_b(
                    task, config, k, seed, selected_variant
                )
                system, provenance = base.deck.load_source(
                    task, load_config, k, seed, device
                )
                for parameter in system.parameters():
                    parameter.requires_grad_(False)
                initial_model_state = _state_digest(system.model)
                initial_system_state = base.calibrated._module_digest(system)
                fingerprint = _fingerprint(
                    config,
                    implementation,
                    selected_variant,
                    stage_a_sha256,
                    k,
                    seed,
                    provenance,
                    reference_sha256,
                    prior_sha256,
                    dataset_hashes,
                )
                result_dir = output / "runs" / f"k{k}" / f"seed_{seed}"
                result_path = result_dir / "result.json"
                existing = _reusable_result(result_path, fingerprint, implementation)
                if existing is not None:
                    results.append(existing)
                    reused += 1
                    del system
                    continue
                regime_results: dict[str, Any] = {}
                diagnostic_arrays: dict[str, np.ndarray] = {}
                for regime in REGIMES:
                    reference_wrapper = {
                        "cuts": {
                            cut: {
                                "causal": {
                                    "orbit_average": {
                                        "causal_classification": reference["cuts"][cut][
                                            "causal"
                                        ][regime]["orbit_average"]["causal_classification"]
                                    }
                                }
                            }
                            for cut in CUTS
                        }
                    }
                    regime_result, arrays = base.analyze_regime(
                        system,
                        cohorts[regime],
                        reference_wrapper,
                        _base_config(config),
                        load_config,
                        device,
                    )
                    regime_result["prior_rotate_all_first_preserved_cut"] = prior[
                        "regimes"
                    ][regime]["observed_first_preserved_cut"]
                    prior_index = prior["regimes"][regime][
                        "observed_first_preserved_index"
                    ]
                    selected_index = regime_result["observed_first_preserved_index"]
                    regime_result["front_movement_from_rotate_all"] = (
                        int(selected_index - prior_index)
                        if selected_index is not None and prior_index is not None
                        else None
                    )
                    regime_results[regime] = regime_result
                    diagnostic_arrays.update(
                        {f"{regime}__{name}": value for name, value in arrays.items()}
                    )
                diagnostics_path = result_dir / "action_semantics_diagnostics.npz"
                _write_npz(diagnostics_path, diagnostic_arrays)
                diagnostics_sha256 = _sha256(diagnostics_path)
                state_unchanged = bool(
                    _state_digest(system.model) == initial_model_state
                    and base.calibrated._module_digest(system) == initial_system_state
                )
                replay_pass = all(
                    regime_results[regime]["maximum_replay_error"]
                    <= config.replay_tolerance
                    for regime in REGIMES
                )
                finite = _finite(regime_results)
                validity = bool(state_unchanged and replay_pass and finite)
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
                semantic_control = all(
                    regime_results[regime]["cuts"]["full"][
                        "control_twirl_source_target"
                    ]["causal_classification"]
                    == "preserved"
                    for regime in REGIMES
                )
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "hypothesis_id": HYPOTHESIS_ID,
                    "experiment_id": f"tinyllm-action-semantics-k{k}-seed{seed}",
                    "status": "completed",
                    "evidence_role": EVIDENCE_ROLE,
                    "completed_at": _utc_now(),
                    "k": k,
                    "seed": seed,
                    "selected_variant": selected_variant,
                    "configuration": _json_config(config),
                    "implementation_sha256": implementation,
                    "scientific_fingerprint": fingerprint,
                    "provenance": {
                        **provenance,
                        "reference_result": str(reference_path),
                        "reference_result_sha256": reference_sha256,
                        "prior_observed_result_sha256": prior_sha256,
                    },
                    "dataset_hashes": dataset_hashes,
                    "stage_a_sha256": stage_a_sha256,
                    "regimes": regime_results,
                    "seed_gates": {
                        "mature_observed_twirl": mature,
                        "computational_cover": cover,
                        "front_replication": front_match,
                        "control_validity": control_valid,
                        "semantic_control_preserved": semantic_control,
                    },
                    "gates": {
                        "stage_a_selection": True,
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
                    f"{selected_variant} k{k} seed {seed}: mature={mature} "
                    f"cover={cover} front={front_match} control={control_valid} "
                    f"source_control={semantic_control} valid={validity}",
                    flush=True,
                )
                del system
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if _implementation_digest() != implementation:
                    raise RuntimeError("action-semantics implementation changed during run")

    degree_aggregates: dict[str, Any] = {}
    for k in config.degrees:
        selected = [item for item in results if item["k"] == k]
        if not selected:
            degree_aggregates[str(k)] = {"causal_stage_not_run": True}
            continue
        counts = {
            name: sum(int(item["seed_gates"][name]) for item in selected)
            for name in (
                "mature_observed_twirl",
                "computational_cover",
                "front_replication",
                "control_validity",
                "semantic_control_preserved",
            )
        }
        degree_aggregates[str(k)] = {
            **{f"{name}_count": value for name, value in counts.items()},
            "all_primary_gates_pass": bool(
                counts["mature_observed_twirl"] >= config.required_seed_passes
                and counts["computational_cover"] >= config.required_seed_passes
                and counts["front_replication"] >= config.required_seed_passes
                and counts["control_validity"] >= config.required_seed_passes
                and counts["semantic_control_preserved"]
                <= config.maximum_control_seed_passes
            ),
        }
    valid = bool(
        stage_a["finite"]
        and (
            not causal_authorized
            or all(item["gates"]["validity"] for item in results)
        )
    )
    if config.allow_underpowered:
        classification, primary_pass = (
            "systems_lifecycle_only_not_quality_evidence",
            False,
        )
    elif selected_variant is None:
        classification, primary_pass = (
            "no_observable_action_semantics_candidate",
            False,
        )
    elif config.stage_a_only:
        classification, primary_pass = ("stage_a_candidate_selected", False)
    else:
        primary_pass = bool(
            all(
                degree_aggregates[str(k)].get("all_primary_gates_pass") is True
                for k in config.degrees
            )
            and degree_aggregates["3"].get("front_replication_count", 0)
            >= config.required_seed_passes
        )
        classification = (
            "observable_residual_transport_restores_front"
            if primary_pass
            else "observable_candidate_does_not_restore_front"
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
            "systems_lifecycle_only_not_quality_evidence_stage_a"
            if config.allow_underpowered and not causal_authorized
            else "systems_lifecycle_only_not_quality_evidence"
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
                torch.cuda.get_device_name(device)
                if causal_authorized and device.type == "cuda"
                else "not_loaded_stage_a_only"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if causal_authorized and device.type == "cuda"
                else 0
            ),
        },
        "provenance": {
            "source_ladder_campaign_sha256": base.SOURCE_LADDER_CAMPAIGN_SHA256,
            "source_deck_campaign_sha256": base.SOURCE_DECK_CAMPAIGN_SHA256,
            "source_observed_campaign_sha256": SOURCE_OBSERVED_CAMPAIGN_SHA256,
            "source_observed_implementation_sha256": SOURCE_OBSERVED_IMPLEMENTATION_SHA256,
            "source_observed_result_manifest_sha256": SOURCE_OBSERVED_RESULT_MANIFEST_SHA256,
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
        "task_config": asdict(task),
        "stage_a": stage_a,
        "stage_a_sha256": stage_a_sha256,
        "summary": {
            "requested": len(config.degrees) * len(config.seeds) if causal_authorized else 0,
            "scheduled": len(config.degrees) * len(config.seeds) - reused if causal_authorized else 0,
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
            "selected_variant": selected_variant,
            "causal_stage_authorized": causal_authorized,
            "degrees": degree_aggregates,
        },
        "results": result_entries,
        "result_manifest_sha256": manifest_sha256,
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "Stage A selects only from registered input geometry, analytic character, group-law, norm, and support metrics before any candidate model output is inspected.",
            "The observed residual-fixed estimator uses deterministic full-history demodulation and no fitted parameter.",
            "Oracle phase variants are attribution controls and cannot enter Stage B.",
            "The third sensor channel is outside the analytic carrier action and is excluded from planar distance selection.",
            "The retained d6 C2/C3 systems do not establish other groups, sensors, or architecture-population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "stage_a": str(stage_a_path),
            "stage_a_arrays": str(stage_a_arrays_path),
            "stage_a_arrays_sha256": stage_a_arrays_sha256,
        },
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
            "data/experiments/tinyllm_observed_cyclic_action_semantics/"
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
    parser.add_argument("--force-variant", choices=SELECTABLE_VARIANTS)
    parser.add_argument("--stage-a-only", action="store_true")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = ActionSemanticsConfig(
        degrees=args.degrees,
        seeds=args.seeds,
        evaluation_orbits=args.evaluation_orbits,
        map_points=args.map_points,
        activation_batch_size=args.activation_batch_size,
        required_seed_passes=args.required_seed_passes,
        maximum_control_seed_passes=args.maximum_control_seed_passes,
        forced_variant=args.force_variant,
        stage_a_only=args.stage_a_only,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
