#!/usr/bin/env python3
"""Swap calibrated symmetry features across frozen TinyLLM checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
import experiments.structure_net.tinyllm_nuisance_support_scaling as nuisance
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig, _state_digest
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-cross-seed-symmetry-feature-swap.v1"
HYPOTHESIS_ID = "tinyllm-cross-seed-symmetry-feature-gauge-v1"
SOURCE_SCHEMA_VERSION = calibrated.SCHEMA_VERSION
SOURCE_HYPOTHESIS_ID = calibrated.HYPOTHESIS_ID
CONDITION = "learned_calibrated_equivariant"
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
COHORT_SPECS: tuple[tuple[str, str, int], ...] = (
    ("a_composition", "composition", 91_301),
    ("a_extrapolation", "extrapolation", 92_301),
    ("b_composition", "composition", 91_302),
    ("b_extrapolation", "extrapolation", 92_302),
)
GROUP_ACTIONS: tuple[dict[str, Any], ...] = (
    {
        "angle": 0.37,
        "scale": 1.70,
        "offset": (0.23, -0.31),
        "drift": (0.11, 0.07),
    },
    {
        "angle": -1.13,
        "scale": 0.61,
        "offset": (-0.19, 0.29),
        "drift": (-0.08, 0.13),
    },
)


@dataclass(frozen=True)
class SymmetryFeatureSwapConfig:
    source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    sample_count: int = 512
    batch_size: int = 128
    direct_replay_tolerance: float = 1e-6
    group_contract_tolerance: float = 1e-5
    correlation_floor: float = 0.90
    correlation_loss_ceiling: float = 0.02
    accuracy_loss_ceiling: float = 0.03
    cross_entropy_increase_ceiling: float = 0.05
    fisher_distance_ceiling: float = 0.10
    control_cross_entropy_margin: float = 0.10
    required_pair_passes: int = 16
    required_target_passes: int = 4
    required_sources_per_target: int = 3
    device: str = "cuda:0"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("the preregistered campaign fixes five source seeds")
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2:
            raise ValueError("at least two distinct checkpoints are required")
        if self.sample_count != 512 and not self.allow_underpowered:
            raise ValueError("the preregistered campaign fixes 512 examples per cell")
        if self.sample_count < 8 or self.sample_count % 2:
            raise ValueError("sample count must be even and at least eight")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if not self.allow_underpowered:
            if self.required_pair_passes != 16:
                raise ValueError("primary pair gate is fixed at 16/20")
            if self.required_target_passes != 4 or self.required_sources_per_target != 3:
                raise ValueError("primary target-level gate is fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "nuisance_generator": Path(nuisance.__file__),
        "tinyllm_model": Path(inspect.getfile(TinyLLMModel)),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(source_digests: Mapping[str, str] | None = None) -> str:
    values = dict(source_digests or _source_digests())
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_role(config: SymmetryFeatureSwapConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_frozen_checkpoint_causal_diagnostic"
    )


def canonical_feature(
    system: calibrated.CalibratedTinyLLM,
    sensor: torch.Tensor,
    calibration: torch.Tensor,
) -> torch.Tensor:
    if system.encoder is None:
        raise ValueError("canonical feature requires the learned equivariant encoder")
    vector = system.encoder.equivariant_vector(sensor, calibration)
    orientation = calibration[:, :2]
    dot = (vector * orientation).sum(-1)
    cross = orientation[:, 0] * vector[:, 1] - orientation[:, 1] * vector[:, 0]
    return torch.stack((dot, cross, calibration[:, 2]), -1)


def _rotate(values: torch.Tensor, angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y = values.unbind(-1)
    return torch.stack((cosine * x - sine * y, sine * x + cosine * y), -1)


def apply_acquisition_action(
    sensor: torch.Tensor,
    calibration_packet: torch.Tensor,
    normalized_history: torch.Tensor,
    action: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the declared positive-similarity acquisition action."""
    angle = float(action["angle"])
    scale = float(action["scale"])
    if scale <= 0.0:
        raise ValueError("acquisition scale must be positive")
    offset = torch.as_tensor(action["offset"], dtype=sensor.dtype, device=sensor.device)
    drift = torch.as_tensor(action["drift"], dtype=sensor.dtype, device=sensor.device)
    transformed_sensor = sensor.clone()
    transformed_sensor[..., :2] = (
        scale * _rotate(sensor[..., :2], angle)
        + offset[None, None, :]
        + drift[None, None, :] * normalized_history[None, :, None]
    )
    transformed_calibration = calibration_packet.clone()
    transformed_calibration[:, :2] = _rotate(calibration_packet[:, :2], angle)
    transformed_calibration[:, 3] = scale * calibration_packet[:, 3]
    transformed_calibration[:, 4:6] = (
        scale * _rotate(calibration_packet[:, 4:6], angle) + offset[None, :]
    )
    transformed_calibration[:, 6:8] = (
        scale * _rotate(calibration_packet[:, 6:8], angle) + drift[None, :]
    )
    return transformed_sensor, transformed_calibration


def _safe_pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-12:
        return 0.0
    return float((left @ right) / denominator)


def _posterior_metrics(
    posterior: torch.Tensor,
    target: torch.Tensor,
    true_cosine: torch.Tensor,
    direct: torch.Tensor,
) -> dict[str, float]:
    posterior = posterior.double()
    target = target.double()
    direct = direct.double()
    centers = torch.linspace(-1.0, 1.0, posterior.shape[1], dtype=torch.float64)
    prediction = posterior @ centers
    direct_prediction = direct @ centers
    target_bins = target.argmax(-1)
    predicted_bins = posterior.argmax(-1)
    affinity = torch.sqrt(posterior.clamp_min(0.0) * direct.clamp_min(0.0)).sum(-1)
    fisher = 2.0 * torch.arccos(affinity.clamp(-1.0, 1.0))
    moment_difference = (prediction - direct_prediction).abs()
    return {
        "exact_bin_accuracy": float((predicted_bins == target_bins).double().mean()),
        "target_cross_entropy": float(
            -(target * posterior.clamp_min(1e-12).log()).sum(-1).mean()
        ),
        "posterior_mean_cosine_correlation": _safe_pearson(prediction, true_cosine),
        "posterior_mean_cosine_rmse": float(
            torch.sqrt(torch.square(prediction - true_cosine.double()).mean())
        ),
        "mean_fisher_rao_from_direct": float(fisher.mean()),
        "mean_absolute_posterior_mean_difference": float(moment_difference.mean()),
        "p95_absolute_posterior_mean_difference": float(
            torch.quantile(moment_difference, 0.95)
        ),
    }


def _with_direct_deltas(
    metrics: Mapping[str, float], direct: Mapping[str, float]
) -> dict[str, float]:
    return {
        **dict(metrics),
        "accuracy_loss_from_direct": (
            direct["exact_bin_accuracy"] - metrics["exact_bin_accuracy"]
        ),
        "cross_entropy_increase_from_direct": (
            metrics["target_cross_entropy"] - direct["target_cross_entropy"]
        ),
        "correlation_loss_from_direct": (
            direct["posterior_mean_cosine_correlation"]
            - metrics["posterior_mean_cosine_correlation"]
        ),
    }


def feature_geometry(
    source: torch.Tensor,
    target: torch.Tensor,
    source_scalar: torch.Tensor,
    target_scalar: torch.Tensor,
) -> dict[str, Any]:
    source = source.double()
    target = target.double()
    source_angle = torch.atan2(source[:, 1], source[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    difference = torch.atan2(
        torch.sin(source_angle - target_angle),
        torch.cos(source_angle - target_angle),
    )
    resultant = torch.exp(1j * difference).mean()
    scalar_difference = (source_scalar.double() - target_scalar.double()).abs().flatten()
    return {
        "mean_absolute_angular_displacement_radians": float(difference.abs().mean()),
        "p95_absolute_angular_displacement_radians": float(
            torch.quantile(difference.abs(), 0.95)
        ),
        "circular_concentration": float(resultant.abs()),
        "circular_mean_offset_radians": float(torch.angle(resultant)),
        "component_correlations": [
            _safe_pearson(source[:, index], target[:, index]) for index in range(3)
        ],
        "mean_feature_l2_difference": float(
            torch.linalg.vector_norm(source - target, dim=-1).mean()
        ),
        "source_scalar_target_scalar_correlation": _safe_pearson(
            source_scalar, target_scalar
        ),
        "mean_absolute_source_target_scalar_difference": float(
            scalar_difference.mean()
        ),
        "p95_absolute_source_target_scalar_difference": float(
            torch.quantile(scalar_difference, 0.95)
        ),
    }


def cell_gate(
    swap: Mapping[str, float],
    direct: Mapping[str, float],
    shuffled: Mapping[str, float],
    half_turn: Mapping[str, float],
    replay_max_error: float,
    config: SymmetryFeatureSwapConfig,
) -> dict[str, Any]:
    checks = {
        "direct_replay": replay_max_error <= config.direct_replay_tolerance,
        "correlation_floor": (
            swap["posterior_mean_cosine_correlation"] >= config.correlation_floor
        ),
        "correlation_preservation": (
            direct["posterior_mean_cosine_correlation"]
            - swap["posterior_mean_cosine_correlation"]
            <= config.correlation_loss_ceiling
        ),
        "accuracy_preservation": (
            direct["exact_bin_accuracy"] - swap["exact_bin_accuracy"]
            <= config.accuracy_loss_ceiling
        ),
        "cross_entropy_preservation": (
            swap["target_cross_entropy"] - direct["target_cross_entropy"]
            <= config.cross_entropy_increase_ceiling
        ),
        "posterior_fidelity": (
            swap["mean_fisher_rao_from_direct"] <= config.fisher_distance_ceiling
        ),
        "shuffled_specificity": (
            shuffled["target_cross_entropy"] - swap["target_cross_entropy"]
            >= config.control_cross_entropy_margin
        ),
        "half_turn_specificity": (
            half_turn["target_cross_entropy"] - swap["target_cross_entropy"]
            >= config.control_cross_entropy_margin
        ),
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def scalar_cell_gate(
    scalar_swap: Mapping[str, float],
    direct: Mapping[str, float],
    replay_max_error: float,
    config: SymmetryFeatureSwapConfig,
) -> dict[str, Any]:
    checks = {
        "direct_replay": replay_max_error <= config.direct_replay_tolerance,
        "correlation_floor": (
            scalar_swap["posterior_mean_cosine_correlation"] >= config.correlation_floor
        ),
        "correlation_preservation": (
            direct["posterior_mean_cosine_correlation"]
            - scalar_swap["posterior_mean_cosine_correlation"]
            <= config.correlation_loss_ceiling
        ),
        "accuracy_preservation": (
            direct["exact_bin_accuracy"] - scalar_swap["exact_bin_accuracy"]
            <= config.accuracy_loss_ceiling
        ),
        "cross_entropy_preservation": (
            scalar_swap["target_cross_entropy"] - direct["target_cross_entropy"]
            <= config.cross_entropy_increase_ceiling
        ),
        "posterior_fidelity": (
            scalar_swap["mean_fisher_rao_from_direct"]
            <= config.fisher_distance_ceiling
        ),
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def _dataset_hash(dataset: calibrated.CalibratedDataset) -> str:
    return invariant._tensor_digest(
        dataset.paired.circle.input_ids,
        dataset.paired.circle.target_posteriors,
        dataset.paired.fiber.cosine,
        dataset.paired.fiber.branch,
        dataset.calibration,
    )


def _load_systems(
    config: SymmetryFeatureSwapConfig, device: torch.device
) -> tuple[
    CircleTaskConfig,
    calibrated.CalibratedFrontendConfig,
    dict[int, calibrated.CalibratedTinyLLM],
    dict[int, dict[str, Any]],
    dict[str, Any],
    Path,
]:
    root = Path(config.source_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign.get("schema_version") != SOURCE_SCHEMA_VERSION
        or campaign.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or int(campaign.get("summary", {}).get("completed", -1)) != 15
        or int(campaign.get("summary", {}).get("failed", -1)) != 0
    ):
        raise ValueError(f"invalid calibrated predecessor campaign {campaign_path}")
    source_seeds = tuple(int(seed) for seed in campaign["configuration"]["seeds"])
    if not set(config.seeds).issubset(source_seeds):
        raise ValueError("requested seeds are absent from predecessor campaign")
    task = CircleTaskConfig(**campaign["task_config"])
    frontend_config = calibrated._config_from_mapping(campaign["configuration"])
    systems: dict[int, calibrated.CalibratedTinyLLM] = {}
    provenance: dict[int, dict[str, Any]] = {}
    for seed in config.seeds:
        directory = root / "runs" / CONDITION / f"seed_{seed}"
        result_path = directory / "result.json"
        model_path = directory / "model.pt"
        frontend_path = directory / "frontend.pt"
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            detail.get("schema_version") != SOURCE_SCHEMA_VERSION
            or detail.get("hypothesis_id") != SOURCE_HYPOTHESIS_ID
            or detail.get("condition") != CONDITION
            or int(detail.get("seed", -1)) != seed
            or detail.get("status") != "completed"
            or detail.get("training", {}).get("checkpoint") != str(model_path)
            or detail.get("training", {}).get("frontend_checkpoint")
            != str(frontend_path)
        ):
            raise ValueError(f"invalid retained result {result_path}")
        model = TinyLLMModel.from_checkpoint(model_path, map_location="cpu")
        system = calibrated.CalibratedTinyLLM(
            model, CONDITION, task, frontend_config
        )
        frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
        if frontend.get("condition") != CONDITION:
            raise ValueError(f"frontend condition mismatch {frontend_path}")
        if system.scalar_embedding is None or system.encoder is None:
            raise AssertionError("learned calibrated system is incomplete")
        system.scalar_embedding.load_state_dict(frontend["scalar_embedding"])
        system.encoder.load_state_dict(frontend["encoder"])
        model_state = _state_digest(system.model)
        system_state = calibrated._module_digest(system)
        if model_state != detail["training"]["final_model_state_sha256"]:
            raise ValueError(f"model state mismatch for seed {seed}")
        if system_state != detail["training"]["final_system_state_sha256"]:
            raise ValueError(f"system state mismatch for seed {seed}")
        systems[seed] = system.to(device).eval()
        provenance[seed] = {
            "result": str(result_path),
            "result_sha256": _sha256(result_path),
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "model_checkpoint": str(model_path),
            "model_checkpoint_sha256": _sha256(model_path),
            "frontend_checkpoint": str(frontend_path),
            "frontend_checkpoint_sha256": _sha256(frontend_path),
            "model_state_sha256": model_state,
            "system_state_sha256": system_state,
        }
    return task, frontend_config, systems, provenance, campaign, campaign_path


@torch.no_grad()
def _continue_from_scalar(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    scalar: torch.Tensor,
    task: CircleTaskConfig,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if system.scalar_embedding is None:
        raise AssertionError("structured scalar embedding missing")
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    output = []
    for start in range(0, len(input_ids), batch_size):
        stop = min(len(input_ids), start + batch_size)
        inputs = input_ids[start:stop].to(device)
        value_scalar = scalar[start:stop].to(device)
        bos = system.model.transformer["wte"](inputs[:, :1])
        scalar_token = system.scalar_embedding(value_scalar)[:, None, :]
        query = system.model.transformer["wte"](inputs[:, -1:])
        value = torch.cat((bos, scalar_token, query), 1)
        positions = torch.arange(value.shape[1], device=device)
        value = value + system.model.transformer["wpe"](positions)
        for block in system.model.transformer["h"]:
            value = block(value)
        logits = invariant._task_logits(system.model, value[:, -1], answer_ids)
        output.append(torch.softmax(logits, -1).cpu())
    return torch.cat(output)


@torch.no_grad()
def _continue_from_feature(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    feature: torch.Tensor,
    task: CircleTaskConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if system.encoder is None:
        raise AssertionError("learned encoder missing")
    scalars = []
    for start in range(0, len(feature), batch_size):
        stop = min(len(feature), start + batch_size)
        scalars.append(system.encoder.scalar_map(feature[start:stop].to(device)).cpu())
    scalar = torch.cat(scalars)
    return _continue_from_scalar(
        system, input_ids, scalar, task, batch_size, device
    ), scalar


@torch.no_grad()
def _extract_checkpoint_cell(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: SymmetryFeatureSwapConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    features = []
    scalars = []
    ordinary = []
    for start in range(0, len(sensor), config.batch_size):
        stop = min(len(sensor), start + config.batch_size)
        inputs = dataset.paired.circle.input_ids[start:stop].to(device)
        observed = sensor[start:stop].to(device)
        packet = dataset.calibration[start:stop].to(device)
        feature = canonical_feature(system, observed, packet)
        scalar = system.encoder.scalar_map(feature)  # type: ignore[union-attr]
        residual = system.forward_cuts(inputs, observed, packet)["full"]
        posterior = torch.softmax(
            invariant._task_logits(system.model, residual, answer_ids), -1
        )
        features.append(feature.cpu())
        scalars.append(scalar.cpu())
        ordinary.append(posterior.cpu())
    return {
        "feature": torch.cat(features),
        "scalar": torch.cat(scalars),
        "ordinary_posterior": torch.cat(ordinary),
    }


@torch.no_grad()
def _group_contract_cell(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: SymmetryFeatureSwapConfig,
    device: torch.device,
) -> dict[str, Any]:
    if system.encoder is None:
        raise AssertionError("learned encoder missing")
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    action_errors = []
    for action in GROUP_ACTIONS:
        maximum = 0.0
        for start in range(0, len(sensor), config.batch_size):
            stop = min(len(sensor), start + config.batch_size)
            observed = sensor[start:stop].to(device)
            packet = dataset.calibration[start:stop].to(device)
            baseline = canonical_feature(system, observed, packet)
            transformed_sensor, transformed_packet = apply_acquisition_action(
                observed,
                packet,
                system.encoder.normalized_history,
                action,
            )
            transformed = canonical_feature(
                system, transformed_sensor, transformed_packet
            )
            maximum = max(maximum, float((baseline - transformed).abs().max()))
        action_errors.append({"action": dict(action), "maximum_absolute_error": maximum})
    maximum = max(item["maximum_absolute_error"] for item in action_errors)
    return {
        "actions": action_errors,
        "maximum_absolute_error": maximum,
        "pass": maximum <= config.group_contract_tolerance,
    }


def _permutation(count: int, dataset_seed: int, source_seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        5_000_003 + dataset_seed * 101 + source_seed * 10_007
    )
    return torch.randperm(count, generator=generator)


def _fingerprint(
    config: SymmetryFeatureSwapConfig,
    source_campaign_sha256: str,
    source_digests: Mapping[str, str],
    provenance: Mapping[int, Mapping[str, Any]],
    dataset_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "cohort_specs": COHORT_SPECS,
        "group_actions": GROUP_ACTIONS,
        "source_campaign_sha256": source_campaign_sha256,
        "source_digests": dict(source_digests),
        "checkpoint_artifacts": {
            str(seed): {
                key: value
                for key, value in values.items()
                if key.endswith("sha256") or key == "scientific_fingerprint"
            }
            for seed, values in provenance.items()
        },
        "dataset_hashes": dict(dataset_hashes),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _result_reusable(
    value: Mapping[str, Any],
    fingerprint: str,
    implementation_sha256: str,
    source_seed: int,
    target_seed: int,
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("implementation_sha256") == implementation_sha256
        and int(value.get("source_seed", -1)) == source_seed
        and int(value.get("target_seed", -1)) == target_seed
    )


def aggregate_pair_results(
    results: Sequence[Mapping[str, Any]],
    group_contracts: Mapping[int, Mapping[str, Any]],
    config: SymmetryFeatureSwapConfig,
) -> dict[str, Any]:
    pair_pass_count = sum(bool(result["gates"]["pair_pass"]) for result in results)
    scalar_pair_pass_count = sum(
        bool(result["gates"]["scalar_pair_pass"]) for result in results
    )
    target_summary = {}
    for target in config.seeds:
        target_results = [result for result in results if result["target_seed"] == target]
        passing_sources = [
            int(result["source_seed"])
            for result in target_results
            if result["gates"]["pair_pass"]
        ]
        scalar_passing_sources = [
            int(result["source_seed"])
            for result in target_results
            if result["gates"]["scalar_pair_pass"]
        ]
        target_summary[str(target)] = {
            "passing_sources": passing_sources,
            "pass_count": len(passing_sources),
            "target_pass": len(passing_sources) >= config.required_sources_per_target,
            "scalar_passing_sources": scalar_passing_sources,
            "scalar_pass_count": len(scalar_passing_sources),
            "scalar_target_pass": (
                len(scalar_passing_sources) >= config.required_sources_per_target
            ),
        }
    target_pass_count = sum(value["target_pass"] for value in target_summary.values())
    scalar_target_pass_count = sum(
        value["scalar_target_pass"] for value in target_summary.values()
    )
    group_count = sum(bool(value["pass"]) for value in group_contracts.values())
    full_group_contract = group_count == len(config.seeds)
    confirmed = bool(
        not config.allow_underpowered
        and full_group_contract
        and pair_pass_count >= config.required_pair_passes
        and target_pass_count >= config.required_target_passes
    )
    scalar_campaign_pass = bool(
        not config.allow_underpowered
        and full_group_contract
        and scalar_pair_pass_count >= config.required_pair_passes
        and scalar_target_pass_count >= config.required_target_passes
    )
    check_failures: dict[str, int] = {}
    for result in results:
        for cell in result["cells"].values():
            for name, passed in cell["feature_swap_gate"]["checks"].items():
                check_failures[name] = check_failures.get(name, 0) + int(not passed)
    if config.allow_underpowered:
        conclusion = "systems_lifecycle_only_not_quality_evidence"
    elif confirmed:
        conclusion = "confirmed_portable_symmetry_fixed_feature_gauge"
    elif scalar_campaign_pass:
        conclusion = "feature_gauge_checkpoint_local_scalar_bottleneck_portable"
    else:
        conclusion = "feature_and_scalar_interfaces_checkpoint_coadapted"
    return {
        "group_contract_pass_count": group_count,
        "group_contract_required": len(config.seeds),
        "pair_pass_count": pair_pass_count,
        "pair_pass_required": config.required_pair_passes,
        "scalar_pair_pass_count": scalar_pair_pass_count,
        "target_pass_count": target_pass_count,
        "target_pass_required": config.required_target_passes,
        "scalar_target_pass_count": scalar_target_pass_count,
        "target_summary": target_summary,
        "feature_cell_check_failure_counts": check_failures,
        "confirmed": confirmed,
        "scalar_campaign_pass": scalar_campaign_pass,
        "conclusion": conclusion,
    }


@torch.no_grad()
def run_campaign(
    config: SymmetryFeatureSwapConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    (
        task,
        frontend_config,
        systems,
        provenance,
        source_campaign,
        source_campaign_path,
    ) = _load_systems(config, device)
    datasets = {
        name: calibrated.generate_calibrated_dataset(
            task,
            sample_count=config.sample_count,
            seed=seed,
            regime=regime,
            shuffle=True,
        )
        for name, regime, seed in COHORT_SPECS
    }
    dataset_hashes = {name: _dataset_hash(dataset) for name, dataset in datasets.items()}
    fingerprint = _fingerprint(
        config,
        _sha256(source_campaign_path),
        source_digests,
        provenance,
        dataset_hashes,
    )
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("hypothesis_id") == HYPOTHESIS_ID
            and existing.get("status") == "completed"
            and existing.get("scientific_fingerprint") == fingerprint
            and existing.get("implementation_sha256") == implementation
            and all(
                Path(entry["path"]).is_file()
                and _sha256(Path(entry["path"])) == entry["result_sha256"]
                for entry in existing.get("results", [])
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    extracted: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    group_contracts: dict[int, dict[str, Any]] = {
        seed: {"cells": {}, "maximum_absolute_error": 0.0, "pass": True}
        for seed in config.seeds
    }
    for name, _, _ in COHORT_SPECS:
        extracted[name] = {}
        for seed in config.seeds:
            extracted[name][seed] = _extract_checkpoint_cell(
                systems[seed], datasets[name], task, config, device
            )
            contract = _group_contract_cell(
                systems[seed], datasets[name], task, config, device
            )
            group_contracts[seed]["cells"][name] = contract
            group_contracts[seed]["maximum_absolute_error"] = max(
                group_contracts[seed]["maximum_absolute_error"],
                contract["maximum_absolute_error"],
            )
            group_contracts[seed]["pass"] = bool(
                group_contracts[seed]["pass"] and contract["pass"]
            )

    results = []
    reused = 0
    for source_seed in config.seeds:
        for target_seed in config.seeds:
            if source_seed == target_seed:
                continue
            result_path = (
                output
                / "runs"
                / f"source_{source_seed}"
                / f"target_{target_seed}"
                / "result.json"
            )
            pair_fingerprint = hashlib.sha256(
                f"{fingerprint}:{source_seed}:{target_seed}".encode()
            ).hexdigest()
            if result_path.is_file():
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if _result_reusable(
                    existing,
                    pair_fingerprint,
                    implementation,
                    source_seed,
                    target_seed,
                ):
                    results.append(existing)
                    reused += 1
                    print(f"reuse {source_seed}->{target_seed}", flush=True)
                    continue
                raise ValueError(f"incompatible completed pair {result_path}")
            cells = {}
            for name, regime, dataset_seed in COHORT_SPECS:
                dataset = datasets[name]
                source = extracted[name][source_seed]
                target = extracted[name][target_seed]
                inputs = dataset.paired.circle.input_ids
                target_posterior = dataset.paired.circle.target_posteriors
                true_cosine = dataset.paired.fiber.cosine
                target_direct, target_scalar_replay = _continue_from_feature(
                    systems[target_seed],
                    inputs,
                    target["feature"],
                    task,
                    config.batch_size,
                    device,
                )
                replay_error = float(
                    (target_direct - target["ordinary_posterior"]).abs().max()
                )
                source_feature_swap, target_map_source_scalar = _continue_from_feature(
                    systems[target_seed],
                    inputs,
                    source["feature"],
                    task,
                    config.batch_size,
                    device,
                )
                source_scalar_swap = _continue_from_scalar(
                    systems[target_seed],
                    inputs,
                    source["scalar"],
                    task,
                    config.batch_size,
                    device,
                )
                permutation = _permutation(
                    config.sample_count, dataset_seed, source_seed
                )
                shuffled, _ = _continue_from_feature(
                    systems[target_seed],
                    inputs,
                    source["feature"][permutation],
                    task,
                    config.batch_size,
                    device,
                )
                half_turn_feature = source["feature"].clone()
                half_turn_feature[:, :2] *= -1.0
                half_turn, _ = _continue_from_feature(
                    systems[target_seed],
                    inputs,
                    half_turn_feature,
                    task,
                    config.batch_size,
                    device,
                )
                direct_metrics = _posterior_metrics(
                    target_direct,
                    target_posterior,
                    true_cosine,
                    target_direct,
                )
                condition_posteriors = {
                    "source_feature_swap": source_feature_swap,
                    "source_scalar_swap": source_scalar_swap,
                    "shuffled_source_feature": shuffled,
                    "half_turn_source_feature": half_turn,
                }
                condition_metrics = {
                    condition: _with_direct_deltas(
                        _posterior_metrics(
                            posterior,
                            target_posterior,
                            true_cosine,
                            target_direct,
                        ),
                        direct_metrics,
                    )
                    for condition, posterior in condition_posteriors.items()
                }
                condition_metrics["target_direct"] = _with_direct_deltas(
                    direct_metrics, direct_metrics
                )
                feature_gate = cell_gate(
                    condition_metrics["source_feature_swap"],
                    condition_metrics["target_direct"],
                    condition_metrics["shuffled_source_feature"],
                    condition_metrics["half_turn_source_feature"],
                    replay_error,
                    config,
                )
                scalar_gate = scalar_cell_gate(
                    condition_metrics["source_scalar_swap"],
                    condition_metrics["target_direct"],
                    replay_error,
                    config,
                )
                cells[name] = {
                    "cohort": name[0],
                    "regime": regime,
                    "dataset_seed": dataset_seed,
                    "dataset_sha256": dataset_hashes[name],
                    "direct_replay_max_absolute_posterior_error": replay_error,
                    "direct_scalar_replay_max_absolute_error": float(
                        (target_scalar_replay - target["scalar"]).abs().max()
                    ),
                    "conditions": condition_metrics,
                    "feature_geometry": feature_geometry(
                        source["feature"],
                        target["feature"],
                        source["scalar"],
                        target["scalar"],
                    ),
                    "target_map_source_scalar": {
                        "correlation_with_target_direct_scalar": _safe_pearson(
                            target_map_source_scalar, target["scalar"]
                        ),
                        "mean_absolute_difference_from_target_direct_scalar": float(
                            (target_map_source_scalar - target["scalar"]).abs().mean()
                        ),
                    },
                    "control_permutation_sha256": hashlib.sha256(
                        permutation.numpy().tobytes()
                    ).hexdigest(),
                    "feature_swap_gate": feature_gate,
                    "scalar_swap_gate": scalar_gate,
                }
            pair_group_contract = bool(
                group_contracts[source_seed]["pass"]
                and group_contracts[target_seed]["pass"]
            )
            pair_pass = bool(
                pair_group_contract
                and all(cell["feature_swap_gate"]["pass"] for cell in cells.values())
            )
            scalar_pair_pass = bool(
                pair_group_contract
                and all(cell["scalar_swap_gate"]["pass"] for cell in cells.values())
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": (
                    f"tinyllm-symmetry-feature-swap-{source_seed}-to-{target_seed}"
                ),
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "source_seed": source_seed,
                "target_seed": target_seed,
                "configuration": asdict(config),
                "scientific_fingerprint": pair_fingerprint,
                "implementation_sha256": implementation,
                "provenance": {
                    "source": provenance[source_seed],
                    "target": provenance[target_seed],
                    "predecessor_campaign": str(source_campaign_path),
                    "predecessor_campaign_sha256": _sha256(source_campaign_path),
                },
                "cells": cells,
                "gates": {
                    "pair_group_contract": pair_group_contract,
                    "pair_pass": pair_pass,
                    "scalar_pair_pass": scalar_pair_pass,
                },
                "trained_models": 0,
                "fitted_maps": 0,
                "artifacts": {"result": str(result_path)},
            }
            _write_json(result_path, result)
            results.append(result)
            print(
                f"{source_seed}->{target_seed}: feature={'pass' if pair_pass else 'fail'} "
                f"scalar={'pass' if scalar_pair_pass else 'fail'}",
                flush=True,
            )
            if _implementation_digest() != implementation:
                raise RuntimeError("implementation changed during campaign")

    aggregates = aggregate_pair_results(results, group_contracts, config)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "cohort_specs": [
            {"name": name, "regime": regime, "seed": seed}
            for name, regime, seed in COHORT_SPECS
        ],
        "group_actions": list(GROUP_ACTIONS),
        "scientific_fingerprint": fingerprint,
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_allocated_gb": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        },
        "source_campaign": {
            "path": str(source_campaign_path),
            "sha256": _sha256(source_campaign_path),
            "schema_version": source_campaign["schema_version"],
            "hypothesis_id": source_campaign["hypothesis_id"],
            "implementation_sha256": source_campaign["implementation_sha256"],
        },
        "provenance_by_seed": provenance,
        "dataset_hashes": dataset_hashes,
        "group_contracts": group_contracts,
        "summary": {
            "requested": len(config.seeds) * (len(config.seeds) - 1),
            "completed": len(results),
            "failed": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_maps": 0,
        },
        "aggregates": aggregates,
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "source_seed": result["source_seed"],
                "target_seed": result["target_seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "pair_pass": result["gates"]["pair_pass"],
                "scalar_pair_pass": result["gates"]["scalar_pair_pass"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "This reuses five known successful learned calibrated-equivariant checkpoints.",
            "Directed pairs are repeated interventions; target checkpoints are the replication unit.",
            "The learned equivariant vector is not uniquely gauge-fixed by equivariance alone.",
            "Task effects are conditioned on each target scalar map, embedding, transformer, and decoder.",
            "A negative swap does not refute within-checkpoint acquisition equivariance.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "pair_results": str(output / "runs" / "source_*" / "target_*" / "result.json"),
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", default=SymmetryFeatureSwapConfig.source_root
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_cross_seed_symmetry_feature_swap/"
            "20260806_d8_preregistered"
        ),
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=PRIMARY_SEEDS)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = SymmetryFeatureSwapConfig(
        source_root=args.source_root,
        seeds=args.seeds,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
        required_pair_passes=(
            16 if not args.allow_underpowered else len(args.seeds) * (len(args.seeds) - 1)
        ),
        required_target_passes=(4 if not args.allow_underpowered else len(args.seeds)),
        required_sources_per_target=(3 if not args.allow_underpowered else len(args.seeds) - 1),
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
