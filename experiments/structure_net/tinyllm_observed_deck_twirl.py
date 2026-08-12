#!/usr/bin/env python3
"""Test an oracle-free observed C2 twirl in frozen calibrated TinyLLMs."""

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
from typing import Any, Mapping

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibrated_frontend_causal_closure as closure
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-observed-deck-twirl-causal-closure.v1"
HYPOTHESIS_ID = "tinyllm-observed-deck-twirl-causal-closure-v1"
EVIDENCE_ROLE = "preregistered_frozen_observed_action_intervention"
SOURCE_CLOSURE_SCHEMA = "nal.tinyllm-calibrated-frontend-causal-closure.v1"
SOURCE_CLOSURE_CAMPAIGN_SHA256 = (
    "1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14"
)
SOURCE_CLOSURE_IMPLEMENTATION_SHA256 = (
    "5060b45674430351dabb6cd67af5e41a215f883d09b9702edd3d36b3d1d51260"
)
SOURCE_CLOSURE_RESULT_MANIFEST_SHA256 = (
    "baed34a16dca206536b2e9cd221fd9f7556f4c063f85ee857352522e770844f4"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-observed-deck-twirl-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "9c397fe42b6bdcb1952d9ed7a5865889e0f919bc5dd0bffce1cb0ef56e484030"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = closure.REGIMES
CUTS = closure.CUTS
TRANSITIONS = closure.TRANSITIONS


@dataclass(frozen=True)
class ObservedDeckTwirlConfig:
    source_closure_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
        "20260810_d15_preregistered"
    )
    calibrated_source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    accuracy_loss_ceiling: float = 0.03
    circular_error_increase_ceiling: float = math.pi / 16.0
    cross_entropy_increase_ceiling: float = 0.10
    required_seed_passes: int = 4
    maximum_control_seed_passes: int = 1
    replay_tolerance: float = 2e-6
    source_metric_tolerance: float = 2e-6
    action_involution_tolerance: float = 2e-6
    calibration_involution_tolerance: float = 1e-7
    target_cosine_tolerance: float = 1e-7
    corrected_norm_tolerance: float = 1e-6
    analytic_feature_tolerance: float = 1e-6
    transformed_planar_limit: float = 2.0
    minimum_oracle_mate_relative_rms: float = 0.5
    batch_size: int = 256
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.conditions or set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown or empty structured condition")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("checkpoint seeds must be non-empty and distinct")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed passes is outside selected population")
        if not 0 <= self.maximum_control_seed_passes <= len(self.seeds):
            raise ValueError("control ceiling is outside selected population")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if not self.allow_underpowered:
            if self.conditions != CONDITIONS or self.seeds != SEEDS:
                raise ValueError("primary conditions and five checkpoints are fixed")
            if self.required_seed_passes != 4:
                raise ValueError("primary population gate is four of five")
            if self.maximum_control_seed_passes != 1:
                raise ValueError("primary control ceiling is one of five")
            if self.batch_size != 256:
                raise ValueError("primary continuation batch size is fixed")


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


def _json_config(config: ObservedDeckTwirlConfig) -> dict[str, Any]:
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
        raise ValueError("observed-deck preregistration changed")
    if closure._implementation_digest() != SOURCE_CLOSURE_IMPLEMENTATION_SHA256:
        raise ValueError("causal-closure source implementation changed")
    paths = {
        "runner": Path(__file__),
        "causal_closure": Path(closure.__file__),
        "calibrated_frontend": Path(calibrated.__file__),
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


def _load_sources(
    config: ObservedDeckTwirlConfig,
) -> tuple[
    dict[str, Any],
    Path,
    CircleTaskConfig,
    calibrated.CalibratedFrontendConfig,
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    closure.CausalClosureConfig,
]:
    root = Path(config.source_closure_root)
    campaign_path = root / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_CLOSURE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SOURCE_CLOSURE_SCHEMA
        or campaign.get("hypothesis_id") != closure.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != SOURCE_CLOSURE_IMPLEMENTATION_SHA256
        or campaign.get("result_manifest_sha256")
        != SOURCE_CLOSURE_RESULT_MANIFEST_SHA256
        or campaign.get("aggregates", {}).get("classification")
        != "frontend_causal_quotient_closed"
        or campaign.get("aggregates", {}).get("primary_hypothesis_pass") is not True
        or campaign.get("aggregates", {}).get("valid") is not True
        or campaign.get("summary", {}).get("completed") != 15
        or campaign.get("summary", {}).get("failed") != 0
        or campaign.get("summary", {}).get("fitted_parameters") != 0
    ):
        raise ValueError(f"invalid causal-closure source campaign {campaign_path}")
    entries = campaign.get("results", [])
    expected = {
        (condition, seed)
        for condition in closure.CONDITIONS
        for seed in closure.SEEDS
    }
    indexed = {
        (entry.get("condition"), int(entry.get("seed", -1))): entry
        for entry in entries
    }
    if set(indexed) != expected:
        raise ValueError("causal-closure source result index changed")
    current_details: dict[tuple[str, int], dict[str, Any]] = {}
    for key, entry in indexed.items():
        result_path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            _sha256(result_path) != entry.get("result_sha256")
            or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
            or detail.get("status") != "completed"
            or detail.get("condition") != key[0]
            or int(detail.get("seed", -1)) != key[1]
            or detail.get("gates", {}).get("validity") is not True
        ):
            raise ValueError(f"invalid causal-closure source result {result_path}")
        if key[0] in config.conditions and key[1] in config.seeds:
            detail["_result_path"] = str(result_path)
            detail["_result_sha256"] = entry["result_sha256"]
            detail["_diagnostics_path"] = str(diagnostics_path)
            detail["_diagnostics_sha256"] = entry["diagnostics_sha256"]
            current_details[key] = detail

    load_config = closure.CausalClosureConfig(
        source_root=config.calibrated_source_root,
        conditions=config.conditions,
        seeds=config.seeds,
        accuracy_loss_ceiling=config.accuracy_loss_ceiling,
        circular_error_increase_ceiling=config.circular_error_increase_ceiling,
        cross_entropy_increase_ceiling=config.cross_entropy_increase_ceiling,
        required_seed_passes=config.required_seed_passes,
        replay_tolerance=config.replay_tolerance,
        batch_size=config.batch_size,
        device="cpu",
        allow_underpowered=True,
    )
    _, _, task, source_config, original_details = closure._load_source_campaign(
        load_config
    )
    if campaign.get("task_config") != asdict(task):
        raise ValueError("causal-closure task configuration changed")
    if campaign.get("dataset_hashes") != closure.EXPECTED_DATASET_HASHES:
        raise ValueError("causal-closure cohort hashes changed")
    return (
        campaign,
        campaign_path,
        task,
        source_config,
        original_details,
        current_details,
        load_config,
    )


def observed_deck_action(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    task: CircleTaskConfig,
    *,
    orthogonal_axis: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the declared action using only decoded planar data and calibration."""
    history = (
        torch.arange(task.sensor_steps, dtype=sensor.dtype, device=sensor.device)
        / max(1.0, float(task.sensor_steps - 1))
        - 1.0
    )
    axis = calibration[:, :2]
    if orthogonal_axis:
        axis = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
    amplitude = calibration[:, 3:4]
    offset = calibration[:, 4:6]
    drift = calibration[:, 6:8]
    corrected = (
        sensor[..., :2]
        - offset[:, None, :]
        - drift[:, None, :] * history[None, :, None]
    ) / amplitude[:, None, :]
    reflected = (
        2.0
        * (corrected * axis[:, None, :]).sum(-1, keepdim=True)
        * axis[:, None, :]
        - corrected
    )
    transformed = sensor.clone()
    transformed[..., :2] = (
        amplitude[:, None, :] * reflected
        + offset[:, None, :]
        + drift[:, None, :] * history[None, :, None]
    )
    transformed_calibration = calibration.clone()
    transformed_calibration[:, 2] = -transformed_calibration[:, 2]
    return transformed, transformed_calibration


def _corrected_planar(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    task: CircleTaskConfig,
) -> torch.Tensor:
    history = (
        torch.arange(task.sensor_steps, dtype=sensor.dtype)
        / max(1.0, float(task.sensor_steps - 1))
        - 1.0
    )
    return (
        sensor[..., :2]
        - calibration[:, None, 4:6]
        - calibration[:, None, 6:8] * history[None, :, None]
    ) / calibration[:, None, 3:4]


def action_contract(
    datasets: Mapping[str, calibrated.CalibratedDataset],
    task: CircleTaskConfig,
    config: ObservedDeckTwirlConfig,
) -> dict[str, Any]:
    regimes = {}
    for regime in REGIMES:
        dataset = datasets[regime]
        sensor = calibrated.decode_sensor_tokens(
            dataset.paired.circle.input_ids, task
        )
        transformed, transformed_calibration = observed_deck_action(
            sensor, dataset.calibration, task
        )
        restored, restored_calibration = observed_deck_action(
            transformed, transformed_calibration, task
        )
        control, control_calibration = observed_deck_action(
            sensor, dataset.calibration, task, orthogonal_axis=True
        )
        control_restored, control_calibration_restored = observed_deck_action(
            control, control_calibration, task, orthogonal_axis=True
        )
        future = (
            dataset.paired.circle.phases.double()
            + dataset.paired.circle.directions.double() * task.future_delta
        )
        transformed_future = (
            -dataset.paired.circle.phases.double()
            - dataset.paired.circle.directions.double() * task.future_delta
        )
        target_error = float(
            (torch.cos(future) - torch.cos(transformed_future)).abs().max()
        )
        analytic = calibrated.AnalyticCalibratedCanonicalizer(task)
        feature_error = float(
            (
                analytic(sensor, dataset.calibration)
                - analytic(transformed, transformed_calibration)
            )
            .abs()
            .max()
        )
        corrected = _corrected_planar(sensor, dataset.calibration, task)
        transformed_corrected = _corrected_planar(
            transformed, transformed_calibration, task
        )
        _, inverse = closure._fiber_mapping(dataset)
        mate = torch.empty(len(sensor), dtype=torch.long)
        for index in range(int(inverse.max()) + 1):
            rows = torch.nonzero(inverse == index, as_tuple=False).flatten()
            mate[rows[0]], mate[rows[1]] = rows[1], rows[0]
        mate_rms = torch.sqrt(
            torch.mean(
                (transformed[..., :2] - sensor[mate, ..., :2])
                .double()
                .square()
            )
        )
        state_rms = torch.sqrt(torch.mean(sensor[..., :2].double().square()))
        values = {
            "sensor_involution_maximum_error": float(
                (restored - sensor).abs().max()
            ),
            "calibration_involution_maximum_error": float(
                (restored_calibration - dataset.calibration).abs().max()
            ),
            "control_sensor_involution_maximum_error": float(
                (control_restored - sensor).abs().max()
            ),
            "control_calibration_involution_maximum_error": float(
                (control_calibration_restored - dataset.calibration).abs().max()
            ),
            "target_cosine_maximum_error": target_error,
            "analytic_feature_maximum_error": feature_error,
            "corrected_norm_maximum_error": float(
                (
                    corrected.norm(dim=-1)
                    - transformed_corrected.norm(dim=-1)
                )
                .abs()
                .max()
            ),
            "transformed_planar_maximum_absolute_value": float(
                torch.maximum(
                    transformed[..., :2].abs().max(),
                    control[..., :2].abs().max(),
                )
            ),
            "oracle_mate_relative_rms": float(mate_rms / state_rms),
        }
        values["pass"] = bool(
            values["sensor_involution_maximum_error"]
            <= config.action_involution_tolerance
            and values["calibration_involution_maximum_error"]
            <= config.calibration_involution_tolerance
            and values["control_sensor_involution_maximum_error"]
            <= config.action_involution_tolerance
            and values["control_calibration_involution_maximum_error"]
            <= config.calibration_involution_tolerance
            and values["target_cosine_maximum_error"]
            <= config.target_cosine_tolerance
            and values["analytic_feature_maximum_error"]
            <= config.analytic_feature_tolerance
            and values["corrected_norm_maximum_error"]
            <= config.corrected_norm_tolerance
            and values["transformed_planar_maximum_absolute_value"]
            <= config.transformed_planar_limit
            and values["oracle_mate_relative_rms"]
            >= config.minimum_oracle_mate_relative_rms
        )
        regimes[regime] = values
    return {
        "construction_inputs": [
            "decoded_planar_history",
            "calibration_orientation",
            "calibration_signed_speed",
            "calibration_amplitude",
            "calibration_offset",
            "calibration_drift",
            "fixed_history_grid",
        ],
        "forbidden_inputs": [
            "latent_phase",
            "target_posterior",
            "target_bin",
            "branch",
            "fiber_id",
            "independent_nuisance_draw",
        ],
        "regimes": regimes,
        "pass": all(item["pass"] for item in regimes.values()),
    }


@torch.no_grad()
def _capture_observed(
    system: calibrated.CalibratedTinyLLM,
    input_ids: torch.Tensor,
    sensor: torch.Tensor,
    calibration_packet: torch.Tensor,
    task: CircleTaskConfig,
    config: ObservedDeckTwirlConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    captured = {cut: [] for cut in CUTS}
    posteriors = []
    for start in range(0, len(sensor), config.batch_size):
        stop = min(len(sensor), start + config.batch_size)
        value = closure._initial_sequence(
            system,
            input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            calibration_packet[start:stop].to(device),
        )
        captured["pre_block"].append(value.cpu())
        value = closure._apply_attention(system, value)
        captured["block0_post_attention"].append(value.cpu())
        value = closure._apply_mlp(system, value)
        captured["block0_post_mlp"].append(value.cpu())
        for block in system.model.transformer["h"][1:]:
            value = block(value)
        captured["full"].append(value.cpu())
        logits = calibrated._task_logits(system.model, value[:, -1], answer_ids)
        posteriors.append(torch.softmax(logits, -1).double().cpu())
    return (
        {cut: torch.cat(parts).float() for cut, parts in captured.items()},
        torch.cat(posteriors),
    )


def _state_geometry(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = left.double() - right.double()
    state_rms = torch.sqrt(torch.mean(left.double().square())).clamp_min(1e-12)
    difference_rms = torch.sqrt(torch.mean(difference.square()))
    return {
        "difference_rms": float(difference_rms),
        "difference_relative_rms": float(difference_rms / state_rms),
        "maximum_absolute_difference": float(difference.abs().max()),
    }


@torch.no_grad()
def analyze_regime(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    config: ObservedDeckTwirlConfig,
    device: torch.device,
    oracle_source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    input_ids = dataset.paired.circle.input_ids
    sensor = calibrated.decode_sensor_tokens(input_ids, task)
    correct_sensor, correct_calibration = observed_deck_action(
        sensor, dataset.calibration, task
    )
    control_sensor, control_calibration = observed_deck_action(
        sensor, dataset.calibration, task, orthogonal_axis=True
    )
    identity, baseline_posterior = _capture_observed(
        system,
        input_ids,
        sensor,
        dataset.calibration,
        task,
        config,
        device,
    )
    correct, correct_posterior = _capture_observed(
        system,
        input_ids,
        correct_sensor,
        correct_calibration,
        task,
        config,
        device,
    )
    control, control_posterior = _capture_observed(
        system,
        input_ids,
        control_sensor,
        control_calibration,
        task,
        config,
        device,
    )
    baseline_metrics = closure.posterior_metrics(baseline_posterior, dataset)
    cut_records: dict[str, Any] = {}
    posterior_records: dict[str, dict[str, torch.Tensor]] = {}
    arrays: dict[str, np.ndarray] = {
        "baseline_posterior": baseline_posterior.float().numpy(),
        "correct_action_full_posterior": correct_posterior.float().numpy(),
        "control_action_full_posterior": control_posterior.float().numpy(),
    }
    maximum_replay_error = 0.0
    for cut in CUTS:
        identity_replay = closure._continue_from_cut(
            system, cut, identity[cut], task, config, device
        )
        correct_replay = closure._continue_from_cut(
            system, cut, correct[cut], task, config, device
        )
        control_replay = closure._continue_from_cut(
            system, cut, control[cut], task, config, device
        )
        replay_error = max(
            float((identity_replay - baseline_posterior).abs().max()),
            float((correct_replay - correct_posterior).abs().max()),
            float((control_replay - control_posterior).abs().max()),
        )
        maximum_replay_error = max(maximum_replay_error, replay_error)
        correct_twirl = 0.5 * (identity[cut] + correct[cut])
        control_twirl = 0.5 * (identity[cut] + control[cut])
        correct_twirl_posterior = closure._continue_from_cut(
            system, cut, correct_twirl, task, config, device
        )
        control_twirl_posterior = closure._continue_from_cut(
            system, cut, control_twirl, task, config, device
        )
        variants = {
            "correct_action": correct_replay,
            "correct_twirl": correct_twirl_posterior,
            "orthogonal_action": control_replay,
            "orthogonal_twirl": control_twirl_posterior,
        }
        records = {}
        for name, posterior in variants.items():
            metrics = closure.posterior_metrics(posterior, dataset)
            passed, gate = closure.task_sufficiency(metrics, baseline_metrics, config)
            records[name] = {
                "task_metrics": metrics,
                "task_sufficiency": gate,
                "task_gate": passed,
                "posterior_js_from_baseline": closure.jensen_shannon(
                    posterior, baseline_posterior
                ),
            }
            arrays[f"{cut}__{name}_posterior"] = posterior.float().numpy()
        cut_records[cut] = {
            "identity_vs_correct_action_geometry": _state_geometry(
                identity[cut], correct[cut]
            ),
            "identity_vs_orthogonal_action_geometry": _state_geometry(
                identity[cut], control[cut]
            ),
            "replay_maximum_absolute_posterior_error": replay_error,
            **records,
            "oracle_independent_nuisance_barycenter": oracle_source["cuts"][cut][
                "orbit_average"
            ],
        }
        posterior_records[cut] = {
            "correct_twirl": correct_twirl_posterior,
            "orthogonal_twirl": control_twirl_posterior,
        }

    correct_pre = 0.5 * (identity["pre_block"] + correct["pre_block"])
    propagated_attention = closure._apply_attention(
        system, correct_pre.to(device)
    ).cpu()
    actual_attention = 0.5 * (
        identity["block0_post_attention"] + correct["block0_post_attention"]
    )
    propagated_mlp = closure._apply_mlp(
        system, actual_attention.to(device)
    ).cpu()
    actual_mlp = 0.5 * (
        identity["block0_post_mlp"] + correct["block0_post_mlp"]
    )
    transitions = {
        "block0_attention": {
            "propagated_pass": cut_records["pre_block"]["correct_twirl"]["task_gate"],
            "actual_pass": cut_records["block0_post_attention"]["correct_twirl"]["task_gate"],
            "causal_regime": closure._regime_name(
                cut_records["pre_block"]["correct_twirl"]["task_gate"],
                cut_records["block0_post_attention"]["correct_twirl"]["task_gate"],
            ),
            "defect_geometry": _state_geometry(actual_attention, propagated_attention),
            "posterior_js_actual_vs_propagated": closure.jensen_shannon(
                posterior_records["block0_post_attention"]["correct_twirl"],
                posterior_records["pre_block"]["correct_twirl"],
            ),
        },
        "block0_mlp": {
            "propagated_pass": cut_records["block0_post_attention"]["correct_twirl"]["task_gate"],
            "actual_pass": cut_records["block0_post_mlp"]["correct_twirl"]["task_gate"],
            "causal_regime": closure._regime_name(
                cut_records["block0_post_attention"]["correct_twirl"]["task_gate"],
                cut_records["block0_post_mlp"]["correct_twirl"]["task_gate"],
            ),
            "defect_geometry": _state_geometry(actual_mlp, propagated_mlp),
            "posterior_js_actual_vs_propagated": closure.jensen_shannon(
                posterior_records["block0_post_mlp"]["correct_twirl"],
                posterior_records["block0_post_attention"]["correct_twirl"],
            ),
        },
    }
    del identity, correct, control
    return {
        "baseline_task_metrics": baseline_metrics,
        "cuts": cut_records,
        "transitions": transitions,
        "maximum_replay_error": maximum_replay_error,
    }, arrays


def classify_campaign(
    *,
    valid: bool,
    twirl_pre_counts: Mapping[str, int],
    action_pre_counts: Mapping[str, int],
    config: ObservedDeckTwirlConfig,
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    required = config.required_seed_passes
    analytic_twirl = twirl_pre_counts.get("analytic_calibrated", 0) >= required
    learned_twirl = (
        twirl_pre_counts.get("learned_calibrated_equivariant", 0) >= required
    )
    analytic_action = action_pre_counts.get("analytic_calibrated", 0) >= required
    learned_action = (
        action_pre_counts.get("learned_calibrated_equivariant", 0) >= required
    )
    if analytic_twirl and learned_twirl and analytic_action and learned_action:
        return "observable_twirl_closed_action_invariant", True
    if analytic_twirl and learned_twirl:
        return "observable_twirl_closed_action_variant", True
    if analytic_twirl and not learned_twirl:
        return "analytic_only_observable_twirl", False
    if learned_twirl and not analytic_twirl:
        return "learned_only_observable_twirl", False
    return "observable_twirl_not_causally_sufficient", False


def _fingerprint(
    config: ObservedDeckTwirlConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    closure_source: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    contract_sha256: str,
) -> str:
    material = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "condition": condition,
        "seed": seed,
        "calibrated_source": dict(provenance),
        "closure_source_result_sha256": closure_source["_result_sha256"],
        "closure_source_diagnostics_sha256": closure_source[
            "_diagnostics_sha256"
        ],
        "dataset_hashes": dict(dataset_hashes),
        "action_contract_sha256": contract_sha256,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
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
        raise ValueError(f"incompatible completed observed-deck result {path}")
    return value


def _campaign_reusable(
    campaign: Mapping[str, Any],
    config: ObservedDeckTwirlConfig,
    implementation: str,
) -> bool:
    entries = campaign.get("results", [])
    expected = {
        (condition, seed)
        for condition in config.conditions
        for seed in config.seeds
    }
    observed = {
        (entry.get("condition"), entry.get("seed")) for entry in entries
    }
    return bool(
        campaign.get("status") == "completed"
        and campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("configuration") == _json_config(config)
        and campaign.get("implementation_sha256") == implementation
        and len(entries) == len(expected)
        and observed == expected
        and all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry["result_sha256"]
            and Path(entry["diagnostics_path"]).is_file()
            and _sha256(Path(entry["diagnostics_path"]))
            == entry["diagnostics_sha256"]
            for entry in entries
        )
    )


def run_campaign(
    config: ObservedDeckTwirlConfig, output: Path
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

    (
        source_closure,
        source_closure_path,
        task,
        source_config,
        original_details,
        closure_details,
        load_config,
    ) = _load_sources(config)
    datasets = closure._datasets(task)
    dataset_hashes = {
        regime: closure._dataset_hash(dataset)
        for regime, dataset in datasets.items()
    }
    if dataset_hashes != closure.EXPECTED_DATASET_HASHES:
        raise ValueError("observed-deck held-out cohorts changed")
    contract = action_contract(datasets, task, config)
    if not contract["pass"]:
        raise ValueError(f"observed-deck action contract failed: {contract}")
    contract_sha256 = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_root = Path(config.calibrated_source_root)
    preflight, preflight_manifest = closure._preflight_sources(
        load_config,
        source_root,
        task,
        source_config,
        original_details,
    )
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    results: list[dict[str, Any]] = []
    reused = 0
    for condition in config.conditions:
        for seed in config.seeds:
            cell_started = time.perf_counter()
            system, provenance = closure._load_system(
                source_root,
                condition,
                seed,
                task,
                source_config,
                original_details[(condition, seed)],
                device,
            )
            if provenance != preflight[(condition, seed)]:
                raise ValueError(
                    f"source changed after preflight for {condition} seed {seed}"
                )
            for parameter in system.parameters():
                parameter.requires_grad_(False)
            closure_source = closure_details[(condition, seed)]
            fingerprint = _fingerprint(
                config,
                implementation,
                condition,
                seed,
                provenance,
                closure_source,
                dataset_hashes,
                contract_sha256,
            )
            result_dir = output / "runs" / condition / f"seed_{seed}"
            result_path = result_dir / "result.json"
            existing = _reusable_result(result_path, fingerprint, implementation)
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"resuming {condition} seed {seed}", flush=True)
                del system
                continue

            regime_results = {}
            diagnostic_arrays: dict[str, np.ndarray] = {}
            for regime in REGIMES:
                regime_result, arrays = analyze_regime(
                    system,
                    datasets[regime],
                    task,
                    config,
                    device,
                    closure_source["regimes"][regime],
                )
                source_metrics = closure_source["regimes"][regime][
                    "baseline_task_metrics"
                ]
                source_error = max(
                    abs(
                        float(regime_result["baseline_task_metrics"][metric])
                        - float(source_metrics[metric])
                    )
                    for metric in (
                        "exact_bin_accuracy",
                        "mean_circular_error_radians",
                        "mean_target_cross_entropy",
                    )
                )
                regime_result["source_task_replay"] = {
                    "maximum_absolute_error": source_error,
                    "pass": source_error <= config.source_metric_tolerance,
                }
                regime_results[regime] = regime_result
                diagnostic_arrays.update(
                    {f"{regime}__{name}": value for name, value in arrays.items()}
                )

            diagnostics_path = result_dir / "observed_deck_diagnostics.npz"
            _write_npz(diagnostics_path, diagnostic_arrays)
            diagnostics_sha256 = _sha256(diagnostics_path)
            state_unchanged = bool(
                calibrated._state_digest(system.model)
                == provenance["model_state_sha256"]
                and calibrated._module_digest(system)
                == provenance["system_state_sha256"]
            )
            replay_pass = all(
                regime_results[regime]["maximum_replay_error"]
                <= config.replay_tolerance
                and regime_results[regime]["source_task_replay"]["pass"]
                for regime in REGIMES
            )
            finite = _finite(regime_results)
            validity = bool(state_unchanged and replay_pass and finite)
            twirl_seed_gates = {
                cut: all(
                    regime_results[regime]["cuts"][cut]["correct_twirl"][
                        "task_gate"
                    ]
                    for regime in REGIMES
                )
                for cut in CUTS
            }
            action_seed_gates = {
                cut: all(
                    regime_results[regime]["cuts"][cut]["correct_action"][
                        "task_gate"
                    ]
                    for regime in REGIMES
                )
                for cut in CUTS
            }
            control_twirl_seed_gates = {
                cut: all(
                    regime_results[regime]["cuts"][cut]["orthogonal_twirl"][
                        "task_gate"
                    ]
                    for regime in REGIMES
                )
                for cut in CUTS
            }
            control_action_seed_gates = {
                cut: all(
                    regime_results[regime]["cuts"][cut]["orthogonal_action"][
                        "task_gate"
                    ]
                    for regime in REGIMES
                )
                for cut in CUTS
            }
            transition_seed_gates = {
                transition: all(
                    regime_results[regime]["transitions"][transition][
                        "causal_regime"
                    ]
                    == "quotient_already_closed"
                    for regime in REGIMES
                )
                for transition in TRANSITIONS
            }
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-observed-deck-{condition}-seed{seed}",
                "status": "completed",
                "evidence_role": EVIDENCE_ROLE,
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": _json_config(config),
                "implementation_sha256": implementation,
                "scientific_fingerprint": fingerprint,
                "provenance": {
                    **provenance,
                    "closure_source_result": closure_source["_result_path"],
                    "closure_source_result_sha256": closure_source[
                        "_result_sha256"
                    ],
                    "closure_source_diagnostics": closure_source[
                        "_diagnostics_path"
                    ],
                    "closure_source_diagnostics_sha256": closure_source[
                        "_diagnostics_sha256"
                    ],
                },
                "dataset_hashes": dataset_hashes,
                "action_contract_sha256": contract_sha256,
                "regimes": regime_results,
                "twirl_seed_gates": twirl_seed_gates,
                "action_seed_gates": action_seed_gates,
                "control_twirl_seed_gates": control_twirl_seed_gates,
                "control_action_seed_gates": control_action_seed_gates,
                "transition_seed_gates": transition_seed_gates,
                "gates": {
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
                f"{condition} seed {seed}: "
                f"twirl={twirl_seed_gates['pre_block']} "
                f"action={action_seed_gates['pre_block']} "
                f"control={control_twirl_seed_gates['pre_block']} "
                f"valid={validity}",
                flush=True,
            )
            del system
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("observed-deck implementation changed during run")

    arms = {}
    for condition in config.conditions:
        selected = [item for item in results if item["condition"] == condition]
        arms[condition] = {
            "twirl_pass_counts": {
                cut: sum(int(item["twirl_seed_gates"][cut]) for item in selected)
                for cut in CUTS
            },
            "action_pass_counts": {
                cut: sum(int(item["action_seed_gates"][cut]) for item in selected)
                for cut in CUTS
            },
            "control_twirl_pass_counts": {
                cut: sum(
                    int(item["control_twirl_seed_gates"][cut])
                    for item in selected
                )
                for cut in CUTS
            },
            "control_action_pass_counts": {
                cut: sum(
                    int(item["control_action_seed_gates"][cut])
                    for item in selected
                )
                for cut in CUTS
            },
            "transition_closed_counts": {
                transition: sum(
                    int(item["transition_seed_gates"][transition])
                    for item in selected
                )
                for transition in TRANSITIONS
            },
        }
    controls_pass = all(
        arms[condition]["control_twirl_pass_counts"]["pre_block"]
        <= config.maximum_control_seed_passes
        for condition in config.conditions
    )
    valid = bool(
        all(item["gates"]["validity"] for item in results)
        and contract["pass"]
        and controls_pass
    )
    if config.allow_underpowered:
        classification, primary_pass = (
            "systems_lifecycle_only_not_quality_evidence",
            False,
        )
    else:
        classification, primary_pass = classify_campaign(
            valid=valid,
            twirl_pre_counts={
                condition: arms[condition]["twirl_pass_counts"]["pre_block"]
                for condition in config.conditions
            },
            action_pre_counts={
                condition: arms[condition]["action_pass_counts"]["pre_block"]
                for condition in config.conditions
            },
            config=config,
        )
    result_entries = [
        {
            "experiment_id": item["experiment_id"],
            "condition": item["condition"],
            "seed": item["seed"],
            "scientific_fingerprint": item["scientific_fingerprint"],
            "path": item["artifacts"]["result"],
            "result_sha256": _sha256(Path(item["artifacts"]["result"])),
            "diagnostics_path": item["artifacts"]["diagnostics"],
            "diagnostics_sha256": item["artifacts"]["diagnostics_sha256"],
        }
        for item in results
    ]
    result_manifest_sha256 = hashlib.sha256(
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
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "provenance": {
            "source_closure_campaign": str(source_closure_path),
            "source_closure_campaign_sha256": SOURCE_CLOSURE_CAMPAIGN_SHA256,
            "source_closure_implementation_sha256": (
                SOURCE_CLOSURE_IMPLEMENTATION_SHA256
            ),
            "source_closure_result_manifest_sha256": (
                SOURCE_CLOSURE_RESULT_MANIFEST_SHA256
            ),
            "source_preflight_manifest_sha256": preflight_manifest,
            "source_preflight_completed_before_interventions": True,
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
        "task_config": asdict(task),
        "dataset_hashes": dataset_hashes,
        "action_contract": contract,
        "action_contract_sha256": contract_sha256,
        "summary": {
            "requested": len(config.conditions) * len(config.seeds),
            "scheduled": len(config.conditions) * len(config.seeds) - reused,
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
            "controls_pass": controls_pass,
            "required_seed_passes": config.required_seed_passes,
            "maximum_control_seed_passes": config.maximum_control_seed_passes,
            "arms": arms,
        },
        "results": result_entries,
        "result_manifest_sha256": result_manifest_sha256,
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The action reads only decoded structured planar history, observed calibration, and the fixed time grid.",
            "Latent phase, target, branch, fiber ID, and independent nuisance pairs are forbidden intervention inputs.",
            "The transform is applied after token decoding and is not re-quantized.",
            "The raw three-channel token model is outside the declared action domain.",
            "TinyLLM, front ends, embeddings, answer rows, probes, and observers remain frozen.",
            "Five retained checkpoints do not establish architecture-population prevalence.",
        ],
        "artifacts": {"campaign": str(campaign_path)},
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_observed_deck_twirl/"
            "20260810_d10_preregistered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--conditions", type=_strings, default=CONDITIONS)
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--required-seed-passes", type=int, default=4)
    parser.add_argument("--maximum-control-seed-passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = ObservedDeckTwirlConfig(
        conditions=args.conditions,
        seeds=args.seeds,
        required_seed_passes=args.required_seed_passes,
        maximum_control_seed_passes=args.maximum_control_seed_passes,
        batch_size=args.batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
