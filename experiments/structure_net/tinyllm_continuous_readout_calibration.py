#!/usr/bin/env python3
"""Separate TinyLLM continuous C2 carrier rank from discrete bin calibration."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import torch

import experiments.structure_net.tinyllm_defect_boundary_audit as boundary
import experiments.structure_net.tinyllm_defect_boundary_basis as boundary_basis
import experiments.structure_net.tinyllm_defect_subspace_rank as rank
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-continuous-readout-calibration.v1.1"
HYPOTHESIS_ID = "tinyllm-c2-semantic-carrier-readout-separation-v1"
REGIMES = ("composition", "extrapolation")
HELDOUT_COHORTS = ("heldout_a", "heldout_b")
PRIMARY_SEEDS = (7, 29, 53)
RANKS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ContinuousReadoutCalibrationConfig:
    source_root: str = rank.DefectSubspaceRankConfig.source_root
    character_root: str = rank.DefectSubspaceRankConfig.character_root
    rank_root: str = "data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered"
    boundary_root: str = "data/experiments/tinyllm_defect_boundary_audit/20260806_d6_preregistered_v3"
    boundary_basis_root: str = "data/experiments/tinyllm_defect_boundary_basis/20260806_d6_preregistered"
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    first_blocks: int = 6
    ranks: tuple[int, ...] = RANKS
    alignment_loss_ceiling: float = 0.005
    mean_shift_ceiling_bins: float = 0.125
    p95_shift_ceiling_bins: float = 0.50
    winding_tolerance: float = 0.10
    accuracy_loss_ceiling: float = 0.03
    calibration_grid_points: int = 4097
    task_effect_floor: float = 1e-6
    decomposition_tolerance: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False
    post_outcome_corrective_replication: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("confirmatory semantic-carrier seeds are fixed to 7,29,53")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if tuple(sorted(set(self.ranks))) != self.ranks or self.ranks[0] != 1:
            raise ValueError("ranks must be unique, increasing, and start at one")
        if self.ranks[-1] < 5 and not self.allow_underpowered:
            raise ValueError("the confirmatory rank grid must include rank five")
        if self.calibration_grid_points < 3 or self.calibration_grid_points % 2 == 0:
            raise ValueError("calibration grid must have an odd number of at least three points")
        if min(
            self.alignment_loss_ceiling,
            self.mean_shift_ceiling_bins,
            self.p95_shift_ceiling_bins,
            self.winding_tolerance,
            self.accuracy_loss_ceiling,
        ) <= 0:
            raise ValueError("endpoint thresholds must be positive")
        if self.allow_underpowered and self.post_outcome_corrective_replication:
            raise ValueError(
                "systems-only and post-outcome corrective evidence roles are exclusive"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _json_compatible(value: Any) -> Any:
    """Return the strict-JSON representation used by persisted artifacts."""
    return json.loads(json.dumps(value, allow_nan=False))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(rank.__file__), Path(boundary.__file__), Path(boundary_basis.__file__)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: ContinuousReadoutCalibrationConfig) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    if config.post_outcome_corrective_replication:
        return "post_outcome_corrective_replication_evidence"
    return "preregistered_underpowered_mechanistic_evidence"


def posterior_moment(posterior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return circular moment angle and magnitude for categorical posteriors."""
    posterior = posterior.double()
    angles = torch.arange(posterior.shape[-1], device=posterior.device, dtype=torch.float64)
    angles = angles * (2.0 * math.pi / posterior.shape[-1])
    real = posterior @ torch.cos(angles)
    imaginary = posterior @ torch.sin(angles)
    return torch.atan2(imaginary, real), torch.sqrt(real.square() + imaginary.square())


def _wrapped_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(left - right), torch.cos(left - right))


def continuous_metrics(
    posterior: torch.Tensor,
    exact: torch.Tensor,
    diagnostics: Mapping[str, Any],
    exact_diagnostics: Mapping[str, Any],
    config: ContinuousReadoutCalibrationConfig,
) -> dict[str, Any]:
    angle, magnitude = posterior_moment(posterior)
    exact_angle, _ = posterior_moment(exact)
    width = 2.0 * math.pi / posterior.shape[-1]
    displacement = _wrapped_difference(angle, exact_angle).abs() / width
    alignment_loss = float(exact_diagnostics["circular_alignment"] - diagnostics["circular_alignment"])
    passed = bool(
        alignment_loss <= config.alignment_loss_ceiling
        and float(displacement.mean()) <= config.mean_shift_ceiling_bins
        and float(torch.quantile(displacement, 0.95)) <= config.p95_shift_ceiling_bins
        and diagnostics["sampling_resolved"]
        and abs(float(diagnostics["winding_degree"]) - 2.0) <= config.winding_tolerance
    )
    return {
        "alignment_loss_from_exact": alignment_loss,
        "mean_moment_shift_bins": float(displacement.mean()),
        "median_moment_shift_bins": float(displacement.median()),
        "p95_moment_shift_bins": float(torch.quantile(displacement, 0.95)),
        "maximum_moment_shift_bins": float(displacement.max()),
        "minimum_moment_magnitude": float(magnitude.min()),
        "continuous_pass": passed,
    }


def quantize_moment_angles(
    angles: torch.Tensor, bins: int, rotation_bins: float
) -> torch.Tensor:
    coordinate = angles.double() / (2.0 * math.pi / bins)
    return torch.floor(torch.remainder(coordinate + 0.5 + rotation_bins, bins)).long()


def fit_boundary_rotation(
    angles: torch.Tensor,
    targets: torch.Tensor,
    bins: int,
    grid_points: int,
) -> dict[str, Any]:
    """Fit one global bin-boundary rotation with deterministic tie breaking."""
    grid = torch.linspace(-0.5, 0.5, grid_points, dtype=torch.float64)
    predictions = quantize_moment_angles(angles[:, None], bins, grid[None, :])
    accuracy = (predictions == targets.long()[:, None]).double().mean(0)
    best = torch.nonzero(accuracy == accuracy.max(), as_tuple=False).flatten()
    candidates = grid[best]
    order = sorted(range(len(best)), key=lambda index: (abs(float(candidates[index])), float(candidates[index])))
    selected = int(best[order[0]])
    return {
        "rotation_bins": float(grid[selected]),
        "source_accuracy": float(accuracy[selected]),
        "best_grid_count": len(best),
    }


def discrete_readout_metrics(
    angles: torch.Tensor,
    targets: torch.Tensor,
    bins: int,
    rotation_bins: float,
    baseline_accuracy: float,
    accuracy_loss_ceiling: float,
) -> dict[str, Any]:
    predictions = quantize_moment_angles(angles, bins, rotation_bins)
    accuracy = float((predictions.cpu() == targets.long().cpu()).double().mean())
    loss = baseline_accuracy - accuracy
    return {
        "rotation_bins": rotation_bins,
        "exact_bin_accuracy": accuracy,
        "accuracy_loss_from_untouched": loss,
        "discrete_pass": bool(loss <= accuracy_loss_ceiling),
    }


def select_source_rank(
    source_cells: Sequence[Mapping[str, Any]], ranks: Sequence[int]
) -> Optional[int]:
    for candidate in ranks:
        if all(cell["ranks"][str(candidate)]["continuous"]["continuous_pass"] for cell in source_cells):
            return int(candidate)
    return None


def _rank_config(config: ContinuousReadoutCalibrationConfig) -> rank.DefectSubspaceRankConfig:
    return rank.DefectSubspaceRankConfig(
        source_root=config.source_root,
        character_root=config.character_root,
        seeds=rank.PRIMARY_SEEDS,
        orbit_count=config.orbit_count,
        first_blocks=config.first_blocks,
        ranks=rank.DEFAULT_RANKS,
        task_effect_floor=config.task_effect_floor,
        decomposition_tolerance=config.decomposition_tolerance,
        activation_batch_size=config.activation_batch_size,
        continuation_batch_size=config.continuation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _load_predecessors(
    config: ContinuousReadoutCalibrationConfig,
    seed: int,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    specifications = (
        (
            "rank",
            Path(config.rank_root),
            (rank.SCHEMA_VERSION,),
            rank.HYPOTHESIS_ID,
            "preregistered_primary_evidence",
        ),
        (
            "boundary",
            Path(config.boundary_root),
            (boundary.SCHEMA_VERSION,),
            boundary.HYPOTHESIS_ID,
            "post_outcome_corrective_replication_evidence",
        ),
        (
            "boundary_basis",
            Path(config.boundary_basis_root),
            (boundary_basis.SCHEMA_VERSION,),
            boundary_basis.HYPOTHESIS_ID,
            "preregistered_underpowered_mechanistic_evidence",
        ),
    )
    output = {}
    for name, root, schemas, hypothesis, evidence_role in specifications:
        campaign_path = root / "campaign_results.json"
        campaign = json.loads(campaign_path.read_text())
        result_path = root / "runs" / f"seed_{seed}" / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
        if (
            campaign.get("schema_version") not in schemas
            or campaign.get("hypothesis_id") != hypothesis
            or campaign.get("status") != "completed"
            or campaign.get("evidence_role") != evidence_role
            or not campaign.get("implementation_sha256")
        ):
            raise ValueError(f"invalid {name} predecessor for seed {seed}")
        if result is None and name != "boundary":
            raise ValueError(f"missing {name} predecessor result for seed {seed}")
        if result is not None and (
            result.get("schema_version") not in schemas
            or result.get("hypothesis_id") != hypothesis
            or result.get("status") != "completed"
            or result.get("evidence_role") != evidence_role
            or int(result.get("seed", -1)) != seed
            or result.get("implementation_sha256")
            != campaign.get("implementation_sha256")
            or result.get("provenance", {}).get("checkpoint_sha256")
            != checkpoint_sha256
        ):
            raise ValueError(f"invalid {name} predecessor result for seed {seed}")
        output[name] = {
            "campaign": str(campaign_path),
            "campaign_sha256": _sha256(campaign_path),
            "implementation_sha256": campaign["implementation_sha256"],
            "result": None if result is None else str(result_path),
            "result_sha256": None if result is None else _sha256(result_path),
        }
    return output


@torch.no_grad()
def evaluate_rank_cell(
    system: Any,
    task: CircleTaskConfig,
    config: ContinuousReadoutCalibrationConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    defect = cell["full_defect"]
    propagated = cell["propagated"]
    flat = defect.reshape(config.orbit_count, -1)
    states = {"zero": propagated, "exact": cell["exact"]}
    for candidate in config.ranks:
        projected = rank.project_onto_basis(flat, basis, candidate).reshape_as(defect)
        states[f"rank_{candidate}"] = propagated + projected
    keys = tuple(states)
    repeated = torch.cat([coupling._repeat_patch(states[key], 2) for key in keys])
    combined = coupling._continue_tensor(
        system, cell["target_cut"], repeated, task, config.continuation_batch_size
    )
    per_state = config.orbit_count * 2
    raw = dict(zip(keys, combined.split(per_state)))
    posteriors = {
        key: value.reshape(config.orbit_count, 2, -1)[:, 0]
        for key, value in raw.items()
    }
    repeated_exact = posteriors["exact"][:, None].expand(-1, 2, -1).reshape(per_state, -1)
    exact_diagnostics = coupling._diagnostics(repeated_exact, cell["dataset"])
    ranks = {}
    for candidate in config.ranks:
        posterior = posteriors[f"rank_{candidate}"]
        repeated_posterior = posterior[:, None].expand(-1, 2, -1).reshape(per_state, -1)
        diagnostics = coupling._diagnostics(repeated_posterior, cell["dataset"])
        ranks[str(candidate)] = {
            "diagnostics": diagnostics,
            "continuous": continuous_metrics(
                posterior, posteriors["exact"], diagnostics, exact_diagnostics, config
            ),
            "fisher_effect": rank.heads._effect_preservation(
                posterior,
                posteriors["exact"],
                posteriors["zero"],
                config.task_effect_floor,
            ),
        }
    targets = cell["dataset"].target_bins.reshape(config.orbit_count, 2)[:, 0]
    summary = {
        "cohort": cell["cohort"],
        "regime": cell["regime"],
        "evaluation_seed": cell["evaluation_seed"],
        "baseline": cell["baseline"],
        "decomposition_relative_error": cell["decomposition_relative_error"],
        "exact_diagnostics": exact_diagnostics,
        "ranks": ranks,
    }
    tensors = {**posteriors, "targets": targets}
    return summary, tensors


def _fingerprint(
    config: ContinuousReadoutCalibrationConfig,
    task: CircleTaskConfig,
    seed: int,
    checkpoint_sha: str,
    character_result_sha256: str,
    predecessors: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "character_result_sha256": character_result_sha256,
        "predecessors": predecessors,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def analyze_seed(
    task: CircleTaskConfig,
    config: ContinuousReadoutCalibrationConfig,
    seed: int,
    output: Path,
    device: torch.device,
    implementation_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    rank_config = _rank_config(config)
    bridge = rank._bridge_config(rank_config)
    system, provenance = rank.deck.load_source(task, bridge, 2, seed, device)
    model_parameter_count = sum(parameter.numel() for parameter in system.model.parameters())
    frontend_parameter_count = sum(
        parameter.numel() for parameter in system.vector_embedding.parameters()
    )
    frozen, frozen_path = rank._load_character_source(rank_config, seed, provenance["checkpoint_sha256"])
    character_sha256 = _sha256(frozen_path)
    predecessors = _load_predecessors(
        config, seed, provenance["checkpoint_sha256"]
    )

    source_raw = [
        rank._defect_cell(system, task, rank_config, bridge, frozen, seed, "source_selection", regime, device)
        for regime in REGIMES
    ]
    source_matrix = torch.cat(
        [cell["full_defect"].reshape(config.orbit_count, -1) for cell in source_raw], dim=0
    )
    basis, energy = rank.fit_right_singular_basis(source_matrix, rank_config.numerical_rank_rtol)
    source_pairs = [
        evaluate_rank_cell(system, task, config, cell, basis) for cell in source_raw
    ]
    source_summaries = [item[0] for item in source_pairs]
    source_tensors = [item[1] for item in source_pairs]
    selected_rank = select_source_rank(source_summaries, config.ranks)
    if selected_rank is None:
        selected_rank = config.ranks[-1]
        source_selection_failed = True
    else:
        source_selection_failed = False

    selected_angles = torch.cat(
        [posterior_moment(item[f"rank_{selected_rank}"])[0].cpu() for item in source_tensors]
    )
    exact_angles = torch.cat([posterior_moment(item["exact"])[0].cpu() for item in source_tensors])
    source_targets = torch.cat([item["targets"].cpu() for item in source_tensors])
    bins = len(task.answer_token_ids)
    selected_calibration = fit_boundary_rotation(
        selected_angles, source_targets, bins, config.calibration_grid_points
    )
    exact_calibration = fit_boundary_rotation(
        exact_angles, source_targets, bins, config.calibration_grid_points
    )

    heldout_records = []
    for cohort in HELDOUT_COHORTS:
        for regime in REGIMES:
            raw = rank._defect_cell(system, task, rank_config, bridge, frozen, seed, cohort, regime, device)
            summary, tensors = evaluate_rank_cell(system, task, config, raw, basis)
            selected_posterior = tensors[f"rank_{selected_rank}"]
            selected_angle, _ = posterior_moment(selected_posterior)
            exact_angle, _ = posterior_moment(tensors["exact"])
            targets = tensors["targets"]
            summary["selected_rank"] = selected_rank
            summary["selected_readouts"] = {
                "uncalibrated_moment": discrete_readout_metrics(
                    selected_angle, targets, bins, 0.0,
                    summary["baseline"]["exact_bin_accuracy"], config.accuracy_loss_ceiling,
                ),
                "calibrated_moment": discrete_readout_metrics(
                    selected_angle, targets, bins, selected_calibration["rotation_bins"],
                    summary["baseline"]["exact_bin_accuracy"], config.accuracy_loss_ceiling,
                ),
                "exact_calibrated_control": discrete_readout_metrics(
                    exact_angle, targets, bins, exact_calibration["rotation_bins"],
                    summary["baseline"]["exact_bin_accuracy"], config.accuracy_loss_ceiling,
                ),
            }
            heldout_records.append(summary)
            print(f"seed {seed} {cohort} {regime} complete", flush=True)

    selected_key = str(selected_rank)
    gates = {
        "predecessor_identity": True,
        "decomposition_contract": max(
            [cell["decomposition_relative_error"] for cell in source_summaries + heldout_records]
        ) <= config.decomposition_tolerance,
        "source_rank_selected_without_fallback": not source_selection_failed,
        "selected_rank_at_most_three": selected_rank <= 3,
        "heldout_continuous_sufficiency": all(
            cell["ranks"][selected_key]["continuous"]["continuous_pass"]
            for cell in heldout_records
        ),
        "exact_calibrator_endpoint": all(
            cell["selected_readouts"]["exact_calibrated_control"]["discrete_pass"]
            for cell in heldout_records
        ),
        "selected_calibrator_endpoint": all(
            cell["selected_readouts"]["calibrated_moment"]["discrete_pass"]
            for cell in heldout_records
        ),
    }
    if seed == 29:
        target_cell = next(
            cell for cell in heldout_records
            if cell["cohort"] == "heldout_b" and cell["regime"] == "extrapolation"
        )
        gates["seed29_margin_repair"] = bool(
            target_cell["selected_readouts"]["calibrated_moment"]["exact_bin_accuracy"]
            - target_cell["selected_readouts"]["uncalibrated_moment"]["exact_bin_accuracy"]
            >= 1.0 / config.orbit_count - 1e-12
            and target_cell["selected_readouts"]["calibrated_moment"]["discrete_pass"]
        )
    maximum_heldout_mean_shift = max(
        cell["ranks"][selected_key]["continuous"]["mean_moment_shift_bins"]
        for cell in heldout_records
    )
    maximum_heldout_p95_shift = max(
        cell["ranks"][selected_key]["continuous"]["p95_moment_shift_bins"]
        for cell in heldout_records
    )
    worst_selected_calibrated_loss = max(
        cell["selected_readouts"]["calibrated_moment"][
            "accuracy_loss_from_untouched"
        ]
        for cell in heldout_records
    )
    worst_exact_calibrated_loss = max(
        cell["selected_readouts"]["exact_calibrated_control"][
            "accuracy_loss_from_untouched"
        ]
        for cell in heldout_records
    )
    result_path = output / "runs" / f"seed_{seed}" / "result.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-continuous-readout-calibration-seed{seed}",
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": _fingerprint(
            config,
            task,
            seed,
            provenance["checkpoint_sha256"],
            character_sha256,
            predecessors,
        ),
        "implementation_sha256": implementation_sha256,
        "primary_metric": {
            "name": "maximum_heldout_mean_moment_shift_bins",
            "value": maximum_heldout_mean_shift,
            "direction": "lower_is_better",
            "threshold": config.mean_shift_ceiling_bins,
        },
        "headline_metrics": {
            "selected_source_rank": selected_rank,
            "maximum_heldout_mean_moment_shift_bins": maximum_heldout_mean_shift,
            "maximum_heldout_p95_moment_shift_bins": maximum_heldout_p95_shift,
            "worst_selected_calibrated_accuracy_loss": worst_selected_calibrated_loss,
            "worst_exact_calibrated_accuracy_loss": worst_exact_calibrated_loss,
        },
        "architecture_provenance": {
            "family": "frozen_degree_two_tinyllm",
            "model_parameter_count": model_parameter_count,
            "frontend_parameter_count": frontend_parameter_count,
            "source_result": provenance["source_result"],
            "checkpoint": provenance["checkpoint"],
        },
        "provenance": {
            **provenance,
            "character_result": str(frozen_path),
            "character_result_sha256": character_sha256,
            "predecessors": predecessors,
        },
        "source_basis": {
            "rank": int(basis.shape[0]),
            "features": int(basis.shape[1]),
            "cumulative_energy_first_five": [
                float(value) for value in energy[:5].cpu().cumsum(0)
            ],
        },
        "source_selection": {
            "selected_rank": selected_rank,
            "fallback_used": source_selection_failed,
            "selected_calibration": selected_calibration,
            "exact_calibration": exact_calibration,
            "cells": source_summaries,
        },
        "heldout_cells": heldout_records,
        "gates": gates,
        "all_primary_gates_passed": all(
            gates[name]
            for name in (
                "predecessor_identity",
                "decomposition_contract",
                "source_rank_selected_without_fallback",
                "selected_rank_at_most_three",
                "heldout_continuous_sufficiency",
                "exact_calibrator_endpoint",
                "selected_calibrator_endpoint",
            )
        ),
        "artifacts": {"result": str(result_path)},
        "observations": [
            {
                "kind": "source_selected_rank",
                "value": selected_rank,
                "fallback_used": source_selection_failed,
            },
            {
                "kind": "source_selected_boundary_rotation_bins",
                "value": selected_calibration["rotation_bins"],
            },
        ],
        "errors": [],
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate_names = (
        "predecessor_identity",
        "decomposition_contract",
        "source_rank_selected_without_fallback",
        "selected_rank_at_most_three",
        "heldout_continuous_sufficiency",
        "exact_calibrator_endpoint",
        "selected_calibrator_endpoint",
    )
    counts = {name: sum(bool(run["gates"][name]) for run in runs) for name in gate_names}
    passes = {name: count == len(runs) for name, count in counts.items()}
    seed29 = next((run for run in runs if int(run["seed"]) == 29), None)
    seed29_repair = bool(seed29 and seed29["gates"].get("seed29_margin_repair", False))
    return {
        "primary_seed_count": len(runs),
        "gate_counts": counts,
        "gate_passes": passes,
        "seed29_margin_repair": seed29_repair,
        "selected_ranks": {str(run["seed"]): run["source_selection"]["selected_rank"] for run in runs},
        "calibration_rotations_bins": {
            str(run["seed"]): run["source_selection"]["selected_calibration"]["rotation_bins"]
            for run in runs
        },
        "confirmed": bool(all(passes.values()) and seed29_repair),
        "per_seed": {
            str(run["seed"]): {
                "selected_rank": run["source_selection"]["selected_rank"],
                "selected_rotation_bins": run["source_selection"][
                    "selected_calibration"
                ]["rotation_bins"],
                "headline_metrics": run["headline_metrics"],
                "gates": run["gates"],
            }
            for run in runs
        },
    }


def _campaign_is_reusable(
    campaign: Mapping[str, Any],
    config: ContinuousReadoutCalibrationConfig,
    implementation: str,
    runs: Sequence[Mapping[str, Any]],
) -> bool:
    expected = [
        (
            run["experiment_id"],
            run["scientific_fingerprint"],
            _sha256(Path(run["artifacts"]["result"])),
        )
        for run in runs
    ]
    observed = [
        (
            item.get("experiment_id"),
            item.get("scientific_fingerprint"),
            item.get("result_sha256"),
        )
        for item in campaign.get("results", [])
    ]
    return bool(
        campaign.get("schema_version") == SCHEMA_VERSION
        and campaign.get("hypothesis_id") == HYPOTHESIS_ID
        and campaign.get("status") == "completed"
        and campaign.get("implementation_sha256") == implementation
        and campaign.get("configuration") == _json_compatible(asdict(config))
        and observed == expected
    )


def run_campaign(config: ContinuousReadoutCalibrationConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    implementation = _implementation_digest()
    runs = []
    reused = 0
    for seed in config.seeds:
        path = output / "runs" / f"seed_{seed}" / "result.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            checkpoint = Path(config.source_root) / "runs" / "k2" / f"seed_{seed}" / "model.pt"
            checkpoint_sha256 = _sha256(checkpoint)
            _, character_path = rank._load_character_source(
                _rank_config(config), seed, checkpoint_sha256
            )
            character_sha256 = _sha256(character_path)
            predecessors = _load_predecessors(config, seed, checkpoint_sha256)
            expected = _fingerprint(
                config,
                task,
                seed,
                checkpoint_sha256,
                character_sha256,
                predecessors,
            )
            if (
                existing.get("status") == "completed"
                and existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("hypothesis_id") == HYPOTHESIS_ID
                and int(existing.get("seed", -1)) == seed
                and existing.get("evidence_role") == _evidence_role(config)
                and existing.get("scientific_fingerprint") == expected
                and existing.get("implementation_sha256") == implementation
                and existing.get("provenance", {}).get(
                    "character_result_sha256"
                )
                == character_sha256
            ):
                runs.append(existing)
                reused += 1
                print(f"resuming {existing['experiment_id']}", flush=True)
                continue
            raise ValueError(
                f"incompatible completed result {path}; use a new campaign root"
            )
        runs.append(analyze_seed(task, config, seed, output, device, implementation))
        if _implementation_digest() != implementation:
            raise RuntimeError("analysis implementation changed during campaign")

    campaign_path = output / "campaign_results.json"
    if reused == len(config.seeds) and campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text())
        if _campaign_is_reusable(
            existing_campaign, config, implementation, runs
        ):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing_campaign
        raise ValueError(
            f"incompatible completed aggregate {campaign_path}; use a new campaign root"
        )

    aggregates = aggregate(runs)
    aggregates["conclusion"] = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else (
            "confirmed_semantic_carrier_readout_separation"
            if aggregates["confirmed"]
            else "not_confirmed_semantic_carrier_readout_separation"
        )
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "task_config": asdict(task),
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "summary": {
            "requested": len(config.seeds),
            "scheduled": len(config.seeds) - reused,
            "completed": len(runs),
            "failed": 0,
            "reused": reused,
            "retries": 0,
            "excluded": 0,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_scalar_calibrators": 2 * len(runs),
        },
        "aggregates": aggregates,
        "results": [
            {
                "experiment_id": run["experiment_id"],
                "seed": run["seed"],
                "scientific_fingerprint": run["scientific_fingerprint"],
                "analysis_seconds": run["analysis_seconds"],
                "headline_metrics": run["headline_metrics"],
                "gates": run["gates"],
                "path": run["artifacts"]["result"],
                "result_sha256": _sha256(Path(run["artifacts"]["result"])),
            }
            for run in runs
        ],
        "execution": {
            "runner": "single_process_sequential_frozen_checkpoint_analysis",
            "scheduler": None,
        },
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
        },
        "method_boundaries": [
            "The selected stable three-checkpoint cohort is underpowered for a population-prevalence claim.",
            "The scalar boundary rotation is supervised on source target bins and is a readout calibration, not a representation probe.",
            "Held-out states use exact held-out defects projected into a source-fitted basis and do not establish independent computability.",
            "Continuous and discrete endpoints remain conditioned on the frozen answer-token decoder probabilities.",
        ],
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=ContinuousReadoutCalibrationConfig.source_root)
    parser.add_argument("--character-root", default=ContinuousReadoutCalibrationConfig.character_root)
    parser.add_argument("--rank-root", default=ContinuousReadoutCalibrationConfig.rank_root)
    parser.add_argument("--boundary-root", default=ContinuousReadoutCalibrationConfig.boundary_root)
    parser.add_argument("--boundary-basis-root", default=ContinuousReadoutCalibrationConfig.boundary_basis_root)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_continuous_readout_calibration/20260806_d6_preregistered"))
    parser.add_argument("--seeds", default="7,29,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--ranks", default="1,2,3,4,5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--post-outcome-corrective-replication", action="store_true")
    args = parser.parse_args(argv)
    config = ContinuousReadoutCalibrationConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        rank_root=args.rank_root,
        boundary_root=args.boundary_root,
        boundary_basis_root=args.boundary_basis_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        ranks=_ints(args.ranks),
        device=args.device,
        allow_underpowered=args.allow_underpowered,
        post_outcome_corrective_replication=args.post_outcome_corrective_replication,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
