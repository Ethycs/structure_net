#!/usr/bin/env python3
"""Audit whether TinyLLM defect-rank near misses are decoder-boundary corrections."""

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

import numpy as np
import torch

import experiments.structure_net.tinyllm_defect_subspace_rank as rank
import experiments.structure_net.tinyllm_predictive_circle as circle
import experiments.structure_net.tinyllm_reynolds_character_coupling as coupling
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-defect-boundary-audit.v1.1"
HYPOTHESIS_ID = "tinyllm-c2-defect-boundary-correction-v1"
REGIMES = ("composition", "extrapolation")
COHORTS = ("heldout_a", "heldout_b")
PRIMARY_SEEDS = (29, 53)
NEAR_RANK = {29: 4, 53: 2}
SUFFICIENT_RANK = {29: 8, 53: 4}
PRIMARY_FAILURES = {
    (29, "heldout_b", "extrapolation"),
    (53, "heldout_a", "composition"),
    (53, "heldout_b", "composition"),
}
PATH_COEFFICIENTS = (0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0)


@dataclass(frozen=True)
class DefectBoundaryAuditConfig:
    source_root: str = rank.DefectSubspaceRankConfig.source_root
    character_root: str = rank.DefectSubspaceRankConfig.character_root
    rank_root: str = "data/experiments/tinyllm_defect_subspace_rank/20260806_d6_preregistered"
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    first_blocks: int = 6
    path_coefficients: tuple[float, ...] = PATH_COEFFICIENTS
    disagreement_ceiling: float = 0.10
    adjacent_fraction_floor: float = 0.90
    mean_moment_shift_ceiling_bins: float = 0.125
    alignment_loss_ceiling: float = 0.005
    distortion_alignment_loss: float = 0.01
    distortion_moment_shift_bins: float = 0.25
    task_effect_floor: float = 1e-6
    decomposition_tolerance: float = 1e-6
    predecessor_fisher_tolerance: float = 1e-8
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("confirmatory audit seeds are fixed to 29,53")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.path_coefficients[0] != 0.0 or self.path_coefficients[-1] != 1.0:
            raise ValueError("path coefficients must include zero and one endpoints")
        if tuple(sorted(set(self.path_coefficients))) != self.path_coefficients:
            raise ValueError("path coefficients must be unique and increasing")
        if not 0 < self.disagreement_ceiling < 1:
            raise ValueError("disagreement ceiling must lie in (0,1)")
        if not 0 < self.adjacent_fraction_floor <= 1:
            raise ValueError("adjacent fraction floor must lie in (0,1]")
        if not self.alignment_loss_ceiling < self.distortion_alignment_loss:
            raise ValueError("alignment classification thresholds must be ordered")
        if not self.mean_moment_shift_ceiling_bins < self.distortion_moment_shift_bins:
            raise ValueError("moment-shift classification thresholds must be ordered")
        if self.predecessor_fisher_tolerance <= 0.0:
            raise ValueError("predecessor Fisher tolerance must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(rank.__file__),
        Path(coupling.__file__),
        Path(circle.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(rank._implementation_digest().encode())
    return digest.hexdigest()


def _rank_config(config: DefectBoundaryAuditConfig) -> rank.DefectSubspaceRankConfig:
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


def _load_rank_predecessor(config: DefectBoundaryAuditConfig, seed: int) -> tuple[dict[str, Any], Path]:
    campaign_path = Path(config.rank_root) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    path = Path(config.rank_root) / "runs" / f"seed_{seed}" / "result.json"
    value = json.loads(path.read_text())
    checkpoint = Path(value.get("provenance", {}).get("checkpoint", ""))
    if (
        campaign.get("schema_version") != rank.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != rank.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != "preregistered_primary_evidence"
        or value.get("schema_version") != rank.SCHEMA_VERSION
        or value.get("hypothesis_id") != rank.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != "preregistered_primary_evidence"
        or value.get("implementation_sha256") != campaign.get("implementation_sha256")
        or int(value.get("seed", -1)) != seed
        or not value.get("scientific_fingerprint")
        or not checkpoint.is_file()
        or _sha256(checkpoint) != value.get("provenance", {}).get("checkpoint_sha256")
        or int(value.get("gates", {}).get("minimum_fixed_sufficient_rank", -1))
        != SUFFICIENT_RANK[seed]
    ):
        raise ValueError(f"invalid defect-rank predecessor {path}")
    return value, path


def _path_key(coefficient: float) -> str:
    return "path_" + (f"{coefficient:.3f}".rstrip("0").rstrip(".").replace(".", "p"))


def _wrapped_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(left - right), torch.cos(left - right))


def prediction_boundary_metrics(
    near: torch.Tensor, exact: torch.Tensor, target_bins: torch.Tensor
) -> dict[str, Any]:
    """Describe categorical boundary crossings between near and exact posteriors."""
    near = near.double().cpu()
    exact = exact.double().cpu()
    target = target_bins.long().cpu()
    bins = exact.shape[1]
    near_pred = near.argmax(-1)
    exact_pred = exact.argmax(-1)
    disagreement = near_pred != exact_pred
    count = int(disagreement.sum())
    circular_delta = (near_pred - exact_pred).abs()
    circular_delta = torch.minimum(circular_delta, bins - circular_delta)
    introduced = (exact_pred == target) & (near_pred != target)
    rescued = (exact_pred != target) & (near_pred == target)

    angles = torch.arange(bins, dtype=torch.float64) * (2.0 * math.pi / bins)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    near_angle = torch.atan2(near @ sine, near @ cosine)
    exact_angle = torch.atan2(exact @ sine, exact @ cosine)
    moment_shift_bins = _wrapped_difference(near_angle, exact_angle).abs() / (2.0 * math.pi / bins)

    exact_log = exact.clamp_min(1e-15).log()
    top_two = torch.topk(exact_log, 2, dim=-1).values
    margins = top_two[:, 0] - top_two[:, 1]
    lower_quartile = torch.quantile(margins, 0.25)
    if count:
        disagreement_margins = margins[disagreement]
        adjacent = float((circular_delta[disagreement] == 1).double().mean())
        low_margin_capture = float((disagreement_margins <= lower_quartile).double().mean())
        disagreement_margin_median = float(disagreement_margins.median())
        exact_class = exact_pred[disagreement]
        near_class = near_pred[disagreement]
        row = torch.arange(count)
        exact_normal_margin = (
            exact_log[disagreement][row, exact_class]
            - exact_log[disagreement][row, near_class]
        )
        near_log = near.clamp_min(1e-15).log()[disagreement]
        near_normal_margin = near_log[row, exact_class] - near_log[row, near_class]
        boundary_normal_change = exact_normal_margin - near_normal_margin
    else:
        adjacent = 1.0
        low_margin_capture = 1.0
        disagreement_margin_median = None
        exact_normal_margin = torch.empty(0, dtype=torch.float64)
        boundary_normal_change = torch.empty(0, dtype=torch.float64)
    return {
        "example_count": len(near),
        "disagreement_count": count,
        "disagreement_rate": count / len(near),
        "introduced_error_count": int(introduced.sum()),
        "introduced_error_rate": float(introduced.double().mean()),
        "rescued_error_count": int(rescued.sum()),
        "rescued_error_rate": float(rescued.double().mean()),
        "adjacent_disagreement_fraction": adjacent,
        "mean_moment_shift_bins": float(moment_shift_bins.mean()),
        "median_moment_shift_bins": float(moment_shift_bins.median()),
        "p95_moment_shift_bins": float(torch.quantile(moment_shift_bins, 0.95)),
        "maximum_moment_shift_bins": float(moment_shift_bins.max()),
        "exact_top_two_margin_q25": float(lower_quartile),
        "exact_top_two_margin_median": float(margins.median()),
        "disagreement_margin_median": disagreement_margin_median,
        "bottom_quartile_margin_capture": low_margin_capture,
        "boundary_normal_exact_margin_median": None if not count else float(exact_normal_margin.median()),
        "boundary_normal_change_median": None if not count else float(boundary_normal_change.median()),
    }


def classify_boundary_mechanism(
    metrics: Mapping[str, Any], near_diagnostics: Mapping[str, Any], exact_diagnostics: Mapping[str, Any], config: DefectBoundaryAuditConfig
) -> str:
    alignment_loss = float(exact_diagnostics["circular_alignment"] - near_diagnostics["circular_alignment"])
    boundary_only = bool(
        metrics["disagreement_rate"] <= config.disagreement_ceiling
        and metrics["adjacent_disagreement_fraction"] >= config.adjacent_fraction_floor
        and metrics["mean_moment_shift_bins"] <= config.mean_moment_shift_ceiling_bins
        and alignment_loss <= config.alignment_loss_ceiling
    )
    if boundary_only:
        return "boundary_only"
    if (
        alignment_loss > config.distortion_alignment_loss
        or metrics["mean_moment_shift_bins"] > config.distortion_moment_shift_bins
    ):
        return "continuous_map_distortion"
    return "distributed_boundary_sensitivity"


def _crossing_summary(
    posteriors: Mapping[str, torch.Tensor], coefficients: Sequence[float]
) -> dict[str, Any]:
    exact_pred = posteriors[_path_key(1.0)].argmax(-1)
    near_pred = posteriors[_path_key(0.0)].argmax(-1)
    mask = near_pred != exact_pred
    first = []
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        crossing = None
        for coefficient in coefficients[1:]:
            if int(posteriors[_path_key(coefficient)][index].argmax()) == int(exact_pred[index]):
                crossing = coefficient
                break
        first.append(crossing)
    histogram = {
        _path_key(coefficient): sum(value == coefficient for value in first)
        for coefficient in coefficients[1:]
    }
    histogram["never"] = sum(value is None for value in first)
    return {
        "near_disagreement_count": int(mask.sum()),
        "first_exact_match_histogram": histogram,
        "fraction_matching_only_at_or_after_0p75": (
            None if not first else sum(value is not None and value >= 0.75 for value in first) / len(first)
        ),
    }


def predecessor_replication_contract(
    cells: Sequence[Mapping[str, Any]],
    predecessor: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Verify regenerated near/sufficient/exact endpoints against rank evidence."""
    previous = {
        (cell["cohort"], cell["regime"]): cell
        for cell in predecessor["heldout_cells"]
    }
    maximum_fisher_error = 0.0
    maximum_diagnostic_error = 0.0
    causal_matches = True
    comparisons = 0
    for cell in cells:
        prior = previous[(cell["cohort"], cell["regime"])]
        mappings = (
            (_path_key(0.0), f"rank_{cell['near_rank']}"),
            ("sufficient_rank", f"rank_{cell['sufficient_rank']}"),
            (_path_key(1.0), "exact"),
        )
        for current_key, prior_key in mappings:
            current = cell["interventions"][current_key]
            reference = prior["interventions"][prior_key]
            causal_matches = causal_matches and bool(current["causal_pass"]) == bool(
                reference["causal_pass"]
            )
            current_value = float(
                current["effect_preservation"]["preserved_fraction"]
            )
            reference_value = float(
                reference["effect_preservation"]["preserved_fraction"]
            )
            maximum_fisher_error = max(
                maximum_fisher_error, abs(current_value - reference_value)
            )
            for metric in (
                "circular_alignment",
                "exact_bin_accuracy",
                "maximum_angular_increment",
                "minimum_moment_magnitude",
                "winding_degree",
            ):
                maximum_diagnostic_error = max(
                    maximum_diagnostic_error,
                    abs(
                        float(current["diagnostics"][metric])
                        - float(reference["diagnostics"][metric])
                    ),
                )
            comparisons += 1
    return {
        "passed": bool(causal_matches and maximum_fisher_error <= tolerance),
        "causal_classifications_match": causal_matches,
        "maximum_absolute_fisher_error": maximum_fisher_error,
        "maximum_absolute_auxiliary_diagnostic_error": maximum_diagnostic_error,
        "comparison_count": comparisons,
        "tolerance": tolerance,
    }


@torch.no_grad()
def evaluate_cell(
    system: Any,
    task: CircleTaskConfig,
    config: DefectBoundaryAuditConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    near_rank = NEAR_RANK[seed]
    sufficient_rank = SUFFICIENT_RANK[seed]
    defect = cell["full_defect"]
    propagated = cell["propagated"]
    shape = defect.shape
    flat = defect.reshape(config.orbit_count, -1)
    near = rank.project_onto_basis(flat, basis, near_rank).reshape(shape)
    exact = defect
    omitted = exact - near
    sufficient = rank.project_onto_basis(flat, basis, sufficient_rank).reshape(shape)
    states: dict[str, torch.Tensor] = {
        _path_key(coefficient): propagated + near + coefficient * omitted
        for coefficient in config.path_coefficients
    }
    states["sufficient_rank"] = propagated + sufficient
    for direction in range(near_rank, sufficient_rank):
        component = rank.project_onto_basis(flat, basis, direction + 1) - rank.project_onto_basis(flat, basis, direction)
        states[f"add_direction_{direction + 1}"] = propagated + near + component.reshape(shape)

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
    zero = coupling._continue_tensor(
        system,
        cell["target_cut"],
        coupling._repeat_patch(propagated, 2),
        task,
        config.continuation_batch_size,
    ).reshape(config.orbit_count, 2, -1)[:, 0]
    exact_posterior = posteriors[_path_key(1.0)]
    interventions = {}
    for key, posterior in posteriors.items():
        repeated_posterior = posterior[:, None].expand(-1, 2, -1).reshape(config.orbit_count * 2, -1)
        diagnostics = coupling._diagnostics(repeated_posterior, cell["dataset"])
        causal_pass, _ = coupling._causal_classification(diagnostics, cell["baseline"], 2)
        interventions[key] = {
            "diagnostics": diagnostics,
            "causal_pass": causal_pass,
            "effect_preservation": rank.heads._effect_preservation(
                posterior, exact_posterior, zero, config.task_effect_floor
            ),
            "boundary_to_exact": prediction_boundary_metrics(
                posterior, exact_posterior, cell["dataset"].target_bins.reshape(config.orbit_count, 2)[:, 0]
            ),
        }
    near_key = _path_key(0.0)
    exact_key = _path_key(1.0)
    boundary = interventions[near_key]["boundary_to_exact"]
    classification = classify_boundary_mechanism(
        boundary,
        interventions[near_key]["diagnostics"],
        interventions[exact_key]["diagnostics"],
        config,
    )
    primary_failure = (seed, cell["cohort"], cell["regime"]) in PRIMARY_FAILURES
    return {
        "cohort": cell["cohort"],
        "regime": cell["regime"],
        "evaluation_seed": cell["evaluation_seed"],
        "primary_failure_cell": primary_failure,
        "near_rank": near_rank,
        "sufficient_rank": sufficient_rank,
        "decomposition_relative_error": cell["decomposition_relative_error"],
        "baseline": cell["baseline"],
        "mechanism_classification": classification,
        "crossing_summary": _crossing_summary(posteriors, config.path_coefficients),
        "interventions": interventions,
    }


def _fingerprint(
    config: DefectBoundaryAuditConfig,
    task: CircleTaskConfig,
    seed: int,
    checkpoint_sha: str,
    predecessor_sha: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "rank_predecessor_sha256": predecessor_sha,
        "near_rank": NEAR_RANK[seed],
        "sufficient_rank": SUFFICIENT_RANK[seed],
        "primary_failures": sorted(PRIMARY_FAILURES),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def analyze_seed(
    task: CircleTaskConfig,
    config: DefectBoundaryAuditConfig,
    seed: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    rank_config = _rank_config(config)
    bridge = rank._bridge_config(rank_config)
    system, provenance = rank.deck.load_source(task, bridge, 2, seed, device)
    frozen, frozen_path = rank._load_character_source(rank_config, seed, provenance["checkpoint_sha256"])
    predecessor, predecessor_path = _load_rank_predecessor(config, seed)
    predecessor_sha = _sha256(predecessor_path)
    source_cells = [
        rank._defect_cell(system, task, rank_config, bridge, frozen, seed, "source_selection", regime, device)
        for regime in REGIMES
    ]
    source_matrix = torch.cat(
        [cell["full_defect"].reshape(config.orbit_count, -1) for cell in source_cells], dim=0
    )
    basis, energy = rank.fit_right_singular_basis(source_matrix)
    cells = []
    for cohort in COHORTS:
        for regime in REGIMES:
            cell = rank._defect_cell(system, task, rank_config, bridge, frozen, seed, cohort, regime, device)
            cells.append(evaluate_cell(system, task, config, cell, basis, seed))
            print(f"seed {seed} {cohort} {regime} complete", flush=True)
    predecessor_replication = predecessor_replication_contract(
        cells, predecessor, config.predecessor_fisher_tolerance
    )
    primary = [cell for cell in cells if cell["primary_failure_cell"]]
    next_rank_key = f"add_direction_{NEAR_RANK[seed] + 1}"
    next_rank_sufficient = all(
        cell["interventions"][next_rank_key]["causal_pass"]
        and not cell["interventions"][next_rank_key]["effect_preservation"]["degenerate"]
        and cell["interventions"][next_rank_key]["effect_preservation"]["preserved_fraction"]
        >= 0.90
        for cell in cells
    )
    controls_pass = all(
        not cell["interventions"][_path_key(0.0)]["causal_pass"]
        and cell["interventions"]["sufficient_rank"]["causal_pass"]
        and cell["interventions"][_path_key(1.0)]["causal_pass"]
        for cell in primary
    )
    gates = {
        "predecessor_identity": bool(
            predecessor["implementation_sha256"]
            == json.loads((Path(config.rank_root) / "campaign_results.json").read_text())["implementation_sha256"]
        ),
        "predecessor_endpoint_replication": predecessor_replication["passed"],
        "exact_decomposition": max(cell["decomposition_relative_error"] for cell in cells)
        <= config.decomposition_tolerance,
        "primary_endpoint_controls": controls_pass,
        "all_primary_failures_boundary_only": bool(
            primary and all(cell["mechanism_classification"] == "boundary_only" for cell in primary)
        ),
        "primary_boundary_only_count": sum(
            cell["mechanism_classification"] == "boundary_only" for cell in primary
        ),
        "primary_classifications": [cell["mechanism_classification"] for cell in primary],
        "next_singular_direction_sufficient_all_cells": next_rank_sufficient,
        "refined_minimum_sufficient_rank": (
            NEAR_RANK[seed] + 1 if next_rank_sufficient else SUFFICIENT_RANK[seed]
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-defect-boundary-audit-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "post_outcome_corrective_replication_evidence"
        ),
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": _fingerprint(
            config, task, seed, provenance["checkpoint_sha256"], predecessor_sha
        ),
        "implementation_sha256": _implementation_digest(),
        "provenance": {
            **provenance,
            "character_result": str(frozen_path),
            "character_result_sha256": _sha256(frozen_path),
            "rank_predecessor": str(predecessor_path),
            "rank_predecessor_sha256": predecessor_sha,
        },
        "source_basis": {
            "rank": int(basis.shape[0]),
            "near_rank": NEAR_RANK[seed],
            "sufficient_rank": SUFFICIENT_RANK[seed],
            "near_cumulative_energy": float(energy[: NEAR_RANK[seed]].sum().cpu()),
            "sufficient_cumulative_energy": float(energy[: SUFFICIENT_RANK[seed]].sum().cpu()),
        },
        "predecessor_replication": predecessor_replication,
        "cells": cells,
        "gates": gates,
        "all_primary_gates_passed": bool(
            gates["predecessor_identity"]
            and gates["predecessor_endpoint_replication"]
            and gates["exact_decomposition"]
            and gates["primary_endpoint_controls"]
            and gates["all_primary_failures_boundary_only"]
        ),
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary_cells = [cell for run in runs for cell in run["cells"] if cell["primary_failure_cell"]]
    observed_seeds = {int(run["seed"]) for run in runs}
    classifications = {
        label: sum(cell["mechanism_classification"] == label for cell in primary_cells)
        for label in ("boundary_only", "distributed_boundary_sensitivity", "continuous_map_distortion")
    }
    return {
        "primary_failure_cell_count": len(primary_cells),
        "classification_counts": classifications,
        "predecessor_identity_count": sum(run["gates"]["predecessor_identity"] for run in runs),
        "predecessor_endpoint_replication_count": sum(
            run["gates"]["predecessor_endpoint_replication"] for run in runs
        ),
        "exact_decomposition_count": sum(run["gates"]["exact_decomposition"] for run in runs),
        "endpoint_control_count": sum(run["gates"]["primary_endpoint_controls"] for run in runs),
        "joint_seed_gate_count": sum(bool(run["all_primary_gates_passed"]) for run in runs),
        "expected_primary_seeds_present": observed_seeds == set(PRIMARY_SEEDS),
        "confirmed": bool(
            observed_seeds == set(PRIMARY_SEEDS)
            and len(primary_cells) == len(PRIMARY_FAILURES)
            and all(bool(run["all_primary_gates_passed"]) for run in runs)
            and classifications["boundary_only"] == len(primary_cells)
        ),
        "refined_minimum_sufficient_ranks": {
            str(run["seed"]): run["gates"]["refined_minimum_sufficient_rank"]
            for run in runs
        },
        "per_cell": {
            f"seed{run['seed']}:{cell['cohort']}:{cell['regime']}": {
                "primary_failure_cell": cell["primary_failure_cell"],
                "classification": cell["mechanism_classification"],
                "boundary": cell["interventions"][_path_key(0.0)]["boundary_to_exact"],
                "crossing_summary": cell["crossing_summary"],
            }
            for run in runs
            for cell in run["cells"]
        },
    }


def run_campaign(config: DefectBoundaryAuditConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    campaign_path = output / "campaign_results.json"
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    implementation = _implementation_digest()
    existing_campaign = None
    if campaign_path.is_file():
        existing_campaign = json.loads(campaign_path.read_text())
        if (
            existing_campaign.get("schema_version") != SCHEMA_VERSION
            or existing_campaign.get("hypothesis_id") != HYPOTHESIS_ID
            or existing_campaign.get("status") != "completed"
            or existing_campaign.get("implementation_sha256") != implementation
        ):
            raise ValueError(
                "completed campaign root is immutable; use a new output path"
            )
    runs = []
    reused = 0
    for seed in config.seeds:
        path = output / "runs" / f"seed_{seed}" / "result.json"
        if path.is_file():
            existing = json.loads(path.read_text())
            checkpoint = Path(config.source_root) / "runs" / "k2" / f"seed_{seed}" / "model.pt"
            predecessor = Path(config.rank_root) / "runs" / f"seed_{seed}" / "result.json"
            expected = _fingerprint(config, task, seed, _sha256(checkpoint), _sha256(predecessor))
            if (
                existing.get("status") == "completed"
                and existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("scientific_fingerprint") == expected
                and existing.get("implementation_sha256") == implementation
            ):
                runs.append(existing)
                reused += 1
                print(f"resuming {existing['experiment_id']}", flush=True)
                continue
        runs.append(analyze_seed(task, config, seed, output, device))
    if existing_campaign is not None:
        if reused != len(config.seeds):
            raise ValueError("completed campaign has missing or non-reusable cells")
        print("preserving immutable completed campaign aggregate", flush=True)
        return existing_campaign
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else "post_outcome_corrective_replication_evidence"
        ),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
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
        },
        "aggregates": aggregate(runs),
        "results": [
            {
                "experiment_id": run["experiment_id"],
                "seed": run["seed"],
                "scientific_fingerprint": run["scientific_fingerprint"],
                "analysis_seconds": run["analysis_seconds"],
                "gates": run["gates"],
                "path": str(output / "runs" / f"seed_{run['seed']}" / "result.json"),
            }
            for run in runs
        ],
        "execution": {
            "runner": "single_process_sequential_frozen_checkpoint_analysis",
            "scheduler": None,
        },
        "method_boundaries": [
            "The audit conditions on the three rank-titration failures and is not an independent quotient-front replication.",
            "Interpolation uses the exact omitted activation defect and tests causal effect, not independent computability.",
            "Decoder-boundary classifications are specific to the preregistered thresholds and frozen exact-bin task.",
            "Singular directions are checkpoint-local and are not aligned across seeds.",
        ],
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DefectBoundaryAuditConfig.source_root)
    parser.add_argument("--character-root", default=DefectBoundaryAuditConfig.character_root)
    parser.add_argument("--rank-root", default=DefectBoundaryAuditConfig.rank_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_defect_boundary_audit/"
            "20260806_d6_preregistered_v3"
        ),
    )
    parser.add_argument("--seeds", default="29,53")
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--path-coefficients", default="0,0.125,0.25,0.5,0.75,0.875,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = DefectBoundaryAuditConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        rank_root=args.rank_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        path_coefficients=_floats(args.path_coefficients),
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
