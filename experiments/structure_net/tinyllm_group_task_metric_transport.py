#!/usr/bin/env python3
"""Test group-anchored Fisher-metric transport of TinyLLM C2 carriers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-group-task-metric-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-group-task-metric-carrier-transport-v1"
PRIMARY_SEEDS = transport.PRIMARY_SEEDS
REGIMES = transport.REGIMES
COHORT_SEEDS = transport.COHORT_SEEDS
HELDOUT_COHORTS = transport.HELDOUT_COHORTS
PREDECESSOR_CAMPAIGN_SHA256 = (
    "44707fd4bcd810e63614671aa491095fae735ee52359d464ab25abb10a2bc228"
)


@dataclass(frozen=True)
class GroupTaskMetricTransportConfig:
    source_root: str = transport.CrossSeedCarrierTransportConfig.source_root
    character_root: str = transport.CrossSeedCarrierTransportConfig.character_root
    readout_root: str = transport.CrossSeedCarrierTransportConfig.readout_root
    transport_root: str = (
        "data/experiments/tinyllm_cross_seed_causal_carrier_transport/"
        "20260806_d6_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    first_blocks: int = 6
    metric_step_fraction: float = 1e-3
    metric_minimum_step: float = 1e-4
    metric_isotropic_floor: float = 0.01
    metric_trace_floor: float = 1e-10
    metric_stability_ceiling: float = 0.05
    metric_psd_tolerance: float = 1e-8
    metric_ridge: float = 1e-6
    coordinate_r2_floor: float = 0.80
    baseline_dominance_ratio: float = 0.75
    shuffled_mean_margin_bins: float = 0.125
    predecessor_replay_tolerance: float = 1e-6
    alignment_loss_ceiling: float = 0.005
    mean_shift_ceiling_bins: float = 0.125
    p95_shift_ceiling_bins: float = 0.50
    winding_tolerance: float = 0.10
    accuracy_loss_ceiling: float = 0.03
    whitening_ridge: float = 1e-6
    affine_ridge: float = 1e-6
    specificity_margin: float = 0.20
    task_effect_floor: float = 1e-6
    decomposition_tolerance: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary task-metric seeds are fixed to 7,29,53")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.carrier_rank != 3 and not self.allow_underpowered:
            raise ValueError("primary carrier rank is fixed to three")
        if not 0.0 < self.metric_step_fraction < 0.1:
            raise ValueError("metric step fraction must be small and positive")
        if self.metric_minimum_step <= 0.0:
            raise ValueError("metric minimum step must be positive")
        if not 0.0 < self.metric_isotropic_floor < 1.0:
            raise ValueError("metric isotropic floor must lie between zero and one")
        if min(self.metric_trace_floor, self.metric_ridge) <= 0.0:
            raise ValueError("metric trace floor and ridge must be positive")
        if not 0.0 < self.metric_stability_ceiling < 1.0:
            raise ValueError("metric stability ceiling must lie between zero and one")
        if not 0.0 < self.baseline_dominance_ratio < 1.0:
            raise ValueError("baseline dominance ratio must lie between zero and one")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
        Path(transport.readout.__file__),
        Path(transport.coupling.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: GroupTaskMetricTransportConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_underpowered_mechanistic_evidence"
    )


def _transport_config(
    config: GroupTaskMetricTransportConfig,
) -> transport.CrossSeedCarrierTransportConfig:
    return transport.CrossSeedCarrierTransportConfig(
        source_root=config.source_root,
        character_root=config.character_root,
        readout_root=config.readout_root,
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        carrier_rank=config.carrier_rank,
        first_blocks=config.first_blocks,
        coordinate_r2_floor=config.coordinate_r2_floor,
        specificity_margin=config.specificity_margin,
        whitening_ridge=config.whitening_ridge,
        affine_ridge=config.affine_ridge,
        alignment_loss_ceiling=config.alignment_loss_ceiling,
        mean_shift_ceiling_bins=config.mean_shift_ceiling_bins,
        p95_shift_ceiling_bins=config.p95_shift_ceiling_bins,
        winding_tolerance=config.winding_tolerance,
        accuracy_loss_ceiling=config.accuracy_loss_ceiling,
        task_effect_floor=config.task_effect_floor,
        decomposition_tolerance=config.decomposition_tolerance,
        activation_batch_size=config.activation_batch_size,
        continuation_batch_size=config.continuation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _mapping_from_summary(
    value: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "linear": torch.tensor(value["linear"], dtype=torch.float64, device=device),
        "intercept": torch.tensor(
            value["intercept"], dtype=torch.float64, device=device
        ),
    }


def _load_predecessor_campaign(
    config: GroupTaskMetricTransportConfig,
) -> tuple[dict[str, Any], Path]:
    path = Path(config.transport_root) / "campaign_results.json"
    value = json.loads(path.read_text())
    expected_hash = PREDECESSOR_CAMPAIGN_SHA256
    if (
        value.get("schema_version") != transport.SCHEMA_VERSION
        or value.get("hypothesis_id") != transport.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role")
        != "preregistered_underpowered_mechanistic_evidence"
        or value.get("implementation_sha256")
        != "798a85b50f7dc8489e28cc66a498a5c3a8af193f859b80b43b1c5cc551ba0a89"
        or value.get("aggregates", {}).get("confirmed") is not False
        or _sha256(path) != expected_hash
    ):
        raise ValueError(f"invalid cross-seed transport predecessor {path}")
    return value, path


def _load_predecessor_pair(
    campaign: Mapping[str, Any], source_seed: int, target_seed: int
) -> tuple[dict[str, Any], Path]:
    matching = [
        item
        for item in campaign.get("results", [])
        if int(item.get("source_seed", -1)) == source_seed
        and int(item.get("target_seed", -1)) == target_seed
    ]
    if len(matching) != 1:
        raise ValueError(f"missing predecessor pair {source_seed}->{target_seed}")
    reference = matching[0]
    path = Path(reference["path"])
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != transport.SCHEMA_VERSION
        or value.get("hypothesis_id") != transport.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("source_seed", -1)) != source_seed
        or int(value.get("target_seed", -1)) != target_seed
        or value.get("scientific_fingerprint")
        != reference.get("scientific_fingerprint")
        or _sha256(path) != reference.get("result_sha256")
    ):
        raise ValueError(f"invalid predecessor pair {path}")
    return value, path


@torch.no_grad()
def _continuation_logits(
    system: Any,
    cut: str,
    values: torch.Tensor,
    task: CircleTaskConfig,
    batch_size: int,
) -> torch.Tensor:
    device = values.device
    answer = torch.tensor(task.answer_token_ids, device=device)
    output = []
    for start in range(0, len(values), batch_size):
        value = values[start : start + batch_size]
        pieces = cut.split("_")
        block_index = int(pieces[1])
        if cut.endswith("post_attention"):
            block = system.model.transformer["h"][block_index]
            value = value + block.mlp(block.ln_2(value))
        next_block = block_index + 1
        for block in system.model.transformer["h"][next_block:]:
            value = block(value)
        output.append(
            transport.coupling.io_source._task_logits(
                system.model, value[:, -1, :], answer
            )
        )
    return torch.cat(output)


def _coordinate_states(
    cell: Mapping[str, Any], coordinates: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    flat = cell["full_defect"].reshape(cell["full_defect"].shape[0], -1)
    defect = (coordinates.double() @ basis.double()).to(dtype=flat.dtype)
    return cell["propagated"] + defect.reshape_as(cell["full_defect"])


def fisher_metric_from_logits_jacobian(
    base_logits: torch.Tensor,
    jacobian: torch.Tensor,
    config: GroupTaskMetricTransportConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return regularized pullback metrics, raw traces, and raw eigenvalues."""
    base_logits = base_logits.double()
    jacobian = jacobian.double()
    if (
        base_logits.ndim != 2
        or jacobian.ndim != 3
        or jacobian.shape[:2] != base_logits.shape
        or jacobian.shape[2] != config.carrier_rank
    ):
        raise ValueError("logit and Jacobian shapes are incompatible")
    probabilities = torch.softmax(base_logits, dim=-1)
    mean = (probabilities[..., None] * jacobian).sum(1, keepdim=True)
    centered = jacobian - mean
    raw = torch.einsum("nk,nki,nkj->nij", probabilities, centered, centered)
    raw = 0.5 * (raw + raw.transpose(-1, -2))
    traces = torch.diagonal(raw, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(
        config.carrier_rank, dtype=raw.dtype, device=raw.device
    )
    normalized = raw / traces.clamp_min(config.metric_trace_floor)[:, None, None]
    normalized = normalized + (
        config.metric_isotropic_floor / config.carrier_rank
    ) * identity
    return normalized, traces, torch.linalg.eigvalsh(raw)


@torch.no_grad()
def pullback_fisher_metrics(
    system: Any,
    task: CircleTaskConfig,
    config: GroupTaskMetricTransportConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    *,
    step_multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate trace-normalized rank-three pullback Fisher metrics."""
    coordinates = transport._coordinates(cell, basis)
    standard_deviation = coordinates.std(0, unbiased=True)
    steps = torch.maximum(
        standard_deviation * (config.metric_step_fraction * step_multiplier),
        torch.full_like(standard_deviation, config.metric_minimum_step),
    )
    base_logits = _continuation_logits(
        system,
        cell["target_cut"],
        _coordinate_states(cell, coordinates, basis),
        task,
        config.continuation_batch_size,
    ).double()
    derivatives = []
    for direction in range(config.carrier_rank):
        offset = torch.zeros_like(coordinates)
        offset[:, direction] = steps[direction]
        plus = _continuation_logits(
            system,
            cell["target_cut"],
            _coordinate_states(cell, coordinates + offset, basis),
            task,
            config.continuation_batch_size,
        ).double()
        minus = _continuation_logits(
            system,
            cell["target_cut"],
            _coordinate_states(cell, coordinates - offset, basis),
            task,
            config.continuation_batch_size,
        ).double()
        derivatives.append((plus - minus) / (2.0 * steps[direction]))
    jacobian = torch.stack(derivatives, dim=-1)
    normalized, traces, eigenvalues = fisher_metric_from_logits_jacobian(
        base_logits, jacobian, config
    )
    return normalized, {
        "step_multiplier": step_multiplier,
        "steps": steps.detach().cpu().tolist(),
        "minimum_raw_trace": float(traces.min()),
        "median_raw_trace": float(traces.median()),
        "maximum_raw_trace": float(traces.max()),
        "minimum_raw_eigenvalue": float(eigenvalues.min()),
        "maximum_raw_eigenvalue": float(eigenvalues.max()),
        "finite": bool(torch.isfinite(normalized).all()),
    }


def metric_stability(
    full: torch.Tensor,
    half: torch.Tensor,
) -> dict[str, float]:
    numerator = torch.linalg.matrix_norm(full - half, dim=(-2, -1))
    denominator = torch.linalg.matrix_norm(half, dim=(-2, -1)).clamp_min(1e-24)
    relative = numerator / denominator
    return {
        "median_relative_error": float(relative.median()),
        "maximum_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
    }


def fit_task_metric_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    metrics: torch.Tensor,
    ridge: float,
) -> dict[str, torch.Tensor]:
    """Fit an affine map under observation-specific target quadratic metrics."""
    source = source.double()
    target = target.double()
    metrics = metrics.double()
    if (
        source.ndim != 2
        or target.shape != source.shape
        or metrics.shape != (source.shape[0], target.shape[1], target.shape[1])
    ):
        raise ValueError("source, target, and metric shapes are incompatible")
    observations, outputs = target.shape
    ones = torch.ones((observations, 1), dtype=source.dtype, device=source.device)
    design = torch.cat((source, ones), dim=1)
    predictors = design.shape[1]
    normal = torch.zeros(
        (outputs * predictors, outputs * predictors),
        dtype=source.dtype,
        device=source.device,
    )
    right = torch.zeros(
        outputs * predictors, dtype=source.dtype, device=source.device
    )
    outer = torch.einsum("ni,nj->nij", design, design)
    for left_output in range(outputs):
        left_slice = slice(
            left_output * predictors, (left_output + 1) * predictors
        )
        right[left_slice] = torch.einsum(
            "ni,nb,nb->i",
            design,
            metrics[:, left_output, :],
            target,
        )
        for right_output in range(outputs):
            right_slice = slice(
                right_output * predictors, (right_output + 1) * predictors
            )
            normal[left_slice, right_slice] = torch.einsum(
                "n,nij->ij", metrics[:, left_output, right_output], outer
            )
    penalty = torch.zeros_like(normal)
    for output in range(outputs):
        start = output * predictors
        penalty[start : start + predictors - 1, start : start + predictors - 1] = (
            torch.eye(predictors - 1, dtype=source.dtype, device=source.device)
        )
    solution = torch.linalg.solve(normal + ridge * penalty, right)
    by_output = solution.reshape(outputs, predictors)
    singular = torch.linalg.svdvals(normal + ridge * penalty)
    return {
        "linear": by_output[:, :-1].T,
        "intercept": by_output[:, -1],
        "normal_condition_number": singular.max() / singular.min().clamp_min(1e-24),
    }


def _mapping_summary(mapping: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    output = transport._mapping_summary(mapping)
    if "normal_condition_number" in mapping:
        output["normal_condition_number"] = float(
            mapping["normal_condition_number"].detach().cpu()
        )
    return output


def _metric_contract(
    summaries: Sequence[Mapping[str, Any]],
    stability: Mapping[str, float],
    config: GroupTaskMetricTransportConfig,
) -> bool:
    return bool(
        summaries
        and all(item["finite"] for item in summaries)
        and min(float(item["minimum_raw_trace"]) for item in summaries)
        > config.metric_trace_floor
        and min(float(item["minimum_raw_eigenvalue"]) for item in summaries)
        >= -config.metric_psd_tolerance
        and stability["median_relative_error"] <= config.metric_stability_ceiling
    )


def _continuous_pass(record: Mapping[str, Any]) -> bool:
    return bool(record["continuous"]["continuous_pass"])


def _predecessor_replay_error(
    heldout: Sequence[Mapping[str, Any]], predecessor: Mapping[str, Any]
) -> float:
    previous_cells = {
        (cell["cohort"], cell["regime"]): cell
        for cell in predecessor["heldout_cells"]
    }
    maximum = 0.0
    for cell in heldout:
        previous = previous_cells[(cell["cohort"], cell["regime"])]
        for state in ("zero", "exact", "direct_rank3", "paired", "shuffled", "affine_ridge"):
            for metric in (
                "alignment_loss_from_exact",
                "mean_moment_shift_bins",
                "p95_moment_shift_bins",
            ):
                current = float(cell["states"][state]["continuous"][metric])
                reference = float(previous["states"][state]["continuous"][metric])
                maximum = max(maximum, abs(current - reference))
    return maximum


def _aggregate_mean_shift(
    heldout: Sequence[Mapping[str, Any]], state: str
) -> float:
    return sum(
        float(cell["states"][state]["continuous"]["mean_moment_shift_bins"])
        for cell in heldout
    ) / len(heldout)


def pair_gates(
    heldout: Sequence[Mapping[str, Any]],
    metric_contract_passed: bool,
    predecessor_replay_error: float,
    config: GroupTaskMetricTransportConfig,
) -> dict[str, Any]:
    controls = all(
        not _continuous_pass(cell["states"]["zero"])
        and _continuous_pass(cell["states"]["exact"])
        and _continuous_pass(cell["states"]["direct_rank3"])
        and cell["decomposition_relative_error"] <= config.decomposition_tolerance
        for cell in heldout
    )
    coordinates = all(
        cell["coordinate_metrics"]["task_metric"]["variance_explained"]
        >= config.coordinate_r2_floor
        for cell in heldout
    )
    causal = all(
        _continuous_pass(cell["states"]["task_metric"]) for cell in heldout
    )
    task_mean = _aggregate_mean_shift(heldout, "task_metric")
    paired_mean = _aggregate_mean_shift(heldout, "paired")
    ridge_mean = _aggregate_mean_shift(heldout, "affine_ridge")
    shuffled_mean = _aggregate_mean_shift(heldout, "task_metric_shuffled")
    dominance = task_mean <= config.baseline_dominance_ratio * min(
        paired_mean, ridge_mean
    )
    shuffled_fails = any(
        cell["coordinate_metrics"]["task_metric_shuffled"]["variance_explained"]
        < config.coordinate_r2_floor
        or not _continuous_pass(cell["states"]["task_metric_shuffled"])
        for cell in heldout
    )
    specificity = bool(
        shuffled_fails
        and task_mean + config.shuffled_mean_margin_bins <= shuffled_mean
    )
    replay = predecessor_replay_error <= config.predecessor_replay_tolerance
    return {
        "metric_contract": metric_contract_passed,
        "continuous_target_control_contract": controls,
        "task_metric_coordinate_transport": coordinates,
        "task_metric_causal_transport": causal,
        "task_metric_dominates_euclidean_baselines": dominance,
        "task_metric_shuffled_specificity": specificity,
        "predecessor_replay_contract": replay,
        "task_metric_aggregate_mean_shift_bins": task_mean,
        "paired_aggregate_mean_shift_bins": paired_mean,
        "affine_ridge_aggregate_mean_shift_bins": ridge_mean,
        "task_metric_shuffled_aggregate_mean_shift_bins": shuffled_mean,
        "predecessor_replay_maximum_absolute_error": predecessor_replay_error,
    }


def _fingerprint(
    config: GroupTaskMetricTransportConfig,
    task: CircleTaskConfig,
    source_seed: int,
    target_seed: int,
    source_provenance: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    predecessor_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "cohort_seeds": COHORT_SEEDS,
        "source_seed": source_seed,
        "target_seed": target_seed,
        "source_checkpoint_sha256": source_provenance["checkpoint_sha256"],
        "target_checkpoint_sha256": target_provenance["checkpoint_sha256"],
        "predecessor_sha256": predecessor_sha256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def _campaign_is_reusable(
    campaign: Mapping[str, Any],
    config: GroupTaskMetricTransportConfig,
    implementation: str,
) -> bool:
    required_pairs = len(config.seeds) * (len(config.seeds) - 1)
    if (
        campaign.get("schema_version") != SCHEMA_VERSION
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != _evidence_role(config)
        or campaign.get("implementation_sha256") != implementation
        or campaign.get("configuration") != _json_compatible(asdict(config))
        or int(campaign.get("summary", {}).get("completed", -1)) != required_pairs
        or len(campaign.get("results", [])) != required_pairs
    ):
        return False
    pairs = set()
    fingerprints = set()
    for entry in campaign["results"]:
        path = Path(entry.get("path", ""))
        if not path.is_file() or _sha256(path) != entry.get("result_sha256"):
            return False
        detail = json.loads(path.read_text())
        pair = (int(entry["source_seed"]), int(entry["target_seed"]))
        fingerprint = entry.get("scientific_fingerprint")
        if (
            pair[0] == pair[1]
            or pair in pairs
            or not fingerprint
            or fingerprint in fingerprints
            or detail.get("schema_version") != SCHEMA_VERSION
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != _evidence_role(config)
            or detail.get("implementation_sha256") != implementation
            or detail.get("scientific_fingerprint") != fingerprint
        ):
            return False
        pairs.add(pair)
        fingerprints.add(fingerprint)
    return pairs == {
        (source, target)
        for source in config.seeds
        for target in config.seeds
        if source != target
    }


def run_campaign(
    config: GroupTaskMetricTransportConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text())
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(
            f"incompatible completed aggregate {campaign_path}; use a new root"
        )

    base_config = _transport_config(config)
    rank_config = transport._rank_config(base_config)
    bridge = transport.rank._bridge_config(rank_config)
    predecessor_campaign: dict[str, Any] | None = None
    predecessor_campaign_path: Path | None = None
    if not config.allow_underpowered:
        predecessor_campaign, predecessor_campaign_path = _load_predecessor_campaign(
            config
        )

    datasets = {
        cohort: {
            regime: transport.rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=config.orbit_count,
                seed=COHORT_SEEDS[cohort][regime],
                regime=regime,
            )
            for regime in REGIMES
        }
        for cohort in COHORT_SEEDS
    }

    seed_records: dict[int, dict[str, Any]] = {}
    for seed in config.seeds:
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        frozen, frozen_path = transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        if any(
            int(frozen["regimes"][regime]["synthesis_front_index"]) != 0
            for regime in REGIMES
        ):
            raise ValueError(f"seed {seed} is not a stable block-0 front")
        basis, basis_summary = transport._fit_seed_basis(
            system,
            task,
            base_config,
            rank_config,
            bridge,
            frozen,
            seed,
            device,
        )
        readout_predecessor = transport._load_readout_predecessor(
            base_config, seed, provenance["checkpoint_sha256"]
        )
        cells = {
            cohort: {
                regime: transport._extract_cell(
                    system,
                    task,
                    base_config,
                    bridge,
                    datasets[cohort][regime],
                    cohort,
                    regime,
                    device,
                )
                for regime in REGIMES
            }
            for cohort in COHORT_SEEDS
        }
        full_metrics = []
        half_metrics = []
        metric_summaries = []
        half_summaries = []
        for regime in REGIMES:
            full, summary = pullback_fisher_metrics(
                system,
                task,
                config,
                cells["alignment_fit"][regime],
                basis,
            )
            half, half_summary = pullback_fisher_metrics(
                system,
                task,
                config,
                cells["alignment_fit"][regime],
                basis,
                step_multiplier=0.5,
            )
            full_metrics.append(full)
            half_metrics.append(half)
            metric_summaries.append({"regime": regime, **summary})
            half_summaries.append({"regime": regime, **half_summary})
        metrics = torch.cat(full_metrics)
        half = torch.cat(half_metrics)
        stability = metric_stability(metrics, half)
        seed_records[seed] = {
            "system": system,
            "provenance": {
                **provenance,
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis,
            "basis_summary": basis_summary,
            "readout": readout_predecessor,
            "cells": cells,
            "fit_metrics": metrics,
            "metric_summaries": metric_summaries,
            "half_metric_summaries": half_summaries,
            "metric_stability": stability,
            "metric_contract": _metric_contract(
                metric_summaries + half_summaries, stability, config
            ),
        }
        print(f"seed {seed} task metric extracted", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("analysis implementation changed during campaign")

    pair_results = []
    for source_seed in config.seeds:
        for target_seed in config.seeds:
            if source_seed == target_seed:
                continue
            source_record = seed_records[source_seed]
            target_record = seed_records[target_seed]
            source_fit = torch.cat(
                [
                    transport._coordinates(
                        source_record["cells"]["alignment_fit"][regime],
                        source_record["basis"],
                    )
                    for regime in REGIMES
                ]
            )
            target_fit = torch.cat(
                [
                    transport._coordinates(
                        target_record["cells"]["alignment_fit"][regime],
                        target_record["basis"],
                    )
                    for regime in REGIMES
                ]
            )
            permutation = transport.regime_preserving_permutation(
                config.orbit_count, source_seed, target_seed
            ).to(device)
            metric_mapping = fit_task_metric_affine(
                source_fit,
                target_fit,
                target_record["fit_metrics"],
                config.metric_ridge,
            )
            metric_shuffled = fit_task_metric_affine(
                source_fit,
                target_fit[permutation],
                target_record["fit_metrics"][permutation],
                config.metric_ridge,
            )

            predecessor: dict[str, Any] | None = None
            predecessor_path: Path | None = None
            if predecessor_campaign is not None:
                predecessor, predecessor_path = _load_predecessor_pair(
                    predecessor_campaign, source_seed, target_seed
                )
                paired = _mapping_from_summary(
                    predecessor["alignment_fit"]["paired_map"], device
                )
                shuffled = _mapping_from_summary(
                    predecessor["alignment_fit"]["shuffled_map"], device
                )
                affine = _mapping_from_summary(
                    predecessor["alignment_fit"]["affine_ridge_map"], device
                )
            else:
                paired = transport.fit_whitened_orthogonal(
                    source_fit, target_fit, config.whitening_ridge
                )
                shuffled = transport.fit_whitened_orthogonal(
                    source_fit,
                    target_fit[permutation],
                    config.whitening_ridge,
                )
                affine = transport.fit_affine_ridge(
                    source_fit, target_fit, config.affine_ridge
                )
            mappings = {
                "paired": paired,
                "shuffled": shuffled,
                "affine_ridge": affine,
                "task_metric": metric_mapping,
                "task_metric_shuffled": metric_shuffled,
            }
            heldout = []
            for cohort in HELDOUT_COHORTS:
                for regime in REGIMES:
                    heldout.append(
                        transport._evaluate_transport_cell(
                            target_record["system"],
                            task,
                            base_config,
                            source_record["cells"][cohort][regime],
                            target_record["cells"][cohort][regime],
                            source_record["basis"],
                            target_record["basis"],
                            mappings,
                            target_record["readout"]["rotation_bins"],
                        )
                    )
            replay_error = (
                0.0
                if predecessor is None
                else _predecessor_replay_error(heldout, predecessor)
            )
            gates = pair_gates(
                heldout,
                bool(target_record["metric_contract"]),
                replay_error,
                config,
            )
            predecessor_sha256 = (
                "systems-only-no-quality-predecessor"
                if predecessor_path is None
                else _sha256(predecessor_path)
            )
            fingerprint = _fingerprint(
                config,
                task,
                source_seed,
                target_seed,
                source_record["provenance"],
                target_record["provenance"],
                predecessor_sha256,
            )
            result_path = (
                output
                / "runs"
                / f"source_{source_seed}_target_{target_seed}"
                / "result.json"
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": (
                    f"tinyllm-c2-task-metric-transport-{source_seed}-to-{target_seed}"
                ),
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "source_seed": source_seed,
                "target_seed": target_seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "provenance": {
                    "source": source_record["provenance"],
                    "target": target_record["provenance"],
                    "source_readout": source_record["readout"],
                    "target_readout": target_record["readout"],
                    "predecessor_campaign": (
                        None
                        if predecessor_campaign_path is None
                        else str(predecessor_campaign_path)
                    ),
                    "predecessor_campaign_sha256": (
                        None
                        if predecessor_campaign_path is None
                        else _sha256(predecessor_campaign_path)
                    ),
                    "predecessor_result": (
                        None if predecessor_path is None else str(predecessor_path)
                    ),
                    "predecessor_result_sha256": predecessor_sha256,
                },
                "source_basis": source_record["basis_summary"],
                "target_basis": target_record["basis_summary"],
                "target_task_metric": {
                    "full_step": target_record["metric_summaries"],
                    "half_step": target_record["half_metric_summaries"],
                    "stability": target_record["metric_stability"],
                    "contract_passed": target_record["metric_contract"],
                },
                "alignment_fit": {
                    "task_metric_metrics": transport.coordinate_metrics(
                        transport.apply_affine(source_fit, metric_mapping), target_fit
                    ),
                    "task_metric_shuffled_metrics_against_true_pairs": (
                        transport.coordinate_metrics(
                            transport.apply_affine(source_fit, metric_shuffled),
                            target_fit,
                        )
                    ),
                    "task_metric_map": _mapping_summary(metric_mapping),
                    "task_metric_shuffled_map": _mapping_summary(metric_shuffled),
                    "permutation_sha256": hashlib.sha256(
                        permutation.cpu().numpy().tobytes()
                    ).hexdigest(),
                },
                "heldout_cells": heldout,
                "gates": gates,
                "primary_metric": float(
                    all(
                        gates[name]
                        for name in (
                            "metric_contract",
                            "continuous_target_control_contract",
                            "task_metric_coordinate_transport",
                            "task_metric_causal_transport",
                            "task_metric_dominates_euclidean_baselines",
                            "task_metric_shuffled_specificity",
                            "predecessor_replay_contract",
                        )
                    )
                ),
                "analysis_seconds": time.perf_counter() - started,
                "artifacts": {"result": str(result_path)},
            }
            if result_path.is_file():
                existing = json.loads(result_path.read_text())
                if (
                    existing.get("schema_version") == SCHEMA_VERSION
                    and existing.get("hypothesis_id") == HYPOTHESIS_ID
                    and existing.get("status") == "completed"
                    and existing.get("evidence_role") == _evidence_role(config)
                    and existing.get("implementation_sha256") == implementation
                    and existing.get("scientific_fingerprint") == fingerprint
                ):
                    result = existing
                    print(f"resuming {existing['experiment_id']}", flush=True)
                else:
                    raise ValueError(
                        f"incompatible completed result {result_path}; use a new root"
                    )
            else:
                _write_json(result_path, result)
            pair_results.append(result)
            print(f"source {source_seed} -> target {target_seed} complete", flush=True)
            if _implementation_digest() != implementation:
                raise RuntimeError("analysis implementation changed during campaign")

    gate_names = (
        "metric_contract",
        "continuous_target_control_contract",
        "task_metric_coordinate_transport",
        "task_metric_causal_transport",
        "task_metric_dominates_euclidean_baselines",
        "task_metric_shuffled_specificity",
        "predecessor_replay_contract",
    )
    gate_counts = {
        name: sum(bool(result["gates"][name]) for result in pair_results)
        for name in gate_names
    }
    required_pairs = len(config.seeds) * (len(config.seeds) - 1)
    confirmed = bool(
        not config.allow_underpowered
        and required_pairs == 6
        and all(gate_counts[name] == required_pairs for name in gate_names)
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "cohort_seeds": COHORT_SEEDS,
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "cpu"
            ),
        },
        "summary": {
            "requested": required_pairs,
            "scheduled": required_pairs,
            "completed": len(pair_results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_new_coordinate_maps": required_pairs * 2,
            "evaluated_coordinate_maps": required_pairs * 5,
        },
        "aggregates": {
            "gate_counts": gate_counts,
            "required_pairs": required_pairs,
            "confirmed": confirmed,
            "conclusion": (
                "systems_lifecycle_only_not_quality_evidence"
                if config.allow_underpowered
                else (
                    "confirmed_group_task_metric_transport"
                    if confirmed
                    else "not_confirmed_group_task_metric_transport"
                )
            ),
            "per_pair": {
                f"{result['source_seed']}->{result['target_seed']}": result["gates"]
                for result in pair_results
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "source_seed": result["source_seed"],
                "target_seed": result["target_seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
                "primary_metric": result["primary_metric"],
                "gates": result["gates"],
            }
            for result in pair_results
        ],
        "method_boundaries": [
            "Only three previously selected stable C2 block-0 checkpoints are tested.",
            "The task metric is label-free but requires paired target-checkpoint access on source-fit orbits.",
            "The intervention is an off-manifold frozen residual patch.",
            "A successful map would be a diagnostic gauge fixing, not an independently computable encoder.",
            "The frozen discrete scalar calibration is secondary because its seed-7 fresh-cohort control is already unstable.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "source_*_target_*" / "result.json"),
        },
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_group_task_metric_transport/"
            "20260806_d6_preregistered"
        ),
    )
    parser.add_argument("--source-root", default=GroupTaskMetricTransportConfig.source_root)
    parser.add_argument(
        "--character-root", default=GroupTaskMetricTransportConfig.character_root
    )
    parser.add_argument("--readout-root", default=GroupTaskMetricTransportConfig.readout_root)
    parser.add_argument(
        "--transport-root", default=GroupTaskMetricTransportConfig.transport_root
    )
    parser.add_argument("--seeds", default=",".join(map(str, PRIMARY_SEEDS)))
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = GroupTaskMetricTransportConfig(
        source_root=args.source_root,
        character_root=args.character_root,
        readout_root=args.readout_root,
        transport_root=args.transport_root,
        seeds=_ints(args.seeds),
        orbit_count=args.orbit_count,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
