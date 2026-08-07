#!/usr/bin/env python3
"""Resolve quotient curvature versus invariant-context-conditioned writing."""

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
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_fixed_gauge_error_decomposition as predecessor
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-frozen-writer-capacity.v1"
HYPOTHESIS_ID = "tinyllm-c2-frozen-writer-capacity-v1"
PREDECESSOR_CAMPAIGN_SHA256 = (
    "be4cf87248550c8ace7fc474efff2ce168bce3c9002932ecf6063977c91f0fa6"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "37175d78dd4768e448bea482a271309e67f00863ffa6202a799346fad008ac38"
)
LOW_ORDERS = (1, 2, 3, 4)
MATCHED_ORDERS = (6, 10, 14, 18)
ALL_FOURIER_ORDERS = tuple(sorted(set(LOW_ORDERS + MATCHED_ORDERS)))
MATCHED_ORDER_BY_CONTEXT = dict(zip(LOW_ORDERS, MATCHED_ORDERS))


@dataclass(frozen=True)
class FrozenWriterCapacityConfig:
    predecessor_root: str = (
        "data/experiments/tinyllm_fixed_gauge_error_decomposition/"
        "20260806_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    low_orders: tuple[int, ...] = LOW_ORDERS
    matched_orders: tuple[int, ...] = MATCHED_ORDERS
    context_rank: int = 3
    writer_ridge: float = 1e-6
    context_scale_floor: float = 1e-8
    specificity_margin_bins: float = 0.125
    replay_tolerance: float = 1e-6
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary diagnostic seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64 and not self.allow_underpowered:
            raise ValueError("primary diagnostic fixes 64 exact orbits per cell")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if tuple(self.low_orders) != LOW_ORDERS:
            raise ValueError("low-order ladder is preregistered as 1,2,3,4")
        if tuple(self.matched_orders) != MATCHED_ORDERS:
            raise ValueError("capacity controls are preregistered as 6,10,14,18")
        if self.context_rank != 3:
            raise ValueError("the invariant context rank is preregistered as three")
        if min(
            self.writer_ridge,
            self.context_scale_floor,
            self.specificity_margin_bins,
            self.replay_tolerance,
        ) <= 0.0:
            raise ValueError("all numerical thresholds must be positive")


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
        Path(predecessor.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
        Path(transport.rank.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _evidence_role(config: FrozenWriterCapacityConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
    )


def fourier_features(theta: torch.Tensor, order: int) -> torch.Tensor:
    """Return [1, cos(theta), sin(theta), ..., cos(m theta), sin(m theta)]."""
    if theta.ndim != 1 or order < 1:
        raise ValueError("theta must be one-dimensional and order positive")
    values = [torch.ones_like(theta)]
    for harmonic in range(1, order + 1):
        values.extend((torch.cos(harmonic * theta), torch.sin(harmonic * theta)))
    return torch.stack(values, dim=-1).double()


def context_features(
    theta: torch.Tensor, context: torch.Tensor, order: int
) -> torch.Tensor:
    """Tensor-product Fourier features with an explicit constant context term."""
    if context.ndim != 2 or context.shape[0] != theta.shape[0]:
        raise ValueError("context must have one row per quotient angle")
    phase = fourier_features(theta, order)
    augmented = torch.cat(
        (
            torch.ones(
                (len(context), 1), dtype=context.dtype, device=context.device
            ),
            context.double(),
        ),
        dim=1,
    )
    return torch.einsum("ni,nj->nij", phase, augmented).reshape(len(theta), -1)


def fit_ridge_writer(
    features: torch.Tensor, coordinates: torch.Tensor, ridge: float
) -> dict[str, torch.Tensor]:
    """Fit a no-intercept ridge map; feature maps carry their own constant."""
    features = features.double()
    coordinates = coordinates.double()
    if (
        features.ndim != 2
        or coordinates.ndim != 2
        or len(features) != len(coordinates)
        or ridge <= 0.0
    ):
        raise ValueError("invalid ridge-writer inputs")
    identity = torch.eye(
        features.shape[1], dtype=features.dtype, device=features.device
    )
    linear = torch.linalg.solve(
        features.T @ features + ridge * identity,
        features.T @ coordinates,
    )
    return {
        "linear": linear,
        "intercept": torch.zeros(
            coordinates.shape[1], dtype=coordinates.dtype, device=coordinates.device
        ),
    }


def _canonicalize_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    result = basis.clone()
    for row in range(len(result)):
        pivot = int(torch.argmax(result[row].abs()))
        if float(result[row, pivot]) < 0.0:
            result[row].neg_()
    return result


def fit_invariant_context(
    propagated: torch.Tensor, rank: int, scale_floor: float
) -> dict[str, torch.Tensor]:
    """Fit a source-only PCA chart of the propagated orbit barycenter."""
    matrix = propagated.reshape(len(propagated), -1).double()
    center = matrix.mean(0)
    centered = matrix - center
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    if len(singular) < rank or float(singular[rank - 1]) <= scale_floor:
        raise ValueError("propagated context has insufficient numerical rank")
    basis = _canonicalize_basis_signs(right[:rank])
    raw = centered @ basis.T
    scale = raw.std(0, unbiased=False).clamp_min(scale_floor)
    return {
        "center": center,
        "basis": basis,
        "scale": scale,
        "singular_values": singular,
    }


def apply_invariant_context(
    propagated: torch.Tensor, model: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    matrix = propagated.reshape(len(propagated), -1).double()
    return ((matrix - model["center"]) @ model["basis"].T) / model["scale"]


def _context_summary(model: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    basis = model["basis"]
    singular = model["singular_values"]
    total = singular.square().sum().clamp_min(1e-24)
    orthogonality = torch.linalg.matrix_norm(
        basis @ basis.T
        - torch.eye(len(basis), dtype=basis.dtype, device=basis.device)
    )
    return {
        "rank": int(len(basis)),
        "input_features": int(basis.shape[1]),
        "scale": model["scale"].detach().cpu().tolist(),
        "minimum_scale": float(model["scale"].min()),
        "orthogonality_error": float(orthogonality),
        "cumulative_energy_fraction": float(singular[: len(basis)].square().sum() / total),
        "model_sha256": _tensor_digest(
            model["center"], model["basis"], model["scale"]
        ),
    }


def _embed_mapping(
    mapping: Mapping[str, torch.Tensor], offset: int, total_features: int
) -> dict[str, torch.Tensor]:
    native = mapping["linear"]
    if offset < 0 or offset + native.shape[0] > total_features:
        raise ValueError("mapping does not fit in the combined feature chart")
    linear = torch.zeros(
        (total_features, native.shape[1]), dtype=native.dtype, device=native.device
    )
    linear[offset : offset + native.shape[0]] = native
    return {"linear": linear, "intercept": mapping["intercept"]}


def _writer_summary(
    heldout: Sequence[Mapping[str, Any]],
    name: str,
    shuffled_name: str,
    specificity_margin_bins: float,
) -> dict[str, Any]:
    true_records = [cell["states"][name]["continuous"] for cell in heldout]
    shuffled_records = [
        cell["states"][shuffled_name]["continuous"] for cell in heldout
    ]
    checkpoint_pass = all(record["continuous_pass"] for record in true_records)
    shuffled_any_failure = any(
        not record["continuous_pass"] for record in shuffled_records
    )
    true_mean = sum(
        record["mean_moment_shift_bins"] for record in true_records
    ) / len(true_records)
    shuffled_mean = sum(
        record["mean_moment_shift_bins"] for record in shuffled_records
    ) / len(shuffled_records)
    specificity = bool(
        shuffled_any_failure
        and true_mean + specificity_margin_bins <= shuffled_mean
    )
    return {
        "checkpoint_pass": checkpoint_pass,
        "passing_cells": sum(record["continuous_pass"] for record in true_records),
        "aggregate_mean_shift_bins": true_mean,
        "aggregate_p95_shift_bins": sum(
            record["p95_moment_shift_bins"] for record in true_records
        )
        / len(true_records),
        "shuffled_any_failure": shuffled_any_failure,
        "shuffled_aggregate_mean_shift_bins": shuffled_mean,
        "specificity": specificity,
        "complete_pass": bool(checkpoint_pass and specificity),
    }


def classify_checkpoint(
    *,
    valid: bool,
    summaries: Mapping[str, Mapping[str, Any]],
    specificity_margin_bins: float,
) -> tuple[str, str | None]:
    if not valid:
        return "invalid", None
    for order in LOW_ORDERS:
        name = f"fourier_m{order:02d}"
        if summaries[name]["complete_pass"]:
            return "low_order_curvature_sufficient", name
    for order in MATCHED_ORDERS:
        name = f"fourier_m{order:02d}"
        if summaries[name]["complete_pass"]:
            return "high_order_quotient_capacity_sufficient", name
    for order in LOW_ORDERS:
        context_name = f"context_m{order:02d}"
        matched_name = f"fourier_m{MATCHED_ORDER_BY_CONTEXT[order]:02d}"
        if (
            summaries[context_name]["complete_pass"]
            and not summaries[matched_name]["complete_pass"]
        ):
            return "invariant_context_required", context_name
    baseline = summaries["fourier_m01"]["aggregate_mean_shift_bins"]
    candidates = [
        (summaries[f"context_m{order:02d}"]["aggregate_mean_shift_bins"], order)
        for order in LOW_ORDERS
    ]
    best_mean, best_order = min(candidates)
    if best_mean + specificity_margin_bins <= baseline:
        return "context_helpful_not_decisive", f"context_m{best_order:02d}"
    return "small_writer_insufficient", None


def _load_predecessor(
    config: FrozenWriterCapacityConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    campaign_path = Path(config.predecessor_root) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    if (
        _sha256(campaign_path) != PREDECESSOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != predecessor.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != predecessor.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != PREDECESSOR_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 3
    ):
        raise ValueError(f"invalid predecessor campaign {campaign_path}")
    details: dict[int, tuple[dict[str, Any], Path]] = {}
    for entry in campaign["results"]:
        path = Path(entry["path"])
        detail = json.loads(path.read_text())
        if (
            _sha256(path) != entry["result_sha256"]
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "writer_or_carrier_limited"
        ):
            raise ValueError(f"invalid predecessor result {path}")
        details[int(detail["seed"])] = (detail, path)
    if not set(config.seeds).issubset(details):
        raise ValueError("predecessor lacks a requested checkpoint")
    return campaign, campaign_path, details


def _fingerprint(
    config: FrozenWriterCapacityConfig,
    seed: int,
    predecessor_campaign_sha256: str,
    predecessor_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "cohort_seeds": fixed.COHORT_SEEDS,
        "seed": seed,
        "predecessor_campaign_sha256": predecessor_campaign_sha256,
        "predecessor_result_sha256": predecessor_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _mapping_record(
    mapping: Mapping[str, torch.Tensor],
    fit_features: torch.Tensor,
    fit_coordinates: torch.Tensor,
) -> dict[str, Any]:
    predicted = transport.apply_affine(fit_features, mapping)
    return {
        "feature_count": int(fit_features.shape[1]),
        "fitted_scalar_parameters": int(mapping["linear"].numel()),
        "linear": mapping["linear"].detach().cpu().tolist(),
        "intercept": mapping["intercept"].detach().cpu().tolist(),
        "fit_coordinate_metrics": transport.coordinate_metrics(
            predicted, fit_coordinates
        ),
    }


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: FrozenWriterCapacityConfig,
    implementation: str,
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("summary", {}).get("completed", -1)) == len(config.seeds)
        and len(value.get("results", [])) == len(config.seeds)
        and all(
            Path(item.get("path", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            for item in value.get("results", [])
        )
    )


@torch.no_grad()
def run_campaign(
    config: FrozenWriterCapacityConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text())
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    predecessor_campaign, predecessor_path, predecessor_details = _load_predecessor(
        config
    )
    base_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    transport_config = fixed._transport_config(base_config)
    rank_config = transport._rank_config(transport_config)
    bridge = transport.rank._bridge_config(rank_config)
    datasets = {
        cohort: {
            regime: transport.rank.deck.generate_exact_orbits(
                task,
                k=2,
                orbit_count=config.orbit_count,
                seed=fixed.COHORT_SEEDS[cohort][regime],
                regime=regime,
            )
            for regime in fixed.REGIMES
        }
        for cohort in fixed.COHORT_SEEDS
    }

    results = []
    max_fourier_order = max(ALL_FOURIER_ORDERS)
    max_context_order = max(LOW_ORDERS)
    fourier_width = 1 + 2 * max_fourier_order
    context_width = (1 + 2 * max_context_order) * (1 + config.context_rank)
    combined_width = fourier_width + context_width

    for seed in config.seeds:
        prior, prior_path = predecessor_details[seed]
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        if provenance["checkpoint_sha256"] != prior["provenance"][
            "checkpoint_sha256"
        ]:
            raise ValueError(f"checkpoint mismatch for seed {seed}")
        frozen, frozen_path = transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        basis, basis_summary = transport._fit_seed_basis(
            system,
            task,
            transport_config,
            rank_config,
            bridge,
            frozen,
            seed,
            device,
        )
        readout_predecessor = transport._load_readout_predecessor(
            transport_config, seed, provenance["checkpoint_sha256"]
        )
        cells = {
            cohort: {
                regime: transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    datasets[cohort][regime],
                    cohort,
                    regime,
                    device,
                )
                for regime in fixed.REGIMES
            }
            for cohort in fixed.COHORT_SEEDS
        }

        fit_coordinates = torch.cat(
            [
                transport._coordinates(cells["alignment_fit"][regime], basis)
                for regime in fixed.REGIMES
            ]
        )
        fit_theta = torch.cat(
            [
                datasets["alignment_fit"][regime]
                .quotient_phase.to(device)
                .reshape(config.orbit_count, 2)[:, 0]
                .double()
                for regime in fixed.REGIMES
            ]
        )
        fit_propagated = torch.cat(
            [cells["alignment_fit"][regime]["propagated"] for regime in fixed.REGIMES]
        )
        context_model = fit_invariant_context(
            fit_propagated, config.context_rank, config.context_scale_floor
        )
        context_summary = _context_summary(context_model)
        fit_context = apply_invariant_context(fit_propagated, context_model)
        permutation = fixed.regime_preserving_writer_permutation(
            config.orbit_count, seed
        ).to(device)

        native_mappings: dict[str, dict[str, torch.Tensor]] = {}
        mapping_records: dict[str, dict[str, Any]] = {}
        embedded_mappings: dict[str, dict[str, torch.Tensor]] = {}
        for order in ALL_FOURIER_ORDERS:
            features = fourier_features(fit_theta, order)
            for suffix, targets in (
                ("", fit_coordinates),
                ("_shuffled", fit_coordinates[permutation]),
            ):
                name = f"fourier_m{order:02d}{suffix}"
                mapping = fit_ridge_writer(features, targets, config.writer_ridge)
                native_mappings[name] = mapping
                mapping_records[name] = _mapping_record(
                    mapping, features, fit_coordinates
                )
                embedded_mappings[name] = _embed_mapping(
                    mapping, 0, combined_width
                )
        for order in LOW_ORDERS:
            features = context_features(fit_theta, fit_context, order)
            for suffix, targets in (
                ("", fit_coordinates),
                ("_shuffled", fit_coordinates[permutation]),
            ):
                name = f"context_m{order:02d}{suffix}"
                mapping = fit_ridge_writer(features, targets, config.writer_ridge)
                native_mappings[name] = mapping
                mapping_records[name] = _mapping_record(
                    mapping, features, fit_coordinates
                )
                embedded_mappings[name] = _embed_mapping(
                    mapping, fourier_width, combined_width
                )

        heldout = []
        replay_errors = []
        prior_by_cell = {
            (cell["cohort"], cell["regime"]): cell
            for cell in prior["heldout_cells"]
        }
        for cohort in fixed.HELDOUT_COHORTS:
            for regime in fixed.REGIMES:
                dataset = datasets[cohort][regime]
                theta = (
                    dataset.quotient_phase.to(device)
                    .reshape(config.orbit_count, 2)[:, 0]
                    .double()
                )
                local_context = apply_invariant_context(
                    cells[cohort][regime]["propagated"], context_model
                )
                combined = torch.cat(
                    (
                        fourier_features(theta, max_fourier_order),
                        context_features(theta, local_context, max_context_order),
                    ),
                    dim=1,
                )
                evaluated = transport._evaluate_transport_cell(
                    system,
                    task,
                    transport_config,
                    {"full_defect": combined},
                    cells[cohort][regime],
                    torch.eye(combined_width, dtype=torch.float64, device=device),
                    basis,
                    embedded_mappings,
                    readout_predecessor["rotation_bins"],
                )
                prior_cell = prior_by_cell[(cohort, regime)]
                replay = max(
                    predecessor._numeric_max_difference(
                        evaluated["coordinate_metrics"]["fourier_m01"],
                        prior_cell["oracle_evaluation"]["coordinate_metrics"][
                            "oracle_fit_oracle_eval"
                        ],
                    ),
                    predecessor._numeric_max_difference(
                        evaluated["states"]["fourier_m01"]["continuous"],
                        prior_cell["oracle_evaluation"]["states"][
                            "oracle_fit_oracle_eval"
                        ]["continuous"],
                    ),
                )
                replay_errors.append(replay)
                evaluated["predecessor_replay_maximum_absolute_error"] = replay
                heldout.append(evaluated)

        summaries = {}
        for order in ALL_FOURIER_ORDERS:
            name = f"fourier_m{order:02d}"
            summaries[name] = _writer_summary(
                heldout,
                name,
                f"{name}_shuffled",
                config.specificity_margin_bins,
            )
            summaries[name]["order"] = order
            summaries[name]["feature_count"] = 1 + 2 * order
            summaries[name]["family"] = "quotient_only_fourier"
        for order in LOW_ORDERS:
            name = f"context_m{order:02d}"
            summaries[name] = _writer_summary(
                heldout,
                name,
                f"{name}_shuffled",
                config.specificity_margin_bins,
            )
            summaries[name]["order"] = order
            summaries[name]["feature_count"] = (1 + 2 * order) * (
                1 + config.context_rank
            )
            summaries[name]["matched_fourier_order"] = MATCHED_ORDER_BY_CONTEXT[
                order
            ]
            summaries[name]["family"] = "quotient_times_invariant_context"

        replay_pass = max(replay_errors) <= config.replay_tolerance
        context_contract = bool(
            context_summary["rank"] == config.context_rank
            and context_summary["minimum_scale"] > config.context_scale_floor
            and context_summary["orthogonality_error"] <= 1e-6
            and torch.isfinite(fit_context).all()
        )
        controls_pass = all(
            not cell["states"]["zero"]["continuous"]["continuous_pass"]
            and cell["states"]["exact"]["continuous"]["continuous_pass"]
            and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        classification, selected = classify_checkpoint(
            valid=bool(replay_pass and context_contract and controls_pass),
            summaries=summaries,
            specificity_margin_bins=config.specificity_margin_bins,
        )
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(predecessor_path),
            _sha256(prior_path),
            provenance["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-frozen-writer-capacity-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "predecessor_campaign": str(predecessor_path),
                "predecessor_campaign_sha256": _sha256(predecessor_path),
                "predecessor_result": str(prior_path),
                "predecessor_result_sha256": _sha256(prior_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "frontend_checkpoint": provenance["frontend_checkpoint"],
                "frontend_checkpoint_sha256": provenance[
                    "frontend_checkpoint_sha256"
                ],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
                "readout_result": readout_predecessor["result"],
                "readout_result_sha256": readout_predecessor["result_sha256"],
            },
            "basis": basis_summary,
            "invariant_context": context_summary,
            "alignment_fit": {
                "permutation_sha256": hashlib.sha256(
                    permutation.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "mappings": mapping_records,
            },
            "heldout_cells": heldout,
            "writer_summaries": summaries,
            "gates": {
                "predecessor_replay_contract": replay_pass,
                "maximum_predecessor_replay_error": max(replay_errors),
                "invariant_context_numerical_contract": context_contract,
                "continuous_target_control_contract": controls_pass,
            },
            "classification": classification,
            "selected_writer": selected,
            "primary_metric": float(classification != "invalid"),
            "analysis_seconds": time.perf_counter() - started,
            "artifacts": {"result": str(result_path)},
        }
        _write_json(result_path, result)
        results.append(result)
        print(
            f"seed {seed}: {classification}"
            + (f" ({selected})" if selected else ""),
            flush=True,
        )
        if _implementation_digest() != implementation:
            raise RuntimeError("writer-capacity implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    valid = all(name != "invalid" for name in classifications)
    common = classifications[0] if len(set(classifications)) == 1 else None
    conclusion = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else (
            common
            if valid and common is not None
            else (
                "checkpoint_stratified_writer_mechanism"
                if valid
                else "invalid_campaign"
            )
        )
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "cohort_seeds": fixed.COHORT_SEEDS,
        "implementation_sha256": implementation,
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
            "predecessor_campaign": str(predecessor_path),
            "predecessor_campaign_sha256": _sha256(predecessor_path),
            "predecessor_implementation_sha256": predecessor_campaign[
                "implementation_sha256"
            ],
        },
        "summary": {
            "requested": len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_writers": len(config.seeds)
            * 2
            * (len(ALL_FOURIER_ORDERS) + len(LOW_ORDERS)),
        },
        "aggregates": {
            "conclusion": conclusion,
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
            },
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "selected_writer_by_seed": {
                str(result["seed"]): result["selected_writer"] for result in results
            },
            "gate_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in (
                    "predecessor_replay_contract",
                    "invariant_context_numerical_contract",
                    "continuous_target_control_contract",
                )
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "classification": result["classification"],
                "selected_writer": result["selected_writer"],
                "gates": result["gates"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The exact quotient angle uses latent phase and is not deployable.",
            "PCA and writers use only the locked alignment-fit cells.",
            "The propagated-barycenter context is invariant to sheet permutation but checkpoint local.",
            "The four held-out cells were reused by preceding post-outcome diagnostics.",
            "This three-checkpoint diagnostic does not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
        },
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
            "data/experiments/tinyllm_frozen_writer_capacity/"
            "20260807_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = FrozenWriterCapacityConfig(
        seeds=args.seeds,
        orbit_count=args.orbit_count,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
