#!/usr/bin/env python3
"""Decompose fixed-gauge writer failure into curvature versus context."""

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

import experiments.structure_net.tinyllm_fixed_gauge_error_decomposition as decomposition
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-fixed-gauge-writer-capacity.v1"
HYPOTHESIS_ID = "tinyllm-c2-fixed-gauge-writer-capacity-v1"
PREDECESSOR_CAMPAIGN_SHA256 = (
    "be4cf87248550c8ace7fc474efff2ce168bce3c9002932ecf6063977c91f0fa6"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "37175d78dd4768e448bea482a271309e67f00863ffa6202a799346fad008ac38"
)
ARM_NAMES = (
    "quotient_order1",
    "quotient_order2",
    "quotient_order4",
    "quotient_order40",
    "quotient_order4_context",
)
CANDIDATE_NAMES = ARM_NAMES[1:]


@dataclass(frozen=True)
class FixedGaugeWriterCapacityConfig:
    predecessor_root: str = (
        "data/experiments/tinyllm_fixed_gauge_error_decomposition/"
        "20260806_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    low_orders: tuple[int, ...] = (2, 4)
    high_order: int = 40
    writer_ridge: float = 1e-6
    context_std_floor: float = 1e-8
    specificity_margin_bins: float = 0.125
    replay_tolerance: float = 1e-6
    oracle_mean_shift_ceiling_bins: float = 1e-8
    oracle_p95_shift_ceiling_bins: float = 1e-8
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary writer-capacity seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.low_orders != (2, 4) or self.high_order != 40:
            raise ValueError("the preregistered Fourier ladder is fixed to 2,4,40")
        if min(self.writer_ridge, self.context_std_floor, self.replay_tolerance) <= 0:
            raise ValueError("ridge, standard-deviation floor, and tolerance must be positive")


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


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(decomposition.__file__),
        Path(fixed.__file__),
        Path(transport.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: FixedGaugeWriterCapacityConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
    )


def fourier_features(carrier: torch.Tensor, order: int) -> torch.Tensor:
    """Return cos/sin harmonics through ``order`` plus one constant channel."""
    if carrier.ndim != 2 or carrier.shape[1] < 2 or order < 1:
        raise ValueError("carrier must have two phase channels and order must be positive")
    angle = torch.atan2(carrier[:, 1].double(), carrier[:, 0].double())
    values = []
    for harmonic in range(1, order + 1):
        values.extend((torch.cos(harmonic * angle), torch.sin(harmonic * angle)))
    values.append(torch.ones_like(angle))
    return torch.stack(values, dim=-1)


def orbit_calibration_context(
    dataset: Any, orbit_count: int, device: torch.device
) -> tuple[torch.Tensor, float]:
    packet = dataset.calibration.to(device).reshape(orbit_count, 2, -1).double()
    sheet_difference = float((packet[:, 0] - packet[:, 1]).abs().max())
    return packet.mean(1), sheet_difference


def standardize_context(
    value: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (value.double() - mean.double()) / scale.double()


def conditional_features(
    quotient: torch.Tensor, standardized_context: torch.Tensor
) -> torch.Tensor:
    if quotient.shape[0] != standardized_context.shape[0]:
        raise ValueError("quotient and context row counts must match")
    one = torch.ones(
        (len(standardized_context), 1),
        dtype=standardized_context.dtype,
        device=standardized_context.device,
    )
    augmented = torch.cat((one, standardized_context), dim=1)
    return torch.einsum("ni,nj->nij", quotient.double(), augmented).reshape(
        len(quotient), -1
    )


def fit_writer(
    features: torch.Tensor, coordinates: torch.Tensor, ridge: float
) -> dict[str, torch.Tensor]:
    features = features.double()
    coordinates = coordinates.double()
    if features.ndim != 2 or coordinates.ndim != 2 or len(features) != len(coordinates):
        raise ValueError("features and coordinates must be row-aligned matrices")
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


def _writer_from_summary(
    value: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "linear": torch.tensor(value["linear"], dtype=torch.float64, device=device),
        "intercept": torch.tensor(
            value["intercept"], dtype=torch.float64, device=device
        ),
    }


def _pad_writer(
    writer: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
    total_width: int,
) -> dict[str, torch.Tensor]:
    linear = torch.zeros(
        (total_width, writer["linear"].shape[1]),
        dtype=writer["linear"].dtype,
        device=writer["linear"].device,
    )
    linear[start:stop] = writer["linear"]
    return {"linear": linear, "intercept": writer["intercept"]}


def classify_checkpoint(
    *,
    contracts_pass: bool,
    candidate_specific_pass: Mapping[str, bool],
) -> str:
    if not contracts_pass:
        return "invalid"
    if candidate_specific_pass.get("quotient_order2"):
        return "low_order_curvature_limited"
    if candidate_specific_pass.get("quotient_order4"):
        return "low_order_curvature_limited"
    if candidate_specific_pass.get("quotient_order40"):
        return "high_order_curvature_limited"
    if candidate_specific_pass.get("quotient_order4_context"):
        return "calibration_context_limited"
    return "unresolved_writer_limited"


def _load_predecessor(
    config: FixedGaugeWriterCapacityConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    path = Path(config.predecessor_root) / "campaign_results.json"
    campaign = json.loads(path.read_text())
    if (
        _sha256(path) != PREDECESSOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != decomposition.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != decomposition.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != PREDECESSOR_IMPLEMENTATION_SHA256
        or campaign.get("aggregates", {}).get("classification_counts", {}).get(
            "writer_or_carrier_limited"
        )
        != 3
    ):
        raise ValueError(f"invalid predecessor decomposition {path}")
    details: dict[int, tuple[dict[str, Any], Path]] = {}
    for entry in campaign["results"]:
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "writer_or_carrier_limited"
        ):
            raise ValueError(f"invalid predecessor result {detail_path}")
        details[int(detail["seed"])] = (detail, detail_path)
    if not set(config.seeds).issubset(details):
        raise ValueError("predecessor campaign lacks a requested checkpoint")
    return campaign, path, details


def _fingerprint(
    config: FixedGaugeWriterCapacityConfig,
    seed: int,
    predecessor_campaign_sha256: str,
    predecessor_result_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "cohort_seeds": fixed.COHORT_SEEDS,
        "seed": seed,
        "predecessor_campaign_sha256": predecessor_campaign_sha256,
        "predecessor_result_sha256": predecessor_result_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_gates(
    heldout: Sequence[Mapping[str, Any]],
    candidate: str,
    specificity_margin: float,
) -> dict[str, Any]:
    shuffled = f"{candidate}_shuffled"
    causal = all(
        cell["states"][candidate]["continuous"]["continuous_pass"]
        for cell in heldout
    )
    candidate_mean = sum(
        cell["states"][candidate]["continuous"]["mean_moment_shift_bins"]
        for cell in heldout
    ) / len(heldout)
    shuffled_mean = sum(
        cell["states"][shuffled]["continuous"]["mean_moment_shift_bins"]
        for cell in heldout
    ) / len(heldout)
    shuffled_fails = any(
        not cell["states"][shuffled]["continuous"]["continuous_pass"]
        for cell in heldout
    )
    specificity = bool(
        shuffled_fails and shuffled_mean - candidate_mean >= specificity_margin
    )
    return {
        "causal_all_cells": causal,
        "shuffled_specificity": specificity,
        "specific_causal_pass": bool(causal and specificity),
        "aggregate_mean_shift_bins": candidate_mean,
        "shuffled_aggregate_mean_shift_bins": shuffled_mean,
        "specificity_margin_bins": shuffled_mean - candidate_mean,
        "worst_coordinate_variance_explained": min(
            cell["coordinate_metrics"][candidate]["variance_explained"]
            for cell in heldout
        ),
    }


@torch.no_grad()
def run_campaign(
    config: FixedGaugeWriterCapacityConfig, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text())
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("hypothesis_id") == HYPOTHESIS_ID
            and existing.get("status") == "completed"
            and existing.get("implementation_sha256") == implementation
            and existing.get("configuration") == json.loads(json.dumps(asdict(config)))
            and all(
                Path(item["path"]).is_file()
                and _sha256(Path(item["path"])) == item["result_sha256"]
                for item in existing.get("results", [])
            )
        ):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    predecessor, predecessor_path, predecessor_details = _load_predecessor(config)
    base_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        writer_ridge=config.writer_ridge,
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
                orbit_count=base_config.orbit_count,
                seed=fixed.COHORT_SEEDS[cohort][regime],
                regime=regime,
            )
            for regime in fixed.REGIMES
        }
        for cohort in fixed.COHORT_SEEDS
    }

    oracle_carriers: dict[str, dict[str, torch.Tensor]] = {}
    oracle_audit: dict[str, dict[str, dict[str, float]]] = {}
    raw_contexts: dict[str, dict[str, torch.Tensor]] = {}
    context_sheet_difference: dict[str, dict[str, float]] = {}
    for cohort in fixed.COHORT_SEEDS:
        oracle_carriers[cohort] = {}
        oracle_audit[cohort] = {}
        raw_contexts[cohort] = {}
        context_sheet_difference[cohort] = {}
        for regime in fixed.REGIMES:
            carrier, audit = decomposition.oracle_orbit_carrier(
                datasets[cohort][regime], task, base_config.orbit_count, device
            )
            context, difference = orbit_calibration_context(
                datasets[cohort][regime], base_config.orbit_count, device
            )
            oracle_carriers[cohort][regime] = carrier
            oracle_audit[cohort][regime] = audit
            raw_contexts[cohort][regime] = context
            context_sheet_difference[cohort][regime] = difference

    fit_context = torch.cat(
        [raw_contexts["alignment_fit"][regime] for regime in fixed.REGIMES]
    )
    context_mean = fit_context.mean(0)
    context_scale = fit_context.std(0, unbiased=False).clamp_min(
        config.context_std_floor
    )
    feature_cells: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for cohort in fixed.COHORT_SEEDS:
        feature_cells[cohort] = {}
        for regime in fixed.REGIMES:
            carrier = oracle_carriers[cohort][regime]
            q1 = fourier_features(carrier, 1)
            q2 = fourier_features(carrier, 2)
            q4 = fourier_features(carrier, 4)
            q40 = fourier_features(carrier, 40)
            context = standardize_context(
                raw_contexts[cohort][regime], context_mean, context_scale
            )
            feature_cells[cohort][regime] = {
                "quotient_order1": q1,
                "quotient_order2": q2,
                "quotient_order4": q4,
                "quotient_order40": q40,
                "quotient_order4_context": conditional_features(q4, context),
            }

    results = []
    for seed in config.seeds:
        predecessor_detail, predecessor_detail_path = predecessor_details[seed]
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        if provenance["checkpoint_sha256"] != predecessor_detail["provenance"][
            "checkpoint_sha256"
        ]:
            raise ValueError(f"checkpoint mismatch for seed {seed}")
        frozen, _ = transport.rank._load_character_source(
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
        permutation = fixed.regime_preserving_writer_permutation(
            base_config.orbit_count, seed
        ).to(device)
        writers: dict[str, dict[str, torch.Tensor]] = {}
        fit_metrics = {}
        for arm in ARM_NAMES:
            features = torch.cat(
                [feature_cells["alignment_fit"][regime][arm] for regime in fixed.REGIMES]
            )
            writer = fit_writer(features, fit_coordinates, config.writer_ridge)
            writers[arm] = writer
            fit_metrics[arm] = transport.coordinate_metrics(
                transport.apply_affine(features, writer), fit_coordinates
            )
            if arm in CANDIDATE_NAMES:
                writers[f"{arm}_shuffled"] = fit_writer(
                    features, fit_coordinates[permutation], config.writer_ridge
                )

        predecessor_writer = _writer_from_summary(
            predecessor_detail["alignment_fit"]["oracle_writer"], device
        )
        q1_mapping_error = max(
            float(
                (
                    writers["quotient_order1"]["linear"]
                    - predecessor_writer["linear"]
                )
                .abs()
                .max()
            ),
            float(
                (
                    writers["quotient_order1"]["intercept"]
                    - predecessor_writer["intercept"]
                )
                .abs()
                .max()
            ),
        )
        writers["quotient_order1"] = predecessor_writer

        widths = {
            arm: feature_cells["alignment_fit"][fixed.REGIMES[0]][arm].shape[1]
            for arm in ARM_NAMES
        }
        offsets = {}
        cursor = 0
        for arm in ARM_NAMES:
            offsets[arm] = (cursor, cursor + widths[arm])
            cursor += widths[arm]
        padded = {}
        for name, writer in writers.items():
            arm = name.removesuffix("_shuffled")
            start, stop = offsets[arm]
            padded[name] = _pad_writer(writer, start, stop, cursor)

        predecessor_cells = {
            (cell["cohort"], cell["regime"]): cell
            for cell in predecessor_detail["heldout_cells"]
        }
        heldout = []
        replay_errors = []
        for cohort in fixed.HELDOUT_COHORTS:
            for regime in fixed.REGIMES:
                master = torch.cat(
                    [feature_cells[cohort][regime][arm] for arm in ARM_NAMES], dim=1
                )
                evaluated = transport._evaluate_transport_cell(
                    system,
                    task,
                    transport_config,
                    {"full_defect": master},
                    cells[cohort][regime],
                    torch.eye(cursor, dtype=torch.float64, device=device),
                    basis,
                    padded,
                    readout_predecessor["rotation_bins"],
                )
                predecessor_cell = predecessor_cells[(cohort, regime)]
                replay = max(
                    decomposition._numeric_max_difference(
                        evaluated["coordinate_metrics"]["quotient_order1"],
                        predecessor_cell["oracle_evaluation"]["coordinate_metrics"]
                        ["oracle_fit_oracle_eval"],
                    ),
                    decomposition._numeric_max_difference(
                        evaluated["states"]["quotient_order1"]["continuous"],
                        predecessor_cell["oracle_evaluation"]["states"]
                        ["oracle_fit_oracle_eval"]["continuous"],
                    ),
                )
                replay_errors.append(replay)
                evaluated["predecessor_replay_maximum_absolute_error"] = replay
                heldout.append(evaluated)

        replay_pass = bool(
            max(replay_errors) <= config.replay_tolerance
            and q1_mapping_error <= config.replay_tolerance
        )
        oracle_pass = all(
            oracle_audit[cohort][regime]["mean_shift_bins"]
            <= config.oracle_mean_shift_ceiling_bins
            and oracle_audit[cohort][regime]["p95_shift_bins"]
            <= config.oracle_p95_shift_ceiling_bins
            and context_sheet_difference[cohort][regime]
            <= config.replay_tolerance
            for cohort in fixed.COHORT_SEEDS
            for regime in fixed.REGIMES
        )
        controls_pass = all(
            not cell["states"]["zero"]["continuous"]["continuous_pass"]
            and cell["states"]["exact"]["continuous"]["continuous_pass"]
            and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
            for cell in heldout
        )
        candidate_gates = {
            arm: _candidate_gates(
                heldout, arm, config.specificity_margin_bins
            )
            for arm in CANDIDATE_NAMES
        }
        contracts = bool(replay_pass and oracle_pass and controls_pass)
        classification = classify_checkpoint(
            contracts_pass=contracts,
            candidate_specific_pass={
                arm: value["specific_causal_pass"]
                for arm, value in candidate_gates.items()
            },
        )
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(predecessor_path),
            _sha256(predecessor_detail_path),
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-fixed-gauge-writer-capacity-seed{seed}",
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
                "predecessor_result": str(predecessor_detail_path),
                "predecessor_result_sha256": _sha256(predecessor_detail_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
            },
            "basis": basis_summary,
            "feature_contract": {
                "arm_widths": widths,
                "total_evaluation_width": cursor,
                "context_mean": context_mean.detach().cpu().tolist(),
                "context_scale": context_scale.detach().cpu().tolist(),
                "maximum_context_sheet_difference": max(
                    difference
                    for cohort in context_sheet_difference.values()
                    for difference in cohort.values()
                ),
                "oracle_audit": oracle_audit,
            },
            "alignment_fit": {
                "coordinate_metrics": fit_metrics,
                "writers": {
                    name: transport._mapping_summary(writer)
                    for name, writer in writers.items()
                },
                "order1_predecessor_mapping_maximum_absolute_error": q1_mapping_error,
                "permutation_sha256": hashlib.sha256(
                    permutation.cpu().numpy().tobytes()
                ).hexdigest(),
            },
            "heldout_cells": heldout,
            "gates": {
                "predecessor_replay_contract": replay_pass,
                "maximum_predecessor_replay_error": max(replay_errors),
                "oracle_and_context_contract": oracle_pass,
                "continuous_target_control_contract": controls_pass,
                "candidates": candidate_gates,
            },
            "classification": classification,
            "primary_metric": float(classification != "invalid"),
            "analysis_seconds": time.perf_counter() - started,
            "artifacts": {"result": str(result_path)},
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("writer-capacity implementation changed during campaign")

    classifications = (
        "low_order_curvature_limited",
        "high_order_curvature_limited",
        "calibration_context_limited",
        "unresolved_writer_limited",
        "invalid",
    )
    classification_counts = {
        name: sum(result["classification"] == name for result in results)
        for name in classifications
    }
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
        },
        "provenance": {
            "predecessor_campaign": str(predecessor_path),
            "predecessor_campaign_sha256": _sha256(predecessor_path),
            "predecessor_implementation_sha256": predecessor[
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
            "fitted_linear_writers": len(config.seeds) * 9,
        },
        "aggregates": {
            "classification_counts": classification_counts,
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "contract_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in (
                    "predecessor_replay_contract",
                    "oracle_and_context_contract",
                    "continuous_target_control_contract",
                )
            },
            "candidate_specific_pass_counts": {
                arm: sum(
                    bool(result["gates"]["candidates"][arm]["specific_causal_pass"])
                    for result in results
                )
                for arm in CANDIDATE_NAMES
            },
            "conclusion": (
                "systems_lifecycle_only_not_quality_evidence"
                if config.allow_underpowered
                else "mechanistic_writer_capacity_decomposition"
            ),
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "classification": result["classification"],
                "gates": result["gates"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "Latent phase makes every quotient Fourier feature diagnostic rather than deployable.",
            "Calibration context is observed but writers retain target-local alignment-fit access.",
            "The order-40 control is a flexible parameter-count control, not a proposed architecture.",
            "Only three selected frozen checkpoints and reused held-out cells are tested.",
            "Off-manifold patch sufficiency does not establish natural use.",
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
            "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
            "20260806_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = FixedGaugeWriterCapacityConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
