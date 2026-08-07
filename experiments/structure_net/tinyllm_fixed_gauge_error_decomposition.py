#!/usr/bin/env python3
"""Decompose fixed-gauge failures into sensor versus writer limitations."""

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

import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-fixed-gauge-error-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-c2-fixed-gauge-error-decomposition-v1"
PRIMARY_CAMPAIGN_SHA256 = (
    "de80e30c23e06801c75d6fae899c67d0da82b86fdaff9158d94270597df8379c"
)
PRIMARY_IMPLEMENTATION_SHA256 = (
    "4508847ca4fa85c2220e3691d8e9922d714e768b0f4891ce5237a4a74089c808"
)


@dataclass(frozen=True)
class FixedGaugeErrorDecompositionConfig:
    primary_root: str = (
        "data/experiments/tinyllm_fixed_semantic_gauge_writer/"
        "20260806_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    oracle_mean_shift_ceiling_bins: float = 1e-8
    oracle_p95_shift_ceiling_bins: float = 1e-8
    replay_tolerance: float = 1e-6
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary diagnostic seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.replay_tolerance <= 0.0:
            raise ValueError("replay tolerance must be positive")


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
    for path in (Path(__file__), Path(fixed.__file__)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: FixedGaugeErrorDecompositionConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
    )


def oracle_orbit_carrier(
    dataset: Any, task: CircleTaskConfig, orbit_count: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    phase = dataset.phase.to(device).reshape(orbit_count, 2)[:, 0].double()
    angle = 2.0 * phase
    carrier = torch.stack(
        (torch.cos(angle), torch.sin(angle), torch.ones_like(angle)), dim=-1
    )
    predicted = torch.atan2(carrier[:, 1], carrier[:, 0])
    difference = torch.atan2(
        torch.sin(predicted - angle), torch.cos(predicted - angle)
    )
    shifts = difference.abs() / (2.0 * math.pi / task.phase_bins)
    return carrier, {
        "circular_alignment": float(torch.cos(difference).mean()),
        "mean_shift_bins": float(shifts.mean()),
        "p95_shift_bins": float(torch.quantile(shifts, 0.95)),
        "maximum_shift_bins": float(shifts.max()),
        "maximum_energy_deviation": float((carrier[:, 2] - 1.0).abs().max()),
    }


def _primary_writer(value: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    mapping = value["alignment_fit"]["writer"]
    return {
        "linear": torch.tensor(mapping["linear"], dtype=torch.float64, device=device),
        "intercept": torch.tensor(
            mapping["intercept"], dtype=torch.float64, device=device
        ),
    }


def _numeric_max_difference(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        common = set(left).intersection(right)
        return max(
            (_numeric_max_difference(left[key], right[key]) for key in common),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if bool(left) == bool(right) else float("inf")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def replay_error(
    current_cell: Mapping[str, Any], primary_cell: Mapping[str, Any]
) -> float:
    return max(
        _numeric_max_difference(
            current_cell["coordinate_metrics"]["observed_primary"],
            primary_cell["coordinate_metrics"]["fixed_gauge"],
        ),
        _numeric_max_difference(
            current_cell["states"]["observed_primary"]["continuous"],
            primary_cell["states"]["fixed_gauge"]["continuous"],
        ),
    )


def classify_checkpoint(
    *,
    replay_pass: bool,
    oracle_contract: bool,
    controls_pass: bool,
    observed_fit_oracle_pass: bool,
    oracle_fit_oracle_pass: bool,
) -> str:
    if not replay_pass or not oracle_contract or not controls_pass:
        return "invalid"
    if observed_fit_oracle_pass and oracle_fit_oracle_pass:
        return "sensor_limited"
    if oracle_fit_oracle_pass:
        return "sensor_and_fit_mismatch"
    return "writer_or_carrier_limited"


def _load_primary(
    config: FixedGaugeErrorDecompositionConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    campaign_path = Path(config.primary_root) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text())
    if (
        _sha256(campaign_path) != PRIMARY_CAMPAIGN_SHA256
        or campaign.get("schema_version") != fixed.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != fixed.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != PRIMARY_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 3
    ):
        raise ValueError(f"invalid primary fixed-gauge campaign {campaign_path}")
    details: dict[int, tuple[dict[str, Any], Path]] = {}
    for entry in campaign["results"]:
        path = Path(entry["path"])
        detail = json.loads(path.read_text())
        if (
            _sha256(path) != entry["result_sha256"]
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
        ):
            raise ValueError(f"invalid primary fixed-gauge result {path}")
        details[int(detail["seed"])] = (detail, path)
    if not set(config.seeds).issubset(details):
        raise ValueError("primary campaign lacks a requested checkpoint")
    return campaign, campaign_path, details


def _fingerprint(
    config: FixedGaugeErrorDecompositionConfig,
    seed: int,
    primary_campaign_sha256: str,
    primary_result_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "cohort_seeds": fixed.COHORT_SEEDS,
        "seed": seed,
        "primary_campaign_sha256": primary_campaign_sha256,
        "primary_result_sha256": primary_result_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@torch.no_grad()
def run_campaign(
    config: FixedGaugeErrorDecompositionConfig, output: Path
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
            and existing.get("configuration")
            == json.loads(json.dumps(asdict(config)))
            and all(
                Path(entry["path"]).is_file()
                and _sha256(Path(entry["path"])) == entry["result_sha256"]
                for entry in existing.get("results", [])
            )
        ):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    primary_campaign, primary_campaign_path, primary_details = _load_primary(config)
    base_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
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

    results = []
    for seed in config.seeds:
        primary, primary_path = primary_details[seed]
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        if provenance["checkpoint_sha256"] != primary["provenance"][
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
        observed_carriers = {}
        oracle_carriers = {}
        oracle_audit = {}
        for cohort in fixed.COHORT_SEEDS:
            observed_carriers[cohort] = {}
            oracle_carriers[cohort] = {}
            oracle_audit[cohort] = {}
            for regime in fixed.REGIMES:
                observed, _ = fixed.semantic_orbit_carrier(
                    datasets[cohort][regime], task, base_config.orbit_count, device
                )
                oracle, audit = oracle_orbit_carrier(
                    datasets[cohort][regime], task, base_config.orbit_count, device
                )
                observed_carriers[cohort][regime] = observed
                oracle_carriers[cohort][regime] = oracle
                oracle_audit[cohort][regime] = audit

        fit_coordinates = torch.cat(
            [
                transport._coordinates(cells["alignment_fit"][regime], basis)
                for regime in fixed.REGIMES
            ]
        )
        fit_oracle = torch.cat(
            [oracle_carriers["alignment_fit"][regime] for regime in fixed.REGIMES]
        )
        observed_writer = _primary_writer(primary, device)
        oracle_writer = fixed.fit_linear_writer(
            fit_oracle, fit_coordinates, base_config.writer_ridge
        )
        heldout = []
        replay_errors = []
        primary_by_cell = {
            (cell["cohort"], cell["regime"]): cell
            for cell in primary["heldout_cells"]
        }
        for cohort in fixed.HELDOUT_COHORTS:
            for regime in fixed.REGIMES:
                observed_eval = transport._evaluate_transport_cell(
                    system,
                    task,
                    transport_config,
                    {"full_defect": observed_carriers[cohort][regime]},
                    cells[cohort][regime],
                    torch.eye(3, dtype=torch.float64, device=device),
                    basis,
                    {
                        "observed_primary": observed_writer,
                        "oracle_fit_observed_eval": oracle_writer,
                    },
                    readout_predecessor["rotation_bins"],
                )
                oracle_eval = transport._evaluate_transport_cell(
                    system,
                    task,
                    transport_config,
                    {"full_defect": oracle_carriers[cohort][regime]},
                    cells[cohort][regime],
                    torch.eye(3, dtype=torch.float64, device=device),
                    basis,
                    {
                        "observed_fit_oracle_eval": observed_writer,
                        "oracle_fit_oracle_eval": oracle_writer,
                    },
                    readout_predecessor["rotation_bins"],
                )
                replay = replay_error(
                    observed_eval, primary_by_cell[(cohort, regime)]
                )
                replay_errors.append(replay)
                heldout.append(
                    {
                        "cohort": cohort,
                        "regime": regime,
                        "replay_maximum_absolute_error": replay,
                        "observed_evaluation": {
                            "coordinate_metrics": observed_eval[
                                "coordinate_metrics"
                            ],
                            "states": {
                                name: observed_eval["states"][name]
                                for name in (
                                    "zero",
                                    "exact",
                                    "direct_rank3",
                                    "observed_primary",
                                    "oracle_fit_observed_eval",
                                )
                            },
                        },
                        "oracle_evaluation": {
                            "coordinate_metrics": oracle_eval["coordinate_metrics"],
                            "states": {
                                name: oracle_eval["states"][name]
                                for name in (
                                    "zero",
                                    "exact",
                                    "direct_rank3",
                                    "observed_fit_oracle_eval",
                                    "oracle_fit_oracle_eval",
                                )
                            },
                        },
                    }
                )

        replay_pass = max(replay_errors) <= config.replay_tolerance
        oracle_contract_pass = all(
            audit["mean_shift_bins"] <= config.oracle_mean_shift_ceiling_bins
            and audit["p95_shift_bins"] <= config.oracle_p95_shift_ceiling_bins
            for cohort in fixed.HELDOUT_COHORTS
            for audit in oracle_audit[cohort].values()
        )
        controls_pass = all(
            not cell["oracle_evaluation"]["states"]["zero"]["continuous"][
                "continuous_pass"
            ]
            and cell["oracle_evaluation"]["states"]["exact"]["continuous"][
                "continuous_pass"
            ]
            and cell["oracle_evaluation"]["states"]["direct_rank3"][
                "continuous"
            ]["continuous_pass"]
            for cell in heldout
        )
        observed_fit_oracle_pass = all(
            cell["oracle_evaluation"]["states"]["observed_fit_oracle_eval"][
                "continuous"
            ]["continuous_pass"]
            for cell in heldout
        )
        oracle_fit_oracle_pass = all(
            cell["oracle_evaluation"]["states"]["oracle_fit_oracle_eval"][
                "continuous"
            ]["continuous_pass"]
            for cell in heldout
        )
        classification = classify_checkpoint(
            replay_pass=replay_pass,
            oracle_contract=oracle_contract_pass,
            controls_pass=controls_pass,
            observed_fit_oracle_pass=observed_fit_oracle_pass,
            oracle_fit_oracle_pass=oracle_fit_oracle_pass,
        )
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(primary_campaign_path),
            _sha256(primary_path),
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-fixed-gauge-error-decomposition-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "primary_campaign": str(primary_campaign_path),
                "primary_campaign_sha256": _sha256(primary_campaign_path),
                "primary_result": str(primary_path),
                "primary_result_sha256": _sha256(primary_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
            },
            "basis": basis_summary,
            "oracle_audit": oracle_audit,
            "alignment_fit": {
                "observed_writer": transport._mapping_summary(observed_writer),
                "oracle_writer": transport._mapping_summary(oracle_writer),
                "oracle_coordinate_metrics": transport.coordinate_metrics(
                    transport.apply_affine(fit_oracle, oracle_writer),
                    fit_coordinates,
                ),
            },
            "heldout_cells": heldout,
            "gates": {
                "primary_replay_contract": replay_pass,
                "maximum_primary_replay_error": max(replay_errors),
                "oracle_carrier_contract": oracle_contract_pass,
                "continuous_target_control_contract": controls_pass,
                "observed_fit_oracle_eval": observed_fit_oracle_pass,
                "oracle_fit_oracle_eval": oracle_fit_oracle_pass,
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
            raise RuntimeError("diagnostic implementation changed during campaign")

    classification_counts = {
        name: sum(result["classification"] == name for result in results)
        for name in (
            "sensor_limited",
            "sensor_and_fit_mismatch",
            "writer_or_carrier_limited",
            "invalid",
        )
    }
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
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
            "primary_campaign": str(primary_campaign_path),
            "primary_campaign_sha256": _sha256(primary_campaign_path),
            "primary_implementation_sha256": primary_campaign[
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
            "fitted_linear_writers": len(config.seeds),
        },
        "aggregates": {
            "classification_counts": classification_counts,
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "gate_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in (
                    "primary_replay_contract",
                    "oracle_carrier_contract",
                    "continuous_target_control_contract",
                    "observed_fit_oracle_eval",
                    "oracle_fit_oracle_eval",
                )
            },
            "conclusion": (
                "systems_lifecycle_only_not_quality_evidence"
                if config.allow_underpowered
                else (
                    "sensor_limited_all_checkpoints"
                    if classification_counts["sensor_limited"] == len(results)
                    else "mixed_sensor_and_writer_limitations"
                )
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
            "The oracle carrier uses latent phase and is not deployable.",
            "This post-outcome diagnostic reuses prior fit and held-out cells.",
            "Target-local linear writers retain paired target-checkpoint access.",
            "The result does not test a learned encoder or context-conditioned writer.",
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
            "data/experiments/tinyllm_fixed_gauge_error_decomposition/"
            "20260806_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = FixedGaugeErrorDecompositionConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
