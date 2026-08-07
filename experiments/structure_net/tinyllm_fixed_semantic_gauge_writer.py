#!/usr/bin/env python3
"""Test a fixed observation-derived C2 carrier with local frozen-model writers."""

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

import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
import experiments.structure_net.tinyllm_degree_k_ladder as ladder
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-fixed-semantic-gauge-writer.v1"
HYPOTHESIS_ID = "tinyllm-c2-fixed-semantic-gauge-writer-v1"
PRIMARY_SEEDS = transport.PRIMARY_SEEDS
REGIMES = transport.REGIMES
HELDOUT_COHORTS = transport.HELDOUT_COHORTS
COHORT_SEEDS = transport.COHORT_SEEDS
PREDECESSOR_CAMPAIGN_SHA256 = (
    "44707fd4bcd810e63614671aa491095fae735ee52359d464ab25abb10a2bc228"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "798a85b50f7dc8489e28cc66a498a5c3a8af193f859b80b43b1c5cc551ba0a89"
)


@dataclass(frozen=True)
class FixedSemanticGaugeWriterConfig:
    source_root: str = transport.CrossSeedCarrierTransportConfig.source_root
    character_root: str = transport.CrossSeedCarrierTransportConfig.character_root
    readout_root: str = transport.CrossSeedCarrierTransportConfig.readout_root
    predecessor_root: str = (
        "data/experiments/tinyllm_cross_seed_causal_carrier_transport/"
        "20260806_d6_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    orbit_count: int = 64
    carrier_rank: int = 3
    first_blocks: int = 6
    writer_ridge: float = 1e-6
    observation_alignment_floor: float = 0.99
    observation_mean_shift_ceiling_bins: float = 0.125
    observation_p95_shift_ceiling_bins: float = 0.50
    shuffled_mean_margin_bins: float = 0.125
    alignment_loss_ceiling: float = 0.005
    mean_shift_ceiling_bins: float = 0.125
    p95_shift_ceiling_bins: float = 0.50
    winding_tolerance: float = 0.10
    accuracy_loss_ceiling: float = 0.03
    task_effect_floor: float = 1e-6
    decomposition_tolerance: float = 1e-6
    activation_batch_size: int = 256
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary writer seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("writer seeds must be non-empty and distinct")
        if self.orbit_count < 8:
            raise ValueError("at least eight exact orbits are required")
        if self.carrier_rank != 3:
            raise ValueError("the fixed semantic gauge has exactly three channels")
        if self.writer_ridge <= 0.0:
            raise ValueError("writer ridge must be positive")


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
        Path(ladder.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: FixedSemanticGaugeWriterConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_mechanistic_positive_control"
    )


def _transport_config(
    config: FixedSemanticGaugeWriterConfig,
) -> transport.CrossSeedCarrierTransportConfig:
    return transport.CrossSeedCarrierTransportConfig(
        source_root=config.source_root,
        character_root=config.character_root,
        readout_root=config.readout_root,
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        carrier_rank=config.carrier_rank,
        first_blocks=config.first_blocks,
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


def neutral_c2_carrier(vector: torch.Tensor) -> torch.Tensor:
    """Exact neutral fusion (x^2-y^2, 2xy, x^2+y^2)."""
    if vector.ndim != 2 or vector.shape[1] != 2:
        raise ValueError("charged carrier must have shape [observations, 2]")
    x, y = vector.unbind(-1)
    return torch.stack((x.square() - y.square(), 2.0 * x * y, x.square() + y.square()), -1)


def semantic_orbit_carrier(
    dataset: Any, task: CircleTaskConfig, orbit_count: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    phase_vector = ladder.AnalyticPhaseCarrier(task)(
        dataset.sensor.to(device), dataset.calibration.to(device)
    )
    sheets = neutral_c2_carrier(phase_vector).reshape(orbit_count, 2, 3)
    carrier = sheets.mean(1).double()
    predicted = torch.atan2(carrier[:, 1], carrier[:, 0])
    phase = dataset.phase.to(device).reshape(orbit_count, 2)[:, 0].double()
    target = torch.remainder(2.0 * phase, 2.0 * math.pi)
    difference = torch.atan2(
        torch.sin(predicted - target), torch.cos(predicted - target)
    )
    width = 2.0 * math.pi / task.phase_bins
    shifts = difference.abs() / width
    sheet_difference = torch.linalg.vector_norm(sheets[:, 0] - sheets[:, 1], dim=1)
    return carrier, {
        "circular_alignment": float(torch.cos(difference).mean()),
        "mean_shift_bins": float(shifts.mean()),
        "p95_shift_bins": float(torch.quantile(shifts, 0.95)),
        "maximum_shift_bins": float(shifts.max()),
        "mean_sheet_difference": float(sheet_difference.mean()),
        "maximum_sheet_difference": float(sheet_difference.max()),
        "mean_energy_channel": float(carrier[:, 2].mean()),
        "maximum_energy_deviation": float((carrier[:, 2] - 1.0).abs().max()),
    }


def fit_linear_writer(
    carrier: torch.Tensor, coordinates: torch.Tensor, ridge: float
) -> dict[str, torch.Tensor]:
    """Fit a no-intercept ridge writer from fixed carrier to target coordinates."""
    carrier = carrier.double()
    coordinates = coordinates.double()
    if carrier.ndim != 2 or carrier.shape != coordinates.shape:
        raise ValueError("carrier and coordinate matrices must have equal shape")
    identity = torch.eye(carrier.shape[1], dtype=carrier.dtype, device=carrier.device)
    linear = torch.linalg.solve(
        carrier.T @ carrier + ridge * identity,
        carrier.T @ coordinates,
    )
    return {
        "linear": linear,
        "intercept": torch.zeros(
            coordinates.shape[1], dtype=coordinates.dtype, device=coordinates.device
        ),
    }


def regime_preserving_writer_permutation(
    count_per_regime: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(17_000_000 + 7919 * seed)
    return torch.cat(
        [
            torch.randperm(count_per_regime, generator=generator)
            + regime * count_per_regime
            for regime in range(len(REGIMES))
        ]
    )


def observation_contract(
    metrics: Sequence[Mapping[str, float]],
    config: FixedSemanticGaugeWriterConfig,
) -> bool:
    return bool(
        metrics
        and all(
            item["circular_alignment"] >= config.observation_alignment_floor
            and item["mean_shift_bins"]
            <= config.observation_mean_shift_ceiling_bins
            and item["p95_shift_bins"] <= config.observation_p95_shift_ceiling_bins
            for item in metrics
        )
    )


def checkpoint_gates(
    heldout: Sequence[Mapping[str, Any]],
    observation_metrics: Sequence[Mapping[str, float]],
    config: FixedSemanticGaugeWriterConfig,
) -> dict[str, Any]:
    controls = all(
        not cell["states"]["zero"]["continuous"]["continuous_pass"]
        and cell["states"]["exact"]["continuous"]["continuous_pass"]
        and cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
        and cell["decomposition_relative_error"] <= config.decomposition_tolerance
        for cell in heldout
    )
    causal = all(
        cell["states"]["fixed_gauge"]["continuous"]["continuous_pass"]
        for cell in heldout
    )
    writer_mean = sum(
        cell["states"]["fixed_gauge"]["continuous"]["mean_moment_shift_bins"]
        for cell in heldout
    ) / len(heldout)
    shuffled_mean = sum(
        cell["states"]["fixed_gauge_shuffled"]["continuous"][
            "mean_moment_shift_bins"
        ]
        for cell in heldout
    ) / len(heldout)
    shuffled_fails = any(
        not cell["states"]["fixed_gauge_shuffled"]["continuous"][
            "continuous_pass"
        ]
        for cell in heldout
    )
    specificity = bool(
        shuffled_fails
        and writer_mean + config.shuffled_mean_margin_bins <= shuffled_mean
    )
    return {
        "observation_contract": observation_contract(observation_metrics, config),
        "continuous_target_control_contract": controls,
        "fixed_gauge_causal_writer": causal,
        "shuffled_specificity": specificity,
        "fixed_gauge_aggregate_mean_shift_bins": writer_mean,
        "shuffled_aggregate_mean_shift_bins": shuffled_mean,
        "worst_heldout_coordinate_variance_explained": min(
            cell["coordinate_metrics"]["fixed_gauge"]["variance_explained"]
            for cell in heldout
        ),
    }


def _load_predecessor(
    config: FixedSemanticGaugeWriterConfig,
) -> tuple[dict[str, Any], Path, dict[int, dict[str, Any]]]:
    path = Path(config.predecessor_root) / "campaign_results.json"
    campaign = json.loads(path.read_text())
    if (
        _sha256(path) != PREDECESSOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != transport.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != transport.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != PREDECESSOR_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 6
    ):
        raise ValueError(f"invalid fixed-gauge predecessor {path}")
    identities: dict[int, dict[str, Any]] = {}
    for entry in campaign.get("results", []):
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
        ):
            raise ValueError(f"invalid predecessor pair {detail_path}")
        for role in ("source", "target"):
            seed = int(detail[f"{role}_seed"])
            provenance = detail["provenance"][role]
            existing = identities.get(seed)
            if existing is not None and existing["checkpoint_sha256"] != provenance[
                "checkpoint_sha256"
            ]:
                raise ValueError(f"inconsistent predecessor identity for seed {seed}")
            identities[seed] = provenance
    if not set(config.seeds).issubset(identities):
        raise ValueError("predecessor does not identify every requested seed")
    return campaign, path, identities


def _fingerprint(
    config: FixedSemanticGaugeWriterConfig,
    task: CircleTaskConfig,
    seed: int,
    predecessor_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "task": asdict(task),
        "cohort_seeds": COHORT_SEEDS,
        "seed": seed,
        "predecessor_sha256": predecessor_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: FixedSemanticGaugeWriterConfig,
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
    config: FixedSemanticGaugeWriterConfig, output: Path
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
    predecessor, predecessor_path, predecessor_identities = _load_predecessor(config)
    base_config = _transport_config(config)
    rank_config = transport._rank_config(base_config)
    bridge = transport.rank._bridge_config(rank_config)
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

    results = []
    for seed in config.seeds:
        system, provenance = transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        expected = predecessor_identities[seed]
        if (
            provenance["checkpoint_sha256"] != expected["checkpoint_sha256"]
            or provenance["frontend_checkpoint_sha256"]
            != expected["frontend_checkpoint_sha256"]
        ):
            raise ValueError(f"checkpoint identity mismatch for seed {seed}")
        frozen, frozen_path = transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
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
        carriers: dict[str, dict[str, torch.Tensor]] = {}
        observation: dict[str, dict[str, dict[str, float]]] = {}
        for cohort in COHORT_SEEDS:
            carriers[cohort] = {}
            observation[cohort] = {}
            for regime in REGIMES:
                value, metrics = semantic_orbit_carrier(
                    datasets[cohort][regime], task, config.orbit_count, device
                )
                carriers[cohort][regime] = value
                observation[cohort][regime] = metrics

        fit_carrier = torch.cat(
            [carriers["alignment_fit"][regime] for regime in REGIMES]
        )
        fit_coordinates = torch.cat(
            [transport._coordinates(cells["alignment_fit"][regime], basis) for regime in REGIMES]
        )
        writer = fit_linear_writer(fit_carrier, fit_coordinates, config.writer_ridge)
        permutation = regime_preserving_writer_permutation(config.orbit_count, seed).to(
            device
        )
        shuffled = fit_linear_writer(
            fit_carrier, fit_coordinates[permutation], config.writer_ridge
        )
        mappings = {"fixed_gauge": writer, "fixed_gauge_shuffled": shuffled}
        identity_basis = torch.eye(3, dtype=torch.float64, device=device)
        heldout = []
        observation_heldout = []
        for cohort in HELDOUT_COHORTS:
            for regime in REGIMES:
                carrier = carriers[cohort][regime]
                source_cell = {"full_defect": carrier}
                heldout.append(
                    transport._evaluate_transport_cell(
                        system,
                        task,
                        base_config,
                        source_cell,
                        cells[cohort][regime],
                        identity_basis,
                        basis,
                        mappings,
                        readout_predecessor["rotation_bins"],
                    )
                )
                observation_heldout.append(observation[cohort][regime])
        gates = checkpoint_gates(heldout, observation_heldout, config)
        fingerprint = _fingerprint(
            config,
            task,
            seed,
            _sha256(predecessor_path),
            provenance["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-fixed-semantic-gauge-writer-seed{seed}",
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
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "frontend_checkpoint": provenance["frontend_checkpoint"],
                "frontend_checkpoint_sha256": provenance[
                    "frontend_checkpoint_sha256"
                ],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
                "readout_campaign": readout_predecessor["campaign"],
                "readout_campaign_sha256": readout_predecessor[
                    "campaign_sha256"
                ],
                "readout_result": readout_predecessor["result"],
                "readout_result_sha256": readout_predecessor["result_sha256"],
            },
            "basis": basis_summary,
            "observation_audit": observation,
            "alignment_fit": {
                "writer": transport._mapping_summary(writer),
                "writer_coordinate_metrics": transport.coordinate_metrics(
                    transport.apply_affine(fit_carrier, writer), fit_coordinates
                ),
                "shuffled_writer": transport._mapping_summary(shuffled),
                "shuffled_metrics_against_true_pairs": transport.coordinate_metrics(
                    transport.apply_affine(fit_carrier, shuffled), fit_coordinates
                ),
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
                        "observation_contract",
                        "continuous_target_control_contract",
                        "fixed_gauge_causal_writer",
                        "shuffled_specificity",
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
                and int(existing.get("seed", -1)) == seed
            ):
                result = existing
                print(f"resuming {existing['experiment_id']}", flush=True)
            else:
                raise ValueError(f"incompatible completed result {result_path}")
        else:
            _write_json(result_path, result)
        results.append(result)
        print(
            f"seed {seed}: writer={'pass' if result['primary_metric'] else 'fail'}",
            flush=True,
        )
        if _implementation_digest() != implementation:
            raise RuntimeError("writer implementation changed during campaign")

    gate_names = (
        "observation_contract",
        "continuous_target_control_contract",
        "fixed_gauge_causal_writer",
        "shuffled_specificity",
    )
    gate_counts = {
        name: sum(bool(result["gates"][name]) for result in results)
        for name in gate_names
    }
    required = len(config.seeds)
    confirmed = bool(
        not config.allow_underpowered
        and required == 3
        and all(gate_counts[name] == required for name in gate_names)
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
            "requested": required,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "trained_models": 0,
            "fitted_predictive_observers": 0,
            "fitted_linear_writers": required * 2,
        },
        "aggregates": {
            "gate_counts": gate_counts,
            "confirmed": confirmed,
            "conclusion": (
                "systems_lifecycle_only_not_quality_evidence"
                if config.allow_underpowered
                else (
                    "confirmed_fixed_semantic_gauge_writer"
                    if confirmed
                    else "not_confirmed_fixed_semantic_gauge_writer"
                )
            ),
            "per_seed": {str(result["seed"]): result["gates"] for result in results},
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
                "primary_metric": result["primary_metric"],
                "gates": result["gates"],
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "This is an analytic positive control rather than a learned sidecar.",
            "The fixed carrier requires the observed calibration packet and exact sensor decoder.",
            "Checkpoint-local linear writers have target access on alignment-fit orbits.",
            "The same held-out cells were used by earlier post-outcome diagnostics.",
            "Only three selected stable block-0 checkpoints are tested.",
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
            "data/experiments/tinyllm_fixed_semantic_gauge_writer/"
            "20260806_d6_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=PRIMARY_SEEDS)
    parser.add_argument("--orbit-count", type=int, default=64)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = FixedSemanticGaugeWriterConfig(
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
