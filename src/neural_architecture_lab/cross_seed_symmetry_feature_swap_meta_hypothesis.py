"""Build and store TinyLLM cross-seed symmetry-feature swap evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-cross-seed-symmetry-feature-swap.v1"
HYPOTHESIS_ID = "tinyllm-cross-seed-symmetry-feature-gauge-v1"
EVIDENCE_ROLE = "preregistered_frozen_checkpoint_causal_diagnostic"
IMPLEMENTATION_SHA256 = (
    "0508821da71b2129b5a7a437c2f1d9aa60f3fa4b400ff3a74a59a968b9d65a50"
)
SEEDS = (7, 17, 29, 41, 53)
DIRECTED_PAIRS = tuple(
    (source, target) for source in SEEDS for target in SEEDS if source != target
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    aggregates = value.get("aggregates", {})
    summary = value.get("summary", {})
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or int(summary.get("requested", -1)) != 20
        or int(summary.get("completed", -1)) != 20
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_maps", -1)) != 0
        or aggregates.get("confirmed") is not False
        or aggregates.get("conclusion")
        != "feature_and_scalar_interfaces_checkpoint_coadapted"
        or int(aggregates.get("group_contract_pass_count", -1)) != 5
        or int(aggregates.get("pair_pass_count", -1)) != 0
        or int(aggregates.get("target_pass_count", -1)) != 0
        or aggregates.get("scalar_campaign_pass") is not False
        or int(aggregates.get("scalar_pair_pass_count", -1)) != 0
        or int(aggregates.get("scalar_target_pass_count", -1)) != 0
    ):
        raise ValueError(f"invalid symmetry-feature swap campaign {path}")
    contracts = value.get("group_contracts", {})
    if set(contracts) != {str(seed) for seed in SEEDS} or not all(
        record.get("pass") is True
        and float(record.get("maximum_absolute_error", float("inf"))) <= 1e-5
        for record in contracts.values()
    ):
        raise ValueError(f"invalid symmetry-feature group contracts {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (int(item["source_seed"]), int(item["target_seed"])): item
        for item in campaign.get("results", [])
    }
    if set(entries) != set(DIRECTED_PAIRS):
        raise ValueError(f"invalid symmetry-feature directed-pair index {path}")
    output = []
    fingerprints: set[str] = set()
    artifacts: dict[Path, str] = {}
    feature_cell_passes = 0
    scalar_cell_passes = 0
    for pair in DIRECTED_PAIRS:
        entry = entries[pair]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        cells = detail.get("cells", {})
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or (int(detail.get("source_seed", -1)), int(detail.get("target_seed", -1)))
            != pair
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("gates", {}).get("pair_group_contract") is not True
            or detail.get("gates", {}).get("pair_pass") is not False
            or detail.get("gates", {}).get("scalar_pair_pass") is not False
            or int(detail.get("trained_models", -1)) != 0
            or int(detail.get("fitted_maps", -1)) != 0
            or set(cells) != {
                "a_composition",
                "a_extrapolation",
                "b_composition",
                "b_extrapolation",
            }
            or any(
                float(cell.get("direct_replay_max_absolute_posterior_error", 1.0))
                > 1e-6
                for cell in cells.values()
            )
        ):
            raise ValueError(f"invalid symmetry-feature result {detail_path}")
        feature_cell_passes += sum(
            cell["feature_swap_gate"]["pass"] is True for cell in cells.values()
        )
        scalar_cell_passes += sum(
            cell["scalar_swap_gate"]["pass"] is True for cell in cells.values()
        )
        provenance = detail["provenance"]
        for role in ("source", "target"):
            for field in ("result", "model_checkpoint", "frontend_checkpoint"):
                artifact = Path(provenance[role][field])
                expected = provenance[role][f"{field}_sha256"]
                previous = artifacts.get(artifact)
                if previous is not None and previous != expected:
                    raise ValueError(f"inconsistent artifact hash for {artifact}")
                artifacts[artifact] = expected
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    if feature_cell_passes != 8 or scalar_cell_passes != 0:
        raise ValueError(f"unexpected symmetry-feature cell split {path}")
    for artifact, expected in artifacts.items():
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"invalid retained artifact {artifact}")
    return output


def build_cross_seed_symmetry_feature_swap_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-cross-seed-symmetry-feature-swap.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = []
    for detail in details:
        direct_tests.append(
            {
                "experiment_id": detail["experiment_id"],
                "source_seed": int(detail["source_seed"]),
                "target_seed": int(detail["target_seed"]),
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "gates": detail["gates"],
                "cells": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "feature_swap_gate": cell["feature_swap_gate"],
                        "scalar_swap_gate": cell["scalar_swap_gate"],
                        "feature_geometry": cell["feature_geometry"],
                        "feature_swap_metrics": cell["conditions"][
                            "source_feature_swap"
                        ],
                        "scalar_swap_metrics": cell["conditions"][
                            "source_scalar_swap"
                        ],
                    }
                    for cell in detail["cells"].values()
                ],
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM cross-seed calibrated symmetry-feature gauge",
            "category": "representation",
            "description": (
                "The calibrated-equivariant three-channel front-end feature fixes "
                "one portable causal gauge across independent TinyLLM checkpoints."
            ),
            "question": (
                "Can a source checkpoint's no-fit dot/cross/signed-speed feature drive "
                "a different target checkpoint's frozen scalar map and continuation?"
            ),
            "prediction": (
                "At least sixteen of twenty directed feature swaps and four of five "
                "target-level joint gates pass across composition and extrapolation."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_exact_equivariance_checkpoint_local_gauges"
            ),
            "evidence_count": 20,
            "direct_experiment_count": 20,
            "independent_checkpoint_count": 5,
            "power_profile": "five_checkpoint_preregistered_frozen_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "all twenty directed swaps among five retained d8 calibrated-equivariant "
                "checkpoints; two held-out cohorts under composition and extrapolation"
            ),
            "subclaims": {
                "exact_acquisition_group_contract": "supported_five_of_five",
                "portable_three_channel_feature_gauge": "rejected_zero_of_twenty_pairs",
                "portable_scalar_bottleneck": "rejected_zero_of_twenty_pairs",
                "composition_cell_portability": "supported_eight_of_forty_cells",
                "extrapolation_cell_portability": "rejected_zero_of_forty_cells",
                "feature_sign_classes": "supported_7_17_vs_29_41_53",
                "exact_direct_replay": "supported_eighty_of_eighty_cells",
            },
            "tags": [
                "tinyllm",
                "equivariance",
                "symmetry",
                "cross-seed",
                "causal-swap",
                "gauge",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A matched architecture with sign, scale, metric, scalar map, and embedding fixed during training.",
                "Population prevalence beyond these five retained checkpoints.",
                "Nonlinear post-hoc cross-seed alignment.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The acquisition-group contract passes in all five checkpoints, but "
                "feature and scalar portability each pass zero of twenty directed pairs "
                "and zero of five target-level joint gates."
            ),
            "confidence": 0.96,
            "confidence_assessment": (
                "exact_preregistered_frozen_intervention_with_five_checkpoint_units"
            ),
            "independent_seed_count": 5,
            "num_direct_experiments": 20,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Exact positive-similarity equivariance holds to at most 6.56e-7 in every checkpoint.",
                "The invariant vectors split into two sign classes: seeds 7/17 and seeds 29/41/53.",
                "Only eight composition cells from 7<->17 and 29<->53 pass; every extrapolation cell fails.",
                "The learned one-dimensional scalar is not a portable downstream interface in any directed pair.",
                "Equivariance fixes transformation law but not a unique causal gauge or continuation convention.",
            ],
            "suggested_hypotheses": [
                "A fixed observed sign anchor plus shared analytic neutral fusion removes the discrete cross-seed gauge class.",
                "A fixed typed scalar embedding is required in addition to a fixed equivariant sensor carrier.",
                "Support-relative temporal filtering causes the remaining within-class extrapolation drift.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
        },
    }


def build_cross_seed_symmetry_feature_swap_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        cells = list(detail["cells"].values())
        feature_passes = sum(cell["feature_swap_gate"]["pass"] for cell in cells)
        scalar_passes = sum(cell["scalar_swap_gate"]["pass"] for cell in cells)
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "pair_group_contract": float(
                        detail["gates"]["pair_group_contract"]
                    ),
                    "feature_pair_pass": float(detail["gates"]["pair_pass"]),
                    "scalar_pair_pass": float(
                        detail["gates"]["scalar_pair_pass"]
                    ),
                    "feature_cell_pass_count": float(feature_passes),
                    "scalar_cell_pass_count": float(scalar_passes),
                    "mean_feature_angle_radians": sum(
                        cell["feature_geometry"][
                            "mean_absolute_angular_displacement_radians"
                        ]
                        for cell in cells
                    )
                    / len(cells),
                    "mean_feature_fisher_distance": sum(
                        cell["conditions"]["source_feature_swap"][
                            "mean_fisher_rao_from_direct"
                        ]
                        for cell in cells
                    )
                    / len(cells),
                },
                primary_metric=float(detail["gates"]["pair_pass"]),
                model_architecture=[8, 8, 512],
                model_parameters=50_976_625,
                training_time=0.0,
                model_checkpoint=detail["provenance"]["target"][
                    "model_checkpoint"
                ],
                observations=[
                    "No-fit swap of the calibrated dot/cross/signed-speed feature across two composition and two extrapolation cells.",
                    "Exact acquisition symmetry passes; no directed pair passes all four cells.",
                ],
                anomalies=(
                    ["Both composition cells pass, but both extrapolation cells fail."]
                    if feature_passes == 2
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_cross_seed_symmetry_feature_swap_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_cross_seed_symmetry_feature_swap_meta_hypothesis(results_path)
    experiments = build_cross_seed_symmetry_feature_swap_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir="data/experiment_queue",
                sent_dir="data/experiment_sent",
                rejected_dir="data/experiment_rejected",
                enable_wandb=False,
                auto_upload=False,
                enable_chromadb=True,
                chromadb_path=str(chromadb_path),
            )
        )
        logger.log_hypothesis(record["hypothesis"])
        storage["result_hashes"] = [
            logger.log_experiment_result(item) for item in experiments
        ]
        storage["chromadb_path"] = str(chromadb_path)
        hypothesis_readback = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        experiment_readback = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in experiment_readback.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("symmetry-feature swap ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_cross_seed_symmetry_feature_swap_meta_hypothesis",
    "build_cross_seed_symmetry_feature_swap_experiment_results",
    "store_cross_seed_symmetry_feature_swap_meta_hypothesis",
]
