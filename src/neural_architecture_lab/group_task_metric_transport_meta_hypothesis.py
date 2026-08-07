"""Build and store the TinyLLM group/task-metric transport evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-group-task-metric-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-group-task-metric-carrier-transport-v1"
EVIDENCE_ROLE = "preregistered_underpowered_mechanistic_evidence"
IMPLEMENTATION_SHA256 = (
    "7e70d42d978fec1f05d2d7c5ee9d9c8d02acb70e9e27291f44942dcbdcd1cea8"
)
DIRECTED_PAIRS = ((7, 29), (7, 53), (29, 7), (29, 53), (53, 7), (53, 29))
GATE_COUNTS = {
    "continuous_target_control_contract": 6,
    "metric_contract": 6,
    "predecessor_replay_contract": 6,
    "task_metric_causal_transport": 2,
    "task_metric_coordinate_transport": 3,
    "task_metric_dominates_euclidean_baselines": 2,
    "task_metric_shuffled_specificity": 6,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or int(value.get("summary", {}).get("requested", -1)) != 6
        or int(value.get("summary", {}).get("completed", -1)) != 6
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_predictive_observers", -1))
        != 0
        or int(value.get("summary", {}).get("fitted_new_coordinate_maps", -1))
        != 12
        or value.get("aggregates", {}).get("confirmed") is not False
        or value.get("aggregates", {}).get("gate_counts") != GATE_COUNTS
    ):
        raise ValueError(f"invalid group/task-metric transport campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (int(item["source_seed"]), int(item["target_seed"])): item
        for item in campaign.get("results", [])
    }
    if set(entries) != set(DIRECTED_PAIRS):
        raise ValueError(f"invalid group/task-metric directed-pair index {path}")
    output = []
    fingerprints: set[str] = set()
    for pair in DIRECTED_PAIRS:
        entry = entries[pair]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        expected_pass = set(pair) == {7, 53}
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
            or detail.get("gates", {}).get("metric_contract") is not True
            or detail.get("gates", {}).get("continuous_target_control_contract")
            is not True
            or detail.get("gates", {}).get("predecessor_replay_contract") is not True
            or detail.get("gates", {}).get("task_metric_shuffled_specificity")
            is not True
            or detail.get("gates", {}).get("task_metric_causal_transport")
            is not expected_pass
            or len(detail.get("heldout_cells", [])) != 4
        ):
            raise ValueError(f"invalid group/task-metric result {detail_path}")
        provenance = detail.get("provenance", {})
        for role in ("source", "target"):
            for field in (
                "checkpoint",
                "frontend_checkpoint",
                "source_result",
                "character_result",
            ):
                artifact = Path(provenance.get(role, {}).get(field, ""))
                if not artifact.is_file() or _sha256(artifact) != provenance.get(
                    role, {}
                ).get(f"{field}_sha256"):
                    raise ValueError(f"invalid {role} {field} in {detail_path}")
        for field in ("predecessor_campaign", "predecessor_result"):
            artifact = Path(provenance.get(field, ""))
            if not artifact.is_file() or _sha256(artifact) != provenance.get(
                f"{field}_sha256"
            ):
                raise ValueError(f"invalid {field} in {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_group_task_metric_transport_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-group-task-metric-transport.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    evidence = []
    for detail in details:
        evidence.append(
            {
                "experiment_id": detail["experiment_id"],
                "source_seed": int(detail["source_seed"]),
                "target_seed": int(detail["target_seed"]),
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "gates": detail["gates"],
                "metric_stability": detail["target_task_metric"]["stability"],
                "heldout": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "coordinate_metrics": cell["coordinate_metrics"][
                            "task_metric"
                        ],
                        "continuous": cell["states"]["task_metric"]["continuous"],
                    }
                    for cell in detail["heldout_cells"]
                ],
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM group-anchored task-metric carrier transport",
            "category": "representation",
            "description": (
                "Label-free per-orbit pullback Fisher metrics fix the linear "
                "multiplicity-space gauge of rank-three C2 Reynolds carriers."
            ),
            "question": (
                "Does target task geometry turn the three selected checkpoint-local "
                "carriers into one causally transportable chart?"
            ),
            "prediction": (
                "All six directed task-metric maps pass continuous causal, coordinate, "
                "control, dominance, numerical, replay, and shuffled gates."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_pairwise_7_53_gauge_class_only",
            "evidence_count": 6,
            "direct_experiment_count": 6,
            "power_profile": "underpowered_preregistered_mechanistic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "six directed maps among three stable d6 C2 block-0 fronts; "
                "two held-out cohorts under composition and extrapolation"
            ),
            "subclaims": {
                "global_task_metric_causal_chart": "rejected_two_of_six",
                "pairwise_seed7_seed53_gauge_equivalence": "supported_both_directions",
                "seed29_linear_task_metric_equivalence": "rejected_all_four_directions",
                "metric_numerical_contract": "supported_six_of_six",
                "continuous_target_controls": "supported_six_of_six",
                "shuffled_specificity": "supported_six_of_six",
                "seed29_extrapolation_cells": "rejected_zero_of_eight",
            },
            "tags": [
                "tinyllm",
                "c2",
                "representation-theory",
                "pullback-fisher",
                "cross-seed",
                "causal-patching",
                "partial-positive",
                "negative-global-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Nonlinear or held-out-adapted carrier transport.",
                "A deliberately gauge-fixed equivariant sidecar trained from scratch.",
                "Population prevalence beyond the selected three checkpoints.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Only the 7-to-53 and 53-to-7 maps pass every causal cell and "
                "dominance gate; all four directions involving seed 29 fail."
            ),
            "confidence": 0.9,
            "confidence_assessment": (
                "exact_preregistered_underpowered_pairwise_partial_support_global_rejection"
            ),
            "independent_seed_count": 3,
            "num_direct_experiments": 6,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "A stable label-free task metric causally aligns checkpoints 7 and 53 in both directions and all eight held-out cells.",
                "The 7/53 task-metric maps halve mean moment error relative to the better Euclidean affine baseline.",
                "All four maps involving seed 29 fail the all-cell endpoint and every seed-29 extrapolation cell fails.",
                "All metric, replay, continuous-control, and shuffled-specificity contracts pass, ruling out the main numerical and marginal-distribution explanations.",
                "Shared irrep type does not fix a universal multiplicity-space gauge or causal metric across independently trained checkpoints.",
            ],
            "suggested_hypotheses": [
                "An explicitly C2-typed sidecar with neutral bilinear fusion and fixed normalization creates a portable three-channel carrier.",
                "Seed 29 differs from seeds 7 and 53 by a nonlinear support-relative carrier chart rather than a linear metric gauge.",
                "Fixing the carrier gauge during training will reduce cross-seed chart stratification without representation penalties.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
        },
    }


def build_group_task_metric_transport_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        gates = detail["gates"]
        metrics = {name: float(gates[name]) for name in GATE_COUNTS}
        metrics.update(
            {
                "task_metric_aggregate_mean_shift_bins": float(
                    gates["task_metric_aggregate_mean_shift_bins"]
                ),
                "paired_aggregate_mean_shift_bins": float(
                    gates["paired_aggregate_mean_shift_bins"]
                ),
                "affine_ridge_aggregate_mean_shift_bins": float(
                    gates["affine_ridge_aggregate_mean_shift_bins"]
                ),
                "metric_median_relative_error": float(
                    detail["target_task_metric"]["stability"][
                        "median_relative_error"
                    ]
                ),
            }
        )
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(detail["primary_metric"]),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["target"]["checkpoint"],
                observations=[
                    "Frozen C2 Reynolds carrier; label-free target pullback Fisher metric; two untouched held-out cohorts under both shifts.",
                    "Both seed-7/53 directions pass; the global three-checkpoint hypothesis fails.",
                ],
                anomalies=(
                    [
                        "Every extrapolation cell involving seed 29 fails the continuous task-metric endpoint."
                    ]
                    if 29 in (int(detail["source_seed"]), int(detail["target_seed"]))
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_group_task_metric_transport_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_group_task_metric_transport_meta_hypothesis(results_path)
    experiments = build_group_task_metric_transport_experiment_results(
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
            raise RuntimeError("group/task-metric transport ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


__all__ = [
    "build_group_task_metric_transport_meta_hypothesis",
    "build_group_task_metric_transport_experiment_results",
    "store_group_task_metric_transport_meta_hypothesis",
]
