"""Persist the frozen TinyLLM internal-quotient probe evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-internal-cosine-quotient-erases-branch-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-internal-quotient-probe.v1"


def _read_results(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported internal-quotient probe schema")
    if campaign.get("status") != "completed" or len(campaign.get("runs", [])) != 4:
        raise ValueError("internal-quotient probe campaign is incomplete")
    return campaign


def build_internal_quotient_probe_meta_hypothesis(
    results_path: Path,
) -> dict[str, Any]:
    campaign = _read_results(results_path)
    runs = campaign["runs"]
    evidence = [
        {
            "evidence_role": "direct_frozen_probe",
            "experiment_id": run["experiment_id"],
            "preset": run["preset"],
            "quotient": run["quotient"],
            "seed": run["seed"],
            "checkpoint": run["checkpoint"],
            "model_frozen": run["model_frozen"],
            **{
                key: float(value)
                for key, value in run["primary_metrics"].items()
            },
        }
        for run in runs
    ]
    by_quotient = {
        quotient: [run for run in runs if run["quotient"] == quotient]
        for quotient in ("phase_circle", "cosine_interval")
    }

    def mean(quotient: str, metric: str) -> float:
        selected = by_quotient[quotient]
        return sum(float(run["primary_metrics"][metric]) for run in selected) / len(
            selected
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Cosine training erases the phase branch from TinyLLM residuals",
            "category": "representation",
            "description": (
                "Frozen nonlinear probes at every residual layer test the phase branch "
                "on exact cosine-matched pairs, with cross-decoding and independently "
                "derived persistent-cohomology coordinates."
            ),
            "question": (
                "Does the cosine-trained model internally identify opposite phases, or "
                "does only its supervised posterior collapse them?"
            ),
            "prediction": campaign["pre_registered_prediction"],
            "created_at": campaign["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "strong_single_seed_internal_asymmetry_not_confirmed",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "saved seed-7 d6/d8 synthetic quotient checkpoints",
            "success_criteria": {
                preset: aggregate["pre_registered_criteria"]
                for preset, aggregate in campaign["aggregates"].items()
            },
            "subclaims": {
                "conditional_branch_asymmetry": "supported_in_both_model_classes",
                "cross_decoding_asymmetry": "supported_in_both_model_classes",
                "disjoint_nuisance_branch_asymmetry": "supported_in_both_model_classes",
                "independent_phase_circle_recovery": "supported_in_both_phase_models",
                "all_frozen_probe_criteria": "not_supported_d6_boundary_failure",
                "multi_seed_repeatability": "not_tested",
            },
            "tags": [
                "tinyllm",
                "frozen-probe",
                "conditional-branch",
                "cross-decoding",
                "persistent-cohomology",
                "disjoint-nuisance",
                "single-seed",
            ],
            "explicitly_not_tested": [
                "independent checkpoints for seeds 17 and 29",
                "real sensor transfer",
                "causal information erasure under intervention",
                "linear-only probe selectivity controls",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The predicted branch and cross-decoding asymmetries were large in d6 "
                "and d8, but only seed-7 checkpoints were retained and d6 cosine "
                "cohomology alignment was 0.603, just above its frozen 0.600 cutoff."
            ),
            "confidence": 0.0,
            "confidence_assessment": "descriptive_single_seed_paired_checkpoints",
            "effect_size": mean(
                "phase_circle", "final_id_conditional_branch_accuracy"
            )
            - mean("cosine_interval", "final_id_conditional_branch_accuracy"),
            "effect_size_kind": "phase_minus_cosine_conditional_branch_accuracy",
            "independent_seed_count": 1,
            "model_class_count": 2,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "phase_id_branch_accuracy_mean": mean(
                    "phase_circle", "final_id_conditional_branch_accuracy"
                ),
                "cosine_id_branch_accuracy_mean": mean(
                    "cosine_interval", "final_id_conditional_branch_accuracy"
                ),
                "phase_ood_branch_accuracy_mean": mean(
                    "phase_circle", "final_ood_conditional_branch_accuracy"
                ),
                "cosine_ood_branch_accuracy_mean": mean(
                    "cosine_interval", "final_ood_conditional_branch_accuracy"
                ),
                "phase_id_cosine_pearson_mean": mean(
                    "phase_circle", "final_id_cosine_pearson"
                ),
                "cosine_id_cosine_pearson_mean": mean(
                    "cosine_interval", "final_id_cosine_pearson"
                ),
                "phase_id_phase_alignment_mean": mean(
                    "phase_circle", "final_id_phase_alignment"
                ),
                "cosine_id_phase_alignment_mean": mean(
                    "cosine_interval", "final_id_phase_alignment"
                ),
                "phase_independent_ph_h1_mean": mean(
                    "phase_circle", "final_ph_normalized_h1_lifetime"
                ),
                "cosine_independent_ph_h1_mean": mean(
                    "cosine_interval", "final_ph_normalized_h1_lifetime"
                ),
            },
            "key_insights": [
                "Phase-model final branch accuracy was 99.9% in both d6 and d8.",
                "Cosine-model final branch accuracy remained within 3.2 points of 50% in all ID/OOD cells.",
                "The asymmetry appeared at the first transformer block and persisted through the final block.",
                "Independent cohomology found strong aligned phase circles only in phase-trained final residuals.",
            ],
            "unexpected_findings": [
                "Disjoint-family cosine decoding degraded in both model types even while branch asymmetry persisted.",
                "The d6 cosine coordinate had weak H1 but alignment 0.603, missing a brittle 0.600 cutoff by 0.003.",
            ],
            "suggested_hypotheses": [
                "The conditional branch asymmetry repeats across at least five independently trained checkpoints.",
                "Probe selectivity remains after matched-capacity random-label and untrained-model controls.",
                "The branch information is causally unavailable, not merely inaccessible to the probe family.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path)],
    }


def build_internal_quotient_probe_experiment_results(
    record: Mapping[str, Any],
    results_path: Path,
) -> list[ExperimentResult]:
    campaign = _read_results(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    results = []
    for run in campaign["runs"]:
        metrics = {
            key: float(value) for key, value in run["primary_metrics"].items()
        }
        metrics["accuracy"] = metrics["final_id_conditional_branch_accuracy"]
        results.append(
            ExperimentResult(
                experiment_id=run["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=metrics["final_id_conditional_branch_accuracy"],
                model_architecture=[
                    int(run["preset"][1:]),
                    int(run["model_parameters"]),
                ],
                model_parameters=int(run["model_parameters"]),
                training_time=float(run["analysis_seconds"]),
                model_checkpoint=run["checkpoint"],
                observations=[
                    f"Frozen source quotient: {run['quotient']}.",
                    "Nonlinear probes were selected on a held-out validation set.",
                    "Branch labels were balanced within exact cosine-matched pairs.",
                ],
                anomalies=(
                    ["Only one retained training seed was available."]
                    + (
                        ["d6 cosine PH alignment missed its cutoff by 0.0033."]
                        if run["preset"] == "d6"
                        and run["quotient"] == "cosine_interval"
                        else []
                    )
                ),
                timestamp=completed_at,
            )
        )
    return results


def store_internal_quotient_probe_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_internal_quotient_probe_meta_hypothesis(results_path)
    experiment_results = build_internal_quotient_probe_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [result.experiment_id for result in experiment_results],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(queue_dir),
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
            logger.log_experiment_result(result) for result in experiment_results
        ]
        storage["chromadb_path"] = str(chromadb_path)
        storage["hypotheses_collection"] = "hypotheses"
        storage["experiments_collection"] = "experiments"
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


__all__ = [
    "META_HYPOTHESIS_ID",
    "build_internal_quotient_probe_experiment_results",
    "build_internal_quotient_probe_meta_hypothesis",
    "store_internal_quotient_probe_meta_hypothesis",
]
