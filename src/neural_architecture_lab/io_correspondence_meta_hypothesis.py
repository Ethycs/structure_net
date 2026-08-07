"""Persist the corrected TinyLLM I/O-correspondence evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-io-correspondence-gauge-descent-v1"
META_SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA_VERSION = "nal.tinyllm-io-correspondence.v1.1"
SEEDS = (7, 17, 29, 41, 53)
CONDITIONS = (
    "uncalibrated_absolute_raw",
    "uncalibrated_absolute_equivariant",
    "calibrated_absolute_raw",
    "calibrated_absolute_analytic",
    "calibrated_absolute_equivariant",
    "uncalibrated_relative_raw",
    "uncalibrated_relative_equivariant",
)
CUTS = ("frontend", "post_attention", "post_mlp", "full")
PRIMARY_REGIMES = ("composition", "extrapolation")


def _read_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported corrected I/O-correspondence schema")
    if campaign.get("status") != "completed":
        raise ValueError("I/O-correspondence campaign is incomplete")
    summary = campaign.get("summary", {})
    if summary.get("completed") != 35 or summary.get("failed") != 0:
        raise ValueError("corrected I/O-correspondence campaign must have 35 successes")
    correction = campaign.get("correction", {})
    if correction.get("claim_status") != "disclosed_post-outcome_implementation_correction":
        raise ValueError("campaign does not preserve the implementation correction boundary")
    return campaign


def _read_details(results_path: Path) -> list[dict[str, Any]]:
    campaign = _read_campaign(results_path)
    implementation = campaign["implementation_sha256"]
    details = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = results_path.parent / "runs" / condition / f"seed_{seed}" / "result.json"
            detail = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = Path(detail.get("training", {}).get("checkpoint", ""))
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA_VERSION
                or detail.get("status") != "completed"
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("implementation_sha256") != implementation
                or not detail.get("scientific_fingerprint")
                or not checkpoint.is_file()
                or not detail.get("io_relation_contract", {}).get("passed")
            ):
                raise ValueError(f"invalid corrected I/O-correspondence cell: {path}")
            details.append(detail)
    return details


def _cell_metrics(detail: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cut in CUTS:
        for regime in PRIMARY_REGIMES:
            prefix = f"{cut}_{regime}"
            probe = detail["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]
            geometry = detail["analysis"]["cuts"][cut]["io_map_geometry"][regime]
            metrics[f"{prefix}_cosine"] = float(probe["cosine_pearson"])
            metrics[f"{prefix}_branch"] = float(probe["balanced_accuracy"])
            metrics[f"{prefix}_log_loss_gain"] = float(
                probe["conditional_log_loss_gain_over_cosine_only"]
            )
            metrics[f"{prefix}_mapper_interval_like"] = float(
                geometry["mapper"]["interval_like"]
            )
            metrics[f"{prefix}_paired_distortion"] = float(
                geometry["paired_distortion"]["fiber_averaged_whitened_distortion"]
            )
    for regime in ("in_distribution", *PRIMARY_REGIMES):
        metrics[f"{regime}_task_accuracy"] = float(
            detail["analysis"]["task_metrics"][regime]["exact_bin_accuracy"]
        )
    return metrics


def build_io_correspondence_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-io-correspondence.md"
    ),
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    details = _read_details(results_path)
    aggregate = campaign["aggregates"]
    arms = aggregate["arms"]
    evidence = [
        {
            "evidence_role": (
                "frozen_observation_repair_control"
                if detail["condition"].startswith("calibrated")
                else "corrected_direct_test"
            ),
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "checkpoint": detail["training"]["checkpoint"],
            "model_parameters": detail["model_parameters"],
            "system_parameters": detail["system_parameters"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            **_cell_metrics(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Descent-compatible I/O relations are stably realized through TinyLLM layers",
            "category": "representation",
            "description": (
                "A seven-arm d8/N3 study compares an unidentifiable absolute target, "
                "observation-side calibration repair, and target-side gauge quotienting "
                "using layerwise base/fiber probes, target-lens Mapper, and paired distortion."
            ),
            "question": (
                "Do observation refinement and target quotienting both make the paired "
                "task relation stable through an equivariant TinyLLM?"
            ),
            "prediction": (
                "Calibrated and relative-target structured arms pass the same all-cut "
                "composition/extrapolation base-and-fiber gate in at least four of five seeds."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "partially_supported_observation_repair_only",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "d8 synthetic N3; seven arms; five seeds; four cuts; corrected schema v1.1",
            "success_criteria": aggregate["preregistered_gate"],
            "subclaims": {
                "uncalibrated_absolute_target_descends": "contradicted_by_exact_gauge",
                "calibrated_absolute_relation_descends": "supported_by_contract",
                "relative_target_relation_descends": "supported_by_contract",
                "calibrated_structured_stable_quotient": "supported_five_of_five",
                "relative_equivariant_stable_quotient": "not_supported_zero_of_five",
                "mapper_alone_certifies_fiber_descent": "contradicted_too_coarse",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "io-correspondence",
                "gauge",
                "identifiability",
                "equivariance",
                "mapper",
                "paired-distortion",
                "multi-seed",
                "implementation-correction",
            ],
            "explicitly_not_tested": [
                "an analytic positive control for the relative target",
                "a vector-valued task head supervised before scalar projection",
                "certified Reeb cosheaves or conditional mutual information",
                "real-world sensor transfer",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Observation-side calibrated structured arms passed in five of five seeds, "
                "but the corrected relative-target equivariant arm passed in zero of five "
                "and failed its task-degradation control."
            ),
            "confidence": 0.0,
            "confidence_assessment": "five_seed_failure_of_full_joint_prediction_after_disclosed_correction",
            "independent_seed_count": 5,
            "model_class_count": 1,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "arms": arms,
                "task_controls": aggregate["task_controls"],
                "prediction_checks": aggregate["prediction_checks"],
                "joint_gate": aggregate["preregistered_gate"],
            },
            "key_insights": [
                "The calibrated analytic and learned arms realized a stable quotient at all four cuts.",
                "The relative target passed the exact descent contract but its corrected equivariant arm retained branch and lost base under extrapolation.",
                "Raw relative TinyLLM showed support-local composition success and global extrapolation failure.",
                "Finite target-lens Mapper often looked interval-like despite substantial conditional branch decodability.",
                "Paired distortion, tested decodability, and Mapper topology measured distinct properties across depth.",
            ],
            "unexpected_findings": [
                "The corrected relative equivariant encoder underperformed raw relative TinyLLM on composition task accuracy by 31 points.",
                "Successful calibrated models increased paired distortion across depth while improving cosine decodability.",
            ],
            "suggested_hypotheses": [
                "An analytic sensor-frame relative canonicalizer passes the global relation gate under the full nuisance generator.",
                "Supervising a future two-vector before scalar projection prevents the learned relative arm from retaining the phase cover.",
            ],
            "implementation_correction": campaign["correction"],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [
            str(results_path),
            str(report_path),
            campaign["correction"]["original_artifact"],
        ],
    }


def build_io_correspondence_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _read_campaign(results_path)
    completed_at = datetime.fromisoformat(campaign["completed_at"].replace("Z", "+00:00"))
    results = []
    for detail in _read_details(results_path):
        metrics = _cell_metrics(detail)
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=min(
                    metrics["full_composition_cosine"],
                    metrics["full_extrapolation_cosine"],
                ),
                model_architecture=[8, int(detail["model_parameters"])],
                model_parameters=int(detail["model_parameters"]),
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["training"]["checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "The three-relation I/O descent contract passed before interpretation.",
                    "Schema v1.1 preserves a disclosed post-outcome equivariance correction.",
                    f"Implementation SHA-256: {detail['implementation_sha256']}.",
                ],
                anomalies=[],
                timestamp=completed_at,
            )
        )
    return results


def store_io_correspondence_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-io-correspondence.md"
    ),
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_io_correspondence_meta_hypothesis(results_path, report_path)
    experiment_results = build_io_correspondence_experiment_results(record, results_path)
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
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


__all__ = [
    "META_HYPOTHESIS_ID",
    "build_io_correspondence_experiment_results",
    "build_io_correspondence_meta_hypothesis",
    "store_io_correspondence_meta_hypothesis",
]
