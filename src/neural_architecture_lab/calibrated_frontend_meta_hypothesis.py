"""Persist the TinyLLM calibrated-front-end causal evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-calibrated-reference-stable-cosine-quotient-v1"
META_SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA_VERSION = "nal.tinyllm-calibrated-frontend-causal.v1"
SEEDS = (7, 17, 29, 41, 53)
CONDITIONS = (
    "raw_calibrated",
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
CUTS = ("frontend", "full")
PRIMARY_REGIMES = ("composition", "extrapolation")


def _read_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported calibrated-front-end campaign schema")
    if campaign.get("status") != "completed":
        raise ValueError("calibrated-front-end campaign is incomplete")
    summary = campaign.get("summary", {})
    if summary.get("completed") != 15 or summary.get("failed") != 0:
        raise ValueError("calibrated-front-end campaign must contain 15 successes")
    if not campaign.get("implementation_sha256"):
        raise ValueError("campaign lacks a frozen implementation digest")
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
            contract = detail.get("identifiability_contract", {})
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA_VERSION
                or detail.get("status") != "completed"
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("implementation_sha256") != implementation
                or not detail.get("scientific_fingerprint")
                or not checkpoint.is_file()
                or not contract.get("passed")
                or contract.get("violations") != 0
            ):
                raise ValueError(f"invalid calibrated-front-end cell: {path}")
            if condition != "raw_calibrated":
                frontend = Path(detail["training"].get("frontend_checkpoint", ""))
                if not frontend.is_file():
                    raise ValueError(f"missing structured front-end checkpoint: {path}")
            details.append(detail)
    return details


def _probe(detail: Mapping[str, Any], cut: str, regime: str) -> Mapping[str, float]:
    return detail["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]


def _cell_metrics(detail: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for cut in CUTS:
        for regime in PRIMARY_REGIMES:
            probe = _probe(detail, cut, regime)
            prefix = f"{cut}_{regime}"
            result[f"{prefix}_cosine"] = float(probe["cosine_pearson"])
            result[f"{prefix}_branch"] = float(probe["balanced_accuracy"])
            result[f"{prefix}_log_loss_gain"] = float(
                probe["conditional_log_loss_gain_over_cosine_only"]
            )
    for regime in ("in_distribution", *PRIMARY_REGIMES):
        result[f"{regime}_task_accuracy"] = float(
            detail["analysis"]["task_metrics"][regime]["exact_bin_accuracy"]
        )
    return result


def build_calibrated_frontend_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-calibrated-identifiability-causal.md"
    ),
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    details = _read_details(results_path)
    aggregate = campaign["aggregates"]
    arms = aggregate["arms"]
    raw = arms["raw_calibrated"]
    analytic = arms["analytic_calibrated"]
    learned = arms["learned_calibrated_equivariant"]
    evidence = [
        {
            "evidence_role": {
                "raw_calibrated": "matched_information_control",
                "analytic_calibrated": "analytic_positive_control",
                "learned_calibrated_equivariant": "direct_architectural_intervention",
            }[detail["condition"]],
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "checkpoint": detail["training"]["checkpoint"],
            "frontend_checkpoint": detail["training"].get("frontend_checkpoint"),
            "model_parameters": detail["model_parameters"],
            "system_parameters": detail["system_parameters"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "identifiability_contract": detail["identifiability_contract"],
            **_cell_metrics(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Gauge repair makes the absolute-cosine quotient constructible",
            "category": "representation",
            "description": (
                "A three-arm d8/N3 causal comparison supplies a phase-independent "
                "calibration reference to raw TinyLLM, an analytic canonicalizer, and a "
                "structurally equivariant learned sensor encoder."
            ),
            "question": (
                "Does fixing the observation gauge and respecting the nuisance symmetry "
                "stabilize cosine retention while removing conditional phase branch?"
            ),
            "prediction": (
                "At least four of five structured seeds simultaneously retain cosine at "
                "least 0.90 and limit conditional branch accuracy to at most 0.55 at the "
                "front end and full depth on composition and extrapolation."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": "confirmed_gauge_repair_constructible",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": (
                "d8 synthetic absolute-cosine task; N3; observed calibration reference; "
                "three arms; five matched seeds"
            ),
            "success_criteria": aggregate["preregistered_gate"],
            "subclaims": {
                "calibrated_target_identifiable_on_observation_quotient": "supported_by_contract",
                "raw_information_availability_is_sufficient": "contradicted_zero_of_five",
                "analytic_positive_control_joint_gate": "supported_five_of_five",
                "learned_equivariant_joint_gate": "supported_five_of_five",
                "learned_composition_stable_quotient": "supported_five_of_five",
                "learned_extrapolation_stable_quotient": "supported_five_of_five",
                "full_preregistered_hypothesis": "confirmed",
            },
            "tags": [
                "tinyllm",
                "causal-intervention",
                "identifiability",
                "gauge-fixing",
                "equivariance",
                "analytic-canonicalizer",
                "sensor-encoder",
                "extrapolation",
                "multi-seed",
            ],
            "explicitly_not_tested": [
                "noisy or missing calibration fields",
                "a gauge-invariant relative-cosine target without calibration",
                "real-world sensor transfer",
                "all possible equivariant encoder families",
            ],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Both the analytic and learned structured arms passed every primary cut "
                "and shift in all five seeds; the matched raw calibrated arm passed in zero."
            ),
            "confidence": 1.0,
            "confidence_assessment": "five_seed_pass_of_preregistered_joint_gate",
            "effect_sizes": {
                "learned_minus_raw_full_extrapolation_cosine": float(
                    learned["cuts"]["full"]["extrapolation"]["cosine_pearson_mean"]
                    - raw["cuts"]["full"]["extrapolation"]["cosine_pearson_mean"]
                ),
                "learned_minus_raw_full_composition_branch": float(
                    learned["cuts"]["full"]["composition"]["branch_balanced_accuracy_mean"]
                    - raw["cuts"]["full"]["composition"]["branch_balanced_accuracy_mean"]
                ),
                "learned_minus_analytic_full_extrapolation_cosine": float(
                    learned["cuts"]["full"]["extrapolation"]["cosine_pearson_mean"]
                    - analytic["cuts"]["full"]["extrapolation"]["cosine_pearson_mean"]
                ),
            },
            "independent_seed_count": 5,
            "model_class_count": 1,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "raw": raw,
                "analytic": analytic,
                "learned": learned,
                "joint_gate": aggregate["preregistered_gate"],
            },
            "key_insights": [
                "The exact calibration reference removed every target-changing ambiguity in the declared gauge grid.",
                "The analytic and learned structured arms passed the complete joint gate in all five seeds.",
                "The raw transformer received the same calibration data but did not learn a stable quotient.",
                "The learned encoder approached the analytic control in full-depth extrapolation geometry.",
            ],
            "unexpected_findings": [
                "The learned representation nearly matched the analytic control while retaining a sizable extrapolation task-accuracy gap.",
                "Transformer depth sharpened the analytic canonicalizer's extrapolation cosine correlation.",
            ],
            "suggested_hypotheses": [
                "The joint gate degrades continuously with calibration noise until an identifiable-reference threshold is crossed.",
                "A gauge-invariant relative target can produce the same stable quotient without a calibration packet.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_calibrated_frontend_experiment_results(
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
                    "The identifiability contract passed before training.",
                    "Primary measurements use fresh frozen cosine-conditioned nonlinear probes.",
                    f"Implementation SHA-256: {detail['implementation_sha256']}.",
                ],
                anomalies=[],
                timestamp=completed_at,
            )
        )
    return results


def store_calibrated_frontend_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-calibrated-identifiability-causal.md"
    ),
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_calibrated_frontend_meta_hypothesis(results_path, report_path)
    experiment_results = build_calibrated_frontend_experiment_results(record, results_path)
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
    "build_calibrated_frontend_experiment_results",
    "build_calibrated_frontend_meta_hypothesis",
    "store_calibrated_frontend_meta_hypothesis",
]
