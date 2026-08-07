"""Persist the TinyLLM invariant-front-end causal evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-invariant-frontend-stable-cosine-quotient-v1"
META_SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA_VERSION = "nal.tinyllm-invariant-frontend-causal.v1"
SEEDS = (7, 17, 29, 41, 53)
CONDITIONS = ("raw_n3", "analytic_invariant", "learned_equivariant")
CUTS = ("frontend", "post_attention", "post_mlp", "full")
PRIMARY_REGIMES = ("composition", "extrapolation")


def _read_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported invariant-front-end campaign schema")
    if campaign.get("status") != "completed":
        raise ValueError("invariant-front-end campaign is incomplete")
    summary = campaign.get("summary", {})
    if summary.get("completed") != 15 or summary.get("failed") != 0:
        raise ValueError("invariant-front-end campaign must contain 15 successes")
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
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA_VERSION
                or detail.get("status") != "completed"
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("implementation_sha256") != implementation
                or not detail.get("scientific_fingerprint")
                or not checkpoint.is_file()
            ):
                raise ValueError(f"invalid invariant-front-end cell: {path}")
            if condition != "raw_n3":
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


def build_invariant_frontend_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-invariant-frontend-causal.md"
    ),
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    details = _read_details(results_path)
    aggregate = campaign["aggregates"]
    arms = aggregate["arms"]
    learned = arms["learned_equivariant"]
    raw = arms["raw_n3"]
    analytic = arms["analytic_invariant"]
    evidence = [
        {
            "evidence_role": {
                "raw_n3": "retained_matched_control",
                "analytic_invariant": "analytic_positive_control",
                "learned_equivariant": "direct_architectural_intervention",
            }[detail["condition"]],
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "checkpoint": detail["training"]["checkpoint"],
            "frontend_checkpoint": detail["training"].get("frontend_checkpoint"),
            "model_parameters": detail["model_parameters"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            **_cell_metrics(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Architectural nuisance symmetry creates a stable cosine quotient",
            "category": "representation",
            "description": (
                "A three-arm d8/N3 causal comparison tests raw TinyLLM, an observation-only "
                "analytic canonicalizer, and a task-trained structurally equivariant sensor encoder."
            ),
            "question": (
                "Can restricting the sensor architecture to respect the nuisance symmetry "
                "stabilize cosine-base retention while removing conditional phase branch?"
            ),
            "prediction": (
                "The same four of five learned seeds retain cosine at least 0.90 and limit "
                "conditional branch accuracy to at most 0.55 at four cuts on composition "
                "and extrapolation."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_support_relative_only",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "d8 synthetic absolute-cosine task; N3; three arms; five matched seeds",
            "success_criteria": aggregate["preregistered_gate"],
            "subclaims": {
                "raw_contains_conditional_branch": "supported",
                "analytic_branch_contraction": "supported",
                "analytic_base_retention": "not_supported",
                "learned_composition_joint_four_of_five": "supported",
                "learned_extrapolation_base_threshold": "not_supported_zero_of_five",
                "learned_extrapolation_branch_contraction": "supported",
                "full_preregistered_hypothesis": "not_confirmed_zero_of_five_joint",
                "absolute_cosine_identifiable_on_observation_quotient": "contradicted_by_exact_gauge",
            },
            "tags": [
                "tinyllm",
                "causal-intervention",
                "equivariance",
                "analytic-canonicalizer",
                "sensor-encoder",
                "compositional-generalization",
                "extrapolation",
                "identifiability",
                "multi-seed",
            ],
            "explicitly_not_tested": [
                "an observed orientation calibration channel",
                "a gauge-invariant relative-cosine target",
                "real-world sensor transfer",
                "certified conditional mutual information",
                "all possible invariant or equivariant encoder families",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "No arm had a seed pass every cut on both primary shifts. Four learned "
                "seeds passed every composition cut, but zero passed extrapolation because "
                "cosine retention remained below 0.90."
            ),
            "confidence": 0.0,
            "confidence_assessment": "five_seed_failure_of_preregistered_joint_gate",
            "effect_sizes": {
                "learned_minus_raw_full_composition_branch": float(
                    learned["cuts"]["full"]["composition"]["branch_balanced_accuracy_mean"]
                    - raw["cuts"]["full"]["composition"]["branch_balanced_accuracy_mean"]
                ),
                "learned_minus_raw_full_extrapolation_cosine": float(
                    learned["cuts"]["full"]["extrapolation"]["cosine_pearson_mean"]
                    - raw["cuts"]["full"]["extrapolation"]["cosine_pearson_mean"]
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
                "The learned equivariant bottleneck erased tested branch information at every cut under both shifts.",
                "Four learned seeds formed a compositional quotient at all four cuts.",
                "All learned seeds lost the cosine base under extrapolation, so the quotient remained support-relative.",
                "The analytic positive control erased branch but did not retain the base under quantized noisy observation.",
                "An exact generator gauge maps identical observations to different absolute-cosine targets.",
            ],
            "unexpected_findings": [
                "One learned seed collapsed on composition while the other four passed every cut.",
                "The learned arm contracted branch more cleanly than the analytic arm retained cosine.",
                "Extrapolation task accuracy was similar to raw despite much lower probed cosine geometry.",
            ],
            "suggested_hypotheses": [
                "An observed calibration reference that fixes planar orientation makes the absolute-cosine quotient identifiable and extrapolatable.",
                "A gauge-invariant cosine-relative-to-reference target is retained by the equivariant encoder under outside-range shift.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_invariant_frontend_experiment_results(
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
                    "Primary measurements use fresh frozen cosine-conditioned nonlinear probes.",
                    f"Implementation SHA-256: {detail['implementation_sha256']}.",
                ],
                anomalies=(
                    ["The absolute target is not constant on the exact observation gauge."]
                    if detail["condition"] != "raw_n3"
                    else []
                ),
                timestamp=completed_at,
            )
        )
    return results


def store_invariant_frontend_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-invariant-frontend-causal.md"
    ),
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_invariant_frontend_meta_hypothesis(results_path, report_path)
    experiment_results = build_invariant_frontend_experiment_results(record, results_path)
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
    "build_invariant_frontend_experiment_results",
    "build_invariant_frontend_meta_hypothesis",
    "store_invariant_frontend_meta_hypothesis",
]
