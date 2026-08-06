"""Persist the causal TinyLLM block-1 quotient-control evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-block1-horizontal-vertical-control-v1"
META_SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA_VERSION = "nal.tinyllm-block1-quotient-control.v1"
SEEDS = (7, 17, 29, 41, 53)
CONDITIONS = ("ordinary_n3", "block1_quotient_controlled_n3")


def _read_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported block-1 quotient-control schema")
    if campaign.get("status") != "completed":
        raise ValueError("block-1 quotient-control campaign is incomplete")
    summary = campaign.get("summary", {})
    if summary.get("completed") != 10 or summary.get("failed") != 0:
        raise ValueError("block-1 quotient-control campaign must contain 10 successes")
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
                raise ValueError(f"invalid block-1 quotient-control cell: {path}")
            details.append(detail)
    return details


def _probe(detail: Mapping[str, Any], cut: str, regime: str) -> Mapping[str, float]:
    return detail["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]


def _cell_metrics(detail: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for cut in ("post_mlp", "full"):
        for regime in ("composition", "extrapolation"):
            probe = _probe(detail, cut, regime)
            prefix = f"{cut}_{regime}"
            result[f"{prefix}_cosine"] = float(probe["cosine_pearson"])
            result[f"{prefix}_branch"] = float(probe["balanced_accuracy"])
            result[f"{prefix}_log_loss_gain"] = float(
                probe["conditional_log_loss_gain_over_cosine_only"]
            )
    for regime in ("in_distribution", "composition", "extrapolation"):
        result[f"full_{regime}_task_accuracy"] = float(
            detail["task_metrics"][regime]["full"]["exact_bin_accuracy"]
        )
        result[f"post_mlp_{regime}_q_local"] = float(
            detail["local_derivative"][regime]["post_mlp"]["median_q_local"]
        )
    return result


def build_block1_quotient_control_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-block1-quotient-control.md"
    ),
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    details = _read_details(results_path)
    aggregate = campaign["aggregates"]
    gate = aggregate["preregistered_gate"]
    cells = aggregate["cells"]
    baseline = cells["ordinary_n3"]["cuts"]
    controlled = cells["block1_quotient_controlled_n3"]["cuts"]
    evidence = [
        {
            "evidence_role": (
                "retained_matched_control"
                if detail["condition"] == "ordinary_n3"
                else "direct_block1_quotient_intervention"
            ),
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "checkpoint": detail["training"]["checkpoint"],
            "model_parameters": detail["model_parameters"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            **_cell_metrics(detail),
        }
        for detail in details
    ]
    effect = (
        controlled["full"]["extrapolation"]["cosine_pearson_mean"]
        - baseline["full"]["extrapolation"]["cosine_pearson_mean"]
    )
    return {
        "schema_version": META_SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Block-1 horizontal/vertical control creates a stable cosine quotient",
            "category": "representation",
            "description": (
                "A matched d8/N3 intervention preserves cosine with a temporary linear "
                "head and contracts its phase branch with a cosine-conditioned gradient-"
                "reversal adversary at block-1 post-MLP."
            ),
            "question": (
                "Can independent control of semantic-base rank and nuisance-fiber "
                "contraction create a quotient stable to composition and extrapolation?"
            ),
            "prediction": (
                "At post-MLP and full depth, cosine remains at least 0.90 while tested "
                "conditional branch accuracy is at most 0.55 in four of five seeds."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": bool(gate["overall_success"]),
            "confirmation_status": "not_confirmed_no_joint_improvement",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "d8 synthetic cosine task; N3; matched two-condition five-seed campaign",
            "success_criteria": gate,
            "subclaims": {
                "composition_base_preserved": "supported",
                "extrapolation_base_threshold": "not_supported",
                "composition_branch_contracted": "contradicted_descriptively",
                "extrapolation_branch_contracted": "not_supported",
                "task_noninferiority_joint_four_of_five": "not_supported_three_of_five",
                "local_horizontal_vertical_ratio_improved": "not_supported",
                "full_preregistered_hypothesis": "not_confirmed_zero_of_five",
            },
            "tags": [
                "tinyllm",
                "causal-intervention",
                "gradient-reversal",
                "conditional-adversary",
                "block-1",
                "compositional-generalization",
                "extrapolation",
                "multi-seed",
            ],
            "explicitly_not_tested": [
                "equivariant or invariant sensor encoder",
                "real-world sensor transfer",
                "certified conditional mutual information",
                "universal adversarial invariance",
            ],
        },
        "result": {
            "confirmed": bool(gate["overall_success"]),
            "confirmation_reason": (
                "No controlled seed passed the joint representation gate. Extrapolation "
                "cosine remained below 0.90 in every seed, and compositional branch "
                "leakage increased in mean."
            ),
            "confidence": 0.0,
            "confidence_assessment": "five_seed_failure_of_preregistered_joint_gate",
            "effect_size": float(effect),
            "effect_size_kind": "controlled_minus_ordinary_full_extrapolation_cosine",
            "independent_seed_count": 5,
            "model_class_count": 1,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "ordinary": baseline,
                "controlled": controlled,
                "task_performance": aggregate["task_performance"],
                "local_q_mean": aggregate["local_q_mean"],
                "joint_gate": gate,
            },
            "key_insights": [
                "The base head modestly improved mean extrapolation cosine but not near the 0.90 threshold.",
                "The controlled model leaked more compositional branch information in mean.",
                "The online adversary approached chance while a fresh frozen probe decoded the branch.",
                "The block-1 MLP still suppressed attention-exposed branch signal but did not create a stable quotient.",
            ],
            "unexpected_findings": [
                "Mean compositional branch leakage increased under the branch adversary.",
                "The local composition quotient ratio decreased under the intervention.",
                "Only three seeds passed task noninferiority jointly despite small mean task changes.",
            ],
            "suggested_hypotheses": [
                "An equivariant sensor encoder with invariant readout preserves cosine under outside-range nuisance shift.",
                "A fresh-adversary or adversary-ensemble objective is necessary if residual adversarial training is revisited.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_block1_quotient_control_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _read_campaign(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    results = []
    for detail in _read_details(results_path):
        metrics = _cell_metrics(detail)
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=min(
                    metrics["post_mlp_extrapolation_cosine"],
                    metrics["full_extrapolation_cosine"],
                ),
                model_architecture=[8, int(detail["model_parameters"])],
                model_parameters=int(detail["model_parameters"]),
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["training"]["checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "Primary measurements use fresh frozen conditional nonlinear probes.",
                    f"Implementation SHA-256: {detail['implementation_sha256']}.",
                ],
                anomalies=[
                    "Online adversary behavior does not substitute for fresh-probe decodability."
                ],
                timestamp=completed_at,
            )
        )
    return results


def store_block1_quotient_control_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-block1-quotient-control.md"
    ),
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_block1_quotient_control_meta_hypothesis(
        results_path, report_path
    )
    experiment_results = build_block1_quotient_control_experiment_results(
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
    "build_block1_quotient_control_experiment_results",
    "build_block1_quotient_control_meta_hypothesis",
    "store_block1_quotient_control_meta_hypothesis",
]
