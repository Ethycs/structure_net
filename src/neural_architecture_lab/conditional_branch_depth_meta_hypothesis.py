"""Persist the multi-seed TinyLLM conditional branch-depth evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-joint-depth-internal-cosine-quotient-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-conditional-branch-depth-scan.v1"


def _read_results(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported conditional branch-depth schema")
    if campaign.get("status") != "completed" or len(campaign.get("runs", [])) != 18:
        raise ValueError("conditional branch-depth campaign is incomplete")
    cosine = [run for run in campaign["runs"] if run["quotient"] == "cosine_interval"]
    phase = [run for run in campaign["runs"] if run["quotient"] == "phase_circle"]
    if len(cosine) != 15 or len(phase) != 3:
        raise ValueError("campaign must contain 15 cosine runs and three phase controls")
    return campaign


def _final_metrics(run: Mapping[str, Any]) -> dict[str, float]:
    cell = run["depth_cells"][-1]
    identity = cell["probes"]["nonlinear"]["evaluations"][
        "overlapping_held_out"
    ]
    shifted = cell["probes"]["nonlinear"]["evaluations"]["disjoint_nuisance"]
    post_attention = run["transition_sublayer_scan"]["stages"][1]["probes"][
        "nonlinear"
    ]["evaluations"]["overlapping_held_out"]
    post_mlp = run["transition_sublayer_scan"]["stages"][2]["probes"][
        "nonlinear"
    ]["evaluations"]["overlapping_held_out"]
    return {
        "full_id_branch_accuracy": float(identity["balanced_accuracy"]),
        "full_ood_branch_accuracy": float(shifted["balanced_accuracy"]),
        "full_id_cosine_pearson": float(identity["cosine_pearson"]),
        "full_ood_cosine_pearson": float(shifted["cosine_pearson"]),
        "full_id_conditional_log_loss_gain": float(
            identity["conditional_log_loss_gain_over_cosine_only"]
        ),
        "full_posterior_h1": float(cell["decoder"]["posterior_normalized_h1_lifetime"]),
        "block1_post_attention_branch_accuracy": float(
            post_attention["balanced_accuracy"]
        ),
        "block1_post_attention_log_loss_gain": float(
            post_attention["conditional_log_loss_gain_over_cosine_only"]
        ),
        "block1_post_mlp_branch_accuracy": float(post_mlp["balanced_accuracy"]),
    }


def _mean_by_arm(
    runs: list[Mapping[str, Any]], arm: str, metric: str
) -> float:
    values = [_final_metrics(run)[metric] for run in runs if run["training_arm"] == arm]
    return float(np.mean(values))


def build_conditional_branch_depth_meta_hypothesis(
    results_path: Path,
) -> dict[str, Any]:
    campaign = _read_results(results_path)
    runs = campaign["runs"]
    cosine = [run for run in runs if run["quotient"] == "cosine_interval"]
    phase = [run for run in runs if run["quotient"] == "phase_circle"]
    aggregates = campaign["aggregates"]
    arm_metrics: dict[str, Any] = {}
    for arm, aggregate in aggregates["cosine_arms"].items():
        arm_metrics[arm] = {
            "median_residual_quotient_front_id": aggregate[
                "median_residual_quotient_front_id"
            ],
            "median_decoder_supported_quotient_front_id": aggregate[
                "median_decoder_supported_quotient_front_id"
            ],
            "nuisance_robust_front_found_fraction": aggregate[
                "nuisance_robust_front_found_fraction"
            ],
            "full_id_branch_accuracy_mean": _mean_by_arm(
                cosine, arm, "full_id_branch_accuracy"
            ),
            "full_ood_branch_accuracy_mean": _mean_by_arm(
                cosine, arm, "full_ood_branch_accuracy"
            ),
            "full_ood_cosine_pearson_mean": _mean_by_arm(
                cosine, arm, "full_ood_cosine_pearson"
            ),
            "block1_post_attention_branch_accuracy_mean": _mean_by_arm(
                cosine, arm, "block1_post_attention_branch_accuracy"
            ),
            "block1_post_attention_log_loss_gain_mean": _mean_by_arm(
                cosine, arm, "block1_post_attention_log_loss_gain"
            ),
            "block1_post_mlp_branch_accuracy_mean": _mean_by_arm(
                cosine, arm, "block1_post_mlp_branch_accuracy"
            ),
        }
    phase_final = [_final_metrics(run) for run in phase]
    evidence = []
    for run in runs:
        metrics = _final_metrics(run)
        evidence.append(
            {
                "evidence_role": "direct_frozen_conditional_probe",
                "experiment_id": run["experiment_id"],
                "training_arm": run["training_arm"],
                "quotient": run["quotient"],
                "seed": run["seed"],
                "checkpoint": run["checkpoint"],
                "model_frozen": run["model_frozen"],
                "residual_quotient_front_id": run.get(
                    "residual_quotient_front_id"
                ),
                "decoder_supported_quotient_front_id": run.get(
                    "decoder_supported_quotient_front_id"
                ),
                "nuisance_robust_quotient_front": run.get(
                    "nuisance_robust_quotient_front"
                ),
                **metrics,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Joint-depth training moves an internal cosine quotient earlier",
            "category": "representation",
            "description": (
                "Five seeds of three cosine-training arms are frozen and probed at "
                "real depths on exact cosine-matched phase branches, with phase controls, "
                "linear/nonlinear and shuffled probes, shifted nuisances, posterior H1, "
                "finite-fiber components, and block-1 attention/MLP cuts."
            ),
            "question": (
                "Does the cosine quotient form in the residual stream, and does joint-"
                "depth supervision move branch erasure toward the network entrance?"
            ),
            "prediction": campaign["pre_registered_prediction"],
            "created_at": campaign["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "internal_quotient_supported_id_nuisance_robust_claim_not_confirmed"
            ),
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "d8 synthetic circle/cosine task; five cosine seeds; seed-7 phase controls",
            "success_criteria": aggregates["pre_registered_criteria"],
            "subclaims": {
                "cosine_residual_branch_erasure_full_depth": "supported_five_seeds_all_arms",
                "phase_control_branch_retention": "supported_seed7_all_arms",
                "in_distribution_residual_quotient": "supported_five_seeds_all_arms",
                "joint_depth_moves_median_id_residual_front_earlier": "supported_descriptively",
                "ordinary_posterior_front_marks_internal_erasure_onset": "contradicted",
                "nuisance_robust_internal_quotient": "not_supported_cosine_not_preserved",
                "attention_mlp_mechanism": "ordinary_attention_branch_transient_removed_by_mlp",
                "reeb_component_agreement": "not_resolved_finite_mst_proxy_insensitive_to_transient",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "multi-seed",
                "conditional-branch",
                "real-depth",
                "linear-probe",
                "nonlinear-probe",
                "label-shuffle-control",
                "attention-mlp-localization",
                "nuisance-shift",
            ],
            "explicitly_not_tested": [
                "causal intervention on branch-bearing residual directions",
                "certified conditional mutual information",
                "certified Reeb graph or cosheaf",
                "real-world sensor transfer",
                "phase controls beyond seed 7",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All cosine arms erased conditional branch information at full depth and "
                "joint-depth arms had an earlier median in-distribution residual front. "
                "However, no run passed the preregistered nuisance-robust front because "
                "disjoint-family cosine decoding remained below 0.9."
            ),
            "confidence": 0.0,
            "confidence_assessment": "multi_seed_partial_support_with_failed_robustness_gate",
            "effect_size": arm_metrics["standard_final"][
                "block1_post_attention_branch_accuracy_mean"
            ]
            - arm_metrics["continuous_gate"][
                "block1_post_attention_branch_accuracy_mean"
            ],
            "effect_size_kind": "standard_minus_continuous_block1_post_attention_branch_accuracy",
            "independent_seed_count": 5,
            "phase_control_seed_count": 1,
            "model_class_count": 1,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "cosine_arms": arm_metrics,
                "phase_full_id_branch_accuracy_mean": float(
                    np.mean([item["full_id_branch_accuracy"] for item in phase_final])
                ),
                "phase_full_ood_branch_accuracy_mean": float(
                    np.mean([item["full_ood_branch_accuracy"] for item in phase_final])
                ),
            },
            "key_insights": [
                "All 15 cosine checkpoints had a finite in-distribution residual quotient front.",
                "Median residual fronts were 0.020 ordinary and 0.005 for both joint-depth arms.",
                "Ordinary block-1 attention exposed a nonlinear branch transient that its MLP removed.",
                "Phase controls retained branch information at approximately 100% ID and 95.9% OOD.",
            ],
            "unexpected_findings": [
                "The earlier ordinary cosine front at 1.85 tracked decoder-supported geometry, not first branch erasure.",
                "No cosine arm preserved cosine above 0.9 on the disjoint nuisance family at full depth.",
                "The finite-fiber MST proxy reported one component even where nonlinear probes detected a branch transient.",
            ],
            "suggested_hypotheses": [
                "Ordinary block-1 attention transiently reconstructs phase branch information that the MLP cancels.",
                "Nuisance-robust cosine representations require explicit nuisance-family training or invariance objectives.",
                "Causal ablation of nonlinear branch-readable directions leaves cosine decoding intact.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path)],
    }


def build_conditional_branch_depth_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _read_results(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    results = []
    for run in campaign["runs"]:
        metrics = _final_metrics(run)
        residual_front = run.get("residual_quotient_front_id")
        decoder_front = run.get("decoder_supported_quotient_front_id")
        metrics["residual_quotient_front_id"] = (
            float(residual_front) if residual_front is not None else -1.0
        )
        metrics["decoder_supported_quotient_front_id"] = (
            float(decoder_front) if decoder_front is not None else -1.0
        )
        metrics["accuracy"] = metrics["full_id_branch_accuracy"]
        is_cosine = run["quotient"] == "cosine_interval"
        results.append(
            ExperimentResult(
                experiment_id=run["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=(
                    1.0 - abs(metrics["full_id_branch_accuracy"] - 0.5) * 2.0
                    if is_cosine
                    else metrics["full_id_branch_accuracy"]
                ),
                model_architecture=[8, int(run["model_parameters"])],
                model_parameters=int(run["model_parameters"]),
                training_time=float(run["analysis_seconds"]),
                model_checkpoint=run["checkpoint"],
                observations=[
                    f"Frozen task quotient: {run['quotient']}.",
                    f"Depth-training arm: {run['training_arm']}.",
                    "Branch labels were balanced within exact cosine-matched pairs.",
                ],
                anomalies=(
                    ["No nuisance-robust quotient front passed the 0.9 shifted-cosine gate."]
                    if is_cosine
                    else ["Phase controls were retained only for seed 7."]
                ),
                timestamp=completed_at,
            )
        )
    return results


def store_conditional_branch_depth_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_conditional_branch_depth_meta_hypothesis(results_path)
    experiment_results = build_conditional_branch_depth_experiment_results(
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
    "build_conditional_branch_depth_experiment_results",
    "build_conditional_branch_depth_meta_hypothesis",
    "store_conditional_branch_depth_meta_hypothesis",
]
