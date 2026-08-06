"""Persist the primary TinyLLM nuisance-support scaling evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-nuisance-invariant-internal-cosine-quotient-v1"
META_SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA_VERSION = "nal.tinyllm-nuisance-support-scaling.v1"
SEEDS = (7, 17, 29, 41, 53)
SUPPORTS = ("N0", "N1", "N2", "N3")
ARMS = (
    "standard_final",
    "discrete_multi_exit",
    "continuous_gate",
    "quotient_contrastive",
)


def _read_campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported nuisance-support scaling schema")
    if campaign.get("status") != "completed":
        raise ValueError("nuisance-support scaling campaign is incomplete")
    summary = campaign.get("summary", {})
    if summary.get("completed") != 35 or summary.get("failed") != 0:
        raise ValueError("primary nuisance-support campaign must contain 35 successes")
    return campaign


def _primary_cells() -> list[tuple[str, str, int]]:
    cells = [
        (support, "standard_final", seed)
        for support in SUPPORTS
        for seed in SEEDS
    ]
    cells.extend(
        ("N3", arm, seed)
        for arm in ARMS
        if arm != "standard_final"
        for seed in SEEDS
    )
    return cells


def _read_primary_details(results_path: Path) -> list[dict[str, Any]]:
    root = results_path.parent / "runs"
    details = []
    for support, arm, seed in _primary_cells():
        path = root / support / arm / f"seed_{seed}" / "result.json"
        detail = json.loads(path.read_text(encoding="utf-8"))
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA_VERSION
            or detail.get("status") != "completed"
            or detail.get("support") != support
            or detail.get("training_arm") != arm
            or int(detail.get("seed", -1)) != seed
        ):
            raise ValueError(f"invalid primary cell: {path}")
        details.append(detail)
    return details


def _cell_metrics(detail: Mapping[str, Any]) -> dict[str, float]:
    cuts = detail["analysis"]["cuts"]

    def probe(cut: str, regime: str) -> Mapping[str, float]:
        return cuts[cut]["probe"]["evaluations"][regime]

    def gamma(cut: str, regime: str) -> float:
        return float(
            cuts[cut]["fiber_geometry"][regime]["fisher_rao"]["median_gamma"]
        )

    return {
        "early_composition_cosine": float(probe("early", "composition")["cosine_pearson"]),
        "early_composition_branch": float(probe("early", "composition")["balanced_accuracy"]),
        "early_extrapolation_cosine": float(probe("early", "extrapolation")["cosine_pearson"]),
        "early_extrapolation_branch": float(probe("early", "extrapolation")["balanced_accuracy"]),
        "full_composition_cosine": float(probe("full", "composition")["cosine_pearson"]),
        "full_composition_branch": float(probe("full", "composition")["balanced_accuracy"]),
        "full_composition_fisher_gamma": gamma("full", "composition"),
        "full_extrapolation_cosine": float(probe("full", "extrapolation")["cosine_pearson"]),
        "full_extrapolation_branch": float(probe("full", "extrapolation")["balanced_accuracy"]),
        "full_extrapolation_fisher_gamma": gamma("full", "extrapolation"),
        "post_attention_composition_branch": float(
            probe("post_attention", "composition")["balanced_accuracy"]
        ),
        "post_mlp_composition_branch": float(
            probe("post_mlp", "composition")["balanced_accuracy"]
        ),
    }


def build_nuisance_support_scaling_meta_hypothesis(
    results_path: Path,
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    details = _read_primary_details(results_path)
    aggregate = campaign["aggregates"]
    cells = aggregate["cells"]
    n0 = cells["N0:standard_final"]["cuts"]["full"]
    n3 = cells["N3:standard_final"]["cuts"]["full"]
    evidence = [
        {
            "evidence_role": "direct_nuisance_support_intervention",
            "experiment_id": detail["experiment_id"],
            "support": detail["support"],
            "training_arm": detail["training_arm"],
            "seed": detail["seed"],
            "checkpoint": detail["training"]["checkpoint"],
            "model_parameters": detail["model_parameters"],
            **_cell_metrics(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Broader nuisance support yields an invariant internal cosine quotient",
            "category": "representation",
            "description": (
                "Five-seed d8 interventions scale nested nuisance support and compare "
                "ordinary, joint-depth, and quotient-contrastive objectives at N3."
            ),
            "question": (
                "Does broader nuisance support retain cosine while erasing its phase "
                "branch under composition and outside-range extrapolation?"
            ),
            "prediction": (
                "Increasing nuisance diversity moves base retention toward one while "
                "conditional branch accuracy remains at chance."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "failure_despite_broad_coverage",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "d8 synthetic cosine task; primary-35 design; five seeds",
            "success_criteria": aggregate["preregistered_gates"],
            "subclaims": {
                "composition_base_retention_at_N3": "supported_all_objectives_in_mean",
                "composition_fiber_erasure_at_N3": "not_supported",
                "extrapolation_base_retention_at_N3": "not_supported",
                "extrapolation_branch_erasure_at_N3": "supported_descriptively",
                "monotonic_support_scaling": "not_supported_N2_regression",
                "objective_dependent_success": "not_supported",
                "block1_mlp_branch_suppression_under_shift": "supported_descriptively",
                "fiber_gamma_sufficient_for_identification": "contradicted",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "nuisance-support",
                "compositional-generalization",
                "extrapolation",
                "conditional-branch",
                "fiber-geometry",
                "multi-seed",
            ],
            "explicitly_not_tested": [
                "real-world sensor transfer",
                "certified conditional mutual information",
                "certified Reeb graph or cosheaf",
                "equivariant or invariant encoder intervention",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "N3 retained cosine on composition but leaked the phase branch; on "
                "extrapolation the branch was near chance but cosine remained below 0.9."
            ),
            "confidence": 0.0,
            "confidence_assessment": "five_seed_failure_of_preregistered_joint_gate",
            "effect_size": (
                n3["composition"]["cosine_pearson_mean"]
                - n0["composition"]["cosine_pearson_mean"]
            ),
            "effect_size_kind": "N3_minus_N0_full_composition_cosine_pearson",
            "independent_seed_count": 5,
            "model_class_count": 1,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "ordinary_N0_full_composition": n0["composition"],
                "ordinary_N3_full_composition": n3["composition"],
                "ordinary_N3_full_extrapolation": n3["extrapolation"],
                "N3_objectives": {
                    arm: cells[f"N3:{arm}"]["cuts"]["full"]
                    for arm in ARMS
                },
            },
            "key_insights": [
                "N3 causes a sharp transition to approximately 0.97 compositional cosine retention.",
                "Composition retains conditional branch information under every objective.",
                "Extrapolation loses cosine under every objective despite near-chance branch probes.",
                "Block-1 MLP computation suppresses most attention-exposed branch information.",
            ],
            "unexpected_findings": [
                "N2 performed worse than N1, so coverage improvement was not monotonic.",
                "Fisher, Euclidean, and cosine fiber gamma remained near one despite highly decodable branch signals.",
                "The explicit quotient objective did not cross the composition fiber-erasure gate.",
            ],
            "suggested_hypotheses": [
                "An equivariant sensor encoder preserves cosine under outside-range nuisance shift.",
                "A block-1 adversarial branch objective removes compositional fiber leakage while preserving cosine.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path)],
    }


def build_nuisance_support_scaling_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _read_campaign(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    results = []
    for detail in _read_primary_details(results_path):
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
                    f"Training nuisance support: {detail['support']}.",
                    f"Training objective: {detail['training_arm']}.",
                    "Exact cosine-matched conditional branch probes were frozen after training.",
                ],
                anomalies=[
                    "Fisher gamma near one is not sufficient evidence of branch erasure."
                ],
                timestamp=completed_at,
            )
        )
    return results


def store_nuisance_support_scaling_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_nuisance_support_scaling_meta_hypothesis(results_path)
    experiment_results = build_nuisance_support_scaling_experiment_results(
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
    "build_nuisance_support_scaling_experiment_results",
    "build_nuisance_support_scaling_meta_hypothesis",
    "store_nuisance_support_scaling_meta_hypothesis",
]
