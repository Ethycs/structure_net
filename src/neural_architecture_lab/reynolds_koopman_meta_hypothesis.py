"""Persist the Reynolds–Koopman quotient-closure result conservatively."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-reynolds-koopman.v1"
META_HYPOTHESIS_ID = "tinyllm-reynolds-koopman-quotient-closure-v1"
SEEDS = (7, 17, 29, 41, 53)
DEGREES = (2, 3)


def _read_campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != META_HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("summary") != {"requested": 10, "completed": 10, "failed": 0}
    ):
        raise ValueError("invalid or incomplete Reynolds–Koopman campaign")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _read_campaign(path)
    output = []
    for k in DEGREES:
        for seed in SEEDS:
            detail_path = path.parent / "runs" / f"k{k}" / f"seed_{seed}" / "result.json"
            detail = json.loads(detail_path.read_text())
            model_path = Path(detail["models_path"])
            if (
                detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("status") != "completed"
                or detail.get("implementation_sha256") != campaign["implementation_sha256"]
                or int(detail.get("k", -1)) != k
                or int(detail.get("seed", -1)) != seed
                or not model_path.is_file()
                or detail.get("models_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"invalid Reynolds–Koopman cell {detail_path}")
            output.append(detail)
    return output


def build_reynolds_koopman_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-reynolds-koopman.md"),
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    details = _details(results_path)
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "k": int(detail["k"]),
            "seed": int(detail["seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "causal_fronts": detail["causal_fronts"],
            "task_closure_fronts": detail["task_closure_fronts"],
            "gates": detail["gates"],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM Reynolds–Koopman quotient closure",
            "category": "representation",
            "description": "Exact task-orbit barycenters and low-rank invariant character products fitted as a nonstationary held-out observable family through depth.",
            "question": "Does barycentric observable closure identify the causal quotient front and predict unseen cover interventions?",
            "prediction": "Every primary front, cover-transition, autonomous-closure, intervention, and degree-ordering gate passes at the preregistered four-of-five threshold.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_predictive_barycenter_front_precedes_causal_quotient_front",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "retained d6 degree-2/3 ladder; five seeds; exact orbits; composition and extrapolation",
            "subclaims": {
                "cover_gain_transition_k2": "supported_five_of_five",
                "cover_gain_transition_k3": "supported_five_of_five",
                "causal_front_agreement_k2": "not_supported_three_of_five",
                "causal_front_agreement_k3": "not_supported_one_of_five",
                "autonomous_closure_k2": "not_supported_one_of_five",
                "autonomous_closure_k3": "supported_five_of_five_near_predictive_front",
                "unseen_cover_response_k2": "not_supported_one_of_five",
                "unseen_cover_response_k3": "not_supported_one_of_five",
                "degree_three_cubic_synthesis": "not_supported",
                "full_hypothesis": "not_confirmed",
            },
            "tags": ["tinyllm", "reynolds-operator", "koopman", "deck-group", "causal-front", "multi-seed"],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "The cover-gain transition replicated, but predictive fronts failed causal agreement, autonomous k2 closure, and unseen-lambda response gates.",
            "confidence": 0.0,
            "confidence_assessment": "complete_five_seed_preregistered_gate_failure_with_supported_secondary_transition",
            "independent_seed_count": 5,
            "num_direct_experiments": len(evidence),
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Cover-generated invariants dominate at the frontend and become unnecessary after block-0 attention in every seed.",
                "An external linear observer often reads the task before exact orbit averaging becomes causally safe for the frozen transformer.",
                "Autonomous next-barycenter closure and final-task readability are distinct, especially for degree two.",
                "The finite lift does not stably predict unseen cover contraction and amplification.",
                "Degree-three response is not cubic-dominant, and generic random features match the structured lift predictively.",
            ],
            "suggested_hypotheses": [
                "The block-0 barycenter contains a readable but downstream-misaligned task chart before the causal quotient front.",
                "Causal closure requires modeling the frozen downstream decoder action, not only observability of the final task.",
                "A local nonlinear response atlas is necessary outside the lambda neighborhood used for fitting.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_reynolds_koopman_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_read_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    output = []
    for detail in _details(results_path):
        agreements = [
            item["cut_distance"]
            for item in detail["front_agreement"].values()
            if item["cut_distance"] is not None
        ]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=META_HYPOTHESIS_ID,
                metrics={
                    "maximum_front_distance": float(max(agreements, default=-1)),
                    "cover_transition_pass": float(detail["gates"]["cover_transition_both_shifts"]),
                    "autonomous_closure_pass": float(detail["gates"]["autonomous_one_step_near_front_both_shifts"]),
                    "intervention_prediction_pass": float(detail["gates"]["unseen_lambda_intervention_both_shifts"]),
                },
                primary_metric=float(detail["gates"]["front_agreement_both_shifts"]),
                model_architecture=[6, int(detail["k"])],
                model_parameters=29_956_224,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=["Frozen checkpoint; exact task orbits; nonstationary held-out observable models."],
                anomalies=["Predictive barycenter closure often precedes causal orbit-average sufficiency."],
                timestamp=completed,
            )
        )
    return output


def store_reynolds_koopman_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_reynolds_koopman_meta_hypothesis(results_path)
    experiments = build_reynolds_koopman_experiment_results(record, results_path)
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
        storage["result_hashes"] = [logger.log_experiment_result(item) for item in experiments]
        storage["chromadb_path"] = str(chromadb_path)
        storage["hypotheses_collection"] = "hypotheses"
        storage["experiments_collection"] = "experiments"
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


__all__ = [
    "build_reynolds_koopman_meta_hypothesis",
    "build_reynolds_koopman_experiment_results",
    "store_reynolds_koopman_meta_hypothesis",
]
