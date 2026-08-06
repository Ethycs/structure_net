"""Persist the TinyLLM depth-graded quotient campaign conservatively."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-depth-graded-task-fronts-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-depth-graded-quotient.v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_campaign(path: Path) -> dict[str, Any]:
    campaign = _read_json(path)
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported depth-graded campaign schema")
    if campaign.get("hypothesis_id") != META_HYPOTHESIS_ID:
        raise ValueError("depth-graded campaign has the wrong hypothesis id")
    if campaign.get("status") != "completed" or len(campaign.get("runs", [])) != 6:
        raise ValueError("depth-graded campaign is incomplete")
    return campaign


def _refined_fronts(results_path: Path) -> tuple[dict[str, Any], list[str]]:
    root = results_path.parent
    full_path = root / "front_refinement.json"
    early_path = root / "front_refinement_early.json"
    full = _read_json(full_path)
    early = _read_json(early_path)
    full_by_id = {item["experiment_id"]: item for item in full["records"]}
    early_by_id = {item["experiment_id"]: item for item in early["records"]}
    refined = {}
    for experiment_id, full_record in full_by_id.items():
        early_record = early_by_id[experiment_id]
        early_front = early_record["refined_front_depth"]
        refined[experiment_id] = {
            "refined_front_depth": (
                early_front
                if early_front is not None
                else full_record["refined_front_depth"]
            ),
            "front_resolution": (
                float(early["depth_step"])
                if early_front is not None
                else float(full["depth_step"])
            ),
            "full_sweep_front_depth": full_record["refined_front_depth"],
            "early_sweep_front_depth": early_front,
        }
    return refined, [str(full_path), str(early_path)]


def build_depth_graded_quotient_meta_hypothesis(
    results_path: Path,
) -> dict[str, Any]:
    campaign = _read_campaign(results_path)
    criteria = campaign["pre_registered_criteria"]
    supported = all(value is True for value in criteria.values())
    if supported != bool(
        campaign["claim_status"]["numerical_depth_family_supported"]
    ):
        raise ValueError("campaign claim status disagrees with its criteria")
    refined, refinement_paths = _refined_fronts(results_path)
    evidence = []
    for run in campaign["runs"]:
        front = refined[run["experiment_id"]]
        final = run["checkpoints"][-1]
        evidence.append(
            {
                "evidence_role": "direct_depth_graded_training_arm",
                "experiment_id": run["experiment_id"],
                "preset": run["preset"],
                "seed": int(run["seed"]),
                "training_arm": run["training_arm"],
                "quotient": run["quotient"],
                "model_parameters": int(run["model_parameters"]),
                "coarse_front_depth": run["front_trajectory"][-1]["front_depth"],
                **front,
                "final_shallow_depth_accuracy": float(
                    run["final_shallow_depth_accuracy"]
                ),
                "final_full_depth_accuracy": float(run["final_full_depth_accuracy"]),
                "final_integer_depth_one_h1": float(
                    final["depth_cells"][4]["posterior_normalized_h1_lifetime"]
                ),
                "final_state_sha256": run["final_state_sha256"],
                "standard_state_matches_source_campaign": run[
                    "standard_state_matches_source_campaign"
                ],
                "all_trained_depth_charge_identities_hold": (
                    all(
                        item["depth_defect_charge"]["charge_identity_holds"]
                        for item in run["checkpoints"]
                        if item["step"] > 0
                    )
                    if run["quotient"] == "phase_circle"
                    else None
                ),
            }
        )
    keyed = {
        (item["training_arm"], item["quotient"]): item for item in evidence
    }
    standard_cos = keyed[("standard_final", "cosine_interval")]
    continuous_cos = keyed[("continuous_gate", "cosine_interval")]
    discrete_phase = keyed[("discrete_multi_exit", "phase_circle")]
    continuous_phase = keyed[("continuous_gate", "phase_circle")]
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM task geometry forms moving fronts in continuous gated depth",
            "category": "training",
            "description": (
                "A matched d8 comparison of final-depth, integer multi-exit, and "
                "continuous residual-gate supervision with one shared task head."
            ),
            "question": (
                "Can one transformer be trained as a compatible real-depth family, "
                "and how do phase carriers and cosine quotients develop across depth?"
            ),
            "prediction": campaign["pre_registered_prediction"],
            "created_at": campaign["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "numerical_depth_family_supported_single_seed_not_confirmed"
                if supported
                else "numerical_depth_family_criteria_failed"
            ),
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "one seed-7 d8 phase/cosine matched depth-family campaign",
            "success_criteria": criteria,
            "subclaims": {
                "exact_integer_depth_compatibility": "unit_tested",
                "continuous_partial_block_family": "numerically_supported",
                "joint_supervision_moves_cosine_quotient_shallower": "supported",
                "joint_supervision_improves_shallow_task_accuracy": "supported",
                "continuous_gate_uniquely_outperforms_discrete_multi_exit": "not_supported",
                "trained_phase_depth_charge_identity": "numerically_supported",
                "conditional_branch_mutual_information": "not_measured",
                "reeb_cosheaf": "not_constructed",
                "neural_ode_refinement_limit": "not_tested",
                "multi_seed_repeatability": "not_tested",
            },
            "tags": [
                "tinyllm",
                "continuous-depth",
                "mapping-telescope",
                "residual-gates",
                "multi-exit",
                "task-front",
                "defect-charge",
                "single-seed",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": campaign["claim_status"]["confirmation_limit"],
            "confidence": 0.0,
            "confidence_assessment": "numerical_single_seed_depth_family",
            "effect_size": float(
                standard_cos["refined_front_depth"]
                - continuous_cos["refined_front_depth"]
            ),
            "effect_size_kind": "standard_minus_continuous_cosine_front_depth",
            "independent_seed_count": 1,
            "model_class_count": 1,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                f"{item['training_arm']}:{item['quotient']}": {
                    "refined_front_depth": item["refined_front_depth"],
                    "front_resolution": item["front_resolution"],
                    "shallow_accuracy": item["final_shallow_depth_accuracy"],
                    "full_accuracy": item["final_full_depth_accuracy"],
                }
                for item in evidence
            },
            "key_insights": [
                "Ordinary phase training already forms a degree-one carrier within 0.045 of block 1.",
                "Joint supervision moves the cosine quotient proxy from depth 1.85 to at most 0.005.",
                "Discrete multi-exit gives the earliest phase front, while continuous gating preserves the best full-depth phase accuracy.",
                "Every resolved trained phase-depth cylinder has indexed charge equal to endpoint degree change.",
            ],
            "unexpected_findings": [
                "The standard phase topological front was nearly immediate despite much weaker depth-1 classification accuracy.",
                "The ordinary cosine threshold front was nonmonotone across training checkpoints.",
                "Continuous gating did not dominate discrete multi-exit supervision on phase onset.",
            ],
            "suggested_hypotheses": [
                "The shallow-accuracy and front-location tradeoff repeats across at least five seeds.",
                "A held-out conditional branch probe confirms the cosine front localized by paired geometry.",
                "Weight interpolation between checkpoints continues the depth defects into curves in phase-depth-training space.",
                "A weight-tied neural-ODE control preserves branch information until a noninvertible projection.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [
            str(results_path),
            str(results_path.parent / "initial_resolution.json"),
            *refinement_paths,
        ],
    }


def build_depth_graded_quotient_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _read_campaign(results_path)
    refined, _ = _refined_fronts(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    results = []
    for run in campaign["runs"]:
        metrics = {
            "accuracy": float(run["final_full_depth_accuracy"]),
            "full_depth_accuracy": float(run["final_full_depth_accuracy"]),
            "shallow_depth_accuracy": float(run["final_shallow_depth_accuracy"]),
            "refined_front_depth": float(
                refined[run["experiment_id"]]["refined_front_depth"]
            ),
            "front_resolution": float(
                refined[run["experiment_id"]]["front_resolution"]
            ),
        }
        if run["quotient"] == "phase_circle":
            metrics["trained_depth_charge_identity_fraction"] = float(
                sum(
                    item["depth_defect_charge"]["charge_identity_holds"]
                    for item in run["checkpoints"]
                    if item["step"] > 0
                )
                / sum(item["step"] > 0 for item in run["checkpoints"])
            )
        results.append(
            ExperimentResult(
                experiment_id=run["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=metrics["shallow_depth_accuracy"],
                model_architecture=[
                    int(run["model_config"]["n_layer"]),
                    int(run["model_parameters"]),
                ],
                model_parameters=int(run["model_parameters"]),
                training_time=float(run["training_seconds"]),
                model_checkpoint=str(results_path.parent / run["model_checkpoint"]),
                observations=[
                    f"Training arm: {run['training_arm']}.",
                    f"Task quotient: {run['quotient']}.",
                    f"Refined task front: {metrics['refined_front_depth']}.",
                ],
                anomalies=[
                    "Front thresholds and branch geometry are operational proxies, not Reeb or induced-map proofs."
                ],
                timestamp=completed_at,
            )
        )
    return results


def store_depth_graded_quotient_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_depth_graded_quotient_meta_hypothesis(results_path)
    experiment_results = build_depth_graded_quotient_experiment_results(
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
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "META_HYPOTHESIS_ID",
    "build_depth_graded_quotient_experiment_results",
    "build_depth_graded_quotient_meta_hypothesis",
    "store_depth_graded_quotient_meta_hypothesis",
]
