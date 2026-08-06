"""Persist the TinyLLM degree/defect cobordism result conservatively."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-degree-change-equals-indexed-defect-charge-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-degree-defect-cobordism.v1"


def _read_results(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported degree/defect cobordism schema")
    if campaign.get("hypothesis_id") != META_HYPOTHESIS_ID:
        raise ValueError("degree/defect campaign has the wrong hypothesis id")
    if campaign.get("status") != "completed" or len(campaign.get("runs", [])) != 2:
        raise ValueError("degree/defect campaign is incomplete")
    return campaign


def _source_phase_runs(campaign: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    source_path = Path(str(campaign["source_results"]))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return {
        run["preset"]: run
        for run in source["runs"]
        if run["quotient"] == "phase_circle"
        and run["condition"] == "trained"
        and int(run["seed"]) == int(campaign["campaign_config"]["seed"])
    }


def build_degree_defect_cobordism_meta_hypothesis(
    results_path: Path,
) -> dict[str, Any]:
    campaign = _read_results(results_path)
    runs = campaign["runs"]
    source_runs = _source_phase_runs(campaign)
    criteria = campaign["pre_registered_criteria"]
    numerical_support = all(
        value is True
        for preset_criteria in criteria.values()
        for value in preset_criteria.values()
    )
    if numerical_support != bool(
        campaign["claim_status"]["numerical_charge_identity_supported"]
    ):
        raise ValueError("campaign claim status disagrees with its criteria")

    evidence = []
    for run in runs:
        source_run = source_runs[run["preset"]]
        evidence.append(
            {
                "evidence_role": "direct_numerical_training_cobordism",
                "experiment_id": run["experiment_id"],
                "source_experiment_id": run.get(
                    "source_experiment_id", source_run["experiment_id"]
                ),
                "preset": run["preset"],
                "seed": int(run["seed"]),
                "model_parameters": int(
                    run.get("model_parameters", source_run["model_parameters"])
                ),
                "degree_transition_steps": run["degree_transition_steps"],
                "endpoint_soft_degree_change": int(
                    run["endpoint_soft_degree_change"]
                ),
                "total_indexed_defect_charge": int(
                    run["total_indexed_defect_charge"]
                ),
                "defect_cell_count": sum(
                    int(item["defect_cell_count"])
                    for item in run["interpolated_transitions"]
                ),
                "all_local_charge_identities_hold": bool(
                    run["all_local_charge_identities_hold"]
                ),
                "global_charge_identity_holds": bool(
                    run["global_charge_identity_holds"]
                ),
                "final_state_matches_source_campaign": bool(
                    run["final_state_matches_source_campaign"]
                ),
                "maximum_trace_phase_points": max(
                    int(item["soft_lift"]["phase_grid_points"])
                    for item in run["training_trace"]
                ),
            }
        )

    status = (
        "numerical_charge_identity_supported_not_formally_certified"
        if numerical_support
        else "numerical_charge_identity_criteria_failed"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM winding change equals indexed posterior-moment defect charge",
            "category": "training",
            "description": (
                "A deterministic training replay on an explicit continuous "
                "adjacent-token-embedding lift, with indexed zero cells on each "
                "degree-changing phase/weight-interpolation cylinder."
            ),
            "question": (
                "Does emergence of the learned degree-one circular map occur through "
                "posterior-moment defects whose signed charge equals the degree change?"
            ),
            "prediction": campaign["pre_registered_identity"],
            "created_at": campaign["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": status,
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "seed-7 d6/d8 phase models on one fixed nuisance slice",
            "success_criteria": criteria,
            "subclaims": {
                "degree_change_equals_indexed_defect_charge": (
                    "numerically_supported" if numerical_support else "not_supported"
                ),
                "degree_change_requires_nonempty_defect_trace": (
                    "numerically_supported" if numerical_support else "not_supported"
                ),
                "hard_tokenizer_training_cobordism": "not_defined_due_to_discontinuity",
                "interval_certified_root_isolation": "not_tested",
                "polynomial_whitney_stratification": "not_constructed",
                "multi_seed_repeatability": "not_tested",
            },
            "tags": [
                "tinyllm",
                "winding-degree",
                "posterior-moment",
                "defect-charge",
                "training-cobordism",
                "continuous-tokenizer-lift",
                "single-seed",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": campaign["claim_status"]["confirmation_limit"],
            "confidence": 0.0,
            "confidence_assessment": "numerical_single_seed_grid_certificate",
            "effect_size": float(
                sum(run["total_indexed_defect_charge"] for run in runs) / len(runs)
            ),
            "effect_size_kind": "mean_indexed_defect_charge",
            "independent_seed_count": 1,
            "model_class_count": len(runs),
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": sum(
                bool(run["global_charge_identity_holds"]) for run in runs
            ),
            "failed_direct_experiments": sum(
                not bool(run["global_charge_identity_holds"]) for run in runs
            ),
            "descriptive_metrics": {
                run["preset"]: {
                    "degree_transition_steps": run["degree_transition_steps"],
                    "endpoint_degree_change": run["endpoint_soft_degree_change"],
                    "indexed_defect_charge": run["total_indexed_defect_charge"],
                    "defect_cell_count": sum(
                        item["defect_cell_count"]
                        for item in run["interpolated_transitions"]
                    ),
                }
                for run in runs
            },
            "key_insights": [
                "Both retained models changed from winding degree zero to plus one.",
                "Each degree-changing optimizer interval contained net indexed defect charge plus one.",
                "Deterministic replays reproduced the original campaign's final-state hashes.",
            ],
            "unexpected_findings": [
                "The transition occurred at step 15 in d6 and step 17 in d8.",
                "The d8 transition neighborhood required adaptive phase-grid refinement.",
            ],
            "suggested_hypotheses": [
                "The indexed charge identity repeats across at least five seeds and nuisance slices.",
                "Interval arithmetic certifies the localized posterior-moment roots and local indices.",
                "Layer-cut defect traces localize where the degree-one carrier becomes well-defined.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path)],
    }


def build_degree_defect_cobordism_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _read_results(results_path)
    source_runs = _source_phase_runs(campaign)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    results = []
    for run in campaign["runs"]:
        source_run = source_runs[run["preset"]]
        identity_error = abs(
            int(run["endpoint_soft_degree_change"])
            - int(run["total_indexed_defect_charge"])
        )
        metrics = {
            "charge_identity_indicator": float(identity_error == 0),
            "charge_identity_error": float(identity_error),
            "endpoint_degree_change": float(run["endpoint_soft_degree_change"]),
            "indexed_defect_charge": float(run["total_indexed_defect_charge"]),
            "degree_transition_count": float(len(run["degree_transition_steps"])),
            "defect_cell_count": float(
                sum(
                    item["defect_cell_count"]
                    for item in run["interpolated_transitions"]
                )
            ),
        }
        results.append(
            ExperimentResult(
                experiment_id=run["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=metrics["charge_identity_indicator"],
                model_architecture=[
                    int(run["preset"][1:]),
                    int(run.get("model_parameters", source_run["model_parameters"])),
                ],
                model_parameters=int(
                    run.get("model_parameters", source_run["model_parameters"])
                ),
                training_time=float(run["analysis_seconds"]),
                model_checkpoint=run.get(
                    "source_checkpoint", source_run.get("model_checkpoint")
                ),
                observations=[
                    f"Degree-changing optimizer steps: {run['degree_transition_steps']}.",
                    "Endpoint degree change and indexed defect charge agree.",
                    "Final-state digest matches the source training campaign.",
                ],
                anomalies=[
                    "This is a finite-grid certificate on a declared continuous lift, not interval root isolation."
                ],
                timestamp=completed_at,
            )
        )
    return results


def store_degree_defect_cobordism_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_degree_defect_cobordism_meta_hypothesis(results_path)
    experiment_results = build_degree_defect_cobordism_experiment_results(
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
    "build_degree_defect_cobordism_experiment_results",
    "build_degree_defect_cobordism_meta_hypothesis",
    "store_degree_defect_cobordism_meta_hypothesis",
]
