"""Persist the TinyLLM layer-by-task geometry atlas conservatively."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-layer-task-carrier-and-quotient-atlas-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-task-geometry-atlas.v1"


def _read_results(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported task-geometry atlas schema")
    if campaign.get("status") != "completed" or len(campaign.get("runs", [])) != 4:
        raise ValueError("task-geometry atlas campaign is incomplete")
    return campaign


def _stage_index(run: Mapping[str, Any], stage_name: str | None) -> int:
    if stage_name is None:
        return -1
    for stage in run["stagewise_atlas"]:
        if stage["stage"] == stage_name:
            return int(stage["stage_index"])
    raise ValueError(f"missing localized stage {stage_name!r}")


def build_task_geometry_atlas_meta_hypothesis(results_path: Path) -> dict[str, Any]:
    campaign = _read_results(results_path)
    runs = campaign["runs"]
    evidence = []
    for run in runs:
        final = run["stagewise_atlas"][-1]
        probe = final["probe"]["evaluations"]["in_distribution"]
        geometry = final["geometry"]["training_support"]
        fiber = final["fiber_geometry"]["in_distribution"]
        evidence.append(
            {
                "evidence_role": "direct_frozen_sublayer_atlas",
                "experiment_id": run["experiment_id"],
                "preset": run["preset"],
                "quotient": run["quotient"],
                "seed": run["seed"],
                "checkpoint": run["checkpoint"],
                "localized_stage": (
                    run["localization"]["first_phase_task_carrier"]
                    if run["quotient"] == "phase_circle"
                    else run["localization"]["first_cosine_internal_quotient"]
                ),
                "final_conditional_branch_accuracy": probe[
                    "conditional_branch_accuracy"
                ],
                "final_cosine_pearson": probe["cosine_pearson"],
                "final_phase_alignment": probe["phase_alignment"],
                "final_reference_distance_spearman": geometry["paired_geometry"][
                    "distance_spearman"
                ],
                "final_normalized_h1_lifetime": geometry[
                    "normalized_h1_lifetime"
                ],
                "final_persistent_coordinate_alignment": geometry[
                    "persistent_coordinate_phase_alignment"
                ],
                "final_fiber_collapse_ratio": fiber["nuisance_collapse_ratio"],
            }
        )

    d6 = {run["quotient"]: run for run in runs if run["preset"] == "d6"}
    d8 = {run["quotient"]: run for run in runs if run["preset"] == "d8"}
    criteria = {
        preset: aggregate["pre_registered_criteria"]
        for preset, aggregate in campaign["aggregates"].items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM layers localize task carriers and internal quotients",
            "category": "interpretability",
            "description": (
                "A frozen post-attention/post-MLP atlas using paired task-reference "
                "geometry, nonlinear decoders, matched fibers, and independently "
                "discovered persistent coordinates."
            ),
            "question": (
                "Where does the phase carrier first appear, and where do opposite "
                "phase branches merge into the cosine quotient?"
            ),
            "prediction": campaign["pre_registered_interpretation"],
            "created_at": campaign["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "proxy_atlas_localized_single_seed_not_confirmed",
            "evidence_count": len(evidence),
            "direct_experiment_count": len(evidence),
            "tested_scope": "saved seed-7 d6/d8 synthetic phase and cosine checkpoints",
            "success_criteria": criteria,
            "subclaims": {
                "phase_task_carrier_localized": "supported_in_both_model_classes",
                "cosine_internal_quotient_localized": "supported_in_both_model_classes",
                "attention_mlp_event_localization": "supported_descriptively",
                "chain_level_induced_map_rank": "not_tested",
                "homotopy_retract": "not_proved",
                "multi_seed_repeatability": "not_tested",
            },
            "tags": [
                "tinyllm",
                "task-geometry-atlas",
                "post-attention",
                "post-mlp",
                "paired-correspondence",
                "persistent-cohomology",
                "fiber-collapse",
                "single-seed",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Every frozen proxy criterion passed in d6 and d8, but only seed-7 "
                "checkpoints were retained and no validated chain-level induced map or "
                "homotopy retraction was computed."
            ),
            "confidence": 0.0,
            "confidence_assessment": "descriptive_single_seed_proxy_atlas",
            "effect_size": float(
                _stage_index(d6["cosine_interval"], d6["cosine_interval"]["localization"]["first_cosine_internal_quotient"])
                - _stage_index(d6["phase_circle"], d6["phase_circle"]["localization"]["first_phase_task_carrier"])
            ),
            "effect_size_kind": "d6_cosine_quotient_minus_phase_carrier_stage_index",
            "independent_seed_count": 1,
            "model_class_count": 2,
            "num_direct_experiments": len(evidence),
            "successful_direct_experiments": len(evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "d6_phase_carrier_stage_index": _stage_index(
                    d6["phase_circle"],
                    d6["phase_circle"]["localization"]["first_phase_task_carrier"],
                ),
                "d6_cosine_quotient_stage_index": _stage_index(
                    d6["cosine_interval"],
                    d6["cosine_interval"]["localization"]["first_cosine_internal_quotient"],
                ),
                "d8_phase_carrier_stage_index": _stage_index(
                    d8["phase_circle"],
                    d8["phase_circle"]["localization"]["first_phase_task_carrier"],
                ),
                "d8_cosine_quotient_stage_index": _stage_index(
                    d8["cosine_interval"],
                    d8["cosine_interval"]["localization"]["first_cosine_internal_quotient"],
                ),
            },
            "key_insights": [
                "The phase carrier localized at d6 block 2 attention and d8 block 1 attention.",
                "Conditional branch information collapsed after block 1 MLP in both cosine models.",
                "The full interval-geometry quotient localized later: d6 block 5 attention and d8 block 3 attention.",
                "Every localized carrier/quotient survived to the final block and the held-out branch criterion passed.",
            ],
            "unexpected_findings": [
                "Branch removal preceded mature interval metric geometry by several sublayers.",
                "The deeper d8 model formed both the phase carrier and cosine quotient earlier than d6.",
            ],
            "suggested_hypotheses": [
                "The attention-localized carrier and MLP-localized branch collapse repeat across at least five seeds.",
                "A witness-complex chain map validates the phase carrier rank proxy.",
                "A Reeb cosheaf confirms the one-component cosine fibers after the localized MLP event.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path)],
    }


def build_task_geometry_atlas_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    campaign = _read_results(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    results = []
    for run in campaign["runs"]:
        final = run["stagewise_atlas"][-1]
        probe = final["probe"]["evaluations"]["in_distribution"]
        geometry = final["geometry"]["training_support"]
        fiber = final["fiber_geometry"]["in_distribution"]
        localized = (
            run["localization"]["first_phase_task_carrier"]
            if run["quotient"] == "phase_circle"
            else run["localization"]["first_cosine_internal_quotient"]
        )
        metrics = {
            "accuracy": float(probe["conditional_branch_accuracy"]),
            "conditional_branch_accuracy": float(
                probe["conditional_branch_accuracy"]
            ),
            "cosine_pearson": float(probe["cosine_pearson"]),
            "phase_alignment": float(probe["phase_alignment"]),
            "reference_distance_spearman": float(
                geometry["paired_geometry"]["distance_spearman"]
            ),
            "normalized_h1_lifetime": float(
                geometry["normalized_h1_lifetime"]
            ),
            "fiber_collapse_ratio": float(fiber["nuisance_collapse_ratio"]),
            "localized_stage_index": float(_stage_index(run, localized)),
        }
        results.append(
            ExperimentResult(
                experiment_id=run["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=metrics["localized_stage_index"],
                model_architecture=[int(run["preset"][1:]), int(run["model_parameters"])],
                model_parameters=int(run["model_parameters"]),
                training_time=float(run["analysis_seconds"]),
                model_checkpoint=run["checkpoint"],
                observations=[
                    f"Frozen quotient: {run['quotient']}.",
                    f"Localized stage: {localized}.",
                    "Every attention and MLP residual was analyzed separately.",
                ],
                anomalies=[
                    "Carrier rank, retract, and fiber-component values are operational proxies rather than chain-level proofs."
                ],
                timestamp=completed_at,
            )
        )
    return results


def store_task_geometry_atlas_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    record = build_task_geometry_atlas_meta_hypothesis(results_path)
    experiment_results = build_task_geometry_atlas_experiment_results(
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
    "build_task_geometry_atlas_experiment_results",
    "build_task_geometry_atlas_meta_hypothesis",
    "store_task_geometry_atlas_meta_hypothesis",
]
