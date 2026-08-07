"""Persist the TinyLLM degree-two character-fusion radius evidence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-character-fusion-radius.v1"
HYPOTHESIS_ID = "tinyllm-c2-character-fusion-radius-v1"
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    summary = value.get("summary", {})
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != "preregistered_primary_evidence"
        or summary.get("requested") != 5
        or summary.get("completed") != 5
        or summary.get("failed") != 0
        or not value.get("implementation_sha256")
    ):
        raise ValueError(f"invalid C2 fusion-radius campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    details = []
    for seed in SEEDS:
        detail_path = path.parent / "runs" / f"seed_{seed}" / "result.json"
        detail = json.loads(detail_path.read_text())
        checkpoint = Path(detail.get("provenance", {}).get("checkpoint", ""))
        character = Path(detail.get("provenance", {}).get("character_result", ""))
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != "preregistered_primary_evidence"
            or int(detail.get("seed", -1)) != seed
            or detail.get("implementation_sha256") != campaign["implementation_sha256"]
            or not detail.get("scientific_fingerprint")
            or not checkpoint.is_file()
            or not character.is_file()
        ):
            raise ValueError(f"invalid C2 fusion-radius cell {detail_path}")
        details.append(detail)
    return details


def _seed_primary_pass(detail: Mapping[str, Any]) -> bool:
    gates = detail["gates"]
    return bool(
        gates["cohort_prediction"]
        and gates["shift_stable_onset"]
        and gates["monotone_causal_response"]
        and gates["control_specificity"]
        and gates["exchange_invariance"]
    )


def build_c2_fusion_radius_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-c2-fusion-radius.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    aggregates = campaign["aggregates"]
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": int(detail["seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "character_source": detail["provenance"]["character_result"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "onset_radii": {
                regime: detail["regimes"][regime]["onset_radius"]
                for regime in REGIMES
            },
            "target_cuts": {
                regime: detail["regimes"][regime]["target_cut"]
                for regime in REGIMES
            },
            "gates": detail["gates"],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM degree-two character-fusion radius",
            "category": "representation",
            "description": (
                "Frozen-checkpoint causal scale sweep of the exact nontrivial C2 "
                "character at previously measured quotient-synthesis fronts."
            ),
            "question": (
                "Is degree-two invariant synthesis a local, monotone, shift-stable "
                "response whose onset radius is predicted by front depth?"
            ),
            "prediction": (
                "Early block-zero fronts turn on by radius 0.50, later fronts require "
                "at least 0.75, and onset, monotonicity, exchange, and specificity gates pass."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_exact_character_specific_but_radius_not_depth_local_or_shift_stable"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "tested_scope": (
                "five retained d6 degree-two checkpoints; fresh 64-orbit composition "
                "and extrapolation cohorts; frozen synthesis sublayer"
            ),
            "subclaims": {
                "early_front_locality": "not_supported_zero_of_three",
                "later_front_finite_radius": "not_supported_zero_of_two",
                "shift_stable_onset": "not_supported_three_of_five",
                "monotone_through_observed_radius": "supported_four_of_five",
                "character_direction_control_specificity": "supported_five_of_five",
                "sheet_exchange_invariance": "supported_five_of_five_exact_zero_error",
                "full_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "c2",
                "character-theory",
                "reynolds-operator",
                "causal-intervention",
                "frozen-checkpoint",
                "symmetry",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All early fronts required radius 0.75, both later-front predictions "
                "failed, and only three onsets were shift-stable; exact directions "
                "remained monotone and control-specific in the narrower tested sense."
            ),
            "confidence": 0.0,
            "confidence_assessment": (
                "complete_five_seed_preregistered_failure_with_specific_causal_direction"
            ),
            "independent_seed_count": 5,
            "num_direct_experiments": 5,
            "descriptive_metrics": aggregates,
            "key_insights": [
                "The predicted association between quotient-front depth and exact character onset radius did not replicate.",
                "The three early block-zero fronts all required radius 0.75 on both shifts.",
                "Seed 17's frozen front changed causal regime across fresh composition and extrapolation cohorts.",
                "No transplanted or norm-matched random direction reached the 0.70 control-reproduction threshold.",
                "Smooth Fisher and alignment recovery preceded the hard exact-bin causal pass, exposing endpoint sensitivity.",
            ],
            "suggested_hypotheses": [
                "A typed character carrier with learned radial gating is more faithful than a universal local quadratic invariant layer.",
                "Causal quotient fronts should be replicated across multiple fresh within-family cohorts before being treated as fixed checkpoint properties.",
                "A smooth posterior endpoint will produce more stable fusion-radius estimates than a hard exact-bin accuracy-loss gate.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_c2_fusion_radius_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        composition = detail["regimes"]["composition"]["onset_radius"]
        extrapolation = detail["regimes"]["extrapolation"]["onset_radius"]
        gates = detail["gates"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "composition_onset_radius": -1.0 if composition is None else float(composition),
                    "extrapolation_onset_radius": -1.0 if extrapolation is None else float(extrapolation),
                    "cohort_prediction": float(gates["cohort_prediction"]),
                    "shift_stable_onset": float(gates["shift_stable_onset"]),
                    "monotone_causal_response": float(gates["monotone_causal_response"]),
                    "control_specificity": float(gates["control_specificity"]),
                    "exchange_invariance": float(gates["exchange_invariance"]),
                },
                primary_metric=float(_seed_primary_pass(detail)),
                model_architecture=[6, 2],
                model_parameters=29_956_224,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen checkpoint; 64 fresh exact C2 orbits per shift; exact radial intervention."
                ],
                anomalies=[
                    "Seed 17 composition failed at radius one while extrapolation already passed at radius zero."
                ] if int(detail["seed"]) == 17 else [],
                timestamp=completed,
            )
        )
    return output


def store_c2_fusion_radius_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c2_fusion_radius_meta_hypothesis(results_path)
    experiments = build_c2_fusion_radius_experiment_results(record, results_path)
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
        storage["result_hashes"] = [
            logger.log_experiment_result(item) for item in experiments
        ]
        storage["chromadb_path"] = str(chromadb_path)
        hypothesis_readback = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        experiment_readback = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        experiment_hypotheses = {
            metadata.get("hypothesis_id")
            for metadata in experiment_readback.get("metadatas", [])
        }
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or experiment_hypotheses != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C2 fusion-radius ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


__all__ = [
    "build_c2_fusion_radius_meta_hypothesis",
    "build_c2_fusion_radius_experiment_results",
    "store_c2_fusion_radius_meta_hypothesis",
]
