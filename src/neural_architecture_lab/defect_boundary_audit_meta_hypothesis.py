"""Build the TinyLLM defect-subspace boundary-audit evidence record."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-defect-boundary-audit.v1.1"
HYPOTHESIS_ID = "tinyllm-c2-defect-boundary-correction-v1"
PRIMARY_SEEDS = (29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role")
        != "post_outcome_corrective_replication_evidence"
        or int(value.get("summary", {}).get("requested", -1)) != 2
        or int(value.get("summary", {}).get("completed", -1)) != 2
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_predictive_observers", -1))
        != 0
        or int(value.get("aggregates", {}).get("primary_failure_cell_count", -1)) != 3
        or int(
            value.get("aggregates", {}).get(
                "predecessor_endpoint_replication_count", -1
            )
        )
        != 2
        or value.get("aggregates", {}).get("confirmed") is not False
    ):
        raise ValueError(f"invalid defect-boundary audit campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    output = []
    fingerprints = set()
    for seed in PRIMARY_SEEDS:
        detail_path = path.parent / "runs" / f"seed_{seed}" / "result.json"
        detail = json.loads(detail_path.read_text())
        checkpoint = Path(detail.get("provenance", {}).get("checkpoint", ""))
        predecessor = Path(
            detail.get("provenance", {}).get("rank_predecessor", "")
        )
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role")
            != "post_outcome_corrective_replication_evidence"
            or detail.get("implementation_sha256") != campaign["implementation_sha256"]
            or int(detail.get("seed", -1)) != seed
            or not detail.get("scientific_fingerprint")
            or not checkpoint.is_file()
            or _sha256(checkpoint)
            != detail.get("provenance", {}).get("checkpoint_sha256")
            or not predecessor.is_file()
            or _sha256(predecessor)
            != detail.get("provenance", {}).get("rank_predecessor_sha256")
            or not detail.get("predecessor_replication", {}).get("passed")
        ):
            raise ValueError(f"invalid defect-boundary audit cell {detail_path}")
        if detail["scientific_fingerprint"] in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(detail["scientific_fingerprint"])
        output.append(detail)
    return output


def build_defect_boundary_audit_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-defect-boundary-audit.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": int(detail["seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "evidence_role": detail["evidence_role"],
            "predecessor_replication": detail["predecessor_replication"],
            "gates": detail["gates"],
            "source_basis": detail["source_basis"],
            "primary_cells": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "mechanism_classification": cell["mechanism_classification"],
                    "boundary": cell["interventions"]["path_0"]["boundary_to_exact"],
                    "crossing_summary": cell["crossing_summary"],
                }
                for cell in detail["cells"]
                if cell["primary_failure_cell"]
            ],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Extra causal defect rank is decoder-boundary correction",
            "category": "representation",
            "description": "Omitted-subspace interpolation and individual singular-direction patches on three high-Fisher rank near misses.",
            "question": "Are defect directions beyond smooth low rank only correcting frozen exact-bin margins?",
            "prediction": "All three registered failures are boundary-local under fixed disagreement, moment-shift, and alignment criteria.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_mixed_boundary_and_third_semantic_direction",
            "evidence_count": 2,
            "direct_experiment_count": 2,
            "power_profile": "underpowered_post_outcome_corrective_replication",
            "evidence_role": "post_outcome_corrective_replication_evidence",
            "tested_scope": "seeds 29 and 53; three preregistered rank failures; two held-out cohorts; composition and extrapolation controls",
            "subclaims": {
                "endpoint_controls": "supported_three_of_three",
                "boundary_only": "supported_one_of_three",
                "continuous_map_distortion": "supported_two_of_three",
                "next_singular_direction_sufficient": "supported_two_of_two_checkpoints",
                "refined_stable_ranks": "seed29_rank5_seed53_rank3",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm", "causal-patching", "decoder-boundary", "singular-direction",
                "semantic-carrier", "margin-correction", "defect-subspace",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "The boundary classifications were visible before the schema-v1.1 reproduction-gate correction, so this is not fresh confirmatory evidence.",
                "The two-checkpoint conditional audit is underpowered for a population claim."
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "Seed 29 is a one-example boundary correction, but both seed-53 composition failures change 42--48 percent of predictions and shift the continuous map by 0.36--0.41 bins.",
            "confidence": 0.0,
            "confidence_assessment": "post_outcome_corrective_three_cell_rejection_with_exact_predecessor_and_sufficient_rank_controls",
            "independent_seed_count": 2,
            "num_direct_experiments": 2,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Seed 29 rank 4 misses by one adjacent-bin prediction and source direction 5 repairs it.",
                "Seed 53 rank 2 broadly distorts the continuous posterior map rather than only low-margin examples.",
                "Source direction 3 alone restores seed 53 across all four held-out cells; direction 4 does not repair composition.",
                "The refined exact tested stable causal ranks are 2, 5, and 3 across seeds 7, 29, and 53.",
                "The quotient front separates into a 2--3 dimensional semantic core and a checkpoint-dependent decoder-margin tail.",
            ],
            "suggested_hypotheses": [
                "A continuous circular endpoint is sufficient from ranks 2--3 while exact-bin calibration alone needs the seed-29 fifth direction.",
                "A group-typed three-dimensional carrier captures the common semantic defect core across stable checkpoints.",
                "Decoder-side margin calibration can recover exact-bin accuracy without changing the frozen transformer carrier.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [
            str(results_path),
            str(results_path.parent.parent / "20260806_d6_preregistered_v2" / "campaign_results.json"),
            str(results_path.parent.parent / "20260806_d6_preregistered" / "campaign_results.json"),
            str(report_path),
        ],
    }


def build_defect_boundary_audit_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics={
                "primary_boundary_only_fraction": float(
                    detail["gates"]["primary_boundary_only_count"]
                    / max(1, len(detail["gates"]["primary_classifications"]))
                ),
                "endpoint_controls": float(detail["gates"]["primary_endpoint_controls"]),
                "refined_minimum_sufficient_rank": float(detail["gates"]["refined_minimum_sufficient_rank"]),
            },
            primary_metric=float(detail["gates"]["all_primary_failures_boundary_only"]),
            model_architecture=[6, 6, 384],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen checkpoint; omitted-defect interpolation; individual source singular-direction interventions."],
            anomalies=["Seed 29 is boundary-local while seed 53 contains one broad third semantic direction."],
            timestamp=completed,
        )
        for detail in _details(results_path)
    ]


def store_defect_boundary_audit_meta_hypothesis(
    results_path: Path, output_path: Path, *, chromadb_path: Path | None = None
) -> dict[str, Any]:
    record = build_defect_boundary_audit_meta_hypothesis(results_path)
    experiments = build_defect_boundary_audit_experiment_results(record, results_path)
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
            raise RuntimeError("defect-boundary audit ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


__all__ = [
    "build_defect_boundary_audit_meta_hypothesis",
    "build_defect_boundary_audit_experiment_results",
    "store_defect_boundary_audit_meta_hypothesis",
]
