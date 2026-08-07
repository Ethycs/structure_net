"""Build the TinyLLM stable-defect subspace-rank evidence record."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-defect-subspace-rank.v1"
HYPOTHESIS_ID = "tinyllm-c2-defect-subspace-rank-v1"
PRIMARY_SEEDS = (7, 29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != "preregistered_primary_evidence"
        or int(value.get("summary", {}).get("requested", -1)) != 3
        or int(value.get("summary", {}).get("completed", -1)) != 3
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_predictive_observers", -1)) != 0
        or not value.get("implementation_sha256")
    ):
        raise ValueError(f"invalid defect-subspace rank campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    output = []
    fingerprints = set()
    for seed in PRIMARY_SEEDS:
        detail_path = path.parent / "runs" / f"seed_{seed}" / "result.json"
        detail = json.loads(detail_path.read_text())
        checkpoint = Path(detail.get("provenance", {}).get("checkpoint", ""))
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != "preregistered_primary_evidence"
            or detail.get("implementation_sha256") != campaign["implementation_sha256"]
            or int(detail.get("seed", -1)) != seed
            or not detail.get("scientific_fingerprint")
            or not checkpoint.is_file()
            or _sha256(checkpoint)
            != detail.get("provenance", {}).get("checkpoint_sha256")
        ):
            raise ValueError(f"invalid defect-subspace rank cell {detail_path}")
        if detail["scientific_fingerprint"] in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(detail["scientific_fingerprint"])
        output.append(detail)
    return output


def build_defect_subspace_rank_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-defect-subspace-rank.md"),
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
            "gates": detail["gates"],
            "source_basis": detail["source_basis"],
            "heldout_cells": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "rank_one": cell["interventions"]["rank_1"],
                    "rank_one_complement": cell["interventions"]["rank_1_complement"],
                    "minimum_fixed_rank": cell["interventions"][
                        f"rank_{detail['gates']['minimum_fixed_sufficient_rank']}"
                    ],
                }
                for cell in detail["heldout_cells"]
            ],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Scalar stable Reynolds-defect subspace",
            "category": "representation",
            "description": "Source-fitted singular-subspace interventions on stable block-0 degree-two attention Reynolds defects.",
            "question": "Do distributed attention heads write one shift-stable causal defect direction?",
            "prediction": "Rank one is sufficient and its orthogonal complement weak in all three stable checkpoints across two held-out cohorts and both shifts.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_low_dimensional_vector_valued_defect",
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "power_profile": "underpowered_three_checkpoint_mechanistic_cohort",
            "tested_scope": "three cross-cohort-stable d6 C2 block-0 attention fronts; source-fit basis; two held-out cohorts; composition and extrapolation",
            "subclaims": {
                "exact_endpoint_replication": "supported_three_of_three",
                "rank_one_sufficiency": "not_supported_zero_of_three",
                "rank_one_complement_weak": "not_supported_one_of_three",
                "random_rank_one_specificity": "supported_three_of_three",
                "low_dimensional_fixed_subspace": "supported_ranks_two_eight_four",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm", "reynolds-defect", "causal-patching", "subspace-rank",
                "low-dimensional-circuit", "multi-head-attention", "scalar-hypothesis-falsification",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "The three-seed stable-front cohort is underpowered for a new five-seed population claim."
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "Rank one failed every held-out cell in all three checkpoints; fixed ranks 2, 8, and 4 were the first to pass all task and Fisher endpoints.",
            "confidence": 0.0,
            "confidence_assessment": "complete_three_checkpoint_rank_one_rejection_with_heldout_causal_controls",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The leading source defect direction carries only 46--57 percent of source energy and 31--51 percent of held-out Fisher effect.",
                "Four source directions carry at least 99.76 percent of defect energy in every checkpoint.",
                "Fixed checkpoint-local ranks 2, 8, and 4 pass every held-out causal and Fisher endpoint.",
                "Near-perfect Fisher recovery can still fail the frozen exact-bin task gate, inflating causal rank above smooth geometric rank.",
                "Stable quotient synthesis is compact and vector-valued rather than scalar or diffusely high-dimensional.",
            ],
            "suggested_hypotheses": [
                "The seed-29 rank-4 and seed-53 rank-2 misses are confined to decoder-boundary examples rather than missing semantic directions.",
                "A group-typed 4-dimensional neutral carrier is sufficient across stable C2 fronts after checkpoint-local alignment.",
                "Task-boundary normals, rather than activation energy, explain the extra causal rank required by hard exact-bin gates.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [
            str(results_path),
            str(results_path.parent / "resume_audit_20260806.json"),
            str(report_path),
        ],
    }


def build_defect_subspace_rank_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics={
                **{
                    gate: float(value)
                    for gate, value in detail["gates"].items()
                    if isinstance(value, bool)
                },
                "minimum_fixed_sufficient_rank": float(detail["gates"]["minimum_fixed_sufficient_rank"]),
                "rank_one_source_energy": float(detail["source_basis"]["cumulative_energy"][0]),
            },
            primary_metric=float(detail["gates"]["rank_one_sufficiency"]),
            model_architecture=[6, 6, 384],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen checkpoint; source-fit singular basis; two untouched cohorts under both shifts."],
            anomalies=["Hard task gates require higher rank than the smooth Fisher geometry in seeds 29 and 53."],
            timestamp=completed,
        )
        for detail in _details(results_path)
    ]


def store_defect_subspace_rank_meta_hypothesis(
    results_path: Path, output_path: Path, *, chromadb_path: Path | None = None
) -> dict[str, Any]:
    record = build_defect_subspace_rank_meta_hypothesis(results_path)
    experiments = build_defect_subspace_rank_experiment_results(record, results_path)
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
            raise RuntimeError("defect-subspace rank ChromaDB read-back failed")
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
    "build_defect_subspace_rank_meta_hypothesis",
    "build_defect_subspace_rank_experiment_results",
    "store_defect_subspace_rank_meta_hypothesis",
]
