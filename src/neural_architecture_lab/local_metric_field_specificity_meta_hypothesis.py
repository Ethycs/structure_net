"""Build and store the corrected TinyLLM local metric-field evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-local-metric-field-specificity.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-metric-field-specificity-v1"
EVIDENCE_ROLE = "preregistered_post_outcome_corrective_frozen_artifact_diagnostic"
IMPLEMENTATION_SHA256 = (
    "01e15e21fc3bae68553ac81d5dbffc8252a18628ab58b8c6be19fb53515ae5f0"
)
CAMPAIGN_SHA256 = (
    "b2be5e8bcbeacd82baab6193a123c3f0e2002836edde161e0b371c162b548fe5"
)
SOURCE_CAMPAIGN_SHA256 = (
    "2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55"
)
SEEDS = (7, 29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    aggregates = value.get("aggregates", {})
    expected_classes = {
        "7": "nuisance_variant_field",
        "29": "nuisance_invariant_phase_specific_field",
        "53": "nuisance_variant_field",
    }
    expected_gates = {
        "source_valid": 3,
        "paired_geometry_all_cells": 1,
        "nuisance_equivalence_all_cells": 3,
        "semantic_phase_specificity_all_cells": 3,
        "random_plane_specificity_all_cells": 3,
        "corrected_checkpoint_gate": 1,
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("provenance", {}).get("source_campaign_sha256")
        != SOURCE_CAMPAIGN_SHA256
        or int(value.get("summary", {}).get("requested", -1)) != 3
        or int(value.get("summary", {}).get("completed", -1)) != 3
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_writers", -1)) != 0
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != expected_gates
        or int(aggregates.get("corrected_checkpoint_pass_count", -1)) != 1
        or aggregates.get("conclusion")
        != "checkpoint_stratified_metric_field_specificity"
    ):
        raise ValueError(f"invalid metric-field specificity campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid metric-field specificity seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        expected_class = (
            "nuisance_invariant_phase_specific_field"
            if seed == 29
            else "nuisance_variant_field"
        )
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or int(detail.get("seed", -1)) != seed
            or detail.get("classification") != expected_class
            or gates.get("source_valid") is not True
            or gates.get("nuisance_equivalence_all_cells") is not True
            or gates.get("semantic_phase_specificity_all_cells") is not True
            or gates.get("random_plane_specificity_all_cells") is not True
            or gates.get("paired_geometry_all_cells") is not (seed == 29)
            or gates.get("corrected_checkpoint_gate") is not (seed == 29)
            or len(detail.get("cells", [])) != 8
        ):
            raise ValueError(f"invalid metric-field specificity result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate metric-field fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def _summary(detail: Mapping[str, Any]) -> dict[str, float]:
    cells = detail["cells"]
    return {
        "minimum_paired_median_cosine": min(
            cell["geometry"]["paired_median_kernel_cosine"] for cell in cells
        ),
        "minimum_paired_p10_cosine": min(
            cell["geometry"]["paired_p10_kernel_cosine"] for cell in cells
        ),
        "maximum_paired_p95_distance": max(
            cell["geometry"]["paired_p95_projector_distance"] for cell in cells
        ),
        "minimum_nuisance_equivalence_cosine": min(
            cell["geometry"]["nuisance_equivalence_median_kernel_cosine"]
            for cell in cells
        ),
        "minimum_semantic_phase_margin": min(
            cell["geometry"]["paired_over_semantic_phase_margin"] for cell in cells
        ),
        "minimum_random_plane_margin": min(
            cell["geometry"]["paired_over_random_margin"] for cell in cells
        ),
    }


def build_local_metric_field_specificity_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-local-metric-field-specificity.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": detail["seed"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "classification": detail["classification"],
            "gates": detail["gates"],
            "geometry_summary": _summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM nuisance-invariant local task-metric field",
            "category": "mechanistic_interpretability",
            "description": (
                "A corrected frozen-artifact audit tests whether rank-two local "
                "task projectors are nuisance invariant and semantic-phase specific."
            ),
            "question": (
                "Does one identity action transport the local task plane across "
                "the declared acquisition nuisance group in all three checkpoints?"
            ),
            "prediction": (
                "All eight cells pass paired geometry, nuisance equivalence, "
                "semantic-phase specificity, and random-plane specificity in 3/3 checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "rejected_checkpoint_stratified_one_of_three",
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_corrective_frozen_artifact_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 checkpoints; stored fresh composition and "
                "extrapolation Jacobian projectors under amplitude, orientation, offset, and composition"
            ),
            "subclaims": {
                "original_negative_control_validity": "rejected_structurally",
                "same_phase_nuisance_equivalence": "supported_three_of_three",
                "semantic_phase_specificity": "supported_three_of_three",
                "random_plane_specificity": "supported_three_of_three",
                "paired_nuisance_geometry": "supported_one_of_three",
                "universal_identity_transport_law": "rejected_one_of_three",
                "fresh_local_tangent_sufficiency": "supported_three_of_three_in_source_campaign",
            },
            "tags": [
                "tinyllm",
                "c2",
                "local-task-metric",
                "nuisance-equivariance",
                "projector-transport",
                "corrective-control",
                "frozen-artifact",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A corrected phase-shifted causal continuation control.",
                "A singular-value-weighted pullback metric transport law.",
                "A trained equivariant sidecar.",
                "Link cobordism or any codimension-two defect locus.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Corrected geometry passes every cell only in seed 29; seeds 7 and 53 "
                "fail paired tail geometry despite valid equivalence and specificity controls."
            ),
            "confidence": 0.99,
            "confidence_assessment": "exact_locked_frozen_artifact_corrective",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The original nuisance-replicate negative control was incompatible with nuisance invariance.",
                "Same-phase nuisance equivalence and both negative specificities pass 3/3.",
                "Seed 29 passes all paired cells; seeds 7 and 53 fail all eight tail gates.",
                "Paired medians remain high, so distribution tails—not averages—reject portability.",
                "The original causal endpoint is nonspecific because nuisance-shuffled and random rank-two controls also pass.",
            ],
            "suggested_hypotheses": [
                "A normalized pullback metric or dominant task covector transports more stably than an unweighted rank-two projector.",
                "Weak second-singular directions account for the paired tail failures in seeds 7 and 53.",
                "No universal sidecar should be trained unless a corrected task-weighted transport gate passes all checkpoints.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {
            "direct_tests": direct_tests,
            "invalidated_antecedent": {
                "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
                "reason": (
                    "Its negative permutation remained within the hypothesized nuisance-equivalence class."
                ),
                "quality_evidence_used": False,
            },
        },
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        },
    }


def build_local_metric_field_specificity_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summary = _summary(detail)
        gates = detail["gates"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    **summary,
                    "source_valid": float(gates["source_valid"]),
                    "paired_geometry_all_cells": float(
                        gates["paired_geometry_all_cells"]
                    ),
                    "nuisance_equivalence_all_cells": float(
                        gates["nuisance_equivalence_all_cells"]
                    ),
                    "semantic_phase_specificity_all_cells": float(
                        gates["semantic_phase_specificity_all_cells"]
                    ),
                    "random_plane_specificity_all_cells": float(
                        gates["random_plane_specificity_all_cells"]
                    ),
                    "corrected_checkpoint_gate": float(
                        gates["corrected_checkpoint_gate"]
                    ),
                },
                primary_metric=float(gates["corrected_checkpoint_gate"]),
                model_architecture=[3, 2],
                model_parameters=0,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint_sha256"],
                observations=[
                    "Corrected frozen-projector geometry across eight acquisition-action cells."
                ],
                anomalies=(
                    ["Paired medians are high but tail gates reject every cell."]
                    if not gates["paired_geometry_all_cells"]
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_local_metric_field_specificity_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_local_metric_field_specificity_meta_hypothesis(results_path)
    experiments = build_local_metric_field_specificity_experiment_results(
        record, results_path
    )
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
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in experiment_readback.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("metric-field specificity ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_local_metric_field_specificity_meta_hypothesis",
    "build_local_metric_field_specificity_experiment_results",
    "store_local_metric_field_specificity_meta_hypothesis",
]

