"""Build and store TinyLLM scalar action-defect decomposition evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-scalar-action-defect-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-c2-scalar-action-defect-decomposition-v1"
EVIDENCE_ROLE = (
    "preregistered_stored_array_post_outcome_underpowered_mechanistic_evidence"
)
IMPLEMENTATION_SHA256 = (
    "dae6ccef726a8b0909067399b2a6228896c1256a68d763a17a318f08772616b0"
)
CAMPAIGN_SHA256 = (
    "bcd826ebe195c55a9b42aaaae68f7bcfc6923be3114c2cff7119b25675007f82"
)
SOURCE_CAMPAIGN_SHA256 = (
    "1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f"
)
SEEDS = (7, 29, 53)
CLASSIFICATION = "residual_coordinate_defect"
EXPECTED_GATES = {
    "control_specificity_all_cells_pass": True,
    "covector_component_all_cells_pass": False,
    "joint_first_order_all_cells_pass": True,
    "residual_component_all_cells_pass": True,
    "residual_coordinate_defect_gate": True,
    "source_provenance_and_validity_contract": True,
    "stored_array_numerical_contract": True,
}
EXPECTED_GATE_COUNTS = {
    name: 3 if value else 0 for name, value in EXPECTED_GATES.items()
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    expected_classes = {str(seed): CLASSIFICATION for seed in SEEDS}
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or provenance.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("source_implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_encoders", -1)) != 0
        or int(summary.get("fitted_observers", -1)) != 0
        or int(summary.get("source_action_cells", -1)) != 24
        or int(summary.get("derived_component_fields", -1)) != 72
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != EXPECTED_GATE_COUNTS
        or int(aggregates.get("primary_checkpoint_pass_count", -1)) != 3
        or int(aggregates.get("joint_first_order_checkpoint_pass_count", -1)) != 3
        or int(aggregates.get("required_checkpoint_pass_count", -1)) != 3
        or aggregates.get("conclusion")
        != "supported_residual_coordinate_defect_three_of_three"
    ):
        raise ValueError(f"invalid scalar action-defect campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid scalar action-defect seed index {path}")
    output: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        cells = detail.get("cells", [])
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or not arrays_path.is_file()
            or _sha256(arrays_path) != entry.get("arrays_sha256")
            or detail.get("artifacts", {}).get("arrays_sha256")
            != entry.get("arrays_sha256")
            or detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or int(detail.get("seed", -1)) != seed
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != CLASSIFICATION
            or detail.get("gates") != EXPECTED_GATES
            or float(detail.get("primary_metric", 0.0)) != 1.0
            or len(cells) != 8
            or any(cell.get("classification") != CLASSIFICATION for cell in cells)
            or any(cell.get("gates", {}).get("joint_prediction") is not True for cell in cells)
            or any(cell.get("gates", {}).get("residual_prediction") is not True for cell in cells)
            or any(cell.get("gates", {}).get("covector_prediction") is not False for cell in cells)
            or any(cell.get("gates", {}).get("control_specificity") is not True for cell in cells)
            or any(cell.get("gates", {}).get("action_defect_nondegenerate") is not True for cell in cells)
            or any(cell.get("gates", {}).get("algebraic_identity") is not True for cell in cells)
            or any(float(cell["metrics"]["joint"]["zero_referenced_r2"]) < 0.998 for cell in cells)
            or any(float(cell["metrics"]["residual"]["zero_referenced_r2"]) < 0.998 for cell in cells)
            or any(float(cell["metrics"]["covector"]["zero_referenced_r2"]) > 0.01 for cell in cells)
            or any(float(cell["metrics"]["joint"]["sign_agreement"]) != 1.0 for cell in cells)
            or any(float(cell["metrics"]["residual"]["sign_agreement"]) != 1.0 for cell in cells)
        ):
            raise ValueError(f"invalid scalar action-defect result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate scalar action-defect fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def _summary(detail: Mapping[str, Any]) -> dict[str, float]:
    cells = detail["cells"]
    return {
        "minimum_joint_zero_referenced_r2": min(
            cell["metrics"]["joint"]["zero_referenced_r2"] for cell in cells
        ),
        "minimum_residual_zero_referenced_r2": min(
            cell["metrics"]["residual"]["zero_referenced_r2"] for cell in cells
        ),
        "maximum_covector_zero_referenced_r2": max(
            cell["metrics"]["covector"]["zero_referenced_r2"] for cell in cells
        ),
        "maximum_joint_relative_l2": max(
            cell["metrics"]["joint"]["relative_l2"] for cell in cells
        ),
        "maximum_residual_relative_l2": max(
            cell["metrics"]["residual"]["relative_l2"] for cell in cells
        ),
        "minimum_target_rms": min(
            cell["metrics"]["joint"]["target_rms"] for cell in cells
        ),
        "maximum_algebraic_relative_error": max(
            cell["algebraic_relative_error"] for cell in cells
        ),
        "best_negative_control_r2": max(
            max(
                control["zero_referenced_r2"]
                for control in cell["controls"].values()
            )
            for cell in cells
        ),
    }


def build_scalar_action_defect_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-scalar-action-defect-decomposition.md"
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
            "component_summary": _summary(detail),
            "cells": [
                {
                    "regime": cell["regime"],
                    "action": cell["action"],
                    "classification": cell["classification"],
                    "metrics": cell["metrics"],
                    "controls": cell["controls"],
                    "gates": cell["gates"],
                    "component_cosine": cell["component_cosine"],
                }
                for cell in detail["cells"]
            ],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM scalar action-defect decomposition",
            "category": "mechanistic_interpretability",
            "description": (
                "A symmetric no-fit decomposition separates nuisance-dependent "
                "signed correction into rank-three residual-coordinate and local "
                "task-covector changes."
            ),
            "question": (
                "Does action dependence live in the writer-coordinate residual, "
                "the local task covector, or their coupling?"
            ),
            "prediction": (
                "Residual-coordinate change alone predicts every action defect, "
                "while covector change alone does not, in all three checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_residual_coordinate_defect_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": (
                "underpowered_preregistered_stored_array_mechanistic_diagnostic"
            ),
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 checkpoints; four exact observed-similarity "
                "actions under composition and extrapolation"
            ),
            "subclaims": {
                "source_and_numerical_contracts": "supported_three_of_three",
                "joint_first_order_action_defect": (
                    "supported_three_of_three_twenty_four_of_twenty_four_cells"
                ),
                "residual_coordinate_sufficiency": (
                    "supported_three_of_three_twenty_four_of_twenty_four_cells"
                ),
                "covector_change_sufficiency": "not_supported_zero_of_twenty_four_cells",
                "semantic_phase_and_sign_controls": "specific_three_of_three",
                "prospective_output_type": "action_conditioned_scalar_amplitude",
                "prospective_learnability": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c2",
                "scalar-residual",
                "action-defect",
                "task-covector",
                "stored-array",
                "output-type-selection",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Whether an observable analytic residual closes the frozen continuation.",
                "Whether a prospective calibration-conditioned scalar head is learnable.",
                "Natural-language tasks or checkpoint-population prevalence.",
            ],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Residual-coordinate change alone predicts all 24 action defects "
                "with R2 above 0.9984 and perfect sign agreement; covector change "
                "alone has approximately zero R2 in every cell."
            ),
            "confidence": 0.995,
            "confidence_assessment": (
                "exact_preregistered_stored_array_component_identification"
            ),
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The nuisance-dependent scalar is residual-coordinate driven in all checkpoints.",
                "The phase-conditioned task covector need not carry the action type.",
                "An invariant amplitude remains structurally wrong despite the scalar interface being sufficient.",
                "The correct prospective type is an action/calibration-conditioned scalar amplitude on a fixed portable covector.",
                "No training or fitting was needed to select this interface type.",
            ],
            "suggested_hypotheses": [
                "An analytic residual between a calibrated invariant semantic estimate and the observable frozen writer prediction closes the existing continuation.",
                "A prospectively trained calibration-conditioned scalar head outperforms invariant and parameter-matched untyped controls under extrapolation.",
                "If the analytic observable residual fails, the calibrated invariant front end should replace rather than repair the frozen writer.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
        },
    }


def build_scalar_action_defect_decomposition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summary = _summary(detail)
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    **{name: float(value) for name, value in detail["gates"].items()},
                    **{name: float(value) for name, value in summary.items()},
                },
                primary_metric=float(
                    detail["gates"]["residual_coordinate_defect_gate"]
                ),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Stored exact-group scalar, residual, and task-covector arrays were decomposed without fitting."
                ],
                anomalies=[
                    "Covector-change-only prediction has approximately zero R2 in all eight cells."
                ],
                timestamp=completed,
            )
        )
    return output


def store_scalar_action_defect_decomposition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_scalar_action_defect_decomposition_meta_hypothesis(results_path)
    experiments = build_scalar_action_defect_decomposition_experiment_results(
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
            raise RuntimeError("scalar action-defect ChromaDB read-back failed")
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
    "build_scalar_action_defect_decomposition_meta_hypothesis",
    "build_scalar_action_defect_decomposition_experiment_results",
    "store_scalar_action_defect_decomposition_meta_hypothesis",
]
