"""Build and store TinyLLM scalar groupoid-defect evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-scalar-groupoid-defect.v1"
HYPOTHESIS_ID = "tinyllm-c2-scalar-groupoid-defect-v1"
EVIDENCE_ROLE = (
    "preregistered_existing_artifact_post_outcome_underpowered_mechanistic_evidence"
)
IMPLEMENTATION_SHA256 = (
    "81faf6321c87b452f22b8f155a437bbb0d4511f8826c3950f75768e0aad630e4"
)
CAMPAIGN_SHA256 = (
    "aeda0f8d90057a5983c75e81398000bfec4a5c133c5e2a476dc91cda97fc89d4"
)
SOURCE_CAMPAIGN_SHA256 = (
    "1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f"
)
SEEDS = (7, 29, 53)
CLASSIFICATION = "two_term_groupoid_defect"
EXPECTED_GATES = {
    "basis_gauge_and_target_replay_contract": True,
    "direct_term_negligible_all_cells": False,
    "exact_groupoid_identity_all_cells": True,
    "finite_angle_contract": True,
    "input_identity_replay_contract": True,
    "provenance_and_inherited_contract": True,
    "signed_scalar_replay_contract": True,
    "writer_only_prediction_all_cells": False,
    "writer_only_specificity_all_cells": False,
    "writer_symmetry_defect_dominance_gate": False,
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
        or provenance.get("transformation_campaign_sha256")
        != SOURCE_CAMPAIGN_SHA256
        or provenance.get("transformation_implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_encoders", -1)) != 0
        or int(summary.get("fitted_observers", -1)) != 0
        or int(summary.get("groupoid_cells", -1)) != 24
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != EXPECTED_GATE_COUNTS
        or int(aggregates.get("primary_checkpoint_pass_count", -1)) != 0
        or int(aggregates.get("required_checkpoint_pass_count", -1)) != 3
        or aggregates.get("conclusion") != CLASSIFICATION
    ):
        raise ValueError(f"invalid scalar groupoid-defect campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid scalar groupoid-defect seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        result_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        cells = detail.get("cells", [])
        if (
            _sha256(result_path) != entry.get("result_sha256")
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
            or float(detail.get("primary_metric", 1.0)) != 0.0
            or detail.get("maximum_signed_scalar_replay_error_bins", 1.0) > 1e-5
            or detail.get("basis_gauge_replay", {}).get(
                "maximum_all_cell_target_replay_error", 1.0
            )
            > 1e-5
            or len(cells) != 8
            or any(
                cell.get("gates", {}).get("action_defect_nondegenerate") is not True
                or cell.get("gates", {}).get("exact_groupoid_identity") is not True
                or cell.get("gates", {}).get("direct_term_negligible") is not False
                or cell.get("gates", {}).get("writer_only_prediction") is not False
                or cell.get("gates", {}).get("primary_cell_pass") is not False
                for cell in cells
            )
            or any(float(cell["direct_to_total_rms_ratio"]) <= 0.80 for cell in cells)
            or any(float(cell["groupoid_max_abs_error_bins"]) > 1e-6 for cell in cells)
            or any(float(cell["writer_only"]["zero_referenced_r2"]) >= 0.90 for cell in cells)
        ):
            raise ValueError(f"invalid scalar groupoid-defect result {result_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate scalar groupoid fingerprint {result_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def _summary(detail: Mapping[str, Any]) -> dict[str, float]:
    cells = detail["cells"]
    return {
        "minimum_direct_to_total_rms_ratio": min(
            cell["direct_to_total_rms_ratio"] for cell in cells
        ),
        "maximum_direct_to_total_rms_ratio": max(
            cell["direct_to_total_rms_ratio"] for cell in cells
        ),
        "maximum_writer_only_zero_referenced_r2": max(
            cell["writer_only"]["zero_referenced_r2"] for cell in cells
        ),
        "minimum_writer_only_sign_agreement": min(
            cell["writer_only"]["sign_agreement"] for cell in cells
        ),
        "maximum_groupoid_error_bins": max(
            cell["groupoid_max_abs_error_bins"] for cell in cells
        ),
        "minimum_action_defect_rms_bins": min(
            cell["delta_scalar_rms_bins"] for cell in cells
        ),
    }


def build_scalar_groupoid_defect_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-scalar-groupoid-defect.md"
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
                    "action": cell["arm"],
                    "delta_scalar_rms_bins": cell["delta_scalar_rms_bins"],
                    "delta_direct_rms_bins": cell["delta_direct_rms_bins"],
                    "delta_writer_rms_bins": cell["delta_writer_rms_bins"],
                    "direct_to_total_rms_ratio": cell[
                        "direct_to_total_rms_ratio"
                    ],
                    "groupoid_max_abs_error_bins": cell[
                        "groupoid_max_abs_error_bins"
                    ],
                    "writer_only": cell["writer_only"],
                    "gates": cell["gates"],
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
            "name": "TinyLLM scalar groupoid-defect decomposition",
            "category": "mechanistic_interpretability",
            "description": (
                "An exact no-fit task-angle decomposition separates the correction's "
                "observed-action defect into direct-rank-three and frozen-writer terms."
            ),
            "question": (
                "Is the action-dependent correction almost entirely the negative "
                "symmetry defect of the frozen order-four writer?"
            ),
            "prediction": (
                "The direct-state term is negligible and the negative writer defect "
                "predicts all action cells in all checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_two_term_groupoid_defect_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": (
                "underpowered_preregistered_existing_artifact_mechanistic_diagnostic"
            ),
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 checkpoints; four exact observed-similarity "
                "actions under composition and extrapolation"
            ),
            "subclaims": {
                "exact_groupoid_identity": "supported_three_of_three_twenty_four_cells",
                "direct_state_term_negligible": "rejected_zero_of_twenty_four_cells",
                "writer_only_defect_prediction": "rejected_zero_of_twenty_four_cells",
                "writer_symmetry_defect_dominance": "rejected_zero_of_three",
                "two_term_groupoid_defect": "supported_three_of_three",
                "frozen_sidecar_observability": "closed_in_tested_scope",
                "calibrated_frontend": "preferred_constructive_path",
            },
            "tags": [
                "tinyllm",
                "c2",
                "groupoid-defect",
                "task-angle",
                "writer-defect",
                "output-type-falsification",
                "no-fit",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Whether an analytic observable semantic residual closes the frozen continuation.",
                "Whether a prospective action-conditioned scalar interface is learnable.",
                "Natural-language tasks or checkpoint-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The exact identity holds, but the direct-state term is material in "
                "all 24 cells and the writer-only approximation passes no cell."
            ),
            "confidence": 0.995,
            "confidence_assessment": "exact_preregistered_two_term_decomposition",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The direct-state action term is 0.81--1.00 times total defect RMS.",
                "The frozen writer's action defect alone has at most 0.348 R2.",
                "The algebraic two-term identity closes to machine precision.",
                "Writer-only sidecar training is ruled out before optimization.",
                "The frozen sidecar branch is closed unless a separately justified observable direct-state estimator is introduced.",
                "The already validated calibrated invariant front end is the preferred constructive path.",
            ],
            "suggested_hypotheses": [
                "A prospective calibration-degradation study identifies the minimum reference quality needed by the validated invariant front end.",
                "A learned pilot estimator can approach the analytic calibrated front end without re-opening the frozen-writer sidecar branch.",
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


def build_scalar_groupoid_defect_experiment_results(
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
                    detail["gates"]["writer_symmetry_defect_dominance_gate"]
                ),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Exact direct-state and writer task-angle defects were recomputed without fitting."
                ],
                anomalies=[
                    "The direct-state term is material in every cell despite the semantic target being invariant."
                ],
                timestamp=completed,
            )
        )
    return output


def store_scalar_groupoid_defect_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_scalar_groupoid_defect_meta_hypothesis(results_path)
    experiments = build_scalar_groupoid_defect_experiment_results(record, results_path)
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        storage_root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(storage_root / "experiment_queue"),
                sent_dir=str(storage_root / "experiment_sent"),
                rejected_dir=str(storage_root / "experiment_rejected"),
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
            raise RuntimeError("scalar groupoid-defect ChromaDB read-back failed")
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
    "build_scalar_groupoid_defect_meta_hypothesis",
    "build_scalar_groupoid_defect_experiment_results",
    "store_scalar_groupoid_defect_meta_hypothesis",
]
