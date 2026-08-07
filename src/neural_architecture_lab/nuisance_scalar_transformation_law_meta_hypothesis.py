"""Build and store TinyLLM nuisance-scalar transformation-law evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-nuisance-scalar-transformation-law.v1"
HYPOTHESIS_ID = "tinyllm-c2-nuisance-scalar-transformation-law-v1"
EVIDENCE_ROLE = (
    "preregistered_existing_artifact_post_outcome_underpowered_mechanistic_evidence"
)
IMPLEMENTATION_SHA256 = (
    "e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f"
)
CAMPAIGN_SHA256 = (
    "1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589"
)
SOURCE_CAMPAIGN_SHA256 = (
    "2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "e3a05118971bba51e50e07e0fc426a8b321f0ecf823ded7dfec6b3539b630bfd"
)
SEEDS = (7, 29, 53)
REGIMES = ("composition", "extrapolation")
ACTIONS = ("amplitude", "orientation", "offset", "composed")
CLASSIFICATION = "scalar_action_dependent"
EXPECTED_GATES = {
    "basis_gauge_replay_contract": True,
    "correspondence_specificity_all_cells_pass": False,
    "input_and_pair_contract": True,
    "input_identity_replay_contract": True,
    "local_linearization_contract": True,
    "nuisance_scalar_transformation_gate": False,
    "provenance_contract": True,
    "scalar_invariance_all_cells_pass": False,
    "target_control_contract": True,
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
        or provenance.get("group_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("group_implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_encoders", -1)) != 0
        or int(summary.get("fitted_observers", -1)) != 0
        or int(summary.get("group_cells", -1)) != 24
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != EXPECTED_GATE_COUNTS
        or int(aggregates.get("primary_checkpoint_pass_count", -1)) != 0
        or int(aggregates.get("required_checkpoint_pass_count", -1)) != 3
        or aggregates.get("conclusion") != CLASSIFICATION
    ):
        raise ValueError(f"invalid nuisance-scalar transformation campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid nuisance-scalar seed index {path}")

    output: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    expected_cells = {(regime, action) for regime in REGIMES for action in ACTIONS}
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        cells = detail.get("cells", [])
        indexed_cells = {(cell.get("regime"), cell.get("arm")) for cell in cells}
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
            or detail.get("basis_gauge_replay", {}).get(
                "maximum_all_cell_target_replay_error", math.inf
            )
            > 1e-5
            or float(detail.get("primary_metric", 1.0)) != 0.0
            or len(cells) != 8
            or indexed_cells != expected_cells
            or any(cell.get("reference_finite") is not True for cell in cells)
            or any(cell.get("transformed_finite") is not True for cell in cells)
            or any(
                cell.get("reference_linearization", {}).get("adequate") is not True
                or cell.get("transformed_linearization", {}).get("adequate") is not True
                for cell in cells
            )
            or any(cell.get("gate", {}).get("invariance_pass") is not False for cell in cells)
            or any(float(cell["paired"]["zero_referenced_r2"]) >= 0.90 for cell in cells)
            or any(float(cell["paired"]["relative_l2"]) <= math.sqrt(0.10) for cell in cells)
            or any(float(cell["paired"]["sign_agreement"]) >= 0.90 for cell in cells)
        ):
            raise ValueError(f"invalid nuisance-scalar transformation result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate nuisance-scalar fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def _summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    cells = detail["cells"]
    by_regime: dict[str, Any] = {}
    by_action: dict[str, Any] = {}
    for regime in REGIMES:
        selected = [cell for cell in cells if cell["regime"] == regime]
        by_regime[regime] = {
            "mean_zero_referenced_r2": sum(
                cell["paired"]["zero_referenced_r2"] for cell in selected
            )
            / len(selected),
            "mean_relative_l2": sum(
                cell["paired"]["relative_l2"] for cell in selected
            )
            / len(selected),
            "mean_sign_agreement": sum(
                cell["paired"]["sign_agreement"] for cell in selected
            )
            / len(selected),
        }
    for action in ACTIONS:
        selected = [cell for cell in cells if cell["arm"] == action]
        by_action[action] = {
            "mean_zero_referenced_r2": sum(
                cell["paired"]["zero_referenced_r2"] for cell in selected
            )
            / len(selected),
            "invariance_pass_count": sum(
                cell["gate"]["invariance_pass"] for cell in selected
            ),
        }
    return {
        "minimum_zero_referenced_r2": min(
            cell["paired"]["zero_referenced_r2"] for cell in cells
        ),
        "maximum_zero_referenced_r2": max(
            cell["paired"]["zero_referenced_r2"] for cell in cells
        ),
        "minimum_relative_l2": min(
            cell["paired"]["relative_l2"] for cell in cells
        ),
        "maximum_sign_agreement": max(
            cell["paired"]["sign_agreement"] for cell in cells
        ),
        "invariance_pass_count": sum(
            cell["gate"]["invariance_pass"] for cell in cells
        ),
        "correspondence_specificity_pass_count": sum(
            cell["gate"]["correspondence_specific"] for cell in cells
        ),
        "by_regime": by_regime,
        "by_action": by_action,
    }


def build_nuisance_scalar_transformation_law_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-nuisance-scalar-transformation-law.md"
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
            "scalar_summary": _summary(detail),
            "cells": [
                {
                    "regime": cell["regime"],
                    "action": cell["arm"],
                    "paired": cell["paired"],
                    "phase_matched_shuffled": cell["phase_matched_shuffled"],
                    "gate": cell["gate"],
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
            "name": "TinyLLM nuisance-scalar transformation law",
            "category": "mechanistic_interpretability",
            "description": (
                "A no-fit frozen-artifact diagnostic tests whether the exact signed "
                "correction multiplying a portable phase-conditioned task covector "
                "is invariant under target-preserving observed similarity actions."
            ),
            "question": (
                "Is an invariant scalar output type compatible with the exact "
                "checkpoint-local correction target before prospective training?"
            ),
            "prediction": (
                "Amplitude, orientation, offset, and composed actions preserve the "
                "signed correction in composition and extrapolation in all checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "rejected_scalar_action_dependent_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": (
                "underpowered_preregistered_existing_artifact_mechanistic_diagnostic"
            ),
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 checkpoints; exact amplitude, orientation, "
                "offset, and composed actions under composition and extrapolation"
            ),
            "subclaims": {
                "provenance_input_numerical_and_target_contracts": (
                    "supported_three_of_three"
                ),
                "amplitude_invariance": "rejected_zero_of_six_cells",
                "orientation_invariance": "rejected_zero_of_six_cells",
                "offset_invariance": "rejected_zero_of_six_cells",
                "composed_action_invariance": "rejected_zero_of_six_cells",
                "invariant_scalar_output_type": "structurally_incompatible",
                "action_conditioned_sensor_learnability": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c2",
                "scalar-residual",
                "transformation-law",
                "observed-similarity-group",
                "output-type-falsification",
                "frozen-artifact",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A groupoid-cocycle decomposition of direct-state and writer action defects.",
                "A prospectively trained action-conditioned symmetry-defect sensor.",
                "Natural-language tasks or checkpoint-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All validity contracts pass, but every amplitude, orientation, "
                "offset, and composed action cell fails the scalar invariance gate "
                "in every checkpoint."
            ),
            "confidence": 0.995,
            "confidence_assessment": (
                "exact_locked_existing_artifact_output_type_falsification"
            ),
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The correction scalar is action-dependent in all three checkpoints and all 24 cells.",
                "The best paired R2 remains negative and the best sign agreement remains below 66%.",
                "Exact local-linearization and causal target controls pass throughout.",
                "The semantic target may be invariant while a frozen writer-relative error is not.",
                "Prospective invariant scalar training is disallowed by the measured target type.",
            ],
            "suggested_hypotheses": [
                "The scalar difference is a groupoid cocycle equal to the direct-state action defect minus the frozen-writer action defect.",
                "A calibration-conditioned defect channel can predict the action-dependent correction where an invariant head cannot.",
                "If no observable cocycle is stable, a calibrated invariant front end should replace rather than repair the frozen writer.",
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


def build_nuisance_scalar_transformation_law_experiment_results(
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
                    "minimum_paired_zero_referenced_r2": float(
                        summary["minimum_zero_referenced_r2"]
                    ),
                    "maximum_paired_zero_referenced_r2": float(
                        summary["maximum_zero_referenced_r2"]
                    ),
                    "minimum_paired_relative_l2": float(
                        summary["minimum_relative_l2"]
                    ),
                    "maximum_paired_sign_agreement": float(
                        summary["maximum_sign_agreement"]
                    ),
                    "cell_invariance_pass_count": float(
                        summary["invariance_pass_count"]
                    ),
                    "composition_mean_zero_referenced_r2": float(
                        summary["by_regime"]["composition"][
                            "mean_zero_referenced_r2"
                        ]
                    ),
                    "extrapolation_mean_zero_referenced_r2": float(
                        summary["by_regime"]["extrapolation"][
                            "mean_zero_referenced_r2"
                        ]
                    ),
                },
                primary_metric=float(
                    detail["gates"]["nuisance_scalar_transformation_gate"]
                ),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Exact group actions were evaluated without fitting a model, writer, encoder, or observer."
                ],
                anomalies=[
                    "All eight action cells fail scalar invariance despite valid local linearization."
                ],
                timestamp=completed,
            )
        )
    return output


def store_nuisance_scalar_transformation_law_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_nuisance_scalar_transformation_law_meta_hypothesis(results_path)
    experiments = build_nuisance_scalar_transformation_law_experiment_results(
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
            raise RuntimeError("nuisance-scalar transformation ChromaDB read-back failed")
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
    "build_nuisance_scalar_transformation_law_meta_hypothesis",
    "build_nuisance_scalar_transformation_law_experiment_results",
    "store_nuisance_scalar_transformation_law_meta_hypothesis",
]
