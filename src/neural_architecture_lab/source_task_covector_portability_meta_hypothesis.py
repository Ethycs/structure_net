"""Build and store TinyLLM source task-covector portability evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-source-task-covector-portability.v1"
HYPOTHESIS_ID = "tinyllm-c2-source-task-covector-portability-v1"
EVIDENCE_ROLE = (
    "preregistered_fresh_cohort_post_outcome_underpowered_mechanistic_evidence"
)
IMPLEMENTATION_SHA256 = (
    "6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046"
)
CAMPAIGN_SHA256 = (
    "fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
LOCAL_TANGENT_CAMPAIGN_SHA256 = (
    "824a655b5c6d74f3c77259b9b7cacce3b4b3ea868ba74f48ba63fd5a24395130"
)
SEEDS = (7, 29, 53)
CLASSIFICATION = "source_covector_portable_scalar_not"
EXPECTED_GATE_COUNTS = {
    "all_controls_specific": 0,
    "continuous_target_control_contract": 3,
    "fresh_local_linearization_adequate": 3,
    "local_covector_source_error_all_fresh_cells_pass": 0,
    "local_oracle_all_fresh_cells_pass": 3,
    "numerical_contract": 3,
    "source_covector_oracle_error_all_fresh_cells_pass": 3,
    "source_covector_portability_gate": 0,
    "source_local_linearization_adequate": 3,
    "source_predicted_all_fresh_cells_pass": 0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {str(seed): CLASSIFICATION for seed in SEEDS}
    provenance = value.get("provenance", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or provenance.get("predecessor_campaign_sha256")
        != PREDECESSOR_CAMPAIGN_SHA256
        or provenance.get("local_tangent_campaign_sha256")
        != LOCAL_TANGENT_CAMPAIGN_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_predictive_observers", -1)) != 6
        or int(summary.get("fresh_primary_cells", -1)) != 6
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != EXPECTED_GATE_COUNTS
        or int(aggregates.get("source_covector_portability_pass_count", -1)) != 0
        or int(aggregates.get("required_checkpoint_count", -1)) != 3
        or aggregates.get("supported") is not False
        or aggregates.get("conclusion") != CLASSIFICATION
    ):
        raise ValueError(f"invalid source task-covector portability campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid source task-covector seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        source_fit = detail.get("source_fit", {})
        fresh_fit = detail.get("fresh_map_diagnostics", {})
        state_gates = detail.get("state_gates", {})
        cells = detail.get("fresh_cells", [])
        source_cell_passes = sum(
            cell["states"]["source_predicted"]["continuous"]["continuous_pass"]
            for cell in cells
        )
        expected_source_cell_passes = 1 if seed == 7 else 0
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or int(detail.get("seed", -1)) != seed
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("classification") != CLASSIFICATION
            or gates.get("provenance_contract") is not True
            or gates.get("numerical_contract") is not True
            or gates.get("continuous_target_control_contract") is not True
            or gates.get("source_local_linearization_adequate") is not True
            or gates.get("fresh_local_linearization_adequate") is not True
            or gates.get("local_oracle_all_fresh_cells_pass") is not True
            or gates.get("source_covector_oracle_error_all_fresh_cells_pass")
            is not True
            or gates.get("local_covector_source_error_all_fresh_cells_pass")
            is not False
            or gates.get("source_predicted_all_fresh_cells_pass") is not False
            or gates.get("all_controls_specific") is not False
            or gates.get("source_covector_portability_gate") is not False
            or float(source_fit["covector"]["zero_referenced_r2"]) < 0.98
            or float(fresh_fit["covector"]["zero_referenced_r2"]) < 0.98
            or float(fresh_fit["covector"]["mean_row_cosine"]) < 0.99
            or float(fresh_fit["signed_error"]["zero_referenced_r2"]) > 0.05
            or float(fresh_fit["signed_error"]["sign_agreement"]) > 0.61
            or state_gates.get("all_primary_pass") is not False
            or state_gates.get("all_controls_specific") is not False
            or len(cells) != 2
            or source_cell_passes != expected_source_cell_passes
            or any(
                cell["states"]["local_oracle"]["continuous"]["continuous_pass"]
                is not True
                for cell in cells
            )
            or any(
                cell["states"]["source_covector_oracle_error"]["continuous"]
                ["continuous_pass"]
                is not True
                for cell in cells
            )
            or len(detail.get("source_maps", {}).get("covector", {}).get("linear", []))
            != 9
            or len(
                detail.get("source_maps", {}).get("signed_error", {}).get("linear", [])
            )
            != 9
        ):
            raise ValueError(f"invalid source task-covector result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate source task-covector fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_source_task_covector_portability_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-source-task-covector-portability.md"
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
            "source_fit": detail["source_fit"],
            "fresh_map_diagnostics": detail["fresh_map_diagnostics"],
            "source_linearization": detail["source_linearization"],
            "fresh_linearization": detail["fresh_linearization"],
            "state_gates": detail["state_gates"],
            "fresh_cells": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "evaluation_seed": cell["evaluation_seed"],
                    "map_diagnostics": cell["map_diagnostics"],
                    "states": {
                        name: cell["states"][name]["continuous"]
                        for name in (
                            "order4",
                            "local_oracle",
                            "source_covector_oracle_error",
                            "local_covector_source_error",
                            "source_predicted",
                            "source_mean_covector",
                            "source_shuffled_error",
                            "source_flipped",
                            "source_random_direction",
                            "direct_rank3",
                        )
                    },
                }
                for cell in detail["fresh_cells"]
            ],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM source task-covector portability",
            "category": "mechanistic_interpretability",
            "description": (
                "Source A/B phase maps separately predict the decoder-conditioned "
                "task covector and signed correction amplitude, then intervene on "
                "fresh cohort C without using its exact residual."
            ),
            "question": (
                "Can both the task covector and signed correction amplitude be "
                "predicted from source cohorts and jointly repair fresh composition "
                "and extrapolation examples?"
            ),
            "prediction": (
                "The fully source-predicted correction passes both fresh cells with "
                "specific controls in all three checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_source_covector_portable_scalar_amplitude_failed_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_preregistered_fresh_cohort_post_outcome_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; source A/B Fourier phase maps "
                "evaluated on fresh cohort C composition and extrapolation seeds"
            ),
            "subclaims": {
                "provenance_numerical_and_target_contracts": "supported_three_of_three",
                "fresh_local_tangent_replication": "supported_three_of_three_six_of_six_cells",
                "source_covector_prediction": "supported_three_of_three",
                "source_covector_causal_sufficiency_with_fresh_error": (
                    "supported_three_of_three_six_of_six_cells"
                ),
                "source_signed_error_prediction": "not_supported_three_of_three",
                "fully_source_predicted_correction": "not_supported_zero_of_three_one_of_six_cells",
                "full_preregistered_gate": "not_confirmed_zero_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "task-covector",
                "fresh-cohort",
                "causal-patching",
                "metric-field",
                "scalar-residual",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "An observable replacement for the oracle quotient-phase feature.",
                "A scalar residual sensor using continuation or activation context.",
                "Cross-checkpoint gauge sharing or population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Source-predicted covectors with fresh signed errors pass all six "
                "fresh cells, but the phase-only signed-error map has near-zero "
                "fresh R2 and the fully source-predicted correction passes only one."
            ),
            "confidence": 0.97,
            "confidence_assessment": "exact_underpowered_fresh_cohort_causal_component_split",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The source phase map predicts the fresh task covector at R2 0.989--0.996 and mean cosine 0.9969--0.9995.",
                "Source covector plus fresh signed error passes all six fresh cells.",
                "The phase-only signed-error map has fresh R2 -0.053--0.041 and 54.8--59.8% sign agreement.",
                "The completely source-predicted correction passes one of six cells, matching the order-four baseline count.",
                "Source corrections are only 27.5--45.6% of the local-oracle norm, isolating an example-specific scalar bottleneck.",
            ],
            "suggested_hypotheses": [
                "A minimal observable continuation-residual sensor predicts the signed scalar correction on a fresh cohort while the covector stays frozen.",
                "The task metric field is phase-portable even when carrier displacement magnitude depends on nuisance or local activation context.",
                "If no observable scalar sensor predicts correction sign, the local covector is explanatory but cannot support a source-only sidecar.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "predecessor_campaign_sha256": PREDECESSOR_CAMPAIGN_SHA256,
            "local_tangent_campaign_sha256": LOCAL_TANGENT_CAMPAIGN_SHA256,
        },
    }


def build_source_task_covector_portability_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        gates = detail["gates"]
        fresh = detail["fresh_map_diagnostics"]
        states = detail["state_gates"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "numerical_contract": float(gates["numerical_contract"]),
                    "fresh_local_linearization_adequate": float(
                        gates["fresh_local_linearization_adequate"]
                    ),
                    "fresh_covector_zero_referenced_r2": float(
                        fresh["covector"]["zero_referenced_r2"]
                    ),
                    "fresh_covector_mean_row_cosine": float(
                        fresh["covector"]["mean_row_cosine"]
                    ),
                    "fresh_signed_error_zero_referenced_r2": float(
                        fresh["signed_error"]["zero_referenced_r2"]
                    ),
                    "fresh_signed_error_sign_agreement": float(
                        fresh["signed_error"]["sign_agreement"]
                    ),
                    "local_oracle_all_cells_pass": float(
                        gates["local_oracle_all_fresh_cells_pass"]
                    ),
                    "source_covector_oracle_error_all_cells_pass": float(
                        gates["source_covector_oracle_error_all_fresh_cells_pass"]
                    ),
                    "source_predicted_all_cells_pass": float(
                        gates["source_predicted_all_fresh_cells_pass"]
                    ),
                    "source_covector_portability_gate": float(
                        gates["source_covector_portability_gate"]
                    ),
                    "source_predicted_mean_shift_bins": float(
                        states["aggregate_mean_shift_bins"]["source_predicted"]
                    ),
                    "source_covector_oracle_error_mean_shift_bins": float(
                        states["aggregate_mean_shift_bins"]
                        ["source_covector_oracle_error"]
                    ),
                },
                primary_metric=float(gates["source_covector_portability_gate"]),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen source-fitted covector and scalar maps evaluated by causal rank-three writes on fresh cohort C."
                ],
                anomalies=[
                    "The signed scalar error map fails despite near-exact fresh covector prediction."
                ],
                timestamp=completed,
            )
        )
    return output


def store_source_task_covector_portability_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_source_task_covector_portability_meta_hypothesis(results_path)
    experiments = build_source_task_covector_portability_experiment_results(
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
            raise RuntimeError("source task-covector ChromaDB read-back failed")
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
    "build_source_task_covector_portability_meta_hypothesis",
    "build_source_task_covector_portability_experiment_results",
    "store_source_task_covector_portability_meta_hypothesis",
]
