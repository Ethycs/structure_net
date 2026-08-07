"""Build and store TinyLLM task-activation scalar-sensor evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-task-activation-scalar-sensor.v2"
HYPOTHESIS_ID = "tinyllm-c2-task-activation-scalar-sensor-v2"
EVIDENCE_ROLE = (
    "preregistered_fresh_e_corrective_underpowered_mechanistic_evidence"
)
IMPLEMENTATION_SHA256 = (
    "698744c6e1791cec7f9e1cdd0504346ad84211a1313509c82dbdc1eddc29d03c"
)
CAMPAIGN_SHA256 = (
    "a9a1b910b1d5f86a834fb3f03c2b9bb050e365bb5a65b8c35b50cbbda6ea37a8"
)
SOURCE_CAMPAIGN_SHA256 = (
    "fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5"
)
SEEDS = (7, 29, 53)
CLASSIFICATION = "observable_scalar_not_identified"
EXPECTED_GATES = {
    "all_controls_specific": False,
    "causal_activation_causal_gate": False,
    "causal_activation_predictive_gate": False,
    "continuous_target_control_contract": True,
    "fresh_local_linearization_adequate": True,
    "local_oracle_all_fresh_cells_pass": True,
    "lookahead_any_joint_pass": False,
    "numerical_contract": True,
    "pca_contract": True,
    "provenance_contract": True,
    "source_covector_oracle_error_all_fresh_cells_pass": True,
    "source_local_linearization_adequate": True,
    "task_activation_scalar_sensor_gate": False,
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
    expected_classes = {str(seed): CLASSIFICATION for seed in SEEDS}
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("evaluation_cohort") != "heldout_e"
        or value.get("evaluation_seeds")
        != {"composition": 630007, "extrapolation": 630008}
        or value.get("provenance", {}).get("source_campaign_sha256")
        != SOURCE_CAMPAIGN_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_scalar_observers", -1)) != 24
        or int(summary.get("fitted_pca_summaries", -1)) != 9
        or int(summary.get("fresh_primary_cells", -1)) != 6
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != EXPECTED_GATE_COUNTS
        or int(aggregates.get("primary_checkpoint_pass_count", -1)) != 0
        or int(aggregates.get("required_checkpoint_pass_count", -1)) != 3
        or aggregates.get("conclusion") != CLASSIFICATION
    ):
        raise ValueError(f"invalid task-activation scalar-sensor campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid task-activation scalar-sensor seed index {path}")

    output: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        cells = detail.get("fresh_cells", [])
        causal = detail.get("causal_gates", {})
        lookahead = detail.get("lookahead_gates", {})
        regimes = {cell.get("regime"): cell for cell in cells}
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or int(detail.get("seed", -1)) != seed
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != CLASSIFICATION
            or detail.get("gates") != EXPECTED_GATES
            or _sha256(detail_path) != entry.get("result_sha256")
            or not arrays_path.is_file()
            or _sha256(arrays_path) != entry.get("arrays_sha256")
            or detail.get("artifacts", {}).get("arrays_sha256")
            != entry.get("arrays_sha256")
            or len(detail.get("pca", {})) != 3
            or any(
                int(record.get("rank", -1)) != 8
                or float(record.get("orthogonality_error", 1.0)) > 1e-8
                for record in detail.get("pca", {}).values()
            )
            or detail.get("source_linearization", {}).get("adequate") is not True
            or detail.get("fresh_linearization", {}).get("adequate") is not True
            or set(regimes) != {"composition", "extrapolation"}
            or any(
                float(cell["covector_diagnostics"]["zero_referenced_r2"])
                < 0.97
                for cell in cells
            )
            or any(
                cell["states"]["local_oracle"]["continuous"]["continuous_pass"]
                is not True
                or cell["states"]["source_covector_oracle_error"]["continuous"]
                ["continuous_pass"]
                is not True
                for cell in cells
            )
            or regimes["composition"]["states"]["causal_combined"]["continuous"]
            ["continuous_pass"]
            is not True
            or regimes["extrapolation"]["states"]["causal_combined"]["continuous"]
            ["continuous_pass"]
            is not False
            or causal.get("primary_causal_scalar_gate") is not False
            or causal.get("all_controls_specific") is not False
            or any(
                record != {"causal_all_cells_pass": False, "predictive_pass": False}
                for record in lookahead.values()
            )
        ):
            raise ValueError(f"invalid task-activation scalar-sensor result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(
                f"duplicate task-activation scalar-sensor fingerprint {detail_path}"
            )
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_task_activation_scalar_sensor_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-task-activation-scalar-sensor-v2.md"
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
            "source_linearization": detail["source_linearization"],
            "fresh_linearization": detail["fresh_linearization"],
            "causal_gates": detail["causal_gates"],
            "lookahead_gates": detail["lookahead_gates"],
            "fresh_cells": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "evaluation_seed": cell["evaluation_seed"],
                    "covector_diagnostics": cell["covector_diagnostics"],
                    "scalar_diagnostics": cell["scalar_diagnostics"],
                    "states": {
                        name: cell["states"][name]["continuous"]
                        for name in (
                            "order4",
                            "local_oracle",
                            "source_covector_oracle_error",
                            "causal_combined",
                            "post_mlp_lookahead",
                            "output_lookahead",
                            "full_lookahead",
                            "causal_shuffled",
                            "causal_flipped",
                            "causal_random_direction",
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
            "name": "TinyLLM task-activation scalar sensor",
            "category": "mechanistic_interpretability",
            "description": (
                "Source-only PCA summaries of task-cut activations and observed "
                "calibration features predict the signed scalar multiplying a "
                "frozen phase-conditioned task covector, then intervene on fresh E."
            ),
            "question": (
                "Do observable task-cut activations contain a source-portable "
                "estimate of the example-specific signed task correction?"
            ),
            "prediction": (
                "The pre-write activation sensor meets predictive and frozen "
                "causal gates on composition and extrapolation in all checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_observable_scalar_not_identified_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": (
                "underpowered_preregistered_fresh_e_corrective_mechanistic_diagnostic"
            ),
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; source A/B scalar fits "
                "evaluated once on fresh-E composition and extrapolation"
            ),
            "subclaims": {
                "provenance_numerical_pca_and_target_contracts": (
                    "supported_three_of_three"
                ),
                "portable_covector_with_exact_scalar": (
                    "supported_three_of_three_six_of_six_cells"
                ),
                "primary_activation_scalar_prediction": (
                    "not_supported_zero_of_three"
                ),
                "primary_activation_scalar_causal_composition": (
                    "descriptively_supported_three_of_three"
                ),
                "primary_activation_scalar_causal_extrapolation": (
                    "not_supported_zero_of_three"
                ),
                "lookahead_scalar_sensor": "not_supported_zero_of_three",
                "full_preregistered_gate": "not_confirmed_zero_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "activation-analysis",
                "task-covector",
                "scalar-residual",
                "fresh-cohort",
                "causal-patching",
                "support-relative",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Nonlinear or outcome-selected retrospective scalar observers.",
                "A prospectively trained typed equivariant scalar sensor.",
                "Natural-language tasks or checkpoint-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All exact-scalar covector controls pass, but no registered "
                "activation observer meets the predictive gate, causal success "
                "is composition-only, and no look-ahead arm jointly passes."
            ),
            "confidence": 0.98,
            "confidence_assessment": (
                "exact_underpowered_fresh_cohort_component_falsification"
            ),
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The frozen task covector plus exact scalar repairs all six fresh-E cells.",
                "The primary activation scalar sensor fails prediction in all three checkpoints.",
                "Primary activation corrections pass composition but fail extrapolation in every checkpoint.",
                "Post-MLP and output-posterior look-ahead summaries provide no joint rescue.",
                "The portable covector remains explanatory, but the tested activations do not expose a portable scalar sidecar interface.",
            ],
            "suggested_hypotheses": [
                "A prospectively trained symmetry-typed scalar sensor makes correction amplitude explicit and extrapolation-stable.",
                "A parameter-matched untyped scalar head will remain support-relative under the same fresh gates.",
                "If typed training fails, retain the local covector only as a diagnostic rather than a deployable module.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        },
    }


def build_task_activation_scalar_sensor_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        cells = {cell["regime"]: cell for cell in detail["fresh_cells"]}
        causal = detail["causal_gates"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    **{name: float(value) for name, value in detail["gates"].items()},
                    "source_primary_zero_referenced_r2": float(
                        detail["source_fit"]["causal_combined"]["zero_referenced_r2"]
                    ),
                    "composition_primary_zero_referenced_r2": float(
                        cells["composition"]["scalar_diagnostics"]["causal_combined"]
                        ["zero_referenced_r2"]
                    ),
                    "extrapolation_primary_zero_referenced_r2": float(
                        cells["extrapolation"]["scalar_diagnostics"]["causal_combined"]
                        ["zero_referenced_r2"]
                    ),
                    "composition_primary_sign_agreement": float(
                        cells["composition"]["scalar_diagnostics"]["causal_combined"]
                        ["sign_agreement"]
                    ),
                    "extrapolation_primary_sign_agreement": float(
                        cells["extrapolation"]["scalar_diagnostics"]["causal_combined"]
                        ["sign_agreement"]
                    ),
                    "composition_primary_causal_pass": float(
                        cells["composition"]["states"]["causal_combined"]
                        ["continuous"]["continuous_pass"]
                    ),
                    "extrapolation_primary_causal_pass": float(
                        cells["extrapolation"]["states"]["causal_combined"]
                        ["continuous"]["continuous_pass"]
                    ),
                    "primary_mean_shift_bins": float(
                        causal["aggregate_mean_shift_bins"]["causal_combined"]
                    ),
                    "covector_exact_scalar_mean_shift_bins": float(
                        causal["aggregate_mean_shift_bins"]
                        ["source_covector_oracle_error"]
                    ),
                },
                primary_metric=float(
                    detail["gates"]["task_activation_scalar_sensor_gate"]
                ),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Source-only activation observers and frozen causal patches evaluated on fresh cohort E."
                ],
                anomalies=[
                    "The primary correction passes composition but fails extrapolation in every checkpoint."
                ],
                timestamp=completed,
            )
        )
    return output


def store_task_activation_scalar_sensor_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_task_activation_scalar_sensor_meta_hypothesis(results_path)
    experiments = build_task_activation_scalar_sensor_experiment_results(
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
            raise RuntimeError("task-activation scalar-sensor ChromaDB read-back failed")
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
    "build_task_activation_scalar_sensor_meta_hypothesis",
    "build_task_activation_scalar_sensor_experiment_results",
    "store_task_activation_scalar_sensor_meta_hypothesis",
]
