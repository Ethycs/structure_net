"""Build and store TinyLLM local continuation tangent evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-local-continuation-tangent.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-continuation-tangent-v1"
EVIDENCE_ROLE = "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
IMPLEMENTATION_SHA256 = (
    "9cf3d7cc51f19397792da9be2cbe163d573931d3452d31d44bfcaa5583cc4a8e"
)
CAMPAIGN_SHA256 = (
    "de69c302c6418d0b6e0cf7b8254c73fcc4b97fed57472a318e75fe9584ea9e85"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
SEEDS = (7, 29, 53)
EXPECTED_CLASSIFICATIONS = {
    7: "local_task_tangent_sufficient",
    29: "tangent_kernel_interaction_or_endpoint_curvature",
    53: "local_task_tangent_sufficient",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {
        str(seed): classification
        for seed, classification in EXPECTED_CLASSIFICATIONS.items()
    }
    expected_gates = {
        "context_replay_contract": 3,
        "coordinate_scale_contract": 3,
        "local_linearization_adequate": 3,
        "numerical_chart_contract": 3,
        "predecessor_replay_contract": 3,
        "primary_checkpoint_pass": 2,
        "tangent_checkpoint_pass": 3,
        "tangent_specificity": 2,
        "target_control_contract": 3,
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("provenance", {}).get("predecessor_campaign_sha256")
        != PREDECESSOR_CAMPAIGN_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_predictive_observers", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != expected_gates
        or int(aggregates.get("primary_checkpoint_pass_count", -1)) != 2
        or int(aggregates.get("required_checkpoint_pass_count", -1)) != 3
        or aggregates.get("conclusion")
        != "checkpoint_stratified_local_continuation_geometry"
    ):
        raise ValueError(f"invalid local continuation tangent campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid local tangent seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        local = detail.get("local_linearization", {})
        kernel = detail.get("kernel_analysis", {})
        cells = detail.get("heldout_cells", [])
        expected_primary = seed != 29
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
            or detail.get("classification") != EXPECTED_CLASSIFICATIONS[seed]
            or gates.get("predecessor_replay_contract") is not True
            or gates.get("context_replay_contract") is not True
            or gates.get("coordinate_scale_contract") is not True
            or gates.get("numerical_chart_contract") is not True
            or gates.get("target_control_contract") is not True
            or gates.get("local_linearization_adequate") is not True
            or gates.get("tangent_checkpoint_pass") is not True
            or gates.get("random_tangent_any_failure") is not True
            or gates.get("tangent_specificity") is not expected_primary
            or gates.get("primary_checkpoint_pass") is not expected_primary
            or float(gates.get("maximum_predecessor_replay_error", 1.0)) != 0.0
            or float(gates.get("maximum_full_residual_replay_error", 1.0)) != 0.0
            or local.get("adequate") is not True
            or float(local.get("signed_error_zero_referenced_r2", 0.0)) < 0.96
            or float(local.get("sign_agreement", 0.0)) != 1.0
            or kernel.get("material_kernel_effect") is not False
            or float(kernel.get("kernel_change_fraction", 1.0)) >= 0.01
            or len(cells) != 4
            or any(
                cell["states"]["tangent"]["continuous"]["continuous_pass"]
                is not True
                or cell["states"]["full_residual"]["continuous"][
                    "continuous_pass"
                ]
                is not True
                for cell in cells
            )
            or sum(
                cell["states"]["random_tangent"]["continuous"][
                    "continuous_pass"
                ]
                is True
                for cell in cells
            )
            != 1
        ):
            raise ValueError(f"invalid local continuation tangent result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate local tangent fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_local_continuation_tangent_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-local-continuation-tangent.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = []
    for detail in details:
        direct_tests.append(
            {
                "experiment_id": detail["experiment_id"],
                "seed": detail["seed"],
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "classification": detail["classification"],
                "gates": detail["gates"],
                "local_linearization": detail["local_linearization"],
                "kernel_analysis": detail["kernel_analysis"],
                "heldout": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "coordinate_metrics": cell["coordinate_metrics"],
                        "local_chart": cell["local_chart"],
                        "states": {
                            name: cell["states"][name]["continuous"]
                            for name in (
                                "predicted",
                                "tangent",
                                "kernel",
                                "full_residual",
                                "random_tangent",
                                "random_kernel",
                            )
                        },
                    }
                    for cell in detail["heldout_cells"]
                ],
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM local continuation task-tangent sufficiency",
            "category": "mechanistic_interpretability",
            "description": (
                "A frozen causal intervention decomposes the residual of a fixed "
                "state-conditioned rank-three writer into the row space and kernel "
                "of the local circular-output Jacobian."
            ),
            "question": (
                "Does the one-dimensional local task-tangent correction, with "
                "norm-matched random specificity, close every held-out writer error?"
            ),
            "prediction": (
                "The tangent passes all four cells with random specificity in all "
                "three checkpoints, while the first-order kernel is immaterial."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_specificity_two_of_three_despite_tangent_endpoint_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; two held-out cohorts under "
                "composition and extrapolation"
            ),
            "subclaims": {
                "predecessor_replay": "supported_three_of_three",
                "local_linearization": "supported_three_of_three",
                "tangent_continuous_endpoint": "supported_three_of_three_twelve_of_twelve_cells",
                "random_control_any_failure": "supported_three_of_three",
                "random_control_margin": "supported_two_of_three",
                "nominal_kernel_materiality": "rejected_zero_of_three",
                "full_preregistered_gate": "not_confirmed_two_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "local-jacobian",
                "task-tangent",
                "first-order-kernel",
                "causal-patching",
                "threshold-miss",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A deployable predictor of the exact tangent coefficient.",
                "Fresh checkpoints, cells, or population prevalence.",
                "A decoder-independent intrinsic representation metric.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The tangent passes all twelve held-out cells and every numerical "
                "contract, but seed 29 improves over its norm-matched random control "
                "by 0.1196 rather than the preregistered 0.1250-bin margin."
            ),
            "confidence": 0.95,
            "confidence_assessment": "exact_underpowered_frozen_causal_decomposition",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The local task-tangent correction passes all 12 held-out cells.",
                "The local Jacobian explains 96.4--99.5% of zero-referenced signed writer error.",
                "The nominal first-order kernel changes only 0.13--0.93% of the writer gap.",
                "Each norm-matched random tangent passes only heldout-B composition.",
                "Seed 29 misses the fixed specificity margin by 0.0054 bins, so the complete gate remains false.",
            ],
            "suggested_hypotheses": [
                "A single portable scalar tangent coefficient is sufficient for the frozen correction.",
                "If that scalar is not portable, the task metric must be fixed architecturally rather than post hoc.",
                "Broad kernel Hessian terms are unnecessary at the measured writer states.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "predecessor_campaign_sha256": PREDECESSOR_CAMPAIGN_SHA256,
        },
    }


def build_local_continuation_tangent_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        gates = detail["gates"]
        local = detail["local_linearization"]
        kernel = detail["kernel_analysis"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "predecessor_replay_contract": float(
                        gates["predecessor_replay_contract"]
                    ),
                    "local_linearization_adequate": float(
                        gates["local_linearization_adequate"]
                    ),
                    "signed_error_zero_referenced_r2": float(
                        local["signed_error_zero_referenced_r2"]
                    ),
                    "prediction_residual_mae_fraction": float(
                        local["prediction_residual_mae_fraction"]
                    ),
                    "tangent_checkpoint_pass": float(
                        gates["tangent_checkpoint_pass"]
                    ),
                    "tangent_specificity": float(gates["tangent_specificity"]),
                    "primary_checkpoint_pass": float(
                        gates["primary_checkpoint_pass"]
                    ),
                    "tangent_mean_shift_bins": float(
                        gates["tangent_aggregate_mean_shift_bins"]
                    ),
                    "random_tangent_mean_shift_bins": float(
                        gates["random_tangent_aggregate_mean_shift_bins"]
                    ),
                    "kernel_change_fraction": float(
                        kernel["kernel_change_fraction"]
                    ),
                },
                primary_metric=float(gates["primary_checkpoint_pass"]),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen context_m04 residual decomposed into local circular-output tangent and first-order kernel."
                ],
                anomalies=(
                    [
                        "Tangent passes every cell but misses the preregistered random-control margin by 0.0054 bins."
                    ]
                    if detail["seed"] == 29
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_local_continuation_tangent_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_local_continuation_tangent_meta_hypothesis(results_path)
    experiments = build_local_continuation_tangent_experiment_results(
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
            raise RuntimeError("local continuation tangent ChromaDB read-back failed")
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
    "build_local_continuation_tangent_meta_hypothesis",
    "build_local_continuation_tangent_experiment_results",
    "store_local_continuation_tangent_meta_hypothesis",
]
