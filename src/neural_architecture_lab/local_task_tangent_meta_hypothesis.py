"""Build and store TinyLLM fixed-gauge local task-tangent evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-local-task-tangent.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-task-tangent-v1"
EVIDENCE_ROLE = "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
IMPLEMENTATION_SHA256 = (
    "a89039dd2999b933c69067b76f3f9b311af65f516a6b956fe391d0b1a68d5063"
)
CAMPAIGN_SHA256 = (
    "824a655b5c6d74f3c77259b9b7cacce3b4b3ea868ba74f48ba63fd5a24395130"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
SEEDS = (7, 29, 53)
EXPECTED_CLASSIFICATIONS = {
    7: "nonunique_or_curved_correction",
    29: "local_task_tangent_sufficient",
    53: "local_task_tangent_sufficient",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {
        str(seed): value for seed, value in EXPECTED_CLASSIFICATIONS.items()
    }
    expected_gates = {
        "continuous_target_control_contract": 3,
        "local_linearization_adequate": 3,
        "local_task_tangent_gate": 2,
        "numerical_decomposition_contract": 3,
        "predecessor_replay_contract": 3,
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
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_predictive_observers", -1)) != 0
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != expected_gates
        or int(aggregates.get("local_task_tangent_pass_count", -1)) != 2
        or int(aggregates.get("required_checkpoint_count", -1)) != 3
        or aggregates.get("supported") is not False
        or aggregates.get("conclusion") != "checkpoint_stratified_local_geometry"
    ):
        raise ValueError(f"invalid local task-tangent campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid local task-tangent seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        local = detail.get("local_linearization", {})
        state_gates = detail.get("state_gates", {})
        cells = detail.get("heldout_cells", [])
        expected_primary = seed != 7
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
            or gates.get("numerical_decomposition_contract") is not True
            or gates.get("continuous_target_control_contract") is not True
            or gates.get("local_linearization_adequate") is not True
            or gates.get("local_task_tangent_gate") is not expected_primary
            or float(gates.get("maximum_predecessor_replay_error", 1.0)) != 0.0
            or local.get("adequate") is not True
            or float(local.get("signed_error_zero_referenced_r2", 0.0)) < 0.98
            or float(local.get("prediction_residual_mae_fraction", 1.0)) > 0.06
            or state_gates.get("tangent_all_cells_pass") is not True
            or state_gates.get("all_controls_specific") is not expected_primary
            or len(cells) != 4
            or any(
                cell["states"]["tangent_only"]["continuous"]["continuous_pass"]
                is not True
                for cell in cells
            )
            or any(
                state_gates["controls"][name]["any_failure"] is not True
                for name in (
                    "kernel_only",
                    "tangent_flipped",
                    "tangent_shuffled",
                    "tangent_random",
                )
            )
        ):
            raise ValueError(f"invalid local task-tangent result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate local task-tangent fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_local_task_tangent_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-local-task-tangent.md"
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
            "local_linearization": detail["local_linearization"],
            "state_gates": detail["state_gates"],
            "heldout": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "states": {
                        name: cell["states"][name]["continuous"]
                        for name in (
                            "order4",
                            "tangent_only",
                            "kernel_only",
                            "tangent_flipped",
                            "tangent_shuffled",
                            "tangent_random",
                            "direct_rank3",
                        )
                    },
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
            "name": "TinyLLM fixed-gauge local task-tangent sufficiency",
            "category": "mechanistic_interpretability",
            "description": (
                "A scalar circular-angle gradient decomposes the fixed-gauge "
                "order-four writer residual and tests kernel, flipped, shuffled, "
                "and norm-matched random controls."
            ),
            "question": (
                "Does the scalar task-tangent uniquely repair the older fixed-gauge "
                "writer across every held-out composition and extrapolation cell?"
            ),
            "prediction": (
                "Tangent passes all four cells and beats every declared control by "
                "at least 0.125 bins in all three checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_control_margin_two_of_three_despite_tangent_endpoint_three_of_three"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; fixed-gauge order-four writer "
                "on two held-out cohorts under composition and extrapolation"
            ),
            "subclaims": {
                "predecessor_and_numerical_contracts": "supported_three_of_three",
                "local_linearization": "supported_three_of_three",
                "tangent_continuous_endpoint": "supported_three_of_three_twelve_of_twelve_cells",
                "flipped_shuffled_random_control_failure": "supported_three_of_three",
                "all_absolute_control_margins": "supported_two_of_three",
                "full_preregistered_gate": "not_confirmed_two_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "local-task-gradient",
                "causal-patching",
                "flipped-control",
                "shuffled-control",
                "threshold-miss",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A source-predicted tangent coefficient on a fresh cohort.",
                "A decoder-independent or cross-checkpoint covector.",
                "Population prevalence beyond selected checkpoints.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Tangent closes every held-out cell and every control fails at "
                "least one cell, but seed 7's kernel advantage is 0.1191 rather "
                "than the preregistered 0.125-bin margin."
            ),
            "confidence": 0.96,
            "confidence_assessment": "exact_underpowered_frozen_causal_control_suite",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Scalar tangent correction passes all 12 held-out cells.",
                "The local model explains 98.59--99.73% of signed task error.",
                "Flipped, shuffled, random, and kernel controls each fail a cell in every checkpoint.",
                "The full 0.125-bin uniqueness margin passes two of three checkpoints.",
                "Together with the activation-conditioned protocol, tangent closure holds in six of six checkpoint-protocol pairs.",
            ],
            "suggested_hypotheses": [
                "A source/alignment task covector can predict a scalar correction on a fresh cohort.",
                "A symmetry-typed task covector is more portable than a fixed carrier writer.",
                "A universal sidecar is unjustified until covector transport passes without exact held-out residuals.",
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


def build_local_task_tangent_experiment_results(
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
        states = detail["state_gates"]
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
                    "tangent_all_cells_pass": float(
                        states["tangent_all_cells_pass"]
                    ),
                    "all_controls_specific": float(states["all_controls_specific"]),
                    "local_task_tangent_gate": float(
                        gates["local_task_tangent_gate"]
                    ),
                    "tangent_mean_shift_bins": float(
                        states["aggregate_mean_shift_bins"]["tangent_only"]
                    ),
                    "smallest_control_margin_bins": min(
                        float(value["margin_over_tangent_bins"])
                        for value in states["controls"].values()
                    ),
                },
                primary_metric=float(gates["local_task_tangent_gate"]),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen fixed-gauge order-four residual decomposed by the local scalar task gradient with expanded controls."
                ],
                anomalies=(
                    ["Kernel margin is 0.1191 bins, below the fixed 0.125 threshold."]
                    if detail["seed"] == 7
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_local_task_tangent_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_local_task_tangent_meta_hypothesis(results_path)
    experiments = build_local_task_tangent_experiment_results(record, results_path)
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
            raise RuntimeError("local task-tangent ChromaDB read-back failed")
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
    "build_local_task_tangent_meta_hypothesis",
    "build_local_task_tangent_experiment_results",
    "store_local_task_tangent_meta_hypothesis",
]
