"""Build and store TinyLLM frozen activation-conditioned writer evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-frozen-writer-capacity.v1"
HYPOTHESIS_ID = "tinyllm-c2-frozen-writer-capacity-v1"
EVIDENCE_ROLE = "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
IMPLEMENTATION_SHA256 = (
    "d53edaedd49ae553af9f8393d92254664239e5100246ac0fd3a06cb420ca80ed"
)
CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "be4cf87248550c8ace7fc474efff2ce168bce3c9002932ecf6063977c91f0fa6"
)
SEEDS = (7, 29, 53)
FOURIER_ORDERS = (1, 2, 3, 4, 6, 10, 14, 18)
CONTEXT_ORDERS = (1, 2, 3, 4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _writer_names() -> tuple[str, ...]:
    return tuple(f"fourier_m{order:02d}" for order in FOURIER_ORDERS) + tuple(
        f"context_m{order:02d}" for order in CONTEXT_ORDERS
    )


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {str(seed): "small_writer_insufficient" for seed in SEEDS}
    expected_gates = {
        "predecessor_replay_contract": 3,
        "invariant_context_numerical_contract": 3,
        "continuous_target_control_contract": 3,
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
        or int(summary.get("fitted_writers", -1)) != 72
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != expected_gates
        or aggregates.get("conclusion") != "small_writer_insufficient"
    ):
        raise ValueError(f"invalid frozen writer-capacity campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid frozen writer-capacity seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        summaries = detail.get("writer_summaries", {})
        cells = detail.get("heldout_cells", [])
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
            or detail.get("classification") != "small_writer_insufficient"
            or detail.get("selected_writer") is not None
            or gates.get("predecessor_replay_contract") is not True
            or gates.get("invariant_context_numerical_contract") is not True
            or gates.get("continuous_target_control_contract") is not True
            or set(summaries) != set(_writer_names())
            or any(
                item.get("complete_pass") is not False
                or item.get("specificity") is not True
                for item in summaries.values()
            )
            or len(cells) != 4
            or any(
                cell["states"]["zero"]["continuous"]["continuous_pass"] is not False
                or cell["states"]["exact"]["continuous"]["continuous_pass"] is not True
                or cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
                is not True
                for cell in cells
            )
            or int(detail.get("invariant_context", {}).get("rank", -1)) != 3
            or not 0.93
            <= float(
                detail.get("invariant_context", {}).get(
                    "cumulative_energy_fraction", 0.0
                )
            )
            <= 0.97
        ):
            raise ValueError(f"invalid frozen writer-capacity result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate frozen writer fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_frozen_writer_capacity_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-frozen-writer-capacity.md"
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
                "invariant_context": detail["invariant_context"],
                "writer_summaries": detail["writer_summaries"],
                "heldout": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "arms": {
                            name: {
                                "coordinate_metrics": cell["coordinate_metrics"][name],
                                "continuous": cell["states"][name]["continuous"],
                            }
                            for name in _writer_names()
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
            "name": "TinyLLM activation-conditioned frozen writer capacity",
            "category": "representation",
            "description": (
                "A frozen Fourier ladder tests whether exact quotient phase plus "
                "the top three propagated-barycenter activation axes can reproduce "
                "the task-effective rank-three Reynolds defect."
            ),
            "question": (
                "Is quotient curvature or a low-rank chart of the recipient task "
                "activation sufficient for a shift-stable causal write?"
            ),
            "prediction": (
                "At least one low-order phase or activation-conditioned writer "
                "passes all four held-out causal cells with shuffled specificity."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_small_writer_insufficient_three_of_three",
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
                "activation_context_numerics": "supported_three_of_three",
                "target_controls": "supported_three_of_three",
                "low_order_phase_curvature": "rejected_zero_of_three",
                "matched_high_order_phase_capacity": "rejected_zero_of_three",
                "rank_three_propagated_activation_context": "rejected_zero_of_three",
                "paired_specificity": "supported_all_writer_checkpoint_gates",
                "small_writer_insufficiency": "supported_three_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "task-activation",
                "propagated-barycenter",
                "fourier-writer",
                "causal-patching",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Nonlinear or higher-rank recipient-state context.",
                "The downstream continuation task Jacobian and curvature.",
                "A learned model or population prevalence beyond the selected checkpoints.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Every provenance and target-control contract passes, but all phase-only "
                "and rank-three activation-conditioned writers fail the four-cell causal "
                "gate in all three checkpoints."
            ),
            "confidence": 0.96,
            "confidence_assessment": "exact_underpowered_frozen_activation_decomposition",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "The top three propagated-barycenter axes capture 93.85--96.08% of source activation variance.",
                "Activation context improves seed 29 substantially but passes only heldout-B composition.",
                "Best activation writers retain worst-cell coordinate R2 of 0.9868--0.9936 while failing causal use.",
                "Matched Fourier capacity through order 18 does not rescue any checkpoint.",
                "Actual example-specific rank-three defects remain causally sufficient in every cell.",
            ],
            "suggested_hypotheses": [
                "The frozen continuation sharply amplifies a small task-tangent component of writer error.",
                "The nominal carrier kernel has causal higher-order effects under extrapolation.",
                "A local task metric at the continuation is more mechanistically stable than global activation variance.",
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


def build_frozen_writer_capacity_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        gates = detail["gates"]
        metrics: dict[str, float] = {
            "predecessor_replay_contract": float(
                gates["predecessor_replay_contract"]
            ),
            "invariant_context_numerical_contract": float(
                gates["invariant_context_numerical_contract"]
            ),
            "continuous_target_control_contract": float(
                gates["continuous_target_control_contract"]
            ),
            "invariant_context_energy_fraction": float(
                detail["invariant_context"]["cumulative_energy_fraction"]
            ),
        }
        for name, summary in detail["writer_summaries"].items():
            metrics[f"{name}_complete_pass"] = float(summary["complete_pass"])
            metrics[f"{name}_mean_shift_bins"] = float(
                summary["aggregate_mean_shift_bins"]
            )
            metrics[f"{name}_passing_cells"] = float(summary["passing_cells"])
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=0.0,
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen activation-conditioned writer ladder on two held-out cohorts under composition and extrapolation."
                ],
                anomalies=[
                    "Held-out defect-coordinate R2 remains high while every complete causal writer gate fails."
                ],
                timestamp=completed,
            )
        )
    return output


def store_frozen_writer_capacity_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_frozen_writer_capacity_meta_hypothesis(results_path)
    experiments = build_frozen_writer_capacity_experiment_results(
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
            raise RuntimeError("frozen writer-capacity ChromaDB read-back failed")
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
    "build_frozen_writer_capacity_meta_hypothesis",
    "build_frozen_writer_capacity_experiment_results",
    "store_frozen_writer_capacity_meta_hypothesis",
]
