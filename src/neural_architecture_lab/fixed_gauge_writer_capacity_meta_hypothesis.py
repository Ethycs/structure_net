"""Build and store the TinyLLM fixed-gauge writer-capacity evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-fixed-gauge-writer-capacity.v1"
HYPOTHESIS_ID = "tinyllm-c2-fixed-gauge-writer-capacity-v1"
EVIDENCE_ROLE = (
    "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
)
IMPLEMENTATION_SHA256 = (
    "7c284e35b5afc225eea45309262ab83c5f6d276736a557ebf20675ed3ccbfe7b"
)
CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "be4cf87248550c8ace7fc474efff2ce168bce3c9002932ecf6063977c91f0fa6"
)
SEEDS = (7, 29, 53)
CANDIDATES = (
    "quotient_order2",
    "quotient_order4",
    "quotient_order40",
    "quotient_order4_context",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {str(seed): "unresolved_writer_limited" for seed in SEEDS}
    expected_contracts = {
        "predecessor_replay_contract": 3,
        "oracle_and_context_contract": 3,
        "continuous_target_control_contract": 3,
    }
    expected_candidates = {candidate: 0 for candidate in CANDIDATES}
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
        or int(summary.get("fitted_linear_writers", -1)) != 27
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("contract_counts") != expected_contracts
        or aggregates.get("candidate_specific_pass_counts")
        != expected_candidates
    ):
        raise ValueError(f"invalid fixed-gauge writer-capacity campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid writer-capacity seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        candidates = gates.get("candidates", {})
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
            or detail.get("classification") != "unresolved_writer_limited"
            or gates.get("predecessor_replay_contract") is not True
            or gates.get("oracle_and_context_contract") is not True
            or gates.get("continuous_target_control_contract") is not True
            or set(candidates) != set(CANDIDATES)
            or any(
                candidate.get("causal_all_cells") is not False
                or candidate.get("shuffled_specificity") is not True
                or candidate.get("specific_causal_pass") is not False
                for candidate in candidates.values()
            )
            or len(cells) != 4
            or any(
                cell["states"]["zero"]["continuous"]["continuous_pass"] is not False
                or cell["states"]["exact"]["continuous"]["continuous_pass"] is not True
                or cell["states"]["direct_rank3"]["continuous"]["continuous_pass"]
                is not True
                for cell in cells
            )
        ):
            raise ValueError(f"invalid fixed-gauge writer-capacity result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate writer-capacity fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_fixed_gauge_writer_capacity_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-fixed-gauge-writer-capacity.md"
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
                "heldout": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "arms": {
                            arm: {
                                "coordinate_metrics": cell["coordinate_metrics"][arm],
                                "continuous": cell["states"][arm]["continuous"],
                            }
                            for arm in ("quotient_order1",) + CANDIDATES
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
            "name": "TinyLLM fixed-gauge writer-capacity decomposition",
            "category": "representation",
            "description": (
                "A frozen Fourier and observed-context ladder tests whether the "
                "failed exact quotient-to-residual write is a small curved or "
                "calibration-conditioned map."
            ),
            "question": (
                "Can fixed phase-only curvature through order 40, or an equal-width "
                "phase-plus-calibration map, reproduce the task-effective rank-three defect?"
            ),
            "prediction": (
                "At least one preregistered candidate passes all four held-out causal "
                "cells with its matched shuffled-specificity control."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "diagnostic_unresolved_writer_limited_three_of_three",
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
                "oracle_and_context_contract": "supported_three_of_three",
                "target_controls": "supported_three_of_three",
                "low_order_phase_curvature": "rejected_zero_of_three",
                "high_order_phase_curvature": "rejected_zero_of_three",
                "declared_calibration_context": "rejected_zero_of_three",
                "shuffled_specificity": "supported_all_twelve_candidate_checkpoint_gates",
                "unresolved_writer_limitation": "supported_three_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "fixed-gauge",
                "fourier-writer",
                "context-conditioned-write",
                "causal-patching",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Every nonlinear quotient-only function.",
                "Hidden recipient-state context at the causal front.",
                "A learned model or population prevalence beyond the selected checkpoints.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All contracts and specificity controls pass, but every Fourier and "
                "declared-context candidate fails the four-cell causal gate in all "
                "three checkpoints."
            ),
            "confidence": 0.96,
            "confidence_assessment": "exact_underpowered_frozen_capacity_decomposition",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Orders two and four reduce aggregate error but pass only two of twelve cells.",
                "Order forty passes zero cells and generalizes worse, rejecting a simple missing-harmonics account.",
                "The equal-width calibration-context arm passes zero cells and underperforms order four.",
                "Actual example-specific rank-three coordinates remain causally sufficient in every cell.",
                "The remaining write is likely recipient-state-conditioned rather than an absolute semantic injection."
            ],
            "suggested_hypotheses": [
                "Actual rank-three defects transfer between identical-phase states only when their nuisance context matches.",
                "A small invariant summary of the recipient propagated barycenter predicts the context-dependent defect correction.",
                "A relative equivariant update from local state is more portable than a fixed absolute semantic embedding."
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


def build_fixed_gauge_writer_capacity_experiment_results(
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
            "oracle_and_context_contract": float(
                gates["oracle_and_context_contract"]
            ),
            "continuous_target_control_contract": float(
                gates["continuous_target_control_contract"]
            ),
        }
        for candidate in CANDIDATES:
            candidate_gate = gates["candidates"][candidate]
            metrics[f"{candidate}_specific_causal_pass"] = float(
                candidate_gate["specific_causal_pass"]
            )
            metrics[f"{candidate}_mean_shift_bins"] = float(
                candidate_gate["aggregate_mean_shift_bins"]
            )
            metrics[f"{candidate}_worst_variance_explained"] = float(
                candidate_gate["worst_coordinate_variance_explained"]
            )
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(
                    any(
                        gates["candidates"][candidate]["specific_causal_pass"]
                        for candidate in CANDIDATES
                    )
                ),
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Frozen d6 C2 writer-capacity ladder on two untouched held-out cohorts under both shifts."
                ],
                anomalies=[
                    "Low-order writers improve mean error but no candidate passes all four cells."
                ],
                timestamp=completed,
            )
        )
    return output


def store_fixed_gauge_writer_capacity_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_fixed_gauge_writer_capacity_meta_hypothesis(results_path)
    experiments = build_fixed_gauge_writer_capacity_experiment_results(
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
            raise RuntimeError("writer-capacity ChromaDB read-back failed")
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
    "build_fixed_gauge_writer_capacity_meta_hypothesis",
    "build_fixed_gauge_writer_capacity_experiment_results",
    "store_fixed_gauge_writer_capacity_meta_hypothesis",
]
