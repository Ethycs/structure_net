"""Build and store TinyLLM nuisance-fiber task-metric transport evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-nuisance-fiber-task-metric-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-nuisance-fiber-task-metric-transport-v1"
EVIDENCE_ROLE = "preregistered_post_outcome_underpowered_mechanistic_diagnostic"
IMPLEMENTATION_SHA256 = (
    "2c18c1e330f974d59a6e07367b223961322b8ea973c9008374a5431f3b2d0e60"
)
CAMPAIGN_SHA256 = (
    "ccd4444a4091eda06ce699fdb219e751fac44cca91f47142dd73e73a8a6abc55"
)
WRITER_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
LOCAL_METHOD_CAMPAIGN_SHA256 = (
    "8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a"
)
SEEDS = (7, 29, 53)
EXPECTED_CLASSIFICATIONS = {
    7: "invalid_numerical_or_target_controls",
    29: "causal_equivalence_without_projector_equality",
    53: "invalid_exact_action_contract",
}
EXPECTED_RANDOM_CHECKPOINT_PASSES = {7: 8, 29: 6, 53: 8}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or value.get("provenance", {}).get("writer_campaign_sha256")
        != WRITER_CAMPAIGN_SHA256
        or value.get("provenance", {}).get("local_method_campaign_sha256")
        != LOCAL_METHOD_CAMPAIGN_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_predictive_observers", -1)) != 0
        or aggregates.get("confirmed") is not False
        or aggregates.get("conclusion")
        != "not_confirmed_nuisance_fiber_task_metric_transport"
        or aggregates.get("gate_counts")
        != {
            "causal_transport": 3,
            "exact_calibration_action_contract": 2,
            "geometric_transport": 0,
            "numerical_target_control_contract": 1,
            "specificity": 0,
        }
        or aggregates.get("classification_by_seed")
        != {str(seed): EXPECTED_CLASSIFICATIONS[seed] for seed in SEEDS}
    ):
        raise ValueError(f"invalid nuisance-fiber metric campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid nuisance-fiber seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        cells = detail.get("heldout_cells", [])
        expected_action = seed != 53
        expected_controls = seed == 29
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
            or gates.get("exact_calibration_action_contract") is not expected_action
            or gates.get("numerical_target_control_contract") is not expected_controls
            or gates.get("geometric_transport") is not False
            or gates.get("causal_transport") is not True
            or gates.get("specificity") is not False
            or sum(gates.get("random_checkpoint_passes", []))
            != EXPECTED_RANDOM_CHECKPOINT_PASSES[seed]
            or gates.get("shuffled_checkpoint_pass") is not True
            or len(cells) != 4
            or any(
                cell["states"]["local_tangent"]["continuous"]["continuous_pass"]
                is not True
                or cell["states"]["transported_tangent"]["continuous"][
                    "continuous_pass"
                ]
                is not True
                or cell["states"]["shuffled_tangent"]["continuous"][
                    "continuous_pass"
                ]
                is not True
                for cell in cells
            )
        ):
            raise ValueError(f"invalid nuisance-fiber metric result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate nuisance-fiber fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    if (
        sum(
            cell["geometry"]["median_kernel_line_absolute_cosine"] >= 0.90
            and cell["geometry"]["p10_kernel_line_absolute_cosine"] >= 0.75
            for detail in output
            for cell in detail["heldout_cells"]
        )
        != 8
    ):
        raise ValueError("unexpected nuisance-fiber geometric cell count")
    return output


def build_nuisance_fiber_task_metric_transport_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-nuisance-fiber-task-metric-transport.md"
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
            "exact_action_contract": detail["exact_action_contract"],
            "gates": detail["gates"],
            "heldout": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "geometry": cell["geometry"],
                    "numerical": cell["numerical"],
                    "states": {
                        name: cell["states"][name]["continuous"]
                        for name in (
                            "predicted",
                            "local_tangent",
                            "transported_tangent",
                            "shuffled_tangent",
                            "local_kernel",
                            "full",
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
            "name": "TinyLLM nuisance-fiber task-metric transport",
            "category": "mechanistic_interpretability",
            "description": (
                "A no-fit causal audit transports the rank-two continuation-moment "
                "row-space projector between phase-matched N3 nuisance fibers inside "
                "each frozen checkpoint."
            ),
            "question": (
                "Does the local task-metric projector descend along the nuisance "
                "groupoid with the trivial action on the neutral C2 carrier?"
            ),
            "prediction": (
                "Exact calibration action, target controls, projector geometry, "
                "causal rescue, and shuffled/random specificity all pass in three "
                "selected checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_nonspecific_causal_transport_and_extrapolation_tail_geometry"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; two fresh phase-matched "
                "nuisance cohorts under composition and extrapolation"
            ),
            "subclaims": {
                "exact_calibration_action_projector_contract": "supported_two_of_three",
                "numerical_and_target_controls": "supported_one_of_three",
                "transported_tangent_continuous_endpoint": (
                    "supported_three_of_three_twelve_of_twelve_cells"
                ),
                "geometric_projector_transport": "rejected_zero_of_three_eight_of_twelve_cells",
                "rank_two_specificity": "rejected_zero_of_three_random_94_of_96_cells_pass",
                "full_preregistered_gate": "not_confirmed_zero_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "symmetry",
                "nuisance-groupoid",
                "task-metric",
                "causal-transport",
                "nonspecific-control",
                "extrapolation-tail",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A rank-one angular covector nuisance-transport law.",
                "A prospectively constrained equivariant sidecar.",
                "Population prevalence beyond selected checkpoints.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Transported tangents pass all twelve cells, but projector geometry "
                "fails every checkpoint, shuffled tangents pass all twelve cells, "
                "and random rank-two controls pass 94 of 96 cell interventions."
            ),
            "confidence": 0.97,
            "confidence_assessment": "exact_underpowered_frozen_causal_control_suite",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Source-fiber and target-local tangents both pass all 12 fresh cells.",
                "Transported aggregate mean shift differs from local by at most 0.0013 bins per checkpoint.",
                "All six composition geometry cells pass, versus two of six extrapolation cells.",
                "Shuffled tangents pass all 12 cells and random rank-two controls pass 94 of 96 cells.",
                "The exact analytic action is accurate before projector amplification, but its locked projector gate passes two of three checkpoints.",
            ],
            "suggested_hypotheses": [
                "A minimal observable scalar can predict sign and magnitude along the independently validated portable angular covector.",
                "Continuation confidence and circular-posterior residual statistics should be tested before calibration or activation summaries.",
                "A prospectively constrained equivariant sidecar can fix the metric gauge even when post-hoc transport is nonunique.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "writer_campaign_sha256": WRITER_CAMPAIGN_SHA256,
            "local_method_campaign_sha256": LOCAL_METHOD_CAMPAIGN_SHA256,
        },
    }


def build_nuisance_fiber_task_metric_transport_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        gates = detail["gates"]
        cells = detail["heldout_cells"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "exact_action_contract": float(
                        gates["exact_calibration_action_contract"]
                    ),
                    "numerical_target_controls": float(
                        gates["numerical_target_control_contract"]
                    ),
                    "geometric_transport": float(gates["geometric_transport"]),
                    "causal_transport": float(gates["causal_transport"]),
                    "specificity": float(gates["specificity"]),
                    "transported_mean_shift_bins": float(
                        gates["transported_tangent_aggregate_mean_shift_bins"]
                    ),
                    "local_mean_shift_bins": float(
                        gates["local_tangent_aggregate_mean_shift_bins"]
                    ),
                    "minimum_p10_kernel_line_cosine": min(
                        float(cell["geometry"]["p10_kernel_line_absolute_cosine"])
                        for cell in cells
                    ),
                    "random_checkpoint_pass_count": float(
                        sum(gates["random_checkpoint_passes"])
                    ),
                },
                primary_metric=0.0,
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Rank-two local task projector transported between fresh phase-matched nuisance fibers."
                ],
                anomalies=[
                    "Causal transport passes while shuffled and nearly all random rank-two controls also pass."
                ],
                timestamp=completed,
            )
        )
    return output


def store_nuisance_fiber_task_metric_transport_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_nuisance_fiber_task_metric_transport_meta_hypothesis(results_path)
    experiments = build_nuisance_fiber_task_metric_transport_experiment_results(
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
            raise RuntimeError("nuisance-fiber metric ChromaDB read-back failed")
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
    "build_nuisance_fiber_task_metric_transport_meta_hypothesis",
    "build_nuisance_fiber_task_metric_transport_experiment_results",
    "store_nuisance_fiber_task_metric_transport_meta_hypothesis",
]
