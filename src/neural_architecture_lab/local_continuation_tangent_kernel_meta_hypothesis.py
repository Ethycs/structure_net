"""Build and store TinyLLM local continuation tangent/kernel evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-local-continuation-tangent-kernel.v1.1"
HYPOTHESIS_ID = "tinyllm-c2-local-continuation-tangent-kernel-v1"
EVIDENCE_ROLE = "post_outcome_corrective_replication_evidence"
IMPLEMENTATION_SHA256 = (
    "31ef191a31bbd5d509fb912cd5164385d846039a3b7c98aca04e6f5e29835c38"
)
CAMPAIGN_SHA256 = (
    "8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
SEEDS = (7, 29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {
        str(seed): "mixed_local_continuation_geometry" for seed in SEEDS
    }
    expected_gates = {
        "predecessor_replay_contract": 3,
        "target_control_contract": 3,
        "jacobian_numerical_contract": 3,
        "finite_difference_contract": 3,
        "decomposition_contract": 3,
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
        or int(summary.get("jacobian_cells", -1)) != 12
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != expected_gates
        or int(aggregates.get("stable_gate_count", -1)) != 0
        or aggregates.get("conclusion") != "mixed_local_continuation_geometry"
    ):
        raise ValueError(f"invalid local continuation campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid local continuation seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        arrays_path = Path(entry["jacobian_arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        aggregate = detail.get("aggregates", {})
        cells = detail.get("heldout_cells", [])
        random = detail.get("random_controls", {})
        stability = detail.get("task_metric_stability", {})
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
            or not arrays_path.is_file()
            or _sha256(arrays_path) != entry.get("jacobian_arrays_sha256")
            or detail.get("classification") != "mixed_local_continuation_geometry"
            or gates.get("predecessor_replay_contract") is not True
            or gates.get("target_control_contract") is not True
            or gates.get("jacobian_numerical_contract") is not True
            or gates.get("finite_difference_contract") is not True
            or gates.get("decomposition_contract") is not True
            or gates.get("stable_local_task_metric_gate") is not False
            or aggregate.get("predicted_checkpoint_pass") is not False
            or aggregate.get("full_checkpoint_pass") is not True
            or aggregate.get("tangent_checkpoint_pass") is not True
            or aggregate.get("kernel_output_inert") is not True
            or aggregate.get("tangent_random_specificity") is not True
            or aggregate.get("kernel_random_specificity") is not False
            or aggregate.get("task_metric_stable") is not False
            or aggregate.get("kernel_causally_active") is not False
            or len(cells) != 4
            or any(
                cell["states"]["tangent"]["continuous"]["continuous_pass"]
                is not True
                or cell["states"]["full"]["continuous"]["continuous_pass"]
                is not True
                or cell["jacobian"]["minimum_jacobian_rank"] != 2
                for cell in cells
            )
            or sum(bool(item) for item in random.get("tangent_checkpoint_passes", []))
            != 0
            or float(stability.get("minimum_top_eigenvector_absolute_cosine", 1.0))
            >= 0.90
            or float(stability.get("maximum_normalized_frobenius_distance", 1.0))
            > 0.35
        ):
            raise ValueError(f"invalid local continuation result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate local continuation fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_local_continuation_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-local-continuation-tangent-kernel.md"
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
                "aggregates": detail["aggregates"],
                "task_metric_stability": detail["task_metric_stability"],
                "random_controls": detail["random_controls"],
                "heldout": [
                    {
                        "cohort": cell["cohort"],
                        "regime": cell["regime"],
                        "coordinate_metrics": cell["coordinate_metrics"],
                        "jacobian": cell["jacobian"],
                        "states": {
                            name: cell["states"][name]
                            for name in ("predicted", "tangent", "kernel", "full")
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
            "name": "TinyLLM local continuation tangent/kernel geometry",
            "category": "mechanistic_interpretability",
            "description": (
                "A no-fit frozen intervention decomposes each failed order-four "
                "writer residual into the row space and kernel of the local "
                "three-coordinate-to-circular-moment Jacobian."
            ),
            "question": (
                "Does one stable local task metric isolate a tangent correction "
                "that repairs the writer while its kernel stays causally inert?"
            ),
            "prediction": (
                "Tangent-only patches pass all four held-out cells with random "
                "specificity, kernel patches remain inert, and the normalized "
                "metric is stable across cells in all three checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_stable_metric_zero_of_three_local_tangent_supported"
            ),
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_post_outcome_corrective_replication",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; orbit-local Jacobians on "
                "two held-out cohorts under composition and extrapolation"
            ),
            "subclaims": {
                "corrective_validity_contracts": "supported_three_of_three",
                "orbit_local_jacobian_rank_two": "supported_all_768_orbits",
                "tangent_causal_sufficiency": "supported_three_of_three_all_12_cells",
                "tangent_random_specificity": "supported_three_of_three",
                "finite_kernel_inertness": "supported_three_of_three",
                "absolute_kernel_random_specificity": "not_supported_zero_of_three",
                "cross_cell_metric_stability": "rejected_zero_of_three",
                "one_stable_local_task_metric": "rejected_zero_of_three",
            },
            "tags": [
                "tinyllm",
                "c2",
                "local-jacobian",
                "task-tangent",
                "task-kernel",
                "causal-patching",
                "metric-field",
                "negative-primary-positive-mechanism",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A deployable estimator of the exact residual.",
                "A declared equivariant transport law for the orbit-local metric field.",
                "Fresh checkpoints or population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Tangent correction and kernel inertness hold in all three "
                "checkpoints, but the preregistered leading-direction stability "
                "and absolute kernel-random specificity gates fail in every checkpoint."
            ),
            "confidence": 0.97,
            "confidence_assessment": (
                "exact_underpowered_post_outcome_corrective_causal_decomposition"
            ),
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "All 768 orbit-local circular-moment Jacobians have rank two.",
                "Tangent-only patches pass all 12 composition/extrapolation cells.",
                "Zero of eight norm-matched random tangent policies passes a checkpoint in every seed.",
                "The finite kernel moves outputs by only 0.00005--0.00021 bins on average per checkpoint.",
                "Minimum leading-direction cosine is 0.759--0.811, rejecting one stable checkpoint-level task metric.",
            ],
            "suggested_hypotheses": [
                "The orbit-local task metric field is equivariant under a declared C2 transport law.",
                "The one-dimensional kernel line is phase-conditioned but nuisance-invariant after group transport.",
                "A typed sidecar must predict a metric field rather than use one fixed writer metric.",
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


def build_local_continuation_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        gates = detail["gates"]
        aggregate = detail["aggregates"]
        stability = detail["task_metric_stability"]
        metrics = {
            "predecessor_replay_contract": float(
                gates["predecessor_replay_contract"]
            ),
            "target_control_contract": float(gates["target_control_contract"]),
            "jacobian_numerical_contract": float(
                gates["jacobian_numerical_contract"]
            ),
            "finite_difference_contract": float(
                gates["finite_difference_contract"]
            ),
            "decomposition_contract": float(gates["decomposition_contract"]),
            "stable_local_task_metric_gate": float(
                gates["stable_local_task_metric_gate"]
            ),
            "tangent_checkpoint_pass": float(aggregate["tangent_checkpoint_pass"]),
            "kernel_output_inert": float(aggregate["kernel_output_inert"]),
            "tangent_random_specificity": float(
                aggregate["tangent_random_specificity"]
            ),
            "kernel_random_specificity": float(
                aggregate["kernel_random_specificity"]
            ),
            "task_metric_stable": float(aggregate["task_metric_stable"]),
            "predicted_mean_shift_bins": float(
                aggregate["predicted_aggregate_mean_shift_bins"]
            ),
            "tangent_mean_shift_bins": float(
                aggregate["tangent_aggregate_mean_shift_bins"]
            ),
            "kernel_mean_movement_bins": float(
                aggregate["kernel_aggregate_mean_movement_bins"]
            ),
            "maximum_metric_frobenius_distance": float(
                stability["maximum_normalized_frobenius_distance"]
            ),
            "minimum_metric_top_cosine": float(
                stability["minimum_top_eigenvector_absolute_cosine"]
            ),
        }
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
                    "Frozen orbit-local tangent/kernel interventions on two held-out cohorts under composition and extrapolation."
                ],
                anomalies=[
                    "The local tangent is sufficient while one cross-cell leading metric direction is not stable."
                ],
                timestamp=completed,
            )
        )
    return output


def store_local_continuation_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_local_continuation_meta_hypothesis(results_path)
    experiments = build_local_continuation_experiment_results(record, results_path)
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
            raise RuntimeError("local continuation ChromaDB read-back failed")
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
    "build_local_continuation_meta_hypothesis",
    "build_local_continuation_experiment_results",
    "store_local_continuation_meta_hypothesis",
]
