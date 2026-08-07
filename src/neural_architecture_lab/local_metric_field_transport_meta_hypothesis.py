"""Build and store TinyLLM local metric-field transport evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c2-local-metric-field-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-local-metric-field-transport-v1"
EVIDENCE_ROLE = "preregistered_fresh_cohort_underpowered_mechanistic_diagnostic"
IMPLEMENTATION_SHA256 = (
    "e3a05118971bba51e50e07e0fc426a8b321f0ecf823ded7dfec6b3539b630bfd"
)
CAMPAIGN_SHA256 = (
    "2aa80cff5882cb79754d732e9012c810c2624934bc5be9e03c9e77de4f525f55"
)
WRITER_CAMPAIGN_SHA256 = (
    "7ab6079d56bc8dc78cc5e1f4011c581b8f507bcd8ec20573bbf8b11227df8d1b"
)
CORRECTIVE_CAMPAIGN_SHA256 = (
    "8b2f25dc61d1ffb93ba9dc66f1a85f36c07bda5a43ed3ed34a9627ca1485c12a"
)
SPECIFICITY_CORRECTIVE_CAMPAIGN_SHA256 = (
    "b2be5e8bcbeacd82baab6193a123c3f0e2002836edde161e0b371c162b548fe5"
)
SEEDS = (7, 29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {str(seed): "nuisance_specific_metric_field" for seed in SEEDS}
    expected_gates = {
        "input_and_pair_contract": 3,
        "numerical_contract": 3,
        "full_control_pass": 3,
        "local_tangent_fresh_cohort_pass": 3,
        "geometric_transport_pass": 0,
        "causal_transport_pass": 0,
        "primary_checkpoint_pass": 0,
    }
    provenance = value.get("provenance", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or provenance.get("writer_campaign_sha256") != WRITER_CAMPAIGN_SHA256
        or provenance.get("corrective_campaign_sha256")
        != CORRECTIVE_CAMPAIGN_SHA256
        or int(summary.get("requested", -1)) != 3
        or int(summary.get("completed", -1)) != 3
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_writers", -1)) != 0
        or int(summary.get("fitted_predictive_observers", -1)) != 0
        or int(summary.get("fresh_group_cells", -1)) != 24
        or int(summary.get("fresh_jacobian_fields", -1)) != 30
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != expected_gates
        or int(aggregates.get("primary_checkpoint_pass_count", -1)) != 0
        or aggregates.get("conclusion") != "nuisance_specific_metric_field"
    ):
        raise ValueError(f"invalid local metric-field transport campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid local metric-field seed index {path}")

    output: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        geometry = detail.get("geometry_cells", [])
        causal = detail.get("causal_cells", [])
        regimes = detail.get("causal_by_regime", {})
        expected_gates = {
            "input_and_pair_contract": True,
            "numerical_contract": True,
            "full_control_pass": True,
            "local_tangent_fresh_cohort_pass": True,
            "geometric_transport_pass": False,
            "causal_transport_pass": False,
            "primary_checkpoint_pass": False,
        }
        all_rank_two_controls_pass = all(
            cell["states"][state]["continuous"]["continuous_pass"] is True
            for cell in causal
            for state in (
                "full",
                "local_tangent",
                "transported_tangent",
                "shuffled_tangent",
                "random_tangent",
            )
        )
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or int(detail.get("seed", -1)) != seed
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "nuisance_specific_metric_field"
            or gates != expected_gates
            or len(geometry) != 8
            or len(causal) != 8
            or any(cell.get("geometry_pass") is not False for cell in geometry)
            or any(
                cell["jacobian"]["minimum_jacobian_rank"] != 2
                or cell["jacobian"]["maximum_jacobian_rank"] != 2
                or cell["input"]["nonfinite_count"] != 0
                or cell["input"]["clipping_fraction"] > 0.01
                or cell["input"]["changed_token_fraction"] < 0.02
                for cell in geometry
            )
            or not all_rank_two_controls_pass
            or set(regimes) != {"composition", "extrapolation"}
            or any(
                record.get("transported_all_cells_pass") is not True
                or record.get("shuffled_any_failure") is not False
                or record.get("random_any_failure") is not False
                or record.get("causal_transport_pass") is not False
                for record in regimes.values()
            )
            or _sha256(detail_path) != entry.get("result_sha256")
            or not arrays_path.is_file()
            or _sha256(arrays_path) != entry.get("arrays_sha256")
            or detail.get("artifacts", {}).get("arrays_sha256")
            != entry.get("arrays_sha256")
        ):
            raise ValueError(f"invalid local metric-field result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate local metric-field fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_local_metric_field_transport_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-local-metric-field-transport.md"
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
                "geometry_cells": detail["geometry_cells"],
                "causal_by_regime": detail["causal_by_regime"],
                "causal_cells": [
                    {
                        "regime": cell["regime"],
                        "arm": cell["arm"],
                        "continuous": {
                            name: cell["states"][name]["continuous"]
                            for name in (
                                "predicted",
                                "full",
                                "local_tangent",
                                "transported_tangent",
                                "shuffled_tangent",
                                "random_tangent",
                            )
                        },
                    }
                    for cell in detail["causal_cells"]
                ],
            }
        )

    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM local task-metric field transport",
            "category": "mechanistic_interpretability",
            "description": (
                "A no-fit frozen intervention compares rank-two continuation "
                "projectors on fresh sensor histories related by exact amplitude, "
                "orientation, offset, and composed observed-group actions."
            ),
            "question": (
                "Does the neutral rank-three carrier's local task-metric field "
                "obey a specific nuisance-invariant identity transport law?"
            ),
            "prediction": (
                "Exact-pair projectors align pointwise and their transported "
                "tangents causally outperform phase-matched shuffled and isotropic "
                "rank-two controls in all three checkpoints."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "control_invalid_corrective_required",
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_preregistered_fresh_cohort_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; 24 fresh exact "
                "similarity-group cells under composition and extrapolation"
            ),
            "subclaims": {
                "fresh_input_and_numerical_contracts": "supported_three_of_three",
                "local_tangent_fresh_cohort_sufficiency": (
                    "supported_three_of_three_twenty_four_of_twenty_four_cells"
                ),
                "identity_transported_tangent_task_equivalence": (
                    "descriptively_supported_three_of_three_twenty_four_of_twenty_four_cells"
                ),
                "registered_nuisance_shuffle_control": "invalidated_structurally",
                "pointwise_projector_geometry": "not_adjudicated_by_registered_joint_gate",
                "rank_two_causal_specificity": (
                    "not_adjudicated_nuisance_shuffle_is_positive_equivalence"
                ),
                "unique_invariant_metric_field": "deferred_to_locked_specificity_corrective",
            },
            "tags": [
                "tinyllm",
                "c2",
                "task-metric",
                "group-action",
                "activation-jacobian",
                "causal-patching",
                "rank-two-specificity",
                "negative-primary-positive-mechanism",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A valid semantic-phase negative control within this campaign.",
                "A rank-one task-covector transport law.",
                "An observable estimator of the signed residual amplitude.",
                "Prospectively trained equivariant architecture or population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The registered nuisance-replicate shuffle remains in the "
                "hypothesized equivalence class, so its required failure cannot "
                "adjudicate nuisance invariance. Identity, nuisance-shuffled, and "
                "random rank-two tangents all pass descriptively."
            ),
            "confidence": 1.0,
            "confidence_assessment": "structural_control_incompatibility_exact",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "All fresh input, target-pair, finite-difference, and rank contracts pass.",
                "Local and identity-transported rank-two tangents pass all 24 transformed cells.",
                "The nuisance-replicate permutation is a positive equivalence under the hypothesis, not a negative control.",
                "Isotropic rank-two controls also pass all 24 cells, leaving the causal endpoint nonspecific.",
                "The locked semantic-phase corrective, not this registered joint gate, adjudicates projector transport.",
            ],
            "suggested_hypotheses": [
                "Observable task confidence and activation summaries predict the example-specific signed scalar correction.",
                "The portable phase-conditioned rank-one covector is the minimal continuation interface.",
                "If the scalar remains unpredictable, the correction is diagnostic rather than a deployable sidecar target.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {
            "direct_tests": direct_tests,
            "validity": {
                "quality_evidence_for_transport_specificity": False,
                "reason": (
                    "The registered negative permutation stays at fixed semantic "
                    "phase and changes only nuisance replicate, which the hypothesis "
                    "declares equivalent."
                ),
                "corrective_campaign_sha256": (
                    SPECIFICITY_CORRECTIVE_CAMPAIGN_SHA256
                ),
                "corrective_hypothesis_id": (
                    "tinyllm-c2-local-metric-field-specificity-v1"
                ),
            },
        },
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "writer_campaign_sha256": WRITER_CAMPAIGN_SHA256,
            "corrective_campaign_sha256": CORRECTIVE_CAMPAIGN_SHA256,
            "specificity_corrective_campaign_sha256": (
                SPECIFICITY_CORRECTIVE_CAMPAIGN_SHA256
            ),
        },
    }


def build_local_metric_field_transport_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        geometry = detail["geometry_cells"]
        regime_records = detail["causal_by_regime"]
        state_means = {
            state: sum(
                record["aggregate_mean_shift_bins"][state]
                for record in regime_records.values()
            )
            / len(regime_records)
            for state in (
                "predicted",
                "local_tangent",
                "transported_tangent",
                "shuffled_tangent",
                "random_tangent",
            )
        }
        metrics = {
            **{name: float(value) for name, value in detail["gates"].items()},
            "transported_all_cells_pass": float(
                all(record["transported_all_cells_pass"] for record in regime_records.values())
            ),
            "shuffled_all_cells_pass": float(
                all(not record["shuffled_any_failure"] for record in regime_records.values())
            ),
            "random_all_cells_pass": float(
                all(not record["random_any_failure"] for record in regime_records.values())
            ),
            "minimum_p10_kernel_cosine": min(
                cell["geometry"]["p10_kernel_cosine"] for cell in geometry
            ),
            "maximum_p95_projector_distance": max(
                cell["geometry"]["p95_projector_distance"] for cell in geometry
            ),
            "maximum_pairing_margin": max(
                cell["geometry"]["paired_over_shuffled_median_margin"]
                for cell in geometry
            ),
            **{f"{name}_mean_shift_bins": value for name, value in state_means.items()},
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
                    "Frozen rank-two metric-field transport across eight fresh exact observed-group cells."
                ],
                anomalies=[
                    "Identity, nuisance-shuffled, and isotropic rank-two projectors all pass the frozen task endpoint."
                ],
                timestamp=completed,
            )
        )
    return output


def store_local_metric_field_transport_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_local_metric_field_transport_meta_hypothesis(results_path)
    experiments = build_local_metric_field_transport_experiment_results(
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
            raise RuntimeError("local metric-field ChromaDB read-back failed")
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
    "build_local_metric_field_transport_meta_hypothesis",
    "build_local_metric_field_transport_experiment_results",
    "store_local_metric_field_transport_meta_hypothesis",
]
