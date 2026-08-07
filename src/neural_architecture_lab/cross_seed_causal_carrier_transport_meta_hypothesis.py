"""Build and store the TinyLLM cross-seed causal-carrier transport record."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-cross-seed-causal-carrier-transport.v1"
HYPOTHESIS_ID = "tinyllm-c2-cross-seed-causal-carrier-transport-v1"
EVIDENCE_ROLE = "preregistered_underpowered_mechanistic_evidence"
IMPLEMENTATION_SHA256 = (
    "798a85b50f7dc8489e28cc66a498a5c3a8af193f859b80b43b1c5cc551ba0a89"
)
DIRECTED_PAIRS = ((7, 29), (7, 53), (29, 7), (29, 53), (53, 7), (53, 29))
GATE_NAMES = (
    "paired_coordinate_transport",
    "paired_causal_transport",
    "shuffled_specificity",
    "target_control_contract",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    counts = value.get("aggregates", {}).get("gate_counts", {})
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or int(value.get("summary", {}).get("requested", -1)) != 6
        or int(value.get("summary", {}).get("completed", -1)) != 6
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_predictive_observers", -1))
        != 0
        or int(value.get("summary", {}).get("fitted_coordinate_maps", -1)) != 18
        or value.get("aggregates", {}).get("confirmed") is not False
        or counts
        != {
            "paired_causal_transport": 0,
            "paired_coordinate_transport": 6,
            "shuffled_specificity": 6,
            "target_control_contract": 4,
        }
    ):
        raise ValueError(f"invalid cross-seed carrier-transport campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {
        (int(item["source_seed"]), int(item["target_seed"])): item
        for item in campaign.get("results", [])
    }
    if set(entries) != set(DIRECTED_PAIRS):
        raise ValueError(f"invalid directed-pair index {path}")

    details: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for pair in DIRECTED_PAIRS:
        entry = entries[pair]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text())
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or (int(detail.get("source_seed", -1)), int(detail.get("target_seed", -1)))
            != pair
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("gates", {}).get("paired_coordinate_transport") is not True
            or detail.get("gates", {}).get("paired_causal_transport") is not False
            or detail.get("gates", {}).get("shuffled_specificity") is not True
            or len(detail.get("heldout_cells", [])) != 4
        ):
            raise ValueError(f"invalid cross-seed carrier-transport result {detail_path}")

        for role in ("source", "target"):
            provenance = detail.get("provenance", {}).get(role, {})
            for field in ("checkpoint", "frontend_checkpoint", "source_result", "character_result"):
                artifact = Path(provenance.get(field, ""))
                if not artifact.is_file() or _sha256(artifact) != provenance.get(
                    f"{field}_sha256"
                ):
                    raise ValueError(f"invalid {role} {field} in {detail_path}")
        for role in ("source_readout", "target_readout"):
            provenance = detail.get("provenance", {}).get(role, {})
            for field in ("campaign", "result"):
                artifact = Path(provenance.get(field, ""))
                if not artifact.is_file() or _sha256(artifact) != provenance.get(
                    f"{field}_sha256"
                ):
                    raise ValueError(f"invalid {role} {field} in {detail_path}")

        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        details.append(detail)
    return details


def build_cross_seed_causal_carrier_transport_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-06_tinyllm-cross-seed-causal-carrier-transport.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = []
    for detail in details:
        paired_means = [
            cell["states"]["paired"]["continuous"]["mean_moment_shift_bins"]
            for cell in detail["heldout_cells"]
        ]
        paired_p95 = [
            cell["states"]["paired"]["continuous"]["p95_moment_shift_bins"]
            for cell in detail["heldout_cells"]
        ]
        direct_tests.append(
            {
                "experiment_id": detail["experiment_id"],
                "evidence_role": EVIDENCE_ROLE,
                "source_seed": int(detail["source_seed"]),
                "target_seed": int(detail["target_seed"]),
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "gates": detail["gates"],
                "worst_paired_variance_explained": min(
                    cell["coordinate_metrics"]["paired"]["variance_explained"]
                    for cell in detail["heldout_cells"]
                ),
                "maximum_paired_mean_moment_shift_bins": max(paired_means),
                "maximum_paired_p95_moment_shift_bins": max(paired_p95),
            }
        )

    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM cross-seed causal-carrier transport",
            "category": "representation",
            "description": (
                "Label-free whitened orthogonal transport between source-fitted "
                "rank-three C2 causal carriers in frozen TinyLLM checkpoints."
            ),
            "question": (
                "Do the checkpoint-local carriers share one causal coordinate "
                "function up to a label-free linear change of basis?"
            ),
            "prediction": (
                "Every directed pair passes held-out coordinate, continuous, "
                "discrete, target-control, and shuffled-specificity gates."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_geometric_but_not_causal_transport",
            "evidence_count": 6,
            "direct_experiment_count": 6,
            "power_profile": "underpowered_preregistered_mechanistic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "six directed maps among three selected stable d6 C2 block-0 "
                "fronts; two fresh held-out cohorts under both shifts"
            ),
            "subclaims": {
                "paired_coordinate_transport": "supported_six_of_six",
                "paired_causal_transport": "rejected_zero_of_six",
                "shuffled_specificity": "supported_six_of_six",
                "target_control_contract": "supported_four_of_six",
                "common_linear_causal_chart": "rejected",
                "common_statistical_carrier_geometry": "supported_in_selected_cohort",
            },
            "tags": [
                "tinyllm",
                "cross-seed",
                "causal-patching",
                "carrier-transport",
                "c2",
                "procrustes",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A task-Jacobian- or Fisher-weighted group-constrained map.",
                "Population prevalence beyond the selected three checkpoints.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All six maps pass the coordinate and shuffled-specificity gates, "
                "but zero of six pass causal transport and only four of six pass "
                "the target control contract."
            ),
            "confidence": 0.9,
            "confidence_assessment": "exact_preregistered_underpowered_mechanistic_rejection",
            "independent_seed_count": 3,
            "num_direct_experiments": 6,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Every directed map explains at least 0.956 of held-out target-coordinate variance.",
                "All paired maps strongly outperform regime-preserving shuffled correspondence.",
                "No paired or unconstrained affine-ridge map passes the continuous causal endpoint.",
                "Direct target rank-three patches remain close to exact patches, isolating transport as the failure.",
                "The selected carriers share statistical geometry but require checkpoint-local causal charts under ordinary linear metrics.",
            ],
            "suggested_hypotheses": [
                "A deck-group-anchored target-Jacobian metric aligns the causally important carrier directions.",
                "A shared equivariant carrier type can coexist with checkpoint-local readout metrics.",
                "Training an explicitly typed sidecar yields a portable chart only when its causal metric is fixed architecturally.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": _sha256(results_path),
            "implementation_sha256": IMPLEMENTATION_SHA256,
        },
    }


def build_cross_seed_causal_carrier_transport_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output: list[ExperimentResult] = []
    for detail in _details(results_path):
        gates = detail["gates"]
        metrics = {
            name: float(gates[name]) for name in GATE_NAMES
        }
        metrics.update(
            {
                "paired_worst_variance_explained": float(
                    gates["paired_worst_variance_explained"]
                ),
                "shuffled_worst_variance_explained": float(
                    gates["shuffled_worst_variance_explained"]
                ),
                "specificity_margin": float(gates["specificity_margin"]),
                "maximum_paired_mean_moment_shift_bins": max(
                    float(
                        cell["states"]["paired"]["continuous"][
                            "mean_moment_shift_bins"
                        ]
                    )
                    for cell in detail["heldout_cells"]
                ),
                "maximum_paired_p95_moment_shift_bins": max(
                    float(
                        cell["states"]["paired"]["continuous"][
                            "p95_moment_shift_bins"
                        ]
                    )
                    for cell in detail["heldout_cells"]
                ),
            }
        )
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=0.0,
                model_architecture=[6, 6, 384],
                model_parameters=29_956_608,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["target"]["checkpoint"],
                observations=[
                    "Frozen source and target checkpoints; fit on label-free paired source orbits and evaluated on two untouched cohorts under both shifts.",
                    "Coordinate transport passes while causal transport fails.",
                ],
                anomalies=(
                    [
                        "Fresh target-control calibration fails in two composition cells for target seed 7."
                    ]
                    if int(detail["target_seed"]) == 7
                    else []
                ),
                timestamp=completed,
            )
        )
    return output


def store_cross_seed_causal_carrier_transport_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_cross_seed_causal_carrier_transport_meta_hypothesis(results_path)
    experiments = build_cross_seed_causal_carrier_transport_experiment_results(
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
            raise RuntimeError("cross-seed carrier-transport ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


__all__ = [
    "build_cross_seed_causal_carrier_transport_meta_hypothesis",
    "build_cross_seed_causal_carrier_transport_experiment_results",
    "store_cross_seed_causal_carrier_transport_meta_hypothesis",
]
