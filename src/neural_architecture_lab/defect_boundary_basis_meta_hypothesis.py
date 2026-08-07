"""Build the TinyLLM source-fitted decoder-boundary basis evidence record."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-defect-boundary-basis.v1"
HYPOTHESIS_ID = "tinyllm-c2-defect-boundary-basis-v1"
PRIMARY_SEEDS = (7, 29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or int(value.get("summary", {}).get("completed", -1)) != 3
        or int(value.get("summary", {}).get("failed", -1)) != 0
        or int(value.get("summary", {}).get("trained_models", -1)) != 0
        or int(value.get("summary", {}).get("fitted_predictive_observers", -1)) != 0
    ):
        raise ValueError(f"invalid defect-boundary basis campaign {path}")
    return value


def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    output = []
    fingerprints = set()
    for seed in PRIMARY_SEEDS:
        detail_path = path.parent / "runs" / f"seed_{seed}" / "result.json"
        detail = json.loads(detail_path.read_text())
        checkpoint = Path(detail.get("provenance", {}).get("checkpoint", ""))
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("implementation_sha256") != campaign["implementation_sha256"]
            or int(detail.get("seed", -1)) != seed
            or not detail.get("scientific_fingerprint")
            or not checkpoint.is_file()
            or _sha256(checkpoint) != detail.get("provenance", {}).get("checkpoint_sha256")
        ):
            raise ValueError(f"invalid defect-boundary basis cell {detail_path}")
        if detail["scientific_fingerprint"] in fingerprints:
            raise ValueError(f"duplicate scientific fingerprint {detail_path}")
        fingerprints.add(detail["scientific_fingerprint"])
        output.append(detail)
    return output


def build_defect_boundary_basis_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path("docs/08 - Analysis/2026-08-06_tinyllm-defect-boundary-basis.md"),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    evidence = [
        {
            "experiment_id": detail["experiment_id"],
            "seed": int(detail["seed"]),
            "checkpoint": detail["provenance"]["checkpoint"],
            "checkpoint_sha256": detail["provenance"]["checkpoint_sha256"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "gates": detail["gates"],
            "source_basis": {
                "base_rank": detail["source_basis"]["base_rank"],
                "boundary_rank": detail["source_basis"]["boundary_rank"],
                "boundary_cumulative_energy": detail["source_basis"]["boundary_cumulative_energy"],
                "source_boundary_cells": detail["source_basis"]["source_boundary_cells"],
            },
            "heldout_cells": [
                {
                    "cohort": cell["cohort"],
                    "regime": cell["regime"],
                    "base_geometric": cell["interventions"]["base_geometric"],
                    "geometric_plus_1": cell["interventions"]["geometric_plus_1"],
                    "boundary_plus_1": cell["interventions"]["boundary_plus_1"],
                    "shuffled_plus_1": cell["interventions"]["shuffled_plus_1"],
                    "random_plus_1": cell["interventions"]["random_plus_1"],
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
            "name": "Decoder-boundary-specific defect correction basis",
            "category": "representation",
            "description": "Source-only frozen-decoder margin-normal basis interventions against geometric, shuffled-pair, and random residual controls.",
            "question": "Do local decoder-boundary normals specifically identify the one extra stable defect direction?",
            "prediction": "A paired boundary direction is sufficient in all checkpoints while shuffled and random same-rank controls fail predecessor cells.",
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_ordered_geometry_not_boundary_specific",
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "power_profile": "underpowered_three_checkpoint_mechanistic_cohort",
            "tested_scope": "three selected stable d6 C2 block-0 fronts; source-only local margin normals; two held-out cohorts and both shifts",
            "subclaims": {
                "basis_and_endpoint_contracts": "supported_three_of_three",
                "paired_boundary_rank_one_sufficiency": "supported_three_of_three",
                "shuffled_and_random_specificity": "not_supported_zero_of_three",
                "ordinary_geometric_next_direction": "supported_three_of_three",
                "adjacent_rank_refinement": "supported_ranks_two_five_three",
                "full_preregistered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm", "decoder-boundary", "causal-patching", "task-weighted-basis",
                "specificity-control", "ordered-defect-geometry", "singular-direction",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + ["The selected three-checkpoint cohort is underpowered for a population-prevalence claim."],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": "One paired boundary direction repairs all checkpoints, but the next geometric and shuffled-pair directions do too; specificity fails in every checkpoint.",
            "confidence": 0.0,
            "confidence_assessment": "three_checkpoint_rejection_with_geometric_shuffled_and_random_controls",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "A one-direction correction is sufficient in all four held-out cells of all three checkpoints.",
                "The next ordinary geometric singular direction is equally sufficient and independently confirms exact ranks 2, 5, and 3.",
                "Shuffling source residual/normal membership does not destroy repair, rejecting example-specific boundary pairing.",
                "Matched random directions in the remaining source span fail, so successful repair still reflects ordered stable geometry.",
                "A three-dimensional semantic carrier is better supported than a learned decoder-boundary sidecar.",
            ],
            "suggested_hypotheses": [
                "Continuous circular performance is captured by the 2--3 dimensional semantic core independently of exact-bin calibration.",
                "A fixed decoder margin calibration repairs seed 29 without adding activation carrier dimensions.",
                "The ordered geometric directions align under a group-typed three-dimensional neutral carrier across checkpoints.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": evidence},
        "source_artifacts": [str(results_path), str(report_path)],
    }


def build_defect_boundary_basis_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    completed = datetime.fromisoformat(_campaign(results_path)["completed_at"].replace("Z", "+00:00"))
    gate_names = (
        "basis_contract",
        "exact_endpoint_replication",
        "predecessor_failure_replication",
        "boundary_rank_one_sufficiency",
        "shuffled_and_random_specificity",
    )
    return [
        ExperimentResult(
            experiment_id=detail["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics={
                **{name: float(detail["gates"][name]) for name in gate_names},
                "base_rank": float(detail["source_basis"]["base_rank"]),
                "minimum_geometric_correction": float(detail["gates"]["minimum_geometric_correction"]),
                "minimum_boundary_correction": float(detail["gates"]["minimum_boundary_correction"]),
                "minimum_shuffled_correction": float(detail["gates"]["minimum_shuffled_correction"]),
            },
            primary_metric=float(all(detail["gates"][name] for name in gate_names)),
            model_architecture=[6, 6, 384],
            model_parameters=29_956_224,
            training_time=float(detail["analysis_seconds"]),
            model_checkpoint=detail["provenance"]["checkpoint"],
            observations=["Frozen checkpoint; source-only decoder-normal basis with geometric, shuffled, and random controls."],
            anomalies=["Paired and shuffled boundary bases are both sufficient, rejecting local-pair specificity."],
            timestamp=completed,
        )
        for detail in _details(results_path)
    ]


def store_defect_boundary_basis_meta_hypothesis(
    results_path: Path, output_path: Path, *, chromadb_path: Path | None = None
) -> dict[str, Any]:
    record = build_defect_boundary_basis_meta_hypothesis(results_path)
    experiments = build_defect_boundary_basis_experiment_results(record, results_path)
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
        storage["result_hashes"] = [logger.log_experiment_result(item) for item in experiments]
        storage["chromadb_path"] = str(chromadb_path)
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


__all__ = [
    "build_defect_boundary_basis_meta_hypothesis",
    "build_defect_boundary_basis_experiment_results",
    "store_defect_boundary_basis_meta_hypothesis",
]

