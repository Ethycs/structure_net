"""Build and store TinyLLM cyclic writer-intertwiner evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c16-writer-intertwiner.v1"
HYPOTHESIS_ID = "tinyllm-c16-writer-intertwiner-v1"
EVIDENCE_ROLE = "preregistered_post_outcome_underpowered_algebraic_diagnostic"
IMPLEMENTATION_SHA256 = (
    "acec5b9dd0291ddf603c1b65bf2f526f57ef9ae274a39b4249219a80be894702"
)
CAMPAIGN_SHA256 = (
    "75cd3fc0c41a673fde883798bb9a6bfe9e86a65c7e8e155e50b20e09935f3eed"
)
PREDECESSOR_CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
SEEDS = (7, 29, 53)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    expected_classes = {
        str(seed): "harmonic_mixed_writer_image" for seed in SEEDS
    }
    expected_gates = {
        "provenance_and_analytic_contract": 3,
        "writer_rank_three": 3,
        "orbit_obstruction": 0,
        "induced_closure": 2,
        "random_subspace_specificity": 3,
        "writer_intertwiner_gate": 0,
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
        or int(summary.get("random_subspaces", -1)) != 768
        or aggregates.get("classification_by_seed") != expected_classes
        or aggregates.get("gate_counts") != expected_gates
        or int(aggregates.get("writer_intertwiner_pass_count", -1)) != 0
        or aggregates.get("supported") is not False
        or aggregates.get("conclusion") != "harmonic_mixed_writer_image"
        or value.get("analytic_controls", {}).get("passed") is not True
    ):
        raise ValueError(f"invalid cyclic writer-intertwiner campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if set(entries) != set(SEEDS):
        raise ValueError(f"invalid cyclic writer seed index {path}")
    output = []
    fingerprints: set[str] = set()
    for seed in SEEDS:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        gates = detail.get("gates", {})
        writer = detail.get("writer", {})
        random = detail.get("random_controls", {})
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
            or detail.get("classification") != "harmonic_mixed_writer_image"
            or detail.get("analytic_controls", {}).get("passed") is not True
            or gates.get("provenance_and_analytic_contract") is not True
            or gates.get("writer_rank_three") is not True
            or gates.get("orbit_obstruction") is not False
            or gates.get("random_subspace_specificity") is not True
            or gates.get("writer_intertwiner_gate") is not False
            or int(writer.get("writer_rank", -1)) != 3
            or float(writer.get("maximum_nonidentity_obstruction", 0.0)) <= 0.05
            or float(writer.get("maximum_nonidentity_obstruction", 2.0)) >= float(
                random.get("fifth_percentile", 0.0)
            )
            or float(
                writer.get("canonical_harmonic_overlaps", {}).get("1", 0.0)
            )
            < 0.98
            or int(random.get("count", -1)) != 256
        ):
            raise ValueError(f"invalid cyclic writer result {detail_path}")
        fingerprint = detail["scientific_fingerprint"]
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate cyclic writer fingerprint {detail_path}")
        fingerprints.add(fingerprint)
        output.append(detail)
    return output


def build_cyclic_writer_intertwiner_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-cyclic-writer-intertwiner.md"
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
            "writer": detail["writer"],
            "random_controls": detail["random_controls"],
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM C16 quotient-writer intertwiner",
            "category": "mechanistic_interpretability",
            "description": (
                "An exact finite-group audit tests whether the stored rank-three "
                "order-four writer image is invariant under cyclic task-phase rotations."
            ),
            "question": (
                "Does the current writer admit a closed three-dimensional C16 "
                "action through which local task covectors can be transported?"
            ),
            "prediction": (
                "All three writer images have at most 0.05 orbit obstruction, "
                "close under the induced action, and beat random subspaces."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "rejected_harmonic_mixed_writer_image_three_of_three",
            "evidence_count": 3,
            "direct_experiment_count": 3,
            "independent_checkpoint_count": 3,
            "power_profile": "underpowered_preregistered_post_outcome_algebraic_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "three selected d6 C2 block-0 fronts; stored quotient-only "
                "order-four 9x3 writers under the exact C16 output-phase group"
            ),
            "subclaims": {
                "analytic_c16_contract": "supported_exactly",
                "writer_rank_three": "supported_three_of_three",
                "writer_more_structured_than_random": "supported_three_of_three",
                "first_harmonic_dominance": "supported_three_of_three",
                "writer_image_c16_invariance": "rejected_zero_of_three",
                "induced_action_closure": "supported_two_of_three",
                "portable_c16_writer_representation": "rejected_zero_of_three",
            },
            "tags": [
                "tinyllm",
                "c16",
                "representation-theory",
                "intertwiner",
                "harmonic-mixing",
                "writer-interface",
                "algebraic-falsifier",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A typed direct-sum harmonic fusion architecture.",
                "Causal continuation after an exactly equivariant writer.",
                "The full acquisition nuisance group.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All writer images are rank three and first-harmonic dominated, "
                "but maximum C16 orbit obstruction is 0.089--0.201 in every checkpoint."
            ),
            "confidence": 0.99,
            "confidence_assessment": "exact_frozen_algebraic_subspace_audit",
            "independent_seed_count": 3,
            "num_direct_experiments": 3,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                "Canonical feature actions and harmonic controls close to below 1.5e-15.",
                "All writers have rank three with condition numbers 2.16--4.55.",
                "First-harmonic-plus-constant overlap is 0.988--0.997.",
                "Small higher-charge leakage raises maximum orbit obstruction to 0.089--0.201.",
                "The writers beat all random fifth-percentile gates but still fail absolute invariance.",
            ],
            "suggested_hypotheses": [
                "Keeping harmonic irreps as a typed direct sum permits exact equivariant fusion without losing higher-order corrections.",
                "A matched typed fusion writer can improve on the failed order-one projection while retaining exact C16 closure.",
                "Post-hoc 3x3 metric transport through the current writer should remain disallowed.",
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


def build_cyclic_writer_intertwiner_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        writer = detail["writer"]
        gates = detail["gates"]
        random = detail["random_controls"]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "analytic_contract": float(
                        gates["provenance_and_analytic_contract"]
                    ),
                    "writer_rank_three": float(gates["writer_rank_three"]),
                    "orbit_obstruction_pass": float(gates["orbit_obstruction"]),
                    "induced_closure_pass": float(gates["induced_closure"]),
                    "random_specificity": float(
                        gates["random_subspace_specificity"]
                    ),
                    "writer_intertwiner_gate": float(
                        gates["writer_intertwiner_gate"]
                    ),
                    "maximum_orbit_obstruction": float(
                        writer["maximum_nonidentity_obstruction"]
                    ),
                    "rms_orbit_obstruction": float(
                        writer["rms_nonidentity_obstruction"]
                    ),
                    "induced_closure_error": float(
                        writer["induced_closure_error"]
                    ),
                    "first_harmonic_overlap": float(
                        writer["canonical_harmonic_overlaps"]["1"]
                    ),
                    "random_fifth_percentile": float(random["fifth_percentile"]),
                },
                primary_metric=0.0,
                model_architecture=[9, 3],
                model_parameters=27,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["checkpoint"],
                observations=[
                    "Exact C16 subspace and induced-action audit of one stored order-four writer."
                ],
                anomalies=[
                    "The writer is nearly first-harmonic invariant and much better than random, but fails the absolute closure interface."
                ],
                timestamp=completed,
            )
        )
    return output


def store_cyclic_writer_intertwiner_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_cyclic_writer_intertwiner_meta_hypothesis(results_path)
    experiments = build_cyclic_writer_intertwiner_experiment_results(
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
            raise RuntimeError("cyclic writer-intertwiner ChromaDB read-back failed")
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
    "build_cyclic_writer_intertwiner_meta_hypothesis",
    "build_cyclic_writer_intertwiner_experiment_results",
    "store_cyclic_writer_intertwiner_meta_hypothesis",
]
