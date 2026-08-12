"""Build and store the C3 connection-observation identifiability evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-connection-observation-identifiability.v1"
HYPOTHESIS_ID = "tinyllm-c3-connection-observation-identifiability-v1"
CLASSIFICATION = "total_holonomy_minimal_known_noise_analytic_no_training_scope"
SEEDS = (1453, 1471, 1483, 1531, 1543)
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_connection_observation_identifiability/"
    "20260811_preregistered/result.json"
)
RESULT_SHA256 = "23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c"
RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_connection_observation_identifiability.py"
)
RUNNER_SHA256 = "a7ecb704dad4f472d57ee94dbda432a632ee9d5a4657dcbfb49307e5beedb819"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-connection-observation-identifiability-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "e2c3a75b50403bfcdf86d89db8b3fdb144ac6b324a89221de326023ec003b938"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-connection-observation-identifiability.md"
)
REPORT_SHA256 = "93b8fd6a6fb945725a72ecd12ced58f9bb0c1ffdf92109e56b646b76376c53eb"
SOURCE_CAMPAIGN_SHA256 = (
    "b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite(item) for item in value)
    return True


def _validated_result(results_path: Path = RESULT_PATH) -> dict[str, Any]:
    if results_path != RESULT_PATH:
        raise ValueError("connection-observation meta record requires sealed result path")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    gates = result.get("gates", {})
    aggregates = result.get("aggregates", {})
    accounting = result.get("accounting", {})
    total = result.get("total_holonomy", {})
    erasure = result.get("single_edge_erasure", {})
    noise = result.get("known_symmetric_noise", {})
    if (
        _sha256(results_path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role")
        != "preregistered_no_training_identifiability_and_artifact_audit"
        or not gates
        or not all(gates.values())
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("primary_hypothesis_pass") is not True
        or aggregates.get("missing_or_partial_connection_training_licensed")
        is not False
        or aggregates.get("known_symmetric_noise_training_licensed") is not False
        or aggregates.get("unrestricted_tinyllm_training_licensed") is not False
        or accounting.get("source_models_loaded") != 5
        or accounting.get("fresh_evaluation_cells") != 10
        or accounting.get("optimizer_steps") != 0
        or accounting.get("learned_probe_fits") != 0
        or accounting.get("tinyllm_models_instantiated") != 0
        or total.get("analytic_clean_pass_count") != 10
        or total.get("fresh_integrity_pass_count") != 10
        or total.get("maximum_full_to_total_prediction_error") != 0.0
        or erasure.get("pass_count") != 7
        or erasure.get("pass") is not True
        or len(noise.get("cells", [])) != 5
        or noise.get("enumeration_pass") is not True
        or noise.get("current_gate_consequence_pass") is not True
        or result.get("source_campaign", {}).get("sha256")
        != SOURCE_CAMPAIGN_SHA256
        or not _finite(result)
    ):
        raise ValueError("invalid C3 connection-observation identifiability result")
    cells = total.get("cells", [])
    if (
        len(cells) != 10
        or {int(item["source_seed"]) for item in cells} != set(SEEDS)
        or not all(item.get("integrity_pass") is True for item in cells)
        or not all(item.get("analytic", {}).get("endpoint_pass") is True for item in cells)
    ):
        raise ValueError("connection-observation fresh population changed")
    for source, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"source artifact changed: {source}")
    return result


def build_c3_connection_observation_identifiability_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    if report_path != REPORT_PATH:
        raise ValueError("connection-observation meta record requires sealed report path")
    result = _validated_result(results_path)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "Total C3 holonomy is the minimal point-identifying connection "
                "statistic and known symmetric noise is analytically closed"
            ),
            "category": "preregistered identifiability and frozen-artifact audit",
            "description": (
                "A no-training audit tests full-to-total connection equivalence, "
                "constructs exact collisions for every one-edge erasure, and "
                "enumerates the known independent symmetric C3 noise law."
            ),
            "question": (
                "Do missing, partial, or known-noisy connections create learned "
                "value over fixed analytic transport in the current generator?"
            ),
            "prediction": (
                "Total holonomy is sufficient; every one-edge erasure destroys "
                "point identification; known symmetric noise has a closed-form "
                "Bayes attenuation and irreducible error."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_total_holonomy_minimal_erasure_nonidentifiable_"
                "known_noise_closed"
            ),
            "evidence_count": 7,
            "direct_experiment_count": 5,
            "evidence_pedigree": (
                "preregistered deterministic algebra plus five frozen source seeds "
                "on two fresh shifts"
            ),
            "tested_scope": (
                "independent-phase C3 endpoint relation, seven discrete edge "
                "connections, exact one-edge erasure, and known independent "
                "symmetric additive edge noise"
            ),
            "success_criteria": {
                "fresh_analytic_endpoint_cells": 10,
                "full_to_total_prediction_error_maximum": 1e-6,
                "exact_erasure_witnesses": 7,
                "erasure_target_separation_minimum": 1.49,
                "noise_enumeration_error_maximum": 1e-12,
                "optimizer_steps": 0,
                "tinyllm_models_instantiated": 0,
            },
            "subclaims": {
                "total_holonomy_sufficiency": "supported_10_of_10_exact_zero_error",
                "total_holonomy_minimality": "supported_7_of_7_exact_collisions",
                "known_noise_closed_form": "supported_5_of_5_enumerations",
                "missing_or_partial_connection_training": "not_licensed",
                "known_symmetric_noise_training": "not_licensed",
                "unknown_observation_dependent_uncertainty": "untested_scope_open",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "connection",
                "holonomy",
                "identifiability",
                "missing-data",
                "bayes-ceiling",
                "no-training",
            ],
            "explicitly_not_tested": [
                "observation-dependent or correlated connection errors",
                "unknown temporal dynamics",
                "repeated views that identify missing holonomy",
                "tasks invariant to connection error",
                "learned robustness or TinyLLM utility",
            ],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "All preregistered total-sufficiency, exact-erasure, known-noise, "
                "gate-consequence, integrity, and zero-training endpoints passed."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "exact_algebra_and_five_seed_frozen_replay_under_declared_law"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "gates": result["gates"],
                "total_holonomy": {
                    "analytic_clean_pass_count": result["total_holonomy"][
                        "analytic_clean_pass_count"
                    ],
                    "maximum_full_to_total_prediction_error": result[
                        "total_holonomy"
                    ]["maximum_full_to_total_prediction_error"],
                },
                "single_edge_erasure_pass_count": result["single_edge_erasure"][
                    "pass_count"
                ],
                "known_noise": result["known_symmetric_noise"],
            },
            "key_insights": [
                (
                    "The full edge path is over-specified for the endpoint task; "
                    "one C3 total holonomy is sufficient for every frozen module."
                ),
                (
                    "That total is minimal under independent phases: hiding any one "
                    "edge creates an exact same-observation target collision."
                ),
                (
                    "Known symmetric noise has a closed-form conditional mean, and "
                    "the clean-data gate becomes information-theoretically impossible "
                    "above per-edge probability 9.5247e-6."
                ),
                (
                    "A learned successor must address identifiable, observation-"
                    "dependent uncertainty that a fixed or scalar adaptive estimator "
                    "cannot already close."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A declared observation-dependent unknown connection-noise law "
                    "with repeated structure may permit a compact learned uncertainty "
                    "estimator to improve over fixed and scalar-adaptive controls."
                )
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {
            "direct_tests": [
                {
                    "source_seed": seed,
                    "fresh_cells": [
                        cell
                        for cell in result["total_holonomy"]["cells"]
                        if int(cell["source_seed"]) == seed
                    ],
                    "shared_exact_gates": {
                        "single_edge_erasure_nonidentifiability": result["gates"][
                            "single_edge_erasure_nonidentifiability"
                        ],
                        "known_noise_enumeration": result["gates"][
                            "known_noise_enumeration"
                        ],
                    },
                }
                for seed in SEEDS
            ],
            "exact_erasure_witnesses": result["single_edge_erasure"],
            "known_noise_enumeration": result["known_symmetric_noise"],
        },
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "artifact_validation": (
                "sealed result/runner/preregistration/report hashes, source campaign "
                "and checkpoint manifests, ten fresh cells, seven exact collisions, "
                "five exhaustive noise laws, and zero-training accounting revalidated"
            ),
        },
    }


def build_c3_connection_observation_identifiability_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _validated_result(results_path)
    completed = datetime.fromisoformat(result["completed_at"].replace("Z", "+00:00"))
    values = []
    for seed in SEEDS:
        cells = [
            item
            for item in result["total_holonomy"]["cells"]
            if int(item["source_seed"]) == seed
        ]
        maximum_error = max(
            max(
                float(item["analytic_full_to_total_maximum_error"]),
                float(item["learned_full_to_total_maximum_error"]),
            )
            for item in cells
        )
        checkpoint = cells[0]["source_checkpoint"]
        metrics = {
            "fresh_shift_count": float(len(cells)),
            "analytic_clean_joint_pass": float(
                len(cells) == 2
                and all(item["analytic"]["endpoint_pass"] for item in cells)
            ),
            "full_to_total_maximum_error": maximum_error,
            "total_holonomy_pass": float(maximum_error <= 1e-6),
            "exact_erasure_witness_pass_count": 7.0,
            "known_noise_enumeration_pass_count": 5.0,
            "optimizer_steps": 0.0,
            "tinyllm_models_instantiated": 0.0,
        }
        values.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-connection-observation-identifiability-"
                    f"source-seed{seed}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(
                    metrics["analytic_clean_joint_pass"]
                    and metrics["total_holonomy_pass"]
                ),
                model_architecture=[1, 16, 8, 2, 1],
                model_parameters=187,
                training_time=0.0,
                model_checkpoint=checkpoint["path"],
                observations=[
                    "Frozen learned-true checkpoint evaluated on two fresh shifts.",
                    "Full connection replaced by canonical total holonomy.",
                    "Shared exact erasure and known-noise algebra performed without fitting.",
                ],
                timestamp=completed,
            )
        )
    return values


def store_c3_connection_observation_identifiability_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_connection_observation_identifiability_meta_hypothesis(
        results_path
    )
    experiments = build_c3_connection_observation_identifiability_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import (
            LoggingConfig,
            StandardizedLogger,
        )

        root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(root / "experiment_queue"),
                sent_dir=str(root / "experiment_sent"),
                rejected_dir=str(root / "experiment_rejected"),
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
        hypothesis = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        stored = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis.get("ids") != [HYPOTHESIS_ID]
            or len(stored.get("ids", [])) != 5
            or {item.get("hypothesis_id") for item in stored.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C3 connection-observation ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": 5,
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return record


__all__ = [
    "build_c3_connection_observation_identifiability_experiment_results",
    "build_c3_connection_observation_identifiability_meta_hypothesis",
    "store_c3_connection_observation_identifiability_meta_hypothesis",
]
