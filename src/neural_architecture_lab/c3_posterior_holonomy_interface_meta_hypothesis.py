"""Build and store the C3 posterior-holonomy interface evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-posterior-holonomy-interface.v1"
HYPOTHESIS_ID = "tinyllm-c3-posterior-holonomy-interface-v1"
CLASSIFICATION = "posterior_holonomy_moment_exact_soft_interface"
SEEDS = (1453, 1471, 1483, 1531, 1543)
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_posterior_holonomy_interface/"
    "20260811_preregistered/result.json"
)
RESULT_SHA256 = "102125d3c465a30be64a51b6a3b3a59ebb8c350dfb92a562f72684431b4601fc"
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_posterior_holonomy_interface.py"
)
RUNNER_SHA256 = "8556317b17637e784eae0746335bcb4de6cbafb8fa0285508b003e37bef7708c"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-posterior-holonomy-interface-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "c95670b166b7ba1368b2241e0a17ea73c84bd07ec51d232b71ee09e0a5321e02"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-c3-posterior-holonomy-interface.md"
)
REPORT_SHA256 = "0c97b14dff4879b49ea1c46fb2b5ad99d7a23c90ed5187e071998da87a963ffd"
PREDECESSOR_RESULT_SHA256 = (
    "23a8989e820d73d1b72c8abaf3f5b4fde0664b854fb03a17ff6df3c5e2d24c7c"
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
        raise ValueError("posterior-holonomy meta record requires sealed result path")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    gates = result.get("gates", {})
    aggregates = result.get("aggregates", {})
    accounting = result.get("accounting", {})
    inverse = result.get("simplex_moment_inverse", {})
    factorization = result.get("conditional_mean_and_risk", {})
    frozen = result.get("frozen_module_replay", {})
    if (
        _sha256(results_path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role")
        != "preregistered_deterministic_soft_holonomy_interface_audit"
        or not gates
        or not all(gates.values())
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("primary_hypothesis_pass") is not True
        or aggregates.get("soft_holonomy_interface_supported") is not True
        or aggregates.get(
            "compact_posterior_estimator_function_class_design_licensed"
        )
        is not True
        or aggregates.get("posterior_estimator_training_licensed") is not False
        or aggregates.get("uncertainty_law_preflight_required") is not True
        or aggregates.get("unrestricted_tinyllm_training_licensed") is not False
        or accounting.get("simplex_points") != 2_080
        or accounting.get("simplex_phase_cells") != 4_259_840
        or accounting.get("frozen_source_models_loaded") != 5
        or accounting.get("frozen_replay_cells") != 10
        or accounting.get("optimizer_steps") != 0
        or accounting.get("learned_probe_fits") != 0
        or accounting.get("tinyllm_models_instantiated") != 0
        or inverse.get("pass") is not True
        or inverse.get("maximum_probability_reconstruction_error", 1.0) > 1e-12
        or factorization.get("pass") is not True
        or factorization.get("minimum_nonvertex_hard_minus_soft_regret", 0.0)
        <= 0.0
        or frozen.get("pass") is not True
        or frozen.get("pass_count") != 10
        or frozen.get("maximum_replay_error", 1.0) > 1e-6
        or frozen.get("all_source_states_unchanged") is not True
        or result.get("predecessor", {}).get("sha256")
        != PREDECESSOR_RESULT_SHA256
        or not _finite(result)
    ):
        raise ValueError("invalid C3 posterior-holonomy interface result")
    cells = frozen.get("cells", [])
    if (
        len(cells) != 10
        or {int(item["source_seed"]) for item in cells} != set(SEEDS)
        or not all(item.get("pass") is True for item in cells)
    ):
        raise ValueError("posterior-holonomy frozen population changed")
    for source, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"source artifact changed: {source}")
    return result


def build_c3_posterior_holonomy_interface_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    if report_path != REPORT_PATH:
        raise ValueError("posterior-holonomy meta record requires sealed report path")
    result = _validated_result(results_path)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "One complex C3 posterior-holonomy moment is an exact minimal "
                "soft connection interface"
            ),
            "category": "preregistered deterministic uncertainty-interface audit",
            "description": (
                "An exhaustive C3 posterior-simplex and phase-grid calculation "
                "tests probability invertibility, Bayes mean/risk factorization, "
                "coordinate covariance, and frozen linear-head replay."
            ),
            "question": (
                "Does arbitrary total-connection uncertainty collapse without loss "
                "to one typed complex moment before a task head?"
            ),
            "prediction": (
                "The moment reconstructs every C3 posterior, produces the exact "
                "Bayes cosine and risk, strictly dominates hard selection off the "
                "vertices, and matches every frozen neutral linear head."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": "confirmed_exact_c3_soft_holonomy_moment_interface",
            "evidence_count": 6,
            "direct_experiment_count": 5,
            "evidence_pedigree": (
                "preregistered exhaustive algebra plus five frozen source seeds on "
                "two hash-matched shifts"
            ),
            "tested_scope": (
                "arbitrary C3 total-error posterior conditional on a charged neutral "
                "character, cosine squared loss, and the five frozen relational "
                "modules with linear neutral heads"
            ),
            "success_criteria": {
                "posterior_simplex_points": 2_080,
                "phase_points": 2_048,
                "algebra_error_maximum": 1e-12,
                "nonvertex_hard_regret_minimum_exclusive": 0.0,
                "frozen_replay_error_maximum": 1e-6,
                "optimizer_steps": 0,
                "tinyllm_models_instantiated": 0,
            },
            "subclaims": {
                "c3_posterior_moment_invertibility": "supported_2080_of_2080",
                "bayes_mean_and_risk_factorization": (
                    "supported_4259840_posterior_phase_cells"
                ),
                "soft_over_hard_selection": "strict_for_every_nonvertex_grid_posterior",
                "typed_coordinate_covariance": "supported_3_of_3",
                "frozen_soft_interface": "supported_10_of_10",
                "posterior_estimator_training": "not_yet_licensed",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "connection",
                "holonomy",
                "posterior-moment",
                "bayes-risk",
                "soft-interface",
                "no-training",
            ],
            "explicitly_not_tested": [
                "an identifiable unknown uncertainty law",
                "a learned posterior estimator",
                "joint physical-phase and connection uncertainty",
                "cyclic groups larger than C3",
                "TinyLLM utility",
            ],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Every simplex inverse, conditional mean/risk, covariance, frozen "
                "replay, integrity, and zero-training gate passed."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "exhaustive_declared_grid_and_five_seed_frozen_implementation_replay"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "gates": result["gates"],
                "simplex_moment_inverse": result["simplex_moment_inverse"],
                "conditional_mean_and_risk": result[
                    "conditional_mean_and_risk"
                ],
                "coordinate_covariance": result["coordinate_covariance"],
                "frozen_replay": {
                    "pass_count": result["frozen_module_replay"]["pass_count"],
                    "maximum_replay_error": result["frozen_module_replay"][
                        "maximum_replay_error"
                    ],
                },
            },
            "key_insights": [
                (
                    "For C3, one complex first-character moment is equivalent to "
                    "the full three-class posterior because probabilities sum to one."
                ),
                (
                    "Soft character transport is the coordinate-covariant posterior "
                    "operation; averaging integer connection labels is not."
                ),
                (
                    "Hard connection selection adds exactly half the squared distance "
                    "from the posterior moment to the selected root."
                ),
                (
                    "A future model should estimate only the typed posterior moment "
                    "and must first beat fixed, global, and adaptive low-dimensional controls."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A statistically identifiable observation-dependent uncertainty "
                    "law may require a compact covariant estimator of the complex "
                    "posterior holonomy moment."
                )
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {
            "direct_tests": [
                {
                    "source_seed": seed,
                    "frozen_cells": [
                        cell
                        for cell in result["frozen_module_replay"]["cells"]
                        if int(cell["source_seed"]) == seed
                    ],
                    "shared_algebra_gates": {
                        key: result["gates"][key]
                        for key in (
                            "simplex_moment_invertibility",
                            "conditional_mean_and_risk_factorization",
                            "coordinate_covariance",
                        )
                    },
                }
                for seed in SEEDS
            ],
            "exhaustive_simplex": result["simplex_moment_inverse"],
            "exhaustive_mean_and_risk": result["conditional_mean_and_risk"],
            "coordinate_covariance": result["coordinate_covariance"],
        },
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "predecessor_result_sha256": PREDECESSOR_RESULT_SHA256,
            "artifact_validation": (
                "sealed result/runner/preregistration/report hashes, predecessor "
                "classification, exhaustive simplex/phase contracts, ten frozen "
                "cells, state immutability, and zero-training accounting revalidated"
            ),
        },
    }


def build_c3_posterior_holonomy_interface_experiment_results(
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
            for item in result["frozen_module_replay"]["cells"]
            if int(item["source_seed"]) == seed
        ]
        maximum_error = max(
            float(item["posterior_hard_average_to_soft_injection_maximum_error"])
            for item in cells
        )
        checkpoint = cells[0]["source_checkpoint"]
        metrics = {
            "frozen_shift_count": float(len(cells)),
            "frozen_soft_interface_pass": float(
                len(cells) == 2 and all(item["pass"] for item in cells)
            ),
            "frozen_replay_maximum_error": maximum_error,
            "simplex_inverse_pass": 1.0,
            "bayes_factorization_pass": 1.0,
            "coordinate_covariance_pass": 1.0,
            "optimizer_steps": 0.0,
            "tinyllm_models_instantiated": 0.0,
        }
        values.append(
            ExperimentResult(
                experiment_id=f"tinyllm-c3-posterior-holonomy-source-seed{seed}",
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=metrics["frozen_soft_interface_pass"],
                model_architecture=[1, 16, 8, 2, 1],
                model_parameters=187,
                training_time=0.0,
                model_checkpoint=checkpoint["path"],
                observations=[
                    "Frozen source checkpoint replayed on two hash-matched shifts.",
                    "Six posterior-weighted hard counterfactuals matched soft-carrier injection.",
                    "Shared posterior-simplex and Bayes-risk algebra used no fitting.",
                ],
                timestamp=completed,
            )
        )
    return values


def store_c3_posterior_holonomy_interface_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_posterior_holonomy_interface_meta_hypothesis(results_path)
    experiments = build_c3_posterior_holonomy_interface_experiment_results(
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
            raise RuntimeError("C3 posterior-holonomy ChromaDB read-back failed")
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
    "build_c3_posterior_holonomy_interface_experiment_results",
    "build_c3_posterior_holonomy_interface_meta_hypothesis",
    "store_c3_posterior_holonomy_interface_meta_hypothesis",
]
