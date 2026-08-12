"""Build and store the observed C3 relational-connection preflight result."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-relational-connection-preflight.v1"
HYPOTHESIS_ID = "tinyllm-c3-relational-connection-preflight-v1"
EVIDENCE_ROLE = "prospective_no_training_identifiability_and_fixed_ceiling"
CLASSIFICATION = "observed_edge_connection_identifies_nonpointwise_c3_relation"
CONFIRMATION_STATUS = (
    "confirmed_observed_connection_identifiability_and_specificity"
)
RESULT_SHA256 = "ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e"
RUNNER_SHA256 = "2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214"
PREREGISTRATION_SHA256 = (
    "702196e0eaaaacda90293d202859e258889e1bed553dea4e2f7986f1ac8a57cc"
)
REPORT_SHA256 = "4631f3ab2f99702e384d8b66c1dac4251cb63f4207ddbf0dca03cbe413a40aff"
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_preflight/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_relational_connection_preflight.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-preflight-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-relational-connection-preflight.md"
)
SEEDS = (1213, 1231, 1277, 1301, 1321)
REGIMES = ("composition", "extrapolation")


@lru_cache(maxsize=64)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns)


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


def _expected_aggregates() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "connection_conditioned_function_class_preflight_licensed": True,
        "connection_free_shortcut_seed_pass_count": 0,
        "connection_specificity_seed_pass_count": 5,
        "matched_training_directly_licensed": False,
        "observed_connection_seed_pass_count": 5,
        "pointwise_insufficiency_seed_pass_count": 5,
        "required_seed_passes": 4,
        "unrestricted_tinyllm_training_licensed": False,
        "valid": True,
    }


def _expected_accounting() -> dict[str, int]:
    return {
        "checkpoints_loaded": 0,
        "local_action_counterfactual_evaluations": 40_960,
        "new_evaluation_examples": 40_960,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "reusable_parameter_fits": 0,
        "target_using_fits": 0,
        "tinyllm_models_instantiated": 0,
    }


def _load_and_validate(path: Path = RESULT_PATH) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    sources = result.get("source_hashes", {})
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("aggregates") != _expected_aggregates()
        or result.get("accounting") != _expected_accounting()
        or sources.get("runner") != RUNNER_SHA256
        or sources.get("preregistration") != PREREGISTRATION_SHA256
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 relational-connection result {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")
    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime) for seed in SEEDS for regime in REGIMES
    }:
        raise ValueError("relational-connection population changed")
    for cell in cells:
        observed = cell.get("arms", {}).get("observed_connection", {})
        if (
            cell.get("valid") is not True
            or cell.get("observed_connection_pass") is not True
            or cell.get("connection_specificity_pass") is not True
            or cell.get("pointwise_insufficiency_pass") is not True
            or cell.get("connection_free_shortcut_pass") is not False
            or cell.get("observed_pair_contracts", {}).get("pass") is not True
            or cell.get("continuous_oracle_maximum_error", math.inf) > 1e-12
            or cell.get("local_action_prediction_maximum_error", math.inf) > 2e-12
            or observed.get("positive_task_pass") is not True
            or observed.get("scalar", {}).get("rmse", math.inf) > 0.01
        ):
            raise ValueError("relational-connection cell changed")
        for control in (
            "no_connection",
            "wrong_sign_connection",
            "shuffled_connection",
            "principal_pointwise_invariant",
        ):
            if (
                cell["arms"].get(control, {}).get("failure_control_pass")
                is not True
            ):
                raise ValueError("relational-connection control changed")
        if (
            cell["arms"]["pointwise_invariant_bayes_zero"].get(
                "bayes_zero_pass"
            )
            is not True
            or cell.get("shuffled_target", {}).get("pass") is not True
        ):
            raise ValueError("pointwise or target control changed")
    witness = result.get("identifiability", {}).get("collision_witness", {})
    if (
        witness.get("pass") is not True
        or witness.get("token_error_count_without_connection") != 0
        or witness.get("connection_difference_count") != 1
        or witness.get("target_absolute_separation", 0.0) < 1.0
    ):
        raise ValueError("identifiability collision changed")
    return result


def _cells_by_seed(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexed = {
        (int(cell["seed"]), cell["regime"]): cell for cell in result["cells"]
    }
    return [
        {
            "seed": seed,
            "composition": indexed[(seed, "composition")],
            "extrapolation": indexed[(seed, "extrapolation")],
        }
        for seed in SEEDS
    ]


def _seed_pass(pair: Mapping[str, Any]) -> bool:
    return all(
        pair[regime]["valid"]
        and pair[regime]["observed_connection_pass"]
        and pair[regime]["connection_specificity_pass"]
        and pair[regime]["pointwise_insufficiency_pass"]
        and not pair[regime]["connection_free_shortcut_pass"]
        for regime in REGIMES
    )


def build_c3_relational_connection_preflight_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _load_and_validate(results_path)
    direct = _cells_by_seed(result)
    successful = sum(_seed_pass(pair) for pair in direct)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Observed C3 connection identifies a non-pointwise relation",
            "category": "prospective no-training identifiability and causal preflight",
            "description": (
                "A calibrated charged endpoint relation is evaluated with an "
                "observed discrete edge connection after pointwise invariant "
                "sufficiency is removed by independent interior phases."
            ),
            "question": (
                "Does an observed C3 edge connection supply the missing local-frame "
                "comparison for cos(theta7-theta0), while every connection-free "
                "and pointwise-invariant control fails?"
            ),
            "prediction": (
                "The analytic connection arm passes both shifts in at least four "
                "of five seeds, all registered specificity controls fail, and an "
                "exact token collision proves pointwise insufficiency."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": CONFIRMATION_STATUS,
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five fresh seeds, composition and extrapolation nuisances, 4,096 "
                "examples per cell, exact C3 local frames, observed edge differences, "
                "calibrated quantized three-channel endpoints"
            ),
            "success_criteria": {
                "joint_seed_passes_minimum": 4,
                "observed_scalar_correlation_minimum": 0.999,
                "observed_scalar_rmse_maximum": 0.01,
                "observed_exact_bin_accuracy_minimum": 0.98,
                "control_absolute_correlation_maximum": 0.10,
                "control_rmse_minimum": 0.80,
                "local_action_error_maximum": 2e-12,
            },
            "subclaims": {
                "observed_connection_fixed_ceiling": "supported_five_of_five",
                "connection_specificity": "supported_five_of_five",
                "pointwise_invariant_insufficiency": "supported_five_of_five_plus_exact_collision",
                "connection_free_shortcut": "rejected_zero_of_five",
                "continuous_observed_pair_identifiability": "supported_symbolically_and_numerically",
                "learned_connection_use": "not_tested",
                "function_class_lifecycle_preflight": "licensed",
                "matched_training": "not_yet_licensed",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "connection",
                "relational-target",
                "identifiability",
                "gauge-covariance",
                "pointwise-insufficiency",
                "causal-control",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "The observed connection passes the full joint gate in 5/5 fresh "
                "seeds; every no/wrong/shuffled/pointwise control fails, and two "
                "identical token observations have targets separated by 1.5 until "
                "the differing edge connection is included."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "preregistered_five_seed_result_plus_exact_collision_and_action_law"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": successful,
            "failed_direct_experiments": 5 - successful,
            "primary_seed_passes": successful,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "Connections pay task rent only when the target compares "
                    "charged states and no sufficient pointwise invariant exists."
                ),
                (
                    "An exact collision separates observation insufficiency from "
                    "the performance of any learned or fixed decoder."
                ),
                (
                    "The observed edge is a frame-comparison convention, not a "
                    "target-correlated auxiliary feature."
                ),
                (
                    "The fixed analytic solution remains cheaper than TinyLLM, so "
                    "learning requires a function-class prerequisite first."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "Build the smallest connection-conditioned exact-equivariant "
                    "function class, show it contains the analytic transport rule "
                    "and receives a finite downhill task gradient, then decide "
                    "whether a matched acquisition campaign is warranted."
                )
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {"direct_tests": direct},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "source_hashes": result["source_hashes"],
            "artifact_validation": (
                "sealed result/source hashes, exact collision, deterministic fresh "
                "streams, token and connection group laws, local action covariance, "
                "all registered fixed/control gates, and zero-training accounting "
                "revalidated"
            ),
        },
    }


def build_c3_relational_connection_preflight_experiment_results(
    record: Mapping[str, Any], results_path: Path = RESULT_PATH
) -> list[ExperimentResult]:
    del record
    result = _load_and_validate(results_path)
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    experiments = []
    for pair in _cells_by_seed(result):
        primary_pass = _seed_pass(pair)
        metrics: dict[str, float] = {
            "validity": 1.0,
            "relational_connection_gate": float(primary_pass),
            "observed_connection_both_shifts": float(
                all(pair[regime]["observed_connection_pass"] for regime in REGIMES)
            ),
            "specificity_both_shifts": float(
                all(pair[regime]["connection_specificity_pass"] for regime in REGIMES)
            ),
            "pointwise_insufficiency_both_shifts": float(
                all(pair[regime]["pointwise_insufficiency_pass"] for regime in REGIMES)
            ),
            "fresh_examples": 2.0 * 4_096.0,
            "models_instantiated": 0.0,
        }
        for regime in REGIMES:
            cell = pair[regime]
            metrics[f"{regime}_observed_rmse"] = float(
                cell["arms"]["observed_connection"]["scalar"]["rmse"]
            )
            metrics[f"{regime}_observed_accuracy"] = float(
                cell["arms"]["observed_connection"]["task"][
                    "exact_bin_accuracy"
                ]
            )
            metrics[f"{regime}_no_connection_rmse"] = float(
                cell["arms"]["no_connection"]["scalar"]["rmse"]
            )
            metrics[f"{regime}_local_action_error"] = float(
                cell["local_action_prediction_maximum_error"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-relational-connection-preflight-"
                    f"seed{pair['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(primary_pass),
                model_architecture=[3, 8, 7, 1],
                model_parameters=0,
                training_time=0.0,
                model_checkpoint="",
                observations=[
                    f"Fresh seed={pair['seed']} relational connection preflight.",
                    "Observed connection passes and all registered controls fail on both shifts.",
                    "No model, checkpoint, fitted parameter, or optimizer was used.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_relational_connection_preflight_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_relational_connection_preflight_meta_hypothesis(
        results_path
    )
    experiments = build_c3_relational_connection_preflight_experiment_results(
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
            or {
                item.get("hypothesis_id")
                for item in stored.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C3 relational-connection ChromaDB read-back failed")
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
    "build_c3_relational_connection_preflight_experiment_results",
    "build_c3_relational_connection_preflight_meta_hypothesis",
    "store_c3_relational_connection_preflight_meta_hypothesis",
]
