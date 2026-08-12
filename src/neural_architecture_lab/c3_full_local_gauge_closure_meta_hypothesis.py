"""Build and store the full framewise C3 gauge-closure result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-full-local-gauge-closure.v1"
HYPOTHESIS_ID = "tinyllm-c3-local-gauge-invariant-closure-v1"
EVIDENCE_ROLE = "artifact_lineage_no_training_full_local_gauge_causal_contract"
CLASSIFICATION = "pointwise_cubic_quotient_closes_full_local_c3_gauge"
CONFIRMATION_STATUS = "confirmed_full_c3_eight_local_gauge_task_closure"
RESULT_SHA256 = "31f7c7301c889db67e436fba4b8de1909dfbc372573681a9950b5b330f22db35"
RUNNER_SHA256 = "dae960759a2412451f8b15e15b0b6fb479603938a323ea54c86c77c68839005d"
PREREGISTRATION_SHA256 = (
    "0f620df51c2f1d4278a8bbd82f8a6071c3159bdeb5e9420f4e08722c41115f90"
)
REPORT_SHA256 = "e1a237410bd3dfb050d22aadbfdf5b68f8b19a83c2977ae483fdebdbe468718e"
SOURCE_RESULT_SHA256 = (
    "f52ce2103a07086a7118975d69f49b7cbeca01ac0ca7c5a15fd6d2a96fbc51fa"
)
SOURCE_RUNNER_SHA256 = (
    "6a9b1b849c97fc30bef7292ed0bbc097c4829db2920d3abe8f698e7546489944"
)
SOURCE_PREREGISTRATION_SHA256 = (
    "e4629a11cac991b1bd64d641f3276b4517296ee31ac3b9e0a3837e5cb5ce4663"
)
SOURCE_REPORT_SHA256 = (
    "281f375c069fb58b9949b7e2d0c98c895e4bfedf8a9f4e7160204e2dc3bb852b"
)
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_full_local_gauge_closure/"
    "20260811_artifact_audit/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_full_local_gauge_closure.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-full-local-gauge-closure-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-full-local-gauge-closure.md"
)
SOURCE_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_gauge_jump_joint_typed_score/"
    "20260811_preregistered/result.json"
)
SOURCE_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_gauge_jump_joint_typed_score.py"
)
SOURCE_PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score-"
    "preregistration.md"
)
SOURCE_REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score.md"
)
SEEDS = (773, 821, 1003, 1031, 1039)
REGIMES = ("composition", "extrapolation")


@lru_cache(maxsize=128)
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
        "certified_local_group_order": 6561,
        "classification": CLASSIFICATION,
        "compact_connection_model_licensed": False,
        "exact_local_gauge_seed_pass_count": 5,
        "generator_action_count": 16,
        "inherited_fixed_invariant_seed_pass_count": 5,
        "multiple_jump_experiment_licensed": False,
        "numeric_local_gauge_seed_pass_count": 5,
        "required_seed_passes": 4,
        "tinyllm_training_licensed": False,
        "valid": True,
    }


def _expected_accounting() -> dict[str, int]:
    return {
        "arbitrary_local_counterfactual_evaluations": 40_960,
        "checkpoints_loaded": 0,
        "fresh_examples": 0,
        "generator_counterfactual_evaluations": 655_360,
        "models_instantiated": 0,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "reusable_parameter_fits": 0,
        "source_examples_reused": 40_960,
        "target_using_fits": 0,
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
        or sources.get("joint_typed_score_result") != SOURCE_RESULT_SHA256
        or sources.get("joint_typed_score_runner") != SOURCE_RUNNER_SHA256
        or sources.get("joint_typed_score_preregistration")
        != SOURCE_PREREGISTRATION_SHA256
        or sources.get("joint_typed_score_report") != SOURCE_REPORT_SHA256
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 full local-gauge result {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (SOURCE_RESULT_PATH, SOURCE_RESULT_SHA256),
        (SOURCE_RUNNER_PATH, SOURCE_RUNNER_SHA256),
        (SOURCE_PREREGISTRATION_PATH, SOURCE_PREREGISTRATION_SHA256),
        (SOURCE_REPORT_PATH, SOURCE_REPORT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")
    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime) for seed in SEEDS for regime in REGIMES
    }:
        raise ValueError("full local-gauge population changed")
    for cell in cells:
        if (
            cell.get("valid") is not True
            or cell.get("source_hash_match") is not True
            or cell.get("identity_metric_replay") is not True
            or cell.get("exact_algebraic_closure") is not True
            or cell.get("numeric_local_gauge_closure") is not True
            or cell.get("maximum_cubic_carrier_error", math.inf) > 2e-12
            or cell.get("maximum_invariant_forecast_error", math.inf) > 2e-12
            or cell.get("token_group_contracts", {}).get("pass") is not True
            or any(
                measurement.get("exact_cube_integer_error_count") != 0
                or measurement.get("pass") is not True
                for measurement in (
                    *cell.get("generator_measurements", {}).values(),
                    cell.get("arbitrary_local_action", {}),
                )
            )
        ):
            raise ValueError("full local-gauge cell changed")
    if sum(int(cell["total_selector_changes"]) for cell in cells) != 42:
        raise ValueError("descriptive selector-change count changed")
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
        and pair[regime]["exact_algebraic_closure"]
        and pair[regime]["numeric_local_gauge_closure"]
        and pair[regime]["source_fixed_invariant_pass"]
        for regime in REGIMES
    )


def build_c3_full_local_gauge_closure_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _load_and_validate(results_path)
    direct = _cells_by_seed(result)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Pointwise cubic quotient closes the full local C3 gauge",
            "category": "artifact-lineage causal group-action certificate",
            "description": (
                "Sixteen single-frame generators and arbitrary per-example "
                "actions test the existing sufficient invariant decoder under "
                "the complete framewise C3^8 observation gauge."
            ),
            "question": (
                "Can any same-law sequence of hidden gauge jumps alter the "
                "pointwise cubic carrier or its fixed physical task computation?"
            ),
            "prediction": (
                "Exact integer cubes, numeric invariant forecasts, and inherited "
                "task closure remain unchanged under every group generator in at "
                "least four of five source seeds."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": CONFIRMATION_STATUS,
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "ten sealed switching/corruption cells; eight observation frames; "
                "full C3^8 local gauge generated by sixteen nonidentity frame "
                "actions; invariant fixed decoder"
            ),
            "success_criteria": {
                "joint_seed_passes_minimum": 4,
                "exact_cube_integer_error_count": 0,
                "token_group_law_error_count": 0,
                "carrier_error_maximum": 2e-12,
                "forecast_error_maximum": 2e-12,
                "certified_group_order": 6561,
            },
            "subclaims": {
                "exact_pointwise_cubic_local_invariance": "supported_five_of_five",
                "numeric_invariant_forecast_closure": "supported_five_of_five",
                "inherited_fixed_task_closure": "supported_five_of_five",
                "full_local_group_certificate": "supported_order_6561",
                "selector_identity_invariance": "not_required_42_tie_changes",
                "same_law_multiple_jump_experiment": "not_licensed",
                "compact_connection_model": "not_licensed",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "local-gauge",
                "pointwise-invariant",
                "multiple-jumps",
                "group-action",
                "causal-contract",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Exact pointwise cubes and invariant forecasts close all sixteen "
                "generators in every sealed cell; those generators certify all "
                "6,561 elements of C3^8, while source task closure remains 5/5."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "algebraic_identity_plus_complete_generator_and_five_seed_numeric_audit"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": sum(_seed_pass(pair) for pair in direct),
            "failed_direct_experiments": sum(not _seed_pass(pair) for pair in direct),
            "primary_seed_passes": sum(_seed_pass(pair) for pair in direct),
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "A connection is required to compare charged coordinates "
                    "across local gauges, but not to compute this invariant task."
                ),
                (
                    "Any number of same-law suffix jumps lies inside the certified "
                    "local group and cannot change the sufficient cubic sequence."
                ),
                (
                    "Forty-two selector-label changes leave forecasts unchanged, "
                    "again separating latent recovery from causal sufficiency."
                ),
                (
                    "A legitimate connection-learning task must remove pointwise "
                    "invariant sufficiency while preserving identifiability."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "Use an identifiable relational or holonomy target with an "
                    "observed or partially observed edge connection, or an unknown "
                    "group for which no sufficient pointwise invariant is supplied."
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
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_runner_sha256": SOURCE_RUNNER_SHA256,
            "source_preregistration_sha256": SOURCE_PREREGISTRATION_SHA256,
            "source_report_sha256": SOURCE_REPORT_SHA256,
            "artifact_validation": (
                "sealed source hashes and metrics, exact local token group laws, "
                "integer cubic closure, numeric carrier/forecast closure, full "
                "generator set, accounting, and conservative training stop revalidated"
            ),
        },
    }


def build_c3_full_local_gauge_closure_experiment_results(
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
            "full_local_gauge_gate": float(primary_pass),
            "exact_both_shifts": float(
                all(pair[regime]["exact_algebraic_closure"] for regime in REGIMES)
            ),
            "numeric_both_shifts": float(
                all(
                    pair[regime]["numeric_local_gauge_closure"]
                    for regime in REGIMES
                )
            ),
            "certified_group_order": 6561.0,
            "fresh_examples": 0.0,
            "models_instantiated": 0.0,
        }
        for regime in REGIMES:
            cell = pair[regime]
            metrics[f"{regime}_maximum_carrier_error"] = float(
                cell["maximum_cubic_carrier_error"]
            )
            metrics[f"{regime}_maximum_forecast_error"] = float(
                cell["maximum_invariant_forecast_error"]
            )
            metrics[f"{regime}_selector_changes"] = float(
                cell["total_selector_changes"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-full-local-gauge-closure-"
                    f"seed{pair['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(primary_pass),
                model_architecture=[3, 8, 1],
                model_parameters=0,
                training_time=0.0,
                model_checkpoint="",
                observations=[
                    f"Sealed source seed={pair['seed']} local-gauge audit.",
                    "All sixteen C3^8 generators close both shifts.",
                    "No fresh example, fitted model, checkpoint, or optimizer was used.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_full_local_gauge_closure_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_full_local_gauge_closure_meta_hypothesis(results_path)
    experiments = build_c3_full_local_gauge_closure_experiment_results(
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
            raise RuntimeError("C3 full local-gauge ChromaDB read-back failed")
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
    "build_c3_full_local_gauge_closure_experiment_results",
    "build_c3_full_local_gauge_closure_meta_hypothesis",
    "store_c3_full_local_gauge_closure_meta_hypothesis",
]
