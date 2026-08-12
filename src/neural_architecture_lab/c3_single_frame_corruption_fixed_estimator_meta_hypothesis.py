"""Build and store the C3 single-frame-corruption fixed-estimator result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-single-frame-corruption-fixed-estimator.v1"
HYPOTHESIS_ID = "tinyllm-c3-single-frame-corruption-fixed-estimator-v1"
EVIDENCE_ROLE = "prospective_no_training_observation_scope_preflight"
CLASSIFICATION = "fixed_robust_estimator_closes_single_frame_corruption"
RESULT_SHA256 = "59681f2764b988f05b0916965898b87d5b233b2165151fe19e0c97391fe467b9"
RUNNER_SHA256 = "8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8"
PREREGISTRATION_SHA256 = (
    "29b47e522fc46d5993cfba811eeba2d6bd2cc8eca189a1f6205c422f159b572e"
)
REPORT_SHA256 = "eb971ce111eedaff85fb6d7601bb221219af3c3aa7ea86020e542c81549e7d58"
ACCELERATION_RUNNER_SHA256 = (
    "6ea952f386b82b12355c3aa2e9552af6bf73e03e7cd47310fec764ce49d0d5e2"
)
ACCELERATION_RESULT_SHA256 = (
    "b04a5574efc658ec1ed73f70fa494041ad16c0ae1342423cdde32925c1c7bc53"
)
GENERATOR_SHA256 = "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
INTERVAL_SHA256 = "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
SEEDS = (107, 127, 149, 173, 197)
REGIMES = ("composition", "extrapolation")
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_single_frame_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_single_frame_corruption_fixed_estimator.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-single-frame-corruption-fixed-estimator-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-single-frame-corruption-fixed-estimator.md"
)
ACCELERATION_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_constant_acceleration_fixed_operator.py"
)
ACCELERATION_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_constant_acceleration_fixed_operator/"
    "20260811_preregistered/result.json"
)
GENERATOR_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_preflight.py"
)
INTERVAL_PATH = Path(
    "experiments/structure_net/tinyllm_joint_physical_scalar_interface.py"
)


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
        "classification": CLASSIFICATION,
        "clean_fixed_ceiling_seed_pass_count": 5,
        "compact_robust_sensor_comparison_licensed": False,
        "corrupted_naive_fixed_ceiling_seed_pass_count": 0,
        "corruption_material_seed_pass_count": 5,
        "oracle_fidelity_seed_pass_count": 5,
        "oracle_fixed_ceiling_seed_pass_count": 5,
        "required_seed_passes": 4,
        "robust_fixed_ceiling_seed_pass_count": 5,
        "robust_repair_seed_pass_count": 5,
        "tinyllm_training_licensed": False,
        "valid": True,
    }


def _result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    sources = result.get("source_hashes", {})
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("aggregates") != _expected_aggregates()
        or sources.get("runner") != RUNNER_SHA256
        or sources.get("preregistration") != PREREGISTRATION_SHA256
        or sources.get("constant_acceleration_runner")
        != ACCELERATION_RUNNER_SHA256
        or sources.get("constant_acceleration_result")
        != ACCELERATION_RESULT_SHA256
        or sources.get("generator") != GENERATOR_SHA256
        or sources.get("interval_likelihood") != INTERVAL_SHA256
        or result.get("accounting")
        != {
            "checkpoints_loaded": 0,
            "closed_form_observation_only_fits": 368_640,
            "matched_base_examples_replayed": 40_960,
            "models_instantiated": 0,
            "new_base_evaluation_examples": 0,
            "new_corrupted_evaluations": 40_960,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "target_using_fits": 0,
        }
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 single-frame-corruption result {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (ACCELERATION_RUNNER_PATH, ACCELERATION_RUNNER_SHA256),
        (ACCELERATION_RESULT_PATH, ACCELERATION_RESULT_SHA256),
        (GENERATOR_PATH, GENERATOR_SHA256),
        (INTERVAL_PATH, INTERVAL_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")

    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime) for seed in SEEDS for regime in REGIMES
    }:
        raise ValueError("C3 single-frame-corruption population changed")
    for cell in cells:
        operators = cell.get("operators", {})
        if (
            cell.get("valid") is not True
            or cell.get("base_dataset_hash_replayed") is not True
            or cell.get("corruption_deterministic") is not True
            or cell.get("donor_fixed_points") != 0
            or int(cell.get("minimum_frame_count", 0)) < 400
            or float(cell.get("minimum_clean_chart_margin", 0.0)) < 0.20
            or cell.get("continuous_corruption_index_recovery_count") != 4_096
            or sum(cell.get("corruption_equivariance_token_errors", {}).values())
            != 0
            or cell.get("corruption_material") is not True
            or operators.get("clean_all_frame_degree2", {}).get(
                "fixed_ceiling_pass"
            )
            is not True
            or operators.get("corrupted_all_frame_degree2", {}).get(
                "fixed_ceiling_pass"
            )
            is not False
            or any(
                operators.get(arm, {}).get("fixed_ceiling_pass") is not True
                for arm in (
                    "oracle_drop_one_quadratic",
                    "robust_drop_one_quadratic",
                )
            )
            or cell.get("robust_repair", {}).get("pass") is not True
            or cell.get("oracle_fidelity", {}).get("pass") is not True
            or float(cell.get("quantized_corruption_index_recovery_rate", 0.0))
            < 0.99
        ):
            raise ValueError(
                f"invalid C3 corruption cell {cell.get('seed')}/{cell.get('regime')}"
            )
        if any(
            operator.get("action_pass") is not True
            or operator.get("shuffle_pass") is not True
            or operator.get("shuffled_task_pass") is not False
            for operator in operators.values()
        ):
            raise ValueError("invalid C3 corruption action/shuffle control")
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


def build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    direct = _cells_by_seed(result)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "A fixed robust C3 estimator closes one unmarked frame "
                "substitution without TinyLLM"
            ),
            "category": "prospective no-training observation-scope preflight",
            "description": (
                "Five matched composition/extrapolation replicates compare the "
                "clean group operator, its corrupted naive use, oracle deletion, "
                "and exhaustive target-free robust deletion."
            ),
            "question": (
                "Does one unmarked same-time donor-frame substitution create "
                "learned temporal work under known constant acceleration?"
            ),
            "prediction": (
                "The corruption is material, while robust deletion reaches the "
                "fixed ceiling and remains oracle-faithful in at least four seeds."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_fixed_robust_estimator_closes_single_frame_"
                "corruption_five_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five matched 4096-example cohorts per shift; exactly one "
                "unmarked marginally matched donor-frame substitution; known "
                "constant acceleration; calibrated exact-C3 observations"
            ),
            "success_criteria": {
                "fixed_ceiling_seed_passes_minimum": 4,
                "scalar_rmse_maximum": 0.020,
                "exact_bin_accuracy_minimum": 0.90,
                "material_repair_required_both_shifts": True,
                "oracle_fidelity_required_both_shifts": True,
            },
            "subclaims": {
                "single_frame_corruption_materiality": "supported_five_of_five",
                "oracle_recoverability": "supported_five_of_five",
                "robust_fixed_sufficiency": "supported_five_of_five",
                "robust_oracle_fidelity": "supported_five_of_five",
                "tinyllm_training": "not_licensed",
                "compact_learned_sensor_comparison": "not_licensed",
                "unknown_or_switching_law": "not_tested",
                "multiple_or_state_dependent_corruptions": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "single-frame-corruption",
                "robust-estimation",
                "leave-one-out",
                "fixed-baseline",
                "no-training",
                "analytic-ceiling",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "The unprotected operator fails every seed, while robust deletion "
                "passes every ceiling, repair, and oracle-fidelity population gate."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "prospective_five_of_five_both_shift_robust_fixed_closure"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 5,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "A severe naive-operator failure does not establish learned-model "
                    "rent until a robust fixed estimator is tested."
                ),
                (
                    "Known group-polynomial law plus seven redundant clean frames "
                    "makes one unmarked gross outlier identifiable without labels."
                ),
                (
                    "Robust deletion matches the oracle under both composition and "
                    "outside-range extrapolation."
                ),
                (
                    "Observation uncertainty alone remains analytically closed in "
                    "this registered single-corruption regime."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A compact typed selector earns rent only when both temporal-law "
                    "order and observation integrity are unknown within sequence."
                ),
                (
                    "State-dependent or stochastic corruption breaks fixed robust "
                    "closure before it breaks oracle recoverability."
                ),
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
            "constant_acceleration_runner_sha256": ACCELERATION_RUNNER_SHA256,
            "constant_acceleration_result_sha256": ACCELERATION_RESULT_SHA256,
            "generator_sha256": GENERATOR_SHA256,
            "interval_sha256": INTERVAL_SHA256,
            "artifact_validation": (
                "result, runner, preregistration, report, predecessor artifacts, "
                "ten matched cohort hashes, deterministic corruption, exact deck "
                "equivariance, continuous recovery, shuffled targets, fixed "
                "ceilings, and locked population classification revalidated"
            ),
        },
    }


def build_c3_single_frame_corruption_fixed_estimator_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    experiments = []
    for pair in _cells_by_seed(result):
        metrics: dict[str, float] = {
            "validity": 1.0,
            "corruption_material_both_shifts": 1.0,
            "robust_ceiling_both_shifts": 1.0,
            "robust_repair_both_shifts": 1.0,
            "oracle_fidelity_both_shifts": 1.0,
            "optimizer_steps": 0.0,
            "models_instantiated": 0.0,
            "target_using_fits": 0.0,
        }
        for regime in REGIMES:
            cell = pair[regime]
            for arm in (
                "corrupted_all_frame_degree2",
                "oracle_drop_one_quadratic",
                "robust_drop_one_quadratic",
            ):
                operator = cell["operators"][arm]
                metrics[f"{regime}_{arm}_rmse"] = float(
                    operator["scalar"]["rmse"]
                )
                metrics[f"{regime}_{arm}_accuracy"] = float(
                    operator["task"]["exact_bin_accuracy"]
                )
            metrics[f"{regime}_robust_repair_rmse_ratio"] = float(
                cell["robust_repair"]["rmse_ratio"]
            )
            metrics[f"{regime}_corruption_index_recovery_rate"] = float(
                cell["quantized_corruption_index_recovery_rate"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    f"tinyllm-c3-single-frame-corruption-fixed-estimator-seed"
                    f"{pair['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=1.0,
                model_architecture=[3, 8, 1],
                model_parameters=0,
                training_time=0.0,
                model_checkpoint="",
                observations=[
                    f"No-training single-frame-corruption seed={pair['seed']}.",
                    "Robust target-free deletion versus naive and oracle controls.",
                    "No model, checkpoint, optimizer step, or reusable fitted parameter.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
        results_path
    )
    experiments = (
        build_c3_single_frame_corruption_fixed_estimator_experiment_results(
            record, results_path
        )
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
            raise RuntimeError("C3 corruption ChromaDB read-back failed")
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
    "build_c3_single_frame_corruption_fixed_estimator_experiment_results",
    "build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis",
    "store_c3_single_frame_corruption_fixed_estimator_meta_hypothesis",
]
