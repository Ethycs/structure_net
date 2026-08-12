"""Build and store the C3 constant-acceleration fixed-operator result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-constant-acceleration-fixed-operator.v1"
HYPOTHESIS_ID = "tinyllm-c3-constant-acceleration-fixed-operator-v1"
EVIDENCE_ROLE = "prospective_no_training_dynamics_scope_preflight"
CLASSIFICATION = "fixed_all_frame_degree2_closes_constant_acceleration"
RESULT_SHA256 = "b04a5574efc658ec1ed73f70fa494041ad16c0ae1342423cdde32925c1c7bc53"
RUNNER_SHA256 = "6ea952f386b82b12355c3aa2e9552af6bf73e03e7cd47310fec764ce49d0d5e2"
PREREGISTRATION_SHA256 = (
    "ae4e15e88fc16ec3cd1cc3a52724e042402e80348af6057df5b65b4719c4ee7b"
)
REPORT_SHA256 = "a2d5c6ad412a667ef0abe1cf48a879b7a49cc1477f4e780e2ee26e8f8a3b51b1"
GENERATOR_SHA256 = "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
INTERVAL_SHA256 = "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
FIXED_OPERATOR_RESULT_SHA256 = (
    "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
)
FEATURE_TRAJECTORY_RESULT_SHA256 = (
    "a0ac3315b03aa65df273539a24d8c08f51f12e8ceb702859cc16a282886ddf27"
)
SEEDS = (107, 127, 149, 173, 197)
REGIMES = ("composition", "extrapolation")
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_constant_acceleration_fixed_operator/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_constant_acceleration_fixed_operator.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-constant-acceleration-fixed-operator-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-constant-acceleration-fixed-operator.md"
)
GENERATOR_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_preflight.py"
)
INTERVAL_PATH = Path(
    "experiments/structure_net/tinyllm_joint_physical_scalar_interface.py"
)
FIXED_OPERATOR_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)
FEATURE_TRAJECTORY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_feature_trajectory_causal/"
    "20260811_d6_preregistered/result.json"
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
        "all_frame_degree2_fixed_ceiling_seed_pass_count": 5,
        "all_frame_material_dominance_seed_pass_count": 5,
        "classification": CLASSIFICATION,
        "constant_acceleration_tinyllm_training_licensed": False,
        "constant_speed_fixed_ceiling_seed_pass_count": 0,
        "recent_degree2_fixed_ceiling_seed_pass_count": 5,
        "required_seed_passes": 4,
        "robust_typed_estimator_preflight_licensed": False,
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
        or sources.get("generator") != GENERATOR_SHA256
        or sources.get("interval_likelihood") != INTERVAL_SHA256
        or sources.get("fixed_operator_result") != FIXED_OPERATOR_RESULT_SHA256
        or sources.get("feature_trajectory_result")
        != FEATURE_TRAJECTORY_RESULT_SHA256
        or result.get("accounting")
        != {
            "checkpoints_loaded": 0,
            "models_instantiated": 0,
            "new_evaluation_examples": 40_960,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "target_using_fits": 0,
        }
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 constant-acceleration result {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (GENERATOR_PATH, GENERATOR_SHA256),
        (INTERVAL_PATH, INTERVAL_SHA256),
        (FIXED_OPERATOR_RESULT_PATH, FIXED_OPERATOR_RESULT_SHA256),
        (FEATURE_TRAJECTORY_RESULT_PATH, FEATURE_TRAJECTORY_RESULT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")

    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime) for seed in SEEDS for regime in REGIMES
    }:
        raise ValueError("C3 constant-acceleration cohort population changed")
    for cell in cells:
        operators = cell.get("operators", {})
        if (
            cell.get("valid") is not True
            or cell.get("dataset_deterministic") is not True
            or cell.get("saturation_count") != 0
            or cell.get("shuffle_fixed_points") != 0
            or cell.get("group_contracts", {}).get("pass") is not True
            or operators.get("constant_speed_mean", {}).get("fixed_ceiling_pass")
            is not False
            or any(
                operators.get(arm, {}).get("fixed_ceiling_pass") is not True
                for arm in ("recent_degree2", "all_frame_degree2")
            )
            or cell.get("all_frame_vs_recent", {}).get(
                "material_dominance_pass"
            )
            is not True
            or float(cell.get("all_frame_vs_recent", {}).get("rmse_ratio", 1.0))
            >= 0.50
        ):
            raise ValueError(
                f"invalid C3 constant-acceleration cell "
                f"{cell.get('seed')}/{cell.get('regime')}"
            )
        for arm, operator in operators.items():
            if (
                operator.get("action_pass") is not True
                or operator.get("shuffle_pass") is not True
                or operator.get("shuffled_task_pass") is not False
            ):
                raise ValueError(f"invalid C3 acceleration control {arm}")
        for arm in ("recent_degree2", "all_frame_degree2"):
            if float(cell["continuous_operator_maximum_errors"][arm]) > 1e-12:
                raise ValueError(f"invalid continuous degree-2 operator {arm}")
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


def build_c3_constant_acceleration_fixed_operator_meta_hypothesis(
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
                "A fixed degree-2 C3 group operator closes calibrated constant "
                "acceleration without TinyLLM"
            ),
            "category": "prospective no-training dynamics-scope preflight",
            "description": (
                "Five independent composition/extrapolation replicates compare a "
                "misspecified constant-speed mean, a recent degree-2 recurrence, "
                "and one all-frame degree-2 group-polynomial estimator."
            ),
            "question": (
                "Does constant acceleration create learned temporal work, or does "
                "the known group law still close under fixed finite differences?"
            ),
            "prediction": (
                "The all-frame degree-2 operator reaches the fixed ceiling and "
                "materially dominates the recent operator in at least four seeds."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_fixed_all_frame_degree2_closes_constant_"
                "acceleration_five_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five independent 4096-example cohorts per shift; calibrated "
                "noiseless exact-C3 observations; disjoint velocity and acceleration "
                "ranges; fixed sixteen-bin physical decoder"
            ),
            "success_criteria": {
                "fixed_ceiling_seed_passes_minimum": 4,
                "scalar_rmse_maximum": 0.020,
                "exact_bin_accuracy_minimum": 0.90,
                "material_rmse_ratio_maximum": 0.75,
                "complete_task_gate_required_both_shifts": True,
            },
            "subclaims": {
                "constant_speed_misspecification": "supported_zero_of_five_ceiling",
                "recent_degree2_sufficiency": "supported_five_of_five",
                "all_frame_degree2_sufficiency": "supported_five_of_five",
                "all_frame_material_dominance": "supported_five_of_five",
                "constant_acceleration_tinyllm_training": "not_licensed",
                "polynomial_degree_ladder": "not_licensed_known_law_closes_analytically",
                "unknown_or_piecewise_dynamics": "not_tested",
                "missing_or_corrupted_observations": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "constant-acceleration",
                "group-polynomial",
                "finite-difference",
                "fixed-baseline",
                "no-training",
                "analytic-ceiling",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Both degree-2 operators clear the strong ceiling in every seed, "
                "and the all-frame estimator halves quantization error while the "
                "constant-speed comparator fails."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "prospective_five_of_five_both_shift_degree2_fixed_closure"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 5,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "Known temporal-law order, not nonconstancy alone, determines "
                    "the fixed group-difference operator required for prediction."
                ),
                (
                    "Transporting all increment estimates to the final time before "
                    "circular averaging cuts degree-2 scalar error roughly in half."
                ),
                (
                    "The misspecified constant-speed estimator fails every ceiling "
                    "cell, so the scope change is empirically material."
                ),
                (
                    "A learned continuation has no demonstrated rent while the law "
                    "is known, exact, and finite-order."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A typed learned temporal estimator earns incremental utility "
                    "when dynamics order changes within a sequence and is not given."
                ),
                (
                    "Robust learned estimation becomes useful under observable "
                    "missingness or unobserved outlier corruption after fixed robust "
                    "baselines are exhausted."
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
            "generator_sha256": GENERATOR_SHA256,
            "interval_sha256": INTERVAL_SHA256,
            "fixed_operator_result_sha256": FIXED_OPERATOR_RESULT_SHA256,
            "feature_trajectory_result_sha256": FEATURE_TRAJECTORY_RESULT_SHA256,
            "artifact_validation": (
                "result, runner, preregistration, report, generator, decoder, "
                "predecessor results, ten deterministic cohorts, exact group "
                "actions, continuous degree-2 algebra, target derangements, fixed "
                "ceilings, and locked population classification revalidated"
            ),
        },
    }


def build_c3_constant_acceleration_fixed_operator_experiment_results(
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
            "all_frame_ceiling_both_shifts": 1.0,
            "all_frame_material_both_shifts": 1.0,
            "optimizer_steps": 0.0,
            "models_instantiated": 0.0,
            "target_using_fits": 0.0,
        }
        for regime in REGIMES:
            cell = pair[regime]
            for arm in (
                "constant_speed_mean",
                "recent_degree2",
                "all_frame_degree2",
            ):
                operator = cell["operators"][arm]
                metrics[f"{regime}_{arm}_rmse"] = float(
                    operator["scalar"]["rmse"]
                )
                metrics[f"{regime}_{arm}_accuracy"] = float(
                    operator["task"]["exact_bin_accuracy"]
                )
            metrics[f"{regime}_all_frame_rmse_ratio"] = float(
                cell["all_frame_vs_recent"]["rmse_ratio"]
            )
            metrics[f"{regime}_all_frame_accuracy_delta"] = float(
                cell["all_frame_vs_recent"]["accuracy_delta"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    f"tinyllm-c3-constant-acceleration-fixed-operator-seed"
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
                    f"No-training constant-acceleration seed={pair['seed']}.",
                    "Fixed degree-2 group operator versus misspecified constant-speed mean.",
                    "No model, checkpoint, fit, or changed parameter.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_constant_acceleration_fixed_operator_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_constant_acceleration_fixed_operator_meta_hypothesis(
        results_path
    )
    experiments = build_c3_constant_acceleration_fixed_operator_experiment_results(
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
            raise RuntimeError("C3 acceleration ChromaDB read-back failed")
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
    "build_c3_constant_acceleration_fixed_operator_experiment_results",
    "build_c3_constant_acceleration_fixed_operator_meta_hypothesis",
    "store_c3_constant_acceleration_fixed_operator_meta_hypothesis",
]
