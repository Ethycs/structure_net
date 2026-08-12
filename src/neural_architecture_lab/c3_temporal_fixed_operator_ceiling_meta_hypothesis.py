"""Build and store the C3 temporal fixed-operator ceiling result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-temporal-fixed-operator-ceiling.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-fixed-operator-ceiling-v1"
EVIDENCE_ROLE = "prospective_no_training_group_operator_comparison"
CLASSIFICATION = "fixed_multistep_group_operator_dominates_last_step"
RESULT_SHA256 = "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
RUNNER_SHA256 = "9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37"
PREREGISTRATION_SHA256 = (
    "07d54cb5b4d65b080fe59a5e06d0853ee59a7d0132fc04faa68f303775a2f8bf"
)
REPORT_SHA256 = "0cc4561f203eb5a258e768f07e05cc772a6ad46c992681673385fcf7d4351be0"
GENERATOR_SHA256 = "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
INTERVAL_SHA256 = "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
SENSOR_CAMPAIGN_SHA256 = (
    "4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012"
)
MECHANISM_RESULT_SHA256 = (
    "c3dbfecd7a6381c2129e4d99f135557f003ad8a225fa2b4d3f4fa0cb429f669b"
)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_fixed_operator_ceiling.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-c3-temporal-fixed-operator-ceiling.md"
)
GENERATOR_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_preflight.py"
)
INTERVAL_PATH = Path(
    "experiments/structure_net/tinyllm_joint_physical_scalar_interface.py"
)
SENSOR_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_only/"
    "20260811_preregistered/campaign_results.json"
)
MECHANISM_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_mechanism/"
    "20260811_preregistered/campaign_results.json"
)


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


def _result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    aggregates = result.get("aggregates", {})
    accounting = result.get("accounting", {})
    sources = result.get("source_hashes", {})
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role") != EVIDENCE_ROLE
        or sources.get("runner") != RUNNER_SHA256
        or sources.get("preregistration") != PREREGISTRATION_SHA256
        or sources.get("generator") != GENERATOR_SHA256
        or sources.get("interval_likelihood") != INTERVAL_SHA256
        or sources.get("sensor_campaign") != SENSOR_CAMPAIGN_SHA256
        or sources.get("sensor_mechanism_result") != MECHANISM_RESULT_SHA256
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("valid") is not True
        or aggregates.get("material_dominance_seed_pass_count") != 5
        or aggregates.get("baseline_ceiling_seed_pass_count") != 5
        or aggregates.get("same_task_unrestricted_tinyllm_training_licensed")
        is not False
        or aggregates.get("typed_residual_function_class_preflight_licensed")
        is not False
        or accounting
        != {
            "checkpoints_loaded": 0,
            "new_evaluation_examples": 40960,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "target_using_fits": 0,
            "tinyllm_models_instantiated": 0,
        }
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 temporal fixed-operator result {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (GENERATOR_PATH, GENERATOR_SHA256),
        (INTERVAL_PATH, INTERVAL_SHA256),
        (SENSOR_CAMPAIGN_PATH, SENSOR_CAMPAIGN_SHA256),
        (MECHANISM_RESULT_PATH, MECHANISM_RESULT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")

    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime) for seed in SEEDS for regime in REGIMES
    }:
        raise ValueError("C3 fixed-operator cohort population changed")
    for cell in cells:
        if (
            cell.get("valid") is not True
            or cell.get("dataset_deterministic") is not True
            or cell.get("saturation_count") != 0
            or cell.get("shuffle_fixed_points") != 0
            or cell.get("comparison", {}).get("material_dominance_pass")
            is not True
            or float(cell.get("comparison", {}).get("rmse_ratio", 1.0)) > 0.75
            or cell.get("baseline_ceiling_pass") is not True
            or max(cell.get("continuous_positive_control_maximum_errors", {}).values())
            > 1e-12
        ):
            raise ValueError(
                f"invalid C3 fixed-operator cell {cell.get('seed')}/"
                f"{cell.get('regime')}"
            )
        for operator in cell.get("operators", {}).values():
            if (
                operator.get("task_pass") is not True
                or operator.get("action_pass") is not True
                or operator.get("shuffle_pass") is not True
            ):
                raise ValueError("invalid fixed-operator cell control")
    return result


def _cells_by_seed(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexed = {
        (int(cell["seed"]), cell["regime"]): cell
        for cell in result["cells"]
    }
    return [
        {
            "seed": seed,
            "composition": indexed[(seed, "composition")],
            "extrapolation": indexed[(seed, "extrapolation")],
        }
        for seed in SEEDS
    ]


def build_c3_temporal_fixed_operator_ceiling_meta_hypothesis(
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
                "A fixed all-frame C3 group operator exhausts the current "
                "same-task temporal continuation opportunity"
            ),
            "category": "prospective no-training group-operator comparison",
            "description": (
                "Five independent composition/extrapolation replicates compare "
                "the registered last increment with one fixed circular mean of all "
                "seven observable group increments."
            ),
            "question": (
                "Does the noiseless sequence leave work for a learned continuation, "
                "or can a fixed all-frame group statistic consume the redundancy?"
            ),
            "prediction": (
                "If the all-increment operator materially dominates in at least "
                "four seeds, use it and close same-task TinyLLM training."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_fixed_multistep_group_operator_dominance_five_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five independent 4096-example replicates under both registered "
                "shifts; exact analytic C3 carrier; last-increment versus one fixed "
                "unweighted all-increment circular mean; fixed interval decoder"
            ),
            "success_criteria": {
                "material_dominance_seed_passes_minimum": 4,
                "rmse_ratio_maximum": 0.75,
                "accuracy_delta_minimum": -0.005,
                "cross_entropy_delta_maximum": 0.005,
                "complete_task_gate_required_both_shifts": True,
            },
            "subclaims": {
                "fixed_all_increment_dominance": "supported_five_of_five",
                "temporal_redundancy_is_observably_useful": (
                    "supported_both_shifts_five_of_five"
                ),
                "same_task_unrestricted_tinyllm_training": "not_licensed",
                "typed_residual_function_class_preflight": "not_licensed",
                "tinyllm_incremental_utility_current_task": (
                    "not_supported_known_fixed_operator_is_better_and_cheaper"
                ),
                "fixed_operator_global_optimality": "not_tested",
                "nonconstant_or_corrupted_dynamics": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "temporal-operator",
                "circular-mean",
                "fixed-baseline",
                "no-training",
                "prospective",
                "negative-training-decision",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "The single preregistered all-increment circular mean cuts scalar "
                "RMSE roughly in half and improves exact-bin accuracy in every "
                "seed and shift without fitting or training."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "prospective_five_of_five_both_shift_fixed_operator_dominance"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 5,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "Seven observable copies of the constant C3 increment provide "
                    "useful quantization-noise reduction through a circular mean."
                ),
                (
                    "The fixed operator raises mean exact-bin accuracy to .9778 on "
                    "composition and .9804 on extrapolation."
                ),
                (
                    "The earlier unrestricted continuation failed to exploit a "
                    "simple symmetry-typed statistic that needs no learning."
                ),
                (
                    "Current-task continuation retraining has no demonstrated rent; "
                    "the observation or dynamics must change before learning."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A learned typed temporal estimator becomes useful under "
                    "nonconstant speed, missing frames, or corrupted increments."
                ),
                (
                    "The same all-increment group estimator transfers to another "
                    "identifiable cyclic task with a declared temporal law."
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
            "sensor_campaign_sha256": SENSOR_CAMPAIGN_SHA256,
            "mechanism_result_sha256": MECHANISM_RESULT_SHA256,
            "artifact_validation": (
                "result, runner, preregistration, report, generator, decoder, both "
                "predecessor results, ten deterministic data hashes, continuous "
                "algebra, deck actions, shuffled targets, complete task gates, and "
                "locked population classification revalidated"
            ),
        },
    }


def build_c3_temporal_fixed_operator_ceiling_experiment_results(
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
            "material_dominance_both_shifts": 1.0,
            "optimizer_steps": 0.0,
            "tinyllm_models_instantiated": 0.0,
            "target_using_fits": 0.0,
        }
        for regime in REGIMES:
            cell = pair[regime]
            baseline = cell["operators"]["last_increment"]
            candidate = cell["operators"]["mean_increment"]
            metrics[f"{regime}_rmse_ratio"] = float(
                cell["comparison"]["rmse_ratio"]
            )
            metrics[f"{regime}_last_accuracy"] = float(
                baseline["task"]["exact_bin_accuracy"]
            )
            metrics[f"{regime}_mean_accuracy"] = float(
                candidate["task"]["exact_bin_accuracy"]
            )
            metrics[f"{regime}_accuracy_delta"] = float(
                cell["comparison"]["accuracy_delta"]
            )
            metrics[f"{regime}_cross_entropy_delta"] = float(
                cell["comparison"]["cross_entropy_delta"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    f"tinyllm-c3-temporal-fixed-operator-ceiling-seed{pair['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=1.0,
                model_architecture=[3, 8, 1],
                model_parameters=0,
                training_time=0.0,
                model_checkpoint="",
                observations=[
                    f"No-training fixed C3 group-operator seed={pair['seed']}.",
                    "One preregistered all-increment circular mean versus last increment.",
                    "No checkpoint, fit, parameter, or TinyLLM instance.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_temporal_fixed_operator_ceiling_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_temporal_fixed_operator_ceiling_meta_hypothesis(results_path)
    experiments = build_c3_temporal_fixed_operator_ceiling_experiment_results(
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
            raise RuntimeError("C3 fixed-operator ChromaDB read-back failed")
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
    "build_c3_temporal_fixed_operator_ceiling_experiment_results",
    "build_c3_temporal_fixed_operator_ceiling_meta_hypothesis",
    "store_c3_temporal_fixed_operator_ceiling_meta_hypothesis",
]
