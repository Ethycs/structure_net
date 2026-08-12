"""Build and store the fresh C3 hidden-order corruption corrective result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-hidden-order-corruption-corrective.v1"
HYPOTHESIS_ID = "tinyllm-c3-hidden-order-corruption-corrective-v1"
EVIDENCE_ROLE = "prospective_no_training_corrective_replication"
CLASSIFICATION = "fresh_corrective_confirms_common_robust_nested_law_closure"
RESULT_SHA256 = "bd5af6550e65c0b9048030a8f6d01cf80e06e9f027edb89066d8338bb8b47c2a"
RUNNER_SHA256 = "6d8e408f2347b86440ba6307345b86cad9f2740ee0945c358dea122d1c8d2789"
PREREGISTRATION_SHA256 = (
    "2778eb0e9ade9c0f084e194865df17b2b9ece1558d396d4e9227f3d599e91834"
)
REPORT_SHA256 = "e3cad7cf119397a0d3193fc211708f7a82f0a8fa0e31438c74e7969c039b02b7"
PREDECESSOR_RUNNER_SHA256 = (
    "318cf81497960d37f86b1be58af3d819076e7dc9b620fe2ba73ae215ae0adea7"
)
PREDECESSOR_RESULT_SHA256 = (
    "88e44e76c44d654331ea647ccedd8dad505030a65ab7c4ac241d8b98d98bd02e"
)
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_corrective/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_hidden_order_corruption_corrective.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-hidden-order-corruption-corrective-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-hidden-order-corruption-corrective.md"
)
PREDECESSOR_RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_hidden_order_corruption_fixed_estimator.py"
)
PREDECESSOR_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
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
        "common_robust_endpoint_seed_pass_count": 5,
        "compact_typed_selector_comparison_licensed": False,
        "corrupted_naive_mixture_ceiling_seed_pass_count": 0,
        "materiality_seed_pass_count": 5,
        "oracle_endpoint_seed_pass_count": 5,
        "oracle_fidelity_seed_pass_count": 5,
        "pareto_repair_seed_pass_count": 5,
        "required_seed_passes": 4,
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
        or sources.get("hidden_order_runner") != PREDECESSOR_RUNNER_SHA256
        or sources.get("hidden_order_result") != PREDECESSOR_RESULT_SHA256
        or result.get("accounting")
        != {
            "adaptive_selector_decisions": 0,
            "checkpoints_loaded": 0,
            "closed_form_observation_only_fits": 737_280,
            "fresh_base_examples": 81_920,
            "fresh_corrupted_evaluations": 81_920,
            "law_label_reads_by_primary_estimator": 0,
            "models_instantiated": 0,
            "old_primary_examples_pooled": 0,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "target_using_fits": 0,
        }
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 hidden-order corrective result {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (PREDECESSOR_RUNNER_PATH, PREDECESSOR_RUNNER_SHA256),
        (PREDECESSOR_RESULT_PATH, PREDECESSOR_RESULT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")
    cells = result.get("cells", [])
    identities = {(cell.get("replicate"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (replicate, regime)
        for replicate in range(1, 6)
        for regime in ("composition", "extrapolation")
    }:
        raise ValueError("C3 corrective population changed")
    for cell in cells:
        if (
            cell.get("valid") is not True
            or cell.get("common_robust_endpoint_pass") is not True
            or cell.get("oracle_endpoint_pass") is not True
            or cell.get("materiality_pass") is not True
            or cell.get("pareto_repair_pass") is not True
            or cell.get("oracle_fidelity_pass") is not True
        ):
            raise ValueError("C3 corrective joint cell changed")
        for population in cell.get("populations", {}).values():
            operators = population.get("operators", {})
            if (
                population.get("valid") is not True
                or population.get("corruption_material") is not True
                or population.get("pareto_repair", {}).get("pass") is not True
                or population.get("oracle_fidelity", {}).get("pass") is not True
                or operators.get("clean_law_specific", {}).get(
                    "fixed_ceiling_pass"
                )
                is not True
                or operators.get("corrupted_law_specific", {}).get(
                    "fixed_ceiling_pass"
                )
                is not False
                or operators.get("oracle_drop_one_quadratic", {}).get(
                    "fixed_ceiling_pass"
                )
                is not True
                or operators.get("robust_drop_one_quadratic", {}).get(
                    "fixed_ceiling_pass"
                )
                is not True
                or any(
                    operator.get("action_pass") is not True
                    or operator.get("shuffle_pass") is not True
                    or operator.get("shuffled_task_pass") is not False
                    for operator in operators.values()
                )
            ):
                raise ValueError("C3 corrective population cell changed")
    return result


def _cells_by_replicate(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexed = {
        (int(cell["replicate"]), cell["regime"]): cell
        for cell in result["cells"]
    }
    return [
        {
            "replicate": replicate,
            "composition": indexed[(replicate, "composition")],
            "extrapolation": indexed[(replicate, "extrapolation")],
        }
        for replicate in range(1, 6)
    ]


def build_c3_hidden_order_corruption_corrective_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    direct = _cells_by_replicate(result)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "Fresh corrective confirms common robust closure of nested C3 "
                "laws under one corrupted frame"
            ),
            "category": "prospective no-training corrective replication",
            "description": (
                "Five fresh paired replicates test one law-blind robust quadratic "
                "estimator on speed, acceleration, and their actual mixture with a "
                "pre-existing Pareto nondegradation guard."
            ),
            "question": (
                "Does the frozen common estimator Pareto-repair the nested law "
                "family on fresh data without pooling the inconclusive cohorts?"
            ),
            "prediction": (
                "Common absolute, materiality, Pareto repair, and oracle fidelity "
                "all pass at least four of five fresh paired replicates."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_fresh_common_robust_nested_law_closure_five_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five fresh 4096-example cohorts per law and shift; exact 50:50 "
                "mixture; one unmarked donor-frame substitution; zero predecessor "
                "examples pooled; frozen law-blind estimator"
            ),
            "success_criteria": {
                "joint_seed_passes_minimum": 4,
                "strong_scalar_rmse_maximum": 0.020,
                "strong_exact_bin_accuracy_minimum": 0.90,
                "pareto_rmse_ratio_maximum": 0.50,
                "pareto_accuracy_delta_minimum": -0.005,
                "pareto_cross_entropy_delta_maximum": -0.10,
            },
            "subclaims": {
                "fresh_common_absolute_endpoint": "supported_five_of_five",
                "fresh_oracle_absolute_endpoint": "supported_five_of_five",
                "fresh_corruption_materiality": "supported_five_of_five",
                "fresh_pareto_repair": "supported_five_of_five",
                "fresh_oracle_fidelity": "supported_five_of_five",
                "old_primary_pooling": "zero",
                "original_inconclusive_classification": "unchanged",
                "typed_selector_comparison": "not_licensed",
                "tinyllm_training": "not_licensed",
                "non_nested_or_switching_law": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "hidden-law-order",
                "single-frame-corruption",
                "corrective-replication",
                "robust-estimation",
                "fixed-baseline",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "All four joint corrective endpoints pass both shifts in every "
                "fresh paired replicate, with no old examples or law labels used."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "prospective_fresh_five_of_five_both_shift_corrective_closure"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 5,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "Nested candidate laws need not create model-selection work: one "
                    "maximal typed chart can cover every subfamily."
                ),
                (
                    "The frozen estimator repairs a material observation corruption "
                    "without task tradeoff on fresh composition and extrapolation data."
                ),
                (
                    "A corrective replication can resolve a redundant effect-size "
                    "gate without rewriting or pooling the original result."
                ),
                (
                    "Selector and TinyLLM training remain unjustified until dynamics "
                    "are genuinely non-nested or switch within sequence."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A fixed enumerative or change-point estimator fails before the "
                    "oracle on a non-nested within-sequence switching law."
                ),
                (
                    "A compact typed continuation earns incremental utility only after "
                    "that fixed adaptive ceiling fails on fresh data."
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
            "predecessor_runner_sha256": PREDECESSOR_RUNNER_SHA256,
            "predecessor_result_sha256": PREDECESSOR_RESULT_SHA256,
            "artifact_validation": (
                "result, runner, preregistration, report, predecessor identity, fresh "
                "seed population, deterministic generation, exact action/inclusion, "
                "continuous recovery, shuffles, four joint endpoints, and locked "
                "corrective classification revalidated"
            ),
        },
    }


def build_c3_hidden_order_corruption_corrective_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    experiments = []
    for pair in _cells_by_replicate(result):
        metrics: dict[str, float] = {
            "validity": 1.0,
            "common_endpoint_both_shifts": 1.0,
            "materiality_both_shifts": 1.0,
            "pareto_repair_both_shifts": 1.0,
            "oracle_fidelity_both_shifts": 1.0,
            "old_primary_examples_pooled": 0.0,
            "models_instantiated": 0.0,
        }
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            for population in (
                "constant_speed",
                "constant_acceleration",
                "mixture",
            ):
                value = cell["populations"][population]
                robust = value["operators"]["robust_drop_one_quadratic"]
                metrics[f"{regime}_{population}_robust_rmse"] = float(
                    robust["scalar"]["rmse"]
                )
                metrics[f"{regime}_{population}_robust_accuracy"] = float(
                    robust["task"]["exact_bin_accuracy"]
                )
                metrics[f"{regime}_{population}_repair_rmse_ratio"] = float(
                    value["pareto_repair"]["rmse_ratio"]
                )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-hidden-order-corruption-corrective-"
                    f"replicate{pair['replicate']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=1.0,
                model_architecture=[3, 8, 1],
                model_parameters=0,
                training_time=0.0,
                model_checkpoint="",
                observations=[
                    f"Fresh no-training corrective replicate={pair['replicate']}.",
                    "All common absolute, materiality, Pareto, and oracle gates pass.",
                    "No predecessor example, law selector, model, or optimizer used.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_hidden_order_corruption_corrective_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_hidden_order_corruption_corrective_meta_hypothesis(
        results_path
    )
    experiments = build_c3_hidden_order_corruption_corrective_experiment_results(
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
            raise RuntimeError("C3 corrective ChromaDB read-back failed")
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
    "build_c3_hidden_order_corruption_corrective_experiment_results",
    "build_c3_hidden_order_corruption_corrective_meta_hypothesis",
    "store_c3_hidden_order_corruption_corrective_meta_hypothesis",
]
