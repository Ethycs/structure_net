"""Build and store the C3 hidden-order corruption fixed-estimator result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-hidden-order-corruption-fixed-estimator.v1"
HYPOTHESIS_ID = "tinyllm-c3-hidden-order-corruption-fixed-estimator-v1"
EVIDENCE_ROLE = "prospective_no_training_joint_scope_preflight"
CLASSIFICATION = "inconclusive_hidden_order_corruption_preflight"
RESULT_SHA256 = "88e44e76c44d654331ea647ccedd8dad505030a65ab7c4ac241d8b98d98bd02e"
RUNNER_SHA256 = "318cf81497960d37f86b1be58af3d819076e7dc9b620fe2ba73ae215ae0adea7"
PREREGISTRATION_SHA256 = (
    "5530806f4f71ff72eba9abdcc3001e5ebe072bfb0217891662fc4e748fe95da5"
)
REPORT_SHA256 = "7a5586e91740e6774aa9a84f8f54b1343d0a0375be2ad1416883a25e566ed800"
SPEED_RUNNER_SHA256 = (
    "9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37"
)
SPEED_RESULT_SHA256 = (
    "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
)
CORRUPTION_RUNNER_SHA256 = (
    "8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8"
)
CORRUPTION_RESULT_SHA256 = (
    "59681f2764b988f05b0916965898b87d5b233b2165151fe19e0c97391fe467b9"
)
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_hidden_order_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_hidden_order_corruption_fixed_estimator.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-hidden-order-corruption-fixed-estimator-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-hidden-order-corruption-fixed-estimator.md"
)
SPEED_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_fixed_operator_ceiling.py"
)
SPEED_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)
CORRUPTION_RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_single_frame_corruption_fixed_estimator.py"
)
CORRUPTION_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_single_frame_corruption_fixed_estimator/"
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
        "repair_seed_pass_count": 0,
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
        or sources.get("constant_speed_runner") != SPEED_RUNNER_SHA256
        or sources.get("constant_speed_result") != SPEED_RESULT_SHA256
        or sources.get("single_frame_corruption_runner")
        != CORRUPTION_RUNNER_SHA256
        or sources.get("single_frame_corruption_result")
        != CORRUPTION_RESULT_SHA256
        or result.get("accounting")
        != {
            "adaptive_selector_decisions": 0,
            "checkpoints_loaded": 0,
            "closed_form_observation_only_fits": 737_280,
            "law_label_reads_by_primary_estimator": 0,
            "matched_base_examples_replayed": 81_920,
            "models_instantiated": 0,
            "new_corrupted_evaluations": 81_920,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "target_using_fits": 0,
        }
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 hidden-order result {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (SPEED_RUNNER_PATH, SPEED_RUNNER_SHA256),
        (SPEED_RESULT_PATH, SPEED_RESULT_SHA256),
        (CORRUPTION_RUNNER_PATH, CORRUPTION_RUNNER_SHA256),
        (CORRUPTION_RESULT_PATH, CORRUPTION_RESULT_SHA256),
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
        raise ValueError("C3 hidden-order population changed")
    for cell in cells:
        if (
            cell.get("valid") is not True
            or cell.get("common_robust_endpoint_pass") is not True
            or cell.get("oracle_endpoint_pass") is not True
            or cell.get("materiality_pass") is not True
            or cell.get("repair_pass") is not False
            or cell.get("oracle_fidelity_pass") is not True
        ):
            raise ValueError("C3 hidden-order joint cell changed")
        populations = cell.get("populations", {})
        if set(populations) != {
            "constant_speed",
            "constant_acceleration",
            "mixture",
        }:
            raise ValueError("C3 hidden-order population names changed")
        for name, population in populations.items():
            operators = population.get("operators", {})
            if (
                population.get("valid") is not True
                or population.get("corruption_material") is not True
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
                or population.get("oracle_fidelity", {}).get("pass") is not True
                or any(
                    operator.get("action_pass") is not True
                    or operator.get("shuffle_pass") is not True
                    or operator.get("shuffled_task_pass") is not False
                    for operator in operators.values()
                )
            ):
                raise ValueError(f"C3 hidden-order {name} cell changed")
        speed_population = populations["constant_speed"]
        if (
            speed_population.get("robust_repair", {}).get("pass") is not False
            or float(
                speed_population.get("robust_repair", {}).get(
                    "accuracy_delta", 1.0
                )
            )
            >= 0.20
            or float(
                speed_population.get("robust_repair", {}).get("rmse_ratio", 1.0)
            )
            >= 0.02
            or populations["constant_acceleration"].get("robust_repair", {}).get(
                "pass"
            )
            is not True
            or populations["mixture"].get("robust_repair", {}).get("pass")
            is not True
        ):
            raise ValueError("C3 hidden-order registered repair failure changed")
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


def build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
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
                "One robust quadratic estimator closes hidden degree-1/degree-2 "
                "dynamics plus one corrupted frame"
            ),
            "category": "prospective no-training joint-scope preflight",
            "description": (
                "Five paired two-shift replicates apply one law-blind robust "
                "quadratic estimator to constant speed, constant acceleration, "
                "and their actual 50:50 mixture."
            ),
            "question": (
                "Does hidden nested law order plus one unmarked corrupted frame "
                "exceed the common fixed estimator?"
            ),
            "prediction": (
                "Common absolute endpoint, materiality, repair, and oracle fidelity "
                "all pass at least four of five paired replicates."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "inconclusive_registered_speed_repair_accuracy_effect_size_"
                "zero_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five paired 4096-example cohorts per law and shift; one hidden "
                "donor-frame substitution; exact 50:50 nested-law mixture; common "
                "law-blind robust quadratic estimator"
            ),
            "success_criteria": {
                "joint_seed_passes_minimum": 4,
                "common_scalar_rmse_maximum": 0.020,
                "common_exact_bin_accuracy_minimum": 0.90,
                "repair_accuracy_delta_minimum": 0.20,
                "all_populations_and_both_shifts_required": True,
            },
            "subclaims": {
                "degree_one_nested_in_degree_two_chart": "supported_exactly",
                "common_absolute_endpoint": "supported_five_of_five",
                "corruption_materiality": "supported_five_of_five",
                "oracle_fidelity": "supported_five_of_five",
                "full_registered_repair": "not_supported_zero_of_five",
                "failure_localization": (
                    "constant_speed_accuracy_gain_below_point_two_only"
                ),
                "typed_selector_comparison": "not_licensed",
                "tinyllm_training": "not_licensed",
                "non_nested_or_switching_law": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "hidden-law-order",
                "single-frame-corruption",
                "nested-function-class",
                "robust-estimation",
                "no-training",
                "inconclusive",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The common absolute endpoint, materiality, and oracle fidelity "
                "pass 5/5, but the preregistered all-population repair conjunction "
                "passes 0/5 because speed accuracy gains are below 0.20."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "prospective_valid_inconclusive_registered_joint_gate"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 0,
            "failed_direct_experiments": 5,
            "primary_seed_passes": 0,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "The proposed degree-1/degree-2 selector is algebraically "
                    "unnecessary because both families share one quadratic chart."
                ),
                (
                    "One law-blind estimator clears every absolute task ceiling and "
                    "matches the oracle across both shifts."
                ),
                (
                    "A borrowed absolute accuracy-gain guard can fail even when the "
                    "candidate repairs RMSE by more than 98 percent and reaches the "
                    "strong endpoint."
                ),
                (
                    "Post-outcome gate diagnosis does not authorize reclassification, "
                    "selector training, or TinyLLM training."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A fresh corrective replication with a nonredundant Pareto repair "
                    "guard confirms common fixed closure of the nested family."
                ),
                (
                    "A fixed enumerative estimator fails before the oracle on a "
                    "genuinely non-nested or within-sequence switching law."
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
            "constant_speed_runner_sha256": SPEED_RUNNER_SHA256,
            "constant_speed_result_sha256": SPEED_RESULT_SHA256,
            "corruption_runner_sha256": CORRUPTION_RUNNER_SHA256,
            "corruption_result_sha256": CORRUPTION_RESULT_SHA256,
            "artifact_validation": (
                "result, runner, preregistration, report, predecessor hashes, "
                "twenty base cohorts, actual mixtures, common endpoints, exact "
                "inclusion, action/shuffle controls, and locked inconclusive "
                "classification revalidated"
            ),
        },
    }


def build_c3_hidden_order_corruption_fixed_estimator_experiment_results(
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
            "oracle_fidelity_both_shifts": 1.0,
            "registered_repair_both_shifts": 0.0,
            "optimizer_steps": 0.0,
            "models_instantiated": 0.0,
            "law_label_reads": 0.0,
        }
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            for population in (
                "constant_speed",
                "constant_acceleration",
                "mixture",
            ):
                record_population = cell["populations"][population]
                robust = record_population["operators"][
                    "robust_drop_one_quadratic"
                ]
                metrics[f"{regime}_{population}_robust_rmse"] = float(
                    robust["scalar"]["rmse"]
                )
                metrics[f"{regime}_{population}_robust_accuracy"] = float(
                    robust["task"]["exact_bin_accuracy"]
                )
            metrics[f"{regime}_speed_repair_accuracy_delta"] = float(
                cell["populations"]["constant_speed"]["robust_repair"][
                    "accuracy_delta"
                ]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-hidden-order-corruption-fixed-estimator-"
                    f"replicate{pair['replicate']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=0.0,
                model_architecture=[3, 8, 1],
                model_parameters=0,
                training_time=0.0,
                model_checkpoint="",
                observations=[
                    f"No-training hidden-order replicate={pair['replicate']}.",
                    "Common endpoint passes; registered speed repair effect size fails.",
                    "No law selector, model, checkpoint, optimizer, or target fit.",
                ],
                anomalies=[
                    "Registered all-population repair gate fails solely on speed accuracy delta <0.20."
                ],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis(
        results_path
    )
    experiments = (
        build_c3_hidden_order_corruption_fixed_estimator_experiment_results(
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
            raise RuntimeError("C3 hidden-order ChromaDB read-back failed")
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
    "build_c3_hidden_order_corruption_fixed_estimator_experiment_results",
    "build_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis",
    "store_c3_hidden_order_corruption_fixed_estimator_meta_hypothesis",
]
