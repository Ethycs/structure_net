"""Build and store the C3 switching-corruption fixed-decoder result."""

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
EXPERIMENT_SCHEMA = (
    "nal.tinyllm-c3-switching-corruption-fixed-decoder-corrective.v1"
)
HYPOTHESIS_ID = "tinyllm-c3-switching-corruption-fixed-decoder-corrective-v1"
EVIDENCE_ROLE = "prospective_fresh_cohort_numerical_corrective"
CLASSIFICATION = "recoverable_switching_exceeds_fixed_change_point_decoder"
CONFIRMATION_STATUS = (
    "not_confirmed_fixed_decoder_full_gate_one_of_five_"
    "oracle_recoverable_five_of_five"
)
RESULT_SHA256 = "20d40a54ce42a904766cab9eb533f6e2baccff79f75a2703e6bd57ff9638675b"
RUNNER_SHA256 = "452b0fa8f8ff54ddb0afa98e32eef9dc38da9f240eccf6a867386d9b65197939"
PREREGISTRATION_SHA256 = (
    "46e40a526c4adf71410beff4d7333da97b4829c2e03f14923744508da602c828"
)
REPORT_SHA256 = "bb082ec3965d5e8badb3e08d645a0fb958990b8c65667f2da8440cbe6f02295a"
INVALID_RESULT_SHA256 = (
    "2dd8f23a2eb7bd5ad7e2c224ee8c08201f648ce905e2fd7d806d7d676badf20c"
)
INVALID_RUNNER_SHA256 = (
    "5ebe462c27989e0f09f38bba8ee0e885ea5fb76190bc86c1d7143d79376e2128"
)
ORIGINAL_PREREGISTRATION_SHA256 = (
    "d7e7b0cd3774a0b661e331c0c6bf56631734152e05fdc156584952f56d6b6ee2"
)
RESULT_PATH = Path(
    "data/experiments/"
    "tinyllm_c3_switching_corruption_fixed_decoder_corrective/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_switching_corruption_fixed_decoder_corrective.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-"
    "corrective-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder.md"
)
INVALID_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_switching_corruption_fixed_decoder/"
    "20260811_preregistered/result.json"
)
INVALID_RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_switching_corruption_fixed_decoder.py"
)
ORIGINAL_PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-switching-corruption-fixed-decoder-"
    "preregistration.md"
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
        "clean_global_quadratic_seed_pass_count": 0,
        "clean_known_switch_seed_pass_count": 5,
        "compact_typed_continuation_comparison_licensed": True,
        "corrupted_global_quadratic_seed_pass_count": 0,
        "corrupted_known_switch_no_drop_seed_pass_count": 0,
        "corruption_materiality_seed_pass_count": 5,
        "dynamics_materiality_seed_pass_count": 5,
        "fixed_switch_drop_seed_pass_count": 3,
        "identifiability_contract_pass": True,
        "late_support_target_identifiable_under_one_corruption": False,
        "oracle_fidelity_seed_pass_count": 1,
        "oracle_switch_drop_seed_pass_count": 5,
        "pareto_repair_seed_pass_count": 3,
        "primary_support_one_corruption_correctable": True,
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
        or sources.get("invalid_result") != INVALID_RESULT_SHA256
        or sources.get("invalid_runner") != INVALID_RUNNER_SHA256
        or sources.get("original_preregistration")
        != ORIGINAL_PREREGISTRATION_SHA256
        or result.get("accounting")
        != {
            "checkpoints_loaded": 0,
            "continuous_validation_candidate_fits": 983_040,
            "fresh_base_examples": 40_960,
            "fresh_corrupted_evaluations": 40_960,
            "invalid_primary_examples_pooled": 0,
            "law_label_reads_by_fixed_decoder": 0,
            "models_instantiated": 0,
            "observation_only_candidate_fits": 1_720_320,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "reusable_parameter_fits": 0,
            "target_using_fits": 0,
        }
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 switching-corruption result {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (INVALID_RESULT_PATH, INVALID_RESULT_SHA256),
        (INVALID_RUNNER_PATH, INVALID_RUNNER_SHA256),
        (ORIGINAL_PREREGISTRATION_PATH, ORIGINAL_PREREGISTRATION_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")
    audit = result.get("identifiability_audit", {})
    if (
        audit.get("pass") is not True
        or audit.get("late_inclusive_support", {}).get(
            "minimum_future_phase_code_distance"
        )
        != 2
        or audit.get("primary_support", {}).get(
            "minimum_future_phase_code_distance"
        )
        != 3
        or audit.get("explicit_late_collision", {}).get(
            "token_collision_error_count"
        )
        != 0
        or audit.get("explicit_late_collision", {}).get(
            "absolute_target_separation"
        )
        < 0.25
    ):
        raise ValueError("switching identifiability audit changed")
    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime)
        for seed in (401, 419, 433, 449, 467)
        for regime in ("composition", "extrapolation")
    }:
        raise ValueError("switching corrective population changed")
    for cell in cells:
        if (
            cell.get("valid") is not True
            or cell.get("dynamics_material") is not True
            or cell.get("corruption_material") is not True
            or cell.get("continuous_future_equivalent_count") != 4_096
            or cell.get("stabilization_displacements", {}).get("maximum")
            > 1e-12
            or any(
                operator.get("action_pass") is not True
                or operator.get("shuffle_pass") is not True
                or operator.get("shuffled_task_pass") is not False
                for operator in cell.get("operators", {}).values()
            )
            or cell.get("operators", {}).get("oracle_switch_drop", {}).get(
                "fixed_ceiling_pass"
            )
            is not True
        ):
            raise ValueError("switching corrective cell changed")
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
        for seed in (401, 419, 433, 449, 467)
    ]


def _seed_primary_pass(pair: Mapping[str, Any]) -> bool:
    cells = (pair["composition"], pair["extrapolation"])
    return all(
        cell["dynamics_material"]
        and cell["corruption_material"]
        and cell["operators"]["oracle_switch_drop"]["fixed_ceiling_pass"]
        and cell["operators"]["fixed_switch_drop"]["fixed_ceiling_pass"]
        and cell["pareto_repair"]["pass"]
        and cell["oracle_fidelity"]["pass"]
        for cell in cells
    )


def build_c3_switching_corruption_fixed_decoder_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    direct = _cells_by_seed(result)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Fixed decoding of identifiable C3 switching dynamics under corruption",
            "category": "prospective no-training identifiability and fixed-ceiling preflight",
            "description": (
                "Exact code-distance analysis and five fresh corrective replicates "
                "test whether exhaustive switch/deletion decoding closes a "
                "within-sequence velocity change after one corrupted frame."
            ),
            "question": (
                "After repairing late-switch non-identifiability, does a fixed "
                "change-point decoder reach the oracle ceiling?"
            ),
            "prediction": (
                "Exact primary code distance is three and the fixed decoder "
                "passes all materiality, absolute, Pareto, and oracle-fidelity "
                "gates in at least four of five seeds."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": CONFIRMATION_STATUS,
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "exact calibrated C3; eight observations; switches after frames "
                "2/3/4; one unmarked donor-frame substitution; fresh five-seed "
                "corrective after a preserved invalid numerical run"
            ),
            "success_criteria": {
                "joint_seed_passes_minimum": 4,
                "late_inclusive_code_distance": 2,
                "primary_code_distance_minimum": 3,
                "strong_scalar_rmse_maximum": 0.020,
                "strong_exact_bin_accuracy_minimum": 0.90,
                "oracle_rmse_slack_maximum": 0.002,
            },
            "subclaims": {
                "late_inclusive_target_identifiability": (
                    "contradicted_by_exact_token_collision"
                ),
                "three_post_change_frame_identifiability": (
                    "supported_by_exact_code_distance_three"
                ),
                "dynamics_materiality": "supported_five_of_five",
                "corruption_materiality": "supported_five_of_five",
                "oracle_recoverability": "supported_five_of_five",
                "fixed_decoder_closure": "not_supported_three_of_five",
                "oracle_fidelity": "not_supported_one_of_five",
                "compact_typed_continuation_comparison": "licensed",
                "unrestricted_tinyllm_training": "not_licensed",
                "invalid_predecessor": "preserved_not_pooled",
            },
            "tags": [
                "tinyllm",
                "c3",
                "within-sequence-switching",
                "single-frame-corruption",
                "identifiability",
                "error-correcting-code",
                "change-point",
                "fixed-decoder",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The repaired target is identifiable and the oracle passes "
                "five of five, but the fixed decoder passes only three seeds "
                "and the complete oracle-fidelity conjunction only one."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "valid_fresh_five_seed_negative_with_exact_identifiability"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": sum(
                _seed_primary_pass(pair) for pair in direct
            ),
            "failed_direct_experiments": sum(
                not _seed_primary_pass(pair) for pair in direct
            ),
            "primary_seed_passes": sum(
                _seed_primary_pass(pair) for pair in direct
            ),
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "An acquisition horizon with only two post-switch frames is "
                    "provably non-identifiable after one arbitrary corruption."
                ),
                (
                    "Three post-switch frames restore exact one-error correction "
                    "and the oracle reaches the task ceiling in every seed."
                ),
                (
                    "The fixed invariant-carrier residual selector is not "
                    "quantization-robust enough outside range."
                ),
                (
                    "Any learned comparison has a narrow job: infer the discrete "
                    "chart while retaining the exact sensor, dynamics, and decoder."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A fixed charged-first-character decoder removes cubing-induced "
                    "phase-noise amplification and closes the oracle gap."
                ),
                (
                    "Only if that fixed decoder fails, a compact typed chart-mixture "
                    "selector outperforms it on fresh cohorts."
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
            "invalid_result_sha256": INVALID_RESULT_SHA256,
            "invalid_runner_sha256": INVALID_RUNNER_SHA256,
            "original_preregistration_sha256": (
                ORIGINAL_PREREGISTRATION_SHA256
            ),
            "artifact_validation": (
                "result, fresh population, exact code distances, explicit token "
                "collision, invalid-run lineage, numerical correction, group/action "
                "contracts, continuous future equivalence, shuffles, primary gates, "
                "and conservative negative classification revalidated"
            ),
        },
    }


def build_c3_switching_corruption_fixed_decoder_experiment_results(
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
        primary_pass = _seed_primary_pass(pair)
        metrics: dict[str, float] = {
            "validity": 1.0,
            "exact_primary_code_distance": 3.0,
            "late_inclusive_code_distance": 2.0,
            "joint_primary_gate": float(primary_pass),
            "oracle_both_shifts": float(
                all(
                    pair[regime]["operators"]["oracle_switch_drop"][
                        "fixed_ceiling_pass"
                    ]
                    for regime in ("composition", "extrapolation")
                )
            ),
            "fixed_both_shifts": float(
                all(
                    pair[regime]["operators"]["fixed_switch_drop"][
                        "fixed_ceiling_pass"
                    ]
                    for regime in ("composition", "extrapolation")
                )
            ),
            "oracle_fidelity_both_shifts": float(
                all(
                    pair[regime]["oracle_fidelity"]["pass"]
                    for regime in ("composition", "extrapolation")
                )
            ),
            "invalid_primary_examples_pooled": 0.0,
            "models_instantiated": 0.0,
        }
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            fixed = cell["operators"]["fixed_switch_drop"]
            oracle = cell["operators"]["oracle_switch_drop"]
            metrics[f"{regime}_fixed_rmse"] = float(fixed["scalar"]["rmse"])
            metrics[f"{regime}_fixed_accuracy"] = float(
                fixed["task"]["exact_bin_accuracy"]
            )
            metrics[f"{regime}_oracle_rmse"] = float(oracle["scalar"]["rmse"])
            metrics[f"{regime}_joint_hidden_index_recovery"] = float(
                cell["quantized_joint_hidden_index_recovery_rate"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-switching-corruption-fixed-decoder-"
                    f"corrective-seed{pair['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(primary_pass),
                model_architecture=[3, 8, 1],
                model_parameters=0,
                training_time=0.0,
                model_checkpoint="",
                observations=[
                    f"Fresh no-training corrective seed={pair['seed']}.",
                    "Oracle switch/drop passes both shifts.",
                    (
                        "Complete fixed-decoder gate passes."
                        if primary_pass
                        else "Complete fixed-decoder gate fails."
                    ),
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_switching_corruption_fixed_decoder_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_switching_corruption_fixed_decoder_meta_hypothesis(
        results_path
    )
    experiments = build_c3_switching_corruption_fixed_decoder_experiment_results(
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
            raise RuntimeError("C3 switching-corruption ChromaDB read-back failed")
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
    "build_c3_switching_corruption_fixed_decoder_experiment_results",
    "build_c3_switching_corruption_fixed_decoder_meta_hypothesis",
    "store_c3_switching_corruption_fixed_decoder_meta_hypothesis",
]
