"""Build and store the fresh C3 gauge-jump joint typed-score result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-gauge-jump-joint-typed-score.v1"
HYPOTHESIS_ID = "tinyllm-c3-gauge-jump-joint-typed-score-v1"
EVIDENCE_ROLE = "fresh_prospective_no_training_joint_typed_score_replication"
CLASSIFICATION = "both_fixed_connection_scores_close_fresh_scope"
CONFIRMATION_STATUS = (
    "confirmed_fixed_time_varying_gauge_closure_without_unique_typed_advantage"
)
RESULT_SHA256 = "f52ce2103a07086a7118975d69f49b7cbeca01ac0ca7c5a15fd6d2a96fbc51fa"
RUNNER_SHA256 = "6a9b1b849c97fc30bef7292ed0bbc097c4829db2920d3abe8f698e7546489944"
PREREGISTRATION_SHA256 = (
    "e4629a11cac991b1bd64d641f3276b4517296ee31ac3b9e0a3837e5cb5ce4663"
)
REPORT_SHA256 = "281f375c069fb58b9949b7e2d0c98c895e4bfedf8a9f4e7160204e2dc3bb852b"
PRIOR_RESULT_SHA256 = (
    "16f98f5c3cbf09fedfc18f12eca24a5fe69da46411d587c48c5d9072c912aca7"
)
PRIOR_RUNNER_SHA256 = (
    "5b35658103481645aba809f5575d38159dcddc9dc7330ebfa6764ad65ba170a4"
)
PRIOR_PREREGISTRATION_SHA256 = (
    "caab4704bb5c11a3f7353c31e20a80da33f14e1a925402df055b652916de10d9"
)
PRIOR_REPORT_SHA256 = (
    "2c2fb44afaf9bd35a63cd82d4331943bdbbfa1a2c366080f50c5a5e8fe0bb6b1"
)
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_gauge_jump_joint_typed_score/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_gauge_jump_joint_typed_score.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-gauge-jump-joint-typed-score.md"
)
PRIOR_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_gauge_jump_corruption_fixed_decoder/"
    "20260811_preregistered/result.json"
)
PRIOR_RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_gauge_jump_corruption_fixed_decoder.py"
)
PRIOR_PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-gauge-jump-corruption-fixed-decoder-"
    "preregistration.md"
)
PRIOR_REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-gauge-jump-corruption-fixed-decoder.md"
)
SEEDS = (773, 821, 1003, 1031, 1039)
PRIOR_SEEDS = (673, 691, 709, 727, 751)
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
        "classification": CLASSIFICATION,
        "compact_typed_chart_mixture_licensed": False,
        "connection_fixed_seed_pass_count": 4,
        "connection_oracle_seed_pass_count": 5,
        "invariant_fixed_seed_pass_count": 5,
        "invariant_oracle_seed_pass_count": 5,
        "joint_typed_material_repair_seed_pass_count": 0,
        "joint_typed_oracle_fidelity_seed_pass_count": 5,
        "joint_typed_seed_pass_count": 5,
        "no_connection_seed_pass_count": 0,
        "required_seed_passes": 4,
        "tinyllm_training_licensed": False,
        "valid": True,
    }


def _expected_accounting() -> dict[str, int]:
    return {
        "checkpoints_loaded": 0,
        "continuous_validation_candidate_fits": 12_779_520,
        "fresh_base_examples": 40_960,
        "fresh_corrupted_evaluations": 40_960,
        "models_instantiated": 0,
        "optimizer_steps": 0,
        "outcome_known_diagnostic_examples_pooled": 0,
        "parameters_changed": 0,
        "primary_observation_only_candidate_fits": 13_762_560,
        "reusable_parameter_fits": 0,
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
        or sources.get("gauge_jump_primary_result") != PRIOR_RESULT_SHA256
        or sources.get("gauge_jump_primary_runner") != PRIOR_RUNNER_SHA256
        or sources.get("gauge_jump_primary_preregistration")
        != PRIOR_PREREGISTRATION_SHA256
        or sources.get("gauge_jump_primary_report") != PRIOR_REPORT_SHA256
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 gauge-jump joint typed result {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (PRIOR_RESULT_PATH, PRIOR_RESULT_SHA256),
        (PRIOR_RUNNER_PATH, PRIOR_RUNNER_SHA256),
        (PRIOR_PREREGISTRATION_PATH, PRIOR_PREREGISTRATION_SHA256),
        (PRIOR_REPORT_PATH, PRIOR_REPORT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")
    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime) for seed in SEEDS for regime in REGIMES
    }:
        raise ValueError("joint typed fresh population changed")
    for cell in cells:
        operators = cell.get("operators", {})
        typed = operators.get("fixed_joint_typed_connection", {})
        if (
            cell.get("valid") is not True
            or typed.get("fixed_ceiling_pass") is not True
            or typed.get("maximum_deck_action_error") != 0.0
            or cell.get("joint_typed_oracle_fidelity", {}).get("pass")
            is not True
            or any(
                operator.get("action_pass") is not True
                or operator.get("shuffle_pass") is not True
                or operator.get("shuffled_task_pass") is not False
                for operator in operators.values()
            )
            or max(cell.get("continuous_prediction_maximum_errors", {}).values())
            > 1e-10
            or cell.get("stabilization_displacements", {}).get("maximum")
            > 1e-12
        ):
            raise ValueError("joint typed fresh cell changed")
    original_failures = {
        (cell["seed"], cell["regime"])
        for cell in cells
        if not cell["operators"]["fixed_charged_connection"][
            "fixed_ceiling_pass"
        ]
    }
    if original_failures != {(821, "composition")}:
        raise ValueError("fresh original-selector comparator changed")
    predecessor = json.loads(PRIOR_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        predecessor.get("status") != "completed"
        or predecessor.get("aggregates", {}).get("classification")
        != "recoverable_time_varying_gauge_exceeds_fixed_connection_decoder"
        or predecessor.get("aggregates", {}).get(
            "connection_fixed_seed_pass_count"
        )
        != 1
        or predecessor.get("aggregates", {}).get(
            "connection_oracle_seed_pass_count"
        )
        != 5
        or predecessor.get("aggregates", {}).get(
            "invariant_fixed_seed_pass_count"
        )
        != 5
    ):
        raise ValueError("gauge-jump predecessor changed")
    return result


def _cells_by_seed(
    result: Mapping[str, Any], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    indexed = {
        (int(cell["seed"]), cell["regime"]): cell for cell in result["cells"]
    }
    return [
        {
            "seed": seed,
            "composition": indexed[(seed, "composition")],
            "extrapolation": indexed[(seed, "extrapolation")],
        }
        for seed in seeds
    ]


def _typed_seed_pass(pair: Mapping[str, Any]) -> bool:
    return all(
        pair[regime]["valid"]
        and pair[regime]["operators"]["fixed_joint_typed_connection"][
            "fixed_ceiling_pass"
        ]
        and pair[regime]["joint_typed_oracle_fidelity"]["pass"]
        for regime in REGIMES
    )


def build_c3_gauge_jump_joint_typed_score_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _load_and_validate(results_path)
    predecessor = json.loads(PRIOR_RESULT_PATH.read_text(encoding="utf-8"))
    direct = _cells_by_seed(result, SEEDS)
    context = _cells_by_seed(predecessor, PRIOR_SEEDS)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "Fixed symmetry-typed estimators close one-jump C3 gauge scope"
            ),
            "category": "fresh no-training time-varying-gauge replication",
            "description": (
                "A fresh five-seed replication compares the original charged "
                "connection selector with a fixed score that independently "
                "combines charged connection and invariant physical-chart residuals."
            ),
            "question": (
                "Does a hidden suffix gauge jump require learned chart mixing, "
                "or do fixed symmetry-typed scores close the observable task?"
            ),
            "prediction": (
                "The fixed joint typed score and its charged-oracle fidelity "
                "pass at least four of five fresh seeds; learned work is licensed "
                "only if both fixed charged scores remain below four."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": CONFIRMATION_STATUS,
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "context_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "exact calibrated C3; one hidden suffix jump; one donor-frame "
                "corruption; switching physical law; five fresh 4096-example "
                "seeds per shift; exact 288-candidate connection bank"
            ),
            "success_criteria": {
                "joint_seed_passes_minimum": 4,
                "strong_scalar_rmse_maximum": 0.020,
                "strong_exact_bin_accuracy_minimum": 0.90,
                "oracle_rmse_slack_maximum": 0.002,
                "oracle_accuracy_slack_maximum": 0.01,
                "oracle_cross_entropy_slack_maximum": 0.005,
                "deck_action_error_maximum": 2e-12,
            },
            "subclaims": {
                "fresh_invariant_oracle_recoverability": "supported_five_of_five",
                "fresh_charged_connection_oracle": "supported_five_of_five",
                "fresh_original_fixed_connection": "supported_four_of_five",
                "fresh_joint_typed_connection": "supported_five_of_five",
                "fresh_joint_typed_oracle_fidelity": "supported_five_of_five",
                "unique_joint_typed_tail_repair": "not_supported_zero_of_five",
                "predecessor_fixed_connection_failure": "preserved_one_of_five",
                "predecessor_examples_in_fresh_result": "zero",
                "compact_typed_chart_mixture": "not_licensed",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "time-varying-gauge",
                "discrete-connection",
                "charged-character",
                "invariant-residual",
                "fixed-decoder",
                "fresh-replication",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "The joint typed score passes closure and oracle fidelity in all "
                "five fresh seeds, while the original fixed charged selector also "
                "reaches the locked four-of-five population gate."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "fresh_five_seed_fixed_time_varying_gauge_closure_without_unique_repair"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": sum(
                _typed_seed_pass(pair) for pair in direct
            ),
            "failed_direct_experiments": sum(
                not _typed_seed_pass(pair) for pair in direct
            ),
            "primary_seed_passes": sum(_typed_seed_pass(pair) for pair in direct),
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "A hidden local gauge connection is causally material for a "
                    "charged carrier, but the physical invariant remains jump-immune."
                ),
                (
                    "The original hard-selector tail was valid in its population "
                    "but did not replicate as a sub-threshold population failure."
                ),
                (
                    "A normalized invariant physical-chart residual can be added "
                    "without breaking exact charged deck closure or fitting a model."
                ),
                (
                    "Neither the compact typed mixture nor unrestricted TinyLLM "
                    "has empirical rent in the declared one-jump generator."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A new learned job requires multiple or unknown gauge jumps, "
                    "an unknown group, partial group observations, or a connection "
                    "family that cannot be analytically enumerated."
                )
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {
            "direct_tests": direct,
            "predecessor_context_tests": context,
        },
        "source_artifacts": [
            str(results_path),
            str(report_path),
            str(PRIOR_RESULT_PATH),
            str(PRIOR_REPORT_PATH),
        ],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "predecessor_result_sha256": PRIOR_RESULT_SHA256,
            "predecessor_runner_sha256": PRIOR_RUNNER_SHA256,
            "predecessor_preregistration_sha256": PRIOR_PREREGISTRATION_SHA256,
            "predecessor_report_sha256": PRIOR_REPORT_SHA256,
            "artifact_validation": (
                "fresh and predecessor results, runners, preregistrations, "
                "reports, exact action and continuous contracts, shuffles, "
                "population gates, accounting, and conservative training stop "
                "revalidated"
            ),
        },
    }


def build_c3_gauge_jump_joint_typed_score_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _load_and_validate(results_path)
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    experiments = []
    for pair in _cells_by_seed(result, SEEDS):
        primary_pass = _typed_seed_pass(pair)
        metrics: dict[str, float] = {
            "validity": 1.0,
            "joint_typed_gate": float(primary_pass),
            "joint_typed_both_shifts": float(primary_pass),
            "original_fixed_both_shifts": float(
                all(
                    pair[regime]["operators"]["fixed_charged_connection"][
                        "fixed_ceiling_pass"
                    ]
                    for regime in REGIMES
                )
            ),
            "joint_typed_oracle_fidelity_both_shifts": float(
                all(
                    pair[regime]["joint_typed_oracle_fidelity"]["pass"]
                    for regime in REGIMES
                )
            ),
            "joint_typed_material_repair_both_shifts": float(
                all(
                    pair[regime]["joint_typed_material_repair"]
                    for regime in REGIMES
                )
            ),
            "exact_joint_typed_action_error": 0.0,
            "predecessor_examples_pooled": 0.0,
            "models_instantiated": 0.0,
        }
        for regime in REGIMES:
            cell = pair[regime]
            typed = cell["operators"]["fixed_joint_typed_connection"]
            original = cell["operators"]["fixed_charged_connection"]
            oracle = cell["operators"]["oracle_charged_connection"]
            metrics[f"{regime}_joint_typed_rmse"] = float(
                typed["scalar"]["rmse"]
            )
            metrics[f"{regime}_joint_typed_accuracy"] = float(
                typed["task"]["exact_bin_accuracy"]
            )
            metrics[f"{regime}_original_fixed_rmse"] = float(
                original["scalar"]["rmse"]
            )
            metrics[f"{regime}_oracle_rmse"] = float(
                oracle["scalar"]["rmse"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-gauge-jump-joint-typed-score-"
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
                    f"Fresh joint typed-score seed={pair['seed']}.",
                    "Joint typed closure and oracle fidelity pass both shifts.",
                    "No predecessor example, fitted model, or optimizer was used.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_gauge_jump_joint_typed_score_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_gauge_jump_joint_typed_score_meta_hypothesis(results_path)
    experiments = build_c3_gauge_jump_joint_typed_score_experiment_results(
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
            raise RuntimeError("C3 gauge-jump typed-score ChromaDB read-back failed")
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
    "build_c3_gauge_jump_joint_typed_score_experiment_results",
    "build_c3_gauge_jump_joint_typed_score_meta_hypothesis",
    "store_c3_gauge_jump_joint_typed_score_meta_hypothesis",
]
