"""Build and store the exact C3 charged-character corrective result."""

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
    "nal.tinyllm-c3-charged-character-fixed-decoder-corrective.v1"
)
HYPOTHESIS_ID = "tinyllm-c3-charged-character-fixed-decoder-corrective-v1"
EVIDENCE_ROLE = "fresh_numerical_corrective_no_training_replication"
CLASSIFICATION = (
    "exact_charged_character_corrective_closes_invariant_quantization_gap"
)
CONFIRMATION_STATUS = (
    "confirmed_exact_charged_closure_five_of_five_"
    "invariant_comparator_three_of_five"
)
RESULT_SHA256 = "d8076bd7bfc42819a17438e188300e8ea0131dcc72d9ceac0bee39ede72fcaaf"
RUNNER_SHA256 = "5fcd691d3fd910a619bd61aa8dd0432e0050efc44699657ec98f5d7e2e01de97"
PREREGISTRATION_SHA256 = (
    "1fa97f74a0f88c0016a8b0309f1a392f9b8d45d545cdb0e39bb7754ecafe2020"
)
REPORT_SHA256 = "9acd88cf100b67f04a27c5415dcbce755b97c3545d6ba78ce114e97885c5ba10"
INVALID_PREREGISTRATION_SHA256 = (
    "9f1a6b280498f254e51324c8d87d385fa5ab72090cc2487db57d5f84b8925633"
)
INVALID_RUNNER_SHA256 = (
    "009c23bf287cf233f16d3a40094c011f546c8982c14706ab59b6ddcdc818e5a3"
)
INVALID_RESULT_SHA256 = (
    "0cf9bd7ca8e22bb5712453aab679dfa6aa76e067fd2e242c07874b42b854264a"
)
RESULT_PATH = Path(
    "data/experiments/"
    "tinyllm_c3_charged_character_fixed_decoder_corrective/"
    "20260811_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_charged_character_fixed_decoder_corrective.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-charged-character-fixed-decoder-"
    "corrective-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-charged-character-fixed-decoder.md"
)
INVALID_PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-charged-character-fixed-decoder-"
    "preregistration.md"
)
INVALID_RUNNER_PATH = Path(
    "experiments/structure_net/"
    "tinyllm_c3_charged_character_fixed_decoder.py"
)
INVALID_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_charged_character_fixed_decoder/"
    "20260811_preregistered/result.json"
)
SEEDS = (577, 593, 613, 631, 653)
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
        "valid": True,
        "required_seed_passes": 4,
        "invariant_oracle_seed_pass_count": 5,
        "invariant_fixed_seed_pass_count": 3,
        "charged_oracle_seed_pass_count": 5,
        "charged_fixed_seed_pass_count": 5,
        "charged_oracle_fidelity_seed_pass_count": 5,
        "invariant_oracle_fidelity_seed_pass_count": 3,
        "compact_typed_continuation_comparison_licensed": False,
        "tinyllm_training_licensed": False,
    }


def _expected_accounting() -> dict[str, int]:
    return {
        "checkpoints_loaded": 0,
        "continuous_validation_candidate_fits": 1_966_080,
        "fresh_base_examples": 40_960,
        "fresh_corrupted_evaluations": 40_960,
        "invalid_primary_examples_pooled": 0,
        "models_instantiated": 0,
        "observation_only_candidate_fits": 1_966_080,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "reusable_parameter_fits": 0,
        "target_using_fits": 0,
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
        or result.get("accounting") != _expected_accounting()
        or sources.get("runner") != RUNNER_SHA256
        or sources.get("preregistration") != PREREGISTRATION_SHA256
        or sources.get("invalid_preregistration")
        != INVALID_PREREGISTRATION_SHA256
        or sources.get("invalid_runner") != INVALID_RUNNER_SHA256
        or sources.get("invalid_result") != INVALID_RESULT_SHA256
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 charged-character corrective result {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (INVALID_PREREGISTRATION_PATH, INVALID_PREREGISTRATION_SHA256),
        (INVALID_RUNNER_PATH, INVALID_RUNNER_SHA256),
        (INVALID_RESULT_PATH, INVALID_RESULT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")
    invalid = json.loads(INVALID_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        invalid.get("status") != "invalid"
        or invalid.get("aggregates", {}).get("classification")
        != "invalid_charged_character_fixed_decoder"
        or sum(not cell.get("valid", False) for cell in invalid.get("cells", []))
        != 7
    ):
        raise ValueError("invalid charged predecessor changed")
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
    ):
        raise ValueError("charged corrective identifiability audit changed")
    cells = result.get("cells", [])
    identities = {(cell.get("seed"), cell.get("regime")) for cell in cells}
    if len(cells) != 10 or identities != {
        (seed, regime) for seed in SEEDS for regime in REGIMES
    }:
        raise ValueError("charged corrective population changed")
    for cell in cells:
        operators = cell.get("operators", {})
        if (
            cell.get("valid") is not True
            or any(
                contract.get("pass") is not True
                for contract in cell.get(
                    "exact_integer_pair_contracts", {}
                ).values()
            )
            or operators.get("oracle_invariant_switch_drop", {}).get(
                "fixed_ceiling_pass"
            )
            is not True
            or operators.get("oracle_charged_switch_drop", {}).get(
                "fixed_ceiling_pass"
            )
            is not True
            or operators.get("fixed_charged_switch_drop", {}).get(
                "fixed_ceiling_pass"
            )
            is not True
            or cell.get("charged_oracle_fidelity", {}).get("pass") is not True
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
            raise ValueError("charged corrective cell changed")
        for arm in (
            "oracle_charged_switch_drop",
            "fixed_charged_switch_drop",
        ):
            if operators[arm]["maximum_deck_action_error"] != 0.0:
                raise ValueError("exact charged action contract changed")
    invariant_failures = {
        cell["seed"]
        for cell in cells
        if not cell["operators"]["fixed_invariant_switch_drop"][
            "fixed_ceiling_pass"
        ]
    }
    if invariant_failures != {577, 593}:
        raise ValueError("invariant comparator pattern changed")
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


def _seed_primary_pass(pair: Mapping[str, Any]) -> bool:
    return all(
        pair[regime]["valid"]
        and pair[regime]["operators"]["oracle_invariant_switch_drop"][
            "fixed_ceiling_pass"
        ]
        and pair[regime]["operators"]["oracle_charged_switch_drop"][
            "fixed_ceiling_pass"
        ]
        and pair[regime]["operators"]["fixed_charged_switch_drop"][
            "fixed_ceiling_pass"
        ]
        and pair[regime]["charged_oracle_fidelity"]["pass"]
        for regime in REGIMES
    )


def build_c3_charged_character_fixed_decoder_meta_hypothesis(
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
                "Exact charged C3 character closes the quantized switching "
                "fixed-decoder gap"
            ),
            "category": "prospective no-training symmetry-typed corrective",
            "description": (
                "Five fresh seeds compare fixed switch/corruption selection in "
                "the cubed invariant carrier with the deck-charged first "
                "character, followed by the same invariant forecast decoder."
            ),
            "question": (
                "Does preserving the charged first character through discrete "
                "chart selection remove premature-cubing noise amplification?"
            ),
            "prediction": (
                "Both oracles, fixed charged closure, and charged oracle fidelity "
                "pass at least four of five seeds while fixed invariant closure "
                "remains below four."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": CONFIRMATION_STATUS,
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "exact calibrated C3; eight observations; switches after frames "
                "2/3/4; one unmarked donor-frame substitution; exact Eisenstein "
                "neutral products; five fresh 4096-example seeds per shift"
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
                "invariant_oracle_recoverability": "supported_five_of_five",
                "charged_oracle_recoverability": "supported_five_of_five",
                "fixed_invariant_closure": "not_supported_three_of_five",
                "fixed_charged_closure": "supported_five_of_five",
                "charged_oracle_fidelity": "supported_five_of_five",
                "exact_charged_deck_closure": "supported_zero_action_error",
                "unique_charged_population_advantage": "supported",
                "invalid_primary": "preserved_not_pooled",
                "compact_typed_continuation_comparison": "not_licensed",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "charged-character",
                "equivariance",
                "invariance",
                "change-point",
                "single-frame-corruption",
                "quantization",
                "exact-arithmetic",
                "fixed-decoder",
                "no-training",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "The exact charged fixed decoder and its oracle-fidelity gate pass "
                "both shifts in all five fresh seeds, while the invariant fixed "
                "comparator independently replicates at only three of five."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "prospective_fresh_five_seed_symmetry_typed_corrective_closure"
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
                    "A representation can be exactly invariant yet numerically "
                    "inferior for a discrete intermediate decision."
                ),
                (
                    "Keeping the first character equivariant until after chart "
                    "selection avoids cubic phase-noise amplification."
                ),
                (
                    "Exact Eisenstein neutral products enforce bitwise deck "
                    "closure without relaxing the numerical action gate."
                ),
                (
                    "The prior oracle/fixed gap was a representation-ordering "
                    "problem, not evidence that learned capacity was needed."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A genuinely new learned job requires an unknown group, "
                    "partial group observations, or no known charged sufficient "
                    "statistic after a fresh identifiability and fixed-ceiling audit."
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
            "invalid_preregistration_sha256": (
                INVALID_PREREGISTRATION_SHA256
            ),
            "invalid_runner_sha256": INVALID_RUNNER_SHA256,
            "invalid_result_sha256": INVALID_RESULT_SHA256,
            "artifact_validation": (
                "result, runner, preregistrations, report, invalid lineage, fresh "
                "population, exact integer action identities, continuous forecast, "
                "shuffles, joint charged gates, invariant comparator, accounting, "
                "and conservative training stop revalidated"
            ),
        },
    }


def build_c3_charged_character_fixed_decoder_experiment_results(
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
            "joint_charged_gate": float(primary_pass),
            "charged_both_shifts": float(
                all(
                    pair[regime]["operators"]["fixed_charged_switch_drop"][
                        "fixed_ceiling_pass"
                    ]
                    for regime in REGIMES
                )
            ),
            "invariant_both_shifts": float(
                all(
                    pair[regime]["operators"]["fixed_invariant_switch_drop"][
                        "fixed_ceiling_pass"
                    ]
                    for regime in REGIMES
                )
            ),
            "charged_oracle_fidelity_both_shifts": float(
                all(
                    pair[regime]["charged_oracle_fidelity"]["pass"]
                    for regime in REGIMES
                )
            ),
            "exact_charged_action_error": 0.0,
            "invalid_primary_examples_pooled": 0.0,
            "models_instantiated": 0.0,
        }
        for regime in REGIMES:
            cell = pair[regime]
            charged = cell["operators"]["fixed_charged_switch_drop"]
            invariant = cell["operators"]["fixed_invariant_switch_drop"]
            oracle = cell["operators"]["oracle_charged_switch_drop"]
            metrics[f"{regime}_charged_rmse"] = float(
                charged["scalar"]["rmse"]
            )
            metrics[f"{regime}_charged_accuracy"] = float(
                charged["task"]["exact_bin_accuracy"]
            )
            metrics[f"{regime}_invariant_rmse"] = float(
                invariant["scalar"]["rmse"]
            )
            metrics[f"{regime}_oracle_rmse"] = float(
                oracle["scalar"]["rmse"]
            )
            metrics[f"{regime}_charged_oracle_selection_rate"] = float(
                cell["charged_selected_oracle_rate"]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    "tinyllm-c3-charged-character-fixed-decoder-"
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
                    f"Fresh exact charged corrective seed={pair['seed']}.",
                    "Charged fixed and oracle-fidelity gates pass both shifts.",
                    "No invalid example, learned model, or optimizer was used.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_charged_character_fixed_decoder_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_charged_character_fixed_decoder_meta_hypothesis(
        results_path
    )
    experiments = build_c3_charged_character_fixed_decoder_experiment_results(
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
            raise RuntimeError("C3 charged-character ChromaDB read-back failed")
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
    "build_c3_charged_character_fixed_decoder_experiment_results",
    "build_c3_charged_character_fixed_decoder_meta_hypothesis",
    "store_c3_charged_character_fixed_decoder_meta_hypothesis",
]
