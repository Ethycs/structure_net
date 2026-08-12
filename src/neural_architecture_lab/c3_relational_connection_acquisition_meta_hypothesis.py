"""Build and store the C3 relational connection acquisition evidence."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-relational-connection-acquisition.v1"
AUDIT_SCHEMA = "nal.tinyllm-c3-relational-connection-readout-audit.v1"
HYPOTHESIS_ID = "tinyllm-c3-relational-connection-acquisition-v1"
EVIDENCE_ROLE = "prospective_matched_connection_invariant_acquisition"
PRIMARY_CLASSIFICATION = "exact_function_class_but_population_acquisition_unreliable"
AUDIT_CLASSIFICATION = (
    "posthoc_public_scale_readout_reaches_four_of_five_one_wrong_winding_remains"
)
SEEDS = (1453, 1471, 1483, 1531, 1543)
ARMS = (
    "learned_true",
    "learned_no_connection",
    "learned_connection_shuffled",
    "learned_target_shuffled",
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_acquisition/"
    "20260811_preregistered/campaign_results.json"
)
PRIMARY_RESULT_SHA256 = (
    "b4b6e94392a866f7a94188d18935db6602fbc83b936e14324b1060e56ce61a4a"
)
PRIMARY_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_relational_connection_acquisition.py"
)
PRIMARY_RUNNER_SHA256 = (
    "cf425970a3424a32e410492ea79d7d17fd579a83cea78bacea9b8a58674116f0"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-acquisition-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "d1ca34b1d251dc06a69aa293bfbfac2c3ee63fb80de3d4b7ed822e01ad10c015"
)
AUDIT_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_relational_connection_readout_audit/"
    "20260811_artifact_audit/result.json"
)
AUDIT_RESULT_SHA256 = (
    "1fb139ebd13b1ac78d77fa0b82d206f237b3d14ef86173cd7e8d6825dd1731a5"
)
AUDIT_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_relational_connection_readout_audit.py"
)
AUDIT_RUNNER_SHA256 = (
    "0c08b9ccd1062fa173e45769ebaa0396614ff81eee327486f3936b657fffa75a"
)
AUDIT_REGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-relational-connection-readout-audit-registration.md"
)
AUDIT_REGISTRATION_SHA256 = (
    "17d76393374a7898943fe9044ea8019281b2bceab693a372c2af6f14c458cf9d"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-relational-connection-acquisition.md"
)
REPORT_SHA256 = "411ff9b6d3fbdde9e002c8d99f7f7320cb6f1ee09bf0bc8ca2107c7292f0f64d"
DETAIL_SHA256 = {
    1453: "aa30cd1e3e657819536f5bcdfe8db6d74788e919eeb837055011f29bec269205",
    1471: "abae3326b3fa43a82971ab2b54eab5a0de9ef0b7b2eb774e1e7c5bcd06ea379d",
    1483: "82d864c151823943ed2632e56938f377d17e8005fe3998d810d420871b832ec8",
    1531: "d4226a86fd99cb04a6fc7ca66e2948ad4c86898e43a71a5295a511af1a903f36",
    1543: "05a1ffd0891543488ea0fd967e3bf47145d76331297092187e20f94c806e9d19",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _detail_path(seed: int) -> Path:
    return PRIMARY_RESULT_PATH.parent / "runs" / f"seed_{seed}" / "result.json"


def _primary() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    result = json.loads(PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    aggregates = result.get("aggregates", {})
    summary = result.get("summary", {})
    expected_counts = {
        "learned_true": 1,
        "learned_no_connection": 0,
        "learned_connection_shuffled": 0,
        "learned_target_shuffled": 0,
    }
    if (
        _sha256(PRIMARY_RESULT_PATH) != PRIMARY_RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("status") != "completed"
        or aggregates.get("valid") is not True
        or aggregates.get("classification") != PRIMARY_CLASSIFICATION
        or aggregates.get("analytic_joint_pass_count") != 5
        or aggregates.get("analytic_stop_gate_pass") is not True
        or aggregates.get("arm_joint_pass_counts") != expected_counts
        or aggregates.get("control_specificity_pass") is not True
        or aggregates.get("primary_hypothesis_pass") is not False
        or aggregates.get("unrestricted_tinyllm_training_licensed") is not False
        or summary.get("completed") != 5
        or summary.get("primary_optimizer_steps") != 48_000
        or summary.get("resume_verification_optimizer_steps") != 24_000
        or summary.get("tinyllm_models_instantiated") != 0
        or result.get("source_hashes", {}).get("runner") != PRIMARY_RUNNER_SHA256
        or not _finite(result)
    ):
        raise ValueError("invalid C3 relational connection acquisition campaign")
    analytic = result.get("analytic_positive_control", [])
    if len(analytic) != 5 or not all(item.get("joint_pass") is True for item in analytic):
        raise ValueError("analytic connection population changed")
    details = {}
    for seed in SEEDS:
        path = _detail_path(seed)
        if _sha256(path) != DETAIL_SHA256[seed]:
            raise ValueError(f"connection acquisition seed changed: {seed}")
        detail = json.loads(path.read_text(encoding="utf-8"))
        if (
            detail.get("seed") != seed
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("matched_initialization") is not True
            or any(
                arm.get("lifecycle_valid") is not True
                or arm.get("action_pass") is not True
                or arm.get("reload", {}).get("pass") is not True
                or arm.get("exact_resume", {}).get("pass") is not True
                for arm in detail.get("arms", {}).values()
            )
            or len(detail.get("arms", {})) != 4
            or not _finite(detail)
        ):
            raise ValueError(f"invalid connection acquisition seed: {seed}")
        details[seed] = detail
    for source, expected in (
        (PRIMARY_RUNNER_PATH, PRIMARY_RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"source artifact changed: {source}")
    return result, details


def _audit() -> dict[str, Any]:
    result = json.loads(AUDIT_RESULT_PATH.read_text(encoding="utf-8"))
    aggregates = result.get("aggregates", {})
    expected_counts = {
        "neutral_linear": {
            "learned_connection_shuffled": 0,
            "learned_no_connection": 0,
            "learned_target_shuffled": 0,
            "learned_true": 4,
        },
        "scalar_affine": {
            "learned_connection_shuffled": 0,
            "learned_no_connection": 0,
            "learned_target_shuffled": 0,
            "learned_true": 4,
        },
    }
    if (
        _sha256(AUDIT_RESULT_PATH) != AUDIT_RESULT_SHA256
        or result.get("schema_version") != AUDIT_SCHEMA
        or result.get("status") != "completed"
        or result.get("evidence_role")
        != "post_outcome_corrective_artifact_only_readout_audit"
        or aggregates.get("valid") is not True
        or aggregates.get("classification") != AUDIT_CLASSIFICATION
        or aggregates.get("joint_pass_counts") != expected_counts
        or aggregates.get("newly_repaired_true_seed_counts")
        != {"neutral_linear": 3, "scalar_affine": 3}
        or aggregates.get("persistent_true_failure_seeds")
        != {"neutral_linear": [1453], "scalar_affine": [1453]}
        or aggregates.get("primary_classification_unchanged") is not True
        or aggregates.get("unrestricted_tinyllm_training_licensed") is not False
        or aggregates.get("further_optimizer_tuning_licensed") is not False
        or result.get("configuration", {}).get("optimizer_steps") != 0
        or result.get("configuration", {}).get("tinyllm_models_instantiated") != 0
        or not _finite(result)
    ):
        raise ValueError("invalid C3 relational connection readout audit")
    cells = result.get("cells", [])
    if len(cells) != 5 or [int(item["seed"]) for item in cells] != list(SEEDS):
        raise ValueError("readout audit population changed")
    if not all(
        cell.get("pass") is True
        and all(arm.get("replay_pass") is True for arm in cell["arms"].values())
        for cell in cells
    ):
        raise ValueError("readout audit replay changed")
    for source, expected in (
        (AUDIT_RUNNER_PATH, AUDIT_RUNNER_SHA256),
        (AUDIT_REGISTRATION_PATH, AUDIT_REGISTRATION_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"audit artifact changed: {source}")
    return result


def build_c3_relational_connection_acquisition_meta_hypothesis(
    results_path: Path = PRIMARY_RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    if results_path != PRIMARY_RESULT_PATH or report_path != REPORT_PATH:
        raise ValueError("C3 acquisition meta record requires sealed artifact paths")
    primary, details = _primary()
    audit = _audit()
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "Ordinary joint training reliably acquires the exact C3 "
                "connection-invariant endpoint relation"
            ),
            "category": "prospective matched compact typed acquisition",
            "description": (
                "A fixed analytic ceiling and four matched 187-parameter learned "
                "arms test population trainability after identifiability, exact "
                "function-class containment, gradients, and CUDA were established."
            ),
            "question": (
                "Does random-initialized gradient training acquire the known "
                "connection-conditioned relation in at least four of five seeds "
                "without information-shuffled controls succeeding?"
            ),
            "prediction": (
                "The true learned arm passes both shifts in at least four seeds; "
                "no-connection, connection-shuffled, and target-shuffled arms pass "
                "at most one seed each."
            ),
            "created_at": primary["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "rejected_population_acquisition_one_of_five_despite_exact_class"
            ),
            "evidence_count": 6,
            "direct_experiment_count": 5,
            "evidence_pedigree": (
                "preregistered five-seed primary plus post-outcome zero-optimizer "
                "frozen-readout audit"
            ),
            "tested_scope": (
                "known calibrated C3 endpoint relation, exact observed edge "
                "connection, 187-parameter typed module, five fresh seeds, "
                "composition and extrapolation, 2400 AdamW scalar-MSE steps"
            ),
            "success_criteria": {
                "analytic_seed_passes_minimum": 4,
                "learned_true_seed_passes_minimum": 4,
                "each_control_seed_passes_maximum": 1,
                "scalar_correlation_minimum": 0.999,
                "scalar_rmse_maximum": 0.01,
                "exact_bin_accuracy_minimum": 0.95,
                "target_cross_entropy_maximum": 1.35,
                "predicted_bin_coverage": 16,
                "action_error_maximum": 2e-5,
            },
            "subclaims": {
                "analytic_connection_ceiling": "supported_5_of_5",
                "learned_population_acquisition": "rejected_1_of_5",
                "information_specificity": "supported_all_controls_0_of_5",
                "architectural_action_invariance": "supported_20_of_20",
                "checkpoint_and_resume_lifecycle": "supported_20_of_20",
                "posthoc_public_scale_readability": (
                    "corrective_4_of_5_with_three_new_repairs"
                ),
                "persistent_wrong_winding_failure": "seed_1453",
                "primary_result_rescued_by_posthoc_fit": "no",
                "further_optimizer_tuning": "not_licensed",
                "unrestricted_tinyllm_training": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "c3",
                "connection",
                "exact-equivariance",
                "trainability",
                "winding",
                "scale-calibration",
                "negative-result",
                "matched-controls",
            ],
            "explicitly_not_tested": [
                "noisy, missing, partial, or unknown connection data",
                "learning the group action or connection itself",
                "natural-task transfer",
                "TinyLLM utility",
                "optimizer or step-count alternatives after the primary result",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The analytic ceiling passes 5/5 and every control 0/5, but the "
                "true learned arm passes only 1/5 under the frozen protocol."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "valid_preregistered_population_rejection_with_exact_lifecycle"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 1,
            "failed_direct_experiments": 4,
            "primary_seed_passes": 1,
            "descriptive_metrics": {
                "primary_aggregates": primary["aggregates"],
                "primary_summary": primary["summary"],
                "corrective_audit_aggregates": audit["aggregates"],
            },
            "key_insights": [
                (
                    "Identifiability, exact equivariance, function-class containment, "
                    "and a finite true-task gradient do not guarantee reliable global "
                    "acquisition."
                ),
                (
                    "Four true arms enter degree -2 at midpoint; three return to "
                    "degree 1 with compressed public scale, while one remains in the "
                    "wrong winding sector."
                ),
                (
                    "A zero-optimizer affine readout reaches 4/5 and repairs three "
                    "failures, but cannot alter the negative primary classification."
                ),
                (
                    "The six-weight analytic connection remains strictly cheaper and "
                    "more reliable under the known observation law."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "Only a new observation scope with missing, noisy, partial, or "
                    "unknown connection data can justify testing whether a learned "
                    "typed interface adds value over fixed analytic transport."
                )
            ],
            "completed_at": audit["completed_at"],
        },
        "evidence": {
            "direct_tests": [
                {
                    "seed": seed,
                    "classification": PRIMARY_CLASSIFICATION,
                    "gates": details[seed]["gates"],
                    "arms": details[seed]["arms"],
                    "accounting": details[seed]["accounting"],
                }
                for seed in SEEDS
            ],
            "corrective_artifact_audit": {
                "classification": audit["aggregates"]["classification"],
                "aggregates": audit["aggregates"],
                "configuration": audit["configuration"],
            },
        },
        "source_artifacts": [
            str(results_path),
            str(AUDIT_RESULT_PATH),
            str(report_path),
        ],
        "provenance": {
            "primary_result_sha256": PRIMARY_RESULT_SHA256,
            "primary_runner_sha256": PRIMARY_RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "audit_result_sha256": AUDIT_RESULT_SHA256,
            "audit_runner_sha256": AUDIT_RUNNER_SHA256,
            "audit_registration_sha256": AUDIT_REGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "detail_sha256": {str(key): value for key, value in DETAIL_SHA256.items()},
            "artifact_validation": (
                "sealed source/result/report hashes, five analytic controls, five "
                "primary seed results, twenty lifecycle/action/reload/resume cells, "
                "primary accounting, corrective same-device replay, fit counts, and "
                "scope locks revalidated"
            ),
        },
    }


def build_c3_relational_connection_acquisition_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = PRIMARY_RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    if results_path != PRIMARY_RESULT_PATH:
        raise ValueError("C3 acquisition experiment results require sealed path")
    primary, details = _primary()
    audit = _audit()
    audit_by_seed = {int(cell["seed"]): cell for cell in audit["cells"]}
    analytic_by_seed = {
        int(cell["seed"]): cell for cell in primary["analytic_positive_control"]
    }
    completed = datetime.fromisoformat(primary["completed_at"].replace("Z", "+00:00"))
    values = []
    for seed in SEEDS:
        detail = details[seed]
        true = detail["arms"]["learned_true"]
        comp = true["regimes"]["composition"]
        extra = true["regimes"]["extrapolation"]
        corrective = audit_by_seed[seed]["arms"]["learned_true"]["fits"]
        maximum_action = max(
            float(value)
            for arm in detail["arms"].values()
            for value in arm["action_errors"].values()
        )
        metrics: dict[str, float] = {
            "primary_valid": 1.0,
            "analytic_joint_pass": float(analytic_by_seed[seed]["joint_pass"]),
            "learned_true_joint_pass": float(true["joint_pass"]),
            "learned_no_connection_joint_pass": float(
                detail["arms"]["learned_no_connection"]["joint_pass"]
            ),
            "learned_connection_shuffled_joint_pass": float(
                detail["arms"]["learned_connection_shuffled"]["joint_pass"]
            ),
            "learned_target_shuffled_joint_pass": float(
                detail["arms"]["learned_target_shuffled"]["joint_pass"]
            ),
            "all_lifecycle_valid": float(
                all(arm["lifecycle_valid"] for arm in detail["arms"].values())
            ),
            "maximum_action_error": maximum_action,
            "composition_correlation": float(comp["scalar"]["correlation"]),
            "composition_rmse": float(comp["scalar"]["rmse"]),
            "composition_accuracy": float(comp["task"]["exact_bin_accuracy"]),
            "extrapolation_correlation": float(extra["scalar"]["correlation"]),
            "extrapolation_rmse": float(extra["scalar"]["rmse"]),
            "extrapolation_accuracy": float(extra["task"]["exact_bin_accuracy"]),
            "initial_winding": float(
                true["winding_diagnostic"]["initial"]["winding_number"]
            ),
            "midpoint_winding": float(
                true["winding_diagnostic"]["midpoint"]["winding_number"]
            ),
            "final_winding": float(
                true["winding_diagnostic"]["final"]["winding_number"]
            ),
            "posthoc_scalar_affine_joint_pass": float(
                corrective["scalar_affine"]["joint_pass"]
            ),
            "posthoc_neutral_linear_joint_pass": float(
                corrective["neutral_linear"]["joint_pass"]
            ),
            "primary_optimizer_steps": 9_600.0,
            "resume_verification_optimizer_steps": 4_800.0,
            "tinyllm_models_instantiated": 0.0,
        }
        values.append(
            ExperimentResult(
                experiment_id=f"tinyllm-c3-relational-connection-acquisition-seed{seed}",
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(true["joint_pass"]),
                model_architecture=[1, 16, 8, 2, 1],
                model_parameters=187,
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["arms"]["learned_true"]["final_checkpoint"]["path"],
                observations=[
                    "Analytic positive control and four matched learned arms.",
                    "Exact action, final reload, and midpoint resume lifecycle.",
                    "Post-outcome scalar-affine and neutral-linear frozen-readout audit.",
                ],
                anomalies=(
                    []
                    if true["joint_pass"]
                    else [
                        "Primary learned true arm failed the simultaneous two-shift endpoint."
                    ]
                ),
                timestamp=completed,
            )
        )
    return values


def store_c3_relational_connection_acquisition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_relational_connection_acquisition_meta_hypothesis(
        results_path
    )
    experiments = build_c3_relational_connection_acquisition_experiment_results(
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
            raise RuntimeError("C3 acquisition ChromaDB read-back failed")
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
    "build_c3_relational_connection_acquisition_experiment_results",
    "build_c3_relational_connection_acquisition_meta_hypothesis",
    "store_c3_relational_connection_acquisition_meta_hypothesis",
]
