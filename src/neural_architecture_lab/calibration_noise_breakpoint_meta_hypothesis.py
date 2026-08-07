"""Build and store TinyLLM representation-only calibration breakpoint evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-calibration-degradation.v1"
HYPOTHESIS_ID = "tinyllm-calibration-noise-breakpoint-v1"
EVIDENCE_ROLE = "preregistered_sequential_frozen_checkpoint_robustness_evidence"
IMPLEMENTATION_SHA256 = (
    "175160f9ca3c51bffb713b8fc9251c3eb888d0f9121424772811566aee77a329"
)
CAMPAIGN_SHA256 = (
    "87a556e61db4a584b9cd423af9cb9d663f3f0225757a31a69a7158788137ef86"
)
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
LEVELS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
LEVEL_KEYS = ("0", "0.125", "0.25", "0.5", "1", "2", "4")
REGIMES = ("composition", "extrapolation")
CUTS = ("frontend", "full")
EXPECTED_GATES = {
    "finite_numerical_contract": True,
    "source_checkpoint_state_contract": True,
    "source_provenance_contract": True,
    "zero_calibration_identity_contract": True,
    "zero_source_metric_replay_contract": True,
}
EXPECTED_COUNTS = {
    "analytic_calibrated": (5, 5, 5, 5, 5, 0, 0),
    "learned_calibrated_equivariant": (5, 5, 5, 5, 3, 0, 0),
}
EXPECTED_BREAKPOINTS = {
    "analytic_calibrated": {
        "any_nonzero_pass": True,
        "breakpoint_interval": [1, 2],
        "first_failed_level": 2,
        "gate_reentry": False,
        "last_passing_level": 1,
        "maximum_level_fails": True,
        "right_censored": False,
        "zero_pass": True,
    },
    "learned_calibrated_equivariant": {
        "any_nonzero_pass": True,
        "breakpoint_interval": [0.5, 1],
        "first_failed_level": 1,
        "gate_reentry": False,
        "last_passing_level": 0.5,
        "maximum_level_fails": True,
        "right_censored": False,
        "zero_pass": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite_numbers(item) for item in value)
    return True


def _endpoint_pass(evaluation: Mapping[str, Any]) -> bool:
    return bool(
        float(evaluation["cosine_pearson"]) >= 0.90
        and float(evaluation["balanced_accuracy"]) <= 0.55
    )


def _evaluation_joint_pass(evaluation: Mapping[str, Any]) -> bool:
    return all(
        _endpoint_pass(
            evaluation["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]
        )
        for cut in CUTS
        for regime in REGIMES
    )


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    configuration = value.get("configuration", {})
    source_campaign = Path(provenance.get("source_campaign", ""))
    arms = aggregates.get("arms", {})
    aggregate_contract = set(arms) == set(CONDITIONS)
    for condition in CONDITIONS:
        arm = arms.get(condition, {})
        level_records = arm.get("levels", {})
        aggregate_contract = bool(
            aggregate_contract
            and set(level_records) == set(LEVEL_KEYS)
            and tuple(
                int(level_records[key].get("joint_pass_count", -1))
                for key in LEVEL_KEYS
            )
            == EXPECTED_COUNTS[condition]
            and arm.get("bounded_stable_breakpoint") is True
            and arm.get("breakpoint") == EXPECTED_BREAKPOINTS[condition]
            and int(arm.get("shuffled_control", {}).get("joint_pass_count", -1)) == 0
            and arm.get("shuffled_control", {}).get("negative_control_pass") is True
        )
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or provenance.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("source_implementation_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or not source_campaign.is_file()
        or _sha256(source_campaign) != SOURCE_CAMPAIGN_SHA256
        or tuple(configuration.get("conditions", ())) != CONDITIONS
        or tuple(configuration.get("seeds", ())) != SEEDS
        or tuple(float(item) for item in configuration.get("noise_levels", ()))
        != LEVELS
        or int(configuration.get("required_seed_passes", -1)) != 4
        or int(summary.get("requested", -1)) != 10
        or int(summary.get("completed", -1)) != 10
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("fitted_frontends", -1)) != 0
        or int(summary.get("fitted_evaluation_probes", -1)) != 170
        or aggregates.get("supported") is not True
        or aggregates.get("validity_contract") is not True
        or aggregates.get("conclusion")
        != "bounded_stable_calibration_breakpoints_both_arms"
        or aggregates.get("gate_counts")
        != {name: 10 for name in EXPECTED_GATES}
        or not aggregate_contract
    ):
        raise ValueError(f"invalid calibration-noise breakpoint campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    indexed = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"invalid calibration-noise result index {path}")

    output = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            source_result_path = Path(provenance.get("result", ""))
            model_path = Path(provenance.get("model_checkpoint", ""))
            frontend_path = Path(provenance.get("frontend_checkpoint", ""))
            source_result = (
                json.loads(source_result_path.read_text(encoding="utf-8"))
                if source_result_path.is_file()
                else {}
            )
            evaluations = detail.get("evaluations", {})
            expected_keys = {*LEVEL_KEYS, "shuffled"}
            expected_passes = tuple(
                bool(
                    campaign["aggregates"]["arms"][condition]["levels"][key][
                        "pass_by_seed"
                    ][str(seed)]
                )
                for key in LEVEL_KEYS
            )
            actual_passes = tuple(
                bool(evaluations.get(key, {}).get("joint_seed_pass"))
                for key in LEVEL_KEYS
            )
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("experiment_id") != entry.get("experiment_id")
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or detail.get("gates") != EXPECTED_GATES
                or float(
                    detail.get("maximum_zero_source_metric_replay_error", math.inf)
                )
                != 0.0
                or set(evaluations) != expected_keys
                or tuple(
                    float(evaluations[key].get("level", math.nan))
                    for key in LEVEL_KEYS
                )
                != LEVELS
                or any(
                    bool(evaluations[key].get("joint_seed_pass"))
                    != _evaluation_joint_pass(evaluations[key])
                    for key in expected_keys
                )
                or actual_passes != expected_passes
                or bool(evaluations["shuffled"].get("joint_seed_pass"))
                or not source_result_path.is_file()
                or _sha256(source_result_path) != provenance.get("result_sha256")
                or source_result.get("condition") != condition
                or int(source_result.get("seed", -1)) != seed
                or not model_path.is_file()
                or _sha256(model_path) != provenance.get("model_checkpoint_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path)
                != provenance.get("frontend_checkpoint_sha256")
                or not _finite_numbers(detail)
            ):
                raise ValueError(f"invalid calibration-noise result {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(
                    f"duplicate calibration-noise fingerprint {result_path}"
                )
            fingerprints.add(fingerprint)
            detail["_source_model_parameters"] = int(source_result["model_parameters"])
            output.append(detail)
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    evaluations = detail["evaluations"]
    pass_pattern = [
        bool(evaluations[key]["joint_seed_pass"]) for key in LEVEL_KEYS
    ]
    last_passing_level = max(
        level for level, passed in zip(LEVELS, pass_pattern) if passed
    )
    primary_values = [
        evaluations[key]["analysis"]["cuts"][cut]["probe"]["evaluations"][regime]
        for key in LEVEL_KEYS
        for cut in CUTS
        for regime in REGIMES
    ]
    first_failure_index = next(
        index for index, passed in enumerate(pass_pattern) if not passed
    )
    return {
        "pass_pattern": pass_pattern,
        "last_passing_level": float(last_passing_level),
        "first_failed_level": float(LEVELS[first_failure_index]),
        "maximum_branch_accuracy": max(
            float(value["balanced_accuracy"]) for value in primary_values
        ),
        "maximum_log_loss_gain": max(
            float(value["conditional_log_loss_gain_over_cosine_only"])
            for value in primary_values
        ),
        "minimum_cosine_at_last_pass": min(
            float(
                evaluations[str(int(last_passing_level)) if last_passing_level.is_integer() else str(last_passing_level)][
                    "analysis"
                ]["cuts"][cut]["probe"]["evaluations"][regime]["cosine_pearson"]
            )
            for cut in CUTS
            for regime in REGIMES
        ),
        "shuffled_joint_pass": bool(
            evaluations["shuffled"]["joint_seed_pass"]
        ),
    }


def build_calibration_noise_breakpoint_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-calibration-degradation.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = [
        {
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "provenance": detail["provenance"],
            "gates": detail["gates"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM calibration-noise representation breakpoint",
            "category": "mechanistic_interpretability",
            "description": (
                "A preregistered frozen-checkpoint curve perturbs all observed "
                "calibration fields and measures the probe-defined quotient endpoint."
            ),
            "question": (
                "Do both calibrated mechanisms have a bounded stable interval over "
                "which cosine retention and conditional branch contraction coexist?"
            ),
            "prediction": (
                "Both arms pass at zero and some nonzero level, fail by q=4 without "
                "re-entry, and fail a shuffled-calibration control."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": "confirmed_bounded_stable_representation_breakpoints",
            "evidence_count": len(direct_tests),
            "direct_experiment_count": len(direct_tests),
            "independent_checkpoint_count": 5,
            "power_profile": "five_seed_preregistered_sequential_checkpoint_curve",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten selected d8/N3 structured checkpoints; joint synthetic "
                "calibration-error grid q=0--4; composition and extrapolation"
            ),
            "subclaims": {
                "analytic_representation_breakpoint": "supported_interval_1_to_2",
                "learned_representation_breakpoint": "supported_interval_0p5_to_1",
                "bounded_stable_breakpoints_both_arms": "supported",
                "gate_reentry": "rejected_both_arms",
                "shuffled_calibration_negative_control": "supported_zero_of_five_both_arms",
                "frozen_task_utility_robustness": "not_an_endpoint",
                "end_to_end_reference_robustness": "not_established",
            },
            "tags": [
                "tinyllm",
                "calibration-noise",
                "representation-breakpoint",
                "gauge-reference",
                "frozen-checkpoint",
                "conditional-branch",
                "multi-seed",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Frozen task-accuracy preservation as a primary endpoint.",
                "A real-instrument calibration uncertainty model.",
                "Population prevalence outside the retained checkpoint cohort.",
            ],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "The analytic and learned arms produced bounded monotone population "
                "breakpoint intervals [1,2] and [0.5,1], while both shuffled controls "
                "passed zero of five checkpoints."
            ),
            "confidence": 0.99,
            "confidence_assessment": "preregistered_five_seed_representation_gate",
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {"campaign": campaign["aggregates"]},
            "key_insights": [
                "Both structured mechanisms have finite, stable representation-only calibration breakpoints.",
                "The analytic interval is [1,2] and the learned interval is [0.5,1] on the registered joint-error scale.",
                "Conditional branch accuracy remains near chance; breakpoint failure is loss of the semantic cosine base.",
                "This endpoint omits frozen task-accuracy preservation and therefore does not contradict the stricter zero-radius task-plus-representation result.",
            ],
            "suggested_hypotheses": [
                "Readout-only noisy-reference calibration can close the gap between representation and complete task robustness.",
                "Orientation error sets the earliest complete-system failure even when the quotient representation remains usable."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
        },
    }


def build_calibration_noise_breakpoint_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        metrics = {
            "last_passing_level": float(summary["last_passing_level"]),
            "first_failed_level": float(summary["first_failed_level"]),
            "maximum_branch_accuracy": float(summary["maximum_branch_accuracy"]),
            "maximum_log_loss_gain": float(summary["maximum_log_loss_gain"]),
            "minimum_cosine_at_last_pass": float(
                summary["minimum_cosine_at_last_pass"]
            ),
            "shuffled_joint_pass": float(summary["shuffled_joint_pass"]),
            **{
                f"pass_q_{index}": float(value)
                for index, value in enumerate(summary["pass_pattern"])
            },
        }
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(summary["last_passing_level"]),
                model_architecture=[8, int(detail["_source_model_parameters"])],
                model_parameters=int(detail["_source_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "Inference-only joint calibration perturbation; no model or front end was trained.",
                    "The primary metric is representation-only and does not include frozen task utility.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "Task accuracy declines before the registered representation breakpoint and is descriptive only."
                ],
                timestamp=completed,
            )
        )
    return output


def store_calibration_noise_breakpoint_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_calibration_noise_breakpoint_meta_hypothesis(results_path)
    experiments = build_calibration_noise_breakpoint_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

        storage_root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(storage_root / "experiment_queue"),
                sent_dir=str(storage_root / "experiment_sent"),
                rejected_dir=str(storage_root / "experiment_rejected"),
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
        hypothesis_readback = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        experiment_readback = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in experiment_readback.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("calibration-noise ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_calibration_noise_breakpoint_experiment_results",
    "build_calibration_noise_breakpoint_meta_hypothesis",
    "store_calibration_noise_breakpoint_meta_hypothesis",
]
