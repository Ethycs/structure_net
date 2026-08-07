"""Build and store TinyLLM calibration-orientation noise evidence."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-calibrated-orientation-noise.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-orientation-noise-radius-v1"
EVIDENCE_ROLE = "preregistered_frozen_checkpoint_causal_titration"
IMPLEMENTATION_SHA256 = (
    "990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70"
)
CAMPAIGN_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
NOISE_SHA256 = (
    "b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
SIGMAS = (0.0, 0.035, 0.087, 0.175, 0.349, 0.524, 0.785)
REGIMES = ("composition", "extrapolation")
CUTS = ("frontend", "full")
EXPECTED_GATES = {
    "all_finite_contract": True,
    "clean_joint_pass": True,
    "feature_width_contract": True,
    "noise_contract": True,
    "source_provenance_contract": True,
    "validity_contract": True,
}
EXPECTED_COMPLETE_COUNTS = {
    "analytic_calibrated": (5, 0, 0, 0, 0, 0, 0),
    "learned_calibrated_equivariant": (5, 2, 0, 0, 0, 0, 0),
}
EXPECTED_REPRESENTATION_PATTERN = (True, True, True, True, False, False, False)


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


def _representation_cell_pass(value: Mapping[str, Any]) -> bool:
    return bool(
        float(value["cosine_pearson"]) >= 0.90
        and float(value["balanced_accuracy"]) <= 0.55
        and float(value["conditional_log_loss_gain_over_cosine_only"]) <= 0.02
    )


def _level_contract(level: Mapping[str, Any]) -> bool:
    representation_passes = []
    for cut in CUTS:
        evaluations = level.get("cuts", {}).get(cut, {}).get("probe", {}).get(
            "evaluations", {}
        )
        declared = level.get("cuts", {}).get(cut, {}).get("passes", {})
        for regime in REGIMES:
            actual = _representation_cell_pass(evaluations.get(regime, {}))
            if declared.get(regime) is not actual:
                return False
            representation_passes.append(actual)
    representation_gate = all(representation_passes)
    task_gate = all(
        float(level.get("accuracy_loss_from_clean", {}).get(regime, math.inf))
        <= 0.03
        for regime in REGIMES
    )
    return bool(
        level.get("finite_contract") is True
        and level.get("representation_gate") is representation_gate
        and level.get("task_gate") is task_gate
        and level.get("complete_gate") is (representation_gate and task_gate)
        and _finite_numbers(level)
    )


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    configuration = value.get("configuration", {})
    artifacts = value.get("artifacts", {})
    source_campaign = Path(provenance.get("source_campaign", ""))
    noise_arrays = Path(artifacts.get("noise_arrays", ""))
    arms = aggregates.get("arms", {})
    aggregate_contract = set(arms) == set(CONDITIONS)
    for condition in CONDITIONS:
        arm = arms.get(condition, {})
        levels = arm.get("levels", [])
        aggregate_contract = bool(
            aggregate_contract
            and len(levels) == len(SIGMAS)
            and tuple(float(item.get("sigma_radians", math.nan)) for item in levels)
            == SIGMAS
            and tuple(int(item.get("pass_count", -1)) for item in levels)
            == EXPECTED_COMPLETE_COUNTS[condition]
            and all(
                bool(item.get("population_pass")) == (int(item["pass_count"]) >= 4)
                for item in levels
            )
            and float(arm.get("robustness_radius_radians", math.nan)) == 0.0
            and arm.get("nonmonotone") is False
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
        or artifacts.get("noise_arrays_sha256") != NOISE_SHA256
        or not noise_arrays.is_file()
        or _sha256(noise_arrays) != NOISE_SHA256
        or tuple(configuration.get("seeds", ())) != SEEDS
        or tuple(float(item) for item in configuration.get("sigmas", ())) != SIGMAS
        or int(configuration.get("required_seed_passes", -1)) != 4
        or float(configuration.get("accuracy_loss_ceiling", -1)) != 0.03
        or float(configuration.get("cosine_floor", -1)) != 0.90
        or float(configuration.get("branch_accuracy_ceiling", -1)) != 0.55
        or float(configuration.get("log_loss_gain_ceiling", -1)) != 0.02
        or int(summary.get("requested", -1)) != 10
        or int(summary.get("completed", -1)) != 10
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("trained_frontends", -1)) != 0
        or int(summary.get("fitted_diagnostic_probe_sets", -1)) != 210
        or int(summary.get("primary_representation_cells", -1)) != 280
        or aggregates.get("classification") != "reference_precision_critical"
        or aggregates.get("clean_replay_contract") is not True
        or aggregates.get("clean_replay_pass_counts")
        != {condition: 5 for condition in CONDITIONS}
        or int(aggregates.get("required_seed_passes", -1)) != 4
        or not aggregate_contract
    ):
        raise ValueError(f"invalid calibration-orientation campaign {path}")
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
        raise ValueError(f"invalid calibration-orientation result index {path}")

    output = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            source_result_path = Path(provenance.get("source_result", ""))
            model_path = Path(provenance.get("model_checkpoint", ""))
            frontend_path = Path(provenance.get("frontend_checkpoint", ""))
            noise_path = Path(detail.get("noise_arrays", ""))
            levels = detail.get("levels", [])
            source_result = (
                json.loads(source_result_path.read_text(encoding="utf-8"))
                if source_result_path.is_file()
                else {}
            )
            level_sigmas = tuple(
                float(level.get("sigma_radians", math.nan)) for level in levels
            )
            representation_pattern = tuple(
                bool(level.get("representation_gate")) for level in levels
            )
            pass_pattern = tuple(bool(item) for item in detail.get("pass_pattern", []))
            radius = 0.0
            for sigma, passed in zip(SIGMAS, pass_pattern):
                if not passed:
                    break
                radius = sigma
            expected_pattern = tuple(
                bool(
                    campaign["aggregates"]["arms"][condition]["levels"][index][
                        "pass_by_seed"
                    ][str(seed)]
                )
                for index in range(len(SIGMAS))
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
                or detail.get("nonmonotone") is not False
                or level_sigmas != SIGMAS
                or not all(_level_contract(level) for level in levels)
                or representation_pattern != EXPECTED_REPRESENTATION_PATTERN
                or pass_pattern != expected_pattern
                or float(detail.get("robustness_radius_radians", math.nan)) != radius
                or float(entry.get("robustness_radius_radians", math.nan)) != radius
                or entry.get("pass_pattern") != list(pass_pattern)
                or entry.get("nonmonotone") is not False
                or entry.get("gates") != EXPECTED_GATES
                or detail.get("noise_arrays_sha256") != NOISE_SHA256
                or not noise_path.is_file()
                or _sha256(noise_path) != NOISE_SHA256
                or not source_result_path.is_file()
                or _sha256(source_result_path)
                != provenance.get("source_result_sha256")
                or source_result.get("condition") != condition
                or int(source_result.get("seed", -1)) != seed
                or not model_path.is_file()
                or _sha256(model_path) != provenance.get("model_checkpoint_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path)
                != provenance.get("frontend_checkpoint_sha256")
                or not _finite_numbers(detail)
            ):
                raise ValueError(
                    f"invalid calibration-orientation result {result_path}"
                )
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(
                    f"duplicate calibration-orientation fingerprint {result_path}"
                )
            fingerprints.add(fingerprint)
            detail["_source_model_parameters"] = int(source_result["model_parameters"])
            output.append(detail)
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    levels = detail["levels"]
    first_noise = levels[1]
    last_representation_pass = levels[3]
    first_representation_failure = levels[4]
    primary_evaluations = [
        level["cuts"][cut]["probe"]["evaluations"][regime]
        for level in levels
        for cut in CUTS
        for regime in REGIMES
    ]
    return {
        "complete_robustness_radius_radians": float(
            detail["robustness_radius_radians"]
        ),
        "complete_pass_pattern": detail["pass_pattern"],
        "first_noise_representation_pass": bool(
            first_noise["representation_gate"]
        ),
        "first_noise_task_pass": bool(first_noise["task_gate"]),
        "first_noise_complete_pass": bool(first_noise["complete_gate"]),
        "first_noise_maximum_task_accuracy_loss": max(
            float(value) for value in first_noise["accuracy_loss_from_clean"].values()
        ),
        "first_noise_minimum_cosine": min(
            float(first_noise["cuts"][cut]["probe"]["evaluations"][regime][
                "cosine_pearson"
            ])
            for cut in CUTS
            for regime in REGIMES
        ),
        "maximum_primary_branch_accuracy": max(
            float(value["balanced_accuracy"]) for value in primary_evaluations
        ),
        "maximum_primary_log_loss_gain": max(
            float(value["conditional_log_loss_gain_over_cosine_only"])
            for value in primary_evaluations
        ),
        "last_representation_pass_sigma_radians": float(
            last_representation_pass["sigma_radians"]
        ),
        "first_representation_failure_sigma_radians": float(
            first_representation_failure["sigma_radians"]
        ),
    }


def _campaign_split_summary(details: list[Mapping[str, Any]]) -> dict[str, Any]:
    representation_passes = 0
    branch_passes = 0
    log_loss_passes = 0
    first_noise_task_passes = {condition: 0 for condition in CONDITIONS}
    first_noise_complete_passes = {condition: 0 for condition in CONDITIONS}
    for detail in details:
        first_noise = detail["levels"][1]
        first_noise_task_passes[detail["condition"]] += int(first_noise["task_gate"])
        first_noise_complete_passes[detail["condition"]] += int(
            first_noise["complete_gate"]
        )
        for level in detail["levels"]:
            for cut in CUTS:
                for regime in REGIMES:
                    evaluation = level["cuts"][cut]["probe"]["evaluations"][regime]
                    representation_passes += int(_representation_cell_pass(evaluation))
                    branch_passes += int(float(evaluation["balanced_accuracy"]) <= 0.55)
                    log_loss_passes += int(
                        float(
                            evaluation[
                                "conditional_log_loss_gain_over_cosine_only"
                            ]
                        )
                        <= 0.02
                    )
    return {
        "representation_passes": representation_passes,
        "representation_cells": 280,
        "representation_passes_through_ten_degrees": sum(
            int(
                _representation_cell_pass(
                    detail["levels"][index]["cuts"][cut]["probe"][
                        "evaluations"
                    ][regime]
                )
            )
            for detail in details
            for index in range(4)
            for cut in CUTS
            for regime in REGIMES
        ),
        "representation_cells_through_ten_degrees": 160,
        "branch_passes": branch_passes,
        "branch_cells": 280,
        "log_loss_passes": log_loss_passes,
        "log_loss_cells": 280,
        "first_noise_task_passes": first_noise_task_passes,
        "first_noise_complete_passes": first_noise_complete_passes,
    }


def build_calibration_orientation_noise_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-calibration-orientation-noise.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    split = _campaign_split_summary(details)
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
            "name": "TinyLLM calibrated-orientation noise radius",
            "category": "mechanistic_interpretability",
            "description": (
                "A preregistered inference-only intervention titrates a common "
                "orientation-reference error in ten frozen calibrated TinyLLM systems."
            ),
            "question": (
                "How much orientation-reference error can the validated analytic and "
                "learned front ends tolerate before the joint representation and "
                "frozen task gate fails?"
            ),
            "prediction": (
                "The registered curve selects one prespecified robustness class from "
                "clean-only through learned/analytic radius matching."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": "confirmed_reference_precision_critical",
            "evidence_count": len(direct_tests),
            "direct_experiment_count": len(direct_tests),
            "independent_checkpoint_count": 5,
            "power_profile": "five_seed_preregistered_retained_checkpoint_titration",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained d8/N3 calibrated checkpoints; orientation-only Gaussian "
                "reference error from zero through 0.785 radians; two shifts and cuts"
            ),
            "subclaims": {
                "analytic_complete_radius": "zero_population_radius",
                "learned_complete_radius": "zero_population_radius",
                "reference_precision_critical": "supported_both_arms",
                "representation_quotient_through_ten_degrees": (
                    "supported_160_of_160_cells"
                ),
                "representation_quotient_at_twenty_degrees": (
                    "rejected_zero_of_ten_checkpoints"
                ),
                "conditional_branch_contraction_across_curve": (
                    "supported_280_of_280_cells_under_registered_probes"
                ),
                "first_failure_localizes_to_frozen_task_output": "supported",
                "full_tinyllm_retraining_needed": "not_supported",
            },
            "tags": [
                "tinyllm",
                "orientation-noise",
                "gauge-reference",
                "frozen-checkpoint",
                "causal-titration",
                "quotient-geometry",
                "readout-calibration",
                "multi-seed",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A real-instrument orientation uncertainty model.",
                "Whether readout-only noisy-reference calibration restores utility.",
                "Robustness to missing or noisy non-orientation calibration fields.",
                "Natural-language tasks or architecture-population prevalence.",
            ],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "Both arms failed the four-of-five complete gate at the first nonzero "
                "registered error (0.035 radians), selecting the preregistered "
                "reference_precision_critical class; representation gates still passed "
                "in every checkpoint through 0.175 radians."
            ),
            "confidence": 0.99,
            "confidence_assessment": "preregistered_five_seed_causal_classification",
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "representation_task_split": split,
            },
            "key_insights": [
                "The complete robustness radius is zero for both structured mechanisms on the registered grid.",
                "At about two degrees, all representation cells pass but the analytic task gate passes 0/5 and the learned gate 2/5.",
                "The quotient representation survives through ten degrees in all ten checkpoints and fails by twenty degrees.",
                "Conditional branch accuracy and conditional log-loss pass all 280 primary cells; failure is semantic-coordinate loss, never branch re-emergence.",
                "The earliest causal bottleneck is chart-to-bin calibration rather than quotient formation.",
            ],
            "suggested_hypotheses": [
                "Readout-only training on declared orientation noise restores exact-bin utility while the frozen quotient representation already passes.",
                "A capacity-matched scalar calibration should recover utility if the defect is only circular-boundary miscalibration.",
                "If an oracle readout cannot recover the task, reference measurement rather than decoder calibration is limiting."
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
            "noise_sha256": NOISE_SHA256,
        },
    }


def build_calibration_orientation_noise_experiment_results(
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
            "complete_robustness_radius_radians": float(
                summary["complete_robustness_radius_radians"]
            ),
            "first_noise_representation_pass": float(
                summary["first_noise_representation_pass"]
            ),
            "first_noise_task_pass": float(summary["first_noise_task_pass"]),
            "first_noise_complete_pass": float(
                summary["first_noise_complete_pass"]
            ),
            "first_noise_maximum_task_accuracy_loss": float(
                summary["first_noise_maximum_task_accuracy_loss"]
            ),
            "first_noise_minimum_cosine": float(
                summary["first_noise_minimum_cosine"]
            ),
            "maximum_primary_branch_accuracy": float(
                summary["maximum_primary_branch_accuracy"]
            ),
            "maximum_primary_log_loss_gain": float(
                summary["maximum_primary_log_loss_gain"]
            ),
            "last_representation_pass_sigma_radians": float(
                summary["last_representation_pass_sigma_radians"]
            ),
            "first_representation_failure_sigma_radians": float(
                summary["first_representation_failure_sigma_radians"]
            ),
            **{
                f"complete_pass_sigma_{index}": float(value)
                for index, value in enumerate(summary["complete_pass_pattern"])
            },
        }
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(detail["robustness_radius_radians"]),
                model_architecture=[8, int(detail["_source_model_parameters"])],
                model_parameters=int(detail["_source_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "Inference-only orientation-reference intervention; no model or front end was trained.",
                    "Pair-shared common noise and fresh per-level diagnostic probes were used.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "The complete gate fails before the representation quotient: task calibration is the first bottleneck."
                ],
                timestamp=completed,
            )
        )
    return output


def store_calibration_orientation_noise_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_calibration_orientation_noise_meta_hypothesis(results_path)
    experiments = build_calibration_orientation_noise_experiment_results(
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
            raise RuntimeError("calibration-orientation ChromaDB read-back failed")
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
    "build_calibration_orientation_noise_experiment_results",
    "build_calibration_orientation_noise_meta_hypothesis",
    "store_calibration_orientation_noise_meta_hypothesis",
]
