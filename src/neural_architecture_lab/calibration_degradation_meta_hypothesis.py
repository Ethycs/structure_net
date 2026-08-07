"""Build and store TinyLLM calibration-degradation causal evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-calibration-degradation-causal.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-reference-robustness-curve-v1"
EVIDENCE_ROLE = "preregistered_frozen_checkpoint_underpowered_causal_diagnostic"
IMPLEMENTATION_SHA256 = (
    "f273abf83231e4fc37583154c24158c81bba15a29e11ca3f96daf0b579be0f70"
)
CAMPAIGN_SHA256 = (
    "170d99553058ab544d95d6abf4d26ea1062bfe1b79fda2deefc531dfaebd0f6e"
)
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_MANIFEST_SHA256 = (
    "23598aaef5a0d16825ff9a928de57857e7bd58b10feab8853739448acfd983fc"
)
NOISE_SHA256 = (
    "1896c852644f724f5a2d214d5e69380eef3e46abb2e7b295eb0d4bace51a0c82"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
CUTS = ("frontend", "full")
NOISE_LEVELS = (
    "sigma_0p00",
    "sigma_0p05",
    "sigma_0p10",
    "sigma_0p20",
    "sigma_0p40",
    "sigma_0p80",
)
ABLATIONS = (
    "orientation_default",
    "speed_default",
    "amplitude_default",
    "offset_default",
    "drift_default",
    "all_default",
)
EXPECTED_ARMS = {
    "analytic_calibrated": {
        "noise_pass_counts": {
            "sigma_0p00": 5,
            "sigma_0p05": 1,
            "sigma_0p10": 0,
            "sigma_0p20": 0,
            "sigma_0p40": 0,
            "sigma_0p80": 0,
        },
        "ablation_pass_counts": {
            "orientation_default": 0,
            "speed_default": 0,
            "amplitude_default": 5,
            "offset_default": 0,
            "drift_default": 5,
            "all_default": 0,
        },
        "robust_radius": 0.0,
        "valid_checkpoint_count": 5,
    },
    "learned_calibrated_equivariant": {
        "noise_pass_counts": {
            "sigma_0p00": 5,
            "sigma_0p05": 4,
            "sigma_0p10": 0,
            "sigma_0p20": 0,
            "sigma_0p40": 0,
            "sigma_0p80": 0,
        },
        "ablation_pass_counts": {
            "orientation_default": 0,
            "speed_default": 2,
            "amplitude_default": 0,
            "offset_default": 0,
            "drift_default": 5,
            "all_default": 0,
        },
        "robust_radius": 0.05,
        "valid_checkpoint_count": 5,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    configuration = value.get("configuration", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or provenance.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or provenance.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or provenance.get("source_manifest_entries") != 30
        or provenance.get("noise_sha256") != NOISE_SHA256
        or tuple(configuration.get("conditions", ())) != CONDITIONS
        or tuple(configuration.get("seeds", ())) != SEEDS
        or tuple(configuration.get("noise_levels", ()))
        != (0, 0.05, 0.1, 0.2, 0.4, 0.8)
        or float(configuration.get("target_robust_radius", -1)) != 0.2
        or int(configuration.get("required_seed_passes", -1)) != 4
        or int(summary.get("requested", -1)) != 10
        or int(summary.get("completed", -1)) != 10
        or int(summary.get("failed", -1)) != 0
        or int(summary.get("excluded", -1)) != 0
        or int(summary.get("trained_models", -1)) != 0
        or int(summary.get("trained_frontends", -1)) != 0
        or int(summary.get("fitted_diagnostic_nulls", -1)) != 10
        or int(summary.get("fitted_diagnostic_suites", -1)) != 20
        or aggregates.get("classification") != "exact_calibration_required"
        or aggregates.get("primary_hypothesis_pass") is not False
        or aggregates.get("required_seed_passes") != 4
        or float(aggregates.get("target_robust_radius", -1)) != 0.2
        or aggregates.get("arms") != EXPECTED_ARMS
    ):
        raise ValueError(f"invalid calibration-degradation campaign {path}")
    return value


def _finite_metrics(detail: Mapping[str, Any]) -> bool:
    for cell in detail.get("cells", []):
        task = cell.get("task", {})
        for name in (
            "clean_exact_bin_accuracy",
            "exact_bin_accuracy",
            "exact_bin_accuracy_drop",
            "mean_circular_error_radians",
            "mean_target_cross_entropy",
        ):
            if not math.isfinite(float(task.get(name, math.nan))):
                return False
        for representation in cell.get("representation", {}).values():
            for value in representation.get("probe", {}).values():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    return False
            for value in representation.get("activation_drift", {}).values():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    return False
    return True


def _endpoint_pass(representation: Mapping[str, Any]) -> bool:
    probe = representation["probe"]
    return bool(
        float(probe["cosine_pearson"]) >= 0.90
        and float(probe["balanced_accuracy"]) <= 0.55
        and float(probe["conditional_log_loss_gain_over_cosine_only"]) <= 0.02
    )


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    indexed = {
        (item["condition"], int(item["seed"])): item
        for item in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"invalid calibration-degradation result index {path}")

    output = []
    fingerprints: set[str] = set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            activations_path = Path(entry["activations"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            source_result = Path(detail.get("source", {}).get("result", ""))
            cells = detail.get("cells", [])
            cell_index = {
                (cell.get("intervention"), cell.get("regime")): cell
                for cell in cells
            }
            expected_cells = {
                (intervention, regime)
                for intervention in (*NOISE_LEVELS, *ABLATIONS)
                for regime in REGIMES
            }
            source_detail = (
                json.loads(source_result.read_text(encoding="utf-8"))
                if source_result.is_file()
                else {}
            )
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or not activations_path.is_file()
                or _sha256(activations_path) != entry.get("activations_sha256")
                or detail.get("artifacts", {}).get("activations_sha256")
                != entry.get("activations_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("classification") != "checkpoint_fails_before_target"
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or not detail.get("data", {}).get("input_target_identity_pass")
                or detail.get("data", {}).get("noise_sha256") != NOISE_SHA256
                or not detail.get("clean_replay", {}).get("pass")
                or float(
                    detail.get("clean_replay", {}).get(
                        "maximum_absolute_metric_error", math.inf
                    )
                )
                != 0.0
                or set(cell_index) != expected_cells
                or not _finite_metrics(detail)
                or _sha256(source_result) != detail.get("source", {}).get("result_sha256")
                or source_detail.get("condition") != condition
                or int(source_detail.get("seed", -1)) != seed
                or not Path(detail.get("source", {}).get("model", "")).is_file()
                or not Path(detail.get("source", {}).get("frontend", "")).is_file()
                or any(
                    bool(representation.get("endpoint_pass"))
                    != _endpoint_pass(representation)
                    for cell in cells
                    for representation in cell.get("representation", {}).values()
                )
                or any(
                    not cell["representation"][cut]["endpoint_pass"]
                    for intervention in NOISE_LEVELS[:4]
                    for regime in REGIMES
                    for cut in CUTS
                    for cell in [cell_index[(intervention, regime)]]
                )
                or any(
                    float(cell_index[(intervention, regime)]["representation"][cut][
                        "probe"
                    ]["balanced_accuracy"])
                    > 0.55
                    or float(cell_index[(intervention, regime)]["representation"][cut][
                        "probe"
                    ]["conditional_log_loss_gain_over_cosine_only"])
                    > 0.02
                    for intervention in NOISE_LEVELS
                    for regime in REGIMES
                    for cut in CUTS
                )
            ):
                raise ValueError(f"invalid calibration-degradation result {result_path}")
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate calibration-degradation fingerprint {result_path}")
            fingerprints.add(fingerprint)
            detail["_source_model_parameters"] = int(source_detail["model_parameters"])
            output.append(detail)
    return output


def _cell_index(detail: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (cell["intervention"], cell["regime"]): cell for cell in detail["cells"]
    }


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    cells = _cell_index(detail)
    primary_cells = [
        cells[(intervention, regime)]
        for intervention in NOISE_LEVELS
        for regime in REGIMES
    ]
    target_cells = [cells[("sigma_0p20", regime)] for regime in REGIMES]
    target_representations = [
        cell["representation"][cut] for cell in target_cells for cut in CUTS
    ]
    return {
        "robust_radius": float(detail["robust_radius"]),
        "seed_pass_by_intervention": detail["seed_pass_by_intervention"],
        "target_representation_pass_count": sum(
            bool(value["endpoint_pass"]) for value in target_representations
        ),
        "target_task_shift_pass_count": sum(
            bool(cell["task"]["utility_pass"]) for cell in target_cells
        ),
        "target_minimum_cosine": min(
            float(value["probe"]["cosine_pearson"])
            for value in target_representations
        ),
        "maximum_primary_branch_accuracy": max(
            float(cell["representation"][cut]["probe"]["balanced_accuracy"])
            for cell in primary_cells
            for cut in CUTS
        ),
        "maximum_primary_log_loss_gain": max(
            float(
                cell["representation"][cut]["probe"][
                    "conditional_log_loss_gain_over_cosine_only"
                ]
            )
            for cell in primary_cells
            for cut in CUTS
        ),
        "target_maximum_task_accuracy_drop": max(
            float(cell["task"]["exact_bin_accuracy_drop"]) for cell in target_cells
        ),
        "target_mean_full_depth_cka": sum(
            float(
                cell["representation"]["full"]["activation_drift"][
                    "centered_linear_cka_to_clean"
                ]
            )
            for cell in target_cells
        )
        / len(target_cells),
    }


def _campaign_split_summary(details: list[Mapping[str, Any]]) -> dict[str, Any]:
    representation_target_passes = 0
    representation_through_target_passes = 0
    primary_branch_passes = 0
    primary_log_loss_passes = 0
    target_task_shift_passes = {condition: 0 for condition in CONDITIONS}
    first_noise_task_shift_passes = {condition: 0 for condition in CONDITIONS}
    for detail in details:
        cells = _cell_index(detail)
        for intervention in NOISE_LEVELS:
            for regime in REGIMES:
                cell = cells[(intervention, regime)]
                for cut in CUTS:
                    representation = cell["representation"][cut]
                    primary_branch_passes += int(
                        representation["probe"]["balanced_accuracy"] <= 0.55
                    )
                    primary_log_loss_passes += int(
                        representation["probe"][
                            "conditional_log_loss_gain_over_cosine_only"
                        ]
                        <= 0.02
                    )
                    if intervention in NOISE_LEVELS[:4]:
                        representation_through_target_passes += int(
                            representation["endpoint_pass"]
                        )
                    if intervention == "sigma_0p20":
                        representation_target_passes += int(
                            representation["endpoint_pass"]
                        )
                if intervention == "sigma_0p20":
                    target_task_shift_passes[detail["condition"]] += int(
                        cell["task"]["utility_pass"]
                    )
                if intervention == "sigma_0p05":
                    first_noise_task_shift_passes[detail["condition"]] += int(
                        cell["task"]["utility_pass"]
                    )
    return {
        "target_representation_passes": representation_target_passes,
        "target_representation_cells": 40,
        "through_target_representation_passes": representation_through_target_passes,
        "through_target_representation_cells": 160,
        "primary_branch_passes": primary_branch_passes,
        "primary_branch_cells": 240,
        "primary_log_loss_passes": primary_log_loss_passes,
        "primary_log_loss_cells": 240,
        "target_task_shift_passes": target_task_shift_passes,
        "target_task_shift_cells_per_arm": 10,
        "first_noise_task_shift_passes": first_noise_task_shift_passes,
        "first_noise_task_shift_cells_per_arm": 10,
    }


def build_calibration_degradation_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-calibration-degradation-causal.md"
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
            "source": detail["source"],
            "clean_replay": detail["clean_replay"],
            "classification": detail["classification"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM calibrated-reference robustness curve",
            "category": "mechanistic_interpretability",
            "description": (
                "A preregistered inference-only causal curve perturbs the calibration "
                "packets of ten retained analytic and learned-equivariant TinyLLM systems."
            ),
            "question": (
                "Do both structured systems retain the joint quotient and frozen task "
                "utility through calibration-noise level 0.20?"
            ),
            "prediction": (
                "Both arms reach robust radius at least 0.20 in four of five seeds, "
                "remain within one grid level, and fail the all-default control."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_exact_calibration_required",
            "evidence_count": len(direct_tests),
            "direct_experiment_count": len(direct_tests),
            "independent_seed_count": 5,
            "power_profile": "five_seed_preregistered_retained_checkpoint_diagnostic",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained d8/N3 structured checkpoints; analytic and learned "
                "calibrated front ends; two shifts; synthetic calibration sigma 0--0.80"
            ),
            "subclaims": {
                "analytic_robust_radius_at_least_0p20": "rejected_zero_radius",
                "learned_robust_radius_at_least_0p20": "rejected_radius_0p05",
                "full_preregistered_hypothesis": "rejected",
                "representation_joint_gate_through_0p20": "supported_160_of_160_cells",
                "conditional_branch_contraction_across_primary_curve": (
                    "supported_240_of_240_cells_under_frozen_observer"
                ),
                "task_utility_at_0p20": "rejected_both_arms_zero_of_five_seeds",
                "all_default_dynamic_range": "supported_both_arms_zero_of_five_pass",
                "learned_frontend_tracks_analytic_robustness_target": "rejected",
                "failure_localizes_first_to_frozen_task_output": "supported",
            },
            "tags": [
                "tinyllm",
                "calibration-noise",
                "gauge-reference",
                "frozen-checkpoint",
                "causal-intervention",
                "activation-geometry",
                "readout-calibration",
                "multi-seed",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "A calibrated real-instrument uncertainty model.",
                "Whether a no-fit continuous readout transport repairs noisy utility.",
                "Whether perturbation-aware retraining would improve robustness.",
                "Natural-language tasks or architecture-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The analytic and learned robust radii were 0.00 and 0.05, below "
                "the preregistered 0.20 target; failure occurred in frozen exact-bin "
                "utility before the representation endpoint failed."
            ),
            "confidence": 0.99,
            "confidence_assessment": "preregistered_five_seed_joint_gate_rejection",
            "independent_seed_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "representation_task_split": split,
            },
            "key_insights": [
                "Both structured arms require more accurate calibration than sigma 0.20 for frozen task utility.",
                "All 160 representation cells through sigma 0.20 retain cosine and contract conditional branch under the frozen probes.",
                "All 240 primary branch and conditional-log-loss cells pass across the complete noise curve.",
                "Task utility is the first failing endpoint: at sigma 0.20 the analytic arm passes 0/10 shift cells and the learned arm 1/10.",
                "The learned arm is modestly more robust at sigma 0.05 but does not approach the target radius.",
                "Orientation and offset are indispensable to both systems; drift is dispensable, while amplitude dependence differs by arm.",
            ],
            "suggested_hypotheses": [
                "A no-fit continuous task-coordinate transport can recover noisy-calibration utility while the quotient probe still passes.",
                "If an oracle scalar recalibration succeeds, a prospective calibration-aware readout can improve robustness without changing the invariant front end.",
                "If the oracle fails, improving the orientation/speed/offset reference process is required before model optimization.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "noise_sha256": NOISE_SHA256,
        },
    }


def build_calibration_degradation_experiment_results(
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
            "robust_radius": float(summary["robust_radius"]),
            "clean_replay_pass": float(detail["clean_replay"]["pass"]),
            "target_representation_pass_count": float(
                summary["target_representation_pass_count"]
            ),
            "target_task_shift_pass_count": float(
                summary["target_task_shift_pass_count"]
            ),
            "target_minimum_cosine": float(summary["target_minimum_cosine"]),
            "maximum_primary_branch_accuracy": float(
                summary["maximum_primary_branch_accuracy"]
            ),
            "maximum_primary_log_loss_gain": float(
                summary["maximum_primary_log_loss_gain"]
            ),
            "target_maximum_task_accuracy_drop": float(
                summary["target_maximum_task_accuracy_drop"]
            ),
            "target_mean_full_depth_cka": float(
                summary["target_mean_full_depth_cka"]
            ),
            **{
                f"pass_{name}": float(value)
                for name, value in summary["seed_pass_by_intervention"].items()
            },
        }
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(detail["robust_radius"]),
                model_architecture=[8, int(detail["_source_model_parameters"])],
                model_parameters=int(detail["_source_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["source"]["model"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "Inference-only calibration intervention; no model, front end, or task head was trained.",
                    "Clean observers were frozen across every perturbation and clean replay error was zero.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "The representation endpoint remains valid through sigma 0.20 while frozen task utility fails earlier."
                ],
                timestamp=completed,
            )
        )
    return output


def store_calibration_degradation_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_calibration_degradation_meta_hypothesis(results_path)
    experiments = build_calibration_degradation_experiment_results(record, results_path)
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
            raise RuntimeError("calibration-degradation ChromaDB read-back failed")
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
    "build_calibration_degradation_experiment_results",
    "build_calibration_degradation_meta_hypothesis",
    "store_calibration_degradation_meta_hypothesis",
]
