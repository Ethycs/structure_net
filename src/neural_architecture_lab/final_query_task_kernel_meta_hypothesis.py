"""Build and store the TinyLLM final-query task/kernel result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-final-query-task-kernel-barycenter.v1"
HYPOTHESIS_ID = "tinyllm-final-query-task-kernel-barycenter-v1"
EVIDENCE_ROLE = "preregistered_underpowered_frozen_final_query_decomposition"
CLASSIFICATION = "task_rowspace_contains_material_fiber_chord"
CAMPAIGN_SHA256 = (
    "93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde"
)
DIAGNOSTICS_SHA256 = (
    "ad1798b960375503381cb59725dfb84db9d823f488ab8073fea178d0399e0974"
)
IMPLEMENTATION_SHA256 = (
    "2e6a8312d73c31be8dd0ea9df066dcd375a83c5bc3659d2436baa605554f2161"
)
RUNNER_SHA256 = (
    "52b59f96e53e76c1794e9d2d251760a76eb2300f8e64e70dbe165090a5899895"
)
PREREGISTRATION_SHA256 = (
    "6c99f8b4e6feb91fe5ca90a6846cc2b1472825874cd201d3fb5f0f51504347c3"
)
PARENT_CAMPAIGN_SHA256 = (
    "b5d6b613683326024eeb00944a3d0aba4dd7a251dd22f5fa7e8561b5e9d6aae4"
)
PARENT_D8_RESULT_SHA256 = (
    "79119a4fa6df16a2fe2d65c34ec1aca30072a0b53d3d8ea312c92b29f86d1476"
)
PARENT_D8_DIAGNOSTICS_SHA256 = (
    "ef86c31d418c04b7ffd4bec3b66135057c4ca825b3e4da8263a920cbce0de1dd"
)
PARENT_RUNNER_SHA256 = (
    "67e0854162cfd5ef4ff7dbc18f2cd2b7e9c6e88e10543e30bf0baf6b87b1c8ee"
)
CHECKPOINT_SHA256 = (
    "8170856da5e6b1f8b7a7f1c2c2121ffd4c53edaaf2ea5aabe01fa14d2f063f3f"
)
SCIENTIFIC_FINGERPRINT = (
    "a1655b1267cab7ae266ec00b13592b6f5f06df2f15e9cb77dff70087c08ecf2c"
)
REGIMES = ("training_support", "outside_range")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_final_query_task_kernel.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-final-query-task-kernel-preregistration.md"
)
PARENT_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/campaign_results.json"
)
PARENT_D8_RESULT_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/runs/d8/result.json"
)
PARENT_D8_DIAGNOSTICS_PATH = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/runs/d8/diagnostics.npz"
)
PARENT_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_task_relative_activation_barycenter.py"
)
CHECKPOINT_PATH = Path(
    "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/"
    "checkpoints/d8_cosine_interval_trained_seed7.pt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _require_source_digest(
    path: Path, expected: str, size: int, modified_ns: int
) -> None:
    del size, modified_ns
    if _sha256(path) != expected:
        raise ValueError(f"source artifact changed: {path}")


def _require_source(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"source artifact changed: {path}")
    stat = path.stat()
    _require_source_digest(path, expected, stat.st_size, stat.st_mtime_ns)


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


def _expected_configuration() -> dict[str, Any]:
    return {
        "accuracy_loss_ceiling": 0.03,
        "allow_underpowered": False,
        "baseline_accuracy_floor": 0.15,
        "batch_size": 64,
        "cross_entropy_increase_ceiling": 0.05,
        "decomposition_relative_error_ceiling": 1e-8,
        "device": "cuda:1",
        "fiber_pairs": 512,
        "finite_difference_pairs": 32,
        "finite_difference_relative_error_ceiling": 0.02,
        "finite_difference_step": 1e-3,
        "jacobian_atol": 1e-10,
        "jacobian_batch_pairs": 32,
        "jacobian_rtol": 1e-6,
        "kernel_contraction_ratio_ceiling": 0.25,
        "kernel_effect_mean_ratio_ceiling": 0.25,
        "kernel_effect_p95_ratio_ceiling": 0.75,
        "kernel_leakage_ceiling": 1e-6,
        "posterior_js_ceiling": 0.02,
        "random_js_disadvantage_floor": 0.01,
        "regimes": list(REGIMES),
        "replay_tolerance": 2e-6,
        "task_effect_mean_residual_ceiling": 0.25,
        "task_effect_median_cosine_floor": 0.9,
        "task_effect_p95_residual_ceiling": 0.75,
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    diagnostics_path = Path(campaign.get("artifacts", {}).get("diagnostics", ""))
    expected_summary = {
        "completed": 1,
        "excluded": 0,
        "failed": 0,
        "fitted_carriers": 0,
        "fitted_decoders": 0,
        "jacobian_cells": 1024,
        "requested": 1,
        "retries": 0,
        "reused": 0,
        "scheduled": 1,
        "trained_models": 0,
        "trained_probes": 0,
    }
    expected_top_gates = {
        "primary_hypothesis_pass": False,
        "state_unchanged": True,
        "validity": True,
    }
    source_hashes = campaign.get("source_hashes", {})
    source_digests = campaign.get("source_digests", {})
    state = campaign.get("state_record", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("classification") != CLASSIFICATION
        or campaign.get("scientific_fingerprint") != SCIENTIFIC_FINGERPRINT
        or campaign.get("configuration") != _expected_configuration()
        or campaign.get("summary") != expected_summary
        or campaign.get("gates") != expected_top_gates
        or set(campaign.get("regimes", {})) != set(REGIMES)
        or source_hashes.get("parent_campaign") != PARENT_CAMPAIGN_SHA256
        or source_hashes.get("parent_d8_result") != PARENT_D8_RESULT_SHA256
        or source_hashes.get("parent_d8_diagnostics")
        != PARENT_D8_DIAGNOSTICS_SHA256
        or source_hashes.get("checkpoint_d8_cosine_interval")
        != CHECKPOINT_SHA256
        or source_digests.get("runner") != RUNNER_SHA256
        or source_digests.get("preregistration") != PREREGISTRATION_SHA256
        or source_digests.get("parent_runner") != PARENT_RUNNER_SHA256
        or campaign.get("artifacts", {}).get("diagnostics_sha256")
        != DIAGNOSTICS_SHA256
        or not diagnostics_path.is_file()
        or _sha256(diagnostics_path) != DIAGNOSTICS_SHA256
        or state.get("unchanged") is not True
        or state.get("initial_model_state_sha256")
        != state.get("final_model_state_sha256")
        or state.get("initial_system_state_sha256")
        != state.get("final_system_state_sha256")
        or campaign.get("parent_summary")
        != {
            "classification": "observational_task_geometry_not_causally_sufficient",
            "mature_causal_front": None,
            "valid_d8_null": True,
        }
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid final-query task/kernel campaign {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (PARENT_CAMPAIGN_PATH, PARENT_CAMPAIGN_SHA256),
        (PARENT_D8_RESULT_PATH, PARENT_D8_RESULT_SHA256),
        (PARENT_D8_DIAGNOSTICS_PATH, PARENT_D8_DIAGNOSTICS_SHA256),
        (PARENT_RUNNER_PATH, PARENT_RUNNER_SHA256),
        (CHECKPOINT_PATH, CHECKPOINT_SHA256),
    ):
        _require_source(source_path, expected)

    expected_regime_gates = {
        "baseline_validity": True,
        "cohort_contract": True,
        "finite": True,
        "jacobian_numerical_contract": True,
        "kernel_pair_contraction": False,
        "kernel_task_sufficient": True,
        "primary_regime_gate": False,
        "random_projector_specificity": True,
        "replay": True,
        "task_effect_attribution": True,
        "task_only_causally_active": True,
        "validity": True,
    }
    config = campaign["configuration"]
    for regime in REGIMES:
        detail = campaign["regimes"][regime]
        arms = detail.get("arms", {})
        jacobian = detail.get("jacobian", {})
        replay = detail.get("replay", {})
        attribution = detail.get("attribution", {})
        geometry = detail.get("geometry", {})
        random_disadvantage = (
            float(arms["random_kernel"]["gates"]["mean_posterior_js"])
            - float(arms["kernel_only"]["gates"]["mean_posterior_js"])
        )
        if (
            detail.get("regime") != regime
            or detail.get("gates") != expected_regime_gates
            or detail.get("cohort_contract", {}).get("pass") is not True
            or set(arms) != {"full", "task_only", "kernel_only", "random_kernel"}
            or arms["kernel_only"]["gates"].get("task_sufficient") is not True
            or any(
                arms[name]["gates"].get("task_sufficient") is not False
                for name in ("full", "task_only", "random_kernel")
            )
            or float(geometry.get("kernel_pair_contraction_ratio", 0.0))
            <= float(config["kernel_contraction_ratio_ceiling"])
            or attribution.get("pass") is not True
            or random_disadvantage < float(config["random_js_disadvantage_floor"])
            or int(jacobian.get("minimum_rank", -1)) != 15
            or int(jacobian.get("maximum_rank", -1)) != 15
            or int(jacobian.get("random_control_rank", -1)) != 15
            or jacobian.get("rank_counts") != {"15": 512}
            or float(jacobian.get("maximum_finite_difference_relative_error", 1.0))
            > float(config["finite_difference_relative_error_ceiling"])
            or float(jacobian.get("maximum_reconstruction_relative_error", 1.0))
            > float(config["decomposition_relative_error_ceiling"])
            or float(jacobian.get("maximum_kernel_leakage", 1.0))
            > float(config["kernel_leakage_ceiling"])
            or replay.get("parent_replay_required") is not True
            or any(
                float(replay.get(name, 1.0)) != 0.0
                for name in (
                    "maximum_capture_replay_error",
                    "maximum_direct_replay_error",
                    "maximum_parent_baseline_error",
                    "maximum_parent_full_error",
                )
            )
        ):
            raise ValueError(
                f"invalid final-query task/kernel regime {regime} in {path}"
            )
    return campaign


def _summary(campaign: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "validity": float(campaign["gates"]["validity"]),
        "primary_hypothesis_pass": float(
            campaign["gates"]["primary_hypothesis_pass"]
        ),
        "state_unchanged": float(campaign["gates"]["state_unchanged"]),
        "fiber_pairs_per_regime": float(campaign["configuration"]["fiber_pairs"]),
        "jacobian_rank": 15.0,
    }
    for regime in REGIMES:
        detail = campaign["regimes"][regime]
        arms = detail["arms"]
        attribution = detail["attribution"]
        geometry = detail["geometry"]
        jacobian = detail["jacobian"]
        prefix = regime
        kernel_js = float(arms["kernel_only"]["gates"]["mean_posterior_js"])
        random_js = float(arms["random_kernel"]["gates"]["mean_posterior_js"])
        metrics.update(
            {
                f"{prefix}_baseline_accuracy": float(
                    detail["baseline"]["exact_bin_accuracy"]
                ),
                f"{prefix}_kernel_accuracy": float(
                    arms["kernel_only"]["metrics"]["exact_bin_accuracy"]
                ),
                f"{prefix}_kernel_cross_entropy_increase": float(
                    arms["kernel_only"]["gates"]["cross_entropy_increase"]
                ),
                f"{prefix}_kernel_posterior_js": kernel_js,
                f"{prefix}_full_posterior_js": float(
                    arms["full"]["gates"]["mean_posterior_js"]
                ),
                f"{prefix}_task_posterior_js": float(
                    arms["task_only"]["gates"]["mean_posterior_js"]
                ),
                f"{prefix}_random_posterior_js": random_js,
                f"{prefix}_random_js_disadvantage": random_js - kernel_js,
                f"{prefix}_kernel_task_sufficient": float(
                    arms["kernel_only"]["gates"]["task_sufficient"]
                ),
                f"{prefix}_remaining_pair_ratio": float(
                    geometry["kernel_pair_contraction_ratio"]
                ),
                f"{prefix}_removed_pair_fraction": 1.0
                - float(geometry["kernel_pair_contraction_ratio"]),
                f"{prefix}_task_displacement_norm_fraction": float(
                    geometry["mean_task_displacement_norm_fraction"]
                ),
                f"{prefix}_kernel_displacement_norm_fraction": float(
                    geometry["mean_kernel_displacement_norm_fraction"]
                ),
                f"{prefix}_mean_relative_task_residual": float(
                    attribution["mean_relative_task_residual"]
                ),
                f"{prefix}_p95_relative_task_residual": float(
                    attribution["p95_relative_task_residual"]
                ),
                f"{prefix}_median_task_full_cosine": float(
                    attribution["median_task_full_cosine"]
                ),
                f"{prefix}_mean_kernel_full_effect_ratio": float(
                    attribution["mean_kernel_full_effect_ratio"]
                ),
                f"{prefix}_p95_kernel_full_effect_ratio": float(
                    attribution["p95_kernel_full_effect_ratio"]
                ),
                f"{prefix}_maximum_finite_difference_relative_error": float(
                    jacobian["maximum_finite_difference_relative_error"]
                ),
                f"{prefix}_maximum_reconstruction_relative_error": float(
                    jacobian["maximum_reconstruction_relative_error"]
                ),
                f"{prefix}_maximum_kernel_leakage": float(
                    jacobian["maximum_kernel_leakage"]
                ),
            }
        )
    return metrics


def build_final_query_task_kernel_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-final-query-task-kernel.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    metrics = _summary(campaign)
    evidence = {
        "experiment_id": "tinyllm-final-query-task-kernel-d8-seed7",
        "preset": "d8",
        "seed": 7,
        "evidence_role": EVIDENCE_ROLE,
        "classification": campaign["classification"],
        "scientific_fingerprint": campaign["scientific_fingerprint"],
        "metrics": metrics,
    }
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM final-query task/kernel barycenter decomposition",
            "category": "mechanistic_interpretability",
            "description": (
                "An analytic no-fit decomposition tests whether the failed final "
                "query barycenter removes a small answer-sensitive component from "
                "an otherwise answer-map-null opposite-phase chord."
            ),
            "question": (
                "Does the local centered-answer-logit kernel contain most of the "
                "same-cosine phase chord while preserving the frozen posterior?"
            ),
            "prediction": (
                "The kernel-only patch preserves the task and leaves at most 25% "
                "of pair distance, while task-only and random rank-15 controls fail."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_valid_kernel_preservation_but_contraction_failed"
            ),
            "evidence_count": 1,
            "direct_quality_experiment_count": 1,
            "fixed_checkpoint_count": 1,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": (
                "one_seed_one_depth_two_shift_1024_fiber_local_causal_decomposition"
            ),
            "tested_scope": (
                "one retained d8 cosine seed-7 checkpoint, 512 exact fibers in "
                "each of two regimes, final post-MLP query activations, and the "
                "centered native 16-answer-logit map"
            ),
            "subclaims": {
                "local_task_kernel_preserves_output": "yes_both_regimes",
                "local_task_kernel_contains_most_fiber_chord": "no",
                "final_query_task_rowspace_contains_material_fiber_chord": "yes",
                "task_component_explains_full_output_change": "yes_both_regimes",
                "random_rank15_specificity": "yes_both_regimes",
                "same_scope_retraining": "no",
            },
            "tags": [
                "tinyllm",
                "task-activations",
                "task-kernel",
                "answer-logit-jacobian",
                "causal-patching",
                "fiber-chord",
                "frozen-checkpoint",
                "preregistered-null",
                "underpowered",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The answer-map kernel preserves the task in both shifts, but "
                "removing it leaves 80.6% and 84.7% of the paired chord, far above "
                "the preregistered 25% ceiling."
            ),
            "confidence": 0.98,
            "confidence_assessment": (
                "strong_exact_local_null_for_retained_d8_checkpoint_not_population_prevalence"
            ),
            "num_direct_experiments": 1,
            "descriptive_metrics": metrics,
            "key_insights": [
                "A genuine rank-497 local answer-map kernel is finite-patch task-preserving in both declared shifts.",
                "That kernel contains only a minority of the same-cosine phase-reflected query chord.",
                "The rank-15 task row space contains a material majority component of the chord despite identical targets.",
                "The task-only component reproduces the full nonlinear output change while a rank-matched random kernel does not preserve it.",
                "Conditional branch-probe collapse does not imply that exact fiber chords lie in the frozen output kernel.",
            ],
            "suggested_hypotheses": [
                "The one-dimensional semantic target quotient and the full 15-dimensional posterior quotient need not coincide.",
                "Useful task averaging can change answer-sensitive coordinates without producing autonomous frozen-computation closure.",
                "Future work must declare whether it preserves a scalar semantic coordinate or the complete native posterior before defining its task kernel.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": [evidence]},
        "source_artifacts": [
            str(results_path),
            str(Path(campaign["artifacts"]["diagnostics"])),
            str(report_path),
            str(PREREGISTRATION_PATH),
        ],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "diagnostics_sha256": DIAGNOSTICS_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "parent_d8_result_sha256": PARENT_D8_RESULT_SHA256,
            "parent_d8_diagnostics_sha256": PARENT_D8_DIAGNOSTICS_SHA256,
            "parent_runner_sha256": PARENT_RUNNER_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "scientific_fingerprint": SCIENTIFIC_FINGERPRINT,
        },
    }


def build_final_query_task_kernel_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    metrics = _summary(campaign)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id="tinyllm-final-query-task-kernel-d8-seed7",
            hypothesis_id=HYPOTHESIS_ID,
            metrics=metrics,
            primary_metric=0.0,
            model_architecture=[8, 512],
            model_parameters=50_964_992,
            training_time=float(campaign["analysis_seconds"]),
            model_checkpoint=str(CHECKPOINT_PATH),
            observations=[
                "One retained d8 seed-7 checkpoint was frozen; no parameter, carrier, probe, decoder, or threshold was fit.",
                "The analytic centered-answer-logit kernel preserved the frozen task in support and outside range.",
                "The task-only component explained the full nonlinear logit change with rank-matched random specificity.",
                f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
            ],
            anomalies=[
                "Kernel removal left 80.6% of the pair chord in support and 84.7% outside range, failing the 25% ceiling."
            ],
            timestamp=completed,
        )
    ]


def store_final_query_task_kernel_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_final_query_task_kernel_meta_hypothesis(results_path)
    experiments = build_final_query_task_kernel_experiment_results(
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
        results = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis.get("ids") != [HYPOTHESIS_ID]
            or len(results.get("ids", [])) != len(experiments)
            or {item.get("hypothesis_id") for item in results.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("final-query task/kernel ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(results["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_final_query_task_kernel_experiment_results",
    "build_final_query_task_kernel_meta_hypothesis",
    "store_final_query_task_kernel_meta_hypothesis",
]
