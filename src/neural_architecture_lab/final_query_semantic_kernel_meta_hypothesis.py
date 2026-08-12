"""Build and store the TinyLLM final-query semantic-kernel result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-final-query-semantic-kernel.v1"
HYPOTHESIS_ID = "tinyllm-final-query-semantic-kernel-v1"
EVIDENCE_ROLE = "preregistered_underpowered_frozen_semantic_posterior_separation"
CLASSIFICATION = "scalar_and_complete_posterior_quotients_not_separated"
CAMPAIGN_SHA256 = (
    "e668f2d9d45212334a24d10b420977927e68e280b5241745578eb46bf29558c8"
)
DIAGNOSTICS_SHA256 = (
    "a2aa5c52f8dbba7027f68012a36ec1d6533e9846813b957ae1797a5025acf394"
)
IMPLEMENTATION_SHA256 = (
    "f634e645cff4fa1e43cd727ed11efe040e21c642af7ab61061b3730f85645e13"
)
RUNNER_SHA256 = (
    "34ef7544fe8660a8ddbbfb048595ed61be58df048e8b0f649e82bd75ba53c449"
)
PREREGISTRATION_SHA256 = (
    "fcd7e6aedb6e7d53de111fb0ffedafda86f8702bb538ad7d06d60e754e3d7e59"
)
POSTERIOR_CAMPAIGN_SHA256 = (
    "93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde"
)
POSTERIOR_DIAGNOSTICS_SHA256 = (
    "ad1798b960375503381cb59725dfb84db9d823f488ab8073fea178d0399e0974"
)
POSTERIOR_RUNNER_SHA256 = (
    "52b59f96e53e76c1794e9d2d251760a76eb2300f8e64e70dbe165090a5899895"
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
CHECKPOINT_SHA256 = (
    "8170856da5e6b1f8b7a7f1c2c2121ffd4c53edaaf2ea5aabe01fa14d2f063f3f"
)
SCIENTIFIC_FINGERPRINT = (
    "09daf6331f0234888f56d0abbcba2d2cbd8d6fb8c3d90bd734655696263693de"
)
REGIMES = ("training_support", "outside_range")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_final_query_semantic_kernel.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-final-query-semantic-kernel-preregistration.md"
)
POSTERIOR_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_final_query_task_kernel.py"
)
POSTERIOR_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/campaign_results.json"
)
POSTERIOR_DIAGNOSTICS_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/diagnostics.npz"
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
        "material_full_effect_floor": 1e-5,
        "material_row_fraction_floor": 0.10,
        "nesting_leakage_ceiling": 1e-8,
        "posterior_js_ceiling": 0.02,
        "posterior_kernel_ratio_floor": 0.75,
        "random_semantic_disadvantage_floor": 0.01,
        "regimes": list(REGIMES),
        "replay_tolerance": 2e-6,
        "root_mean_squared_error_increase_ceiling": 0.01,
        "semantic_mean_change_ceiling": 0.01,
        "semantic_p95_change_ceiling": 0.03,
        "task_effect_mean_residual_ceiling": 0.25,
        "task_effect_p95_residual_ceiling": 0.75,
        "task_effect_sign_agreement_floor": 0.90,
        "task_map_score_loss_ceiling": 0.01,
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
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
    diagnostics_path = Path(campaign.get("artifacts", {}).get("diagnostics", ""))
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("scientific_fingerprint") != SCIENTIFIC_FINGERPRINT
        or campaign.get("classification") != CLASSIFICATION
        or campaign.get("configuration") != _expected_configuration()
        or campaign.get("summary") != expected_summary
        or campaign.get("gates") != expected_top_gates
        or set(campaign.get("regimes", {})) != set(REGIMES)
        or source_hashes.get("posterior_campaign")
        != POSTERIOR_CAMPAIGN_SHA256
        or source_hashes.get("posterior_diagnostics")
        != POSTERIOR_DIAGNOSTICS_SHA256
        or source_hashes.get("parent_campaign") != PARENT_CAMPAIGN_SHA256
        or source_hashes.get("parent_d8_result") != PARENT_D8_RESULT_SHA256
        or source_hashes.get("parent_d8_diagnostics")
        != PARENT_D8_DIAGNOSTICS_SHA256
        or source_hashes.get("checkpoint_d8_cosine_interval")
        != CHECKPOINT_SHA256
        or source_digests.get("runner") != RUNNER_SHA256
        or source_digests.get("preregistration") != PREREGISTRATION_SHA256
        or source_digests.get("posterior_runner") != POSTERIOR_RUNNER_SHA256
        or campaign.get("artifacts", {}).get("diagnostics_sha256")
        != DIAGNOSTICS_SHA256
        or not diagnostics_path.is_file()
        or _sha256(diagnostics_path) != DIAGNOSTICS_SHA256
        or state.get("unchanged") is not True
        or state.get("initial_model_state_sha256")
        != state.get("final_model_state_sha256")
        or state.get("initial_system_state_sha256")
        != state.get("final_system_state_sha256")
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid final-query semantic-kernel campaign {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (POSTERIOR_RUNNER_PATH, POSTERIOR_RUNNER_SHA256),
        (POSTERIOR_CAMPAIGN_PATH, POSTERIOR_CAMPAIGN_SHA256),
        (POSTERIOR_DIAGNOSTICS_PATH, POSTERIOR_DIAGNOSTICS_SHA256),
        (PARENT_CAMPAIGN_PATH, PARENT_CAMPAIGN_SHA256),
        (PARENT_D8_RESULT_PATH, PARENT_D8_RESULT_SHA256),
        (PARENT_D8_DIAGNOSTICS_PATH, PARENT_D8_DIAGNOSTICS_SHA256),
        (CHECKPOINT_PATH, CHECKPOINT_SHA256),
    ):
        _require_source(source_path, expected)

    common_gates = {
        "baseline_validity": True,
        "cohort_contract": True,
        "finite": True,
        "jacobian_numerical_and_nesting_contract": True,
        "posterior_kernel_noncontraction_replay": True,
        "primary_regime_gate": False,
        "random_projector_specificity": True,
        "replay": True,
        "semantic_kernel_pair_contraction": False,
        "semantic_kernel_scalar_preservation": False,
        "semantic_only_causally_active": True,
        "validity": True,
    }
    for regime in REGIMES:
        detail = campaign["regimes"][regime]
        gates = detail.get("gates", {})
        expected_gates = {
            **common_gates,
            "complete_posterior_separation": regime == "outside_range",
            "semantic_effect_attribution": regime == "training_support",
        }
        jacobian = detail.get("jacobian", {})
        replay = detail.get("replay", {})
        arms = detail.get("arms", {})
        geometry = detail.get("geometry", {})
        if (
            detail.get("regime") != regime
            or gates != expected_gates
            or detail.get("cohort_contract", {}).get("pass") is not True
            or set(arms)
            != {"full", "semantic_only", "semantic_kernel", "random_kernel"}
            or any(
                arms[name]["scalar_preservation"].get("pass") is not False
                for name in arms
            )
            or float(
                geometry.get("semantic_kernel_pair_contraction_ratio", 0.0)
            )
            <= 0.25
            or float(
                geometry.get("posterior_kernel_pair_contraction_ratio", 0.0)
            )
            < 0.75
            or int(jacobian.get("minimum_scalar_rank", -1)) != 1
            or int(jacobian.get("maximum_scalar_rank", -1)) != 1
            or jacobian.get("scalar_rank_counts") != {"1": 512}
            or int(jacobian.get("minimum_posterior_rank", -1)) != 15
            or int(jacobian.get("maximum_posterior_rank", -1)) != 15
            or jacobian.get("posterior_rank_counts") != {"15": 512}
            or float(jacobian.get("maximum_finite_difference_relative_error", 1.0))
            > 0.02
            or float(jacobian.get("maximum_reconstruction_relative_error", 1.0))
            > 1e-8
            or float(jacobian.get("maximum_kernel_leakage", 1.0)) > 1e-6
            or float(
                jacobian.get("maximum_posterior_rowspace_nesting_leakage", 1.0)
            )
            > 1e-8
            or replay.get("source_replay_required") is not True
            or any(
                float(replay.get(name, 1.0)) != 0.0
                for name in (
                    "maximum_capture_replay_error",
                    "maximum_direct_replay_error",
                    "maximum_parent_baseline_error",
                    "maximum_parent_full_error",
                    "maximum_posterior_baseline_error",
                    "maximum_posterior_full_error",
                )
            )
        ):
            raise ValueError(
                f"invalid final-query semantic-kernel regime {regime} in {path}"
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
        "scalar_jacobian_rank": 1.0,
        "posterior_jacobian_rank": 15.0,
    }
    for regime in REGIMES:
        detail = campaign["regimes"][regime]
        arms = detail["arms"]
        semantic = arms["semantic_kernel"]
        random = arms["random_kernel"]
        geometry = detail["geometry"]
        attribution = detail["attribution"]
        jacobian = detail["jacobian"]
        prefix = regime
        semantic_change = float(
            semantic["scalar_preservation"]["mean_absolute_coordinate_change"]
        )
        random_change = float(
            random["scalar_preservation"]["mean_absolute_coordinate_change"]
        )
        metrics.update(
            {
                f"{prefix}_baseline_accuracy": float(
                    detail["baseline"]["exact_bin_accuracy"]
                ),
                f"{prefix}_semantic_kernel_accuracy": float(
                    semantic["metrics"]["exact_bin_accuracy"]
                ),
                f"{prefix}_mean_scalar_change": semantic_change,
                f"{prefix}_p95_scalar_change": float(
                    semantic["scalar_preservation"][
                        "p95_absolute_coordinate_change"
                    ]
                ),
                f"{prefix}_task_map_score_loss": float(
                    semantic["scalar_preservation"]["task_map_score_loss"]
                ),
                f"{prefix}_rmse_increase": float(
                    semantic["scalar_preservation"][
                        "root_mean_squared_error_increase"
                    ]
                ),
                f"{prefix}_semantic_kernel_posterior_js": float(
                    semantic["gates"]["mean_posterior_js"]
                ),
                f"{prefix}_semantic_remaining_pair_ratio": float(
                    geometry["semantic_kernel_pair_contraction_ratio"]
                ),
                f"{prefix}_posterior_remaining_pair_ratio": float(
                    geometry["posterior_kernel_pair_contraction_ratio"]
                ),
                f"{prefix}_mean_semantic_displacement_fraction": float(
                    geometry["mean_semantic_displacement_norm_fraction"]
                ),
                f"{prefix}_mean_kernel_displacement_fraction": float(
                    geometry["mean_kernel_displacement_norm_fraction"]
                ),
                f"{prefix}_mean_relative_semantic_residual": float(
                    attribution["mean_relative_semantic_residual"]
                ),
                f"{prefix}_p95_relative_semantic_residual": float(
                    attribution["p95_relative_semantic_residual"]
                ),
                f"{prefix}_semantic_full_sign_agreement": float(
                    attribution["semantic_full_sign_agreement"]
                ),
                f"{prefix}_mean_kernel_full_effect_ratio": float(
                    attribution["mean_kernel_full_effect_ratio"]
                ),
                f"{prefix}_random_scalar_disadvantage": random_change
                - semantic_change,
                f"{prefix}_maximum_nesting_leakage": float(
                    jacobian["maximum_posterior_rowspace_nesting_leakage"]
                ),
                f"{prefix}_maximum_finite_difference_error": float(
                    jacobian["maximum_finite_difference_relative_error"]
                ),
            }
        )
    return metrics


def build_final_query_semantic_kernel_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-final-query-semantic-kernel.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    metrics = _summary(campaign)
    evidence = {
        "experiment_id": "tinyllm-final-query-semantic-kernel-d8-seed7",
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
            "name": "TinyLLM final-query semantic-kernel decomposition",
            "category": "mechanistic_interpretability",
            "description": (
                "An analytic no-fit decomposition asks whether the local kernel "
                "of the frozen posterior-mean cosine coordinate supplies a scalar "
                "quotient despite failure of complete-posterior closure."
            ),
            "question": (
                "Does the rank-1 semantic kernel preserve the model's scalar "
                "coordinate while containing most of the same-cosine query chord?"
            ),
            "prediction": (
                "The semantic-kernel patch preserves the scalar coordinate, leaves "
                "at most 25% of pair distance, and still changes the full posterior."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_scalar_kernel_no_contraction_or_shift_stable_preservation"
            ),
            "evidence_count": 1,
            "direct_quality_experiment_count": 1,
            "fixed_checkpoint_count": 1,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": (
                "one_seed_one_depth_two_shift_1024_fiber_nested_causal_decomposition"
            ),
            "tested_scope": (
                "one retained d8 cosine seed-7 checkpoint, 512 exact fibers in "
                "each of two regimes, final post-MLP query activations, and the "
                "fixed posterior-mean cosine coordinate"
            ),
            "subclaims": {
                "scalar_kernel_preserves_coordinate": "no_both_regimes",
                "scalar_kernel_contains_most_fiber_chord": "no_both_regimes",
                "scalar_posterior_quotient_separation": "no_not_shift_stable",
                "semantic_tangent_nested_in_posterior_tangent": "yes_both_regimes",
                "semantic_tangent_finite_attribution": "support_only",
                "random_rank1_specificity": "yes_both_regimes",
                "raw_final_query_autonomous_scalar_quotient": "no",
                "same_scope_retraining": "no",
            },
            "tags": [
                "tinyllm",
                "task-activations",
                "semantic-kernel",
                "posterior-mean",
                "nested-jacobian",
                "causal-patching",
                "frozen-checkpoint",
                "preregistered-null",
                "underpowered",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The semantic-kernel patch leaves 51.2% and 57.0% of pair "
                "distance and fails scalar preservation in both regimes; its "
                "finite nonlinear attribution also fails outside range."
            ),
            "confidence": 0.98,
            "confidence_assessment": (
                "strong_exact_local_null_for_retained_d8_checkpoint_not_population_prevalence"
            ),
            "num_direct_experiments": 1,
            "descriptive_metrics": metrics,
            "key_insights": [
                "The declared rank-1 semantic tangent is numerically nested inside the rank-15 centered-answer-logit tangent.",
                "That single semantic direction nevertheless contains roughly half of the exact same-target chord.",
                "The finite semantic-kernel patch improves target-facing correlation and RMSE while failing to preserve the model's scalar computation.",
                "Nonlinear tangent attribution is accurate in support and fails under outside-range extrapolation.",
                "Neither scalar nor complete-posterior local-kernel projection yields autonomous closure in the raw final query.",
            ],
            "suggested_hypotheses": [
                "Raw final-query same-target motion is support-relative in both scalar semantic and posterior-shape coordinates.",
                "Architectural gauge repair creates a quotient before the transformer rather than revealing a hidden autonomous quotient at the raw final state.",
                "Future quotient claims must distinguish target improvement from preservation of the frozen computation being averaged.",
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
            "posterior_campaign_sha256": POSTERIOR_CAMPAIGN_SHA256,
            "posterior_diagnostics_sha256": POSTERIOR_DIAGNOSTICS_SHA256,
            "posterior_runner_sha256": POSTERIOR_RUNNER_SHA256,
            "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "parent_d8_result_sha256": PARENT_D8_RESULT_SHA256,
            "parent_d8_diagnostics_sha256": PARENT_D8_DIAGNOSTICS_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "scientific_fingerprint": SCIENTIFIC_FINGERPRINT,
        },
    }


def build_final_query_semantic_kernel_experiment_results(
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
            experiment_id="tinyllm-final-query-semantic-kernel-d8-seed7",
            hypothesis_id=HYPOTHESIS_ID,
            metrics=metrics,
            primary_metric=0.0,
            model_architecture=[8, 512],
            model_parameters=50_964_992,
            training_time=float(campaign["analysis_seconds"]),
            model_checkpoint=str(CHECKPOINT_PATH),
            observations=[
                "One retained d8 seed-7 checkpoint was frozen; no parameter, probe, decoder, carrier, metric, or threshold was fit.",
                "The rank-1 semantic tangent is nested in the rank-15 posterior tangent and is task-specific against a random control.",
                "The scalar-kernel patch fails contraction and scalar preservation in both shifts.",
                f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
            ],
            anomalies=[
                "Support scalar preservation is near individual thresholds, but extrapolation failure is large and the registered joint gate fails."
            ],
            timestamp=completed,
        )
    ]


def store_final_query_semantic_kernel_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_final_query_semantic_kernel_meta_hypothesis(results_path)
    experiments = build_final_query_semantic_kernel_experiment_results(
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
            raise RuntimeError("final-query semantic-kernel ChromaDB read-back failed")
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
    "build_final_query_semantic_kernel_experiment_results",
    "build_final_query_semantic_kernel_meta_hypothesis",
    "store_final_query_semantic_kernel_meta_hypothesis",
]
