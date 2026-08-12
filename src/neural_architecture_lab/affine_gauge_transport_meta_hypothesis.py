"""Build and store the TinyLLM affine-gauge transport result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-affine-gauge-transport.v1"
HYPOTHESIS_ID = "tinyllm-affine-gauge-transport-v1"
EVIDENCE_ROLE = "registered_post_outcome_artifact_only_gauge_audit"
CLASSIFICATION = "support_relative_affine_gauge_insufficient"
CAMPAIGN_SHA256 = (
    "d4ff8b69474d09a47a767a655e8d89610f3e09a0415c0334882bc8ec2babb017"
)
RESULT_MANIFEST_SHA256 = (
    "600e71c8506d4ce15643a5f9c7b99ecccd496c86980ddefd9518508919078833"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "7304fa026ea3dcdf928ef53c1098f79645e5b36007ffdf23a140d6e3756f9027"
)
IMPLEMENTATION_SHA256 = (
    "dfa81f86af3e30c0b24eba4a66cc25b5aa9afda9794565100610ae8ec3a7f89b"
)
CAMPAIGN_FINGERPRINT = (
    "56907481c9c3fb14f8272948e197494dc236b1433322f3d0bc3ec4da15d887a3"
)
RUNNER_SHA256 = (
    "94f5068644f8ba5c84ab74796219bfa37e036da47fec8334539a975913fad040"
)
PREREGISTRATION_SHA256 = (
    "aa7624e98859822592552e0d2a645c8e90a2946fdd5bcf64a72ef37f938a4813"
)
PARENT_CAMPAIGN_SHA256 = (
    "cf8f27e088f9022b78f36d285f2ddb49920bd6bc740d71e6efac7b04ab877cc1"
)
PRESETS = ("d6", "d10")
CONDITION = "learned_calibrated_equivariant"
SEEDS = (7, 17, 29, 41, 53)
ARMS = ("physical_true", "pair_shuffled")
REGIMES = ("composition", "extrapolation")
EXPECTED_JOINT_PASSES = {"d6": 2, "d10": 1}
EXPECTED_FRONT_PASSES = {"d6": 2, "d10": 2}
RUNNER_PATH = Path("experiments/structure_net/tinyllm_affine_gauge_transport.py")
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-affine-gauge-transport-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-affine-gauge-transport.md"
)


@lru_cache(maxsize=256)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns)


def _manifest_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
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


def _detail_paths(campaign_path: Path) -> list[Path]:
    root = campaign_path.parent
    return [
        root / "runs" / preset / CONDITION / f"seed_{seed}" / "result.json"
        for preset in PRESETS
        for seed in SEEDS
    ]


def _campaign(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    aggregate = campaign.get("aggregates", {})
    summary = campaign.get("summary", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or campaign.get("parent", {}).get("campaign_sha256")
        != PARENT_CAMPAIGN_SHA256
        or summary.get("requested_cells") != 10
        or summary.get("completed_cells") != 10
        or summary.get("failed_cells") != 0
        or summary.get("analyzed_arms") != 20
        or summary.get("optimizer_steps") != 0
        or summary.get("trained_parameters") != 0
        or aggregate.get("valid") is not True
        or aggregate.get("complete_primary_population") is not True
        or aggregate.get("classification") != CLASSIFICATION
        or aggregate.get("primary_hypothesis_pass") is not False
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("derived_diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid affine-gauge campaign {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")

    for preset in PRESETS:
        stratum = aggregate["strata"][preset]
        if (
            stratum["valid_count"] != 5
            or stratum["physical_true_canonical_front_pass_count"]
            != EXPECTED_FRONT_PASSES[preset]
            or stratum["physical_true_joint_pass_count"]
            != EXPECTED_JOINT_PASSES[preset]
            or stratum["pair_shuffled_joint_pass_count"] != 0
            or stratum["canonical_front_family_gate"] is not False
            or stratum["family_gate"] is not False
            or stratum["shuffled_specificity_gate"] is not True
        ):
            raise ValueError(f"affine-gauge stratum changed: {preset}")

    result_paths = _detail_paths(path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError("affine-gauge result manifest changed")
    details = [json.loads(item.read_text(encoding="utf-8")) for item in result_paths]
    diagnostics = [Path(item["artifacts"]["derived_diagnostics"]) for item in details]
    if _manifest_sha256(diagnostics) != DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("affine-gauge diagnostics manifest changed")

    physical_counts = {preset: 0 for preset in PRESETS}
    front_counts = {preset: 0 for preset in PRESETS}
    for detail, result_path in zip(details, result_paths):
        gates = detail.get("gates", {})
        arms = detail.get("arms", {})
        diagnostics_path = Path(detail["artifacts"]["derived_diagnostics"])
        parent_result = Path(detail["parent"]["result"])
        parent_diagnostics = Path(detail["parent"]["diagnostics"])
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or Path(detail["artifacts"]["result"]) != result_path
            or gates.get("validity") is not True
            or gates.get("transport_identity") is not True
            or gates.get("derived_diagnostics_reload") is not True
            or gates.get("finite") is not True
            or detail.get("optimizer_steps") != 0
            or detail.get("trained_parameters") != 0
            or any(not arms[arm]["transport_pass"] for arm in ARMS)
            or any(
                float(arms[arm]["transport"]["maximum_float32_absolute_error"])
                > 2e-6
                for arm in ARMS
            )
            or float(arms["physical_true"]["training_fit"]["r_squared"])
            < 0.99
            or arms["pair_shuffled"]["joint_seed_pass"] is not False
            or _sha256(diagnostics_path)
            != detail["artifacts"]["derived_diagnostics_sha256"]
            or _sha256(parent_result) != detail["parent"]["result_sha256"]
            or _sha256(parent_diagnostics)
            != detail["parent"]["diagnostics_sha256"]
            or not _finite(detail)
        ):
            raise ValueError(
                f"invalid affine-gauge cell {detail.get('experiment_id')}"
            )
        preset = str(detail["preset"])
        physical_counts[preset] += int(arms["physical_true"]["joint_seed_pass"])
        front_counts[preset] += int(
            arms["physical_true"]["canonical_front_joint_pass"]
        )
    if (
        physical_counts != EXPECTED_JOINT_PASSES
        or front_counts != EXPECTED_FRONT_PASSES
    ):
        raise ValueError("affine-gauge direct pass counts changed")
    return campaign, details


def _direct_summary(
    campaign: Mapping[str, Any], details: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    cells = []
    for detail in details:
        physical = detail["arms"]["physical_true"]
        shuffled = detail["arms"]["pair_shuffled"]
        regimes = {}
        for regime in REGIMES:
            record = physical["canonical_front"][regime]
            regimes[regime] = {
                "endpoint_pass": record["endpoint_pass"],
                "cosine_correlation": record["scalar_metrics"]["cosine_pearson"],
                "cosine_rmse": record["scalar_metrics"]["cosine_rmse"],
                "branch_balanced_accuracy": record["conditional_branch"][
                    "balanced_accuracy"
                ],
                "conditional_log_loss_gain": record["conditional_branch"][
                    "conditional_log_loss_gain_over_cosine_only"
                ],
                "exact_bin_accuracy": record["task_metrics"]["exact_bin_accuracy"],
                "task_floor": record["task_accuracy_floor"],
                "best_shift_affine_r_squared": physical["shift_best_affine_laws"][
                    regime
                ]["r_squared"],
            }
        cells.append(
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "seed": detail["seed"],
                "training_alpha": physical["training_fit"]["alpha"],
                "training_beta": physical["training_fit"]["beta"],
                "training_r_squared": physical["training_fit"]["r_squared"],
                "orientation": physical["training_fit"]["orientation"],
                "maximum_float32_transport_error": physical["transport"][
                    "maximum_float32_absolute_error"
                ],
                "canonical_front_joint_pass": physical[
                    "canonical_front_joint_pass"
                ],
                "parent_full_depth_joint_pass": physical[
                    "parent_full_depth_joint_pass"
                ],
                "physical_true_joint_seed_pass": physical["joint_seed_pass"],
                "pair_shuffled_joint_seed_pass": shuffled["joint_seed_pass"],
                "pair_shuffled_training_r_squared": shuffled["training_fit"][
                    "r_squared"
                ],
                "regimes": regimes,
                "scientific_fingerprint": detail["scientific_fingerprint"],
            }
        )
    return {
        "classification": campaign["aggregates"]["classification"],
        "campaign_fingerprint": campaign["campaign_fingerprint"],
        "strata": campaign["aggregates"]["strata"],
        "cells": cells,
    }


def build_affine_gauge_transport_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    direct = _direct_summary(campaign, details)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "A learned TinyLLM sensor differs from physical cosine only "
                "by an affine gauge"
            ),
            "category": "registered artifact-only interface decomposition",
            "description": (
                "A sealed training-cohort affine chart is transported inversely "
                "through the saved scalar embedding and evaluated unchanged on "
                "composition and extrapolation."
            ),
            "question": (
                "Is the failed learned physical interface only a checkpoint-local "
                "sign, scale, and offset convention?"
            ),
            "prediction": (
                "At least four of five physical seeds pass jointly in both d6 and "
                "d10, with at most one shuffled pass per preset."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_training_affine_chart_fails_population_transport"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained learned d6/d10 full-interface checkpoints; physical "
                "and pair-shuffled targets; sealed training fit and two held-out shifts"
            ),
            "success_criteria": campaign["aggregates"]["preregistered_gate"],
            "subclaims": {
                "pure_affine_gauge_sufficiency": (
                    "contradicted_d6_two_of_five_d10_one_of_five"
                ),
                "physical_training_affine_relation": (
                    "supported_ten_of_ten_r_squared_above_point_99"
                ),
                "exact_inverse_embedding_transport": "supported_twenty_of_twenty",
                "d6_canonical_front_transport": "insufficient_two_of_five",
                "d10_canonical_front_transport": "insufficient_two_of_five",
                "shuffled_specificity": "supported_zero_of_ten",
                "no_training_or_parameter_change": "supported_all_cells",
                "post_hoc_scalar_map_branch": (
                    "closed_require_fixed_chart_by_construction"
                ),
            },
            "tags": [
                "tinyllm",
                "affine-gauge",
                "inverse-transport",
                "support-relative",
                "artifact-only",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Physical transport reaches only two d6 and one d10 joint seed "
                "passes despite near-affine training fits; all shuffled controls fail."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_within_registered_post_outcome_artifact_only_scope"
            ),
            "num_direct_experiments": 10,
            "successful_direct_experiments": 10,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                (
                    "The private sensor chart has a genuine affine gauge but is "
                    "not exhausted by it."
                ),
                (
                    "All physical canonical fronts pass composition, while only "
                    "two per preset pass extrapolation."
                ),
                (
                    "Continuous cosine order remains strong even where exact-bin "
                    "utility fails."
                ),
                (
                    "The inverse scalar-embedding transport is function-preserving "
                    "to float precision."
                ),
                (
                    "A constructive successor must type the physical chart in its "
                    "function class."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A fixed-chart sensor whose allowed corrections preserve "
                    "orientation, scale, endpoints, and interval decoding passes "
                    "prospectively across d6 and d10."
                ),
                (
                    "A richer calibrated group exposes whether causal quotient "
                    "closure extends beyond the retained C2 reflection."
                ),
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct["cells"]},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "diagnostics_manifest_sha256": DIAGNOSTICS_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "artifact_validation": (
                "campaign, ten results, ten diagnostics, parent result and "
                "diagnostic identities, transport gates, endpoint counts, and "
                "shuffled controls revalidated"
            ),
        },
    }


def build_affine_gauge_transport_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    _campaign_record, details = _campaign(results_path)
    widths = {"d6": 384, "d10": 640}
    layers = {"d6": 6, "d10": 10}
    results = []
    for detail in details:
        physical = detail["arms"]["physical_true"]
        shuffled = detail["arms"]["pair_shuffled"]
        parent_result = json.loads(
            Path(detail["parent"]["result"]).read_text(encoding="utf-8")
        )
        completed = datetime.fromisoformat(
            detail["completed_at"].replace("Z", "+00:00")
        )
        metrics: dict[str, float] = {
            "validity": 1.0,
            "physical_true_joint_seed_pass": float(physical["joint_seed_pass"]),
            "pair_shuffled_joint_seed_pass": float(shuffled["joint_seed_pass"]),
            "canonical_front_joint_pass": float(
                physical["canonical_front_joint_pass"]
            ),
            "parent_full_depth_joint_pass": float(
                physical["parent_full_depth_joint_pass"]
            ),
            "training_r_squared": float(physical["training_fit"]["r_squared"]),
            "training_alpha": float(physical["training_fit"]["alpha"]),
            "maximum_float32_transport_error": float(
                physical["transport"]["maximum_float32_absolute_error"]
            ),
            "optimizer_steps": 0.0,
            "trained_parameters": 0.0,
        }
        for regime in REGIMES:
            cell = physical["canonical_front"][regime]
            metrics[f"{regime}_cosine_correlation"] = float(
                cell["scalar_metrics"]["cosine_pearson"]
            )
            metrics[f"{regime}_exact_accuracy"] = float(
                cell["task_metrics"]["exact_bin_accuracy"]
            )
            metrics[f"{regime}_task_floor"] = float(cell["task_accuracy_floor"])
            metrics[f"{regime}_endpoint_pass"] = float(cell["endpoint_pass"])
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(physical["joint_seed_pass"]),
                model_architecture=[
                    layers[detail["preset"]],
                    widths[detail["preset"]],
                    1,
                ],
                model_parameters=int(
                    parent_result["arms"]["physical_true"]["training"][
                        "trainable_parameter_count"
                    ]
                ),
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=physical["checkpoint"]["path"],
                observations=[
                    f"Preset={detail['preset']} seed={detail['seed']}.",
                    (
                        "Artifact-only affine fit; zero optimizer steps and zero "
                        "trained parameters."
                    ),
                    "Primary metric is the registered physical joint seed pass.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_affine_gauge_transport_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_affine_gauge_transport_meta_hypothesis(results_path)
    experiments = build_affine_gauge_transport_experiment_results(
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
            or len(stored.get("ids", [])) != 10
            or {item.get("hypothesis_id") for item in stored.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("affine-gauge ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": 10,
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
    "build_affine_gauge_transport_experiment_results",
    "build_affine_gauge_transport_meta_hypothesis",
    "store_affine_gauge_transport_meta_hypothesis",
]
