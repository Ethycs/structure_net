"""Build and store the TinyLLM joint full-interface physical-typing result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-joint-full-interface.v1"
HYPOTHESIS_ID = "tinyllm-joint-full-interface-physical-typing-v1"
EVIDENCE_ROLE = "prospective_full_interface_physical_typing"
CLASSIFICATION = "flexible_full_interface_physical_typing_insufficient"
CAMPAIGN_SHA256 = (
    "cf8f27e088f9022b78f36d285f2ddb49920bd6bc740d71e6efac7b04ab877cc1"
)
RESULT_MANIFEST_SHA256 = (
    "658b47f2311035139df5cf038e612f6d7c6c2d495fc1267094dce1617df78d2c"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "4489e7dba9e6db55646268b023ab3d143b37bdb02c1f63db0525d89319638e9a"
)
FULL_INTERFACE_MANIFEST_SHA256 = (
    "4dba8670d5622b65b1eeb1df54eff8d2b7a5137cda614211de4d62ec3a96fd22"
)
IMPLEMENTATION_SHA256 = (
    "e719cfd42c3e14be5a37700d55272dd0f6172698fd302b8b5f9be136c1ea11f9"
)
CAMPAIGN_FINGERPRINT = (
    "dbefa30cfd6ba1cda963ff771fdbf2fa364815265bd43a7b06bffab6854464e8"
)
RUNNER_SHA256 = (
    "b4509a24436e5767afa96a42672f6ab7d0cc857230e62fc205be5214049bd65a"
)
PREREGISTRATION_SHA256 = (
    "1921afdeb0fca28c80ff9fb151c767dd6b8baa8aac39967b3ae614bef5df9329"
)
PRESETS = ("d6", "d10")
CONDITION = "learned_calibrated_equivariant"
SEEDS = (7, 17, 29, 41, 53)
ARMS = ("physical_true", "pair_shuffled")
CUTS = ("frontend", "full")
REGIMES = ("composition", "extrapolation")
RUNNER_PATH = Path("experiments/structure_net/tinyllm_joint_full_interface.py")
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-full-interface-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-joint-full-interface.md"
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
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or campaign.get("summary", {}).get("completed_source_cells") != 10
        or campaign.get("summary", {}).get("completed_full_interface_fits") != 20
        or campaign.get("summary", {}).get("failed_source_cells") != 0
        or campaign.get("summary", {}).get("changed_continuations") != 10
        or aggregate.get("valid") is not True
        or aggregate.get("complete_primary_population") is not True
        or aggregate.get("classification") != CLASSIFICATION
        or aggregate.get("primary_hypothesis_pass") is not False
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or campaign.get("full_interface_manifest_sha256")
        != FULL_INTERFACE_MANIFEST_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid full-interface campaign {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")
    for preset in PRESETS:
        stratum = aggregate["strata"][preset]
        if (
            stratum["valid_count"] != 5
            or stratum["physical_true_pass_count"] != 0
            or stratum["pair_shuffled_pass_count"] != 0
            or stratum["shuffled_specificity_gate"] is not True
        ):
            raise ValueError(f"full-interface stratum changed: {preset}")

    result_paths = _detail_paths(path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError("full-interface result manifest changed")
    details = [json.loads(item.read_text(encoding="utf-8")) for item in result_paths]
    diagnostics = [Path(item["artifacts"]["diagnostics"]) for item in details]
    checkpoints = [
        Path(item["artifacts"]["full_interfaces"][arm]["path"])
        for item in details
        for arm in ARMS
    ]
    if _manifest_sha256(diagnostics) != DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("full-interface diagnostics manifest changed")
    if _manifest_sha256(checkpoints) != FULL_INTERFACE_MANIFEST_SHA256:
        raise ValueError("full-interface checkpoint manifest changed")
    for detail, result_path in zip(details, result_paths):
        arms = detail.get("arms", {})
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or Path(detail["artifacts"]["result"]) != result_path
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("stage_a_initial_state_replay")
            is not True
            or detail.get("gates", {}).get("source_model_state_replay") is not True
            or detail.get("gates", {}).get("continuation_changed") is not True
            or detail.get("gates", {}).get("model_topology_unchanged") is not True
            or detail.get("gates", {}).get("full_parameter_route") is not True
            or detail.get("gates", {}).get("checkpoint_reload") is not True
            or detail.get("gates", {}).get("diagnostics_reload") is not True
            or detail.get("gates", {}).get("finite") is not True
            or detail.get("gates", {}).get("physical_true_joint_seed_pass")
            is not False
            or detail.get("gates", {}).get("pair_shuffled_joint_seed_pass")
            is not False
            or any(not arms[arm]["training"]["model_changed"] for arm in ARMS)
            or any(
                not arms[arm]["training"]["model_topology_unchanged"]
                for arm in ARMS
            )
            or not _finite(detail)
        ):
            raise ValueError(
                f"invalid full-interface cell {detail.get('experiment_id')}"
            )
    return campaign, details


def _direct_summary(
    campaign: Mapping[str, Any], details: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    cells = []
    for detail in details:
        cuts = {}
        for cut in CUTS:
            cuts[cut] = {}
            for regime in REGIMES:
                record = detail["arms"]["physical_true"]["analysis"]["cuts"][
                    cut
                ][regime]
                cuts[cut][regime] = {
                    "endpoint_pass": record["endpoint_pass"],
                    "cosine_correlation": record["scalar_metrics"][
                        "cosine_pearson"
                    ],
                    "cosine_rmse": record["scalar_metrics"]["cosine_rmse"],
                    "calibration_slope": record["scalar_metrics"]["slope"],
                    "branch_balanced_accuracy": record["conditional_branch"][
                        "balanced_accuracy"
                    ],
                    "conditional_log_loss_gain": record["conditional_branch"][
                        "conditional_log_loss_gain_over_cosine_only"
                    ],
                    "exact_bin_accuracy": record["task_metrics"][
                        "exact_bin_accuracy"
                    ],
                    "task_floor": record["task_accuracy_floor"],
                }
        training = detail["arms"]["physical_true"]["training"]
        cells.append(
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "seed": detail["seed"],
                "physical_true_joint_seed_pass": False,
                "pair_shuffled_joint_seed_pass": False,
                "trainable_parameter_count": training[
                    "trainable_parameter_count"
                ],
                "group_parameter_counts": training["group_parameter_counts"],
                "model_changed": training["model_changed"],
                "model_topology_unchanged": training[
                    "model_topology_unchanged"
                ],
                "final_logged_sensor_mse": training["history"][-1]["sensor_mse"],
                "final_logged_final_mse": training["history"][-1]["final_mse"],
                "cuts": cuts,
                "scientific_fingerprint": detail["scientific_fingerprint"],
            }
        )
    return {
        "classification": campaign["aggregates"]["classification"],
        "campaign_fingerprint": campaign["campaign_fingerprint"],
        "strata": campaign["aggregates"]["strata"],
        "cells": cells,
    }


def build_joint_full_interface_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    direct = _direct_summary(campaign, details)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "A flexible TinyLLM continuation repairs learned physical typing",
            "category": "prospective full-interface causal intervention",
            "description": (
                "The learned sensor, scalar embedding, complete TinyLLM residual "
                "continuation, tied token/LM-head embedding, and final scalar head "
                "train jointly from exact Stage A initialization."
            ),
            "question": (
                "Was the frozen continuation the remaining obstruction to a stable "
                "declared physical cosine chart?"
            ),
            "prediction": (
                "Physical-true passes at least four of five seeds in both d6 and "
                "d10 while pair-shuffled controls remain specific."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_physical_true_zero_of_five_both_presets"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "ten learned d6/d10 source checkpoints; twenty matched full-model "
                "fits; exact Stage A data, objective, schedule, and gates"
            ),
            "success_criteria": campaign["aggregates"]["preregistered_gate"],
            "subclaims": {
                "architecture_family_full_interface_repair": (
                    "contradicted_zero_of_ten"
                ),
                "d6_full_interface_repair": "contradicted_zero_of_five",
                "d10_full_interface_repair": "contradicted_zero_of_five",
                "semantic_specificity": "supported_zero_of_ten_pair_shuffled",
                "full_parameter_intervention": "supported_twenty_of_twenty_arms",
                "continuation_state_change": "supported_twenty_of_twenty_arms",
                "topology_and_tied_alias_identity": (
                    "supported_twenty_of_twenty_arms"
                ),
                "front_sensor_physical_chart": "contradicted_zero_of_ten",
                "full_depth_cosine_retention": (
                    "supported_all_ten_both_held_out_shifts"
                ),
                "branch_contraction": (
                    "supported_all_ten_both_cuts_and_held_out_shifts"
                ),
                "flexible_joint_supervision_branch": (
                    "closed_require_fixed_chart_by_construction"
                ),
            },
            "tags": [
                "tinyllm",
                "full-interface",
                "physical-typing",
                "gauge",
                "coadaptation",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Every continuation changes and remains valid, but neither learned "
                "architecture population passes one joint physical seed."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_within_preregistered_equal_weight_600_update_scope"
            ),
            "num_direct_experiments": 10,
            "successful_direct_experiments": 10,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                "The frozen continuation is not the sole obstruction to physical typing.",
                "All d10 sensors retain a sign-reversed chart while full-depth outputs recover positive physical orientation.",
                "D6 retains seed-dependent gauge: three positive and two sign-reversed sensor charts.",
                "Branch contraction and full-depth cosine retention survive; front chart and task calibration fail.",
                "Flexible joint supervision is closed as a construction method under the registered protocol.",
            ],
            "suggested_hypotheses": [
                "An orientation-preserving fixed-chart sensor head prevents sign and scale regauging.",
                "A no-training affine gauge audit with inverse scalar-embedding transport isolates pure gauge from nonlinear sensor error.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct["cells"]},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "diagnostics_manifest_sha256": DIAGNOSTICS_MANIFEST_SHA256,
            "full_interface_manifest_sha256": FULL_INTERFACE_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "artifact_validation": (
                "campaign, ten results, ten diagnostics, twenty full checkpoints, "
                "source and Stage A identities, parameter routes, topology, endpoint "
                "gates, and shuffled controls revalidated"
            ),
        },
    }


def build_joint_full_interface_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    _campaign_record, details = _campaign(results_path)
    widths = {"d6": 384, "d10": 640}
    layers = {"d6": 6, "d10": 10}
    results = []
    for detail in details:
        completed = datetime.fromisoformat(
            detail["completed_at"].replace("Z", "+00:00")
        )
        metrics: dict[str, float] = {
            "validity": 1.0,
            "physical_true_joint_seed_pass": 0.0,
            "pair_shuffled_joint_seed_pass": 0.0,
            "continuation_changed": 1.0,
            "model_topology_unchanged": 1.0,
        }
        for cut in CUTS:
            for regime in REGIMES:
                cell = detail["arms"]["physical_true"]["analysis"]["cuts"][cut][
                    regime
                ]
                prefix = f"{cut}_{regime}"
                metrics[f"{prefix}_cosine_correlation"] = float(
                    cell["scalar_metrics"]["cosine_pearson"]
                )
                metrics[f"{prefix}_exact_accuracy"] = float(
                    cell["task_metrics"]["exact_bin_accuracy"]
                )
                metrics[f"{prefix}_branch_accuracy"] = float(
                    cell["conditional_branch"]["balanced_accuracy"]
                )
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=0.0,
                model_architecture=[
                    layers[detail["preset"]],
                    widths[detail["preset"]],
                    1,
                ],
                model_parameters=int(
                    detail["arms"]["physical_true"]["training"][
                        "trainable_parameter_count"
                    ]
                ),
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["artifacts"]["full_interfaces"][
                    "physical_true"
                ]["path"],
                observations=[
                    f"Preset={detail['preset']} seed={detail['seed']}.",
                    "Two matched fits; complete TinyLLM continuation trainable.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_joint_full_interface_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_joint_full_interface_meta_hypothesis(results_path)
    experiments = build_joint_full_interface_experiment_results(record, results_path)
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
            raise RuntimeError("full-interface ChromaDB read-back failed")
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
    "build_joint_full_interface_experiment_results",
    "build_joint_full_interface_meta_hypothesis",
    "store_joint_full_interface_meta_hypothesis",
]
