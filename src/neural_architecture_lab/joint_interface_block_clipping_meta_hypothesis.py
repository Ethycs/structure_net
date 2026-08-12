"""Build and store the TinyLLM parameter-block clipping result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-joint-interface-block-clipping.v1"
HYPOTHESIS_ID = "tinyllm-joint-interface-block-clipping-v1"
EVIDENCE_ROLE = "prospective_frozen_backbone_parameter_block_clipping"
CLASSIFICATION = "parameter_block_clipping_insufficient"
CAMPAIGN_SHA256 = (
    "2f7c7cdd5494322ff89e20fb55407c6d4d8de66dde852ca9a8ec67fbc22a2349"
)
RESULT_MANIFEST_SHA256 = (
    "a78a57e3a2bf0f44946a8b3081a64a3bb915ef1e3426c068a765bba70b0e6d69"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "16ec8c5e312b96d1b2880049fcadadc4ee1f466be172a4aa052c1f0b3dd160df"
)
INTERFACE_MANIFEST_SHA256 = (
    "156144e9f6a1bdbebb90262f5c38417c1c238101f3f471df6c8fab33c146ecfd"
)
IMPLEMENTATION_SHA256 = (
    "e69d570234fb727c8ee243d7fb5e72f5a4abaa8ad749f6d9dce0f54794b2a895"
)
CAMPAIGN_FINGERPRINT = (
    "f25912d98c0e1ad11bd0bf115e093443ecf761cd544abe07fc608d8bad3aeb26"
)
RUNNER_SHA256 = (
    "7dc67af033523e7819b0f7cadeac9848508ce2cbe9cce201bef67c6e26dcacf6"
)
PREREGISTRATION_SHA256 = (
    "4264a56a0be6f70fc5c4e812f2fef8aadcdc541450d51993f5f1bbc52fc26f6f"
)
PRESETS = ("d6", "d10")
CONDITION = "learned_calibrated_equivariant"
SEEDS = (7, 17, 29, 41, 53)
ARMS = ("physical_true", "pair_shuffled")
CUTS = ("frontend", "full")
REGIMES = ("composition", "extrapolation")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_joint_interface_block_clipping.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-interface-block-clipping-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-joint-interface-block-clipping.md"
)


@lru_cache(maxsize=128)
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
        or campaign.get("summary", {}).get("completed_interface_fits") != 20
        or campaign.get("summary", {}).get("failed_source_cells") != 0
        or aggregate.get("valid") is not True
        or aggregate.get("complete_primary_population") is not True
        or aggregate.get("classification") != CLASSIFICATION
        or aggregate.get("primary_hypothesis_pass") is not False
        or aggregate.get("full_interface_extension_licensed") is not True
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or campaign.get("interface_manifest_sha256")
        != INTERFACE_MANIFEST_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid block-clipping campaign {path}")
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
            raise ValueError(f"block-clipping stratum changed: {preset}")

    result_paths = _detail_paths(path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError("block-clipping result manifest changed")
    details = [json.loads(item.read_text(encoding="utf-8")) for item in result_paths]
    diagnostics = [Path(item["artifacts"]["diagnostics"]) for item in details]
    interfaces = [
        Path(item["artifacts"]["interfaces"][arm]["path"])
        for item in details
        for arm in ARMS
    ]
    if _manifest_sha256(diagnostics) != DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("block-clipping diagnostics manifest changed")
    if _manifest_sha256(interfaces) != INTERFACE_MANIFEST_SHA256:
        raise ValueError("block-clipping interface manifest changed")
    for detail, result_path in zip(details, result_paths):
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
            or detail.get("gates", {}).get("frozen_model_unchanged") is not True
            or detail.get("gates", {}).get("checkpoint_reload") is not True
            or detail.get("gates", {}).get("diagnostics_reload") is not True
            or detail.get("gates", {}).get("finite") is not True
            or detail.get("gates", {}).get("physical_true_joint_seed_pass")
            is not False
            or detail.get("gates", {}).get("pair_shuffled_joint_seed_pass")
            is not False
            or not _finite(detail)
        ):
            raise ValueError(
                f"invalid block-clipping cell {detail.get('experiment_id')}"
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
        history = detail["arms"]["physical_true"]["training"]["history"]
        cells.append(
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "seed": detail["seed"],
                "physical_true_joint_seed_pass": False,
                "pair_shuffled_joint_seed_pass": False,
                "step_one_global_equivalent_clip": history[0][
                    "global_equivalent_clip_coefficient"
                ],
                "step_one_encoder_block_clip": history[0]["clip_coefficients"][
                    "encoder"
                ],
                "final_logged_sensor_mse": history[-1]["sensor_mse"],
                "final_logged_final_mse": history[-1]["final_mse"],
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


def build_joint_interface_block_clipping_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    direct = _direct_summary(campaign, details)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Parameter-block clipping repairs the learned physical TinyLLM interface",
            "category": "prospective frozen-backbone optimizer intervention",
            "description": (
                "The learned encoder, scalar embedding, and final scalar head "
                "receive independent norm clipping while all backbone parameters, "
                "data, losses, schedules, and endpoints remain fixed."
            ),
            "question": (
                "Was cross-block gradient-norm coupling causally sufficient for "
                "the Stage A physical-interface failure?"
            ),
            "prediction": (
                "The physical arm passes at least four of five seeds in d6 and "
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
                "ten learned d6/d10 Stage A source checkpoints; twenty matched "
                "block-clipped fits; frozen backbones"
            ),
            "success_criteria": campaign["aggregates"]["preregistered_gate"],
            "subclaims": {
                "architecture_family_block_clipping_repair": "contradicted_zero_of_ten",
                "d6_block_clipping_repair": "contradicted_zero_of_five",
                "d10_block_clipping_repair": "contradicted_zero_of_five",
                "semantic_specificity": "supported_zero_of_ten_pair_shuffled",
                "material_optimizer_intervention": "supported_all_ten_cells",
                "stage_a_initial_state_identity": "supported_twenty_of_twenty_arms",
                "frozen_backbone_identity": "supported_twenty_of_twenty_arms",
                "full_interface_extension": "licensed_and_optimizer_branch_closed",
            },
            "tags": [
                "tinyllm",
                "parameter-block-clipping",
                "adamw",
                "typed-interface",
                "frozen-backbone",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Independent block clipping materially changes raw gradient routing "
                "but neither learned architecture population passes one seed."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_within_preregistered_equal_weight_adamw_frozen_backbone_scope"
            ),
            "num_direct_experiments": 10,
            "successful_direct_experiments": 10,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                "Raw cross-block suppression is not causally sufficient for the typed-interface failure.",
                "AdamW approximately cancels uniform positive gradient scaling through its adaptive denominator.",
                "The learned sensor remains compressed or sign-reversed while the final scalar is often useful.",
                "Optimizer repair is closed; the licensed full-interface stage is next.",
            ],
            "suggested_hypotheses": [
                "Unfreezing the residual continuation allows the declared physical chart to remain stable end to end.",
                "If full-interface training fails, physical sign and scale must be fixed architecturally rather than through flexible joint loss."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct["cells"]},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "diagnostics_manifest_sha256": DIAGNOSTICS_MANIFEST_SHA256,
            "interface_manifest_sha256": INTERFACE_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "artifact_validation": (
                "campaign, ten results, ten diagnostics, twenty checkpoints, "
                "source and Stage A identities, endpoint gates, and shuffled "
                "controls revalidated"
            ),
        },
    }


def build_joint_interface_block_clipping_experiment_results(
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
                model_checkpoint=detail["artifacts"]["interfaces"][
                    "physical_true"
                ]["path"],
                observations=[
                    f"Preset={detail['preset']} seed={detail['seed']}.",
                    "Two matched fits; parameter-block clipping; backbone frozen.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_joint_interface_block_clipping_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_joint_interface_block_clipping_meta_hypothesis(results_path)
    experiments = build_joint_interface_block_clipping_experiment_results(
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
            raise RuntimeError("block-clipping ChromaDB read-back failed")
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
    "build_joint_interface_block_clipping_experiment_results",
    "build_joint_interface_block_clipping_meta_hypothesis",
    "store_joint_interface_block_clipping_meta_hypothesis",
]

