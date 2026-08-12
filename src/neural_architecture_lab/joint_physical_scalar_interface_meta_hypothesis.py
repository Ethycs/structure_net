"""Build and store the TinyLLM joint physical-interface population result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-joint-physical-scalar-interface.v1"
HYPOTHESIS_ID = "tinyllm-joint-physical-scalar-interface-v1"
EVIDENCE_ROLE = "prospective_frozen_backbone_joint_physical_interface"
CLASSIFICATION = "frozen_backbone_joint_interface_insufficient"
CAMPAIGN_SHA256 = (
    "65ab4b4e887212c4754cf918908cd5e3f04727af4d7876de1d3aa749bc50ac51"
)
RESULT_MANIFEST_SHA256 = (
    "3299a6cd2edf8816b8bb65ef1ddfb7dfc18f0f6edd41ff09730af632580fc9f3"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "1f69d3223ae8eb75ff52235be2db326ccf60ecc7f978f1ee89649c7ddb5ba778"
)
INTERFACE_MANIFEST_SHA256 = (
    "50dec4731b55d118561e36f0ff35e8bbcadb3fbca856f7cd234151332e09322a"
)
RUNNER_SHA256 = (
    "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
)
PREREGISTRATION_SHA256 = (
    "1f83fb2802340a51f4dc281f898e99122b99089c48e6c207bd46626c20aab838"
)
IMPLEMENTATION_SHA256 = (
    "ac1f5b42e2e8bcfc645de7dba048ae258e692db79f09b20f4e25f8d00940e1a1"
)
CAMPAIGN_FINGERPRINT = (
    "5c4a9258690182813c8d0701daad308c118b7024407e642c09e30a920543a690"
)
PRESETS = ("d6", "d10")
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
ARMS = ("physical_true", "pair_shuffled")
CUTS = ("frontend", "full")
REGIMES = ("composition", "extrapolation")
EXPECTED_TRUE_COUNTS = {
    "d6/analytic_calibrated": 5,
    "d6/learned_calibrated_equivariant": 0,
    "d10/analytic_calibrated": 4,
    "d10/learned_calibrated_equivariant": 0,
}
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_joint_physical_scalar_interface.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-physical-scalar-interface-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-joint-physical-scalar-interface.md"
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
        root / "runs" / preset / condition / f"seed_{seed}" / "result.json"
        for preset in PRESETS
        for condition in CONDITIONS
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
        or campaign.get("summary")
        != {
            "completed_interface_fits": 40,
            "completed_source_cells": 20,
            "failed_source_cells": 0,
            "frozen_backbones": 20,
            "requested_interface_fits": 40,
            "requested_source_cells": 20,
            "reused_source_cells": 0,
            "scheduled_source_cells": 20,
        }
        or aggregate.get("classification") != CLASSIFICATION
        or aggregate.get("valid") is not True
        or aggregate.get("complete_primary_population") is not True
        or aggregate.get("primary_hypothesis_pass") is not False
        or aggregate.get("full_interface_extension_licensed") is not True
        or aggregate.get("stop_before_transformer_finetuning") is not False
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or campaign.get("interface_manifest_sha256") != INTERFACE_MANIFEST_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid joint physical-interface campaign {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")

    strata = aggregate["strata"]
    for key, expected in EXPECTED_TRUE_COUNTS.items():
        record = strata[key]
        if (
            record["physical_true_pass_count"] != expected
            or record["pair_shuffled_pass_count"] != 0
            or record["valid_count"] != 5
            or record["shuffled_specificity_gate"] is not True
        ):
            raise ValueError(f"joint physical-interface stratum changed: {key}")

    result_paths = _detail_paths(path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError("joint physical-interface result manifest changed")
    details = [json.loads(item.read_text(encoding="utf-8")) for item in result_paths]
    diagnostics = [Path(item["artifacts"]["diagnostics"]) for item in details]
    interfaces = [
        Path(item["artifacts"]["interfaces"][arm]["path"])
        for item in details
        for arm in ARMS
    ]
    if _manifest_sha256(diagnostics) != DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("joint physical-interface diagnostics manifest changed")
    if _manifest_sha256(interfaces) != INTERFACE_MANIFEST_SHA256:
        raise ValueError("joint physical-interface checkpoint manifest changed")
    for detail in details:
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("source_replay") is not True
            or detail.get("gates", {}).get("frozen_model_unchanged") is not True
            or detail.get("gates", {}).get("checkpoint_reload") is not True
            or detail.get("gates", {}).get("diagnostics_reload") is not True
            or detail.get("gates", {}).get("finite") is not True
            or not _finite(detail)
        ):
            raise ValueError(
                f"invalid joint physical-interface cell {detail.get('experiment_id')}"
            )
        if detail["gates"]["pair_shuffled_joint_seed_pass"] is not False:
            raise ValueError("joint physical-interface shuffled specificity changed")
        for arm in ARMS:
            if detail["arms"][arm]["model_unchanged"] is not True:
                raise ValueError("frozen model identity changed")
            if detail["arms"][arm]["checkpoint"]["reload_pass"] is not True:
                raise ValueError("interface checkpoint reload changed")
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
                record = detail["arms"]["physical_true"]["analysis"]["cuts"][cut][
                    regime
                ]
                cuts[cut][regime] = {
                    "endpoint_pass": record["endpoint_pass"],
                    "cosine_correlation": record["scalar_metrics"][
                        "cosine_pearson"
                    ],
                    "cosine_rmse": record["scalar_metrics"]["cosine_rmse"],
                    "calibration_slope": record["scalar_metrics"]["slope"],
                    "calibration_intercept": record["scalar_metrics"]["intercept"],
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
        cells.append(
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "condition": detail["condition"],
                "seed": detail["seed"],
                "physical_true_joint_seed_pass": detail["gates"][
                    "physical_true_joint_seed_pass"
                ],
                "pair_shuffled_joint_seed_pass": False,
                "trainable_parameter_count": detail["arms"]["physical_true"][
                    "training"
                ]["trainable_parameter_count"],
                "final_logged_sensor_mse": detail["arms"]["physical_true"][
                    "training"
                ]["history"][-1]["sensor_mse"],
                "final_logged_final_mse": detail["arms"]["physical_true"][
                    "training"
                ]["history"][-1]["final_mse"],
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


def build_joint_physical_scalar_interface_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    direct = _direct_summary(campaign, details)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "A jointly supervised physical scalar remains typed through a frozen TinyLLM continuation",
            "category": "prospective frozen-backbone interface intervention",
            "description": (
                "The learned equivariant sensor, scalar injection, and final scalar "
                "extractor are jointly supervised while every TinyLLM backbone "
                "parameter remains frozen."
            ),
            "question": (
                "Can one physical cosine convention remain stable from observed "
                "sensor through a frozen TinyLLM continuation under composition "
                "and extrapolation?"
            ),
            "prediction": (
                "The joint endpoint passes at both sensor and full-depth cuts in "
                "at least four of five seeds in every analytic and learned stratum."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_learned_frontend_zero_of_five_both_presets"
            ),
            "evidence_count": 20,
            "direct_experiment_count": 20,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "twenty d6/d10 analytic and learned calibrated source checkpoints; "
                "two matched interface fits per cell; frozen backbones"
            ),
            "success_criteria": campaign["aggregates"]["preregistered_gate"],
            "subclaims": {
                "architecture_family_joint_physical_interface": (
                    "contradicted_learned_zero_of_five_both_presets"
                ),
                "analytic_positive_control": "supported_d6_five_of_five_d10_four_of_five",
                "learned_frontend_physical_typing": "contradicted_zero_of_ten",
                "learned_full_depth_scalar": "partial_two_of_five_both_presets_joint_shifts",
                "conditional_branch_collapse": "supported_at_reported_scalar_cuts",
                "semantic_specificity": "supported_zero_of_twenty_pair_shuffled",
                "frozen_backbone_identity": "supported_forty_of_forty_fits",
                "full_interface_extension": "licensed_by_preregistered_stop_rule",
                "gradient_conflict_explanation": "not_yet_tested",
            },
            "tags": [
                "tinyllm",
                "prospective",
                "frozen-backbone",
                "physical-gauge",
                "typed-interface",
                "equivariant-sensor",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Analytic controls and shuffled specificity pass, but neither "
                "learned population contains a seed whose sensor and full-depth "
                "physical scalar jointly pass both held-out shifts."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_within_preregistered_equal_weight_frozen_backbone_protocol"
            ),
            "num_direct_experiments": 20,
            "successful_direct_experiments": 20,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                "The analytic physical interface is transportable through both backbone families.",
                "The learned interface recreates a compressed or sign-reversed sensor gauge even when its final scalar is useful.",
                "A decodable final scalar is not evidence of one shared physical coordinate across modules.",
                "Pair-preserving shuffled targets pass no seed, excluding generic fit capacity as the result.",
            ],
            "suggested_hypotheses": [
                "Global clipping causes final/task gradients to dominate direct sensor calibration in the learned interface.",
                "A fixed physical sensor output or separately normalized parameter-block updates prevent gauge re-creation.",
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
                "campaign, twenty result files, twenty diagnostics, forty interface "
                "checkpoints, frozen states, gates, and shuffled controls revalidated"
            ),
        },
    }


def build_joint_physical_scalar_interface_experiment_results(
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
        gate = bool(detail["gates"]["physical_true_joint_seed_pass"])
        metrics: dict[str, float] = {
            "validity": 1.0,
            "physical_true_joint_seed_pass": float(gate),
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
        checkpoint = detail["artifacts"]["interfaces"]["physical_true"]["path"]
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(gate),
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
                model_checkpoint=checkpoint,
                observations=[
                    f"Preset={detail['preset']} condition={detail['condition']} seed={detail['seed']}.",
                    "Two matched interface fits; TinyLLM backbone frozen.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_joint_physical_scalar_interface_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_joint_physical_scalar_interface_meta_hypothesis(results_path)
    experiments = build_joint_physical_scalar_interface_experiment_results(
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
            or len(stored.get("ids", [])) != 20
            or {item.get("hypothesis_id") for item in stored.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("joint physical-interface ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": 20,
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
    "build_joint_physical_scalar_interface_experiment_results",
    "build_joint_physical_scalar_interface_meta_hypothesis",
    "store_joint_physical_scalar_interface_meta_hypothesis",
]
