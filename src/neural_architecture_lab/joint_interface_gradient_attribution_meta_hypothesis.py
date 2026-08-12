"""Build and store the TinyLLM joint-interface gradient attribution result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-joint-interface-gradient-attribution.v2"
HYPOTHESIS_ID = "tinyllm-joint-interface-gradient-attribution-v2"
EVIDENCE_ROLE = "registered_post_outcome_no_training_gradient_attribution"
CLASSIFICATION = "no_registered_gradient_failure_mechanism"
CAMPAIGN_SHA256 = (
    "a3540216800a0cccf0d3725cf349f8a5c91bf01b8680d44c814afd8f4fa6ba25"
)
RESULT_MANIFEST_SHA256 = (
    "84af30bd069bc62d963090442383330af5c8acc3f935fdd346d855f4600c31b2"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "638047a25c77ffba507c59a9b5442d0002a45e98831b36976d7576ed9146daf4"
)
IMPLEMENTATION_SHA256 = (
    "550f25b276ecd502f73f8b220af4ac1eff5c1ec73eb8838c1f6d27234a2d7183"
)
CAMPAIGN_FINGERPRINT = (
    "e9a3fc91320ce5dada8c2fe5f610b0c45a0310279a7afba26a5c683a7928faad"
)
RUNNER_SHA256 = (
    "75b9222673e543657efd801226149fc154e3c3993f39d05148519c883eebcc23"
)
PREREGISTRATION_SHA256 = (
    "74aef578a00b4cd71cfdd7a94852cb65ba1bef76e3b13be6e1b12b9acab4cbdd"
)
PRESETS = ("d6", "d10")
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
STATES = ("initial", "final")
BATCHES = ("first", "last")
EXPECTED_LEARNED_COUNTS = {
    "d6": {"initial": 5, "persistent": 0},
    "d10": {"initial": 2, "persistent": 2},
}
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_joint_interface_gradient_attribution.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-joint-interface-gradient-attribution-v2-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-joint-interface-gradient-attribution.md"
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
        or campaign.get("summary", {}).get("completed") != 20
        or campaign.get("summary", {}).get("failed") != 0
        or campaign.get("summary", {}).get("optimizer_steps") != 0
        or campaign.get("summary", {}).get("gradient_snapshots") != 80
        or aggregate.get("valid") is not True
        or aggregate.get("complete_primary_population") is not True
        or aggregate.get("classification") != CLASSIFICATION
        or aggregate.get("initial_cross_block_starvation_population_gate")
        is not False
        or aggregate.get("persistent_learned_state_conflict_population_gate")
        is not False
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid gradient-attribution campaign {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")

    for preset, expected in EXPECTED_LEARNED_COUNTS.items():
        stratum = aggregate["strata"][
            f"{preset}/learned_calibrated_equivariant"
        ]
        if (
            stratum["valid_count"] != 5
            or stratum["initial_cross_block_starvation_count"]
            != expected["initial"]
            or stratum["persistent_learned_state_conflict_count"]
            != expected["persistent"]
        ):
            raise ValueError(f"gradient-attribution stratum changed: {preset}")

    result_paths = _detail_paths(path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError("gradient-attribution result manifest changed")
    details = [json.loads(item.read_text(encoding="utf-8")) for item in result_paths]
    diagnostics = [Path(item["artifacts"]["diagnostics"]) for item in details]
    if _manifest_sha256(diagnostics) != DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("gradient-attribution diagnostics manifest changed")
    for detail, result_path in zip(details, result_paths):
        diagnostics_path = Path(detail["artifacts"]["diagnostics"])
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or Path(detail["artifacts"]["result"]) != result_path
            or _sha256(diagnostics_path)
            != detail["artifacts"]["diagnostics_sha256"]
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("gradient_additivity") is not True
            or detail.get("gates", {}).get("state_unchanged") is not True
            or detail.get("gates", {}).get("diagnostics_reload") is not True
            or detail.get("gates", {}).get("finite") is not True
            or not _finite(detail)
        ):
            raise ValueError(
                f"invalid gradient-attribution cell {detail.get('experiment_id')}"
            )
    return campaign, details


def _direct_summary(
    campaign: Mapping[str, Any], details: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    cells = []
    for detail in details:
        learned = detail["condition"] == "learned_calibrated_equivariant"
        snapshots = {}
        for state in STATES:
            snapshots[state] = {}
            for batch in BATCHES:
                record = detail["records"][state][batch]
                encoder = record["blocks"]["encoder"]
                snapshots[state][batch] = {
                    "global_clip_coefficient": record["global_clip_coefficient"],
                    "encoder_block_clip_coefficient": encoder[
                        "block_clip_coefficient"
                    ],
                    "encoder_cross_block_suppression": encoder[
                        "cross_block_suppression"
                    ],
                    "encoder_sensor_gradient_norm": encoder["objective_norms"][
                        "sensor"
                    ],
                    "encoder_downstream_gradient_norm": encoder["downstream_norm"],
                    "encoder_sensor_descent_ratio": encoder["sensor_descent_ratio"],
                    "encoder_sensor_downstream_cosine": encoder[
                        "objective_cosines"
                    ]["sensor_downstream"],
                    "final_scalar_total_gradient_norm": record["blocks"][
                        "final_scalar"
                    ]["total_norm"],
                }
        cells.append(
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "condition": detail["condition"],
                "seed": detail["seed"],
                "learned_primary_cell": learned,
                "initial_cross_block_starvation": detail["gates"][
                    "initial_cross_block_starvation"
                ],
                "persistent_learned_state_conflict": detail["gates"][
                    "persistent_learned_state_conflict"
                ],
                "maximum_absolute_additivity_error": detail["gates"][
                    "maximum_gradient_additivity_error"
                ],
                "maximum_relative_additivity_error": detail["gates"][
                    "maximum_relative_gradient_additivity_error"
                ],
                "snapshots": snapshots,
                "scientific_fingerprint": detail["scientific_fingerprint"],
            }
        )
    return {
        "classification": campaign["aggregates"]["classification"],
        "campaign_fingerprint": campaign["campaign_fingerprint"],
        "registered_gate": campaign["aggregates"]["registered_gate"],
        "strata": campaign["aggregates"]["strata"],
        "cells": cells,
    }


def build_joint_interface_gradient_attribution_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    direct = _direct_summary(campaign, details)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "One registered gradient failure mechanism explains the learned physical-interface failure across TinyLLM architectures",
            "category": "registered post-outcome no-training causal diagnostic",
            "description": (
                "Exact objective gradients at saved initial and final interface "
                "states test pure cross-block clipping starvation and persistent "
                "sensor-versus-downstream conflict without fitting any parameter."
            ),
            "question": (
                "Does one registered optimizer mechanism recur in at least four "
                "of five learned d6 and d10 checkpoints?"
            ),
            "prediction": (
                "Initial starvation or final conflict passes at least four of "
                "five seeds separately in both learned architecture strata."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_no_population_wide_registered_gradient_mechanism"
            ),
            "evidence_count": 20,
            "direct_experiment_count": 20,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "twenty sealed d6/d10 analytic and learned Stage A cells; "
                "eighty exact local gradient snapshots; no optimizer step"
            ),
            "success_criteria": campaign["aggregates"]["registered_gate"],
            "subclaims": {
                "population_wide_registered_mechanism": "contradicted",
                "d6_initial_pure_cross_block_starvation": "supported_five_of_five",
                "d10_initial_pure_cross_block_starvation": "not_supported_two_of_five",
                "d6_persistent_trained_state_conflict": "contradicted_zero_of_five",
                "d10_persistent_trained_state_conflict": "not_supported_two_of_five",
                "objective_gradient_additivity": "supported_twenty_of_twenty",
                "source_and_interface_state_identity": "supported_twenty_of_twenty",
                "universal_descriptive_initial_extra_suppression": (
                    "observed_post_outcome_not_a_registered_population_gate"
                ),
                "full_interface_extension": "remains_licensed_by_stage_a",
            },
            "tags": [
                "tinyllm",
                "gradient-attribution",
                "global-clipping",
                "typed-interface",
                "no-training",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Initial starvation is d6-specific under the frozen pure gate, "
                "and persistent conflict reaches only two d10 seeds and no d6 seed."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_within_registered_saved_state_and_two_batch_local_scope"
            ),
            "num_direct_experiments": 20,
            "successful_direct_experiments": 20,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                "No registered optimizer failure mechanism is architecture-wide.",
                "The zero final head nevertheless induces severe extra initial encoder suppression in every learned cell.",
                "Trained-state sensor/downstream conflict is checkpoint- and minibatch-dependent.",
                "A prospective optimizer intervention is required before assigning causality to the descriptive suppression.",
            ],
            "suggested_hypotheses": [
                "Parameter-block clipping prevents the zero-head gradient from suppressing direct physical-sensor calibration.",
                "If block clipping fails, full-interface fine-tuning is necessary rather than optimizer repair alone.",
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
            "artifact_validation": (
                "campaign, twenty results, twenty diagnostic arrays, eighty "
                "snapshots, state identities, numerical validity, and locked "
                "population gates revalidated"
            ),
        },
    }


def build_joint_interface_gradient_attribution_experiment_results(
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
        initial = detail["gates"]["initial_cross_block_starvation"]
        persistent = detail["gates"]["persistent_learned_state_conflict"]
        learned = detail["condition"] == "learned_calibrated_equivariant"
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics={
                    "validity": 1.0,
                    "learned_primary_cell": float(learned),
                    "initial_cross_block_starvation": float(bool(initial)),
                    "persistent_learned_state_conflict": float(bool(persistent)),
                    "maximum_absolute_additivity_error": float(
                        detail["gates"]["maximum_gradient_additivity_error"]
                    ),
                    "maximum_relative_additivity_error": float(
                        detail["gates"][
                            "maximum_relative_gradient_additivity_error"
                        ]
                    ),
                },
                primary_metric=float(bool(initial)),
                model_architecture=[
                    layers[detail["preset"]],
                    widths[detail["preset"]],
                    1,
                ],
                model_parameters=0,
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["source"]["physical_interface_checkpoint"],
                observations=[
                    f"Preset={detail['preset']} condition={detail['condition']} seed={detail['seed']}.",
                    "Four exact gradient snapshots; no optimizer step.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_joint_interface_gradient_attribution_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_joint_interface_gradient_attribution_meta_hypothesis(
        results_path
    )
    experiments = build_joint_interface_gradient_attribution_experiment_results(
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
            or {
                item.get("hypothesis_id")
                for item in stored.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("gradient-attribution ChromaDB read-back failed")
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
    "build_joint_interface_gradient_attribution_experiment_results",
    "build_joint_interface_gradient_attribution_meta_hypothesis",
    "store_joint_interface_gradient_attribution_meta_hypothesis",
]

