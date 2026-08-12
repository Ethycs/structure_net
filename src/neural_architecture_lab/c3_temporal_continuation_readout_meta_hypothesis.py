"""Build and store the frozen C3 temporal continuation/readout result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-temporal-continuation-readout.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-continuation-readout-v1"
EVIDENCE_ROLE = (
    "prospective_artifact_only_c3_continuation_readout_decomposition"
)
CLASSIFICATION = "analytic_sensor_valid_frozen_continuation_not_affinely_typed"
CAMPAIGN_SHA256 = (
    "0da3dcad91a7b9eac34a24a23f687b598b5315349e0b0302a5cc6a0bebab2dea"
)
RESULT_MANIFEST_SHA256 = (
    "3a61695e502813d0b134695edafc4139bad98e86474b8e03b402ce73d932660d"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "71718d4730f1a41e1302acb01e194430a804520091a4547b0741facd2883d225"
)
IMPLEMENTATION_SHA256 = (
    "11db63d258e663eed8654e144cbbb3cd5c70284488962e7dab6a458107e46110"
)
CAMPAIGN_FINGERPRINT = (
    "813a234e5103c806a1d5c788ef9b44a4c84677c91ed761e33e3f8f625b68b210"
)
RUNNER_SHA256 = (
    "3ccb922b8e0fb5119cc8c327024f6b9e1e957bb34b959dcb9e523518ac41746e"
)
PREREGISTRATION_SHA256 = (
    "f4833ffcc7455fd7b72c23d49da1db92920cad5e3030928cccef6af53dbc20ae"
)
REPORT_SHA256 = (
    "811414cb29ee541901dda9cf253a575188795b5c3463cb1fc2ef5a6e146377c9"
)
SOURCE_CAMPAIGN_SHA256 = (
    "e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc"
)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
ARMS = (
    "output_scalar_recalibration",
    "untyped_final_readout",
    "typed_final_readout",
)
EXPECTED_TRUE_PASSES = {
    "output_scalar_recalibration": 0,
    "untyped_final_readout": 0,
    "typed_final_readout": 1,
}
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_continuation_readout.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-continuation-readout-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-temporal-continuation-readout.md"
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


def _result_paths(campaign_path: Path) -> list[Path]:
    root = campaign_path.parent
    return [
        root / "runs" / "analytic" / f"seed_{seed}" / "result.json"
        for seed in SEEDS
    ]


def _diagnostic_paths(campaign_path: Path) -> list[Path]:
    root = campaign_path.parent
    return [
        root / "runs" / "analytic" / f"seed_{seed}" / "diagnostics.npz"
        for seed in SEEDS
    ]


def _campaign(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    aggregates = campaign.get("aggregates", {})
    summary = campaign.get("summary", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or campaign.get("source", {}).get("campaign_sha256")
        != SOURCE_CAMPAIGN_SHA256
        or summary
        != {
            "completed": 5,
            "failed": 0,
            "optimizer_steps": 0,
            "requested": 5,
            "reused": 0,
            "trained_model_parameters": 0,
        }
        or aggregates.get("valid") is not True
        or aggregates.get("exact_temporal_bypass_pass") is not True
        or aggregates.get("specificity_pass") is not True
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("primary_hypothesis_pass") is not False
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid C3 continuation/readout campaign {path}")
    for arm, expected in EXPECTED_TRUE_PASSES.items():
        record = aggregates.get("arms", {}).get(arm, {})
        if (
            record.get("true_pass_count") != expected
            or record.get("shuffled_pass_count") != 0
            or record.get("population_pass") is not False
        ):
            raise ValueError(f"C3 continuation/readout aggregate changed: {arm}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
    ):
        if _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")
    results = _result_paths(path)
    diagnostics = _diagnostic_paths(path)
    if _manifest_sha256(results) != RESULT_MANIFEST_SHA256:
        raise ValueError("C3 continuation/readout result manifest changed")
    if _manifest_sha256(diagnostics) != DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("C3 continuation/readout diagnostic manifest changed")
    details = [json.loads(item.read_text(encoding="utf-8")) for item in results]
    counts = {arm: 0 for arm in ARMS}
    shuffled = {arm: 0 for arm in ARMS}
    for detail, result_path, diagnostic_path in zip(details, results, diagnostics):
        seed = int(detail.get("seed", -1))
        gates = detail.get("gates", {})
        measurements = detail.get("measurements", {})
        artifacts = detail.get("artifacts", {})
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or seed not in SEEDS
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or Path(artifacts.get("result", "")) != result_path
            or Path(artifacts.get("diagnostics", "")) != diagnostic_path
            or _sha256(diagnostic_path) != artifacts.get("diagnostics_sha256")
            or gates.get("validity") is not True
            or gates.get("source_state_unchanged") is not True
            or gates.get("source_replay") is not True
            or gates.get("exact_decoder") is not True
            or gates.get("diagnostics_reload") is not True
            or gates.get("exact_temporal_bypass") is not True
            or measurements.get("optimizer_steps") != 0
            or measurements.get("trained_model_parameters") != 0
            or measurements.get("state_sha256_before")
            != measurements.get("state_sha256_after")
            or not _finite(detail)
        ):
            raise ValueError(f"invalid C3 continuation/readout seed {seed}")
        for arm in ARMS:
            arm_gate = gates.get("arms", {}).get(arm, {})
            counts[arm] += int(arm_gate.get("true_both_shifts", False))
            shuffled[arm] += int(arm_gate.get("shuffled_both_shifts", False))
            if not all(
                detail["regimes"][regime]["arms"][arm]["task_gate_true"]
                for regime in ("composition",)
            ):
                raise ValueError(
                    f"registered composition fit changed for {arm}/seed {seed}"
                )
    if counts != EXPECTED_TRUE_PASSES or any(shuffled.values()):
        raise ValueError("C3 continuation/readout direct pass counts changed")
    return campaign, details


def _direct_summary(
    campaign: Mapping[str, Any], details: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    cells = []
    for detail in details:
        regimes = {}
        for regime in REGIMES:
            arms = {}
            for arm in ARMS:
                record = detail["regimes"][regime]["arms"][arm]
                arms[arm] = {
                    "task_gate_true": record["task_gate_true"],
                    "task_gate_shuffled": record["task_gate_shuffled"],
                    "true_metrics": record["true"],
                    "shuffled_metrics": record["shuffled"],
                    "scalar_true": record.get("scalar_true"),
                    "scalar_shuffled": record.get("scalar_shuffled"),
                }
            regimes[regime] = {
                "source": detail["regimes"][regime]["source"],
                "exact_temporal_bypass": detail["regimes"][regime][
                    "exact_temporal_bypass"
                ],
                "arms": arms,
            }
        cells.append(
            {
                "experiment_id": (
                    f"tinyllm-c3-temporal-continuation-readout-seed{detail['seed']}"
                ),
                "seed": detail["seed"],
                "source_natural_task_pass": detail["source"][
                    "known_natural_task_pass"
                ],
                "arm_gates": detail["gates"]["arms"],
                "exact_temporal_bypass_pass": detail["gates"][
                    "exact_temporal_bypass"
                ],
                "regimes": regimes,
                "scientific_fingerprint": detail["scientific_fingerprint"],
            }
        )
    return {
        "classification": campaign["aggregates"]["classification"],
        "campaign_fingerprint": campaign["campaign_fingerprint"],
        "population": campaign["aggregates"],
        "cells": cells,
    }


def build_c3_temporal_continuation_readout_meta_hypothesis(
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
                "The frozen C3 TinyLLM final state exposes a reliable affine "
                "physical temporal coordinate"
            ),
            "category": "prospective artifact-only continuation/readout decomposition",
            "description": (
                "Closed-form source-only readouts separate inherited output "
                "calibration, free answer rows, and a typed physical scalar "
                "without changing any checkpoint parameter."
            ),
            "question": (
                "Is the analytic C3 population failure confined to its answer "
                "interface, or is outside-support temporal precision already "
                "insufficient in the frozen final state?"
            ),
            "prediction": (
                "A typed affine final-state readout passes the complete task gate "
                "in at least four of five seeds with at most one shuffled pass."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_typed_final_readout_one_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five retained analytic d6 checkpoints; sealed source training "
                "cohorts; composition and outside-speed extrapolation"
            ),
            "success_criteria": {
                "typed_true_joint_seed_passes_minimum": 4,
                "typed_shuffled_seed_passes_maximum": 1,
                "complete_task_gate_required_both_shifts": True,
                "frozen_source_identity_required": True,
            },
            "subclaims": {
                "typed_affine_final_interface": (
                    "not_supported_one_of_five_complete_seed_passes"
                ),
                "free_answer_row_replacement": "not_supported_zero_of_five",
                "inherited_output_scalar_recalibration": (
                    "not_supported_zero_of_five"
                ),
                "exact_analytic_temporal_bypass": "supported_complete_gate",
                "target_shuffle_specificity": "supported_zero_of_fifteen",
                "composition_fit": "supported_all_arms_five_of_five",
                "approximate_extrapolating_physical_coordinate": (
                    "supported_typed_corr_and_accuracy_five_of_five"
                ),
                "complete_extrapolating_metric_precision": (
                    "not_supported_cross_entropy_four_seed_failure"
                ),
                "source_state_unchanged": "supported_five_of_five",
                "raw_and_learned_c3_arms": "untested_predecessor_stop_retained",
            },
            "tags": [
                "tinyllm",
                "c3",
                "temporal-continuation",
                "typed-readout",
                "closed-form-ridge",
                "artifact-only",
                "negative-result",
            ],
            "explicitly_not_tested": [
                "No nonlinear final readout was fitted.",
                "No interval width, ridge value, cohort, or threshold was swept.",
                "No TinyLLM, front-end, raw, or learned-C3 parameter was trained.",
                "D10 and other temporal tasks were not tested.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The typed affine final readout passes only seed 7; output "
                "recalibration and free answer rows pass no seed, while all "
                "shuffled controls fail and the exact temporal bypass passes."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_within_registered_affine_five_seed_artifact_only_scope"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 1,
            "descriptive_metrics": direct,
            "key_insights": [
                (
                    "All registered affine interfaces fit composition but fail "
                    "population-reliable extrapolation."
                ),
                (
                    "The typed scalar retains high correlation and adequate exact "
                    "accuracy in every seed; four failures are cross-entropy precision "
                    "failures rather than loss of ordering."
                ),
                (
                    "A free answer-row replacement is insufficient, so the inherited "
                    "head is not the sole obstruction."
                ),
                (
                    "The fixed analytic temporal operator and decoder remain the "
                    "valid constructive positive control."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A learned exact-C3 sensor feeding a frozen analytic temporal "
                    "operator and fixed physical decoder can acquire the invariant "
                    "carrier without support-relative continuation calibration."
                )
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
            "report_sha256": REPORT_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "artifact_validation": (
                "campaign, five results, five diagnostics, source identities, "
                "zero-change state, exact decoder, source replay, arm pass counts, "
                "and shuffled specificity revalidated"
            ),
        },
    }


def build_c3_temporal_continuation_readout_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    _campaign_record, details = _campaign(results_path)
    results = []
    for detail in details:
        typed = detail["gates"]["arms"]["typed_final_readout"]
        completed = datetime.fromisoformat(
            detail["completed_at"].replace("Z", "+00:00")
        )
        metrics: dict[str, float] = {
            "validity": 1.0,
            "typed_true_joint_pass": float(typed["true_both_shifts"]),
            "typed_shuffled_joint_pass": float(typed["shuffled_both_shifts"]),
            "untyped_true_joint_pass": float(
                detail["gates"]["arms"]["untyped_final_readout"][
                    "true_both_shifts"
                ]
            ),
            "recalibration_true_joint_pass": float(
                detail["gates"]["arms"]["output_scalar_recalibration"][
                    "true_both_shifts"
                ]
            ),
            "exact_temporal_bypass_pass": 1.0,
            "optimizer_steps": 0.0,
            "trained_model_parameters": 0.0,
        }
        for regime in REGIMES:
            cell = detail["regimes"][regime]["arms"]["typed_final_readout"]
            metrics[f"typed_{regime}_accuracy"] = float(
                cell["true"]["exact_bin_accuracy"]
            )
            metrics[f"typed_{regime}_correlation"] = float(
                cell["true"]["posterior_mean_correlation"]
            )
            metrics[f"typed_{regime}_cross_entropy"] = float(
                cell["true"]["target_cross_entropy"]
            )
            metrics[f"typed_{regime}_scalar_correlation"] = float(
                cell["scalar_true"]["cosine_pearson"]
            )
        results.append(
            ExperimentResult(
                experiment_id=(
                    f"tinyllm-c3-temporal-continuation-readout-seed{detail['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(typed["true_both_shifts"]),
                model_architecture=[6, 384, 16],
                model_parameters=int(
                    detail["source"].get("model_parameters", 29_951_232)
                ),
                training_time=float(detail["measurements"]["wall_seconds"]),
                model_checkpoint=detail["source"]["checkpoint"],
                observations=[
                    f"Frozen analytic d6 source seed={detail['seed']}.",
                    "Zero optimizer steps and zero changed model parameters.",
                    (
                        "Primary metric is the registered typed affine final "
                        "readout joint seed pass."
                    ),
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_c3_temporal_continuation_readout_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_temporal_continuation_readout_meta_hypothesis(results_path)
    experiments = build_c3_temporal_continuation_readout_experiment_results(
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
            or len(stored.get("ids", [])) != 5
            or {item.get("hypothesis_id") for item in stored.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C3 continuation/readout ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": 5,
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
    "build_c3_temporal_continuation_readout_experiment_results",
    "build_c3_temporal_continuation_readout_meta_hypothesis",
    "store_c3_temporal_continuation_readout_meta_hypothesis",
]
