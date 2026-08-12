"""Build and store the TinyLLM frozen interval-readout result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-frozen-interval-readout-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-frozen-interval-readout-decomposition-v1"
EVIDENCE_ROLE = "prospective_frozen_backbone_closed_form_interface_fit"
CLASSIFICATION = "partial_frozen_interface_repair"
CAMPAIGN_SHA256 = (
    "3f15245386d1fb41e797f0688ff512aadc7c9690a552c3d58c8ba92754ee9208"
)
RESULT_MANIFEST_SHA256 = (
    "8a567df1952634a0be782b0dbd52e8dd92bd8d5d39f3f2d4f23c4f6df2a04638"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "7110c85b90c96581e3d65aaffba0a2c2d829e763a69f7f2dd7eac40403759454"
)
RUNNER_SHA256 = (
    "bb6a73c203fcf4e654295bf7567f205826a6f054153ad50bbdd36851297de926"
)
PREREGISTRATION_SHA256 = (
    "e5e1ee52bdccb420403643cf071adf3a49c80854c70adf5c632bba56e683d62f"
)
IMPLEMENTATION_SHA256 = (
    "67e225b46a52d5724b84689dc411db507bc3d51dc5b0f81db0f0f707765b0d79"
)
CAMPAIGN_FINGERPRINT = (
    "176b00d5cf9a97537ed36bbfea75fb609dcef41f62de8972d1eeed4b1424def6"
)
PRESETS = ("d6", "d10")
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
ARMS = (
    "input_affine_gauge",
    "untyped_final_readout",
    "typed_interval_readout",
    "frontend_typed_bypass",
)
EXPECTED_TYPED_COUNTS = {
    "d6/analytic_calibrated": 5,
    "d6/learned_calibrated_equivariant": 4,
    "d10/analytic_calibrated": 5,
    "d10/learned_calibrated_equivariant": 1,
}
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_frozen_interval_readout_decomposition.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-interval-readout-decomposition-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-frozen-interval-readout-decomposition.md"
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
            "closed_form_fit_count": 120,
            "completed": 20,
            "failed": 0,
            "fitted_parameter_count": 348920,
            "iterative_optimizers": 0,
            "requested": 20,
            "source_checkpoints_loaded": 20,
            "trained_frontends": 0,
            "trained_models": 0,
        }
        or aggregate.get("classification") != CLASSIFICATION
        or aggregate.get("valid") is not True
        or aggregate.get("complete_primary_population") is not True
        or aggregate.get("source_failed_count") != 10
        or aggregate.get("source_failure_repair_counts")
        != {
            "frontend_typed_bypass": 4,
            "input_affine_gauge": 1,
            "typed_interval_readout": 5,
            "untyped_final_readout": 3,
        }
        or aggregate.get("gates")
        != {
            "frontend_bypass_family_pass": False,
            "input_affine_gauge_family_pass": False,
            "joint_interface_retraining_licensed": True,
            "primary_hypothesis_pass": False,
            "stop_before_joint_retraining": False,
            "untyped_readout_family_pass": False,
        }
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid frozen interval-readout campaign {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")

    strata = aggregate["strata"]
    for key, expected in EXPECTED_TYPED_COUNTS.items():
        typed = strata[key]["arms"]["typed_interval_readout"]
        if typed["true_pass_count"] != expected or typed["shuffled_pass_count"] != 0:
            raise ValueError(f"typed interval-readout stratum changed: {key}")
        for arm in ARMS:
            if strata[key]["arms"][arm]["shuffled_pass_count"] != 0:
                raise ValueError(f"shuffled specificity changed: {key}/{arm}")

    result_paths = [Path(item["result"]) for item in campaign["results"]]
    if len(result_paths) != 20 or _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError("frozen interval-readout result manifest changed")
    details = [json.loads(item.read_text(encoding="utf-8")) for item in result_paths]
    diagnostics = [Path(item["artifacts"]["diagnostics"]) for item in details]
    if _manifest_sha256(diagnostics) != DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("frozen interval-readout diagnostics manifest changed")
    for detail in details:
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("source_replay") is not True
            or detail.get("gates", {}).get("exact_decoder_fidelity") is not True
            or detail.get("gates", {}).get("state_unchanged") is not True
            or detail.get("gates", {}).get("diagnostics_reload") is not True
            or detail.get("fit_protocol", {}).get("trained_model_parameters") != 0
            or not _finite(detail)
        ):
            raise ValueError(
                "invalid frozen interval-readout cell "
                f"{detail.get('experiment_id')}"
            )
        for arm in ARMS:
            if detail["gates"]["arms"][arm]["shuffled_both_shifts"] is not False:
                raise ValueError(
                    f"frozen interval-readout shuffled cell changed: {arm}"
                )
    return campaign, details


def _direct_summary(
    campaign: Mapping[str, Any], details: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    cells = []
    minimum_typed_correlation = 1.0
    for detail in details:
        regimes = {}
        for regime in REGIMES:
            record = detail["regimes"][regime]
            typed = record["arms"]["typed_interval_readout"]
            minimum_typed_correlation = min(
                minimum_typed_correlation,
                float(typed["scalar_true"]["cosine_pearson"]),
            )
            regimes[regime] = {
                "task_floor": record["task_accuracy_floor"],
                "source_accuracy": record["source"]["metrics"][
                    "exact_bin_accuracy"
                ],
                "typed_accuracy": typed["true"]["exact_bin_accuracy"],
                "typed_cosine_correlation": typed["scalar_true"][
                    "cosine_pearson"
                ],
                "typed_cosine_rmse": typed["scalar_true"]["cosine_rmse"],
                "untyped_accuracy": record["arms"]["untyped_final_readout"][
                    "true"
                ]["exact_bin_accuracy"],
                "bypass_accuracy": record["arms"]["frontend_typed_bypass"][
                    "true"
                ]["exact_bin_accuracy"],
                "input_gauge_accuracy": record["arms"]["input_affine_gauge"][
                    "true"
                ]["exact_bin_accuracy"],
            }
        cells.append(
            {
                "experiment_id": detail["experiment_id"],
                "preset": detail["preset"],
                "condition": detail["condition"],
                "seed": detail["seed"],
                "source_task_adequacy_pass": detail["source"][
                    "source_task_adequacy_pass"
                ],
                "arm_gates": detail["gates"]["arms"],
                "regimes": regimes,
                "scientific_fingerprint": detail["scientific_fingerprint"],
            }
        )
    return {
        "classification": campaign["aggregates"]["classification"],
        "campaign_fingerprint": campaign["campaign_fingerprint"],
        "strata": campaign["aggregates"]["strata"],
        "source_failure_repair_counts": campaign["aggregates"][
            "source_failure_repair_counts"
        ],
        "minimum_typed_cosine_correlation": minimum_typed_correlation,
        "cells": cells,
    }


def build_frozen_interval_readout_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    direct = _direct_summary(campaign, details)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "A fixed physical interval readout makes the calibrated TinyLLM family task-stable",
            "category": "prospective frozen-interface causal decomposition",
            "description": (
                "Closed-form physical-cosine and free-logit readouts intervene "
                "only after frozen d6/d10 calibrated TinyLLM states."
            ),
            "question": (
                "Can the exact ordered interval chart recover architecture-family "
                "task adequacy from the frozen quotient-sufficient final state?"
            ),
            "prediction": (
                "The typed interval arm passes both shifts in at least four of "
                "five seeds in every preset/condition stratum, with no shuffled success."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_typed_readout_fails_d10_learned_one_of_five"
            ),
            "evidence_count": 20,
            "direct_experiment_count": 20,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "twenty frozen structured d6/d10 checkpoints; analytic and learned "
                "calibrated front ends; composition and extrapolation"
            ),
            "success_criteria": {
                "typed_scalar_correlation_minimum": 0.90,
                "task_floor": "inherited per checkpoint and regime",
                "required_seed_passes_per_stratum": 4,
                "shuffled_seed_pass_ceiling_per_stratum": 1,
            },
            "subclaims": {
                "architecture_family_typed_readout": "contradicted_d10_learned_one_of_five",
                "analytic_population_typed_readout": "supported_ten_of_ten",
                "d6_learned_typed_readout": "supported_four_of_five",
                "d10_learned_typed_readout": "contradicted_one_of_five",
                "source_failure_repair": "supported_five_of_ten_partial",
                "typed_scalar_ordering": "supported_forty_of_forty_regime_cells",
                "untyped_answer_rows_sufficient": "contradicted_family_gate",
                "input_affine_gauge_sufficient": "contradicted_family_gate",
                "semantic_specificity": "supported_zero_shuffled_passes_all_strata",
                "joint_interface_retraining": "licensed_by_preregistered_stop_rule",
            },
            "tags": [
                "tinyllm",
                "prospective",
                "frozen-backbone",
                "closed-form-ridge",
                "typed-interface",
                "interval-chart",
                "partial-repair",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The typed readout passes both analytic strata and d6 learned, "
                "but only one of five d10 learned seeds meets both task floors."
            ),
            "confidence": 0.99,
            "confidence_assessment": "high_within_preregistered_frozen_affine_readout_family",
            "num_direct_experiments": 20,
            "successful_direct_experiments": 20,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                "A physically typed ordered readout repairs five of ten source failures but is not model-family stable.",
                "All typed scalar correlations exceed 0.95; absolute extrapolation calibration, not ordering, fails.",
                "Untying sixteen answer rows does not outperform the scalar chart at the population boundary.",
                "Mapping learned input scalars to physical cosine breaks their co-adapted frozen continuation.",
            ],
            "suggested_hypotheses": [
                "A learned equivariant sensor and frozen continuation trained under one explicit physical scalar gauge retain calibrated cosine under extrapolation.",
                "A prospectively typed scalar embedding and output chart outperform endpoint-only fits in d10 learned checkpoints.",
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
                "campaign, twenty result files, twenty diagnostics, source identity, "
                "fit controls, seed gates, and shuffled specificity revalidated"
            ),
        },
    }


def build_frozen_interval_readout_decomposition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    _campaign_record, details = _campaign(results_path)
    results = []
    widths = {"d6": 384, "d10": 640}
    layers = {"d6": 6, "d10": 10}
    for detail in details:
        completed = datetime.fromisoformat(
            detail["completed_at"].replace("Z", "+00:00")
        )
        typed_gate = bool(
            detail["gates"]["arms"]["typed_interval_readout"][
                "true_both_shifts"
            ]
        )
        metrics: dict[str, float] = {
            "validity": 1.0,
            "typed_interval_both_shifts": float(typed_gate),
            "untyped_final_both_shifts": float(
                detail["gates"]["arms"]["untyped_final_readout"][
                    "true_both_shifts"
                ]
            ),
            "input_affine_both_shifts": float(
                detail["gates"]["arms"]["input_affine_gauge"][
                    "true_both_shifts"
                ]
            ),
            "frontend_bypass_both_shifts": float(
                detail["gates"]["arms"]["frontend_typed_bypass"][
                    "true_both_shifts"
                ]
            ),
        }
        for regime in REGIMES:
            typed = detail["regimes"][regime]["arms"][
                "typed_interval_readout"
            ]
            metrics[f"{regime}_typed_accuracy"] = float(
                typed["true"]["exact_bin_accuracy"]
            )
            metrics[f"{regime}_typed_cosine_correlation"] = float(
                typed["scalar_true"]["cosine_pearson"]
            )
        source_result = json.loads(
            Path(detail["source"]["result"]).read_text(encoding="utf-8")
        )
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(typed_gate),
                model_architecture=[
                    layers[detail["preset"]],
                    widths[detail["preset"]],
                    16,
                ],
                model_parameters=int(source_result["model_parameters"]),
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["source"]["model_checkpoint"],
                observations=[
                    f"Preset={detail['preset']} condition={detail['condition']} seed={detail['seed']}.",
                    "Six closed-form interface fits; frozen checkpoint unchanged.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_frozen_interval_readout_decomposition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_frozen_interval_readout_decomposition_meta_hypothesis(
        results_path
    )
    experiments = build_frozen_interval_readout_decomposition_experiment_results(
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
            raise RuntimeError("frozen interval-readout ChromaDB read-back failed")
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
    "build_frozen_interval_readout_decomposition_experiment_results",
    "build_frozen_interval_readout_decomposition_meta_hypothesis",
    "store_frozen_interval_readout_decomposition_meta_hypothesis",
]
