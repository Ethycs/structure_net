"""Build and store the TinyLLM frozen scalar-interface diagnostic."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-frozen-scalar-interface-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-frozen-scalar-interface-decomposition-v1"
EVIDENCE_ROLE = "registered_post_outcome_frozen_scalar_causal_diagnostic"
CLASSIFICATION = "continuation_capacity_failure_present"
CAMPAIGN_SHA256 = (
    "f71b85bc51f9694346fa23482bc63e1631e0918d55f5cae4f3a0d5ce09a47ba2"
)
RESULT_MANIFEST_SHA256 = (
    "201f6bb780e5a92938df1596e61f9f75164bb0e3bfcd0554f6ad03ba74f177b4"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "bd9adc8e1103835af3c8264465fc0fe1723f06b769b8245ba1d914fe7de6145a"
)
IMPLEMENTATION_SHA256 = (
    "605798a7966691dfa10dbdd3af1fccee818850c141d4a797f78b11d57a816b4a"
)
CAMPAIGN_FINGERPRINT = (
    "f64e6c9e6440037c922e83cf4c81603aafef336cd0c8a3e938d787bf69a049e9"
)
RUNNER_SHA256 = (
    "0228836047e99c8af865e377e8cae329ddb56c70a4de22207087b8d10f2da128"
)
PREREGISTRATION_SHA256 = (
    "157d5b1c0a41f3d89f79530a4f3a92123b2c1c0da265a9cd87c5945f0882ea40"
)
SOURCE_CAMPAIGN_SHA256 = (
    "656d9814a032d1899810e81d398adf935cea3e1116712460e2062da188a0c9e2"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "f87740e70062f5fecf5238f00dd00774246e4f3e155dceb87752b099ce4ca80a"
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "5c08c771d04aae513ad9605d9e4818867ab0a8b0303680337dabbf87dce352e0"
)
PRESETS = ("d6", "d10")
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
SOURCE_FAILURES = {
    ("d6", "analytic_calibrated"): (),
    ("d6", "learned_calibrated_equivariant"): (7, 17, 41, 53),
    ("d10", "analytic_calibrated"): (17, 29),
    ("d10", "learned_calibrated_equivariant"): (7, 29, 41, 53),
}
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_frozen_scalar_interface_decomposition.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-scalar-interface-decomposition-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-frozen-scalar-interface-decomposition.md"
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


def _expected_aggregate() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "classification_counts": {
            "continuation_or_answer_row_failure": 1,
            "scalar_coordinate_or_boundary_failure": 8,
            "sensor_scalar_estimation_failure": 1,
            "source_passing_positive_control": 10,
        },
        "exact_cosine_repair_count": 1,
        "gates": {
            "one_dimensional_interface_expressively_sufficient": False,
            "primary_hypothesis_pass": False,
            "semantic_specificity": True,
            "uniformly_upstream_of_scalar_embedding": False,
        },
        "negative_cosine_pass_count": 0,
        "oracle_repair_count": 9,
        "positive_control_oracle_recovery_count": 10,
        "shuffled_cosine_pass_count": 0,
        "source_failed_count": 10,
        "source_passing_control_count": 10,
        "strata": {
            "d10/analytic_calibrated": {
                "exact_cosine_repair_count": 1,
                "oracle_repair_count": 2,
                "source_failed_count": 2,
            },
            "d10/learned_calibrated_equivariant": {
                "exact_cosine_repair_count": 0,
                "oracle_repair_count": 3,
                "source_failed_count": 4,
            },
            "d6/learned_calibrated_equivariant": {
                "exact_cosine_repair_count": 0,
                "oracle_repair_count": 4,
                "source_failed_count": 4,
            },
        },
        "valid": True,
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    source = campaign.get("source", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("diagnostics_manifest_sha256")
        != DIAGNOSTICS_MANIFEST_SHA256
        or source.get("campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or source.get("result_manifest_sha256")
        != SOURCE_RESULT_MANIFEST_SHA256
        or source.get("artifact_manifest_sha256")
        != SOURCE_ARTIFACT_MANIFEST_SHA256
        or campaign.get("summary")
        != {
            "completed": 20,
            "failed": 0,
            "fitted_parameters": 0,
            "requested": 20,
            "trained_frontends": 0,
            "trained_models": 0,
            "trained_probes": 0,
        }
        or campaign.get("aggregates") != _expected_aggregate()
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid frozen scalar-interface campaign {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")
    return campaign


def _details(results_path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(results_path)
    details: list[dict[str, Any]] = []
    result_paths: list[Path] = []
    diagnostic_paths: list[Path] = []
    source_failed_count = 0
    reachability_pass_count = 0
    for preset in PRESETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                path = (
                    results_path.parent
                    / "runs"
                    / preset
                    / condition
                    / f"seed_{seed}"
                    / "result.json"
                )
                detail = json.loads(path.read_text(encoding="utf-8"))
                diagnostics = Path(detail.get("artifacts", {}).get("diagnostics", ""))
                source = detail.get("source", {})
                source_result = Path(source.get("result", ""))
                expected_failed = seed in SOURCE_FAILURES[(preset, condition)]
                if (
                    detail.get("schema_version") != EXPERIMENT_SCHEMA
                    or detail.get("hypothesis_id") != HYPOTHESIS_ID
                    or detail.get("status") != "completed"
                    or detail.get("evidence_role") != EVIDENCE_ROLE
                    or detail.get("preset") != preset
                    or detail.get("condition") != condition
                    or int(detail.get("seed", -1)) != seed
                    or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                    or not detail.get("scientific_fingerprint")
                    or detail.get("gates", {}).get("validity") is not True
                    or bool(detail["gates"]["source_failed"]) != expected_failed
                    or source.get("campaign_sha256") != SOURCE_CAMPAIGN_SHA256
                    or not diagnostics.is_file()
                    or _sha256(diagnostics)
                    != detail.get("artifacts", {}).get("diagnostics_sha256")
                    or not source_result.is_file()
                    or _sha256(source_result) != source.get("result_sha256")
                    or detail.get("state_record", {}).get("before")
                    != detail.get("state_record", {}).get("after")
                    or not _finite(detail)
                ):
                    raise ValueError(f"invalid frozen scalar-interface cell {path}")
                replay_error = max(
                    float(detail["regimes"][regime]["replay"][key])
                    for regime in REGIMES
                    for key in (
                        "direct_vs_injected_natural_maximum_absolute_posterior_error",
                        "direct_vs_stored_task_metrics_maximum_absolute_error",
                    )
                )
                if replay_error > 2e-6:
                    raise ValueError(f"invalid replay in {path}")
                if expected_failed:
                    source_failed_count += 1
                    if all(
                        float(detail["regimes"][regime]["oracle"]["exact_bin_reachability"])
                        >= float(detail["regimes"][regime]["task_accuracy_floor"])
                        for regime in REGIMES
                    ):
                        reachability_pass_count += 1
                detail["_source_model_parameters"] = int(
                    json.loads(source_result.read_text(encoding="utf-8"))[
                        "model_parameters"
                    ]
                )
                details.append(detail)
                result_paths.append(path)
                diagnostic_paths.append(diagnostics)
    if (
        len(details) != 20
        or source_failed_count != 10
        or reachability_pass_count != 10
        or _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256
        or _manifest_sha256(diagnostic_paths) != DIAGNOSTICS_MANIFEST_SHA256
    ):
        raise ValueError("frozen scalar-interface evidence manifest changed")
    listed = {
        item["experiment_id"]: item for item in campaign.get("results", [])
    }
    if set(listed) != {detail["experiment_id"] for detail in details}:
        raise ValueError("frozen scalar-interface result index changed")
    exceptional = next(
        detail
        for detail in details
        if detail["preset"] == "d10"
        and detail["condition"] == "learned_calibrated_equivariant"
        and detail["seed"] == 29
    )
    expected_bins = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    if (
        exceptional["classification"] != "continuation_or_answer_row_failure"
        or any(
            exceptional["regimes"][regime]["oracle"]["contexts"][0][
                "reachable_target_bins"
            ]
            != expected_bins
            for regime in REGIMES
        )
    ):
        raise ValueError("exceptional four-bin scalar-interface result changed")
    return details


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "source_failed": float(detail["gates"]["source_failed"]),
        "exact_cosine_both_shifts": float(
            detail["gates"]["exact_cosine_both_shifts"]
        ),
        "oracle_both_shifts": float(detail["gates"]["oracle_both_shifts"]),
        "negative_cosine_both_shifts": float(
            detail["gates"]["negative_cosine_both_shifts"]
        ),
        "shuffled_cosine_both_shifts": float(
            detail["gates"]["shuffled_cosine_both_shifts"]
        ),
        "validity": float(detail["gates"]["validity"]),
    }
    for regime in REGIMES:
        record = detail["regimes"][regime]
        prefix = "composition" if regime == "composition" else "extrapolation"
        result[f"{prefix}_natural_accuracy"] = float(
            record["natural_metrics"]["exact_bin_accuracy"]
        )
        result[f"{prefix}_exact_cosine_accuracy"] = float(
            record["exact_cosine_metrics"]["exact_bin_accuracy"]
        )
        result[f"{prefix}_oracle_accuracy"] = float(
            record["oracle"]["minimum_cross_entropy_selection_metrics"][
                "exact_bin_accuracy"
            ]
        )
        result[f"{prefix}_target_bin_reachability"] = float(
            record["oracle"]["exact_bin_reachability"]
        )
    return result


def build_frozen_scalar_interface_decomposition_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    summaries = [
        {
            "experiment_id": detail["experiment_id"],
            "preset": detail["preset"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "classification": detail["classification"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "source_result": detail["source"]["result"],
            "source_result_sha256": detail["source"]["result_sha256"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "Architecture-family task failures are upstream of the scalar embedding",
            "category": "frozen causal interface decomposition",
            "description": (
                "A registered post-outcome diagnostic substitutes exact cosine and "
                "searches the existing one-dimensional scalar interface in twenty "
                "frozen d6/d10 structured checkpoints."
            ),
            "question": (
                "Do failed architecture-family cells recover under exact cosine, or "
                "does failure lie in the scalar coordinate or frozen continuation?"
            ),
            "prediction": (
                "Exact cosine repairs at least eight of ten source failures, including "
                "the declared stratum minima, with semantic controls remaining negative."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_exact_cosine_repairs_one_of_ten",
            "evidence_count": 20,
            "direct_experiment_count": 20,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "all twenty structured d6/d10 source checkpoints; frozen scalar "
                "embedding and continuation; composition and extrapolation cohorts"
            ),
            "success_criteria": {
                "uniform_exact_cosine_repair": "at least 8/10 source failures with registered stratum minima",
                "one_dimensional_oracle_repair": "10/10 source failures and 10/10 positive controls",
                "negative_control_ceiling": 1,
            },
            "subclaims": {
                "uniformly_upstream_exact_cosine": "contradicted_one_of_ten",
                "registered_minimum_ce_oracle_sufficiency": "not_supported_nine_of_ten",
                "target_bin_reachability_clears_source_floor": "supported_ten_of_ten_secondary",
                "scalar_coordinate_or_boundary_failure": "supported_eight_cells",
                "sensor_scalar_estimation_failure": "supported_one_cell",
                "local_continuation_or_answer_row_hole": "supported_one_cell",
                "semantic_specificity": "supported_zero_of_ten_per_control",
                "full_registered_hypothesis": "not_confirmed",
            },
            "tags": [
                "tinyllm",
                "registered-post-outcome",
                "frozen-checkpoint",
                "causal-intervention",
                "scalar-interface",
                "negative-result",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Exact cosine repairs 1/10 source failures; the registered scalar "
                "oracle repairs 9/10, and one d10 learned checkpoint omits four bins "
                "on the admissible interval."
            ),
            "confidence": 0.99,
            "confidence_assessment": "high_confidence_registered_post_outcome_localization",
            "num_direct_experiments": 20,
            "successful_direct_experiments": 20,
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "checkpoint_summaries": summaries,
            },
            "key_insights": [
                "The architecture-family failures are not uniformly sensor-estimation failures.",
                "An admissible scalar coordinate repairs nine of ten failed frozen systems.",
                "Target-bin reachability clears the source accuracy floor in all ten failed systems.",
                "D10 learned seed 29 has a localized four-bin hole on [-1, 1].",
            ],
            "suggested_hypotheses": [
                "A wider frozen scalar-domain scan separates encoder saturation from an answer-curve hole."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": summaries},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "diagnostics_manifest_sha256": DIAGNOSTICS_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "artifact_validation": "all 20 result, diagnostic, and source-result digests revalidated",
        },
    }


def build_frozen_scalar_interface_decomposition_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    architectures = {"d6": [6, 384], "d10": [10, 640]}
    output = []
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=summary,
                primary_metric=float(summary["exact_cosine_both_shifts"]),
                model_architecture=architectures[detail["preset"]],
                model_parameters=detail["_source_model_parameters"],
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["source"]["model_checkpoint"],
                observations=[
                    f"Preset: {detail['preset']}.",
                    f"Condition: {detail['condition']}.",
                    f"Classification: {detail['classification']}.",
                    "No model, front end, probe, observer, or map was trained or fit.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return output


def store_frozen_scalar_interface_decomposition_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_frozen_scalar_interface_decomposition_meta_hypothesis(results_path)
    experiments = build_frozen_scalar_interface_decomposition_experiment_results(
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
            raise RuntimeError("frozen scalar-interface ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(results["ids"]),
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
    "build_frozen_scalar_interface_decomposition_experiment_results",
    "build_frozen_scalar_interface_decomposition_meta_hypothesis",
    "store_frozen_scalar_interface_decomposition_meta_hypothesis",
]
