"""Build and store the TinyLLM frozen scalar-domain extension result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-frozen-scalar-domain-extension.v1"
HYPOTHESIS_ID = "tinyllm-frozen-scalar-domain-extension-v1"
EVIDENCE_ROLE = "registered_outcome_informed_frozen_scalar_domain_diagnostic"
CLASSIFICATION = "continuation_answer_curve_hole_persists_to_radius_8"
RESULT_SHA256 = (
    "e0bbb6120272627de59acf639e886d434ab53ec7ed0fc11874d77864a4dd1312"
)
DIAGNOSTICS_SHA256 = (
    "95334b9475bd73111f20f07747ec716fe0bf3c5ff44c14c2a91b78c825218d69"
)
RUNNER_SHA256 = (
    "d252cff4522c2908b38446cb1d9cf32d83333801660951066753bd8ec49f856e"
)
PREREGISTRATION_SHA256 = (
    "1eec9eec7832124d5e4fd8d66632e2bede641cba1f4559f2ae677d909670a96e"
)
IMPLEMENTATION_SHA256 = (
    "494a6de53e14ebff9d285aa0d86af74ab4a6808ba5549c78b452dbf42e77bc4a"
)
FINGERPRINT = (
    "d4b0c2ab9fe0dfd0c1b1b30911e429e6c9d4fdf38456ca44ca8b3553a11e4c55"
)
SOURCE_CAMPAIGN_SHA256 = (
    "f71b85bc51f9694346fa23482bc63e1631e0918d55f5cae4f3a0d5ce09a47ba2"
)
SOURCE_RESULT_SHA256 = (
    "16deb857f4449c66a22eac347c2c7e2b2cf23fbd5a8a1ad0daaf39660bf910f0"
)
SOURCE_DIAGNOSTICS_SHA256 = (
    "3bfbf496e1cb85d79548849b533b4139c0737e7079a53e4e5bb2f974de7f9476"
)
REGIMES = ("composition", "extrapolation")
RADII = ("1", "2", "4", "8")
MISSING_BINS = (0, 1, 6, 15)
REACHABLE_BINS = (2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_frozen_scalar_domain_extension.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-scalar-domain-extension-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-frozen-scalar-domain-extension.md"
)


@lru_cache(maxsize=64)
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


def _result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = Path(result.get("artifacts", {}).get("diagnostics", ""))
    source_result = Path(result.get("source", {}).get("result", ""))
    source_diagnostics = Path(result.get("source", {}).get("diagnostics", ""))
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("classification") != CLASSIFICATION
        or result.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or result.get("scientific_fingerprint") != FINGERPRINT
        or result.get("gates")
        != {
            "cuda_valid": True,
            "finite": True,
            "source_bin_replay": True,
            "source_curve_replay": True,
            "state_unchanged": True,
            "validity": True,
        }
        or result.get("summary")
        != {
            "common_external_discovered_bin_count": 0,
            "fitted_parameters": 0,
            "source_missing_bin_count": 4,
            "trained_frontends": 0,
            "trained_models": 0,
            "trained_probes": 0,
        }
        or not diagnostics.is_file()
        or _sha256(diagnostics) != DIAGNOSTICS_SHA256
        or not source_result.is_file()
        or _sha256(source_result) != SOURCE_RESULT_SHA256
        or not source_diagnostics.is_file()
        or _sha256(source_diagnostics) != SOURCE_DIAGNOSTICS_SHA256
        or result.get("source", {}).get("campaign_sha256")
        != SOURCE_CAMPAIGN_SHA256
        or result.get("state_record", {}).get("before")
        != result.get("state_record", {}).get("after")
        or not _finite(result)
    ):
        raise ValueError(f"invalid frozen scalar-domain result {path}")
    if _sha256(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError(f"source artifact changed: {RUNNER_PATH}")
    if _sha256(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ValueError(f"source artifact changed: {PREREGISTRATION_PATH}")
    for regime in REGIMES:
        record = result["regimes"][regime]
        if (
            float(record["source_curve_replay_maximum_absolute_error"]) > 2e-6
            or tuple(record["source_reachable_bins_replayed"]) != REACHABLE_BINS
            or tuple(record["source_missing_bins"]) != MISSING_BINS
            or record["source_missing_bins_discovered_outside_encoder_image"] != []
        ):
            raise ValueError(f"invalid frozen scalar-domain replay for {regime}")
        for radius in RADII:
            ladder = record["ladders"][radius]
            if (
                tuple(ladder["reachable_target_bins"]) != REACHABLE_BINS
                or tuple(ladder["missing_target_bins"]) != MISSING_BINS
            ):
                raise ValueError(f"scalar-domain bin set changed at {regime}/{radius}")
    return result


def _summary(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = {
        "validity": 1.0,
        "common_external_discovered_bin_fraction": 0.0,
        "source_curve_replay_maximum_absolute_error": max(
            float(result["regimes"][regime]["source_curve_replay_maximum_absolute_error"])
            for regime in REGIMES
        ),
    }
    for regime in REGIMES:
        prefix = "composition" if regime == "composition" else "extrapolation"
        for radius in RADII:
            oracle = result["regimes"][regime]["ladders"][radius]["oracle"]
            metrics[f"{prefix}_radius_{radius}_reachability"] = float(
                oracle["exact_bin_reachability"]
            )
            metrics[f"{prefix}_radius_{radius}_oracle_accuracy"] = float(
                oracle["minimum_cross_entropy_selection_metrics"][
                    "exact_bin_accuracy"
                ]
            )
    return metrics


def build_frozen_scalar_domain_extension_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    metrics = _summary(result)
    direct = {
        "experiment_id": result["experiment_id"],
        "classification": result["classification"],
        "scientific_fingerprint": result["scientific_fingerprint"],
        "source_missing_bins": list(MISSING_BINS),
        "discovered_external_bins": {
            regime: result["regimes"][regime][
                "source_missing_bins_discovered_outside_encoder_image"
            ]
            for regime in REGIMES
        },
        **metrics,
    }
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "The exceptional TinyLLM scalar hole is caused by bounded encoder range",
            "category": "frozen scalar continuation localization",
            "description": (
                "An outcome-informed, registered, no-fit scan extends the frozen "
                "d10 learned seed-29 scalar input from radius one to radius eight."
            ),
            "question": (
                "Do answer bins 0, 1, 6, and 15 acquire winning scalar regions "
                "outside the retained tanh encoder image?"
            ),
            "prediction": (
                "All four source-missing bins appear at resolved scalars with "
                "1 < |s| <= 8 under both held-out shifts."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "not_confirmed_no_missing_bin_appears_through_radius_8",
            "evidence_count": 1,
            "direct_experiment_count": 1,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "one outcome-selected d10 learned-equivariant seed-29 checkpoint; "
                "frozen scalar embedding and continuation; resolved domain [-8,8]"
            ),
            "success_criteria": {
                "missing_bins": list(MISSING_BINS),
                "required_external_discoveries": 4,
                "required_shifts": list(REGIMES),
                "maximum_radius": 8,
                "grid_step": 1 / 2048,
            },
            "subclaims": {
                "bounded_encoder_range_explains_hole": "contradicted_zero_of_four_bins",
                "continuation_answer_curve_hole_persists": "supported_to_radius_8_at_registered_resolution",
                "winning_bin_set_stable_across_ladder": "supported_both_shifts",
                "source_curve_replay": "supported_max_error_2.0862e-7",
                "population_frequency": "not_tested_single_outcome_selected_checkpoint",
            },
            "tags": [
                "tinyllm",
                "registered-post-outcome",
                "frozen-checkpoint",
                "scalar-domain",
                "answer-curve",
                "negative-result",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "None of the four source-missing answer bins becomes posterior "
                "argmax through radius eight in either held-out shift."
            ),
            "confidence": 0.99,
            "confidence_assessment": "high_within_registered_single_checkpoint_grid",
            "num_direct_experiments": 1,
            "successful_direct_experiments": 1,
            "failed_direct_experiments": 0,
            "descriptive_metrics": direct,
            "key_insights": [
                "The missing decision regions are not just beyond the tanh encoder boundary.",
                "The winning-bin set is unchanged at radii one, two, four, and eight.",
                "The frozen composite scalar-to-posterior chart remains incomplete.",
            ],
            "suggested_hypotheses": [
                "A prospectively typed scalar/readout interface with explicit full-bin coverage prevents checkpoint-local answer-curve holes."
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {"direct_tests": [direct]},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "diagnostics_sha256": DIAGNOSTICS_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "scientific_fingerprint": FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_diagnostics_sha256": SOURCE_DIAGNOSTICS_SHA256,
            "artifact_validation": "result, diagnostics, source artifacts, state, and all ladder bin sets revalidated",
        },
    }


def build_frozen_scalar_domain_extension_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    source_result = json.loads(
        Path(result["source"]["result"]).read_text(encoding="utf-8")
    )
    architecture_result = json.loads(
        Path(source_result["source"]["result"]).read_text(encoding="utf-8")
    )
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id=result["experiment_id"],
            hypothesis_id=HYPOTHESIS_ID,
            metrics=_summary(result),
            primary_metric=0.0,
            model_architecture=[10, 640],
            model_parameters=int(architecture_result["model_parameters"]),
            training_time=float(result["wall_seconds"]),
            model_checkpoint=result["source"]["model_checkpoint"],
            observations=[
                f"Classification: {CLASSIFICATION}.",
                "Primary metric is the fraction of four source-missing bins discovered outside [-1,1].",
                "No model, front end, probe, observer, or map was trained or fit.",
                f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
            ],
            anomalies=[],
            timestamp=completed,
        )
    ]


def store_frozen_scalar_domain_extension_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_frozen_scalar_domain_extension_meta_hypothesis(results_path)
    experiments = build_frozen_scalar_domain_extension_experiment_results(
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
            or len(results.get("ids", [])) != 1
            or {item.get("hypothesis_id") for item in results.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("frozen scalar-domain ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": 1,
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
    "build_frozen_scalar_domain_extension_experiment_results",
    "build_frozen_scalar_domain_extension_meta_hypothesis",
    "store_frozen_scalar_domain_extension_meta_hypothesis",
]
