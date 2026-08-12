"""Build and store the C3 temporal feature-trajectory causal result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-temporal-feature-trajectory-causal.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-feature-trajectory-causal-v1"
EVIDENCE_ROLE = "registered_post_outcome_artifact_only_feature_trajectory_causal"
CLASSIFICATION = (
    "fixed_operator_available_but_frozen_continuation_cannot_use_"
    "projected_trajectory"
)
RESULT_SHA256 = "a0ac3315b03aa65df273539a24d8c08f51f12e8ceb702859cc16a282886ddf27"
RUNNER_SHA256 = "bcb10634a4928848d55608f3ef3fbe054fb3ed84c145f8311af465fb8a1fd17f"
PREREGISTRATION_SHA256 = (
    "73bf3b792ec44872b8362b69fb51e88826be44073739615a38a28fbdfe2a97ca"
)
REPORT_SHA256 = "eebc8e3ac6e6f5250c206d2f39d752272d252309c969704c2aa1975bf657860a"
SOURCE_CAMPAIGN_SHA256 = (
    "e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc"
)
FIXED_OPERATOR_RESULT_SHA256 = (
    "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
)
SEEDS = (7, 17, 29, 41, 53)
PASSING_SEEDS = (7, 53)
REGIMES = ("composition", "extrapolation")
ARMS = ("source", "last_consistent", "mean_consistent", "early_deranged")
RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_feature_trajectory_causal/"
    "20260811_d6_preregistered/result.json"
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_feature_trajectory_causal.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-feature-trajectory-causal-"
    "preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/"
    "2026-08-11_tinyllm-c3-temporal-feature-trajectory-causal.md"
)
SOURCE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_quotient/"
    "20260811_d6_preregistered/campaign_results.json"
)
FIXED_OPERATOR_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)


@lru_cache(maxsize=128)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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


def _expected_aggregates() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "fixed_mean_bypass_pass_count": 5,
        "last_consistent_task_pass_count": 2,
        "mean_consistent_shuffled_pass_count": 0,
        "mean_consistent_task_pass_count": 2,
        "required_seed_passes": 4,
        "shuffled_seed_pass_maximum": 1,
        "source_natural_task_pass_count": 2,
        "valid": True,
    }


def _result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    sources = result.get("source_hashes", {})
    if (
        _sha256(path) != RESULT_SHA256
        or result.get("schema_version") != EXPERIMENT_SCHEMA
        or result.get("hypothesis_id") != HYPOTHESIS_ID
        or result.get("status") != "completed"
        or result.get("evidence_role") != EVIDENCE_ROLE
        or result.get("aggregates") != _expected_aggregates()
        or sources.get("runner") != RUNNER_SHA256
        or sources.get("preregistration") != PREREGISTRATION_SHA256
        or sources.get("source_campaign") != SOURCE_CAMPAIGN_SHA256
        or sources.get("fixed_operator_result") != FIXED_OPERATOR_RESULT_SHA256
        or result.get("accounting")
        != {
            "checkpoints_loaded": 5,
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "target_using_fits": 0,
            "tinyllm_models_instantiated": 5,
        }
        or not _finite(result)
    ):
        raise ValueError(f"invalid C3 feature-trajectory result {path}")

    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (SOURCE_CAMPAIGN_PATH, SOURCE_CAMPAIGN_SHA256),
        (FIXED_OPERATOR_RESULT_PATH, FIXED_OPERATOR_RESULT_SHA256),
    ):
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")

    cells = result.get("cells", [])
    if len(cells) != 5 or {cell.get("seed") for cell in cells} != set(SEEDS):
        raise ValueError("C3 feature-trajectory checkpoint population changed")
    for cell in cells:
        seed = int(cell["seed"])
        expected_pass = seed in PASSING_SEEDS
        gates = cell.get("arm_seed_gates", {})
        if (
            cell.get("valid") is not True
            or cell.get("state_unchanged") is not True
            or cell.get("state_sha256_before") != cell.get("state_sha256_after")
            or cell.get("fixed_mean_bypass_both_shifts") is not True
            or float(cell.get("maximum_source_posterior_error", 1.0)) > 2e-6
            or float(cell.get("maximum_source_metric_error", 1.0)) > 2e-6
            or any(
                gates.get(arm, {}).get("true_both_shifts") is not expected_pass
                for arm in ("source", "last_consistent", "mean_consistent")
            )
            or gates.get("early_deranged", {}).get("true_both_shifts") is not False
            or any(
                gates.get(arm, {}).get("shuffled_both_shifts") is not False
                for arm in ARMS
            )
        ):
            raise ValueError(f"invalid C3 feature-trajectory seed {seed}")
        for regime in REGIMES:
            measured = cell.get("regimes", {}).get(regime, {})
            contracts = measured.get("feature_contracts", {})
            if (
                contracts.get("pass") is not True
                or measured.get("target_derangement_fixed_points") != 0
                or measured.get("fixed_mean_bypass", {}).get("task_gate") is not True
                or measured.get("arms", {}).get("early_deranged", {}).get("task_gate")
                is not False
                or any(
                    measured.get("arms", {}).get(arm, {}).get("shuffled_task_gate")
                    is not False
                    for arm in ARMS
                )
            ):
                raise ValueError(
                    f"invalid C3 feature-trajectory cell {seed}/{regime}"
                )
        for artifact, digest in (
            (cell["source"]["checkpoint"], cell["source"]["checkpoint_sha256"]),
            (
                cell["source"]["frontend_checkpoint"],
                cell["source"]["frontend_checkpoint_sha256"],
            ),
        ):
            artifact_path = Path(artifact)
            if not artifact_path.is_file() or _sha256(artifact_path) != digest:
                raise ValueError(f"source checkpoint changed: {artifact_path}")
    return result


def build_c3_temporal_feature_trajectory_causal_meta_hypothesis(
    results_path: Path = RESULT_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    result = _result(results_path)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "A fixed all-frame group-consistent feature trajectory repairs "
                "the frozen TinyLLM continuation"
            ),
            "category": "registered post-outcome artifact-only causal diagnostic",
            "description": (
                "The all-increment C3 statistic is converted into a denoised "
                "constant-speed carrier trajectory and patched before the learned "
                "sequence embedding of five frozen analytic d6 checkpoints."
            ),
            "question": (
                "Can the failed continuation use the sufficient all-frame temporal "
                "statistic when it is supplied at its existing feature interface?"
            ),
            "prediction": (
                "The mean-consistent trajectory passes both shifts in at least four "
                "seeds while target-deranged controls pass at most one."
            ),
            "created_at": result["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_fixed_operator_available_but_frozen_continuation_"
                "cannot_use_projected_trajectory"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five frozen analytic d6 TinyLLM checkpoints; unchanged 1024-example "
                "composition and extrapolation cohorts; source, last-consistent, "
                "mean-consistent, and early-deranged feature trajectories"
            ),
            "success_criteria": {
                "mean_consistent_seed_passes_minimum": 4,
                "last_consistent_specificity_comparator": True,
                "mean_consistent_shuffled_seed_passes_maximum": 1,
                "fixed_mean_bypass_seed_passes_required": 5,
                "complete_task_gate_required_both_shifts": True,
            },
            "subclaims": {
                "all_frame_trajectory_repairs_frozen_continuation": (
                    "not_supported_two_of_five"
                ),
                "last_step_trajectory_repairs_frozen_continuation": (
                    "not_supported_two_of_five"
                ),
                "fixed_all_frame_bypass_available": "supported_five_of_five",
                "target_derangement_specificity": "supported_zero_of_five",
                "early_frame_causal_sensitivity": "supported_five_of_five",
                "failure_downstream_of_analytic_carrier_and_temporal_projection": (
                    "supported_in_five_frozen_d6_checkpoints"
                ),
                "same_task_tinyllm_repair": "not_licensed",
                "nonconstant_or_corrupted_dynamics": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "feature-trajectory",
                "causal-patch",
                "frozen-checkpoint",
                "artifact-only",
                "no-training",
                "negative-result",
                "downstream-localization",
            ],
            "explicitly_not_tested": result["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The mean-consistent patch passes only the two already-successful "
                "seeds, while the matched fixed physical bypass passes all five."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "valid_registered_five_checkpoint_negative_with_positive_bypass"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 2,
            "failed_direct_experiments": 3,
            "primary_seed_passes": 2,
            "descriptive_metrics": result["aggregates"],
            "key_insights": [
                (
                    "A sufficient all-frame temporal statistic does not become a "
                    "sufficient TinyLLM computation merely by projecting the input "
                    "carrier sequence onto the corresponding trajectory."
                ),
                (
                    "The feature patch leaves the continuation population at the "
                    "same two-of-five natural utility result."
                ),
                (
                    "Deranging early carrier states strongly changes the posterior, "
                    "so the continuation uses early detail without extracting the "
                    "registered all-frame denoising benefit."
                ),
                (
                    "The fixed operator and physical decoder remain the supported "
                    "same-task implementation."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A typed learned temporal model earns incremental utility only "
                    "under identifiable nonconstant, missing-frame, or corrupted "
                    "dynamics where the fixed circular mean is insufficient."
                ),
                (
                    "A new task with an unknown group law separates useful learned "
                    "temporal structure from a known analytic baseline."
                ),
            ],
            "completed_at": result["completed_at"],
        },
        "evidence": {"direct_tests": result["cells"]},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "fixed_operator_result_sha256": FIXED_OPERATOR_RESULT_SHA256,
            "artifact_validation": (
                "result, runner, preregistration, report, source campaign, fixed "
                "operator result, five checkpoint pairs, dataset hashes, source "
                "replay, feature algebra, state identity, positive bypass, shuffled "
                "controls, and locked population classification revalidated"
            ),
        },
    }


def build_c3_temporal_feature_trajectory_causal_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = RESULT_PATH,
) -> list[ExperimentResult]:
    del record
    result = _result(results_path)
    completed = datetime.fromisoformat(
        result["completed_at"].replace("Z", "+00:00")
    )
    experiments = []
    for cell in result["cells"]:
        gates = cell["arm_seed_gates"]
        metrics: dict[str, float] = {
            "validity": 1.0,
            "source_true_both_shifts": float(gates["source"]["true_both_shifts"]),
            "last_true_both_shifts": float(
                gates["last_consistent"]["true_both_shifts"]
            ),
            "mean_true_both_shifts": float(
                gates["mean_consistent"]["true_both_shifts"]
            ),
            "mean_shuffled_both_shifts": float(
                gates["mean_consistent"]["shuffled_both_shifts"]
            ),
            "fixed_mean_bypass_both_shifts": 1.0,
            "optimizer_steps": 0.0,
            "parameters_changed": 0.0,
        }
        for regime in REGIMES:
            measured = cell["regimes"][regime]["arms"]
            for arm in ("source", "last_consistent", "mean_consistent"):
                arm_metrics = measured[arm]["metrics"]
                prefix = f"{regime}_{arm}"
                metrics[f"{prefix}_accuracy"] = float(
                    arm_metrics["exact_bin_accuracy"]
                )
                metrics[f"{prefix}_correlation"] = float(
                    arm_metrics["posterior_mean_correlation"]
                )
                metrics[f"{prefix}_cross_entropy"] = float(
                    arm_metrics["target_cross_entropy"]
                )
            metrics[f"{regime}_mean_js_from_source"] = float(
                measured["mean_consistent"]["effect_from_source"][
                    "mean_jensen_shannon_divergence"
                ]
            )
            metrics[f"{regime}_early_argmax_change"] = float(
                measured["early_deranged"]["effect_from_source"][
                    "argmax_change_fraction"
                ]
            )
        experiments.append(
            ExperimentResult(
                experiment_id=(
                    f"tinyllm-c3-temporal-feature-trajectory-causal-seed"
                    f"{cell['seed']}"
                ),
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=metrics["mean_true_both_shifts"],
                model_architecture=[6, 384, 16],
                model_parameters=29_951_232,
                training_time=0.0,
                model_checkpoint=cell["source"]["checkpoint"],
                observations=[
                    f"Frozen analytic d6 source seed={cell['seed']}.",
                    "All-frame group-consistent trajectory patched before sequence embedding.",
                    "Zero optimizer steps and zero changed parameters.",
                    "Primary metric is the registered mean-consistent joint seed pass.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return experiments


def store_c3_temporal_feature_trajectory_causal_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_temporal_feature_trajectory_causal_meta_hypothesis(
        results_path
    )
    experiments = build_c3_temporal_feature_trajectory_causal_experiment_results(
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
            or {
                item.get("hypothesis_id")
                for item in stored.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("C3 feature-trajectory ChromaDB read-back failed")
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
    "build_c3_temporal_feature_trajectory_causal_experiment_results",
    "build_c3_temporal_feature_trajectory_causal_meta_hypothesis",
    "store_c3_temporal_feature_trajectory_causal_meta_hypothesis",
]
