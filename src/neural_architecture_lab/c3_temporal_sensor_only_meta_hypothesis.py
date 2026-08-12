"""Build and store the five-seed C3 temporal sensor-only result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-temporal-sensor-only.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-sensor-only-v1"
EVIDENCE_ROLE = "prospective_c3_sensor_only_fixed_operator_training"
CLASSIFICATION = "task_only_sensor_acquisition_supported"
CAMPAIGN_SHA256 = "4d023d9f9d77f64b1fe75970acd63461cfd5a64aa1de60c7954a2c671c427012"
RESULT_MANIFEST_SHA256 = "a23d19892f112645a7d3b5401d1528a0eb612a5d6db26520fe3514246c4c6d1a"
CHECKPOINT_MANIFEST_SHA256 = "e83832872f29d072d710859e022f4e17d1b6da6a9e16b63049f41d4ea2eb01a0"
IMPLEMENTATION_SHA256 = "f4e0755c228475a7edaf52f458e964c6df5f6beb737db81833a5dd7a692e3b88"
CAMPAIGN_FINGERPRINT = "bdad912fedee9e3323ca6e3c29cd330131b80b5015a647eac89cd1cfdc4c07a7"
RUNNER_SHA256 = "7f4a5990f2f9a56bcaad0032d7cf9eca20f74b599a0684337c48dcdf9593b3ed"
PREREGISTRATION_SHA256 = "e0c16420ea79130b22d6940b6f4943b5dda6bd3e048418eff078c0190e69a2bb"
REPORT_SHA256 = "7d6803d028af33bd220bb42133e41fa40dd206afde18ad67807008f88530815d"
CAPACITY_RESULT_SHA256 = "6a01db25ebc2ed15d202884c39f16db685d5218647b0bb209e2e5a737696a383"
SEEDS = (7, 17, 29, 41, 53)
ARMS = ("learned_true", "learned_target_shuffled")
REGIMES = ("composition", "extrapolation")
CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_only/"
    "20260811_preregistered/campaign_results.json"
)
RUNNER_PATH = Path("experiments/structure_net/tinyllm_c3_temporal_sensor_only.py")
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-sensor-only-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-c3-temporal-sensor-only.md"
)
CAPACITY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_function_class/"
    "20260811_preregistered/result.json"
)


@lru_cache(maxsize=256)
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


def _manifest(paths: Iterable[Path]) -> str:
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


def _result_paths(path: Path) -> list[Path]:
    root = path.parent
    return [root / "runs" / f"seed_{seed}" / "result.json" for seed in SEEDS]


def _checkpoint_paths(path: Path) -> list[Path]:
    root = path.parent
    return [
        root / "runs" / f"seed_{seed}" / f"{arm}_{point}.pt"
        for seed in SEEDS
        for arm in ARMS
        for point in ("midpoint", "final")
    ]


def _campaign(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    expected_summary = {
        "completed": 5,
        "failed": 0,
        "primary_optimizer_steps": 6000,
        "requested": 5,
        "resume_verification_optimizer_steps": 3000,
        "reused": 4,
        "scheduled": 1,
        "tinyllm_models_instantiated": 0,
    }
    aggregates = campaign.get("aggregates", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or campaign.get("source_hashes", {}).get("runner") != RUNNER_SHA256
        or campaign.get("source_hashes", {}).get("preregistration")
        != PREREGISTRATION_SHA256
        or campaign.get("source_hashes", {}).get("capacity_result")
        != CAPACITY_RESULT_SHA256
        or campaign.get("capacity_license", {}).get("sha256")
        != CAPACITY_RESULT_SHA256
        or campaign.get("summary") != expected_summary
        or aggregates.get("valid") is not True
        or aggregates.get("analytic_positive_control_pass") is not True
        or aggregates.get("learned_true_joint_pass_count") != 5
        or aggregates.get("learned_target_shuffled_joint_pass_count") != 0
        or aggregates.get("specificity_pass") is not True
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("primary_hypothesis_pass") is not True
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("checkpoint_manifest_sha256")
        != CHECKPOINT_MANIFEST_SHA256
        or float(campaign.get("maximum_peak_cuda_allocated_gb", 1.0))
        > 0.07
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid C3 temporal sensor-only campaign {path}")

    for source, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (REPORT_PATH, REPORT_SHA256),
        (CAPACITY_RESULT_PATH, CAPACITY_RESULT_SHA256),
    ):
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"source artifact changed: {source}")
    results = _result_paths(path)
    checkpoints = _checkpoint_paths(path)
    if _manifest(results) != RESULT_MANIFEST_SHA256:
        raise ValueError("C3 sensor-only result manifest changed")
    if _manifest(checkpoints) != CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("C3 sensor-only checkpoint manifest changed")

    details = [json.loads(item.read_text(encoding="utf-8")) for item in results]
    fingerprints = set()
    for detail, result_path in zip(details, results):
        seed = int(detail.get("seed", -1))
        gates = detail.get("gates", {})
        accounting = detail.get("accounting", {})
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or seed not in SEEDS
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("source_hashes", {}).get("runner") != RUNNER_SHA256
            or gates
            != {
                "analytic_positive_control": True,
                "learned_target_shuffled_joint": False,
                "learned_true_joint": True,
                "matched_initialization": True,
                "validity": True,
            }
            or accounting
            != {
                "primary_optimizer_steps": 1200,
                "resume_verification_optimizer_steps": 600,
                "tinyllm_models_instantiated": 0,
                "trainable_parameters_per_arm": 184,
            }
            or detail.get("analytic_positive_control", {}).get("pass") is not True
            or Path(detail.get("artifacts", {}).get("result", "")) != result_path
            or not _finite(detail)
        ):
            raise ValueError(f"invalid C3 temporal sensor-only seed {seed}")
        fingerprints.add(detail["scientific_fingerprint"])
        if detail.get("protocol_contract", {}).get("pass") is not True:
            raise ValueError(f"invalid C3 sensor protocol seed {seed}")
        for arm in ARMS:
            arm_result = detail.get("arms", {}).get(arm, {})
            expected_joint = arm == "learned_true"
            if (
                arm_result.get("state_changed") is not True
                or arm_result.get("reload", {}).get("pass") is not True
                or arm_result.get("exact_resume", {}).get("pass") is not True
                or arm_result.get("finite") is not True
                or arm_result.get("joint_pass") is not expected_joint
                or len(arm_result.get("training_history", [])) != 600
            ):
                raise ValueError(f"invalid C3 sensor arm {arm}/seed {seed}")
            for point in ("midpoint", "final"):
                artifact = arm_result[f"{point}_checkpoint"]
                artifact_path = Path(artifact["path"])
                if (
                    not artifact_path.is_file()
                    or _sha256(artifact_path) != artifact["sha256"]
                ):
                    raise ValueError(
                        f"invalid C3 sensor checkpoint {arm}/{point}/seed {seed}"
                    )
            for regime in REGIMES:
                cell = arm_result["regimes"][regime]
                if arm == "learned_true" and (
                    cell.get("joint_pass") is not True
                    or cell.get("task_pass") is not True
                    or cell.get("carrier_pass") is not True
                    or cell.get("action_pass") is not True
                    or float(cell.get("mean_aligned_unit_dot", 0.0)) < 0.90
                    or float(cell.get("aligned_coordinate_rmse", 1.0)) > 0.35
                    or float(cell.get("maximum_deck_action_error", 1.0)) > 2e-6
                ):
                    raise ValueError(
                        f"invalid true C3 sensor regime {regime}/seed {seed}"
                    )
    if len(fingerprints) != 5:
        raise ValueError("C3 sensor-only seed fingerprints are not independent")
    return campaign, details


def build_c3_temporal_sensor_only_meta_hypothesis(
    results_path: Path = CAMPAIGN_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details = _campaign(results_path)
    direct = []
    for detail in details:
        direct.append(
            {
                "experiment_id": detail["experiment_id"],
                "seed": detail["seed"],
                "gates": detail["gates"],
                "analytic_positive_control": detail["analytic_positive_control"],
                "arms": {
                    arm: {
                        "joint_pass": detail["arms"][arm]["joint_pass"],
                        "gauge": detail["arms"][arm]["gauge"],
                        "regimes": detail["arms"][arm]["regimes"],
                        "state_changed": detail["arms"][arm]["state_changed"],
                        "reload": detail["arms"][arm]["reload"],
                        "exact_resume": detail["arms"][arm]["exact_resume"],
                    }
                    for arm in ARMS
                },
                "accounting": detail["accounting"],
                "wall_seconds": detail["wall_seconds"],
                "peak_cuda_allocated_gb": detail["peak_cuda_allocated_gb"],
            }
        )
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "Task-only training acquires the exact-C3 temporal carrier when "
                "the operator and physical decoder are fixed"
            ),
            "category": "prospective typed sensor-only causal training",
            "description": (
                "Five randomly initialized 184-parameter exact-C3 sensors train "
                "against a frozen temporal law and interval decoder, with matched "
                "pair-preserving target-shuffled controls."
            ),
            "question": (
                "Can task loss alone recover a support-stable task-equivalent C3 "
                "carrier without a learned transformer continuation?"
            ),
            "prediction": (
                "At least four of five true sensors pass carrier and complete task "
                "gates on both shifts, with at most one shuffled pass."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": True,
            "confirmation_status": (
                "confirmed_task_only_sensor_acquisition_five_of_five"
            ),
            "evidence_count": 5,
            "direct_experiment_count": 5,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five random 184-parameter exact-C3 sensors; 4096 paired training "
                "examples; disjoint carrier reference, composition, and outside-speed "
                "extrapolation cohorts; fixed temporal operator and 16-bin decoder"
            ),
            "success_criteria": {
                "learned_true_joint_seed_passes_minimum": 4,
                "target_shuffled_joint_seed_passes_maximum": 1,
                "carrier_dot_minimum": 0.90,
                "carrier_rmse_maximum": 0.35,
                "deck_action_error_maximum": 2e-6,
                "complete_task_gate_required_both_shifts": True,
            },
            "subclaims": {
                "task_only_sensor_acquisition": "supported_five_of_five",
                "target_shuffle_specificity": "supported_zero_of_five",
                "composition_fixed_task": "supported_five_of_five",
                "extrapolation_fixed_task": "supported_five_of_five",
                "single_reference_gauge_carrier_fidelity": (
                    "supported_five_of_five_both_shifts"
                ),
                "exact_c3_action_after_training": "supported_five_of_five",
                "checkpoint_reload_and_exact_resume": "supported_ten_of_ten_arms",
                "tinyllm_continuation_required": "not_supported_in_this_task_scope",
                "tinyllm_utility": "not_tested_tinyllm_absent",
                "noise_and_new_group_transfer": "not_tested",
            },
            "tags": [
                "tinyllm",
                "c3",
                "sensor-only",
                "architectural-invariance",
                "typed-interface",
                "task-only-training",
                "positive-result",
                "extrapolation",
            ],
            "explicitly_not_tested": [
                "No TinyLLM model or transformer continuation was instantiated.",
                "No carrier supervision, analytic warm start, optimizer sweep, or loss sweep was used.",
                "Sensor noise, missing calibration, approximate actions, other groups, and language tasks were not tested.",
                "The result does not show superiority to the analytic carrier on this noiseless generator.",
            ],
        },
        "result": {
            "confirmed": True,
            "confirmation_reason": (
                "All five true sensors recover an almost exact analytic carrier "
                "and pass the complete fixed task under both shifts; all five "
                "matched target-shuffled controls fail."
            ),
            "confidence": 1.0,
            "confidence_assessment": (
                "preregistered_five_of_five_with_zero_of_five_matched_controls"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 5,
            "descriptive_metrics": campaign["aggregates"],
            "key_insights": [
                (
                    "The existing exact-C3 sensor is reliably learnable from task "
                    "loss once the temporal law and physical metric chart are fixed."
                ),
                (
                    "One reference-fitted global O2 gauge transfers with greater "
                    "than .99999 carrier alignment under both shifts in every seed."
                ),
                (
                    "The failed predecessor is localized to its learned TinyLLM "
                    "continuation/readout rather than invariant sensor acquisition."
                ),
                (
                    "Architectural symmetry pays constructive rent by removing deck "
                    "orientation from the trainable function class."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "The same typed sensor/operator/decoder separation remains "
                    "learnable when calibration or observations are noisy."
                ),
                (
                    "A prospective continuation constrained to preserve the typed "
                    "carrier retains this extrapolation result after reintegration."
                ),
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "capacity_result_sha256": CAPACITY_RESULT_SHA256,
            "artifact_validation": (
                "campaign, five results, twenty checkpoints, source identities, "
                "five independent fingerprints, matched initialization, ten arm "
                "reload/resume contracts, analytic controls, carrier/action/task "
                "gates, and shuffled specificity revalidated"
            ),
        },
    }


def build_c3_temporal_sensor_only_experiment_results(
    record: Mapping[str, Any],
    results_path: Path = CAMPAIGN_PATH,
) -> list[ExperimentResult]:
    del record
    _campaign_record, details = _campaign(results_path)
    results = []
    for detail in details:
        metrics: dict[str, float] = {
            "validity": 1.0,
            "learned_true_joint_pass": 1.0,
            "learned_target_shuffled_joint_pass": 0.0,
            "analytic_positive_control_pass": 1.0,
            "tinyllm_models_instantiated": 0.0,
            "trainable_parameters_per_arm": 184.0,
            "primary_optimizer_steps": 1200.0,
            "resume_verification_optimizer_steps": 600.0,
        }
        for regime in REGIMES:
            true = detail["arms"]["learned_true"]["regimes"][regime]
            shuffled = detail["arms"]["learned_target_shuffled"]["regimes"][regime]
            metrics[f"true_{regime}_accuracy"] = float(
                true["task"]["exact_bin_accuracy"]
            )
            metrics[f"true_{regime}_correlation"] = float(
                true["task"]["posterior_mean_correlation"]
            )
            metrics[f"true_{regime}_cross_entropy"] = float(
                true["task"]["target_cross_entropy"]
            )
            metrics[f"true_{regime}_carrier_dot"] = float(
                true["mean_aligned_unit_dot"]
            )
            metrics[f"true_{regime}_carrier_rmse"] = float(
                true["aligned_coordinate_rmse"]
            )
            metrics[f"shuffled_{regime}_accuracy"] = float(
                shuffled["task"]["exact_bin_accuracy"]
            )
        completed = datetime.fromisoformat(
            detail["completed_at"].replace("Z", "+00:00")
        )
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=1.0,
                model_architecture=[3, 16, 8, 2],
                model_parameters=184,
                training_time=float(detail["wall_seconds"]),
                model_checkpoint=detail["arms"]["learned_true"][
                    "final_checkpoint"
                ]["path"],
                observations=[
                    f"Sensor-only exact-C3 seed={detail['seed']}.",
                    "Matched true and pair-preserving target-shuffled arms.",
                    "TinyLLM absent; temporal operator and interval decoder frozen.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_c3_temporal_sensor_only_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_temporal_sensor_only_meta_hypothesis(results_path)
    experiments = build_c3_temporal_sensor_only_experiment_results(
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
            raise RuntimeError("C3 sensor-only ChromaDB read-back failed")
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
    "build_c3_temporal_sensor_only_experiment_results",
    "build_c3_temporal_sensor_only_meta_hypothesis",
    "store_c3_temporal_sensor_only_meta_hypothesis",
]
