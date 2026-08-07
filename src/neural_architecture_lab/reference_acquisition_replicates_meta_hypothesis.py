"""Build and store TinyLLM repeated-reference acquisition evidence."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-reference-acquisition-replicates.v1"
HYPOTHESIS_ID = "tinyllm-reference-acquisition-replicates-v1"
EVIDENCE_ROLE = "preregistered_frozen_checkpoint_causal_acquisition_intervention"
IMPLEMENTATION_SHA256 = (
    "6c3cc4463b2c515280c778461e4606dca6ffb19f08d11cb5b7a852a942c7df77"
)
CAMPAIGN_SHA256 = (
    "269fd948f0d6fee8916bbe3cb94c1d87f76572e43c103b52fc8775fa9653031e"
)
RESULT_MANIFEST_SHA256 = (
    "8181d48f99850d5e487b5209d648373410bea3b46b2eafcb58f57118ed898c1c"
)
REPEAT_ARRAY_SHA256 = (
    "8886c1f6ad0fd307720748e44b1741edb656fdb9e29e913c7ad059b72397eef4"
)
DENOISER_SHA256 = (
    "7373577fd3a6535c9183eeeac393389dd54f5bf9bb64f5baeb6f4b371e66f702"
)
SOURCE_ORIENTATION_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_ORIENTATION_IMPLEMENTATION_SHA256 = (
    "990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70"
)
SOURCE_NOISE_SHA256 = (
    "b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7"
)
SOURCE_CALIBRATED_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_CALIBRATED_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
METHODS = ("analytic_circular_mean", "learned_equivariant_moment")
COUNTS = (1, 4, 16, 64)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
EXPECTED_ORACLE = {
    "analytic_calibrated": {seed: False for seed in SEEDS},
    "learned_calibrated_equivariant": {
        7: True,
        17: False,
        29: False,
        41: True,
        53: False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite_numbers(item) for item in value)
    return True


def _expected_method_counts() -> dict[str, Any]:
    return {
        method: {
            str(count): {
                condition: (5 if count == 64 else 0)
                for condition in CONDITIONS
            }
            for count in COUNTS
        }
        for method in METHODS
    }


def _campaign(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    config = value.get("configuration", {})
    summary = value.get("summary", {})
    aggregates = value.get("aggregates", {})
    provenance = value.get("provenance", {})
    artifacts = value.get("artifacts", {})
    repeat_contracts = value.get("repeat_noise_contracts", {})
    scaling = value.get("scaling", {})
    comparison = value.get("learned_analytic_comparison", {})
    orientation_path = Path(provenance.get("orientation_campaign", ""))
    repeat_path = Path(artifacts.get("repeat_arrays", ""))
    denoiser_path = Path(artifacts.get("denoisers", ""))
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or value.get("schema_version") != EXPERIMENT_SCHEMA
        or value.get("hypothesis_id") != HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("evidence_role") != EVIDENCE_ROLE
        or value.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or tuple(config.get("seeds", ())) != SEEDS
        or tuple(config.get("repeat_counts", ())) != COUNTS
        or float(config.get("sigma_radians", math.nan)) != 0.175
        or float(config.get("ridge_ratio", math.nan)) != 0.01
        or float(config.get("accuracy_loss_ceiling", math.nan)) != 0.03
        or int(config.get("required_seed_passes", -1)) != 4
        or int(config.get("control_pass_ceiling", -1)) != 1
        or config.get("allow_underpowered") is not False
        or summary
        != {
            "requested": 10,
            "completed": 10,
            "failed": 0,
            "reused": 0,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "new_representation_probes": 0,
            "fitted_acquisition_denoisers": 4,
            "fitted_acquisition_parameters": 12,
        }
        or aggregates.get("valid") is not False
        or aggregates.get("classification") != "invalid"
        or aggregates.get("method_pass_counts") != _expected_method_counts()
        or aggregates.get("misgrouped_control_pass_counts")
        != {condition: 0 for condition in CONDITIONS}
        or aggregates.get("target_oracle_pass_counts")
        != {"analytic_calibrated": 0, "learned_calibrated_equivariant": 2}
        or aggregates.get("target_shuffled_pass_counts")
        != {condition: 0 for condition in CONDITIONS}
        or repeat_contracts.get("pair_shared_repeat_contract") is not True
        or repeat_contracts.get("source_first_draw_contract") is not True
        or float(repeat_contracts.get("maximum_pair_shared_error", math.inf)) != 0.0
        or float(repeat_contracts.get("maximum_source_first_draw_error", math.inf))
        != 0.0
        or scaling.get("contract") is not True
        or not all(
            -0.60 <= float(scaling.get("slopes", {}).get(regime, math.inf)) <= -0.40
            for regime in REGIMES
        )
        or comparison.get("contract") is not True
        or any(
            float(comparison.get("rmse_gap_at_maximum_count", {}).get(regime, math.inf))
            > 0.002
            for regime in REGIMES
        )
        or provenance.get("orientation_campaign_sha256")
        != SOURCE_ORIENTATION_SHA256
        or provenance.get("orientation_implementation_sha256")
        != SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
        or provenance.get("orientation_noise_sha256") != SOURCE_NOISE_SHA256
        or provenance.get("calibrated_campaign_sha256")
        != SOURCE_CALIBRATED_SHA256
        or provenance.get("calibrated_implementation_sha256")
        != SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
        or not orientation_path.is_file()
        or _sha256(orientation_path) != SOURCE_ORIENTATION_SHA256
        or not repeat_path.is_file()
        or _sha256(repeat_path) != REPEAT_ARRAY_SHA256
        or artifacts.get("repeat_arrays_sha256") != REPEAT_ARRAY_SHA256
        or not denoiser_path.is_file()
        or _sha256(denoiser_path) != DENOISER_SHA256
        or artifacts.get("denoisers_sha256") != DENOISER_SHA256
        or not _finite_numbers(value)
    ):
        raise ValueError(f"invalid repeated-reference acquisition campaign {path}")
    return value


@lru_cache(maxsize=8)
def _details(path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(path)
    indexed = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in campaign.get("results", [])
    }
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"invalid repeated-reference result index {path}")

    output: list[dict[str, Any]] = []
    paths: list[Path] = []
    fingerprints: set[str] = set()
    expected_method_passes = {
        method: {str(count): count == 64 for count in COUNTS}
        for method in METHODS
    }
    for condition in CONDITIONS:
        for seed in SEEDS:
            entry = indexed[(condition, seed)]
            result_path = Path(entry["path"])
            detail = json.loads(result_path.read_text(encoding="utf-8"))
            provenance = detail.get("provenance", {})
            gates = detail.get("gates", {})
            source_orientation_path = Path(provenance.get("orientation_result", ""))
            source_result_path = Path(provenance.get("source_result", ""))
            model_path = Path(provenance.get("model_checkpoint", ""))
            frontend_path = Path(provenance.get("frontend_checkpoint", ""))
            source = (
                json.loads(source_result_path.read_text(encoding="utf-8"))
                if source_result_path.is_file()
                else {}
            )
            oracle_expected = EXPECTED_ORACLE[condition][seed]
            if (
                _sha256(result_path) != entry.get("result_sha256")
                or detail.get("schema_version") != EXPERIMENT_SCHEMA
                or detail.get("hypothesis_id") != HYPOTHESIS_ID
                or detail.get("status") != "completed"
                or detail.get("evidence_role") != EVIDENCE_ROLE
                or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
                or detail.get("experiment_id") != entry.get("experiment_id")
                or detail.get("condition") != condition
                or int(detail.get("seed", -1)) != seed
                or detail.get("scientific_fingerprint")
                != entry.get("scientific_fingerprint")
                or detail.get("method_passes") != expected_method_passes
                or entry.get("method_passes") != expected_method_passes
                or any(gates.get(name) is not True for name in (
                    "source_metric_replay_contract",
                    "inherited_representation_contract",
                    "repeat_noise_contract",
                    "finite_numerical_contract",
                    "causal_write_finite_contract",
                    "system_state_unchanged_contract",
                    "validity_contract",
                ))
                or float(detail.get("maximum_source_metric_replay_error", math.inf))
                != 0.0
                or detail.get("misgrouped_control", {}).get("task_gate") is not False
                or detail.get("causal_ceiling", {}).get("target_oracle_gate")
                is not oracle_expected
                or detail.get("causal_ceiling", {}).get("target_shuffled_gate")
                is not False
                or not source_orientation_path.is_file()
                or _sha256(source_orientation_path)
                != provenance.get("orientation_result_sha256")
                or not source_result_path.is_file()
                or _sha256(source_result_path) != provenance.get("source_result_sha256")
                or source.get("condition") != condition
                or int(source.get("seed", -1)) != seed
                or not model_path.is_file()
                or _sha256(model_path) != provenance.get("model_checkpoint_sha256")
                or not frontend_path.is_file()
                or _sha256(frontend_path)
                != provenance.get("frontend_checkpoint_sha256")
                or not _finite_numbers(detail)
            ):
                raise ValueError(
                    f"invalid repeated-reference acquisition result {result_path}"
                )
            fingerprint = detail["scientific_fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(
                    f"duplicate repeated-reference fingerprint {result_path}"
                )
            fingerprints.add(fingerprint)
            detail["_source_model_parameters"] = int(source["model_parameters"])
            output.append(detail)
            paths.append(result_path)
    if _manifest_sha256(paths) != RESULT_MANIFEST_SHA256:
        raise ValueError(f"invalid repeated-reference result manifest {path}")
    return output


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        methods[method] = {}
        for count in (16, 64):
            cell = detail["evaluations"][method][str(count)]
            methods[method][str(count)] = {
                "task_gate": bool(cell["task_gate"]),
                "accuracy_loss": {
                    regime: float(cell["accuracy_loss_from_clean"][regime])
                    for regime in REGIMES
                },
                "task_accuracy": {
                    regime: float(
                        cell["regimes"][regime]["task"]["exact_bin_accuracy"]
                    )
                    for regime in REGIMES
                },
                "frontend_cosine_correlation": {
                    regime: float(
                        cell["regimes"][regime]["frontend"]["cosine_pearson"]
                    )
                    for regime in REGIMES
                },
            }
    oracle = detail["causal_ceiling"]
    return {
        "complete_seed_gate": bool(
            methods["analytic_circular_mean"]["64"]["task_gate"]
            and methods["learned_equivariant_moment"]["64"]["task_gate"]
            and oracle["target_oracle_gate"]
        ),
        "methods": methods,
        "misgrouped_control_pass": bool(
            detail["misgrouped_control"]["task_gate"]
        ),
        "target_oracle_pass": bool(oracle["target_oracle_gate"]),
        "target_shuffled_pass": bool(oracle["target_shuffled_gate"]),
        "target_oracle_accuracy_loss": {
            regime: float(oracle["target_oracle_accuracy_loss_from_clean"][regime])
            for regime in REGIMES
        },
        "target_oracle_patch_norm": {
            regime: float(
                oracle["regimes"][regime]["target_oracle"]["mean_patch_norm"]
            )
            for regime in REGIMES
        },
    }


def _checkpoint_means(details: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        summaries = [
            _detail_summary(detail)
            for detail in details
            if detail["condition"] == condition
        ]
        output[condition] = {
            method: {
                str(count): {
                    regime: mean(
                        summary["methods"][method][str(count)]["accuracy_loss"][
                            regime
                        ]
                        for summary in summaries
                    )
                    for regime in REGIMES
                }
                for count in (16, 64)
            }
            for method in METHODS
        }
        output[condition]["target_oracle_passes"] = sum(
            int(summary["target_oracle_pass"]) for summary in summaries
        )
        output[condition]["mean_target_oracle_patch_norm"] = {
            regime: mean(
                summary["target_oracle_patch_norm"][regime]
                for summary in summaries
            )
            for regime in REGIMES
        }
    return output


def build_reference_acquisition_replicates_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-07_tinyllm-reference-acquisition-replicates.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    direct_tests = [
        {
            "experiment_id": detail["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "scientific_fingerprint": detail["scientific_fingerprint"],
            "provenance": detail["provenance"],
            "gates": detail["gates"],
            **_detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM repeated-reference acquisition",
            "category": "mechanistic_interpretability",
            "description": (
                "A preregistered frozen-system intervention averages nested "
                "independent orientation-reference observations and retains a "
                "one-step residual causal ceiling."
            ),
            "question": (
                "Does inverse-square acquisition precision restore both structured "
                "systems, with the retained true-coordinate write validating the "
                "causal ceiling?"
            ),
            "prediction": (
                "Both acquisition estimators and the true-coordinate positive "
                "control must pass at least four of five seeds per arm."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "invalid_causal_ceiling_nonportable_acquisition_subresult_supported"
            ),
            "evidence_count": len(direct_tests),
            "direct_experiment_count": len(direct_tests),
            "independent_checkpoint_count": 5,
            "power_profile": "five_seed_preregistered_frozen_acquisition_intervention",
            "evidence_role": EVIDENCE_ROLE,
            "tested_scope": (
                "ten retained d8/N3 calibrated checkpoints; 0.175-radian "
                "independent orientation error; m=1,4,16,64"
            ),
            "subclaims": {
                "complete_preregistered_hypothesis": "invalid",
                "analytic_acquisition_repair_at_m64": (
                    "supported_five_of_five_both_arms"
                ),
                "learned_acquisition_repair_at_m64": (
                    "supported_five_of_five_both_arms"
                ),
                "inverse_square_angular_scaling": "supported_both_shifts",
                "learned_higher_moment_advantage": "rejected_matches_analytic_mean",
                "same_reference_membership_specificity": (
                    "supported_zero_of_five_misgrouped_both_arms"
                ),
                "one_step_true_coordinate_ceiling": (
                    "rejected_zero_of_five_analytic_two_of_five_learned"
                ),
                "measurement_precision_bottleneck": (
                    "supported_as_preregistered_subresult_not_full_confirmation"
                ),
                "model_or_frontend_retraining_licensed": "no",
            },
            "tags": [
                "tinyllm",
                "reference-acquisition",
                "orientation-noise",
                "inverse-square-scaling",
                "circular-mean",
                "equivariant-denoiser",
                "frozen-checkpoint",
                "invalid-positive-control",
            ],
            "explicitly_not_tested": campaign["method_boundaries"]
            + [
                "Correlated or biased reference errors.",
                "Real-instrument acquisition cost.",
                "Whether multi-step residual transport repairs the failed local ceiling.",
                "Natural-language behavior or architecture-population prevalence.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Both acquisition methods pass all ten systems at m=64 with exact "
                "inverse-square scaling and clean controls, but the preregistered "
                "true-coordinate residual ceiling passes only 0/5 analytic and "
                "2/5 learned checkpoints, forcing the locked invalid classification."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "preregistered_direct_subresult_with_failed_registered_positive_control"
            ),
            "independent_checkpoint_count": 5,
            "num_direct_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "scaling": campaign["scaling"],
                "learned_analytic_comparison": campaign[
                    "learned_analytic_comparison"
                ],
                "checkpoint_means": _checkpoint_means(details),
            },
            "key_insights": [
                "Both frozen structured systems recover only at m=64 on the locked grid, bracketing the task threshold between 1.25 and 2.51 degrees effective error.",
                "Measured angular error follows inverse-square scaling and correct repeat membership is necessary.",
                "The three-parameter equivariant estimator is indistinguishable from the analytic circular mean, so learned denoising is unnecessary for unbiased Gaussian error.",
                "A one-step local residual task-gradient write is not a global acquisition ceiling at large off-manifold displacement.",
                "The complete hypothesis remains invalid even though its direct acquisition branch is strongly supported."
            ],
            "suggested_hypotheses": [
                "Locally relinearized residual transport follows the actual reference-induced residual path and succeeds where the one-step write fails.",
                "If integrated tangent transport still fails, the scalar task-gradient chart is non-integrable into the trained residual manifold.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_tests},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "repeat_array_sha256": REPEAT_ARRAY_SHA256,
            "denoiser_sha256": DENOISER_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "source_orientation_campaign_sha256": SOURCE_ORIENTATION_SHA256,
            "source_orientation_implementation_sha256": (
                SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
            ),
            "source_noise_sha256": SOURCE_NOISE_SHA256,
            "source_calibrated_campaign_sha256": SOURCE_CALIBRATED_SHA256,
            "source_calibrated_implementation_sha256": (
                SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
            ),
        },
    }


def build_reference_acquisition_replicates_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    completed = datetime.fromisoformat(
        _campaign(results_path)["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        summary = _detail_summary(detail)
        metrics = {
            "campaign_valid": 0.0,
            "complete_seed_gate": float(summary["complete_seed_gate"]),
            "analytic_m64_task_gate": float(
                summary["methods"]["analytic_circular_mean"]["64"]["task_gate"]
            ),
            "learned_m64_task_gate": float(
                summary["methods"]["learned_equivariant_moment"]["64"][
                    "task_gate"
                ]
            ),
            "target_oracle_pass": float(summary["target_oracle_pass"]),
            "target_shuffled_pass": float(summary["target_shuffled_pass"]),
            "misgrouped_control_pass": float(summary["misgrouped_control_pass"]),
        }
        for method in METHODS:
            for count in (16, 64):
                for regime in REGIMES:
                    metrics[f"{method}_m{count}_{regime}_accuracy_loss"] = summary[
                        "methods"
                    ][method][str(count)]["accuracy_loss"][regime]
        for regime in REGIMES:
            metrics[f"target_oracle_{regime}_accuracy_loss"] = summary[
                "target_oracle_accuracy_loss"
            ][regime]
            metrics[f"target_oracle_{regime}_patch_norm"] = summary[
                "target_oracle_patch_norm"
            ][regime]
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(summary["complete_seed_gate"]),
                model_architecture=[8, int(detail["_source_model_parameters"])],
                model_parameters=int(detail["_source_model_parameters"]),
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}.",
                    "All model, front-end, and answer-head parameters remained frozen.",
                    "Both acquisition methods pass at m=64; the complete seed gate also requires the causal ceiling.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "The direct acquisition intervention passes while the registered one-step true-coordinate residual ceiling fails."
                ],
                timestamp=completed,
            )
        )
    return output


def store_reference_acquisition_replicates_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_reference_acquisition_replicates_meta_hypothesis(results_path)
    experiments = build_reference_acquisition_replicates_experiment_results(
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

        storage_root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(storage_root / "experiment_queue"),
                sent_dir=str(storage_root / "experiment_sent"),
                rejected_dir=str(storage_root / "experiment_rejected"),
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
        hypothesis_readback = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        experiment_readback = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis_readback.get("ids") != [HYPOTHESIS_ID]
            or len(experiment_readback.get("ids", [])) != len(experiments)
            or {
                item.get("hypothesis_id")
                for item in experiment_readback.get("metadatas", [])
            }
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("reference-acquisition ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(experiment_readback["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_reference_acquisition_replicates_experiment_results",
    "build_reference_acquisition_replicates_meta_hypothesis",
    "store_reference_acquisition_replicates_meta_hypothesis",
]
