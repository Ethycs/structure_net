"""Build and store the TinyLLM observable-C3 temporal quotient result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-c3-temporal-quotient-d6.v1"
DIAGNOSTIC_SCHEMA = "nal.tinyllm-c3-temporal-positive-control.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-quotient-training-v1"
EVIDENCE_ROLE = "prospective_c3_temporal_quotient_d6"
DIAGNOSTIC_EVIDENCE_ROLE = (
    "registered_result_localization_no_checkpoint_loads"
)
CLASSIFICATION = "c3_positive_control_task_failure"
DIAGNOSTIC_CLASSIFICATION = (
    "invariant_sensor_valid_trained_continuation_readout_"
    "extrapolation_unreliable"
)
CAMPAIGN_SHA256 = (
    "e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc"
)
RESULT_MANIFEST_SHA256 = (
    "7dfdcf1ff80fe20a975fe6a7d1311dc92e3ff1a396a6da9550c91835a568a0ff"
)
ARTIFACT_MANIFEST_SHA256 = (
    "a0b90484863346cf2a5e0ef8be65cac3a221cfa50a373f88c9ead07a0cd351a1"
)
DIAGNOSTIC_SHA256 = (
    "9e3b635b5de997641fec9722989e13047fdf6409a953e0829e8677aa2f4e1e4d"
)
IMPLEMENTATION_SHA256 = (
    "25f8d57e32aa7fd7081836d4eda401d5c84469e882ec60adc196101b94e9c4d3"
)
CAMPAIGN_FINGERPRINT = (
    "6b74b0acb406943eadd21323477a1d609b49d88df60940b3e943b06fdd8c73e0"
)
RUNNER_SHA256 = (
    "9b2cd0e3ce3752b7eea80d5859c11880a9d3732fb48b58306e34eab4f080d5ec"
)
ANALYSIS_SHA256 = (
    "89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3"
)
STAGE0_RUNNER_SHA256 = (
    "dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6"
)
DIAGNOSTIC_RUNNER_SHA256 = (
    "f4635fbe8546788b164c878f8cb422a0ed8fc674d0f1c402125d2cb1021a8577"
)
PREREGISTRATION_SHA256 = (
    "33537d32c4e8361fb325a1792a3779a2cb76a0248ef2945d727ac78dcb17d71a"
)
REPORT_SHA256 = (
    "a9ce3cf9fab481ede50862ac2a2a4edcd98784e748085aa0403b4f380b8907d0"
)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
CUTS = ("frontend", "post_attention", "post_mlp", "full")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_campaign.py"
)
ANALYSIS_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_analysis.py"
)
STAGE0_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_training.py"
)
DIAGNOSTIC_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_c3_temporal_quotient_positive_control.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-quotient-d6-preregistration.md"
)
REPORT_PATH = Path(
    "docs/08 - Analysis/2026-08-11_tinyllm-c3-temporal-quotient-d6.md"
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


def _artifact_paths(campaign_path: Path) -> list[Path]:
    root = campaign_path.parent
    return [
        root / "runs" / "analytic" / f"seed_{seed}" / name
        for seed in SEEDS
        for name in ("model.pt", "frontend.pt", "diagnostics.pt")
    ]


def _maximum_conditional_log_loss_gain(detail: Mapping[str, Any]) -> float:
    return max(
        float(
            detail["representation"]["cuts"][cut]["conditional_deck"]
            ["evaluations"][regime]["conditional_log_loss_gain"]
        )
        for cut in CUTS
        for regime in REGIMES
    )


def _maximum_conditional_deck_accuracy(detail: Mapping[str, Any]) -> float:
    return max(
        float(
            detail["representation"]["cuts"][cut]["conditional_deck"]
            ["evaluations"][regime]["balanced_accuracy"]
        )
        for cut in CUTS
        for regime in REGIMES
    )


def _minimum_registered_target_correlation(detail: Mapping[str, Any]) -> float:
    return min(
        float(
            detail["representation"]["cuts"][cut]["semantic"]
            ["evaluations"][regime]["target_correlation"]
        )
        for cut in ("frontend", "full")
        for regime in REGIMES
    )


def _campaign(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    summary = campaign.get("summary", {})
    aggregates = campaign.get("aggregates", {})
    arms = aggregates.get("arms", {})
    source_hashes = campaign.get("source_hashes", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or source_hashes.get("runner") != RUNNER_SHA256
        or source_hashes.get("analysis") != ANALYSIS_SHA256
        or source_hashes.get("stage0_runner") != STAGE0_RUNNER_SHA256
        or source_hashes.get("preregistration") != PREREGISTRATION_SHA256
        or summary
        != {
            "completed": 5,
            "failed": 0,
            "requested": 15,
            "reused": 0,
            "scheduled": 5,
            "stopped_by_positive_control": 10,
        }
        or aggregates.get("valid") is not True
        or aggregates.get("complete_requested_grid") is not False
        or aggregates.get("controls_pass") is not True
        or aggregates.get("classification") != CLASSIFICATION
        or aggregates.get("primary_hypothesis_pass") is not False
        or aggregates.get("required_seed_passes") != 4
        or arms.get("analytic", {}).get("valid_count") != 5
        or arms.get("analytic", {}).get("natural_task_pass_count") != 2
        or arms.get("analytic", {}).get("representation_pass_count") != 5
        or arms.get("analytic", {}).get("causal_pass_count") != 5
        or arms.get("analytic", {}).get("exact_action_pass_count") != 5
        or arms.get("analytic", {}).get("joint_pass_count") != 2
        or arms.get("analytic", {}).get("success") is not False
        or arms.get("raw", {}).get("valid_count") != 0
        or arms.get("learned_c3", {}).get("valid_count") != 0
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid C3 temporal quotient campaign {path}")

    source_artifacts = {
        RUNNER_PATH: RUNNER_SHA256,
        ANALYSIS_PATH: ANALYSIS_SHA256,
        STAGE0_RUNNER_PATH: STAGE0_RUNNER_SHA256,
        DIAGNOSTIC_RUNNER_PATH: DIAGNOSTIC_RUNNER_SHA256,
        PREREGISTRATION_PATH: PREREGISTRATION_SHA256,
        REPORT_PATH: REPORT_SHA256,
    }
    for source_path, expected in source_artifacts.items():
        if _sha256(source_path) != expected:
            raise ValueError(f"source artifact changed: {source_path}")

    result_paths = _result_paths(path)
    if _manifest_sha256(result_paths) != RESULT_MANIFEST_SHA256:
        raise ValueError("C3 analytic result manifest changed")
    if _manifest_sha256(_artifact_paths(path)) != ARTIFACT_MANIFEST_SHA256:
        raise ValueError("C3 checkpoint/front-end/diagnostic manifest changed")

    details = [json.loads(item.read_text(encoding="utf-8")) for item in result_paths]
    task_count = 0
    fingerprints: set[str] = set()
    for detail, result_path in zip(details, result_paths):
        seed = int(detail.get("seed", -1))
        gates = detail.get("gates", {})
        artifacts = detail.get("artifacts", {})
        if (
            detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("experiment_id")
            != f"tinyllm-c3-temporal-d6-analytic-seed{seed}"
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("arm") != "analytic"
            or seed not in SEEDS
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("source_hashes") != source_hashes
            or gates.get("validity") is not True
            or gates.get("representation") is not True
            or gates.get("causal_all_cuts") is not True
            or gates.get("exact_action") is not True
            or gates.get("identity_replay") is not True
            or any(gates.get("target_derangement_pass_by_regime", {}).values())
            or artifacts.get("checkpoint_reload_pass") is not True
            or artifacts.get("diagnostics_reload_pass") is not True
            or Path(artifacts.get("result", "")) != result_path
            or _sha256(Path(artifacts["checkpoint"]))
            != artifacts.get("checkpoint_sha256")
            or _sha256(Path(artifacts["frontend_checkpoint"]))
            != artifacts.get("frontend_checkpoint_sha256")
            or _sha256(Path(artifacts["diagnostics"]))
            != artifacts.get("diagnostics_sha256")
            or detail.get("feature_contract_before", {}).get("pass") is not True
            or detail.get("feature_contract_after", {}).get("pass") is not True
            or not _finite(detail)
        ):
            raise ValueError(f"invalid C3 analytic cell seed {seed}")
        task_count += int(gates["natural_task"])
        fingerprints.add(str(detail["scientific_fingerprint"]))
    if task_count != 2 or len(fingerprints) != 5:
        raise ValueError("C3 analytic direct result population changed")

    diagnostic_path = path.parent / "positive_control_diagnostic.json"
    if _sha256(diagnostic_path) != DIAGNOSTIC_SHA256:
        raise ValueError("C3 no-checkpoint diagnostic changed")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (
        diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA
        or diagnostic.get("hypothesis_id") != HYPOTHESIS_ID
        or diagnostic.get("evidence_role") != DIAGNOSTIC_EVIDENCE_ROLE
        or diagnostic.get("status") != "completed"
        or diagnostic.get("classification") != DIAGNOSTIC_CLASSIFICATION
        or diagnostic.get("campaign", {}).get("sha256") != CAMPAIGN_SHA256
        or diagnostic.get("campaign", {}).get("result_manifest_sha256")
        != RESULT_MANIFEST_SHA256
        or diagnostic.get("counts")
        != {
            "causal_all_cuts_pass": 5,
            "checkpoints_loaded": 0,
            "natural_task_pass": 2,
            "optimizer_steps": 0,
            "representation_pass": 5,
            "trained_parameters": 0,
        }
        or not all(diagnostic.get("gates", {}).values())
        or not _finite(diagnostic)
    ):
        raise ValueError("invalid C3 no-checkpoint diagnostic")
    return campaign, details, diagnostic


def _direct_summary(
    campaign: Mapping[str, Any],
    details: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = []
    for detail in details:
        task_metrics = {
            regime: {
                "exact_bin_accuracy": detail["task_metrics"][regime][
                    "exact_bin_accuracy"
                ],
                "posterior_mean_correlation": detail["task_metrics"][regime][
                    "posterior_mean_correlation"
                ],
                "target_cross_entropy": detail["task_metrics"][regime][
                    "target_cross_entropy"
                ],
                "predicted_bin_coverage": detail["task_metrics"][regime][
                    "predicted_bin_coverage"
                ],
            }
            for regime in REGIMES
        }
        cells.append(
            {
                "experiment_id": detail["experiment_id"],
                "arm": detail["arm"],
                "seed": detail["seed"],
                "validity": detail["gates"]["validity"],
                "natural_task_pass": detail["gates"]["natural_task"],
                "representation_pass": detail["gates"]["representation"],
                "causal_all_cuts_pass": detail["gates"]["causal_all_cuts"],
                "exact_action_pass": detail["gates"]["exact_action"],
                "identity_replay_pass": detail["gates"]["identity_replay"],
                "target_derangement_pass_by_regime": detail["gates"][
                    "target_derangement_pass_by_regime"
                ],
                "minimum_registered_target_correlation": (
                    _minimum_registered_target_correlation(detail)
                ),
                "maximum_conditional_deck_accuracy": (
                    _maximum_conditional_deck_accuracy(detail)
                ),
                "maximum_conditional_log_loss_gain": (
                    _maximum_conditional_log_loss_gain(detail)
                ),
                "task_metrics": task_metrics,
                "scientific_fingerprint": detail["scientific_fingerprint"],
            }
        )
    return {
        "classification": campaign["aggregates"]["classification"],
        "campaign_fingerprint": campaign["campaign_fingerprint"],
        "population": campaign["aggregates"]["arms"]["analytic"],
        "cells": cells,
    }


def _diagnostic_summary(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evidence_role": diagnostic["evidence_role"],
        "classification": diagnostic["classification"],
        "counts": diagnostic["counts"],
        "gates": diagnostic["gates"],
        "fixed_no_model_positive_control": diagnostic[
            "fixed_no_model_positive_control"
        ],
        "method_boundaries": diagnostic["method_boundaries"],
    }


def build_c3_temporal_quotient_meta_hypothesis(
    results_path: Path,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    campaign, details, diagnostic = _campaign(results_path)
    direct = _direct_summary(campaign, details)
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": (
                "Exact C3 architectural invariance yields a useful temporal "
                "quotient in matched d6 TinyLLM systems"
            ),
            "category": "prospective richer-group architecture experiment",
            "description": (
                "A staged raw, analytic, and exact learned-C3 campaign tests "
                "whether an observable temporal quotient remains task-bearing, "
                "branch-contracted, causally sufficient, and naturally useful "
                "under composition and extrapolation."
            ),
            "question": (
                "Does architectural C3 invariance make the declared temporal "
                "quotient both causally valid and useful to a trained TinyLLM?"
            ),
            "prediction": (
                "The analytic positive control and learned-C3 populations each "
                "pass their complete joint gate in at least four of five seeds."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_analytic_positive_control_two_of_five_stop"
            ),
            "evidence_count": 6,
            "direct_experiment_count": 5,
            "derived_diagnostic_count": 1,
            "evidence_pedigree": EVIDENCE_ROLE,
            "tested_scope": (
                "five analytic d6 seeds under prospective composition and "
                "outside-speed extrapolation gates"
            ),
            "success_criteria": {
                "required_joint_seed_passes": 4,
                "analytic_population_runs_first": True,
                "conditional_cells_require_analytic_pass": True,
                "representation": {
                    "target_correlation_minimum": 0.90,
                    "conditional_deck_accuracy_maximum": 0.3834,
                    "conditional_log_loss_gain_maximum": 0.02,
                },
                "natural_task": {
                    "composition_accuracy_minimum": 0.50,
                    "extrapolation_accuracy_minimum": 0.35,
                    "composition_cross_entropy_maximum": 1.80,
                    "extrapolation_cross_entropy_maximum": 2.20,
                },
            },
            "subclaims": {
                "full_matched_d6_hypothesis": (
                    "not_confirmed_analytic_positive_control_two_of_five"
                ),
                "exact_analytic_c3_representation": "supported_five_of_five",
                "analytic_causal_closure_all_four_cuts": (
                    "supported_five_of_five"
                ),
                "analytic_exact_action_contract": "supported_five_of_five",
                "analytic_natural_extrapolating_utility": (
                    "failed_population_gate_two_of_five"
                ),
                "target_derangement_specificity": (
                    "supported_zero_preservation_passes_in_both_shifts"
                ),
                "fixed_no_model_temporal_solution": (
                    "supported_post_outcome_no_fit_diagnostic"
                ),
                "learned_exact_c3_usefulness": (
                    "untested_preregistered_positive_control_stop"
                ),
                "raw_comparison": "untested_preregistered_positive_control_stop",
                "d10_extension": "unauthorized",
            },
            "tags": [
                "tinyllm",
                "c3",
                "temporal-quotient",
                "architectural-invariance",
                "causal-intervention",
                "positive-control-stop",
                "negative-primary-result",
            ],
            "explicitly_not_tested": [
                "The learned-C3 arm did not run and has no outcome.",
                "The raw matched arm did not run and has no outcome.",
                "D10 was neither scheduled nor licensed.",
                "No step, loss, optimizer, seed, or threshold rescue was tested.",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The analytic representation and causal gates pass five of five, "
                "but the complete natural-task gate passes only two of five; the "
                "prospective stop rule therefore prevents raw and learned cells."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_for_the_registered_five_seed_stop_and_narrow_subclaims"
            ),
            "num_direct_experiments": 5,
            "successful_direct_experiments": 5,
            "failed_direct_experiments": 0,
            "primary_seed_passes": 2,
            "descriptive_metrics": direct,
            "post_outcome_diagnostic": _diagnostic_summary(diagnostic),
            "key_insights": [
                (
                    "Exact C3 invariance creates a target-bearing, branch-neutral, "
                    "causally closed representation in every analytic seed."
                ),
                (
                    "Natural trained extrapolating utility is a separate obligation "
                    "and fails the population gate."
                ),
                (
                    "The no-model analytic temporal rule passes the frozen task "
                    "floors, localizing the current failure downstream of the "
                    "invariant sensor without rescuing the campaign."
                ),
                (
                    "The stop rule avoids six thousand conditional optimizer steps "
                    "and preserves learned-C3 as unknown rather than negative."
                ),
            ],
            "suggested_hypotheses": [
                (
                    "A prospectively typed temporal operator and ordered metric "
                    "decoder can use the exact C3 carrier reliably under outside-"
                    "speed extrapolation."
                ),
                (
                    "After that positive control passes, a task-only exact-C3 "
                    "learned sensor approaches the analytic carrier without a "
                    "support-relative downstream chart."
                ),
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {
            "direct_tests": direct["cells"],
            "derived_diagnostic": _diagnostic_summary(diagnostic),
        },
        "source_artifacts": [
            str(results_path),
            str(results_path.parent / "positive_control_diagnostic.json"),
            str(report_path),
        ],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "checkpoint_frontend_diagnostics_manifest_sha256": (
                ARTIFACT_MANIFEST_SHA256
            ),
            "positive_control_diagnostic_sha256": DIAGNOSTIC_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
            "runner_sha256": RUNNER_SHA256,
            "analysis_sha256": ANALYSIS_SHA256,
            "stage0_runner_sha256": STAGE0_RUNNER_SHA256,
            "diagnostic_runner_sha256": DIAGNOSTIC_RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "report_sha256": REPORT_SHA256,
            "artifact_validation": (
                "campaign, five analytic results, fifteen model/front-end/"
                "diagnostic artifacts, source identities, gate counts, exact "
                "action, replay, causal closure, target-changing controls, and "
                "the separate no-checkpoint diagnostic revalidated"
            ),
        },
    }


def build_c3_temporal_quotient_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    _campaign_record, details, _diagnostic = _campaign(results_path)
    results = []
    for detail in details:
        completed = datetime.fromisoformat(
            detail["completed_at"].replace("Z", "+00:00")
        )
        metrics: dict[str, float] = {
            "validity": 1.0,
            "natural_task_pass": float(detail["gates"]["natural_task"]),
            "representation_pass": 1.0,
            "causal_all_cuts_pass": 1.0,
            "exact_action_pass": 1.0,
            "identity_replay_pass": 1.0,
            "minimum_registered_target_correlation": (
                _minimum_registered_target_correlation(detail)
            ),
            "maximum_conditional_deck_accuracy": (
                _maximum_conditional_deck_accuracy(detail)
            ),
            "maximum_conditional_log_loss_gain": (
                _maximum_conditional_log_loss_gain(detail)
            ),
            "optimizer_steps": 600.0,
            "learned_encoder_parameters": 0.0,
        }
        for regime in REGIMES:
            task = detail["task_metrics"][regime]
            metrics[f"{regime}_exact_accuracy"] = float(
                task["exact_bin_accuracy"]
            )
            metrics[f"{regime}_posterior_mean_correlation"] = float(
                task["posterior_mean_correlation"]
            )
            metrics[f"{regime}_target_cross_entropy"] = float(
                task["target_cross_entropy"]
            )
        results.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=float(detail["gates"]["natural_task"]),
                model_architecture=[6, 384, 16],
                model_parameters=int(detail["parameter_counts"]["total"]),
                training_time=float(detail["training"]["seconds"]),
                model_checkpoint=detail["artifacts"]["checkpoint"],
                observations=[
                    f"Analytic d6 positive-control seed={detail['seed']}.",
                    (
                        "Representation, exact action, replay, and four-cut "
                        "causal closure pass."
                    ),
                    (
                        "Primary metric is the preregistered natural-task seed "
                        "pass; this is not a learned-C3 result."
                    ),
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[],
                timestamp=completed,
            )
        )
    return results


def store_c3_temporal_quotient_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_c3_temporal_quotient_meta_hypothesis(results_path)
    experiments = build_c3_temporal_quotient_experiment_results(
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
            raise RuntimeError("C3 temporal quotient ChromaDB read-back failed")
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
    "build_c3_temporal_quotient_experiment_results",
    "build_c3_temporal_quotient_meta_hypothesis",
    "store_c3_temporal_quotient_meta_hypothesis",
]
