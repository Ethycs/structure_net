"""Persist predictive-circle TinyLLM evidence without overstating the result."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-semantic-quotient-circle-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-predictive-circle.v1"


def _read_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        results = json.load(source)
    if results.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported predictive-circle result schema")
    if results.get("status") != "completed":
        raise ValueError("predictive-circle campaign is not complete")
    runs = results.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("predictive-circle campaign contains no runs")
    return results


def _mean_metric(
    runs: Sequence[Mapping[str, Any]],
    condition: str,
    metric: str,
) -> float:
    values = [
        float(run["primary_metrics"][metric])
        for run in runs
        if run["condition"] == condition
    ]
    if not values:
        raise ValueError(f"campaign is missing condition {condition!r}")
    return statistics.fmean(values)


def build_predictive_circle_meta_hypothesis(
    results_path: Path,
) -> dict[str, Any]:
    """Build a conservative aggregate from the completed d6/d8 campaign."""
    campaign = _read_results(results_path)
    runs = campaign["runs"]
    presets = sorted({run["preset"] for run in runs})
    seeds = sorted({int(run["seed"]) for run in runs})
    conditions = sorted({run["condition"] for run in runs})
    expected = len(presets) * len(seeds) * len(conditions)
    if len(runs) != expected:
        raise ValueError(
            f"campaign is incomplete: expected {expected} matched runs, found {len(runs)}"
        )

    trained_accuracy = _mean_metric(runs, "trained", "in_domain_exact_bin_accuracy")
    label_accuracy = _mean_metric(
        runs, "label_shuffled", "in_domain_exact_bin_accuracy"
    )
    time_accuracy = _mean_metric(runs, "time_shuffled", "in_domain_exact_bin_accuracy")
    trained_shift = _mean_metric(
        runs, "trained", "nuisance_shift_exact_bin_accuracy"
    )
    trained_bottleneck = _mean_metric(
        runs, "trained", "posterior_h1_bottleneck_to_target"
    )
    label_bottleneck = _mean_metric(
        runs, "label_shuffled", "posterior_h1_bottleneck_to_target"
    )
    time_bottleneck = _mean_metric(
        runs, "time_shuffled", "posterior_h1_bottleneck_to_target"
    )
    trained_pullback = _mean_metric(
        runs, "trained", "pullback_fisher_normalized_h1_lifetime"
    )
    time_pullback = _mean_metric(
        runs, "time_shuffled", "pullback_fisher_normalized_h1_lifetime"
    )

    direct_evidence = [
        {
            "evidence_role": "direct_test",
            "experiment_id": run["experiment_id"],
            "preset": run["preset"],
            "seed": run["seed"],
            "condition": run["condition"],
            "model_parameters": run["model_parameters"],
            **{key: float(value) for key, value in run["primary_metrics"].items()},
        }
        for run in runs
    ]
    criteria = {
        preset: campaign["aggregates"][preset]["pre_registered_criteria"]
        for preset in presets
    }
    all_criteria_passed = all(
        value is True
        for preset_criteria in criteria.values()
        for value in preset_criteria.values()
    )
    if all_criteria_passed != bool(campaign["claim_status"]["confirmed"]):
        raise ValueError("campaign claim status disagrees with its criterion records")

    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM recovers a predictive-circle semantic quotient",
            "category": "training",
            "description": (
                "A matched d6/d8 synthetic-sensor study of whether task-posterior "
                "Fisher geometry and final-residual pullback Fisher geometry recover "
                "the known S1 quotient while rejecting shuffled controls."
            ),
            "question": (
                "Does successful circular future-phase prediction recover target-aligned "
                "H1 and remain stable under nuisance shift?"
            ),
            "prediction": (
                "Trained d6 and d8 models exceed 80% exact-bin accuracy, produce a "
                "bootstrap-stable target-aligned H1, beat shuffled controls, and retain "
                "the effect under nuisance shift."
            ),
            "created_at": campaign["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "partial_synthetic_success_not_confirmed",
            "evidence_count": len(direct_evidence),
            "direct_experiment_count": len(direct_evidence),
            "tested_scope": "synthetic predictive S1 with direct token serialization",
            "success_criteria": criteria,
            "tags": [
                "tinyllm",
                "persistent-homology",
                "semantic-quotient",
                "fisher-rao",
                "pullback-fisher",
                "synthetic-sensor",
                "multi-seed",
            ],
            "explicitly_not_tested": campaign["unsupported_or_deferred"]
            + campaign["claim_status"]["does_not_establish"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Every trained model class recovered target-aligned topology and beat "
                "shuffled controls, but d6 and d8 mean exact-bin accuracy remained below "
                "the pre-registered 80% threshold and nuisance-shift performance was weak."
            ),
            "confidence": 0.0,
            "confidence_assessment": "descriptive_three_seed_campaign",
            "effect_size": trained_accuracy - label_accuracy,
            "effect_size_kind": "mean_trained_minus_label_shuffled_exact_bin_accuracy",
            "independent_seed_count": len(seeds),
            "model_class_count": len(presets),
            "num_direct_experiments": len(direct_evidence),
            "successful_direct_experiments": len(direct_evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "trained_accuracy_mean": trained_accuracy,
                "label_shuffled_accuracy_mean": label_accuracy,
                "time_shuffled_accuracy_mean": time_accuracy,
                "trained_nuisance_shift_accuracy_mean": trained_shift,
                "trained_target_bottleneck_mean": trained_bottleneck,
                "label_shuffled_target_bottleneck_mean": label_bottleneck,
                "time_shuffled_target_bottleneck_mean": time_bottleneck,
                "trained_pullback_h1_mean": trained_pullback,
                "time_shuffled_pullback_h1_mean": time_pullback,
            },
            "key_insights": [
                "All six trained d6/d8 runs formed bootstrap-stable posterior H1 classes.",
                "Training reduced posterior-to-target bottleneck distance relative to label and time shuffling.",
                "Both model classes beat label-shuffled chance accuracy by roughly 66 percentage points.",
                "The topology appeared by the first 100-step checkpoint and remained through step 600.",
            ],
            "unexpected_findings": [
                "Time-shuffled models retained strong raw and pullback-Fisher H1 despite much lower task accuracy.",
                "Increasing from d6 to d8 did not improve in-domain exact-bin accuracy.",
                "Nuisance-shift accuracy remained below 31% for every trained run.",
                "The original 80% exact-bin threshold failed for both model classes.",
            ],
            "suggested_hypotheses": [
                "Training with explicit nuisance intervention improves OOD accuracy while preserving target-aligned H1.",
                "Target bottleneck distance predicts OOD accuracy better than raw H1 lifetime.",
                "The result survives at least five independent seeds and a held-out nuisance family.",
                "True zigzag persistence localizes the emergence of the semantic class across checkpoints and blocks.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_evidence},
        "source_artifacts": [str(results_path)],
    }


def build_predictive_circle_experiment_results(
    record: Mapping[str, Any],
    results_path: Path,
) -> list[ExperimentResult]:
    """Convert all 18 matched arms to the NAL experiment-result contract."""
    campaign = _read_results(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    root = results_path.parent
    results: list[ExperimentResult] = []
    for run in campaign["runs"]:
        metrics = {key: float(value) for key, value in run["primary_metrics"].items()}
        metrics["accuracy"] = metrics["in_domain_exact_bin_accuracy"]
        checkpoint = run.get("model_checkpoint")
        observations = [
            f"Condition: {run['condition']}.",
            "Task posterior was restricted to 16 circular answer tokens.",
            "Semantic topology was evaluated at the fixed query position.",
        ]
        anomalies = []
        if run["condition"] == "time_shuffled":
            anomalies.append(
                "Strong H1 may persist without target alignment or high task accuracy."
            )
        results.append(
            ExperimentResult(
                experiment_id=run["experiment_id"],
                hypothesis_id=record["hypothesis"]["id"],
                metrics=metrics,
                primary_metric=metrics["in_domain_exact_bin_accuracy"],
                model_architecture=[
                    int(run["model_config"]["n_layer"]),
                    int(run["model_config"]["n_head"]),
                    int(run["model_config"]["n_embd"]),
                    int(run["model_config"]["vocab_size"]),
                ],
                model_parameters=int(run["model_parameters"]),
                training_time=float(run["training_seconds"]),
                training_history=run["training_history"],
                model_checkpoint=str(root / checkpoint) if checkpoint else None,
                observations=observations,
                anomalies=anomalies,
                timestamp=completed_at,
            )
        )
    return results


def store_predictive_circle_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    """Write the conservative aggregate and optionally upsert it into ChromaDB."""
    record = build_predictive_circle_meta_hypothesis(results_path)
    experiment_results = build_predictive_circle_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [result.experiment_id for result in experiment_results],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import (
            LoggingConfig,
            StandardizedLogger,
        )

        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(queue_dir),
                sent_dir="data/experiment_sent",
                rejected_dir="data/experiment_rejected",
                enable_wandb=False,
                auto_upload=False,
                enable_chromadb=True,
                chromadb_path=str(chromadb_path),
            )
        )
        logger.log_hypothesis(record["hypothesis"])
        storage["result_hashes"] = [
            logger.log_experiment_result(result) for result in experiment_results
        ]
        storage["chromadb_path"] = str(chromadb_path)
        storage["hypotheses_collection"] = "hypotheses"
        storage["experiments_collection"] = "experiments"
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "META_HYPOTHESIS_ID",
    "build_predictive_circle_experiment_results",
    "build_predictive_circle_meta_hypothesis",
    "store_predictive_circle_meta_hypothesis",
]
