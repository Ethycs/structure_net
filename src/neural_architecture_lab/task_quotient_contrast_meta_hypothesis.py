"""Persist the matched TinyLLM task-quotient contrast conservatively."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "tinyllm-task-topology-follows-quotient-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-task-quotient-contrast.v1"


def _read_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        campaign = json.load(source)
    if campaign.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError("unsupported task-quotient result schema")
    if campaign.get("status") != "completed":
        raise ValueError("task-quotient campaign is not complete")
    if campaign.get("hypothesis_id") != META_HYPOTHESIS_ID:
        raise ValueError("task-quotient campaign has the wrong hypothesis id")
    runs = campaign.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("task-quotient campaign contains no runs")
    return campaign


def _mean_metric(
    runs: Sequence[Mapping[str, Any]],
    *,
    quotient: str,
    condition: str,
    metric: str,
) -> float:
    values = [
        float(run["primary_metrics"][metric])
        for run in runs
        if run["quotient"] == quotient and run["condition"] == condition
    ]
    if not values:
        raise ValueError(f"missing cell {quotient}:{condition}")
    return statistics.fmean(values)


def build_task_quotient_contrast_meta_hypothesis(
    results_path: Path,
) -> dict[str, Any]:
    """Build a claim-preserving aggregate from the completed matched campaign."""
    campaign = _read_results(results_path)
    runs = campaign["runs"]
    presets = sorted({run["preset"] for run in runs})
    seeds = sorted({int(run["seed"]) for run in runs})
    quotients = sorted({run["quotient"] for run in runs})
    conditions = sorted({run["condition"] for run in runs})
    expected = len(presets) * len(seeds) * len(quotients) * len(conditions)
    if len(runs) != expected:
        raise ValueError(
            f"campaign is incomplete: expected {expected} matched runs, found {len(runs)}"
        )

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

    circle_accuracy = _mean_metric(
        runs,
        quotient="phase_circle",
        condition="trained",
        metric="in_domain_exact_bin_accuracy",
    )
    interval_accuracy = _mean_metric(
        runs,
        quotient="cosine_interval",
        condition="trained",
        metric="in_domain_exact_bin_accuracy",
    )
    circle_alignment = _mean_metric(
        runs,
        quotient="phase_circle",
        condition="trained",
        metric="map_alignment",
    )
    interval_correlation = _mean_metric(
        runs,
        quotient="cosine_interval",
        condition="trained",
        metric="map_pearson_correlation",
    )
    circle_h1 = _mean_metric(
        runs,
        quotient="phase_circle",
        condition="trained",
        metric="posterior_normalized_h1_lifetime",
    )
    interval_h1 = _mean_metric(
        runs,
        quotient="cosine_interval",
        condition="trained",
        metric="posterior_normalized_h1_lifetime",
    )
    circle_collapse = _mean_metric(
        runs,
        quotient="phase_circle",
        condition="trained",
        metric="posterior_nuisance_collapse_ratio",
    )
    interval_collapse = _mean_metric(
        runs,
        quotient="cosine_interval",
        condition="trained",
        metric="posterior_nuisance_collapse_ratio",
    )
    shuffled_accuracy = statistics.fmean(
        float(run["primary_metrics"]["in_domain_exact_bin_accuracy"])
        for run in runs
        if run["condition"] == "label_shuffled"
    )
    interval_baselines = campaign["target_class_baselines"]["cosine_interval"]

    direct_evidence = [
        {
            "evidence_role": "direct_test",
            "experiment_id": run["experiment_id"],
            "preset": run["preset"],
            "seed": int(run["seed"]),
            "quotient": run["quotient"],
            "condition": run["condition"],
            "model_parameters": int(run["model_parameters"]),
            **{
                key: float(value)
                for key, value in run["primary_metrics"].items()
            },
        }
        for run in runs
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "TinyLLM topology follows the supervised task quotient",
            "category": "training",
            "description": (
                "A matched d6/d8 intervention holding periodic inputs, initialization, "
                "and minibatches fixed while changing the semantic target from phase "
                "on S1 to cos(phase) on an interval."
            ),
            "question": (
                "Does learned task-posterior topology follow the supervised quotient "
                "rather than merely preserve periodic input geometry?"
            ),
            "prediction": (
                "Phase prediction yields an aligned degree-one circle, cosine "
                "prediction yields a correlated interval with negligible H1, both "
                "beat shuffled labels, and both exceed 60% exact-bin accuracy."
            ),
            "created_at": campaign["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "map_aware_task_topology_supported_accuracy_endpoint_failed"
            ),
            "evidence_count": len(direct_evidence),
            "direct_experiment_count": len(direct_evidence),
            "tested_scope": "synthetic periodic sensors with task-defined S1 and interval quotients",
            "success_criteria": criteria,
            "subclaims": {
                "topology_follows_task_quotient": "supported",
                "phase_circle_induced_map": "supported",
                "cosine_interval_induced_map": "supported",
                "trained_tasks_exceed_60_percent_exact_bin_accuracy": "not_supported",
                "posterior_map_collapses_nuisance_fibers_relative_to_shuffled_controls": "supported_in_distribution",
                "residual_stream_erases_quotient_branch_information": "not_tested",
                "invariant_ood_semantic_quotient": "not_established",
            },
            "tags": [
                "tinyllm",
                "semantic-quotient",
                "matched-intervention",
                "persistent-homology",
                "induced-map",
                "fisher-rao",
                "multi-seed",
            ],
            "explicitly_not_tested": campaign["deferred"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "All map-aware and topology-separation criteria passed in d6 and d8, "
                "but mean cosine-interval exact-bin accuracy was 57.23%, below the "
                "pre-registered 60% endpoint."
            ),
            "confidence": 0.0,
            "confidence_assessment": "descriptive_three_seed_matched_campaign",
            "effect_size": circle_h1 - interval_h1,
            "effect_size_kind": "trained_circle_minus_interval_normalized_h1_lifetime",
            "independent_seed_count": len(seeds),
            "model_class_count": len(presets),
            "num_direct_experiments": len(direct_evidence),
            "successful_direct_experiments": len(direct_evidence),
            "failed_direct_experiments": 0,
            "descriptive_metrics": {
                "trained_circle_accuracy_mean": circle_accuracy,
                "trained_interval_accuracy_mean": interval_accuracy,
                "label_shuffled_accuracy_mean": shuffled_accuracy,
                "trained_circle_alignment_mean": circle_alignment,
                "trained_interval_pearson_mean": interval_correlation,
                "trained_circle_normalized_h1_mean": circle_h1,
                "trained_interval_normalized_h1_mean": interval_h1,
                "trained_circle_posterior_nuisance_collapse_mean": circle_collapse,
                "trained_interval_posterior_nuisance_collapse_mean": interval_collapse,
                "interval_uniform_random_accuracy": interval_baselines[
                    "mean_uniform_random_accuracy"
                ],
                "interval_empirical_prior_random_accuracy": interval_baselines[
                    "mean_empirical_prior_random_accuracy"
                ],
                "interval_empirical_majority_class_accuracy": interval_baselines[
                    "mean_empirical_majority_class_accuracy"
                ],
            },
            "key_insights": [
                "All six trained phase runs had alignment above 0.985 and winding degree +1.",
                "All six trained interval runs had Pearson correlation above 0.988 and near-zero posterior H1.",
                "Changing only the task target switched learned posterior topology from a circle to an interval.",
                "Trained posterior-map nuisance-collapse ratios were far below label-shuffled ratios in both tasks.",
            ],
            "unexpected_findings": [
                "The continuous interval map was highly accurate even though its exact-bin endpoint missed by 2.77 percentage points.",
                "Label-shuffled circle runs sometimes had accidental integer winding degree, so degree is not meaningful without alignment and task accuracy.",
                "An empty target H1 diagram makes interval bottleneck distance weak by itself; correlation and nuisance collapse are necessary companion diagnostics.",
            ],
            "suggested_hypotheses": [
                "The task-topology switch survives five or more seeds and a disjoint nuisance family.",
                "A regression-calibrated interval decoder crosses the exact-bin endpoint without changing topology.",
                "Persistent-cohomology-derived coordinates reproduce decoder-derived circle alignment.",
                "Map-aware diagnostics distinguish semantic and nuisance loops on real tokenized sensor data.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"direct_tests": direct_evidence},
        "source_artifacts": [str(results_path)],
    }


def build_task_quotient_contrast_experiment_results(
    record: Mapping[str, Any],
    results_path: Path,
) -> list[ExperimentResult]:
    """Convert all matched arms to the NAL experiment-result contract."""
    campaign = _read_results(results_path)
    completed_at = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    root = results_path.parent
    results: list[ExperimentResult] = []
    for run in campaign["runs"]:
        metrics = {
            key: float(value) for key, value in run["primary_metrics"].items()
        }
        metrics["accuracy"] = metrics["in_domain_exact_bin_accuracy"]
        checkpoint = run.get("model_checkpoint")
        anomalies: list[str] = []
        if (
            run["quotient"] == "cosine_interval"
            and run["condition"] == "trained"
            and metrics["in_domain_exact_bin_accuracy"] < 0.6
        ):
            anomalies.append("The trained interval arm missed the 60% exact-bin endpoint.")
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
                observations=[
                    f"Task quotient: {run['quotient']}.",
                    f"Condition: {run['condition']}.",
                    "Periodic inputs, initialization, and minibatch schedule were matched across quotients.",
                ],
                anomalies=anomalies,
                timestamp=completed_at,
            )
        )
    return results


def store_task_quotient_contrast_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    """Write the aggregate and optionally upsert it into the evidence ledger."""
    record = build_task_quotient_contrast_meta_hypothesis(results_path)
    experiment_results = build_task_quotient_contrast_experiment_results(
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
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


__all__ = [
    "META_HYPOTHESIS_ID",
    "build_task_quotient_contrast_experiment_results",
    "build_task_quotient_contrast_meta_hypothesis",
    "store_task_quotient_contrast_meta_hypothesis",
]
