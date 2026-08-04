"""Build and persist meta-hypothesis records from completed NAL artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from neural_architecture_lab.core import ExperimentResult


META_HYPOTHESIS_ID = "meta-extrema-sidecar-transfer-mnist-fashion-v1"
SCHEMA_VERSION = "nal.meta-hypothesis.v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parameter_count(architecture: list[int]) -> int:
    return sum(
        architecture[index] * architecture[index + 1] + architecture[index + 1]
        for index in range(len(architecture) - 1)
    )


def build_meta_hypothesis_record(
    replication_path: Path,
    checkpoint_audit_path: Path,
) -> dict[str, Any]:
    """Create a conservative aggregate without overstating single-seed evidence."""
    replication = _read_json(replication_path)
    audit = _read_json(checkpoint_audit_path)

    runs_by_dataset = {run["dataset"]: run for run in replication["runs"]}
    expected_datasets = {"mnist", "fashion_mnist"}
    missing = expected_datasets.difference(runs_by_dataset)
    if missing:
        raise ValueError(f"replication artifact is missing datasets: {sorted(missing)}")

    direct_evidence = []
    paired_gains = []
    for dataset in ("mnist", "fashion_mnist"):
        run = runs_by_dataset[dataset]
        metrics = run["metrics"]
        gain = float(metrics["patched_minus_control"])
        paired_gains.append(gain)
        direct_evidence.append(
            {
                "evidence_role": "direct_test",
                "experiment_id": run["experiment_id"],
                "dataset": dataset,
                "seed": run["seed"],
                "scaffold_accuracy": metrics["scaffold_accuracy"],
                "control_accuracy": metrics["control_accuracy"],
                "patched_accuracy": metrics["patched_accuracy"],
                "patched_minus_control": gain,
                "patch_count": metrics["patch_count"],
                "effective_parameters": metrics["effective_parameters"],
                "stored_trainable_parameters": metrics["stored_trainable_parameters"],
            }
        )

    provenance_evidence = []
    for checkpoint in audit["checkpoints"]:
        provenance_evidence.append(
            {
                "evidence_role": "provenance_audit",
                "experiment_id": (
                    f"mnist-checkpoint-audit-seed-{checkpoint['seed']}-"
                    f"{checkpoint['sha256'][:8]}"
                ),
                "checkpoint": checkpoint["path"],
                "sha256": checkpoint["sha256"],
                "seed": checkpoint["seed"],
                "stored_accuracy": checkpoint["stored_accuracy"],
                "canonical_full_test_accuracy": checkpoint[
                    "canonical_full_test_accuracy"
                ],
                "stored_minus_canonical": checkpoint["stored_minus_canonical"],
                "claim_reproduced": False,
            }
        )

    completed_at = replication["experiment"]["completed_at"]
    positive_count = sum(gain > 0 for gain in paired_gains)
    mean_gain = sum(paired_gains) / len(paired_gains)

    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis": {
            "id": META_HYPOTHESIS_ID,
            "name": "Extrema-selected sidecar capacity transfers across MNIST tasks",
            "category": "architecture",
            "description": (
                "A meta-level evidence record for the repaired sidecar-patch prototype "
                "on MNIST and Fashion-MNIST, contextualized by an audit of archived "
                "dense checkpoints. It does not test the mature embedded variable-density "
                "StructureNet growth mechanism."
            ),
            "question": (
                "Does extrema-selected sidecar capacity beat a matched continuation-only "
                "control on both MNIST and Fashion-MNIST?"
            ),
            "prediction": "The patched arm has positive paired accuracy gain on both datasets.",
            "created_at": replication["experiment"]["started_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": "provisional_not_confirmed",
            "evidence_count": len(direct_evidence) + len(provenance_evidence),
            "direct_experiment_count": len(direct_evidence),
            "tags": [
                "mnist",
                "fashion-mnist",
                "extrema",
                "sidecar-patch",
                "single-seed",
                "checkpoint-audit",
            ],
            "success_criteria": {
                "positive_paired_gain_on_each_dataset": True,
                "minimum_independent_seeds": 5,
            },
            "tested_scope": "repaired sidecar-patch prototype",
            "explicitly_not_tested": [
                "embedded variable-density patches",
                "iterative structural growth",
                "vertical cloning",
                "dead-neuron highways",
            ],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The observed direction holds on both datasets, but one seed per dataset "
                "cannot establish repeatability or statistical confidence."
            ),
            "confidence": 0.0,
            "confidence_assessment": "not_estimable_single_seed",
            "effect_size": mean_gain,
            "effect_size_kind": "descriptive_mean_paired_accuracy_gain",
            "num_direct_experiments": len(direct_evidence),
            "successful_direct_experiments": len(direct_evidence),
            "failed_direct_experiments": 0,
            "independent_seed_count": len(
                {evidence["seed"] for evidence in direct_evidence}
            ),
            "positive_paired_gain_count": positive_count,
            "checkpoint_audit_count": len(provenance_evidence),
            "best_metrics": max(
                direct_evidence,
                key=lambda evidence: evidence["patched_minus_control"],
            ),
            "key_insights": [
                "The seed-4 sidecar arm beat its paired control by 14.64 percentage points on MNIST.",
                "The seed-4 sidecar arm beat its paired control by 7.41 percentage points on Fashion-MNIST.",
                "The cross-dataset direction is promising but remains qualitative until replicated across seeds and controls.",
            ],
            "unexpected_findings": [
                "The archived 95% checkpoint evaluated at 14.94% on the canonical full MNIST test set.",
                "The archived 100% checkpoint evaluated at 8.17% on the canonical full MNIST test set.",
                "Masked dense tensors reduced effective connections but not stored parameter memory or dense-kernel compute.",
            ],
            "suggested_hypotheses": [
                "The positive paired gain survives at least five independent seeds per dataset.",
                "Extrema-selected patch locations outperform random-location patches at matched capacity.",
                "A train-only extrema probe preserves the paired gain without test-set leakage.",
                "The mature embedded variable-density growth mechanism outperforms this sidecar ablation at matched effective parameters.",
            ],
            "completed_at": completed_at,
        },
        "evidence": {
            "direct_tests": direct_evidence,
            "provenance_audits": provenance_evidence,
        },
        "source_artifacts": [
            str(replication_path),
            str(checkpoint_audit_path),
            "archive/old_research/notes/Breakthroughs.md",
            "archive/old_research/notes/experiment 1.md",
            "archive/old_research/notes/mod 1.md",
            "archive/old_research/notes/tricks.md",
        ],
    }


def build_experiment_results(
    record: dict[str, Any],
    replication_path: Path,
    checkpoint_audit_path: Path,
) -> list[ExperimentResult]:
    """Convert all direct and provenance evidence to the NAL result contract."""
    replication = _read_json(replication_path)
    audit = _read_json(checkpoint_audit_path)
    runs_by_id = {run["experiment_id"]: run for run in replication["runs"]}
    checkpoints_by_sha = {
        checkpoint["sha256"]: checkpoint for checkpoint in audit["checkpoints"]
    }
    hypothesis_id = record["hypothesis"]["id"]
    completed_at = _parse_timestamp(replication["experiment"]["completed_at"])
    audit_timestamp = _parse_timestamp(audit["created_at"])

    results: list[ExperimentResult] = []
    for evidence in record["evidence"]["direct_tests"]:
        run = runs_by_id[evidence["experiment_id"]]
        metrics = {key: float(value) for key, value in run["metrics"].items()}
        metrics["accuracy"] = metrics["patched_accuracy"]
        results.append(
            ExperimentResult(
                experiment_id=run["experiment_id"],
                hypothesis_id=hypothesis_id,
                metrics=metrics,
                primary_metric=metrics["patched_minus_control"],
                model_architecture=run["model_architecture"],
                model_parameters=int(metrics["stored_trainable_parameters"]),
                training_time=float(run["training_time_seconds"]),
                observations=run["observations"]
                + ["Evidence role: direct paired test."],
                anomalies=run["anomalies"],
                timestamp=completed_at,
            )
        )

    for evidence in record["evidence"]["provenance_audits"]:
        checkpoint = checkpoints_by_sha[evidence["sha256"]]
        architecture = checkpoint["architecture"]
        canonical = float(checkpoint["canonical_full_test_accuracy"])
        results.append(
            ExperimentResult(
                experiment_id=evidence["experiment_id"],
                hypothesis_id=hypothesis_id,
                metrics={
                    "accuracy": canonical,
                    "canonical_full_test_accuracy": canonical,
                    "stored_accuracy": float(checkpoint["stored_accuracy"]),
                    "stored_minus_canonical": float(
                        checkpoint["stored_minus_canonical"]
                    ),
                    "claim_reproduced": 0.0,
                },
                primary_metric=canonical,
                model_architecture=architecture,
                model_parameters=_parameter_count(architecture),
                training_time=0.0,
                model_checkpoint=checkpoint["path"],
                observations=[
                    "Evidence role: provenance audit; no training was performed.",
                    "Evaluated on the canonical full 10,000-example MNIST test set.",
                ],
                anomalies=[
                    "Stored accuracy claim did not reproduce under the canonical evaluation contract."
                ],
                timestamp=audit_timestamp,
            )
        )

    return results


def store_meta_hypothesis(
    replication_path: Path,
    checkpoint_audit_path: Path,
    output_path: Path,
    chromadb_path: Path | None = None,
    queue_dir: Path = Path("data/experiment_queue"),
) -> dict[str, Any]:
    """Write the aggregate record and optionally upsert it into the NAL ledger."""
    record = build_meta_hypothesis_record(replication_path, checkpoint_audit_path)
    experiment_results = build_experiment_results(
        record, replication_path, checkpoint_audit_path
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
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
