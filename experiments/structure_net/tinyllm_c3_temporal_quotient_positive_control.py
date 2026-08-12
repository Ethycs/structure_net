#!/usr/bin/env python3
"""Localize the failed C3 analytic population without loading a checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_temporal_quotient_analysis as analysis
import experiments.structure_net.tinyllm_c3_temporal_quotient_campaign as campaign
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as preflight
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-positive-control.v1"
HYPOTHESIS_ID = campaign.HYPOTHESIS_ID
EVIDENCE_ROLE = "registered_result_localization_no_checkpoint_loads"
CAMPAIGN_SHA256 = (
    "e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc"
)
RESULT_MANIFEST_SHA256 = (
    "7dfdcf1ff80fe20a975fe6a7d1311dc92e3ff1a396a6da9550c91835a568a0ff"
)
SEEDS = campaign.SEEDS
REGIMES = campaign.REGIMES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fixed_predictor(
    task: stage0.C3TaskConfig,
    dataset: stage0.C3TrainingDataset,
) -> dict[str, Any]:
    carrier, _ = preflight.analytic_carrier(dataset.tokens, dataset.calibration)
    prediction = preflight.temporal_prediction(carrier)
    posterior, _ = stage0._targets(prediction, task.phase_bins)
    metrics = analysis.task_metrics_from_logits(
        posterior.clamp_min(1e-12).log(),
        dataset.target_posteriors,
        dataset.target_bins,
        dataset.target,
    )
    correlation = float(
        torch.corrcoef(torch.stack((prediction, dataset.target)))[0, 1]
    )
    rmse = float(torch.sqrt((prediction - dataset.target).square().mean()))
    return {
        "temporal_prediction_correlation": correlation,
        "temporal_prediction_rmse": rmse,
        "fixed_interval_metrics": metrics,
    }


def build_diagnostic(root: Path) -> dict[str, Any]:
    campaign_path = root / "campaign_results.json"
    if _sha256(campaign_path) != CAMPAIGN_SHA256:
        raise RuntimeError("C3 primary campaign changed")
    result_paths = [
        root / "runs" / "analytic" / f"seed_{seed}" / "result.json"
        for seed in SEEDS
    ]
    if _manifest(result_paths) != RESULT_MANIFEST_SHA256:
        raise RuntimeError("C3 analytic result manifest changed")
    campaign_record = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign_record.get("status") != "completed"
        or campaign_record.get("aggregates", {}).get("classification")
        != "c3_positive_control_task_failure"
        or campaign_record.get("summary", {}).get("completed") != 5
        or campaign_record.get("summary", {}).get("stopped_by_positive_control")
        != 10
    ):
        raise RuntimeError("C3 campaign is not the registered positive-control stop")
    task = stage0.C3TaskConfig(**campaign_record["task"])
    final_sets = analysis.build_task_datasets(
        task, campaign_record["configuration"]["probe_test_latents"]
    )
    fixed = {
        regime: _fixed_predictor(task, final_sets[regime]) for regime in REGIMES
    }
    config = campaign._config_from_mapping(campaign_record["configuration"])
    fixed_gate = campaign.natural_task_pass(
        {
            regime: fixed[regime]["fixed_interval_metrics"]
            for regime in REGIMES
        },
        config,
    )
    cells = []
    for path in result_paths:
        detail = json.loads(path.read_text(encoding="utf-8"))
        seed = int(detail["seed"])
        failed_endpoints = []
        for regime in REGIMES:
            metrics = detail["task_metrics"][regime]
            if float(metrics["posterior_mean_correlation"]) < config.target_correlation_minimum:
                failed_endpoints.append(f"{regime}:posterior_mean_correlation")
            accuracy_minimum = (
                config.composition_accuracy_minimum
                if regime == "composition"
                else config.extrapolation_accuracy_minimum
            )
            if float(metrics["exact_bin_accuracy"]) < accuracy_minimum:
                failed_endpoints.append(f"{regime}:exact_bin_accuracy")
            cross_entropy_maximum = (
                config.composition_cross_entropy_maximum
                if regime == "composition"
                else config.extrapolation_cross_entropy_maximum
            )
            if float(metrics["target_cross_entropy"]) > cross_entropy_maximum:
                failed_endpoints.append(f"{regime}:target_cross_entropy")
            coverage_minimum = (
                config.composition_coverage_minimum
                if regime == "composition"
                else config.extrapolation_coverage_minimum
            )
            if int(metrics["predicted_bin_coverage"]) < coverage_minimum:
                failed_endpoints.append(f"{regime}:predicted_bin_coverage")
        cells.append(
            {
                "seed": seed,
                "natural_task_pass": detail["gates"]["natural_task"],
                "representation_pass": detail["gates"]["representation"],
                "causal_all_cuts_pass": detail["gates"]["causal_all_cuts"],
                "exact_action_pass": detail["gates"]["exact_action"],
                "identity_replay_pass": detail["gates"]["identity_replay"],
                "target_derangement_pass_by_regime": detail["gates"][
                    "target_derangement_pass_by_regime"
                ],
                "failed_natural_task_endpoints": failed_endpoints,
                "task_metrics": detail["task_metrics"],
                "full_depth_semantic_probe": {
                    regime: detail["representation"]["cuts"]["full"]["semantic"]
                    ["evaluations"][regime]
                    for regime in REGIMES
                },
                "result_sha256": _sha256(path),
            }
        )
    representation_count = sum(cell["representation_pass"] for cell in cells)
    causal_count = sum(cell["causal_all_cuts_pass"] for cell in cells)
    task_count = sum(cell["natural_task_pass"] for cell in cells)
    if fixed_gate and representation_count == 5 and causal_count == 5 and task_count < 4:
        classification = "invariant_sensor_valid_trained_continuation_readout_extrapolation_unreliable"
    else:
        classification = "positive_control_failure_not_localized"
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "status": "completed",
        "completed_at": _utc_now(),
        "classification": classification,
        "campaign": {
            "path": str(campaign_path),
            "sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
        },
        "fixed_no_model_positive_control": fixed,
        "fixed_no_model_natural_task_gate": fixed_gate,
        "checkpoint_cells": cells,
        "counts": {
            "natural_task_pass": task_count,
            "representation_pass": representation_count,
            "causal_all_cuts_pass": causal_count,
            "checkpoints_loaded": 0,
            "optimizer_steps": 0,
            "trained_parameters": 0,
        },
        "gates": {
            "campaign_positive_control_stop": True,
            "fixed_no_model_task": fixed_gate,
            "all_checkpoint_representations": representation_count == 5,
            "all_checkpoint_causal_closure": causal_count == 5,
            "natural_output_population_failure": task_count < 4,
            "no_checkpoint_loads": True,
            "zero_optimizer_steps": True,
        },
        "method_boundaries": [
            "This is a registered-result and fixed analytic diagnostic selected after the population outcome.",
            "The fixed predictor establishes generator/front-end sufficiency, not trained TinyLLM utility.",
            "The registered nonlinear semantic probe localizes decodability but is not a causal readout repair.",
            "No checkpoint is loaded and no parameter is fitted or optimized in this diagnostic.",
        ],
    }
    if not stage0._finite(record):
        raise RuntimeError("non-finite C3 positive-control diagnostic")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_c3_temporal_quotient/"
            "20260811_d6_preregistered"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.root / "positive_control_diagnostic.json"
    record = build_diagnostic(args.root)
    _write_json(output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "classification": record["classification"],
                "counts": record["counts"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
