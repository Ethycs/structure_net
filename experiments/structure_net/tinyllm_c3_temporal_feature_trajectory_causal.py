#!/usr/bin/env python3
"""Patch fixed C3 carrier trajectories into frozen analytic TinyLLM systems."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import torch

import experiments.structure_net.tinyllm_c3_temporal_continuation_readout as readout
import experiments.structure_net.tinyllm_c3_temporal_fixed_operator_ceiling as fixed
import experiments.structure_net.tinyllm_c3_temporal_quotient_analysis as analysis
import experiments.structure_net.tinyllm_c3_temporal_quotient_campaign as campaign
import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as preflight
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-feature-trajectory-causal.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-feature-trajectory-causal-v1"
EVIDENCE_ROLE = "registered_post_outcome_artifact_only_feature_trajectory_causal"
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
ARMS = ("source", "last_consistent", "mean_consistent", "early_deranged")
PRIMARY_ARMS = ("last_consistent", "mean_consistent")
SOURCE_ROOT = Path(
    "data/experiments/tinyllm_c3_temporal_quotient/"
    "20260811_d6_preregistered"
)
PRIMARY_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_feature_trajectory_causal/"
    "20260811_d6_preregistered/result.json"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-feature-trajectory-causal-"
    "preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "73bf3b792ec44872b8362b69fb51e88826be44073739615a38a28fbdfe2a97ca"
)
SOURCE_CAMPAIGN_SHA256 = (
    "e46710de358a3c9c5d30ddd4a19e36182b94a50cabc699014f79d58d3d3de7cc"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "7dfdcf1ff80fe20a975fe6a7d1311dc92e3ff1a396a6da9550c91835a568a0ff"
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "a0b90484863346cf2a5e0ef8be65cac3a221cfa50a373f88c9ead07a0cd351a1"
)
SOURCE_CAMPAIGN_RUNNER_SHA256 = (
    "9b2cd0e3ce3752b7eea80d5859c11880a9d3732fb48b58306e34eab4f080d5ec"
)
SOURCE_ANALYSIS_SHA256 = (
    "89dacc60d02707678e689c6ce1e8f9c963889af352565a227bb90ed8e367e6a3"
)
SOURCE_TRAINING_SHA256 = (
    "dbc934190b5f725185ff9a99690409ac63fc4f16bd04686f870e7ef21cba03a6"
)
READOUT_LOADER_SHA256 = (
    "3ccb922b8e0fb5119cc8c327024f6b9e1e957bb34b959dcb9e523518ac41746e"
)
FIXED_OPERATOR_RESULT_PATH = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)
FIXED_OPERATOR_RESULT_SHA256 = (
    "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
)
FIXED_OPERATOR_RUNNER_SHA256 = (
    "9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37"
)
FEATURE_TOLERANCE = 2e-6
REPLAY_TOLERANCE = 2e-6
REQUIRED_SEED_PASSES = 4
SHUFFLED_SEED_PASS_MAXIMUM = 1
EARLY_DERANGEMENT_SEEDS = {"composition": 741_000, "extrapolation": 743_000}
TARGET_DERANGEMENT_BASES = {"composition": 751_000, "extrapolation": 753_000}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def validate_sources() -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, str]]:
    source_record, details = readout._source_campaign(SOURCE_ROOT)
    hashes = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "source_campaign": _sha256(SOURCE_ROOT / "campaign_results.json"),
        "source_campaign_runner": _sha256(Path(campaign.__file__)),
        "source_analysis": _sha256(Path(analysis.__file__)),
        "source_training": _sha256(Path(stage0.__file__)),
        "readout_loader": _sha256(Path(readout.__file__)),
        "fixed_operator_runner": _sha256(Path(fixed.__file__)),
        "fixed_operator_result": _sha256(FIXED_OPERATOR_RESULT_PATH),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "source_campaign": SOURCE_CAMPAIGN_SHA256,
        "source_campaign_runner": SOURCE_CAMPAIGN_RUNNER_SHA256,
        "source_analysis": SOURCE_ANALYSIS_SHA256,
        "source_training": SOURCE_TRAINING_SHA256,
        "readout_loader": READOUT_LOADER_SHA256,
        "fixed_operator_runner": FIXED_OPERATOR_RUNNER_SHA256,
        "fixed_operator_result": FIXED_OPERATOR_RESULT_SHA256,
    }
    fixed_result = json.loads(
        FIXED_OPERATOR_RESULT_PATH.read_text(encoding="utf-8")
    )
    if (
        hashes != expected
        or fixed_result.get("status") != "completed"
        or fixed_result.get("aggregates", {}).get("classification")
        != "fixed_multistep_group_operator_dominates_last_step"
    ):
        raise RuntimeError("feature-trajectory source contract changed")
    return source_record, details, hashes


def mean_step(carrier: torch.Tensor) -> torch.Tensor:
    increments = carrier[:, 1:] * carrier[:, :-1].conj()
    total = increments.sum(dim=1)
    return total / total.abs().clamp_min(1e-12)


def project_trajectory(carrier: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    exponents = torch.arange(
        carrier.shape[1], dtype=torch.int64, device=carrier.device
    ) - (carrier.shape[1] - 1)
    return carrier[:, -1, None] * step[:, None].pow(exponents[None])


def feature_arms(
    dataset: stage0.C3TrainingDataset,
    regime: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    carrier, _ = preflight.analytic_carrier(dataset.tokens, dataset.calibration)
    last = carrier[:, -1] * carrier[:, -2].conj()
    averaged = mean_step(carrier)
    last_projection = project_trajectory(carrier, last)
    mean_projection = project_trajectory(carrier, averaged)
    permutation = fixed.sattolo_derangement(
        len(dataset), EARLY_DERANGEMENT_SEEDS[regime]
    )
    early = carrier.clone()
    early[:, :6] = carrier[permutation, :6]
    arms_complex = {
        "source": carrier,
        "last_consistent": last_projection,
        "mean_consistent": mean_projection,
        "early_deranged": early,
    }
    arms = {
        name: torch.view_as_real(value.to(torch.complex64)).float()
        for name, value in arms_complex.items()
    }
    realized = {
        name: torch.view_as_complex(value.contiguous()) for name, value in arms.items()
    }
    fixed_mean = fixed.mean_increment_prediction(carrier)
    projected_mean = preflight.temporal_prediction(realized["mean_consistent"])
    projected_last = preflight.temporal_prediction(realized["last_consistent"])
    source_last = preflight.temporal_prediction(carrier)
    contracts = {
        "last_q6_maximum_error": float(
            (realized["last_consistent"][:, 6] - carrier[:, 6]).abs().max()
        ),
        "last_q7_maximum_error": float(
            (realized["last_consistent"][:, 7] - carrier[:, 7]).abs().max()
        ),
        "mean_q7_maximum_error": float(
            (realized["mean_consistent"][:, 7] - carrier[:, 7]).abs().max()
        ),
        "early_q6_q7_maximum_error": float(
            (realized["early_deranged"][:, 6:] - carrier[:, 6:]).abs().max()
        ),
        "last_scalar_maximum_error": float(
            (projected_last.double() - source_last.double()).abs().max()
        ),
        "mean_scalar_maximum_error": float(
            (projected_mean.double() - fixed_mean.double()).abs().max()
        ),
        "early_derangement_fixed_points": int(
            (permutation == torch.arange(len(dataset))).sum()
        ),
    }
    contracts["pass"] = bool(
        max(
            contracts["last_q6_maximum_error"],
            contracts["last_q7_maximum_error"],
            contracts["mean_q7_maximum_error"],
            contracts["early_q6_q7_maximum_error"],
            contracts["last_scalar_maximum_error"],
            contracts["mean_scalar_maximum_error"],
        )
        <= FEATURE_TOLERANCE
        and contracts["early_derangement_fixed_points"] == 0
    )
    return arms, {**contracts, "fixed_mean_scalar": fixed_mean}


@torch.inference_mode()
def posterior_from_feature(
    system: stage0.C3TemporalTinyLLM,
    feature: torch.Tensor,
    task: stage0.C3TaskConfig,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    system.eval()
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    values = []
    for start in range(0, len(feature), batch_size):
        current = feature[start : start + batch_size].to(device)
        batch = current.shape[0]
        bos_ids = torch.zeros((batch, 1), dtype=torch.long, device=device)
        query_ids = torch.ones((batch, 1), dtype=torch.long, device=device)
        bos = system.model.transformer["wte"](bos_ids)
        sequence = system.sequence_embedding(current)
        query = system.model.transformer["wte"](query_ids)
        hidden = torch.cat((bos, sequence, query), dim=1)
        positions = torch.arange(hidden.shape[1], device=device)
        hidden = hidden + system.model.transformer["wpe"](positions)
        for block in system.model.transformer["h"]:
            hidden = block(hidden)
        query_state = system.model.transformer["ln_f"](hidden[:, -1])
        logits = system.model.lm_head(query_state).index_select(-1, answer_ids)
        values.append(torch.softmax(logits, -1).float().cpu())
    return torch.cat(values)


def posterior_effect(
    source_posterior: torch.Tensor,
    intervention_posterior: torch.Tensor,
) -> dict[str, float]:
    left = source_posterior.double().clamp_min(1e-15)
    right = intervention_posterior.double().clamp_min(1e-15)
    middle = 0.5 * (left + right)
    js = 0.5 * (
        (left * (left.log() - middle.log())).sum(-1)
        + (right * (right.log() - middle.log())).sum(-1)
    )
    centers = torch.linspace(-1.0, 1.0, left.shape[-1], dtype=torch.float64)
    left_mean = left @ centers
    right_mean = right @ centers
    difference = right - left
    return {
        "maximum_absolute_posterior_change": float(difference.abs().max()),
        "mean_absolute_posterior_change": float(difference.abs().mean()),
        "mean_jensen_shannon_divergence": float(js.mean()),
        "posterior_mean_rms_change": float(
            torch.sqrt((right_mean - left_mean).square().mean())
        ),
        "argmax_change_fraction": float(
            (right.argmax(-1) != left.argmax(-1)).double().mean()
        ),
    }


def _shuffled_metrics(
    posterior: torch.Tensor,
    dataset: stage0.C3TrainingDataset,
    permutation: torch.Tensor,
) -> dict[str, Any]:
    logits = posterior.double().clamp_min(1e-15).log()
    return analysis.task_metrics_from_logits(
        logits,
        dataset.target_posteriors[permutation],
        dataset.target_bins[permutation],
        dataset.target[permutation],
    )


def run_seed(
    seed: int,
    source_detail: Mapping[str, Any],
    task: stage0.C3TaskConfig,
    evaluation: Mapping[str, stage0.C3TrainingDataset],
    evaluation_hashes: Mapping[str, str],
    config: readout.C3ReadoutConfig,
    device: torch.device,
) -> dict[str, Any]:
    system = readout._load_system(source_detail, device)
    state_before = stage0._state_digest(system)
    if state_before != source_detail["initialization"]["final_system_sha256"]:
        raise RuntimeError(f"source state changed before trajectory patch seed {seed}")
    regimes = {}
    maximum_source_posterior_error = 0.0
    maximum_source_metric_error = 0.0
    for regime in REGIMES:
        dataset = evaluation[regime]
        features, contracts = feature_arms(dataset, regime)
        direct = readout._extract_state(system, dataset, task, config, device)[
            "source_posterior"
        ]
        posteriors = {
            name: posterior_from_feature(system, feature, task, device, config.batch_size)
            for name, feature in features.items()
        }
        source_error = float((posteriors["source"] - direct).abs().max())
        maximum_source_posterior_error = max(
            maximum_source_posterior_error, source_error
        )
        records = {}
        target_permutation = fixed.sattolo_derangement(
            len(dataset), TARGET_DERANGEMENT_BASES[regime] + seed
        )
        for name in ARMS:
            metrics = readout._posterior_metrics(posteriors[name], dataset)
            shuffled = _shuffled_metrics(
                posteriors[name], dataset, target_permutation
            )
            records[name] = {
                "metrics": metrics,
                "task_gate": readout._task_gate(metrics, regime, config),
                "shuffled_target_metrics": shuffled,
                "shuffled_task_gate": readout._task_gate(
                    shuffled, regime, config
                ),
                "effect_from_source": posterior_effect(
                    posteriors["source"], posteriors[name]
                ),
            }
        metric_error = readout._maximum_stored_metric_error(
            records["source"]["metrics"], source_detail["task_metrics"][regime]
        )
        maximum_source_metric_error = max(maximum_source_metric_error, metric_error)
        fixed_mean_posterior = fixed.joint.interval_posterior_unclipped(
            contracts.pop("fixed_mean_scalar").double(), 16
        )
        fixed_mean_metrics = readout._posterior_metrics(
            fixed_mean_posterior, dataset
        )
        regimes[regime] = {
            "dataset_sha256": evaluation_hashes[regime],
            "feature_contracts": contracts,
            "source_posterior_maximum_error": source_error,
            "source_stored_metric_maximum_error": metric_error,
            "target_derangement_fixed_points": int(
                (target_permutation == torch.arange(len(dataset))).sum()
            ),
            "fixed_mean_bypass": {
                "metrics": fixed_mean_metrics,
                "task_gate": readout._task_gate(
                    fixed_mean_metrics, regime, config
                ),
            },
            "arms": records,
        }
    state_after = stage0._state_digest(system)
    arm_gates = {
        name: {
            "true_both_shifts": all(
                regimes[regime]["arms"][name]["task_gate"] for regime in REGIMES
            ),
            "shuffled_both_shifts": all(
                regimes[regime]["arms"][name]["shuffled_task_gate"]
                for regime in REGIMES
            ),
        }
        for name in ARMS
    }
    fixed_bypass = all(
        regimes[regime]["fixed_mean_bypass"]["task_gate"] for regime in REGIMES
    )
    valid = bool(
        state_before == state_after
        and maximum_source_posterior_error <= REPLAY_TOLERANCE
        and maximum_source_metric_error <= REPLAY_TOLERANCE
        and fixed_bypass
        and all(
            regimes[regime]["feature_contracts"]["pass"] for regime in REGIMES
        )
        and all(
            regimes[regime]["target_derangement_fixed_points"] == 0
            for regime in REGIMES
        )
        and _finite(regimes)
    )
    del system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "source": {
            "result": source_detail["_result_path"],
            "result_sha256": source_detail["_result_sha256"],
            "checkpoint": source_detail["artifacts"]["checkpoint"],
            "checkpoint_sha256": source_detail["artifacts"]["checkpoint_sha256"],
            "frontend_checkpoint": source_detail["artifacts"]["frontend_checkpoint"],
            "frontend_checkpoint_sha256": source_detail["artifacts"][
                "frontend_checkpoint_sha256"
            ],
        },
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_unchanged": state_before == state_after,
        "maximum_source_posterior_error": maximum_source_posterior_error,
        "maximum_source_metric_error": maximum_source_metric_error,
        "regimes": regimes,
        "arm_seed_gates": arm_gates,
        "fixed_mean_bypass_both_shifts": fixed_bypass,
        "valid": valid,
    }


def classify(cells: Sequence[Mapping[str, Any]], valid: bool) -> dict[str, Any]:
    def count(arm: str, field: str = "true_both_shifts") -> int:
        return sum(bool(cell["arm_seed_gates"][arm][field]) for cell in cells)

    source_count = count("source")
    last_count = count("last_consistent")
    mean_count = count("mean_consistent")
    shuffled_mean_count = count("mean_consistent", "shuffled_both_shifts")
    bypass_count = sum(cell["fixed_mean_bypass_both_shifts"] for cell in cells)
    if not valid:
        classification = "invalid_feature_trajectory_causal_contract"
    elif shuffled_mean_count > SHUFFLED_SEED_PASS_MAXIMUM:
        classification = "feature_trajectory_specificity_failed"
    elif mean_count >= REQUIRED_SEED_PASSES and last_count < REQUIRED_SEED_PASSES:
        classification = "all_frame_projection_repairs_frozen_continuation"
    elif mean_count >= REQUIRED_SEED_PASSES and last_count >= REQUIRED_SEED_PASSES:
        classification = (
            "constant_trajectory_projection_repairs_without_all_frame_specificity"
        )
    elif mean_count < REQUIRED_SEED_PASSES and bypass_count == len(SEEDS):
        classification = (
            "fixed_operator_available_but_frozen_continuation_cannot_use_"
            "projected_trajectory"
        )
    else:
        classification = "invalid_feature_trajectory_causal_contract"
    return {
        "classification": classification,
        "valid": valid,
        "source_natural_task_pass_count": source_count,
        "last_consistent_task_pass_count": last_count,
        "mean_consistent_task_pass_count": mean_count,
        "mean_consistent_shuffled_pass_count": shuffled_mean_count,
        "fixed_mean_bypass_pass_count": bypass_count,
        "required_seed_passes": REQUIRED_SEED_PASSES,
        "shuffled_seed_pass_maximum": SHUFFLED_SEED_PASS_MAXIMUM,
    }


def build_result(device_name: str = "cuda:0") -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to replay the source trajectory campaign")
    source_record, source_details, source_hashes = validate_sources()
    task = stage0.C3TaskConfig(**source_record["task"])
    evaluation = analysis.build_task_datasets(task, 1_024)
    evaluation_hashes = {
        regime: analysis.dataset_hash(dataset) for regime, dataset in evaluation.items()
    }
    if evaluation_hashes != source_record["dataset_contract"]["task_hashes"]:
        raise RuntimeError("source trajectory evaluation hashes changed")
    config = readout.C3ReadoutConfig(device=str(device))
    cells = [
        run_seed(
            seed,
            source_details[seed],
            task,
            evaluation,
            evaluation_hashes,
            config,
            device,
        )
        for seed in SEEDS
    ]
    valid = len(cells) == 5 and all(cell["valid"] for cell in cells)
    aggregates = classify(cells, valid)
    source_hashes["runner"] = _sha256(Path(__file__))
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if valid else "invalid",
        "evidence_role": EVIDENCE_ROLE,
        "completed_at": _utc_now(),
        "source": {
            "root": str(SOURCE_ROOT),
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
            "artifact_manifest_sha256": SOURCE_ARTIFACT_MANIFEST_SHA256,
            "evaluation_hashes": evaluation_hashes,
        },
        "source_hashes": source_hashes,
        "configuration": {
            "seeds": list(SEEDS),
            "regimes": list(REGIMES),
            "arms": list(ARMS),
            "evaluation_examples_per_shift": 1_024,
            "batch_size": 256,
            "device": str(device),
            "feature_tolerance": FEATURE_TOLERANCE,
            "replay_tolerance": REPLAY_TOLERANCE,
            "required_seed_passes": REQUIRED_SEED_PASSES,
            "shuffled_seed_pass_maximum": SHUFFLED_SEED_PASS_MAXIMUM,
            "early_derangement_seeds": EARLY_DERANGEMENT_SEEDS,
            "target_derangement_bases": TARGET_DERANGEMENT_BASES,
        },
        "cells": cells,
        "aggregates": aggregates,
        "accounting": {
            "optimizer_steps": 0,
            "parameters_changed": 0,
            "checkpoints_loaded": 5,
            "tinyllm_models_instantiated": 5,
            "target_using_fits": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "method_boundaries": [
            "The trajectory intervention was registered after the fixed-operator result.",
            "Feature patches use no target, phase, speed, fitted coefficient, or model update.",
            "Early-frame derangement is a descriptive causal stress, not a repair arm.",
            "The result localizes five frozen analytic d6 checkpoints only.",
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite feature-trajectory causal result")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=PRIMARY_RESULT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_result(args.device)
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "classification": result["aggregates"]["classification"],
                "source_passes": result["aggregates"][
                    "source_natural_task_pass_count"
                ],
                "last_passes": result["aggregates"][
                    "last_consistent_task_pass_count"
                ],
                "mean_passes": result["aggregates"][
                    "mean_consistent_task_pass_count"
                ],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
