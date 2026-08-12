#!/usr/bin/env python3
"""Preflight a matched d6/d8/d10 calibrated TinyLLM replication.

This command performs no optimization and inspects no new model outcomes.  It
validates the retained d8 anchors, constructs the prospective d6/d10 grid,
checks that examples and minibatch schedules are preset-independent, and
projects resource use from the measured d8 campaign.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TINYLLM_PRESETS


SCHEMA_VERSION = "nal.tinyllm-calibrated-architecture-replication-preflight.v1"
HYPOTHESIS_ID = "tinyllm-calibrated-architecture-replication-v1"

SOURCE_ROOT = Path(
    "data/experiments/tinyllm_calibrated_frontend_causal/"
    "20260806_d8_preregistered"
)
SOURCE_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "34bf25feb896abc9b9e06386b474fc6795c94566a23cd97a06795435fba64d68"
)

CLOSURE_ROOT = Path(
    "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
    "20260810_d15_preregistered"
)
CLOSURE_CAMPAIGN_SHA256 = (
    "1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14"
)

ANCHOR_PRESET = "d8"
PROSPECTIVE_PRESETS = ("d6", "d10")
ALL_PRESETS = ("d6", "d8", "d10")
CONDITIONS = calibrated.CONDITIONS
PRIMARY_CONDITIONS = (
    "analytic_calibrated",
    "learned_calibrated_equivariant",
)
SEEDS = (7, 17, 29, 41, 53)
PRIMARY_REGIMES = ("composition", "extrapolation")

TRAINING_STEPS = 600
TRAIN_SAMPLES = 4_096
BATCH_SIZE = 64
REPRESENTATION_THRESHOLDS = {
    "cosine_pearson_minimum": 0.90,
    "conditional_branch_accuracy_maximum": 0.55,
    "conditional_log_loss_gain_maximum": 0.02,
}
CAUSAL_THRESHOLDS = {
    "accuracy_loss_maximum": 0.03,
    "circular_error_increase_maximum_radians": float(np.pi / 16.0),
    "cross_entropy_increase_maximum_nats": 0.10,
}
TASK_ADEQUACY_MARGIN = 0.03
REQUIRED_SEED_PASSES = 4
ANALYTIC_FIDELITY_GATES = {
    "composition": {"correlation_minimum": 0.99, "rmse_maximum": 0.05},
    "extrapolation": {"correlation_minimum": 0.99, "rmse_maximum": 0.08},
}
GROUP_CONTRACT_TOLERANCE = 1e-5
GROUP_ACTIONS: tuple[dict[str, Any], ...] = (
    {
        "angle": 0.37,
        "scale": 1.70,
        "offset": (0.23, -0.31),
        "drift": (0.11, 0.07),
    },
    {
        "angle": -1.13,
        "scale": 0.61,
        "offset": (-0.19, 0.29),
        "drift": (-0.08, 0.13),
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def model_parameter_count(task: CircleTaskConfig, preset: str) -> int:
    """Return the exact tied-embedding GPT-2 parameter count analytically."""
    layers, _heads, width = TINYLLM_PRESETS[preset]
    block_size = task.sequence_length + 1
    embeddings = task.vocab_size * width + block_size * width
    transformer_blocks = layers * (12 * width * width + 13 * width)
    final_layer_norm = 2 * width
    return embeddings + transformer_blocks + final_layer_norm


def system_parameter_count(task: CircleTaskConfig, preset: str, condition: str) -> int:
    base = model_parameter_count(task, preset)
    width = TINYLLM_PRESETS[preset][2]
    if condition == "raw_calibrated":
        return base + (calibrated.CALIBRATION_WIDTH + 1) * width
    scalar_embedding = 2 * width
    if condition == "analytic_calibrated":
        return base + scalar_embedding
    if condition == "learned_calibrated_equivariant":
        # The learned encoder has 10,097 parameters at the locked 16 channels
        # and eight sensor steps. This is independent of transformer width.
        vector_channels = 16
        sensor_steps = task.sensor_steps
        hidden = max(32, 2 * vector_channels)
        invariant_width = vector_channels * vector_channels
        encoder = (
            vector_channels * sensor_steps
            + invariant_width * hidden
            + hidden
            + hidden * vector_channels
            + vector_channels
            + 3 * hidden
            + hidden
            + hidden * hidden
            + hidden
            + hidden
            + 1
        )
        return base + scalar_embedding + encoder
    raise ValueError(f"unknown condition {condition!r}")


def experiment_grid() -> list[dict[str, Any]]:
    return [
        {
            "preset": preset,
            "condition": condition,
            "seed": seed,
            "evidence_role": "prospective_primary"
            if condition in PRIMARY_CONDITIONS
            else "prospective_matched_comparator",
        }
        for preset in PROSPECTIVE_PRESETS
        for condition in CONDITIONS
        for seed in SEEDS
    ]


def _result_manifest(root: Path) -> str:
    entries = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = root / "runs" / condition / f"seed_{seed}" / "result.json"
            entries.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "result_sha256": _sha256(path),
                }
            )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_anchors() -> dict[str, Any]:
    source_path = SOURCE_ROOT / "campaign_results.json"
    closure_path = CLOSURE_ROOT / "campaign_results.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    checks = {
        "source_campaign_sha256": _sha256(source_path) == SOURCE_CAMPAIGN_SHA256,
        "source_implementation_sha256": (
            _sha256(Path(calibrated.__file__)) == SOURCE_IMPLEMENTATION_SHA256
        ),
        "source_result_manifest_sha256": (
            _result_manifest(SOURCE_ROOT) == SOURCE_RESULT_MANIFEST_SHA256
        ),
        "source_completed": source.get("status") == "completed",
        "source_grid_complete": source.get("summary", {}).get("completed") == 15,
        "source_preset": source.get("configuration", {}).get("preset") == ANCHOR_PRESET,
        "closure_campaign_sha256": _sha256(closure_path) == CLOSURE_CAMPAIGN_SHA256,
        "closure_completed": closure.get("status") == "completed",
        "closure_valid": closure.get("aggregates", {}).get("valid") is True,
        "closure_primary_pass": (
            closure.get("aggregates", {}).get("primary_hypothesis_pass") is True
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"retained d8 anchor validation failed: {failed}")
    return {
        "checks": checks,
        "source_campaign": str(source_path),
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
        "source_result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
        "closure_campaign": str(closure_path),
        "closure_campaign_sha256": CLOSURE_CAMPAIGN_SHA256,
        "known_evidence_role": "retrospective_anchor_not_fresh_replication",
    }


def protocol_hashes(task: CircleTaskConfig) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        rows = {}
        for preset in ALL_PRESETS:
            config = calibrated.CalibratedFrontendConfig(
                preset=preset,
                seeds=SEEDS,
                training_steps=TRAINING_STEPS,
                train_samples=TRAIN_SAMPLES,
                batch_size=BATCH_SIZE,
                allow_underpowered=preset != ANCHOR_PRESET,
            )
            dataset, pair_batches, data_hash, batch_hash = calibrated._protocol_material(
                task, config, seed
            )
            rows[preset] = {
                "training_data_sha256": data_hash,
                "minibatch_schedule_sha256": batch_hash,
            }
            del dataset, pair_batches
        matched = len(
            {
                (row["training_data_sha256"], row["minibatch_schedule_sha256"])
                for row in rows.values()
            }
        ) == 1
        if not matched:
            raise RuntimeError(f"preset protocol mismatch for seed {seed}")
        by_seed[str(seed)] = {"presets": rows, "matched": True}
    return by_seed


def task_accuracy_anchors() -> dict[str, Any]:
    anchors: dict[str, Any] = {}
    for condition in PRIMARY_CONDITIONS:
        anchors[condition] = {}
        for seed in SEEDS:
            path = SOURCE_ROOT / "runs" / condition / f"seed_{seed}" / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            anchors[condition][str(seed)] = {
                regime: {
                    "d8_accuracy": float(
                        result["analysis"]["task_metrics"][regime]["exact_bin_accuracy"]
                    ),
                    "prospective_floor": float(
                        result["analysis"]["task_metrics"][regime]["exact_bin_accuracy"]
                        - TASK_ADEQUACY_MARGIN
                    ),
                }
                for regime in PRIMARY_REGIMES
            }
    return anchors


def _rotate(values: torch.Tensor, angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y = values.unbind(-1)
    return torch.stack((cosine * x - sine * y, sine * x + cosine * y), -1)


def _apply_acquisition_action(
    sensor: torch.Tensor,
    calibration: torch.Tensor,
    normalized_history: torch.Tensor,
    action: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    angle = float(action["angle"])
    scale = float(action["scale"])
    offset = torch.as_tensor(
        action["offset"], dtype=sensor.dtype, device=sensor.device
    )
    drift = torch.as_tensor(
        action["drift"], dtype=sensor.dtype, device=sensor.device
    )
    transformed_sensor = sensor.clone()
    transformed_sensor[..., :2] = (
        scale * _rotate(sensor[..., :2], angle)
        + offset[None, None]
        + drift[None, None] * normalized_history[None, :, None]
    )
    transformed_calibration = calibration.clone()
    transformed_calibration[:, :2] = _rotate(calibration[:, :2], angle)
    transformed_calibration[:, 3] = scale * calibration[:, 3]
    transformed_calibration[:, 4:6] = (
        scale * _rotate(calibration[:, 4:6], angle) + offset[None]
    )
    transformed_calibration[:, 6:8] = (
        scale * _rotate(calibration[:, 6:8], angle) + drift[None]
    )
    return transformed_sensor, transformed_calibration


def _canonical_feature(
    encoder: calibrated.CalibratedEquivariantEncoder,
    sensor: torch.Tensor,
    calibration: torch.Tensor,
) -> torch.Tensor:
    vector = encoder.equivariant_vector(sensor, calibration)
    orientation = calibration[:, :2]
    dot = (vector * orientation).sum(-1)
    cross = orientation[:, 0] * vector[:, 1] - orientation[:, 1] * vector[:, 0]
    return torch.stack((dot, cross, calibration[:, 2]), -1)


@torch.no_grad()
def sensor_positive_controls(task: CircleTaskConfig) -> dict[str, Any]:
    datasets = {
        "composition": calibrated.generate_calibrated_dataset(
            task, sample_count=1_024, seed=1_399, regime="composition", shuffle=True
        ),
        "extrapolation": calibrated.generate_calibrated_dataset(
            task,
            sample_count=1_024,
            seed=2_408,
            regime="extrapolation",
            shuffle=True,
        ),
    }
    canonicalizer = calibrated.AnalyticCalibratedCanonicalizer(task)
    analytic = {}
    for regime, dataset in datasets.items():
        sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
        predicted = canonicalizer(sensor, dataset.calibration).squeeze(-1).double()
        target = dataset.paired.fiber.cosine.double()
        error = predicted - target
        correlation = float(torch.corrcoef(torch.stack((predicted, target)))[0, 1])
        rmse = float(torch.sqrt(torch.mean(error.square())))
        gate = ANALYTIC_FIDELITY_GATES[regime]
        analytic[regime] = {
            "correlation": correlation,
            "rmse": rmse,
            "gate": gate,
            "pass": correlation >= gate["correlation_minimum"]
            and rmse <= gate["rmse_maximum"],
        }

    torch.manual_seed(6_601)
    encoder = calibrated.CalibratedEquivariantEncoder(task.sensor_steps, 16).eval()
    dataset = datasets["composition"]
    sensor = calibrated.decode_sensor_tokens(
        dataset.paired.circle.input_ids[:128], task
    )
    packet = dataset.calibration[:128]
    baseline = _canonical_feature(encoder, sensor, packet)
    action_errors = []
    for action in GROUP_ACTIONS:
        transformed_sensor, transformed_packet = _apply_acquisition_action(
            sensor, packet, encoder.normalized_history, action
        )
        transformed = _canonical_feature(
            encoder, transformed_sensor, transformed_packet
        )
        action_errors.append(
            {
                "action": dict(action),
                "maximum_absolute_error": float((baseline - transformed).abs().max()),
            }
        )
    maximum_group_error = max(
        item["maximum_absolute_error"] for item in action_errors
    )
    result = {
        "analytic_canonicalizer": analytic,
        "learned_encoder_group_contract": {
            "actions": action_errors,
            "maximum_absolute_error": maximum_group_error,
            "tolerance": GROUP_CONTRACT_TOLERANCE,
            "pass": maximum_group_error <= GROUP_CONTRACT_TOLERANCE,
            "weight_scope": "random initialization; the contract is architectural",
        },
    }
    if not all(item["pass"] for item in analytic.values()):
        raise RuntimeError("analytic canonicalizer positive control failed")
    if not result["learned_encoder_group_contract"]["pass"]:
        raise RuntimeError("learned encoder group contract failed")
    return result


def resource_projection(task: CircleTaskConfig) -> dict[str, Any]:
    result_paths = [
        SOURCE_ROOT / "runs" / condition / f"seed_{seed}" / "result.json"
        for condition in CONDITIONS
        for seed in SEEDS
    ]
    source_results = [
        json.loads(path.read_text(encoding="utf-8")) for path in result_paths
    ]
    d8_training_probe_seconds = float(
        sum(item["wall_seconds"] for item in source_results)
    )
    closure_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in CLOSURE_ROOT.glob("runs/*/seed_*/result.json")
    ]
    if len(closure_results) != len(CONDITIONS) * len(SEEDS):
        raise RuntimeError("retained causal-closure timing population is incomplete")
    d8_causal_seconds = float(
        sum(item["analysis_seconds"] for item in closure_results)
    )
    d8_layers, _d8_heads, d8_width = TINYLLM_PRESETS[ANCHOR_PRESET]
    ratios = {}
    projected_training_probe_seconds = 0.0
    projected_causal_seconds = 0.0
    for preset in PROSPECTIVE_PRESETS:
        layers, _heads, width = TINYLLM_PRESETS[preset]
        ratio = (layers * width * width) / (d8_layers * d8_width * d8_width)
        ratios[preset] = ratio
        projected_training_probe_seconds += d8_training_probe_seconds * ratio
        projected_causal_seconds += d8_causal_seconds * ratio
    projected_seconds = projected_training_probe_seconds + projected_causal_seconds
    source_checkpoint_bytes = float(
        np.mean([path.stat().st_size for path in SOURCE_ROOT.glob("runs/*/seed_7/model.pt")])
    )
    d8_parameters = model_parameter_count(task, ANCHOR_PRESET)
    projected_checkpoint_bytes = sum(
        source_checkpoint_bytes
        * model_parameter_count(task, preset)
        / d8_parameters
        * len(CONDITIONS)
        * len(SEEDS)
        for preset in PROSPECTIVE_PRESETS
    )
    measured_peaks = {
        condition: max(
            float(item["peak_cuda_allocated_gb"])
            for item in source_results
            if item["condition"] == condition
        )
        for condition in CONDITIONS
    }
    return {
        "basis": (
            "measured d8 training-plus-probe and causal-analysis times, each "
            "scaled by layers*width^2"
        ),
        "measured_d8_training_probe_gpu_seconds_for_15_cells": (
            d8_training_probe_seconds
        ),
        "measured_d8_causal_gpu_seconds_for_15_cells": d8_causal_seconds,
        "compute_ratios_to_d8": ratios,
        "projected_new_training_probe_gpu_minutes": (
            projected_training_probe_seconds / 60.0
        ),
        "projected_new_causal_gpu_minutes": projected_causal_seconds / 60.0,
        "projected_new_gpu_seconds": projected_seconds,
        "projected_new_gpu_minutes": projected_seconds / 60.0,
        "projected_two_slot_wall_minutes": projected_seconds / 120.0,
        "measured_d8_peak_cuda_allocated_gb": measured_peaks,
        "required_d10_shakedown_reservation_gb_per_cell": 4.0,
        "projected_new_checkpoint_bytes": int(projected_checkpoint_bytes),
        "projected_new_checkpoint_gib": projected_checkpoint_bytes / 1024**3,
        "boundary": (
            "Runtime and memory are planning estimates; a representative d10 CUDA "
            "shakedown must pass before the primary campaign is scheduled."
        ),
    }


def build_preflight() -> dict[str, Any]:
    task = CircleTaskConfig(train_samples=TRAIN_SAMPLES)
    contract = calibrated.identifiability_contract()
    if contract.get("passed") is not True:
        raise RuntimeError("identifiability contract failed")
    anchors = validate_anchors()
    protocols = protocol_hashes(task)
    positive_controls = sensor_positive_controls(task)
    parameters = {
        preset: {
            "layers": TINYLLM_PRESETS[preset][0],
            "heads": TINYLLM_PRESETS[preset][1],
            "width": TINYLLM_PRESETS[preset][2],
            "model_parameters": model_parameter_count(task, preset),
            "system_parameters": {
                condition: system_parameter_count(task, preset, condition)
                for condition in CONDITIONS
            },
        }
        for preset in ALL_PRESETS
    }
    grid = experiment_grid()
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "preflight_passed",
        "generated_at": _utc_now(),
        "optimization_steps_executed": 0,
        "new_outcomes_inspected": 0,
        "scope": {
            "anchor_preset": ANCHOR_PRESET,
            "prospective_presets": list(PROSPECTIVE_PRESETS),
            "co_variation_boundary": (
                "TinyLLM presets jointly vary depth, width, and head count; this is "
                "a family replication, not a factorized depth-versus-width test."
            ),
        },
        "anchors": anchors,
        "identifiability_contract": contract,
        "sensor_positive_controls": positive_controls,
        "task_config": asdict(task),
        "protocol": {
            "training_steps": TRAINING_STEPS,
            "train_samples": TRAIN_SAMPLES,
            "batch_size": BATCH_SIZE,
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "preset_hashes_by_seed": protocols,
        },
        "grid": {
            "new_cell_count": len(grid),
            "reused_anchor_cell_count": len(CONDITIONS) * len(SEEDS),
            "cells": grid,
        },
        "parameters": parameters,
        "locked_primary_gates": {
            "representation": REPRESENTATION_THRESHOLDS,
            "causal_task_sufficiency": CAUSAL_THRESHOLDS,
            "task_adequacy": {
                "comparison": "same-condition same-seed d8 exact-bin accuracy minus margin",
                "margin": TASK_ADEQUACY_MARGIN,
                "anchors": task_accuracy_anchors(),
            },
            "required_seed_passes_per_preset_and_primary_condition": REQUIRED_SEED_PASSES,
            "required_fresh_presets": list(PROSPECTIVE_PRESETS),
            "raw_role": (
                "matched specificity comparator; it changes interpretation but cannot "
                "rescue a failed analytic or learned structured arm"
            ),
        },
        "resource_projection": resource_projection(task),
        "valid": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calibrated_architecture_replication/"
            "20260810_preflight/preflight.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preflight = build_preflight()
    _write_json(args.output, preflight)
    print(
        json.dumps(
            {
                "status": preflight["status"],
                "valid": preflight["valid"],
                "new_cell_count": preflight["grid"]["new_cell_count"],
                "projected_new_gpu_minutes": preflight["resource_projection"][
                    "projected_new_gpu_minutes"
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
