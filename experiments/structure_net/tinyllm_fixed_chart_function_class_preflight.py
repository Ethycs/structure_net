#!/usr/bin/env python3
"""Preflight a non-regaugeable TinyLLM physical-scalar function class.

This command performs no optimization and loads no TinyLLM checkpoint. It
checks the retained analytic source contract, constructs a monotone
fixed-endpoint counterexample, and verifies a typed split in which learned
auxiliary parameters cannot write the public physical scalar.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-fixed-chart-function-class-preflight.v1"
HYPOTHESIS_ID = "tinyllm-fixed-chart-function-class-v1"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-fixed-chart-function-class-preflight.md"
)
PREREGISTRATION_SHA256 = (
    "d300919a678aebb8bc70cb65913d49696815b1fe883f9e7fbf2a63c8e9fb1470"
)
CALIBRATED_SOURCE_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
INTERVAL_SOURCE_SHA256 = (
    "b027eb55d971b87f598bd600131b33ead82463e3e73766fcdce38fdc3a5497c6"
)
REGIMES = ("composition", "extrapolation")
SAMPLE_COUNT = 4_096
DATASET_SEEDS = {"composition": 109_003, "extrapolation": 109_019}
WARP_PARAMETERS = (-0.4, 0.4)
GRID_POINTS = 65_537
ANALYTIC_GATES = {
    "composition": {"correlation_minimum": 0.99, "rmse_maximum": 0.05},
    "extrapolation": {"correlation_minimum": 0.99, "rmse_maximum": 0.08},
}


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


def monotone_endpoint_warp(value: torch.Tensor, parameter: float) -> torch.Tensor:
    """Return an odd, fixed-endpoint warp with a positive derivative."""
    if not -0.5 < parameter < 0.5:
        raise ValueError("the registered warp requires abs(parameter) < 0.5")
    return value + parameter * value * (1.0 - value.square())


class TypedPhysicalSensor(nn.Module):
    """Expose an immutable analytic scalar and a separately typed carrier."""

    def __init__(self, task: CircleTaskConfig, vector_channels: int = 16):
        super().__init__()
        self.analytic = calibrated.AnalyticCalibratedCanonicalizer(task)
        self.auxiliary = calibrated.CalibratedEquivariantEncoder(
            task.sensor_steps, vector_channels
        )

    def forward(
        self, sensor: torch.Tensor, packet: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "physical_scalar": self.analytic(sensor, packet),
            "auxiliary_carrier": self.auxiliary.equivariant_vector(sensor, packet),
        }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.reshape(-1).double()
    y = right.reshape(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x * y).sum() / denominator.clamp_min(1e-12))


def _analytic_fidelity(
    task: CircleTaskConfig,
) -> tuple[dict[str, Any], dict[str, calibrated.CalibratedDataset]]:
    canonicalizer = calibrated.AnalyticCalibratedCanonicalizer(task)
    records: dict[str, Any] = {}
    datasets: dict[str, calibrated.CalibratedDataset] = {}
    for regime in REGIMES:
        dataset = calibrated.generate_calibrated_dataset(
            task,
            sample_count=SAMPLE_COUNT,
            seed=DATASET_SEEDS[regime],
            regime=regime,
            shuffle=False,
        )
        sensor = calibrated.decode_sensor_tokens(
            dataset.paired.circle.input_ids, task
        )
        predicted = canonicalizer(sensor, dataset.calibration).reshape(-1)
        target = dataset.paired.fiber.cosine.reshape(-1)
        correlation = _pearson(predicted, target)
        rmse = float(torch.sqrt(torch.mean((predicted - target).square())))
        gate = ANALYTIC_GATES[regime]
        passed = (
            correlation >= gate["correlation_minimum"]
            and rmse <= gate["rmse_maximum"]
        )
        records[regime] = {
            "sample_count": len(target),
            "dataset_seed": DATASET_SEEDS[regime],
            "cosine_correlation": correlation,
            "cosine_rmse": rmse,
            "pass": passed,
        }
        datasets[regime] = dataset
    return records, datasets


def _warp_witnesses(task: CircleTaskConfig) -> list[dict[str, Any]]:
    grid = torch.linspace(-1.0, 1.0, GRID_POINTS, dtype=torch.float64)
    identity_bins = joint.interval_posterior_unclipped(
        grid, task.phase_bins
    ).argmax(dim=-1)
    records = []
    spacing = float(grid[1] - grid[0])
    for parameter in WARP_PARAMETERS:
        warped = monotone_endpoint_warp(grid, parameter)
        derivative_lower_bound = min(1.0 + parameter, 1.0 - 2.0 * parameter)
        minimum_grid_slope = float(torch.diff(warped).min() / spacing)
        endpoint_error = float(
            max(abs(float(warped[0]) + 1.0), abs(float(warped[-1]) - 1.0))
        )
        oddness_error = float(torch.max(torch.abs(warped + warped.flip(0))))
        range_violation = float(
            torch.maximum(
                (warped.abs() - 1.0).clamp_min(0.0).max(),
                torch.tensor(0.0, dtype=warped.dtype),
            )
        )
        maximum_displacement = float(torch.max(torch.abs(warped - grid)))
        warped_bins = joint.interval_posterior_unclipped(
            warped, task.phase_bins
        ).argmax(dim=-1)
        bin_disagreement = float((warped_bins != identity_bins).double().mean())
        structural_pass = (
            endpoint_error <= 1e-6
            and oddness_error <= 1e-6
            and range_violation <= 1e-6
            and derivative_lower_bound > 0.0
            and minimum_grid_slope > 0.0
        )
        task_change_pass = maximum_displacement >= 0.05 and bin_disagreement >= 0.01
        records.append(
            {
                "parameter": parameter,
                "derivative_lower_bound": derivative_lower_bound,
                "minimum_grid_slope": minimum_grid_slope,
                "endpoint_error": endpoint_error,
                "oddness_error": oddness_error,
                "range_violation": range_violation,
                "maximum_nonidentity_displacement": maximum_displacement,
                "fixed_interval_bin_disagreement_fraction": bin_disagreement,
                "structural_contract_pass": structural_pass,
                "task_change_witness_pass": task_change_pass,
                "pass": structural_pass and task_change_pass,
            }
        )
    return records


def _randomize_auxiliary(module: TypedPhysicalSensor, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in module.auxiliary.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                * 0.2
            )


def _typed_split_contract(
    task: CircleTaskConfig,
    datasets: Mapping[str, calibrated.CalibratedDataset],
) -> dict[str, Any]:
    left = TypedPhysicalSensor(task)
    right = TypedPhysicalSensor(task)
    _randomize_auxiliary(left, 119_003)
    _randomize_auxiliary(right, 119_021)
    auxiliary_parameter_count = sum(
        parameter.numel() for parameter in left.auxiliary.parameters()
    )
    records = {}
    global_physical_error = 0.0
    global_analytic_error = 0.0
    global_auxiliary_difference = 0.0
    for regime in REGIMES:
        dataset = datasets[regime]
        sensor = calibrated.decode_sensor_tokens(
            dataset.paired.circle.input_ids, task
        )
        packet = dataset.calibration
        with torch.no_grad():
            left_output = left(sensor, packet)
            right_output = right(sensor, packet)
            analytic = left.analytic(sensor, packet)
        physical_error = float(
            torch.max(
                torch.abs(
                    left_output["physical_scalar"]
                    - right_output["physical_scalar"]
                )
            )
        )
        analytic_error = float(
            torch.max(torch.abs(left_output["physical_scalar"] - analytic))
        )
        auxiliary_difference = float(
            torch.sqrt(
                torch.mean(
                    (
                        left_output["auxiliary_carrier"]
                        - right_output["auxiliary_carrier"]
                    ).square()
                )
            )
        )
        global_physical_error = max(global_physical_error, physical_error)
        global_analytic_error = max(global_analytic_error, analytic_error)
        global_auxiliary_difference = max(
            global_auxiliary_difference, auxiliary_difference
        )
        records[regime] = {
            "physical_state_independence_maximum_absolute_error": physical_error,
            "analytic_identity_maximum_absolute_error": analytic_error,
            "auxiliary_state_rms_difference": auxiliary_difference,
        }

    gradient_dataset = datasets["composition"]
    gradient_sensor = calibrated.decode_sensor_tokens(
        gradient_dataset.paired.circle.input_ids[:32], task
    ).requires_grad_(True)
    gradient_packet = gradient_dataset.calibration[:32]
    scalar = left(gradient_sensor, gradient_packet)["physical_scalar"]
    parameters = tuple(left.auxiliary.parameters())
    gradients = torch.autograd.grad(
        scalar.sum(), parameters, allow_unused=True, retain_graph=False
    )
    parameter_gradient_leak_count = sum(item is not None for item in gradients)
    passed = (
        auxiliary_parameter_count > 0
        and global_physical_error <= 1e-7
        and global_analytic_error <= 1e-7
        and global_auxiliary_difference >= 1e-4
        and parameter_gradient_leak_count == 0
    )
    return {
        "auxiliary_parameter_count": auxiliary_parameter_count,
        "physical_scalar_parameter_count": 0,
        "parameter_gradient_leak_count": parameter_gradient_leak_count,
        "maximum_physical_state_dependence": global_physical_error,
        "maximum_analytic_identity_error": global_analytic_error,
        "maximum_auxiliary_state_rms_difference": global_auxiliary_difference,
        "regimes": records,
        "pass": passed,
    }


def build_preflight() -> dict[str, Any]:
    sources = {
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "calibrated_frontend": _sha256(Path(calibrated.__file__)),
        "interval_interface": _sha256(Path(joint.__file__)),
    }
    source_checks = {
        "preregistration": sources["preregistration"] == PREREGISTRATION_SHA256,
        "calibrated_frontend": (
            sources["calibrated_frontend"] == CALIBRATED_SOURCE_SHA256
        ),
        "interval_interface": (
            sources["interval_interface"] == INTERVAL_SOURCE_SHA256
        ),
    }
    if not all(source_checks.values()):
        failed = sorted(name for name, passed in source_checks.items() if not passed)
        raise RuntimeError(f"fixed-chart source contract failed: {failed}")

    task = CircleTaskConfig(train_samples=SAMPLE_COUNT)
    identifiability = calibrated.identifiability_contract()
    analytic, datasets = _analytic_fidelity(task)
    warps = _warp_witnesses(task)
    typed_split = _typed_split_contract(task, datasets)
    source_valid = (
        identifiability["passed"]
        and all(record["pass"] for record in analytic.values())
        and all(source_checks.values())
    )
    witness_pass = all(record["pass"] for record in warps)
    if not source_valid:
        classification = "invalid_source_contract"
        valid = False
    elif not typed_split["pass"]:
        classification = "typed_split_implementation_invalid"
        valid = False
    elif witness_pass:
        classification = "exact_physical_chart_requires_frozen_scalar_spine"
        valid = True
    else:
        classification = "monotone_endpoint_typing_sufficient_in_declared_family"
        valid = True
    del datasets
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if valid else "invalid",
        "completed_at": _utc_now(),
        "classification": classification,
        "valid": valid,
        "implementation_sha256": _sha256(Path(__file__)),
        "source_artifacts": sources,
        "source_checks": source_checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "sample_count_per_regime": SAMPLE_COUNT,
            "dataset_seeds": DATASET_SEEDS,
            "warp_parameters": WARP_PARAMETERS,
            "grid_points": GRID_POINTS,
            "analytic_gates": ANALYTIC_GATES,
        },
        "identifiability": identifiability,
        "analytic_positive_control": analytic,
        "monotone_endpoint_witnesses": warps,
        "typed_split": typed_split,
        "gates": {
            "source_valid": source_valid,
            "monotone_counterexample": witness_pass,
            "typed_split": typed_split["pass"],
            "zero_optimizer_steps": True,
            "zero_checkpoint_loads": True,
        },
        "optimizer_steps_executed": 0,
        "checkpoints_loaded": 0,
        "evidence_role": "no_training_architecture_function_class_preflight",
        "method_boundaries": [
            (
                "This is a function-class counterexample and systems preflight, "
                "not a trained-model result."
            ),
            (
                "The typed split proves scalar isolation, not usefulness of its "
                "learned auxiliary carrier."
            ),
            (
                "The monotone family is one explicit witness; it does not "
                "enumerate all constrained scalar heads."
            ),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/tinyllm-fixed-chart-function-class-preflight.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = build_preflight()
    _write_json(args.output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "classification": record["classification"],
                "valid": record["valid"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if record["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
