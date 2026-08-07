#!/usr/bin/env python3
"""Attempt an interval-grade certificate for the d6 step-15 degree defect."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import root
import torch

from experiments.structure_net.tinyllm_degree_defect_cobordism import (
    DefectCampaignConfig,
    capture_transition_states,
    interpolate_state,
    moment_record,
    soft_posterior_moment,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _tinyllm_config,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-defect-certification.v1"
HYPOTHESIS_ID = "tinyllm-d6-step15-defect-certificate-v1"


@dataclass(frozen=True)
class CertificationConfig:
    preset: str = "d6"
    seed: int = 7
    transition_step: int = 15
    phase_interval: tuple[float, float] = (4.3197, 4.3258)
    path_interval: tuple[float, float] = (0.2000, 0.2154)
    expanded_phase_margin: float = 0.0062
    expanded_path_margin: float = 0.0154
    root_tolerance: float = 1e-10
    derivative_step: float = 1e-4
    boundary_samples: int = 1_024
    endpoint_phase_points: int = 4_096
    surrogate_degree: int = 8
    surrogate_audit_points: int = 65
    device: str = "cuda"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Cylinder:
    def __init__(self, task: CircleTaskConfig, config: CertificationConfig, device: torch.device):
        campaign = DefectCampaignConfig(
            presets=(config.preset,), seed=config.seed, training_steps=config.transition_step,
            device=str(device), trace_phase_points=32, interpolation_phase_points=64,
            interpolation_path_points=8,
        )
        states = capture_transition_states(
            config.preset, [config.transition_step], task, campaign, device
        )
        self.left = {name: value.to(device) for name, value in states[config.transition_step - 1].items()}
        self.right = {name: value.to(device) for name, value in states[config.transition_step].items()}
        self.model = TinyLLMModel(_tinyllm_config(config.preset, task, config.seed)).to(device)
        self.task, self.device = task, device

    def __call__(self, phase: float, path: float) -> np.ndarray:
        interpolate_state(self.model, self.left, self.right, float(path), self.device)
        value = soft_posterior_moment(
            self.model, self.task, np.asarray([phase], dtype=np.float64), self.device
        )[0]
        return np.asarray([value.real, value.imag], dtype=np.float64)


def numerical_jacobian(field: Cylinder, phase: float, path: float, step: float) -> np.ndarray:
    phase_column = (field(phase + step, path) - field(phase - step, path)) / (2.0 * step)
    path_column = (field(phase, path + step) - field(phase, path - step)) / (2.0 * step)
    return np.column_stack((phase_column, path_column))


def boundary_audit(field: Cylinder, phase_bounds: tuple[float, float], path_bounds: tuple[float, float], samples: int) -> Dict[str, Any]:
    phase = np.linspace(*phase_bounds, samples)
    path = np.linspace(*path_bounds, samples)
    values = []
    for item in phase:
        values.extend((field(float(item), path_bounds[0]), field(float(item), path_bounds[1])))
    for item in path:
        values.extend((field(phase_bounds[0], float(item)), field(phase_bounds[1], float(item))))
    magnitudes = np.linalg.norm(np.asarray(values), axis=1)
    return {
        "kind": "dense_floating_point_boundary_audit",
        "samples_per_edge": samples,
        "minimum_boundary_magnitude": float(magnitudes.min()),
        "sampled_boundary_avoids_zero": bool(magnitudes.min() > 0.0),
        "outward_rounded_actual_network_enclosure": False,
    }


def fit_chebyshev_surrogate(field: Cylinder, phase_bounds: tuple[float, float], path_bounds: tuple[float, float], degree: int, audit_points: int) -> Dict[str, Any]:
    nodes = np.cos(math.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    phase = np.mean(phase_bounds) + 0.5 * np.ptp(phase_bounds) * nodes
    path = np.mean(path_bounds) + 0.5 * np.ptp(path_bounds) * nodes
    xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
    samples = np.empty((degree + 1, degree + 1, 2))
    for i, p in enumerate(phase):
        for j, s in enumerate(path):
            samples[i, j] = field(float(p), float(s))
    vandermonde = np.polynomial.chebyshev.chebvander2d(xx.ravel(), yy.ravel(), [degree, degree])
    coefficients = [
        np.linalg.lstsq(vandermonde, samples[..., component].ravel(), rcond=None)[0].reshape(degree + 1, degree + 1)
        for component in range(2)
    ]
    audit = np.linspace(-1.0, 1.0, audit_points)
    maximum_error = 0.0
    rms = []
    for x in audit:
        p = np.mean(phase_bounds) + 0.5 * np.ptp(phase_bounds) * x
        for y in audit:
            s = np.mean(path_bounds) + 0.5 * np.ptp(path_bounds) * y
            actual = field(float(p), float(s))
            predicted = np.asarray([
                np.polynomial.chebyshev.chebval2d(x, y, item) for item in coefficients
            ])
            error = np.linalg.norm(actual - predicted)
            maximum_error = max(maximum_error, float(error)); rms.append(float(error * error))
    return {
        "degree": degree,
        "coefficients": [item.tolist() for item in coefficients],
        "audit_grid": [audit_points, audit_points],
        "maximum_sampled_value_error": maximum_error,
        "root_mean_squared_sampled_value_error": float(math.sqrt(np.mean(rms))),
        "actual_network_remainder_bound_kind": "sampled_not_rigorous",
        "outward_rounded_actual_network_remainder_enclosed": False,
    }


def run_certification(config: CertificationConfig, output: Path, source_result: Path) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    source = json.loads(source_result.read_text())
    source_run = next(item for item in source["runs"] if item["preset"] == config.preset)
    provenance = {
        "source_result": str(source_result),
        "source_result_sha256": _sha256(source_result),
        "source_final_state_matches_original_campaign": source_run["final_state_matches_source_campaign"],
        "source_final_state_sha256": source_run["final_state_sha256"],
        "degree_transition_steps": source_run["degree_transition_steps"],
    }
    if config.transition_step not in source_run["degree_transition_steps"]:
        raise ValueError("declared transition is absent from the verified source replay")
    cylinder = Cylinder(task, config, device)
    phase_bounds = (
        config.phase_interval[0] - config.expanded_phase_margin,
        config.phase_interval[1] + config.expanded_phase_margin,
    )
    path_bounds = (
        config.path_interval[0] - config.expanded_path_margin,
        config.path_interval[1] + config.expanded_path_margin,
    )
    center = np.asarray((np.mean(config.phase_interval), np.mean(config.path_interval)))
    solution = root(
        lambda value: cylinder(float(value[0]), float(value[1])),
        center,
        jac=lambda value: numerical_jacobian(cylinder, float(value[0]), float(value[1]), config.derivative_step),
        tol=config.root_tolerance,
    )
    root_value = cylinder(float(solution.x[0]), float(solution.x[1]))
    jacobian = numerical_jacobian(cylinder, float(solution.x[0]), float(solution.x[1]), config.derivative_step)
    determinant_phi_path = float(np.linalg.det(jacobian))
    # The cylinder implementation and complex_defect_charge orient cells as
    # (path, phase), so its local defect index uses the swapped-column sign.
    oriented_jacobian = jacobian[:, [1, 0]]
    determinant_path_phase = float(np.linalg.det(oriented_jacobian))
    boundary = boundary_audit(cylinder, phase_bounds, path_bounds, config.boundary_samples)
    surrogate = fit_chebyshev_surrogate(
        cylinder, phase_bounds, path_bounds, config.surrogate_degree, config.surrogate_audit_points
    )
    phases = np.linspace(0.0, 2.0 * math.pi, config.endpoint_phase_points, endpoint=False)
    interpolate_state(cylinder.model, cylinder.left, cylinder.right, 0.0, device)
    first = moment_record(soft_posterior_moment(cylinder.model, task, phases, device), phases)
    interpolate_state(cylinder.model, cylinder.left, cylinder.right, 1.0, device)
    second = moment_record(soft_posterior_moment(cylinder.model, task, phases, device), phases)
    numerical = {
        "root_solver_success": bool(solution.success),
        "root": {"phase": float(solution.x[0]), "path_fraction": float(solution.x[1])},
        "root_residual_norm": float(np.linalg.norm(root_value)),
        "inside_declared_rectangle": bool(phase_bounds[0] <= solution.x[0] <= phase_bounds[1] and path_bounds[0] <= solution.x[1] <= path_bounds[1]),
        "jacobian_phi_path": jacobian.tolist(),
        "jacobian_determinant_phi_path": determinant_phi_path,
        "oriented_jacobian_path_phase": oriented_jacobian.tolist(),
        "oriented_jacobian_determinant_path_phase": determinant_path_phase,
        "orientation_convention": "defect index uses the cylinder (path, phase) orientation, matching complex_defect_charge",
        "positive_numerical_index": determinant_path_phase > 0.0,
        "boundary": boundary,
        "endpoint_records": [first, second],
        "endpoint_degree_change": second["rounded_degree"] - first["rounded_degree"],
    }
    gates = {
        "actual_network_boundary_interval_excludes_zero": boundary["outward_rounded_actual_network_enclosure"],
        "actual_network_krawczyk_unique_root": False,
        "all_other_boxes_interval_root_free": False,
        "actual_network_jacobian_determinant_interval_positive": False,
        "endpoint_winding_change_plus_one": numerical["endpoint_degree_change"] == 1 and first["sampling_resolved"] and second["sampling_resolved"],
        "actual_network_remainder_rigorously_enclosed": surrogate["outward_rounded_actual_network_remainder_enclosed"],
    }
    result = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed", "completed_at": _utc_now(), "configuration": asdict(config),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "provenance": provenance, "declared_rectangle": {"phase": phase_bounds, "path": path_bounds},
        "numerical_localization": numerical, "chebyshev_surrogate": surrogate,
        "certification_gates": gates, "formally_certified": all(gates.values()),
        "verdict": "unique_positive_actual_network_defect_certified" if all(gates.values()) else (
            "formal_certificate_not_obtained_numerical_regular_positive_root_retained"
            if numerical["positive_numerical_index"]
            else "formal_certificate_not_obtained_numerical_index_not_positive"
        ),
        "method_boundary": "Floating-point root finding and sampled surrogate errors do not enclose the actual transformer map; they cannot satisfy the preregistered theorem gate.",
    }
    _write_json(output / "results.json", result)
    np.savez_compressed(
        output / "surrogate.npz",
        real_coefficients=np.asarray(surrogate["coefficients"][0]),
        imaginary_coefficients=np.asarray(surrogate["coefficients"][1]),
        numerical_root=solution.x,
        numerical_jacobian_phi_path=jacobian,
        numerical_jacobian_path_phase=oriented_jacobian,
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--boundary-samples", type=int, default=1_024)
    parser.add_argument("--endpoint-phase-points", type=int, default=4_096)
    parser.add_argument("--surrogate-degree", type=int, default=8)
    parser.add_argument("--surrogate-audit-points", type=int, default=65)
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--source-result", type=Path, default=Path("data/experiments/tinyllm_degree_defect_cobordism/20260805_d6_d8_seed7/results.json"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_defect_certification/20260806_d6_step15"))
    args = parser.parse_args(argv)
    if args.shakedown:
        args.boundary_samples, args.endpoint_phase_points = 16, 64
        args.surrogate_degree, args.surrogate_audit_points = 2, 5
    config = CertificationConfig(
        device=args.device, boundary_samples=args.boundary_samples,
        endpoint_phase_points=args.endpoint_phase_points,
        surrogate_degree=args.surrogate_degree, surrogate_audit_points=args.surrogate_audit_points,
    )
    result = run_certification(config, args.output, args.source_result)
    print(json.dumps({"verdict": result["verdict"], "root": result["numerical_localization"]["root"], "residual": result["numerical_localization"]["root_residual_norm"], "oriented_determinant": result["numerical_localization"]["oriented_jacobian_determinant_path_phase"]}, indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
