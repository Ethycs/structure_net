#!/usr/bin/env python3
"""Test a reduced normal jet-kernel radius at the d6 step-15 defect."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import torch

from experiments.structure_net.tinyllm_defect_certification import Cylinder
from experiments.structure_net.tinyllm_degree_defect_cobordism import (
    fixed_nuisance_sensor_values,
    interpolate_state,
    soft_sensor_token_embeddings,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.analyzers import circular_winding_degree


SCHEMA_VERSION = "nal.tinyllm-normal-kernel-radius.v1"
HYPOTHESIS_ID = "tinyllm-normal-jet-kernel-structural-radius-v1"


@dataclass(frozen=True)
class NormalKernelConfig:
    seed: int = 7
    transition_step: int = 15
    phase_search_interval: tuple[float, float] = (4.20, 4.45)
    phase_derivative_step: float = 2e-4
    direct_iterations: int = 12
    random_directions: int = 32
    random_search_maxiter: int = 35
    degree_phase_points: int = 1_024
    residual_gate: float = 1e-5
    device: str = "cuda"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def posterior_moment_tensor(model, task: CircleTaskConfig, phases: np.ndarray, device: torch.device) -> torch.Tensor:
    phase_values = np.atleast_1d(np.asarray(phases, dtype=np.float64))
    sensors = fixed_nuisance_sensor_values(task, phase_values)
    sensor_embeddings = soft_sensor_token_embeddings(model, task, sensors, device)
    sample_count = len(phase_values)
    start = model.transformer["wte"](torch.tensor(1, device=device))
    query = model.transformer["wte"](torch.tensor(2, device=device))
    embedded = torch.empty(
        (sample_count, task.sequence_length, model.config.n_embd),
        device=device, dtype=sensor_embeddings.dtype,
    )
    embedded[:, 0, :] = start
    embedded[:, -1, :] = query
    embedded[:, 1:-1, :] = sensor_embeddings.reshape(sample_count, -1, model.config.n_embd)
    positions = torch.arange(task.sequence_length, device=device)
    value, _ = model._run_blocks(embedded + model.transformer["wpe"](positions))
    value = model.transformer["ln_f"](value)
    answer = torch.tensor(task.answer_token_ids, device=device)
    logits = model.lm_head(value[:, -1, :]).index_select(-1, answer)
    probabilities = torch.softmax(logits, -1)
    angles = 2.0 * math.pi * torch.arange(task.phase_bins, device=device, dtype=probabilities.dtype) / task.phase_bins
    return torch.stack((probabilities @ torch.cos(angles), probabilities @ torch.sin(angles)), -1)


def parameter_jacobian(moment: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    rows = []
    for component in range(2):
        gradient = torch.autograd.grad(
            moment[component], parameter, retain_graph=component == 0, create_graph=False
        )[0]
        rows.append(gradient.reshape(-1))
    return torch.stack(rows)


class BiasEditField:
    def __init__(self, cylinder: Cylinder, task: CircleTaskConfig):
        self.cylinder, self.model, self.task, self.device = cylinder, cylinder.model, task, cylinder.device
        interpolate_state(self.model, cylinder.left, cylinder.right, 0.0, self.device)
        self.bias = self.model.transformer["h"][0].mlp.c_proj.bias
        self.base_bias = self.bias.detach().clone()

    def set_delta(self, delta: np.ndarray | torch.Tensor) -> None:
        value = torch.as_tensor(delta, device=self.device, dtype=self.bias.dtype)
        with torch.no_grad():
            self.bias.copy_(self.base_bias + value)

    def numpy(self, phase: float, delta: np.ndarray | torch.Tensor) -> np.ndarray:
        self.set_delta(delta)
        with torch.no_grad():
            value = posterior_moment_tensor(self.model, self.task, np.asarray([phase]), self.device)[0]
        return value.detach().cpu().double().numpy()

    def value_and_parameter_jacobian(self, phase: float, delta: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.set_delta(delta)
        self.model.zero_grad(set_to_none=True)
        value = posterior_moment_tensor(self.model, self.task, np.asarray([phase]), self.device)[0]
        jacobian = parameter_jacobian(value, self.bias)
        return value.detach().cpu().double().numpy(), jacobian.detach().cpu().double().numpy()

    def phase_jacobian(self, phase: float, delta: np.ndarray | torch.Tensor, step: float) -> np.ndarray:
        return (self.numpy(phase + step, delta) - self.numpy(phase - step, delta)) / (2.0 * step)

    def degree(self, delta: np.ndarray | torch.Tensor, points: int) -> Dict[str, Any]:
        self.set_delta(delta)
        phases = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
        values = []
        with torch.no_grad():
            for start in range(0, points, 256):
                tensor = posterior_moment_tensor(self.model, self.task, phases[start:start + 256], self.device)
                values.append(tensor.cpu().double().numpy())
        array = np.concatenate(values)
        complex_values = array[:, 0] + 1j * array[:, 1]
        angles = np.angle(complex_values) % (2.0 * math.pi)
        closed = np.concatenate((complex_values, complex_values[:1]))
        increments = np.angle(closed[1:] * np.conj(closed[:-1]))
        return {
            "degree": float(circular_winding_degree(angles)),
            "minimum_magnitude": float(np.abs(complex_values).min()),
            "maximum_angular_increment": float(np.abs(increments).max()),
            "sampling_resolved": bool(np.abs(increments).max() < math.pi / 2.0),
        }


def direct_gauss_newton(field: BiasEditField, initial_phase: float, config: NormalKernelConfig) -> Dict[str, Any]:
    delta = np.zeros(field.bias.numel(), dtype=np.float64)
    phase = float(initial_phase)
    trace = []
    for iteration in range(config.direct_iterations):
        value, j_parameter = field.value_and_parameter_jacobian(phase, delta)
        j_phase = field.phase_jacobian(phase, delta, config.phase_derivative_step)
        jacobian = np.column_stack((j_parameter, j_phase))
        correction = -jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T) @ value
        delta += correction[:-1]
        phase += float(correction[-1])
        residual = float(np.linalg.norm(field.numpy(phase, delta)))
        trace.append({"iteration": iteration + 1, "residual_norm": residual, "delta_norm": float(np.linalg.norm(delta)), "phase": phase})
        if residual <= config.residual_gate:
            break
    value = field.numpy(phase, delta)
    return {"delta": delta, "phase": phase, "residual_norm": float(np.linalg.norm(value)), "radius": float(np.linalg.norm(delta)), "trace": trace}


def directional_search(field: BiasEditField, direction: np.ndarray, phase0: float, radius: float, config: NormalKernelConfig) -> Dict[str, Any]:
    unit = direction / max(1e-15, np.linalg.norm(direction))
    def objective(value: np.ndarray) -> float:
        scale, phase = float(value[0]), float(value[1])
        return float(np.linalg.norm(field.numpy(phase, scale * unit)))
    optimized = minimize(
        objective, np.asarray((radius, phase0)), method="Nelder-Mead",
        options={"maxiter": config.random_search_maxiter, "xatol": 1e-7, "fatol": 1e-8},
    )
    return {
        "scale": float(abs(optimized.x[0])), "phase": float(optimized.x[1]),
        "residual_norm": float(optimized.fun), "optimizer_success": bool(optimized.success),
        "evaluations": int(optimized.nfev),
    }


def run_experiment(config: NormalKernelConfig, output: Path) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    task = CircleTaskConfig()
    cylinder_config = __import__(
        "experiments.structure_net.tinyllm_defect_certification", fromlist=["CertificationConfig"]
    ).CertificationConfig(device=config.device)
    cylinder = Cylinder(task, cylinder_config, device)
    field = BiasEditField(cylinder, task)
    closest = minimize_scalar(
        lambda phase: np.linalg.norm(field.numpy(float(phase), np.zeros(field.bias.numel()))),
        bounds=config.phase_search_interval, method="bounded", options={"xatol": 1e-9},
    )
    phase0 = float(closest.x)
    value, j_parameter = field.value_and_parameter_jacobian(phase0, np.zeros(field.bias.numel()))
    j_phase = field.phase_jacobian(phase0, np.zeros(field.bias.numel()), config.phase_derivative_step)
    tangent = j_phase / max(1e-15, np.linalg.norm(j_phase))
    normal = np.asarray((-tangent[1], tangent[0]))
    projected_value = float(normal @ value)
    projected_jacobian = normal @ j_parameter
    kernel = float(projected_jacobian @ projected_jacobian)
    predicted_delta = -projected_value * projected_jacobian / max(kernel, 1e-30)
    predicted_radius = float(np.linalg.norm(predicted_delta))
    direct = direct_gauss_newton(field, phase0, config)
    predicted_search = directional_search(
        field, predicted_delta, phase0, max(predicted_radius, direct["radius"]), config
    )
    generator = np.random.default_rng(config.seed + 77_003)
    controls = []
    for index in range(config.random_directions):
        direction = generator.normal(size=field.bias.numel())
        search = directional_search(field, direction, phase0, direct["radius"], config)
        search["index"] = index; controls.append(search)
    residuals = [predicted_search["residual_norm"]] + [item["residual_norm"] for item in controls]
    predicted_rank = 1 + sum(value < residuals[0] for value in residuals[1:])
    crossing = predicted_search["scale"]
    unit = predicted_delta / max(1e-15, predicted_radius)
    before = field.degree(0.95 * crossing * unit, config.degree_phase_points)
    after = field.degree(1.05 * crossing * unit, config.degree_phase_points)
    ratio = predicted_radius / max(direct["radius"], 1e-30)
    gates = {
        "radius_ratio_in_0_75_1_25": 0.75 <= ratio <= 1.25,
        "direct_residual_at_most_1e_5": direct["residual_norm"] <= 1e-5,
        "predicted_direction_residual_at_most_1e_4": predicted_search["residual_norm"] <= 1e-4,
        "predicted_crossing_within_1_25_direct_radius": predicted_search["scale"] <= 1.25 * direct["radius"],
        "predicted_direction_best_ten_percent": predicted_rank <= max(1, math.ceil(0.10 * (config.random_directions + 1))),
        "resolved_degree_transition": before["sampling_resolved"] and after["sampling_resolved"] and round(before["degree"]) != round(after["degree"]),
    }
    result = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed", "completed_at": _utc_now(), "configuration": asdict(config),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "edit_space": {"parameter": "transformer.h.0.mlp.c_proj.bias", "dimension": field.bias.numel(), "metric": "euclidean"},
        "base": {"phase": phase0, "moment": value.tolist(), "moment_norm": float(np.linalg.norm(value)), "phase_jacobian": j_phase.tolist()},
        "normal_kernel": {"projected_value": projected_value, "projected_jacobian_norm": float(np.linalg.norm(projected_jacobian)), "kernel": kernel, "predicted_radius": predicted_radius, "predicted_delta": predicted_delta.tolist()},
        "direct_optimization": {key: value for key, value in direct.items() if key != "delta"},
        "predicted_direction_search": predicted_search,
        "random_direction_controls": controls,
        "predicted_direction_rank": predicted_rank,
        "radius_ratio_predicted_over_direct": ratio,
        "degree_bracket": {"before": before, "after": after},
        "primary_gates": gates, "confirmed_in_declared_edit_space": all(gates.values()),
        "verdict": "normal_kernel_radius_supported_in_declared_edit_space" if all(gates.values()) else "normal_kernel_full_gate_not_confirmed",
        "method_boundaries": ["One transition, seed, parameter subspace, and Euclidean metric are tested.", "Direct feasibility uses nonlinear Gauss-Newton and is not a global distance proof.", "Random controls are directional searches with a finite evaluation budget."],
    }
    _write_json(output / "results.json", result)
    np.savez_compressed(
        output / "vectors.npz", predicted_delta=predicted_delta,
        direct_delta=direct["delta"], projected_jacobian=projected_jacobian,
        random_control_residuals=np.asarray([item["residual_norm"] for item in controls]),
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--random-directions", type=int, default=32)
    parser.add_argument("--degree-phase-points", type=int, default=1_024); parser.add_argument("--direct-iterations", type=int, default=12)
    parser.add_argument("--random-search-maxiter", type=int, default=35); parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_normal_kernel_radius/20260806_d6_step15"))
    args = parser.parse_args(argv)
    if args.shakedown:
        args.random_directions, args.degree_phase_points = 2, 64
        args.direct_iterations, args.random_search_maxiter = 3, 5
    config = NormalKernelConfig(
        device=args.device, random_directions=args.random_directions,
        degree_phase_points=args.degree_phase_points, direct_iterations=args.direct_iterations,
        random_search_maxiter=args.random_search_maxiter,
    )
    result = run_experiment(config, args.output)
    print(json.dumps({"verdict": result["verdict"], "radius_ratio": result["radius_ratio_predicted_over_direct"], "rank": result["predicted_direction_rank"], "gates": result["primary_gates"]}, indent=2, sort_keys=True))
    print(args.output / "results.json"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
