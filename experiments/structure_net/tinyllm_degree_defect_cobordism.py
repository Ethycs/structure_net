#!/usr/bin/env python3
"""Numerically localize degree-changing posterior-moment defects in TinyLLM.

The hard sensor tokenizer is discontinuous in phase, so the smooth cobordism
identity is tested on an explicit continuous, piecewise-linear lift of sensor
values into adjacent token embeddings. Deterministic training is replayed,
degree-changing optimizer steps are identified, and consecutive weights are
joined by straight-line paths. Indexed zero cells on each phase/path cylinder
are then checked against the endpoint winding-degree change.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleTaskConfig,
        _conditioned_training_data,
        _resolve_device,
        _serialize_sensor_values,
        _state_digest,
        _task_logits,
        _tinyllm_config,
        generate_circle_dataset,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleTaskConfig,
        _conditioned_training_data,
        _resolve_device,
        _serialize_sensor_values,
        _state_digest,
        _task_logits,
        _tinyllm_config,
        generate_circle_dataset,
    )
from structure_net.components.analyzers import (
    circular_phase_alignment,
    circular_winding_degree,
    complex_defect_charge,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-degree-defect-cobordism.v1"
HYPOTHESIS_ID = "tinyllm-degree-change-equals-indexed-defect-charge-v1"


@dataclass(frozen=True)
class DefectCampaignConfig:
    presets: tuple[str, ...] = ("d6", "d8")
    seed: int = 7
    training_steps: int = 600
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    trace_phase_points: int = 128
    trace_refinement_limit: int = 8
    interpolation_phase_points: int = 256
    interpolation_path_points: int = 66
    device: str = "cuda:auto"

    def __post_init__(self) -> None:
        if not self.presets or self.training_steps < 1 or self.batch_size < 1:
            raise ValueError("campaign axes and training sizes must be positive")
        if self.trace_phase_points < 16 or self.interpolation_phase_points < 32:
            raise ValueError("phase grids are too small")
        if self.trace_refinement_limit < 1:
            raise ValueError("trace_refinement_limit must be positive")
        if self.interpolation_path_points < 3:
            raise ValueError("interpolation path requires at least three points")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fixed_nuisance_sensor_values(
    task: CircleTaskConfig,
    future_phases: np.ndarray,
) -> np.ndarray:
    """Smooth sensor slice inside the source generator's training support."""
    future = np.asarray(future_phases, dtype=np.float64)
    current = future - task.future_delta
    history = np.arange(task.sensor_steps, dtype=np.float64) - (
        task.sensor_steps - 1
    )
    angles = current[:, None] + history[None, :] * 0.35
    return np.stack(
        (np.cos(angles), np.sin(angles), 0.25 * np.cos(2.0 * angles)),
        axis=-1,
    )


def soft_sensor_token_embeddings(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    sensor_values: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Continuously lift sensor values by interpolating adjacent token embeddings.

    Bin centers map exactly to their hard token embedding. Between centers the
    lift is linear and continuous; this is a declared extension of, not an
    identity with, the discontinuous hard tokenizer.
    """
    values = np.asarray(sensor_values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (task.sensor_steps, 3):
        raise ValueError("sensor_values must have shape (samples, sensor_steps, 3)")
    scaled = (
        (values + task.quantization_limit)
        / (2.0 * task.quantization_limit)
        * task.value_bins
        - 0.5
    )
    scaled = np.clip(scaled, 0.0, task.value_bins - 1.0)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.ceil(scaled).astype(np.int64)
    mixture = torch.from_numpy((scaled - lower).astype(np.float32)).to(device)
    offsets = task.value_bins * np.arange(3, dtype=np.int64) + 100
    lower_ids = torch.from_numpy(lower + offsets[None, None, :]).to(device)
    upper_ids = torch.from_numpy(upper + offsets[None, None, :]).to(device)
    lower_embeddings = model.transformer["wte"](lower_ids)
    upper_embeddings = model.transformer["wte"](upper_ids)
    return (
        (1.0 - mixture[..., None]) * lower_embeddings
        + mixture[..., None] * upper_embeddings
    )


@torch.no_grad()
def soft_posterior_moment(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    future_phases: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate the circular answer moment on the continuous tokenizer lift."""
    if len(future_phases) > 1_024:
        return np.concatenate(
            [
                soft_posterior_moment(model, task, future_phases[start : start + 1_024], device)
                for start in range(0, len(future_phases), 1_024)
            ]
        )
    model.eval()
    sensor_values = fixed_nuisance_sensor_values(task, future_phases)
    sensor_embeddings = soft_sensor_token_embeddings(
        model, task, sensor_values, device
    )
    sample_count = len(future_phases)
    embedded = torch.empty(
        (sample_count, task.sequence_length, model.config.n_embd),
        device=device,
        dtype=sensor_embeddings.dtype,
    )
    start = model.transformer["wte"](torch.tensor(1, device=device))
    query = model.transformer["wte"](torch.tensor(2, device=device))
    embedded[:, 0, :] = start
    embedded[:, -1, :] = query
    embedded[:, 1:-1, :] = sensor_embeddings.reshape(
        sample_count, -1, model.config.n_embd
    )
    positions = torch.arange(task.sequence_length, device=device)
    value, _ = model._run_blocks(embedded + model.transformer["wpe"](positions))
    value = model.transformer["ln_f"](value)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    logits = model.lm_head(value[:, -1, :]).index_select(-1, answer_ids)
    probabilities = torch.softmax(logits, dim=-1)
    angles = 2.0 * math.pi * torch.arange(
        task.phase_bins, device=device, dtype=probabilities.dtype
    ) / task.phase_bins
    real = probabilities @ torch.cos(angles)
    imaginary = probabilities @ torch.sin(angles)
    return (real.cpu().double().numpy() + 1j * imaginary.cpu().double().numpy())


@torch.no_grad()
def hard_posterior_moment(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    future_phases: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = fixed_nuisance_sensor_values(task, future_phases)
    inputs = torch.from_numpy(_serialize_sensor_values(values, task)).to(device)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    logits, _ = _task_logits(model, inputs, answer_ids)
    probabilities = torch.softmax(logits, dim=-1)
    angles = 2.0 * math.pi * torch.arange(
        task.phase_bins, device=device, dtype=probabilities.dtype
    ) / task.phase_bins
    real = probabilities @ torch.cos(angles)
    imaginary = probabilities @ torch.sin(angles)
    return real.cpu().double().numpy() + 1j * imaginary.cpu().double().numpy()


def moment_record(moment: np.ndarray, target_phases: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(moment, dtype=np.complex128)
    phases = np.angle(values) % (2.0 * math.pi)
    closed = np.concatenate((values, values[:1]))
    increments = np.angle(closed[1:] * np.conj(closed[:-1]))
    alignment = circular_phase_alignment(phases, target_phases)
    degree = circular_winding_degree(phases)
    return {
        "degree": degree,
        "rounded_degree": int(np.rint(degree)),
        "minimum_moment_magnitude": float(np.abs(values).min()),
        "maximum_angular_increment": float(np.abs(increments).max()),
        "phase_alignment": alignment["alignment"],
        "orientation": alignment["orientation"],
        "sampling_resolved": bool(np.abs(increments).max() < math.pi / 2.0),
    }


def resolved_soft_moment_record(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    base_phase_points: int,
    refinement_limit: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Adaptively refine phase sampling until winding increments are resolved."""
    phase_points = base_phase_points
    refinements = 0
    while True:
        phases = np.linspace(0.0, 2.0 * math.pi, phase_points, endpoint=False)
        record = moment_record(
            soft_posterior_moment(model, task, phases, device), phases
        )
        record["phase_grid_points"] = phase_points
        record["refinement_levels"] = refinements
        if record["sampling_resolved"] or phase_points >= (
            base_phase_points * refinement_limit
        ):
            return record
        phase_points *= 2
        refinements += 1


def _training_setup(
    preset: str,
    task: CircleTaskConfig,
    campaign: DefectCampaignConfig,
    device: torch.device,
) -> tuple[TinyLLMModel, torch.optim.Optimizer, torch.Tensor, torch.Tensor, torch.Tensor]:
    seed = campaign.seed
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyLLMModel(_tinyllm_config(preset, task, seed)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
    )
    training = generate_circle_dataset(
        task, sample_count=task.train_samples, seed=seed + 1_001
    )
    inputs, targets = _conditioned_training_data(
        training, task, "trained", seed=seed
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    batches = torch.randint(
        0,
        len(inputs),
        (campaign.training_steps, campaign.batch_size),
        generator=generator,
    )
    return model, optimizer, inputs, targets, batches


def _optimizer_step(
    model: TinyLLMModel,
    optimizer: torch.optim.Optimizer,
    training_inputs: torch.Tensor,
    training_targets: torch.Tensor,
    indices: torch.Tensor,
    task: CircleTaskConfig,
    campaign: DefectCampaignConfig,
    device: torch.device,
) -> float:
    model.train()
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    inputs = training_inputs[indices].to(device, non_blocking=True)
    targets = training_targets[indices].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits, _ = _task_logits(model, inputs, answer_ids)
    loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), campaign.gradient_clip)
    optimizer.step()
    return float(loss.detach())


def training_degree_trace(
    preset: str,
    task: CircleTaskConfig,
    campaign: DefectCampaignConfig,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], str]:
    model, optimizer, inputs, targets, batches = _training_setup(
        preset, task, campaign, device
    )
    base_phases = np.linspace(
        0.0, 2.0 * math.pi, campaign.trace_phase_points, endpoint=False
    )
    trace = []
    for step in range(campaign.training_steps + 1):
        soft = resolved_soft_moment_record(
            model,
            task,
            campaign.trace_phase_points,
            campaign.trace_refinement_limit,
            device,
        )
        hard = moment_record(
            hard_posterior_moment(model, task, base_phases, device), base_phases
        )
        hard["phase_grid_points"] = campaign.trace_phase_points
        trace.append({"step": step, "soft_lift": soft, "hard_tokens": hard})
        if step < campaign.training_steps:
            loss = _optimizer_step(
                model,
                optimizer,
                inputs,
                targets,
                batches[step],
                task,
                campaign,
                device,
            )
            trace[-1]["next_step_training_loss"] = loss
    digest = _state_digest(model)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return trace, digest


def capture_transition_states(
    preset: str,
    transition_steps: Sequence[int],
    task: CircleTaskConfig,
    campaign: DefectCampaignConfig,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    requested = {step - 1 for step in transition_steps} | set(transition_steps)
    model, optimizer, inputs, targets, batches = _training_setup(
        preset, task, campaign, device
    )
    states: Dict[int, Dict[str, torch.Tensor]] = {}
    if 0 in requested:
        states[0] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    for step in range(1, max(requested, default=0) + 1):
        _optimizer_step(
            model,
            optimizer,
            inputs,
            targets,
            batches[step - 1],
            task,
            campaign,
            device,
        )
        if step in requested:
            states[step] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return states


@torch.no_grad()
def interpolate_state(
    model: TinyLLMModel,
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    amount: float,
    device: torch.device,
) -> None:
    current = model.state_dict()
    for name, destination in current.items():
        left = first[name].to(device)
        right = second[name].to(device)
        if torch.is_floating_point(destination):
            destination.copy_(torch.lerp(left, right, amount))
        else:
            destination.copy_(left if amount < 0.5 else right)


def analyze_transition(
    preset: str,
    step: int,
    states: Mapping[int, Mapping[str, torch.Tensor]],
    task: CircleTaskConfig,
    campaign: DefectCampaignConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model = TinyLLMModel(_tinyllm_config(preset, task, campaign.seed)).to(device)
    # Move the two endpoint states once. Re-copying every tensor from host for
    # every path sample dominates the actual topology calculation on CUDA.
    first_state = {
        name: value.to(device) for name, value in states[step - 1].items()
    }
    second_state = {
        name: value.to(device) for name, value in states[step].items()
    }
    phases = np.linspace(
        0.0,
        2.0 * math.pi,
        campaign.interpolation_phase_points,
        endpoint=False,
    )
    path = np.linspace(0.0, 1.0, campaign.interpolation_path_points)
    field = np.empty((len(path), len(phases)), dtype=np.complex128)
    hard_endpoints = []
    for index, amount in enumerate(path):
        interpolate_state(model, first_state, second_state, float(amount), device)
        field[index] = soft_posterior_moment(model, task, phases, device)
        if index in (0, len(path) - 1):
            hard_endpoints.append(
                moment_record(
                    hard_posterior_moment(model, task, phases, device), phases
                )
            )
    charge = complex_defect_charge(field, phases, path)
    charge["optimizer_step"] = step
    charge["hard_token_endpoint_records"] = hard_endpoints
    charge["soft_endpoint_records"] = [
        moment_record(field[0], phases),
        moment_record(field[-1], phases),
    ]
    del model, first_state, second_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return charge


def analyze_preset(
    preset: str,
    task: CircleTaskConfig,
    campaign: DefectCampaignConfig,
    device: torch.device,
    source_run: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    trace, final_digest = training_degree_trace(preset, task, campaign, device)
    transition_steps = [
        current["step"]
        for previous, current in zip(trace, trace[1:])
        if previous["soft_lift"]["rounded_degree"]
        != current["soft_lift"]["rounded_degree"]
    ]
    states = capture_transition_states(
        preset, transition_steps, task, campaign, device
    )
    transitions = [
        analyze_transition(
            preset, step, states, task, campaign, device
        )
        for step in transition_steps
    ]
    endpoint_change = (
        trace[-1]["soft_lift"]["rounded_degree"]
        - trace[0]["soft_lift"]["rounded_degree"]
    )
    total_charge = sum(item["defect_charge"] for item in transitions)
    return {
        "experiment_id": f"tinyllm-degree-defect-{preset}-phase_circle-seed{campaign.seed}",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "preset": preset,
        "seed": campaign.seed,
        "model_parameters": int(source_run["model_parameters"]),
        "source_experiment_id": source_run["experiment_id"],
        "source_checkpoint": source_run.get("model_checkpoint"),
        "training_trace": trace,
        "degree_transition_steps": transition_steps,
        "interpolated_transitions": transitions,
        "endpoint_soft_degree_change": endpoint_change,
        "total_indexed_defect_charge": total_charge,
        "global_charge_identity_holds": total_charge == endpoint_change,
        "all_local_charge_identities_hold": all(
            transition["charge_identity_holds"] for transition in transitions
        ),
        "final_state_sha256": final_digest,
        "expected_final_state_sha256": source_run["final_state_sha256"],
        "final_state_matches_source_campaign": (
            final_digest == source_run["final_state_sha256"]
        ),
        "analysis_seconds": time.perf_counter() - started,
    }


def run_campaign(
    task: CircleTaskConfig,
    campaign: DefectCampaignConfig,
    source_results: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(campaign.device)
    torch.use_deterministic_algorithms(True)
    source = json.loads(source_results.read_text(encoding="utf-8"))
    expected = {
        run["preset"]: run
        for run in source["runs"]
        if run["quotient"] == "phase_circle"
        and run["condition"] == "trained"
        and run["seed"] == campaign.seed
    }
    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "running",
        "started_at": _utc_now(),
        "task_config": asdict(task),
        "campaign_config": asdict(campaign),
        "source_results": str(source_results),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
        "pre_registered_identity": (
            "On the declared continuous quantizer lift and piecewise-linear weight "
            "path, the change in posterior-moment winding degree equals the sum of "
            "indexed phase/path zero-defect charges."
        ),
        "method_boundaries": [
            "the trained hard tokenizer is discontinuous in phase",
            "the continuous adjacent-embedding lift is an explicit extension, not unique",
            "zero cells are grid-indexed rather than interval-certified roots",
            "one fixed nuisance slice and seed 7 are tested",
            "same-degree optimizer intervals may contain cancelling defect pairs",
            "weight interpolation is a chosen continuous path between discrete optimizer states",
        ],
        "runs": [],
    }
    partial = output_dir / "results.partial.json"
    _write_json(partial, bundle)
    for preset in campaign.presets:
        if preset not in expected:
            raise ValueError(f"source campaign has no expected digest for {preset}")
        run = analyze_preset(
            preset, task, campaign, device, expected[preset]
        )
        bundle["runs"].append(run)
        _write_json(partial, bundle)
        print(
            run["experiment_id"],
            json.dumps(
                {
                    "degree_transition_steps": run["degree_transition_steps"],
                    "endpoint_degree_change": run["endpoint_soft_degree_change"],
                    "defect_charge": run["total_indexed_defect_charge"],
                    "hash_match": run["final_state_matches_source_campaign"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    criteria = {
        run["preset"]: {
            "initial_soft_degree_zero": run["training_trace"][0]["soft_lift"][
                "rounded_degree"
            ]
            == 0,
            "final_soft_degree_plus_one": run["training_trace"][-1]["soft_lift"][
                "rounded_degree"
            ]
            == 1,
            "at_least_one_degree_transition": bool(run["degree_transition_steps"]),
            "all_local_charge_identities_hold": run[
                "all_local_charge_identities_hold"
            ],
            "global_charge_identity_holds": run["global_charge_identity_holds"],
            "final_state_matches_source_campaign": run[
                "final_state_matches_source_campaign"
            ],
            "all_trace_samples_resolved": all(
                item["soft_lift"]["sampling_resolved"]
                for item in run["training_trace"]
            ),
        }
        for run in bundle["runs"]
    }
    observed = all(all(values.values()) for values in criteria.values())
    bundle["pre_registered_criteria"] = criteria
    bundle["status"] = "completed"
    bundle["completed_at"] = _utc_now()
    bundle["claim_status"] = {
        "confirmed": False,
        "numerical_charge_identity_supported": observed,
        "interpretation": (
            "Every numerical degree/defect identity and provenance criterion passed."
            if observed
            else "At least one numerical degree/defect or provenance criterion failed."
        ),
        "confirmation_limit": (
            "Finite grids on a chosen continuous tokenizer lift are not an interval-certified polynomial cobordism proof."
        ),
    }
    _write_json(output_dir / "results.json", bundle)
    _write_json(partial, bundle)
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="d6,d8")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--trace-phase-points", type=int, default=128)
    parser.add_argument("--trace-refinement-limit", type=int, default=8)
    parser.add_argument("--interpolation-phase-points", type=int, default=256)
    parser.add_argument("--interpolation-path-points", type=int, default=66)
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_task_quotient_contrast/20260805_d6_d8/results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_degree_defect_cobordism/20260805_d6_d8_seed7"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = DefectCampaignConfig(
        presets=tuple(item.strip() for item in args.presets.split(",") if item.strip()),
        seed=args.seed,
        training_steps=args.steps,
        trace_phase_points=args.trace_phase_points,
        trace_refinement_limit=args.trace_refinement_limit,
        interpolation_phase_points=args.interpolation_phase_points,
        interpolation_path_points=args.interpolation_path_points,
        device=args.device,
    )
    result = run_campaign(
        CircleTaskConfig(), campaign, args.source_results, args.output
    )
    print(json.dumps(result["claim_status"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
