#!/usr/bin/env python3
"""Audit local residual transport along a frozen orientation-reference path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibration_orientation_noise as orientation
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
import experiments.structure_net.tinyllm_reference_acquisition_replicates as acquisition
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-reference-path-residual-transport.v1.4"
HYPOTHESIS_ID = "tinyllm-reference-path-residual-transport-v1"
SOURCE_SCHEMA = acquisition.SCHEMA_VERSION
SOURCE_HYPOTHESIS = acquisition.HYPOTHESIS_ID
SOURCE_CAMPAIGN_SHA256 = (
    "269fd948f0d6fee8916bbe3cb94c1d87f76572e43c103b52fc8775fa9653031e"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "6c3cc4463b2c515280c778461e4606dca6ffb19f08d11cb5b7a852a942c7df77"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "8181d48f99850d5e487b5209d648373410bea3b46b2eafcb58f57118ed898c1c"
)
SOURCE_REPEAT_ARRAY_SHA256 = (
    "8886c1f6ad0fd307720748e44b1741edb656fdb9e29e913c7ad059b72397eef4"
)
EVIDENCE_ROLE = "corrective_outcome_exposed_frozen_residual_transport_audit"
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
CONDITIONS = acquisition.CONDITIONS
REGIMES = acquisition.PRIMARY_REGIMES
STEP_COUNTS = (1, 2, 4, 8, 16)
FINE_STEPS = 16
SCHEDULES = ("true_cosine", "path_moment", "shuffled_path_moment")
SHUFFLE_SEED_OFFSET = 15_700_019


@dataclass(frozen=True)
class ReferencePathTransportConfig:
    source_root: str = (
        "data/experiments/tinyllm_reference_acquisition_replicates/"
        "20260807_d8_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    step_counts: tuple[int, ...] = STEP_COUNTS
    fine_steps: int = FINE_STEPS
    activation_batch_size: int = 256
    accuracy_loss_ceiling: float = 0.03
    replay_tolerance: float = 2e-6
    path_sharing_tolerance: float = 1e-12
    gradient_squared_floor: float = 1e-8
    gradient_norm_floor: float = 1e-8
    required_seed_passes: int = 4
    control_pass_ceiling: int = 1
    analysis_seed: int = 31_337
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if (
            not self.step_counts
            or self.step_counts[0] != 1
            or tuple(sorted(set(self.step_counts))) != self.step_counts
        ):
            raise ValueError("step counts must be ordered, distinct, and start at one")
        if self.fine_steps < 1 or any(
            count < 1 or self.fine_steps % count for count in self.step_counts
        ):
            raise ValueError("every step count must divide the fine path grid")
        if self.step_counts[-1] != self.fine_steps:
            raise ValueError("the maximum step count must equal the fine path grid")
        if self.activation_batch_size < 1:
            raise ValueError("activation batch size must be positive")
        if self.required_seed_passes > len(self.seeds):
            raise ValueError("required seed passes exceed selected checkpoints")
        for name in (
            "accuracy_loss_ceiling",
            "replay_tolerance",
            "path_sharing_tolerance",
            "gradient_squared_floor",
            "gradient_norm_floor",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.allow_underpowered:
            if self.seeds != PRIMARY_SEEDS:
                raise ValueError("the primary checkpoint seeds are fixed")
            if self.step_counts != STEP_COUNTS or self.fine_steps != FINE_STEPS:
                raise ValueError("the primary transport grid is fixed")
            if self.required_seed_passes != 4 or self.control_pass_ceiling != 1:
                raise ValueError("the primary population and control gates are fixed")
            if self.analysis_seed != 31_337:
                raise ValueError("the primary analysis seed is fixed")
            if self.path_sharing_tolerance != 1e-12:
                raise ValueError("the primary path-sharing tolerance is fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256(paths: Iterable[Path]) -> str:
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


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite_numbers(item) for item in value)
    return True


def _runner_protocol_digest() -> str:
    """Hash scientific and lifecycle semantics while excluding CLI path defaults."""

    protocol_names = (
        "ReferencePathTransportConfig",
        "_manifest_sha256",
        "_finite_numbers",
        "_evidence_role",
        "shortest_reference_path",
        "minimum_norm_task_delta",
        "nested_step_indices",
        "fiber_block_permutation",
        "residual_curve_geometry",
        "jensen_shannon_divergence",
        "task_gate",
        "classify_campaign",
        "_task_config_from_source",
        "_load_source_campaign",
        "_dataset_hash",
        "_posterior",
        "_forward_packet",
        "_extract_actual_path",
        "_task_metrics",
        "_causal_metrics",
        "_metric_error",
        "_rollout",
        "_local_path_diagnostics",
        "_fingerprint",
        "_reusable_result",
        "_campaign_reusable",
        "_shuffle_seed",
        "run_campaign",
    )
    constants = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "source_schema": SOURCE_SCHEMA,
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
        "source_result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
        "source_repeat_array_sha256": SOURCE_REPEAT_ARRAY_SHA256,
        "evidence_role": EVIDENCE_ROLE,
        "primary_seeds": PRIMARY_SEEDS,
        "conditions": CONDITIONS,
        "regimes": REGIMES,
        "step_counts": STEP_COUNTS,
        "fine_steps": FINE_STEPS,
        "schedules": SCHEDULES,
        "shuffle_seed_offset": SHUFFLE_SEED_OFFSET,
    }
    material = {
        "protocol_names": protocol_names,
        "sources": {
            name: inspect.getsource(globals()[name]) for name in protocol_names
        },
        "constants": constants,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_digests() -> dict[str, str]:
    paths = {
        "acquisition": Path(acquisition.__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "orientation_noise": Path(orientation.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "tinyllm_model": Path(inspect.getfile(TinyLLMModel)),
    }
    return {
        "runner_protocol": _runner_protocol_digest(),
        **{name: _sha256(path) for name, path in paths.items()},
    }


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    material = dict(digests or _source_digests())
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_role(config: ReferencePathTransportConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else EVIDENCE_ROLE
    )


def shortest_reference_path(
    start: torch.Tensor, end: torch.Tensor, fine_steps: int
) -> torch.Tensor:
    """Return the shortest unit-circle path with shape [steps+1, observations]."""

    if start.ndim != 1 or start.shape != end.shape:
        raise ValueError("reference endpoints must be equal one-dimensional tensors")
    if fine_steps < 1:
        raise ValueError("fine_steps must be positive")
    if bool((start.abs() < 1e-12).any() or (end.abs() < 1e-12).any()):
        raise ValueError("reference endpoints must be nonzero")
    exact_start = start.clone()
    exact_end = end.clone()
    start = exact_start / exact_start.abs()
    end = exact_end / exact_end.abs()
    delta = torch.angle(end * start.conj())
    fraction = torch.linspace(0.0, 1.0, fine_steps + 1, dtype=delta.dtype)
    phase = fraction[:, None] * delta[None, :]
    path = start[None, :] * torch.polar(torch.ones_like(phase), phase)
    path[0] = exact_start
    path[-1] = exact_end
    return path


def minimum_norm_task_delta(
    current: torch.Tensor,
    target: torch.Tensor,
    gradient: torch.Tensor,
    squared_floor: float,
) -> torch.Tensor:
    """Construct the minimum-norm first-order update for one scalar chart."""

    if current.ndim != 1 or target.shape != current.shape:
        raise ValueError("current and target coordinates must be equal vectors")
    if gradient.ndim != 2 or len(gradient) != len(current):
        raise ValueError("gradient must have shape [observations, residual_width]")
    squared = gradient.square().sum(1).clamp_min(float(squared_floor))
    return ((target - current) / squared)[:, None] * gradient


def nested_step_indices(fine_steps: int, step_count: int) -> tuple[int, ...]:
    if fine_steps < 1 or step_count < 1 or fine_steps % step_count:
        raise ValueError("step_count must divide fine_steps")
    stride = fine_steps // step_count
    return tuple(range(0, fine_steps + 1, stride))


def fiber_block_permutation(
    fiber_ids: torch.Tensor, branch: torch.Tensor, *, shift: int
) -> torch.Tensor:
    """Map every exact fiber to another fiber while retaining sheet identity."""

    fiber_ids = fiber_ids.long().cpu()
    branch = branch.cpu()
    unique = torch.unique(fiber_ids, sorted=True)
    if len(unique) < 2:
        raise ValueError("fiber permutation requires at least two fibers")
    offset = 1 + (int(shift) % (len(unique) - 1))
    donor_fibers = torch.roll(unique, shifts=offset)
    output = torch.empty(len(fiber_ids), dtype=torch.long)
    for target_fiber, donor_fiber in zip(unique, donor_fibers):
        target_rows = torch.nonzero(fiber_ids == target_fiber, as_tuple=False)[:, 0]
        donor_rows = torch.nonzero(fiber_ids == donor_fiber, as_tuple=False)[:, 0]
        if len(target_rows) != len(donor_rows):
            raise ValueError("every permuted fiber must have the same sheet count")
        for row in target_rows:
            matches = donor_rows[branch[donor_rows] == branch[row]]
            if len(matches) != 1:
                raise ValueError("sheet labels do not identify one donor row")
            output[row] = matches[0]
    if torch.equal(output, torch.arange(len(output))):
        raise RuntimeError("fiber permutation unexpectedly retained every row")
    return output


def residual_curve_geometry(residuals: torch.Tensor) -> dict[str, torch.Tensor]:
    """Measure per-example arc/chord ratio and maximum normalized deviation."""

    if residuals.ndim != 3 or len(residuals) < 2:
        raise ValueError("residual curve must have shape [steps,observations,width]")
    value = residuals.double()
    increments = value[1:] - value[:-1]
    chord = value[-1] - value[0]
    chord_norm = torch.linalg.vector_norm(chord, dim=1).clamp_min(1e-12)
    arc = torch.linalg.vector_norm(increments, dim=2).sum(0)
    fraction = torch.linspace(0.0, 1.0, len(value), dtype=value.dtype)
    linear = value[0][None, :, :] + fraction[:, None, None] * chord[None, :, :]
    deviation = torch.linalg.vector_norm(value - linear, dim=2).max(0).values
    return {
        "arc_chord_ratio": arc / chord_norm,
        "maximum_relative_chord_deviation": deviation / chord_norm,
        "chord_norm": chord_norm,
        "arc_length": arc,
    }


def jensen_shannon_divergence(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("posterior tensors must have equal [observations,bins] shape")
    left = left.double().clamp_min(1e-12)
    right = right.double().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.log() - middle.log())).sum(1)
        + (right * (right.log() - middle.log())).sum(1)
    )


def task_gate(
    clean_accuracy: Mapping[str, float],
    regime_records: Mapping[str, Mapping[str, Any]],
    ceiling: float,
) -> tuple[bool, dict[str, float]]:
    losses = {
        regime: float(clean_accuracy[regime])
        - float(regime_records[regime]["exact_bin_accuracy"])
        for regime in REGIMES
    }
    return all(value <= ceiling + 1e-12 for value in losses.values()), losses


def classify_campaign(
    *,
    valid: bool,
    true_cosine_pass: Mapping[str, int],
    path_moment_pass: Mapping[str, int],
    required: int,
) -> str:
    if not valid:
        return "invalid"
    true_pass = all(int(true_cosine_pass[name]) >= required for name in CONDITIONS)
    path_pass = all(int(path_moment_pass[name]) >= required for name in CONDITIONS)
    if true_pass and path_pass:
        return "local_relinearization_sufficient"
    if path_pass:
        return "terminal_coordinate_mismatch"
    if true_pass:
        return "semantic_schedule_only"
    return "scalar_transport_insufficient"


def _task_config_from_source(source: Mapping[str, Any]) -> acquisition.ReferenceAcquisitionConfig:
    value = dict(source["configuration"])
    value["seeds"] = tuple(int(seed) for seed in value["seeds"])
    value["repeat_counts"] = tuple(int(count) for count in value["repeat_counts"])
    return acquisition.ReferenceAcquisitionConfig(**value)


def _load_source_campaign(
    config: ReferencePathTransportConfig,
) -> tuple[dict[str, Any], Path, dict[tuple[str, int], dict[str, Any]]]:
    path = Path(config.source_root) / "campaign_results.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    artifacts = source.get("artifacts", {})
    repeat_path = Path(artifacts.get("repeat_arrays", ""))
    entries = {
        (entry["condition"], int(entry["seed"])): entry
        for entry in source.get("results", [])
    }
    expected = {
        (condition, seed) for condition in CONDITIONS for seed in PRIMARY_SEEDS
    }
    result_paths = [Path(entry["path"]) for entry in entries.values()]
    if (
        _sha256(path) != SOURCE_CAMPAIGN_SHA256
        or source.get("schema_version") != SOURCE_SCHEMA
        or source.get("hypothesis_id") != SOURCE_HYPOTHESIS
        or source.get("status") != "completed"
        or source.get("implementation_sha256") != SOURCE_IMPLEMENTATION_SHA256
        or int(source.get("summary", {}).get("completed", -1)) != 10
        or int(source.get("summary", {}).get("failed", -1)) != 0
        or set(entries) != expected
        or not repeat_path.is_file()
        or _sha256(repeat_path) != SOURCE_REPEAT_ARRAY_SHA256
        or artifacts.get("repeat_arrays_sha256") != SOURCE_REPEAT_ARRAY_SHA256
        or any(
            not result_path.is_file()
            or _sha256(result_path) != entries[key].get("result_sha256")
            for key, result_path in zip(entries, result_paths)
        )
        or _manifest_sha256(result_paths) != SOURCE_RESULT_MANIFEST_SHA256
        or acquisition._source_digests() != source.get("source_digests")
        or acquisition._implementation_digest(source["source_digests"])
        != SOURCE_IMPLEMENTATION_SHA256
    ):
        raise ValueError(f"invalid repeated-reference source campaign {path}")
    for key, entry in entries.items():
        detail = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        provenance = detail.get("provenance", {})
        for path_key, hash_key in (
            ("model_checkpoint", "model_checkpoint_sha256"),
            ("frontend_checkpoint", "frontend_checkpoint_sha256"),
            ("source_result", "source_result_sha256"),
            ("orientation_result", "orientation_result_sha256"),
        ):
            artifact = Path(provenance.get(path_key, ""))
            if not artifact.is_file() or _sha256(artifact) != provenance.get(hash_key):
                raise ValueError(f"invalid source artifact for {key}: {artifact}")
    return source, path, entries


def _dataset_hash(dataset: calibrated.CalibratedDataset) -> str:
    return orientation._dataset_hash(dataset)


def _posterior(
    model: TinyLLMModel, residual: torch.Tensor, answer_ids: torch.Tensor
) -> torch.Tensor:
    return torch.softmax(invariant._task_logits(model, residual, answer_ids), -1)


def _forward_packet(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    calibration: torch.Tensor,
    task: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    residuals: list[torch.Tensor] = []
    posteriors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(sensor), batch_size):
            stop = min(start + batch_size, len(sensor))
            residual = system.forward_cuts(
                dataset.paired.circle.input_ids[start:stop].to(device),
                sensor[start:stop].to(device),
                calibration[start:stop].to(device),
            )["full"]
            residuals.append(residual.cpu())
            posteriors.append(_posterior(system.model, residual, answer_ids).cpu())
    return torch.cat(residuals), torch.cat(posteriors)


def _extract_actual_path(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    references: torch.Tensor,
    task: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals: list[torch.Tensor] = []
    posteriors: list[torch.Tensor] = []
    for reference in references:
        packet = acquisition.packet_from_orientation(dataset.calibration, reference)
        residual, posterior = _forward_packet(
            system, dataset, packet, task, batch_size, device
        )
        residuals.append(residual)
        posteriors.append(posterior)
    return torch.stack(residuals), torch.stack(posteriors)


def _task_metrics(
    posterior: torch.Tensor,
    dataset: calibrated.CalibratedDataset,
) -> dict[str, float]:
    posterior = posterior.float()
    target = dataset.paired.circle.target_posteriors.float()
    predicted_bin = posterior.argmax(1)
    target_bin = target.argmax(1)
    delta = (predicted_bin - target_bin).abs()
    circular = torch.minimum(delta, target.shape[1] - delta)
    return {
        "exact_bin_accuracy": float((predicted_bin == target_bin).float().mean()),
        "mean_circular_error_radians": float(
            circular.float().mean() * (2.0 * math.pi / target.shape[1])
        ),
        "mean_target_cross_entropy": float(
            -(target * posterior.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def _causal_metrics(
    posterior: torch.Tensor,
    dataset: calibrated.CalibratedDataset,
) -> dict[str, float]:
    posterior = posterior.double()
    target = dataset.paired.circle.target_posteriors.double()
    return {
        "exact_bin_accuracy": float((posterior.argmax(1) == target.argmax(1)).double().mean()),
        "mean_target_cross_entropy": float(
            -(target * posterior.clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def _metric_error(
    observed: Mapping[str, Any], expected: Mapping[str, Any], names: Sequence[str]
) -> float:
    return max(abs(float(observed[name]) - float(expected[name])) for name in names)


def _rollout(
    *,
    model: TinyLLMModel,
    answer_ids: torch.Tensor,
    centers: torch.Tensor,
    start_residual: torch.Tensor,
    endpoint_residual: torch.Tensor,
    endpoint_posterior: torch.Tensor,
    actual_moments: torch.Tensor,
    target_cosine: torch.Tensor,
    dataset: calibrated.CalibratedDataset,
    step_count: int,
    fine_steps: int,
    schedule: str,
    permutation: torch.Tensor,
    batch_size: int,
    device: torch.device,
    squared_floor: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown transport schedule {schedule}")
    indices = nested_step_indices(fine_steps, step_count)
    final_residuals: list[torch.Tensor] = []
    final_posteriors: list[torch.Tensor] = []
    final_moments: list[torch.Tensor] = []
    path_lengths: list[torch.Tensor] = []
    maximum_steps: list[torch.Tensor] = []
    minimum_gradients: list[torch.Tensor] = []
    endpoint_errors: list[torch.Tensor] = []
    posterior_js: list[torch.Tensor] = []
    target_cosine = target_cosine.float().cpu()
    actual_moments = actual_moments.float().cpu()
    permutation = permutation.long().cpu()

    for start in range(0, len(start_residual), batch_size):
        stop = min(start + batch_size, len(start_residual))
        value = start_residual[start:stop].to(device)
        initial_moment = actual_moments[0, start:stop].to(device)
        length = torch.zeros(stop - start, dtype=value.dtype, device=device)
        maximum = torch.zeros_like(length)
        minimum_gradient = torch.full_like(length, float("inf"))
        for index in indices[1:]:
            value = value.detach().requires_grad_(True)
            current_posterior = _posterior(model, value, answer_ids)
            current_moment = current_posterior @ centers
            gradient = torch.autograd.grad(current_moment.sum(), value)[0]
            fraction = float(index) / float(fine_steps)
            if schedule == "true_cosine":
                terminal = target_cosine[start:stop].to(device)
                requested = (
                    terminal
                    if index == fine_steps
                    else initial_moment + fraction * (terminal - initial_moment)
                )
            elif schedule == "path_moment":
                requested = actual_moments[index, start:stop].to(device)
            else:
                donor = permutation[start:stop]
                requested = initial_moment + (
                    actual_moments[index, donor] - actual_moments[0, donor]
                ).to(device)
            delta = minimum_norm_task_delta(
                current_moment.detach(), requested, gradient.detach(), squared_floor
            )
            # Match the locked source producer exactly for K=1 patch-norm replay.
            step_norm = delta.norm(dim=1)
            length += step_norm
            maximum = torch.maximum(maximum, step_norm)
            minimum_gradient = torch.minimum(
                minimum_gradient, torch.linalg.vector_norm(gradient.detach(), dim=1)
            )
            value = value.detach() + delta.detach()

        with torch.no_grad():
            final_posterior = _posterior(model, value, answer_ids)
            final_moment = final_posterior @ centers
        actual_endpoint = endpoint_residual[start:stop].to(device)
        chord = torch.linalg.vector_norm(
            actual_endpoint - start_residual[start:stop].to(device), dim=1
        ).clamp_min(1e-12)
        endpoint_error = torch.linalg.vector_norm(value - actual_endpoint, dim=1) / chord
        js = jensen_shannon_divergence(
            final_posterior.cpu(), endpoint_posterior[start:stop]
        )
        final_residuals.append(value.detach().cpu())
        final_posteriors.append(final_posterior.detach().cpu())
        final_moments.append(final_moment.detach().cpu())
        path_lengths.append(length.detach().cpu())
        maximum_steps.append(maximum.detach().cpu())
        minimum_gradients.append(minimum_gradient.detach().cpu())
        endpoint_errors.append(endpoint_error.detach().cpu())
        posterior_js.append(js.detach().cpu())

    final_residual = torch.cat(final_residuals)
    final_posterior = torch.cat(final_posteriors)
    final_moment = torch.cat(final_moments)
    path_length = torch.cat(path_lengths)
    maximum_step = torch.cat(maximum_steps)
    minimum_gradient = torch.cat(minimum_gradients)
    endpoint_error = torch.cat(endpoint_errors)
    js = torch.cat(posterior_js)
    endpoint_moment = actual_moments[-1]
    target = target_cosine.double()
    metrics = {
        "task": _task_metrics(final_posterior, dataset),
        "causal_task": _causal_metrics(final_posterior, dataset),
        "mean_absolute_target_cosine_error": float(
            (final_moment.double() - target).abs().mean()
        ),
        "mean_absolute_actual_endpoint_moment_error": float(
            (final_moment.double() - endpoint_moment.double()).abs().mean()
        ),
        "mean_posterior_js_from_actual_endpoint": float(js.mean()),
        "mean_relative_residual_endpoint_error": float(endpoint_error.mean()),
        "maximum_relative_residual_endpoint_error": float(endpoint_error.max()),
        "mean_rollout_path_length": float(path_length.mean()),
        "maximum_step_norm": float(maximum_step.max()),
        "minimum_task_gradient_norm": float(minimum_gradient.min()),
        "finite_contract": bool(
            torch.isfinite(final_residual).all()
            and torch.isfinite(final_posterior).all()
            and torch.isfinite(path_length).all()
            and torch.isfinite(minimum_gradient).all()
        ),
    }
    arrays = {
        "final_moment": final_moment.numpy(),
        "predicted_bin": final_posterior.argmax(1).numpy(),
        "path_length": path_length.numpy(),
        "maximum_step_norm": maximum_step.numpy(),
        "minimum_gradient_norm": minimum_gradient.numpy(),
        "relative_residual_endpoint_error": endpoint_error.numpy(),
        "posterior_js_from_actual_endpoint": js.numpy(),
    }
    return metrics, arrays


def _local_path_diagnostics(
    *,
    model: TinyLLMModel,
    answer_ids: torch.Tensor,
    centers: torch.Tensor,
    residuals: torch.Tensor,
    posteriors: torch.Tensor,
    moments: torch.Tensor,
    batch_size: int,
    device: torch.device,
    squared_floor: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    observations = residuals.shape[1]
    parallel = torch.zeros(observations, dtype=torch.double)
    orthogonal = torch.zeros_like(parallel)
    first_order_error = torch.zeros_like(parallel)
    first_order_scale = torch.zeros_like(parallel)
    local_js = torch.zeros_like(parallel)
    local_residual_error = torch.zeros_like(parallel)
    minimum_gradient = torch.full((observations,), float("inf"), dtype=torch.double)
    step_count = len(residuals) - 1

    for step in range(step_count):
        for start in range(0, observations, batch_size):
            stop = min(start + batch_size, observations)
            value = residuals[step, start:stop].to(device).detach().requires_grad_(True)
            posterior = _posterior(model, value, answer_ids)
            moment = posterior @ centers
            gradient = torch.autograd.grad(moment.sum(), value)[0]
            increment = (
                residuals[step + 1, start:stop] - residuals[step, start:stop]
            ).to(device)
            squared = gradient.detach().square().sum(1).clamp_min(squared_floor)
            coefficient = (increment * gradient.detach()).sum(1) / squared
            projected = coefficient[:, None] * gradient.detach()
            increment_norm = torch.linalg.vector_norm(increment, dim=1).clamp_min(1e-12)
            requested = moments[step + 1, start:stop].to(device)
            local_delta = minimum_norm_task_delta(
                moment.detach(), requested, gradient.detach(), squared_floor
            )
            with torch.no_grad():
                patched = _posterior(model, value.detach() + local_delta, answer_ids)
            predicted_change = (gradient.detach() * increment).sum(1)
            actual_change = (
                moments[step + 1, start:stop] - moments[step, start:stop]
            ).to(device)
            region = slice(start, stop)
            parallel[region] += (
                torch.linalg.vector_norm(projected, dim=1) / increment_norm
            ).double().cpu()
            orthogonal[region] += (
                torch.linalg.vector_norm(increment - projected, dim=1) / increment_norm
            ).double().cpu()
            first_order_error[region] += (
                predicted_change - actual_change
            ).abs().double().cpu()
            first_order_scale[region] += actual_change.abs().double().cpu()
            local_js[region] += jensen_shannon_divergence(
                patched.cpu(), posteriors[step + 1, start:stop]
            ).cpu()
            local_residual_error[region] += (
                torch.linalg.vector_norm(local_delta - increment, dim=1) / increment_norm
            ).double().cpu()
            minimum_gradient[region] = torch.minimum(
                minimum_gradient[region],
                torch.linalg.vector_norm(gradient.detach(), dim=1).double().cpu(),
            )

    parallel /= step_count
    orthogonal /= step_count
    first_order_error /= step_count
    first_order_scale /= step_count
    local_js /= step_count
    local_residual_error /= step_count
    summary = {
        "mean_parallel_increment_fraction": float(parallel.mean()),
        "mean_orthogonal_increment_fraction": float(orthogonal.mean()),
        "mean_absolute_first_order_moment_error": float(first_order_error.mean()),
        "relative_first_order_moment_error": float(
            first_order_error.sum() / first_order_scale.sum().clamp_min(1e-12)
        ),
        "mean_local_posterior_js": float(local_js.mean()),
        "mean_local_relative_residual_error": float(local_residual_error.mean()),
        "minimum_on_path_task_gradient_norm": float(minimum_gradient.min()),
    }
    arrays = {
        "parallel_increment_fraction": parallel.numpy(),
        "orthogonal_increment_fraction": orthogonal.numpy(),
        "absolute_first_order_moment_error": first_order_error.numpy(),
        "local_posterior_js": local_js.numpy(),
        "local_relative_residual_error": local_residual_error.numpy(),
        "minimum_on_path_gradient_norm": minimum_gradient.numpy(),
    }
    return summary, arrays


def _fingerprint(
    config: ReferencePathTransportConfig,
    implementation: str,
    condition: str,
    seed: int,
    source_entry: Mapping[str, Any],
    source_detail: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "condition": condition,
        "seed": seed,
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_result_sha256": source_entry["result_sha256"],
        "source_scientific_fingerprint": source_detail["scientific_fingerprint"],
        "source_provenance": source_detail["provenance"],
        "dataset_hashes": dict(dataset_hashes),
        "repeat_arrays_sha256": SOURCE_REPEAT_ARRAY_SHA256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_result(
    path: Path,
    fingerprint: str,
    implementation: str,
    condition: str,
    seed: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    arrays = Path(value.get("artifacts", {}).get("diagnostic_arrays", ""))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("implementation_sha256") == implementation
        and value.get("condition") == condition
        and int(value.get("seed", -1)) == seed
        and arrays.is_file()
        and _sha256(arrays)
        == value.get("artifacts", {}).get("diagnostic_arrays_sha256")
    ):
        raise ValueError(f"incompatible completed transport result {path}")
    return value


def _campaign_reusable(
    value: Mapping[str, Any],
    config: ReferencePathTransportConfig,
    implementation: str,
) -> bool:
    try:
        entries = list(value.get("results", []))
        result_paths = [Path(entry["path"]) for entry in entries]
        indexed = {
            (entry["condition"], int(entry["seed"])) for entry in entries
        }
    except (KeyError, TypeError, ValueError):
        return False
    expected = {
        (condition, seed) for condition in CONDITIONS for seed in config.seeds
    }
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and json.dumps(value.get("configuration"), sort_keys=True)
        == json.dumps(asdict(config), sort_keys=True)
        and indexed == expected
        and len(entries) == len(expected)
        and int(value.get("summary", {}).get("requested", -1)) == len(expected)
        and int(value.get("summary", {}).get("completed", -1)) == len(expected)
        and _manifest_sha256(result_paths)
        == value.get("result_manifest_sha256")
        and all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry.get("result_sha256")
            and Path(entry["arrays_path"]).is_file()
            and _sha256(Path(entry["arrays_path"])) == entry.get("arrays_sha256")
            for entry in value.get("results", [])
        )
    )


def _shuffle_seed(condition: str, seed: int, regime: str, analysis_seed: int) -> int:
    payload = f"{HYPOTHESIS_ID}:{condition}:{seed}:{regime}:{analysis_seed}".encode()
    return SHUFFLE_SEED_OFFSET + int(hashlib.sha256(payload).hexdigest()[:8], 16)


def run_campaign(
    config: ReferencePathTransportConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    implementation = _implementation_digest(source_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_reusable(existing, config, implementation):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed transport campaign {campaign_path}")

    source, source_path, source_entries = _load_source_campaign(config)
    source_acquisition_config = _task_config_from_source(source)
    _, _, _, task, source_frontend_config = acquisition._load_sources(
        source_acquisition_config
    )
    runtime_config = replace(
        source_frontend_config,
        seeds=config.seeds,
        activation_batch_size=config.activation_batch_size,
        allow_underpowered=config.allow_underpowered,
    )
    datasets = {
        regime: calibrated.generate_calibrated_dataset(
            task,
            sample_count=orientation.SPLIT_SPECS[regime][2],
            seed=orientation.SPLIT_SPECS[regime][0],
            regime=orientation.SPLIT_SPECS[regime][1],
            shuffle=True,
        )
        for regime in REGIMES
    }
    dataset_hashes = {name: _dataset_hash(value) for name, value in datasets.items()}
    if any(dataset_hashes[name] != source["dataset_hashes"][name] for name in REGIMES):
        raise ValueError("transport datasets do not replay the source campaign")

    repeat_path = Path(source["artifacts"]["repeat_arrays"])
    reference_paths: dict[str, torch.Tensor] = {}
    path_sharing_errors: dict[str, float] = {}
    path_endpoint_errors: dict[str, float] = {}
    path_unit_errors: dict[str, float] = {}
    with np.load(repeat_path, allow_pickle=False) as archive:
        for regime, dataset in datasets.items():
            stored_fibers = torch.from_numpy(archive[f"{regime}__fiber_id"])
            noise = torch.from_numpy(archive[f"{regime}__repeat_noise"])
            if not torch.equal(stored_fibers.long(), dataset.paired.fiber.fiber_id.long()):
                raise ValueError(f"stored fiber IDs changed for {regime}")
            observations = acquisition.repeated_observations(
                dataset.calibration, noise, source_acquisition_config.sigma_radians
            )
            q1 = acquisition.analytic_circular_mean(observations, 1)
            q64 = acquisition.analytic_circular_mean(observations, 64)
            reference_paths[regime] = shortest_reference_path(
                q1, q64, config.fine_steps
            )
            path_endpoint_errors[regime] = max(
                float((reference_paths[regime][0] - q1).abs().max()),
                float((reference_paths[regime][-1] - q64).abs().max()),
            )
            path_unit_errors[regime] = float(
                (reference_paths[regime].abs() - 1.0).abs().max()
            )
            increments = torch.angle(
                reference_paths[regime]
                * reference_paths[regime][0:1].conj()
            )
            maximum = 0.0
            for fiber in torch.unique(dataset.paired.fiber.fiber_id, sorted=True):
                selected = increments[
                    :, dataset.paired.fiber.fiber_id == fiber
                ]
                maximum = max(maximum, float((selected - selected[:, :1]).abs().max()))
            path_sharing_errors[regime] = maximum
    path_sharing_contract = all(
        value <= config.path_sharing_tolerance
        for value in path_sharing_errors.values()
    )
    path_construction_contract = bool(
        all(
            value <= config.path_sharing_tolerance
            for value in path_endpoint_errors.values()
        )
        and all(
            value <= config.path_sharing_tolerance
            for value in path_unit_errors.values()
        )
    )
    nested_grid_contract = all(
        nested_step_indices(config.fine_steps, count)[0] == 0
        and nested_step_indices(config.fine_steps, count)[-1]
        == config.fine_steps
        and len(nested_step_indices(config.fine_steps, count)) == count + 1
        for count in config.step_counts
    )

    results: list[dict[str, Any]] = []
    reused = 0
    source_model_root = Path(source_acquisition_config.source_root)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    centers = torch.linspace(-1.0, 1.0, len(task.answer_token_ids), device=device)

    for condition in CONDITIONS:
        for seed in config.seeds:
            source_entry = source_entries[(condition, seed)]
            source_detail = json.loads(
                Path(source_entry["path"]).read_text(encoding="utf-8")
            )
            system, provenance = orientation._load_system(
                source_model_root,
                condition,
                seed,
                task,
                runtime_config,
                device,
            )
            for name in (
                "model_checkpoint_sha256",
                "frontend_checkpoint_sha256",
                "model_state_sha256",
                "system_state_sha256",
            ):
                if provenance[name] != source_detail["provenance"][name]:
                    raise ValueError(f"source provenance mismatch {condition} seed {seed}: {name}")
            fingerprint = _fingerprint(
                config,
                implementation,
                condition,
                seed,
                source_entry,
                source_detail,
                dataset_hashes,
            )
            result_dir = output / "runs" / condition / f"seed_{seed}"
            result_path = result_dir / "result.json"
            existing = _reusable_result(
                result_path, fingerprint, implementation, condition, seed
            )
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"reuse {condition} seed {seed}", flush=True)
                del system
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            seed_started = time.perf_counter()
            initial_state = calibrated._module_digest(system)
            regime_records: dict[str, Any] = {}
            diagnostic_arrays: dict[str, np.ndarray] = {}
            maximum_replay_error = 0.0
            minimum_gradient = float("inf")
            finite = True

            for regime_index, regime in enumerate(REGIMES):
                dataset = datasets[regime]
                clean_residual, clean_posterior = _forward_packet(
                    system,
                    dataset,
                    dataset.calibration,
                    task,
                    config.activation_batch_size,
                    device,
                )
                actual_residuals, actual_posteriors = _extract_actual_path(
                    system,
                    dataset,
                    reference_paths[regime],
                    task,
                    config.activation_batch_size,
                    device,
                )
                actual_moments = actual_posteriors.to(device) @ centers
                actual_moments = actual_moments.cpu()
                clean_metrics = _task_metrics(clean_posterior, dataset)
                actual_curve = [
                    _task_metrics(actual_posteriors[index], dataset)
                    for index in range(config.fine_steps + 1)
                ]
                endpoint_replay_posterior_parts: list[torch.Tensor] = []
                with torch.no_grad():
                    for start in range(0, len(dataset.calibration), config.activation_batch_size):
                        stop = min(start + config.activation_batch_size, len(dataset.calibration))
                        endpoint_replay_posterior_parts.append(
                            _posterior(
                                system.model,
                                actual_residuals[-1, start:stop].to(device),
                                answer_ids,
                            ).cpu()
                        )
                endpoint_replay_posterior = torch.cat(endpoint_replay_posterior_parts)
                endpoint_replay_metrics = _task_metrics(endpoint_replay_posterior, dataset)
                endpoint_replay_error = _metric_error(
                    endpoint_replay_metrics,
                    actual_curve[-1],
                    (
                        "exact_bin_accuracy",
                        "mean_circular_error_radians",
                        "mean_target_cross_entropy",
                    ),
                )

                source_clean = source_detail["clean"][regime]["task"]
                source_m1 = source_detail["evaluations"]["analytic_circular_mean"]["1"][
                    "regimes"
                ][regime]["task"]
                source_m64 = source_detail["evaluations"]["analytic_circular_mean"]["64"][
                    "regimes"
                ][regime]["task"]
                maximum_replay_error = max(
                    maximum_replay_error,
                    _metric_error(
                        clean_metrics,
                        source_clean,
                        (
                            "exact_bin_accuracy",
                            "mean_circular_error_radians",
                            "mean_target_cross_entropy",
                        ),
                    ),
                    _metric_error(
                        actual_curve[0],
                        source_m1,
                        (
                            "exact_bin_accuracy",
                            "mean_circular_error_radians",
                            "mean_target_cross_entropy",
                        ),
                    ),
                    _metric_error(
                        actual_curve[-1],
                        source_m64,
                        (
                            "exact_bin_accuracy",
                            "mean_circular_error_radians",
                            "mean_target_cross_entropy",
                        ),
                    ),
                    endpoint_replay_error,
                )

                geometry_arrays = residual_curve_geometry(actual_residuals)
                local_summary, local_arrays = _local_path_diagnostics(
                    model=system.model,
                    answer_ids=answer_ids,
                    centers=centers,
                    residuals=actual_residuals,
                    posteriors=actual_posteriors,
                    moments=actual_moments,
                    batch_size=config.activation_batch_size,
                    device=device,
                    squared_floor=config.gradient_squared_floor,
                )
                minimum_gradient = min(
                    minimum_gradient, local_summary["minimum_on_path_task_gradient_norm"]
                )
                shuffle_index = fiber_block_permutation(
                    dataset.paired.fiber.fiber_id,
                    dataset.paired.fiber.branch,
                    shift=_shuffle_seed(
                        condition, seed, regime, config.analysis_seed + regime_index
                    ),
                )
                rollouts: dict[str, dict[str, Any]] = {
                    schedule: {} for schedule in SCHEDULES
                }
                for schedule in SCHEDULES:
                    for count in config.step_counts:
                        record, arrays = _rollout(
                            model=system.model,
                            answer_ids=answer_ids,
                            centers=centers,
                            start_residual=actual_residuals[0],
                            endpoint_residual=actual_residuals[-1],
                            endpoint_posterior=actual_posteriors[-1],
                            actual_moments=actual_moments,
                            target_cosine=dataset.paired.fiber.cosine,
                            dataset=dataset,
                            step_count=count,
                            fine_steps=config.fine_steps,
                            schedule=schedule,
                            permutation=shuffle_index,
                            batch_size=config.activation_batch_size,
                            device=device,
                            squared_floor=config.gradient_squared_floor,
                        )
                        rollouts[schedule][str(count)] = record
                        minimum_gradient = min(
                            minimum_gradient, record["minimum_task_gradient_norm"]
                        )
                        finite = bool(finite and record["finite_contract"])
                        prefix = f"{regime}__{schedule}__k{count}"
                        for name, value in arrays.items():
                            diagnostic_arrays[f"{prefix}__{name}"] = value

                source_oracle = source_detail["causal_ceiling"]["regimes"][regime][
                    "target_oracle"
                ]
                maximum_replay_error = max(
                    maximum_replay_error,
                    _metric_error(
                        rollouts["true_cosine"]["1"]["causal_task"],
                        source_oracle,
                        ("exact_bin_accuracy", "mean_target_cross_entropy"),
                    ),
                    abs(
                        float(
                            rollouts["true_cosine"]["1"][
                                "mean_rollout_path_length"
                            ]
                        )
                        - float(source_oracle["mean_patch_norm"])
                    ),
                )
                geometry_summary = {
                    name: float(value.mean())
                    for name, value in geometry_arrays.items()
                    if name != "chord_norm"
                }
                geometry_summary["minimum_chord_norm"] = float(
                    geometry_arrays["chord_norm"].min()
                )
                regime_records[regime] = {
                    "clean": clean_metrics,
                    "actual_reference_path": {
                        "task_by_step": actual_curve,
                        "endpoint_replay": endpoint_replay_metrics,
                        "maximum_endpoint_replay_error": endpoint_replay_error,
                        "geometry": geometry_summary,
                        "local_transport": local_summary,
                    },
                    "rollouts": rollouts,
                    "shuffle": {
                        "seed": _shuffle_seed(
                            condition, seed, regime, config.analysis_seed + regime_index
                        ),
                        "fixed_point_count": int(
                            (shuffle_index == torch.arange(len(shuffle_index))).sum()
                        ),
                    },
                }
                diagnostic_arrays[f"{regime}__actual_moment"] = actual_moments.numpy()
                diagnostic_arrays[f"{regime}__actual_predicted_bin"] = (
                    actual_posteriors.argmax(2).numpy()
                )
                diagnostic_arrays[f"{regime}__reference_delta"] = torch.angle(
                    reference_paths[regime][-1]
                    * reference_paths[regime][0].conj()
                ).numpy()
                diagnostic_arrays[f"{regime}__shuffle_index"] = shuffle_index.numpy()
                for name, value in geometry_arrays.items():
                    diagnostic_arrays[f"{regime}__geometry__{name}"] = value.numpy()
                for name, value in local_arrays.items():
                    diagnostic_arrays[f"{regime}__local__{name}"] = value

            clean_accuracy = {
                regime: regime_records[regime]["clean"]["exact_bin_accuracy"]
                for regime in REGIMES
            }
            actual_records = {
                regime: regime_records[regime]["actual_reference_path"]["task_by_step"][-1]
                for regime in REGIMES
            }
            endpoint_records = {
                regime: regime_records[regime]["actual_reference_path"]["endpoint_replay"]
                for regime in REGIMES
            }
            actual_gate, actual_losses = task_gate(
                clean_accuracy, actual_records, config.accuracy_loss_ceiling
            )
            endpoint_gate, endpoint_losses = task_gate(
                clean_accuracy, endpoint_records, config.accuracy_loss_ceiling
            )
            rollout_gates: dict[str, dict[str, Any]] = {
                schedule: {} for schedule in SCHEDULES
            }
            for schedule in SCHEDULES:
                for count in config.step_counts:
                    records = {
                        regime: regime_records[regime]["rollouts"][schedule][str(count)][
                            "task"
                        ]
                        for regime in REGIMES
                    }
                    passed, losses = task_gate(
                        clean_accuracy, records, config.accuracy_loss_ceiling
                    )
                    rollout_gates[schedule][str(count)] = {
                        "task_gate": passed,
                        "accuracy_loss_from_clean": losses,
                    }

            state_unchanged = calibrated._module_digest(system) == initial_state
            replay_contract = maximum_replay_error <= config.replay_tolerance
            gradient_floor_clear = minimum_gradient > config.gradient_norm_floor
            finite = bool(
                finite
                and _finite_numbers(regime_records)
                and all(np.isfinite(value).all() for value in diagnostic_arrays.values())
            )
            systems_valid = bool(
                path_sharing_contract
                and path_construction_contract
                and nested_grid_contract
                and replay_contract
                and finite
                and state_unchanged
            )
            arrays_path = result_dir / "transport_diagnostics.npz"
            _write_npz(arrays_path, diagnostic_arrays)
            arrays_sha256 = _sha256(arrays_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-reference-path-transport-{condition}-seed{seed}",
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "provenance": {
                    **provenance,
                    "source_acquisition_result": source_entry["path"],
                    "source_acquisition_result_sha256": source_entry["result_sha256"],
                    "source_acquisition_scientific_fingerprint": source_detail[
                        "scientific_fingerprint"
                    ],
                },
                "dataset_hashes": dataset_hashes,
                "regimes": regime_records,
                "transport_gates": {
                    "actual_m64_reference": {
                        "task_gate": actual_gate,
                        "accuracy_loss_from_clean": actual_losses,
                    },
                    "exact_endpoint_residual": {
                        "task_gate": endpoint_gate,
                        "accuracy_loss_from_clean": endpoint_losses,
                    },
                    "rollouts": rollout_gates,
                },
                "maximum_source_metric_replay_error": maximum_replay_error,
                "minimum_task_gradient_norm": minimum_gradient,
                "gates": {
                    "source_metric_replay_contract": replay_contract,
                    "exact_fiber_path_sharing_contract": path_sharing_contract,
                    "reference_path_construction_contract": (
                        path_construction_contract
                    ),
                    "nested_step_grid_contract": nested_grid_contract,
                    "finite_numerical_contract": finite,
                    "task_gradient_floor_clear_observation": gradient_floor_clear,
                    "system_state_unchanged_contract": state_unchanged,
                    "validity_contract": systems_valid,
                },
                "artifacts": {
                    "diagnostic_arrays": str(arrays_path),
                    "diagnostic_arrays_sha256": arrays_sha256,
                },
                "analysis_seconds": time.perf_counter() - seed_started,
            }
            _write_json(result_path, result)
            results.append(result)
            if _implementation_digest() != implementation:
                raise RuntimeError("transport implementation changed during campaign")
            print(f"completed {condition} seed {seed}", flush=True)
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()

    required = len(config.seeds) if config.allow_underpowered else config.required_seed_passes
    final_key = str(config.fine_steps)
    rollout_counts = {
        schedule: {
            str(count): {
                condition: sum(
                    int(
                        result["transport_gates"]["rollouts"][schedule][str(count)][
                            "task_gate"
                        ]
                    )
                    for result in results
                    if result["condition"] == condition
                )
                for condition in CONDITIONS
            }
            for count in config.step_counts
        }
        for schedule in SCHEDULES
    }
    actual_counts = {
        condition: sum(
            int(result["transport_gates"]["actual_m64_reference"]["task_gate"])
            for result in results
            if result["condition"] == condition
        )
        for condition in CONDITIONS
    }
    endpoint_counts = {
        condition: sum(
            int(result["transport_gates"]["exact_endpoint_residual"]["task_gate"])
            for result in results
            if result["condition"] == condition
        )
        for condition in CONDITIONS
    }
    one_step_counts = rollout_counts["true_cosine"]["1"]
    positive_control = all(
        actual_counts[condition] == len(config.seeds)
        and endpoint_counts[condition] == len(config.seeds)
        for condition in CONDITIONS
    )
    shuffled_control = all(
        rollout_counts["shuffled_path_moment"][final_key][condition]
        <= config.control_pass_ceiling
        for condition in CONDITIONS
    )
    source_one_step_below_population = all(
        one_step_counts[condition] < required for condition in CONDITIONS
    )
    systems_valid = bool(
        path_sharing_contract
        and all(result["gates"]["validity_contract"] for result in results)
    )
    scientific_valid = bool(
        systems_valid
        and positive_control
        and shuffled_control
        and source_one_step_below_population
    )
    valid = systems_valid if config.allow_underpowered else scientific_valid
    classification = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else classify_campaign(
            valid=valid,
            true_cosine_pass=rollout_counts["true_cosine"][final_key],
            path_moment_pass=rollout_counts["path_moment"][final_key],
            required=required,
        )
    )
    first_passing_steps = {
        result["experiment_id"]: {
            schedule: next(
                (
                    count
                    for count in config.step_counts
                    if result["transport_gates"]["rollouts"][schedule][str(count)][
                        "task_gate"
                    ]
                ),
                None,
            )
            for schedule in SCHEDULES
        }
        for result in results
    }
    result_manifest = _manifest_sha256(
        [
            output / "runs" / result["condition"] / f"seed_{result['seed']}" / "result.json"
            for result in results
        ]
    )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "provenance": {
            "source_campaign": str(source_path),
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
            "source_result_manifest_sha256": SOURCE_RESULT_MANIFEST_SHA256,
            "source_repeat_arrays_sha256": SOURCE_REPEAT_ARRAY_SHA256,
        },
        "dataset_hashes": dataset_hashes,
        "path_contracts": {
            "exact_fiber_path_sharing_contract": path_sharing_contract,
            "maximum_exact_fiber_path_sharing_error": path_sharing_errors,
            "reference_path_construction_contract": path_construction_contract,
            "maximum_reference_path_endpoint_error": path_endpoint_errors,
            "maximum_reference_path_unit_norm_error": path_unit_errors,
            "nested_step_grid_contract": nested_grid_contract,
        },
        "aggregates": {
            "valid": valid,
            "systems_valid": systems_valid,
            "scientific_controls_valid": scientific_valid,
            "classification": classification,
            "required_seed_passes": required,
            "rollout_pass_counts": rollout_counts,
            "actual_m64_reference_pass_counts": actual_counts,
            "exact_endpoint_residual_pass_counts": endpoint_counts,
            "source_one_step_true_cosine_pass_counts": one_step_counts,
            "source_one_step_below_population_contract": (
                source_one_step_below_population
            ),
            "positive_control_contract": positive_control,
            "shuffled_control_contract": shuffled_control,
            "first_passing_steps": first_passing_steps,
        },
        "summary": {
            "requested": len(CONDITIONS) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "trained_probes": 0,
            "fitted_transport_parameters": 0,
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "condition": result["condition"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "gates": result["gates"],
                "transport_gates": result["transport_gates"],
                "path": str(
                    output
                    / "runs"
                    / result["condition"]
                    / f"seed_{result['seed']}"
                    / "result.json"
                ),
                "result_sha256": _sha256(
                    output
                    / "runs"
                    / result["condition"]
                    / f"seed_{result['seed']}"
                    / "result.json"
                ),
                "arrays_path": result["artifacts"]["diagnostic_arrays"],
                "arrays_sha256": result["artifacts"]["diagnostic_arrays_sha256"],
            }
            for result in results
        ],
        "result_manifest_sha256": result_manifest,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "All front ends, TinyLLM checkpoints, embeddings, norms, and answer rows remain frozen.",
            "The true-cosine rollout uses latent task information and is not deployable.",
            "The path-moment rollout reads frozen model outputs and is a mechanistic oracle.",
            "Only final-query residual transport is tested; earlier block geometry is not measured.",
            "The reference path comes from one stored unbiased synthetic Gaussian acquisition array.",
            "No topology, link, or cobordism invariant is computed.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "*" / "seed_*" / "result.json"),
            "diagnostic_arrays": str(
                output / "runs" / "*" / "seed_*" / "transport_diagnostics.npz"
            ),
        },
    }
    if _implementation_digest() != implementation:
        raise RuntimeError("transport implementation changed before campaign write")
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_reference_path_residual_transport/"
            "20260807_d8_corrected_v4"
        ),
    )
    args = parser.parse_args()
    seeds = _ints(args.seeds)
    if args.shakedown:
        seeds = (7,)
        step_counts = (1, 16)
        allow_underpowered = True
    else:
        step_counts = STEP_COUNTS
        allow_underpowered = args.allow_underpowered
    config = ReferencePathTransportConfig(
        seeds=seeds,
        step_counts=step_counts,
        required_seed_passes=(len(seeds) if allow_underpowered else 4),
        control_pass_ceiling=(len(seeds) if allow_underpowered else 1),
        device=args.device,
        allow_underpowered=allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2), flush=True)
    print(campaign["artifacts"]["campaign"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
