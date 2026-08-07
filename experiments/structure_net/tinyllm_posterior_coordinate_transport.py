#!/usr/bin/env python3
"""Test nested answer-coordinate residual transport along the frozen reference path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibration_orientation_noise as orientation
import experiments.structure_net.tinyllm_reference_acquisition_replicates as acquisition
import experiments.structure_net.tinyllm_reference_path_residual_transport as transport
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-posterior-coordinate-transport.v1"
HYPOTHESIS_ID = "tinyllm-posterior-coordinate-transport-v1"
SOURCE_SCHEMA = transport.SCHEMA_VERSION
SOURCE_HYPOTHESIS = transport.HYPOTHESIS_ID
SOURCE_CAMPAIGN_SHA256 = (
    "6b232f523cd570f10ebfcc07c47abae6a724b568d4cfc2852e4b91ccff01321f"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "5cb1944e57db0515dbc7ad5e3956a328be733cdda7f5dda34631d21e8cf3a81c"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "5d765445082346512092dfa0acb00e3e39db369bd71829cbe4d96c8871593b4b"
)
EVIDENCE_ROLE = "preregistered_outcome_directed_posterior_coordinate_transport"
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
CONDITIONS = transport.CONDITIONS
REGIMES = transport.REGIMES
ANSWER_BINS = 16
FULL_RANK = ANSWER_BINS - 1
RANKS = (1, 2, 4, 8, FULL_RANK)
STEP_COUNTS = (4, 16)
FINE_STEPS = 16
PRIMARY_STEP_COUNT = 16
SCHEDULES = ("actual_coordinate", "shuffled_coordinate")
COMPARATOR_STEP_COUNTS = (4, 16)
SHUFFLE_SEED_OFFSET = 977_700_013
TASK_METRIC_NAMES = (
    "exact_bin_accuracy",
    "mean_circular_error_radians",
    "mean_target_cross_entropy",
)

_utc_now = transport._utc_now
_sha256 = transport._sha256
_manifest_sha256 = transport._manifest_sha256
_write_json = transport._write_json
_write_npz = transport._write_npz
_finite_numbers = transport._finite_numbers


def rank_label(rank: int) -> str:
    return "full" if int(rank) == FULL_RANK else str(int(rank))


@dataclass(frozen=True)
class PosteriorCoordinateTransportConfig:
    source_root: str = (
        "data/experiments/tinyllm_reference_path_residual_transport/"
        "20260807_d8_corrected_v4"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    ranks: tuple[int, ...] = RANKS
    step_counts: tuple[int, ...] = STEP_COUNTS
    fine_steps: int = FINE_STEPS
    activation_batch_size: int = 256
    accuracy_loss_ceiling: float = 0.03
    replay_tolerance: float = 2e-6
    orthonormal_tolerance: float = 1e-10
    pinv_rcond: float = 1e-6
    posterior_floor: float = 1e-12
    required_seed_passes: int = 4
    control_pass_ceiling: int = 1
    analysis_seed: int = 41_141
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if (
            not self.ranks
            or tuple(sorted(set(self.ranks))) != self.ranks
            or self.ranks[-1] != FULL_RANK
            or any(rank < 1 or rank > FULL_RANK for rank in self.ranks)
        ):
            raise ValueError("ranks must be ordered, distinct, and end at full")
        if (
            not self.step_counts
            or tuple(sorted(set(self.step_counts))) != self.step_counts
            or any(count < 1 or self.fine_steps % count for count in self.step_counts)
            or self.step_counts[-1] != self.fine_steps
        ):
            raise ValueError("step counts must be ordered divisors ending at the grid")
        if self.activation_batch_size < 1:
            raise ValueError("activation batch size must be positive")
        if self.required_seed_passes > len(self.seeds):
            raise ValueError("required seed passes exceed selected checkpoints")
        for name in (
            "accuracy_loss_ceiling",
            "replay_tolerance",
            "orthonormal_tolerance",
            "pinv_rcond",
            "posterior_floor",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.allow_underpowered:
            if self.seeds != PRIMARY_SEEDS:
                raise ValueError("the primary checkpoint seeds are fixed")
            if self.ranks != RANKS:
                raise ValueError("the primary coordinate rank ladder is fixed")
            if self.step_counts != STEP_COUNTS or self.fine_steps != FINE_STEPS:
                raise ValueError("the primary transport grid is fixed")
            if self.pinv_rcond != 1e-6:
                raise ValueError("the primary pseudoinverse cutoff is fixed")
            if self.posterior_floor != 1e-12:
                raise ValueError("the primary posterior floor is fixed")
            if self.required_seed_passes != 4 or self.control_pass_ceiling != 1:
                raise ValueError("the primary population and control gates are fixed")
            if self.analysis_seed != 41_141:
                raise ValueError("the primary analysis seed is fixed")


def _runner_protocol_digest() -> str:
    """Hash scientific and lifecycle semantics while excluding CLI path defaults."""

    protocol_names = (
        "PosteriorCoordinateTransportConfig",
        "rank_label",
        "_evidence_role",
        "dct_coordinate_basis",
        "centered_log_posterior",
        "coordinate_schedule",
        "classify_campaign",
        "_shuffle_seed",
        "_transport_config_from_source",
        "_load_source_campaign",
        "_coordinate_jacobian",
        "_pseudoinverse_step",
        "_vector_rollout",
        "_fingerprint",
        "_reusable_result",
        "_campaign_reusable",
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
        "evidence_role": EVIDENCE_ROLE,
        "primary_seeds": PRIMARY_SEEDS,
        "conditions": CONDITIONS,
        "regimes": REGIMES,
        "answer_bins": ANSWER_BINS,
        "full_rank": FULL_RANK,
        "ranks": RANKS,
        "step_counts": STEP_COUNTS,
        "fine_steps": FINE_STEPS,
        "primary_step_count": PRIMARY_STEP_COUNT,
        "schedules": SCHEDULES,
        "comparator_step_counts": COMPARATOR_STEP_COUNTS,
        "shuffle_seed_offset": SHUFFLE_SEED_OFFSET,
        "task_metric_names": TASK_METRIC_NAMES,
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
        "reference_path_transport": Path(transport.__file__),
        "acquisition": Path(acquisition.__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "orientation_noise": Path(orientation.__file__),
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


def _evidence_role(config: PosteriorCoordinateTransportConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else EVIDENCE_ROLE
    )


def dct_coordinate_basis(bins: int) -> torch.Tensor:
    """Return the orthonormal DCT-II rows with the constant vector omitted."""

    if bins < 2:
        raise ValueError("bins must be at least two")
    grid = torch.arange(bins, dtype=torch.float64)
    rows = [
        math.sqrt(2.0 / bins)
        * torch.cos(math.pi * (grid + 0.5) * frequency / bins)
        for frequency in range(1, bins)
    ]
    return torch.stack(rows)


def centered_log_posterior(posterior: torch.Tensor, floor: float) -> torch.Tensor:
    logp = posterior.clamp_min(float(floor)).log()
    return logp - logp.mean(dim=-1, keepdim=True)


def coordinate_schedule(
    posteriors: torch.Tensor, basis: torch.Tensor, floor: float
) -> torch.Tensor:
    """Project stored path posteriors onto the full coordinate basis."""

    if posteriors.ndim != 3 or posteriors.shape[-1] != basis.shape[-1]:
        raise ValueError("posterior schedule must have shape [steps,observations,bins]")
    ell = centered_log_posterior(posteriors.double(), floor)
    return ell @ basis.transpose(0, 1)


def classify_campaign(
    *,
    valid: bool,
    shuffled_valid: bool,
    rank_gates: Mapping[str, bool],
    ranks: tuple[int, ...] = RANKS,
) -> str:
    if not valid:
        return "invalid"
    if not shuffled_valid:
        return "invalid_control"
    smallest = next(
        (rank for rank in ranks if bool(rank_gates[rank_label(rank)])), None
    )
    if smallest is None:
        return "answer_coordinates_nonintegrable"
    if smallest <= 4:
        return "compact_vector_chart"
    return "high_rank_answer_chart"


def _shuffle_seed(condition: str, seed: int, regime: str, analysis_seed: int) -> int:
    payload = f"{HYPOTHESIS_ID}:{condition}:{seed}:{regime}:{analysis_seed}".encode()
    return SHUFFLE_SEED_OFFSET + int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _transport_config_from_source(
    source: Mapping[str, Any],
) -> transport.ReferencePathTransportConfig:
    value = dict(source["configuration"])
    value["seeds"] = tuple(int(seed) for seed in value["seeds"])
    value["step_counts"] = tuple(int(count) for count in value["step_counts"])
    return transport.ReferencePathTransportConfig(**value)


def _load_source_campaign(
    config: PosteriorCoordinateTransportConfig,
) -> tuple[dict[str, Any], Path, dict[tuple[str, int], dict[str, Any]]]:
    path = Path(config.source_root) / "campaign_results.json"
    source = json.loads(path.read_text(encoding="utf-8"))
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
        or any(
            not result_path.is_file()
            or _sha256(result_path) != entries[key].get("result_sha256")
            for key, result_path in zip(entries, result_paths)
        )
        or any(
            not Path(entry["arrays_path"]).is_file()
            or _sha256(Path(entry["arrays_path"])) != entry.get("arrays_sha256")
            for entry in entries.values()
        )
        or _manifest_sha256(result_paths) != SOURCE_RESULT_MANIFEST_SHA256
        or transport._source_digests() != source.get("source_digests")
        or transport._implementation_digest(source["source_digests"])
        != SOURCE_IMPLEMENTATION_SHA256
    ):
        raise ValueError(f"invalid reference-path transport source campaign {path}")
    return source, path, entries


def _coordinate_jacobian(
    value: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """Stack per-coordinate gradients into a [batch, rank, width] Jacobian."""

    rows = []
    rank = coordinates.shape[1]
    for index in range(rank):
        rows.append(
            torch.autograd.grad(
                coordinates[:, index].sum(),
                value,
                retain_graph=index + 1 < rank,
            )[0]
        )
    return torch.stack(rows, dim=1)


def _pseudoinverse_step(
    jacobian: torch.Tensor, coordinate_error: torch.Tensor, rcond: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the minimum-norm SVD update, effective rank, and condition number."""

    if jacobian.ndim != 3 or coordinate_error.shape != jacobian.shape[:2]:
        raise ValueError("jacobian must be [batch,rank,width] with matching error")
    left, singular, right = torch.linalg.svd(jacobian.double(), full_matrices=False)
    keep = singular > (float(rcond) * singular[:, :1])
    inverse = torch.where(keep, singular.reciprocal(), torch.zeros_like(singular))
    weights = inverse * (
        left.transpose(1, 2) @ coordinate_error.double().unsqueeze(-1)
    ).squeeze(-1)
    delta = (right.transpose(1, 2) @ weights.unsqueeze(-1)).squeeze(-1)
    effective_rank = keep.sum(1)
    retained_minimum = torch.where(
        keep, singular, torch.full_like(singular, float("inf"))
    ).min(1).values
    condition = torch.where(
        effective_rank > 0,
        singular[:, 0] / retained_minimum,
        torch.zeros_like(singular[:, 0]),
    )
    return delta, effective_rank, condition


def _vector_rollout(
    *,
    model: TinyLLMModel,
    answer_ids: torch.Tensor,
    basis: torch.Tensor,
    rank: int,
    start_residual: torch.Tensor,
    endpoint_residual: torch.Tensor,
    endpoint_posterior: torch.Tensor,
    schedule_coordinates: torch.Tensor,
    dataset: calibrated.CalibratedDataset,
    step_count: int,
    fine_steps: int,
    schedule: str,
    permutation: torch.Tensor,
    batch_size: int,
    device: torch.device,
    rcond: float,
    posterior_floor: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown coordinate transport schedule {schedule}")
    indices = transport.nested_step_indices(fine_steps, step_count)
    basis_rows = basis[:rank].to(device=device, dtype=torch.float32)
    coordinates = schedule_coordinates[:, :, :rank].cpu()
    permutation = permutation.long().cpu()

    final_residuals: list[torch.Tensor] = []
    final_posteriors: list[torch.Tensor] = []
    path_lengths: list[torch.Tensor] = []
    maximum_steps: list[torch.Tensor] = []
    minimum_ranks: list[torch.Tensor] = []
    maximum_conditions: list[torch.Tensor] = []
    incoming_errors: list[torch.Tensor] = []
    schedule_errors: list[torch.Tensor] = []
    endpoint_coordinate_errors: list[torch.Tensor] = []
    endpoint_errors: list[torch.Tensor] = []
    posterior_js: list[torch.Tensor] = []

    for start in range(0, len(start_residual), batch_size):
        stop = min(start + batch_size, len(start_residual))
        value = start_residual[start:stop].to(device)
        own_start = coordinates[0, start:stop].to(device)
        donor = permutation[start:stop]
        length = torch.zeros(stop - start, dtype=torch.float64, device=device)
        maximum = torch.zeros_like(length)
        minimum_rank = torch.full(
            (stop - start,), rank, dtype=torch.long, device=device
        )
        maximum_condition = torch.zeros_like(length)
        incoming = torch.zeros_like(length)
        previous_requested: torch.Tensor | None = None
        for index in indices[1:]:
            value = value.detach().requires_grad_(True)
            posterior = transport._posterior(model, value, answer_ids)
            ell = centered_log_posterior(posterior, posterior_floor)
            current = ell @ basis_rows.transpose(0, 1)
            jacobian = _coordinate_jacobian(value, current)
            if schedule == "actual_coordinate":
                requested = coordinates[index, start:stop].to(device)
            else:
                requested = own_start + (
                    coordinates[index, donor] - coordinates[0, donor]
                ).to(device)
            if previous_requested is not None:
                incoming += torch.linalg.vector_norm(
                    current.detach().double() - previous_requested, dim=1
                )
            delta, effective_rank, condition = _pseudoinverse_step(
                jacobian.detach(),
                requested - current.detach().double(),
                rcond,
            )
            step_norm = torch.linalg.vector_norm(delta, dim=1)
            length += step_norm
            maximum = torch.maximum(maximum, step_norm)
            minimum_rank = torch.minimum(minimum_rank, effective_rank)
            maximum_condition = torch.maximum(maximum_condition, condition)
            previous_requested = requested
            value = value.detach() + delta.to(value.dtype)

        with torch.no_grad():
            final_posterior = transport._posterior(model, value, answer_ids)
            final_coordinates = (
                centered_log_posterior(final_posterior, posterior_floor)
                @ basis_rows.transpose(0, 1)
            ).double()
        actual_endpoint_coordinates = coordinates[-1, start:stop].to(device)
        schedule_error = torch.linalg.vector_norm(
            final_coordinates - previous_requested, dim=1
        )
        endpoint_coordinate_error = torch.linalg.vector_norm(
            final_coordinates - actual_endpoint_coordinates, dim=1
        )
        actual_endpoint = endpoint_residual[start:stop].to(device)
        chord = torch.linalg.vector_norm(
            actual_endpoint - start_residual[start:stop].to(device), dim=1
        ).clamp_min(1e-12)
        endpoint_error = (
            torch.linalg.vector_norm(value - actual_endpoint, dim=1) / chord
        )
        js = transport.jensen_shannon_divergence(
            final_posterior.cpu(), endpoint_posterior[start:stop]
        )
        final_residuals.append(value.detach().cpu())
        final_posteriors.append(final_posterior.detach().cpu())
        path_lengths.append(length.detach().cpu())
        maximum_steps.append(maximum.detach().cpu())
        minimum_ranks.append(minimum_rank.detach().cpu())
        maximum_conditions.append(maximum_condition.detach().cpu())
        incoming_errors.append((incoming / max(step_count - 1, 1)).detach().cpu())
        schedule_errors.append(schedule_error.detach().cpu())
        endpoint_coordinate_errors.append(endpoint_coordinate_error.detach().cpu())
        endpoint_errors.append(endpoint_error.detach().cpu())
        posterior_js.append(js.detach().cpu())

    final_residual = torch.cat(final_residuals)
    final_posterior = torch.cat(final_posteriors)
    path_length = torch.cat(path_lengths)
    maximum_step = torch.cat(maximum_steps)
    minimum_rank = torch.cat(minimum_ranks)
    maximum_condition = torch.cat(maximum_conditions)
    incoming_error = torch.cat(incoming_errors)
    schedule_error = torch.cat(schedule_errors)
    endpoint_coordinate_error = torch.cat(endpoint_coordinate_errors)
    endpoint_error = torch.cat(endpoint_errors)
    js = torch.cat(posterior_js)
    metrics = {
        "task": transport._task_metrics(final_posterior, dataset),
        "causal_task": transport._causal_metrics(final_posterior, dataset),
        "mean_schedule_target_coordinate_error": float(schedule_error.mean()),
        "mean_endpoint_coordinate_error": float(endpoint_coordinate_error.mean()),
        "maximum_endpoint_coordinate_error": float(endpoint_coordinate_error.max()),
        "mean_incoming_coordinate_error": float(incoming_error.mean()),
        "mean_posterior_js_from_actual_endpoint": float(js.mean()),
        "mean_relative_residual_endpoint_error": float(endpoint_error.mean()),
        "maximum_relative_residual_endpoint_error": float(endpoint_error.max()),
        "mean_rollout_path_length": float(path_length.mean()),
        "maximum_step_norm": float(maximum_step.max()),
        "minimum_effective_rank": int(minimum_rank.min()),
        "mean_minimum_effective_rank": float(minimum_rank.double().mean()),
        "maximum_condition_number": float(maximum_condition.max()),
        "finite_contract": bool(
            torch.isfinite(final_residual).all()
            and torch.isfinite(final_posterior).all()
            and torch.isfinite(path_length).all()
            and torch.isfinite(maximum_condition).all()
            and torch.isfinite(endpoint_coordinate_error).all()
        ),
    }
    arrays = {
        "predicted_bin": final_posterior.argmax(1).numpy(),
        "path_length": path_length.numpy(),
        "maximum_step_norm": maximum_step.numpy(),
        "minimum_effective_rank": minimum_rank.numpy(),
        "maximum_condition_number": maximum_condition.numpy(),
        "mean_incoming_coordinate_error": incoming_error.numpy(),
        "schedule_target_coordinate_error": schedule_error.numpy(),
        "endpoint_coordinate_error": endpoint_coordinate_error.numpy(),
        "relative_residual_endpoint_error": endpoint_error.numpy(),
        "posterior_js_from_actual_endpoint": js.numpy(),
    }
    return metrics, arrays


def _fingerprint(
    config: PosteriorCoordinateTransportConfig,
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
        raise ValueError(f"incompatible completed coordinate-transport result {path}")
    return value


def _campaign_reusable(
    value: Mapping[str, Any],
    config: PosteriorCoordinateTransportConfig,
    implementation: str,
) -> bool:
    try:
        entries = list(value.get("results", []))
        result_paths = [Path(entry["path"]) for entry in entries]
        indexed = {(entry["condition"], int(entry["seed"])) for entry in entries}
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
        and _manifest_sha256(result_paths) == value.get("result_manifest_sha256")
        and all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry.get("result_sha256")
            and Path(entry["arrays_path"]).is_file()
            and _sha256(Path(entry["arrays_path"])) == entry.get("arrays_sha256")
            for entry in value.get("results", [])
        )
    )


def run_campaign(
    config: PosteriorCoordinateTransportConfig, output: Path
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
        raise ValueError(
            f"incompatible completed coordinate-transport campaign {campaign_path}"
        )

    source, source_path, source_entries = _load_source_campaign(config)
    transport_config = _transport_config_from_source(source)
    acquisition_source, _, _ = transport._load_source_campaign(transport_config)
    acquisition_config = transport._task_config_from_source(acquisition_source)
    _, _, _, task, source_frontend_config = acquisition._load_sources(
        acquisition_config
    )
    if len(task.answer_token_ids) != ANSWER_BINS:
        raise ValueError("the locked answer-bin count does not match the task")
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
    dataset_hashes = {
        name: transport._dataset_hash(value) for name, value in datasets.items()
    }
    if any(dataset_hashes[name] != source["dataset_hashes"][name] for name in REGIMES):
        raise ValueError("coordinate-transport datasets do not replay the source")

    repeat_path = Path(acquisition_source["artifacts"]["repeat_arrays"])
    reference_paths: dict[str, torch.Tensor] = {}
    path_sharing_errors: dict[str, float] = {}
    path_endpoint_errors: dict[str, float] = {}
    path_unit_errors: dict[str, float] = {}
    with np.load(repeat_path, allow_pickle=False) as archive:
        for regime, dataset in datasets.items():
            stored_fibers = torch.from_numpy(archive[f"{regime}__fiber_id"])
            noise = torch.from_numpy(archive[f"{regime}__repeat_noise"])
            if not torch.equal(
                stored_fibers.long(), dataset.paired.fiber.fiber_id.long()
            ):
                raise ValueError(f"stored fiber IDs changed for {regime}")
            observations = acquisition.repeated_observations(
                dataset.calibration, noise, acquisition_config.sigma_radians
            )
            q1 = acquisition.analytic_circular_mean(observations, 1)
            q64 = acquisition.analytic_circular_mean(observations, 64)
            reference_paths[regime] = transport.shortest_reference_path(
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
                reference_paths[regime] * reference_paths[regime][0:1].conj()
            )
            maximum = 0.0
            for fiber in torch.unique(dataset.paired.fiber.fiber_id, sorted=True):
                selected = increments[:, dataset.paired.fiber.fiber_id == fiber]
                maximum = max(maximum, float((selected - selected[:, :1]).abs().max()))
            path_sharing_errors[regime] = maximum
    path_sharing_contract = all(
        value <= transport_config.path_sharing_tolerance
        for value in path_sharing_errors.values()
    )
    path_construction_contract = bool(
        all(
            value <= transport_config.path_sharing_tolerance
            for value in path_endpoint_errors.values()
        )
        and all(
            value <= transport_config.path_sharing_tolerance
            for value in path_unit_errors.values()
        )
    )
    nested_grid_contract = all(
        transport.nested_step_indices(config.fine_steps, count)[0] == 0
        and transport.nested_step_indices(config.fine_steps, count)[-1]
        == config.fine_steps
        and len(transport.nested_step_indices(config.fine_steps, count)) == count + 1
        for count in config.step_counts
    )

    basis = dct_coordinate_basis(ANSWER_BINS)
    identity_error = float(
        (basis @ basis.transpose(0, 1) - torch.eye(FULL_RANK, dtype=torch.float64))
        .abs()
        .max()
    )
    constant_leak = float(basis.sum(1).abs().max()) / math.sqrt(ANSWER_BINS)
    basis_orthonormal_contract = bool(
        identity_error <= config.orthonormal_tolerance
        and constant_leak <= config.orthonormal_tolerance
    )
    basis_nested_contract = all(
        torch.equal(basis[:rank], basis[: RANKS[-1]][:rank]) for rank in config.ranks
    )

    results: list[dict[str, Any]] = []
    reused = 0
    source_model_root = Path(acquisition_config.source_root)
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
                    raise ValueError(
                        f"source provenance mismatch {condition} seed {seed}: {name}"
                    )
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
            shuffle_pairing_contract = True
            finite = True

            for regime_index, regime in enumerate(REGIMES):
                dataset = datasets[regime]
                source_regime = source_detail["regimes"][regime]
                clean_residual, clean_posterior = transport._forward_packet(
                    system,
                    dataset,
                    dataset.calibration,
                    task,
                    config.activation_batch_size,
                    device,
                )
                actual_residuals, actual_posteriors = transport._extract_actual_path(
                    system,
                    dataset,
                    reference_paths[regime],
                    task,
                    config.activation_batch_size,
                    device,
                )
                actual_moments = (actual_posteriors.to(device) @ centers).cpu()
                clean_metrics = transport._task_metrics(clean_posterior, dataset)
                actual_curve = [
                    transport._task_metrics(actual_posteriors[index], dataset)
                    for index in range(config.fine_steps + 1)
                ]
                maximum_replay_error = max(
                    maximum_replay_error,
                    transport._metric_error(
                        clean_metrics, source_regime["clean"], TASK_METRIC_NAMES
                    ),
                    max(
                        transport._metric_error(
                            actual_curve[index],
                            source_regime["actual_reference_path"]["task_by_step"][
                                index
                            ],
                            TASK_METRIC_NAMES,
                        )
                        for index in range(config.fine_steps + 1)
                    ),
                )

                endpoint_replay_parts: list[torch.Tensor] = []
                with torch.no_grad():
                    for start in range(
                        0, len(dataset.calibration), config.activation_batch_size
                    ):
                        stop = min(
                            start + config.activation_batch_size,
                            len(dataset.calibration),
                        )
                        endpoint_replay_parts.append(
                            transport._posterior(
                                system.model,
                                actual_residuals[-1, start:stop].to(device),
                                answer_ids,
                            ).cpu()
                        )
                endpoint_replay_posterior = torch.cat(endpoint_replay_parts)
                endpoint_replay_metrics = transport._task_metrics(
                    endpoint_replay_posterior, dataset
                )
                endpoint_replay_error = max(
                    transport._metric_error(
                        endpoint_replay_metrics, actual_curve[-1], TASK_METRIC_NAMES
                    ),
                    transport._metric_error(
                        endpoint_replay_metrics,
                        source_regime["actual_reference_path"]["endpoint_replay"],
                        TASK_METRIC_NAMES,
                    ),
                )
                maximum_replay_error = max(maximum_replay_error, endpoint_replay_error)

                shuffle_shift = _shuffle_seed(
                    condition, seed, regime, config.analysis_seed + regime_index
                )
                shuffle_index = transport.fiber_block_permutation(
                    dataset.paired.fiber.fiber_id,
                    dataset.paired.fiber.branch,
                    shift=shuffle_shift,
                )
                shuffle_pairing_contract = bool(
                    shuffle_pairing_contract
                    and torch.equal(
                        dataset.paired.fiber.branch[shuffle_index],
                        dataset.paired.fiber.branch,
                    )
                    and not bool(
                        (shuffle_index == torch.arange(len(shuffle_index))).any()
                    )
                )

                comparator_records: dict[str, Any] = {}
                for count in COMPARATOR_STEP_COUNTS:
                    record, _ = transport._rollout(
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
                        schedule="path_moment",
                        permutation=shuffle_index,
                        batch_size=config.activation_batch_size,
                        device=device,
                        squared_floor=transport_config.gradient_squared_floor,
                    )
                    comparator_records[str(count)] = record
                    maximum_replay_error = max(
                        maximum_replay_error,
                        transport._metric_error(
                            record["task"],
                            source_regime["rollouts"]["path_moment"][str(count)][
                                "task"
                            ],
                            TASK_METRIC_NAMES,
                        ),
                    )
                    finite = bool(finite and record["finite_contract"])

                schedule_coordinates = coordinate_schedule(
                    actual_posteriors, basis, config.posterior_floor
                )
                rollouts: dict[str, dict[str, Any]] = {
                    schedule: {rank_label(rank): {} for rank in config.ranks}
                    for schedule in SCHEDULES
                }
                for schedule in SCHEDULES:
                    for rank in config.ranks:
                        for count in config.step_counts:
                            record, arrays = _vector_rollout(
                                model=system.model,
                                answer_ids=answer_ids,
                                basis=basis,
                                rank=rank,
                                start_residual=actual_residuals[0],
                                endpoint_residual=actual_residuals[-1],
                                endpoint_posterior=actual_posteriors[-1],
                                schedule_coordinates=schedule_coordinates,
                                dataset=dataset,
                                step_count=count,
                                fine_steps=config.fine_steps,
                                schedule=schedule,
                                permutation=shuffle_index,
                                batch_size=config.activation_batch_size,
                                device=device,
                                rcond=config.pinv_rcond,
                                posterior_floor=config.posterior_floor,
                            )
                            rollouts[schedule][rank_label(rank)][str(count)] = record
                            finite = bool(finite and record["finite_contract"])
                            prefix = (
                                f"{regime}__{schedule}__r{rank_label(rank)}__k{count}"
                            )
                            for name, value in arrays.items():
                                diagnostic_arrays[f"{prefix}__{name}"] = value

                regime_records[regime] = {
                    "clean": clean_metrics,
                    "actual_reference_path": {
                        "start": actual_curve[0],
                        "endpoint": actual_curve[-1],
                        "endpoint_replay": endpoint_replay_metrics,
                        "maximum_endpoint_replay_error": endpoint_replay_error,
                    },
                    "scalar_comparator": comparator_records,
                    "rollouts": rollouts,
                    "shuffle": {
                        "seed": shuffle_shift,
                        "fixed_point_count": int(
                            (shuffle_index == torch.arange(len(shuffle_index))).sum()
                        ),
                    },
                }
                diagnostic_arrays[f"{regime}__actual_coordinates_start"] = (
                    schedule_coordinates[0].numpy()
                )
                diagnostic_arrays[f"{regime}__actual_coordinates_end"] = (
                    schedule_coordinates[-1].numpy()
                )
                diagnostic_arrays[f"{regime}__shuffle_index"] = shuffle_index.numpy()

            clean_accuracy = {
                regime: regime_records[regime]["clean"]["exact_bin_accuracy"]
                for regime in REGIMES
            }
            actual_records = {
                regime: regime_records[regime]["actual_reference_path"]["endpoint"]
                for regime in REGIMES
            }
            endpoint_records = {
                regime: regime_records[regime]["actual_reference_path"][
                    "endpoint_replay"
                ]
                for regime in REGIMES
            }
            actual_gate, actual_losses = transport.task_gate(
                clean_accuracy, actual_records, config.accuracy_loss_ceiling
            )
            endpoint_gate, endpoint_losses = transport.task_gate(
                clean_accuracy, endpoint_records, config.accuracy_loss_ceiling
            )
            comparator_gates: dict[str, Any] = {}
            for count in COMPARATOR_STEP_COUNTS:
                records = {
                    regime: regime_records[regime]["scalar_comparator"][str(count)][
                        "task"
                    ]
                    for regime in REGIMES
                }
                passed, losses = transport.task_gate(
                    clean_accuracy, records, config.accuracy_loss_ceiling
                )
                comparator_gates[str(count)] = {
                    "task_gate": passed,
                    "accuracy_loss_from_clean": losses,
                }
            rollout_gates: dict[str, dict[str, Any]] = {
                schedule: {rank_label(rank): {} for rank in config.ranks}
                for schedule in SCHEDULES
            }
            for schedule in SCHEDULES:
                for rank in config.ranks:
                    for count in config.step_counts:
                        records = {
                            regime: regime_records[regime]["rollouts"][schedule][
                                rank_label(rank)
                            ][str(count)]["task"]
                            for regime in REGIMES
                        }
                        passed, losses = transport.task_gate(
                            clean_accuracy, records, config.accuracy_loss_ceiling
                        )
                        rollout_gates[schedule][rank_label(rank)][str(count)] = {
                            "task_gate": passed,
                            "accuracy_loss_from_clean": losses,
                        }

            state_unchanged = calibrated._module_digest(system) == initial_state
            replay_contract = maximum_replay_error <= config.replay_tolerance
            finite = bool(
                finite
                and _finite_numbers(regime_records)
                and all(
                    np.isfinite(value).all() for value in diagnostic_arrays.values()
                )
            )
            systems_valid = bool(
                path_sharing_contract
                and path_construction_contract
                and nested_grid_contract
                and basis_orthonormal_contract
                and basis_nested_contract
                and shuffle_pairing_contract
                and replay_contract
                and finite
                and state_unchanged
            )
            arrays_path = result_dir / "coordinate_transport_diagnostics.npz"
            _write_npz(arrays_path, diagnostic_arrays)
            arrays_sha256 = _sha256(arrays_path)
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": (
                    f"tinyllm-posterior-coordinate-transport-{condition}-seed{seed}"
                ),
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
                    "source_transport_result": source_entry["path"],
                    "source_transport_result_sha256": source_entry["result_sha256"],
                    "source_transport_scientific_fingerprint": source_detail[
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
                    "scalar_comparator": comparator_gates,
                    "rollouts": rollout_gates,
                },
                "maximum_source_metric_replay_error": maximum_replay_error,
                "gates": {
                    "source_metric_replay_contract": replay_contract,
                    "exact_fiber_path_sharing_contract": path_sharing_contract,
                    "reference_path_construction_contract": path_construction_contract,
                    "nested_step_grid_contract": nested_grid_contract,
                    "coordinate_basis_orthonormal_contract": (
                        basis_orthonormal_contract
                    ),
                    "coordinate_basis_nested_contract": basis_nested_contract,
                    "shuffle_pairing_contract": shuffle_pairing_contract,
                    "finite_numerical_contract": finite,
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
                raise RuntimeError(
                    "coordinate-transport implementation changed during campaign"
                )
            print(f"completed {condition} seed {seed}", flush=True)
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()

    required = (
        len(config.seeds) if config.allow_underpowered else config.required_seed_passes
    )
    primary_key = str(PRIMARY_STEP_COUNT)
    rollout_counts = {
        schedule: {
            rank_label(rank): {
                str(count): {
                    condition: sum(
                        int(
                            result["transport_gates"]["rollouts"][schedule][
                                rank_label(rank)
                            ][str(count)]["task_gate"]
                        )
                        for result in results
                        if result["condition"] == condition
                    )
                    for condition in CONDITIONS
                }
                for count in config.step_counts
            }
            for rank in config.ranks
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
    comparator_counts = {
        str(count): {
            condition: sum(
                int(
                    result["transport_gates"]["scalar_comparator"][str(count)][
                        "task_gate"
                    ]
                )
                for result in results
                if result["condition"] == condition
            )
            for condition in CONDITIONS
        }
        for count in COMPARATOR_STEP_COUNTS
    }
    positive_control = all(
        actual_counts[condition] == len(config.seeds)
        and endpoint_counts[condition] == len(config.seeds)
        for condition in CONDITIONS
    )
    shuffled_valid = all(
        rollout_counts["shuffled_coordinate"][rank_label(rank)][primary_key][condition]
        <= config.control_pass_ceiling
        for rank in config.ranks
        for condition in CONDITIONS
    )
    rank_gates = {
        rank_label(rank): bool(
            all(
                rollout_counts["actual_coordinate"][rank_label(rank)][primary_key][
                    condition
                ]
                >= required
                for condition in CONDITIONS
            )
            and all(
                rollout_counts["shuffled_coordinate"][rank_label(rank)][primary_key][
                    condition
                ]
                <= config.control_pass_ceiling
                for condition in CONDITIONS
            )
        )
        for rank in config.ranks
    }
    smallest_passing_rank = next(
        (rank_label(rank) for rank in config.ranks if rank_gates[rank_label(rank)]),
        None,
    )
    systems_valid = bool(
        path_sharing_contract
        and basis_orthonormal_contract
        and basis_nested_contract
        and all(result["gates"]["validity_contract"] for result in results)
    )
    scientific_valid = bool(systems_valid and positive_control)
    valid = systems_valid if config.allow_underpowered else scientific_valid
    classification = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else classify_campaign(
            valid=valid,
            shuffled_valid=shuffled_valid,
            rank_gates=rank_gates,
            ranks=config.ranks,
        )
    )
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
        "coordinate_contracts": {
            "answer_bins": ANSWER_BINS,
            "full_rank": FULL_RANK,
            "basis_identity_error": identity_error,
            "basis_constant_leak": constant_leak,
            "coordinate_basis_orthonormal_contract": basis_orthonormal_contract,
            "coordinate_basis_nested_contract": basis_nested_contract,
        },
        "aggregates": {
            "valid": valid,
            "systems_valid": systems_valid,
            "scientific_controls_valid": scientific_valid,
            "shuffled_control_contract": shuffled_valid,
            "classification": classification,
            "required_seed_passes": required,
            "rank_gates": rank_gates,
            "smallest_passing_rank": smallest_passing_rank,
            "rollout_pass_counts": rollout_counts,
            "actual_m64_reference_pass_counts": actual_counts,
            "exact_endpoint_residual_pass_counts": endpoint_counts,
            "scalar_comparator_pass_counts": comparator_counts,
            "positive_control_contract": positive_control,
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
            "Coordinate schedules read frozen model outputs along the stored path and are mechanistic oracles, not deployable sensors.",
            "The DCT basis is example-free but the transported chart remains conditioned on the frozen answer decoder.",
            "Only final-query residual transport is tested; earlier block geometry is not measured.",
            "The reference path comes from one stored unbiased synthetic Gaussian acquisition array.",
            "No topology, link, or cobordism invariant is computed.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "*" / "seed_*" / "result.json"),
            "diagnostic_arrays": str(
                output / "runs" / "*" / "seed_*" / "coordinate_transport_diagnostics.npz"
            ),
        },
    }
    if _implementation_digest() != implementation:
        raise RuntimeError(
            "coordinate-transport implementation changed before campaign write"
        )
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
            "data/experiments/tinyllm_posterior_coordinate_transport/"
            "20260807_d8_preregistered"
        ),
    )
    args = parser.parse_args()
    seeds = _ints(args.seeds)
    if args.shakedown:
        seeds = (7,)
        allow_underpowered = True
    else:
        allow_underpowered = args.allow_underpowered
    config = PosteriorCoordinateTransportConfig(
        seeds=seeds,
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
