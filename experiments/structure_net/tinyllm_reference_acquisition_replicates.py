#!/usr/bin/env python3
"""Test repeated orientation-reference acquisition in frozen TinyLLM systems."""

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
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-reference-acquisition-replicates.v1"
HYPOTHESIS_ID = "tinyllm-reference-acquisition-replicates-v1"
SOURCE_ORIENTATION_SCHEMA = orientation.SCHEMA_VERSION
SOURCE_ORIENTATION_HYPOTHESIS = orientation.HYPOTHESIS_ID
SOURCE_ORIENTATION_CAMPAIGN_SHA256 = (
    "876af062f9d0bdd365e2f1ffbb959d9ff2e5e4e277321b510bbe6a956456877f"
)
SOURCE_ORIENTATION_IMPLEMENTATION_SHA256 = (
    "990975365c7f0c468c3895029c1a87a98fa14d5a28853a1fc445070769218c70"
)
SOURCE_ORIENTATION_NOISE_SHA256 = (
    "b8d959169f2f249d635037a362c70522e514ace13d3bf656a91bdaea99ef43e7"
)
SOURCE_CALIBRATED_CAMPAIGN_SHA256 = (
    "80623a7283fed3d902a1cbf9fb58afe2a2fef82ab86406212f907de40a96c501"
)
SOURCE_CALIBRATED_IMPLEMENTATION_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
EVIDENCE_ROLE = "preregistered_frozen_checkpoint_causal_acquisition_intervention"
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
COUNTS = (1, 4, 16, 64)
METHODS = ("analytic_circular_mean", "learned_equivariant_moment")
REGIMES = ("in_distribution", "composition", "extrapolation")
PRIMARY_REGIMES = ("composition", "extrapolation")
SIGMA_RADIANS = 0.175
RIDGE_RATIO = 0.01
REPEAT_NOISE_SEED_OFFSET = 9_400_019
CONTROL_SEED_OFFSET = 9_700_019
SOURCE_LEVEL_INDEX = orientation.LOCKED_SIGMAS.index(SIGMA_RADIANS)


@dataclass(frozen=True)
class ReferenceAcquisitionConfig:
    source_root: str = (
        "data/experiments/tinyllm_calibrated_frontend_causal/"
        "20260806_d8_preregistered"
    )
    source_orientation_root: str = (
        "data/experiments/tinyllm_calibration_orientation_noise/"
        "20260807_d8_preregistered"
    )
    seeds: tuple[int, ...] = PRIMARY_SEEDS
    repeat_counts: tuple[int, ...] = COUNTS
    sigma_radians: float = SIGMA_RADIANS
    ridge_ratio: float = RIDGE_RATIO
    analysis_seed: int = 23_711
    activation_batch_size: int = 256
    accuracy_loss_ceiling: float = 0.03
    source_replay_tolerance: float = 2e-6
    scaling_slope_minimum: float = -0.60
    scaling_slope_maximum: float = -0.40
    learned_rmse_gap_ceiling: float = 0.002
    required_seed_passes: int = 4
    control_pass_ceiling: int = 1
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if (
            not self.repeat_counts
            or self.repeat_counts[0] != 1
            or tuple(sorted(set(self.repeat_counts))) != self.repeat_counts
        ):
            raise ValueError("repeat counts must be distinct, ordered, and start at one")
        if any(value < 1 or value > 64 for value in self.repeat_counts):
            raise ValueError("repeat counts must lie between one and 64")
        if self.required_seed_passes > len(self.seeds):
            raise ValueError("required seed passes exceed selected checkpoints")
        if not self.allow_underpowered:
            if self.seeds != PRIMARY_SEEDS or self.repeat_counts != COUNTS:
                raise ValueError("primary seeds and repeat-count grid are fixed")
            if (
                self.sigma_radians != SIGMA_RADIANS
                or self.ridge_ratio != RIDGE_RATIO
                or self.analysis_seed != 23_711
            ):
                raise ValueError("primary acquisition protocol is fixed")
            if self.required_seed_passes != 4 or self.control_pass_ceiling != 1:
                raise ValueError("primary population and control gates are fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
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


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
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


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "calibrated_frontend": Path(calibrated.__file__),
        "orientation_noise": Path(orientation.__file__),
        "invariant_frontend": Path(invariant.__file__),
        "tinyllm_model": Path(inspect.getfile(TinyLLMModel)),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(digests: Mapping[str, str] | None = None) -> str:
    material = dict(digests or _source_digests())
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_role(config: ReferenceAcquisitionConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else EVIDENCE_ROLE
    )


def _dataset_hash(dataset: calibrated.CalibratedDataset) -> str:
    return orientation._dataset_hash(dataset)


def _load_sources(
    config: ReferenceAcquisitionConfig,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    CircleTaskConfig,
    calibrated.CalibratedFrontendConfig,
]:
    orientation_path = Path(config.source_orientation_root) / "campaign_results.json"
    orientation_campaign = json.loads(orientation_path.read_text(encoding="utf-8"))
    noise_path = Path(orientation_campaign.get("artifacts", {}).get("noise_arrays", ""))
    if (
        _sha256(orientation_path) != SOURCE_ORIENTATION_CAMPAIGN_SHA256
        or orientation_campaign.get("schema_version") != SOURCE_ORIENTATION_SCHEMA
        or orientation_campaign.get("hypothesis_id")
        != SOURCE_ORIENTATION_HYPOTHESIS
        or orientation_campaign.get("status") != "completed"
        or orientation_campaign.get("implementation_sha256")
        != SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
        or orientation_campaign.get("provenance", {}).get("source_campaign_sha256")
        != SOURCE_CALIBRATED_CAMPAIGN_SHA256
        or orientation_campaign.get("provenance", {}).get(
            "source_implementation_sha256"
        )
        != SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
        or int(orientation_campaign.get("summary", {}).get("completed", -1)) != 10
        or not noise_path.is_file()
        or _sha256(noise_path) != SOURCE_ORIENTATION_NOISE_SHA256
        or orientation_campaign.get("artifacts", {}).get("noise_arrays_sha256")
        != SOURCE_ORIENTATION_NOISE_SHA256
    ):
        raise ValueError(f"invalid orientation source campaign {orientation_path}")

    source_config = orientation.OrientationNoiseConfig(
        source_root=config.source_root,
        seeds=config.seeds,
        allow_underpowered=config.allow_underpowered,
    )
    calibrated_campaign, calibrated_path, task, calibrated_config = (
        orientation._load_source_campaign(source_config)
    )
    if _sha256(calibrated_path) != SOURCE_CALIBRATED_CAMPAIGN_SHA256:
        raise ValueError("calibrated source campaign digest changed")
    return (
        orientation_campaign,
        orientation_path,
        calibrated_campaign,
        task,
        calibrated_config,
    )


def build_repeat_noise(
    dataset: calibrated.CalibratedDataset,
    source_base_noise: torch.Tensor,
    *,
    split_seed: int,
    maximum_repeats: int,
) -> torch.Tensor:
    """Build nested pair-shared repeats while preserving the source first draw."""

    fiber_ids = dataset.paired.fiber.fiber_id.long()
    unique, inverse = torch.unique(fiber_ids, sorted=True, return_inverse=True)
    if len(source_base_noise) != len(fiber_ids):
        raise ValueError("source orientation noise length differs from dataset")
    first_indices = torch.stack(
        [torch.nonzero(fiber_ids == fiber, as_tuple=False)[0, 0] for fiber in unique]
    )
    first = source_base_noise.double()[first_indices, None]
    if maximum_repeats == 1:
        unique_noise = first
    else:
        generator = np.random.default_rng(REPEAT_NOISE_SEED_OFFSET + split_seed)
        additional = torch.from_numpy(
            generator.standard_normal((len(unique), maximum_repeats - 1))
        ).double()
        unique_noise = torch.cat((first, additional), dim=1)
    expanded = unique_noise[inverse]
    if not torch.equal(expanded[:, 0], source_base_noise.double()):
        raise RuntimeError("first repeated-reference draw does not replay source")
    return expanded


def _ensure_repeat_arrays(
    path: Path,
    datasets: Mapping[str, calibrated.CalibratedDataset],
    noises: Mapping[str, torch.Tensor],
) -> str:
    arrays: dict[str, np.ndarray] = {}
    for name, dataset in datasets.items():
        arrays[f"{name}__fiber_id"] = dataset.paired.fiber.fiber_id.numpy()
        arrays[f"{name}__repeat_noise"] = noises[name].numpy()
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(archive[name], value)
                for name, value in arrays.items()
            ):
                raise ValueError(f"incompatible repeated-reference arrays {path}")
    else:
        _write_npz(path, arrays)
    return _sha256(path)


def repeated_observations(
    calibration: torch.Tensor,
    repeat_noise: torch.Tensor,
    sigma_radians: float,
) -> torch.Tensor:
    if len(calibration) != len(repeat_noise):
        raise ValueError("calibration and repeated-reference noise lengths differ")
    angle = torch.atan2(calibration[:, 1].double(), calibration[:, 0].double())
    observed_angles = angle[:, None] + float(sigma_radians) * repeat_noise.double()
    return torch.polar(torch.ones_like(observed_angles), observed_angles)


def charge_one_moment_features(observations: torch.Tensor, count: int) -> torch.Tensor:
    selected = observations[:, :count]
    first = selected.mean(1)
    second = selected.square().mean(1)
    third = selected.pow(3).mean(1)
    return torch.stack(
        (first, second * first.conj(), third * first.conj().square()), dim=1
    )


def fit_equivariant_moment_denoiser(
    observations: torch.Tensor,
    clean_orientation: torch.Tensor,
    count: int,
    ridge_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = charge_one_moment_features(observations, count)
    target = torch.complex(
        clean_orientation[:, 0].double(), clean_orientation[:, 1].double()
    )
    design = torch.cat((features.real, features.imag), dim=0)
    response = torch.cat((target.real, target.imag), dim=0)
    gram = design.T @ design
    ridge = float(ridge_ratio) * float(torch.trace(gram)) / features.shape[1]
    prior = torch.tensor((1.0, 0.0, 0.0), dtype=torch.double)
    coefficients = torch.linalg.solve(
        gram + ridge * torch.eye(features.shape[1], dtype=torch.double),
        design.T @ response + ridge * prior,
    )
    prediction = features @ coefficients.to(features.dtype)
    fit_rmse = torch.sqrt((prediction - target).abs().square().mean())
    return coefficients, {
        "ridge_ratio": float(ridge_ratio),
        "ridge_penalty": ridge,
        "fit_vector_rmse": float(fit_rmse),
        "parameter_count": int(len(coefficients)),
    }


def analytic_circular_mean(observations: torch.Tensor, count: int) -> torch.Tensor:
    value = observations[:, :count].mean(1)
    return value / value.abs().clamp_min(1e-12)


def apply_equivariant_moment_denoiser(
    observations: torch.Tensor, count: int, coefficients: torch.Tensor
) -> torch.Tensor:
    features = charge_one_moment_features(observations, count)
    value = features @ coefficients.double().to(features.dtype)
    return value / value.abs().clamp_min(1e-12)


def misgrouped_circular_mean(
    observations: torch.Tensor, fiber_ids: torch.Tensor, count: int
) -> torch.Tensor:
    """Destroy same-reference membership while preserving repeat marginals."""

    unique, inverse = torch.unique(fiber_ids.long(), sorted=True, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("misgrouped control requires at least two fibers")
    first_indices = torch.stack(
        [torch.nonzero(fiber_ids == fiber, as_tuple=False)[0, 0] for fiber in unique]
    )
    grouped = observations[first_indices, :count].clone()
    for repeat in range(1, count):
        grouped[:, repeat] = torch.roll(
            grouped[:, repeat], shifts=1 + (repeat % (len(unique) - 1)), dims=0
        )
    expanded = grouped[inverse]
    return analytic_circular_mean(expanded, count)


def packet_from_orientation(
    calibration: torch.Tensor, orientation_vector: torch.Tensor
) -> torch.Tensor:
    value = calibration.clone()
    value[:, 0] = orientation_vector.real.to(value.dtype)
    value[:, 1] = orientation_vector.imag.to(value.dtype)
    return value


def orientation_audit(
    estimated: torch.Tensor, calibration: torch.Tensor, count: int, sigma: float
) -> dict[str, float]:
    target = torch.complex(calibration[:, 0].double(), calibration[:, 1].double())
    delta = torch.angle(estimated * target.conj())
    return {
        "repeat_count": int(count),
        "predicted_standard_error_radians": float(sigma / math.sqrt(count)),
        "angular_rmse_radians": float(torch.sqrt(delta.square().mean())),
        "angular_mae_radians": float(delta.abs().mean()),
        "maximum_absolute_error_radians": float(delta.abs().max()),
        "maximum_unit_norm_error": float((estimated.abs() - 1.0).abs().max()),
    }


def scaling_slope(counts: Sequence[int], rmses: Sequence[float]) -> float:
    if len(counts) < 2 or len(counts) != len(rmses):
        raise ValueError("scaling slope requires matched multi-count inputs")
    return float(np.polyfit(np.log(np.asarray(counts)), np.log(np.asarray(rmses)), 1)[0])


def frontend_metrics(
    features: np.ndarray, dataset: calibrated.CalibratedDataset
) -> dict[str, float]:
    predicted = np.asarray(features, dtype=np.float64).reshape(-1)
    target = dataset.paired.fiber.cosine.double().numpy()
    correlation = 0.0
    if np.ptp(predicted) > 1e-12 and np.ptp(target) > 1e-12:
        correlation = float(np.corrcoef(predicted, target)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    error = predicted - target
    return {
        "cosine_pearson": correlation,
        "cosine_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "cosine_mae": float(np.mean(np.abs(error))),
    }


def task_gate(
    clean_accuracy: Mapping[str, float],
    evaluations: Mapping[str, Mapping[str, Any]],
    ceiling: float,
) -> tuple[bool, dict[str, float]]:
    losses = {
        regime: float(clean_accuracy[regime])
        - float(evaluations[regime]["task"]["exact_bin_accuracy"])
        for regime in PRIMARY_REGIMES
    }
    return all(value <= ceiling for value in losses.values()), losses


def _posterior_metrics(
    posterior: torch.Tensor, dataset: calibrated.CalibratedDataset
) -> dict[str, float]:
    target = dataset.paired.circle.target_posteriors.double()
    prediction = posterior.argmax(1)
    target_bin = target.argmax(1)
    return {
        "exact_bin_accuracy": float((prediction == target_bin).double().mean()),
        "mean_target_cross_entropy": float(
            -(target * posterior.double().clamp_min(1e-12).log()).sum(1).mean()
        ),
    }


def causal_coordinate_write(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    batch_size: int,
    device: torch.device,
    control_seed: int,
) -> dict[str, Any]:
    """Patch the frozen residual along its local ordered-cosine task gradient."""

    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    centers = torch.linspace(-1.0, 1.0, len(task.answer_token_ids), device=device)
    target_coordinate = dataset.paired.fiber.cosine.float()
    generator = torch.Generator(device="cpu").manual_seed(control_seed)
    permutation = torch.randperm(len(target_coordinate), generator=generator)
    shuffled_coordinate = target_coordinate[permutation]
    posteriors: dict[str, list[torch.Tensor]] = {
        "target_oracle": [],
        "target_shuffled": [],
    }
    patch_norms: dict[str, list[torch.Tensor]] = {
        "target_oracle": [],
        "target_shuffled": [],
    }
    finite = True
    for start in range(0, len(sensor), batch_size):
        stop = min(start + batch_size, len(sensor))
        residual = system.forward_cuts(
            dataset.paired.circle.input_ids[start:stop].to(device),
            sensor[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )["full"].detach().requires_grad_(True)
        logits = invariant._task_logits(system.model, residual, answer_ids)
        posterior = torch.softmax(logits, -1)
        moment = posterior @ centers
        gradient = torch.autograd.grad(moment.sum(), residual, create_graph=False)[0]
        gradient_sq = gradient.square().sum(1).clamp_min(1e-8)
        coordinates = {
            "target_oracle": target_coordinate[start:stop].to(device),
            "target_shuffled": shuffled_coordinate[start:stop].to(device),
        }
        for name, coordinate in coordinates.items():
            alpha = (coordinate - moment.detach()) / gradient_sq
            delta = alpha[:, None] * gradient
            patched_logits = invariant._task_logits(
                system.model, residual.detach() + delta.detach(), answer_ids
            )
            posteriors[name].append(torch.softmax(patched_logits, -1).detach().cpu())
            patch_norms[name].append(delta.detach().norm(dim=1).cpu())
            finite = bool(finite and torch.isfinite(delta).all())
    return {
        name: {
            **_posterior_metrics(torch.cat(parts), dataset),
            "mean_patch_norm": float(torch.cat(patch_norms[name]).mean()),
        }
        for name, parts in posteriors.items()
    } | {"finite_contract": finite}


def classify_campaign(
    *,
    valid: bool,
    analytic_pass: Mapping[str, int],
    learned_pass: Mapping[str, int],
    required: int,
) -> str:
    if not valid:
        return "invalid"
    analytic_both = all(int(analytic_pass[condition]) >= required for condition in CONDITIONS)
    learned_both = all(int(learned_pass[condition]) >= required for condition in CONDITIONS)
    if analytic_both and learned_both:
        return "acquisition_precision_sufficient"
    if analytic_both:
        return "analytic_reference_averaging_sufficient"
    if learned_both:
        return "learned_denoising_required"
    any_condition_passes = any(
        int(analytic_pass[condition]) >= required
        or int(learned_pass[condition]) >= required
        for condition in CONDITIONS
    )
    return (
        "arm_stratified_acquisition_repair"
        if any_condition_passes
        else "reference_averaging_insufficient"
    )


def _source_result(
    campaign: Mapping[str, Any], condition: str, seed: int
) -> tuple[dict[str, Any], Path, str]:
    entries = [
        entry
        for entry in campaign.get("results", [])
        if entry.get("condition") == condition and int(entry.get("seed", -1)) == seed
    ]
    if len(entries) != 1:
        raise ValueError(f"missing orientation source result {condition} seed {seed}")
    entry = entries[0]
    path = Path(entry["path"])
    if not path.is_file() or _sha256(path) != entry.get("result_sha256"):
        raise ValueError(f"invalid orientation source result {path}")
    detail = json.loads(path.read_text(encoding="utf-8"))
    return detail, path, str(entry["result_sha256"])


def _source_replay_error(
    source: Mapping[str, Any],
    clean: Mapping[str, Mapping[str, float]],
    single: Mapping[str, Mapping[str, Any]],
) -> float:
    maximum = 0.0
    for regime in REGIMES:
        for actual, level_index in ((clean, 0), (single, SOURCE_LEVEL_INDEX)):
            expected = source["levels"][level_index]["task_metrics"][regime]
            observed = actual[regime]["task"] if "task" in actual[regime] else actual[regime]
            for name in (
                "exact_bin_accuracy",
                "mean_circular_error_radians",
                "mean_target_cross_entropy",
            ):
                maximum = max(maximum, abs(float(observed[name]) - float(expected[name])))
    return maximum


def _fingerprint(
    config: ReferenceAcquisitionConfig,
    implementation: str,
    condition: str,
    seed: int,
    provenance: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    arrays_sha256: str,
    denoisers_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "condition": condition,
        "seed": seed,
        "provenance": dict(provenance),
        "dataset_hashes": dict(dataset_hashes),
        "repeat_arrays_sha256": arrays_sha256,
        "denoisers_sha256": denoisers_sha256,
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
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("implementation_sha256") == implementation
        and value.get("condition") == condition
        and int(value.get("seed", -1)) == seed
    ):
        raise ValueError(f"incompatible completed acquisition result {path}")
    return value


def _campaign_reusable(value: Mapping[str, Any], implementation: str) -> bool:
    arrays = Path(value.get("artifacts", {}).get("repeat_arrays", ""))
    denoisers = Path(value.get("artifacts", {}).get("denoisers", ""))
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("implementation_sha256") == implementation
        and arrays.is_file()
        and _sha256(arrays) == value["artifacts"].get("repeat_arrays_sha256")
        and denoisers.is_file()
        and _sha256(denoisers) == value["artifacts"].get("denoisers_sha256")
        and all(
            Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["result_sha256"]
            for item in value.get("results", [])
        )
    )


def _evaluate_dataset(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    source_config: calibrated.CalibratedFrontendConfig,
    device: torch.device,
) -> dict[str, Any]:
    features = calibrated.extract_features(system, dataset, task, source_config, device)
    return {
        "frontend": frontend_metrics(features["frontend"], dataset),
        "task": orientation._task_metrics_from_full(
            system,
            features["full"],
            dataset,
            task,
            source_config.activation_batch_size,
            device,
        ),
    }


def run_campaign(
    config: ReferenceAcquisitionConfig, output: Path
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
        if _campaign_reusable(existing, implementation):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed acquisition campaign {campaign_path}")

    (
        orientation_campaign,
        orientation_path,
        _,
        task,
        source_config,
    ) = _load_sources(config)
    runtime_config = replace(
        source_config,
        seeds=config.seeds,
        activation_batch_size=config.activation_batch_size,
        allow_underpowered=config.allow_underpowered,
    )
    datasets = {
        name: calibrated.generate_calibrated_dataset(
            task,
            sample_count=count,
            seed=seed,
            regime=regime,
            shuffle=True,
        )
        for name, (seed, regime, count) in orientation.SPLIT_SPECS.items()
    }
    dataset_hashes = {name: _dataset_hash(dataset) for name, dataset in datasets.items()}
    if dataset_hashes != orientation_campaign.get("dataset_hashes"):
        raise ValueError("repeated-reference datasets do not replay orientation source")

    source_noise_path = Path(orientation_campaign["artifacts"]["noise_arrays"])
    source_base: dict[str, torch.Tensor] = {}
    with np.load(source_noise_path, allow_pickle=False) as archive:
        for name in datasets:
            source_base[name] = torch.from_numpy(archive[f"{name}__base_noise"])
    repeat_noise = {
        name: build_repeat_noise(
            datasets[name],
            source_base[name],
            split_seed=orientation.SPLIT_SPECS[name][0],
            maximum_repeats=max(config.repeat_counts),
        )
        for name in datasets
    }
    repeat_path = output / "reference_repeat_arrays.npz"
    repeat_sha256 = _ensure_repeat_arrays(repeat_path, datasets, repeat_noise)
    observations = {
        name: repeated_observations(
            datasets[name].calibration, repeat_noise[name], config.sigma_radians
        )
        for name in datasets
    }

    pair_shared_maximum = max(
        float(
            torch.stack(
                [
                    repeat_noise[name][datasets[name].paired.fiber.fiber_id == fiber]
                    .sub(
                        repeat_noise[name][
                            datasets[name].paired.fiber.fiber_id == fiber
                        ][0]
                    )
                    .abs()
                    .max()
                    for fiber in torch.unique(
                        datasets[name].paired.fiber.fiber_id, sorted=True
                    )
                ]
            ).max()
        )
        for name in datasets
    )
    first_draw_maximum = max(
        float((repeat_noise[name][:, 0] - source_base[name].double()).abs().max())
        for name in datasets
    )
    noise_contract = pair_shared_maximum == 0.0 and first_draw_maximum == 0.0

    denoiser_coefficients: dict[int, torch.Tensor] = {}
    denoiser_fits: dict[str, Any] = {}
    for count in config.repeat_counts:
        coefficients, fit = fit_equivariant_moment_denoiser(
            observations["train"],
            datasets["train"].calibration[:, :2],
            count,
            config.ridge_ratio,
        )
        denoiser_coefficients[count] = coefficients
        denoiser_fits[str(count)] = {
            **fit,
            "coefficients": coefficients.tolist(),
        }
    denoiser_path = output / "equivariant_moment_denoisers.pt"
    _write_torch(
        denoiser_path,
        {
            "schema_version": SCHEMA_VERSION,
            "ridge_ratio": config.ridge_ratio,
            "coefficients": denoiser_coefficients,
        },
    )
    denoiser_sha256 = _sha256(denoiser_path)

    packets: dict[str, dict[int, dict[str, calibrated.CalibratedDataset]]] = {
        method: {} for method in METHODS
    }
    acquisition_audits: dict[str, dict[str, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        for count in config.repeat_counts:
            packets[method][count] = {}
            acquisition_audits[method][str(count)] = {}
            for name, dataset in datasets.items():
                if method == "analytic_circular_mean":
                    estimate = analytic_circular_mean(observations[name], count)
                else:
                    estimate = apply_equivariant_moment_denoiser(
                        observations[name], count, denoiser_coefficients[count]
                    )
                packet = packet_from_orientation(dataset.calibration, estimate)
                packets[method][count][name] = calibrated.CalibratedDataset(
                    paired=dataset.paired, calibration=packet
                )
                audit = orientation_audit(
                    estimate, dataset.calibration, count, config.sigma_radians
                )
                if method == "analytic_circular_mean":
                    resultant = observations[name][:, :count].mean(1).abs()
                    audit["mean_resultant_length"] = float(resultant.mean())
                    audit["minimum_resultant_length"] = float(resultant.min())
                acquisition_audits[method][str(count)][name] = audit

    slopes = {
        regime: scaling_slope(
            config.repeat_counts,
            [
                acquisition_audits["analytic_circular_mean"][str(count)][regime][
                    "angular_rmse_radians"
                ]
                for count in config.repeat_counts
            ],
        )
        for regime in PRIMARY_REGIMES
    }
    scaling_contract = all(
        config.scaling_slope_minimum <= value <= config.scaling_slope_maximum
        for value in slopes.values()
    )
    maximum_count = max(config.repeat_counts)
    learned_rmse_gaps = {
        regime: acquisition_audits["learned_equivariant_moment"][
            str(maximum_count)
        ][regime]["angular_rmse_radians"]
        - acquisition_audits["analytic_circular_mean"][str(maximum_count)][regime][
            "angular_rmse_radians"
        ]
        for regime in PRIMARY_REGIMES
    }
    learned_rmse_contract = all(
        gap <= config.learned_rmse_gap_ceiling for gap in learned_rmse_gaps.values()
    )

    misgrouped_datasets = {
        name: calibrated.CalibratedDataset(
            paired=dataset.paired,
            calibration=packet_from_orientation(
                dataset.calibration,
                misgrouped_circular_mean(
                    observations[name], dataset.paired.fiber.fiber_id, maximum_count
                ),
            ),
        )
        for name, dataset in datasets.items()
    }

    results: list[dict[str, Any]] = []
    reused = 0
    root = Path(config.source_root)
    for condition_index, condition in enumerate(CONDITIONS):
        for seed in config.seeds:
            source_orientation, source_result_path, source_result_sha256 = (
                _source_result(orientation_campaign, condition, seed)
            )
            system, provenance = orientation._load_system(
                root, condition, seed, task, source_config, device
            )
            provenance = {
                **provenance,
                "orientation_result": str(source_result_path),
                "orientation_result_sha256": source_result_sha256,
                "orientation_scientific_fingerprint": source_orientation[
                    "scientific_fingerprint"
                ],
            }
            fingerprint = _fingerprint(
                config,
                implementation,
                condition,
                seed,
                provenance,
                dataset_hashes,
                repeat_sha256,
                denoiser_sha256,
            )
            result_path = output / "runs" / condition / f"seed_{seed}" / "result.json"
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
            clean = {
                regime: _evaluate_dataset(
                    system, datasets[regime], task, runtime_config, device
                )
                for regime in REGIMES
            }
            clean_accuracy = {
                regime: clean[regime]["task"]["exact_bin_accuracy"]
                for regime in PRIMARY_REGIMES
            }
            evaluations: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
            method_passes: dict[str, dict[str, bool]] = {method: {} for method in METHODS}
            for method in METHODS:
                for count in config.repeat_counts:
                    values = {
                        regime: _evaluate_dataset(
                            system,
                            packets[method][count][regime],
                            task,
                            runtime_config,
                            device,
                        )
                        for regime in REGIMES
                    }
                    passed, losses = task_gate(
                        clean_accuracy, values, config.accuracy_loss_ceiling
                    )
                    if method == "learned_equivariant_moment" and count == maximum_count:
                        passed = bool(passed and learned_rmse_contract)
                    evaluations[method][str(count)] = {
                        "regimes": values,
                        "accuracy_loss_from_clean": losses,
                        "task_gate": passed,
                    }
                    method_passes[method][str(count)] = passed

            misgrouped = {
                regime: _evaluate_dataset(
                    system, misgrouped_datasets[regime], task, runtime_config, device
                )
                for regime in PRIMARY_REGIMES
            }
            misgrouped_pass, misgrouped_losses = task_gate(
                clean_accuracy, misgrouped, config.accuracy_loss_ceiling
            )

            single_packet = packets["analytic_circular_mean"][1]
            causal_writes = {
                regime: causal_coordinate_write(
                    system,
                    single_packet[regime],
                    task,
                    config.activation_batch_size,
                    device,
                    CONTROL_SEED_OFFSET
                    + config.analysis_seed
                    + condition_index * 1_000_003
                    + seed * 10_007
                    + regime_index * 101,
                )
                for regime_index, regime in enumerate(PRIMARY_REGIMES)
            }
            oracle_losses = {
                regime: clean_accuracy[regime]
                - causal_writes[regime]["target_oracle"]["exact_bin_accuracy"]
                for regime in PRIMARY_REGIMES
            }
            shuffled_losses = {
                regime: clean_accuracy[regime]
                - causal_writes[regime]["target_shuffled"]["exact_bin_accuracy"]
                for regime in PRIMARY_REGIMES
            }
            oracle_pass = all(
                value <= config.accuracy_loss_ceiling for value in oracle_losses.values()
            )
            shuffled_pass = all(
                value <= config.accuracy_loss_ceiling for value in shuffled_losses.values()
            )
            causal_finite = all(
                causal_writes[regime]["finite_contract"] for regime in PRIMARY_REGIMES
            )

            single = evaluations["analytic_circular_mean"]["1"]["regimes"]
            replay_error = _source_replay_error(source_orientation, clean, single)
            inherited_representation = bool(
                source_orientation["levels"][SOURCE_LEVEL_INDEX][
                    "representation_gate"
                ]
            )
            state_unchanged = calibrated._module_digest(system) == initial_state
            finite = _finite_numbers(
                {
                    "clean": clean,
                    "evaluations": evaluations,
                    "misgrouped": misgrouped,
                    "causal_writes": causal_writes,
                }
            )
            validity = bool(
                noise_contract
                and replay_error <= config.source_replay_tolerance
                and inherited_representation
                and state_unchanged
                and finite
                and causal_finite
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": (
                    f"tinyllm-reference-acquisition-{condition}-seed{seed}"
                ),
                "status": "completed",
                "evidence_role": _evidence_role(config),
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": asdict(config),
                "scientific_fingerprint": fingerprint,
                "implementation_sha256": implementation,
                "provenance": provenance,
                "dataset_hashes": dataset_hashes,
                "clean": clean,
                "evaluations": evaluations,
                "method_passes": method_passes,
                "misgrouped_control": {
                    "regimes": misgrouped,
                    "accuracy_loss_from_clean": misgrouped_losses,
                    "task_gate": misgrouped_pass,
                },
                "causal_ceiling": {
                    "regimes": causal_writes,
                    "target_oracle_accuracy_loss_from_clean": oracle_losses,
                    "target_shuffled_accuracy_loss_from_clean": shuffled_losses,
                    "target_oracle_gate": oracle_pass,
                    "target_shuffled_gate": shuffled_pass,
                },
                "maximum_source_metric_replay_error": replay_error,
                "gates": {
                    "source_metric_replay_contract": (
                        replay_error <= config.source_replay_tolerance
                    ),
                    "inherited_representation_contract": inherited_representation,
                    "repeat_noise_contract": noise_contract,
                    "finite_numerical_contract": finite,
                    "causal_write_finite_contract": causal_finite,
                    "system_state_unchanged_contract": state_unchanged,
                    "validity_contract": validity,
                },
                "analysis_seconds": time.perf_counter() - seed_started,
            }
            _write_json(result_path, result)
            results.append(result)
            if _implementation_digest() != implementation:
                raise RuntimeError("acquisition implementation changed during campaign")
            print(f"completed {condition} seed {seed}", flush=True)
            del system
            if device.type == "cuda":
                torch.cuda.empty_cache()

    required = len(config.seeds) if config.allow_underpowered else config.required_seed_passes
    maximum_key = str(maximum_count)
    method_counts = {
        method: {
            str(count): {
                condition: sum(
                    int(result["method_passes"][method][str(count)])
                    for result in results
                    if result["condition"] == condition
                )
                for condition in CONDITIONS
            }
            for count in config.repeat_counts
        }
        for method in METHODS
    }
    misgrouped_counts = {
        condition: sum(
            int(result["misgrouped_control"]["task_gate"])
            for result in results
            if result["condition"] == condition
        )
        for condition in CONDITIONS
    }
    oracle_counts = {
        condition: sum(
            int(result["causal_ceiling"]["target_oracle_gate"])
            for result in results
            if result["condition"] == condition
        )
        for condition in CONDITIONS
    }
    shuffled_counts = {
        condition: sum(
            int(result["causal_ceiling"]["target_shuffled_gate"])
            for result in results
            if result["condition"] == condition
        )
        for condition in CONDITIONS
    }
    source_replay_contract = all(
        result["gates"]["source_metric_replay_contract"] for result in results
    )
    validity_contract = all(result["gates"]["validity_contract"] for result in results)
    control_contract = all(
        misgrouped_counts[condition] <= config.control_pass_ceiling
        and shuffled_counts[condition] <= config.control_pass_ceiling
        for condition in CONDITIONS
    )
    oracle_contract = all(oracle_counts[condition] >= required for condition in CONDITIONS)
    systems_valid = bool(
        noise_contract
        and source_replay_contract
        and validity_contract
        and scaling_contract
    )
    scientific_valid = bool(
        systems_valid
        and control_contract
        and oracle_contract
    )
    valid = systems_valid if config.allow_underpowered else scientific_valid
    classification = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else classify_campaign(
            valid=valid,
            analytic_pass=method_counts["analytic_circular_mean"][maximum_key],
            learned_pass=method_counts["learned_equivariant_moment"][maximum_key],
            required=required,
        )
    )
    first_passing_counts = {
        result["experiment_id"]: {
            method: next(
                (
                    count
                    for count in config.repeat_counts
                    if result["method_passes"][method][str(count)]
                ),
                None,
            )
            for method in METHODS
        }
        for result in results
    }

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
            "orientation_campaign": str(orientation_path),
            "orientation_campaign_sha256": SOURCE_ORIENTATION_CAMPAIGN_SHA256,
            "orientation_implementation_sha256": (
                SOURCE_ORIENTATION_IMPLEMENTATION_SHA256
            ),
            "orientation_noise_sha256": SOURCE_ORIENTATION_NOISE_SHA256,
            "calibrated_campaign_sha256": SOURCE_CALIBRATED_CAMPAIGN_SHA256,
            "calibrated_implementation_sha256": (
                SOURCE_CALIBRATED_IMPLEMENTATION_SHA256
            ),
        },
        "dataset_hashes": dataset_hashes,
        "repeat_noise_contracts": {
            "pair_shared_repeat_contract": pair_shared_maximum == 0.0,
            "source_first_draw_contract": first_draw_maximum == 0.0,
            "maximum_pair_shared_error": pair_shared_maximum,
            "maximum_source_first_draw_error": first_draw_maximum,
        },
        "denoiser_fits": denoiser_fits,
        "acquisition_audits": acquisition_audits,
        "scaling": {
            "slopes": slopes,
            "contract": scaling_contract,
            "bounds": [
                config.scaling_slope_minimum,
                config.scaling_slope_maximum,
            ],
        },
        "learned_analytic_comparison": {
            "rmse_gap_at_maximum_count": learned_rmse_gaps,
            "ceiling": config.learned_rmse_gap_ceiling,
            "contract": learned_rmse_contract,
        },
        "aggregates": {
            "valid": valid,
            "classification": classification,
            "required_seed_passes": required,
            "method_pass_counts": method_counts,
            "misgrouped_control_pass_counts": misgrouped_counts,
            "target_oracle_pass_counts": oracle_counts,
            "target_shuffled_pass_counts": shuffled_counts,
            "first_passing_counts": first_passing_counts,
        },
        "summary": {
            "requested": len(CONDITIONS) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "reused": reused,
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_task_heads": 0,
            "new_representation_probes": 0,
            "fitted_acquisition_denoisers": len(config.repeat_counts),
            "fitted_acquisition_parameters": 3 * len(config.repeat_counts),
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "condition": result["condition"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "gates": result["gates"],
                "method_passes": result["method_passes"],
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
            }
            for result in results
        ],
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
            "Only repeated orientation-reference acquisition is changed; every front end, TinyLLM, and answer head remains frozen.",
            "The learned denoiser uses clean calibration-orientation supervision but no phase, cosine, target-bin, or task label.",
            "The true-coordinate write is a label-using causal ceiling and is not deployable.",
            "The synthetic repeat errors are unbiased and independent rather than a measured instrument-noise model.",
            "The selected checkpoints are replication units, not a random architecture population.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "*" / "seed_*" / "result.json"),
            "repeat_arrays": str(repeat_path),
            "repeat_arrays_sha256": repeat_sha256,
            "denoisers": str(denoiser_path),
            "denoisers_sha256": denoiser_sha256,
        },
    }
    if _implementation_digest() != implementation:
        raise RuntimeError("acquisition implementation changed before campaign write")
    _write_json(campaign_path, campaign)
    return campaign


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
            "data/experiments/tinyllm_reference_acquisition_replicates/"
            "20260807_d8_preregistered"
        ),
    )
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if args.shakedown:
        seeds = (7,)
        counts = (1, 4)
        allow_underpowered = True
    else:
        counts = COUNTS
        allow_underpowered = args.allow_underpowered
    config = ReferenceAcquisitionConfig(
        seeds=seeds,
        repeat_counts=counts,
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
