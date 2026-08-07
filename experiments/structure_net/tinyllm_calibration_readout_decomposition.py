#!/usr/bin/env python3
"""Localize calibrated TinyLLM failures with frozen readouts and causal writes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr
import torch
from torch import nn
import torch.nn.functional as F

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibration_degradation_causal as degradation
import experiments.structure_net.tinyllm_conditional_branch_depth_scan as probes
import experiments.structure_net.tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-calibration-readout-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-calibration-readout-decomposition-v1"
SOURCE_CAMPAIGN_SHA256 = (
    "170d99553058ab544d95d6abf4d26ea1062bfe1b79fda2deefc531dfaebd0f6e"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "f273abf83231e4fc37583154c24158c81bba15a29e11ca3f96daf0b579be0f70"
)
SOURCE_MANIFEST_SHA256 = (
    "23598aaef5a0d16825ff9a928de57857e7bd58b10feab8853739448acfd983fc"
)
SOURCE_NOISE_SHA256 = (
    "1896c852644f724f5a2d214d5e69380eef3e46abb2e7b295eb0d4bace51a0c82"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
LEVELS = (0.0, 0.05, 0.10, 0.20)
READOUTS = ("model_argmax", "model_posterior_mean", "frontend", "observer")
PATCHES = ("observer", "target_oracle", "flipped", "shuffled", "kernel")
PRIMARY_EVIDENCE_ROLE = (
    "preregistered_existing_checkpoint_post_outcome_mechanistic_diagnostic"
)


@dataclass(frozen=True)
class ReadoutDecompositionConfig:
    source_root: str = (
        "data/experiments/tinyllm_calibration_degradation_causal/"
        "20260807_d8_existing_checkpoints"
    )
    conditions: tuple[str, ...] = CONDITIONS
    seeds: tuple[int, ...] = SEEDS
    levels: tuple[float, ...] = LEVELS
    task_accuracy_drop_ceiling: float = 0.03
    required_seed_passes: int = 4
    control_pass_ceiling: int = 1
    clean_replay_tolerance: float = 1e-5
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.conditions) != CONDITIONS and not self.allow_underpowered:
            raise ValueError("the two primary arms are fixed")
        if tuple(self.seeds) != SEEDS and not self.allow_underpowered:
            raise ValueError("the five primary seeds are fixed")
        if tuple(self.levels) != LEVELS:
            raise ValueError("the readout noise levels are fixed")
        if set(self.conditions).difference(CONDITIONS):
            raise ValueError("unknown condition")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.required_seed_passes > len(self.seeds):
            raise ValueError("required seed passes exceeds selected seeds")


class FrozenCosineObserver(nn.Module):
    """The cosine member of the predecessor's nonlinear conditional suite."""

    def __init__(self, features: int, width: int):
        super().__init__()
        self.network = probes._SmallProbe(features, width, 1, "nonlinear")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


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


def _write_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(calibrated.__file__),
        Path(degradation.__file__),
        Path(probes.__file__),
        Path(invariant.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def noise_key(level: float) -> str:
    return degradation.noise_key(level)


def evaluation_key(regime: str, level: float) -> str:
    return degradation.evaluation_key(regime, noise_key(level))


def interval_centers(count: int, *, device: torch.device | None = None) -> torch.Tensor:
    if count < 2:
        raise ValueError("at least two answer centers are required")
    return torch.linspace(-1.0, 1.0, count, device=device)


def interval_posterior(coordinate: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    width = 2.0 / float(len(centers) - 1)
    logits = -0.5 * ((centers[None, :] - coordinate[:, None]) / width).square()
    return torch.softmax(logits, dim=-1)


def coordinate_metrics(
    coordinate: torch.Tensor,
    target_cosine: torch.Tensor,
    targets: torch.Tensor,
    centers: torch.Tensor,
) -> dict[str, float]:
    coordinate = coordinate.detach().double().cpu().clamp(-1.0, 1.0)
    target_cosine = target_cosine.detach().double().cpu()
    targets = targets.detach().double().cpu()
    centers = centers.detach().double().cpu()
    posterior = interval_posterior(coordinate, centers)
    predicted_bin = (coordinate[:, None] - centers[None, :]).abs().argmin(1)
    target_bin = targets.argmax(1)
    correlation = 0.0
    x = coordinate.numpy()
    y = target_cosine.numpy()
    if np.ptp(x) > 1e-8 and np.ptp(y) > 1e-8:
        correlation = float(pearsonr(x, y).statistic)
        if not np.isfinite(correlation):
            correlation = 0.0
    error = coordinate - target_cosine
    return {
        "exact_bin_accuracy": float((predicted_bin == target_bin).double().mean()),
        "target_cross_entropy": float(
            -(targets * posterior.clamp_min(1e-12).log()).sum(1).mean()
        ),
        "cosine_mae": float(error.abs().mean()),
        "cosine_rmse": float(error.square().mean().sqrt()),
        "cosine_pearson": correlation,
    }


def posterior_metrics(
    posterior: torch.Tensor,
    target_cosine: torch.Tensor,
    targets: torch.Tensor,
    centers: torch.Tensor,
    *,
    use_argmax: bool,
) -> dict[str, float]:
    posterior = posterior.detach().double().cpu()
    targets = targets.detach().double().cpu()
    centers = centers.detach().double().cpu()
    mean = posterior @ centers
    prediction = posterior.argmax(1) if use_argmax else (
        mean[:, None] - centers[None, :]
    ).abs().argmin(1)
    target_bin = targets.argmax(1)
    continuous = coordinate_metrics(mean, target_cosine, targets, centers)
    continuous["exact_bin_accuracy"] = float(
        (prediction == target_bin).double().mean()
    )
    continuous["target_cross_entropy"] = float(
        -(targets * posterior.clamp_min(1e-12).log()).sum(1).mean()
    )
    return continuous


def utility_pass(
    clean_accuracy: float,
    noisy_accuracy: float,
    clean_model_accuracy: float,
    ceiling: float,
) -> tuple[bool, dict[str, float]]:
    adequacy_gap = clean_model_accuracy - clean_accuracy
    noisy_drop = clean_accuracy - noisy_accuracy
    return (
        bool(adequacy_gap <= ceiling and noisy_drop <= ceiling),
        {
            "clean_adequacy_gap": float(adequacy_gap),
            "accuracy_drop_from_own_clean": float(noisy_drop),
        },
    )


def patch_utility_pass(
    clean_patch_accuracy: float,
    noisy_patch_accuracy: float,
    clean_model_accuracy: float,
    ceiling: float,
) -> tuple[bool, dict[str, float]]:
    """Apply the preregistered unchanged-clean-model baseline to a patch."""
    adequacy_gap = clean_model_accuracy - clean_patch_accuracy
    noisy_drop = clean_model_accuracy - noisy_patch_accuracy
    return (
        bool(adequacy_gap <= ceiling and noisy_drop <= ceiling),
        {
            "clean_adequacy_gap": float(adequacy_gap),
            "accuracy_drop_from_clean_model": float(noisy_drop),
        },
    )


def classify_campaign(
    *,
    valid: bool,
    argmax_pass: Mapping[str, int],
    posterior_pass: Mapping[str, int],
    observer_pass: Mapping[str, int],
    patch_pass: Mapping[str, int],
    target_oracle_pass: Mapping[str, int],
    controls_specific: bool,
    required: int,
) -> str:
    if not valid:
        return "invalid"
    both = lambda values: all(int(values[arm]) >= required for arm in CONDITIONS)
    analytic_only = (
        int(observer_pass[CONDITIONS[0]]) >= required
        and int(observer_pass[CONDITIONS[1]]) < required
    )
    learned_only = (
        int(observer_pass[CONDITIONS[1]]) >= required
        and int(observer_pass[CONDITIONS[0]]) < required
    )
    argmax_fails = all(int(argmax_pass[arm]) < required for arm in CONDITIONS)
    posterior_fails = all(
        int(posterior_pass[arm]) < required for arm in CONDITIONS
    )
    if argmax_fails and both(posterior_pass):
        return "posterior_shape_calibration_limited"
    if (
        both(observer_pass)
        and both(patch_pass)
        and both(target_oracle_pass)
        and posterior_fails
        and controls_specific
    ):
        return "decoder_relation_limited"
    if both(observer_pass):
        return "coordinate_decodable_but_causal_write_unresolved"
    if all(int(observer_pass[arm]) < required for arm in CONDITIONS):
        return "reference_coordinate_precision_limited"
    if learned_only:
        return "analytic_reference_precision_limited"
    if analytic_only:
        return "learned_reference_precision_limited"
    return "mixed_readout_reference_limit"


def _source_degradation_config(campaign: Mapping[str, Any]) -> degradation.CalibrationDegradationConfig:
    values = dict(campaign["configuration"])
    for key in ("conditions", "seeds", "noise_levels"):
        values[key] = tuple(values[key])
    return degradation.CalibrationDegradationConfig(**values)


def _load_sources(config: ReadoutDecompositionConfig) -> tuple[
    dict[str, Any], degradation.CalibrationDegradationConfig, dict[str, Any],
    CircleTaskConfig, calibrated.CalibratedFrontendConfig, dict[str, calibrated.CalibratedDataset], str
]:
    source_path = Path(config.source_root) / "campaign_results.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_results = source.get("results", [])
    if not (
        _sha256(source_path) == SOURCE_CAMPAIGN_SHA256
        and source.get("schema_version") == degradation.SCHEMA_VERSION
        and source.get("hypothesis_id") == degradation.HYPOTHESIS_ID
        and source.get("implementation_sha256") == SOURCE_IMPLEMENTATION_SHA256
        and source.get("status") == "completed"
        and source.get("provenance", {}).get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256
        and source.get("provenance", {}).get("noise_sha256") == SOURCE_NOISE_SHA256
        and int(source.get("summary", {}).get("requested", -1)) == 10
        and int(source.get("summary", {}).get("completed", -1)) == 10
        and int(source.get("summary", {}).get("failed", -1)) == 0
        and int(source.get("summary", {}).get("trained_models", -1)) == 0
        and int(source.get("summary", {}).get("trained_frontends", -1)) == 0
        and len(source_results) == 10
        and all(
            Path(item.get("path", "")).is_file()
            and Path(item.get("activations", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            and _sha256(Path(item["activations"])) == item.get("activations_sha256")
            for item in source_results
        )
    ):
        raise ValueError("invalid calibration-degradation source campaign")
    degradation_config = _source_degradation_config(source)
    calibrated_campaign, _, manifest, _ = degradation._load_source_campaign(
        degradation_config
    )
    if manifest != SOURCE_MANIFEST_SHA256:
        raise ValueError("source checkpoint manifest mismatch")
    task = CircleTaskConfig(**calibrated_campaign["task_config"])
    source_config = calibrated._config_from_mapping(calibrated_campaign["configuration"])
    datasets = degradation._datasets(task, source_config)
    interventions, noise_sha256 = degradation._interventions(
        datasets, source_config, degradation_config
    )
    if noise_sha256 != SOURCE_NOISE_SHA256:
        raise ValueError("source common-noise replay mismatch")
    selected = {
        evaluation_key(regime, level): interventions[evaluation_key(regime, level)]
        for regime in REGIMES
        for level in LEVELS
    }
    input_target_hashes = {
        key: invariant._tensor_digest(
            value.paired.circle.input_ids,
            value.paired.circle.target_posteriors,
            value.paired.fiber.cosine,
            value.paired.fiber.branch,
        )
        for key, value in selected.items()
    }
    for regime in REGIMES:
        expected = input_target_hashes[evaluation_key(regime, 0.0)]
        if any(
            digest != expected
            for key, digest in input_target_hashes.items()
            if key.startswith(regime + "__")
        ):
            raise ValueError(f"input/target identity failed for {regime}")
    return (
        source,
        degradation_config,
        calibrated_campaign,
        task,
        source_config,
        {**datasets, **selected},
        noise_sha256,
    )


def _source_result(
    campaign: Mapping[str, Any],
    source_root: Path,
    condition: str,
    seed: int,
) -> tuple[dict[str, Any], Path]:
    path = source_root / "runs" / condition / f"seed_{seed}" / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = [
        item
        for item in campaign.get("results", [])
        if item.get("condition") == condition and int(item.get("seed", -1)) == seed
    ]
    matching = [
        item for item in value.get("cells", [])
        if item.get("intervention") == noise_key(0.0)
    ]
    if (
        len(entries) != 1
        or _sha256(path) != entries[0].get("result_sha256")
        or value.get("schema_version") != degradation.SCHEMA_VERSION
        or value.get("hypothesis_id") != degradation.HYPOTHESIS_ID
        or value.get("status") != "completed"
        or value.get("condition") != condition
        or int(value.get("seed", -1)) != seed
        or len(matching) != len(REGIMES)
    ):
        raise ValueError(f"source clean cells missing in {path}")
    return value, path


def _fit_observer(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_cosine: torch.Tensor,
    validation_cosine: torch.Tensor,
    source_config: calibrated.CalibratedFrontendConfig,
    seed: int,
    device: torch.device,
) -> tuple[FrozenCosineObserver, np.ndarray, np.ndarray, dict[str, Any]]:
    """Exactly replay the cosine half of the predecessor joint suite.

    The unused branch head is retained during fitting because its initialization
    consumes RNG state before the cosine head in the source implementation.
    """
    torch.manual_seed(seed)
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    deviation = np.maximum(
        train_features.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5
    )
    train_x = torch.from_numpy((train_features - mean) / deviation).float().to(device)
    validation_x = torch.from_numpy(
        (validation_features - mean) / deviation
    ).float().to(device)
    train_y = train_cosine.to(device)
    validation_y = validation_cosine.to(device)
    width = min(source_config.probe_width, max(16, train_features.shape[1]))
    suite = probes._ConditionalSuite(train_features.shape[1], width, "nonlinear").to(device)
    optimizer = torch.optim.AdamW(
        suite.parameters(),
        lr=source_config.learning_rate,
        weight_decay=source_config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 401)
    best_loss = float("inf")
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_step = 0
    stale = 0
    steps_run = 0
    # Use the real paired branch labels so the optimizer and RNG match exactly.
    train_labels = torch.zeros_like(train_y)
    validation_labels = torch.zeros_like(validation_y)
    # Callers replace these placeholders below by attaching attributes. This is
    # kept internal to make the fit contract explicit and easy to test.
    if not hasattr(train_cosine, "_branch_labels"):
        raise ValueError("observer fit requires paired branch labels")
    train_labels = getattr(train_cosine, "_branch_labels").to(device)
    validation_labels = getattr(validation_cosine, "_branch_labels").to(device)

    def loss(values: torch.Tensor, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        branch_logits = suite.branch(torch.cat((values, cosine[:, None]), 1)).squeeze(1)
        predicted = suite.cosine(values).squeeze(1)
        return F.binary_cross_entropy_with_logits(branch_logits, labels) + F.mse_loss(predicted, cosine)

    for step in range(1, source_config.probe_steps + 1):
        indices = torch.randint(
            0, len(train_x), (source_config.probe_batch_size,), generator=generator
        ).to(device)
        suite.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss(train_x[indices], train_y[indices], train_labels[indices])
        train_loss.backward()
        optimizer.step()
        steps_run = step
        if step % source_config.validation_interval == 0 or step == source_config.probe_steps:
            suite.eval()
            with torch.no_grad():
                validation_loss = float(loss(validation_x, validation_y, validation_labels))
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in suite.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= source_config.early_stopping_patience:
                    break
    if best_state is None:
        raise RuntimeError("observer fit produced no validation checkpoint")
    suite.load_state_dict(best_state)
    observer = FrozenCosineObserver(train_features.shape[1], width).to(device)
    observer.network.load_state_dict(suite.cosine.state_dict())
    observer.eval()
    return observer, mean, deviation, {
        "best_step": best_step,
        "steps_run": steps_run,
        "best_validation_loss": best_loss,
        "width": width,
        "seed": seed,
    }


def _attach_branch_labels(cosine: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    # torch.Tensor instances accept transient Python attributes; neither tensor
    # bytes nor the scientific data contract are changed.
    setattr(cosine, "_branch_labels", branch)
    return cosine


@torch.no_grad()
def _observer_predict(
    observer: FrozenCosineObserver,
    features: np.ndarray,
    mean: np.ndarray,
    deviation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    values = torch.from_numpy((features - mean) / deviation).float()
    parts = []
    for start in range(0, len(values), batch_size):
        parts.append(observer(values[start : start + batch_size].to(device)).cpu())
    return torch.cat(parts)


def _forward_outputs(
    system: calibrated.CalibratedTinyLLM,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    source_config: calibrated.CalibratedFrontendConfig,
    observer_coordinate: torch.Tensor,
    device: torch.device,
    control_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float]]]:
    observer_coordinate = observer_coordinate.clamp(-1.0, 1.0)
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    centers = interval_centers(len(task.answer_token_ids), device=device)
    model_posteriors: list[torch.Tensor] = []
    frontend_coordinates: list[torch.Tensor] = []
    patch_posteriors: dict[str, list[torch.Tensor]] = {name: [] for name in PATCHES}
    patch_diagnostics: dict[str, list[float]] = {
        "gradient_norm": [], "patch_norm": [], "first_order_error": [],
        "post_patch_mean_error": [], "finite": [],
    }
    generator = torch.Generator(device="cpu").manual_seed(control_seed)
    permutation = torch.randperm(len(sensor), generator=generator)
    shuffled = observer_coordinate[permutation]
    random_directions = torch.randn(
        (len(sensor), system.model.config.n_embd), generator=generator
    )
    target_cosine = dataset.paired.fiber.cosine
    batch_size = source_config.activation_batch_size
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
        gradient_norm = gradient_sq.sqrt()
        coordinates = {
            "observer": observer_coordinate[start:stop].to(device),
            "target_oracle": target_cosine[start:stop].to(device),
            "flipped": observer_coordinate[start:stop].to(device),
            "shuffled": shuffled[start:stop].to(device),
        }
        alpha = {
            key: (value - moment.detach()) / gradient_sq
            for key, value in coordinates.items()
        }
        deltas = {
            "observer": alpha["observer"][:, None] * gradient,
            "target_oracle": alpha["target_oracle"][:, None] * gradient,
            "flipped": -alpha["flipped"][:, None] * gradient,
            "shuffled": alpha["shuffled"][:, None] * gradient,
        }
        random = random_directions[start:stop].to(device)
        random = random - (
            (random * gradient).sum(1) / gradient_sq
        )[:, None] * gradient
        random = random / random.norm(dim=1, keepdim=True).clamp_min(1e-8)
        deltas["kernel"] = random * deltas["observer"].norm(dim=1, keepdim=True)
        for name, delta in deltas.items():
            patched_logits = invariant._task_logits(
                system.model, residual.detach() + delta.detach(), answer_ids
            )
            patch_posteriors[name].append(torch.softmax(patched_logits, -1).detach().cpu())
        patched_observer = patch_posteriors["observer"][-1].to(device) @ centers
        intended = coordinates["observer"]
        first_order = moment.detach() + (gradient * deltas["observer"]).sum(1)
        patch_diagnostics["gradient_norm"].extend(gradient_norm.detach().cpu().tolist())
        patch_diagnostics["patch_norm"].extend(
            deltas["observer"].norm(dim=1).detach().cpu().tolist()
        )
        patch_diagnostics["first_order_error"].extend(
            (first_order - intended).abs().detach().cpu().tolist()
        )
        patch_diagnostics["post_patch_mean_error"].extend(
            (patched_observer - intended).abs().detach().cpu().tolist()
        )
        finite = torch.isfinite(residual).all(1) & torch.isfinite(gradient).all(1)
        finite &= torch.isfinite(deltas["observer"]).all(1)
        patch_diagnostics["finite"].extend(finite.detach().cpu().float().tolist())
        model_posteriors.append(posterior.detach().cpu())
        frontend_coordinates.append(system.feature(
            sensor[start:stop].to(device), dataset.calibration[start:stop].to(device)
        ).detach().cpu().squeeze(1))
    diagnostics = {
        "observer": {
            "mean_task_gradient_norm": float(np.mean(patch_diagnostics["gradient_norm"])),
            "minimum_task_gradient_norm": float(np.min(patch_diagnostics["gradient_norm"])),
            "mean_patch_norm": float(np.mean(patch_diagnostics["patch_norm"])),
            "mean_absolute_first_order_target_error": float(np.mean(patch_diagnostics["first_order_error"])),
            "mean_absolute_post_patch_moment_error": float(np.mean(patch_diagnostics["post_patch_mean_error"])),
            "finite_fraction": float(np.mean(patch_diagnostics["finite"])),
        }
    }
    return {
        "model_posterior": torch.cat(model_posteriors),
        "frontend_coordinate": torch.cat(frontend_coordinates),
        **{f"patch_{name}": torch.cat(parts) for name, parts in patch_posteriors.items()},
    }, diagnostics


def _readout_records(
    outputs: Mapping[str, torch.Tensor],
    observer_coordinate: torch.Tensor,
    dataset: calibrated.CalibratedDataset,
    task: CircleTaskConfig,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    centers = interval_centers(len(task.answer_token_ids))
    target_cosine = dataset.paired.fiber.cosine
    targets = dataset.paired.circle.target_posteriors
    readouts = {
        "model_argmax": posterior_metrics(outputs["model_posterior"], target_cosine, targets, centers, use_argmax=True),
        "model_posterior_mean": posterior_metrics(outputs["model_posterior"], target_cosine, targets, centers, use_argmax=False),
        "frontend": coordinate_metrics(outputs["frontend_coordinate"], target_cosine, targets, centers),
        "observer": coordinate_metrics(observer_coordinate, target_cosine, targets, centers),
    }
    patches = {
        name: posterior_metrics(outputs[f"patch_{name}"], target_cosine, targets, centers, use_argmax=True)
        for name in PATCHES
    }
    return readouts, patches


def _fingerprint(
    config: ReadoutDecompositionConfig,
    implementation: str,
    condition: str,
    seed: int,
    source_result_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "condition": condition,
        "seed": seed,
        "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
        "source_result_sha256": source_result_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _reusable_result(
    result_path: Path,
    observer_path: Path,
    fingerprint: str,
    implementation: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("implementation_sha256") == implementation
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("artifacts", {}).get("result") == str(result_path)
        and value.get("artifacts", {}).get("observer") == str(observer_path)
        and observer_path.is_file()
        and value.get("artifacts", {}).get("observer_sha256") == _sha256(observer_path)
    ):
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return value


def run_campaign(config: ReadoutDecompositionConfig, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("hypothesis_id") == HYPOTHESIS_ID
            and existing.get("status") == "completed"
            and existing.get("configuration")
            == json.loads(json.dumps(asdict(config), allow_nan=False))
            and existing.get("implementation_sha256") == implementation
            and int(existing.get("summary", {}).get("completed", -1))
            == len(config.conditions) * len(config.seeds)
            and len(existing.get("results", []))
            == len(config.conditions) * len(config.seeds)
            and all(
                Path(item["path"]).is_file()
                and _sha256(Path(item["path"])) == item["result_sha256"]
                and Path(item["observer"]).is_file()
                and _sha256(Path(item["observer"])) == item["observer_sha256"]
                for item in existing.get("results", [])
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    (
        source,
        degradation_config,
        calibrated_campaign,
        task,
        source_config,
        datasets,
        noise_sha256,
    ) = _load_sources(config)
    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    results: list[dict[str, Any]] = []
    reused = 0
    for condition in config.conditions:
        for seed in config.seeds:
            cell_started = time.perf_counter()
            source_result, source_result_path = _source_result(
                source, Path(config.source_root), condition, seed
            )
            source_result_sha256 = _sha256(source_result_path)
            fingerprint = _fingerprint(config, implementation, condition, seed, source_result_sha256)
            directory = output / "runs" / condition / f"seed_{seed}"
            result_path = directory / "result.json"
            observer_path = directory / "observer.pt"
            existing = _reusable_result(result_path, observer_path, fingerprint, implementation)
            if existing is not None:
                results.append(existing)
                reused += 1
                print(f"resuming {condition} seed {seed}", flush=True)
                continue
            system, provenance = degradation._load_system(
                calibrated_campaign, degradation_config, condition, seed, device
            )
            for parameter in system.parameters():
                parameter.requires_grad_(False)
            train_features = calibrated.extract_features(system, datasets["train"], task, source_config, device)["full"]
            validation_features = calibrated.extract_features(system, datasets["validation"], task, source_config, device)["full"]
            train_cosine = _attach_branch_labels(
                datasets["train"].paired.fiber.cosine,
                datasets["train"].paired.fiber.branch,
            )
            validation_cosine = _attach_branch_labels(
                datasets["validation"].paired.fiber.cosine,
                datasets["validation"].paired.fiber.branch,
            )
            probe_seed = source_config.analysis_seed + seed * 10_007 + 1_009
            observer, mean, deviation, fit = _fit_observer(
                train_features, validation_features, train_cosine, validation_cosine,
                source_config, probe_seed, device,
            )
            evaluations: dict[str, Any] = {}
            replay_error = 0.0
            for regime in REGIMES:
                for level in config.levels:
                    key = evaluation_key(regime, level)
                    dataset = datasets[key]
                    full = calibrated.extract_features(system, dataset, task, source_config, device)["full"]
                    observer_coordinate = _observer_predict(
                        observer, full, mean, deviation, device, source_config.activation_batch_size
                    )
                    outputs, diagnostics = _forward_outputs(
                        system, dataset, task, source_config, observer_coordinate, device,
                        source_config.analysis_seed + seed * 100_003 + int(level * 10_000) + (0 if regime == "composition" else 1_000_003),
                    )
                    readouts, patches = _readout_records(outputs, observer_coordinate, dataset, task)
                    evaluations[key] = {
                        "regime": regime,
                        "noise_level": level,
                        "readouts": readouts,
                        "patches": patches,
                        "causal_diagnostics": diagnostics,
                    }
                    if level == 0.0:
                        source_cell = next(
                            item for item in source_result["cells"]
                            if item["intervention"] == noise_key(0.0) and item["regime"] == regime
                        )
                        source_probe = source_cell["representation"]["full"]["probe"]
                        replay_error = max(
                            replay_error,
                            abs(readouts["observer"]["cosine_pearson"] - source_probe["cosine_pearson"]),
                            abs(readouts["observer"]["cosine_rmse"] - source_probe["cosine_rmse"]),
                            abs(readouts["model_argmax"]["exact_bin_accuracy"] - source_cell["task"]["exact_bin_accuracy"]),
                        )
            clean_replay = replay_error <= config.clean_replay_tolerance
            # Attach gates after every measurement exists, using each readout's own clean baseline.
            seed_passes: dict[str, dict[str, bool]] = {
                "readouts": {}, "patches": {}
            }
            for name in READOUTS:
                passed = True
                for regime in REGIMES:
                    clean = evaluations[evaluation_key(regime, 0.0)]["readouts"][name]
                    clean_model = evaluations[evaluation_key(regime, 0.0)]["readouts"]["model_argmax"]["exact_bin_accuracy"]
                    for level in config.levels:
                        record = evaluations[evaluation_key(regime, level)]["readouts"][name]
                        gate, detail = utility_pass(
                            clean["exact_bin_accuracy"], record["exact_bin_accuracy"], clean_model,
                            config.task_accuracy_drop_ceiling,
                        )
                        record.update(detail)
                        record["utility_pass"] = gate
                        if level == 0.20:
                            passed = bool(passed and gate)
                seed_passes["readouts"][name] = passed
            for name in PATCHES:
                passed = True
                for regime in REGIMES:
                    clean_model = evaluations[evaluation_key(regime, 0.0)]["readouts"]["model_argmax"]["exact_bin_accuracy"]
                    clean_patch = evaluations[evaluation_key(regime, 0.0)]["patches"][name]["exact_bin_accuracy"]
                    for level in config.levels:
                        record = evaluations[evaluation_key(regime, level)]["patches"][name]
                        gate, detail = patch_utility_pass(
                            clean_patch, record["exact_bin_accuracy"], clean_model,
                            config.task_accuracy_drop_ceiling,
                        )
                        record.update(detail)
                        record["utility_pass"] = gate
                        if level == 0.20:
                            passed = bool(passed and gate)
                seed_passes["patches"][name] = passed
            finite = all(
                cell["causal_diagnostics"]["observer"]["finite_fraction"] == 1.0
                for cell in evaluations.values()
            )
            finite = bool(
                finite
                and all(
                    np.isfinite(
                        [
                            float(value)
                            for section in (cell["readouts"], cell["patches"])
                            for record in section.values()
                            for value in record.values()
                            if isinstance(value, (int, float))
                        ]
                    ).all()
                    for cell in evaluations.values()
                )
            )
            source_state_pass = bool(
                provenance["system_state_sha256"]
                == source_result.get("source", {}).get("system_state_sha256")
                and provenance["model_state_sha256"]
                == source_result.get("source", {}).get("model_state_sha256")
            )
            valid = bool(clean_replay and finite and source_state_pass)
            _write_torch(observer_path, {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "condition": condition,
                "seed": seed,
                "normalization_mean": torch.from_numpy(mean),
                "normalization_deviation": torch.from_numpy(deviation),
                "state_dict": {key: value.detach().cpu() for key, value in observer.state_dict().items()},
                "fit": fit,
                "scientific_fingerprint": fingerprint,
                "source_degradation_result_sha256": source_result_sha256,
            })
            result = {
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "experiment_id": f"tinyllm-calibration-readout-{condition}-seed{seed}",
                "status": "completed",
                "evidence_role": PRIMARY_EVIDENCE_ROLE if not config.allow_underpowered else "systems_lifecycle_only_not_quality_evidence",
                "completed_at": _utc_now(),
                "condition": condition,
                "seed": seed,
                "configuration": asdict(config),
                "implementation_sha256": implementation,
                "scientific_fingerprint": fingerprint,
                "source": {key: value for key, value in provenance.items() if key != "detail"},
                "source_degradation_result": str(source_result_path),
                "source_degradation_result_sha256": source_result_sha256,
                "observer_fit": fit,
                "clean_replay": {"pass": clean_replay, "maximum_absolute_error": replay_error, "tolerance": config.clean_replay_tolerance},
                "finite_pass": finite,
                "source_state_pass": source_state_pass,
                "input_target_identity_pass": True,
                "evaluations": evaluations,
                "seed_passes_at_0p20": seed_passes,
                "valid": valid,
                "analysis_seconds": time.perf_counter() - cell_started,
                "artifacts": {"result": str(result_path), "observer": str(observer_path), "observer_sha256": _sha256(observer_path)},
            }
            _write_json(result_path, result)
            results.append(result)
            print(f"{condition} seed {seed}: valid={valid}", flush=True)
            del system, observer
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _implementation_digest() != implementation:
                raise RuntimeError("readout-decomposition implementation changed during run")

    valid = all(item["valid"] for item in results)
    counts = {
        "readouts": {
            name: {
                condition: sum(
                    bool(item["seed_passes_at_0p20"]["readouts"][name])
                    for item in results if item["condition"] == condition
                ) for condition in config.conditions
            } for name in READOUTS
        },
        "patches": {
            name: {
                condition: sum(
                    bool(item["seed_passes_at_0p20"]["patches"][name])
                    for item in results if item["condition"] == condition
                ) for condition in config.conditions
            } for name in PATCHES
        },
    }
    pass_counts_by_level = {
        section: {
            name: {
                condition: {
                    noise_key(level): sum(
                        all(
                            bool(
                                item["evaluations"][evaluation_key(regime, level)][
                                    section
                                ][name]["utility_pass"]
                            )
                            for regime in REGIMES
                        )
                        for item in results
                        if item["condition"] == condition
                    )
                    for level in config.levels
                }
                for condition in config.conditions
            }
            for name in (READOUTS if section == "readouts" else PATCHES)
        }
        for section in ("readouts", "patches")
    }
    controls_specific = all(
        counts["patches"][name][condition] <= config.control_pass_ceiling
        for name in ("flipped", "shuffled", "kernel")
        for condition in config.conditions
    )
    classification = classify_campaign(
        valid=valid,
        argmax_pass=counts["readouts"]["model_argmax"],
        posterior_pass=counts["readouts"]["model_posterior_mean"],
        observer_pass=counts["readouts"]["observer"],
        patch_pass=counts["patches"]["observer"],
        target_oracle_pass=counts["patches"]["target_oracle"],
        controls_specific=controls_specific,
        required=config.required_seed_passes,
    )
    campaign_result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": PRIMARY_EVIDENCE_ROLE if not config.allow_underpowered else "systems_lifecycle_only_not_quality_evidence",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "provenance": {
            "source_campaign": str(Path(config.source_root) / "campaign_results.json"),
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_noise_sha256": noise_sha256,
            "dvc_source_root": "ca2abd03b528233760a5f1cb23686dca.dir",
            "lakefs_source_commit": "4fca75c78b6ec1f9e83c1548bb15aca79d076de909b9e0dc7547c55223bb816f",
        },
        "summary": {
            "requested": len(config.conditions) * len(config.seeds),
            "scheduled": len(config.conditions) * len(config.seeds) - reused,
            "completed": len(results), "failed": 0, "excluded": 0, "retries": 0, "reused": reused,
            "trained_models": 0, "trained_frontends": 0, "trained_task_heads": 0,
            "fitted_diagnostic_observers": len(results),
        },
        "aggregates": {
            "classification": classification,
            "valid": valid,
            "pass_counts_at_sigma_0p20": counts,
            "pass_counts_by_level": pass_counts_by_level,
            "controls_specific": controls_specific,
            "required_seed_passes": config.required_seed_passes,
            "primary_hypothesis_pass": classification == "decoder_relation_limited",
        },
        "results": [
            {
                "experiment_id": item["experiment_id"], "condition": item["condition"], "seed": item["seed"],
                "path": item["artifacts"]["result"], "result_sha256": _sha256(Path(item["artifacts"]["result"])),
                "observer": item["artifacts"]["observer"], "observer_sha256": item["artifacts"]["observer_sha256"],
            } for item in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "No TinyLLM model, sensor front end, or task head was trained or updated.",
            "The fitted observer is a clean-split diagnostic estimator, not part of the deployed system.",
            "The target-oracle write is label-using positive control evidence only.",
            "A one-step local task-gradient write does not establish global decoder linearity.",
        ],
        "artifacts": {"campaign": str(campaign_path)},
    }
    _write_json(campaign_path, campaign_result)
    return campaign_result


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/experiments/tinyllm_calibration_readout_decomposition/20260807_d8_existing_checkpoints"),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = ReadoutDecompositionConfig(
        seeds=args.seeds,
        required_seed_passes=len(args.seeds) if args.allow_underpowered else 4,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    result = run_campaign(config, args.output)
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
